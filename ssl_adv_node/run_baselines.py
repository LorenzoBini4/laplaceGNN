"""CCA-SSG (Zhang et al., 2021) and GraphMAE (Hou et al., 2022) under the same data, protocol and logging as LaplaceGNN."""
import copy
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app, flags
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from tqdm import tqdm

from laplaceGNN import protocol
from laplaceGNN.objectives import cca_loss
from laplaceGNN.models import DualFrequencyConv
from laplaceGNN.runlog import RunLogger, default_logdir, flags_to_dict
from laplaceGNN.transform import get_graph_drop_transform
from laplaceGNN.utils import set_random_seeds
import ssl_adv_node.run_adv_node as node

FLAGS = flags.FLAGS
flags.DEFINE_enum('baseline', 'ccassg', ['ccassg', 'graphmae', 'polygcl'], 'Baseline to train.')
flags.DEFINE_float('poly_lr_prop', 1e-2, 'PolyGCL: learning rate of the polynomial filter, alpha and beta.')
flags.DEFINE_float('poly_wd_prop', 0.0, 'PolyGCL: weight decay of the polynomial filter.')
flags.DEFINE_integer('poly_K', 10, 'PolyGCL: polynomial order.')
flags.DEFINE_float('poly_dropout', 0.5, 'PolyGCL: dropout after propagation.')
flags.DEFINE_float('poly_dprate', 0.5, 'PolyGCL: dropout before propagation.')
flags.DEFINE_integer('gat_heads', 4, 'GraphMAE: attention heads of the hidden encoder layers.')
flags.DEFINE_float('feat_drop', 0.2, 'GraphMAE: input feature dropout.')
flags.DEFINE_float('attn_drop', 0.1, 'GraphMAE: attention dropout.')
flags.DEFINE_integer('eval_points', 10, 'Number of evaluations over training (checkpoint chosen on validation).')


class PlainGCN(nn.Module):
    def __init__(self, sizes, conv='gcn'):
        super().__init__()
        make = (lambda a, b: GCNConv(a, b)) if conv == 'gcn' else (lambda a, b: DualFrequencyConv(a, b, FLAGS.gate_init))
        self.convs = nn.ModuleList(make(a, b) for a, b in zip(sizes[:-1], sizes[1:]))

    def forward(self, data):
        x = data.x
        for i, conv in enumerate(self.convs):
            x = conv(x, data.edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


class GAT(nn.Module):
    def __init__(self, sizes, heads, feat_drop, attn_drop, encoding=True):
        super().__init__()
        self.layers, self.acts = nn.ModuleList(), nn.ModuleList()
        self.feat_drop = feat_drop
        n = len(sizes) - 1
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = i == n - 1
            h = heads if (encoding or not last) else 1
            self.layers.append(GATConv(a, b // h, heads=h, concat=True, dropout=attn_drop))
            self.acts.append(nn.PReLU() if (encoding or not last) else nn.Identity())

    def forward(self, x, edge_index):
        for layer, act in zip(self.layers, self.acts):
            x = act(layer(F.dropout(x, self.feat_drop, self.training), edge_index))
        return x


class GraphMAE(nn.Module):
    def __init__(self, in_dim, sizes, heads, feat_drop, attn_drop, replace_rate=0.05):
        super().__init__()
        self.encoder = GAT([in_dim] + sizes, heads, feat_drop, attn_drop, encoding=True)
        self.enc2dec = nn.Linear(sizes[-1], sizes[-1], bias=False)
        self.decoder = GAT([sizes[-1], in_dim], 1, feat_drop, attn_drop, encoding=False)
        self.token = nn.Parameter(torch.zeros(1, in_dim))
        self.replace_rate = replace_rate

    def loss(self, data, mask_rate, alpha):
        n = data.num_nodes
        perm = torch.randperm(n, device=data.x.device)
        masked = perm[:int(mask_rate * n)]
        n_rep = int(self.replace_rate * masked.numel())
        x = data.x.clone()
        token_nodes, noise_nodes = masked[n_rep:], masked[:n_rep]
        x[token_nodes] = self.token
        x[noise_nodes] = data.x[torch.randint(0, n, (n_rep,), device=data.x.device)]
        h = self.enc2dec(self.encoder(x, data.edge_index))
        h[masked] = 0
        rec = self.decoder(h, data.edge_index)
        return ((1 - F.cosine_similarity(rec[masked], data.x[masked], dim=-1)) ** alpha).mean()

    def forward(self, data):
        return self.encoder(data.x, data.edge_index)


def main(argv):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_random_seeds(FLAGS.model_seed)
    logdir = os.path.join(FLAGS.logdir, f'seed{FLAGS.model_seed}-{time.strftime("%Y%m%d-%H%M%S")}') if FLAGS.logdir \
        else default_logdir(FLAGS.baseline, FLAGS.dataset, FLAGS.model_seed)
    runlog = RunLogger(logdir, flags_to_dict(FLAGS), pipeline=f'ssl_adv_node/run_baselines.py:{FLAGS.baseline}')
    data, splits = node.load_data()
    clean = Data(x=data.x, edge_index=data.edge_index).to(device)
    sizes = list(FLAGS.graph_encoder_layer)

    if FLAGS.baseline == 'ccassg':
        model = PlainGCN([data.x.shape[1]] + sizes, conv=FLAGS.encoder_type).to(device)
        t1 = get_graph_drop_transform(FLAGS.drop_edge_p_1, FLAGS.drop_feat_p_1)
        t2 = get_graph_drop_transform(FLAGS.drop_edge_p_2, FLAGS.drop_feat_p_2)
        step_loss = lambda: cca_loss(model(t1(clean)), model(t2(clean)), FLAGS.cca_lambda)
        embed = lambda m: m(clean)
    elif FLAGS.baseline == 'graphmae':
        model = GraphMAE(data.x.shape[1], sizes, FLAGS.gat_heads, FLAGS.feat_drop, FLAGS.attn_drop).to(device)
        step_loss = lambda: model.loss(clean, FLAGS.mask_rate, FLAGS.recon_alpha)
        embed = lambda m: m(clean)
    else:
        from third_party.polygcl import Model as PolyGCL
        model = PolyGCL(in_dim=data.x.shape[1], out_dim=sizes[-1], K=FLAGS.poly_K, dprate=FLAGS.poly_dprate,
                        dropout=FLAGS.poly_dropout, is_bns=False, act_fn='relu').to(device)
        n = clean.num_nodes
        lbl = torch.cat([torch.ones(2 * n), torch.zeros(2 * n)]).to(device)
        bce = nn.BCEWithLogitsLoss()
        step_loss = lambda: bce(model(clean.edge_index, clean.x, clean.x[torch.randperm(n, device=device)]), lbl)
        embed = lambda m: m.get_embedding(clean.edge_index, clean.x)

    if FLAGS.baseline == 'polygcl':
        opt = torch.optim.Adam([
            {'params': model.encoder.lin1.parameters(), 'weight_decay': FLAGS.weight_decay, 'lr': FLAGS.lr},
            {'params': model.disc.parameters(), 'weight_decay': FLAGS.weight_decay, 'lr': FLAGS.lr},
            {'params': model.encoder.prop1.parameters(), 'weight_decay': FLAGS.poly_wd_prop, 'lr': FLAGS.poly_lr_prop},
            {'params': [model.alpha, model.beta], 'weight_decay': FLAGS.poly_wd_prop, 'lr': FLAGS.poly_lr_prop}])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (1 + math.cos(e * math.pi / FLAGS.epochs)) * 0.5) \
        if FLAGS.baseline == 'graphmae' else None
    selector = protocol.ValSelector('val_mean')
    labels = data.y.numpy()
    every = max(1, FLAGS.epochs // FLAGS.eval_points)
    train_seconds = 0.0
    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        t0 = time.time()
        model.train()
        opt.zero_grad()
        loss = step_loss()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        train_seconds += time.time() - t0
        if epoch % every == 0 or epoch == FLAGS.epochs:
            m = copy.deepcopy(model).eval()
            with torch.no_grad():
                reps = embed(m)
            res = protocol.evaluate_node_embeddings(reps.cpu().numpy(), labels, splits, backend=FLAGS.probe_backend,
                                                    metric=node.metric_name())
            selector.update(epoch, res)
            runlog.log({'epoch': epoch, 'loss': loss.item(), 'val_mean': res['val_mean'], 'test_mean': res['test_mean']})
            print(f'Epoch {epoch}: val {res["val_mean"]:.4f} | test {res["test_mean"]:.4f}')
    best = selector.best
    summary = {'dataset': FLAGS.dataset, 'model_seed': FLAGS.model_seed, 'metric': 'rocauc' if node.metric_name() == 'auc' else 'accuracy',
               'protocol': 'linear probe; C and checkpoint selected on validation', 'best_epoch': best['epoch'],
               'val_mean': best['val_mean'], 'test_mean': best['test_mean'], 'test_std_over_splits': best['test_std'],
               'train_seconds': train_seconds, 'selector': selector.summary()}
    runlog.finish(summary)
    return summary


if __name__ == '__main__':
    app.run(main)
