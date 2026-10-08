"""LaplaceGNN graph-level pre-training on ogbg-mol* with probe evaluation (Phase-0 protocol)."""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from LaplaceGNN4Graph import LaplaceGNN_Graph
from models_ogb import GNN
from transforms import get_graph_drop_transform
from ..augmentations_graph import LaplaceGNN_Augmentation_Graph
from laplaceGNN import protocol
from laplaceGNN.runlog import RunLogger, default_logdir
from laplaceGNN.utils import set_random_seeds, drop_feature
from laplacian_augmentations.view_sampler import sampler_from_sparse, expected_flips_per_graph, warn_if_noop

parser = argparse.ArgumentParser(description='LaplaceGNN on ogbg-mol* datasets')
parser.add_argument('--gnn', type=str, default='gcn', help='gin or gcn')
parser.add_argument('--drop_ratio', type=float, default=0, help='dropout ratio')
parser.add_argument('--decay', type=float, default=0.99, help='moving average decay of the target encoder')
parser.add_argument('--num_layer', type=int, default=2, help='number of GNN layers')
parser.add_argument('--emb_dim', type=int, default=512, help='GNN hidden dimension')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--dataset', type=str, default="ogbg-moltox21", help='ogbg-molbbbp, ogbg-molhiv, ogbg-moltox21, ogbg-moltoxcast')
parser.add_argument('--pp', type=str, default="H", help='perturb position: X (features) or H (hidden layer)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--hidden_channels', type=int, default=512, help='hidden size of the MLP probe (--probe=mlp)')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--step_size', type=float, default=8e-3)
parser.add_argument('--delta', type=float, default=8e-3)
parser.add_argument('--m', type=int, default=3)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--projection_hidden_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=77)
parser.add_argument('--projection_size', type=int, default=512)
parser.add_argument('--prediction_size', type=int, default=512)
parser.add_argument('--view_mode', type=str, default='spectral', choices=['spectral', 'spectral+feat', 'random', 'spectral+random'],
                    help="random = what the original script trained on (DropEdges/DropFeatures of the original batch)")
parser.add_argument('--drop_edge_p_1', type=float, default=0.1)
parser.add_argument('--drop_feat_p_1', type=float, default=0.1)
parser.add_argument('--drop_edge_p_2', type=float, default=0.3)
parser.add_argument('--drop_feat_p_2', type=float, default=0.1)
parser.add_argument('--lapl_max_lr', type=float, default=100, help='augmentation learning rate for laplacian max strategy')
parser.add_argument('--lapl_min_lr', type=float, default=0.1, help='augmentation learning rate for laplacian min strategy')
parser.add_argument('--lapl_epoch', type=int, default=10, help='iteration for augmentation')
parser.add_argument('--prob_feat', type=float, default=0.4, help='feature masking probability (--view_mode=spectral+feat)')
parser.add_argument('--threshold', type=float, default=0.3, help='budget ratio r for edge perturbation')
parser.add_argument('--legacy_view_swap', action='store_true', help='restore the original online/target view swap')
parser.add_argument('--probe', type=str, default='linear', choices=['linear', 'mlp'])
parser.add_argument('--probe_epochs', type=int, default=100)
parser.add_argument('--probe_lr', type=float, default=1e-3)
parser.add_argument('--cache_dir', type=str, default='./laplacian_dataset')
parser.add_argument('--logdir', type=str, default=None)
parser.add_argument('--spectral_backend', type=str, default='batched', choices=['batched', 'legacy'])
parser.add_argument('--view_pair', type=str, default='max_min', choices=['max_min', 'max_max', 'min_min'])
parser.add_argument('--sv_budget_ratio', type=float, default=0.2)
parser.add_argument('--sv_iters', type=int, default=20)
parser.add_argument('--sv_gamma', type=float, default=1.0)
parser.add_argument('--sv_protect', type=str, default='hubs', choices=['hubs', 'periphery', 'uniform'])
parser.add_argument('--sv_init', type=str, default='centrality', choices=['centrality', 'uniform'])
args = parser.parse_args()

SPECTRAL_CACHE_VERSION = 'phase0-v1'


def spectral_views(batch, device):
    views = []
    for dis_type in ('max', 'min'):
        sampler = sampler_from_sparse(batch[dis_type], batch.edge_index, batch.num_nodes, device,
                                      edge_attr=batch.edge_attr, batch=batch.batch, ptr=batch.ptr)
        edge_index, edge_attr, _ = sampler.sample()
        view = batch.clone()
        view.edge_index, view.edge_attr = edge_index, edge_attr
        if args.view_mode == 'spectral+feat':
            view.x = drop_feature(view.x, args.prob_feat)
        views.append(view)
    return views


def make_views(batch, device, transform_1, transform_2):
    if args.view_mode in ('spectral', 'spectral+feat', 'spectral+random'):
        batch_1, batch_2 = spectral_views(batch, device)
    else:
        batch_1, batch_2 = batch, batch
    if args.view_mode in ('random', 'spectral+random'):
        batch_1, batch_2 = transform_1(batch_1), transform_2(batch_2)
    return batch_1, batch_2


def train_adv_bootstrap(model, device, loader, optimizer, transform_1, transform_2):
    total_loss, n_batches = 0.0, 0
    model.train()
    for batch in loader:
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            continue
        batch_1, batch_2 = make_views(batch, device, transform_1, transform_2)
        optimizer.zero_grad()

        perturb = torch.FloatTensor(batch_1.x.shape[0], args.emb_dim).uniform_(-args.delta, args.delta).to(device)
        perturb.requires_grad_()
        loss = model(batch_1, batch_2, perturb)
        for _ in range(args.m - 1):
            loss.backward()
            perturb.data = (perturb.detach() + args.step_size * torch.sign(perturb.grad.detach())).data
            perturb.grad[:] = 0
            loss = model(batch_1, batch_2, perturb)
            loss /= args.m
        loss.backward()
        optimizer.step()
        model.update_moving_average()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def embed_all(model, loader, device):
    model.eval()
    embs, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        embs.append(model.embed(batch))
        ys.append(batch.y.view(batch.num_graphs, -1).float())
    return torch.cat(embs), torch.cat(ys)


def spectral_cache_path(dataset_name):
    cfg = {'dataset': dataset_name, 'lapl_max_lr': args.lapl_max_lr, 'lapl_min_lr': args.lapl_min_lr,
           'lapl_epoch': args.lapl_epoch, 'threshold': args.threshold, 'version': SPECTRAL_CACHE_VERSION}
    digest = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    return os.path.join(args.cache_dir, f'laplacian_{dataset_name}-{digest}.pt'), cfg


def precompute_spectral(dataset, device):
    if args.spectral_backend == 'batched':
        from laplacian_augmentations.spectral_views import BatchedSpectralViewGenerator
        cfg = {k: getattr(args, k) for k in ('sv_budget_ratio', 'sv_iters', 'sv_gamma', 'sv_protect', 'sv_init', 'view_pair')}
        digest = hashlib.sha1(json.dumps(dict(cfg, dataset=args.dataset), sort_keys=True).encode()).hexdigest()[:12]
        path = os.path.join(args.cache_dir, f'sv_{args.dataset}-{digest}.pt')
        if os.path.exists(path):
            return torch.load(path), {'cache': path, 'loaded_from_cache': True}
        graphs = [dataset.get(i) for i in range(len(dataset))]
        gen = BatchedSpectralViewGenerator(budget_ratio=args.sv_budget_ratio, iters=args.sv_iters, gamma=args.sv_gamma,
                                           protect=args.sv_protect, init=args.sv_init, seed=args.seed, device=device)
        signs = {'max_min': (1, -1), 'max_max': (1, 1), 'min_min': (-1, -1)}[args.view_pair]
        stats = {}
        for key, sign in zip(('max', 'min'), signs):
            probs, stats[key] = gen.generate(graphs, sign)
            for d, pr in zip(graphs, probs):
                d[key] = pr
        os.makedirs(args.cache_dir, exist_ok=True)
        torch.save(graphs, path)
        return graphs, {'cache': path, 'loaded_from_cache': False, 'config': cfg, **stats}
    path, cfg = spectral_cache_path(args.dataset)
    if os.path.exists(path):
        print(f'Loaded precomputed flip probabilities from {path}')
        return torch.load(path), {'cache': path, 'loaded_from_cache': True}
    centrality_types = ['degree', 'pagerank', 'eigenvector']
    centrality_weights = [0.2, 0.3, 0.5]
    views = [LaplaceGNN_Augmentation_Graph(ratio=args.threshold, lr=lr, iteration=args.lapl_epoch, dis_type=dis,
                                           device=device, centrality_types=centrality_types,
                                           centrality_weights=centrality_weights, precomputed_centrality=None, sample='no')
             for dis, lr in (('max', args.lapl_max_lr), ('min', args.lapl_min_lr))]
    t0 = time.time()
    updated = []
    for i in tqdm(range(len(dataset)), desc='Precomputing flip probabilities'):
        data = dataset.get(i)
        for view in views:
            data = view.calc_prob(data, silence=True)
        updated.append(data)
    os.makedirs(args.cache_dir, exist_ok=True)
    torch.save(updated, path)
    return updated, {'cache': path, 'loaded_from_cache': False, 'seconds': time.time() - t0, 'config': cfg}


def main():
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    set_random_seeds(args.seed)
    logdir = os.path.join(args.logdir, f'seed{args.seed}-{time.strftime("%Y%m%d-%H%M%S")}') if args.logdir \
        else default_logdir('ogb', args.dataset, args.seed)
    runlog = RunLogger(logdir, vars(args), pipeline='ssl_adv_graph/ogb/run_adv_graph.py')

    dataset = PygGraphPropPredDataset(name=args.dataset)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(args.dataset)
    metric = dataset.eval_metric

    if args.view_mode == 'random':
        graphs = [dataset.get(i) for i in range(len(dataset))]
        spectral_info = None
    else:
        graphs, spectral_info = precompute_spectral(dataset, device)
        spectral_info['expected_flips_per_graph'] = expected_flips_per_graph(graphs)
        spectral_info['noop'] = warn_if_noop(spectral_info['expected_flips_per_graph'])
    runlog.update_config(spectral=spectral_info, metric=metric,
                         split_sizes={k: len(v) for k, v in split_idx.items()})

    train_graphs = [graphs[i] for i in split_idx['train'].tolist()]
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    eval_loader = DataLoader([dataset.get(i) for i in range(len(dataset))], batch_size=args.batch_size, shuffle=False)
    transform_1 = get_graph_drop_transform(drop_edge_p=args.drop_edge_p_1, drop_feat_p=args.drop_feat_p_1)
    transform_2 = get_graph_drop_transform(drop_edge_p=args.drop_edge_p_2, drop_feat_p=args.drop_feat_p_2)

    feat_dim = dataset.data.x.shape[-1]
    if args.gnn not in ('gin', 'gcn'):
        raise ValueError('Invalid GNN-encoder type')
    encoder = GNN(gnn_type=args.gnn, num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
                  drop_ratio=args.drop_ratio, feat_dim=feat_dim, perturb_position=args.pp).to(device)
    model = LaplaceGNN_Graph(encoder, num_tasks=dataset.num_tasks, emb_dim=args.emb_dim, projection_size=args.projection_size,
                             prediction_size=args.prediction_size, projection_hidden_size=args.projection_hidden_size,
                             moving_average_decay=args.decay, legacy_view_swap=args.legacy_view_swap).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    selector = protocol.ValSelector('val')
    train_seconds = 0.0
    for epoch in tqdm(range(1, args.epochs + 1)):
        t0 = time.time()
        loss = train_adv_bootstrap(model, device, train_loader, optimizer, transform_1, transform_2)
        train_seconds += time.time() - t0
        if epoch % args.test_freq == 0 or epoch == args.epochs:
            emb, y = embed_all(model, eval_loader, device)
            result = protocol.ogb_probe_eval(emb, y, split_idx, evaluator, metric, probe=args.probe, hidden=args.hidden_channels,
                                             epochs=args.probe_epochs, lr=args.probe_lr, seed=args.seed)
            selector.update(epoch, result)
            runlog.log(dict(result, epoch=epoch, loss=loss))
            print(f'epoch {epoch}: loss {loss:.4f} | {metric} train {result["train"]:.4f} val {result["val"]:.4f} '
                  f'test {result["test"]:.4f} | selected epoch {selector.best["epoch"]}')

    best = selector.best
    runlog.finish({'dataset': args.dataset, 'seed': args.seed, 'metric': metric,
                   'protocol': f'fresh {args.probe} probe per evaluation on frozen embeddings; probe epoch and checkpoint selected on validation',
                   'best_epoch': best['epoch'], 'val': best['val'], 'test': best['test'], 'train': best['train'],
                   'train_seconds': train_seconds, 'selector': selector.summary()})


if __name__ == "__main__":
    main()
