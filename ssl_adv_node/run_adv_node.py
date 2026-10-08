"""LaplaceGNN node-level pre-training with linear-probe evaluation (Phase-0 protocol)."""
import copy
import gc
import hashlib
import json
import logging
import os
import sys
import time

import numpy as np
import torch
from absl import app
from absl import flags
from torch.nn.functional import cosine_similarity
from torch.optim import AdamW
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from laplaceGNN.models import Encoder_Adversarial_GCN, Encoder_DualFrequency, Encoder_MLP, Encoder_SAGE
from laplaceGNN.laplaceGNN import LaplaceGNN_v1
from laplaceGNN.utils import set_random_seeds, CosineDecayScheduler, drop_feature
from laplaceGNN.data import get_heterophilous, get_dataset, get_ogbn_arxiv, get_wiki_cs, get_citeseer, get_cora, get_pubmed, get_ogbn_papers100M
from laplaceGNN.transform import get_graph_drop_transform
from laplaceGNN import protocol
from laplaceGNN.objectives import MaskedReconstruction, variance_covariance, FrequencyFilter, cca_loss, LatentPredictor, laplacian_pe
from laplaceGNN.runlog import RunLogger, default_logdir, flags_to_dict
from laplacian_augmentations.laplacian_node import LaplaceGNN_Augmentation_Node
from laplacian_augmentations.view_sampler import sampler_from_data
from laplacian_augmentations.structural_adversary import StructuralAdversary
from ssl_adv_graph.tudataset.predictors import MLP_Predictor

log = logging.getLogger(__name__)
FLAGS = flags.FLAGS
flags.DEFINE_integer('model_seed', 77, 'Random seed used for model initialization and training.')
flags.DEFINE_integer('data_seed', 7, 'Random seed used to generate train/val/test split.')
flags.DEFINE_integer('num_eval_splits', 10, 'Number of random train/val/test splits (datasets without official splits).')

# Dataset.
flags.DEFINE_enum('dataset', 'coauthor-cs',
                  ['amazon-computers', 'amazon-photos', 'coauthor-cs', 'coauthor-physics', 'wiki-cs', 'ogbn-arxiv', 'cora', 'citeseer',
                   'pubmed', 'ogbn-papers100M', 'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions'],
                  'Which graph dataset to use.')
flags.DEFINE_string('dataset_dir', './data', 'Where the dataset resides.')
flags.DEFINE_bool('feat_standardize', True, 'Standardise node features per dimension (WikiCS, Planetoid, OGB loaders).')
flags.DEFINE_enum('planetoid_split', 'public', ['public', 'random'], 'Cora/CiteSeer/PubMed: public split or random 10/10/80 splits.')

# Architecture.
flags.DEFINE_multi_integer('graph_encoder_layer', None, 'Conv layer sizes.')
flags.DEFINE_integer('predictor_hidden_size', 512, 'Hidden size of projector.')
flags.DEFINE_enum('encoder_forward', 'full', ['full', 'legacy'],
                  "'full': conv->norm->PReLU per layer. 'legacy': original forward (first two convs only, no norm/activation).")
flags.DEFINE_enum('encoder_type', 'gcn', ['gcn', 'dualfreq', 'mlp', 'sage'],
                  "'dualfreq': low-pass GCN + gated high-pass channel per layer; 'mlp': no graph; 'sage': GraphSAGE (self/neighbour weights).")
flags.DEFINE_float('gate_init', 0.0, 'Initial logit of the dual-frequency gate (0 -> 0.5; -4 -> ~0.02, i.e. nearly closed).')
flags.DEFINE_string('edge_index_file', None, 'Replace the graph with this edge_index (.pt), e.g. a poisoned graph.')
flags.DEFINE_enum('encoder_norm', 'auto', ['auto', 'batchnorm', 'layernorm'],
                  "'auto' keeps the original rule: LayerNorm + weight standardisation for OGB, BatchNorm otherwise.")

# Training hyperparameters.
flags.DEFINE_integer('epochs', 10000, 'The number of training epochs.')
flags.DEFINE_float('lr', 1e-5, 'The learning rate for model training.')
flags.DEFINE_float('weight_decay', 8e-4, 'Weight decay passed to AdamW (the original script always used 8e-4).')
flags.DEFINE_float('mm', 0.99, 'The momentum for moving average.')
flags.DEFINE_bool('centering', False, 'Whether to center the momentum.')
flags.DEFINE_bool('sharpening', False, 'Whether to sharpen the target weights.')
flags.DEFINE_float('center_mm', 0.9, 'Momentum for centering.')
flags.DEFINE_float('temperature', 0.04, 'Temperature for sharpening.')
flags.DEFINE_integer('lr_warmup_epochs', 1000, 'Warmup period for learning rate.')

# Views.
flags.DEFINE_enum('view_mode', 'spectral', ['spectral', 'spectral+feat', 'random', 'spectral+random'],
                  "spectral: Bernoulli samples of the max/min spectral flip probabilities. "
                  "spectral+feat: plus feature masking with --prob_feat (as in ssl_adv_graph/tudataset/run_adv_node.py). "
                  "random: DropEdges/DropFeatures only (what the original run_adv_node.py trained on). "
                  "spectral+random: spectral edges followed by DropEdges/DropFeatures.")
flags.DEFINE_float('prob_feat', 0.4, 'Feature masking probability for --view_mode=spectral+feat.')
flags.DEFINE_float('drop_edge_p_1', 0., 'Probability of edge dropout 1 (random view modes only).')
flags.DEFINE_float('drop_feat_p_1', 0., 'Probability of node feature dropout 1 (random view modes only).')
flags.DEFINE_float('drop_edge_p_2', 0., 'Probability of edge dropout 2 (random view modes only).')
flags.DEFINE_float('drop_feat_p_2', 0., 'Probability of node feature dropout 2 (random view modes only).')

# Spectral augmentation (previously hardcoded in `Args`).
flags.DEFINE_float('lapl_budget_ratio', 0.3, 'Budget r: expected flips <= r * |E| (was Args.treshold).')
flags.DEFINE_float('lapl_max_lr', 100., 'Step size of the max-view optimisation.')
flags.DEFINE_float('lapl_min_lr', 0.1, 'Step size of the min-view optimisation.')
flags.DEFINE_integer('lapl_iters', 7, 'Iterations T of the spectral optimisation (was Args.lapl_epoch).')
flags.DEFINE_list('centrality_types', ['degree', 'pagerank', 'eigenvector'], 'Centralities combined into the initialisation.')
flags.DEFINE_list('centrality_weights', ['0.2', '0.3', '0.5'], 'Fixed (not learned) weights alpha_k of the centralities.')
flags.DEFINE_enum('centrality_norm', 'sum', ['sum', 'minmax'], "'sum' is the original normalisation (divide by the sum).")
flags.DEFINE_bool('spectral_fast', True, 'Use top-k singular values (svd_lowrank) instead of the full spectrum.')
flags.DEFINE_integer('spectral_k', 10, 'Number of singular values when --spectral_fast.')
flags.DEFINE_float('spectral_min_prob', 1e-7, 'Flip probabilities <= this are not stored (dropped mass is logged).')
flags.DEFINE_enum('spectral_backend', 'dense', ['dense', 'sparse'], "'dense': original module. 'sparse': laplacian_augmentations/spectral_views.py.")
flags.DEFINE_enum('view_pair', 'max_min', ['max_min', 'max_max', 'max_orig', 'min_min'], 'Objectives of the two views (sparse backend).')
flags.DEFINE_float('sv_budget_ratio', 0.2, 'Sparse backend: expected flips per view = ratio * |E|.')
flags.DEFINE_integer('sv_k', 32, 'Sparse backend: eigenvalues used at each end of the spectrum.')
flags.DEFINE_integer('sv_iters', 20, 'Sparse backend: projected gradient iterations.')
flags.DEFINE_float('sv_step', 0.2, 'Sparse backend: step on the max-normalised gradient.')
flags.DEFINE_float('sv_gamma', 1.0, 'Sparse backend: weight of the centrality-preservation term.')
flags.DEFINE_enum('sv_protect', 'hubs', ['hubs', 'periphery', 'uniform'], 'Sparse backend: nodes whose centrality is preserved.')
flags.DEFINE_enum('sv_init', 'centrality', ['centrality', 'uniform', 'random'], 'Sparse backend: initialisation of Delta.')
flags.DEFINE_float('sv_new_ratio', 1.0, 'Sparse backend: absent candidate pairs per existing edge.')
flags.DEFINE_float('sv_hub_frac', 0.05, 'Sparse backend: fraction of nodes treated as hubs for candidate pairs.')
flags.DEFINE_float('view_mix', -1.0, 'Dose-response: both views sample alpha*Delta_max + (1-alpha)*Delta_min (same budget); <0 = off.')
flags.DEFINE_bool('measure_views', False, 'Log the realised spectral change of sampled views w.r.t. the input graph.')
flags.DEFINE_integer('max_dense_nodes', 20000, 'The spectral module builds dense n x n matrices; refuse larger graphs.')
flags.DEFINE_string('delta_cache_dir', './data/delta_cache', 'Cache for the optimised flip probabilities.')

# Adversarial bootstrapping (previously hardcoded in `Args`).
flags.DEFINE_float('adv_delta', 8e-5, 'L_inf bound epsilon of the hidden-layer perturbation (was Args.delta).')
flags.DEFINE_float('adv_step_size', 8e-3, 'Sign-gradient ascent step on the perturbation (was Args.step_size).')
flags.DEFINE_integer('adv_m', 3, 'Forward passes per step; adv_m - 1 ascent steps (was Args.m).')
flags.DEFINE_integer('accumulation_steps', 2, 'Gradient accumulation steps (was Args.accumulation_steps).')

# New objective terms and adversary variants.
flags.DEFINE_enum('objective', 'byol', ['byol', 'cca', 'byol+cca'], 'Self-supervised objective on the two views.')
flags.DEFINE_float('cca_lambda', 1e-3, 'Decorrelation weight of the CCA objective (also used by the CCA-SSG baseline).')
flags.DEFINE_float('cca_weight', 1.0, 'Weight of the CCA term when --objective=byol+cca.')
flags.DEFINE_float('mask_rate', 0.0, 'Fraction of nodes whose features the online encoder sees as a learnable [MASK] token.')
flags.DEFINE_float('recon_weight', 0.0, 'Weight of the masked-feature reconstruction loss (GCN decoder, scaled cosine error).')
flags.DEFINE_float('recon_alpha', 2.0, 'Exponent of the scaled cosine error.')
flags.DEFINE_float('latent_weight', 0.0, 'JEPA-style term: predict the target embeddings of masked nodes (needs --mask_rate > 0).')
flags.DEFINE_integer('latent_pe_dim', 0, 'Laplacian-eigenvector positions given to the latent predictor (0 = none).')
flags.DEFINE_float('var_weight', 0.0, 'Weight of the variance term on online embeddings.')
flags.DEFINE_float('cov_weight', 0.0, 'Weight of the covariance term on online embeddings.')
flags.DEFINE_enum('adv_filter', 'none', ['none', 'low', 'high'], 'Frequency band of the hidden-layer adversarial perturbation.')
flags.DEFINE_integer('sadv_every', 0, 'Structural adversary: update the max view against the online encoder every k epochs (0 = off).')
flags.DEFINE_float('sadv_radius', 0.1, 'Structural adversary: max change of a flip probability w.r.t. the spectral solution.')
flags.DEFINE_float('sadv_step', 0.02, 'Structural adversary: signed ascent step on the flip probabilities.')
flags.DEFINE_float('sv_curriculum', 1.0, 'Initial fraction of the flip budget; ramps linearly to 1 over training.')

# Evaluation and saving.
flags.DEFINE_integer('eval_epochs', 5, 'Evaluate every eval_epochs.')
flags.DEFINE_enum('probe_backend', 'auto', ['auto', 'torch', 'sklearn', 'liblinear-ovr'], 'Linear probe implementation.')
flags.DEFINE_string('logdir', None, 'Run directory root; a seed/timestamp sub-directory is created inside it.')
flags.DEFINE_bool('save_encoder', False, 'Save the encoder weights of the selected (best validation) checkpoint.')

SPECTRAL_MODES = ('spectral', 'spectral+feat', 'spectral+random')
HETEROPHILOUS = {'roman-empire': 'Roman-empire', 'amazon-ratings': 'Amazon-ratings', 'minesweeper': 'Minesweeper',
                 'tolokers': 'Tolokers', 'questions': 'Questions'}
AUC_DATASETS = ('minesweeper', 'tolokers', 'questions')
DELTA_CACHE_VERSION = 'phase0-v1'


def load_data():
    name = FLAGS.dataset
    if name in ['amazon-computers', 'amazon-photos', 'coauthor-cs', 'coauthor-physics']:
        data = get_dataset(FLAGS.dataset_dir, name)[0]
        splits = protocol.random_splits(data.num_nodes, FLAGS.num_eval_splits, FLAGS.data_seed)
    elif name == 'wiki-cs':
        dataset, train_masks, val_masks, test_masks = get_wiki_cs(FLAGS.dataset_dir, standardize=FLAGS.feat_standardize)
        data = dataset[0]
        splits = protocol.masks_to_splits(train_masks, val_masks, test_masks)
    elif name in ['cora', 'citeseer', 'pubmed']:
        loader = {'cora': get_cora, 'citeseer': get_citeseer, 'pubmed': get_pubmed}[name]
        dataset, train_masks, val_masks, test_masks = loader(FLAGS.dataset_dir, standardize=FLAGS.feat_standardize)
        data = dataset[0]
        if FLAGS.planetoid_split == 'public':
            splits = protocol.masks_to_splits(train_masks, val_masks, test_masks)
        else:
            splits = protocol.random_splits(data.num_nodes, FLAGS.num_eval_splits, FLAGS.data_seed)
    elif name in HETEROPHILOUS:
        dataset, train_masks, val_masks, test_masks = get_heterophilous(FLAGS.dataset_dir, HETEROPHILOUS[name], FLAGS.feat_standardize)
        data = dataset[0]
        splits = protocol.masks_to_splits(train_masks, val_masks, test_masks)
    elif name == 'ogbn-arxiv':
        dataset, train_idx, val_idx, test_idx = get_ogbn_arxiv(FLAGS.dataset_dir, standardize=FLAGS.feat_standardize)
        data = dataset[0]
        splits = protocol.index_split(train_idx, val_idx, test_idx)
    else:
        dataset, train_idx, val_idx, test_idx = get_ogbn_papers100M(FLAGS.dataset_dir, standardize=FLAGS.feat_standardize)
        data = dataset[0]
        splits = protocol.index_split(train_idx, val_idx, test_idx)
    if FLAGS.edge_index_file:
        data.edge_index = torch.load(FLAGS.edge_index_file)
    protocol.check_splits(splits, data.num_nodes)
    return data, splits


def metric_name():
    return 'auc' if FLAGS.dataset in AUC_DATASETS else 'acc'


def spectral_cache_key(name):
    if FLAGS.spectral_backend == 'sparse':
        keys = ['dataset', 'spectral_backend', 'view_pair', 'sv_budget_ratio', 'sv_k', 'sv_iters', 'sv_step', 'sv_gamma',
                'sv_protect', 'sv_init', 'sv_new_ratio', 'sv_hub_frac', 'centrality_types', 'centrality_weights', 'model_seed',
                'view_mix', 'edge_index_file']
    else:
        keys = ['dataset', 'lapl_budget_ratio', 'lapl_max_lr', 'lapl_min_lr', 'lapl_iters', 'centrality_types',
                'centrality_weights', 'centrality_norm', 'spectral_fast', 'spectral_k', 'spectral_min_prob', 'edge_index_file']
    cfg = {k: getattr(FLAGS, k) for k in keys}
    cfg['version'] = DELTA_CACHE_VERSION
    digest = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    return os.path.join(FLAGS.delta_cache_dir, f'{name}-{digest}.pt'), cfg


def sparse_spectral_probs(data, device):
    from laplacian_augmentations.spectral_views import SpectralViewGenerator
    weights = [float(w) for w in FLAGS.centrality_weights]
    gen = SpectralViewGenerator(budget_ratio=FLAGS.sv_budget_ratio, k=FLAGS.sv_k, iters=FLAGS.sv_iters, step=FLAGS.sv_step,
                                gamma=FLAGS.sv_gamma, protect=FLAGS.sv_protect, init=FLAGS.sv_init,
                                centrality_types=FLAGS.centrality_types, centrality_weights=weights,
                                new_ratio=FLAGS.sv_new_ratio, hub_frac=FLAGS.sv_hub_frac, seed=FLAGS.model_seed, device=device)
    signs = {'max_min': (1, -1), 'max_max': (1, 1), 'max_orig': (1, None), 'min_min': (-1, -1)}[FLAGS.view_pair]
    if FLAGS.view_mix >= 0:
        signs = (1, -1)
    tensors, stats = {}, {}
    for view, sign in zip(('max', 'min'), signs):
        if sign is None:
            row = col = torch.zeros(0, dtype=torch.long)
            prob, st = torch.zeros(0), {'identity_view': True}
        else:
            gen.seed = FLAGS.model_seed + (0 if (view == 'max' or FLAGS.view_mix >= 0) else 1)
            row, col, prob, st = gen.generate(data.edge_index, data.num_nodes, sign)
        tensors.update({f'{view}_row': row, f'{view}_col': col, f'{view}_prob': prob})
        stats[view] = st
    if FLAGS.view_mix >= 0:
        n, a = data.num_nodes, FLAGS.view_mix
        keys = torch.cat([tensors['max_row'] * n + tensors['max_col'], tensors['min_row'] * n + tensors['min_col']])
        vals = torch.cat([a * tensors['max_prob'], (1 - a) * tensors['min_prob']])
        uniq, inv = torch.unique(keys, return_inverse=True)
        mixed = torch.zeros(uniq.numel()).index_add_(0, inv, vals)
        for view in ('max', 'min'):
            tensors.update({f'{view}_row': uniq // n, f'{view}_col': uniq % n, f'{view}_prob': mixed})
        stats['mix'] = {'alpha': a, 'expected_flips': float(mixed.sum())}
    return tensors, stats


def compute_spectral_probs(data, device):
    if FLAGS.spectral_backend == 'sparse':
        path, cfg = spectral_cache_key(FLAGS.dataset)
        if os.path.exists(path):
            cached = torch.load(path)
            tensors, stats = cached['tensors'], dict(cached['stats'], loaded_from_cache=True)
        else:
            tensors, stats = sparse_spectral_probs(data, device)
            os.makedirs(FLAGS.delta_cache_dir, exist_ok=True)
            torch.save({'config': cfg, 'stats': stats, 'tensors': tensors}, path)
        for k, v in tensors.items():
            data[k] = v
        return data, dict(stats, cache=path)
    if data.num_nodes > FLAGS.max_dense_nodes:
        raise ValueError(
            f'{FLAGS.dataset} has {data.num_nodes} nodes; the current spectral module builds dense n x n matrices '
            f'and is limited to --max_dense_nodes={FLAGS.max_dense_nodes}. Use --view_mode=random to run what the '
            f'original script ran, or wait for the sparse backend (Phase 1).')
    path, cfg = spectral_cache_key(FLAGS.dataset)
    if os.path.exists(path):
        cached = torch.load(path)
        for k, v in cached['tensors'].items():
            data[k] = v
        print(f'Loaded spectral flip probabilities from {path}')
        return data, dict(cached['stats'], cache=path, loaded_from_cache=True)

    weights = [float(w) for w in FLAGS.centrality_weights]
    common = dict(ratio=FLAGS.lapl_budget_ratio, iteration=FLAGS.lapl_iters, device=device,
                  centrality_types=FLAGS.centrality_types, centrality_weights=weights, precomputed_centrality=None,
                  sample='no', centrality_norm=FLAGS.centrality_norm, spectral_fast=FLAGS.spectral_fast,
                  spectral_k=FLAGS.spectral_k, store='compact', min_prob=FLAGS.spectral_min_prob)
    stats, tensors = {}, {}
    work = Data(x=data.x, edge_index=data.edge_index, num_nodes=data.num_nodes)
    for dis_type, lr in (('max', FLAGS.lapl_max_lr), ('min', FLAGS.lapl_min_lr)):
        view = LaplaceGNN_Augmentation_Node(lr=lr, dis_type=dis_type, **common)
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        t0 = time.time()
        work = view.calc_prob(work)
        stats[dis_type] = dict(view.stats, seconds=time.time() - t0,
                               peak_gpu_gb=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None)
        for suffix in ('row', 'col', 'prob'):
            tensors[f'{dis_type}_{suffix}'] = work[f'{dis_type}_{suffix}']
        torch.cuda.empty_cache()
        gc.collect()
    os.makedirs(FLAGS.delta_cache_dir, exist_ok=True)
    torch.save({'config': cfg, 'stats': stats, 'tensors': tensors}, path)
    for k, v in tensors.items():
        data[k] = v
    return data, dict(stats, cache=path, loaded_from_cache=False)


def main(argv):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_random_seeds(random_seed=FLAGS.model_seed)

    logdir = os.path.join(FLAGS.logdir, f'seed{FLAGS.model_seed}-{time.strftime("%Y%m%d-%H%M%S")}') if FLAGS.logdir \
        else default_logdir('node', FLAGS.dataset, FLAGS.model_seed)
    runlog = RunLogger(logdir, flags_to_dict(FLAGS), pipeline='ssl_adv_node/run_adv_node.py')

    data, splits = load_data()
    runlog.update_config(num_nodes=data.num_nodes, num_edges=int(data.edge_index.shape[1]), num_splits=len(splits),
                         split_sizes=[{k: len(v) for k, v in s.items()} for s in splits])

    ######################### Views ############################
    uses_spectral = FLAGS.view_mode in SPECTRAL_MODES
    uses_random = FLAGS.view_mode in ('random', 'spectral+random')
    if uses_spectral:
        data, spectral_stats = compute_spectral_probs(data, device)
        runlog.update_config(spectral=spectral_stats)
        sampler_max = sampler_from_data(data, 'max', device)
        sampler_min = sampler_from_data(data, 'min', device)
    transform_1 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_1, drop_feat_p=FLAGS.drop_feat_p_1)
    transform_2 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_2, drop_feat_p=FLAGS.drop_feat_p_2)
    clean = Data(x=data.x, edge_index=data.edge_index, y=data.y).to(device)

    def make_views(epoch):
        if uses_spectral:
            scale = FLAGS.sv_curriculum + (1 - FLAGS.sv_curriculum) * min(1.0, epoch / FLAGS.epochs)
            (ei1, _, s1), (ei2, _, s2) = sampler_max.sample(scale=scale), sampler_min.sample(scale=scale)
            v1, v2 = Data(x=clean.x, edge_index=ei1), Data(x=clean.x, edge_index=ei2)
            if FLAGS.view_mode == 'spectral+feat':
                v1.x, v2.x = drop_feature(clean.x, FLAGS.prob_feat), drop_feature(clean.x, FLAGS.prob_feat)
            stats = {'max_added': s1['added'], 'max_removed': s1['removed'], 'min_added': s2['added'], 'min_removed': s2['removed']}
        else:
            v1, v2, stats = clean, clean, {}
        if uses_random:
            v1, v2 = transform_1(v1), transform_2(v2)
        return v1, v2, stats

    if FLAGS.measure_views:
        from laplacian_augmentations.spectral_views import extremal_spectrum, lower_pairs
        def spectral_distance(edge_index):
            r, c = lower_pairs(edge_index.cpu(), clean.num_nodes)
            lam, _ = extremal_spectrum(r, c, torch.ones(r.numel(), dtype=torch.float64), clean.num_nodes, FLAGS.sv_k)
            return lam
        lam0 = spectral_distance(clean.edge_index)
        dists = []
        for _ in range(5):
            w1, w2, _ = make_views(FLAGS.epochs)
            for w in (w1, w2):
                lam = spectral_distance(w.edge_index)
                m = min(lam.numel(), lam0.numel())
                dists.append(float(((lam[:m] - lam0[:m]) ** 2).sum() / (lam0[:m] ** 2).sum()))
        runlog.update_config(realised_spectral_change=float(np.mean(dists)), realised_spectral_change_std=float(np.std(dists)))

    ######################### Model ############################
    input_size, representation_size = data.x.size(1), FLAGS.graph_encoder_layer[-1]
    norm = FLAGS.encoder_norm
    if norm == 'auto':
        norm = 'layernorm' if FLAGS.dataset in ['ogbn-arxiv', 'ogbn-papers100M'] else 'batchnorm'
    weight_std = FLAGS.dataset in ['ogbn-arxiv', 'ogbn-papers100M']
    if FLAGS.encoder_type == 'dualfreq':
        encoder = Encoder_DualFrequency([input_size] + FLAGS.graph_encoder_layer, batchnorm=norm == 'batchnorm',
                                        layernorm=norm == 'layernorm', gate_init=FLAGS.gate_init)
    elif FLAGS.encoder_type == 'sage':
        encoder = Encoder_SAGE([input_size] + FLAGS.graph_encoder_layer, batchnorm=norm == 'batchnorm',
                               layernorm=norm == 'layernorm')
    elif FLAGS.encoder_type == 'mlp':
        encoder = Encoder_MLP([input_size] + FLAGS.graph_encoder_layer, batchnorm=norm == 'batchnorm',
                              layernorm=norm == 'layernorm')
    else:
        encoder = Encoder_Adversarial_GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=norm == 'batchnorm',
                                          layernorm=norm == 'layernorm', weight_standardization=weight_std,
                                          forward_mode=FLAGS.encoder_forward)
    predictor = MLP_Predictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size)
    model = LaplaceGNN_v1(encoder, predictor).to(device)
    print(model.online_encoder.model)
    print(model.predictor)

    aux = MaskedReconstruction(input_size, representation_size, FLAGS.recon_alpha).to(device)
    adv_filter = FrequencyFilter(clean.edge_index, clean.num_nodes, FLAGS.adv_filter)
    pe = laplacian_pe(clean.edge_index, clean.num_nodes, FLAGS.latent_pe_dim).to(device) if FLAGS.latent_pe_dim > 0 else None
    latent = LatentPredictor(representation_size, pe).to(device)
    params = model.trainable_parameters() + (list(aux.parameters()) if FLAGS.mask_rate > 0 else []) + \
        (list(latent.parameters()) if FLAGS.latent_weight > 0 else [])
    optimizer = AdamW(params, lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)
    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.epochs)

    def bootstrap_loss(v1, v2, y1, y2, perturb, masks):
        delta = adv_filter(perturb)
        h1, h2 = model.online_encoder(v1, delta, None), model.online_encoder(v2, delta, None)
        loss = h1.new_zeros(())
        if FLAGS.objective != 'cca':
            q1, q2 = model.predictor(h1), model.predictor(h2)
            loss = 2 - cosine_similarity(q1, y2, dim=-1).mean() - cosine_similarity(q2, y1, dim=-1).mean()
        if FLAGS.objective != 'byol':
            loss = loss + (FLAGS.cca_weight if FLAGS.objective == 'byol+cca' else 1.0) * cca_loss(h1, h2, FLAGS.cca_lambda)
        if FLAGS.latent_weight > 0 and masks[0] is not None:
            loss = loss + FLAGS.latent_weight * (latent.loss(h1, y1, masks[0]) + latent.loss(h2, y2, masks[1]))
        if FLAGS.recon_weight > 0:
            loss = loss + FLAGS.recon_weight * (aux.loss(h1, v1.edge_index, clean.x, masks[0]) +
                                                aux.loss(h2, v2.edge_index, clean.x, masks[1]))
        if FLAGS.var_weight > 0 or FLAGS.cov_weight > 0:
            for h in (h1, h2):
                var, cov = variance_covariance(h)
                loss = loss + FLAGS.var_weight * var + FLAGS.cov_weight * cov
        return loss

    def train_step(step, v1, v2):
        model.train()
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        mm = 1 - mm_scheduler.get(step)
        optimizer.zero_grad()

        perturb = torch.empty(v1.x.shape[0], model.online_encoder.model[0].out_channels, device=device)
        perturb.uniform_(-FLAGS.adv_delta, FLAGS.adv_delta).requires_grad_()
        # target weights only change at optimizer.step()
        y1 = y2 = None
        if FLAGS.objective != 'cca' or FLAGS.latent_weight > 0:
            with torch.no_grad():
                y1, y2 = model.target_encoder(v1).detach(), model.target_encoder(v2).detach()
        masks = (None, None)
        if FLAGS.mask_rate > 0:
            (x1, m1), (x2, m2) = aux.mask(v1.x, FLAGS.mask_rate), aux.mask(v2.x, FLAGS.mask_rate)
            v1, v2, masks = Data(x=x1, edge_index=v1.edge_index), Data(x=x2, edge_index=v2.edge_index), (m1, m2)
        for acc_step in range(FLAGS.accumulation_steps):
            if acc_step > 0:
                optimizer.zero_grad()
            loss = bootstrap_loss(v1, v2, y1, y2, perturb, masks)
            for _ in range(FLAGS.adv_m - 1):
                loss.backward(retain_graph=True)
                with torch.no_grad():
                    perturb.data += FLAGS.adv_step_size * torch.sign(perturb.grad)
                    perturb.data = perturb.data.clamp(-FLAGS.adv_delta, FLAGS.adv_delta)
                perturb.grad.zero_()
                loss = bootstrap_loss(v1, v2, y1, y2, perturb, masks)
            loss.backward(retain_graph=acc_step < FLAGS.accumulation_steps - 1)
            if (acc_step + 1) % FLAGS.accumulation_steps == 0:
                optimizer.step()
                model.update_target_network(mm, centering=FLAGS.centering, sharpening=FLAGS.sharpening,
                                            center_momentum=FLAGS.center_mm, temperature=FLAGS.temperature)
        return loss.item()

    adversary = StructuralAdversary(sampler_max, FLAGS.sadv_radius, FLAGS.sadv_step) if uses_spectral and FLAGS.sadv_every else None

    def structural_loss(edge_index, edge_weight):
        ei2, _, _ = sampler_min.sample()
        with torch.no_grad():
            y2 = model.target_encoder(Data(x=clean.x, edge_index=ei2)).detach()
        q1 = model.predictor(model.online_encoder(Data(x=clean.x, edge_index=edge_index, edge_weight=edge_weight)))
        return 1 - cosine_similarity(q1, y2, dim=-1).mean()

    selector = protocol.ValSelector('val_mean')
    labels = clean.y.cpu().numpy()

    def evaluate(epoch):
        tmp_encoder = copy.deepcopy(model.online_encoder).eval()
        with torch.no_grad():
            reps = tmp_encoder(clean)
        result = protocol.evaluate_node_embeddings(reps.cpu().numpy(), labels, splits, backend=FLAGS.probe_backend,
                                                   metric=metric_name())
        is_best = selector.update(epoch, result)
        if is_best and FLAGS.save_encoder:
            torch.save({'epoch': epoch, 'model': tmp_encoder.state_dict()}, os.path.join(logdir, 'encoder_best_val.pt'))
        print(f'Epoch {epoch}: val {result["val_mean"]:.4f} | test {result["test_mean"]:.4f} +- {result["test_std"]:.4f}'
              f' | selected epoch {selector.best["epoch"]} (val {selector.best["val_mean"]:.4f})')
        return result

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_seconds = 0.0
    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        t0 = time.time()
        sadv_stats = adversary.update(structural_loss) if adversary is not None and epoch % FLAGS.sadv_every == 0 else None
        v1, v2, view_stats = make_views(epoch)
        if sadv_stats:
            view_stats.update(sadv_stats)
        loss = train_step(epoch - 1, v1, v2)
        train_seconds += time.time() - t0
        if epoch % FLAGS.eval_epochs == 0 or epoch == FLAGS.epochs:
            result = evaluate(epoch)
            runlog.log({'epoch': epoch, 'loss': loss, 'val_mean': result['val_mean'], 'test_mean': result['test_mean'],
                        'test_std': result['test_std'], 'per_split': result['per_split'], 'views': view_stats})

    best = selector.best
    summary = {
        'dataset': FLAGS.dataset, 'model_seed': FLAGS.model_seed, 'data_seed': FLAGS.data_seed,
        'metric': 'rocauc' if metric_name() == 'auc' else 'accuracy', 'protocol': 'linear probe; C and checkpoint selected on validation',
        'best_epoch': best['epoch'], 'val_mean': best['val_mean'],
        'test_mean': best['test_mean'], 'test_std_over_splits': best['test_std'],
        'train_seconds': train_seconds, 'seconds_per_epoch': train_seconds / FLAGS.epochs,
        'peak_gpu_gb_training': torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        'selector': selector.summary(),
    }
    runlog.finish(summary)
    return summary


if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
