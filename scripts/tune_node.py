"""Equal-budget random search on validation accuracy for the node pipeline.

    python scripts/tune_node.py --dataset cora --method laplacegnn --trials 60 --workers 4
    python scripts/tune_node.py --dataset cora --method laplacegnn --final_seeds 10   # rerun the best config

Test accuracy is stored with each trial but never used for selection.
"""
import argparse
import glob
import json
import math
import os
import random
import subprocess
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TRAINING_SPACE = {
    'lr': ('log', 1e-4, 5e-3),
    'weight_decay': ('log', 1e-6, 1e-3),
    'epochs': ('choice', [500, 1000, 2000]),
    'mm': ('choice', [0.9, 0.99, 0.999]),
    'lr_warmup_epochs': ('choice', [0, 50, 100]),
    'hidden': ('choice', [256, 512]),
    'out': ('choice', [256, 512]),
    'predictor_hidden_size': ('choice', [256, 512]),
    'prob_feat': ('uniform', 0.0, 0.5),
}
RANDOM_VIEWS = {
    'drop_edge_p_1': ('uniform', 0.0, 0.5), 'drop_edge_p_2': ('uniform', 0.0, 0.5),
    'drop_feat_p_1': ('uniform', 0.0, 0.5), 'drop_feat_p_2': ('uniform', 0.0, 0.5),
}
ADVERSARIAL = {
    'adv_delta': ('log', 1e-4, 1e-1),
    'adv_m': ('choice', [2, 3, 4]),
}
SPECTRAL = {
    'sv_budget_ratio': ('uniform', 0.05, 0.5),
    'sv_gamma': ('choice', [0.0, 0.1, 1.0, 10.0]),
    'sv_protect': ('choice', ['hubs', 'periphery', 'uniform']),
    'sv_init': ('choice', ['centrality', 'uniform']),
    'view_pair': ('choice', ['max_min', 'max_max']),
}
OBJECTIVE = {
    'mask_rate': ('uniform', 0.0, 0.7),
    'recon_weight': ('choice', [0.0, 0.5, 1.0, 2.0]),
    'var_weight': ('choice', [0.0, 0.1, 1.0]),
    'cov_weight': ('choice', [0.0, 0.01, 0.04]),
    'adv_filter': ('choice', ['none', 'low', 'high']),
    'sv_curriculum': ('choice', [1.0, 0.5, 0.25]),
    'encoder_type': ('choice', ['gcn', 'dualfreq']),
}
STRUCTURAL = {
    'sadv_every': ('choice', [0, 5, 10, 25]),
    'sadv_radius': ('choice', [0.05, 0.1, 0.2]),
    'sadv_step': ('choice', [0.01, 0.02, 0.05]),
}
CCASSG = {
    'lr': ('log', 1e-4, 5e-3), 'weight_decay': ('log', 1e-6, 1e-3), 'epochs': ('choice', [20, 50, 100, 200, 500]),
    'hidden': ('choice', [256, 512]), 'out': ('choice', [256, 512]), 'cca_lambda': ('log', 1e-4, 1e-2), **RANDOM_VIEWS,
}
GRAPHMAE = {
    'lr': ('log', 1e-4, 5e-3), 'weight_decay': ('log', 1e-5, 1e-3), 'epochs': ('choice', [300, 500, 1000, 1500]),
    'hidden': ('choice', [256, 512]), 'out': ('choice', [256, 512]), 'gat_heads': ('choice', [2, 4]),
    'mask_rate': ('uniform', 0.3, 0.8), 'recon_alpha': ('choice', [1.0, 2.0, 3.0]),
    'feat_drop': ('uniform', 0.0, 0.3), 'attn_drop': ('uniform', 0.0, 0.3),
}
POLYGCL = {
    'lr': ('choice', [1e-4, 5e-4, 1e-3]), 'poly_lr_prop': ('choice', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
    'weight_decay': ('choice', [0.0, 1e-5]), 'poly_wd_prop': ('choice', [0.0, 1e-5]), 'epochs': ('choice', [200, 500]),
    'out': ('choice', [256, 512]), 'poly_K': ('choice', [5, 10]),
    'poly_dropout': ('uniform', 0.0, 0.6), 'poly_dprate': ('uniform', 0.0, 0.6),
}
METHODS = {
    # BGRL-style bootstrap: random drops, no adversarial step
    'bgrl': ({'view_mode': 'random', 'adv_m': 1}, {**TRAINING_SPACE, **RANDOM_VIEWS}),
    'bgrl_adv': ({'view_mode': 'random'}, {**TRAINING_SPACE, **RANDOM_VIEWS, **ADVERSARIAL}),
    'laplacegnn': ({'view_mode': 'spectral+feat', 'spectral_backend': 'sparse'}, {**TRAINING_SPACE, **SPECTRAL, **ADVERSARIAL, **OBJECTIVE}),
    'laplacegnn_base': ({'view_mode': 'spectral+feat', 'spectral_backend': 'sparse'}, {**TRAINING_SPACE, **SPECTRAL, **ADVERSARIAL}),
    'laplacegnn_noadv': ({'view_mode': 'spectral+feat', 'spectral_backend': 'sparse', 'adv_m': 1}, {**TRAINING_SPACE, **SPECTRAL}),
    'laplacegnn_v2': ({'view_mode': 'spectral+feat', 'spectral_backend': 'sparse'},
                      {**TRAINING_SPACE, **SPECTRAL, **ADVERSARIAL, **OBJECTIVE, **STRUCTURAL}),
    'laplacegnn_cca': ({'view_mode': 'spectral+feat', 'spectral_backend': 'sparse'},
                       {**TRAINING_SPACE, **SPECTRAL, **ADVERSARIAL, **OBJECTIVE, **STRUCTURAL,
                        'epochs': ('choice', [50, 100, 200, 500, 1000]), 'objective': ('choice', ['cca', 'byol+cca']),
                        'cca_lambda': ('log', 1e-4, 1e-2), 'cca_weight': ('choice', [0.1, 1.0])}),
    'laplacegnn_full': ({'view_mode': 'spectral+feat', 'spectral_backend': 'sparse'},
                        {**TRAINING_SPACE, **SPECTRAL, **ADVERSARIAL, **OBJECTIVE, **STRUCTURAL,
                         'epochs': ('choice', [100, 200, 500, 1000, 2000]),
                         'objective': ('choice', ['byol', 'cca', 'byol+cca']),
                         'cca_lambda': ('log', 1e-4, 1e-2), 'cca_weight': ('choice', [0.1, 1.0]),
                         'latent_weight': ('choice', [0.0, 0.5, 1.0]), 'latent_pe_dim': ('choice', [0, 16])}),
    'ccassg': ({'baseline': 'ccassg'}, CCASSG),
    'polygcl': ({'baseline': 'polygcl'}, POLYGCL),
    'ccassg_dualfreq': ({'baseline': 'ccassg', 'encoder_type': 'dualfreq'}, CCASSG),
    'bgrl_dualfreq': ({'view_mode': 'random', 'adv_m': 1, 'encoder_type': 'dualfreq'}, {**TRAINING_SPACE, **RANDOM_VIEWS}),
    'bgrl_dualfreq_gate': ({'view_mode': 'random', 'adv_m': 1, 'encoder_type': 'dualfreq', 'gate_init': -4.0},
                           {**TRAINING_SPACE, **RANDOM_VIEWS}),
    'ccassg_dualfreq_gate': ({'baseline': 'ccassg', 'encoder_type': 'dualfreq', 'gate_init': -4.0}, CCASSG),
    'bgrl_mlp': ({'view_mode': 'random', 'adv_m': 1, 'encoder_type': 'mlp'}, {**TRAINING_SPACE, **RANDOM_VIEWS}),
    'bgrl_sage': ({'view_mode': 'random', 'adv_m': 1, 'encoder_type': 'sage'}, {**TRAINING_SPACE, **RANDOM_VIEWS}),
    'graphmae': ({'baseline': 'graphmae'}, GRAPHMAE),
    'laplacegnn_v0': ({'view_mode': 'spectral+feat', 'spectral_backend': 'dense'}, {**TRAINING_SPACE, **ADVERSARIAL}),
}


EPOCH_CAP = {'roman-empire': 500, 'amazon-ratings': 500, 'minesweeper': 500, 'tolokers': 500, 'questions': 500}


def sample(space, rng):
    cfg = {}
    for k, (kind, *a) in space.items():
        if kind == 'log':
            cfg[k] = math.exp(rng.uniform(math.log(a[0]), math.log(a[1])))
        elif kind == 'uniform':
            cfg[k] = rng.uniform(a[0], a[1])
        else:
            cfg[k] = rng.choice(a[0])
    return cfg


def to_flags(cfg):
    cfg = {k: v for k, v in cfg.items() if k not in ('hidden', 'out')}
    if 'adv_delta' in cfg:
        cfg['adv_step_size'] = cfg['adv_delta']
    return [f'--{k}={v}' for k, v in cfg.items()]


def run_trial(dataset, fixed, cfg, logdir, seed, eval_epochs):
    from absl import flags as absl_flags
    if 'baseline' in fixed:
        import ssl_adv_node.run_baselines as node
    else:
        import ssl_adv_node.run_adv_node as node
    F = absl_flags.FLAGS
    F.unparse_flags()
    if dataset in EPOCH_CAP and 'epochs' in cfg:
        cfg = {**cfg, 'epochs': min(int(cfg['epochs']), EPOCH_CAP[dataset])}
    eval_epochs = max(eval_epochs, int(cfg.get('epochs', 0)) // 10)
    F(['tune', f'--flagfile={ROOT}/config_node/{dataset}.cfg', f'--logdir={logdir}', f'--model_seed={seed}',
       f'--eval_epochs={eval_epochs}'] + to_flags({**fixed, **cfg}))
    if 'hidden' in cfg:
        F.graph_encoder_layer = [cfg['hidden'], cfg['out']]
    elif 'out' in cfg:
        F.graph_encoder_layer = [cfg['out']]
    return node.main([])


def run_with_oom_retry(fn, retries=3, wait=120):
    import time
    import torch
    for attempt in range(retries + 1):
        try:
            return fn()
        except RuntimeError as exc:
            if 'out of memory' not in str(exc) or attempt == retries:
                raise
            torch.cuda.empty_cache()
            time.sleep(wait)


def worker(args):
    study = os.path.join(ROOT, 'runs', 'tune', args.dataset, args.method)
    fixed, space = METHODS[args.method]
    rng = random.Random(args.search_seed)
    configs = [sample(space, rng) for _ in range(args.trials)]
    path = os.path.join(study, f'trials-shard{args.shard}.jsonl')
    done = {json.loads(l)['trial'] for f in glob.glob(os.path.join(study, 'trials-shard*.jsonl')) for l in open(f)}
    out = open(path, 'a')
    for i in range(args.shard, args.trials, args.workers):
        if i in done:
            continue
        try:
            res = run_with_oom_retry(lambda: run_trial(args.dataset, fixed, configs[i], os.path.join(study, f'trial{i:03d}'),
                                                       args.seed, args.eval_epochs))
            rec = {'trial': i, 'config': configs[i], 'val': res['val_mean'], 'test': res['test_mean'], 'best_epoch': res['best_epoch']}
        except Exception as exc:
            rec = {'trial': i, 'config': configs[i], 'val': 0.0, 'error': f'{type(exc).__name__}: {exc}', 'trace': traceback.format_exc()[-800:]}
        out.write(json.dumps(rec) + '\n')
        out.flush()
        print(json.dumps({k: rec[k] for k in ('trial', 'val') if k in rec}), flush=True)


def load_trials(study):
    trials = []
    for f in sorted(os.listdir(study)):
        if f.startswith('trials-shard'):
            trials += [json.loads(l) for l in open(os.path.join(study, f))]
    return sorted(trials, key=lambda t: -t['val'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cora')
    p.add_argument('--method', default='laplacegnn', choices=list(METHODS))
    p.add_argument('--trials', type=int, default=60)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--shard', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--search_seed', type=int, default=0)
    p.add_argument('--eval_epochs', type=int, default=50)
    p.add_argument('--final_seeds', type=int, default=0)
    args = p.parse_args()
    study = os.path.join(ROOT, 'runs', 'tune', args.dataset, args.method)
    os.makedirs(study, exist_ok=True)

    if args.final_seeds:
        best = load_trials(study)[0]
        json.dump(best, open(os.path.join(study, 'best.json'), 'w'), indent=2)
        fixed, _ = METHODS[args.method]
        for seed in range(args.final_seeds):
            cmd = [sys.executable, os.path.abspath(__file__), '--dataset', args.dataset, '--method', args.method,
                   '--seed', str(seed), '--shard', '-1']
            subprocess.run(cmd, check=True, env=dict(os.environ, TUNE_FINAL=json.dumps(best['config'])))
        return
    if args.shard == -1:
        fixed, _ = METHODS[args.method]
        cfg = json.loads(os.environ['TUNE_FINAL'])
        run_trial(args.dataset, fixed, cfg, os.path.join(ROOT, 'runs', 'final', args.dataset, args.method), args.seed, args.eval_epochs)
        return
    if args.shard is not None:
        worker(args)
        return
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), *sys.argv[1:], '--shard', str(s)]) for s in range(args.workers)]
    for pr in procs:
        pr.wait()
    trials = load_trials(study)
    print(f'{len(trials)} trials; best val {trials[0]["val"]:.4f} (trial {trials[0]["trial"]})')


if __name__ == '__main__':
    main()
