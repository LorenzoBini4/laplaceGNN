"""TU graph classification: paired random search (spectral vs random views on the same sampled configs), selection on
the mean inner-validation accuracy of the 10-fold CV, then fresh seeds of each mode's best config.

    python scripts/tune_tu.py --dataset MUTAG --trials 12 --seeds 5
Output: runs/tu_tune/<dataset>/<mode>/, results/tu_tuned.csv
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE = {
    'lr': [1e-4, 5e-4, 1e-3, 5e-3], 'epoch': [100, 200], 'dim': [128, 256, 512], 'batch_size': [32, 128],
    'sv_budget_ratio': [0.1, 0.2, 0.3], 'delta': [1e-3, 1e-2, 8e-2], 'm': [0, 1, 2], 'mm': [0.9, 0.99],
}
MODES = ('spectral', 'random')


def configs(n):
    rng = random.Random(0)
    return [{k: rng.choice(v) for k, v in SPACE.items()} for _ in range(n)]


def run(dataset, mode, cfg, seed, logdir):
    os.makedirs(logdir, exist_ok=True)
    if glob.glob(os.path.join(logdir, '*', 'final.json')):
        path = glob.glob(os.path.join(logdir, '*', 'final.json'))[0]
    else:
        cmd = [sys.executable, '-m', 'ssl_adv_graph.tudataset.run_adv_graph', '--dataset', dataset, '--view_mode', mode,
               '--seed', str(seed), '--lr', str(cfg['lr']), '--epoch', str(cfg['epoch']), '--batch_size', str(cfg['batch_size']),
               '--gnn1_num_layers', '2', '--gnn2_num_layers', '2', '--gnn1_dim', str(cfg['dim']), '--gnn2_dim', str(cfg['dim']),
               '--mlp_dim', str(cfg['dim']), '--sv_budget_ratio', str(cfg['sv_budget_ratio']), '--delta', str(cfg['delta']),
               '--step_size', str(cfg['delta']), '--m', str(cfg['m']), '--mm', str(cfg['mm']), '--logdir', logdir]
        with open(os.path.join(logdir, 'run.log'), 'w') as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        path = glob.glob(os.path.join(logdir, '*', 'final.json'))[0]
    r = json.load(open(path))
    return float(np.mean([f['inner_val_acc'] for f in r['folds']])), r['test_mean']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--trials', type=int, default=12)
    p.add_argument('--seeds', type=int, default=5)
    args = p.parse_args()
    cfgs = configs(args.trials)
    base = os.path.join(ROOT, 'runs', 'tu_tune', args.dataset)
    rows = []
    for mode in MODES:
        trials = []
        for i, cfg in enumerate(cfgs):
            val, test = run(args.dataset, mode, cfg, 0, os.path.join(base, mode, f'trial{i:02d}'))
            trials.append((val, test, i))
            print(f'{args.dataset} {mode} trial {i}: val {val:.4f} test {test:.4f}', flush=True)
        val, _, best = max(trials)
        tests = [run(args.dataset, mode, cfgs[best], s, os.path.join(base, mode, f'best_seed{s}'))[1] for s in range(1, args.seeds + 1)]
        rows.append({'dataset': args.dataset, 'mode': mode, 'trials': args.trials, 'best_trial': best, 'best_val': round(100 * val, 2),
                     'seeds': args.seeds, 'test': round(100 * float(np.mean(tests)), 2), 'test_std': round(100 * float(np.std(tests)), 2),
                     'config': json.dumps(cfgs[best])})
        print(rows[-1], flush=True)
    path = os.path.join(ROOT, 'results', 'tu_tuned.csv')
    old = [r for r in csv.DictReader(open(path))] if os.path.exists(path) else []
    keep = [r for r in old if r['dataset'] != args.dataset]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(keep + rows)


if __name__ == '__main__':
    main()
