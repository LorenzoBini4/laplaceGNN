"""Accuracy vs realised spectral change at a fixed flip budget: views sample alpha*Delta_max + (1-alpha)*Delta_min.

    python scripts/dose_response.py --dataset cora --method laplacegnn_full --seeds 3 --workers 4
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def levels(cfg):
    base = {'sadv_every': 0, 'sv_curriculum': 1.0, 'measure_views': True}
    v = {f'alpha_{a}': {**base, 'view_mix': a} for a in ALPHAS}
    budget = cfg.get('sv_budget_ratio', 0.2)
    v['random_same_budget'] = {**base, 'view_mode': 'random', 'drop_edge_p_1': budget, 'drop_edge_p_2': budget,
                               'drop_feat_p_1': cfg.get('prob_feat', 0.0), 'drop_feat_p_2': cfg.get('prob_feat', 0.0)}
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--method', default='laplacegnn_full')
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--job', default=None)
    args = p.parse_args()
    import tune_node as T
    from ablate import best_config
    fixed, _ = T.METHODS[args.method]
    cfg = best_config(args.dataset, args.method)['config']
    out_root = os.path.join(ROOT, 'runs', 'dose', args.dataset)
    vs = levels(cfg)

    if args.job is not None:
        name, seed = args.job.rsplit(':', 1)
        res_file = os.path.join(out_root, name, f'seed{seed}.json')
        if os.path.exists(res_file):
            return
        override = dict(vs[name])
        run_fixed = {**fixed, **({'view_mode': override.pop('view_mode')} if 'view_mode' in override else {})}
        logdir = os.path.join(out_root, name)
        res = T.run_trial(args.dataset, run_fixed, {**cfg, **override}, logdir, int(seed), 50)
        run_dir = max(glob.glob(os.path.join(logdir, 'seed*-*')), key=os.path.getmtime)
        extra = json.load(open(os.path.join(run_dir, 'config.json'))).get('extra', {})
        json.dump({'val': res['val_mean'], 'test': res['test_mean'],
                   'spectral_change': extra.get('realised_spectral_change')}, open(res_file, 'w'))
        return

    for n in vs:
        os.makedirs(os.path.join(out_root, n), exist_ok=True)
    running = []
    for job in [f'{n}:{s}' for n in vs for s in range(args.seeds)]:
        log = open(os.path.join(out_root, job.replace(':', '-seed') + '.log'), 'w')
        running.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), '--dataset', args.dataset,
                                         '--method', args.method, '--job', job], stdout=log, stderr=subprocess.STDOUT))
        if len(running) >= args.workers:
            running.pop(0).wait()
    for pr in running:
        pr.wait()

    rows = []
    for n in vs:
        res = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out_root, n, 'seed*.json')))]
        if not res:
            continue
        t = 100 * np.array([r['test'] for r in res])
        sc = np.array([r['spectral_change'] for r in res if r['spectral_change'] is not None])
        rows.append({'level': n, 'seeds': len(res), 'spectral_change': float(sc.mean()) if sc.size else '',
                     'test': round(t.mean(), 2), 'test_std': round(t.std(), 2)})
    with open(os.path.join(ROOT, 'results', f'dose_{args.dataset}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'== {args.dataset}')
    for r in rows:
        print(f"   {r['level']:<20} spectral change {r['spectral_change']!s:<24} test {r['test']:6.2f} ± {r['test_std']:.2f}")


if __name__ == '__main__':
    main()
