"""Leave-one-out ablation of the best tuned config: each active component switched off, several seeds.

    python scripts/ablate.py --dataset cora --method laplacegnn_full --seeds 3 --workers 4
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


def variants(cfg):
    v = {'full': {}}
    if cfg.get('sv_gamma', 0) > 0:
        v['no_centrality_preservation'] = {'sv_gamma': 0.0}
    if cfg.get('sadv_every', 0) > 0:
        v['no_structural_adversary'] = {'sadv_every': 0}
    if cfg.get('adv_m', 1) > 1:
        v['no_hidden_adversary'] = {'adv_m': 1}
        if cfg.get('adv_filter', 'none') != 'none':
            v['no_frequency_filter'] = {'adv_filter': 'none'}
    if cfg.get('recon_weight', 0) > 0 and cfg.get('mask_rate', 0) > 0:
        v['no_masked_reconstruction'] = {'recon_weight': 0.0}
    if cfg.get('latent_weight', 0) > 0 and cfg.get('mask_rate', 0) > 0:
        v['no_latent_prediction'] = {'latent_weight': 0.0}
        if cfg.get('latent_pe_dim', 0) > 0:
            v['no_spectral_positions'] = {'latent_pe_dim': 0}
    if cfg.get('var_weight', 0) > 0 or cfg.get('cov_weight', 0) > 0:
        v['no_variance_covariance'] = {'var_weight': 0.0, 'cov_weight': 0.0}
    if cfg.get('sv_curriculum', 1.0) < 1.0:
        v['no_curriculum'] = {'sv_curriculum': 1.0}
    if cfg.get('encoder_type') == 'dualfreq':
        v['gcn_encoder'] = {'encoder_type': 'gcn'}
    if cfg.get('objective') == 'byol+cca':
        v['byol_only'] = {'objective': 'byol'}
        v['cca_only'] = {'objective': 'cca'}
    if cfg.get('view_pair') != 'max_min':
        v['pair_max_min'] = {'view_pair': 'max_min'}
    budget = cfg.get('sv_budget_ratio', 0.2)
    v['random_views_same_budget'] = {'view_mode': 'random', 'drop_edge_p_1': budget, 'drop_edge_p_2': budget,
                                     'drop_feat_p_1': cfg.get('prob_feat', 0.0), 'drop_feat_p_2': cfg.get('prob_feat', 0.0),
                                     'sadv_every': 0}
    return v


def best_config(dataset, method):
    """Best validation trial from runs/tune; falls back to results/tuning_summary_pre_deletion.csv, regenerating the
    config from its trial index (configs are sampled deterministically with random.Random(0))."""
    from summarize_tuning import capped
    trials = capped(dataset, [json.loads(l) for f in glob.glob(os.path.join(ROOT, 'runs', 'tune', dataset, method, 'trials-shard*.jsonl')) for l in open(f)])
    ok = [t for t in trials if 'error' not in t]
    snap = [r for r in csv.DictReader(open(os.path.join(ROOT, 'results', 'tuning_summary_pre_deletion.csv')))
            if r['dataset'] == dataset and r['method'] == method]
    if ok and (not snap or len(ok) >= int(snap[0]['trials'])):
        return max(ok, key=lambda t: t['val'])
    import random
    import tune_node as T
    for r in snap:
        if True:
            i = int(r['best_trial'])
            rng = random.Random(0)
            configs = [T.sample(T.METHODS[method][1], rng) for _ in range(i + 1)]
            return {'trial': i, 'config': configs[i], 'val': float(r['val']) / 100, 'test': float(r['test']) / 100}
    raise KeyError(f'no tuning record for {dataset}/{method}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--method', default='laplacegnn_full')
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--job', default=None)
    args = p.parse_args()
    import tune_node as T
    fixed, _ = T.METHODS[args.method]
    best = best_config(args.dataset, args.method)
    cfg = best['config']
    out_root = os.path.join(ROOT, 'runs', 'ablation', args.dataset)
    vs = variants(cfg)

    if args.job is not None:
        name, seed = args.job.rsplit(':', 1)
        res_file = os.path.join(out_root, name, f'seed{seed}.json')
        if os.path.exists(res_file):
            return
        override = dict(vs[name])
        run_fixed = {**fixed, **{k: override.pop(k) for k in list(override) if k in ('view_mode',)}}
        res = T.run_trial(args.dataset, run_fixed, {**cfg, **override}, os.path.join(out_root, name), int(seed), 50)
        json.dump({'val': res['val_mean'], 'test': res['test_mean']}, open(res_file, 'w'))
        return

    jobs = [f'{n}:{s}' for n in vs for s in range(args.seeds)]
    for n in vs:
        os.makedirs(os.path.join(out_root, n), exist_ok=True)
    running = []
    for job in jobs:
        log = open(os.path.join(out_root, job.replace(':', '-seed') + '.log'), 'w')
        running.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), '--dataset', args.dataset,
                                         '--method', args.method, '--job', job], stdout=log, stderr=subprocess.STDOUT))
        if len(running) >= args.workers:
            running.pop(0).wait()
    for pr in running:
        pr.wait()

    rows, full = [], None
    for n in vs:
        res = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out_root, n, 'seed*.json')))]
        if not res:
            continue
        t = 100 * np.array([r['test'] for r in res])
        v = 100 * np.array([r['val'] for r in res])
        rows.append({'variant': n, 'seeds': len(res), 'val': round(v.mean(), 2), 'test': round(t.mean(), 2), 'test_std': round(t.std(), 2)})
        if n == 'full':
            full = rows[-1]
    for r in rows:
        r['delta_test'] = round(r['test'] - full['test'], 2) if full else ''
    with open(os.path.join(ROOT, 'results', f'ablation_{args.dataset}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'== {args.dataset} (best {args.method} trial {best["trial"]})')
    for r in rows:
        print(f"   {r['variant']:<28} test {r['test']:6.2f} ± {r['test_std']:.2f}  delta {r['delta_test']:+}")


if __name__ == '__main__':
    main()
