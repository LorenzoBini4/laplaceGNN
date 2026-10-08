"""Mean +- std over seeds of every run under runs/, grouped by pipeline, dataset and hyperparameters."""
import argparse
import csv
import glob
import hashlib
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_KEYS = {'model_seed', 'seed', 'logdir', 'dataset_dir', 'delta_cache_dir', 'cache_dir', 'device', 'save_encoder'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default=os.path.join(ROOT, 'runs'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'results', 'summary.csv'))
    args = parser.parse_args()

    groups = defaultdict(list)
    for final in glob.glob(os.path.join(args.runs, '**', 'final.json'), recursive=True):
        run = os.path.dirname(final)
        cfg = json.load(open(os.path.join(run, 'config.json')))
        res = json.load(open(final))
        hp = {k: v for k, v in cfg['hparams'].items() if k not in SEED_KEYS}
        key = hashlib.sha1(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:8]
        groups[(cfg['pipeline'], res['dataset'], key)].append((res, cfg, run))

    rows = []
    for (pipeline, dataset, key), runs in sorted(groups.items()):
        tests = np.array([r['test_mean'] if 'test_mean' in r else r['test'] for r, _, _ in runs]) * 100
        vals = [r.get('val_mean', r.get('val')) for r, _, _ in runs]
        hp = runs[0][1]['hparams']
        rows.append({
            'pipeline': pipeline, 'dataset': dataset, 'config': key, 'n_seeds': len(runs),
            'metric': runs[0][0].get('metric', 'accuracy'),
            'test_mean': round(float(tests.mean()), 2), 'test_std': round(float(tests.std()), 2),
            'val_mean': round(float(np.mean(vals)) * 100, 2) if all(v is not None for v in vals) else '',
            'view_mode': hp.get('view_mode', ''), 'dirty_code': any(c['git']['dirty'] for _, c, _ in runs),
            'commits': ' '.join(sorted({(c['git']['commit'] or '')[:8] for _, c, _ in runs})),
            'runs': ' '.join(sorted(os.path.relpath(r, ROOT) for _, _, r in runs)),
        })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['pipeline'])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['dataset']:<18} {r['view_mode']:<16} {r['config']}  n={r['n_seeds']:<2}  {r['metric']} {r['test_mean']:.2f} +- {r['test_std']:.2f}")


if __name__ == '__main__':
    main()
