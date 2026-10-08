"""Runs the best tuned config of each method on fresh seeds (1..S; seed 0 was used for tuning), optionally with
overrides (e.g. a poisoned graph). Resumable and parallel.

    python scripts/run_best.py --dataset cora --methods bgrl,ccassg --seeds 3 --workers 6 --tag clean
    python scripts/run_best.py --dataset cora --methods bgrl --tag dice-0.1 --set edge_index_file=data/attacks/cora/dice-0.1.pt
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


def parse_set(items):
    out = {}
    for item in items or []:
        k, v = item.split('=', 1)
        out[k] = v
    return out


def summarize(dataset, tag, methods):
    rows = []
    for m in methods:
        res = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, 'runs', 'best', tag, dataset, m, 'seed*.json')))]
        if not res:
            continue
        t = 100 * np.array([r['test'] for r in res])
        rows.append({'dataset': dataset, 'tag': tag, 'method': m, 'seeds': len(res), 'test': round(t.mean(), 2), 'test_std': round(t.std(), 2)})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--methods', required=True)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--tag', default='clean')
    p.add_argument('--set', action='append')
    p.add_argument('--job', default=None)
    args = p.parse_args()
    methods = args.methods.split(',')
    overrides = parse_set(args.set)

    if args.job is not None:
        import tune_node as T
        from ablate import best_config
        method, seed = args.job.rsplit(':', 1)
        out = os.path.join(ROOT, 'runs', 'best', args.tag, args.dataset, method)
        res_file = os.path.join(out, f'seed{seed}.json')
        if os.path.exists(res_file):
            return
        fixed, _ = T.METHODS[method]
        cfg = {**best_config(args.dataset, method)['config'], **overrides}
        res = T.run_with_oom_retry(lambda: T.run_trial(args.dataset, fixed, cfg, out, int(seed), 50))
        json.dump({'val': res['val_mean'], 'test': res['test_mean']}, open(res_file, 'w'))
        return

    jobs = [f'{m}:{s}' for s in range(1, args.seeds + 1) for m in methods]
    running = []
    for job in jobs:
        method = job.rsplit(':', 1)[0]
        out = os.path.join(ROOT, 'runs', 'best', args.tag, args.dataset, method)
        os.makedirs(out, exist_ok=True)
        if os.path.exists(os.path.join(out, f"seed{job.rsplit(':', 1)[1]}.json")):
            continue
        cmd = [sys.executable, os.path.abspath(__file__), '--dataset', args.dataset, '--methods', args.methods,
               '--tag', args.tag, '--job', job] + sum((['--set', s] for s in (args.set or [])), [])
        log = open(os.path.join(out, job.replace(':', '-seed') + '.log'), 'w')
        running.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        if len(running) >= args.workers:
            running.pop(0).wait()
    for pr in running:
        pr.wait()
    rows = summarize(args.dataset, args.tag, methods)
    path = os.path.join(ROOT, 'results', 'best_runs.csv')
    old = [r for r in csv.DictReader(open(path))] if os.path.exists(path) else []
    keep = [r for r in old if not (r['dataset'] == args.dataset and r['tag'] == args.tag and r['method'] in methods)]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'tag', 'method', 'seeds', 'test', 'test_std'])
        w.writeheader()
        w.writerows(keep + rows)
    for r in rows:
        print(f"{r['dataset']:<14} {r['tag']:<14} {r['method']:<20} {r['test']:6.2f} ± {r['test_std']:.2f} ({r['seeds']} seeds)", flush=True)


if __name__ == '__main__':
    main()
