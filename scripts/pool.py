"""Pool of GPU workers over single tuning trials, interleaved by trial index across studies.

    python scripts/pool.py --workers 7 --study cora:bgrl_dualfreq_gate:20 --study tolokers:polygcl:10 ...
Each job runs exactly one trial (tune_node shard i of N), so studies advance evenly and resume after interruption.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def done_trials(dataset, method):
    files = glob.glob(os.path.join(ROOT, 'runs', 'tune', dataset, method, 'trials-shard*.jsonl'))
    return {json.loads(l)['trial'] for f in files for l in open(f)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=7)
    p.add_argument('--study', action='append', required=True, help='dataset:method:trials')
    args = p.parse_args()
    studies = [(d, m, int(n)) for d, m, n in (s.split(':') for s in args.study)]
    jobs = []
    for i in range(max(n for _, _, n in studies)):
        for d, m, n in studies:
            if i < n and i not in done_trials(d, m):
                jobs.append((d, m, n, i))
    print(f'{len(jobs)} trials to run', flush=True)
    os.makedirs(os.path.join(ROOT, 'logs', 'pool'), exist_ok=True)
    running = []
    for k, (d, m, n, i) in enumerate(jobs):
        while len(running) >= args.workers:
            running = [r for r in running if r.poll() is None]
            time.sleep(2)
        os.makedirs(os.path.join(ROOT, 'runs', 'tune', d, m), exist_ok=True)
        log = open(os.path.join(ROOT, 'logs', 'pool', f'{d}-{m}-{i}.log'), 'w')
        cmd = [sys.executable, os.path.join(ROOT, 'scripts', 'tune_node.py'), '--dataset', d, '--method', m,
               '--trials', str(n), '--workers', str(n), '--shard', str(i)]
        running.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT))
        if k % 20 == 0:
            print(f'{time.strftime("%H:%M")} launched {k + 1}/{len(jobs)}', flush=True)
    for r in running:
        r.wait()
    print('POOL_DONE', flush=True)


if __name__ == '__main__':
    main()
