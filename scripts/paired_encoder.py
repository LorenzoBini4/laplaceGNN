"""Paired effect of an encoder variant (dual-frequency, closed gate, GraphSAGE, MLP): for every trial index, test(variant) - test(plain GCN) with the same
sampled hyperparameters. Output: results/dualfreq_vs_homophily.csv"""
import csv
import glob
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = {r['dataset']: float(r['adjusted_homophily']) for r in csv.DictReader(open(os.path.join(ROOT, 'results', 'raw_and_homophily.csv')))}


def trials(d, m):
    T = [json.loads(l) for f in glob.glob(os.path.join(ROOT, 'runs', 'tune', d, m, 'trials-shard*.jsonl')) for l in open(f)]
    return {t['trial']: t for t in T if 'error' not in t}


rows = []
for d in H:
    for base in ('bgrl', 'ccassg'):
        a = trials(d, base)
        for var in ('dualfreq', 'dualfreq_gate', 'sage', 'mlp'):
            b = trials(d, f'{base}_{var}')
            common = sorted(set(a) & set(b))
            if len(common) < 5:
                continue
            diff = 100 * np.array([b[i]['test'] - a[i]['test'] for i in common])
            rows.append({'dataset': d, 'adjusted_homophily': H[d], 'base': base, 'variant': var, 'trials': len(common),
                         'mean_delta': round(diff.mean(), 2), 'sem': round(diff.std(ddof=1) / np.sqrt(len(diff)), 2),
                         'frac_better': round(float((diff > 0).mean()), 2)})
with open(os.path.join(ROOT, 'results', 'dualfreq_vs_homophily.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
for base in ('bgrl', 'ccassg'):
    for var in ('dualfreq', 'dualfreq_gate', 'sage', 'mlp'):
        print(f'\n{base} + {var}   (paired mean Δ over trials, ± s.e.m.; fraction of trials where better)')
        for r in sorted((r for r in rows if r['base'] == base and r['variant'] == var), key=lambda r: r['adjusted_homophily']):
            print(f"   {r['dataset']:<15} homophily {r['adjusted_homophily']:+.2f}  n={r['trials']:2d}  Δ {r['mean_delta']:+6.2f} ± {r['sem']:.2f}   {r['frac_better']:.0%}")
