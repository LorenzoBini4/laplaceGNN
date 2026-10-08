"""Peak GPU memory per stage of the sparse view generator on random graphs of growing size (average degree ~10).
Output: results/generator_memory_profile.csv"""
import collections
import csv
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from laplacian_augmentations.spectral_views import SpectralViewGenerator
from scripts.scaling import random_graph

rows = []
for n in (20000, 30000, 40000, 50000):
    ei = random_graph(n)
    gen = SpectralViewGenerator(budget_ratio=0.2, k=32, iters=3, gamma=1.0, device='cuda')
    gen.mem_log = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _, _, _, st = gen.generate(ei, n, 1)
    peak = collections.defaultdict(float)
    for stage, gb in gen.mem_log:
        peak[stage] = max(peak[stage], gb)
    for stage, gb in peak.items():
        rows.append({'nodes': n, 'edges': ei.shape[1] // 2, 'candidates': st['candidates'], 'stage': stage, 'peak_gb': round(gb, 3)})
    print(n, st['candidates'], dict((k, round(v, 2)) for k, v in peak.items()), flush=True)
with open(os.path.join(ROOT, 'results', 'generator_memory_profile.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
