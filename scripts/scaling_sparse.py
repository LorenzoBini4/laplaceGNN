"""Sparse view generation: time and peak GPU memory measured in a fresh process per graph (avg degree ~10, plus ogbn-arXiv).
Output: results/scaling_sparse.csv"""
import csv
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SIZES = ['1000', '2000', '5000', '10000', '20000', '50000', '100000', 'ogbn-arxiv']


def one(size):
    import torch
    from laplacian_augmentations.spectral_views import SpectralViewGenerator
    if size == 'ogbn-arxiv':
        from laplaceGNN.data import get_ogbn_arxiv
        d = get_ogbn_arxiv('./data')[0][0]
        ei, n = d.edge_index, d.num_nodes
    else:
        from scripts.scaling import random_graph
        n = int(size)
        ei = random_graph(n)
    gen = SpectralViewGenerator(budget_ratio=0.2, k=32, iters=10, gamma=1.0, device='cuda')
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    _, _, _, st = gen.generate(ei, n, 1)
    torch.cuda.synchronize()
    print(json.dumps({'graph': size, 'nodes': n, 'edges': ei.shape[1] // 2, 'candidates': st['candidates'],
                      'seconds': time.time() - t0, 'peak_gb': torch.cuda.max_memory_allocated() / 2**30}))


def main():
    if len(sys.argv) > 2 and sys.argv[1] == '--one':
        one(sys.argv[2])
        return
    rows = []
    for size in SIZES:
        out = subprocess.run([sys.executable, os.path.abspath(__file__), '--one', size], cwd=ROOT, capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith('{')]
        rows.append(json.loads(line[-1]) if line else {'graph': size, 'status': 'failed: ' + out.stderr[-200:]})
        print(rows[-1], flush=True)
    with open(os.path.join(ROOT, 'results', 'scaling_sparse.csv'), 'w', newline='') as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
