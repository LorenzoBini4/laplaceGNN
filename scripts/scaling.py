"""Time and peak GPU memory of view generation vs graph size: original dense module vs sparse generator.

Random graphs with average degree ~10, plus ogbn-arXiv. 10 optimisation iterations each. Output: results/scaling.csv
"""
import csv
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops
from laplacian_augmentations.spectral_views import SpectralViewGenerator
from laplacian_augmentations.laplacian_node import LaplaceGNN_Augmentation_Node


def random_graph(n, deg=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    ei = torch.randint(0, n, (2, n * deg // 2), generator=g)
    ei, _ = remove_self_loops(to_undirected(ei, num_nodes=n))
    return ei


def measure(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        fn()
        torch.cuda.synchronize()
        return time.time() - t0, torch.cuda.max_memory_allocated() / 2**30, 'ok'
    except RuntimeError as exc:
        return float('nan'), float('nan'), 'OOM' if 'out of memory' in str(exc) else f'error: {exc}'[:80]


def main():
    dev = torch.device('cuda')
    graphs = [(f'random-{n}', n, random_graph(n)) for n in (1000, 2000, 5000, 10000, 20000, 50000, 100000)]
    from laplaceGNN.data import get_ogbn_arxiv
    arxiv = get_ogbn_arxiv('./data')[0][0]
    graphs.append(('ogbn-arxiv', arxiv.num_nodes, arxiv.edge_index))
    rows, dense_failed = [], False
    for name, n, ei in graphs:
        gen = SpectralViewGenerator(budget_ratio=0.2, k=32, iters=10, gamma=1.0, device=dev)
        t, m, status = measure(lambda: gen.generate(ei, n, 1))
        rows.append({'graph': name, 'nodes': n, 'edges': ei.shape[1] // 2, 'method': 'sparse (ours)', 'seconds': t, 'peak_gb': m, 'status': status})
        print(rows[-1], flush=True)
        if dense_failed:
            rows.append({'graph': name, 'nodes': n, 'edges': ei.shape[1] // 2, 'method': 'dense (original)', 'seconds': float('nan'), 'peak_gb': float('nan'), 'status': 'skipped (OOM at smaller n)'})
            continue
        aug = LaplaceGNN_Augmentation_Node(ratio=0.2, lr=100.0, iteration=10, dis_type='max', device=dev,
                                           centrality_types=['degree', 'pagerank', 'eigenvector'], centrality_weights=[0.2, 0.3, 0.5],
                                           store='compact', min_prob=1e-7)
        t, m, status = measure(lambda: aug.calc_prob(Data(x=torch.zeros(n, 1), edge_index=ei, num_nodes=n), silence=True))
        dense_failed = status != 'ok'
        rows.append({'graph': name, 'nodes': n, 'edges': ei.shape[1] // 2, 'method': 'dense (original)', 'seconds': t, 'peak_gb': m, 'status': status})
        print(rows[-1], flush=True)
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    with open(os.path.join(ROOT, 'results', 'scaling.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
