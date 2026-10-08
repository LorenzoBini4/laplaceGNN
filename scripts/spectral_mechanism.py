"""Relaxed vs sampled spectral change of views at a fixed flip budget (no training).

For each dataset: our generator (max and min objective), the original dense SPAN-style module
(graphs <= 12k nodes) and random edge dropping at the same budget. Relaxed = spectrum of the expected weighted graph a + (1 - 2a) p;
sampled = mean over Bernoulli samples of the sampled graph's spectrum. Output: results/spectral_mechanism.csv
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from laplacian_augmentations.spectral_views import SpectralViewGenerator, extremal_spectrum, lower_pairs
from laplacian_augmentations.view_sampler import SpectralViewSampler
from laplacian_augmentations.laplacian_node import LaplaceGNN_Augmentation_Node
from torch_geometric.data import Data


def load(name):
    from laplaceGNN.data import get_cora, get_citeseer, get_pubmed, get_dataset, get_heterophilous
    if name in ('cora', 'citeseer', 'pubmed'):
        return {'cora': get_cora, 'citeseer': get_citeseer, 'pubmed': get_pubmed}[name]('./data/datasets')[0][0]
    if name in ('amazon-photos',):
        return get_dataset('./data', name)[0]
    return get_heterophilous('./data/datasets', {'roman-empire': 'Roman-empire', 'amazon-ratings': 'Amazon-ratings'}[name])[0][0]


def distance(lam, lam0):
    m = min(lam.numel(), lam0.numel())
    return float(((lam[:m] - lam0[:m]) ** 2).sum() / (lam0[:m] ** 2).sum())


def relaxed_change(sampler, n, k, lam0):
    keys = sampler.row * n + sampler.col
    a = torch.isin(keys, sampler.orig_keys).double()
    fixed = sampler.orig_keys[~torch.isin(sampler.orig_keys, keys)]
    r = torch.cat([fixed // n, sampler.row]).cpu()
    c = torch.cat([fixed % n, sampler.col]).cpu()
    w = torch.cat([torch.ones(fixed.numel(), dtype=torch.float64), (a + (1 - 2 * a) * sampler.prob.double()).cpu()])
    lam, _ = extremal_spectrum(r, c, w, n, k)
    return distance(lam, lam0)


def sampled_change(sample_fn, n, k, lam0, samples):
    out = []
    for _ in range(samples):
        ei = sample_fn()
        r, c = lower_pairs(ei.cpu(), n)
        lam, _ = extremal_spectrum(r, c, torch.ones(r.numel(), dtype=torch.float64), n, k)
        out.append(distance(lam, lam0))
    return float(np.mean(out)), float(np.std(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', default='cora,citeseer,pubmed,amazon-photos,roman-empire,amazon-ratings')
    p.add_argument('--budget', type=float, default=0.2)
    p.add_argument('--k', type=int, default=32)
    p.add_argument('--samples', type=int, default=10)
    args = p.parse_args()
    dev = torch.device('cuda')
    rows = []
    for name in args.datasets.split(','):
        data = load(name)
        n = data.num_nodes
        er, ec = lower_pairs(data.edge_index, n)
        lam0, _ = extremal_spectrum(er, ec, torch.ones(er.numel(), dtype=torch.float64), n, args.k)
        E = er.numel()
        gen = SpectralViewGenerator(budget_ratio=args.budget, k=args.k, iters=20, device=dev)
        for sign, label in ((1, 'ours-max'), (-1, 'ours-min')):
            t0 = time.time()
            row, col, prob, _ = gen.generate(data.edge_index, n, sign)
            sampler = SpectralViewSampler(data.edge_index, n, row, col, prob, dev)
            rel = relaxed_change(sampler, n, args.k, lam0)
            smp, sd = sampled_change(lambda: sampler.sample()[0], n, args.k, lam0, args.samples)
            rows.append({'dataset': name, 'views': label, 'expected_flips': float(prob.sum()), 'relaxed_change': rel,
                         'sampled_change': smp, 'sampled_std': sd, 'seconds': time.time() - t0})
        if n <= 12000:
            for dis, lr, label in (('max', 100.0, 'span-style-max'), ('min', 0.1, 'span-style-min')):
                t0 = time.time()
                aug = LaplaceGNN_Augmentation_Node(ratio=args.budget, lr=lr, iteration=20, dis_type=dis, device=dev,
                                                   centrality_types=['degree', 'pagerank', 'eigenvector'],
                                                   centrality_weights=[0.2, 0.3, 0.5], store='compact', min_prob=1e-7)
                d = aug.calc_prob(Data(x=data.x, edge_index=data.edge_index, num_nodes=n), silence=True)
                sampler = SpectralViewSampler(data.edge_index, n, d[f'{dis}_row'], d[f'{dis}_col'], d[f'{dis}_prob'], dev)
                rel = relaxed_change(sampler, n, args.k, lam0)
                smp, sd = sampled_change(lambda: sampler.sample()[0], n, args.k, lam0, args.samples)
                rows.append({'dataset': name, 'views': label, 'expected_flips': float(d[f'{dis}_prob'].sum()),
                             'relaxed_change': rel, 'sampled_change': smp, 'sampled_std': sd, 'seconds': time.time() - t0})
                torch.cuda.empty_cache()
        drop = torch.full((er.numel(),), min(1.0, args.budget), device=dev)
        rnd = SpectralViewSampler(data.edge_index, n, er.to(dev), ec.to(dev), drop, dev)
        rel = relaxed_change(rnd, n, args.k, lam0)
        smp, sd = sampled_change(lambda: rnd.sample()[0], n, args.k, lam0, args.samples)
        rows.append({'dataset': name, 'views': 'random-drop', 'expected_flips': float(drop.sum()), 'relaxed_change': rel,
                     'sampled_change': smp, 'sampled_std': sd, 'seconds': 0.0})
        for r in rows[-5:]:
            if r['dataset'] == name:
                print(f"{name:<14} {r['views']:<16} flips {r['expected_flips']:9.1f} relaxed {r['relaxed_change']:.2e} "
                      f"sampled {r['sampled_change']:.2e} ± {r['sampled_std']:.1e}", flush=True)
        print(f'{name:<14} |E| = {E}, budget {args.budget}', flush=True)
        os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
        with open(os.path.join(ROOT, 'results', 'spectral_mechanism.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
