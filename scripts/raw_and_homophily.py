"""Per dataset: edge homophily, adjusted homophily (Platonov et al., 2023) and linear probes on raw features and on
SGC features (A^2 X), under the same splits and probe as every method. Output: results/raw_and_homophily.csv
"""
import csv
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_scatter import scatter_add
from laplaceGNN import protocol
from laplaceGNN.data import get_cora, get_citeseer, get_pubmed, get_dataset, get_heterophilous

HET = {'roman-empire': 'Roman-empire', 'amazon-ratings': 'Amazon-ratings', 'minesweeper': 'Minesweeper',
       'tolokers': 'Tolokers', 'questions': 'Questions'}
AUC = ('minesweeper', 'tolokers', 'questions')


def load(name):
    if name in ('cora', 'citeseer', 'pubmed'):
        ds, tr, va, te = {'cora': get_cora, 'citeseer': get_citeseer, 'pubmed': get_pubmed}[name]('./data/datasets')
        return ds[0], protocol.masks_to_splits(tr, va, te)
    if name == 'amazon-photos':
        d = get_dataset('./data', name)[0]
        return d, protocol.random_splits(d.num_nodes, 10, 7)
    ds, tr, va, te = get_heterophilous('./data/datasets', HET[name])
    return ds[0], protocol.masks_to_splits(tr, va, te)


def homophily(edge_index, y):
    y = y.long()
    same = (y[edge_index[0]] == y[edge_index[1]]).float()
    h_edge = float(same.mean())
    k = int(y.max()) + 1
    deg = torch.bincount(edge_index[0], minlength=y.numel()).float()
    d_k = scatter_add(deg, y, dim=0, dim_size=k)
    total = deg.sum()
    expected = float(((d_k / total) ** 2).sum())
    return h_edge, (h_edge - expected) / (1 - expected)


def sgc(data, k=2):
    ei, w = gcn_norm(data.edge_index, num_nodes=data.num_nodes, add_self_loops=True)
    x = data.x
    for _ in range(k):
        x = scatter_add(w.unsqueeze(-1) * x[ei[0]], ei[1], dim=0, dim_size=data.num_nodes)
    return x


def write(rows):
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    with open(os.path.join(ROOT, 'results', 'raw_and_homophily.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    rows = []
    for name in ['cora', 'citeseer', 'pubmed', 'amazon-photos', 'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        data, splits = load(name)
        metric = 'auc' if name in AUC else 'acc'
        h_edge, h_adj = homophily(data.edge_index, data.y)
        raw = protocol.evaluate_node_embeddings(data.x.numpy(), data.y.numpy(), splits, metric=metric)
        prop = protocol.evaluate_node_embeddings(sgc(data).numpy(), data.y.numpy(), splits, metric=metric)
        rows.append({'dataset': name, 'nodes': data.num_nodes, 'edges': data.edge_index.shape[1] // 2, 'metric': metric,
                     'edge_homophily': round(h_edge, 4), 'adjusted_homophily': round(h_adj, 4),
                     'raw_features_test': round(100 * raw['test_mean'], 2), 'sgc_features_test': round(100 * prop['test_mean'], 2)})
        print(rows[-1], flush=True)
        write(rows)


if __name__ == '__main__':
    main()
