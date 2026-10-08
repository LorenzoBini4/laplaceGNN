"""Poisoned graphs for the robustness study: random insertion, DICE and PR-BCD (Geisler et al., 2021) at several budgets.

PR-BCD attacks a 2-layer GCN surrogate trained on the training labels, with self-training labels for the other nodes;
its perturbed graph is then used as poisoned input for self-supervised pre-training (transfer of evasion perturbations).
Output: data/attacks/<dataset>/<attack>-<budget>.pt (undirected edge_index).
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce
from laplaceGNN.data import get_cora, get_citeseer, get_pubmed


class GCN(torch.nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.c1, self.c2 = GCNConv(i, h), GCNConv(h, o)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(F.relu(self.c1(x, edge_index, edge_weight)), 0.5, self.training)
        return self.c2(x, edge_index, edge_weight)


def undirected(ei, n):
    ei, _ = remove_self_loops(to_undirected(ei, num_nodes=n))
    return coalesce(ei, num_nodes=n)


def random_insert(ei, n, k, g):
    new = torch.randint(0, n, (2, 3 * k), generator=g)
    new = new[:, new[0] != new[1]][:, :k]
    return undirected(torch.cat([ei, new], 1), n)


def dice(ei, y, n, k, g):
    r, c = ei[:, ei[0] < ei[1]]
    same = y[r] == y[c]
    internal = torch.nonzero(same).flatten()
    n_del = min(k // 2, internal.numel())
    drop = internal[torch.randperm(internal.numel(), generator=g)[:n_del]]
    keep = torch.ones(r.numel(), dtype=torch.bool)
    keep[drop] = False
    added = []
    while sum(a.shape[1] for a in added) < k - n_del:
        cand = torch.randint(0, n, (2, 4 * k), generator=g)
        added.append(cand[:, y[cand[0]] != y[cand[1]]])
    added = torch.cat(added, 1)[:, :k - n_del]
    return undirected(torch.cat([torch.stack([r[keep], c[keep]]), added], 1), n)


def prbcd(x, ei, y, train, n, k, dev):
    from torch_geometric.contrib.nn import PRBCDAttack
    model = GCN(x.shape[1], 64, int(y.max()) + 1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    x, ei, y = x.to(dev), ei.to(dev), y.to(dev)
    for _ in range(200):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x, ei)[train], y[train]).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pseudo = model(x, ei).argmax(1)
    pseudo[train] = y[train]
    attack = PRBCDAttack(model, block_size=250_000, lr=100)
    victims = torch.nonzero(~torch.isin(torch.arange(n, device=dev), train)).flatten()
    pert, _ = attack.attack(x, ei, pseudo, budget=k, idx_attack=victims)
    return undirected(pert.cpu(), n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', default='cora,citeseer')
    p.add_argument('--budgets', default='0.05,0.1,0.2')
    args = p.parse_args()
    dev = torch.device('cuda')
    loaders = {'cora': get_cora, 'citeseer': get_citeseer, 'pubmed': get_pubmed}
    for name in args.datasets.split(','):
        ds, tr, _, _ = loaders[name]('./data/datasets')
        data = ds[0]
        n = data.num_nodes
        ei = undirected(data.edge_index, n)
        E = ei.shape[1] // 2
        out = os.path.join(ROOT, 'data', 'attacks', name)
        os.makedirs(out, exist_ok=True)
        train = torch.as_tensor(tr).nonzero().flatten().to(dev)
        for b in (float(v) for v in args.budgets.split(',')):
            k = int(b * E)
            for attack in ('random', 'dice', 'prbcd'):
                path = os.path.join(out, f'{attack}-{b}.pt')
                if os.path.exists(path):
                    continue
                g = torch.Generator().manual_seed(0)
                if attack == 'random':
                    new = random_insert(ei, n, k, g)
                elif attack == 'dice':
                    new = dice(ei, data.y, n, k, g)
                else:
                    new = prbcd(data.x, ei, data.y, train, n, k, dev)
                torch.save(new, path)
                a = set(map(tuple, ei.T.tolist()))
                bset = set(map(tuple, new.T.tolist()))
                print(f'{name} {attack}-{b}: |E| {E} -> {new.shape[1] // 2}, flipped {len(a ^ bset) // 2} (target {k})', flush=True)


if __name__ == '__main__':
    main()
