"""Sparse centrality-preserving spectral views.

Relaxed graph: pair p=(i,j) in a candidate set has weight w_p = a_p + s_p * Delta_p, s_p = 1 - 2 a_p.
View objective (maximised):  sign * sum_k (lambda_k(Delta) - lambda_k(0))^2 / sum_k lambda_k(0)^2
                             - gamma * sum_i omega_i (c_i(Delta) - c_i(0))^2 / sum_i omega_i c_i(0)^2
subject to Delta in [0,1]^P and sum(Delta) = B (equal budget for both views).
lambda_k are the k smallest and k largest eigenvalues of the normalised Laplacian, with the exact
first-order gradient  d lambda / d w_ij = (f_i - f_j)^2 - lambda (f_i^2 + f_j^2),  f = D^{-1/2} u.
"""
import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import eigsh
from torch_scatter import scatter_add


def lower_pairs(edge_index, n):
    r, c = edge_index
    mask = r > c
    keys = torch.unique(r[mask] * n + c[mask])
    return keys // n, keys % n


def build_candidates(edge_index, n, centrality, new_ratio=1.0, hub_frac=0.05, seed=0):
    """Existing edges plus ~new_ratio*|E| absent pairs: half random 2-hop pairs, half pairs among hubs."""
    g = torch.Generator().manual_seed(seed)
    er, ec = lower_pairs(edge_index.cpu(), n)
    existing = er * n + ec
    n_new = int(new_ratio * existing.numel())

    src, dst = edge_index.cpu()
    order = torch.argsort(src)
    src, dst = src[order], dst[order]
    ptr = torch.zeros(n + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(torch.bincount(src, minlength=n), 0)
    deg = ptr[1:] - ptr[:-1]
    m = 4 * n_new
    start = torch.randint(0, n, (m,), generator=g)
    start = start[deg[start] > 0]
    mid = dst[ptr[start] + (torch.rand(start.numel(), generator=g) * deg[start]).long()]
    mid = mid[deg[mid] > 0]
    start = start[:mid.numel()]
    end = dst[ptr[mid] + (torch.rand(mid.numel(), generator=g) * deg[mid]).long()]
    two_hop = torch.stack([torch.maximum(start, end), torch.minimum(start, end)])

    k = max(2, int(hub_frac * n))
    hubs = torch.topk(centrality.cpu(), k).indices
    hi = hubs[torch.randint(0, k, (m,), generator=g)]
    hj = hubs[torch.randint(0, k, (m,), generator=g)]
    hub_pairs = torch.stack([torch.maximum(hi, hj), torch.minimum(hi, hj)])

    new = []
    for pairs in (two_hop, hub_pairs):
        pairs = pairs[:, pairs[0] > pairs[1]]
        keys = torch.unique(pairs[0] * n + pairs[1])
        keys = keys[~torch.isin(keys, existing)]
        keys = keys[torch.randperm(keys.numel(), generator=g)[:n_new // 2]]
        new.append(keys)
    new = torch.unique(torch.cat(new))
    keys = torch.cat([existing, new])
    base = torch.cat([torch.ones(existing.numel()), torch.zeros(new.numel())])
    return keys // n, keys % n, base


def _sym(row, col, w, n):
    return torch.cat([row, col]), torch.cat([col, row]), torch.cat([w, w])


def degree(row, col, w, n):
    r, _, ww = _sym(row, col, w, n)
    return scatter_add(ww, r, dim=0, dim_size=n)


def pagerank(row, col, w, n, alpha=0.85, iters=30):
    r, c, ww = _sym(row, col, w, n)
    deg = scatter_add(ww, r, dim=0, dim_size=n).clamp_min(1e-12)
    x = torch.full((n,), 1.0 / n, device=w.device, dtype=w.dtype)
    for _ in range(iters):
        x = (1 - alpha) / n + alpha * scatter_add(ww * x[c] / deg[c], r, dim=0, dim_size=n)
    return x


def _minmax(x):
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-12)


def eigenvector_centrality(row, col, w, n):
    r, c, ww = _sym(row, col, w.detach(), n)
    A = sp.csr_matrix((ww.cpu().numpy(), (r.cpu().numpy(), c.cpu().numpy())), shape=(n, n))
    _, vec = eigsh(A, k=1, which='LA')
    return torch.as_tensor(np.abs(vec[:, 0]), dtype=w.dtype, device=w.device)


def combined_centrality(row, col, w, n, types, weights):
    fns = {'degree': degree, 'pagerank': pagerank, 'eigenvector': eigenvector_centrality}
    return sum(a * _minmax(fns[t](row, col, w, n)) for t, a in zip(types, weights))


def extremal_spectrum(row, col, w, n, k):
    """k smallest and k largest eigenvalues of the normalised Laplacian, and f = D^{-1/2} u."""
    r, c, ww = _sym(row, col, w.detach().double(), n)
    deg = scatter_add(ww, r, dim=0, dim_size=n).clamp_min(1e-12)
    dinv = deg.rsqrt()
    vals = (dinv[r] * ww * dinv[c]).cpu().numpy()
    A = sp.csr_matrix((vals, (r.cpu().numpy(), c.cpu().numpy())), shape=(n, n))
    k = min(k, (n - 2) // 2)
    mu_hi, u_hi = eigsh(A, k=k, which='LA', tol=1e-6)
    mu_lo, u_lo = eigsh(A, k=k, which='SA', tol=1e-6)
    mu, u = np.concatenate([mu_hi, mu_lo]), np.concatenate([u_hi, u_lo], 1)
    order = np.argsort(-mu)
    lam = torch.as_tensor(1.0 - mu[order], device=w.device)
    f = torch.as_tensor(u[:, order], device=w.device) * dinv.unsqueeze(1)
    return lam, f


def eigen_grad(lam, f, row, col, sign_p, coef, chunk=200_000):
    """sum_k coef_k * d lambda_k / d Delta_p, exact to first order; computed in chunks of pairs to bound memory."""
    out = torch.empty(row.numel(), dtype=f.dtype, device=f.device)
    for a in range(0, row.numel(), chunk):
        fi, fj = f[row[a:a + chunk]], f[col[a:a + chunk]]
        out[a:a + chunk] = ((fi - fj) ** 2 - lam.unsqueeze(0) * (fi ** 2 + fj ** 2)) @ coef
    return sign_p * out


def project_budget(delta, budget, lo=0.0, hi=1.0, iters=60):
    """Euclidean projection onto {delta in [lo,hi]^P, sum = budget} by bisection on the shift."""
    a, b = (delta.min() - hi).item(), (delta.max() - lo).item()
    for _ in range(iters):
        tau = (a + b) / 2
        if (delta - tau).clamp(lo, hi).sum().item() > budget:
            a = tau
        else:
            b = tau
    return (delta - (a + b) / 2).clamp(lo, hi)


class SpectralViewGenerator:
    def __init__(self, budget_ratio=0.2, k=32, iters=20, step=0.2, gamma=1.0, protect='hubs',
                 init='centrality', centrality_types=('degree', 'pagerank', 'eigenvector'),
                 centrality_weights=(1 / 3, 1 / 3, 1 / 3), preserve_types=('degree', 'pagerank'),
                 new_ratio=1.0, hub_frac=0.05, seed=0, device='cuda'):
        self.__dict__.update(locals())
        del self.__dict__['self']

    def _pres(self, row, col, w, n, c0, omega):
        c = sum(_minmax(pagerank(row, col, w, n)) if t == 'pagerank' else _minmax(degree(row, col, w, n))
                for t in self.preserve_types)
        return (omega * (c - c0) ** 2).sum() / (omega * c0 ** 2).sum().clamp_min(1e-12)

    def _mark(self, stage):
        if getattr(self, 'mem_log', None) is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.mem_log.append((stage, torch.cuda.max_memory_allocated() / 2**30))
            torch.cuda.reset_peak_memory_stats()

    def generate(self, edge_index, n, sign):
        dev = torch.device(self.device)
        t0 = time.time()
        er, ec = lower_pairs(edge_index.cpu(), n)
        ones = torch.ones(er.numel(), device=dev, dtype=torch.float64)
        er, ec = er.to(dev), ec.to(dev)
        cent = combined_centrality(er, ec, ones, n, self.centrality_types, self.centrality_weights)
        self._mark('centrality')
        row, col, base = build_candidates(edge_index, n, cent, self.new_ratio, self.hub_frac, self.seed)
        row, col, base = row.to(dev), col.to(dev), base.to(dev, torch.float64)
        self._mark('candidates')
        s = 1 - 2 * base
        budget = self.budget_ratio * er.numel()

        if self.init == 'centrality':
            delta = cent[row] * cent[col]
        elif self.init == 'uniform':
            delta = torch.ones_like(base)
        else:
            delta = torch.rand(base.shape, device=dev, dtype=torch.float64,
                               generator=torch.Generator(device=dev).manual_seed(self.seed))
        delta = project_budget(delta, budget)

        lam0, _ = extremal_spectrum(row, col, base, n, self.k)
        self._mark('init+spectrum0')
        c0 = sum(_minmax(pagerank(row, col, base, n)) if t == 'pagerank' else _minmax(degree(row, col, base, n))
                 for t in self.preserve_types)
        omega = {'hubs': cent, 'periphery': 1 - cent, 'uniform': torch.ones_like(cent)}[self.protect]
        trace = []
        for t in range(1, self.iters + 1):
            w = base + s * delta
            lam, f = extremal_spectrum(row, col, w, n, self.k)
            m = min(lam.numel(), lam0.numel())
            spec = ((lam[:m] - lam0[:m]) ** 2).sum() / (lam0[:m] ** 2).sum()
            self._mark('spectrum')
            grad = sign * eigen_grad(lam[:m], f[:, :m], row, col, s, 2 * (lam[:m] - lam0[:m]) / (lam0[:m] ** 2).sum())
            pres = torch.zeros((), device=dev)
            self._mark('eigen_grad')
            if self.gamma > 0:
                d = delta.clone().requires_grad_()
                pres = self._pres(row, col, base + s * d, n, c0, omega)
                grad = grad - self.gamma * torch.autograd.grad(pres, d)[0]
            self._mark('preservation')
            delta = project_budget(delta + self.step / t ** 0.5 * grad / grad.abs().max().clamp_min(1e-12), budget)
            trace.append({'iter': t, 'spectral': float(spec), 'preserve': float(pres)})
        keep = delta > 1e-6
        stats = {'candidates': int(base.numel()), 'existing': int(base.sum()), 'budget': budget,
                 'expected_flips': float(delta.sum()), 'expected_removed': float((delta * base).sum()),
                 'expected_added': float((delta * (1 - base)).sum()), 'stored_pairs': int(keep.sum()),
                 'seconds': time.time() - t0, 'trace': trace}
        return row[keep].cpu(), col[keep].cpu(), delta[keep].float().cpu(), stats


def _dense_parts(A, mask):
    deg = A.sum(-1)
    dinv = torch.where(deg > 0, deg.clamp_min(1e-12).rsqrt(), torch.zeros_like(deg))
    L = mask.unsqueeze(-1) * mask.unsqueeze(-2) * (torch.eye(A.shape[-1], device=A.device, dtype=A.dtype) - dinv.unsqueeze(-1) * A * dinv.unsqueeze(-2))
    return deg, L


def _dense_pagerank(A, mask, alpha=0.85, iters=30):
    n = mask.sum(-1, keepdim=True).clamp_min(1)
    deg = A.sum(-1).clamp_min(1e-12)
    P = A / deg.unsqueeze(-2)
    x = mask / n
    for _ in range(iters):
        x = mask * ((1 - alpha) / n + alpha * (P @ x.unsqueeze(-1)).squeeze(-1))
    return x


def _dense_minmax(x, mask):
    big = torch.finfo(x.dtype).max
    lo = torch.where(mask > 0, x, torch.full_like(x, big)).min(-1, keepdim=True).values
    hi = torch.where(mask > 0, x, torch.full_like(x, -big)).max(-1, keepdim=True).values
    return mask * (x - lo) / (hi - lo).clamp_min(1e-12)


def _dense_project(delta, pair_mask, budget, iters=50):
    a = (delta.amin((-1, -2)) - 1).view(-1, 1, 1)
    b = delta.amax((-1, -2)).view(-1, 1, 1)
    for _ in range(iters):
        tau = (a + b) / 2
        over = ((delta - tau).clamp(0, 1) * pair_mask).sum((-1, -2)).view(-1, 1, 1) > budget.view(-1, 1, 1)
        a, b = torch.where(over, tau, a), torch.where(over, b, tau)
    return (delta - (a + b) / 2).clamp(0, 1) * pair_mask


class BatchedSpectralViewGenerator:
    """The same view objective for many small graphs at once: all pairs are candidates, exact batched eigvalsh."""

    def __init__(self, budget_ratio=0.2, iters=20, step=0.2, gamma=1.0, protect='hubs', init='centrality',
                 centrality_weights=(0.5, 0.5), seed=0, device='cuda', batch_graphs=512):
        self.__dict__.update(locals())
        del self.__dict__['self']

    def _centrality(self, A, mask):
        return sum(w * _dense_minmax(c, mask) for w, c in zip(self.centrality_weights, (A.sum(-1), _dense_pagerank(A, mask))))

    def _bucket(self, graphs, sign):
        dev = torch.device(self.device)
        nmax = max(g.num_nodes for g in graphs)
        B = len(graphs)
        A = torch.zeros(B, nmax, nmax, device=dev, dtype=torch.float64)
        mask = torch.zeros(B, nmax, device=dev, dtype=torch.float64)
        for b, g in enumerate(graphs):
            if g.edge_index.numel():
                A[b, g.edge_index[0], g.edge_index[1]] = 1.0
            mask[b, :g.num_nodes] = 1.0
        A = ((A + A.transpose(1, 2)) > 0).double()
        A = A * (1 - torch.eye(nmax, device=dev, dtype=A.dtype))
        pair_mask = torch.tril(mask.unsqueeze(-1) * mask.unsqueeze(-2), diagonal=-1)
        S = 1 - 2 * A
        budget = (self.budget_ratio * torch.tril(A, -1).sum((-1, -2))).clamp_min(1.0)

        cent = self._centrality(A, mask)
        if self.init == 'centrality':
            delta = cent.unsqueeze(-1) * cent.unsqueeze(-2) * pair_mask
        else:
            delta = pair_mask.clone()
        delta = _dense_project(delta, pair_mask, budget)
        _, L0 = _dense_parts(A, mask)
        lam0 = torch.linalg.eigvalsh(L0)
        norm0 = (lam0 ** 2).sum(-1).clamp_min(1e-12)
        c0 = _dense_minmax(A.sum(-1), mask) + _dense_minmax(_dense_pagerank(A, mask), mask)
        omega = {'hubs': cent, 'periphery': mask * (1 - cent), 'uniform': mask}[self.protect]
        for t in range(1, self.iters + 1):
            d = delta.clone().requires_grad_()
            sym = d + d.transpose(1, 2)
            W = A + S * sym
            _, L = _dense_parts(W, mask)
            lam = torch.linalg.eigvalsh(L)
            spec = ((lam - lam0) ** 2).sum(-1) / norm0
            obj = sign * spec
            if self.gamma > 0:
                c = _dense_minmax(W.sum(-1), mask) + _dense_minmax(_dense_pagerank(W, mask), mask)
                pres = (omega * (c - c0) ** 2).sum(-1) / (omega * c0 ** 2).sum(-1).clamp_min(1e-12)
                obj = obj - self.gamma * pres
            g = torch.autograd.grad(obj.sum(), d)[0] * pair_mask
            g = g / g.abs().amax((-1, -2), keepdim=True).clamp_min(1e-12)
            delta = _dense_project(delta + self.step / t ** 0.5 * g, pair_mask, budget)
        return delta, float(spec.mean()), budget

    def generate(self, graphs, sign):
        """Returns one symmetric SparseTensor of flip probabilities per graph, plus summary statistics."""
        from torch_sparse import SparseTensor
        order = sorted(range(len(graphs)), key=lambda i: graphs[i].num_nodes)
        out = [None] * len(graphs)
        t0, flips, spec_sum = time.time(), 0.0, 0.0
        for s in range(0, len(order), self.batch_graphs):
            idx = order[s:s + self.batch_graphs]
            delta, spec, budget = self._bucket([graphs[i] for i in idx], sign)
            flips += float(delta.sum())
            spec_sum += spec * len(idx)
            delta = (delta + delta.transpose(1, 2)).float().cpu()
            for b, i in enumerate(idx):
                n = graphs[i].num_nodes
                out[i] = SparseTensor.from_dense(delta[b, :n, :n])
        return out, {'graphs': len(graphs), 'expected_flips_per_graph': flips / len(graphs),
                     'spectral_change': spec_sum / len(graphs), 'seconds': time.time() - t0}
