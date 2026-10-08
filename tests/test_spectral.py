"""Run with `python tests/test_spectral.py`."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from laplacian_augmentations.spectral_views import (eigen_grad, project_budget, extremal_spectrum, build_candidates,
                                                    SpectralViewGenerator, lower_pairs)


def _dense_lsym(row, col, w, n):
    W = torch.zeros(n, n, dtype=w.dtype)
    W = W.index_put((row, col), w).index_put((col, row), w)
    dinv = W.sum(1).clamp_min(1e-12).rsqrt()
    return torch.eye(n, dtype=w.dtype) - dinv[:, None] * W * dinv[None, :]


def test_eigen_grad_matches_autograd():
    torch.manual_seed(0)
    n = 12
    row, col = torch.tril_indices(n, n, -1)
    base = (torch.rand(row.numel()) < 0.4).double()
    s = 1 - 2 * base
    delta = (torch.rand(row.numel(), dtype=torch.float64) * 0.3).requires_grad_()
    lam = torch.linalg.eigvalsh(_dense_lsym(row, col, base + s * delta, n))
    coef = torch.randn(n, dtype=torch.float64)
    auto = torch.autograd.grad((lam * coef).sum(), delta)[0]
    L = _dense_lsym(row, col, (base + s * delta).detach(), n)
    D = torch.diag(torch.eye(n, dtype=torch.float64) - L)
    W = torch.zeros(n, n, dtype=torch.float64).index_put((row, col), (base + s * delta).detach())
    W = W + W.T
    ev, u = torch.linalg.eigh(L)
    f = u * W.sum(1).rsqrt()[:, None]
    analytic = eigen_grad(ev, f, row, col, s, coef)
    assert torch.allclose(auto, analytic, atol=1e-8), (auto - analytic).abs().max()


def test_extremal_spectrum_matches_dense():
    torch.manual_seed(1)
    n = 60
    row, col = torch.tril_indices(n, n, -1)
    keep = torch.rand(row.numel()) < 0.15
    row, col = row[keep], col[keep]
    w = torch.rand(row.numel(), dtype=torch.float64) + 0.5
    lam, _ = extremal_spectrum(row, col, w, n, k=5)
    dense = torch.linalg.eigvalsh(_dense_lsym(row, col, w, n))
    ref = torch.cat([dense[:5], dense[-5:]])
    assert torch.allclose(lam, ref, atol=1e-6), (lam, ref)


def test_project_budget():
    x = torch.randn(1000, dtype=torch.float64) * 3
    p = project_budget(x, 123.0)
    assert p.min() >= 0 and p.max() <= 1 and abs(p.sum().item() - 123.0) < 1e-3


def test_generator_spends_budget_and_views_differ():
    torch.manual_seed(2)
    n = 200
    row, col = torch.tril_indices(n, n, -1)
    keep = torch.rand(row.numel()) < 0.05
    ei = torch.stack([torch.cat([row[keep], col[keep]]), torch.cat([col[keep], row[keep]])])
    E = int(keep.sum())
    gen = SpectralViewGenerator(budget_ratio=0.2, k=8, iters=10, device='cpu')
    r1, c1, p1, s1 = gen.generate(ei, n, sign=+1)
    r2, c2, p2, s2 = gen.generate(ei, n, sign=-1)
    for p, st in ((p1, s1), (p2, s2)):
        assert abs(p.sum().item() - 0.2 * E) < 0.01 * E
        assert torch.all(r1 > c1)
    assert s1['trace'][-1]['spectral'] > s2['trace'][-1]['spectral']


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f'PASS {name}')
        except Exception as exc:
            failed += 1
            print(f'FAIL {name}: {type(exc).__name__}: {exc}')
    print(f'{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
