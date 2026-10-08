"""Run with `python tests/test_phase0.py`."""
import os
import sys

import numpy as np
import torch
from torch_sparse import SparseTensor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from laplaceGNN import protocol
from laplacian_augmentations.view_sampler import SpectralViewSampler, sampler_from_sparse


def _random_graph(n, p, seed):
    g = torch.Generator().manual_seed(seed)
    upper = torch.triu(torch.rand(n, n, generator=g) < p, diagonal=1)
    A = (upper | upper.T).float()
    return A, A.nonzero().T.contiguous()


def _dense_modified_adj(A, P):
    n = A.shape[0]
    return (torch.ones_like(A) - torch.eye(n) - A - A) * P + A


def _to_dense(edge_index, n):
    A = torch.zeros(n, n)
    A[edge_index[0], edge_index[1]] = 1
    return A


def test_sampler_matches_dense_flip():
    n = 30
    A, edge_index = _random_graph(n, 0.2, seed=0)
    tril = torch.tril_indices(n, n, offset=-1)
    prob = torch.rand(tril.shape[1], generator=torch.Generator().manual_seed(1))
    sampler = SpectralViewSampler(edge_index, n, tril[0], tril[1], prob, device='cpu')
    flip = torch.rand(prob.shape, generator=torch.Generator().manual_seed(2)) < prob
    ei, attr, stats = sampler.apply_flips(flip)
    P = torch.zeros(n, n)
    P[tril[0][flip], tril[1][flip]] = 1
    P = P + P.T
    assert attr is None
    assert torch.equal(_to_dense(ei, n), _dense_modified_adj(A, P))
    assert stats['added'] + stats['removed'] == int(flip.sum())


def test_sampler_expected_flips():
    n = 40
    A, edge_index = _random_graph(n, 0.1, seed=3)
    tril = torch.tril_indices(n, n, offset=-1)
    prob = torch.full((tril.shape[1],), 0.05)
    sampler = SpectralViewSampler(edge_index, n, tril[0], tril[1], prob, device='cpu')
    torch.manual_seed(0)
    flips = [sum(sampler.sample()[2].values()) for _ in range(200)]
    assert abs(np.mean(flips) - sampler.expected_flips()) < 0.05 * sampler.expected_flips()


def test_sampler_keeps_edge_attributes():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = torch.tensor([[1, 0, 1], [1, 0, 1], [2, 1, 0], [2, 1, 0]])
    row, col = torch.tensor([2, 1, 3]), torch.tensor([0, 0, 2])
    sampler = SpectralViewSampler(edge_index, 4, row, col, torch.ones(3), device='cpu', edge_attr=edge_attr)
    ei, attr, stats = sampler.apply_flips(torch.tensor([True, True, False]))
    got = {(int(a), int(b)): tuple(v.tolist()) for a, b, v in zip(ei[0], ei[1], attr)}
    assert got == {(2, 1): (2, 1, 0), (1, 2): (2, 1, 0), (2, 0): (0, 0, 0), (0, 2): (0, 0, 0)}
    assert stats == {'removed': 1, 'added': 1}


def test_batched_sampler_maps_pairs_to_the_right_graph():
    from torch_geometric.data import Batch, Data
    sizes, graphs, expected = [3, 6, 5], [], set()
    for g, n in enumerate(sizes):
        P = torch.zeros(n, n)
        P[n - 1, n - 2] = P[n - 2, n - 1] = 1.0
        graphs.append(Data(x=torch.ones(n, 1), edge_index=torch.tensor([[0, 1], [1, 0]]), max=SparseTensor.from_dense(P)))
    batch = Batch.from_data_list(graphs)
    for g, n in enumerate(sizes):
        off = int(batch.ptr[g])
        expected |= {(off + 1, off), (off, off + 1), (off + n - 1, off + n - 2), (off + n - 2, off + n - 1)}
    sampler = sampler_from_sparse(batch.max, batch.edge_index, batch.num_nodes, 'cpu', batch=batch.batch, ptr=batch.ptr)
    ei, _, stats = sampler.sample()
    assert set(map(tuple, ei.T.tolist())) == expected
    assert torch.equal(batch.batch[ei[0]], batch.batch[ei[1]]), 'flips must not connect different graphs'


def test_random_splits_are_deterministic_and_disjoint():
    a = protocol.random_splits(1000, 5, seed=7)
    b = protocol.random_splits(1000, 5, seed=7)
    for sa, sb in zip(a, b):
        for k in ('train', 'val', 'test'):
            assert np.array_equal(sa[k], sb[k])
    protocol.check_splits(a, 1000)
    assert len(a[0]['train']) == 100 and len(a[0]['val']) == 100 and len(a[0]['test']) == 800


def test_masks_to_splits_1d_and_2d():
    n = 20
    tr, va, te = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    tr[:5], va[5:10], te[10:] = True, True, True
    assert len(protocol.masks_to_splits(tr, va, te)) == 1
    tr2, va2 = np.stack([tr, np.roll(tr, 1)], 1), np.stack([va, np.roll(va, 1)], 1)
    assert len(protocol.masks_to_splits(tr2, va2, te)) == 2


def _synthetic(n=600, d=16, k=3, seed=0):
    rng = np.random.RandomState(seed)
    centers = rng.randn(k, d) * 2
    y = rng.randint(k, size=n)
    X = centers[y] + rng.randn(n, d) * 2.5
    return X.astype(np.float32), y


def test_probe_backends_agree():
    X, y = _synthetic()
    split = protocol.random_splits(len(y), 1, seed=1, train_ratio=0.3, val_ratio=0.2)[0]
    r_torch = protocol.linear_probe(X, y, split, backend='torch', device=torch.device('cpu'))
    r_sk = protocol.linear_probe(X, y, split, backend='sklearn')
    assert abs(r_torch['test_acc'] - r_sk['test_acc']) < 0.02, (r_torch, r_sk)


def test_probe_selection_ignores_test_labels():
    X, y = _synthetic(seed=2)
    split = protocol.random_splits(len(y), 1, seed=3, train_ratio=0.3, val_ratio=0.2)[0]
    y_scrambled = y.copy()
    y_scrambled[split['test']] = np.random.RandomState(0).permutation(y[split['test']])
    a = protocol.linear_probe(X, y, split, backend='sklearn')
    b = protocol.linear_probe(X, y_scrambled, split, backend='sklearn')
    assert a['C'] == b['C'] and a['val_acc'] == b['val_acc']


def test_probe_rejects_nan():
    X, y = _synthetic()
    X[0, 0] = np.nan
    split = protocol.random_splits(len(y), 1, seed=1)[0]
    try:
        protocol.linear_probe(X, y, split, backend='sklearn')
    except ValueError:
        return
    raise AssertionError('NaN embeddings must be rejected')


def test_val_selector_uses_validation_only():
    sel = protocol.ValSelector('val_mean')
    sel.update(1, {'val_mean': 0.70, 'test_mean': 0.90})
    sel.update(2, {'val_mean': 0.80, 'test_mean': 0.60})
    sel.update(3, {'val_mean': 0.75, 'test_mean': 0.99})
    assert sel.best['epoch'] == 2 and sel.best['test_mean'] == 0.60
    assert sel.last['epoch'] == 3


def test_tu_kfold_covers_every_graph_once():
    X, y = _synthetic(n=300)
    res = protocol.tu_kfold_eval(X, y, seed=0, n_folds=10)
    assert len(res['folds']) == 10
    res2 = protocol.tu_kfold_eval(X, y, seed=0, n_folds=10)
    assert res['test_mean'] == res2['test_mean']
    from sklearn.model_selection import StratifiedKFold
    test_idx = np.concatenate([t for _, t in StratifiedKFold(10, shuffle=True, random_state=0).split(X, y)])
    assert sorted(test_idx.tolist()) == list(range(len(y)))


def test_planetoid_loader_has_no_nan():
    from laplaceGNN.data import get_cora
    root = os.path.join(ROOT, 'data', 'datasets')
    if not os.path.isdir(os.path.join(root, 'Cora')):
        print('  (skipped: Cora not downloaded)')
        return
    data = get_cora(root)[0][0]
    assert torch.isfinite(data.x).all()


def test_encoder_modes():
    from torch_geometric.data import Data
    from laplaceGNN.models import Encoder_Adversarial_GCN
    torch.manual_seed(0)
    data = Data(x=torch.randn(12, 5), edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]))
    legacy = Encoder_Adversarial_GCN([5, 8, 4], batchnorm=True, layernorm=False, forward_mode='legacy').eval()
    manual = legacy.model[3](legacy.model[0](data.x, data.edge_index), data.edge_index)
    assert torch.allclose(legacy(data), manual), 'legacy mode must stay a linear two-conv stack'
    full = Encoder_Adversarial_GCN([5, 8, 8, 4], batchnorm=True, layernorm=False, forward_mode='full').eval()
    assert full(data).shape == (12, 4)
    assert (full(data) != 0).any()


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
