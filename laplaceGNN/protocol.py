"""Evaluation protocol: fixed splits, hyperparameters and checkpoints selected on validation, test read once."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize

C_GRID = 2.0 ** np.arange(-10, 11)


def _mask_to_idx(mask):
    return np.flatnonzero(np.asarray(mask))


def random_splits(num_nodes, n_splits, seed, train_ratio=0.1, val_ratio=0.1):
    rng = np.random.RandomState(seed)
    n_train, n_val = int(train_ratio * num_nodes), int(val_ratio * num_nodes)
    splits = []
    for _ in range(n_splits):
        perm = rng.permutation(num_nodes)
        splits.append({'train': perm[:n_train], 'val': perm[n_train:n_train + n_val], 'test': perm[n_train + n_val:]})
    return splits


def masks_to_splits(train_masks, val_masks, test_masks):
    train_masks, val_masks, test_masks = map(np.asarray, (train_masks, val_masks, test_masks))
    if train_masks.ndim == 1:
        train_masks, val_masks = train_masks[:, None], val_masks[:, None]
    splits = []
    for s in range(train_masks.shape[1]):
        test = test_masks[:, s] if test_masks.ndim == 2 else test_masks
        splits.append({'train': _mask_to_idx(train_masks[:, s]), 'val': _mask_to_idx(val_masks[:, s]), 'test': _mask_to_idx(test)})
    return splits


def index_split(train_idx, val_idx, test_idx):
    to_np = lambda a: np.asarray(a.cpu() if torch.is_tensor(a) else a).reshape(-1)
    return [{'train': to_np(train_idx), 'val': to_np(val_idx), 'test': to_np(test_idx)}]


def check_splits(splits, num_nodes):
    for s in splits:
        tr, va, te = (set(s[k].tolist()) for k in ('train', 'val', 'test'))
        assert not (tr & va) and not (tr & te) and not (va & te), 'train/val/test splits overlap'
        assert max(max(tr), max(va), max(te)) < num_nodes


class _TorchLogRegPath:
    """L2 logistic regression along the C grid (increasing C, warm-started), full-batch L-BFGS on GPU."""
    def __init__(self, num_features, num_classes, device):
        self.W = torch.zeros(num_features, num_classes, device=device, requires_grad=True)
        self.b = torch.zeros(num_classes, device=device, requires_grad=True)

    def fit(self, X, y, C, max_iter=100):
        lam = 1.0 / (C * X.shape[0])
        opt = torch.optim.LBFGS([self.W, self.b], lr=1, max_iter=max_iter, line_search_fn='strong_wolfe',
                                tolerance_grad=1e-6, tolerance_change=1e-9, history_size=20)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(X @ self.W + self.b, y) + 0.5 * lam * (self.W ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        return self

    @torch.no_grad()
    def predict(self, X):
        return (X @ self.W + self.b).argmax(1)


def _accuracy(pred, y):
    if torch.is_tensor(pred):
        return (pred == y).float().mean().item()
    return float((pred == y).mean())


def linear_probe(X, y, split, c_grid=C_GRID, backend='auto', device=None, metric='acc'):
    """C chosen on split['val']; split['test'] is only read for the chosen C. metric: 'acc' or 'auc' (binary)."""
    from sklearn.metrics import roc_auc_score
    if not np.isfinite(np.asarray(X.detach().cpu() if torch.is_tensor(X) else X)).all():
        raise ValueError('embeddings contain NaN/inf')
    if backend == 'auto':
        backend = 'torch' if len(split['train']) > 1000 else 'sklearn'
    if backend == 'torch':
        device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        Xt = F.normalize(torch.as_tensor(X, dtype=torch.float32, device=device), dim=1)
        yt = torch.as_tensor(y, dtype=torch.long, device=device)
        num_classes = int(yt.max().item()) + 1
        tr, va, te = (torch.as_tensor(split[k], dtype=torch.long, device=device) for k in ('train', 'val', 'test'))
        clf = _TorchLogRegPath(Xt.shape[1], num_classes, device)

        def score(idx):
            if metric == 'auc':
                with torch.no_grad():
                    p = torch.softmax(Xt[idx] @ clf.W + clf.b, dim=1)[:, 1]
                return float(roc_auc_score(yt[idx].cpu().numpy(), p.cpu().numpy()))
            return _accuracy(clf.predict(Xt[idx]), yt[idx])

        best = None
        for C in sorted(c_grid):
            clf.fit(Xt[tr], yt[tr], C)
            val = score(va)
            if best is None or val > best['val_acc']:
                best = {'val_acc': val, 'C': float(C), 'train_acc': score(tr), 'test_acc': score(te)}
        return best

    X = normalize(np.asarray(X), norm='l2')
    y = np.asarray(y)
    tr, va, te = split['train'], split['val'], split['test']

    def score(clf, idx):
        if metric == 'auc':
            return float(roc_auc_score(y[idx], clf.predict_proba(X[idx])[:, 1]))
        return _accuracy(clf.predict(X[idx]), y[idx])

    best = None
    for C in c_grid:
        if backend == 'sklearn':
            clf = LogisticRegression(C=C, max_iter=2000)
        elif backend == 'liblinear-ovr':
            clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=C))
        else:
            raise ValueError(f'unknown probe backend {backend}')
        clf.fit(X[tr], y[tr])
        val = score(clf, va)
        if best is None or val > best['val_acc']:
            best = {'val_acc': val, 'C': float(C), 'clf': clf}
    clf = best.pop('clf')
    best['train_acc'] = score(clf, tr)
    best['test_acc'] = score(clf, te)
    return best


def evaluate_node_embeddings(X, y, splits, backend='auto', c_grid=C_GRID, metric='acc'):
    per_split = [linear_probe(X, y, s, c_grid=c_grid, backend=backend, metric=metric) for s in splits]
    val = np.array([r['val_acc'] for r in per_split])
    test = np.array([r['test_acc'] for r in per_split])
    return {'val_mean': float(val.mean()), 'test_mean': float(test.mean()), 'test_std': float(test.std()),
            'per_split': per_split}


class ValSelector:
    """Keeps the evaluation with the best validation score."""
    def __init__(self, key='val_mean'):
        self.key = key
        self.best = None
        self.last = None
        self.history = []

    def update(self, epoch, result):
        record = dict(result, epoch=epoch)
        self.history.append({'epoch': epoch, self.key: result[self.key]})
        self.last = record
        if self.best is None or result[self.key] > self.best[self.key]:
            self.best = record
        return record is self.best

    def summary(self):
        return {'selection': f'checkpoint with best {self.key}', 'selected': self.best, 'last': self.last,
                'val_history': self.history}


def tu_kfold_eval(X, y, seed, n_folds=10, inner_val=0.1, backend='sklearn', c_grid=C_GRID):
    """Stratified k-fold CV; C chosen on an inner validation split of each training fold, then refit."""
    X, y = np.asarray(X), np.asarray(y)
    outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for k, (train_full, test) in enumerate(outer.split(X, y)):
        inner = StratifiedShuffleSplit(n_splits=1, test_size=inner_val, random_state=seed + k)
        tr_rel, va_rel = next(inner.split(X[train_full], y[train_full]))
        sel = linear_probe(X, y, {'train': train_full[tr_rel], 'val': train_full[va_rel], 'test': test},
                           c_grid=c_grid, backend=backend)
        refit = linear_probe(X, y, {'train': train_full, 'val': train_full[va_rel], 'test': test},
                             c_grid=[sel['C']], backend=backend)
        folds.append({'fold': k, 'C': sel['C'], 'inner_val_acc': sel['val_acc'], 'test_acc': refit['test_acc']})
    test = np.array([f['test_acc'] for f in folds])
    return {'test_mean': float(test.mean()), 'test_std': float(test.std()), 'folds': folds,
            'protocol': f'stratified {n_folds}-fold CV, C selected on {inner_val:.0%} inner validation split'}


def ogb_probe_eval(emb, y, split_idx, evaluator, metric, probe='linear', hidden=512, epochs=100,
                   lr=1e-3, eval_every=5, seed=0):
    """Fresh probe on frozen embeddings; probe epoch selected on validation."""
    torch.manual_seed(seed)
    device = emb.device
    num_tasks = y.shape[1]
    if probe == 'linear':
        head = torch.nn.Linear(emb.shape[1], num_tasks)
    elif probe == 'mlp':
        head = torch.nn.Sequential(torch.nn.Linear(emb.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, num_tasks))
    else:
        raise ValueError(probe)
    head = head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    tr, va, te = (torch.as_tensor(split_idx[k], device=device) for k in ('train', 'valid', 'test'))
    y = y.to(device).float()

    def score(idx):
        with torch.no_grad():
            pred = head(emb[idx])
        return evaluator.eval({'y_true': y[idx].cpu().numpy(), 'y_pred': pred.cpu().numpy()})[metric]

    best = {'val': -np.inf}
    for epoch in range(1, epochs + 1):
        head.train()
        opt.zero_grad()
        logits = head(emb[tr])
        labeled = ~torch.isnan(y[tr])
        loss = F.binary_cross_entropy_with_logits(logits[labeled], y[tr][labeled])
        loss.backward()
        opt.step()
        if epoch % eval_every == 0 or epoch == epochs:
            head.eval()
            val = score(va)
            if val > best['val']:
                best = {'val': val, 'test': score(te), 'train': score(tr), 'probe_epoch': epoch}
    return best
