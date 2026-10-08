import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class MaskedReconstruction(nn.Module):
    """Learnable [MASK] token for node features and a GCN decoder trained with the scaled cosine error."""

    def __init__(self, in_dim, hid_dim, alpha=2.0):
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, in_dim))
        self.decoder = GCNConv(hid_dim, in_dim)
        self.alpha = alpha

    def mask(self, x, rate):
        m = torch.rand(x.shape[0], device=x.device) < rate
        x = x.clone()
        x[m] = self.token.to(x.dtype)
        return x, m

    def loss(self, h, edge_index, x, m):
        if not m.any():
            return h.new_zeros(())
        h = h.clone()
        h[m] = 0
        rec = self.decoder(h, edge_index)
        return ((1 - F.cosine_similarity(rec[m], x[m], dim=-1)) ** self.alpha).mean()


def variance_covariance(h, eps=1e-4):
    h = h - h.mean(0)
    std = torch.sqrt(h.var(0) + eps)
    var = F.relu(1 - std).mean()
    cov = (h.T @ h) / (h.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var, (off ** 2).sum() / h.shape[1]


class FrequencyFilter:
    """Low- or high-pass filtering of node signals with the GCN propagation matrix P: low = P z, high = z - P z."""

    def __init__(self, edge_index, num_nodes, mode):
        self.mode = mode
        if mode != 'none':
            ei, w = gcn_norm(edge_index, num_nodes=num_nodes, add_self_loops=True)
            self.P = torch.sparse_coo_tensor(ei, w, (num_nodes, num_nodes)).coalesce()

    def __call__(self, z):
        if self.mode == 'none':
            return z
        pz = torch.sparse.mm(self.P, z)
        return pz if self.mode == 'low' else z - pz


def cca_loss(h1, h2, lam):
    n, d = h1.shape
    z1 = (h1 - h1.mean(0)) / h1.std(0).clamp_min(1e-8)
    z2 = (h2 - h2.mean(0)) / h2.std(0).clamp_min(1e-8)
    eye = torch.eye(d, device=h1.device)
    inv = -torch.diagonal(z1.T @ z2 / n).sum()
    dec = ((eye - z1.T @ z1 / n) ** 2).sum() + ((eye - z2.T @ z2 / n) ** 2).sum()
    return inv + lam * dec


class LatentPredictor(nn.Module):
    """JEPA-style predictor of the target embeddings of masked nodes, optionally given spectral positions."""

    def __init__(self, dim, pe=None, hidden=512):
        super().__init__()
        self.register_buffer('pe', pe if pe is not None else torch.zeros(0, 0))
        in_dim = dim + (pe.shape[1] if pe is not None else 0)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.PReLU(), nn.Linear(hidden, dim))

    def loss(self, h, y, m):
        if y is None or not m.any():
            return h.new_zeros(())
        z = h[m]
        if self.pe.numel():
            sign = (torch.randint(0, 2, (1, self.pe.shape[1]), device=h.device) * 2 - 1).to(h.dtype)
            z = torch.cat([z, self.pe[m] * sign], dim=-1)
        return 1 - F.cosine_similarity(self.net(z), y[m], dim=-1).mean()


def laplacian_pe(edge_index, num_nodes, k):
    """k eigenvectors of the normalised adjacency with the largest eigenvalues, skipping the trivial one."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    ei, w = gcn_norm(edge_index, num_nodes=num_nodes, add_self_loops=False)
    A = sp.csr_matrix((w.cpu().numpy(), (ei[0].cpu().numpy(), ei[1].cpu().numpy())), shape=(num_nodes, num_nodes))
    _, vec = eigsh(A, k=k + 1, which='LA', tol=1e-5)
    return torch.as_tensor(np.ascontiguousarray(vec[:, ::-1][:, 1:]), dtype=torch.float32)
