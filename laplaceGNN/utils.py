import random
import numpy as np
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from typing import *
import os
import torch
import dgl
import random
import numpy as np

def set_random_seeds(random_seed=77):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def get_adj_tensor(edge_index):
    coo = to_scipy_sparse_matrix(edge_index)
    adj = coo.tocsr()
    if adj.shape[0] == 0 or adj.shape[1] == 0:
        return torch.LongTensor(adj.todense())
    else:
        adj = adj + adj.T
        adj = adj.tolil()
        adj[adj > 1] = 1
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()
        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph should be unweighted"
        adj = torch.LongTensor(adj.todense())
        return adj
    
def get_normalize_adj_tensor(adj, device='cuda:0'):
    device = torch.device(device if adj.is_cuda else "cpu")
    mx = adj + torch.eye(adj.shape[0]).to(device)
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    return mx

def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(device)
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def preprocess_adj(adj, device):
    """
    Preprocess adjacency matrix to improve numerical stability.
    Args:
        adj (torch.Tensor): Input adjacency matrix
        device (torch.device): Computation device
    Returns:
        torch.Tensor: Preprocessed adjacency matrix
    """
    # Symmetrize
    adj = (adj + adj.T) / 2.0
    # Add small diagonal for stability
    adj += torch.eye(adj.shape[0], device=device) * 1e-6
    # Normalize
    row_sum = adj.sum(dim=1)
    row_sum[row_sum == 0] = 1.0  # Prevent division by zero
    adj = adj / row_sum.unsqueeze(1)
    return adj

class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))