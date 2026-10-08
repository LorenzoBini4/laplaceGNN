"""Per-epoch Bernoulli sampling of flipped-edge views, A' = A + (1 - I - 2A) * P, on edge lists."""
import torch
from torch_geometric.utils import to_undirected, remove_self_loops


class SpectralViewSampler:
    def __init__(self, edge_index, num_nodes, row, col, prob, device, edge_attr=None):
        assert torch.all(row > col), 'flip probabilities must be stored for the lower triangle (row > col)'
        self.n = num_nodes
        self.device = device
        edge_index = edge_index.to(device)
        if edge_attr is None:
            ei, _ = remove_self_loops(to_undirected(edge_index, num_nodes=num_nodes))
            lower = ei[0] > ei[1]
            self.orig_keys = torch.unique(ei[0][lower] * num_nodes + ei[1][lower])
            self.orig_attr = None
        else:
            edge_attr = edge_attr.to(device)
            lower = edge_index[0] > edge_index[1]
            assert int(lower.sum()) == int((edge_index[0] < edge_index[1]).sum()), 'edge_index must be symmetric'
            keys = edge_index[0][lower] * num_nodes + edge_index[1][lower]
            order = torch.argsort(keys)
            self.orig_keys, self.orig_attr = keys[order], edge_attr[lower][order]
        self.row, self.col, self.prob = row.to(device), col.to(device), prob.to(device).float()

    def expected_flips(self):
        return float(self.prob.sum())

    def apply_flips(self, flip):
        n = self.n
        keys = self.row[flip] * n + self.col[flip]
        in_orig = torch.isin(keys, self.orig_keys)
        keep = ~torch.isin(self.orig_keys, keys[in_orig])
        added = keys[~in_orig]
        new_keys = torch.cat([self.orig_keys[keep], added])
        r, c = new_keys // n, new_keys % n
        edge_index = torch.stack([torch.cat([r, c]), torch.cat([c, r])])
        edge_attr = None
        if self.orig_attr is not None:
            filler = torch.zeros(  # added edges get an all-zero attribute
                (added.numel(),) + self.orig_attr.shape[1:], dtype=self.orig_attr.dtype, device=self.device)
            attr = torch.cat([self.orig_attr[keep], filler])
            edge_attr = torch.cat([attr, attr])
        return edge_index, edge_attr, {'removed': int(in_orig.sum()), 'added': int(added.numel())}

    def sample(self, generator=None, scale=1.0):
        flip = torch.rand(self.prob.shape, device=self.device, generator=generator) < (self.prob * scale).clamp(max=1.0)
        return self.apply_flips(flip)


def sampler_from_data(data, dis_type, device):
    row, col, prob = data[f'{dis_type}_row'], data[f'{dis_type}_col'], data[f'{dis_type}_prob']
    return SpectralViewSampler(data.edge_index, data.num_nodes, row, col, prob, device)


def sampler_from_sparse(ptb_prob, edge_index, num_nodes, device, edge_attr=None, batch=None, ptr=None):
    # PyG stacks per-graph SparseTensors along rows only, so columns stay local and must be offset
    row, col, val = ptb_prob.coo()
    row, col = row.to(device), col.to(device)
    if ptb_prob.sizes()[1] != num_nodes:
        assert batch is not None and ptr is not None, 'batched probabilities need batch and ptr to offset columns'
        col = col + ptr.to(device)[batch.to(device)[row]]
    lower = row > col
    if val is None:
        val = torch.ones_like(row, dtype=torch.float)
    val = val.to(device)
    return SpectralViewSampler(edge_index, num_nodes, row[lower], col[lower], val[lower], device, edge_attr=edge_attr)


def expected_flips_per_graph(graphs, keys=('max', 'min')):
    out = {}
    for k in keys:
        total = 0.0
        for d in graphs:
            val = d[k].storage.value()
            total += float(val.sum()) / 2 if val is not None else 0.0
        out[k] = total / max(len(graphs), 1)
    return out


def warn_if_noop(stats):
    noop = all(v == 0 for v in stats.values())
    if noop:
        print('WARNING: all flip probabilities are zero; the spectral views equal the input graphs')
    return noop
