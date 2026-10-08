import torch

from laplacian_augmentations.spectral_views import project_budget


class StructuralAdversary:
    """Encoder-aware update of a view's flip probabilities: signed ascent on the bootstrap loss of the relaxed
    graph (w = a + (1 - 2a) * p), kept within +-radius of the spectral solution and at its expected flip count."""

    def __init__(self, sampler, radius, step):
        self.sampler, self.radius, self.step = sampler, radius, step
        n = sampler.n
        keys = sampler.row * n + sampler.col
        self.a = torch.isin(keys, sampler.orig_keys).float()
        fixed = sampler.orig_keys[~torch.isin(sampler.orig_keys, keys)]
        self.fixed_r, self.fixed_c = fixed // n, fixed % n
        self.base = sampler.prob.clone()
        self.budget = float(self.base.sum())
        self.zeta = torch.zeros_like(self.base)

    def relaxed(self, prob):
        w = torch.cat([torch.ones_like(self.fixed_r, dtype=prob.dtype), self.a + (1 - 2 * self.a) * prob])
        r = torch.cat([self.fixed_r, self.sampler.row])
        c = torch.cat([self.fixed_c, self.sampler.col])
        return torch.stack([torch.cat([r, c]), torch.cat([c, r])]), torch.cat([w, w])

    def update(self, loss_fn):
        prob = self.sampler.prob.clone().requires_grad_()
        edge_index, edge_weight = self.relaxed(prob)
        grad = torch.autograd.grad(loss_fn(edge_index, edge_weight), prob)[0]
        self.zeta = (self.zeta + self.step * torch.sign(grad)).clamp(-self.radius, self.radius)
        self.sampler.prob = project_budget((self.base + self.zeta).double(), self.budget).float()
        return {'mean_abs_shift': float((self.sampler.prob - self.base).abs().mean())}
