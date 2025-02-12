import copy
import torch
import torch_geometric
import torch_geometric.nn as pyg_nn  
from torch_geometric.data import Data

class LaplaceGNN_Graph(torch.nn.Module):
    r""" LaplaceGNN: LaplaceGNN: Scalable Graph Learning through Spectral Bootstrapping and Adversarial Training
    """
    def __init__(self, encoder, predictor, augmentation):
        super().__init__()

        # Encoders and augmentation
        self.online_encoder = encoder
        self.predictor = predictor
        self.augmentation = augmentation

        # Target networks
        self.target_encoder = copy.deepcopy(encoder)
        self.target_encoder.reset_parameters()
        # Stop gradients for target encoders
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())        

    @staticmethod
    def corruption(x, edge_index, edge_weight=None):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    @torch.no_grad()
    def update_target_network(self, mm, centering=True, sharpening=True, center_momentum=0.9, temperature=0.04):
        r"""Performs a momentum update of the target network's weights, with optional centering and sharpening.
        Args:
            mm (float): Momentum used in moving average update.
            centering (bool): Whether to apply centering to the target network outputs.
            sharpening (bool): Whether to apply sharpening to the target network outputs.
            center_momentum (float): Momentum used for updating the center.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        assert 0.0 <= center_momentum <= 1.0, "Center momentum needs to be between 0.0 and 1.0, got %.5f" % center_momentum

        # Momentum update of the target network's weights
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

        if centering:
            # Initialize center if not already done
            if not hasattr(self, 'center'):
                self.center = torch.zeros_like(next(self.target_encoder.parameters()))

            # Update center with exponential moving average
            with torch.no_grad():
                for param in self.target_encoder.parameters():
                    self.center.mul_(center_momentum).add_(param.data.mean(), alpha=1. - center_momentum)

        if sharpening:
            # Apply sharpening to the target network outputs
            # avoid collapse when the temperature is higher than 0.06
            # temperature = 0.04
            for param in self.target_encoder.parameters():
                # param.data = torch.nn.functional.softmax(param.data / temperature, dim=-1)
                param.data = param.data / (param.data.std() + 1e-6)

    def forward(self, data, perturb_first=None, perturb_last=None):
        x, edge_index = data.x, data.edge_index
        ptb_prob1 = data.max
        ptb_prob2 = data.min

        # Apply augmentations
        L1_view, L2_view = self.augmentation
        x1, edge_index1, _ = L1_view(x, edge_index, ptb_prob1, batch=data.batch)
        x2, edge_index2, _ = L2_view(x, edge_index, ptb_prob2, batch=data.batch)

        # Online encoder forward pass with perturbations
        online_y = self.online_encoder(x1, edge_index1, batch=data.batch, perturb_first=perturb_first, perturb_last=perturb_last)
        if isinstance(online_y, tuple):
            online_y = online_y[0]  # Ensure online_y is a tensor
        online_q = self.predictor(online_y)

        # x_corr, edge_index_corr,_ = self.corruption(x1, edge_index1, edge_weight=None)
        # online_y = self.online_encoder(x_corr, edge_index_corr, batch=data.batch, perturb_first=perturb_first, perturb_last=perturb_last)
        online_q = self.predictor(online_y)

          # Target encoder forward pass (no perturbations)
        with torch.no_grad():
            target_y = self.target_encoder(x2, edge_index2, batch=data.batch)
            if isinstance(target_y, tuple):
                target_y = target_y[0]  # Ensure target_y is a tensor
            target_y = target_y.detach()

        return online_q, target_y
      
