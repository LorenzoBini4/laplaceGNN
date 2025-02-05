import copy
import torch
import torch_geometric
import torch_geometric.nn as pyg_nn  

class LaplaceGNN_v1(torch.nn.Module):
    r""" LaplaceGNN: LaplaceGNN: Scalable Graph Learning through Spectral Bootstrapping and Adversarial Training
    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    .. note::
        encoder must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """
    def __init__(self, encoder, predictor):
        super().__init__()
        # online network
        self.online_encoder = encoder
        self.predictor = predictor

        # target network
        self.target_encoder = copy.deepcopy(encoder)
        # reinitialize weights
        self.target_encoder.reset_parameters()
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

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

    def forward(self, online_x, target_x, perturb_first=None, perturb_last=None):

        online_y = self.online_encoder(online_x, perturb_first, perturb_last)  # Apply perturbations to the online encoder
        online_q = self.predictor(online_y)

        # forward pass for the target encoder (no perturbations)
        with torch.no_grad(): 
            target_y = self.target_encoder(target_x).detach()  # target encoder receives clean inputs

        return online_q, target_y

    def compute_zinc_representations(net, dataset, device, pooling='mean'):
        """
        Pre-computes the graph-level representations for the entire dataset.
        Args:
            net (torch.nn.Module): The encoder network.
            dataset (Dataset): The dataset to process.
            device (torch.device): Device to use for computation.
            pooling (str): Pooling method to apply to node-level representations ('mean' or 'max').
        """
        net.eval()
        reps = []
        labels = []
        pooling_layer = None
    
        # Define a pooling layer if needed
        if pooling == 'mean':
            pooling_layer = pyg_nn.global_mean_pool
        elif pooling == 'max':
            pooling_layer = pyg_nn.global_max_pool
    
        for data in dataset:
            data = data.to(device)
            with torch.no_grad():
                node_reps = net(data)  # Node-level representations
    
                # Apply pooling to get a graph-level representation
                if pooling_layer is not None:
                    graph_rep = pooling_layer(node_reps, data.batch)  # Apply pooling
                else:
                    raise ValueError("Unsupported pooling type. Use 'mean' or 'max'.")
    
                reps.append(graph_rep)
                labels.append(data.y.unsqueeze(1))  # Ensure labels are 2D for consistency
    
        # Concatenate all graph-level representations and labels
        reps = torch.cat(reps, dim=0)
        labels = torch.cat(labels, dim=0)
    
        return reps, labels

    def compute_representations(net, dataset, device):
        r"""Pre-computes the representations for the entire dataset.
        """
        net.eval()
        reps = []
        labels = []
    
        for data in dataset:
            # forward
            if isinstance(data, dict):
                data = {k: v.to(device) if hasattr(v, 'to') else v for k, v in data.items()}
                # data = data['valid']
                # print(data.get('x', data.get('node_feat')))
                # print(f"data: {data}")
            else:
                data = data.to(device)
            with torch.no_grad():
                reps.append(net(data))
                # print(data.x)
                labels.append(data.y)
        reps = torch.cat(reps, dim=0)
        labels = torch.cat(labels, dim=0)
        return [reps, labels]

class LaplaceGNN_v2(torch.nn.Module):
    r""" LaplaceGNN: LaplaceGNN: Scalable Graph Learning through Spectral Bootstrapping and Adversarial Training
    Args:
        encoder (torch.nn.Module): First encoder network for augmentations (from module.py).
        augmentor (tuple): Tuple containing two augmentation functions.
        hidden_dim (int): Dimensionality of the hidden representation.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    .. note::
        Encoder must have a `reset_parameters` method, as the weights of the target network will
        be initialized differently from the online network.
    """
    def __init__(self, encoder, predictor, augmentor):
        super().__init__()

        # Encoders and augmentor
        self.online_encoder = encoder
        self.predictor = predictor
        self.augmentor = augmentor

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
        # print(f'data.max = {data.max}')
        # print(f'data.min = {data.min}')
        ptb_prob1 = data.max
        ptb_prob2 = data.min

        # Apply augmentations
        aug1, aug2 = self.augmentor
        x1, edge_index1, _ = aug1(x, edge_index, ptb_prob1, batch=None)
        x2, edge_index2, _ = aug2(x, edge_index, ptb_prob2, batch=None)

        # Online encoder forward pass with perturbations
        # online_y = self.online_encoder(x1, edge_index1, perturb_first, perturb_last)
        x_corr, edge_index_corr,_ = self.corruption(x1, edge_index1, edge_weight=None)
        online_y = self.online_encoder(x_corr, edge_index_corr, perturb_first=perturb_first, perturb_last=perturb_last)
        online_q = self.predictor(online_y)

          # Target encoder forward pass (no perturbations)
        with torch.no_grad():
            target_y = self.target_encoder(x2, edge_index2).detach()

        return online_q, target_y
    
        return encoder.to(device)
    
    def compute_representations(net, dataset, device):
        r"""Pre-computes the representations for the entire dataset.
        """
        net.eval()
        reps = []
        labels = []

        for data in dataset:
            # forward
            if isinstance(data, dict):
                data = {k: v.to(device) if hasattr(v, 'to') else v for k, v in data.items()}
            else:
                data = data.to(device)
            with torch.no_grad():
                reps.append(net(data))
                labels.append(data.y)
        reps = torch.cat(reps, dim=0)
        labels = torch.cat(labels, dim=0)
        return [reps, labels]
