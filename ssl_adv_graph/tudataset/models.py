import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, SAGEConv, Sequential
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn import GATConv

class Encoder_Adversarial_GCN(nn.Module):
    def __init__(self, layer_sizes, batchnorm=False, batchnorm_mm=0.99, layernorm=True, weight_standardization=False):
        super().__init__()
        assert batchnorm != layernorm
        assert len(layer_sizes) >= 2
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self.weight_standardization = weight_standardization
        layers = []
        self.gcn_layers = []  # store references to GCNConv layers for easy access
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            gcn_conv = GCNConv(in_dim, out_dim)
            self.gcn_layers.append(gcn_conv)
            layers.append((gcn_conv, 'x, edge_index -> x'))
            if batchnorm:
                layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            else:
                layers.append(LayerNorm(out_dim))
            layers.append(nn.PReLU())
        self.model = Sequential('x, edge_index', layers)

    def forward(self, x, edge_index, perturb_first=None, perturb_last=None):
        if self.weight_standardization:
            self.standardize_weights()

        # apply perturbations to the first hidden layer
        if perturb_first is not None:
            x = self.model[0](x, edge_index)
            x = x + perturb_first  # add adversarial perturbation to the first layer output
        else:
            x = self.model[0](x, edge_index)
        
        # x = torch.relu(x)  
        # pass through the second layer 
        x = self.model[3](x, edge_index)  
        # x = torch.relu(x)
        # pass through the third layer --> used for ogbn-arxiv only
        # x = self.model[6](x, edge_index)
        # x = torch.relu(x)
        # apply perturbations to the last hidden layer (the second convolution)
        if perturb_last is not None:
            x = x + perturb_last  # add adversarial perturbation to the last layer output

        return x

    def reset_parameters(self):
        self.model.reset_parameters()

    def standardize_weights(self):
        """
        Standardize weights across layers except for the first GCN layer.
        """
        skipped_first_conv = False
        for m in self.model.modules():
            if isinstance(m, GCNConv):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                weight = m.lin.weight.data
                var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                m.lin.weight.data = weight

class Encoder_Adversarial_GraphSAGE(nn.Module):
    def __init__(self, layer_sizes, batchnorm=False, batchnorm_mm=0.99, layernorm=True, weight_standardization=False):
        super().__init__()
        assert batchnorm != layernorm
        assert len(layer_sizes) >= 2
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self.weight_standardization = weight_standardization
        layers = []
        self.sage_layers = []  # store references to SAGEConv layers for easy access
        
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            sage_conv = SAGEConv(in_dim, out_dim)
            self.sage_layers.append(sage_conv)
            layers.append((sage_conv, 'x, edge_index -> x'))

            if batchnorm:
                layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            else:
                layers.append(LayerNorm(out_dim))

            layers.append(nn.PReLU())

        self.model = Sequential('x, edge_index', layers)

    def forward(self, data, perturb_first=None, perturb_last=None):
        if self.weight_standardization:
            self.standardize_weights()
        
        if isinstance(data, dict):
            x = data.get('x', data.get('node_feat'))
        else:
            x = data.x
        
        edge_index = data.get('edge_index') if isinstance(data, dict) else data.edge_index
        if perturb_first is not None:
            x = self.model[0](x, edge_index)
            x = x + perturb_first
        else:
            x = self.model[0](x, edge_index)
        x = self.model[3](x, edge_index)

        if perturb_last is not None:
            x = x + perturb_last
        return x

    def reset_parameters(self):
        self.model.reset_parameters()

    def standardize_weights(self):
        skipped_first_conv = False
        for m in self.model.modules():
            if isinstance(m, SAGEConv):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                # Standardize weights for lin_l
                if hasattr(m, 'lin_l'):
                    weight = m.lin_l.weight.data
                    var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                    weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                    m.lin_l.weight.data = weight
                
                # Standardize weights for lin_r if it exists
                if hasattr(m, 'lin_r'):
                    weight = m.lin_r.weight.data
                    var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                    weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                    m.lin_r.weight.data = weight

class Encoder_Adversarial_GraphGAT(nn.Module):
    def __init__(self, layer_sizes, num_heads=4, batchnorm=False, batchnorm_mm=0.99, layernorm=True, weight_standardization=False):
        super().__init__()
        assert batchnorm != layernorm
        assert len(layer_sizes) >= 2
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self.weight_standardization = weight_standardization
        layers = []
        self.gat_layers = []  # store references to GATConv layers for easy access
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            gat_conv = GATConv(in_dim, out_dim // num_heads, heads=num_heads, concat=True)
            self.gat_layers.append(gat_conv)
            layers.append((gat_conv, 'x, edge_index -> x'))

            if batchnorm:
                layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            else:
                layers.append(LayerNorm(out_dim))
            layers.append(nn.PReLU())
        self.model = Sequential('x, edge_index', layers)

    def forward(self, data, perturb_first=None, perturb_last=None):
        if self.weight_standardization:
            self.standardize_weights()
        
        if isinstance(data, dict):
            x = data.get('x', data.get('node_feat'))
        else:
            x = data.x
        
        edge_index = data.get('edge_index') if isinstance(data, dict) else data.edge_index
        if perturb_first is not None:
            x = self.model[0](x, edge_index)
            # Ensure perturb_first matches the shape of x before adding
            perturb_first = perturb_first.view(x.size(0), -1)  # Reshape to (batch_size, feature_dim)
            x = x + perturb_first
        else:
            x = self.model[0](x, edge_index)

        x = self.model[3](x, edge_index)
        if perturb_last is not None:
            x = x + perturb_last
        return x

    def reset_parameters(self):
        self.model.reset_parameters()

    def standardize_weights(self):
        for m in self.model.modules():
            if isinstance(m, GATConv):
                # Access the attention weights for source and destination
                if hasattr(m, 'att_src'):
                    weight = m.att_src.data  # Direct access to the data
                    var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                    weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                    m.att_src.data = weight

                if hasattr(m, 'att_dst'):
                    weight = m.att_dst.data  # Direct access to the data
                    var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                    weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                    m.att_dst.data = weight
