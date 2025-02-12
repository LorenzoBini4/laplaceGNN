import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, BatchNorm, GCNConv, LayerNorm, SAGEConv, Sequential, global_mean_pool
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn import GATConv

class GCN(nn.Module):
    def __init__(self, layer_sizes, batchnorm=False, batchnorm_mm=0.99, layernorm=True, weight_standardization=False):
        super().__init__()

        assert batchnorm != layernorm
        assert len(layer_sizes) >= 2
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self.weight_standardization = weight_standardization

        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append((GCNConv(in_dim, out_dim), 'x, edge_index -> x'),)

            if batchnorm:
                layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            else:
                layers.append(LayerNorm(out_dim))

            layers.append(nn.PReLU())

        self.model = Sequential('x, edge_index', layers)

    def forward(self, data):
        if self.weight_standardization:
            self.standardize_weights()
        return self.model(data.x, data.edge_index)

    def reset_parameters(self):
        self.model.reset_parameters()

    def standardize_weights(self):
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

########################### No Laplace-module only Adversarial Training #######################################
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

    def forward(self, data, perturb_first=None, perturb_last=None):
        
        if self.weight_standardization:
            self.standardize_weights()
        
         # check if data is a PyG Data object or a dictionary ---> useful for ogbn-arxiv
        if isinstance(data, dict):
            x = data.get('x', data.get('node_feat'))  # 'x' first, then 'node_feat'
        else:
            x = data.x  
        
        edge_index = data.get('edge_index') if isinstance(data, dict) else data.edge_index

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

class Encoder_Adversarial_GAT(nn.Module):
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

########################### LAPLACEGNN - Complete Module #######################################
class Encoder_LaplaceGNN_GCN(nn.Module):
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

########################### FOR PPI ########################################
class Encoder_LaplaceGNN_PPISAGE(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super().__init__()

        self.convs = nn.ModuleList([
            SAGEConv(input_size, hidden_size, root_weight=True),
            SAGEConv(hidden_size, hidden_size, root_weight=True),
            SAGEConv(hidden_size, embedding_size, root_weight=True),
        ])

        self.skip_lins = nn.ModuleList([
            nn.Linear(input_size, hidden_size, bias=False),
            nn.Linear(input_size, hidden_size, bias=False),
        ])

        self.layer_norms = nn.ModuleList([
            LayerNorm(hidden_size),
            LayerNorm(hidden_size),
            LayerNorm(embedding_size),
        ])

        self.activations = nn.ModuleList([
            nn.PReLU(1),
            nn.PReLU(1),
            nn.PReLU(1),
        ])

    def forward(self, data, perturb_first=None, perturb_last=None):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        # first hidden layer with optional perturbation
        h1 = self.convs[0](x, edge_index)
        h1 = self.layer_norms[0](h1, batch)
        if perturb_first is not None:
            h1 = h1 + perturb_first  # apply adversarial perturbation
        h1 = self.activations[0](h1)

        x_skip_1 = self.skip_lins[0](x)

        # second hidden layer
        h2 = self.convs[1](h1 + x_skip_1, edge_index)
        h2 = self.layer_norms[1](h2, batch)
        h2 = self.activations[1](h2)

        x_skip_2 = self.skip_lins[1](x)

        # last layer with optional perturbation
        ret = self.convs[2](h1 + h2 + x_skip_2, edge_index)
        ret = self.layer_norms[2](ret, batch)
        if perturb_last is not None:
            ret = ret + perturb_last  # apply adversarial perturbation
        ret = self.activations[2](ret)

        return ret

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()
        for m in self.skip_lins:
            m.reset_parameters()
        for m in self.activations:
            m.weight.data.fill_(0.25)
        for m in self.layer_norms:
            m.reset_parameters()

############################################## ZINC ##############################################
class Encoder_LaplaceGNN_ZINCSAGE(nn.Module):
    def __init__(self, hidden_size, embedding_size):
        super().__init__()

        self.hidden_size = hidden_size
        # ZINC has 3 possible values as bond types (1, 2, 3)
        self.edge_embedding = nn.Embedding(3, hidden_size)
        # ZINC has 21 possible values has atomic node types
        self.input_embedding = nn.Embedding(21, hidden_size)
        # self.input_embedding = nn.Linear(input_size, hidden_size)

        # Graph convolutional layers
        self.convs = nn.ModuleList([
            SAGEConv(hidden_size, hidden_size, root_weight=True),
            SAGEConv(hidden_size, hidden_size, root_weight=True),
            SAGEConv(hidden_size, embedding_size, root_weight=True),
        ])

        # Skip connection layers
        self.skip_lins = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Linear(hidden_size, hidden_size, bias=False),
        ])

        # # Layer normalization layers
        # self.layer_norms = nn.ModuleList([
        #     LayerNorm(hidden_size),
        #     LayerNorm(hidden_size),
        #     LayerNorm(embedding_size),
        # ])

        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.BatchNorm1d(embedding_size),
        ])

        # Activation layers
        self.activations = nn.ModuleList([
            nn.PReLU(1),
            nn.PReLU(1),
            nn.PReLU(1),
        ])

    def forward(self, data, perturb_first=None, perturb_last=None):
        x = self.input_embedding(data.x)
        x = x.view(-1, self.hidden_size)

        edge_index = data.edge_index
        edge_attr = data.edge_attr.view(-1, 1)
        edge_attr_embedded = self.edge_embedding(edge_attr-1)
        
        row, col = edge_index
        if row.max() >= x.size(0) or col.max() >= x.size(0):
            raise ValueError("Edge index values exceed the number of nodes.")
        # print(f"Size of edge_attr_embedded: {edge_attr_embedded.shape}")
        edge_attr_embedded = edge_attr_embedded.view(-1, self.hidden_size)
        x[row] += edge_attr_embedded
        batch = data.batch if hasattr(data, 'batch') else None

        # First hidden layer with optional perturbation
        h1 = self.convs[0](x, edge_index)
        # h1 = self.layer_norms[0](h1, batch)
        h1 = self.batch_norms[0](h1)
        if perturb_first is not None:
            h1 = h1 + perturb_first  # Apply adversarial perturbation
        h1 = self.activations[0](h1)

        # First skip connection
        x_skip_1 = self.skip_lins[0](x)

        # Second hidden layer
        h2 = self.convs[1](h1 + x_skip_1, edge_index)
        # h2 = self.layer_norms[1](h2, batch)
        h2 = self.batch_norms[1](h2)
        h2 = self.activations[1](h2)

        # Second skip connection
        x_skip_2 = self.skip_lins[1](x)

        # Last layer with optional perturbation
        ret = self.convs[2](h1 + h2 + x_skip_2, edge_index)
        # ret = self.layer_norms[2](ret, batch)
        ret = self.batch_norms[2](ret)
        if perturb_last is not None:
            ret = ret + perturb_last  # Apply adversarial perturbation
        ret = self.activations[2](ret)

        return ret

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()
        for m in self.skip_lins:
            m.reset_parameters()
        for m in self.activations:
            m.weight.data.fill_(0.25)
        for m in self.batch_norms:
            m.reset_parameters()
