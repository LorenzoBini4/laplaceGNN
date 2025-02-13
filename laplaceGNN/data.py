import numpy as np
import torch
from torch_geometric import datasets
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import NormalizeFeatures
from ogb.nodeproppred import NodePropPredDataset
import torch_geometric
import torch_geometric.utils as pyg_utils
import torch_geometric.datasets as datasets
from torch_geometric.data import Data

def get_dataset(root, name, transform=NormalizeFeatures()):
    pyg_dataset_dict = {
        'coauthor-cs': (datasets.Coauthor, 'CS'),
        'coauthor-physics': (datasets.Coauthor, 'physics'),
        'amazon-computers': (datasets.Amazon, 'Computers'),
        'amazon-photos': (datasets.Amazon, 'Photo'),
    }

    dataset_class, name = pyg_dataset_dict[name]
    dataset = dataset_class(root, name=name, transform=transform)

    return dataset

def get_wiki_cs(root, transform=NormalizeFeatures()):
    dataset = datasets.WikiCS(root, transform=transform)
    data = dataset[0]
    std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
    data.x = (data.x - mean) / std
    data.edge_index = to_undirected(data.edge_index)
    
    train_mask = np.array(data.train_mask)
    val_mask = np.array(data.val_mask)
    test_mask = np.array(data.test_mask)
    
    return [data], train_mask, val_mask, test_mask

def get_cora(root, transform=NormalizeFeatures()):
    dataset = datasets.Planetoid(root, name='Cora', transform=transform)
    data = dataset[0]
    std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
    data.x = (data.x - mean) / std
    data.edge_index = to_undirected(data.edge_index)
    return [data], np.array(data.train_mask), np.array(data.val_mask), np.array(data.test_mask)

def get_citeseer(root, transform=NormalizeFeatures()):
    dataset = datasets.Planetoid(root, name='CiteSeer', transform=transform)
    data = dataset[0]
    std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
    data.x = (data.x - mean) / std
    data.edge_index = to_undirected(data.edge_index)
    return [data], np.array(data.train_mask), np.array(data.val_mask), np.array(data.test_mask)

def get_pubmed(root, transform=NormalizeFeatures()):
    dataset = datasets.Planetoid(root, name='PubMed', transform=transform)
    data = dataset[0]
    std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
    data.x = (data.x - mean) / std
    data.edge_index = to_undirected(data.edge_index)
    return [data], np.array(data.train_mask), np.array(data.val_mask), np.array(data.test_mask)

''''
Standard-Protocol-Evaluation
ogn-arxiv
Training set: 54%
Validation set: 18%
Test set: 28%
'''
def get_ogbn_arxiv(root):
    dataset = NodePropPredDataset(name='ogbn-arxiv', root=root)
    split_idx = dataset.get_idx_split()
    
    # OGB returns a tuple (data_dict, label)
    data_dict = dataset[0][0]  # First element is the data dictionary
    labels = dataset[0][1]  # Second element is the labels

    data = Data(
        x=torch.tensor(data_dict['node_feat']),  # Node features
        edge_index=torch.tensor(data_dict['edge_index']),  # Edges
        y=torch.tensor(labels).squeeze()  # Labels
    )

    # Normalize node features
    std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
    data.x = (data.x - mean) / std

    # Convert the graph to an undirected graph 
    data.edge_index = pyg_utils.to_undirected(data.edge_index)

    return [data], split_idx['train'], split_idx['valid'], split_idx['test']

def get_ogbn_papers100M(root):
    dataset = NodePropPredDataset(name='ogbn-papers100M', root=root)
    split_idx = dataset.get_idx_split()
    
    # OGB returns a tuple (data_dict, label)
    data_dict = dataset[0][0]  # First element is the data dictionary
    labels = dataset[0][1]  # Second element is the labels

    data = Data(
        x=torch.tensor(data_dict['node_feat']),  # Node features
        edge_index=torch.tensor(data_dict['edge_index']),  # Edges
        y=torch.tensor(labels).squeeze()  # Labels
    )

    # Normalize node features
    std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
    data.x = (data.x - mean) / std

    # Convert the graph to an undirected graph
    data.edge_index = pyg_utils.to_undirected(data.edge_index)

    return [data], split_idx['train'], split_idx['valid'], split_idx['test']

class ConcatDataset(InMemoryDataset):
    r"""
    PyG class for merging multiple Dataset objects into one, if needed.
    """
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        self.__indices__ = None
        self.__data_list__ = []
        for dataset in datasets:
            self.__data_list__.extend(list(dataset))
        self.data, self.slices = self.collate(self.__data_list__)
