from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List
from tqdm import tqdm
import pickle as pkl
import os
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch_sparse import SparseTensor
from torch_geometric.utils.sparse import to_edge_index
from torch_geometric.utils import unbatch, unbatch_edge_index
from torch_geometric.data import Batch, Data
from utils import get_adj_tensor, preprocess_adj, get_normalize_adj_tensor, to_dense_adj, dense_to_sparse, switch_edge, drop_feature

###################### Standard Class ######################
class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    ptb_prob: Optional[SparseTensor]

    def unfold(self) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[SparseTensor]]:
        return self.x, self.edge_index, self.ptb_prob

class Augmentation(ABC):
    """Standard class for augmentation."""
    def __init__(self):
        pass
    @abstractmethod
    def augment(self, g: Graph, batch: torch.Tensor) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(
        self, 
        x: torch.FloatTensor, 
        edge_index: torch.LongTensor, 
        ptb_prob: Optional[SparseTensor] = None, 
        batch = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.augment(Graph(x, edge_index, ptb_prob), batch).unfold()

###################### Laplacian Max-Min Augmentation Module - LaplaceGNN Class ######################
class Compose(Augmentation):
    def __init__(self, augmentors: List[Augmentation]):
        super(Compose, self).__init__()
        self.augmentors = augmentors

    def augment(self, g: Graph, batch: torch.Tensor) -> Graph:
        for aug in self.augmentors:
            g = aug.augment(g, batch)
        return g

# If also features want to be used
class FeatAugmentation(Augmentation):
    def __init__(self, pf: float):
        super(FeatAugmentation, self).__init__()
        self.pf = pf
    def augment(self, g: Graph, batch: torch.Tensor) -> Graph:
        x, edge_index, _ = g.unfold()
        x = drop_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, ptb_prob=None)
    def get_aug_name(self):
        return 'feature'

class CentralitySpectralAugmentor_Graph(Augmentation):
    def __init__(self, ratio, lr, iteration, dis_type, device, centrality_types, centrality_weights, sample='no', threshold=0.5):
        """
        Centrality-guided Laplacian Spectral Augmentor.
        Args:
            ratio (float): Ratio of edges to perturb.
            lr (float): Learning rate for gradient updates.
            iteration (int): Number of optimization iterations.
            dis_type (str): Distance type ('max', 'min', etc.).
            device (torch.device): Device for computation.
            centrality_types (list): List of centrality measures to compute.
            centrality_weights (list): Weights for combining centrality measures.
            sample (str): Sampling mode ('yes' or 'no').
            threshold (float): Perturbation threshold for projection.
        """
        super(CentralitySpectralAugmentor_Graph, self).__init__()
        self.ratio = ratio
        self.lr = lr
        self.iteration = iteration
        self.dis_type = dis_type
        self.device = device
        self.centrality_types = centrality_types
        self.centrality_weights = centrality_weights
        self.sample = sample
        self.threshold = threshold

    def compute_centrality(self, adj):
        """
        Compute multiple centrality metrics and combine them.
        Args:
            adj (torch.Tensor): Adjacency matrix.
        Returns:
            torch.Tensor: Combined centrality scores normalized to [0, 1].
        """
        adj = (adj + adj.T) / 2.0  # Symmetrize
        adj += torch.eye(adj.shape[0], device=adj.device) * 1e-6  # Regularization
        
        centrality_scores = {}
        for centrality_type in self.centrality_types:
            if centrality_type == 'degree':
                centrality_scores['degree'] = adj.sum(dim=1)
            elif centrality_type == 'pagerank':
                pagerank = torch.linalg.solve(
                    torch.eye(adj.shape[0]).to(self.device) - 0.85 * adj,
                    torch.ones(adj.shape[0]).to(self.device)
                )
                centrality_scores['pagerank'] = pagerank
            elif centrality_type == 'eigenvector':
                try:
                    # Try standard eigenvalue computation
                    eigvals, eigvecs = torch.linalg.eigh(adj.float())
                    centrality_scores['eigenvector'] = torch.abs(eigvecs[:, -1])
                except:
                    try:
                        # Fallback to symmetric eigenvalue computation
                        eigvals, eigvecs = torch.symeig(adj.float(), eigenvectors=True)
                        centrality_scores['eigenvector'] = torch.abs(eigvecs[:, -1])
                    except:
                        # Last resort: use a simpler centrality measure
                        centrality_scores['eigenvector'] = adj.sum(dim=1)
            else:
                raise ValueError(f"Unknown centrality type: {centrality_type}")
        
        # Combine centrality scores
        combined_centrality = sum(
            self.centrality_weights[i] * centrality_scores[ctype]
            for i, ctype in enumerate(self.centrality_types)
        )
        
        # Normalize and handle potential numerical issues
        combined_centrality = combined_centrality + 1e-10  # Prevent division by zero
        return combined_centrality / combined_centrality.sum()

    def calc_prob(self, data, silence=False):
        x, edge_index = data.x, data.edge_index
        x = x.to(self.device)
        ori_adj = get_adj_tensor(edge_index.cpu()).to(self.device)
         # Insert the preprocessing here
        ori_adj = preprocess_adj(ori_adj, self.device)

        # Enforce symmetry and regularization
        ori_adj = (ori_adj + ori_adj.T) / 2.0
        ori_adj += torch.eye(ori_adj.shape[0], device=self.device) * 1e-6

        # Compute combined centrality
        combined_centrality = self.compute_centrality(ori_adj)

        # Initialize perturbation probabilities as a Parameter with requires_grad=True
        nnodes = ori_adj.shape[0]
        tril_indices = torch.tril_indices(nnodes, nnodes, offset=-1)
        adj_changes = Parameter(
            torch.zeros_like(combined_centrality[tril_indices[0]], requires_grad=True)
        ).to(self.device)

        ori_adj_norm = get_normalize_adj_tensor(ori_adj, device=self.device)
        ori_e = torch.linalg.eigvalsh(ori_adj_norm)
        eigen_norm = torch.norm(ori_e)

        n_perturbations = int(self.ratio * (ori_adj.sum() / 2))
        optimizer = torch.optim.Adam([adj_changes], lr=self.lr)

        with tqdm(total=self.iteration, desc='Centrality LaplaceGNN4Graph Augmentation', disable=silence) as pbar:
            for t in range(1, self.iteration + 1):
                optimizer.zero_grad()

                # Reshape adj_changes to full symmetric matrix
                m = self.reshape_m(nnodes, adj_changes)
                modified_adj = self.get_modified_adj(ori_adj, m)
                
                try:
                    adj_norm_noise = get_normalize_adj_tensor(modified_adj, device=self.device)
                    e = torch.linalg.eigvalsh(adj_norm_noise)
                    
                    # Define spectral loss
                    eigen_mse = torch.norm(ori_e - e)
                    reg_loss = eigen_mse / eigen_norm if self.dis_type == 'max' else -eigen_mse / eigen_norm
                    
                    # Compute and apply gradients
                    reg_loss.backward()
                    optimizer.step()
                    
                    self.projection(n_perturbations, adj_changes)
                    
                    pbar.set_postfix({'reg_loss': reg_loss.item(), 'budget': n_perturbations})
                    pbar.update()
                    
                except Exception as e:
                    print(f"Iteration {t} failed: {e}")
                    break

        # data[self.dis_type] = SparseTensor.from_dense(self.reshape_m(nnodes, adj_changes))
        # Store the computed probabilities as an attribute
        ptb_prob = SparseTensor.from_dense(self.reshape_m(nnodes, adj_changes))
        if self.dis_type == "max":
            data.max = ptb_prob
        elif self.dis_type == "min":
            data.min = ptb_prob
        return data
    
    # def __call__(self, data):
    #     """
    #     Apply the precomputed probabilistic augmentation to a given data object.
        
    #     Args:
    #         data (torch_geometric.data.Data): Input graph data object
        
    #     Returns:
    #         torch_geometric.data.Data: Augmented graph data object
    #     """
    #     # Ensure probabilities have been precomputed
    #     if not hasattr(self, 'stored_prob'):
    #         raise RuntimeError("Probabilities must be precomputed using calc_prob() before calling the augmentor")
        
    #     # Create a copy of the original data to avoid modifying the input
    #     augmented_data = data.clone()
        
    #     # Convert SparseTensor to edge_index
    #     row, col, _ = self.stored_prob.coo()
    #     augmented_data.edge_index = torch.stack([row, col])
        
    #     return augmented_data

    def augment(self, g: Graph, batch: torch.Tensor) -> Graph:
        """
        Apply the augmentation to the graph.
        Args:
            g (Graph): Input graph.
            batch (torch.Tensor): Batch information (if any).
        Returns:
            Graph: Augmented graph.
        """
        x, edge_index, ptb_prob = g.unfold()
        ori_adj = to_dense_adj(edge_index, batch)
        ptb_idx, ptb_w = to_edge_index(ptb_prob)
        ptb_m = to_dense_adj(ptb_idx, batch, ptb_w)
        ptb_adj = self.random_sample(ptb_m)
        modified_adj = self.get_modified_adj(ori_adj, ptb_adj).detach()

        if batch is None:
            edge_index, _ = dense_to_sparse(modified_adj)
        else:
            x_unbatched = unbatch(x, batch)
            aug_data = Batch.from_data_list([Data(
                x=x_unbatched[b],
                edge_index=dense_to_sparse(modified_adj[b])[0]
            ) for b in range(modified_adj.shape[0])])
            x = aug_data.x
            edge_index = aug_data.edge_index

        return Graph(x=x, edge_index=edge_index, ptb_prob=None)

    def get_modified_adj(self, ori_adj, m):
        nnodes = ori_adj.shape[1]
        complementary = (torch.ones_like(ori_adj) - torch.eye(nnodes).to(self.device) - ori_adj) - ori_adj
        modified_adj = complementary * m + ori_adj
        return modified_adj

    def reshape_m(self, nnodes, adj_changes):
        m = torch.zeros((nnodes, nnodes)).to(self.device)
        tril_indices = torch.tril_indices(row=nnodes, col=nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = adj_changes
        m = m + m.t()
        return m

    def projection(self, n_perturbations, adj_changes):
        if torch.clamp(adj_changes, 0, self.threshold).sum() > n_perturbations:
            left = adj_changes.min()
            right = adj_changes.max()
            miu = self.bisection(left, right, n_perturbations, 1e-4, adj_changes)
            adj_changes.data.copy_(torch.clamp(adj_changes.data - miu, min=0, max=1))
        else:
            adj_changes.data.copy_(torch.clamp(adj_changes.data, min=0, max=1))

    def bisection(self, a, b, n_perturbations, epsilon, adj_changes):
        def func(x):
            return torch.clamp(adj_changes - x, 0, self.threshold).sum() - n_perturbations

        while (b - a) >= epsilon:
            miu = (a + b) / 2
            if func(miu) == 0.0:
                return miu
            elif func(miu) * func(a) < 0:
                b = miu
            else:
                a = miu
        return (a + b) / 2
        
    def random_sample(self, edge_prop):
        """
        Randomly sample perturbations from the edge probability matrix.
        Args:
            edge_prop (torch.Tensor): Edge probability matrix.
        
        Returns:
            torch.FloatTensor: Binary sampled perturbation matrix.
        """
        with torch.no_grad():
            s = edge_prop.cpu().detach().numpy()
            # s = (s + np.transpose(s))
            if self.sample == 'yes':
                binary = np.random.binomial(1, s)
                mask = np.random.binomial(1, 0.7, s.shape)
                sampled = np.multiply(binary, mask)
            else:
                sampled = np.random.binomial(1, s)
            return torch.FloatTensor(sampled).to(self.device)
