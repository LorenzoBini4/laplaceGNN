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
from utils import get_adj_tensor, get_normalize_adj_tensor, to_dense_adj, dense_to_sparse, switch_edge, drop_feature

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
# If also features want to be used
class FeatAugmentation(Augmentation):
    def __init__(self, prob_feat: float):
        super(FeatAugmentation, self).__init__()
        self.prob_feat = prob_feat

    def augment(self, g: Graph, batch: torch.Tensor) -> Graph:
        x, edge_index, _ = g.unfold()
        x = drop_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, ptb_prob=None)

    def get_aug_name(self):
        return 'feature'

class Compose(Augmentation):
  def __init__(self, augmentations: List[Augmentation]):
        super(Compose, self).__init__()
        self.augmentations = augmentations

    def augment(self, g: Graph, batch: torch.Tensor) -> Graph:
        for aug in self.augmentations:
            g = aug.augment(g, batch)
        return g
    
class LaplaceGNN_Augmentation_Node(Augmentation):
    def __init__(self, ratio, lr, iteration, dis_type, device, centrality_types, centrality_weights, threshold=0.5, precomputed_centrality=None, sample='no'):
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
        super(CentralitySpectralAugmentor_Node, self).__init__()
        self.ratio = ratio
        self.lr = lr
        self.iteration = iteration
        self.dis_type = dis_type
        self.device = device
        self.centrality_types = centrality_types
        self.centrality_weights = centrality_weights
        self.threshold = threshold
        self.precomputed_centrality = precomputed_centrality
        self.sample = sample
        
    def compute_centrality(self, adj):
        """
        Compute multiple centrality metrics and combine them.
        Args:
            adj (torch.Tensor): Adjacency matrix.
        Returns:
            torch.Tensor: Combined centrality scores normalized to [0, 1].
        """
        ####################################### Needed only for WikyCS dataset
        # adj = (adj + adj.T) / 2.0  # Symmetrize
        # adj += torch.eye(adj.shape[0], device=adj.device) * 1e-6  # Regularization
        #######################################
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
                eigvals, eigvecs = torch.linalg.eigh(adj.float())
                centrality_scores['eigenvector'] = eigvecs[:, -1]
            else:
                raise ValueError(f"Unknown centrality type: {centrality_type}")
        
        combined_centrality = sum(
            self.centrality_weights[i] * centrality_scores[ctype]
            for i, ctype in enumerate(self.centrality_types)
        )
        return combined_centrality / combined_centrality.sum()  # Normalize

    def calc_prob(self, data, fast=True, silence=False, precomputed_centrality=None):
        """
        Precompute the perturbation probabilities with centrality-guided initialization using Power Iteration.
        Args:
            data (torch_geometric.data.Data): Graph data object.
            fast (bool): Whether to use a fast approximation for eigenvalue computation.
            silence (bool): Whether to suppress progress output.
            precomputed_centrality (dict, optional): Precomputed centrality measures.
        Returns:
            torch_geometric.data.Data: Updated graph data object with perturbation probabilities.
        """
        def power_iteration(matrix, num_iters=10):
            """
            Approximate the largest eigenvalue using Power Iteration without in-place operations.
            Args:
                matrix (torch.Tensor): Input symmetric matrix.
                num_iters (int): Number of iterations for Power Iteration.
            Returns:
                float: Approximated largest eigenvalue.
            """
            vec = torch.rand(matrix.shape[0], device=matrix.device)
            vec = vec / torch.norm(vec)  # Out-of-place normalization
            for _ in range(num_iters):
                vec = torch.matmul(matrix, vec)
                vec = vec / torch.norm(vec)  # Out-of-place normalization
            eigenvalue = torch.dot(vec, torch.matmul(matrix, vec))
            return eigenvalue

        x, edge_index = data.x, data.edge_index
        x = x.to(self.device)
        ori_adj = get_adj_tensor(edge_index.cpu()).to(self.device)

        # Use precomputed centrality if provided, otherwise compute
        if precomputed_centrality is not None:
            # Convert precomputed centrality to tensor and normalize
            combined_centrality = torch.zeros(ori_adj.shape[0], device=self.device)
            for centrality_type, centrality_values in precomputed_centrality.items():
                # Check if centrality type is in the specified centrality types
                if centrality_type in self.centrality_types:
                    # Convert to tensor and normalize
                    type_centrality = torch.tensor(list(centrality_values.values()), device=self.device)
                    type_centrality = (type_centrality - type_centrality.min()) / (type_centrality.max() - type_centrality.min())
                    
                    # Apply weight if specified
                    weight = self.centrality_weights.get(centrality_type, 1.0)
                    combined_centrality += weight * type_centrality
            
            # Normalize the combined centrality
            combined_centrality = (combined_centrality - combined_centrality.min()) / (combined_centrality.max() - combined_centrality.min())
        else:
            # Fallback to original centrality computation method
            combined_centrality = self.compute_centrality(ori_adj)

        # Initialize perturbation probabilities based on centrality
        nnodes = ori_adj.shape[0]
        tril_indices = torch.tril_indices(nnodes, nnodes, offset=-1)
        adj_changes = Parameter(combined_centrality[tril_indices[0]] * combined_centrality[tril_indices[1]], requires_grad=True).to(self.device)
        # Compute normalized adjacency matrix
        ori_adj_norm = get_normalize_adj_tensor(ori_adj, device=self.device)
        # Use SVD for spectral analysis
        if fast:
            print('Using fast SVD for spectral analysis')
            # For large matrices, use truncated SVD
            k = min(10, min(ori_adj_norm.shape) // 2)
            U, S, V = torch.svd_lowrank(ori_adj_norm, q=k)
            ori_spectrum = S[:k]
        else:
            print('Using full eigenvalues for spectral analysis')
            ori_spectrum = torch.linalg.eigvalsh(ori_adj_norm)

        # Compute norm of the spectrum for loss calculation
        spectrum_norm = torch.norm(ori_spectrum)

        # Compute number of perturbations
        n_perturbations = int(self.ratio * (ori_adj.sum() / 2))

        # Progress tracking
        with tqdm(total=self.iteration, desc='Centrality Spectral Augment', disable=silence) as pbar:
            for t in range(1, self.iteration + 1):
                # Modify adjacency matrix
                modified_adj = self.get_modified_adj(ori_adj, self.reshape_m(nnodes, adj_changes))
                adj_norm_noise = get_normalize_adj_tensor(modified_adj, device=self.device)

                # Compute SVD of modified adjacency
                if fast:
                    print('Using fast SVD for spectral analysis')
                    # Truncated SVD for modified adjacency
                    k = min(10, min(adj_norm_noise.shape) // 2)
                    U_noise, S_noise, V_noise = torch.svd_lowrank(adj_norm_noise, q=k)
                    noise_spectrum = S_noise[:k]
                else:
                    print('Using full eigenvalues for spectral analysis')
                    noise_spectrum = torch.linalg.eigvalsh(adj_norm_noise)

                # Compute spectral difference
                spectrum_mse = torch.norm(ori_spectrum - noise_spectrum)

                # Define spectral loss
                reg_loss = spectrum_mse / spectrum_norm if self.dis_type == 'max' else -spectrum_mse / spectrum_norm
                self.loss = reg_loss

                # Gradient update
                adj_grad = torch.autograd.grad(self.loss, adj_changes)[0]
                lr = self.lr / (t ** 0.5)
                adj_changes.data.add_(lr * adj_grad)
                self.projection(n_perturbations, adj_changes)
                pbar.set_postfix({'reg_loss': reg_loss.item(), 'budget': n_perturbations})
                pbar.update()

        data[self.dis_type] = SparseTensor.from_dense(self.reshape_m(nnodes, adj_changes))
        return data

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
                print('Using sampling for perturbation')
                binary = np.random.binomial(1, s)
                mask = np.random.binomial(1, 0.7, s.shape)
                sampled = np.multiply(binary, mask)
            else:
                print('Not using sampling for perturbation')
                sampled = np.random.binomial(1, s)
            return torch.FloatTensor(sampled).to(self.device)
