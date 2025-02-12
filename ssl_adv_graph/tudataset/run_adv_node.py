import argparse
import numpy as np
import os
import os.path as osp
import sys
sys.path.append('../')
import gc
import torch
from torch import nn
import torch_geometric.transforms as T
from tqdm import tqdm
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS
from torch.optim import AdamW
from torch.nn.functional import cosine_similarity
from scheduler import CosineDecayScheduler

from ../../laplaceGNN.utils import set_random_seeds
from laplacian_eval_graph import get_split, LaplacianLogRegr
from laplaceGNN4Graph import LaplaceGNN_Node
from ../../laplacian_augmentations.laplacian_node import *
from predictors import *
from models import *
import gc

###################### GNN Encoder (shared for teacher and student) ######################
class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z

###################### LaplaceGNN Training and Testing ######################
def test(encoder_model, data):
    encoder_model.eval()
    x, edge_index = data.x, data.edge_index
    z = encoder_model.online_encoder(x, edge_index)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    best_result = {
        'accuracy': 0,
        'micro_f1': 0,
        'macro_f1': 0,
        'accuracy_val': 0,
        'micro_f1_val': 0,
        'macro_f1_val': 0
    }
    for decay in [0.0, 0.001, 0.005, 0.01, 0.1]:
        result = LaplacianLogRegr(weight_decay=decay)(z, data.y, split)
        if result['accuracy_val'] > best_result['accuracy_val']:
            best_result = result
    return best_result

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=77, help='Random seed')
    parser.add_argument('--dataset', type=str, choices=['Cora', 'CiteSeer', 'PubMed', 'Computers', 'Photo', 'CS', 'Physics', 'Wiki-CS'], default='Cora', help='Dataset name')
    parser.add_argument('--graph_encoder_layer', type=int, nargs='+', default=[512, 512], help='Conv layer sizes.')
    parser.add_argument('--predictor_hidden_size', type=int, default=512, help='Hidden size of projector.')
    parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate for model training.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='The value of the weight decay for training.')
    parser.add_argument('--mm', type=float, default=0.99, help='The momentum for moving average.')
    parser.add_argument('--centering', action='store_true', help='Whether to center the momentum.')
    parser.add_argument('--sharpening', action='store_true', help='Whether to sharpen the momentum.')
    parser.add_argument('--center_mm', type=float, default=0.9, help='The momentum for centering.')
    parser.add_argument('--temperature', type=float, default=0.04, help='The temperature for sharpening.')
    parser.add_argument('--lr_warmup_epochs', type=int, default=50, help='Warmup period for learning rate.')
    parser.add_argument('--epoch', type=int, default=500, help='LaplaceGNN number of epochs for training')
    parser.add_argument('--lapl_max_lr ', type=float, default=100, help='augmentation learning rate for laplacian max strategy')
    parser.add_argument('--lapl_min_lr ', type=float, default=0.1, help='augmentation learning rate for laplacian min strategy')
    parser.add_argument('--lapl_epoch ', type=int, default=10, help='iteration for augmentation')
    parser.add_argument('--prob_feat', type=float, default=0.4, help='feature masking probability')  # if standard feature augmentations have been selected to be added 
    parser.add_argument('--threshold', type=float, default=0.3, help='threshold for edge perturbation')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--delta', type=float, default=0.001, help='perturbation magnitude')
    parser.add_argument('--m', type=int, default=1, help='number of inner maximization steps')
    parser.add_argument('--step_size', type=float, default=0.001, help='step size for inner maximization')
    return parser.parse_args()

def main():
    args = arg_parse()
    for arg, value in vars(args).items():
        print(f'{arg}: {value}')
    print(50*'-')    
    set_random_seeds(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = osp.join(osp.expanduser('./data/'), 'datasets')
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(path, name=args.dataset, transform=T.NormalizeFeatures())
    elif args.dataset in ['Computers', 'Photo']:
        dataset = Amazon(path, name=args.dataset, transform=T.NormalizeFeatures())
    elif args.dataset in ['CS', 'Physics']:
        dataset = Coauthor(path, name=args.dataset, transform=T.NormalizeFeatures())
    elif args.dataset == 'Wiki-CS':
        dataset = WikiCS(path, transform=T.NormalizeFeatures())
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    ######################## AUGMENTATION ########################
    centrality_types = ['degree', 'pagerank', 'eigenvector']
    centrality_weights = [0.2, 0.3, 0.5] # randomly initialized, trainable params.
    # centrality_weights={'degree': 0.2, 'pagerank': 0.3, 'eigenvector': 0.5} # uncomment it if precoomputed centrality is being used.
    # precomputed_centrality = {
    #     'degree': nx.degree_centrality(to_networkx(data)),
    #     'pagerank': nx.pagerank(to_networkx(data)),
    #     'eigenvector': nx.eigenvector_centrality(to_networkx(data))
    # }

    L1_view = CentralitySpectralAugmentation_Node(
        ratio=args.threshold,
        lr=args.lapl_max_lr,
        iteration=args.lapl_epoch,
        dis_type='max',
        device=device,
        centrality_types=centrality_types,
        centrality_weights=centrality_weights,
        precomputed_centrality=None,
        sample='no'
    )

    L2_view = CentralitySpectralAugmentation_Node(
        ratio=args.threshold,
        lr=args.lapl_min_lr,
        iteration=args.lapl_epoch,
        dis_type='min',
        device=device,
        centrality_types=centrality_types,
        centrality_weights=centrality_weights,
        precomputed_centrality=None,
        sample='no'
    )

    # Precompute laplacian perturbation or load them
    laplacian_path = osp.join(path, args.dataset+'/laplacian_max{}_min{}_threshold{}.pt'.format(args.lapl_max_lr, args.lapl_min_lr, args.threshold))
    if os.path.exists(laplacian_path):  # Load saved probability matrix
        loaded_laplacian_path = torch.load(laplacian_path)
        print('Laplacian perturbations have beeen loaded!')
        print(f'Data before applying augmentor: {data}')
        print(50*'-')
    else:  
        print('Laplacian perturbations under computation')
        assert dataset.len() == 1  # now it's node classifiction task
        laplacian_path = []
        print(f'Data before applying augmentor: {data}')
        L1_view.calc_prob(data) # now max-laplacian has been encoded as data['max']=ptb_prob1
        torch.cuda.empty_cache()
        gc.collect()  
        L2_view.calc_prob(data) # now min-laplacian has been encoded as data['min']=ptb_prob1
        torch.cuda.empty_cache()
        gc.collect()  
        print(50*'-')
        torch.save(data, laplacian_path)
    print(f'Data after applying augmentor')
    print(50*'-')
    print(f'Data: {data}')
    print(f'Data.max: {data.max}')
    print(f'Data.min: {data.min}')
    print(50*'-')
    # exit()
    data = data.to(device)
    L1 = Compose([L1_view, FeatAugmentation(pf=args.prob_feat)])
    L2 = Compose([L2_view, FeatAugmentation(pf=args.prob_feat)])
    input_size, representation_size = data.x.size(1), args.graph_encoder_layer[-1]
    encoder = Encoder_Adversarial_GCN([input_size] + args.graph_encoder_layer, batchnorm=True, layernorm=False, weight_standardization=False).to(device)    
    predictor = MLP_Predictor(representation_size, representation_size, hidden_size=args.predictor_hidden_size).to(device)
    encoder_model = LaplaceGNN_Node(encoder, predictor, augmentation=(L1,L2)).to(device)
    lr_scheduler = CosineDecayScheduler(args.lr, args.lr_warmup_epochs, args.epoch)
    mm_scheduler = CosineDecayScheduler(1 - args.mm, 0, args.epoch)
    optimizer = AdamW(encoder_model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay) 

    def train(step, data):
            encoder_model.train()
            optimizer.zero_grad()
            lr = lr_scheduler.get(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            mm = 1 - mm_scheduler.get(step)
        
            accumulation_steps = args.accumulation_steps
            step_loss = 0  # to accumulate loss across steps

            perturb_shape_first = (data.x.shape[0], encoder_model.online_encoder.model[0].out_channels)  # first layer
            perturb_shape_last = (data.x.shape[0], encoder_model.online_encoder.model[3].out_channels)   # second and last layer

            perturb_first = torch.Tensor(*perturb_shape_first).uniform_(-args.delta, args.delta).to(device)
            perturb_first.requires_grad_()

            perturb_last = torch.Tensor(*perturb_shape_last).uniform_(-args.delta, args.delta).to(device)
            perturb_last.requires_grad_()

            if perturb_first is not None:
                perturb_first = perturb_first.to(device)
            if perturb_last is not None:
                perturb_last = perturb_last.to(device)

            for acc_step in range(accumulation_steps):
                if acc_step > 0:
                    optimizer.zero_grad()

                q1, y2 = encoder_model(data, perturb_first=perturb_first, perturb_last=None) 
                q2, y1 = encoder_model(data, perturb_first=perturb_first, perturb_last=None)

                loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
                # inner maximization loop for perturbations
                for _ in range(args.m-1):
                    loss.backward(retain_graph=True)
                    with torch.no_grad():
                        perturb_first.data += args.step_size * torch.sign(perturb_first.grad)
                        perturb_first.data = perturb_first.data.clamp(-args.delta, args.delta)
                        # perturb_last.data += args.step_size * torch.sign(perturb_last.grad)
                        # perturb_last.data = perturb_last.data.clamp(-args.delta, args.delta)
                    perturb_first.grad.zero_()
                    # perturb_last.grad.zero_()
                    q1, y2 = encoder_model(data, perturb_first=perturb_first, perturb_last=None)
                    q2, y1 = encoder_model(data, perturb_first=perturb_first, perturb_last=None) 
                    loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
                # for the last backward call in the accumulation loop, we don't retain the graph
                if acc_step == accumulation_steps - 1:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
                step_loss += loss / accumulation_steps
                if (acc_step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    encoder_model.update_target_network(mm, centering=args.centering, sharpening=args.sharpening, center_momentum=args.center_mm, temperature=args.temperature),  
            print(f'Step: {step}, Loss: {loss.item()}') #, Learning Rate: {lr}, Momentum: {mm}')
            return step_loss.item()
    
    with tqdm(total=args.epoch, desc='(T)') as pbar:
        for epoch in range(1, args.epoch+1):
            loss = train(epoch, data)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, data)
    print(f'Test accuracy={test_result["accuracy"]:.4f}')

if __name__ == '__main__':
    main()
