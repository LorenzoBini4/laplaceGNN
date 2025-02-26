import argparse
import numpy as np
import os
import os.path as osp
import torch
from torch import nn
from tqdm import tqdm
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from laplaceGNN.utils import set_random_seeds, CosineDecayScheduler
from laplacian_eval_graph import get_split, LaplacianLogRegr
from ..augmentations_graph import LaplaceGNN_Augmentation_Graph
from laplaceGNN4Graph import LaplaceGNN_Graph
from torch.optim import AdamW
from torch.nn.functional import cosine_similarity
import gc

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                # First layer: input_dim → hidden_dim
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                # Subsequent layers: hidden_dim → hidden_dim
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv in self.layers:
            z = conv(z, edge_index)  
            z = self.activation(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        g = torch.cat(gs, dim=1)
        return z, g

class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)
        
###################### GNN Encoder (shared for teacher and student) ######################
class GCNEncoder(nn.Module):
    def __init__(self, gconv1, gconv2):
        super(GCNEncoder, self).__init__()
        self.gconv1 = gconv1
        self.gconv2 = gconv2

    def forward(self, x, edge_index, batch=None, perturb_first=None, perturb_last=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        # Forward pass through the first GCN layer
        z1, g1 = self.gconv1(x, edge_index, batch)
        # Apply perturbation to the output of the first GCN layer
        if perturb_first is not None:
            z1 = z1 + perturb_first  # <--- Add perturbation to the output of the first GCN layer
        # Forward pass through the second GCN layer
        z2, g2 = self.gconv2(z1, edge_index, batch)
        # Apply perturbation to the output of the second GCN layer (if needed)
        if perturb_last is not None:
            z2 = z2 + perturb_last  # <--- Add perturbation to the output of the second GCN layer
        return z2, g2

    def reset_parameters(self):
        self.gconv1.reset_parameters()
        self.gconv2.reset_parameters()

###################### LaplaceGNN Training and Testing ######################
def test(encoder_model, dataloader, device):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to(device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        # Obtain graph embeddings from the online encoder
        _, g = encoder_model.online_encoder(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    best_result = {
        'accuracy': 0,
        'micro_f1': 0,
        'macro_f1': 0,
        'accuracy_val': 0,
        'micro_f1_val': 0,
        'macro_f1_val': 0
    }
    for decay in [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0]:
        result = LaplacianLogRegr(weight_decay=decay)(x, y, split)
        if result['accuracy_val'] > best_result['accuracy_val']:
            best_result = result
    return best_result

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=77, help='Random seed')
    parser.add_argument('--device', type=int, default=0, help='cuda')
    parser.add_argument('--dataset', type=str, default='PROTEINS', choices=['MUTAG', 'PROTEINS', 'NCI1', 'IMDB-BINARY', 'IMDB-MULTI'])
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--gnn1_dim', type=int, default=64, help='The hidden dimension of the first GNN layer.')
    parser.add_argument('--gnn1_num_layers', type=int, default=2, help='The number of layers of the first GNN.')
    parser.add_argument('--gnn2_dim', type=int, default=64, help='The hidden dimension of the second GNN layer.')
    parser.add_argument('--gnn2_num_layers', type=int, default=2, help='The number of layers of the second GNN.')
    parser.add_argument('--mlp_dim', type=int, default=64, help='The hidden dimension of the MLP.')
    parser.add_argument('--lr', type=float, default=1e-5, help='The learning rate for model training.')
    parser.add_argument('--weight_decay', type=float, default=6e-5, help='The value of the weight decay for training.')
    parser.add_argument('--mm', type=float, default=0.99, help='The momentum for moving average.')
    parser.add_argument('--centering', action='store_true', help='Whether to center the momentum.')
    parser.add_argument('--sharpening', action='store_true', help='Whether to sharpen the momentum.')
    parser.add_argument('--center_mm', type=float, default=0.9, help='The momentum for centering.')
    parser.add_argument('--temperature', type=float, default=0.04, help='The temperature for sharpening.')
    parser.add_argument('--lr_warmup_epochs', type=int, default=50, help='Warmup period for learning rate.')
    parser.add_argument('--epoch', type=int, default=500, help='LaplaceGNN number of epochs for training')
    parser.add_argument('--lapl_max_lr', type=float, default=0.5, help='augmentation learning rate for laplacian max strategy')
    parser.add_argument('--lapl_min_lr', type=float, default=0.5, help='augmentation learning rate for laplacian min strategy')
    parser.add_argument('--lapl_epoch', type=int, default=40, help='iteration for augmentation')
    parser.add_argument('--prob_feat', type=float, default=0.4, help='feature masking probability')  # if standard feature augmentations have been selected to be added 
    parser.add_argument('--threshold', type=float, default=0.2, help='threshold for edge perturbation')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--delta', type=float, default=8e-2, help='perturbation magnitude')
    parser.add_argument('--m', type=int, default=2, help='number of inner maximization steps')
    parser.add_argument('--step_size', type=float, default=8e-2, help='step size for inner maximization')
    return parser.parse_args()

def main():
    args = arg_parse()
    for arg, value in vars(args).items():
        print(f'{arg}: {value}')
    print(50*'-')   
    set_random_seeds(args.seed)
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else "cpu")
    # Load dataset
    path = osp.join(osp.expanduser('./data/'), 'datasets')
    dataset = TUDataset(path, name=args.dataset)

    ######################## AUGMENTATION ########################
    centrality_types = ['degree', 'pagerank', 'eigenvector']
    centrality_weights = [0.2, 0.3, 0.5] # randomly initialized, trainable params.
    # centrality_weights={'degree': 0.2, 'pagerank': 0.3, 'eigenvector': 0.5} # uncomment it if precoomputed centrality is being used.
    # precomputed_centrality = {
    #     'degree': nx.degree_centrality(to_networkx(data)),
    #     'pagerank': nx.pagerank(to_networkx(data)),
    #     'eigenvector': nx.eigenvector_centrality(to_networkx(data))
    # }

    L1_view = LaplaceGNN_Augmentation_Graph(
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

    L2_view = LaplaceGNN_Augmentation_Graph(
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
    else:  
        print('Laplacian perturbations under computation')
        assert dataset.len() > 1  # it's a graph classification
        loaded_laplacian_path = []
        for i in tqdm(range(dataset.len())):
            data = dataset.get(i)        
            L1_view.calc_prob(data, silence=True) # now max-laplacian has been encoded as data['max']=ptb_prob1
            L2_view.calc_prob(data, silence=True) # now min-laplacian has been encoded as data['min']=ptb_prob2
            loaded_laplacian_path.append(data)
        torch.save(loaded_laplacian_path, laplacian_path)
    
    # LaplaceGNN main loop
    dataloader = DataLoader(loaded_laplacian_path, batch_size=args.batch_size, shuffle=True)
    gconv1 = GConv(input_dim=dataset.num_features, hidden_dim=args.gnn1_dim, num_layers=args.gnn1_num_layers).to(device)
    gconv2 = GConv(input_dim=args.gnn1_dim, hidden_dim=args.gnn2_dim, num_layers=args.gnn2_num_layers).to(device)
    gcn_encoder = GCNEncoder(gconv1, gconv2).to(device)
    mlp1 = FC(input_dim=args.gnn2_dim, output_dim=args.mlp_dim)
    mlp2 = FC(input_dim=args.mlp_dim, output_dim=args.gnn2_dim)
    predictor = nn.Sequential(mlp1, mlp2).to(device)
    encoder_model = LaplaceGNN_Graph(gcn_encoder, predictor, augmentation=(L1_view,L2_view)).to(device)
    lr_scheduler = CosineDecayScheduler(args.lr, args.lr_warmup_epochs, args.epoch)
    mm_scheduler = CosineDecayScheduler(1 - args.mm, 0, args.epoch)
    optimizer = AdamW(encoder_model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay) 

    def train(step, dataloader):
            for data in dataloader:
                data = data.to(device)
                encoder_model.train()
                optimizer.zero_grad()
                lr = lr_scheduler.get(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                mm = 1 - mm_scheduler.get(step)
            
                accumulation_steps = args.accumulation_steps
                step_loss = 0  # to accumulate loss across steps

                perturb_shape_first = (data.x.shape[0], encoder_model.online_encoder.gconv1.layers[0].out_channels)  # first layer
                perturb_shape_last = (data.x.shape[0], encoder_model.online_encoder.gconv2.layers[-1].out_channels)   # second and last layer

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
                    # print(perturb_first.grad)

                    # inner maximization loop for perturbations
                    for _ in range(args.m):
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
            loss = train(epoch, dataloader)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, dataloader, device)
    
    print(f'Test accuracy={test_result["accuracy"]:.4f}')

if __name__ == '__main__':
    main()
