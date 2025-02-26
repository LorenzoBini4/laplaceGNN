import logging
import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import os
import argparse
import time
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pdb
from tqdm import tqdm
from tqdm import trange
import sys
# OGB datasets
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from LaplaceGNN4Graph import LaplaceGNN_Graph
from models_ogb import GNN
from transforms import *
from ..augmentations_graph import *

parser = argparse.ArgumentParser(description='GNN baselines on ogbg data with PyG')
parser.add_argument('--gnn', type=str, default='gcn',
                    help='GNN gin, or gcn (default: gin)')
parser.add_argument('--drop_ratio', type=float, default=0,
                    help='dropout ratio (default: 0.5)')
parser.add_argument('--decay', type=float, default=0.99,
                    help='moving_average_decay (default: 0.99)')
parser.add_argument('--num_layer', type=int, default=2,
                    help='number of GNN layers (default: 5)')
parser.add_argument('--emb_dim', type=int, default=512,
                    help='GNNs hidden dimension (default: 512)')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers (default: 0)')
parser.add_argument('--dataset', type=str, default="ogbg-moltox21",
                    help='dataset name (default: ogbg-molbbbp, ogbg-molhiv, ogbg-moltoxcast)')
parser.add_argument('--pp', type=str, default="H",
                    help='perturb_position (default: X(feature), H(hidden layer))')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--hidden_channels', type=int, default=512, help='MLP hidden dim, default:512')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--step_size', type=float, default=8e-3)
parser.add_argument('--delta', type=float, default=8e-3)
parser.add_argument('--m', type=int, default=3)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--num_tasks', type=int, default=512) # for evaluating on standard protocol
parser.add_argument('--projection_hidden_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=77)
parser.add_argument('--projection_size', type=int, default=512)
parser.add_argument('--prediction_size', type=int, default=512)
parser.add_argument('--drop_edge_p_1', type=float, default=0.1)
parser.add_argument('--drop_feat_p_1', type=float, default=0.1)
parser.add_argument('--drop_edge_p_2', type=float, default=0.3)
parser.add_argument('--drop_feat_p_2', type=float, default=0.1)
parser.add_argument('--lapl_max_lr', type=float, default=100, help='augmentation learning rate for laplacian max strategy')
parser.add_argument('--lapl_min_lr', type=float, default=0.1, help='augmentation learning rate for laplacian min strategy')
parser.add_argument('--lapl_epoch', type=int, default=10, help='iteration for augmentation')
parser.add_argument('--prob_feat', type=float, default=0.4, help='feature masking probability')  # if standard feature augmentations have been selected to be added 
parser.add_argument('--threshold', type=float, default=0.3, help='threshold for edge perturbation')

args = parser.parse_args()
print("Arguments:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
print(50*"-")
transform_1 = get_graph_drop_transform(drop_edge_p=args.drop_edge_p_1, drop_feat_p=args.drop_feat_p_1)
transform_2 = get_graph_drop_transform(drop_edge_p=args.drop_edge_p_2, drop_feat_p=args.drop_feat_p_2)

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

def train_adv_bootstrap(model, device, loader, optimizer, task_type, args, L1_view, L2_view, feat_augmentation=None):
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            continue
        ptb_prob1 = batch.max  # Precomputed max probability
        ptb_prob2 = batch.min  # Precomputed min probability

        L1_view  = Compose([L1_view, feat_augmentation])
        L2_view  = Compose([L2_view, feat_augmentation])

        # Create augmented batches
        x1, edge_index1, _ = L1_view(batch.x, batch.edge_index, ptb_prob1, batch=batch.batch)
        x2, edge_index2, _ = L2_view(batch.x, batch.edge_index, ptb_prob2, batch=batch.batch)

        # Create new data objects with augmented features and edges
        batch_1 = batch.clone()
        batch_1.x = x1
        batch_1.edge_index = edge_index1

        batch_2 = batch.clone()
        batch_2.x = x2
        batch_2.edge_index = edge_index2
        batch_1, batch_2 = transform_1(batch), transform_2(batch)

        model.train()
        optimizer.zero_grad()

        perturb_shape = (batch_1.x.shape[0], args.emb_dim)
        perturb = torch.FloatTensor(*perturb_shape).uniform_(-args.delta, args.delta).to(device)
        perturb.requires_grad_()
        loss = model(batch_1, batch_2, perturb)

        for _ in range(args.m - 1):
            loss.backward()
            perturb_data = perturb.detach() + args.step_size * torch.sign(perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0

            loss = model(batch_1, batch_2, perturb)
            loss /= args.m

        total_loss += loss.item()
        loss.backward()
        model.update_moving_average()
        optimizer.step()
    return total_loss / len(loader)

def evaluation(model, logreg, linear_optimizer, device, train_loader, val_loader, test_loader, evaluator, metric):
    optimizer = linear_optimizer
    cls_criterion = torch.nn.BCEWithLogitsLoss()
    best_train = best_val = best_test = 0
    with trange(100) as t:
        for epoch in t:
            t.set_description('epoch %i' % epoch)

            for step, batch in enumerate(train_loader):
                batch = batch.to(device)
                optimizer.zero_grad()
                is_labeled = batch.y == batch.y

                batch_embed = model.embed(batch)
                logits = logreg(batch_embed)
                loss = cls_criterion(logits.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                train_perf = eval_model(model, logreg, device, train_loader, evaluator)
                val_perf = eval_model(model, logreg, device, val_loader, evaluator)
                test_perf = eval_model(model, logreg, device, test_loader, evaluator)

                tra, val, tst = (train_perf[metric], val_perf[metric], test_perf[metric])

                if val > best_val:
                    best_train, best_val, best_test = tra, val, tst

                print(f'epoch:{epoch} Train:{tra:9.5f} val:{val:9.5f}, test:{tst:9.5f}')
                t.set_postfix(tra=best_train, val=best_val, test=best_test)

    print(f'Train:{best_train:9.5f} val:{best_val:9.5f}, test:{best_test:9.5f}')
    return (best_train, best_val, best_test)

def get_one_hot(dataset):
    g_idx = 0
    total_node = 0
    for i in dataset.data.num_nodes:
        total_node += i
    total_degree = np.zeros(total_node)
    node_start = 0
    node_end = 0
    for i in dataset.data.num_nodes:
        node_end += i
        edge_start = dataset.slices['edge_index'][g_idx]
        edge_end = dataset.slices['edge_index'][g_idx+1]
        edges = dataset.data.edge_index[:, edge_start:edge_end]
        in_degree = out_degree = np.zeros(i)

        for ee in edges:
            in_degree[ee] += 1
            out_degree[ee] += 1
        tot_degree = in_degree + out_degree
        total_degree[node_start:node_end] = tot_degree
        node_start = node_end
        g_idx += 1


    total_degree = total_degree.astype(np.int64)
    return torch.nn.functional.one_hot(torch.tensor(torch.from_numpy(total_degree))).float()

def eval_model(model, logreg, device, loader, evaluator):
    model.eval()
    logreg.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            batch_embed = model.embed(batch)
            logits = logreg(batch_embed)
        y_true.append(batch.y.view(logits.shape).detach().cpu())
        y_pred.append(logits.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)

def main():
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    ### protocol dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)
    
    print(f"Dataset: {dataset}")
    print(f"First Graph: {dataset[0]}")
    print(f"Last Graph: {dataset[-1]}")
    print(f"Number of graphs: {len(dataset)}")
    num_nodes = [data.num_nodes for data in dataset]
    num_edges = [data.num_edges for data in dataset]
    avg_num_nodes = sum(num_nodes) / len(num_nodes)
    avg_num_edges = sum(num_edges) / len(num_edges)
    print(f"Average number of nodes per graph: {avg_num_nodes}")
    print(f"Average number of edges per graph: {avg_num_edges}")
    # unique_labels = dataset._data.y.unique()
    # print(f"Number of unique labels: {len(unique_labels)}")
    # print(f"Unique labels: {unique_labels}")
    if dataset._data.x is not None:
        print(f"Number of node features: {dataset._data.x.shape[1]}")
    else:
        print("No node features")
    if dataset._data.edge_attr is not None:
        print(f"Number of edge features: {dataset._data.edge_attr.shape[1]}")
    else:
        print("No edge features")
    print(f"Number of tasks: {dataset.num_tasks}")
    print(f"Task type: {dataset.task_type}")
    split_idx = dataset.get_idx_split()
    print(f"Number of graphs in train set: {len(split_idx['train'])}")
    print(f"Number of graphs in validation set: {len(split_idx['valid'])}")
    print(f"Number of graphs in test set: {len(split_idx['test'])}")
    # exit()

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

    print('Precomputing probability for augmentation')
    assert dataset.len() > 1  # should have multiple data object for graph classification
    updated_dataset = []
    for i in tqdm(range(dataset.len())):
        data = dataset.get(i)        
        #print(f'dataset(0): {dataset.get(0)}')
        data = L1_view.calc_prob(data, silence=True) # note that data is updated with data['max']=ptb_prob1
        #print(f'data.max = {data.max}')
        data = L2_view.calc_prob(data, silence=True) # note that data is further updated with data['min']=ptb_prob2
        #print(f'data.min = {data.min}')
        updated_dataset.append(data)  # Add the modified data object to the new dataset
        # print(batch_1, batch_2)
    print('Done precomputing probability for augmentation')
    #print(updated_dataset[0])
    output_dir = './laplacian_dataset'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset_name = f'laplacian_{args.dataset}'
    output_file = os.path.join(output_dir, f'{dataset_name}.pt')
    torch.save(updated_dataset, output_file)
    #exit()
    # Save the updated data to re-use in future
    ######################## LOAD AUGMENTATION ########################
    dataset_path = os.path.join(output_dir, f'laplacian_{args.dataset}.pt')
    if os.path.exists(dataset_path):
        updated_dataset = torch.load(dataset_path)
        print(f"Loaded precomputed dataset from {dataset_path}")
    else:
        print(f"Precomputed dataset not found at {dataset_path}. Please run the precomputation step first.")
    ######################## LOAD AUGMENTATION ########################

    if dataset.data.x is None:
        dataset.data.x = get_one_hot(dataset)
    all_idx = torch.tensor(range(0, len(dataset)))  # train on all data
    all_loader = DataLoader(dataset[all_idx], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    if 'x' not in all_loader.dataset.slices:
        tmp = torch.LongTensor(len(dataset.data.num_nodes)+1)
        accum_node = 0
        tmp[0] = 0
        for i in range(len(dataset.data.num_nodes)):
            accum_node += dataset.data.num_nodes[i]
            tmp[i+1] = accum_node
        all_loader.dataset.slices['x'] = tmp

    if dataset.data.x is not None:
        feat_dim = dataset.data.x.shape[-1]
    best_result = -1
    all_results = []
    seeds = [args.seed]

    best_acc = -1
    best_std = -1
    best_results = []
    trains, vals, tests = [], [], []
    for run in range(len(seeds)):
        best_train, best_val, best_test = 0, 0, 0
        seed = seeds[run]
        torch.manual_seed(seed)
        np.random.seed(seed)
        if args.gnn == 'gin':
            encoder = GNN(gnn_type='gin', num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
                           drop_ratio=args.drop_ratio, feat_dim=feat_dim, perturb_position=args.pp).to(device)
        elif args.gnn == 'gcn':
            encoder = GNN(gnn_type='gcn', num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
                           drop_ratio=args.drop_ratio, feat_dim=feat_dim, perturb_position=args.pp).to(device)
        else:
            raise ValueError('Invalid GNN-encoder type')

        model = LaplaceGNN_Graph(encoder, num_tasks=dataset.num_tasks, emb_dim = args.emb_dim, projection_size=args.projection_size,
                     prediction_size=args.prediction_size, projection_hidden_size=args.projection_hidden_size,
                     moving_average_decay = args.decay
                     )
        
        model.to(device)
        logreg = MLP(args.emb_dim, args.hidden_channels, dataset.num_tasks)
        # logreg = LogisticRegression(args.emb_dim, dataset.num_tasks)
        logreg = logreg.to(device)
        split_idx = dataset.get_idx_split()
        ### protocol evaluator
        evaluator = Evaluator(args.dataset)

        train_dataset = [updated_dataset[i] for i in split_idx['train'].tolist()]
        val_dataset = [updated_dataset[i] for i in split_idx['valid'].tolist()]
        test_dataset = [updated_dataset[i] for i in split_idx['test'].tolist()]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in tqdm(range(1, args.epochs+1)):
            loss = train_adv_bootstrap(model, device, train_loader, optimizer, None, args, L1_view, L2_view, feat_augmentation=FeatAugmentation(pf=args.prob_feat))
            if epoch % args.test_freq == 0 or epoch == args.epochs:
                linear_optimizer = torch.optim.Adam(logreg.parameters(), lr=3e-4)
                result = evaluation(model, logreg, linear_optimizer, device, train_loader, val_loader, test_loader, evaluator, dataset.eval_metric)
                tra, val, tst = result
                if val > best_val:
                    best_train, best_val, best_test = result
                print(f'Train:{best_train:9.5f} val:{best_val:9.5f} test:{best_test:9.5f}')
        trains.append(best_train)
        vals.append(best_val)
        tests.append(best_test)
    print('')
    print(f"Average train accuracy: {np.mean(trains)}  {np.std(trains)}")
    print(f"Average val accuracy: {np.mean(vals)}  {np.std(vals)}")
    print(f"Average test accuracy: {np.mean(tests)}  {np.std(tests)}")
if __name__ == "__main__":
    main()
    