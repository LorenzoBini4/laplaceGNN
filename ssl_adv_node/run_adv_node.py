import copy
import logging
import os

from absl import app
from absl import flags
import torch
from torch.nn.functional import cosine_similarity
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch_geometric.nn import GCNConv
#from ../laplaceGNN import *
from ../laplaceGNN.model import *
from ../laplaceGNN.laplaceGNN import *
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
from ../laplaceGNN.utils import set_random_seeds
from ../laplaceGNNb.data import get_dataset, get_ogbn_arxiv, get_wiki_cs, get_citeseer,   get_cora, get_pubmed, get_ogbn_papers100M
from ../laplaceGNN.logistic_regression_eval import fit_logistic_regression_ogbn_arxiv_liblinear, fit_logistic_regression_ogbn_arxiv_adam, fit_logistic_regression_ogbn_paper100M_liblinear
from geomloss import SamplesLoss
from ../laplacian_augmentations.label_guide_ssl import *
from ../laplacian_augmentations.laplacian_node import *
import gc
from ../laplaceGNN.laplacian_evaluation import *

log = logging.getLogger(__name__)
FLAGS = flags.FLAGS
flags.DEFINE_integer('model_seed', 77, 'Random seed used for model initialization and training.')
flags.DEFINE_integer('data_seed', 7, 'Random seed used to generate train/val/test split.')
flags.DEFINE_integer('num_eval_splits', 3, 'Number of different train/test splits the model will be evaluated over.')

# Dataset.
flags.DEFINE_enum('dataset', 'coauthor-cs',
                  ['amazon-computers', 'amazon-photos', 'coauthor-cs', 'coauthor-physics', 'wiki-cs', 'ogbn-arxiv', 'cora', 'citeseer', 'pubmed', 'ogbn-papers100M'],
                  'Which graph dataset to use.')
flags.DEFINE_string('dataset_dir', './data', 'Where the dataset resides.')

# Architecture.
flags.DEFINE_multi_integer('graph_encoder_layer', None, 'Conv layer sizes.')
flags.DEFINE_integer('predictor_hidden_size', 512, 'Hidden size of projector.')

# Training hyperparameters.
flags.DEFINE_integer('epochs', 10000, 'The number of training epochs.')
flags.DEFINE_float('lr', 1e-5, 'The learning rate for model training.')
flags.DEFINE_float('weight_decay', 1e-2, 'The value of the weight decay for training.')
flags.DEFINE_float('mm', 0.99, 'The momentum for moving average.')
flags.DEFINE_bool('centering', False, 'Whether to center the momentum.')
flags.DEFINE_bool('sharpening', False, 'Whether to center the momentum.')
flags.DEFINE_float('center_mm', 0.9, 'Whether to center the momentum.')
flags.DEFINE_float('temperature', 0.04, 'Whether to center the momentum.')
flags.DEFINE_integer('lr_warmup_epochs', 1000, 'Warmup period for learning rate.')

# If classical augmentations want to be added.
flags.DEFINE_float('drop_edge_p_1', 0., 'Probability of edge dropout 1.')
flags.DEFINE_float('drop_feat_p_1', 0., 'Probability of node feature dropout 1.')
flags.DEFINE_float('drop_edge_p_2', 0., 'Probability of edge dropout 2.')
flags.DEFINE_float('drop_feat_p_2', 0., 'Probability of node feature dropout 2.')

# Evaluation and Saving
flags.DEFINE_integer('eval_epochs', 5, 'Evaluate every eval_epochs.')
flags.DEFINE_string('logdir', None, 'Where the checkpoint and logs are stored.')

class Args:
    delta = 8e-3 
    step_size = 8e-3 
    m = 3 
    accumulation_steps = 2 # 2-4 accumulation steps is a good starting point to simulate a larger batch size.
                        # if smaller batch sizes, may increase this to 6-8 steps.
                        # if learning rate is high, gradient accumulation over more steps can help stabilize training 
                        # by averaging gradients over a larger number of samples.
    weight_decay = 8e-4 
    lapl_max_lr = 100
    lapl_min_lr = 0.1
    lapl_epoch = 10 #20
    # prob_feat = 0.4 # if standard feature augmentations have been selected to be added 
    treshold = 0.3
args = Args()

def main(argv):
    # use CUDA_VISIBLE_DEVICES to select gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info('Using {} for training.'.format(device))

    # set random seed
    if FLAGS.model_seed is not None:
        log.info('Random seed set to {}.'.format(FLAGS.model_seed))
        set_random_seeds(random_seed=FLAGS.model_seed)

    # load data
    if FLAGS.dataset != 'wiki-cs' and FLAGS.dataset != 'ogbn-arxiv' and FLAGS.dataset != 'cora' and FLAGS.dataset != 'citeseer' and FLAGS.dataset != 'pubmed' and FLAGS.dataset != 'ogbn-papers100M':
        dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset)
        num_eval_splits = FLAGS.num_eval_splits
    elif FLAGS.dataset == 'wiki-cs':
        dataset, train_masks, val_masks, test_masks = get_wiki_cs(FLAGS.dataset_dir)
        num_eval_splits = train_masks.shape[1]
    elif FLAGS.dataset == 'cora':
        dataset, train_masks, val_masks, test_masks = get_cora(FLAGS.dataset_dir)
        num_eval_splits = FLAGS.num_eval_splits
    elif FLAGS.dataset == 'pubmed':
        dataset, train_masks, val_masks, test_masks = get_pubmed(FLAGS.dataset_dir)
        num_eval_splits = FLAGS.num_eval_splits
    elif FLAGS.dataset == 'citeseer':
        dataset, train_masks, val_masks, test_masks = get_citeseer(FLAGS.dataset_dir)
        num_eval_splits = FLAGS.num_eval_splits   
    elif FLAGS.dataset == 'ogbn-papers100M':
        dataset, train_masks, val_masks, test_masks = get_ogbn_papers100M(FLAGS.dataset_dir)
        print(f"train_masks: {train_masks.shape}, val_masks: {val_masks.shape}, test_masks: {test_masks.shape}")
        print(f"train_masks: {train_masks}, val_masks: {val_masks}, test_masks: {test_masks}")
        num_eval_splits = FLAGS.num_eval_splits
    else:
        dataset, train_masks, val_masks, test_masks = get_ogbn_arxiv(FLAGS.dataset_dir)
        print(f"train_masks: {train_masks.shape}, val_masks: {val_masks.shape}, test_masks: {test_masks.shape}")
        print(f"train_masks: {train_masks}, val_masks: {val_masks}, test_masks: {test_masks}")
        num_eval_splits = FLAGS.num_eval_splits

    print(f"--drop_edge_p_1={FLAGS.drop_edge_p_1}")
    print(f"--drop_feat_p_1={FLAGS.drop_feat_p_1}")
    print(f"--drop_edge_p_2={FLAGS.drop_edge_p_2}")
    print(f"--drop_feat_p_2={FLAGS.drop_feat_p_2}")
    print(f"centering={FLAGS.centering}")
    print(f"sharpening={FLAGS.sharpening}")
    print(f"--center_mm={FLAGS.center_mm}")
    print(f"--temperature={FLAGS.temperature}")
    
    ######################### LaplacianGNN-Max-Min Spectral Augmentation Module ############################
    # Initialize centrality-guided augmentors
    data = dataset[0]
    data = data.to(device)  # permanently move in gpu memory
    centrality_types = ['degree', 'pagerank', 'eigenvector']
    centrality_weights = [0.2, 0.3, 0.5] # randomly initialized, trainable params.
    # centrality_weights={'degree': 0.2, 'pagerank': 0.3, 'eigenvector': 0.5} # uncomment it if precoomputed centrality is being used.

    # precomputed_centrality = {
    #     'degree': nx.degree_centrality(to_networkx(data)),
    #     'pagerank': nx.pagerank(to_networkx(data)),
    #     'eigenvector': nx.eigenvector_centrality(to_networkx(data))
    # }

    L1_view = LaplaceGNN_Augmentation_Node(
        ratio=args.treshold,
        lr=args.lapl_max_lr,
        iteration=args.lapl_epoch,
        dis_type='max',
        device=device,
        centrality_types=centrality_types,
        centrality_weights=centrality_weights,
        precomputed_centrality=None,
        sample='no'
    )

    L2_view = LaplaceGNN_Augmentation_Node(
        ratio=args.treshold,
        lr=args.lapl_min_lr,
        iteration=args.lapl_epoch,
        dis_type='min',
        device=device,
        centrality_types=centrality_types,
        centrality_weights=centrality_weights,
        precomputed_centrality=None,
        sample='no'
    )

    x1=L1_view.calc_prob(data, precomputed_centrality=None) # now data contains data['max']=ptb_prob1
    # del x1
    torch.cuda.empty_cache()
    gc.collect()  
    x2=L2_view.calc_prob(data, precomputed_centrality=None) # note data contains data['min']=ptb_prob2
    torch.cuda.empty_cache()
    gc.collect() 
    
    # # data = modify_graph_v2_dc(dataset, percentage=2)
    print(f'Dataset used: {FLAGS.dataset}')
    aug1 = Compose([L1_view]) #, FeatureAugmentor(pf=args.pf)]) --> to add standard feat augmentations as well
    aug2 = Compose([L2_view]) #, FeatureAugmentor(pf=args.pf)]) --> to add standard feat augmentations as well

    # prepare transforms
    transform_1 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_1, drop_feat_p=FLAGS.drop_feat_p_1)
    transform_2 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_2, drop_feat_p=FLAGS.drop_feat_p_2)
    # x1, x2 = modify_graph_v2_dc(dataset, percentage=2), modify_graph_v2_pc(dataset, percentage=2)
    x1, x2 = x1.to(device), x2.to(device)
    ####################################################################
    
    # build teacher-student encoders networks
    input_size, representation_size = data.x.size(1), FLAGS.graph_encoder_layer[-1]
    if FLAGS.dataset in ['ogbn-arxiv', 'ogbn-papers-100M']:
        encoder = Encoder_Adversarial_GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=False, layernorm=True, weight_standardization=True)
    else:
        encoder = Encoder_Adversarial_GraphSAGE([input_size] + FLAGS.graph_encoder_layer, batchnorm=True, layernorm=False, weight_standardization=False)
        # encoder = Encoder_LaplaceGNN_GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=True, layernorm=False, weight_standardization=False)
    predictor = MLP_Predictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size)
    model = LaplaceGNN_v1(encoder, predictor).to(device)
    # model = LaplaceGNN_v2(encoder, predictor, augmentor=(aug1,aug2)).to(device)
    print(model.online_encoder.model)
    print(model.predictor)

    optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=args.weight_decay)  # FLAGS.weight_decay
    print(f"m: {args.m}, step_size: {args.step_size}, delta: {args.delta}, accumulation_steps: {args.accumulation_steps} , lr={FLAGS.lr}, weight_decay={args.weight_decay}")
    print(f'Model seed: {FLAGS.model_seed}, Data seed: {FLAGS.data_seed}') 
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)
    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.epochs)

    def train_laplacian_v1(step, x1, x2):
        torch.cuda.empty_cache()
        model.train()
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        mm = 1 - mm_scheduler.get(step)
        optimizer.zero_grad()
        x1, x2 = transform_1(x1), transform_2(x2)
        # set up gradient accumulation
        accumulation_steps = args.accumulation_steps
        step_loss = 0  # to accumulate loss across steps

        perturb_shape_first = (x1.x.shape[0], model.online_encoder.model[0].out_channels)  # first layer
        perturb_shape_last = (x2.x.shape[0], model.online_encoder.model[3].out_channels)   # second and last layer

        perturb_first = torch.Tensor(*perturb_shape_first).uniform_(-args.delta, args.delta).to(device)
        perturb_first.requires_grad_()

        perturb_last = torch.Tensor(*perturb_shape_last).uniform_(-args.delta, args.delta).to(device)
        perturb_last.requires_grad_()

        # print(perturb_first.requires_grad)
        # print(perturb_last.requires_grad)
        # print(perturb_first)
        # print(perturb_last)
        # print(perturb_first.grad)
        # print(perturb_last.grad)

        x1 = x1.to(device)
        x2 = x2.to(device)

        if perturb_first is not None:
            perturb_first = perturb_first.to(device)
        if perturb_last is not None:
            perturb_last = perturb_last.to(device)

        for acc_step in range(accumulation_steps):
            if acc_step > 0:
                optimizer.zero_grad()

            q1, y2 = model(x1, x2, perturb_first=perturb_first, perturb_last=None)
            q2, y1 = model(x2, x1, perturb_first=perturb_first, perturb_last=None)

            loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
            # loss = torch.nn.functional.mse_loss(q1, y2.detach()) + torch.nn.functional.mse_loss(q2, y1.detach())
            # inner maximization loop for perturbations
            for _ in range(args.m -1):
                loss.backward(retain_graph=True)
                with torch.no_grad():
                    perturb_first.data += args.step_size * torch.sign(perturb_first.grad)
                    perturb_first.data = perturb_first.data.clamp(-args.delta, args.delta)

                    # perturb_last.data += args.step_size * torch.sign(perturb_last.grad)
                    # perturb_last.data = perturb_last.data.clamp(-args.delta, args.delta)

                perturb_first.grad.zero_()
                # perturb_last.grad.zero_()

                q1, y2 = model(x1, x2, perturb_first=perturb_first, perturb_last=None)
                q2, y1 = model(x2, x1, perturb_first=perturb_first, perturb_last=None)
                
                loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
                # loss = torch.nn.functional.mse_loss(q1, y2.detach()) + torch.nn.functional.mse_loss(q2, y1.detach())
            # for the last backward call in the accumulation loop, we don't retain the graph
            if acc_step == accumulation_steps - 1:
                loss.backward()
            else:
                loss.backward(retain_graph=True)
            # accumulate the loss
            step_loss += loss / accumulation_steps

            # optimizer step only after accumulation_steps iterations
            if (acc_step + 1) % accumulation_steps == 0:
                optimizer.step()
                model.update_target_network(mm, centering=FLAGS.centering, sharpening=FLAGS.sharpening, center_momentum=FLAGS.center_mm, temperature=FLAGS.temperature),  

        # return loss.item()
        print(f'Step: {step}, Loss: {loss.item()}') #, Learning Rate: {lr}, Momentum: {mm}')

    def train_laplacian_v2(step, data):
        model.train()
        optimizer.zero_grad()
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        mm = 1 - mm_scheduler.get(step)
    
        accumulation_steps = args.accumulation_steps
        step_loss = 0  # to accumulate loss across steps

        perturb_shape_first = (data.x.shape[0], model.online_encoder.model[0].out_channels)  # first layer
        perturb_shape_last = (data.x.shape[0], model.online_encoder.model[3].out_channels)   # second and last layer

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

            q1, y2 = model(data, perturb_first=perturb_first, perturb_last=None) 
            q2, y1 = model(data, perturb_first=perturb_first, perturb_last=None)

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
                q1, y2 = model(data, perturb_first=perturb_first, perturb_last=None)
                q2, y1 = model(data, perturb_first=perturb_first, perturb_last=None) 
                loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
            # for the last backward call in the accumulation loop, we don't retain the graph
            if acc_step == accumulation_steps - 1:
                loss.backward()
            else:
                loss.backward(retain_graph=True)
            step_loss += loss / accumulation_steps
            if (acc_step + 1) % accumulation_steps == 0:
                optimizer.step()
                model.update_target_network(mm, centering=args.centering, sharpening=args.sharpening, center_momentum=args.center_mm, temperature=args.temperature),  
        print(f'Step: {step}, Loss: {loss.item()}') #, Learning Rate: {lr}, Momentum: {mm}')
        return step_loss.item()
    
    best_accuracy = 0.0  # Initialize best_accuracy at the beginning
    def eval_laplacian_v1(epoch):
        # make temporary copy of encoder
        global best_accuracy
        tmp_encoder = copy.deepcopy(model.online_encoder).eval()
        representations, labels = compute_representations(tmp_encoder, dataset, device)

        if FLAGS.dataset in ['coauthor-cs', 'amazon-computers', 'amazon-photos', 'coauthor-physics']:
            scores = fit_logistic_regression(representations.cpu().numpy(), labels.cpu().numpy(), data_random_seed=FLAGS.data_seed, repeat=FLAGS.num_eval_splits)
            for i, score in enumerate(scores):
                print(f'Epoch: {epoch}, Split: {i}, Accuracy: {score}')
                # best_accuracy = max(best_accuracy, score)  

        elif FLAGS.dataset in ['wiki-cs', 'cora', 'citeseer', 'pubmed']:
            print("Using fit_logistic_regression_liblinear")
            scores = fit_logistic_regression_preset_splits(representations.cpu().numpy(), labels.cpu().numpy(), train_masks, val_masks, test_masks)
            for i, score in enumerate(scores):
                print(f'Epoch: {epoch}, Split: {i}, Accuracy: {score}')
                # best_accuracy = max(best_accuracy, score)  
                # print(f'Best Accuracy Obtained: {best_accuracy}')
        elif FLAGS.dataset == 'ogbn-arxiv':
            print("Using fit_logistic_regression_ogbn_arxiv_adam")
            scores = fit_logistic_regression_ogbn_arxiv_liblinear(representations.cpu().numpy(), labels.cpu().numpy(), train_masks, val_masks, test_masks)
            # score = fit_logistic_regression_ogbn_arxiv_adam(representations.cpu().numpy(), labels.cpu().numpy(), train_masks, val_masks, test_masks)
            # print(f'Epoch: {epoch}, Accuracy: {score}')
            # best_accuracy = max(best_accuracy, score)  
        else:
            print("Using fit_logistic_regression_liblinear")
            scores = fit_logistic_regression_ogbn_paper100M_liblinear(representations.cpu().numpy(), labels.cpu().numpy(), train_masks, val_masks, test_masks)
            for i, score in enumerate(scores):
                print(f'Epoch: {epoch}, Split: {i}, Accuracy: {score}')
                # best_accuracy = max(best_accuracy, score)  

    def eval_laplacian_v2(model, data):
        model.eval()
        # x, edge_index = data.x, data.edge_index
        z = model.online_encoder(data)
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
            result = LREvaluator(weight_decay=decay)(z, data.y, split)
            if result['accuracy_val'] > best_result['accuracy_val']:
                best_result = result
        return best_result
    
    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        train_laplacian_v1(epoch - 1, x1, x2)    
        # train_laplacian_v2(epoch, data)
        if epoch % FLAGS.eval_epochs == 0:
            eval_laplacian_v1(epoch)
            # test_result = eval_laplacian_v2(model, data)
            # print(f'(E): Test accuracy={test_result["accuracy"]:.4f}')
        
    # save encoder weights
    # torch.save({'model': model.online_encoder.state_dict()}, os.path.join(FLAGS.logdir, 'laplaceGNN.pt'))
    # print()

if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
