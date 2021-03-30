from glob import glob
import sys
import os
import os.path as osp
import numpy as np
import pandas as pd
import pickle, math, time, numba, tqdm

#Check if the GPU configuration has been provided
try:
    if not ("CUDA_VISIBLE_DEVICES" in os.environ):
        import setGPU
except Exception as e:
    print("Could not import setGPU, running CPU-only")

import torch
use_gpu = torch.cuda.device_count()>0
multi_gpu = torch.cuda.device_count()>1

if multi_gpu:
    print('Will use multi_gpu..')

#define the global base device
if use_gpu:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print("Working on device:", device)

import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv, MessagePassing, EdgePooling, GATConv, GCNConv, JumpingKnowledge, GraphUNet, DynamicEdgeConv, DenseGCNConv
from torch_geometric.nn import TopKPooling, SAGPooling, SGConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn.inits import reset
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from gravnet import GravNetConv
from torch_geometric.data import Data, DataListLoader, Batch
from torch.utils.data import random_split

import torch_cluster

import matplotlib, mplhep
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix

import args
from args import parse_args

from model import PFNet7
from graph_data_delphes import PFGraphDataset, one_hot_embedding
from data_preprocessing import data_to_loader
import evaluate
from evaluate import make_plots, Evaluate

#Ignore divide by 0 errors
np.seterr(divide='ignore', invalid='ignore')

#Get a unique directory name for the model
def get_model_fname(dataset, model, n_train, lr, target_type):
    model_name = type(model).__name__
    model_params = sum(p.numel() for p in model.parameters())
    import hashlib
    model_cfghash = hashlib.blake2b(repr(model).encode()).hexdigest()[:10]
    model_user = os.environ['USER']

    model_fname = '{}_{}_ntrain_{}'.format(
        model_name,
        target_type,
        n_train)
    return model_fname


def mse_loss(input, target):
    return torch.sum((input - target) ** 2)

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target).sum(axis=1) ** 2)

def compute_weights(target_ids, device):
    vs, cs = torch.unique(target_ids, return_counts=True)
    weights = torch.zeros(output_dim_id).to(device=device)
    for k, v in zip(vs, cs):
        weights[k] = 1.0/math.sqrt(float(v))
    return weights

def make_plot_from_list(l, label, xlabel, ylabel, outpath):
    fig, ax = plt.subplots()
    ax.plot(range(len(l)), l, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    plt.savefig(outpath + '/' + str(l) + '.png')
    plt.close(fig)

    with open(outpath + '/' + str(l) + '.pkl', 'wb') as f:
        pickle.dump(l, f)

@torch.no_grad()
def test(model, loader, epoch, l1m, l2m, l3m, target_type, device):
    with torch.no_grad():
        ret = train(model, loader, epoch, None, l1m, l2m, l3m, target_type, None)
    return ret


def train(model, loader, epoch, optimizer, l1m, l2m, l3m, target_type, device):

    is_train = not (optimizer is None)

    if is_train:
        model.train()
    else:
        model.eval()

    #loss values for each batch: classification, regression
    losses_1 = []
    losses_2 = []
    losses_tot = []

    #accuracy values for each batch (monitor classification performance)
    accuracies_batch = []
    accuracies_batch_msk = []
    accuracies_batch_msk2 = []

    #correlation values for each batch (monitor regression performance)
    corrs_batch = np.zeros(len(loader))

    #epoch confusion matrix
    conf_matrix = np.zeros((output_dim_id, output_dim_id))

    #keep track of how many data points were processed
    num_samples = 0

    for i, batch in enumerate(loader):
        t0 = time.time()

        # for better reading of the code
        if args.target == "cand":
            X = batch.to(device)
            target_ids = batch.ycand_id.to(device)
            target_p4 = batch.ycand.to(device)

        # forwardprop
        cand_ids, cand_p4, new_edge_index = model(X)

        # BACKPROP
        # (1) Predictions where both the predicted and true class label was nonzero
        # In these cases, the true candidate existed and a candidate was predicted
        # msk is a list of booleans of shape [~5000*batch_size] where each boolean correspond to whether a candidate was predicted
        _, indices = torch.max(cand_ids, -1)     # picks the maximum PID location and stores the index (opposite of one_hot_embedding)
        _, target_ids_msk = torch.max(target_ids, -1)
        msk = ((indices != 0) & (target_ids_msk != 0))
        msk2 = ((indices != 0) & (indices == target_ids_msk))

        # (2) computing losses
        weights = compute_weights(torch.max(target_ids,-1)[1], device)
        l1 = l1m * torch.nn.functional.cross_entropy(target_ids, indices, weight=weights)
        l2 = l2m * torch.nn.functional.mse_loss(target_p4[msk2], cand_p4[msk2])
        loss = l1 + l2

        losses_1.append(l1.item())
        losses_2.append(l2.item())
        losses_tot.append(loss.item())

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t1 = time.time()

        num_samples += len(cand_ids)

        accuracies_batch.append(accuracy_score(target_ids_msk.detach().cpu().numpy(), indices.detach().cpu().numpy()))
        accuracies_batch_msk.append(accuracy_score(target_ids_msk[msk].detach().cpu().numpy(), indices[msk].detach().cpu().numpy()))
        accuracies_batch_msk2.append(accuracy_score(target_ids_msk[msk2].detach().cpu().numpy(), indices[msk2].detach().cpu().numpy()))

        print('{}/{} batch_loss={:.2f} dt={:.1f}s'.format(i, len(loader), loss.item(), t1-t0), end='\r', flush=True)

        #Compute correlation of predicted and true pt values for monitoring
        corr_pt = 0.0
        if msk.sum()>0:
            corr_pt = np.corrcoef(
                cand_p4[msk, 0].detach().cpu().numpy(),
                target_p4[msk, 0].detach().cpu().numpy())[0,1]

        corrs_batch[i] = corr_pt

        conf_matrix += confusion_matrix(target_ids_msk.detach().cpu().numpy(),
                                        np.argmax(cand_ids.detach().cpu().numpy(),axis=1), labels=range(6))

    corr = np.mean(corrs_batch)

    acc = np.mean(accuracies_batch)
    acc_msk = np.mean(accuracies_batch_msk)
    acc_msk2 = np.mean(accuracies_batch_msk2)

    losses_1 = np.mean(losses_1)
    losses_2 = np.mean(losses_2)
    losses_tot = np.mean(losses_tot)

    return num_samples, losses_tot, acc, conf_matrix


def train_loop():
    t0_initial = time.time()

    losses_1_train = []
    losses_2_train = []
    losses_tot_train = []

    losses_1_valid = []
    losses_2_valid = []
    losses_tot_valid = []

    accuracies_train = []
    accuracies_valid = []

    conf_matrix_train = np.zeros((output_dim_id, output_dim_id))
    conf_matrix_valid = np.zeros((output_dim_id, output_dim_id))

    best_val_loss = 99999.9
    stale_epochs = 0

    print("Training over {} epochs".format(args.n_epochs))
    for epoch in range(args.n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            print("breaking due to stale epochs")
            break

        # training epoch
        model.train()
        num_samples, losses_tot, acc, conf_matrix = train(model, train_loader, epoch, optimizer, args.l1, args.l2, args.l3, args.target, device)

        losses_tot_train.append(losses_tot)
        accuracies_train.append(acc)
        conf_matrix_train += conf_matrix

        # validation step
        model.eval()
        num_samples_val, losses_tot_v, acc_v, conf_matrix_v = test(model, valid_loader, epoch,
                                                                    args.l1, args.l2, args.l3, args.target, device)
        losses_tot_valid.append(losses_tot_v)
        accuracies_valid.append(acc_v)
        conf_matrix_valid += conf_matrix_v

        # early-stopping
        if losses_tot_v < best_val_loss:
            best_val_loss = losses_tot_v
            stale_epochs = 0
        else:
            stale_epochs += 1

        t1 = time.time()

        epochs_remaining = args.n_epochs - (epoch+1)
        time_per_epoch = (t1 - t0_initial)/(epoch + 1)
        eta = epochs_remaining*time_per_epoch/60

        print("epoch={}/{} dt={:.2f}s train_loss={:.5f} valid_loss={:.5f} train_acc={:.5f} valid_acc={:.5f} stale={} eta={:.1f}m".format(
            epoch, args.n_epochs,
            t1 - t0, losses_tot_train[epoch], losses_tot_valid[epoch], accuracies_train[epoch], accuracies_valid[epoch],
            stale_epochs, eta))

        torch.save(model.state_dict(), "{0}/epoch_{1}_weights.pth".format(outpath, epoch))

    make_plot_from_list(losses_tot_train, 'train loss', 'Epochs', 'Loss', outpath)
    make_plot_from_list(losses_tot_valid, 'valid loss', 'Epochs', 'Loss', outpath)
    make_plot_from_list(accuracies_train, 'train accuracy', 'Epochs', 'Accuracy', outpath)
    make_plot_from_list(accuracies_valid, 'valid accuracy', 'Epochs', 'Accuracy', outpath)

    print('Done with training.')


if __name__ == "__main__":

    args = parse_args()

    # # the next part initializes some args values (to run the script not from terminal)
    # class objectview(object):
    #     def __init__(self, d):
    #         self.__dict__ = d
    #
    # args = objectview({'train': True, 'n_train': 3, 'n_valid': 1, 'n_test': 2, 'n_epochs': 1, 'patience': 100, 'hidden_dim':32, 'encoding_dim': 256,
    # 'batch_size': 1, 'model': 'PFNet7', 'target': 'cand', 'dataset': '../../test_tmp_delphes/data/pythia8_ttbar',
    # 'outpath': '../../test_tmp_delphes/experiments/', 'activation': 'leaky_relu', 'optimizer': 'adam', 'lr': 1e-4, 'l1': 1, 'l2': 0.001, 'l3': 1, 'dropout': 0.5,
    # 'radius': 0.1, 'convlayer': 'gravnet-radius', 'convlayer2': 'none', 'space_dim': 2, 'nearest': 3, 'overwrite': True,
    # 'input_encoding': 0, 'load': None, 'evaluate': True, 'path': '../../test_tmp_delphes/experiments/PFNet7_cand_ntrain_3', 'eval_epoch' : 0})

    # define the dataset (assumes the data exists as .pt files in "processed")
    full_dataset = PFGraphDataset(args.dataset)

    # constructs a loader from the data to iterate over batches
    train_loader, valid_loader, test_loader = data_to_loader(full_dataset, args.n_train, args.n_valid, args.n_test, batch_size=args.batch_size)

    # element parameters
    input_dim = 12

    #one-hot particle ID and momentum
    output_dim_id = 6
    output_dim_p4 = 6

    patience = args.patience

    model_classes = {"PFNet7": PFNet7}

    model_class = model_classes[args.model]
    model_kwargs = {'input_dim': input_dim,
                    'hidden_dim': args.hidden_dim,
                    'encoding_dim': args.encoding_dim,
                    'output_dim_id': output_dim_id,
                    'output_dim_p4': output_dim_p4,
                    'dropout_rate': args.dropout,
                    'convlayer': args.convlayer,
                    'convlayer2': args.convlayer2,
                    'radius': args.radius,
                    'space_dim': args.space_dim,
                    'activation': args.activation,
                    'nearest': args.nearest,
                    'input_encoding': args.input_encoding}

    if args.train:
        #instantiate the model
        model = model_class(**model_kwargs)
        if args.load:
            s1 = torch.load(args.load, map_location=torch.device('cpu'))
            s2 = {k.replace("module.", ""): v for k, v in s1.items()}
            model.load_state_dict(s2)

        if multi_gpu:
            model = torch_geometric.nn.DataParallel(model)

        model.to(device)

        model_fname = get_model_fname(args.dataset, model, args.n_train, args.lr, args.target)

        outpath = osp.join(args.outpath, model_fname)
        if osp.isdir(outpath):
            if args.overwrite:
                print("model output {} already exists, deleting it".format(outpath))
                import shutil
                shutil.rmtree(outpath)
            else:
                print("model output {} already exists, please delete it".format(outpath))
                sys.exit(0)
        try:
            os.makedirs(outpath)
        except Exception as e:
            pass

        with open('{}/model_kwargs.pkl'.format(outpath), 'wb') as f:
            pickle.dump(model_kwargs, f,  protocol=pickle.HIGHEST_PROTOCOL)

        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        print(model)
        print(model_fname)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("params", params)

        # train the model
        model.train()
        train_loop()

    # evaluate the model
    if args.evaluate:
        weights = torch.load("{}/epoch_{}_weights.pth".format(args.path, args.eval_epoch), map_location=device)
        weights = {k.replace("module.", ""): v for k, v in weights.items()}

        with open('{}/model_kwargs.pkl'.format(args.path),'rb') as f:
            model_kwargs = pickle.load(f)

        model_class = model_classes[args.model]
        model = model_class(**model_kwargs)
        model.load_state_dict(weights)
        model = model.to(device)
        model.eval()

        Evaluate(model, test_loader, args.path, args.target)
