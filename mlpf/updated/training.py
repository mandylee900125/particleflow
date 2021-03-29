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

#define the global base device
if use_gpu:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep

import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix

import args
from args import parse_args

from model import PFNet7
from graph_data_delphes import PFGraphDataset, one_hot_embedding
from data_preprocessing import data_to_loader
import evaluate
from evaluate import make_plots

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

@torch.no_grad()
def test(model, loader, epoch, l1m, l2m, l3m, target_type):
    with torch.no_grad():
        ret = train(model, loader, epoch, None, l1m, l2m, l3m, target_type, None)
    return ret


def train(model, loader, epoch, optimizer, l1m, l2m, l3m, target_type, scheduler):

    is_train = not (optimizer is None)

    if is_train:
        model.train()
    else:
        model.eval()

    #loss values for each batch: classification, regression
    losses = np.zeros((len(loader), 3))

    #accuracy values for each batch (monitor classification performance)
    accuracies_batch = np.zeros(len(loader))

    #correlation values for each batch (monitor regression performance)
    corrs_batch = np.zeros(len(loader))

    #epoch confusion matrix
    conf_matrix = np.zeros((output_dim_id, output_dim_id))

    #keep track of how many data points were processed
    num_samples = 0

    for i, batch in enumerate(loader):
        t0 = time.time()

        if not multi_gpu:
            batch = batch.to(device)

        if is_train:
            optimizer.zero_grad()

        # forward pass
        cand_id_onehot, cand_momentum, new_edge_index = model(batch)

        _dev = cand_id_onehot.device                   # store the device in dev
        _, indices = torch.max(cand_id_onehot, -1)     # picks the maximum PID location and stores the index (opposite of one_hot_embedding)

        num_samples += len(cand_id_onehot)

        # concatenate ygen/ycand over the batch to compare with the truth label
        # now: ygen/ycand is of shape [~5000*batch_size, 6] corresponding to the output of the forward pass
        if args.target == "gen":
            target_ids = batch.ygen_id
            target_p4 = batch.ygen
        elif args.target == "cand":
            target_ids = batch.ycand_id
            target_p4 = batch.ycand

        #Predictions where both the predicted and true class label was nonzero
        #In these cases, the true candidate existed and a candidate was predicted
        # target_ids_msk reverts the one_hot_embedding
        # msk is a list of booleans of shape [~5000*batch_size] where each boolean correspond to whether a candidate was predicted
        _, target_ids_msk = torch.max(target_ids, -1)
        msk = ((indices != 0) & (target_ids_msk != 0)).detach().cpu()
        msk2 = ((indices != 0) & (indices == target_ids_msk))

        accuracies_batch[i] = accuracy_score(target_ids_msk[msk].detach().cpu().numpy(), indices[msk].detach().cpu().numpy())

        # a manual rescaling weight given to each class
        weights = compute_weights(torch.max(target_ids,-1)[1], _dev)

        #Loss for output candidate id (multiclass)
        l1 = l1m * torch.nn.functional.cross_entropy(target_ids, indices, weight=weights)

        #Loss for candidate p4 properties (regression)
        l2 = l2m * torch.nn.functional.mse_loss(target_p4[msk2], cand_momentum[msk2])

        batch_loss = l1 + l2
        losses[i, 0] = l1.item()
        losses[i, 1] = l2.item()

        if is_train:
            batch_loss.backward()

        batch_loss_item = batch_loss.item()
        t1 = time.time()

        print('{}/{} batch_loss={:.2f} dt={:.1f}s'.format(i, len(loader), batch_loss_item, t1-t0), end='\r', flush=True)
        if is_train:
            optimizer.step()
            if not scheduler is None:
                scheduler.step()

        #Compute correlation of predicted and true pt values for monitoring
        corr_pt = 0.0
        if msk.sum()>0:
            corr_pt = np.corrcoef(
                cand_momentum[msk, 0].detach().cpu().numpy(),
                target_p4[msk, 0].detach().cpu().numpy())[0,1]

        corrs_batch[i] = corr_pt

        conf_matrix += confusion_matrix(target_ids_msk.detach().cpu().numpy(),
                                        np.argmax(cand_id_onehot.detach().cpu().numpy(),axis=1), labels=range(6))

    corr = np.mean(corrs_batch)
    acc = np.mean(accuracies_batch)
    losses = losses.mean(axis=0)
    return num_samples, losses, corr, acc, conf_matrix

def train_loop():
    t0_initial = time.time()

    losses_train = np.zeros((args.n_epochs, 3))
    losses_val = np.zeros((args.n_epochs, 3))

    corrs = []
    corrs_v = []
    accuracies = []
    accuracies_v = []
    best_val_loss = 99999.9
    stale_epochs = 0

    print("Training over {} epochs".format(args.n_epochs))
    for epoch in range(args.n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            print("breaking due to stale epochs")
            break

        model.train()

        num_samples_train, losses, c, acc, conf_matrix = train(model, train_loader, epoch, optimizer,
                                                               args.l1, args.l2, args.l3, args.target, scheduler)

        l = sum(losses)
        losses_train[epoch] = losses
        corrs += [c]
        accuracies += [acc]

        model.eval()
        num_samples_val, losses_v, c_v, acc_v, conf_matrix_v = test(model, valid_loader, epoch,
                                                                    args.l1, args.l2, args.l3, args.target)
        l_v = sum(losses_v)
        losses_val[epoch] = losses_v
        corrs_v += [c_v]
        accuracies_v += [acc_v]

        if l_v < best_val_loss:
            best_val_loss = l_v
            stale_epochs = 0
        else:
            stale_epochs += 1

        t1 = time.time()
        epochs_remaining = args.n_epochs - epoch
        time_per_epoch = (t1 - t0_initial)/(epoch + 1)
        eta = epochs_remaining*time_per_epoch/60

        spd = (num_samples_val+num_samples_train)/time_per_epoch
        losses_str = "[" + ",".join(["{:.4f}".format(x) for x in losses_v]) + "]"

        torch.save(model.state_dict(), "{0}/epoch_{1}_weights.pth".format(outpath, epoch))

        print("epoch={}/{} dt={:.2f}s loss_train={:.5f} loss_valid={:.5f} c={:.2f}/{:.2f} a={:.6f}/{:.6f} partial_losses={} stale={} eta={:.1f}m spd={:.2f} samples/s lr={}".format(
            epoch+1, args.n_epochs,
            t1 - t0, l, l_v, c, c_v, acc, acc_v,
            losses_str, stale_epochs, eta, spd, optimizer.param_groups[0]['lr']))

    print('Done with training.')


if __name__ == "__main__":

    # args = parse_args()

    # the next part initializes some args values (to run the script not from terminal)
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d

    args = objectview({'n_train': 3, 'n_valid': 1, 'n_test': 2, 'n_epochs': 1, 'patience': 100, 'hidden_dim':32, 'encoding_dim': 256,
    'batch_size': 1, 'model': 'PFNet7', 'target': 'cand', 'dataset': '../../test_tmp_delphes/data/pythia8_ttbar',
    'outpath': '../../test_tmp_delphes/experiments/', 'activation': 'leaky_relu', 'optimizer': 'adam', 'lr': 1e-4, 'l1': 1, 'l2': 0.001, 'l3': 1, 'dropout': 0.5,
    'radius': 0.1, 'convlayer': 'gravnet-radius', 'convlayer2': 'none', 'space_dim': 2, 'nearest': 3, 'overwrite': True,
    'input_encoding': 0, 'load': None, 'scheduler': 'none', 'evaluate': True, 'path': '../../test_tmp_delphes/experiments/PFNet7_cand_ntrain_3', 'eval_epoch' : 0})


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

    scheduler = None
    if args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=int(len(train_loader)),
            epochs=args.n_epochs + 1,
            anneal_strategy='linear',
        )

    print(model)
    print(model_fname)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("params", params)

    model.train()

    train_loop()


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

        # TODO: make another for 'gen'
        if args.target=='cand':
            pred_id_all = []
            pred_p4_all = []
            new_edges_all = []
            ycand_id_all = []
            ycand_all = []

            for batch in test_loader:
                pred_id, pred_p4, new_edges = model(batch)

                pred_id_all.append(pred_id)
                pred_p4_all.append(pred_p4)
                new_edges_all.append(new_edges)
                ycand_id_all.append(batch.ycand_id)
                ycand_all.append(batch.ycand)

            pred_id = pred_id_all[0]
            pred_p4 = pred_p4_all[0]
            new_edges = new_edges_all[0]
            ycand_id = ycand_id_all[0]
            ycand = ycand_all[0]

            for i in range(len(pred_id_all)-1):
                pred_id = torch.cat((pred_id,pred_id_all[i+1]))
                pred_p4 = torch.cat((pred_p4,pred_p4_all[i+1]))
                ycand = torch.cat((ycand,ycand_all[i+1]))
                ycand_id = torch.cat((ycand_id,ycand_id_all[i+1]))

            print('Making plots for evaluation..')
            
            make_plots(ycand_id, ycand, pred_id, pred_p4, out=args.path +'/')



    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    #     train_loop()

    # print(prof.key_averages().table(sort_by="cuda_time_total"))
