from glob import glob
import sys
import os
import os.path as osp
import numpy as np
import pandas as pd
import pickle, math, time, numba, tqdm

#Check if the GPU configuration has been provided
import torch
use_gpu = torch.cuda.device_count()>0
multi_gpu = torch.cuda.device_count()>1

try:
    if not ("CUDA_VISIBLE_DEVICES" in os.environ):
        import setGPU
        if multi_gpu:
            print('Will use multi_gpu..')
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        else:
            print('Will use single_gpu..')
except Exception as e:
    print("Could not import setGPU, running CPU-only")

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
from torch_geometric.nn import GravNetConv
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

from model_from_pytorch import PFNet7
from graph_data_delphes import PFGraphDataset, one_hot_embedding
from data_preprocessing import data_to_loader_ttbar, data_to_loader_qcd
import evaluate
from evaluate import make_plots, Evaluate, plot_confusion_matrix

#Ignore divide by 0 errors
np.seterr(divide='ignore', invalid='ignore')

#Get a unique directory name for the model
def get_model_fname(dataset, model, n_train, n_epochs, lr, target_type, batch_size):
    model_name = type(model).__name__
    model_params = sum(p.numel() for p in model.parameters())
    import hashlib
    model_cfghash = hashlib.blake2b(repr(model).encode()).hexdigest()[:10]
    model_user = os.environ['USER']

    model_fname = '{}_{}_ntrain_{}_nepochs_{}_batch_size_{}_lr_{}'.format(
        model_name,
        target_type,
        n_train,
        n_epochs,
        batch_size,
        lr)
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

def make_plot_from_list(l, label, xlabel, ylabel, outpath, save_as):
    if not os.path.exists(outpath + '/training_plots/'):
        os.makedirs(outpath + '/training_plots/')

    fig, ax = plt.subplots()
    ax.plot(range(len(l)), l, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    plt.savefig(outpath + '/training_plots/' + save_as + '.png')
    plt.close(fig)

    with open(outpath + '/training_plots/' + save_as + '.pkl', 'wb') as f:
        pickle.dump(l, f)

@torch.no_grad()
def test(model, loader, epoch, l1m, l2m, l3m, target_type, device):
    with torch.no_grad():
        ret = train(model, loader, epoch, None, l1m, l2m, l3m, target_type, device)
    return ret


def train(model, loader, epoch, optimizer, l1m, l2m, l3m, target_type, device):

    is_train = not (optimizer is None)

    if is_train:
        model.train()
    else:
        model.eval()

    #loss values for each batch: classification, regression
    losses_1, losses_2, losses_tot = [], [], []

    #accuracy values for each batch (monitor classification performance)
    accuracies_batch, accuracies_batch_msk = [], []

    #correlation values for each batch (monitor regression performance)
    corrs_batch = np.zeros(len(loader))

    #epoch confusion matrix
    conf_matrix = np.zeros((output_dim_id, output_dim_id))

    #keep track of how many data points were processed
    num_samples = 0

    for i, batch in enumerate(loader):
        t0 = time.time()

        if multi_gpu:
            X = batch
        else:
            X = batch.to(device)

        cand_ids, cand_p4, target_ids, target_p4 = model(X)

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
        l1 = l1m * torch.nn.functional.cross_entropy(target_ids, indices, weight=weights) # for classifying PID
        l1.requires_grad = True
        l2 = l2m * torch.nn.functional.mse_loss(target_p4[msk2], cand_p4[msk2])  # for regressing p4

        if args.classification_only:
            loss = l1
        else:
            loss = l1+l2

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

        print('{}/{} batch_loss={:.2f} dt={:.1f}s'.format(i, len(loader), loss.item(), t1-t0), end='\r', flush=True)

        #Compute correlation of predicted and true pt values for monitoring
        corr_pt = 0.0
        if msk.sum()>0:
            corr_pt = np.corrcoef(
                cand_p4[msk, 0].detach().cpu().numpy(),
                target_p4[msk, 0].detach().cpu().numpy())[0,1]

        corrs_batch[i] = corr_pt

        conf_matrix += sklearn.metrics.confusion_matrix(target_ids_msk.detach().cpu().numpy(),
                                        np.argmax(cand_ids.detach().cpu().numpy(),axis=1), labels=range(6))

    corr = np.mean(corrs_batch)

    losses_1 = np.mean(losses_1)
    losses_2 = np.mean(losses_2)
    losses_tot = np.mean(losses_tot)

    acc = np.mean(accuracies_batch)
    acc_msk = np.mean(accuracies_batch_msk)

    conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

    return num_samples, losses_tot, losses_1, losses_2, acc, acc_msk, conf_matrix, conf_matrix_norm


def train_loop():
    t0_initial = time.time()

    losses_1_train, losses_2_train, losses_tot_train = [], [], []
    losses_1_valid, losses_2_valid, losses_tot_valid  = [], [], []

    accuracies_train, accuracies_msk_train = [], []
    accuracies_valid, accuracies_msk_valid = [], []

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
        num_samples, losses_tot, losses_1, losses_2, acc, acc_msk, conf_matrix, conf_matrix_norm = train(model, train_loader, epoch, optimizer, args.l1, args.l2, args.l3, args.target, device)

        losses_tot_train.append(losses_tot)
        losses_1_train.append(losses_1)
        losses_2_train.append(losses_2)

        accuracies_train.append(acc)
        accuracies_msk_train.append(acc_msk)

        # validation step
        model.eval()
        num_samples_val, losses_tot_v, losses_1_v, losses_2_v, acc_v, acc_msk_v, conf_matrix_v, conf_matrix_norm_v = test(model, valid_loader, epoch, args.l1, args.l2, args.l3, args.target, device)

        losses_tot_valid.append(losses_tot_v)
        losses_1_valid.append(losses_1_v)
        losses_2_valid.append(losses_2_v)

        accuracies_valid.append(acc_v)
        accuracies_msk_valid.append(acc_msk_v)

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

        print("epoch={}/{} dt={:.2f}min train_loss={:.5f} valid_loss={:.5f} train_acc={:.5f} valid_acc={:.5f} train_acc_msk={:.5f} valid_acc_msk={:.5f} stale={} eta={:.1f}m".format(
            epoch+1, args.n_epochs,
            (t1-t0)/60, losses_tot_train[epoch], losses_tot_valid[epoch], accuracies_train[epoch], accuracies_valid[epoch],
            accuracies_msk_train[epoch], accuracies_msk_valid[epoch], stale_epochs, eta))

        torch.save(model.state_dict(), "{0}/epoch_{1}_weights.pth".format(outpath, epoch))

        plot_confusion_matrix(conf_matrix, fname= outpath + '/confusion_matrix_plots/cmT_epoch_' + str(epoch), epoch=epoch)
        plot_confusion_matrix(conf_matrix_norm, fname = outpath + '/confusion_matrix_plots/cmT_normed_epoch_' + str(epoch), epoch=epoch)

        with open(outpath + '/confusion_matrix_plots/cmT_epoch_' + str(epoch) + '.pkl', 'wb') as f:
            pickle.dump(conf_matrix, f)
        with open(outpath + '/confusion_matrix_plots/cmT_normed_epoch_' + str(epoch) + '.pkl', 'wb') as f:
            pickle.dump(conf_matrix_norm, f)

        plot_confusion_matrix(conf_matrix_v, fname= outpath + '/confusion_matrix_plots/cmV_epoch_' + str(epoch), epoch=epoch)
        plot_confusion_matrix(conf_matrix_norm_v, fname = outpath + '/confusion_matrix_plots/cmV_normed_epoch_' + str(epoch), epoch=epoch)

        with open(outpath + '/confusion_matrix_plots/cmV_epoch_' + str(epoch) + '.pkl', 'wb') as f:
            pickle.dump(conf_matrix_v, f)
        with open(outpath + '/confusion_matrix_plots/cmV_normed_epoch_' + str(epoch) + '.pkl', 'wb') as f:
            pickle.dump(conf_matrix_norm_v, f)

    make_plot_from_list(losses_tot_train, 'train loss_tot', 'Epochs', 'Loss', outpath, 'losses_tot_train')
    make_plot_from_list(losses_1_train, 'train loss_1', 'Epochs', 'Loss', outpath, 'losses_1_train')
    make_plot_from_list(losses_2_train, 'train loss_2', 'Epochs', 'Loss', outpath, 'losses_2_train')

    make_plot_from_list(losses_tot_valid, 'valid loss_tot', 'Epochs', 'Loss', outpath, 'losses_tot_valid')
    make_plot_from_list(losses_1_valid, 'valid loss_1', 'Epochs', 'Loss', outpath, 'losses_1_valid')
    make_plot_from_list(losses_2_valid, 'valid loss_2', 'Epochs', 'Loss', outpath, 'losses_2_valid')

    make_plot_from_list(accuracies_train, 'train accuracy', 'Epochs', 'Accuracy', outpath, 'accuracies_train')
    make_plot_from_list(accuracies_msk_train, 'train accuracy_msk', 'Epochs', 'Accuracy', outpath, 'accuracies_msk_train')

    make_plot_from_list(accuracies_valid, 'valid accuracy', 'Epochs', 'Accuracy', outpath, 'accuracies_valid')
    make_plot_from_list(accuracies_msk_valid, 'valid accuracy_msk', 'Epochs', 'Accuracy', outpath, 'accuracies_msk_valid')

    print('Done with training.')

    return

if __name__ == "__main__":

    #args = parse_args()

    # the next part initializes some args values (to run the script not from terminal)
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d

    args = objectview({'train': True, 'n_train': 2, 'n_valid': 1, 'n_test': 2, 'n_epochs': 2, 'patience': 100, 'hidden_dim':32, 'encoding_dim': 256,
    'batch_size': 2, 'model': 'PFNet7', 'target': 'gen', 'dataset': '../../../../test_tmp_delphes/data/pythia8_ttbar', 'dataset_qcd': '../../../../test_tmp_delphes/data/pythia8_qcd',
    'outpath': '../../../../test_tmp_delphes/experiments/', 'activation': 'leaky_relu', 'optimizer': 'adam', 'lr': 1e-4, 'l1': 1, 'l2': 1, 'l3': 1, 'dropout': 0.5,
    'radius': 0.1, 'convlayer': 'gravnet-knn', 'convlayer2': 'none', 'space_dim': 2, 'nearest': 3, 'overwrite': True,
    'input_encoding': 0, 'load': False, 'load_epoch': 0, 'load_model': 'PFNet7_gen_ntrain_2_nepochs_3_batch_size_3_lr_0.0001', 'evaluate': True, 'evaluate_on_cpu': False, 'classification_only': False})

    # define the dataset (assumes the data exists as .pt files in "processed")
    print('Creating physics data objects..')
    full_dataset_ttbar = PFGraphDataset(args.dataset)
    full_dataset_qcd = PFGraphDataset(args.dataset_qcd)

    # constructs a loader from the data to iterate over batches
    print('Constructing data loaders..')
    train_loader, valid_loader = data_to_loader_ttbar(full_dataset_ttbar, args.n_train, args.n_valid, batch_size=args.batch_size)
    test_loader = data_to_loader_qcd(full_dataset_qcd, args.n_test, batch_size=args.batch_size)

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
                    'input_encoding': args.input_encoding,
                    'target': args.target}

    if args.train:
        #instantiate the model
        print('Instantiating a model..')
        model = model_class(**model_kwargs)

        if multi_gpu:
            print("Parallelizing the training..")
            model = torch_geometric.nn.DataParallel(model)
            #model = torch.nn.parallel.DistributedDataParallel(model)    ### TODO: make it compatible with DDP

        model.to(device)

        model_fname = get_model_fname(args.dataset, model, args.n_train, args.n_epochs, args.lr, args.target, args.batch_size)

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

        if not os.path.exists(outpath + '/confusion_matrix_plots/'):
            os.makedirs(outpath + '/confusion_matrix_plots/')

        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        print(model)
        print(model_fname)

        # train the model
        model.train()
        train_loop()

    elif args.load:
            print('Loading a previously trained model..')
            model = model_class(**model_kwargs)
            outpath = args.outpath + args.load_model
            PATH = outpath + '/epoch_' + str(args.load_epoch) + '_weights.pth'
            model.load_state_dict(torch.load(PATH, map_location=device))

    # evaluate the model
    if args.evaluate:
        if args.evaluate_on_cpu:
            device = "cpu"

        model = model.to(device)
        model.eval()

        if args.train:
            Evaluate(model, test_loader, outpath, args.target, device, args.n_epochs)
        elif args.load:
            Evaluate(model, test_loader, outpath, args.target, device, args.load_epoch)

## -----------------------------------------------------------
# # to retrieve a stored variable in pkl file
# import pickle
# with open('../../test_tmp_delphes/experiments/PFNet7_gen_ntrain_2_nepochs_3_batch_size_3_lr_0.0001/confusion_matrix_plots/cmT_epoch_0.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     a = pickle.load(f)




#
# # testing a forward pass
# class objectview(object):
#     def __init__(self, d):
#         self.__dict__ = d
#
# args = objectview({'train': True, 'n_train': 2, 'n_valid': 3, 'n_test': 2, 'n_epochs': 3, 'patience': 100, 'hidden_dim':32, 'encoding_dim': 256,
# 'batch_size': 3, 'model': 'PFNet7', 'target': 'gen', 'dataset': '../../test_tmp_delphes/data/pythia8_ttbar', 'dataset_qcd': '../../test_tmp_delphes/data/pythia8_qcd',
# 'outpath': '../../test_tmp_delphes/experiments/', 'activation': 'leaky_relu', 'optimizer': 'adam', 'lr': 1e-4, 'l1': 1, 'l2': 0.001, 'l3': 1, 'dropout': 0.5,
# 'radius': 0.1, 'convlayer': 'gravnet-knn', 'convlayer2': 'none', 'space_dim': 2, 'nearest': 3, 'overwrite': True,
# 'input_encoding': 0, 'load': False, 'load_epoch': 0, 'load_model': 'PFNet7_gen_ntrain_2_nepochs_3_batch_size_3_lr_0.0001', 'evaluate': True, 'evaluate_on_cpu': True})
#
# # define the dataset (assumes the data exists as .pt files in "processed")
# full_dataset_ttbar = PFGraphDataset(args.dataset)
# full_dataset_qcd = PFGraphDataset(args.dataset_qcd)
#
# # constructs a loader from the data to iterate over batches
# train_loader, valid_loader = data_to_loader_ttbar(full_dataset_ttbar, args.n_train, args.n_valid, batch_size=args.batch_size)
# test_loader = data_to_loader_qcd(full_dataset_qcd, args.n_test, batch_size=args.batch_size)
#
# # element parameters
# input_dim = 12
#
# #one-hot particle ID and momentum
# output_dim_id = 6
# output_dim_p4 = 6
#
# patience = args.patience
#
# model_classes = {"PFNet7": PFNet7}
#
# model_class = model_classes[args.model]
# model_kwargs = {'input_dim': input_dim,
#                 'hidden_dim': args.hidden_dim,
#                 'encoding_dim': args.encoding_dim,
#                 'output_dim_id': output_dim_id,
#                 'output_dim_p4': output_dim_p4,
#                 'dropout_rate': args.dropout,
#                 'convlayer': args.convlayer,
#                 'convlayer2': args.convlayer2,
#                 'radius': args.radius,
#                 'space_dim': args.space_dim,
#                 'activation': args.activation,
#                 'nearest': args.nearest,
#                 'input_encoding': args.input_encoding}
#
# model = model_class(**model_kwargs)
#
# model_fname = get_model_fname(args.dataset, model, args.n_train, args.n_epochs, args.lr, args.target, args.batch_size)
#
# outpath = osp.join(args.outpath, model_fname)
#
# for batch in train_loader:
#     X = batch
#     target_ids = batch.ygen_id
#     target_p4 = batch.ygen
#
#     # forwardprop
#     cand_ids, cand_p4 = model(X)
#     break
#
#
#
# batch
#
#
#
# weights = compute_weights(torch.max(target_ids,-1)[1], device)
#
# weights
#
#
# batch
#
#
# batch.ycand_id.argmax(axis=0)
#
#
# len(batch.ycand_id)
#
# c0, c1, c2, c3, c4, c5 = 0, 0, 0, 0, 0, 0
#
# for i in range(len(batch.ygen_id)):
#     if (batch.ygen_id[i][0]==1):
#         c0=c0+1
# for i in range(len(batch.ygen_id)):
#     if (batch.ygen_id[i][1]==1):
#         c1=c1+1
# for i in range(len(batch.ygen_id)):
#     if (batch.ygen_id[i][2]==1):
#         c2=c2+1
# for i in range(len(batch.ygen_id)):
#     if (batch.ygen_id[i][3]==1):
#         c3=c3+1
# for i in range(len(batch.ygen_id)):
#     if (batch.ygen_id[i][4]==1):
#         c4=c4+1
# for i in range(len(batch.ygen_id)):
#     if (batch.ygen_id[i][5]==1):
#         c5=c5+1
#
# c0
# c1
# c2
# c3
# c4
# c5
#
#
# c0+c1+c2+c3+c4+c5
#
#
# c0/15704
# c1/15704
# c2/15704
# c0/15704
#
# def compute_weights(target_ids, device):
#     vs, cs = torch.unique(target_ids, return_counts=True)
#     weights = torch.zeros(output_dim_id).to(device=device)
#     for k, v in zip(vs, cs):
#         weights[k] = 1.0/math.sqrt(float(v))
#     return weights
#
#
# compute_weights(torch.max(target_ids,-1)[1], device)
#
#
# torch.max(target_ids,-1)[1]
#
# torch.unique(torch.max(target_ids,-1)[1], return_counts=True)
#
#
#
# torch.zeros(output_dim_id).to(device=device)
# for k, v in zip(vs, cs):
#     weights[k] = 1.0/math.sqrt(float(v))
#
#
# weights
#
# 602
# math.sqrt(900)
