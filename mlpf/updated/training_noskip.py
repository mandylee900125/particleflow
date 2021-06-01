from glob import glob
import sys, os
import os.path as osp
import pickle, math, time, numba, tqdm
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib, mplhep
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn.inits import reset
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from torch_geometric.nn import GravNetConv
from torch.utils.data import random_split
import torch_cluster

sys.path.insert(1, '../../plotting/')
sys.path.insert(1, '../../mlpf/plotting/')
import args
from args import parse_args
from graph_data_delphes import PFGraphDataset, one_hot_embedding
from data_preprocessing import data_to_loader_ttbar, data_to_loader_qcd
import evaluate
from evaluate import make_plots, Evaluate
from plot_utils import plot_confusion_matrix
from model_noskip import PFNet7

#Ignore divide by 0 errors
np.seterr(divide='ignore', invalid='ignore')

#Get a unique directory name for the model
def get_model_fname(dataset, model, n_train, n_epochs, lr, target_type, batch_size, task, title):
    model_name = type(model).__name__
    model_params = sum(p.numel() for p in model.parameters())
    import hashlib
    model_cfghash = hashlib.blake2b(repr(model).encode()).hexdigest()[:10]
    model_user = os.environ['USER']

    model_fname = '{}_{}_ntrain_{}_nepochs_{}_batch_size_{}_lr_{}_{}_{}'.format(
        model_name,
        target_type,
        n_train,
        n_epochs,
        batch_size,
        lr,
        task,
        title)
    return model_fname

def compute_weights(target_ids_one_hot, device):
    vs, cs = torch.unique(target_ids_one_hot, return_counts=True)
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
def test(model, loader, epoch, alpha, target_type, device):
    with torch.no_grad():
        ret = train(model, loader, epoch, None, alpha, target_type, device)
    return ret

def train(model, loader, epoch, optimizer, alpha, target_type, device):

    is_train = not (optimizer is None)

    if is_train:
        model.train()
    else:
        model.eval()

    #loss values for each batch: classification, regression, total
    losses_1, losses_2, losses_tot = [], [], []

    #accuracy values for each batch (monitor classification performance)
    accuracies_batch, accuracies_batch_msk = [], []

    #setup confusion matrix
    conf_matrix = np.zeros((output_dim_id, output_dim_id))

    # for param in model.parameters():
    #     print(param.data)
    #     break

    for i, batch in enumerate(loader):
        t0 = time.time()

        if multi_gpu:
            X = batch
        else:
            X = batch.to(device)

        # Forwardprop
        cand_ids_one_hot, cand_p4, target_ids_one_hot, target_p4 = model(X)

        _, cand_ids = torch.max(cand_ids_one_hot, -1)
        _, target_ids = torch.max(target_ids_one_hot, -1)

        # masking
        msk = ((cand_ids != 0) & (target_ids != 0))
        msk2 = ((cand_ids != 0) & (cand_ids == target_ids))

        # computing loss
        weights = compute_weights(torch.max(target_ids_one_hot,-1)[1], device)
        l1 = torch.nn.functional.cross_entropy(cand_ids_one_hot, target_ids, weight=weights) # for classifying PID
        l2 = alpha * torch.nn.functional.mse_loss(cand_p4[msk2], target_p4[msk2])  # for regressing p4

        if args.classification_only:
            loss = l1
        else:
            loss = l1+l2

        losses_1.append(l1.item())
        losses_2.append(l2.item())
        losses_tot.append(loss.item())

        if is_train:
            # BACKPROP
            #print(list(model.parameters())[1].grad)
            a = list(model.parameters())[1].clone()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b = list(model.parameters())[1].clone()
            if torch.equal(a.data, b.data):
                print('Model is not learning.. weights are not updating..')

        t1 = time.time()

        accuracies_batch.append(accuracy_score(target_ids.detach().cpu().numpy(), cand_ids.detach().cpu().numpy()))
        accuracies_batch_msk.append(accuracy_score(target_ids[msk].detach().cpu().numpy(), cand_ids[msk].detach().cpu().numpy()))

        conf_matrix += sklearn.metrics.confusion_matrix(target_ids.detach().cpu().numpy(),
                                        np.argmax(cand_ids_one_hot.detach().cpu().numpy(),axis=1), labels=range(6))

        print('{}/{} batch_loss={:.2f} dt={:.1f}s'.format(i, len(loader), loss.item(), t1-t0)), end='\r', flush=True)

    losses_1 = np.mean(losses_1)
    losses_2 = np.mean(losses_2)
    losses_tot = np.mean(losses_tot)

    acc = np.mean(accuracies_batch)
    acc_msk = np.mean(accuracies_batch_msk)

    conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

    return losses_tot, losses_1, losses_2, acc, acc_msk, conf_matrix, conf_matrix_norm


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
        losses_tot, losses_1, losses_2, acc, acc_msk, conf_matrix, conf_matrix_norm = train(model, train_loader, epoch, optimizer, args.alpha, args.target, device)

        losses_tot_train.append(losses_tot)
        losses_1_train.append(losses_1)
        losses_2_train.append(losses_2)

        accuracies_train.append(acc)
        accuracies_msk_train.append(acc_msk)

        # validation step
        model.eval()
        losses_tot_v, losses_1_v, losses_2_v, acc_v, acc_msk_v, conf_matrix_v, conf_matrix_norm_v = test(model, valid_loader, epoch, args.alpha, args.target, device)

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

        plot_confusion_matrix(conf_matrix_norm, ["none", "ch.had", "n.had", "g", "el", "mu"], fname = outpath + '/confusion_matrix_plots/cmT_normed_epoch_' + str(epoch), epoch=epoch)
        plot_confusion_matrix(conf_matrix_norm_v, ["none", "ch.had", "n.had", "g", "el", "mu"], fname = outpath + '/confusion_matrix_plots/cmV_normed_epoch_' + str(epoch), epoch=epoch)

        with open(outpath + '/confusion_matrix_plots/cmT_normed_epoch_' + str(epoch) + '.pkl', 'wb') as f:
            pickle.dump(conf_matrix_norm, f)

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

    args = parse_args()

    # # the next part initializes some args values (to run the script not from terminal)
    # class objectview(object):
    #     def __init__(self, d):
    #         self.__dict__ = d
    #
    # args = objectview({'train': True, 'n_train': 1, 'n_valid': 1, 'n_test': 2, 'n_epochs': 1, 'patience': 100, 'hidden_dim':256, 'input_encoding': 12, 'encoding_dim': 125,
    # 'batch_size': 2, 'model': 'PFNet7', 'target': 'gen', 'dataset': '../../test_tmp_delphes/data/pythia8_ttbar', 'dataset_qcd': '../../test_tmp_delphes/data/pythia8_qcd',
    # 'outpath': '../../test_tmp_delphes/experiments/', 'optimizer': 'adam', 'lr': 0.001, 'alpha': 2e-4, 'dropout': 0,
    # 'space_dim': 4, 'propagate_dimensions': 22,'nearest': 16, 'overwrite': True,
    # 'load': True, 'load_epoch': 0 , 'load_model': 'DataParallel_gen_ntrain_400_nepochs_100_batch_size_4_lr_0.0001_both_noskip',
    # 'evaluate': False, 'evaluate_on_cpu': False, 'classification_only': True, 'nn1': False, 'conv2': False, 'nn3': False, 'title': ''})

    # define the dataset (assumes the data exists as .pt files in "processed")
    print('Processing the data..')
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
                    'input_encoding': args.input_encoding,
                    'encoding_dim': args.encoding_dim,
                    'output_dim_id': output_dim_id,
                    'output_dim_p4': output_dim_p4,
                    'dropout_rate': args.dropout,
                    'space_dim': args.space_dim,
                    'propagate_dimensions': args.propagate_dimensions,
                    'nearest': args.nearest,
                    'target': args.target,
                    'nn1': args.nn1,
                    'conv2': args.conv2,
                    'nn3': args.nn3}

    if args.load:
            print('Loading a previously trained model..')
            model = model_class(**model_kwargs)
            outpath = args.outpath + args.load_model
            PATH = outpath + '/epoch_' + str(args.load_epoch) + '_weights.pth'

            state_dict = torch.load(PATH, map_location=device)

            if "DataParallel" in args.load_model:   # if the model was trained using DataParallel then we do this
                state_dict = torch.load(PATH, map_location=device)
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove module.
                    new_state_dict[name] = v
                    # print('name is:', name)
                state_dict=new_state_dict

            model.load_state_dict(state_dict)

    elif args.train:
        #instantiate the model
        print('Instantiating a model..')
        model = model_class(**model_kwargs)

        if multi_gpu:
            print("Parallelizing the training..")
            model = torch_geometric.nn.DataParallel(model)
            #model = torch.nn.parallel.DistributedDataParallel(model)    ### TODO: make it compatible with DDP

        model.to(device)

    if args.train:
        args.title=args.title+'_noskip'
        if args.nn1:
            args.title=args.title+'_nn1'
        if args.nn3:
            args.title=args.title+'_nn3'
        if args.conv2:
            args.title=args.title+'_conv2'

        if args.classification_only:
            model_fname = get_model_fname(args.dataset, model, args.n_train, args.n_epochs, args.lr, args.target, args.batch_size, "clf", args.title)
        else:
            model_fname = get_model_fname(args.dataset, model, args.n_train, args.n_epochs, args.lr, args.target, args.batch_size, "both", args.title)

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

        # # train the model
        # for param in model.parameters():
        #     param.requires_grad = True

        model.train()
        train_loop()



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
# with open('../../test_tmp_delphes/experiments/PFNet7_gen_ntrain_2_nepochs_3_batch_size_3_lr_0.0001/confusion_matrix_plots/cmT_normed_epoch_0.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     a = pickle.load(f)
