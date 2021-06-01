import torch
import torch_sparse
import torch_scatter
import torch_cluster
import torch_geometric

import numpy as np
import matplotlib.pyplot as plt

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


class GraphBuildingLSH(torch.nn.Module):
    def __init__(self, feature_dim, bin_size, max_num_bins, k, **kwargs):
        super(GraphBuildingLSH, self).__init__(**kwargs)

        self.k = k
        self.bin_size = bin_size
        self.max_num_bins = max_num_bins
        self.codebook = torch.randn((feature_dim, max_num_bins//2))

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x):
        shp = x.shape #(batches, nodes, features)
        n_bins = shp[1] // self.bin_size

        assert(n_bins <= self.max_num_bins)

        dev = x.device

        mul = torch.matmul(x, self.codebook[:, :n_bins//2].to(dev))
        cmul = torch.cat([mul, -mul], axis=-1)

        bin_idx = torch.argmax(cmul, axis=-1)
        bins_split = torch.reshape(torch.argsort(bin_idx), (shp[0], n_bins, shp[1]//n_bins))

        points_binned = torch.stack([
            x[ibatch][bins_split[ibatch]]
            for ibatch in range(x.shape[0])]
        ) #(batches, bins, nodes, features)

        #multiply binned feature dimension
        dm_binned = torch.einsum("...ij,...kj->...ik", points_binned, points_binned)
        dm = torch.sigmoid(dm_binned) #(batches, bins, nodes, nodes)

        #(batches, bins, nodes, neighbors)
        topk = torch.topk(dm, self.k, axis=-1)

        sps = []
        for ibatch in range(dm.shape[0]):
            src = []
            dst = []
            val = []
            for ibin in range(dm.shape[1]):
                inds_src = torch.arange(0, dm.shape[2])
                inds_dst = topk.indices[ibatch, ibin]
                global_indices_src = bins_split[ibatch, ibin][inds_src]
                global_indices_dst = bins_split[ibatch, ibin][inds_dst]
                vals = topk.values[ibatch, ibin]

                for ineigh in range(inds_dst.shape[-1]):
                    src.append(global_indices_src)
                    dst.append(global_indices_dst[:, ineigh])
                    val.append(vals[:, ineigh])

            src = torch.cat(src)
            dst = torch.cat(dst)
            val = torch.cat(val)

            sp = torch.sparse_coo_tensor(
                torch.stack([src, dst]), val,
                requires_grad=True, size=(shp[1], shp[1])
            )
            sps.append(sp)

        #Sparse (batches, nodes, nodes)
        sp = torch.stack(sps).coalesce()

        return sp

#take a 3d sparse matrix, and output a 2d sparse matrix,
#where the batch dimension has been stacked in a block-diagonal way
def stacked_sparse(dm):
    #dm.shape: (num_batch, nodes, nodes)

    vals = []
    inds = []
    for ibatch in range(dm.shape[0]):
        ind = dm[ibatch].coalesce().indices()

        ind += ibatch*dm.shape[1]
        inds.append(ind)

    edge_index = torch.cat(inds, axis=-1)  #(2, num_batch*nodes)
    edge_values = dm.values() #(num_batch*nodes)

    return edge_index, edge_values

class Net(torch.nn.Module):
    def __init__(self, num_node_features):
        super(Net, self).__init__()

        feature_dim = 16
        self.lin1 = torch.nn.Linear(num_node_features, feature_dim)
        self.dm = GraphBuildingLSH(
            feature_dim=feature_dim,
            bin_size=100,
            max_num_bins=200,
            k=16
        )
        self.gcn = torch_geometric.nn.GCNConv(num_node_features, 32)
        self.lin2 = torch.nn.Linear(32, 1)


    def forward(self, data):
        x=data.x[None,:4000,:]
        print('x is', x.shape)

        n_batches = x.shape[0]
        n_points = x.shape[1]

        i1 = self.lin1(x) #(n_batches, nodes, feature_dim)
        dm = self.dm(i1) #(n_batches, nodes, nodes)

        edge_index, edge_vals = stacked_sparse(dm)

        xflat = torch.reshape(x, (n_batches*n_points, x.shape[-1]))
        i2 = self.gcn(xflat, edge_index, edge_vals) #(n_batches, nodes, 32)
        i2 = torch.reshape(i2, (n_batches, n_points, i2.shape[-1]))

        i3 = self.lin2(i2) #(n_batches, nodes, 1)

        return i3, dm


# the next part initializes some args values (to run the script not from terminal)
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

args = objectview({'train': True, 'n_train': 1, 'n_valid': 1, 'n_test': 2, 'n_epochs': 1, 'patience': 100, 'hidden_dim':32, 'input_encoding': 12, 'encoding_dim': 256,
'batch_size': 1, 'model': 'PFNet7', 'target': 'gen', 'dataset': '../../../../test_tmp_delphes/data/pythia8_ttbar', 'dataset_qcd': '../../../../test_tmp_delphes/data/pythia8_qcd',
'outpath': '../../../../test_tmp_delphes/experiments/', 'optimizer': 'adam', 'lr': 0.001, 'alpha': 1, 'dropout': 0,
'space_dim': 4, 'propagate_dimensions': 22,'nearest': 16, 'overwrite': True,
'load': False, 'load_epoch': 20 , 'load_model': 'DataParallel_gen_ntrain_400_nepochs_100_batch_size_4_lr_0.0001_both_noskip_noskip',
'evaluate': False, 'evaluate_on_cpu': False, 'classification_only': False, 'nn1': False, 'conv2': False, 'nn3': False, 'title': ''})

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

net = Net(12).float()

if multi_gpu:
    print("Parallelizing the training..")
    net = torch_geometric.nn.DataParallel(net)

net.to(device)

for batch in train_loader:
    if multi_gpu:
        X = batch
    else:
        X = batch.to(device)

    # Forwardprop
    cand_ids_one_hot, s = net(X)
    break
