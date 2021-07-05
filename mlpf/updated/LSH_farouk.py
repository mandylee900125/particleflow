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
        multi_gpu=0
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
from model import PFNet7

# the next part initializes some args values (to run the script not from terminal)
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

args = objectview({'train': True, 'n_train': 1, 'n_valid': 1, 'n_test': 2, 'n_epochs': 1, 'patience': 100, 'hidden_dim':32, 'input_encoding': 12, 'encoding_dim': 256,
'batch_size': 1, 'model': 'PFNet7', 'target': 'gen', 'dataset': '../../test_tmp_delphes/data/pythia8_ttbar', 'dataset_qcd': '../../test_tmp_delphes/data/pythia8_qcd',
'outpath': '../../test_tmp_delphes/experiments/', 'optimizer': 'adam', 'lr': 0.001, 'alpha': 1, 'dropout': 0,
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

for batch in train_loader:
    if multi_gpu:
        X = batch
    else:
        X = batch.to(device)
    break

# -----------------------------------------------------------------------------------------------------


eta = X.x[:,2]
eta

sphi= X.x[:,3]
sphi

fig = plt.figure(figsize=(5,5))

%matplotlib inline
plt.plot(eta, sphi)



from collections import defaultdict

def generate_random_vectors(dim, n_vectors):
    """
    generate random projection vectors
    the dims comes first in the matrix's shape,
    so we can use it for matrix multiplication.
    """
    return np.random.randn(n_vectors, dim,)


f = X.x[:,3:5].numpy()

def train_lsh(f, n_vectors, seed=None):
    if seed is not None:
        np.random.seed(seed)

    dim = 2
    random_vectors = generate_random_vectors(dim, n_vectors)

    # partition data points into bins,
    # and encode bin index bits into integers
    bin_indices_bits = random_vectors.dot(f) >= 0
    powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)
    bin_indices = powers_of_two.dot(bin_indices_bits)

    # update `table` so that `table[i]` is the list of document ids with bin index equal to i
    table = defaultdict(list)
    for idx, bin_index in enumerate(bin_indices):
        table[bin_index].append(idx)

    # note that we're storing the bin_indices here
    # so we can do some ad-hoc checking with it,
    # this isn't actually required
    model = {'table': table,
             'random_vectors': random_vectors,
             'bin_indices': bin_indices,
             'bin_indices_bits': bin_indices_bits}
    return model

# train the model
n_vectors = 16
model = train_lsh(X.x[:,3:5].reshape(2,-1).numpy(), n_vectors, seed=143)

model

model.keys()
model['bin_indices_bits'].shape
X.x.shape


np.unique(model['bin_indices'])

import scipy
import sklearn

from sklearn.neighbors import KDTree

import time
import numpy as np

t0=time.time()
for node in range(len(X.x)):
    rng = np.random.RandomState(0)
    tree = KDTree(pos, leaf_size=2)
    dist, ind = tree.query(pos[:node+1], k=16)
    # print(ind)  # indices of 3 closest neighbors
    # print(dist)  # distances to 3 closest neighbors
t1=time.time()
t1-t0


torch.tensor([0, 0, 0, 0])

torch.zeros_like(pos[:,0]).long()


X


X.batch

t0=time.time()
assign_index = knn(X.x, X.x, 16, X.batch, X.batch)
    # print(ind)  # indices of 3 closest neighbors
    # print(dist)  # distances to 3 closest neighbors
t1=time.time()
t1-t0

assign_index
assign_index = knn(X.x, X.x, 35, torch.zeros_like(X.x[:,0]).long(), torch.zeros_like(X.x[:,0]).long())
