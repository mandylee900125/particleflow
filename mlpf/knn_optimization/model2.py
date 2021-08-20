import numpy as np
import mplhep, time, os

import torch
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
from torch.utils.data import random_split

#from torch_geometric.nn import GravNetConv         # if you want to get it from source code (won't be able to retrieve the adjacency matrix)
from gravnet2 import GravNetConv2########################USES GRAVNET2###########################
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

#Model with gravnet clustering
class PFNet7(nn.Module):
    def __init__(self,
        input_dim=12, hidden_dim=256, hidden_dim_nn1=64, input_encoding=12, encoding_dim=64,
        output_dim_id=6,
        output_dim_p4=6,
        space_dim=4, propagate_dimensions=22, nearest=16,
        target="gen"):

        super(PFNet7, self).__init__()

        self.elu = nn.ELU
        self.act_f = torch.nn.functional.leaky_relu

        # (1) DNN: encoding/decoding of all tracks and clusters
        self.nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_nn1),
            self.elu(),
            nn.Linear(hidden_dim_nn1, hidden_dim_nn1),
            self.elu(),
            nn.Linear(hidden_dim_nn1, input_encoding),
        )

        # (2) CNN: Gravnet layer
        self.conv1 = GravNetConv(input_encoding, encoding_dim, space_dim, propagate_dimensions, nearest)

        # (3) DNN layer: classifying PID
        self.nn2 = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            self.elu(),
            nn.Linear(hidden_dim, hidden_dim),
            self.elu(),
            nn.Linear(hidden_dim, hidden_dim),
            self.elu(),
            nn.Linear(hidden_dim, output_dim_id),
        )

        # (4) DNN layer: regressing p4
        self.nn3 = nn.Sequential(
            nn.Linear(encoding_dim + output_dim_id + input_dim, hidden_dim),
            self.elu(),
            nn.Linear(hidden_dim, hidden_dim),
            self.elu(),
            nn.Linear(hidden_dim, hidden_dim),
            self.elu(),
            nn.Linear(hidden_dim, output_dim_p4),
        )

    def forward(self, data):
        x0 = data.x

        # Encoder/Decoder step
        x = self.nn1(x0)

        # Gravnet step
        x, edge_index, edge_weight = self.conv1(x)
        x = self.act_f(x)                 # act by nonlinearity

        # DNN to predict PID
        pred_ids = self.nn2(x)

        # DNN to predict p4
        nn3_input = torch.cat([x, pred_ids, x0], axis=-1)
        pred_p4 = self.nn3(nn3_input)

        return pred_ids, pred_p4, data.ygen_id, data.ygen, data.ycand_id, data.ycand

# # -------------------------------------------------------------------------------------
# testing inference of a forward pass
from graph_data_delphes import PFGraphDataset
from data_preprocessing import data_to_loader_ttbar, data_to_loader_qcd

# get the dataset
full_dataset_ttbar = PFGraphDataset('../../../test_tmp_delphes/data/pythia8_ttbar')
full_dataset_qcd = PFGraphDataset('../../../test_tmp_delphes/data/pythia8_qcd')

# make data loaders
train_loader, valid_loader = data_to_loader_ttbar(full_dataset_ttbar, n_train=1, n_valid=1, batch_size=2)
test_loader = data_to_loader_qcd(full_dataset_qcd, n_test=1, batch_size=2)

# instantiate a model
model = PFNet7()
if multi_gpu:
    print("Parallelizing the inference..")
    model = torch_geometric.nn.DataParallel(model)

model.to(device)
print(model)

T=[]
for i, batch in enumerate(train_loader):
    if multi_gpu:
        X = batch
    else:
        X = batch.to(device)

    t0 = time.time()
    pred_ids_one_hot, pred_p4, gen_ids_one_hot, gen_p4, cand_ids_one_hot, cand_p4 = model(X)
    t1 = time.time()
    T.append(round((t1-t0),5))

    if i==10:
        break

print('Average inference time per event: ', round(sum(T)/len(T),5), 's')
