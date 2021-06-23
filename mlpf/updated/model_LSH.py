import numpy as np
import mplhep

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
from torch.utils.data import random_split

#from torch_geometric.nn import GravNetConv         # if you want to get it from source code (won't be able to retrieve the adjacency matrix)
from torch_geometric.nn import GraphConv

#Model with gravnet clustering
class PFNet7(nn.Module):
    def __init__(self,
        input_dim=12, hidden_dim=125, input_encoding=12, encoding_dim=32,
        output_dim_id=6,
        output_dim_p4=6,
        dropout_rate=0.0,
        space_dim=4, propagate_dimensions=22, nearest=16,
        target="gen", nn1=True, conv2=True, nn3=True):

        super(PFNet7, self).__init__()

        self.target = target
        self.nn1 = nn1

        self.conv2 = conv2
        self.nn3 = nn3

        self.act = nn.LeakyReLU
        self.act_f = torch.nn.functional.leaky_relu

        # dropout layer if needed anywhere
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # (1) DNN
        if self.nn1:
            self.nn1 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                self.act(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.Linear(hidden_dim, hidden_dim),
                self.act(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.Linear(hidden_dim, input_encoding),
            )

        # (2) CNN: Gravnet layer
        self.conv1 = GraphBuildingLSH(
            feature_dim=input_encoding,
            bin_size=100,
            max_num_bins=200,
            k=16
        )
        self.gcn = torch_geometric.nn.GCNConv(input_encoding, encoding_dim)

        # (3) CNN: GraphConv
        if self.conv2:
            self.conv2 = GraphConv(encoding_dim, encoding_dim)

        # (4) DNN layer: classifying PID
        self.nn2 = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            self.act(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            self.act(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim_id),
        )

        # (5) DNN layer: regressing p4
        if self.nn3:
            self.nn3 = nn.Sequential(
                nn.Linear(encoding_dim + output_dim_id, hidden_dim),
                self.act(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.Linear(hidden_dim, hidden_dim),
                self.act(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.Linear(hidden_dim, output_dim_p4),
            )

    def forward(self, data):

        x = data.x
        ygen_id = data.ygen_id
        ygen = data.ygen

        while x.shape[0]<5000:
            x=torch.cat((x,torch.zeros_like(x[1]).reshape(1,-1)),axis=0)
            ygen_id=torch.cat((ygen_id,torch.zeros_like(ygen_id[1]).reshape(1,-1)),axis=0)
            ygen=torch.cat((ygen,torch.zeros_like(ygen[1]).reshape(1,-1)),axis=0)

        x = x[None,:5000,:]

        n_batches = x.shape[0]
        n_points = x.shape[1]

        # Encoder/Decoder step
        if self.nn1:
            x = self.nn1(x)

        # Gravnet step
        dm = self.conv1(x)
        edge_index, edge_weight = stacked_sparse(dm)

        x = self.act_f(x)                 # act by nonlinearity
        xflat = torch.reshape(x, (n_batches*n_points, x.shape[-1]))

        x = self.gcn(x, edge_index, edge_weight)

        # GraphConv step
        if self.conv2:
            x = self.conv2(x, edge_index=edge_index, edge_weight=edge_weight)
            x = self.act_f(x)                 # act by nonlinearity

        # DNN to predict PID (after a dropout)
        cand_ids = self.nn2(self.dropout(x))

        # DNN to predict p4 (after a dropout)
        if self.nn3:
            nn3_input = torch.cat([x, cand_ids], axis=-1)
            cand_p4 = self.nn3(self.dropout(nn3_input))
        else:
            cand_p4=torch.zeros_like(data.ycand)

        if self.target=='cand':
            return cand_ids.reshape(4000,6), cand_p4.reshape(4000,6), data.ycand_id[:4000,:], data.ycand[:4000,:]

        elif self.target=='gen':
            return cand_ids.reshape(5000,6), cand_p4.reshape(5000,6), ygen_id[:5000,:], ygen[:5000,:]

        else:
            print('Target type unknown..')
            return 0

class GraphBuildingLSH(torch.nn.Module):
    def __init__(self, feature_dim, bin_size, max_num_bins, k, **kwargs):
        super(GraphBuildingLSH, self).__init__(**kwargs)

        self.k = k
        self.bin_size = bin_size
        self.max_num_bins = max_num_bins
        self.codebook = torch.randn((feature_dim, max_num_bins//2)).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x):
        shp = x.shape #(batches, nodes, features)
        n_bins = shp[1] // self.bin_size

        assert(n_bins <= self.max_num_bins)
        mul = torch.matmul(x, self.codebook[:, :n_bins//2])
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



# -------------------------------------------------------------------------------------
# # uncomment to test a forward pass
# from graph_data_delphes import PFGraphDataset
# from data_preprocessing import data_to_loader_ttbar
# from data_preprocessing import data_to_loader_qcd
#
# full_dataset = PFGraphDataset('../../test_tmp_delphes/data/pythia8_ttbar')
#
# train_loader, valid_loader = data_to_loader_ttbar(full_dataset, n_train=2, n_valid=1, batch_size=2)
#
# print('Input to the network:', next(iter(train_loader)))
#
# model = PFNet7()
#
# for batch in train_loader:
#     cand_ids, cand_p4, target_ids, target_p4 = model(batch)
#     cand_ids
#     print('Predicted PID:', cand_ids)
#     print('Predicted p4:', cand_p4)
#     break
