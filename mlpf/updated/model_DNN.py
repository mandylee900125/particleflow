import numpy as np
import mplhep

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
from gravnet import GravNetConv
from torch_geometric.nn import GraphConv

#Model with gravnet clustering
class PFNet7(nn.Module):
    def __init__(self,
        input_dim=12, hidden_dim=125,
        output_dim_id=6,
        output_dim_p4=6,
        target="gen"):

        super(PFNet7, self).__init__()

        self.target = target

        self.act = nn.LeakyReLU
        self.act_f = torch.nn.functional.leaky_relu

        self.nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.act(),
            nn.Linear(hidden_dim, hidden_dim),
            self.act(),
            nn.Linear(hidden_dim, output_dim_id),
        )

        self.nn2 = nn.Sequential(
            #nn.Linear(input_dim + output_dim_id, hidden_dim),
            nn.Linear(output_dim_id, hidden_dim),
            self.act(),
            nn.Linear(hidden_dim, hidden_dim),
            self.act(),
            nn.Linear(hidden_dim, output_dim_p4),
        )

    def forward(self, data):

        x = data.x

        # Encoder/Decoder step
        cand_ids = self.nn1(x)

        # DNN to predict p4 (after a dropout)
        # nn2_input = torch.cat([x, cand_ids], axis=-1)
        nn2_input = cand_ids
        cand_p4 = self.nn2(nn2_input)

        if self.target=='cand':
            return cand_ids, cand_p4, data.ycand_id, data.ycand

        elif self.target=='gen':
            return cand_ids, cand_p4, data.ygen_id, data.ygen

        else:
            print('Target type unknown..')
            return 0

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
#     break
