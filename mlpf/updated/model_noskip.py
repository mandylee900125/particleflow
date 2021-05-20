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
from gravnet_noskip import GravNetConv
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
        self.conv1 = GravNetConv(input_encoding, encoding_dim, space_dim, propagate_dimensions, nearest)

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

        # Encoder/Decoder step
        if self.nn1:
            x = self.nn1(x)

        # Gravnet step
        x, edge_index, edge_weight = self.conv1(x)
        x = self.act_f(x)                 # act by nonlinearity

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
#     print('Predicted PID:', cand_ids)
#     print('Predicted p4:', cand_p4)
#     break
