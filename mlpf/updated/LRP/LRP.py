import torch
import torch.nn as nn
from torch.nn import Sequential as Seq,Linear,ReLU,BatchNorm1d
from torch_scatter import scatter_mean
import numpy as np
import json
import model_io
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy
import pickle, math, time

from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils.convert import to_networkx

use_gpu = torch.cuda.device_count()>0
multi_gpu = torch.cuda.device_count()>1

#define the global base device
if use_gpu:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

class LRP:
    EPSILON=1e-9

    def __init__(self,model:model_io):
        self.model=model

    def register_model(model:model_io):
        self.model=model

    """
    LRP rules
    """

    @staticmethod
    def eps_rule(layer,input,R, edge_index, edge_weight, index, after_message, before_message, LeakyReLU):
        print('l1', layer)

        EPSILON=1e-9
        a=copy_tensor(input)
        a.retain_grad()
        z = layer.forward(a)

        if LeakyReLU:
            w = torch.eye(a.shape[1])
        else:
            w = layer.weight

        if 'in_features=256, out_features=6' in str(layer): # for the output layer
            T, W, r = [], [], []

            for i in range(6):
                T.append(R[:,i].reshape(-1,1))
                W.append(w[i,:].reshape(1,-1))

                I = torch.ones_like(R[:,i]).reshape(-1,1)

                Numerator=(a*torch.matmul(T[i],W[i]))
                Denominator=(a*torch.matmul(I,W[i])).sum(axis=1)

                Denominator = Denominator.reshape(-1,1).expand(Denominator.size()[0],Numerator.size()[1])

                r.append(torch.abs(Numerator / (Denominator+EPSILON*torch.sign(Denominator))))

            return r

        else:

            for i in range(6):
                I = torch.ones_like(R[i])

                Numerator=(a*torch.matmul(R[i],w))
                Denominator=(a*torch.matmul(I,w)).sum(axis=1)

                Denominator = Denominator.reshape(-1,1).expand(Denominator.size()[0],Numerator.size()[1])

                R[i]=(torch.abs(Numerator / (Denominator+EPSILON*torch.sign(Denominator))))

            return R

    @staticmethod
    def z_rule(layer,input,R, edge_index, edge_weight, flag):
        w=copy_tensor(layer.weight.data)
        b=copy_tensor(layer.bias.data)

        def f(x):
            x.retain_grad()

            n=x*w
            d=n+b*torch.sign(n)*torch.sign(b)

            return n/d

        frac=f(input)
        return frac*R

    @staticmethod
    def gravnet_rule(layer,input, R, edge_index, edge_weight, index, after_message, before_message, LeakyReLU):

        print('l2', layer)
        BIG_LIST=[]
        for i in range(6):
            big_list=[]
            c=0

            b=Data()
            b['edge_index']=edge_index
            b['num_nodes']=edge_index[0].shape[0]
            b['edge_weight']=edge_weight

            G = to_networkx(b, edge_attrs=['edge_weight'], to_undirected=True, remove_self_loops=False)

            def nodes_connected(u, v):
                return u in G.neighbors(v)

            for node_i in range(R[i].shape[0]):
                R_tensor_per_node = torch.zeros(R[i].shape[0],R[i].shape[1])
                R_tensor_per_node[node_i]=R[i][node_i]
                for node_j in range(R[i].shape[0]):
                    if node_j == node_i:
                        pass
                    # check if neighbor
                    if nodes_connected(node_i,node_j):
                        R_tensor_per_node[node_j]=R[i][node_j]*G[node_i][node_j]['edge_weight']
                        # print('node', node_j, 'is indeed neighbor')

                big_list.append(torch.abs(R_tensor_per_node))
            BIG_LIST.append(big_list)

        return R, BIG_LIST


    """
    explanation functions
    """

    def explain(self,
                to_explain:dict,
                save:bool=True,
                save_to:str="./relevance.pt",
                sort_nodes_by:int=0,
                signal=torch.tensor([1,0,0,0,0,0],dtype=torch.float32).to(device),
                return_result:bool=False):

        inputs=to_explain["inputs"]
        pred=to_explain["pred"]
        edge_index=to_explain["edge_index"]
        edge_weight=to_explain["edge_weight"]
        after_message=to_explain["after_message"]
        before_message=to_explain["before_message"]

        start_index=self.model.n_layers                  ##########################
        print('Total number of linear (+LeakyReLU) layers:', start_index)
        to_explain['R'][start_index]=copy_tensor(pred)

        ### loop over each single layer
        big_list=[]
        R=[]
        for index in range(start_index+1, 1,-1):
            R, big_list=self.explain_single_layer(R, to_explain, edge_index, edge_weight, after_message, before_message, big_list, index)
            # print(to_explain['R'][index-1].shape, to_explain['R'][index-1])

        return to_explain['R'], big_list[0]

    def explain_single_layer(self, R, to_explain, edge_index, edge_weight, after_message, before_message, big_list, index=None,name=None):

        # preparing variables required for computing LRP
        layer=self.model.get_layer(index=index,name=name)
        rule=self.model.get_rule(index=index,layer_name=name)

        if rule=="z":
            rule=LRP.z_rule
        else:             # default to use epsilon rule if provided rule name not supported
            rule=LRP.eps_rule

        if name is None:
            name=self.model.index2name(index)
        if index is None:
            index=self.model.name2index(name)

        input=to_explain['A'][name]
        R=to_explain["R"][index-1]

        # backward pass with specified LRP rule
        if 'Linear' in str(layer):
            # any linear layer (but for this gravnet model skip the space-dim layer)
            if 'lin_s' in name:
                # run LRP for message_passing here.. so it happens that the message passing step that we want to add is at the same index of the lin_s layer we want to remove
                # so we just swap them
                to_explain["R"][index-2]=R
                # R=rule(layer, input, R, edge_index, edge_weight, index, after_message, before_message, LeakyReLU=False, message_passing=True, transpose=False)
                # to_explain["R"][index-2]=R

            elif 'lin_h' in name:
                # recover R-scores from message passing
                R=rule(layer, input, R, edge_index, edge_weight, index, after_message, before_message, LeakyReLU=False)
                to_explain["R"][index-2]=R
                R, list =LRP.gravnet_rule(layer, input, R, edge_index, edge_weight, index, after_message, before_message, LeakyReLU=False)
                to_explain["R"][index-2]=R
                big_list.append(list)
            else:
                R=rule(layer, input, R, edge_index, edge_weight, index, after_message, before_message, LeakyReLU=False)
                to_explain["R"][index-2]=R

        elif 'LeakyReLU' in str(layer):
            R=rule(layer, input, R, edge_index, edge_weight, index, after_message, before_message, LeakyReLU=True)
            to_explain["R"][index-2]=R

        return R, big_list

def copy_tensor(tensor,dtype=torch.float32):
    """
    create a deep copy of the provided tensor,
    outputs the copy with specified dtype
    """

    return tensor.clone().detach().requires_grad_(True).to(device)


#
# edge=torch.tensor([[1,2,3,4,4,4,4,4,4],[4,1,1,1,6,7,8,9,9]])
#
#
#
# edge
#
# edge[1]==4
#
#
# if (edge[0]==4).sum():
#     print('yay')
