import torch
import torch.nn as nn
from torch.nn import Sequential as Seq,Linear,ReLU,BatchNorm1d
from torch_scatter import scatter_mean
import numpy as np
import json
import model_io
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy

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
    def eps_rule(layer,input,R, edge_index, edge_weight, index, after_message, before_message, LeakyReLU, message_passing, transpose):
        EPSILON=1e-9
        a=copy_tensor(input)
        a.retain_grad()

        z = layer.forward(a)

        if LeakyReLU:
            w = torch.eye(a.shape[1])

        elif message_passing:

            edge_index_detached=edge_index.detach()
            edge_weight_detached=edge_weight.detach()

            # need to stack/cat to make the Adjacency matrix symmetric (because this is an undirected graph but scipy doesn't know that)
            # TODO: make it mean not sum.. right now w corrsponds to weighted sum of nodes.. i have to divide w by a scale factor
            edge_indices=torch.stack([torch.cat((edge_index_detached[0], edge_index_detached[1]), axis=0),torch.cat((edge_index_detached[1], edge_index_detached[0]), axis=0)])
            edge_weights = torch.cat([edge_weight_detached,edge_weight_detached])

            w = torch.from_numpy(to_scipy_sparse_matrix(edge_indices,edge_weights).toarray())
            w = w - 0.5*torch.diag(torch.diag(w))    # to avoid double counting when i stacked the edge indices

            z = after_message.transpose(0,1)
            a = before_message.transpose(0,1)
            R = R.transpose(0,1)

        else:
            w = layer.weight

        if transpose:
            R = R.transpose(0,1)

        I = torch.ones_like(R)

        Numerator=(a*torch.matmul(R,w))
        Denominator=(a*torch.matmul(I,w)).sum(axis=1)

        Denominator = Denominator.reshape(-1,1).expand(Denominator.size()[0],Numerator.size()[1])

        R = Numerator / (Denominator+EPSILON*torch.sign(Denominator))
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
        for index in range(start_index+1, 1,-1):
            self.explain_single_layer(to_explain, edge_index, edge_weight, after_message, before_message, index)
            # print(to_explain['R'][index-1].shape, to_explain['R'][index-1])
        return to_explain['R']

    def explain_single_layer(self, to_explain, edge_index, edge_weight, after_message, before_message, index=None,name=None,):

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
                R=rule(layer, input, R, edge_index, edge_weight, index, after_message, before_message, LeakyReLU=False, message_passing=True, transpose=False)
                to_explain["R"][index-2]=R
            elif 'lin_h' in name:
                # flag transpose is needed to recover what was done in message_passing
                R=rule(layer, input, R, edge_index, edge_weight, index, after_message, before_message, LeakyReLU=False, message_passing=False, transpose=True)
                to_explain["R"][index-2]=R
            else:
                R=rule(layer, input, R, edge_index, edge_weight, index, after_message, before_message, LeakyReLU=False, message_passing=False, transpose=False)
                to_explain["R"][index-2]=R

        elif 'LeakyReLU' in str(layer):
            R=rule(layer, input, R, edge_index, edge_weight, index, after_message, before_message, LeakyReLU=True, message_passing=False, transpose=False)
            to_explain["R"][index-2]=R


def copy_tensor(tensor,dtype=torch.float32):
    """
    create a deep copy of the provided tensor,
    outputs the copy with specified dtype
    """

    return tensor.clone().detach().requires_grad_(True).to(device)
