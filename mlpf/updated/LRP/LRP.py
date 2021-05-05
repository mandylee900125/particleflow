import torch
import torch.nn as nn
from torch.nn import Sequential as Seq,Linear,ReLU,BatchNorm1d
from torch_scatter import scatter_mean
import numpy as np
import json
import model_io

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
    def eps_rule(layer,input,R, edge_index, edge_weight, flag):
        a=copy_tensor(input)
        a.retain_grad()

        if flag:
            # index=copy_tensor(edge_index)
            # index.retain_grad()
            weight=copy_tensor(edge_weight)
            weight.retain_grad()

            z=layer.forward(a, edge_index, weight)
            s=R/(z)
            (z*s.data).sum().backward()

        else:
            z = layer.forward(a)[0]
            s=R/(z+LRP.EPSILON*torch.sign(z))
            (z*s.data).sum().backward()

        c=a.grad
        return a*c

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

        start_index=self.model.n_layers                  ##########################
        to_explain['R'][start_index+1]=copy_tensor(pred)

        # # TODO: not 4 but 1
        for index in range(start_index+1,1,-1):
            self.explain_single_layer(to_explain, edge_index, edge_weight, index)

        return to_explain['R']

    def explain_single_layer(self,to_explain, edge_index, edge_weight, index=None,name=None,):
        # todo: deal with special case when previous layer has not been explained

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

        R=to_explain["R"][index]

        # backward pass with specified LRP rule
        if index==3:
            R=rule(layer,input,R, edge_index, edge_weight, flag=True)
        else:
            R=rule(layer,input,R, edge_index, edge_weight, flag=False)

        # store result
        to_explain["R"][index-1]=R

def copy_tensor(tensor,dtype=torch.float32):
    """
    create a deep copy of the provided tensor,
    outputs the copy with specified dtype
    """

    return tensor.clone().detach().requires_grad_(True).to(device)
