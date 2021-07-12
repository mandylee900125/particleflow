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
import _pickle as cPickle
from sys import getsizeof

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

    # this rule is wrong.. it is just here because it is much quicker for experimentation and gives the correct dimensions needed for debugging
    @staticmethod
    def easy_rule(layer,input,R,index,output_layer,activation_layer):
        print('l1', layer)

        EPSILON=1e-9
        # a=copy_tensor(input)
        # a.retain_grad()

        z = layer.forward(input)

        if activation_layer:
            w = torch.eye(input.shape[1])
        else:
            w = layer.weight

        if output_layer: # for the output layer
            T, W, r = [], [], []

            for i in range(R.shape[1]):
                T.append(R[:,i].reshape(-1,1))
                W.append(w[i,:].reshape(1,-1))
                I = torch.ones_like(R[:,i]).reshape(-1,1)

                Numerator=(input*torch.matmul(T[i],W[i]))
                Denominator=(input*torch.matmul(I,W[i])).sum(axis=1)

                Denominator = Denominator.reshape(-1,1).expand(Denominator.size()[0],Numerator.size()[1])
                r.append(torch.abs(Numerator / (Denominator+EPSILON*torch.sign(Denominator))))

                print('- Finished computing R-scores for output neuron #: ', i+1)
                break
            print(f'- Completed layer: {layer}')
            return r
        else:
            for i in range(len(R)):
                I = torch.ones_like(R[i])

                Numerator=(input*torch.matmul(R[i],w))
                Denominator=(input*torch.matmul(I,w)).sum(axis=1)

                Denominator = Denominator.reshape(-1,1).expand(Denominator.size()[0],Numerator.size()[1])
                R[i]=(torch.abs(Numerator / (Denominator+EPSILON*torch.sign(Denominator))))

                print('- Finished computing R-scores for output neuron #: ', i+1)
                break
            print(f'- Completed layer: {layer}')

            return R


    @staticmethod
    def eps_rule_before_gravnet(layer, input, R, index, output_layer, activation_layer):

        # EPSILON=1e-9
        # a=copy_tensor(input)
        # a.retain_grad()

        # z = layer.forward(a)
        # basically layer.forward does this: output=(torch.matmul(a,torch.transpose(w,0,1))+b) , assuming the following w & b are retrieved

        if activation_layer:
            w = torch.eye(input.shape[1])
        else:
            w = layer.weight
            b = layer.bias

        wt = torch.transpose(w,0,1)

        if output_layer:
            R_list = [None]*R.shape[1]
            Wt = [None]*R.shape[1]
            for output_node in range(R.shape[1]):
                R_list[output_node]=(R[:,output_node].reshape(-1,1).clone())
                Wt[output_node]=(wt[:,output_node].reshape(-1,1))
        else:
            R_list = R
            Wt = [wt]*len(R_list)

        R_previous=[None]*len(R_list)
        for output_node in range(len(R_list)):
            # rep stands for repeated
            a_rep = input.reshape(input.shape[0],input.shape[1],1).expand(-1,-1,R_list[output_node].shape[1])
            wt_rep = Wt[output_node].reshape(1,Wt[output_node].shape[0],Wt[output_node].shape[1]).expand(input.shape[0],-1,-1)

            H = a_rep*wt_rep
            # deno = H.sum(axis=1).reshape(H.sum(axis=1).shape[0],1,H.sum(axis=1).shape[1]).repeat(1,a.shape[1],1).float()
            # deno = H.sum(axis=1).reshape(H.sum(axis=1).shape[0],1,H.sum(axis=1).shape[1]).expand(H.sum(axis=1).reshape(H.sum(axis=1).shape[0],1,H.sum(axis=1).shape[1]).shape[0],a.shape[1],H.sum(axis=1).reshape(H.sum(axis=1).shape[0],1,H.sum(axis=1).shape[1]).shape[2]).float()
            deno = H.sum(axis=1).reshape(H.sum(axis=1).shape[0],1,H.sum(axis=1).shape[1]).expand(-1,input.shape[1],-1)

            G = H/deno

            R_previous[output_node] = (torch.matmul(G, R_list[output_node].reshape(R_list[output_node].shape[0],R_list[output_node].shape[1],1).float()))
            R_previous[output_node] = R_previous[output_node].reshape(R_previous[output_node].shape[0], R_previous[output_node].shape[1])

            print('- Finished computing R-scores for output neuron #: ', output_node+1)
            break

        print(f'- Completed layer: {layer}')
        if (torch.allclose(R_previous[output_node].sum(axis=1), R_list[output_node].sum(axis=1))):
            print('- R score is conserved up to relative tolerance 1e-5')
        elif (torch.allclose(R_previous[output_node].sum(axis=1), R_list[output_node].sum(axis=1), rtol=1e-4)):
            print('- R score is conserved up to relative tolerance 1e-4')
        elif (torch.allclose(R_previous[output_node].sum(axis=1), R_list[output_node].sum(axis=1), rtol=1e-3)):
            print('- R score is conserved up to relative tolerance 1e-3')
        elif (torch.allclose(R_previous[output_node].sum(axis=1), R_list[output_node].sum(axis=1), rtol=1e-2)):
            print('- R score is conserved up to relative tolerance 1e-2')
        elif (torch.allclose(R_previous[output_node].sum(axis=1), R_list[output_node].sum(axis=1), rtol=1e-1)):
            print('- R score is conserved up to relative tolerance 1e-1')

        return R_previous

    @staticmethod
    def eps_rule_after_gravnet(layer, input, R, index, activation_layer):

        # EPSILON=1e-9
        # a=copy_tensor(input)
        # a.retain_grad()

        if activation_layer:
            w = torch.eye(input.shape[1])
        else:
            w = layer.weight
            b = layer.bias

        wt = torch.transpose(w,0,1)

        Wt = [wt]*len(R)

        R_previous=[None]*len(R)
        for output_node in range(len(R)):
        # rep stands for repeated

            H = input.reshape(1,input.shape[0],input.shape[1],1).expand(R[output_node].shape[0],-1,-1,R[output_node].shape[2])*Wt[output_node].reshape(1,1,Wt[output_node].shape[0],Wt[output_node].shape[1]).expand(R[output_node].shape[0],input.shape[0],-1,-1)

            # deno=H.sum(axis=2).reshape(H.sum(axis=2).shape[0],H.sum(axis=2).shape[1],1,H.sum(axis=2).shape[2]).repeat(1,1,a.shape[1],1).float()
            # deno=H.sum(axis=2).reshape(H.sum(axis=2).shape[0],H.sum(axis=2).shape[1],1,H.sum(axis=2).shape[2]).expand(H.sum(axis=2).reshape(H.sum(axis=2).shape[0],H.sum(axis=2).shape[1],1,H.sum(axis=2).shape[2]).shape[0],H.sum(axis=2).reshape(H.sum(axis=2).shape[0],H.sum(axis=2).shape[1],1,H.sum(axis=2).shape[2]).shape[1],a.shape[1],H.sum(axis=2).reshape(H.sum(axis=2).shape[0],H.sum(axis=2).shape[1],1,H.sum(axis=2).shape[2]).shape[3]).float()
            deno=H.sum(axis=2).reshape(H.sum(axis=2).shape[0],H.sum(axis=2).shape[1],1,H.sum(axis=2).shape[2]).expand(-1,-1,input.shape[1],-1)

            G = H/deno

            R_previous[output_node] = torch.matmul(G, R[output_node].reshape(R[output_node].shape[0],R[output_node].shape[1],R[output_node].shape[2],1).float())
            R_previous[output_node] = R_previous[output_node].reshape(R_previous[output_node].shape[0], R_previous[output_node].shape[1],R_previous[output_node].shape[2])

            print('- Finished computing R-scores for output neuron #: ', output_node+1)
            break

        print(f'- Completed layer: {layer}')

        if (torch.allclose(R_previous[output_node].sum(axis=2), R[output_node].sum(axis=2))):
            print('- R score is conserved up to relative tolerance 1e-5')
        elif (torch.allclose(R_previous[output_node].sum(axis=2), R[output_node].sum(axis=2), rtol=1e-4)):
            print('- R score is conserved up to relative tolerance 1e-4')
        elif (torch.allclose(R_previous[output_node].sum(axis=2), R[output_node].sum(axis=2), rtol=1e-3)):
            print('- R score is conserved up to relative tolerance 1e-3')
        elif (torch.allclose(R_previous[output_node].sum(axis=2), R[output_node].sum(axis=2), rtol=1e-2)):
            print('- R score is conserved up to relative tolerance 1e-2')
        elif (torch.allclose(R_previous[output_node].sum(axis=2), R[output_node].sum(axis=2), rtol=1e-1)):
            print('- R score is conserved up to relative tolerance 1e-1')

        print('size of R_previous is: ', getsizeof(R_previous))
        return R_previous

    @staticmethod
    def message_passing_rule(layer, input, R, big_list, edge_index, edge_weight, after_message, before_message, index):

        # b=Data()
        # b['edge_index']=edge_index
        # b['num_nodes']=edge_index[0].shape[0]
        # b['edge_weight']=edge_weight
        #
        # G = to_networkx(b, edge_attrs=['edge_weight'], to_undirected=True, remove_self_loops=False)
        #
        # def nodes_connected(u, v):
        #     return u in G.neighbors(v)

        R_tensor_per_all_nodes=[None]*len(R)
        for output_node in range(len(R)):
            R_tensor_per_all_nodes[output_node]=torch.zeros(R[output_node].shape[0],R[output_node].shape[0],R[output_node].shape[1])

            for node_i in range(R[output_node].shape[0]):
                R_tensor_per_all_nodes[output_node][node_i][node_i]=R[output_node][node_i]

            print('- Finished computing R-scores for for all nodes for output neuron # : ', output_node+1)
            break

        print('- Completed layer: Message Passing')

        if (torch.allclose(R_tensor_per_all_nodes[output_node].sum(axis=1).sum(axis=1), R[output_node].sum(axis=1))):
            print('- R score is conserved up to relative tolerance 1e-5')
        elif (torch.allclose(R_tensor_per_all_nodes[output_node].sum(axis=1).sum(axis=1), R[output_node].sum(axis=1), rtol=1e-4)):
            print('- R score is conserved up to relative tolerance 1e-4')
        elif (torch.allclose(R_tensor_per_all_nodes[output_node].sum(axis=1).sum(axis=1), R[output_node].sum(axis=1), rtol=1e-3)):
            print('- R score is conserved up to relative tolerance 1e-3')
        elif (torch.allclose(R_tensor_per_all_nodes[output_node].sum(axis=1).sum(axis=1), R[output_node].sum(axis=1), rtol=1e-2)):
            print('- R score is conserved up to relative tolerance 1e-2')
        elif (torch.allclose(R_tensor_per_all_nodes[output_node].sum(axis=1).sum(axis=1), R[output_node].sum(axis=1), rtol=1e-1)):
            print('- R score is conserved up to relative tolerance 1e-1')

        return R_tensor_per_all_nodes


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

        start_index=self.model.n_layers                  ##########################
        print('Total number of layers (including activation layers):', start_index)

        ### loop over each single layer
        big_list=[]
        for index in range(start_index+1, 1,-1):
            print(f"Explaining layer {1+start_index+1-index}/{start_index+1-1}")
            if index==start_index+1:
                R, big_list  = self.explain_single_layer(to_explain["pred"], to_explain, big_list, start_index+1, index)
            else:
                R, big_list  = self.explain_single_layer(R, to_explain, big_list, start_index+1, index)

            if len(big_list)==0:
                with open(to_explain["outpath"]+'/'+to_explain["load_model"]+f'/R{index}.pkl', 'wb') as f:
                    cPickle.dump(R, f)
            else:
                with open(to_explain["outpath"]+'/'+to_explain["load_model"]+f'/R{index}.pkl', 'wb') as f:
                    cPickle.dump(big_list, f)

        print("Finished explaing all layers.")

    def explain_single_layer(self, R, to_explain, big_list, output_layer_index, index=None, name=None):
        # preparing variables required for computing LRP
        layer=self.model.get_layer(index=index,name=name)

        if name is None:
            name=self.model.index2name(index)
        if index is None:
            index=self.model.name2index(name)

        input=to_explain['A'][name]

        if index==output_layer_index:
            output_layer_bool=True
        else:
            output_layer_bool=False

        # it works out of the box that the conv1.lin_s layer which we don't care about is in the same place of the message passing.. so we can just replace its action
        if 'conv1.lin_s' in str(name):
            big_list = self.message_passing_rule(layer, input, R, big_list, to_explain["edge_index"], to_explain["edge_weight"], to_explain["after_message"], to_explain["before_message"], index)
            return R, big_list

        # if you haven't hit the message passing step yet
        if len(big_list)==0:
            if 'Linear' in str(layer):
                R = self.eps_rule_before_gravnet(layer, input, R, index, output_layer_bool, activation_layer=False)
            elif 'LeakyReLU' or 'ELU' in str(layer):
                R = self.eps_rule_before_gravnet(layer, input, R, index, output_layer_bool, activation_layer=True)
        else:
            if 'Linear' in str(layer):
                big_list = self.eps_rule_after_gravnet(layer, input, big_list, index, activation_layer=False)
            elif 'LeakyReLU' or 'ELU' in str(layer):
                big_list =  self.eps_rule_after_gravnet(layer, input, big_list, index, activation_layer=True)

        return R, big_list

def copy_tensor(tensor,dtype=torch.float32):
    """
    create a deep copy of the provided tensor,
    outputs the copy with specified dtype
    """

    return tensor.clone().detach().requires_grad_(True).to(device)


##-----------------------------------------------------------------------------
#
# arep=torch.transpose(a[0].repeat(6, 1),0,1)   # repeat it 6 times
# H=arep*wt
#
# G = H/H.sum(axis=0).float()
#
# Num = torch.matmul(G, R[0].float())
#
# print('Num.sum()', Num.sum())
#
# print(R[0].sum())
