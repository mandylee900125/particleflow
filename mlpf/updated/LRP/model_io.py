import torch
import torch.nn as nn
from torch.nn import Sequential as Seq,Linear,ReLU,BatchNorm1d
from torch_scatter import scatter_mean
import numpy as np
import json

use_gpu = torch.cuda.device_count()>0
multi_gpu = torch.cuda.device_count()>1

#define the global base device
if use_gpu:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

#Choose special layers from: dict_keys(['.dropout', '.conv1.lin_s', '.conv1.lin_h', '.conv1.lin_p', '.conv2.lin_l', '.conv2.lin_r', '.nn2.0', '.nn2.1', '.nn2.2', '.nn2.3', '.nn2.4', '.nn2.5', '.nn2.6', '.nn3.0', '.nn3.1', '.nn3.2', '.nn3.3', '.nn3.4', '.nn3.5', '.nn3.6'])

class model_io:
    SPECIAL_LAYERS=[
        ".nn2.0",
        # ".conv1.lin_h",
        # ".conv1.lin_p"
    ]

    def __init__(self,model,
                model_state_dict,
                activation_dest, dic):

        self.model=model
        self.model.load_state_dict(model_state_dict)
        self.dest=activation_dest
        self.dic=dic

        # declare variables
        self.L=dict()           # layers
        self.A=activation_dest  # activations
        # self.R=dict()          # relevance scores

        self._rules=dict()     # rules to use for each layer
        self._hook_handles=[]  # collection of all hook handles

        # # extract layers and register hooks
        # self._extract_layers("",model,)

        self.L=dict()
        for name, module in model.named_modules():
            if name=='conv1' or name=='conv2':
                self.L[name]=module
            else:
                self.L['.'+name]=module

        for key, value in list(self.L.items()):
            if key not in self.dic.keys():
                del self.L[key]

        self.n_layers=len(self.L.keys())

        # register rules for each layer
        self._register_rules()

        # register special layers
        self.special_layers=list()
        for key in model_io.SPECIAL_LAYERS:
            full_key=[layer_name for layer_name in self.L.keys() if key in layer_name][0]
            self.special_layers.append(full_key)


    """
    rules functions
    """
    def _register_rules(self):
        for layer_name in self.L.keys():
            layer=self.L[layer_name]
            layer_class=layer.__class__.__name__
            if layer_class=="BatchNorm1d":
                rule="z"
            else:
                rule="eps"
            self._rules[layer_name]=rule

    def get_rule(self,index=None,layer_name=None):
        assert (not index is None) or (not layer_name is None), "at least one of (index,name) must be provided"
        if layer_name is None:
            layer_name=self.index2name(index)

        if hasattr(self,"_rules"):
            return self._rules[layer_name]
        else:
            self._register_rules()
            return self._rules[layer_name]


    """
    layer functions
    """

    def _extract_layers(self,name,model):
        l=list(model.named_children())

        if len(l)==0:
            self.L[name]=copy_layer(model)
        else:
            l=list(model.named_children())
            for i in l:
                self._extract_layers(name+"."+i[0],i[1])

    def get_layer(self,index=None,name=None):
        assert (not index is None) or (not name is None), "at least one of (index,name) must be provided"
        if name is None:
            name=self.index2name(index)
        return self.L[name]


    """
    general getters
    """
    def index2name(self,idx:int)->str:
        if not hasattr(self,"_i2n"):
            self._i2n=[]
            for i,n in enumerate(self.A.keys()):
                self._i2n.append(n)
        return self._i2n[idx-2]

    def name2index(self,name:str)->int:
        if not hasattr(self,"_i2n"):
            self._i2n=[]
            for i,n in enumerate(self.A.keys()):
                self._i2n.append(n)
        return self._i2n.index(name)



    """
    reset and setter functions
    """
    def _clear_hooks(self):
        for hook in self._hook_handles:
            hook.remove()

    def reset(self):
        """
        reset the prepared model
        """
        pass
        # self._clear_hooks()
        # self.A=dict()
        # self.R=dict()

    def set_dest(self,activation_dest):
        self.A=activation_dest

def copy_layer(layer):
    """
    create a deep copy of provided layer
    """
    layer_cp=eval("nn."+layer.__repr__())
    layer_cp.load_state_dict(layer.state_dict())

    return layer_cp.to(device)


def copy_tensor(tensor,dtype=torch.float32):
    """
    create a deep copy of the provided tensor,
    outputs the copy with specified dtype
    """

    return tensor.clone().detach().requires_grad_(True).to(device)

#---------------------------------------------------------------------
# import sys
# sys.path.insert(1, '../')
# sys.path.insert(1, '../../../plotting/')
# sys.path.insert(1, '../../../mlpf/plotting/')
# import args
# from args import parse_args
# from graph_data_delphes import PFGraphDataset, one_hot_embedding
# from data_preprocessing import data_to_loader_ttbar, data_to_loader_qcd
# from model import PFNet7
#
# class objectview(object):
#     def __init__(self, d):
#         self.__dict__ = d
#
# args = objectview({'train': True, 'n_train': 1, 'n_valid': 1, 'n_test': 2, 'n_epochs': 3, 'patience': 100, 'hidden_dim':32, 'input_encoding': 12, 'encoding_dim': 256,
# 'batch_size': 2, 'model': 'PFNet7', 'target': 'gen', 'dataset': '../../../test_tmp_delphes/data/pythia8_ttbar', 'dataset_qcd': '../../../test_tmp_delphes/data/pythia8_qcd',
# 'outpath': '../../../prp/models/', 'optimizer': 'adam', 'lr': 0.001, 'alpha': 1, 'dropout': 0.0,
# 'space_dim': 4, 'propagate_dimensions': 22,'nearest': 16, 'overwrite': True,
# 'load': False, 'load_epoch': 0, 'load_model': 'PFNet7_gen_ntrain_1_nepochs_3_batch_size_2_lr_0.001_clf',
# 'evaluate': False, 'evaluate_on_cpu': False, 'classification_only': True, 'nn1': False,
# 'explain': True})
#
# # define the dataset (assumes the data exists as .pt files in "processed")
# print('Processing the data..')
# full_dataset_ttbar = PFGraphDataset(args.dataset)
# full_dataset_qcd = PFGraphDataset(args.dataset_qcd)
#
# # constructs a loader from the data to iterate over batches
# print('Constructing data loaders..')
# train_loader, valid_loader = data_to_loader_ttbar(full_dataset_ttbar, args.n_train, args.n_valid, batch_size=args.batch_size)
# test_loader = data_to_loader_qcd(full_dataset_qcd, args.n_test, batch_size=args.batch_size)
#
# # element parameters
# input_dim = 12
#
# #one-hot particle ID and momentum
# output_dim_id = 6
# output_dim_p4 = 6
#
# patience = args.patience
#
# model_classes = {"PFNet7": PFNet7}
#
# model_class = model_classes[args.model]
# model_kwargs = {'input_dim': input_dim,
#                 'hidden_dim': args.hidden_dim,
#                 'input_encoding': args.input_encoding,
#                 'encoding_dim': args.encoding_dim,
#                 'output_dim_id': output_dim_id,
#                 'output_dim_p4': output_dim_p4,
#                 'dropout_rate': args.dropout,
#                 'space_dim': args.space_dim,
#                 'propagate_dimensions': args.propagate_dimensions,
#                 'nearest': args.nearest,
#                 'target': args.target,
#                 'nn1': args.nn1}
#
# print('Loading a previously trained model..')
# model = model_class(**model_kwargs)
# outpath = args.outpath + args.load_model
# PATH = outpath + '/epoch_' + str(args.load_epoch) + '_weights.pth'
# model.load_state_dict(torch.load(PATH, map_location=device))
# # print(model)
#
# if args.explain:
#     state_dict=torch.load(PATH,map_location=device)
#     model=model_io(model,state_dict,dict())
