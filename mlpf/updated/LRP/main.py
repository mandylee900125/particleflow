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

sys.path.insert(1, '../')
sys.path.insert(1, '../../../plotting/')
sys.path.insert(1, '../../../mlpf/plotting/')
import args
from args import parse_args
from graph_data_delphes import PFGraphDataset, one_hot_embedding
from data_preprocessing import data_to_loader_ttbar, data_to_loader_qcd
import evaluate
from evaluate import make_plots, Evaluate
from plot_utils import plot_confusion_matrix
from model_LRP import PFNet7

from LRP import LRP
from model_io import model_io

#Ignore divide by 0 errors
np.seterr(divide='ignore', invalid='ignore')

#Get a unique directory name for the model
def get_model_fname(dataset, model, n_train, n_epochs, lr, target_type, batch_size, task, title):
    model_name = type(model).__name__
    model_params = sum(p.numel() for p in model.parameters())
    import hashlib
    model_cfghash = hashlib.blake2b(repr(model).encode()).hexdigest()[:10]
    model_user = os.environ['USER']

    model_fname = '{}_{}_ntrain_{}_nepochs_{}_batch_size_{}_lr_{}_{}'.format(
        model_name,
        target_type,
        n_train,
        n_epochs,
        batch_size,
        lr,
        task,
        title)
    return model_fname


if __name__ == "__main__":

    args = parse_args()

    # # the next part initializes some args values (to run the script not from terminal)
    # class objectview(object):
    #     def __init__(self, d):
    #         self.__dict__ = d
    #
    # args = objectview({'train': False, 'n_train': 1, 'n_valid': 1, 'n_test': 2, 'n_epochs': 2, 'patience': 100, 'hidden_dim':256, 'input_encoding': 12, 'encoding_dim': 125,
    # 'batch_size': 2, 'model': 'PFNet7', 'target': 'gen', 'dataset': '../../../test_tmp_delphes/data/pythia8_ttbar', 'dataset_qcd': '../../../test_tmp_delphes/data/pythia8_qcd',
    # 'outpath': '../../../prp/models/LRP/', 'optimizer': 'adam', 'lr': 0.001, 'alpha': 1, 'dropout': 0,
    # 'space_dim': 4, 'propagate_dimensions': 22,'nearest': 16, 'overwrite': True,
    # 'load': True, 'load_epoch': 19, 'load_model': 'PFNet7_gen_ntrain_1_nepochs_20_batch_size_2_lr_0.001_clf__noskip',
    # 'evaluate': False, 'evaluate_on_cpu': False, 'classification_only': True, 'nn1': False, 'conv2': False, 'nn3': False, 'title': '',
    # 'explain': True})

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

    model_classes = {"PFNet7": PFNet7}

    model_class = model_classes[args.model]
    model_kwargs = {'input_dim': input_dim,
                    'hidden_dim': args.hidden_dim,
                    'input_encoding': args.input_encoding,
                    'encoding_dim': args.encoding_dim,
                    'output_dim_id': output_dim_id,
                    'output_dim_p4': output_dim_p4,
                    'dropout_rate': args.dropout,
                    'space_dim': args.space_dim,
                    'propagate_dimensions': args.propagate_dimensions,
                    'nearest': args.nearest,
                    'target': args.target,
                    'nn1': args.nn1,
                    'conv2': args.conv2,
                    'nn3': args.nn3}

    print('Loading a previously trained model..')
    model = model_class(**model_kwargs)
    outpath = args.outpath + args.load_model
    PATH = outpath + '/epoch_' + str(args.load_epoch) + '_weights.pth'

    state_dict = torch.load(PATH, map_location=device)

    if "DataParallel" in args.load_model:
        state_dict = torch.load(PATH, map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove module.
            new_state_dict[name] = v
        state_dict=new_state_dict

    model.load_state_dict(state_dict)

    if args.explain:
        model.train()
        print(model)

        results = []
        signal =torch.tensor([1,0,0,0,0,0],dtype=torch.float32).to(device)

        # create some hooks to retrieve intermediate activations
        activation = {}
        hooks={}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = input[0]
            return hook

        param=dict()
        for i, parameter in enumerate(model.parameters()):
            param[i]=parameter
            print(parameter.shape)

        for name, module in model.named_modules():
            if (type(module)==nn.Linear) or (type(module)==nn.LeakyReLU):
                hooks[name] = module.register_forward_hook(get_activation("." + name))

        for i, batch in enumerate(train_loader):
            t0 = time.time()

            if multi_gpu:
                X = batch
            else:
                X = batch.to(device)

            if i==0:
                # could be defined better
                # I run at least one forward pass to get the activation to use them in defining the LRP layers
                cand_ids_one_hot, cand_p4, target_ids_one_hot, target_p4, edge_index, edge_weight, after_message, before_message = model(X)
                model=model_io(model,state_dict,dict(),activation)
                explainer=LRP(model)

            else:
                cand_ids_one_hot, cand_p4, target_ids_one_hot, target_p4, edge_index, edge_weight, after_message, before_message = model.model(X)

            to_explain={"A":activation,"inputs":dict(x=X.x,
                                                batch=X.batch),"y":target_ids_one_hot,"R":dict(), "pred":cand_ids_one_hot,
                                                "edge_index":edge_index, "edge_weight":edge_weight, "after_message":after_message, "before_message":before_message}

            print('LRP layers are:', to_explain['A'].keys())

            model.set_dest(to_explain["A"])
            results.append(explainer.explain(to_explain,save=False,return_result=True, signal=signal))

            print(results[i][0])         # 0 indicates the first layer (i.e. relevance scores of the input)
            break

    # evaluate the model
    if args.evaluate:
        if args.evaluate_on_cpu:
            device = "cpu"

        model = model.to(device)
        model.eval()
        Evaluate(model, test_loader, outpath, args.target, device, args.load_epoch)

## -----------------------------------------------------------
# # to retrieve a stored variable in pkl file
# import pickle
# with open('../../test_tmp_delphes/experiments/PFNet7_gen_ntrain_2_nepochs_3_batch_size_3_lr_0.0001/confusion_matrix_plots/cmT_normed_epoch_0.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     a = pickle.load(f)
