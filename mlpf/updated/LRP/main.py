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

import networkx as nx
from torch_geometric.utils.convert import to_networkx
from tabulate import tabulate

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

    # args = parse_args()

    # the next part initializes some args values (to run the script not from terminal)
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d

    args = objectview({'train': False, 'n_train': 1, 'n_valid': 1, 'n_test': 2, 'n_epochs': 2, 'patience': 100, 'hidden_dim':256, 'input_encoding': 12, 'encoding_dim': 125,
    'batch_size': 1, 'model': 'PFNet7', 'target': 'gen', 'dataset': '../../../test_tmp_delphes/data/pythia8_ttbar', 'dataset_qcd': '../../../test_tmp_delphes/data/pythia8_qcd',
    'outpath': '../../../prp/models/LRP/', 'optimizer': 'adam', 'lr': 0.001, 'alpha': 1, 'dropout': 0,
    'space_dim': 4, 'propagate_dimensions': 22,'nearest': 16, 'overwrite': True,
    'load': True, 'load_epoch': 19, 'load_model': 'DataParallel_gen_ntrain_400_nepochs_100_batch_size_4_lr_0.0001_clf_noskip',
    'evaluate': False, 'evaluate_on_cpu': False, 'classification_only': True, 'nn1': False, 'conv2': False, 'nn3': False, 'title': '',
    'explain': True})

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
            # print('name is:', name)
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

        c0, c1, c2, c3, c4, c5 = [0]*12, [0]*12, [0]*12, [0]*12, [0]*12, [0]*12
        c0t, c1t, c2t, c3t, c4t, c5t = 0, 0, 0, 0 ,0 ,0

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

            model.set_dest(to_explain["A"])
            results.append(explainer.explain(to_explain,save=False,return_result=True, signal=signal))

            if i==0:
                print('LRP layers are:', to_explain['A'].keys())
                print(results[i][0])         # 0 indicates the first layer (i.e. relevance scores of the input)

            if True:
                res = torch.abs(results[i][0])
            else:
                res = torch.cat((res,torch.abs(results[i][0])), dim=0)

            # ------------------------------------
            batch['edge_index']=edge_index
            batch['edge_weight']=edge_weight

            batch['type']=res[:,0]

            c0t = c0t+ (torch.argmax(cand_ids_one_hot, axis=1)==0).sum().item()    # use batch.ycand_id instead of cand_ids_one_hot if you want to get the true classes
            c1t = c1t+ (torch.argmax(cand_ids_one_hot, axis=1)==1).sum().item()
            c2t = c2t+ (torch.argmax(cand_ids_one_hot, axis=1)==2).sum().item()
            c3t = c3t+ (torch.argmax(cand_ids_one_hot, axis=1)==3).sum().item()
            c4t = c4t+ (torch.argmax(cand_ids_one_hot, axis=1)==4).sum().item()
            c5t = c5t+ (torch.argmax(cand_ids_one_hot, axis=1)==5).sum().item()

            for index in range(12):
                batch['top_50_index']=torch.topk(res[:,index], 2500)[1]
                color = cand_ids_one_hot.argmax(axis=1)[batch['top_50_index']]

                c0[index] = c0[index]+ (color==0).sum().item()
                c1[index] = c1[index]+ (color==1).sum().item()
                c2[index] = c2[index]+ (color==2).sum().item()
                c3[index] = c3[index]+ (color==3).sum().item()
                c4[index] = c4[index]+ (color==4).sum().item()
                c5[index] = c5[index]+ (color==5).sum().item()


            if i==20:
                break

            ##### to pick the most relevant 50 nodes:
            if False:
                batch['edge_index']=edge_index
                batch['edge_weight']=edge_weight
                batch.num_nodes=500

                batch['top_50_values']=torch.topk(res[:,3], 500)[0]
                batch['top_50_index']=torch.topk(res[:,3], 500)[1]

                indices = torch.topk(res[:,3], 500)[1]

                list1=[]
                list2=[]
                c1=0
                c2=0
                for index in indices:
                    for i in range(len(edge_index[0])):
                        if edge_index[0][i]==index:
                            if edge_index[1][i] in indices:
                                list1.append(index)
                                list2.append(edge_index[1][i])

                new_edge_index = torch.stack((torch.tensor(list1),torch.tensor(list2)))

                unique = torch.unique_consecutive(new_edge_index[0])

                l1=[]
                l2=[0]*new_edge_index[0].shape[0]
                s=0
                c=-1
                for elem in new_edge_index[0]:
                    if elem!=s:
                        c=c+1
                        s=elem
                    l1.append(c)

                for i,elem2 in enumerate(new_edge_index[1]):
                    for j,elem in enumerate(new_edge_index[0]):
                        if elem2==elem:
                            l2[i]=l1[j]

                batch['edge_index'] = torch.stack((torch.tensor(l1),torch.tensor(l2)))

                # to label the nodes by their class
                color = cand_ids_one_hot.argmax(axis=1)[batch['top_50_index']]
                print('color', color)
                G_type_50 = to_networkx(batch, node_attrs=['top_50_values'], edge_attrs=None, to_undirected=True, remove_self_loops=False)
                nx.draw(G_type_50, font_weight='bold', node_color=color)
                plt.savefig("eta_505.png")
                print()


            # G_type = to_networkx(batch, node_attrs=['type'], edge_attrs=['edge_weight'], to_undirected=True, remove_self_loops=False)
            # # nx.draw(G_type)
            # # plt.savefig("type.png")
            # pos=nx.spring_layout(G_type)   #G is my graph
            # nx.draw(G_type,pos,node_color='#A0CBE2',edge_color='#BB0000',width=2,edge_cmap=plt.cm.Blues,with_labels=True)
            # plt.savefig("type.png", dpi=500, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1)
            # print('finished 1st graph..')

            # batch['pt']=res[:,1]
            # G_pt = to_networkx(batch, node_attrs=['pt'], edge_attrs=['edge_weight'], to_undirected=True, remove_self_loops=False)
            # nx.draw(G_pt)
            # plt.savefig("pt.png", dpi=1000)
            # print('finished 2nd graph..')

            # batch['eta']=res[:,2]
            # G_eta = to_networkx(batch, node_attrs=['eta'], edge_attrs=['edge_weight'], to_undirected=True, remove_self_loops=False)
            # nx.draw(G_eta)
            # plt.savefig("eta.png")
        print('c0t is', c0t)
        print('c1t is', c1t)
        print('c2t is', c2t)
        print('c3t is', c3t)
        print('c4t is', c4t)
        print('c5t is', c5t)

        print('c0 is ', c0)
        print('c1 is ', c1)
        print('c2 is ', c2)
        print('c3 is ', c3)
        print('c4 is ', c4)
        print('c5 is ', c5)

        print('c0 as % is ', [int(round((i * 100/c0t),0)) for i in c0])
        print('c1 as % is ', [int(round((i * 100/c1t),0)) for i in c1])
        print('c2 as % is ', [int(round((i * 100/c2t),0)) for i in c2])
        print('c3 as % is ', [int(round((i * 100/c3t),0)) for i in c3])
        print('c4 as % is ', [int(round((i * 100/c4t),0)) for i in c4])
        print('c5 as % is ', [int(round((i * 100/c5t),0)) for i in c5])


        # names = ['None', 'charged_hadron', 'neutral_hadron', 'photon', 'electron', 'muon']
        # titles = ['type', 'Et/pt', 'eta', 'sin phi', 'cos phi', 'E/P', 'Eem/eta_outer', 'Ehad/sin phi_outer', '0/cos phi_outer', '0/charge', '0/is_gen_muon', '0/is_gen_electron']
        #
        # data = [titles] + list(zip(names, weights, costs, unit_costs))
        table = [['PID | feature', 'type', 'Et/pt', 'eta', 'sin phi', 'cos phi', 'E/P', 'Eem/eta_outer', 'Ehad/sin phi_outer', '0/cos phi_outer', '0/charge', '0/is_gen_muon', '0/is_gen_electron'],
        ['None']+c0,
        ['charged_hadron']+c1,
        ['neutral_hadron']+c2,
        ['photon']+c3,
        ['electron']+c4,
        ['muon']+c5]

        print(tabulate(table))


        table = [['PID | feature', 'type', 'Et/pt', 'eta', 'sin phi', 'cos phi', 'E/P', 'Eem/eta_outer', 'Ehad/sin phi_outer', '0/cos phi_outer', '0/charge', '0/is_gen_muon', '0/is_gen_electron'],
        ['None']+[int(round((i * 100/c0t),0)) for i in c0],
        ['charged_hadron']+[int(round((i * 100/c1t),0)) for i in c1],
        ['neutral_hadron']+[int(round((i * 100/c2t),0)) for i in c2],
        ['photon']+[int(round((i * 100/c3t),0)) for i in c3],
        ['electron']+[int(round((i * 100/c4t),0)) for i in c4],
        ['muon']+[int(round((i * 100/c5t),0)) for i in c5]]

        print(tabulate(table))

        # fig = plt.figure(figsize=(5,5))
        # # print('homy', results[i][0][:,0].detach().numpy().shape)
        # plt.hist(res[:,0].detach().numpy(), histtype="step", label="type label (cluster/track)")
        # plt.hist(res[:,1].detach().numpy(), histtype="step", label="pt")
        # plt.hist(res[:,2].detach().numpy(), histtype="step", label="eta")
        # plt.hist(res[:,5].detach().numpy(), histtype="step", label="energy")
        # plt.xlabel('relevance score')
        # plt.legend(loc="best", frameon=False)
        # plt.xlim(0,1e-3)
        # plt.yscale('log')
        # plt.savefig("mygraph.png")
        # plt.close(fig)


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


#
# #
# import numpy as np
# import torch
# a = torch.tensor([[4,3], [2,1], [3,5], [3,4], [2.1,2]])
# b = a[:,0]
#
# print(b)
#
# import matplotlib.pyplot as plt
# rng = np.random.RandomState(10)  # deterministic random data
# a = np.hstack((rng.normal(size=1000),
#                rng.normal(loc=5, scale=2, size=1000)))
# _ = plt.hist(b, bins='auto')  # arguments are passed to np.histogram
# plt.title("Histogram with 'auto' bins")
# plt.savefig("mygraph.png")
#
# np.histogram(b)
#
# res.shape
#
# fig = plt.figure(figsize=(5,5))
# # print('homy', results[i][0][:,0].detach().numpy().shape)
# plt.hist(res[:,0].detach().numpy(), histtype="step", label="type label (cluster/track)")
# plt.hist(res[:,1].detach().numpy(), histtype="step", label="pt")
# plt.hist(res[:,2].detach().numpy(), histtype="step", label="eta")
# plt.xlabel('relevance score')
# plt.legend(loc="best", frameon=False)
# # plt.xlim(0,1e-3)
# plt.yscale('log')
# plt.savefig("mygraph.png")
# plt.close(fig)

# #
# color
# #
# #
# # batch.ycand_id
#
# (torch.argmax(batch.ycand_id, axis=1)==0).sum()
# (torch.argmax(batch.ycand_id, axis=1)==1).sum()
# (torch.argmax(batch.ycand_id, axis=1)==2).sum()
# (torch.argmax(batch.ycand_id, axis=1)==3).sum()
# (torch.argmax(batch.ycand_id, axis=1)==4).sum()
# (torch.argmax(batch.ycand_id, axis=1)==5).sum()
#
# res[:,2]
# res[:,1]
# batch['top_50_index']=torch.topk(res[:,10], 4000)[1]
# color = cand_ids_one_hot.argmax(axis=1)[batch['top_50_index']]
#
# color
# color[(color!=1)]
#
# (color[(color!=1)]==3).sum()
#
# (color==1).sum()
#
# batch['top_50_index']=torch.topk(res[:,0], 2800)[1]
# color = cand_ids_one_hot.argmax(axis=1)[batch['top_50_index']]
#
# color
# color[(color!=1)]
#
# cand_ids_one_hot.argmax(axis=1)[]
