import numpy as np
import torch
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch

use_gpu = torch.cuda.device_count()>0
multi_gpu = torch.cuda.device_count()>1

#define the global base device
if use_gpu:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# define a function that casts the ttbar dataset into a dataloader for efficient NN training
def data_to_loader_ttbar(full_dataset, n_train, n_valid, batch_size):

    train_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=0, stop=n_train))
    valid_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=n_train, stop=n_train+n_valid))

    # preprocessing the train_dataset in a good format for passing correct batches of events to the GNN
    train_data=[]
    for i in range(len(train_dataset)):
        train_data = train_data + train_dataset[i]

    # preprocessing the valid_dataset in a good format for passing correct batches of events to the GNN
    valid_data=[]
    for i in range(len(valid_dataset)):
        valid_data = valid_data + valid_dataset[i]

    #hack for multi-gpu training
    if not multi_gpu:
        def collate(items):
            l = sum([items], [])
            return Batch.from_data_list(l)
    else:
        def collate(items):
            l = sum([items], [])
            return l

    train_loader = DataListLoader(train_data, batch_size, pin_memory=True, shuffle=True)
    train_loader.collate_fn = collate
    valid_loader = DataListLoader(valid_data, batch_size, pin_memory=True, shuffle=False)
    valid_loader.collate_fn = collate

    return train_loader, valid_loader

# define a function that casts the dataset into a dataloader for efficient NN training
def data_to_loader_qcd(full_dataset, n_test, batch_size):

    test_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=0, stop=n_test))

    # preprocessing the test_dataset in a good format for passing correct batches of events to the GNN
    test_data=[]
    for i in range(len(test_dataset)):
        test_data = test_data + test_dataset[i]

    #hack for multi-gpu training
    if not multi_gpu:
        def collate(items):
            l = sum([items], [])
            return Batch.from_data_list(l)
    else:
        def collate(items):
            l = sum([items], [])
            return l

    test_loader = DataListLoader(test_data, batch_size, pin_memory=True, shuffle=False)
    test_loader.collate_fn = collate

    return test_loader
