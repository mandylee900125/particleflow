import args
from args import parse_args
import sklearn
import sklearn.metrics
import numpy as np
import pandas, mplhep
import pickle as pkl
import time, math

import sys
import os.path as osp
sys.path.insert(1, '../../plotting/')
sys.path.insert(1, '../../mlpf/plotting/')

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
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
import mplhep as hep
plt.style.use(hep.style.ROOT)

use_gpu = torch.cuda.device_count()>0
multi_gpu = torch.cuda.device_count()>1

#define the global base device
if use_gpu:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

from plot_utils import plot_confusion_matrix
from make_plots import plot_regression, plot_all_distributions, plot_pt_eta, plot_num_particles_pid, draw_efficiency_fakerate, get_eff, get_fake, plot_reso

def Evaluate(model, test_loader, path, target, device, epoch, which_data):
    outpath=path + '/'
    sample_title_qcd = "QCD, 14 TeV, PU200"
    sample_title_ttbar = "$t\\bar{t}$, 14 TeV, PU200"

    gen_ids_all, gen_p4_all = [], []
    pred_ids_all, pred_p4_all = [], []
    cand_ids_all, cand_p4_all = [], []

    gen_list = {"null":[], "chhadron":[], "nhadron":[], "photon":[], "electron":[], "muon":[]}
    pred_list = {"null":[], "chhadron":[], "nhadron":[], "photon":[], "electron":[], "muon":[]}
    cand_list = {"null":[], "chhadron":[], "nhadron":[], "photon":[], "electron":[], "muon":[]}

    print('Making predictions on ' + which_data)

    for i, batch in enumerate(test_loader):
        if multi_gpu:
            X = batch
        else:
            X = batch.to(device)

        pred_ids_one_hot, pred_p4, gen_ids_one_hot, gen_p4, cand_ids_one_hot, cand_p4 = model(X)

        # to make "num_gen vs num_pred" plots
        _, gen_ids = torch.max(gen_ids_one_hot, -1)
        _, pred_ids = torch.max(pred_ids_one_hot, -1)
        _, cand_ids = torch.max(cand_ids_one_hot, -1)

        gen_list["null"].append((gen_ids==0).sum().item())
        gen_list["chhadron"].append((gen_ids==1).sum().item())
        gen_list["nhadron"].append((gen_ids==2).sum().item())
        gen_list["photon"].append((gen_ids==3).sum().item())
        gen_list["electron"].append((gen_ids==4).sum().item())
        gen_list["muon"].append((gen_ids==5).sum().item())

        pred_list["null"].append((pred_ids==0).sum().item())
        pred_list["chhadron"].append((pred_ids==1).sum().item())
        pred_list["nhadron"].append((pred_ids==2).sum().item())
        pred_list["photon"].append((pred_ids==3).sum().item())
        pred_list["electron"].append((pred_ids==4).sum().item())
        pred_list["muon"].append((pred_ids==5).sum().item())

        cand_list["null"].append((cand_ids==0).sum().item())
        cand_list["chhadron"].append((cand_ids==1).sum().item())
        cand_list["nhadron"].append((cand_ids==2).sum().item())
        cand_list["photon"].append((cand_ids==3).sum().item())
        cand_list["electron"].append((cand_ids==4).sum().item())
        cand_list["muon"].append((cand_ids==5).sum().item())

        # to evaluate
        gen_ids_all.append(gen_ids_one_hot.detach().cpu())
        gen_p4_all.append(gen_p4.detach().cpu())

        pred_ids_all.append(pred_ids_one_hot.detach().cpu())
        pred_p4_all.append(pred_p4.detach().cpu())

        cand_ids_all.append(cand_ids_one_hot.detach().cpu())
        cand_p4_all.append(cand_p4.detach().cpu())

    # store the 3 list dictionaries in a list (this is done only to compute the particle multiplicity plots)
    list = [pred_list, gen_list, cand_list]
    with open(path + '/list_for_multiplicities.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pkl.dump(list, f)

    # for all the other plots
    gen_ids = gen_ids_all[0]
    gen_p4 = gen_p4_all[0]

    pred_ids = pred_ids_all[0]
    pred_p4 = pred_p4_all[0]

    cand_ids = cand_ids_all[0]
    cand_p4 = cand_p4_all[0]

    for i in range(len(pred_ids_all)-1):
        gen_ids = torch.cat((gen_ids,gen_ids_all[i+1]))
        gen_p4 = torch.cat((gen_p4,gen_p4_all[i+1]))

        pred_ids = torch.cat((pred_ids,pred_ids_all[i+1]))
        pred_p4 = torch.cat((pred_p4,pred_p4_all[i+1]))

        cand_ids = torch.cat((cand_ids,cand_ids_all[i+1]))
        cand_p4 = torch.cat((cand_p4,cand_p4_all[i+1]))

    ygen = torch.cat([gen_ids.argmax(axis=1).reshape(-1,1),gen_p4], axis=1)
    ypred = torch.cat([pred_ids.argmax(axis=1).reshape(-1,1),pred_p4], axis=1)
    ycand = torch.cat([cand_ids.argmax(axis=1).reshape(-1,1),cand_p4], axis=1)

    # store the actual predictions to make all the other plots
    predictions = {"ygen":ygen.reshape(1,-1,7).numpy(), "ycand":ycand.reshape(1,-1,7).numpy(), "ypred":ypred.reshape(1,-1,7).numpy()}

    with open(path + '/predictions.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pkl.dump(predictions, f)

    ygen = predictions["ygen"].reshape(-1,7)
    ypred = predictions["ypred"].reshape(-1,7)
    ycand = predictions["ycand"].reshape(-1,7)

    print('Making plots on ' + which_data)

    # make confusion matrix
    conf_matrix_norm = sklearn.metrics.confusion_matrix(torch.max(gen_ids, -1)[1],
                                    np.argmax(pred_ids.detach().cpu().numpy(),axis=1), labels=range(6), normalize="true")

    plot_confusion_matrix(conf_matrix_norm, ["none", "ch.had", "n.had", "g", "el", "mu"], fname = outpath + 'conf_matrix_norm_test' + str(epoch), epoch=epoch)

    with open(outpath + '/conf_matrix_norm_test' + str(epoch) + '.pkl', 'wb') as f:
        pkl.dump(conf_matrix_norm, f)

    # make distribution plots
    plot_all_distributions(gen_ids, gen_p4, pred_ids, pred_p4, cand_ids, cand_p4,
                target, epoch, outpath=path + '/')

    # make pt, eta plots to visualize dataset
    ax, _ = plot_pt_eta(ygen)
    plt.savefig(outpath+"gen_pt_eta.png", bbox_inches="tight")

    # plot particle multiplicity plots
    fig, ax = plt.subplots(1, 1, figsize=(8, 2*8))
    ret_num_particles_null = plot_num_particles_pid(list, "null", ax)
    plt.savefig(outpath+"/multiplicity_plots/num_null.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2*8))
    ret_num_particles_chhad = plot_num_particles_pid(list, "chhadron", ax)
    plt.savefig(outpath+"multiplicity_plots/num_chhadron.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2*8))
    ret_num_particles_nhad = plot_num_particles_pid(list, "nhadron", ax)
    plt.savefig(outpath+"multiplicity_plots/num_nhadron.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2*8))
    ret_num_particles_photon = plot_num_particles_pid(list, "photon", ax)
    plt.savefig(outpath+"multiplicity_plots/num_photon.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2*8))
    ret_num_particles_electron = plot_num_particles_pid(list, "electron", ax)
    plt.savefig(outpath+"multiplicity_plots/num_electron.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2*8))
    ret_num_particles_muon = plot_num_particles_pid(list, "muon", ax)
    plt.savefig(outpath+"multiplicity_plots/num_muon.png", bbox_inches="tight")
    plt.close(fig)

    # make efficiency and fake rate plots for charged hadrons
    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 1, "pt", np.linspace(0, 3, 61), both=True, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"efficiency_plots/eff_fake_pid1_pt.png", bbox_inches="tight")
    plt.close(fig)

    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 1, "eta", np.linspace(-3, 3, 61), both=True, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"efficiency_plots/eff_fake_pid1_eta.png", bbox_inches="tight")
    plt.close(fig)

    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 1, "energy", np.linspace(0, 50, 75), both=True, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"efficiency_plots/eff_fake_pid1_energy.png", bbox_inches="tight")
    plt.close(fig)

    # make efficiency and fake rate plots for neutral hadrons
    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 2, "pt", np.linspace(0, 3, 61), both=True, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"efficiency_plots/eff_fake_pid2_pt.png", bbox_inches="tight")
    plt.close(fig)

    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 2, "eta", np.linspace(-3, 3, 61), both=True, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"efficiency_plots/eff_fake_pid2_eta.png", bbox_inches="tight")
    plt.close(fig)

    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 2, "energy", np.linspace(0, 50, 75), both=True, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"efficiency_plots/eff_fake_pid2_energy.png", bbox_inches="tight")
    plt.close(fig)

    # make resolution plots for chhadrons: pid=1
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    res_chhad_pt = plot_reso(ygen, ypred, ycand, 1, "pt", 2, ax=ax1, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"resolution_plots/res_pid1_pt.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_chhad_eta = plot_reso(ygen, ypred, ycand, 1, "eta", 0.2, ax=ax2, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"resolution_plots/res_pid1_eta.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_chhad_E = plot_reso(ygen, ypred, ycand, 1, "energy", 0.2, ax=ax3, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"resolution_plots/res_pid1_energy.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for nhadrons: pid=2
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    res_nhad_pt = plot_reso(ygen, ypred, ycand, 2, "pt", 2, ax=ax1, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"resolution_plots/res_pid2_pt.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_nhad_eta = plot_reso(ygen, ypred, ycand, 2, "eta", 0.2, ax=ax2, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"resolution_plots/res_pid2_eta.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_nhad_E = plot_reso(ygen, ypred, ycand, 2, "energy", 0.2, ax=ax3, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"resolution_plots/res_pid2_energy.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for photons: pid=3
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    res_photon_pt = plot_reso(ygen, ypred, ycand, 3, "pt", 2, ax=ax1, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"resolution_plots/res_pid3_pt.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_photon_eta = plot_reso(ygen, ypred, ycand, 3, "eta", 0.2, ax=ax2, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"resolution_plots/res_pid3_eta.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_photon_E = plot_reso(ygen, ypred, ycand, 3, "energy", 0.2, ax=ax3, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"resolution_plots/res_pid3_energy.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for electrons: pid=4
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    res_electron_pt = plot_reso(ygen, ypred, ycand, 4, "pt", 2, ax=ax1, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"resolution_plots/res_pid4_pt.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_electron_eta = plot_reso(ygen, ypred, ycand, 4, "eta", 0.2, ax=ax2, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"resolution_plots/res_pid4_eta.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_electron_E = plot_reso(ygen, ypred, ycand, 4, "energy", 0.2, ax=ax3, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"resolution_plots/res_pid4_energy.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for muons: pid=5
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    res_muon_pt = plot_reso(ygen, ypred, ycand, 5, "pt", 2, ax=ax1, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"resolution_plots/res_pid5_pt.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_muon_eta = plot_reso(ygen, ypred, ycand, 5, "eta", 0.2, ax=ax2, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"resolution_plots/res_pid5_eta.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_muon_E = plot_reso(ygen, ypred, ycand, 5, "energy", 0.2, ax=ax3, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"resolution_plots/res_pid5_energy.png", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)
