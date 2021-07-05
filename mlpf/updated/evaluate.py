import args
from args import parse_args

#import setGPU
import torch
import torch_geometric
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
import pandas, mplhep, pickle

import time
import math

import sys
sys.path.insert(1, '../../plotting/')
sys.path.insert(1, '../../mlpf/plotting/')

use_gpu = torch.cuda.device_count()>0
multi_gpu = torch.cuda.device_count()>1

#define the global base device
if use_gpu:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

import sklearn
import sklearn.metrics

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas
import mplhep
import math

import sys
import os.path as osp

from plot_utils import cms_label, particle_label, sample_label
from plot_utils import plot_E_reso, plot_eta_reso, plot_phi_reso, bins
from plot_utils import plot_confusion_matrix
import torch
import seaborn as sns

elem_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
class_labels = [0, 1, 2, 3, 4, 5]

#map these to ids 0...Nclass
class_to_id = {r: class_labels[r] for r in range(len(class_labels))}
# map these to ids 0...Nclass
elem_to_id = {r: elem_labels[r] for r in range(len(elem_labels))}

def deltaphi(phi1, phi2):
    return np.fmod(phi1 - phi2 + np.pi, 2*np.pi) - np.pi

def mse_unreduced(true, pred):
    return torch.square(true-pred)

# computes accuracy of PID predictions given a one_hot_embedding: truth & pred
def accuracy(true_id, pred_id):
    # revert one_hot_embedding
    _, true_id = torch.max(true_id, -1)
    _, pred_id = torch.max(pred_id, -1)

    is_true = (true_id !=0)
    is_same = (true_id == pred_id)

    acc = (is_same&is_true).sum() / is_true.sum()
    return acc

# computes the resolution given a one_hot_embedding truth & pred + p4 of truth & pred
def energy_resolution(true_id, true_p4, pred_id, pred_p4):
    # revert one_hot_embedding
    _,true_id= torch.max(true_id, -1)
    _,pred_id = torch.max(pred_id, -1)

    msk = (true_id!=0)

    return mse_unreduced(true_p4[msk], pred_p4[msk])

def plot_regression(val_x, val_y, var_name, rng, target, fname):
    fig = plt.figure(figsize=(5,5))
    plt.hist2d(
        val_x,
        val_y,
        bins=(rng, rng),
        cmap="Blues",
        #norm=matplotlib.colors.LogNorm()
    );

    if target=='cand':
        plt.xlabel("Cand {}".format(var_name))
    elif target=='gen':
        plt.xlabel("Gen {}".format(var_name))

    plt.ylabel("MLPF {}".format(var_name))

    plt.savefig(fname + '.png')
    plt.close(fig)

    return fig

def plot_distributions(val_x, val_y, var_name, rng, target, fname):
    fig = plt.figure(figsize=(5,5))

    if target=='cand':
        plt.hist(val_x, bins=rng, density=True, histtype="step", lw=2, label="cand");
    elif target=='gen':
        plt.hist(val_x, bins=rng, density=True, histtype="step", lw=2, label="gen");

    plt.hist(val_y, bins=rng, density=True, histtype="step", lw=2, label="MLPF");
    plt.xlabel(var_name)
    plt.legend(loc="best", frameon=False)
    plt.ylim(0,1.5)

    plt.savefig(fname + '.png')
    plt.close(fig)

    return fig

# def plot_resolutions(pred_p4, target_p4, i, rng, target, fname):
#
#     msk = torch.isinf(((pred_p4-target_p4)/target_p4)[:,i])==0
#     val_x = ((pred_p4[msk]-target_p4[msk])/target_p4[msk])[:,i]
#
#     fig = plt.figure(figsize=(5,5))
#
#     plt.hist(val_x, bins=rng, density=True, histtype="step", lw=2);
#
#     plt.savefig(fname + '.png')
#     plt.close(fig)
#
#     return fig

def plot_particles(fname, true_id, true_p4, pred_id, pred_p4, pid=1):
    #Ground truth vs model prediction particles
    fig = plt.figure(figsize=(10,10))

    true_p4 = true_p4.detach().numpy()
    pred_p4 = pred_p4.detach().numpy()

    msk = (true_id == pid)
    plt.scatter(true_p4[msk, 2], np.arctan2(true_p4[msk, 3], true_p4[msk, 4]), s=2*true_p4[msk, 2], marker="o", alpha=0.5)

    msk = (pred_id == pid)
    plt.scatter(pred_p4[msk, 2], np.arctan2(pred_p4[msk, 3], pred_p4[msk, 4]), s=2*pred_p4[msk, 2], marker="o", alpha=0.5)

    plt.xlabel("eta")
    plt.ylabel("phi")
    plt.xlim(-5,5)
    plt.ylim(-4,4)

    plt.savefig(fname + '.png')
    plt.close(fig)

    return fig

def make_plots(true_id, true_p4, pred_id, pred_p4, pf_id, pf_p4,
                cand_ids_list_chhadron, cand_ids_list_nhadron,
                target_ids_list_chhadron, target_ids_list_nhadron,
                pf_ids_list_chhadron, pf_ids_list_nhadron,target, epoch, outpath):

    conf_matrix_norm = sklearn.metrics.confusion_matrix(torch.max(true_id, -1)[1],
                                    np.argmax(pred_id.detach().cpu().numpy(),axis=1), labels=range(6), normalize="true")

    plot_confusion_matrix(conf_matrix_norm, ["none", "ch.had", "n.had", "g", "el", "mu"], fname = outpath + 'conf_matrix_norm_test' + str(epoch), epoch=epoch)

    with open(outpath + '/conf_matrix_norm_test' + str(epoch) + '.pkl', 'wb') as f:
        pickle.dump(conf_matrix_norm, f)

    _, true_id = torch.max(true_id, -1)
    _, pred_id = torch.max(pred_id, -1)
    _, pf_id = torch.max(pf_id, -1)

    msk = (pred_id!=0) & (true_id!=0)

    ch_true = true_p4[msk, 0].flatten().detach().numpy()
    ch_pred = pred_p4[msk, 0].flatten().detach().numpy()

    pt_true = true_p4[msk, 1].flatten().detach().numpy()
    pt_pred = pred_p4[msk, 1].flatten().detach().numpy()

    e_true = true_p4[msk, 5].flatten().detach().numpy()
    e_pred = pred_p4[msk, 5].flatten().detach().numpy()

    eta_true = true_p4[msk, 2].flatten().detach().numpy()
    eta_pred = pred_p4[msk, 2].flatten().detach().numpy()

    sphi_true = true_p4[msk, 3].flatten().detach().numpy()
    sphi_pred = pred_p4[msk, 3].flatten().detach().numpy()

    cphi_true = true_p4[msk, 4].flatten().detach().numpy()
    cphi_pred = pred_p4[msk, 4].flatten().detach().numpy()

    # figure = plot_regression(ch_true, ch_pred, "charge", np.linspace(-2, 2, 100), target, fname = outpath+'charge_regression')

    figure = plot_distributions(ch_true, ch_pred, "charge", np.linspace(0, 5, 100), target, fname = outpath+'charge_distribution')

    # figure = plot_regression(pt_true, pt_pred, "pt", np.linspace(0, 5, 100), target, fname = outpath+'pt_regression')

    figure = plot_distributions(pt_true, pt_pred, "pt", np.linspace(0, 5, 100), target, fname = outpath+'pt_distribution')

    # figure = plot_regression(e_true, e_pred, "E", np.linspace(-1, 5, 100), target, fname = outpath+'energy_regression')

    figure = plot_distributions(e_true, e_pred, "E", np.linspace(-1, 5, 100), target, fname = outpath+'energy_distribution')

    # figure = plot_regression(eta_true, eta_pred, "eta", np.linspace(-5, 5, 100), target, fname = outpath+'eta_regression')

    figure = plot_distributions(eta_true, eta_pred, "eta", np.linspace(-5, 5, 100), target, fname = outpath+'eta_distribution')

    # figure = plot_regression(sphi_true, sphi_pred, "sin phi", np.linspace(-2, 2, 100), target, fname = outpath+'sphi_regression')

    figure = plot_distributions(sphi_true, sphi_pred, "sin phi", np.linspace(-2, 2, 100), target, fname = outpath+'sphi_distribution')

    # figure = plot_regression(cphi_true, cphi_pred, "cos phi", np.linspace(-2, 2, 100), target, fname = outpath+'cphi_regression')

    figure = plot_distributions(cphi_true, cphi_pred, "cos phi", np.linspace(-2, 2, 100), target, fname = outpath+'cphi_distribution')

    figure = plot_particles( outpath+'particleID1', true_id, true_p4, pred_id, pred_p4, pid=1)

    figure = plot_particles( outpath+'particleID2', true_id, true_p4, pred_id, pred_p4, pid=2)

    # figure = plot_resolutions(pred_p4, true_p4, 0, np.linspace(-5, 5, 200), target, fname=outpath+'charge_resolution')
    # figure = plot_resolutions(pred_p4, true_p4, 1, np.linspace(-5, 5, 200), target, fname=outpath+'pt_resolution')
    # figure = plot_resolutions(pred_p4, true_p4, 2, np.linspace(-5, 5, 200), target, fname=outpath+'eta_resolution')
    # figure = plot_resolutions(pred_p4, true_p4, 3, np.linspace(-5, 5, 200), target, fname=outpath+'sphi_resolution')
    # figure = plot_resolutions(pred_p4, true_p4, 4, np.linspace(-5, 5, 200), target, fname=outpath+'cphi_resolution')
    # figure = plot_resolutions(pred_p4, true_p4, 5, np.linspace(-5, 5, 200), target, fname=outpath+'E_resolution')

    # plot num_gen vs num_pred for chhadron
    fig, ax = plt.subplots()
    ax.scatter(cand_ids_list_chhadron, target_ids_list_chhadron, label="MLPF")
    ax.scatter(pf_ids_list_chhadron, target_ids_list_chhadron, label="Rule-based PF")

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against each other
    ax.plot(lims, lims, '--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.title('Charged hadrons')
    plt.xlabel('True')
    plt.ylabel('Reconstructed')
    plt.legend(loc="best", frameon=False)
    plt.savefig(outpath + 'num_chhadron.png')
    plt.close(fig)

    # plot num_gen vs num_pred for nhadron
    fig, ax = plt.subplots()
    ax.scatter(cand_ids_list_nhadron, target_ids_list_nhadron, label="MLPF")
    ax.scatter(pf_ids_list_nhadron, target_ids_list_nhadron, label="Rule-based PF")

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, '--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.title('Neutral hadrons')
    plt.xlabel('True')
    plt.ylabel('Reconstructed')
    plt.legend(loc="best", frameon=False)
    plt.savefig(outpath + 'num_nhadron.png')
    plt.close(fig)


import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
import mplhep as hep

plt.style.use(hep.style.ROOT)

sample_title_qcd = "QCD, 14 TeV, PU200"
sample_title_ttbar = "$t\\bar{t}$, 14 TeV, PU200"

ranges = {
    "pt": np.linspace(0, 10, 61),
    "eta": np.linspace(-5, 5, 61),
    "sphi": np.linspace(-1, 1, 61),
    "cphi": np.linspace(-1, 1, 61),
    "energy": np.linspace(0, 100, 61)
}
pid_names = {
    1: "Charged hadrons",
    2: "Neutral hadrons",
    3: "Photons",
    4: "Electrons",
    5: "Muons",
}
var_names = {
    "pt": r"$p_\mathrm{T}$ [GeV]",
    "eta": r"$\eta$",
    "sphi": r"$\mathrm{sin} \phi$",
    "cphi": r"$\mathrm{cos} \phi$",
    "energy": r"$E$ [GeV]"
}
var_names_nounit = {
    "pt": r"$p_\mathrm{T}$",
    "eta": r"$\eta$",
    "sphi": r"$\mathrm{sin} \phi$",
    "cphi": r"$\mathrm{cos} \phi$",
    "energy": r"$E$"
}
var_names_bare = {
    "pt": "p_\mathrm{T}",
    "eta": "\eta",
    "energy": "E",
}
var_indices = {
    "pt": 2,
    "eta": 3,
    "sphi": 4,
    "cphi": 5,
    "energy": 6
}

def midpoints(x):
    return x[:-1] + np.diff(x)/2

def mask_empty(hist):
    h0 = hist[0].astype(np.float64)
    h0[h0<50] = 0
    return (h0, hist[1])

def divide_zero(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    out = np.zeros_like(a)
    np.divide(a, b, where=b>0, out=out)
    return out

def plot_pt_eta(ygen, legend_title=""):
    b = np.linspace(0, 100, 41)

    msk_pid1 = (ygen[:, 0]==1)
    msk_pid2 = (ygen[:, 0]==2)
    msk_pid3 = (ygen[:, 0]==3)
    msk_pid4 = (ygen[:, 0]==4)
    msk_pid5 = (ygen[:, 0]==5)

    h1 = np.histogram(ygen[msk_pid1, 2], bins=b)
    h2 = np.histogram(ygen[msk_pid2, 2], bins=b)
    h3 = np.histogram(ygen[msk_pid3, 2], bins=b)
    h4 = np.histogram(ygen[msk_pid4, 2], bins=b)
    h5 = np.histogram(ygen[msk_pid5, 2], bins=b)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 2*8))

    xs = midpoints(h1[1])
    width = np.diff(h1[1])

    hep.histplot([h5[0], h4[0], h3[0], h2[0], h1[0]], bins=h1[1], ax=ax1, stack=True, histtype="fill",
        label=["Muons", "Electrons", "Photons", "Neutral hadrons", "Charged hadrons"])

    ax1.legend(loc="best", frameon=False, title=legend_title)
    ax1.set_yscale("log")
    ax1.set_ylim(1e1, 1e9)
    ax1.set_xlabel(r"Truth particle $p_\mathrm{T}$ [GeV]")
    ax1.set_ylabel("Truth particles")

    b = np.linspace(-8, 8, 41)
    h1 = np.histogram(ygen[msk_pid1, 3], bins=b)
    h2 = np.histogram(ygen[msk_pid2, 3], bins=b)
    h3 = np.histogram(ygen[msk_pid3, 3], bins=b)
    h4 = np.histogram(ygen[msk_pid4, 3], bins=b)
    h5 = np.histogram(ygen[msk_pid5, 3], bins=b)
    xs = midpoints(h1[1])
    width = np.diff(h1[1])

    hep.histplot([h5[0], h4[0], h3[0], h2[0], h1[0]], bins=h1[1], ax=ax2, stack=True, histtype="fill",
        label=["Muons", "Electrons", "Photons", "Neutral hadrons", "Charged hadrons"])
    leg = ax2.legend(loc="best", frameon=False, ncol=2, title=legend_title)
    leg._legend_box.align = "left"
    ax2.set_yscale("log")
    ax2.set_ylim(1e1, 1e9)
    ax2.set_xlabel("Truth particle $\eta$")
    ax2.set_ylabel("Truth particles")
    return ax1, ax2

def plot_num_particles_pid(fi, pid=0, ax=None, legend_title=""):
    if not ax:
        plt.figure(figsize=(4,4))
        ax = plt.axes()

    #compute the number of particles per event
    if pid == 0:
        x1=fi["ygen"][:, :, 0]!=pid
        x1_msk=np.sum(x1, axis=1)
        x2=fi["ypred"][:, :, 0]!=pid
        x2_msk=np.sum(x2, axis=1)
        x3=fi["ycand"][:, :, 0]!=pid
        x3_msk=np.sum(x3, axis=1)
    else:
        x1=fi["ygen"][:, :, 0]==pid
        x1_msk=np.sum(x1, axis=1)
        x2=fi["ypred"][:, :, 0]==pid
        x2_msk=np.sum(x2, axis=1)
        x3=fi["ycand"][:, :, 0]==pid
        x3_msk=np.sum(x3, axis=1)

    v0 = np.min([np.min(x1_msk), np.min(x2_msk), np.min(x3_msk)])
    v1 = np.max([np.max(x1_msk), np.max(x2_msk), np.max(x3_msk)])

    #draw only a random sample of the events to avoid overcrowding
    inds = np.random.permutation(len(fi["ygen"][:, :, 0][0]))[:1000]

    ratio_dpf = (x3[inds] - x1[inds]) / x1[inds]
    ratio_dpf[ratio_dpf > 10] = 10
    ratio_dpf[ratio_dpf < -10] = -10
    mu_dpf = np.mean(ratio_dpf)
    sigma_dpf = np.std(ratio_dpf)

    ax.scatter(
        x1[inds],
        x3[inds],
        marker="o",
        label="Rule-based PF, $r={0:.3f}$\n$\mu={1:.3f}\\ \sigma={2:.3f}$".format(
            np.corrcoef(x1, x3)[0,1], mu_dpf, sigma_dpf
        ),
        alpha=0.5
    )

    ratio_mlpf = (x2[inds] - x1[inds]) / x1[inds]
    ratio_mlpf[ratio_mlpf > 10] = 10
    ratio_mlpf[ratio_mlpf < -10] = -10
    mu_mlpf = np.mean(ratio_mlpf)
    sigma_mlpf = np.std(ratio_mlpf)

    ax.scatter(
        x1[inds],
        x2[inds],
        marker="^",
        label="MLPF, $r={0:.3f}$\n$\mu={1:.3f}\\ \sigma={2:.3f}$".format(
            np.corrcoef(x1, x2)[0,1], mu_mlpf, sigma_mlpf
        ),
        alpha=0.5
    )
    leg = ax.legend(loc="best", frameon=False, title=legend_title+pid_names[pid] if pid>0 else "all particles")
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    ax.plot([v0, v1], [v0, v1], color="black", ls="--")
    #ax.set_title(pid_names[pid])
    ax.set_xlabel("Truth particles / event")
    ax.set_ylabel("Reconstructed particles / event")
    #plt.title("Particle multiplicity, {}".format(pid_names[pid]))
    #plt.savefig("plots/num_particles_pid{}.pdf".format(pid), bbox_inches="tight")
    return {"sigma_dpf": sigma_dpf, "sigma_mlpf": sigma_mlpf, "ratio_mlpf": ratio_mlpf, "ratio_dpf": ratio_dpf,
        "x1": x1, "x2": x2, "x3": x3}

def draw_efficiency_fakerate(ygen, ypred, ycand, pid, var, bins, both=True, legend_title=""):
    var_idx = var_indices[var]

    msk_gen = ygen[:, 0]==pid
    msk_pred = ypred[:, 0]==pid
    msk_cand = ycand[:, 0]==pid

    hist_gen = np.histogram(ygen[msk_gen, var_idx], bins=bins);
    hist_cand = np.histogram(ygen[msk_gen & msk_cand, var_idx], bins=bins);
    hist_pred = np.histogram(ygen[msk_gen & msk_pred, var_idx], bins=bins);

    hist_gen = mask_empty(hist_gen)
    hist_cand = mask_empty(hist_cand)
    hist_pred = mask_empty(hist_pred)

    #efficiency plot
    if both:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 2*8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 1*8))
        ax2 = None

    #ax1.set_title("reco efficiency for {}".format(pid_names[pid]))
    ax1.errorbar(
        midpoints(hist_gen[1]),
        divide_zero(hist_cand[0], hist_gen[0]),
        divide_zero(np.sqrt(hist_gen[0]), hist_gen[0]) * divide_zero(hist_cand[0], hist_gen[0]),
        lw=0, label="Rule-based PF", elinewidth=2, marker=".",markersize=10)
    ax1.errorbar(
        midpoints(hist_gen[1]),
        divide_zero(hist_pred[0], hist_gen[0]),
        divide_zero(np.sqrt(hist_gen[0]), hist_gen[0]) * divide_zero(hist_pred[0], hist_gen[0]),
        lw=0, label="MLPF", elinewidth=2, marker=".",markersize=10)
    ax1.legend(frameon=False, loc=0, title=legend_title+pid_names[pid])
    ax1.set_ylim(0,1.2)
    # if var=="energy":
    #     ax1.set_xlim(0,30)
    ax1.set_xlabel(var_names[var])
    ax1.set_ylabel("Efficiency")

    hist_cand2 = np.histogram(ygen[msk_cand & (ygen[:, 0]!=0), var_idx], bins=bins);
    hist_pred2 = np.histogram(ygen[msk_pred & (ygen[:, 0]!=0), var_idx], bins=bins);
    hist_cand_gen2 = np.histogram(ygen[msk_cand & ~msk_gen & (ygen[:, 0]!=0), var_idx], bins=bins);
    hist_pred_gen2 = np.histogram(ygen[msk_pred & ~msk_gen & (ygen[:, 0]!=0), var_idx], bins=bins);

    hist_cand2 = mask_empty(hist_cand2)
    hist_cand_gen2 = mask_empty(hist_cand_gen2)
    hist_pred2 = mask_empty(hist_pred2)
    hist_pred_gen2 = mask_empty(hist_pred_gen2)

    if both:
        #fake rate plot
        #ax2.set_title("reco fake rate for {}".format(pid_names[pid]))
        ax2.errorbar(
            midpoints(hist_cand2[1]),
            divide_zero(hist_cand_gen2[0], hist_cand2[0]),
            divide_zero(np.sqrt(hist_cand_gen2[0]), hist_cand2[0]),
            lw=0, label="Rule-based PF", elinewidth=2, marker=".",markersize=10)
        ax2.errorbar(
            midpoints(hist_pred2[1]),
            divide_zero(hist_pred_gen2[0], hist_pred2[0]),
            divide_zero(np.sqrt(hist_pred_gen2[0]), hist_pred2[0]),
            lw=0, label="MLPF", elinewidth=2, marker=".",markersize=10)
        ax2.legend(frameon=False, loc=0, title=legend_title+pid_names[pid])
        ax2.set_ylim(0, 1.0)
        #plt.yscale("log")
        ax2.set_xlabel(var_names[var])
        ax2.set_ylabel("Fake rate")
    return ax1, ax2

def get_eff(ygen, ypred, ycand):
    msk_gen = (ygen[:, 0]==pid) & (ygen[:, var_indices["pt"]]>5.0)
    msk_pred = ypred[:, 0]==pid
    msk_cand = ycand[:, 0]==pid

    hist_gen = np.histogram(ygen[msk_gen, var_idx], bins=bins);
    hist_cand = np.histogram(ygen[msk_gen & msk_cand, var_idx], bins=bins);
    hist_pred = np.histogram(ygen[msk_gen & msk_pred, var_idx], bins=bins);

    hist_gen = mask_empty(hist_gen)
    hist_cand = mask_empty(hist_cand)
    hist_pred = mask_empty(hist_pred)

    return {
        "x": midpoints(hist_gen[1]),
        "y": divide_zero(hist_pred[0], hist_gen[0]),
        "yerr": divide_zero(np.sqrt(hist_gen[0]), hist_gen[0]) * divide_zero(hist_pred[0], hist_gen[0])
    }

def get_fake(ygen, ypred, ycand):
    msk_gen = ygen[:, 0]==pid
    msk_pred = ypred[:, 0]==pid
    msk_cand = ycand[:, 0]==pid

    hist_cand2 = np.histogram(ygen[msk_cand & (ygen[:, 0]!=0), var_idx], bins=bins);
    hist_pred2 = np.histogram(ygen[msk_pred & (ygen[:, 0]!=0), var_idx], bins=bins);
    hist_cand_gen2 = np.histogram(ygen[msk_cand & ~msk_gen & (ygen[:, 0]!=0), var_idx], bins=bins);
    hist_pred_gen2 = np.histogram(ygen[msk_pred & ~msk_gen & (ygen[:, 0]!=0), var_idx], bins=bins);

    hist_cand2 = mask_empty(hist_cand2)
    hist_cand_gen2 = mask_empty(hist_cand_gen2)
    hist_pred2 = mask_empty(hist_pred2)
    hist_pred_gen2 = mask_empty(hist_pred_gen2)

    return {
        "x": midpoints(hist_pred2[1]),
        "y": divide_zero(hist_pred_gen2[0], hist_pred2[0]),
        "yerr": divide_zero(np.sqrt(hist_pred_gen2[0]), hist_pred2[0])
    }

def plot_reso(ygen, ypred, ycand, pid, var, rng, ax=None, legend_title=""):
    var_idx = var_indices[var]
    msk = (ygen[:, 0]==pid) & (ycand[:, 0]==pid)
    bins = np.linspace(-rng, rng, 100)
    yg = ygen[msk, var_idx]
    yp = ypred[msk, var_idx]

    yc = ycand[msk, var_idx]
    ratio_mlpf = (yp - yg) / yg
    ratio_dpf = (yc - yg) / yg

    #remove outliers for std value computation
    outlier = 10
    ratio_mlpf[ratio_mlpf<-outlier] = -outlier
    ratio_mlpf[ratio_mlpf>outlier] = outlier
    ratio_dpf[ratio_dpf<-outlier] = -outlier
    ratio_dpf[ratio_dpf>outlier] = outlier

    res_dpf = np.mean(ratio_dpf.numpy()), np.std(ratio_dpf.numpy())
    res_mlpf = np.mean(ratio_mlpf.numpy()), np.std(ratio_mlpf.numpy())

    if ax is None:
        plt.figure(figsize=(4, 4))
        ax = plt.axes()

    #plt.title("{} resolution for {}".format(var_names_nounit[var], pid_names[pid]))
    ax.hist(ratio_dpf, bins=bins, histtype="step", lw=2, label="Rule-based PF\n$\mu={:.2f},\\ \sigma={:.2f}$".format(*res_dpf));
    ax.hist(ratio_mlpf, bins=bins, histtype="step", lw=2, label="MLPF\n$\mu={:.2f},\\ \sigma={:.2f}$".format(*res_mlpf));
    ax.legend(frameon=False, title=legend_title+pid_names[pid])
    ax.set_xlabel("{nounit} resolution, $({bare}^\prime - {bare})/{bare}$".format(nounit=var_names_nounit[var],bare=var_names_bare[var]))
    ax.set_ylabel("Particles")
    #plt.ylim(0, ax.get_ylim()[1]*2)
    ax.set_ylim(1, 1e10)
    ax.set_yscale("log")

    return {"dpf": res_dpf, "mlpf": res_mlpf}


def make_other_plots(fi, ygen, ycand, ypred, outpath):

    # make pt, eta plots to visualize dataset
    ax, _ = plot_pt_eta(ygen)
    plt.savefig(outpath+"gen_pt_eta.png", bbox_inches="tight")

    # # plot number of particles
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 2*8))
    # ret_num_particles_ch_had = plot_num_particles_pid(fi, 1, ax1)
    # ret_num_particles_n_had = plot_num_particles_pid(fi, 2, ax2)
    # plt.tight_layout()
    # plt.savefig("num_particles.png", bbox_inches="tight")
    # plt.savefig("num_particles.png", bbox_inches="tight", dpi=200)

    # make efficiency plots for charged hadrons
    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 1, "pt", np.linspace(0, 3, 61), both=False, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"eff_fake_pid1_pt.png", bbox_inches="tight")

    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 1, "eta", np.linspace(-3, 3, 61), both=False, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"eff_fake_pid1_eta.png", bbox_inches="tight")

    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 1, "energy", np.linspace(-3, 3, 61), both=False, legend_title=sample_title_qcd+"\n")
    plt.savefig(outpath+"eff_fake_pid1_energy.png", bbox_inches="tight")

    # make resolution plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 2*8))

    res_ch_had_pt = plot_reso(ygen, ypred, ycand, 1, "pt", 2, ax=ax1, legend_title=sample_title_qcd+"\n")
    res_ch_had_eta = plot_reso(ygen, ypred, ycand, 1, "eta", 0.2, ax=ax2, legend_title=sample_title_qcd+"\n")

    ax1.set_ylim(100, 10**11)
    ax2.set_ylim(100, 10**11)
    plt.tight_layout()
    plt.savefig(outpath+"res_pid1.png", bbox_inches="tight")


def Evaluate(model, test_loader, path, target, device, epoch):
    pred_ids_all, pred_p4_all = [], []
    target_ids_all, target_p4_all = [], []
    pf_ids_all, pf_p4_all = [], []

    cand_ids_list_chhadron, target_ids_list_chhadron, pf_ids_list_chhadron = [], [], []
    cand_ids_list_nhadron, target_ids_list_nhadron, pf_ids_list_nhadron = [], [], []

    print('Making predictions..')

    for batch in test_loader:
        if multi_gpu:
            X = batch
        else:
            X = batch.to(device)

        cand_ids_one_hot, pred_p4, target_ids_one_hot, target_p4, pf_ids_one_hot, pf_p4 = model(X)

        # to make "num_gen vs num_pred" plots
        _, cand_ids = torch.max(cand_ids_one_hot, -1)
        _, target_ids = torch.max(target_ids_one_hot, -1)
        _, pf_ids = torch.max(pf_ids_one_hot, -1)

        cand_ids_list_chhadron.append((cand_ids==1).sum().item())
        cand_ids_list_nhadron.append((cand_ids==2).sum().item())

        target_ids_list_chhadron.append((target_ids==1).sum().item())
        target_ids_list_nhadron.append((target_ids==2).sum().item())

        pf_ids_list_chhadron.append((pf_ids==1).sum().item())
        pf_ids_list_nhadron.append((pf_ids==2).sum().item())

        # to evaluate
        pred_ids_all.append(cand_ids_one_hot.detach().cpu())
        pred_p4_all.append(pred_p4.detach().cpu())

        target_ids_all.append(target_ids_one_hot.detach().cpu())
        target_p4_all.append(target_p4.detach().cpu())

        pf_ids_all.append(pf_ids_one_hot.detach().cpu())
        pf_p4_all.append(pf_p4.detach().cpu())

    pred_ids = pred_ids_all[0]
    pred_p4 = pred_p4_all[0]
    target_ids = target_ids_all[0]
    target_p4 = target_p4_all[0]
    pf_ids = pf_ids_all[0]
    pf_p4 = pf_p4_all[0]

    for i in range(len(pred_ids_all)-1):
        pred_ids = torch.cat((pred_ids,pred_ids_all[i+1]))
        pred_p4 = torch.cat((pred_p4,pred_p4_all[i+1]))
        target_ids = torch.cat((target_ids,target_ids_all[i+1]))
        target_p4 = torch.cat((target_p4,target_p4_all[i+1]))
        pf_ids = torch.cat((pf_ids,pf_ids_all[i+1]))
        pf_p4 = torch.cat((pf_p4,pf_p4_all[i+1]))

    ypred=torch.cat([pred_ids.argmax(axis=1).reshape(-1,1),pred_p4], axis=1)
    ygen=torch.cat([target_ids.argmax(axis=1).reshape(-1,1),target_p4], axis=1)
    ycand=torch.cat([pf_ids.argmax(axis=1).reshape(-1,1),pf_p4], axis=1)

    fi = {"ygen":ygen.reshape(1,-1,7).numpy(), "ycand":ycand.reshape(1,-1,7).numpy(), "ypred":ypred.reshape(1,-1,7).numpy()}

    with open(path + '/fi.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(fi, f)

    print('Making plots..')

    make_other_plots(fi, ygen, ycand, ypred, outpath=path + '/')

    # make_plots(target_ids, target_p4, pred_ids, pred_p4, pf_ids, pf_p4,
    #             cand_ids_list_chhadron, cand_ids_list_nhadron,
    #             target_ids_list_chhadron, target_ids_list_nhadron,
    #             pf_ids_list_chhadron, pf_ids_list_nhadron,
    #             target, epoch, outpath=path + '/')
