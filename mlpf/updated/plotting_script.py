import numpy as np
import mplhep
import pickle as pkl

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

### Given a dictionary with 3 keys "ygen", "ycand" and "ypred"; make some plots
### Each of the 3 keys is a big tensor of shape (1, big_number, 7) where 7 denotes [pid, charge, pt, eta, sphi, cphi, energy]

# retrieve predictions:
with open('../../prp/models/yee/PFNet7_gen_ntrain_1_nepochs_16_batch_size_1_lr_0.001_alpha_0.0002_both__noskip_nn1_nn3/fi.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    fi = pkl.load(f)

ygen = fi["ygen"].reshape(-1,7)
ypred = fi["ypred"].reshape(-1,7)
ycand = fi["ycand"].reshape(-1,7)

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

    res_dpf = np.mean(ratio_dpf), np.std(ratio_dpf)
    res_mlpf = np.mean(ratio_mlpf), np.std(ratio_mlpf)

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


if __name__ == "__main__":

    # make pt, eta plots to visualize dataset
    ax, _ = plot_pt_eta(ygen)
    plt.savefig("gen_pt_eta.png", bbox_inches="tight")

    # # plot number of particles
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 2*8))
    # ret_num_particles_ch_had = plot_num_particles_pid(fi, 1, ax1)
    # ret_num_particles_n_had = plot_num_particles_pid(fi, 2, ax2)
    # plt.tight_layout()
    # plt.savefig("num_particles.png", bbox_inches="tight")
    # plt.savefig("num_particles.png", bbox_inches="tight", dpi=200)

    # make efficiency plots for charged hadrons
    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 1, "pt", np.linspace(0, 3, 61), both=False, legend_title=sample_title_qcd+"\n")
    plt.savefig("eff_fake_pid1_pt.png", bbox_inches="tight")

    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 1, "eta", np.linspace(-3, 3, 61), both=False, legend_title=sample_title_qcd+"\n")
    plt.savefig("eff_fake_pid1_eta.png", bbox_inches="tight")

    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 1, "energy", np.linspace(-3, 3, 61), both=False, legend_title=sample_title_qcd+"\n")
    plt.savefig("eff_fake_pid1_energy.png", bbox_inches="tight")

    # make resolution plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 2*8))

    res_ch_had_pt = plot_reso(ygen, ypred, ycand, 1, "pt", 2, ax=ax1, legend_title=sample_title_qcd+"\n")
    res_ch_had_eta = plot_reso(ygen, ypred, ycand, 1, "eta", 0.2, ax=ax2, legend_title=sample_title_qcd+"\n")

    ax1.set_ylim(100, 10**11)
    ax2.set_ylim(100, 10**11)
    plt.tight_layout()
    plt.savefig("res_pid1.png", bbox_inches="tight")
