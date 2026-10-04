# -*- coding: utf-8 -*-
"""
AtomTransferComparison.py

The chi^2-vs-Ne graph of CriticalDensityFit without and with the population transfer between the
4p (2p) levels and 2p -> 1s by collisions with ground-state Ar (Zhu & Pu 2010,
InputData/Ar_2p_atom_transfer.csv; CRFitNeTe.CONFIG['atom_transfer']).

Needs the CriticalDensityFit outputs of both models: Experimental_Data/Output/CriticalDensity_noAT
(CONFIG['atom_transfer'] = False, after ActinometryNitrogenContent with the same setting) and
CriticalDensity (the default). Writes chi2_profiles_noAT_vs_AT.png and fit_changes.csv to
Experimental_Data/Output/AtomTransfer.  Run from Spyder: F5.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT_DIR, "Experimental_Data", "ExperimentalDataAnalysis"))
import CRFitNeTe as crf                             # noqa: E402

N_C = constants.epsilon_0 * constants.m_e * (2 * np.pi * 2.42e9) ** 2 / constants.e ** 2
MODELS = {False: "without Ar-atom transfer (before)", True: "with Ar-atom 2p transfer (now)"}
OUTDIR = os.path.join(ROOT_DIR, "Experimental_Data", "Output", "AtomTransfer")


def plot_profiles(prof, path):
    fig, axs = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=True)
    for i, eedf in enumerate(("microwave", "dc")):
        for j, at in enumerate(MODELS):
            ax = axs[i, j]
            p = prof[at][prof[at].eedf == eedf]
            n2 = sorted(p[p.sweep == "N2 fraction"].x.unique())
            cn = dict(zip(n2, plt.cm.viridis(np.linspace(0, 0.85, max(len(n2), 1)))))
            pw = sorted(p[p.sweep == "Power"].x.unique())
            cp = dict(zip(pw, plt.cm.autumn(np.linspace(0, 0.8, max(len(pw), 1)))))
            for (sweep, c), g in p.groupby(["sweep", "x"]):
                if sweep == "N2 fraction":
                    ax.semilogx(g.Ne, g.delta_chi2_s2, "-", color="k" if c == 0 else cn[c], lw=2.6 if c == 0 else 1.6,
                                label=f"{c:g} % N$_2$" + (" (pure Ar)" if c == 0 else ""))
                else:
                    ax.semilogx(g.Ne, g.delta_chi2_s2, "--", color=cp[c], lw=1.4, label=f"{c:g} W (2.4 % N$_2$)")
            for lev, ls in ((1, ":"), (4, "--")):
                ax.axhline(lev, color="0.4", lw=0.8, ls=ls)
            ax.axvline(N_C, color="k", lw=1.5)
            ax.axvspan(1e12, N_C, color="0.92", lw=0)
            ax.set_xlim(1e13, 3e19)
            ax.set_ylim(0, 30)
            ax.grid(alpha=0.3, which="both")
            ax.set_title(f"{eedf} EEDF, {MODELS[at]}")
            if i == 1:
                ax.set_xlabel("$N_e$ [m$^{-3}$]")
            if j == 0:
                ax.set_ylabel(r"$\Delta\chi^2 / s^2$ against the best fit")
    h, l = axs[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle(r"$\chi^2$ profiled over E/N vs $N_e$ (CriticalDensityFit, BSR 4s cross sections): without and "
                 r"with the Ar-atom 2p transfer; vertical line: critical density $n_c$ (2.42 GHz)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    dirs = {at: crf.output_dir("CriticalDensity", dict(crf.CONFIG, atom_transfer=at)) for at in MODELS}
    missing = [d for d in dirs.values() if not os.path.isfile(os.path.join(d, "chi2_profiles.csv"))]
    if missing:
        raise FileNotFoundError(f"run CriticalDensityFit for both settings first; missing: {missing}")
    prof = {at: pd.read_csv(os.path.join(d, "chi2_profiles.csv")) for at, d in dirs.items()}
    plot_profiles(prof, os.path.join(OUTDIR, "chi2_profiles_noAT_vs_AT.png"))
    cols = ["eedf", "sweep", "x", "Te_free", "Ne_free", "chi2_red_free", "delta_chi2_at_nc", "Te_pinned",
            "chi2_red_pinned", "n_1s5_free", "n_1s5_pinned"]
    fits = {at: pd.read_csv(os.path.join(d, "pinned_fits.csv"))[cols] for at, d in dirs.items()}
    m = fits[False].merge(fits[True], on=["eedf", "sweep", "x"], suffixes=("_noAT", "_AT"))
    m = m.sort_values(["eedf", "sweep", "x"])
    m.to_csv(os.path.join(OUTDIR, "fit_changes.csv"), index=False)
    show = ["eedf", "sweep", "x"] + [f"{c}_{s}" for c in ("Ne_free", "chi2_red_free", "delta_chi2_at_nc", "Te_pinned",
                                                          "n_1s5_pinned") for s in ("noAT", "AT")]
    with pd.option_context("display.width", 260, "display.max_columns", 30):
        print(m[show].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nfigure and table in {OUTDIR}")
    return m


if __name__ == "__main__":
    RES = main()
