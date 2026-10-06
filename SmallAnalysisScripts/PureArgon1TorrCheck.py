# -*- coding: utf-8 -*-
"""
PureArgon1TorrCheck.py

The pure-Ar spectrum of the N2 fraction sweep (0 %, 85 W, 1 Torr, Tg = 300 K) with the current CR model
(BSR-500 cross sections, Ar-atom 2p transfer, extended escape-factor table), for the two EEDF families
and both escape-factor treatments:
  microwave  BOLSIG+ 2.45 GHz from 3 Td (MicrowaveLowENFit 'lowEN:0')
  dc         BOLSIG+ DC (Ar_Biagi_bolsig)
  escape 'table' (Monte Carlo hemisphere, the default) or 'walsh' (Holstein-Walsh cylinder)
chi^2 profiled over E/N at every Ne (MicrowaveLowENFit.ne_path) with Te_eff and the CR 1s5 density along
the profile, against the critical density n_c and the absorption estimate of 1s5 (1-8e17 m^-3).

Output: Experimental_Data/Output/PureArgon1TorrCheck/ (pure_ar_1torr_profiles.png, profiles.csv,
summary.csv). Run from Spyder (F5) or python SmallAnalysisScripts/PureArgon1TorrCheck.py.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import MicrowaveLowENFit as mlf                     # noqa: E402

crf, he = mlf.crf, mlf.he
JOBS = {"microwave": "lowEN:0", "dc": "ref:Ar_Biagi_bolsig"}
MODES = ("table", "walsh")
STYLE = {("microwave", "table"): ("C3", "-"), ("microwave", "walsh"): ("C3", "--"),
         ("dc", "table"): ("C0", "-"), ("dc", "walsh"): ("C0", "--")}
OUTDIR = os.path.join(mlf.ROOT_DIR, "Experimental_Data", "Output", "PureArgon1TorrCheck")


def run_one(eedf, mode):
    cfg = dict(mlf.job_cfg(JOBS[eedf], build=False), escape_mode=mode)
    if mode != "table":
        cfg["outdir"] = cfg["outdir"] + f"_{mode}"
    fit = mlf.prepare(cfg, with_nu=False)
    d = fit["ft"].query("sweep == 'N2 fraction' and x == 0")
    s, post, _ = crf.fit_block(d, fit["M"], fit["feats"], fit["grid"], fit["Te_f"], cfg)
    path = mlf.ne_path(fit, post, s["chi2_min"], s["birge"])
    j = int(np.argmin(np.abs(np.log(path.Ne / mlf.N_C))))
    pin = path.iloc[j]
    best = path.loc[path.dchi2.idxmin()]
    row = dict(eedf=eedf, escape=mode, chi2_red_free=s["chi2_red"], Ne_free=best.Ne, Te_free=best.Te_eff,
               n_1s5_free=best.n_1s5, dchi2_at_nc=pin.dchi2, chi2_red_nc=(s["chi2_min"] + pin.dchi2 * s["birge"] ** 2)
               / (s["dof"] + 1), Te_nc=pin.Te_eff, EN_nc=pin.EN, n_1s5_nc=pin.n_1s5)
    return path.assign(eedf=eedf, escape=mode), row


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    paths, rows = [], []
    for eedf in JOBS:
        for mode in MODES:
            p, r = run_one(eedf, mode)
            paths.append(p)
            rows.append(r)
            print(f"  {eedf:9s} {mode:5s}: free Ne {r['Ne_free']:.2e}, Te_eff {r['Te_free']:.2f}, chi2/dof {r['chi2_red_free']:.2f}, "
                  f"1s5 {r['n_1s5_free']:.2e} | n_c: dchi2 {r['dchi2_at_nc']:.1f}, chi2/dof {r['chi2_red_nc']:.2f}, "
                  f"Te_eff {r['Te_nc']:.2f}, 1s5 {r['n_1s5_nc']:.2e}", flush=True)
    P = pd.concat(paths, ignore_index=True)
    S = pd.DataFrame(rows)
    P.to_csv(os.path.join(OUTDIR, "profiles.csv"), index=False)
    S.to_csv(os.path.join(OUTDIR, "summary.csv"), index=False)
    fig, axs = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    lo, hi = mlf.MEASURED_1S5
    for (eedf, mode), g in P.groupby(["eedf", "escape"]):
        c, ls = STYLE[eedf, mode]
        lab = f"{eedf} EEDF, escape {mode}"
        axs[0].plot(g.Ne, g.dchi2 + 0.1, color=c, ls=ls, lw=1.5, label=lab)
        axs[1].plot(g.Ne, g.n_1s5, color=c, ls=ls, lw=1.5, label=lab)
        axs[2].plot(g.Ne, g.Te_eff, color=c, ls=ls, lw=1.5, label=lab)
    axs[1].axhspan(lo, hi, color="C2", alpha=0.15, lw=0, label="1s$_5$ absorption estimate")
    for ax in axs:
        ax.axvline(mlf.N_C, color="k", ls=":", lw=1.2)
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    axs[0].axhline(1.1, color="0.6", ls="--", lw=0.8)
    axs[0].set_ylabel(r"$\Delta\chi^2/s^2$ (E/N profiled) + 0.1")
    axs[1].set_ylabel("CR 1s$_5$ density along the profile [m$^{-3}$]")
    axs[2].set_ylabel(r"$T_{e,\mathrm{eff}}$ along the profile [eV]")
    axs[2].set_xlabel("$N_e$ [m$^{-3}$]  (dotted: $n_c$)")
    axs[0].legend(fontsize=8)
    axs[1].legend(fontsize=8)
    fig.suptitle("Pure Ar, 1 Torr, 85 W (Tg 300 K): BSR + Ar-atom transfer + extended escape table", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "pure_ar_1torr_profiles.png"), dpi=140)
    plt.close(fig)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(S.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"output in {OUTDIR}")
    return S, P


if __name__ == "__main__":
    S, P = main()
