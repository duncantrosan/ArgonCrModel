# -*- coding: utf-8 -*-
"""
MicrowaveN2AFit.py

Do superelastic collisions with metastable N2(A3Sigma_u+) rescue the microwave EEDF in Ar/N2?

In a 2.45 GHz field at 1 Torr the electrons below ~5 eV are barely heated (nu_m << omega), and
with N2 the vibrational (2-4 eV) and electronic (6-11 eV) losses drain them: at n_c the
microwave fit needs ~490 Td and overproduces 1s5 ~100x (MicrowaveLowENFit).  N2(A) -> N2
superelastic collisions hand electrons 6.7 eV back, right where the field cannot.

N2(A) estimate (estimate_n2a, from the DC fits at n_c, which match the measured 1s5):
  production  Ar(4s) + N2 -> Ar + N2(C) -> N2(B) -> N2(A) (CR 4s densities x kQ of
              he.ImportArQuenchingData) and electron impact on N2 above 6 eV (BOLSIG+ rates;
              the singlets, which do not cascade to N2(A), are counted too)
  losses      N2(A) + N2(A) pooling K_POOL, electron collisions K_E_OUT Ne, the wall NU_WALL
              (quenching by N atoms left out, so an upper estimate)
  -> n_A ~ 1e18-1e19 m^-3, i.e. n_A / n_N2 ~ 2e-3 to 1e-2 at 2.4 % N2.
Scanned: n_A / n_N2 = F_A, with F_A = 0 the same BOLSIG+ setup without N2(A) (baseline).

Per N2 fraction a BOLSIG+ microwave library (MicrowaveLowENFit variant 'A<fA>': the
N2(A3Sigma, v = 0-4) excitation written reversible, g ratio 3, N2(A) as a species at fA x;
no N2(v) superelastics) and a CR table around n_c (current CR model, incl. 4p quenching).
Per condition, at Ne = n_c along E/N: chi^2/dof, Te_eff and CR 1s5, and the best chi^2/dof
with the CR 1s5 inside the absorption range (1-8e17 m^-3).

Output in Experimental_Data/Output/MicrowaveN2A/:
  n2a_summary.png    best chi^2/dof at n_c (1s5 in range) per condition, E/N and Te_eff there,
                     against DC and the pure-Ar microwave fit
  n2a_nc_rows.png    chi^2/dof, Te_eff and CR 1s5 along E/N at n_c for a few conditions
  n2a_summary.csv, n2a_nc_rows.csv, n2a_estimate.csv
Run from Spyder (F5) or python SmallAnalysisScripts/MicrowaveN2AFit.py (~30 min the first time:
4 x 7 BOLSIG+ libraries and CR tables, MicrowaveLowENFit.N_WORKERS at a time).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import MicrowaveLowENFit as mlf                     # noqa: E402

crf, anc, he = mlf.crf, mlf.anc, mlf.he

# ---- settings -------------------------------------------------------------------
F_A = [0, 1e-3, 3e-3, 1e-2]                         # n_N2(A) / n_N2; 0 = baseline without N2(A)
FRACTIONS = [0.5, 1, 1.5, 2, 2.4, 2.5, 3]           # % N2 of the conditions (power sweep at 2.4 %)
PICKS = [("N2 fraction", 1.0), ("N2 fraction", 3.0), ("Power", 85.0)]
COLORS = {0: "k", 1e-3: "C1", 3e-3: "C3", 1e-2: "C4"}
K_POOL = 2.6e-16                                    # m^3/s, N2(A) + N2(A) -> N2(C, B) (Piper 1988)
K_E_OUT = 1e-14                                     # m^3/s, electron collisions out of N2(A) (order of magnitude)
NU_WALL = 20.0                                      # s^-1, diffusion to the R = 4 cm wall, gamma ~ 1
LOWEN_SUMMARY = os.path.join(mlf.OUTDIR, "summary.csv")  # DC and pure-Ar reference (MicrowaveLowENFit)
LOWEN_NC_ROWS = os.path.join(mlf.OUTDIR, "nc_rows.csv")
OUTDIR = os.path.join(mlf.ROOT_DIR, "Experimental_Data", "Output", "MicrowaveN2A")


def key(fA, pct):
    return f"A{fA:g}:{pct:g}"


# ---- N2(A) estimate -------------------------------------------------------------
def estimate_n2a(ref):
    """n_A from production = losses at each fraction's DC fit at n_c (densities of the saved DC
    tables of ActinometryNitrogenContent, read directly)."""
    rows = []
    N = he.Torr2Volume(anc.CONFIG["fit"]["P_Torr"], anc.CONFIG["fit"]["Tg"])
    Q = he.ImportArQuenchingData("N2", verbose=False)
    for pct in FRACTIONS:
        r = ref[(ref.family == "dc") & (ref.N2_percent == pct)]
        if r.empty:
            continue
        EN = float(r.EN_nc.median())
        cfg = dict(anc.CONFIG["fit"], eedf=str(anc.mixture_library(pct, "dc")), N2_percent=pct,
                   outdir=os.path.join(anc.CONFIG["outdir"], f"fit_dc_{pct:g}pct_N2"))
        tab = dict(np.load(crf._table_path(cfg)))
        i = int(np.argmin(np.abs(np.log(tab["x_grid"] / EN))))
        j = int(np.argmin(np.abs(np.log(tab["Ne_grid"] / mlf.N_C))))
        lev = list(tab["levels"])
        n_N2 = pct / 100 * N
        P_ar = sum(Q[l]["kQ"] * tab["dens"][i, j, lev.index(l)] * n_N2 for l in Q)
        lib = he.ImportBolsigLibrary(cfg["eedf"], verbose=False)
        e = lib["EEDFs"][int(np.argmin(np.abs(np.log(lib["EN_Td"] / EN))))]
        k_trip = sum(q["rate_m3s"] for q in e["rates"] if q["species"] == "N2" and q["process"] == "Excitation"
                     and 6 < (q["threshold_eV"] or 0) < 15.5)    # BOLSIG+ rates carry no state names
        P = P_ar + mlf.N_C * n_N2 * k_trip
        nu = NU_WALL + mlf.N_C * K_E_OUT
        nA = (-nu + np.sqrt(nu ** 2 + 4 * K_POOL * P)) / (2 * K_POOL)
        rows.append(dict(N2_percent=pct, EN_dc_nc=EN, P_Ar4s=P_ar, P_e=P - P_ar, n_A=nA, n_A_over_N2=nA / n_N2))
    return pd.DataFrame(rows)


# ---- fits -----------------------------------------------------------------------
def fit_all():
    rows, ncs = [], []
    lo, hi = mlf.MEASURED_1S5
    for pct in FRACTIONS:
        for fA in F_A:
            cfg = mlf.job_cfg(key(fA, pct), build=False)
            fit = mlf.prepare(cfg, with_nu=False)
            for sw, x in mlf.conditions_of(fit["ft"], pct):
                d = fit["ft"][(fit["ft"].sweep == sw) & (fit["ft"].x == x)]
                s, post, _ = crf.fit_block(d, fit["M"], fit["feats"], fit["grid"], fit["Te_f"], cfg)
                nc = mlf.nc_row(fit, post, s)
                ib = int(nc.chi2_red.idxmin())
                row = dict(fA=fA, sweep=sw, x=x, N2_percent=pct, chi2_nc=nc.chi2_red[ib], EN_nc=nc.EN[ib],
                           Te_nc=nc.Te_eff[ib], n_1s5_nc=nc.n_1s5[ib])
                ok = (nc.n_1s5 >= lo) & (nc.n_1s5 <= hi)
                if ok.any():
                    k = int(nc.chi2_red.where(ok).idxmin())
                    row.update(chi2_1s5=nc.chi2_red[k], EN_1s5=nc.EN[k], Te_1s5=nc.Te_eff[k], n_1s5_1s5=nc.n_1s5[k])
                rows.append(row)
                ncs.append(nc.assign(fA=fA, sweep=sw, x=x))
            print(f"  {pct:g} % N2, n_A/n_N2 = {fA:g}: done")
    return pd.DataFrame(rows), pd.concat(ncs, ignore_index=True)


# ---- plots ----------------------------------------------------------------------
def label(fA):
    return "no N$_2$(A) (baseline)" if fA == 0 else f"N$_2$(A)/N$_2$ = {fA:g}"


def plot_summary(res, ref, path):
    sweeps = crf.CONFIG["sweeps"]
    keys = [("chi2_1s5", r"best $\chi^2$/dof at $n_c$ with 1s$_5$ in 1-8e17", "linear"),
            ("EN_1s5", "E/N there [Td]", "log"),
            ("Te_1s5", r"$T_{e,\mathrm{eff}}$ there [eV]", "linear")]
    fig, axs = plt.subplots(len(keys), len(sweeps), figsize=(7 * len(sweeps), 3.6 * len(keys)), squeeze=False)
    ar = ref[(ref.family == "mw_lowEN") & (ref.N2_percent == 0)]
    for j, sw in enumerate(sweeps):
        r = res[res.sweep == sw["name"]]
        dc = ref[(ref.family == "dc") & (ref.sweep == sw["name"]) & (ref.N2_percent > 0)].sort_values("x")
        for fA in F_A:
            g = r[r.fA == fA].sort_values("x")
            for i, (k, _, _) in enumerate(keys):
                axs[i, j].plot(g.x, g[k], "o-", color=COLORS[fA], lw=1.4, ms=5, label=label(fA))
        for i, k in enumerate(("chi2_red_nc", "EN_nc", "Te_nc")):
            axs[i, j].plot(dc.x, dc[k], "s-.", color="C0", mfc="white", ms=5, lw=1.2, label="DC EEDF (reference)")
        if len(ar):
            for i, k in enumerate(("chi2_red_nc", "EN_nc", "Te_nc")):
                axs[i, j].axhline(ar[k].iloc[0], color="0.5", ls=":", lw=1.2,
                                  label="pure Ar, microwave (n$_c$)" if i == 0 else None)
        for i, (k, ylab, scale) in enumerate(keys):
            axs[i, j].set_yscale(scale)
            axs[i, j].set_ylabel(ylab, fontsize=9)
            axs[i, j].set_xlabel(sw["xlabel"])
            axs[i, j].grid(alpha=0.3, which="both")
        axs[0, j].axhline(1, color="0.7", lw=0.8)
        axs[0, j].set_ylim(0, None)
        axs[0, j].set_title(f"{sw['name']} sweep ({sw['note']}), microwave EEDF + N$_2$(A) superelastics")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_nc_rows(ncs, ref_nc, path):
    fig, axs = plt.subplots(3, len(PICKS), figsize=(5.8 * len(PICKS), 11), sharex=True, squeeze=False)
    lo, hi = mlf.MEASURED_1S5
    for j, (sw, x) in enumerate(PICKS):
        for fA in F_A:
            h = ncs[(ncs.fA == fA) & (ncs.sweep == sw) & (ncs.x == x)].sort_values("EN")
            ok = ((h.n_1s5 >= lo) & (h.n_1s5 <= hi)).to_numpy()
            for i, k in enumerate(("chi2_red", "Te_eff", "n_1s5")):
                axs[i, j].plot(h.EN, h[k], "-", color=COLORS[fA], lw=1.3, label=label(fA))
                axs[i, j].plot(h.EN.where(ok), h[k].where(ok), "-", color=COLORS[fA], lw=4, alpha=0.5)
        dc = ref_nc[(ref_nc.family == "dc") & (ref_nc.sweep == sw) & (ref_nc.x == x)].sort_values("EN")
        for i, k in enumerate(("chi2_red", "Te_eff", "n_1s5")):
            axs[i, j].plot(dc.EN, dc[k], "-.", color="C0", lw=1.2, label="DC EEDF (reference)")
        axs[0, j].axhline(1, color="0.5", ls=":", lw=0.8)
        axs[0, j].set_ylim(0, 8)
        axs[2, j].axhspan(lo, hi, color="C2", alpha=0.15, lw=0)
        axs[2, j].set_yscale("log")
        axs[2, j].set_ylim(1e12, 1e21)
        for ax in axs[:, j]:
            ax.set_xscale("log")
            ax.grid(alpha=0.3, which="both")
        axs[0, j].set_title(f"{sw} = {x:g} ({anc.n2_percent(sw, x):g} % N$_2$), $N_e = n_c$", fontsize=10)
        axs[-1, j].set_xlabel("E/N [Td]")
    axs[0, 0].set_ylabel(r"$\chi^2$/dof at $N_e = n_c$")
    axs[1, 0].set_ylabel(r"$T_{e,\mathrm{eff}}$ [eV]")
    axs[2, 0].set_ylabel("CR 1s$_5$ [m$^{-3}$] (thick: in the measured range)")
    axs[0, -1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    mlf.build_tables([key(fA, pct) for fA in F_A for pct in FRACTIONS])
    ref = pd.read_csv(LOWEN_SUMMARY)
    est = estimate_n2a(ref)
    est.to_csv(os.path.join(OUTDIR, "n2a_estimate.csv"), index=False)
    print("N2(A) estimate at the DC fits at n_c (upper estimate, no quenching by N atoms):")
    print(est.to_string(index=False, float_format=lambda v: f"{v:.2g}"))
    print("fitting ...")
    res, ncs = fit_all()
    res.to_csv(os.path.join(OUTDIR, "n2a_summary.csv"), index=False)
    ncs.to_csv(os.path.join(OUTDIR, "n2a_nc_rows.csv"), index=False)
    plot_summary(res, ref, os.path.join(OUTDIR, "n2a_summary.png"))
    plot_nc_rows(ncs, pd.read_csv(LOWEN_NC_ROWS), os.path.join(OUTDIR, "n2a_nc_rows.png"))
    cols = ["sweep", "x", "fA", "chi2_nc", "EN_nc", "Te_nc", "n_1s5_nc", "chi2_1s5", "EN_1s5", "Te_1s5"]
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(res.sort_values(["sweep", "x", "fA"])[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nfigures in {OUTDIR}")
    return res, ncs


if __name__ == "__main__":
    RES, NC_ROWS = main()
