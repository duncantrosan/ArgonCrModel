# -*- coding: utf-8 -*-
"""
CrossSection4sComparison.py

Electron-impact excitation out of the four 4s levels: the BSR (B-spline R-matrix,
Zatsarinny & Bartschat) cross sections the CR model uses now (he.XSEC_4S = 'BSR') against
the RDW (relativistic distorted wave, NGFSRDW) ones it used before, and the analytic
(Drawin) cross sections the model filled in where RDW had no data.

1. rates_4s.png      Rate coefficient at Te = 1 eV (Maxwellian) of every channel out of each
                     4s level, per upper level: BSR, RDW, analytic.
2. sigma_4s.png      Cross sections of the 4s -> 4s mixing and the strongest 4s -> 4p channels.
3. densities_4s.png  CR model with BSR / with RDW 4s cross sections: level densities over Ne at
                     two EEDFs (BOLSIG+ microwave, pure Ar, Te_eff ~0.9 and ~1.3 eV); 4s levels
                     and the upper levels of the fitted lines.
4. chi2_profiles_RDW_vs_BSR.png, fit_changes.csv
                     The critical-density analysis (CriticalDensityFit.py) with each set: chi^2
                     profile over Ne of every condition, and the free and pinned fit results.
                     Needs the outputs of CriticalDensityFit.py with xsec_4s = 'BSR' and 'RDW'
                     (CRFitNeTe.CONFIG); skipped if one is missing.

Figures in Experimental_Data/Output/CrossSections4s/.  Run from Spyder: F5 (~1 min).
"""
import contextlib
import io
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT_DIR, "Experimental_Data", "ExperimentalDataAnalysis"))
import CRFitNeTe as crf                             # noqa: E402

he = crf._helpers()

# ---- settings -------------------------------------------------------------------
TE_RATE = 1.0                                       # eV, Maxwellian for the rate comparison
SIGMA_CHANNELS = [("4s1", "4s2"), ("4s1", "4s4"), ("4s3", "4s4"), ("4s1", "4p2"), ("4s1", "4p5"),
                  ("4s2", "4p3"), ("4s4", "4p7"), ("4s3", "4p7"), ("4s1", "3d4")]
EEDF_LIBRARY = he.BOLSIG_FOLDER / "Ar_Biagi_bolsig_mw"     # pure Ar, 2.45 GHz
TE_EEDF = (0.9, 1.3)                                # eV, EEDFs of the library closest to these
NE_GRID = np.geomspace(1e15, 3e19, 15)              # m^-3
LEVELS = ["4s1", "4s2", "4s3", "4s4", "4p3", "4p5", "4p6", "4p7", "4p8", "4p9", "4p10", "5p3", "5p5"]
CR = dict(P_Torr=1.0, Tg=300.0, R=0.04)             # as CRFitNeTe.CONFIG
OUTDIR = os.path.join(ROOT_DIR, "Experimental_Data", "Output", "CrossSections4s")
PASCHEN = {"4s1": "1s$_5$", "4s2": "1s$_4$", "4s3": "1s$_3$", "4s4": "1s$_2$"}
SRC = {"BSR": dict(color="C3", marker="o"), "RDW": dict(color="C0", marker="s"),
       "analytic": dict(color="0.5", marker="^")}


def channels(xsec):
    """{(lower, upper): entry} of the excitation channels out of the 4s levels in the model."""
    with contextlib.redirect_stdout(io.StringIO()):
        MD, _ = he.GetData(xsec)
    out = {}
    for lo in he.FOUR_S:
        for r in MD[lo]["Electron Impact CrossSections"]["Reactants"]:
            out[lo, r["partner_label"]] = r
    return MD, out


def rate(entry, Te, E=np.linspace(0, 60, 6001)):
    """Maxwellian rate coefficient [m^3/s] of a model cross-section entry."""
    F = he.Te2EEPF(Te, E)
    s = np.interp(E, entry["energy_eV"], entry["cross_section"], left=0.0, right=0.0)
    s[E <= entry["threshold_eV"]] = 0.0
    return np.sqrt(2 * 1.60217e-19 / 9.10938e-31) * np.trapezoid(np.sqrt(E) * F * s, E)


def source(entry):
    return "analytic" if entry.get("analytic") else ("BSR" if entry.get("database") == "BSR" else "RDW")


def plot_rates(MD, ch_rdw, ch_bsr, path):
    uppers = sorted({u for _, u in list(ch_rdw) + list(ch_bsr)}, key=lambda u: MD[u]["energy_eV"])
    fig, axs = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    xi = {u: i for i, u in enumerate(uppers)}
    for ax, lo in zip(axs, he.FOUR_S):
        for ch, dx, filled in ((ch_rdw, -0.15, False), (ch_bsr, 0.15, True)):
            for (l, u), e in ch.items():
                if l != lo:
                    continue
                src = source(e)
                if ch is ch_bsr and src != "BSR":
                    continue                       # same channel as in the RDW model
                st = SRC[src]
                ax.semilogy(xi[u] + dx, rate(e, TE_RATE), st["marker"], color=st["color"], ms=6,
                            mfc=st["color"] if filled else "white")
        ax.set_ylabel(f"$k$({PASCHEN[lo]} $\\rightarrow$ X) [m$^3$/s]")
        ax.grid(alpha=0.3, which="both")
        ax.set_ylim(1e-20, 1e-11)
    for src, st in SRC.items():
        axs[0].plot([], [], st["marker"], color=st["color"], mfc=st["color"] if src == "BSR" else "white",
                    label={"BSR": "BSR (used now)", "RDW": "RDW (used before)",
                           "analytic": "analytic Drawin (used before, no RDW data)"}[src])
    axs[0].legend(fontsize=9, ncol=3, loc="upper left")
    axs[-1].set_xticks(range(len(uppers)), uppers, rotation=90)
    axs[-1].set_xlabel("upper level")
    fig.suptitle(f"Electron-impact excitation out of the 4s levels: rate coefficients at "
                 f"$T_e$ = {TE_RATE:g} eV (Maxwellian)\nopen symbols: model before (RDW + analytic), "
                 "filled: BSR; 4s $\\rightarrow$ 5p stays RDW")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sigma(ch_rdw, ch_bsr, path):
    n = len(SIGMA_CHANNELS)
    fig, axs = plt.subplots(int(np.ceil(n / 3)), 3, figsize=(15, 3.6 * np.ceil(n / 3)), squeeze=False)
    for ax, key in zip(axs.ravel(), SIGMA_CHANNELS):
        for ch, lab in ((ch_rdw, "before"), (ch_bsr, "now")):
            e = ch.get(key)
            if e is None:
                continue
            src = source(e)
            E = np.geomspace(e["threshold_eV"] * 1.001, 100, 800)    # as the model interpolates
            s = np.interp(E, e["energy_eV"], e["cross_section"], left=0.0, right=0.0)
            ax.loglog(E[s > 0], s[s > 0], "-" if lab == "now" else "--",
                      color=SRC[src]["color"], lw=1.6, label=f"{src} ({lab})")
        ax.set_xlim(0.03, 100)
        ax.set_title(f"{PASCHEN.get(key[0], key[0])} $\\rightarrow$ {PASCHEN.get(key[1], key[1])} "
                     f"({key[0]} $\\rightarrow$ {key[1]})", fontsize=10)
        ax.set_xlabel("electron energy [eV]")
        ax.set_ylabel(r"$\sigma$ [m$^2$]")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    for ax in axs.ravel()[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def densities(xsec, eedfs):
    """{Te_eff: array (Ne, levels)} of CR-model densities at the given EEDFs."""
    cr, _, MD, interp = crf.load_cr_model(OUTDIR, xsec)
    MD = he.AddDiffusionLoss(MD, CR["P_Torr"], CR["Tg"], CR["R"])
    out = {}
    for e in eedfs:
        rows = []
        for ne in NE_GRID:
            with contextlib.redirect_stdout(io.StringIO()):
                D, _, _, _ = cr["CRModel"](MD, e, float(ne), CR["P_Torr"], CR["Tg"], CR["R"], interp)
            rows.append([D[lbl]["density_m^-3"] for lbl in LEVELS])
        out[e["Te_eff"]] = np.array(rows)
    return out


def plot_densities(dens, path):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    cols = dict(zip(LEVELS, plt.cm.tab20(np.linspace(0, 1, 20))))
    for ax, te in zip(axs, sorted(dens["BSR"])):
        ratio = dens["BSR"][te] / dens["RDW"][te]
        for k, lbl in enumerate(LEVELS):
            ax.semilogx(NE_GRID, ratio[:, k], "-" if lbl.startswith("4s") else "--", color=cols[lbl],
                        lw=2.4 if lbl.startswith("4s") else 1.3, label=PASCHEN.get(lbl, lbl))
        ax.axhline(1, color="k", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("$N_e$ [m$^{-3}$]")
        ax.set_title(f"BOLSIG+ microwave EEDF (pure Ar), $T_{{e,\\mathrm{{eff}}}}$ = {te:.2f} eV")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("density with BSR / with RDW 4s cross sections")
    axs[1].legend(fontsize=8, ncol=2)
    fig.suptitle("CR model, 1 Torr: effect of the 4s cross sections on the 4s densities and the upper "
                 "levels of the fitted lines")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fit_comparison(outdir=OUTDIR):
    """chi^2 profiles over Ne and fit results of CriticalDensityFit with the RDW and the BSR set."""
    import pandas as pd
    dirs = {x: crf.output_dir("CriticalDensity", {"xsec_4s": x}) for x in ("RDW", "BSR")}
    if not all(os.path.isfile(os.path.join(d, "chi2_profiles.csv")) for d in dirs.values()):
        print("CriticalDensityFit outputs for both cross-section sets not found - fit comparison skipped")
        return None
    prof = {x: pd.read_csv(os.path.join(d, "chi2_profiles.csv")) for x, d in dirs.items()}
    fig, axs = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=True)
    for i, eedf in enumerate(("microwave", "dc")):
        for j, x in enumerate(("RDW", "BSR")):
            ax = axs[i, j]
            p = prof[x][prof[x].eedf == eedf]
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
            n_c = constants.epsilon_0 * constants.m_e * (2 * np.pi * 2.42e9) ** 2 / constants.e ** 2
            ax.axvline(n_c, color="k", lw=1.5)
            ax.axvspan(1e12, n_c, color="0.92", lw=0)
            ax.set_xlim(1e13, 3e19)
            ax.set_ylim(0, 30)
            ax.grid(alpha=0.3, which="both")
            ax.set_title(f"{eedf} EEDF, {x} 4s cross sections" + (" (before)" if x == "RDW" else " (now)"))
            if i == 1:
                ax.set_xlabel("$N_e$ [m$^{-3}$]")
            if j == 0:
                ax.set_ylabel(r"$\Delta\chi^2 / s^2$ against the best fit")
    h, l = axs[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle(r"$\chi^2$ profiled over E/N vs $N_e$ (CriticalDensityFit): RDW vs BSR excitation out of "
                 r"the 4s levels; vertical line: critical density $n_c$ (2.42 GHz)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(os.path.join(outdir, "chi2_profiles_RDW_vs_BSR.png"), dpi=150)
    plt.close(fig)
    cols = ["eedf", "sweep", "x", "Te_free", "Ne_free", "chi2_red_free", "delta_chi2_at_nc", "Te_pinned",
            "chi2_red_pinned", "n_1s5_free", "n_1s5_pinned"]
    fits = {x: pd.read_csv(os.path.join(d, "pinned_fits.csv"))[cols] for x, d in dirs.items()}
    m = fits["RDW"].merge(fits["BSR"], on=["eedf", "sweep", "x"], suffixes=("_RDW", "_BSR"))
    m.to_csv(os.path.join(outdir, "fit_changes.csv"), index=False)
    show = ["eedf", "sweep", "x"] + [f"{c}_{x}" for c in ("Te_free", "Ne_free", "chi2_red_free",
                                                          "delta_chi2_at_nc", "n_1s5_free") for x in ("RDW", "BSR")]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(m.sort_values(["eedf", "sweep", "x"])[show].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return m


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    MD, ch_rdw = channels("RDW")
    _, ch_bsr = channels("BSR")
    plot_rates(MD, ch_rdw, ch_bsr, os.path.join(OUTDIR, "rates_4s.png"))
    plot_sigma(ch_rdw, ch_bsr, os.path.join(OUTDIR, "sigma_4s.png"))
    print(f"{'channel':<12s} {'before':>9s} {'k before':>10s} {'k BSR':>10s} {'BSR/before':>10s}"
          f"   (Maxwellian Te = {TE_RATE:g} eV)")
    for key in sorted(ch_bsr, key=lambda k: (k[0], MD[k[1]]["energy_eV"])):
        if source(ch_bsr[key]) != "BSR":
            continue
        kb = rate(ch_bsr[key], TE_RATE)
        old = ch_rdw.get(key)
        ko = rate(old, TE_RATE) if old is not None else np.nan
        print(f"{key[0] + '->' + key[1]:<12s} {source(old) if old is not None else 'none':>9s} "
              f"{ko:10.2e} {kb:10.2e} {kb / ko:10.2f}")
    lib = he.ImportBolsigEEDFs(EEDF_LIBRARY, verbose=False)
    eedfs = [min(lib, key=lambda e: abs(e["Te_eff"] - t)) for t in TE_EEDF]
    dens = {x: densities(x, eedfs) for x in ("RDW", "BSR")}
    plot_densities(dens, os.path.join(OUTDIR, "densities_4s.png"))
    fit_comparison()
    print(f"\nfigures in {OUTDIR}")


if __name__ == "__main__":
    main()
