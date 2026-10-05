# -*- coding: utf-8 -*-
"""
PureArgonMicrowaveNe.py

Presentation figures: the CR fit of the pure-Ar spectrum (N2 fraction sweep at 0 %, 85 W,
1 Torr) with the 2.45 GHz microwave EEDFs (BOLSIG+, Biagi Ar, the MicrowaveLowENFit library
from 3 Td), walked in Ne: at every Ne the E/N is refitted (chi^2 profiled over E/N).

  pure_Ar_microwave_Ne.png      all four panels
  chi2_vs_Ne.png                chi^2/dof against Ne (Ne fixed: dof + 1)
  delta_chi2_vs_Ne.png          Delta chi^2 / s^2 against the best fit (1 and 4 marked)
  Te_vs_Ne.png                  Te_eff = 2/3 <E> at the best E/N
  1s5_vs_Ne.png                 CR 1s5 density against the absorption estimate
  pure_Ar_microwave_Ne.csv      the profile (Ne, chi^2/dof, Delta chi^2/s^2, E/N, Te_eff, 1s5)
with the critical density n_c (2.42 GHz) and the fit values there marked.

Needs the CR table of MicrowaveLowENFit (built here if missing, ~20 min).
Output in Experimental_Data/Output/PureArgonMicrowave/.  Run from Spyder (F5) or
python SmallAnalysisScripts/PureArgonMicrowaveNe.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import MicrowaveLowENFit as mlf                     # noqa: E402

crf, he = mlf.crf, mlf.he

# ---- settings -------------------------------------------------------------------
SWEEP, X = "N2 fraction", 0.0                       # pure Ar, 85 W, 1 Torr
NE_RANGE = (1e13, 3e19)                             # m^-3, x range of the figures
COLOR = "C3"
FONT = 13                                           # base font size (slides)
MARK_NC = True                                      # mark and label the fit at n_c
# line_sets_vs_Ne.png: the same profile with fewer lines (air wavelengths kept, response tilt);
# None = every line of the CR fit selection (CRFitNeTe CONFIG)
LINE_SETS = {
    "all 9 lines, response tilt free (CR fit default)": (None, 1),
    "6 lines (no 430.01, 727.29, 794.82), tilt free": ((415.859, 706.722, 750.387, 751.465, 800.616, 801.479), 1),
    "6 lines, calibration trusted (no tilt)": ((415.859, 706.722, 750.387, 751.465, 800.616, 801.479), None),
    "3 lines (415.86, 706.72, 750.39), calibration trusted": ((415.859, 706.722, 750.387), None),
}
OUTDIR = os.path.join(mlf.ROOT_DIR, "Experimental_Data", "Output", "PureArgonMicrowave")


# ---- fit ------------------------------------------------------------------------
def line_cfg(keep=None, tilt=1):
    """Fit settings of the pure-Ar microwave table with only the lines keep (air nm) and the
    response tilt degree tilt (None = trust the intensity calibration)."""
    cfg = dict(mlf.job_cfg("lowEN:0"), response_deg=tilt)
    if keep is not None:
        with mlf.contextlib.redirect_stdout(mlf.io.StringIO()):
            tab = crf.build_model_table(cfg)
        _, comps = crf.select_features(tab, cfg)
        drop = [w for w in comps.wl_air.round(3) if not np.isclose(keep, w, atol=1e-3).any()]
        cfg["exclude_wl"] = tuple(cfg["exclude_wl"]) + tuple(drop)
    return cfg


def profile(cfg=None):
    """The pure-Ar fit profiled over E/N at every Ne, its free best fit and the values at n_c."""
    cfg = cfg or mlf.job_cfg("lowEN:0")
    fit = mlf.prepare(cfg, with_nu=False)
    d = fit["ft"][(fit["ft"].sweep == SWEEP) & (fit["ft"].x == X)]
    s, post, _ = crf.fit_block(d, fit["M"], fit["feats"], fit["grid"], fit["Te_f"], cfg)
    p = mlf.ne_path(fit, post, s["chi2_min"], s["birge"])
    p["chi2_red"] = (s["chi2_min"] + p.dchi2 * s["birge"] ** 2) / (s["dof"] + 1)
    nc = p.iloc[int(np.argmin(np.abs(np.log(p.Ne / mlf.N_C))))]
    return p, s, nc


# ---- plots ----------------------------------------------------------------------
PANELS = {   # column, y label, y scale, y limits
    "chi2_red": (r"$\chi^2$/dof (E/N refitted at each $N_e$)", "linear", (0, 2.0)),
    "dchi2": (r"$\Delta\chi^2/s^2$ against the best fit", "linear", (0, 30)),
    "Te_eff": (r"$T_{e,\mathrm{eff}} = \frac{2}{3}\langle\varepsilon\rangle$ at the best E/N [eV]", "linear", (0, 2.0)),
    "n_1s5": (r"CR 1s$_5$ density [m$^{-3}$]", "log", (1e10, 1e19)),
}
NC_TEXT = {"chi2_red": "{:.2f}", "dchi2": "{:.1f}", "Te_eff": "{:.2f} eV", "n_1s5": "{:.1e} m$^{{-3}}$"}


def draw(ax, p, nc, key):
    ylab, scale, ylim = PANELS[key]
    ax.axvline(mlf.N_C, color="k", lw=1.3)
    ax.text(mlf.N_C * 1.15, ylim[0] + 0.96 * (ylim[1] - ylim[0]) if scale == "linear" else ylim[1] / 3,
            f"$n_c$ = {mlf.N_C:.1e} m$^{{-3}}$", fontsize=FONT - 3, va="top")
    if key == "chi2_red":
        ax.axhline(1, color="0.5", ls=":", lw=1)
    if key == "dchi2":
        ax.axhline(1, color="0.5", ls=":", lw=1)
        ax.axhline(4, color="0.5", ls="--", lw=1)
    if key == "n_1s5":
        ax.axhspan(*mlf.MEASURED_1S5, color="C2", alpha=0.18, lw=0, label="measured (absorption)")
        ax.legend(fontsize=FONT - 3, loc="lower right")
    ax.plot(p.Ne, p[key], color=COLOR, lw=2.6)
    if MARK_NC:
        ax.plot(nc.Ne, nc[key], "o", color=COLOR, mec="k", ms=9, zorder=5)
        y = nc[key] * (3 if scale == "log" else 1) + (0 if scale == "log" else 0.06 * (ylim[1] - ylim[0]))
        ax.text(nc.Ne * 0.8, y, NC_TEXT[key].format(nc[key]), fontsize=FONT - 2, ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale(scale)
    ax.set_xlim(*NE_RANGE)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$N_e$ [m$^{-3}$]", fontsize=FONT)
    ax.set_ylabel(ylab, fontsize=FONT)
    ax.tick_params(labelsize=FONT - 2)
    ax.grid(alpha=0.3, which="both")


def plot_line_sets(path):
    """chi^2/dof and Delta chi^2/s^2 against Ne for each entry of LINE_SETS."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.8))
    for (name, (keep, tilt)), col in zip(LINE_SETS.items(), ("C3", "C1", "C0", "C2")):
        p, s, nc = profile(line_cfg(keep, tilt))
        lab = f"{name}: best $N_e$ {s['Ne_best']:.1e}, at $n_c$ $T_{{e,eff}}$ {nc.Te_eff:.2f} eV"
        axs[0].plot(p.Ne, p.chi2_red, color=col, lw=2.2, label=lab)
        axs[1].plot(p.Ne, p.dchi2, color=col, lw=2.2)
    for ax, (ylab, ylim) in zip(axs, ((r"$\chi^2$/dof (E/N refitted at each $N_e$)", (0, 2.0)),
                                      (r"$\Delta\chi^2/s^2$ against each set's best fit", (0, 30)))):
        ax.axvline(mlf.N_C, color="k", lw=1.3)
        ax.axhline(1, color="0.5", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlim(*NE_RANGE)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$N_e$ [m$^{-3}$] (black: $n_c$)", fontsize=FONT)
        ax.set_ylabel(ylab, fontsize=FONT)
        ax.tick_params(labelsize=FONT - 2)
        ax.grid(alpha=0.3, which="both")
    axs[0].legend(fontsize=FONT - 5, loc="upper left")
    fig.suptitle("Pure Ar, microwave EEDFs: the $N_e$ profile with fewer lines", fontsize=FONT + 1)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plot_line_sets(os.path.join(OUTDIR, "line_sets_vs_Ne.png"))
    p, s, nc = profile()
    p[["Ne", "chi2_red", "dchi2", "EN", "Te_eff", "n_1s5"]].to_csv(
        os.path.join(OUTDIR, "pure_Ar_microwave_Ne.csv"), index=False)
    title = "Pure Ar, 85 W, 1 Torr: CR fit with 2.45 GHz microwave EEDFs"
    fig, axs = plt.subplots(2, 2, figsize=(14, 9.5), sharex=True)
    for ax, key in zip(axs.ravel(), PANELS):
        draw(ax, p, nc, key)
    for ax in axs[0]:
        ax.set_xlabel("")
    fig.suptitle(title, fontsize=FONT + 2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "pure_Ar_microwave_Ne.png"), dpi=200)
    plt.close(fig)
    names = {"chi2_red": "chi2_vs_Ne", "dchi2": "delta_chi2_vs_Ne", "Te_eff": "Te_vs_Ne", "n_1s5": "1s5_vs_Ne"}
    for key, name in names.items():
        fig, ax = plt.subplots(figsize=(8, 5.5))
        draw(ax, p, nc, key)
        ax.set_title(title, fontsize=FONT)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"{name}.png"), dpi=200)
        plt.close(fig)
    print(f"free fit: Ne = {s['Ne_best']:.2e} m^-3, E/N = {s['x_best']:.3g} Td, chi2/dof = {s['chi2_red']:.2f}")
    print(f"at n_c = {mlf.N_C:.2e} m^-3: chi2/dof = {nc.chi2_red:.2f}, Delta chi2/s^2 = {nc.dchi2:.1f}, "
          f"E/N = {nc.EN:.3g} Td, Te_eff = {nc.Te_eff:.2f} eV, CR 1s5 = {nc.n_1s5:.2e} m^-3")
    print(f"figures in {OUTDIR}")
    return p


if __name__ == "__main__":
    PROFILE = main()
