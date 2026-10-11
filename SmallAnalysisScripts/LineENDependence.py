# -*- coding: utf-8 -*-
"""
LineENDependence.py

How the Ar I 5p -> 4s (400-475 nm) and 4p -> 4s (690-860 nm) line intensities of the CR model depend on the EEDF
(E/N of the pure-Ar 2.42 GHz microwave BOLSIG+ library, i.e. Te_eff) at constant Ne = n_c, from the cached CR tables
of N2SweepPureArEEDF (AnalyticFivePFit, analytic 5p cross sections), 1 Torr, 650 K, 0 and 2 % N2.

Per table: the strongest 5p and 4p lines at REF_EN, each normalised to its value at REF_EN, against E/N (top axis
Te_eff); the 5p/4p ratio (summed lines); and the direct (ground-state) share of the upper-level production.
The E/N range of the fits at n_c (31-44 Td in the N2 sweeps) is shaded.

Output: Experimental_Data/Output/LineENDependence/line_EN_dependence_<table>.png, slopes.csv
Run from Spyder (F5) or python SmallAnalysisScripts/LineENDependence.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLE_DIR = os.path.join(ROOT, "Experimental_Data", "Output", "AnalyticFivePFit", "tables",
                         "ArEEDF_Ar_Biagi_bolsig_mw2.42GHz_650K_1000mTorr")
TABLES = {"0 % N$_2$": "cr_model_table_Ar_Biagi_bolsig_mw2.42GHz_650K_1000mTorr.npz",
          "2 % N$_2$": "cr_model_table_Ar_Biagi_bolsig_mw2.42GHz_650K_1000mTorr_N2_2pct.npz"}
OUTDIR = os.path.join(ROOT, "Experimental_Data", "Output", "LineENDependence")
NE = 7.26e16                    # m^-3, critical density of 2.42 GHz
REF_EN = 35.0                   # Td, normalisation (fits at n_c: 31-44 Td)
FIT_EN = (31.0, 44.0)
N_LINES = 8
GROUPS = {"5p": (400.0, 475.0), "4p": (690.0, 860.0)}


def at_ne(v, Ne_grid, Ne=NE):
    """Values on (x, Ne[, ...]) interpolated linearly in ln v and ln Ne to Ne."""
    lv = np.log(np.clip(v, 1e-300, None))
    j = np.searchsorted(Ne_grid, Ne) - 1
    w = (np.log(Ne) - np.log(Ne_grid[j])) / (np.log(Ne_grid[j + 1]) - np.log(Ne_grid[j]))
    return np.exp((1 - w) * lv[:, j] + w * lv[:, j + 1])


def lines(t, group):
    lo, hi = GROUPS[group]
    up, low, wl = (t[k].astype(str) if k != "line_wl" else t[k] for k in ("line_upper", "line_lower", "line_wl"))
    return [k for k in range(len(wl)) if up[k].startswith(group) and low[k].startswith("4s") and lo <= wl[k] <= hi]


def plot_table(name, t):
    EN, Te = t["x_grid"], t["Te_eff"]
    I = at_ne(t["I_obs"], t["Ne_grid"])                     # (x, line), photons m^-3 s^-1
    fd = at_ne(np.nan_to_num(t["fdirect"], nan=1.0), t["Ne_grid"])
    lev = list(t["levels"].astype(str))
    r = int(np.argmin(np.abs(np.log(EN / REF_EN))))
    i0, i1 = np.searchsorted(EN, FIT_EN)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    rows, sums = [], {}
    for ax, (g, col) in zip(axs[0], (("5p", "Blues"), ("4p", "Reds"))):
        ks = sorted(lines(t, g), key=lambda k: -I[r, k])[:N_LINES]
        cols = plt.get_cmap(col)(np.linspace(0.45, 1, len(ks)))
        sums[g] = I[:, lines(t, g)].sum(axis=1)
        for c, k in zip(cols, ks):
            y = I[:, k] / I[r, k]
            slope = np.polyfit(np.log(EN[i0:i1]), np.log(I[i0:i1, k]), 1)[0]
            rows.append(dict(table=name, group=g, wl=round(float(t["line_wl"][k]), 3), upper=str(t["line_upper"][k]),
                             x_vs_EN_31_44=I[i1 - 1, k] / I[i0, k], dlnI_dlnEN=slope))
            ax.plot(EN, y, "-", color=c, lw=1.8, label=f"{t['line_wl'][k]:.2f} ({t['line_upper'][k]}), "
                                                       f"x{I[i1 - 1, k] / I[i0, k]:.2f} over 31-44 Td")
        ax.set(xscale="log", yscale="log", ylim=(1e-2, 1e3), xlabel="E/N of the EEDF [Td]",
               ylabel=f"I / I({REF_EN:g} Td)", title=f"Ar I {g} $\\rightarrow$ 4s lines, {name}, N$_e$ = n$_c$")
    for ax in axs.ravel():
        ax.axvspan(*FIT_EN, color="0.85", zorder=0)
        ax.grid(True, which="both", alpha=0.25)
    for ax in axs[0]:
        ax.legend(fontsize=7, loc="upper left")
    ratio = sums["5p"] / sums["4p"]
    axs[1, 0].plot(EN, ratio / ratio[r], "k-", lw=2.2, label="5p / 4p (all lines summed)")
    tw = axs[1, 0].twinx()                              # Te_eff is not monotonic in E/N below ~20 Td: no top axis
    tw.plot(EN, Te, "g--", lw=1.5)
    tw.set_ylabel(r"$T_{e,\mathrm{eff}} = \frac{2}{3}\langle\varepsilon\rangle$ [eV] (green, dashed)", color="g")
    axs[1, 0].set(xscale="log", yscale="log", xlabel="E/N of the EEDF [Td]", ylabel=f"relative to {REF_EN:g} Td",
                  title=f"5p/4p ratio: x{ratio[i1 - 1] / ratio[i0]:.2f} over 31-44 Td "
                        f"(Te_eff {Te[i0]:.3f} -> {Te[i1 - 1]:.3f} eV)")
    axs[1, 0].legend(fontsize=8)
    for g, c in (("5p", "C0"), ("4p", "C3")):
        ups = sorted({str(u) for u in t["line_upper"][lines(t, g)]})
        f = np.array([fd[:, lev.index(u)] for u in ups])
        axs[1, 1].fill_between(EN, f.min(0), f.max(0), color=c, alpha=0.2)
        axs[1, 1].plot(EN, np.median(f, axis=0), "-", color=c, lw=2, label=f"{g} levels (median, band: min-max)")
    axs[1, 1].set(xscale="log", ylim=(0, 1), xlabel="E/N of the EEDF [Td]",
                  ylabel="direct (ground-state) share of production",
                  title="rest: stepwise from the 1s levels and cascades")
    axs[1, 1].legend(fontsize=8)
    fig.suptitle(f"CR model, pure-Ar 2.42 GHz BOLSIG+ EEDF, 1 Torr, 650 K, {name}, N$_e$ = n$_c$ = {NE:.2e} m$^{{-3}}$; "
                 "grey: E/N of the fits at n$_c$", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"line_EN_dependence_{name.split()[0]}pctN2.png"), dpi=200)
    return rows


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for name, f in TABLES.items():
        with np.load(os.path.join(TABLE_DIR, f)) as t:
            rows += plot_table(name.replace("$_2$", "2"), {k: t[k] for k in t.files})
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUTDIR, "slopes.csv"), index=False)
    print(d.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"output in {OUTDIR}")


if __name__ == "__main__":
    main()
