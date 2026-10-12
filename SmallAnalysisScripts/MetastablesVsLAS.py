# -*- coding: utf-8 -*-
"""
MetastablesVsLAS.py

Ar(1s5) of the CR fit (fits.csv of Main.py runs) against the LAS measurements (Master_AllMeasurements.csv) for the
80 W N2 admixture sweeps at 0.6, 1 and 1.5 Torr: one panel per pressure with every LAS measurement at that pressure
and power (points), their mean per % N2 (diamonds) and the CR 1s5 with its band of each run (RUNS); bottom row the
ratio CR / mean LAS.  Conditions whose free fit sits at the Ne grid edge (the cold, dense
branch, CR 1s5 ~ 1e7 m^-3) are marked at the bottom instead of drawn.

Output: Experimental_Data/Output/N2Sweep80W_metastables_vs_LAS.png
Run from Spyder (F5) or python SmallAnalysisScripts/MetastablesVsLAS.py (after python Main.py input_N2sweep_80W.toml
and input_N2sweep_80W_vibsat.toml)
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "Experimental_Data", "Output")
LAS_FILE = os.path.join(ROOT, "Experimental_Data", "LAS_Data", "Master_AllMeasurements.csv")
RUNS = {"pure-Ar EEDF": ("N2Sweep80W_pureArEEDF", "C0"), "saturated N$_2$(v) EEDF": ("N2Sweep80W_vibsatEEDF", "C3")}
SETS = {0.6: "80W_0.6Torr_0-10pctN2", 1.0: "80W_1Torr_0-10pctN2", 1.5: "80W_1.5Torr_0-10pctN2"}
POWER_W, FREQ_MHZ = 80.0, (2400, 2450)
EDGE_NE = 2.9e19                                     # m^-3, top of the Ne grid


def main():
    las = pd.read_csv(LAS_FILE)
    las = las[np.isclose(las.power, POWER_W) & las.frequency.between(*FREQ_MHZ) & (las.N_s_cm3 > 0)]
    fig, axs = plt.subplots(2, len(SETS), figsize=(6 * len(SETS), 9), sharex=True,
                            gridspec_kw=dict(height_ratios=[2, 1]))
    for col, (p, name) in enumerate(SETS.items()):
        ax, rx = axs[0, col], axs[1, col]
        m = las[np.isclose(las.pressure, p)].assign(x=lambda d: d.Nitrogen_Percent.round(2), n=lambda d: d.N_s_cm3 * 1e6)
        mean = m.groupby("x").n.agg(["mean", "std", "size"]).reset_index()
        ax.plot(m.x, m.n, ".", color="0.6", ms=5, label="LAS, single measurements")
        ax.errorbar(mean.x, mean["mean"], yerr=mean["std"].fillna(0), fmt="D", color="k", mfc="white", ms=7, capsize=3,
                    label="LAS, mean per % N$_2$")
        top = float(np.nanmax(mean["mean"] + mean["std"].fillna(0))) if len(mean) else 0.0
        edges = []
        for lab, (folder, c) in RUNS.items():
            f = os.path.join(OUT, folder, name, "fits.csv")
            if not os.path.exists(f):
                continue
            t = pd.read_csv(f).sort_values("x")
            edge = t.Ne_fit >= EDGE_NE                 # free fit on the cold, dense branch at the Ne grid edge
            e, t = t[edge], t[~edge]
            ax.fill_between(t.x, t.n_1s5_lo, t.n_1s5_hi, color=c, alpha=0.15, lw=0)
            ax.plot(t.x, t.n_1s5, "s-", color=c, ms=6, lw=2, label=f"CR fit, {lab}")
            top = max(top, float(np.nanmax(t.n_1s5_hi)) if len(t) else 0.0)
            if len(e):
                edges.append((e.x, c, lab))
            r = t.merge(mean, on="x")
            rx.plot(r.x, r.n_1s5 / r["mean"], "s-", color=c, ms=6, lw=2, label=lab)
        for k, (ex, c, lab) in enumerate(edges):        # dense-branch fits (CR 1s5 ~ 0) as markers at the bottom
            ax.plot(ex, np.full(len(ex), (0.02 + 0.03 * k) * 1.15 * top), "v", color=c, mfc="white", ms=9,
                    label=f"{lab}: free fit at the N$_e$ grid edge (CR 1s$_5$ ~ 0)")
        ax.set(ylim=(0, 1.15 * top), ylabel="Ar(1s$_5$) [m$^{-3}$]",
               title=f"{POWER_W:g} W, {p:g} Torr: CR fit (N$_e$ fitted) vs LAS")
        rx.axhline(1, color="k", lw=1)
        rx.set(ylim=(0, 3), xlabel="N$_2$ in the feed [%]", ylabel="CR / LAS (mean)")
        for a in (ax, rx):
            a.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "N2Sweep80W_metastables_vs_LAS.png")
    fig.savefig(path, dpi=180)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
