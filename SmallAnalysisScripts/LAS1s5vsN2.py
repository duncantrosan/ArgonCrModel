# -*- coding: utf-8 -*-
"""
LAS1s5vsN2.py

Ar(1s5) density against % N2: CR model (microwave EEDF, Ne pinned at the critical density n_c, gas temperature
from LAS) fitted to the echelle N2 percent sweep (PrelijminaryNitrogenAdmixtureMini/GsaSweep_85W_1_torr, 85 W,
1 Torr, 0-3 % N2), and the LAS 1s5 measured at 85 W (1 Torr run, and the two 0.9 Torr runs for reference).

Reads the fits from LASTgComparison (Experimental_Data/Output/LASTgComparison/comparison.csv) - run that first;
nothing is refitted here.  Output: Experimental_Data/Output/LASTgComparison/figures/1s5_vs_N2_cr_las.png
Run from Spyder (F5) or python SmallAnalysisScripts/LAS1s5vsN2.py
"""
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from LASColumns import read_las

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "Experimental_Data", "Output", "LASTgComparison")
COMP_FILE = os.path.join(OUTDIR, "comparison.csv")
LAS_FILE = os.path.join(ROOT, "Experimental_Data", "LAS_Data", "Master_AllMeasurements.csv")
P_W = 85.0                    # W, power of the OES N2 sweep; LAS rows at exactly this power
LAS_FREQ_MHZ = (2400, 2450)   # as LASTgComparison
X_MAX = 3.5                   # % N2 shown (the OES sweep ends at 3 %)
LAS_SETS = [                  # pressure [Torr], marker, filled, label
    (1.0, "D", True, "LAS, 1 Torr (29 Sep)"),
    (0.9, "o", False, "LAS, 0.9 Torr (26 Jul, 29 Sep)"),
]


def main():
    comp = pd.read_csv(COMP_FILE)
    cr = comp[comp.sweep == "N2 fraction"].sort_values("x")
    las = read_las(LAS_FILE)
    las = las[las.frequency.between(*LAS_FREQ_MHZ) & np.isclose(las.power, P_W)
              & (las.Nitrogen_Percent <= X_MAX) & (las.N_s_cm3 > 0)]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.fill_between(cr.x, cr.n_1s5_lo, cr.n_1s5_hi, color="C3", alpha=0.15, lw=0)
    ax.plot(cr.x, cr.n_1s5, "s-", color="C3", ms=7, lw=2,
            label="CR model: microwave EEDF, $N_e = n_c$, LAS $T_g$\n(band: $\\Delta\\chi^2 \\leq 1\\,s^2$ along E/N)")
    for p, mk, filled, lab in LAS_SETS:
        g = las[np.isclose(las.pressure, p)].sort_values("Nitrogen_Percent")
        ax.errorbar(g.Nitrogen_Percent, g.N_s_cm3 * 1e6, yerr=g.N_s_cm3_err * 1e6, fmt=mk, ms=7, capsize=3,
                    color="k" if filled else "0.45", mfc=None if filled else "white", label=lab)
    for r in cr.itertuples():   # CR / LAS at each OES condition
        ax.annotate(f"{r.ratio_cr_las:.0f}$\\times$" if r.ratio_cr_las >= 10 else f"{r.ratio_cr_las:.1f}$\\times$",
                    (r.x, r.n_1s5_hi), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color="0.3")
    ax.set_yscale("log")
    ax.set_xlim(-0.2, X_MAX)
    ax.set_ylim(1e16, 5e18)
    ax.set_xlabel("N$_2$ in the feed [%]")
    ax.set_ylabel("Ar(1s$_5$) density [m$^{-3}$]")
    ax.set_title(f"Ar(1s$_5$) vs N$_2$ admixture, {P_W:g} W, 1 Torr (echelle N$_2$ sweep)\n"
                 "numbers: CR / LAS at each condition", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    path = os.path.join(OUTDIR, "figures", "1s5_vs_N2_cr_las.png")
    fig.savefig(path, dpi=200)
    print(path)
    print(cr[["x", "Tg", "EN_nc", "chi2_red_nc", "n_1s5", "las_1s5", "las_1s5_how", "ratio_cr_las"]].to_string(index=False))
    plt.show()


if __name__ == "__main__":
    main()
