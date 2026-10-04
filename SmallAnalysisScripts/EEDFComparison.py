# -*- coding: utf-8 -*-
"""
EEDFComparison.py

Ground-state excitation rate coefficients of a few actinometry upper levels for
MultiBolt EEDFs (Ar/N2 mixture, several numbers of Legendre terms) and for a
Maxwellian, plotted against Te_eff = 2/3 <E>.

MultiBolt runs land in InputData/MultiBolt/<RUN_PREFIX>_<N>terms and are reused
when they already exist (delete the folder to rerun).
Run from Spyder: edit the settings, press F5.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "Experimental_Data", "ExperimentalDataAnalysis"))
import ActinometryRates as ar                       # noqa: E402
import CRFitNeTe as crf                             # noqa: E402

he = crf._helpers()

# ---- settings -------------------------------------------------------------------
XSEC_DIR = r"C:\Users\dptro\Documents\Work\Python\Multibolt\MultiBolt-master\MultiBolt-master\cross-sections"
XSEC_FILES = [os.path.join(XSEC_DIR, "Biagi_Ar.txt"), os.path.join(XSEC_DIR, "Biagi_N2.txt")]
MIXTURE = {"Ar": 0.95, "N2": 0.05}
EN_TD = [3,  4, 5,6  , 7,8,9, 10,11,12,13,14, 15, 20, 30, 50, 75, 100, 150, 200]
N_TERMS = [2, 4, 6, 8]
RUN_PREFIX = "ArN2_5pct"
TE_MAXWELL = np.linspace(0.5, 8, 60)            # eV
LEVELS = [("Ar I", "5p5", "Ar I 415.9 nm (5p5)"),
          ("Ar I", "4p10", "Ar I 750.4 nm (4p10)"),
          ("N I", "other21", "N I 744.2 nm (3p 4S)")]
OUTPUT = os.path.join(crf.ROOT_DIR, "Experimental_Data", "Output", "EEDFComparison.png")


def multibolt_run(n_terms):
    name = f"{RUN_PREFIX}_{n_terms}terms"
    folder = os.path.join(he.MULTIBOLT_FOLDER, name)
    if os.path.isdir(os.path.join(folder, "EEDFs_f0")):
        return he.ImportMultiBoltEEDFs(folder, verbose=False)
    return he.RunMultiBolt(XSEC_FILES, name, EN_TD, species=MIXTURE, N_terms=n_terms,
                           remap_span=12, verbose=False)


xs = ar.load_all_cross_sections()
fig, axs = plt.subplots(1, len(LEVELS), figsize=(6 * len(LEVELS), 5))
colors = plt.cm.viridis(np.linspace(0, 0.85, len(N_TERMS)))
for ax, (sp, lbl, title) in zip(axs, LEVELS):
    #ax.semilogy(TE_MAXWELL, ar.rate_coefficient(xs[sp][lbl], Te=TE_MAXWELL), "k--", lw=2, label="Maxwellian")
    ax.set_title(title)
for n, col in zip(N_TERMS, colors):
    eedfs = multibolt_run(n)
    Te_eff = np.array([e["Te_eff"] for e in eedfs])
    for ax, (sp, lbl, _) in zip(axs, LEVELS):
        k = [ar.rate_coefficient(xs[sp][lbl], eedf=(e["E"], e["EEDF"])) for e in eedfs]
        ax.semilogy(Te_eff, k, "o-", color=col, ms=4, label=f"MultiBolt, {n} terms")
for ax in axs:
    ax.set_xlabel(r"$T_{e,\mathrm{eff}} = \frac{2}{3}\langle\varepsilon\rangle$ [eV]")
    ax.set_ylabel(r"ground-state excitation rate $k$ [m$^3$ s$^{-1}$]")
    ax.set_ylim(1e-24, None)
    ax.grid(alpha=0.3, which="both")
axs[0].legend(fontsize=8)
fig.suptitle(f"Maxwellian vs MultiBolt EEDFs ({', '.join(f'{100 * v:g} % {s}' for s, v in MIXTURE.items())}, "
             f"E/N = {EN_TD[0]}-{EN_TD[-1]} Td)")
fig.tight_layout()
fig.savefig(OUTPUT, dpi=150)
plt.show()
print("saved", OUTPUT)
