# -*- coding: utf-8 -*-
"""
EEDFvsEN.py

Why the fits report E/N and not Te: the pure-Ar 2.42 GHz BOLSIG+ EEPFs (1 Torr, 650 K, the library of the N2-sweep
fits) for E/N = 30-60 Td, with Te_eff = 2/3 <eps> of each.  Te_eff barely moves while the tail above the Ar
excitation thresholds changes by orders of magnitude; a Maxwellian of the same Te_eff is shown for comparison.

Output: Experimental_Data/Output/EEDFvsEN/eedf_vs_EN.png
Run from Spyder (F5) or python SmallAnalysisScripts/EEDFvsEN.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "Scripts"))
import HelperFunctions as he                        # noqa: E402

LIBRARY = "Ar_Biagi_bolsig_mw2.42GHz_650K_1000mTorr"
EN_RANGE = (30.0, 60.0)                             # Td
THRESHOLDS = {"1s (4s)": 11.55, "4p": 12.9, "5p": 14.5, "Ar$^+$": 15.76}   # eV
OUTDIR = os.path.join(ROOT, "Experimental_Data", "Output", "EEDFvsEN")


def te_eff(E, f):
    """2/3 <eps> of an EEPF f (eV^-3/2): <eps> = int eps^1.5 f / int eps^0.5 f."""
    return 2 / 3 * np.trapz(E ** 1.5 * f, E) / np.trapz(E ** 0.5 * f, E)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    lib = he.ImportBolsigLibrary(he.BOLSIG_FOLDER / LIBRARY, verbose=False)
    rows = [(en, e) for en, e in zip(lib["EN_Td"], lib["EEDFs"]) if EN_RANGE[0] <= en <= EN_RANGE[1]]
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(rows)))
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    ref = None
    for c, (en, e) in zip(cols, rows):
        E, f = np.asarray(e["E"], float), np.asarray(e["EEPF"], float)
        ok = (E > 0) & (f > 0)
        E, f = E[ok], f[ok] / np.trapz(E[ok] ** 0.5 * f[ok], E[ok])          # normalised: int eps^0.5 f = 1
        T = te_eff(E, f)
        axs[0].plot(E, f, "-", color=c, lw=2, label=f"{en:g} Td: $T_{{e,\\mathrm{{eff}}}}$ = {T:.3f} eV")
        if ref is None:
            ref = (en, E, f, T)
        m = E <= ref[1].max()                         # ratio only where the reference EEPF is defined
        axs[1].plot(E[m], f[m] / np.interp(E[m], ref[1], ref[2]), "-", color=c, lw=2, label=f"{en:g} Td ({T:.3f} eV)")
    en0, E0, f0, T0 = ref
    mx = 2 / np.sqrt(np.pi) * T0 ** -1.5 * np.exp(-E0 / T0)                   # Maxwellian EEPF, same normalisation
    axs[0].plot(E0, mx, "k--", lw=1.5, label=f"Maxwellian, T$_e$ = {T0:.3f} eV")
    for ax in axs:
        for lab, th in THRESHOLDS.items():
            ax.axvline(th, color="0.5", ls=":", lw=1)
            ax.text(th, 0.98, f" {lab}", transform=ax.get_xaxis_transform(), rotation=90, va="top", fontsize=8,
                    color="0.35")
        ax.set(xlim=(0, 22), yscale="log", xlabel="electron energy [eV]")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")
    axs[0].set(ylim=(1e-14, 2), ylabel="EEPF [eV$^{-3/2}$]",
               title="EEPF of the pure-Ar 2.42 GHz BOLSIG+ library (1 Torr, 650 K)")
    axs[1].set(ylim=(0.3, 1e5), ylabel=f"EEPF / EEPF({en0:g} Td)",
               title="relative to the lowest E/N: bulk unchanged, tail up by orders of magnitude")
    fig.suptitle(f"E/N {EN_RANGE[0]:g}-{EN_RANGE[1]:g} Td: $T_{{e,\\mathrm{{eff}}}} = \\frac{{2}}{{3}}\\langle\\varepsilon\\rangle$ "
                 "changes by a few %, the excitation-relevant tail by 10-1000x", fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUTDIR, "eedf_vs_EN.png")
    fig.savefig(path, dpi=200)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
