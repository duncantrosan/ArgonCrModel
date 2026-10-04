# -*- coding: utf-8 -*-
"""
EEDFSolverComparison.py

How the EEDF - and the electron-impact rates the CR model takes from it - changes
with the Boltzmann solver and with the physics in it. All for the Biagi Ar cross
sections at 1 Torr, 300 K (libraries: Scripts/CreateEEDFLibraries.py):

  MultiBolt, 6 terms          InputData/MultiBolt/Ar_Biagi_lowEN_span12
  MultiBolt, 2 terms          InputData/MultiBolt/Ar_Biagi_lowEN_span12_2term
  BOLSIG+                     InputData/Bolsig/Ar_Biagi_bolsig
  BOLSIG+ e-e                 InputData/Bolsig/Ar_Biagi_bolsig_ee      } at Ne = NE_SHOW;
  BOLSIG+ superelastic        InputData/Bolsig/Ar_Biagi_bolsig_se      } 4s populations from
  BOLSIG+ e-e + superelastic  InputData/Bolsig/Ar_Biagi_bolsig_ee_se   } the CR model
  Maxwell-Boltzmann           same mean energy as each EEDF

Figures in Experimental_Data/Output/EEDFSolvers/:
  eepf_vs_energy.png     EEPF at EN_SHOW, with the 4s / 4p / 5p / ionization thresholds
  eepf_over_maxwell.png  each EEPF divided by the Maxwellian of the same mean energy
  rates_vs_EN.png        CR-model rate coefficients (ground -> 4s, 4p, 5p; 4s1 -> 4p, 5p)
                         vs E/N, and relative to plain BOLSIG+
  ne_dependence.png      BOLSIG+ e-e + superelastic EEPF at EN_SCAN for several Ne

Run from Spyder: edit the settings, press F5.
"""
import contextlib
import io
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "Experimental_Data", "ExperimentalDataAnalysis"))
import CRFitNeTe as crf                             # noqa: E402

he = crf._helpers()

# ---- settings -------------------------------------------------------------------
EN_SHOW = [2.5, 3.5, 10, 50]                 # Td, panels of the EEPF figures
NE_SHOW = 1e17                               # m^-3, column of the E/N x Ne libraries
EN_SCAN, NE_SCAN = 3.5, [1e15, 1e16, 1e17, 1e18, 3e18]
SOURCES = [   # label, folder, colour, line style
    ("MultiBolt, 6 terms", he.MULTIBOLT_FOLDER / "Ar_Biagi_lowEN_span12", "0.6", "-"),
    ("MultiBolt, 2 terms", he.MULTIBOLT_FOLDER / "Ar_Biagi_lowEN_span12_2term", "k", "-"),
    ("BOLSIG+", he.BOLSIG_FOLDER / "Ar_Biagi_bolsig", "C0", "--"),
    ("BOLSIG+ e-e", he.BOLSIG_FOLDER / "Ar_Biagi_bolsig_ee", "C2", "-"),
    ("BOLSIG+ superelastic", he.BOLSIG_FOLDER / "Ar_Biagi_bolsig_se", "C1", "-"),
    ("BOLSIG+ e-e + superelastic", he.BOLSIG_FOLDER / "Ar_Biagi_bolsig_ee_se", "C3", "-"),
]
THRESHOLDS = [(11.55, "4s"), (13.08, "4p"), (14.46, "5p"), (15.76, "ion.")]
RATE_GROUPS = [("ground", "4s"), ("ground", "4p"), ("ground", "5p"), ("4s1", "4p"), ("4s1", "5p")]
OUTDIR = os.path.join(ROOT_DIR, "Experimental_Data", "Output", "EEDFSolvers")


def ready(folder):
    """A finished EEDF library: BOLSIG+ (library.json) or MultiBolt (EEDFs_f0/)."""
    return he.IsBolsigLibrary(folder) or os.path.isdir(os.path.join(folder, "EEDFs_f0"))


def load(folder, ne=NE_SHOW):
    """(E/N [Td], EEDFs along E/N, Ne used); E/N x Ne libraries: the column nearest ne."""
    folder = str(folder)
    if he.IsBolsigLibrary(folder):
        lib = he.ImportBolsigLibrary(folder, verbose=False)
        if lib["kind"] == "2D":
            j = int(np.argmin(np.abs(np.log(lib["Ne"] / ne))))
            return lib["EN_Td"], [row[j] for row in lib["EEDFs"]], lib["Ne"][j]
        return lib["EN_Td"], lib["EEDFs"], None
    E = he.ImportMultiBoltEEDFs(folder, verbose=False)
    return np.array([e["sweep"]["value"] for e in E]), E, None


def at(EN, eedfs, en):
    i = int(np.argmin(np.abs(EN - en)))
    return eedfs[i] if np.isclose(EN[i], en, rtol=1e-3) else None


def maxwell_like(eedf):
    """Maxwellian with the mean energy of eedf, on eedf's energy grid."""
    return he.MaxwellianEEDF(eedf["Te_eff"], E=eedf["E"])


def thresholds(ax, labels=True):
    for e, name in THRESHOLDS:
        ax.axvline(e, color="0.75", lw=0.8, zorder=0)
        if labels:
            ax.text(e, 1.0, f" {name}", transform=ax.get_xaxis_transform(), va="top",
                    ha="left", fontsize=8, color="0.45")


def rate_table(eedfs, cr, MD):
    """Summed rate coefficients [m^3/s] of RATE_GROUPS for each EEDF (the CR model's rates)."""
    out = np.zeros((len(eedfs), len(RATE_GROUPS)))
    for i, eedf in enumerate(eedfs):
        with contextlib.redirect_stdout(io.StringIO()):
            md = cr["CreateExcitationReactionRates"](MD, he.AsEEDF(eedf))
        for k, (lower, upper) in enumerate(RATE_GROUPS):
            out[i, k] = sum(r["Rate"] for lbl, s in md.items() if lbl.startswith(upper)
                            for r in s["Electron Impact CrossSections"]["Products"]
                            if r["partner_label"] == lower)
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    data = []
    for label, folder, col, ls in SOURCES:
        if not ready(folder):
            print(f"missing {folder} - run Scripts/CreateEEDFLibraries.py")
            continue
        EN, eedfs, ne = load(folder)
        data.append((label + (f" (Ne = {ne:.1e} m$^{{-3}}$)" if ne else ""), EN, eedfs, col, ls))

    # --- EEPF and EEPF / Maxwellian at EN_SHOW ---------------------------------------
    figs = {k: plt.subplots(1, len(EN_SHOW), figsize=(4.6 * len(EN_SHOW), 4.6), sharey=(k == "ratio"))
            for k in ("eepf", "ratio")}
    print(f"{'source':48s}" + "".join(f"{f'Te_eff@{en:g}Td':>13s}" for en in EN_SHOW))
    for label, EN, eedfs, col, ls in data:
        print(f"{label:48s}" + "".join(f"{e['Te_eff']:13.3f}" if e else f"{'-':>13s}"
                                       for e in (at(EN, eedfs, en) for en in EN_SHOW)))
        for k, en in enumerate(EN_SHOW):
            e = at(EN, eedfs, en)
            if e is None:
                continue
            ok = e["EEPF"] > 0
            figs["eepf"][1][k].semilogy(e["E"][ok], e["EEPF"][ok], color=col, ls=ls, lw=1.4, label=label)
            mx = maxwell_like(e)["EEPF"]
            figs["ratio"][1][k].semilogy(e["E"][ok], e["EEPF"][ok] / mx[ok], color=col, ls=ls, lw=1.4,
                                         label=label)
    plain = next((d for d in data if d[0] == "BOLSIG+"), None)       # reference for the Maxwellian
    for k, en in enumerate(EN_SHOW):
        ref = at(plain[1], plain[2], en) if plain else None
        ax = figs["eepf"][1][k]
        if ref is not None:
            mx = he.MaxwellianEEDF(ref["Te_eff"], E=np.linspace(0, 25, 501))
            ax.semilogy(mx["E"], mx["EEPF"], ":", color="m", lw=1.4,
                        label=f"Maxwellian, Te = {ref['Te_eff']:.2f} eV (BOLSIG+ <E>)")
        ax.set_ylim(1e-22, 1)
        figs["ratio"][1][k].axhline(1, color="m", ls=":", lw=1.4)
        figs["ratio"][1][k].set_ylim(1e-12, 1e6)
        for kind in ("eepf", "ratio"):
            a = figs[kind][1][k]
            a.set_xlim(0, 25)
            thresholds(a)
            a.set_xlabel("Electron energy (eV)")
            a.set_title(f"E/N = {en:g} Td", fontsize=11)
            a.grid(alpha=0.25)
    figs["eepf"][1][0].set_ylabel(r"EEPF  $f_0$ (eV$^{-3/2}$)")
    figs["ratio"][1][0].set_ylabel("EEPF / Maxwellian of the same mean energy")
    for kind, name in (("eepf", "eepf_vs_energy.png"), ("ratio", "eepf_over_maxwell.png")):
        fig, axs = figs[kind]
        axs[0].legend(fontsize=7, loc="lower left")
        fig.suptitle("Biagi Ar, 1 Torr, 300 K: " + ("EEPF" if kind == "eepf" else
                     "shape of the EEPF relative to a Maxwellian"), fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, name), dpi=170)

    # --- CR-model rate coefficients vs E/N ---------------------------------------------
    np.seterr(divide="ignore", invalid="ignore")      # rates are 0 below threshold at low E/N
    cr, _, MD, _ = crf.load_cr_model(OUTDIR)
    rates = {label: (EN, rate_table(eedfs, cr, MD)) for label, EN, eedfs, *_ in data}
    EN_ref, k_ref = rates["BOLSIG+"]
    mx_rates = rate_table([maxwell_like(e) for e in plain[2]], cr, MD)
    fig, axs = plt.subplots(2, len(RATE_GROUPS), figsize=(4.2 * len(RATE_GROUPS), 7.4), sharex=True)
    for label, EN, eedfs, col, ls in data:
        EN_, k = rates[label]
        for g in range(len(RATE_GROUPS)):
            axs[0, g].loglog(EN_, np.where(k[:, g] > 0, k[:, g], np.nan), marker="o", ms=3, color=col, ls=ls, label=label)
            kr = np.interp(EN_, EN_ref, k_ref[:, g])
            axs[1, g].loglog(EN_, np.where((k[:, g] > 0) & (kr > 0), k[:, g] / kr, np.nan), marker="o", ms=3,
                             color=col, ls=ls)
    for g, (lower, upper) in enumerate(RATE_GROUPS):
        axs[0, g].loglog(EN_ref, np.where(mx_rates[:, g] > 0, mx_rates[:, g], np.nan), ":", color="m", lw=1.6,
                         label="Maxwellian, same <E> as BOLSIG+")
        axs[1, g].loglog(EN_ref, mx_rates[:, g] / k_ref[:, g], ":", color="m", lw=1.6)
        axs[0, g].set_title(f"{lower} $\\rightarrow$ {upper} (sum)", fontsize=11)
        axs[1, g].axhline(1, color="0.5", lw=0.8)
        axs[1, g].set_xlabel("E/N (Td)")
        for a in axs[:, g]:
            a.grid(alpha=0.25, which="both")
    axs[0, 0].set_ylabel("Rate coefficient (m$^3$/s)")
    axs[1, 0].set_ylabel("Relative to plain BOLSIG+")
    axs[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle("CR-model electron-impact rate coefficients for each EEDF", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "rates_vs_EN.png"), dpi=170)

    # --- Ne dependence of the e-e + superelastic EEDF ----------------------------------
    folder = str(he.BOLSIG_FOLDER / "Ar_Biagi_bolsig_ee_se")
    if he.IsBolsigLibrary(folder):
        lib = he.ImportBolsigLibrary(folder, verbose=False)
        i = int(np.argmin(np.abs(lib["EN_Td"] - EN_SCAN)))
        fig, ax = plt.subplots(figsize=(7.5, 5))
        cols = plt.cm.viridis(np.linspace(0, 0.9, len(NE_SCAN)))
        base = at(plain[1], plain[2], lib["EN_Td"][i]) if plain else None
        if base is not None:
            ax.semilogy(base["E"], base["EEPF"], "k--", lw=1.4, label="BOLSIG+ (no e-e, no superelastics)")
        frac = np.array(lib["meta"]["fractions"])
        for ne, col in zip(NE_SCAN, cols):
            j = int(np.argmin(np.abs(np.log(lib["Ne"] / ne))))
            e = lib["EEDFs"][i][j]
            ok = e["EEPF"] > 0
            ax.semilogy(e["E"][ok], e["EEPF"][ok], color=col, lw=1.5,
                        label=f"Ne = {lib['Ne'][j]:.1e} m$^{{-3}}$, n(4s)/N = {frac[i, j].sum():.1e}, "
                              f"Te_eff = {e['Te_eff']:.2f} eV")
        thresholds(ax)
        ax.set_xlim(0, 25)
        ax.set_ylim(1e-22, 1)
        ax.set_xlabel("Electron energy (eV)")
        ax.set_ylabel(r"EEPF  $f_0$ (eV$^{-3/2}$)")
        ax.set_title(f"BOLSIG+ e-e + superelastic, E/N = {lib['EN_Td'][i]:g} Td", fontsize=11)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, "ne_dependence.png"), dpi=170)
    print(f"\nfigures in {OUTDIR}")
    plt.show()


if __name__ == "__main__":
    main()
