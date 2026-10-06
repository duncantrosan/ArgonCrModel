# -*- coding: utf-8 -*-
"""
SimpleActinometry.py  -  N / Ar actinometry on the N2 percent sweep (85 W, 1 Torr), kept simple.

  1. read_sweep        reads every echelle .spa file of the sweep
  2. integrate_line    area of one line: sum over a small window minus the flat background next to it
  3. plot              raw intensity ratios N / Ar against N2 percent
  4. line_info         one dict per line: label, wavelength, area, A, branching ratio, cross section
  5. actinometry       k ratio from an EEDF -> n_N / n_Ar and the N2 dissociation fraction, with two EEDFs:
                         a Maxwellian at TE_EV, and the BOLSIG+ microwave EEDF of the Ar/N2 mixture of each
                         percent at Te,eff = MW_TE_EFF (InputData/Bolsig/ArN2_<p>pct_bolsig_mw, read with
                         HelperFunctions.ImportBolsigLibrary)

Actinometry: a line emits  I = n_ground * ne * k * b  photons per volume and time
(k = excitation rate coefficient from the ground state, b = A_line / sum of A out of the upper level), so

    n_N / n_Ar = (I_N / I_Ar) * (k_Ar * b_Ar) / (k_N * b_N)

The spectra are calibrated in energy units, so each area is multiplied by its wavelength to get photons.
Run: python SmallAnalysisScripts/SimpleActinometry.py   (figures in Experimental_Data/Output/SimpleActinometry)
"""
import json
import os
import re
import sys
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "Scripts"))
import HelperFunctions as he                         # noqa: E402  (only to read the BOLSIG+ EEDFs)
SWEEP_DIR = os.path.join(ROOT, "Experimental_Data", "EchelleData", "PrelijminaryNitrogenAdmixtureMini",
                         "GsaSweep_85W_1_torr", "SPA_Files")
N_FILE = os.path.join(ROOT, "Experimental_Data", "EchelleData", "TestData", "AtomicNitrogen_LevelData.json")
AR_XSEC_FILE = os.path.join(ROOT, "InputData", "ArgonCrossSections_BSR.json")
AR_LEVEL_FILE = os.path.join(ROOT, "InputData", "ArgonLevelList.json")
AR_LINE_FILE = os.path.join(ROOT, "InputData", "ArgonReactionList.json")
OUTDIR = os.path.join(ROOT, "Experimental_Data", "Output", "SimpleActinometry")

TE_EV = 2.0          # Maxwellian electron temperature for the k ratio
MW_TE_EFF = 0.9      # eV, Te,eff = 2/3 <energy> of the microwave EEDF picked from each library
HALF_WIDTH = 0.10    # nm, integration window = line centre +- this
BG_WIDTH = 0.20      # nm, background taken from this much beyond the window on each side

# (label, species, air wavelength nm, upper level label in the data files)
# (742.36 nm, the third line of the N triplet, and Ar 811.53 nm lie in echelle order gaps)
N_LINES = [("N 744.2", "N", 744.229, "other21")]      # N I 3p 4S* -> 3s 4P
AR_LINES = [("Ar 750.4 (2p1)", "Ar", 750.387, "4p10"),
            ("Ar 751.5 (2p5)", "Ar", 751.465, "4p6"),
            ("Ar 801.5 (2p8)", "Ar", 801.479, "4p3")]


# ---- 1. read the data ---------------------------------------------------------------
def read_spectrum(path):
    """One .spa file -> (wavelength [nm], intensity).  Order gaps are stored as zeros."""
    data = np.loadtxt(path, skiprows=1)
    return data[:, 0], data[:, 1]


def percent_of(filename):
    """'Percent0_5__2.spa' -> 0.5 ;  'Percent0_1.spa' -> 0.0 ;  'Percent_1_0__1.spa' -> 1.0"""
    m = re.match(r"Percent_?(\d+)(?:_(\d+))?_+\d+\.spa$", os.path.basename(filename))
    return float(f"{m.group(1)}.{m.group(2) or 0}")


def read_sweep(folder=SWEEP_DIR):
    """List of (N2 percent, wavelength, intensity), one entry per spectrum."""
    return [(percent_of(f), *read_spectrum(f)) for f in sorted(glob(os.path.join(folder, "*.spa")))]


# ---- 2. integrate a line ------------------------------------------------------------
def integrate_line(wl, I, centre):
    """Area of the line at `centre`: integral over centre +- HALF_WIDTH minus the median
    background of the two strips just outside the window.  NaN if the window hits an order gap."""
    inside = np.abs(wl - centre) <= HALF_WIDTH
    beside = (np.abs(wl - centre) > HALF_WIDTH) & (np.abs(wl - centre) <= HALF_WIDTH + BG_WIDTH)
    if inside.sum() < 5 or np.any(I[inside] == 0):
        return np.nan
    background = np.median(I[beside])
    return np.trapezoid(I[inside] - background, wl[inside])


def integrate_sweep(sweep, lines):
    """{label: (percent array, area array)} for every line, one value per spectrum."""
    pct = np.array([p for p, _, _ in sweep])
    return {lab: (pct, np.array([integrate_line(wl, I, centre) for _, wl, I in sweep]))
            for lab, _, centre, _ in lines}


def mean_per_percent(pct, values):
    """Average the repeats: (unique percents, mean, standard deviation)."""
    ps = np.unique(pct)
    return ps, np.array([np.nanmean(values[pct == p]) for p in ps]), np.array([np.nanstd(values[pct == p]) for p in ps])


# ---- 3. plot the raw ratios ----------------------------------------------------------
def plot_raw_ratios(areas):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (nlab, *_), col in zip(N_LINES, ("C0", "C1", "C2")):
        for (alab, *_), mk in zip(AR_LINES, ("o", "s", "^")):
            pct, ratio = areas[nlab][0], areas[nlab][1] / areas[alab][1]
            ps, mean, sd = mean_per_percent(pct, ratio)
            ax.errorbar(ps, mean, yerr=sd, fmt=mk + "-", color=col, ms=5, capsize=2, label=f"{nlab} / {alab}")
    ax.set_xlabel("N$_2$ in the feed [%]")
    ax.set_ylabel("raw intensity ratio  I$_N$ / I$_{Ar}$")
    ax.set_title("Raw N / Ar line ratios, 85 W, 1 Torr")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "raw_ratios_vs_percent.png"), dpi=200)
    plt.close(fig)


# ---- 4. line data: A, branching ratio, cross section ----------------------------------
def line_info(lines, areas):
    """{label: dict(label, species, wl, area, A, b, xsec_E, xsec)} for every line.
    A  = Einstein coefficient of the line [1/s]; b = A / (sum of A out of the upper level);
    xsec_E, xsec = ground-state excitation cross section of the upper level [eV], [m^2]."""
    n_levels = json.load(open(N_FILE, encoding="utf-8"))
    ar_xsec = {c["upper_label"]: c for c in json.load(open(AR_XSEC_FILE))["cross_sections"]
               if c["type"] == "excitation" and c["lower_label"] == "ground"}
    ar_levels = json.load(open(AR_LEVEL_FILE))
    ar_lines = json.load(open(AR_LINE_FILE))["transitions"]
    info = {}
    for lab, species, wl, upper in lines:
        if species == "N":
            lev = n_levels[upper]
            A = next(t["Aki"] for t in lev["radiative_out"] if abs(t["wl_nm"] - wl) < 0.01)
            A_sum = lev["A_listed_sum_s-1"]
            x = lev["electron_impact_excitation"][0]
            E, sigma = x["energy_eV"], x["cross_section_m2"]
        else:
            lev = ar_levels[upper]
            same_upper = [t for t in ar_lines if t["upper"]["config"] == lev["configuration"]
                          and t["upper"]["term"] == lev["term"] and t["upper"]["J"] == lev["J"] and t["Aki"]]
            A = next(t["Aki"] for t in same_upper if abs(t["wl_nm"] - wl) < 0.01)
            A_sum = sum(t["Aki"] for t in same_upper)
            E, sigma = ar_xsec[upper]["energy_eV"], ar_xsec[upper]["cross_section"]
        info[lab] = dict(label=lab, species=species, wl=wl, area=areas[lab][1], A=A, b=A / A_sum,
                         xsec_E=np.asarray(E, float), xsec=np.asarray(sigma, float))
    return info


# ---- 5. actinometry -------------------------------------------------------------------
def maxwellian(Te):
    """Maxwellian EEDF at Te [eV] -> (energy grid [eV], f(E) [1/eV])."""
    E = np.linspace(0.0, 60 * Te, 20000)
    return E, 2 * np.sqrt(E / np.pi) * Te ** -1.5 * np.exp(-E / Te)


def microwave_eedf(percent, te_eff=MW_TE_EFF):
    """BOLSIG+ microwave EEDF of the Ar/N2 mixture at this N2 percent: the E/N row whose
    Te,eff is closest to te_eff -> (energy grid [eV], f(E) [1/eV], its Te,eff)."""
    lib = he.ImportBolsigLibrary(he.BOLSIG_FOLDER / f"ArN2_{percent:g}pct_bolsig_mw", verbose=False)
    eedf = min(lib["EEDFs"], key=lambda e: abs(e["Te_eff"] - te_eff))
    return eedf["E"], eedf["EEDF"], eedf["Te_eff"]


def rate_coefficient(E_xs, sigma, E, f):
    """k = integral of sigma(E) v(E) f(E) dE / integral of f(E) dE   [m^3/s]."""
    e, m_e = 1.602176634e-19, 9.1093837015e-31
    v = np.sqrt(2 * E * e / m_e)
    s = np.interp(E, E_xs, sigma, left=0.0, right=0.0)
    return np.trapezoid(s * v * f, E) / np.trapezoid(f, E)


def actinometry(info, pct, k):
    """n_N / n_Ar for every N / Ar line pair and the N2 dissociation fraction
    D = n_N / (2 n_N2), with n_N2 = (percent / (100 - percent)) n_Ar (feed composition).
    k = {line label: rate coefficient}, or {line label: array with one value per spectrum}."""
    x = pct / 100
    out = {}
    for nlab, *_ in N_LINES:
        for alab, *_ in AR_LINES:
            N, Ar = info[nlab], info[alab]
            photon_ratio = (N["area"] * N["wl"]) / (Ar["area"] * Ar["wl"])
            nN_nAr = photon_ratio * (k[alab] * Ar["b"]) / (k[nlab] * N["b"])
            with np.errstate(divide="ignore", invalid="ignore"):
                D = nN_nAr * (1 - x) / (2 * x)
            out[(nlab, alab)] = dict(nN_nAr=nN_nAr, D=D)
    return out


def plot_actinometry(pct, results):
    """results = {EEDF name: actinometry output}: one colour per EEDF, one marker per Ar line."""
    for key, ylab, fname in (("nN_nAr", "n$_N$ / n$_{Ar}$", "nN_over_nAr_vs_percent.png"),
                             ("D", "N$_2$ dissociation fraction  n$_N$ / (2 n$_{N_2}$)", "dissociation_vs_percent.png")):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        keep = pct > 0                                      # no N2 -> nothing to measure
        for (name, out), col in zip(results.items(), ("C0", "C3")):
            for ((nlab, alab), r), mk in zip(out.items(), ("o", "s", "^")):
                ps, mean, sd = mean_per_percent(pct[keep], r[key][keep])
                ax.errorbar(ps, mean, yerr=sd, fmt=mk + "-", color=col, ms=5, capsize=2,
                            label=f"{name}: {nlab} / {alab}")
        ax.set_xlabel("N$_2$ in the feed [%]")
        ax.set_ylabel(ylab)
        ax.set_yscale("log")
        ax.set_title("Actinometry, 85 W, 1 Torr")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, fname), dpi=200)
        plt.close(fig)


# ---- run ------------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    sweep = read_sweep()
    areas = integrate_sweep(sweep, N_LINES + AR_LINES)
    plot_raw_ratios(areas)
    info = line_info(N_LINES + AR_LINES, areas)
    pct = areas[N_LINES[0][0]][0]
    print(f"{len(sweep)} spectra, N2 percents {[float(p) for p in np.unique(pct)]}")

    # rate coefficients: one Maxwellian for all spectra; one microwave EEDF per N2 percent
    E, f = maxwellian(TE_EV)
    k_mx = {lab: rate_coefficient(d["xsec_E"], d["xsec"], E, f) for lab, d in info.items()}
    k_mw = {lab: np.full(len(pct), np.nan) for lab in info}
    print(f"\n{'line':<16s} {'A [1/s]':>10s} {'b':>6s} {'k Maxwell %g eV' % TE_EV:>17s}   k microwave per percent")
    for p in np.unique(pct[pct > 0]):
        E, f, te = microwave_eedf(p)
        print(f"  {p:g} % N2: microwave EEDF with Te,eff = {te:.2f} eV")
        for lab, d in info.items():
            k_mw[lab][pct == p] = rate_coefficient(d["xsec_E"], d["xsec"], E, f)
    percents = np.unique(pct[pct > 0])
    first = [int(np.argmax(pct == p)) for p in percents]      # one spectrum per percent
    for lab, d in info.items():
        print(f"{lab:<16s} {d['A']:10.3g} {d['b']:6.3f} {k_mx[lab]:17.3e}   "
              + " ".join(f"{v:.2e}" for v in k_mw[lab][first]))

    results = {f"Maxwell {TE_EV:g} eV": actinometry(info, pct, k_mx),
               f"microwave {MW_TE_EFF:g} eV": actinometry(info, pct, k_mw)}
    print("\nk_N / k_Ar (Maxwell | microwave at each percent):")
    for alab, *_ in AR_LINES:
        nlab = N_LINES[0][0]
        print(f"  {nlab} / {alab:<16s} {k_mx[nlab] / k_mx[alab]:8.3g} | "
              + " ".join(f"{v:.3g}" for v in (k_mw[nlab] / k_mw[alab])[first]))
    print("\ndissociation fraction, mean over repeats (rows: percent; columns: Ar line)")
    for name, out in results.items():
        print(f"  {name}")
        for p in np.unique(pct[pct > 0]):
            print(f"    {p:4.1f} %  " + "  ".join(f"{np.nanmean(r['D'][pct == p]):9.3g}" for r in out.values()))
    plot_actinometry(pct, results)
    print(f"\nfigures in {OUTDIR}")
