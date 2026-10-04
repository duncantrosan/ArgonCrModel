# -*- coding: utf-8 -*-
"""
EEDFComparisonArN2.py

Three EEDF comparisons, each for pure Ar and for Ar + 10 % N2 (Biagi cross sections,
1 Torr, 300 K - the conditions of Scripts/CreateEEDFLibraries.py):

  1. solver   MultiBolt with 2, 4, 6 and 8 Legendre terms vs BOLSIG+ (two-term)
  2. physics  Maxwell-Boltzmann vs BOLSIG+, BOLSIG+ with e-e collisions and BOLSIG+
              with superelastic collisions from the four 4s levels
  3. all in   BOLSIG+ DC vs 2.45 GHz microwave field, each plain and with
              e-e + superelastic collisions

The EEDFs are compared at the same mean energy, Te_eff = 2/3 <E> (the axis of the
CR fit), not at the same E/N: a microwave field reaches a given <E> at a much higher
E/N than DC, and N2 at a lower <E> than Ar. Each figure has
  row 1  EEPF at the Te_eff of each column (nearest point of each sweep; the
         Maxwellian is at the column's Te_eff)
  row 2  EEPF / Maxwellian of the same mean energy (1 = Maxwellian shape)
  row 3  CR-model rate coefficients / Maxwellian of the same mean energy, vs Te_eff
e-e and superelastic EEDFs depend on Ne; they are shown at NE_SHOW.

Missing libraries are built on the first run and reused afterwards (delete a folder
to rebuild it); the slow ones are the self-consistent superelastic libraries, so
the new E/N x Ne libraries use the short Ne grid NE_LIB (they are not on the Ne grid
the CR fit needs). New libraries:
  InputData/MultiBolt/Ar_Biagi_lowEN_span12_{4,8}term
  InputData/MultiBolt/ArN2_10pct_{2,4,6,8}terms
  InputData/Bolsig/Ar_Biagi_bolsig_mw_ee_se
  InputData/Bolsig/ArN2_10pct_bolsig{,_ee,_se,_ee_se,_mw,_mw_ee_se}

The superelastic 4s populations come from the pure-Ar CR model, also in the N2
mixture (no quenching of Ar(4s) by N2, Ar ground density not diluted), so there the
superelastic effect is an upper bound. Superelastic collisions with vibrationally
excited N2 are not included.

Figures in Experimental_Data/Output/EEDFComparisonArN2/.
Run from Spyder: edit the settings, press F5.
"""
import contextlib
import io
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "Scripts"))
sys.path.insert(0, os.path.join(ROOT_DIR, "Experimental_Data", "ExperimentalDataAnalysis"))
import CRFitNeTe as crf                             # noqa: E402
import CreateEEDFLibraries as cel                   # noqa: E402

he = crf._helpers()

# ---- settings -------------------------------------------------------------------
N_TERMS = [2, 4, 6, 8]
NE_SHOW = 1e17                               # m^-3, Ne of the e-e / superelastic EEDFs
NE_LIB = [cel.NE_GRID[np.argmin(np.abs(np.log(cel.NE_GRID / ne)))] for ne in (1e16, 1e17, 1e18)]
N_TE_SHOW = 3                                # columns (= len(RATE_GROUPS), row 3 uses the same axes),
                                             # placed where all sweeps have a point close by
TE_TOL = 0.15                                # eV, a larger Te_eff mismatch in rows 1-2 is printed
EN_ARN2 = [3, 4, 5, 6, 7, 8.5, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]   # Td, N2 cools the EEDF
MB_XSEC_DIR = os.path.dirname(cel.MULTIBOLT_XSEC)
MIXTURES = {
    "Ar": dict(title="Ar", species=["Ar"], fractions=[1.0], bolsig="Ar_Biagi_bolsig",
               multibolt={n: "Ar_Biagi_lowEN_span12" if n == 6 else f"Ar_Biagi_lowEN_span12_{n}term"
                          for n in N_TERMS},
               EN_multibolt=cel.EN_MULTIBOLT, EN_dc=cel.EN_BOLSIG),
    "ArN2": dict(title="Ar + 10 % N$_2$", species=["Ar", "N2"], fractions=[0.9, 0.1],
                 bolsig="ArN2_10pct_bolsig", multibolt={n: f"ArN2_10pct_{n}terms" for n in N_TERMS},
                 EN_multibolt=EN_ARN2, EN_dc=EN_ARN2),
}
# BOLSIG+ variants: suffix of the library name -> (e-e, superelastic, microwave)
BOLSIG_VARIANTS = {"": (False, False, False), "_ee": (True, False, False), "_se": (False, True, False),
                   "_ee_se": (True, True, False), "_mw": (False, False, True), "_mw_ee_se": (True, True, True)}
FIGURES = [   # file name, title, [(label, source, colour, line style)]
    ("1_solver", "MultiBolt (N Legendre terms) vs BOLSIG+ (two-term)",
     [(f"MultiBolt, {n} terms", f"mb{n}", c, "-")
      for n, c in zip(N_TERMS, plt.cm.Greys(np.linspace(0.4, 0.95, len(N_TERMS))))]
     + [("BOLSIG+", "", "C0", "--")]),
    ("2_physics", "Maxwell-Boltzmann vs BOLSIG+ with e-e and superelastic collisions",
     [("BOLSIG+", "", "C0", "--"), ("BOLSIG+ e-e", "_ee", "C2", "-"),
      ("BOLSIG+ superelastic", "_se", "C1", "-")]),
    ("3_all_physics", "BOLSIG+ DC vs 2.45 GHz microwave, with e-e + superelastic collisions",
     [("BOLSIG+ DC", "", "C0", "--"), ("BOLSIG+ DC, e-e + superelastic", "_ee_se", "C3", "-"),
      ("BOLSIG+ microwave", "_mw", "C4", "--"), ("BOLSIG+ microwave, e-e + superelastic", "_mw_ee_se", "C5", "-")]),
]
THRESHOLDS = [(11.55, "4s"), (13.08, "4p"), (14.46, "5p"), (15.76, "ion.")]
RATE_GROUPS = [("ground", "4p"), ("ground", "5p"), ("4s1", "4p")]
OUTDIR = os.path.join(ROOT_DIR, "Experimental_Data", "Output", "EEDFComparisonArN2")


# ---- libraries ----------------------------------------------------------------------
def xsec_files(mix):
    """(plain, superelastic) BOLSIG+ cross-section files of a mixture."""
    plain, se = cel.biagi_files()
    if mix["species"] == ["Ar"]:
        return plain, se
    plain = he.BOLSIG_XSEC_FOLDER / "Biagi_ArN2.txt"
    if not plain.is_file():
        parts = [open(os.path.join(MB_XSEC_DIR, f"Biagi_{s}.txt"), errors="replace").read().rstrip()
                 for s in mix["species"]]
        plain.write_text("\n\n".join(parts) + "\n")
    se = he.BOLSIG_XSEC_FOLDER / "Biagi_ArN2_superelastic.txt"
    if not se.is_file():
        he.MakeSuperelasticXsecFile(plain, se, {k: g for k, (_, g) in he.AR_PASCHEN_4S.items()})
    return plain, se


def build_missing(mix, need):
    """Build the libraries of mix that the figures need and that do not exist yet."""
    cr_solve = None
    for src in need:
        if src.startswith("mb"):
            n = int(src[2:])
            name = mix["multibolt"][n]
            if not cel.exists(os.path.join(he.MULTIBOLT_FOLDER, name)):
                print(f"\n== MultiBolt, {n} terms: {name}")
                he.RunMultiBolt([os.path.join(MB_XSEC_DIR, f"Biagi_{s}.txt") for s in mix["species"]], name,
                                mix["EN_multibolt"], species=dict(zip(mix["species"], mix["fractions"])),
                                N_terms=n, remap_span=12, verbose=False)
            continue
        name = mix["bolsig"] + src
        if cel.exists(os.path.join(he.BOLSIG_FOLDER, name)):
            continue
        ee, se, mw = BOLSIG_VARIANTS[src]
        plain, se_file = xsec_files(mix)
        extra = dict(omega_N=cel.OMEGA_N) if mw else {}
        EN = cel.EN_MICROWAVE if mw else mix["EN_dc"]
        print(f"\n== BOLSIG+: {name}")
        if not (ee or se):
            he.RunBolsig(plain, name, EN, species=mix["species"], fractions=mix["fractions"],
                         verbose=False, **cel.BOLSIG_SETTINGS, **extra)
            continue
        if se and cr_solve is None:
            cr_solve = crf.cr_density_solver(cel.CR_CFG)
        he.BuildBolsigLibrary(se_file if se else plain, name, EN, NE_LIB, P_Torr=cel.CR_CFG["P_Torr"],
                              Tg=cel.CR_CFG["Tg"], species=mix["species"], fractions=mix["fractions"],
                              electron_electron=ee, superelastic=cel.SUPERELASTIC if se else None,
                              cr_solve=cr_solve if se else None, n_iter=8, tol=0.03,
                              n_workers=cel.N_WORKERS, **cel.BOLSIG_SETTINGS, **extra)


def load(mix, src):
    """EEDFs of one source along its E/N sweep; E/N x Ne libraries: the Ne nearest NE_SHOW."""
    if src.startswith("mb"):
        return he.ImportMultiBoltEEDFs(he.MULTIBOLT_FOLDER / mix["multibolt"][int(src[2:])], verbose=False)
    lib = he.ImportBolsigLibrary(he.BOLSIG_FOLDER / (mix["bolsig"] + src), verbose=False)
    if lib["kind"] == "1D":
        return lib["EEDFs"]
    j = int(np.argmin(np.abs(np.log(lib["Ne"] / NE_SHOW))))
    return [row[j] for row in lib["EEDFs"]]


# ---- analysis -----------------------------------------------------------------------
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


def te_columns(curves):
    """N_TE_SHOW Te_eff values in the range every curve covers: the range is cut into
    N_TE_SHOW equal parts and each column goes to the sweep point in its part that all
    curves come closest to (the sweeps are on different E/N grids)."""
    te = [np.array([e["Te_eff"] for e in eedfs]) for eedfs in curves]
    lo, hi = max(t.min() for t in te), min(t.max() for t in te)
    if lo >= hi:
        raise ValueError(f"the sweeps share no Te_eff range ({lo:.2f} > {hi:.2f} eV) - extend the E/N lists")
    cand = np.unique(np.concatenate(te))
    cand = cand[(cand >= lo) & (cand <= hi)]
    miss = np.array([max(np.min(np.abs(t - c)) for t in te) for c in cand])
    edges = np.linspace(lo, hi, N_TE_SHOW + 1)
    cols = []
    for a, b in zip(edges[:-1], edges[1:]):
        inside = (cand >= a) & (cand <= b)
        c = cand[inside][np.argmin(miss[inside])] if inside.any() else 0.5 * (a + b)
        cols.append(round(float(c), 2))
    return np.array(cols)


def thresholds(ax):
    for e, name in THRESHOLDS:
        ax.axvline(e, color="0.75", lw=0.8, zorder=0)
        ax.text(e, 1.0, f" {name}", transform=ax.get_xaxis_transform(), va="top", ha="left",
                fontsize=8, color="0.45")


def figure(mix_key, mix, fname, title, curves, cr, MD):
    data = [(label, load(mix, src), col, ls) for label, src, col, ls in curves]
    te_show = te_columns([eedfs for _, eedfs, _, _ in data])
    fig, axs = plt.subplots(3, N_TE_SHOW, figsize=(4.8 * N_TE_SHOW, 12))
    E_mx = np.linspace(0, 25, 501)
    print(f"\n{mix_key}: {title}\n  {'':40s}" + "".join(f"{f'Te={t:g}':>10s}" for t in te_show))
    for label, eedfs, col, ls in data:
        te = np.array([e["Te_eff"] for e in eedfs])
        picks = [int(np.argmin(np.abs(te - t))) for t in te_show]
        print(f"  {label:40s}" + "".join(f"{eedfs[i]['sweep']['value']:7.3g} Td" for i in picks))
        for c, (t, i) in enumerate(zip(te_show, picks)):
            e = eedfs[i]
            if abs(e["Te_eff"] - t) > TE_TOL:
                print(f"    {label}: nearest Te_eff to {t} eV is {e['Te_eff']:.2f} eV (no closer sweep point)")
            ok = e["EEPF"] > 0
            mx = he.MaxwellianEEDF(e["Te_eff"], E=e["E"])["EEPF"]
            axs[0, c].semilogy(e["E"][ok], e["EEPF"][ok], color=col, ls=ls, lw=1.4,
                               label=f"{label} ({e['sweep']['value']:g} Td, {e['Te_eff']:.2f} eV)")
            axs[1, c].semilogy(e["E"][ok], e["EEPF"][ok] / mx[ok], color=col, ls=ls, lw=1.4)
        k = rate_table(eedfs, cr, MD)
        k_mx = rate_table([he.MaxwellianEEDF(e["Te_eff"]) for e in eedfs], cr, MD)
        for g in range(len(RATE_GROUPS)):
            r = np.where((k[:, g] > 0) & (k_mx[:, g] > 0), k[:, g] / k_mx[:, g], np.nan)
            axs[2, g].semilogy(te, r, marker="o", ms=3, color=col, ls=ls, lw=1.4, label=label)
    for c, t in enumerate(te_show):
        mx = he.MaxwellianEEDF(t, E=E_mx)
        axs[0, c].semilogy(mx["E"], mx["EEPF"], ":", color="m", lw=1.6, label=f"Maxwellian, {t:g} eV")
        axs[1, c].axhline(1, color="m", ls=":", lw=1.6)
        axs[0, c].set_ylim(1e-22, 1)
        axs[1, c].set_ylim(1e-8, 1e4)
        for a in axs[:2, c]:
            a.set_xlim(0, 25)
            thresholds(a)
            a.set_xlabel("Electron energy (eV)")
            a.grid(alpha=0.25)
        axs[0, c].set_title(rf"$T_{{e,\mathrm{{eff}}}}$ = {t:g} eV", fontsize=11)
        axs[0, c].legend(fontsize=6.5, loc="lower left")
    for g, (lower, upper) in enumerate(RATE_GROUPS):
        a = axs[2, g]
        a.axhline(1, color="m", ls=":", lw=1.6, label="Maxwellian")
        a.set_title(f"{lower} $\\rightarrow$ {upper} (sum)", fontsize=11)
        a.set_xlabel(r"$T_{e,\mathrm{eff}} = \frac{2}{3}\langle\varepsilon\rangle$ (eV)")
        a.grid(alpha=0.25, which="both")
    axs[0, 0].set_ylabel(r"EEPF  $f_0$ (eV$^{-3/2}$)")
    axs[1, 0].set_ylabel("EEPF / Maxwellian of the same mean energy")
    axs[2, 0].set_ylabel("Rate coefficient / Maxwellian of the same mean energy")
    axs[2, 0].legend(fontsize=7, loc="best")
    fig.suptitle(f"{mix['title']}, 1 Torr, 300 K: {title}"
                 + (f"  (e-e / superelastic at Ne = {NE_SHOW:.0e} m$^{{-3}}$)" if fname != "1_solver" else ""),
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"{mix_key}_{fname}.png"), dpi=170)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    np.seterr(divide="ignore", invalid="ignore")      # rates are 0 below threshold at low Te
    need = sorted({src for _, _, curves in FIGURES for _, src, _, _ in curves})
    for mix in MIXTURES.values():
        build_missing(mix, need)
    cr, _, MD, _ = crf.load_cr_model(OUTDIR)
    for key, mix in MIXTURES.items():
        for fname, title, curves in FIGURES:
            figure(key, mix, fname, title, curves, cr, MD)
    print(f"\nfigures in {OUTDIR}")
    plt.show()


if __name__ == "__main__":
    main()
