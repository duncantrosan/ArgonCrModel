# -*- coding: utf-8 -*-
"""
AnalyticFivePFit.py

CR fit of the N2 percent sweep (85 W, 1 Torr, 0-3 % N2; microwave EEDF, Ne = n_c, LAS Tg as LASTgComparison) with
the 29 NGFSRDW fallback channels (ground -> 5p, 4s -> 5p; distorted wave, tabulated only from 20 eV / 5-6 eV) left
out of the cross-section set, so that the model fills them with its analytic Drawin / Bogaerts cross sections
(HelperFunctions.AddAnalyticExcitationCrossSections), as it does for every other pair without data.

1. InputData/ArgonCrossSections_BSR_analytic5p.json = ArgonCrossSections_BSR.json without the NGFSRDW entries.
2. Cross sections of the replaced channels, NGFSRDW against analytic (peak sigma, rate coefficient for the EEDFs of
   the fits at n_c).
3. CR tables of the N2 sweep conditions with that set (Output/AnalyticFivePFit/tables, N_WORKERS at a time), fit at
   n_c without the response tilt (disabled in CRFitNeTe since 2026-10-07; the first run also had tilt variants)
   against the BSR + NGFSRDW tables of LASTgComparison; CR 1s5 against the LAS 1s5; model 5p/4p change against the measured one (FivePLinesVsN2).

Output in Experimental_Data/Output/AnalyticFivePFit/: xsec_comparison.csv, fits.csv, figures/*.png
Run from Spyder (F5) or python SmallAnalysisScripts/AnalyticFivePFit.py
"""
# The pipeline half of this script (table settings per EEDF option, building the CR tables) is
# pipeline/CRTableBuilder.py; this file keeps the analytic-5p cross-section comparison and its fits.  Main.py does not
# use this file.
import contextlib
import io
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pipeline"))
import CRTableBuilder as P                          # noqa: E402 - the pipeline half
from CRTableBuilder import *                        # noqa: E402,F401,F403
import LASTgComparison as L                         # noqa: E402

crf, he = L.crf, L.he

BASE_XSEC = "ArgonCrossSections_BSR.json"
FIGDIR = os.path.join(OUTDIR, "figures")
SWEEP = "N2 fraction"
FIT_5P = (415.859, 430.0101)                        # the two 5p features of the 9-feature fit


def make_xsec():
    src = os.path.join(crf.ROOT_DIR, "InputData", BASE_XSEC)
    dst = os.path.join(crf.ROOT_DIR, "InputData", XSEC)
    d = json.load(open(src))
    drop = [c for c in d["cross_sections"] if c.get("source") == "NGFSRDW"]
    d["cross_sections"] = [c for c in d["cross_sections"] if c.get("source") != "NGFSRDW"]
    d["description"] = (f"{BASE_XSEC} without the {len(drop)} NGFSRDW fallback channels (ground -> 5p, 4s -> 5p); "
                        "the CR model fills them with its analytic cross sections. SmallAnalysisScripts/AnalyticFivePFit.py")
    with open(dst, "w") as f:
        json.dump(d, f, indent=1)
    return drop


def conditions():
    comp = pd.read_csv(os.path.join(L.OUTDIR, "comparison.csv"))
    return comp[comp.sweep == SWEEP].sort_values("x").reset_index(drop=True)


# ---- cross sections ---------------------------------------------------------------------
def channels(MD, pairs):
    """{(lower, upper): (E, sigma, source)} of the excitation channels pairs in the model data MD."""
    out = {}
    for lo, up in pairs:
        src = MD[lo] if lo != "ground" else next(v for v in MD.values() if v["kind"] == "ground")
        for r in src["Electron Impact CrossSections"]["Reactants"]:
            if r["partner_label"] == up:
                out[(lo, up)] = (np.asarray(r["energy_eV"], float), np.asarray(r["cross_section"], float),
                                 r.get("source", "?"))
    return out


def rate(E, s, eedf):
    """Rate coefficient [m^3/s] of sigma(E) for an EEDF (E, EEDF normalised to 1 eV^-1)."""
    e, f = np.asarray(eedf["E"], float), np.asarray(eedf["EEDF"], float)
    v = np.sqrt(2 * e * 1.602176634e-19 / 9.1093837e-31)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(np.interp(e, E, s, left=0, right=0) * v * f, e) / trap(f, e))


def compare_xsec(drop, cond):
    pairs = [(c["lower_label"], c["upper_label"]) for c in drop]
    with contextlib.redirect_stdout(io.StringIO()):
        with xsec_set(BASE_XSEC):
            MD0, _ = he.GetData()
        with xsec_set(XSEC):
            MD1, _ = he.GetData()
    old, new = channels(MD0, pairs), channels(MD1, pairs)
    eedfs = {}
    for c in cond[cond.x.isin([0.0, 1.0])].itertuples():
        with contextlib.redirect_stdout(io.StringIO()):
            e, x, _, _ = crf.eedf_axis(table_cfg(L._key(c.N2, c.p_mTorr, c.Tg)))
        eedfs[c.x] = e[int(np.argmin(np.abs(np.log(x / c.EN_nc))))]
    rows = []
    for p in pairs:
        r = dict(lower=p[0], upper=p[1])
        for tag, ch in (("ngfsrdw", old), ("analytic", new)):
            if p not in ch:
                continue
            E, s, src = ch[p]
            r.update({f"{tag}_source": src, f"{tag}_E0": float(E[np.argmax(s > 0)]), f"{tag}_peak": float(s.max()),
                      f"{tag}_E_peak": float(E[np.argmax(s)])})
            for x, e in eedfs.items():
                r[f"{tag}_k_{x:g}pct"] = rate(E, s, e)
        rows.append(r)
    t = pd.DataFrame(rows)
    for x in eedfs:
        t[f"k_ratio_{x:g}pct"] = t[f"analytic_k_{x:g}pct"] / t[f"ngfsrdw_k_{x:g}pct"]
    return t, old, new


def plot_xsec(old, new, outdir):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, lo in zip(axs, ("ground", "4s1")):
        ks = [p for p in old if p[0] == lo]
        cols = plt.cm.tab10(np.arange(len(ks)) % 10)
        for p, c in zip(ks, cols):
            ax.plot(*old[p][:2], "-", color=c, lw=1.5, label=f"{p[1]}")
            if p in new:
                ax.plot(*new[p][:2], "--", color=c, lw=1.2)
        ax.set_yscale("log")
        ax.set_xlim(0, 60 if lo == "ground" else 25)
        ax.set_ylim(1e-24, 1e-18)
        ax.set_xlabel("Electron energy [eV]")
        ax.set_title(f"{lo} $\\rightarrow$ 5p (solid NGFSRDW, dashed analytic)", fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    axs[0].set_ylabel("cross section [m$^2$]")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "xsec_5p_ngfsrdw_vs_analytic.png"), dpi=200)


# ---- fits --------------------------------------------------------------------------------
def fit_all(cond):
    rows = []
    for xs, eedf, mk in ((BASE_XSEC, "mix", lambda k: L.table_cfg(k, "n2")), (XSEC, "mix", table_cfg),
                         (XSEC, "ar", lambda k: table_cfg(k, "ar"))):
        for deg in (None,):                         # response tilt disabled (CRFitNeTe.no_tilt)
            with xsec_set(xs), contextlib.redirect_stdout(io.StringIO()):
                first = cond.iloc[0]
                feats, ft = L.measurements(dict(mk(L._key(first.N2, first.p_mTorr, first.Tg)), response_deg=deg))
                for c in cond.itertuples():
                    cfg = dict(mk(L._key(c.N2, c.p_mTorr, c.Tg)), response_deg=deg)
                    f = L.fit_condition(cfg, c.sweep, c.x, feats, ft)
                    tab = crf.build_model_table(cfg)
                    fX, fN = crf.fine_grid(tab, cfg)
                    i = int(np.argmin(np.abs(fX - np.log(f["EN_nc"]))))
                    j = int(np.argmin(np.abs(fN - np.log(N_C))))
                    ln = lambda k: crf.on_fine_grid(tab, np.log(np.clip(tab["I_obs"][:, :, k], 1e-300, None)), cfg, k=1)[i, j]
                    k5 = [int(np.argmin(np.abs(tab["line_wl"] - w))) for w in FIT_5P]
                    k4 = [int(np.argmin(np.abs(tab["line_wl"] - w))) for w in (706.7218, 727.2936, 750.3869, 751.4652,
                                                                              794.8176, 800.6157, 801.4786)]
                    rows.append(dict(xsec=xs, eedf=eedf, tilt=deg is not None, x=c.x, EN=f["EN_nc"], Te_eff=f["Te_nc"],
                                     chi2_red=f["chi2_red_nc"], dchi2=f["dchi2_nc"], Ne_free=f["Ne_free"],
                                     chi2_red_free=f["chi2_red_free"], n_1s5=f["n_1s5"], n_1s5_lo=f["n_1s5_lo"],
                                     n_1s5_hi=f["n_1s5_hi"], las_1s5=c.las_1s5,
                                     ln_5p_4p=np.mean([ln(k) for k in k5]) - np.mean([ln(k) for k in k4])))
            print(f"fitted {xs}, {eedf} EEDF, tilt {deg}", flush=True)
    d = pd.DataFrame(rows)
    d["ratio_cr_las"] = d.n_1s5 / d.las_1s5
    return d


def plot_fits(d, outdir):
    style = {(BASE_XSEC, "mix", True): ("C3", "s", "-", "NGFSRDW 5p, Ar/N$_2$ EEDF, with tilt"),
             (BASE_XSEC, "mix", False): ("C3", "s", ":", "NGFSRDW 5p, Ar/N$_2$ EEDF, no tilt"),
             (XSEC, "mix", True): ("C0", "o", "-", "analytic 5p, Ar/N$_2$ EEDF, with tilt"),
             (XSEC, "mix", False): ("C0", "o", ":", "analytic 5p, Ar/N$_2$ EEDF, no tilt"),
             (XSEC, "ar", True): ("C2", "^", "-", "analytic 5p, pure-Ar EEDF, with tilt"),
             (XSEC, "ar", False): ("C2", "^", ":", "analytic 5p, pure-Ar EEDF, no tilt")}
    las = d.drop_duplicates("x").sort_values("x")
    for col, ylab, fname, logy in (("n_1s5", "Ar(1s$_5$) density [m$^{-3}$]", "1s5_vs_N2.png", True),
                                   ("chi2_red", "$\\chi^2$/dof at n$_c$", "chi2_vs_N2.png", False),
                                   ("EN", "E/N of the fit at n$_c$ [Td]", "EN_vs_N2.png", True)):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for (xs, eedf, tilt), g in d.groupby(["xsec", "eedf", "tilt"]):
            c, mk, ls, lab = style[(xs, eedf, tilt)]
            g = g.sort_values("x")
            if col == "n_1s5":
                ax.fill_between(g.x, g.n_1s5_lo, g.n_1s5_hi, color=c, alpha=0.08, lw=0)
            ax.plot(g.x, g[col], mk + ls, color=c, mfc="white" if not tilt else c, ms=6, lw=1.6, label=lab)
        if col == "n_1s5":
            ax.plot(las.x, las.las_1s5, "D", color="k", ms=7, label="LAS 1s$_5$")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("N$_2$ in the feed [%]")
        ax.set_ylabel(ylab)
        ax.set_title("CR fit at N$_e$ = n$_c$ (microwave EEDF, LAS T$_g$), 85 W, 1 Torr:\n"
                     "NGFSRDW vs analytic ground/4s $\\rightarrow$ 5p cross sections", fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=200)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    drop = make_xsec()
    print(f"{XSEC}: {len(drop)} NGFSRDW channels left out", flush=True)
    cond = conditions()
    keys = sorted({L._key(c.N2, c.p_mTorr, c.Tg) for c in cond.itertuples()})
    build_all([(k, e) for e in ("mix", "ar") for k in keys])
    xs, old, new = compare_xsec(drop, cond)
    xs.to_csv(os.path.join(OUTDIR, "xsec_comparison.csv"), index=False)
    plot_xsec(old, new, FIGDIR)
    d = fit_all(cond)
    d.to_csv(os.path.join(OUTDIR, "fits.csv"), index=False)
    plot_fits(d, FIGDIR)
    pd.set_option("display.width", 250)
    print("\ncross sections (rate coefficients for the EEDFs of the fits at n_c, 0 % and 1 % N2):")
    print(xs[["lower", "upper", "ngfsrdw_E0", "ngfsrdw_peak", "analytic_peak", "ngfsrdw_k_0pct", "analytic_k_0pct",
              "k_ratio_0pct", "k_ratio_1pct"]].to_string(float_format=lambda v: f"{v:.3g}"))
    print("\nfits at n_c:")
    print(d[["xsec", "eedf", "tilt", "x", "EN", "Te_eff", "chi2_red", "dchi2", "n_1s5", "las_1s5", "ratio_cr_las", "ln_5p_4p"]]
          .to_string(float_format=lambda v: f"{v:.3g}"))
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--build":
        P.build_table(*sys.argv[2:4])
    else:
        main()
