# -*- coding: utf-8 -*-
"""
QEProNitrogenLines.py

Can the QE Pro measure the N I lines of the actinometry (1.5 Torr, 5 % N2, 10-90 W, spectra taken with the echelle at
the same time)?  At 0.76 nm/px the N2 first positive bands (B -> A) are a continuum under the N I lines.

1. Contrast: the echelle spectrum of each power (mean of the repeats, calibrated) convolved with the QE Pro LSF onto
   the QE Pro pixels, and the N I lines alone (echelle line areas of QEProCalibration.echelle_lines placed as LSF
   profiles): N I share of the QE Pro signal at the pixel of each N I line, where < MAX_GAP_FRAC of that pixel's LSF
   acceptance falls in echelle order gaps.  This is what the QE Pro would have to separate from the bands.
2. Extraction: the N I lines fitted in the QE Pro spectra (response-corrected) like the Ar lines
   (QEProCalibration.fit_window: one LSF component per N I line of the multiplet window - their ratio left free - and
   every CR-model Ar line, polynomial baseline of degree BASE_DEG), against the echelle areas of the same lines.
3. Figures: QE Pro spectrum, echelle convolved (scaled to the QE Pro), and the N I lines alone, around each N I line.

Output in Experimental_Data/Output/QEProComparison/nitrogen/: n_lines.csv (per line and power: echelle area, QE Pro
area, N I share of the QE Pro pixel), figures/n_lines_<P>W.png
Run from Spyder (F5) or python SmallAnalysisScripts/QEProNitrogenLines.py (after QEProCalibration.py)
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import QEProCalibration as qc                       # noqa: E402

OUTDIR = os.path.join(qc.crf.ROOT_DIR, "Experimental_Data", "Output", "QEProComparison", "nitrogen")
FIGDIR = os.path.join(OUTDIR, "figures")
SWEEP = "N2_5pct"
N_LINES = (742.364, 744.229, 746.831, 818.486, 818.801, 820.036, 821.072, 821.634, 822.313, 824.239, 856.774,
           859.400, 862.923, 865.588, 868.028, 868.340, 868.615)   # N I 3p 4S*, 3p 4P*, 3p 2P*, 3p 4D* multiplets
WINDOWS = ((738.0, 753.0), (815.0, 829.0), (853.0, 872.0))    # figure / extraction windows [nm]
BASE_DEG = 3
MAX_GAP_FRAC = 0.02                                 # a QE Pro pixel counts as covered by the echelle when < this of its
                                                    # LSF acceptance falls in order gaps
POWERS_FIG = (10.0, 50.0, 90.0)


def n_line_areas(el):
    """Mean echelle area per N I line and power over the spectra where it is fitted ok / weak (>= 3 spectra)."""
    n = el[(el.species == "N I") & el.status.isin(("ok", "weak"))]
    n = n[n.wl_air.round(3).isin(np.round(N_LINES, 3))]
    g = n.groupby(["wl_air", "x"]).agg(E=("area", "mean"), n_E=("file", "nunique"), snr=("snr", "median"))
    return g[g.n_E >= 3].reset_index()


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    wl, p, R = qc.load_calibration()
    ref = qc.reference_lines()
    feats = qc.features(ref)
    el = qc.echelle_lines(SWEEP, [l for ls in feats.lines for l in ls] + list(N_LINES))
    nA = n_line_areas(el)
    et = qc.echelle_table(SWEEP)
    lam, I0 = qc.arn.read_spa(et.path.iloc[0])
    K = qc.convolution_matrix(lam, wl, p)
    dlam = np.gradient(lam)
    gap_frac = (K @ ((I0 == 0) * dlam)) / np.clip(K @ dlam, 1e-30, None)   # LSF acceptance of a pixel in order gaps
    edges = qc.pixel_edges(wl)
    qp = qc.load_qepro(SWEEP)
    rows = []
    for P, g in et.groupby("P"):
        k = np.isclose(qp["files"].P, P)
        Q = qp["spec"][k].mean(0) / R(wl)                         # QE Pro, echelle units (relative)
        Ec = K @ (np.mean([qc.arn.read_spa(f)[1] for f in g.path], axis=0) * dlam)
        a = nA[np.isclose(nA.x, P)]
        Nc = sum(r.E * qc.line_profile(edges, r.wl_air, p) for r in a.itertuples()) if len(a) else np.zeros_like(wl)
        ok = (gap_frac < MAX_GAP_FRAC) & (Ec > 0)
        s = np.sum(Q[ok & (wl > 700) & (wl < 870)]) / np.sum(Ec[ok & (wl > 700) & (wl < 870)])  # echelle -> QE Pro
        # QE Pro extraction: N I and Ar components free, polynomial baseline
        ext = {}
        for lo, hi in WINDOWS:
            m = (wl > lo) & (wl < hi)
            comps = qc.components(ref, lo - 1, hi + 1) + [dict(wl=l, lines=[l], w=np.array([1.0]))
                                                          for l in N_LINES if lo < l < hi]
            amp, cov, _, _ = qc.fit_window(wl[m], Q[m] * R(wl[m]), qc.pixel_variance(Q[m] * R(wl[m])),
                                          ~qp["sat"][k].any(0)[m], comps, p, deg=BASE_DEG)
            for c, A_, v in zip(comps, amp, np.diag(cov)):
                if c["lines"][0] in N_LINES:
                    ext[c["lines"][0]] = (A_ / R(c["wl"]), np.sqrt(v) / R(c["wl"]))
        for l in N_LINES:
            i = int(np.argmin(np.abs(wl - l)))
            e = a[np.isclose(a.wl_air, l, atol=1e-3)]
            rows.append(dict(P=P, wl=l, echelle_area=float(e.E.iloc[0]) * s if len(e) else np.nan,
                             echelle_snr=float(e.snr.iloc[0]) if len(e) else np.nan,
                             n_share_qepro_pixel=float(Nc[i] / Ec[i]) if ok[i] and len(e) else np.nan,
                             echelle_covers=bool(ok[i]), qepro_area=ext.get(l, (np.nan, np.nan))[0] * 1.0,
                             qepro_err=ext.get(l, (np.nan, np.nan))[1]))
        if P in POWERS_FIG:
            fig, axs = plt.subplots(1, len(WINDOWS), figsize=(18, 5))
            for ax, (lo, hi) in zip(axs, WINDOWS):
                m = (wl > lo) & (wl < hi)
                ax.plot(wl[m], Q[m], "o-", color="k", ms=3, lw=0.8, label="QE Pro (response-corrected)")
                e = np.where(ok, Ec * s, np.nan)
                ax.plot(wl[m], e[m], "s-", color="C0", ms=3, lw=0.8, label="echelle convolved with the LSF")
                ax.plot(wl[m], (Nc * s)[m], "-", color="C3", lw=1.5, label="N I lines only (echelle areas)")
                for l in N_LINES:
                    if lo < l < hi:
                        ax.axvline(l, color="C3", lw=0.5, ls=":")
                ax.set_yscale("log")
                ax.set_ylim(np.nanmax(Q[m]) * 1e-3, np.nanmax(Q[m]) * 2)
                ax.set_xlim(lo, hi)
                ax.grid(alpha=0.3, which="both")
                ax.set_xlabel("Wavelength [nm]")
            axs[0].legend(fontsize=7)
            axs[0].set_ylabel("QE Pro signal (echelle units, relative)")
            fig.suptitle(f"N I lines at QE Pro resolution, 5 % N$_2$, 1.5 Torr, {P:g} W (gaps: echelle order gaps "
                         f"within the LSF reach)", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(FIGDIR, f"n_lines_{P:g}W.png"), dpi=170)
            plt.close(fig)
    t = pd.DataFrame(rows)
    t["qepro_over_echelle"] = t.qepro_area / t.echelle_area
    t.to_csv(os.path.join(OUTDIR, "n_lines.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(t[t.echelle_area.notna()].round(4).to_string(index=False))
    print("written to", OUTDIR)
    return t


if __name__ == "__main__":
    T = main()
