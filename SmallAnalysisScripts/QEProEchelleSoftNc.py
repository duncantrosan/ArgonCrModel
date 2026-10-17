# -*- coding: utf-8 -*-
"""
QEProEchelleSoftNc.py

QE Pro and echelle CR fits of the 1.5 Torr power sweeps (pure Ar and 5 % N2, 10-90 W) with the soft one-sided Ne
penalty above the critical density (CRFitNeTe Ne_penalty_above / Main.py [fit] nc_penalty): Ne stays free, but Ne above
NC_START x n_c costs Delta chi^2 / B^2 = (ln(Ne / (NC_START n_c)) / ln NC_WIDTH)^2 (Main.py defaults 1 and 3).  The fit
then lands on the local chi^2 minimum near n_c where the lines alone would run to the 3e19 grid edge.  Lines-only
results (Ne_lines_only, x_lines_only) and the penalty at the fit are kept in fits.csv.

  QE Pro   qp_4p5p_strong of QEProFivePCRFit: 4p -> 4s groups not contaminated by N2 + the band-free 5p groups measured to
           better than 50 % in 5 % N2 (418.3, 426.4, 430.0, 433.8, 451.3), mean spectrum per power
  echelle  the pipeline's nine features (CRFitNeTe group fitter, 5 spectra per power) without the features that have
           clipped raw pixels in a spectrum (EchelleSaturationCheck) - plotted;  echelle_all (clipped lines in) in fits.csv
CR fit as QEProCRFit (AnalyticFivePFit tables at the LAS-trend Tg, one scale per spectrum, sigma_model per line, no
tilt).  Bands: Delta chi^2 <= B^2 on the penalized surface.  EEDF (--eedf):
  ar      pure-Ar microwave BOLSIG+ EEDF, N2 quenching and dilution in the CR model only (QEProCRFit; default)
  vibsat  Ar/N2 microwave EEDF with saturated N2(v) (LASTgComparison.lib_folder(vibsat=True), CR tables VibSat_*;
          missing libraries and tables are built first); only the sweeps with N2 (pure Ar is the same either way)

Output in Experimental_Data/Output/QEProComparison/soft_nc[_<eedf>]/: fits.csv, figures/EN_Ne_<sweep>.png
Run after QEProCalibration.py, QEProBandModel.py, QEProCRFit.py:
  python SmallAnalysisScripts/QEProEchelleSoftNc.py [--eedf vibsat]
"""
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import QEProCalibration as qc                       # noqa: E402
import QEProCRFit as qf                             # noqa: E402
import QEProFivePCRFit as q5                        # noqa: E402

A, L, crf, cdf = qf.A, qc.L, qc.crf, qc.L.mlf.cdf
EEDF = sys.argv[sys.argv.index("--eedf") + 1] if "--eedf" in sys.argv else "ar"
EEDF_LABEL = {"ar": "pure-Ar microwave EEDF", "vibsat": "Ar/N$_2$ microwave EEDF, saturated N$_2$(v)"}[EEDF]
OUTDIR = os.path.join(q5.OUT_ROOT, "soft_nc" if EEDF == "ar" else f"soft_nc_{EEDF}")
FIGDIR = os.path.join(OUTDIR, "figures")
NC_START = 1.0                                      # penalty starts at this multiple of n_c (Main.py nc_penalty_start)
NC_WIDTH = 3.0                                      # factor in Ne above the start that costs 1 sigma (nc_penalty_width)
QEPRO_SET = "qp_4p5p_strong"
ECHELLE_SET = "echelle_unsaturated"
COLORS = {"QE Pro": "#2a78d6", "echelle": "#eb6834"}   # categorical slots 1, 2 (dataviz reference palette)
MARKERS = {"QE Pro": "o", "echelle": "s"}


def soft_cfg(key):
    cfg = dict(A.table_cfg(key, EEDF), export_flags="?-?-1-*", Ne_penalty_above=NC_START * qf.N_C,
               Ne_penalty_width=NC_WIDTH)
    crf.no_tilt(cfg)
    return cfg


def build_missing(keys):
    """BOLSIG+ libraries and CR tables of EEDF that do not exist yet (as N2SweepPureArEEDF)."""
    if EEDF == "vibsat":
        L.vibsat_xsec()
        libs = [k for k in keys if L._unkey(k)[0] > 0
                and not L.he.IsBolsigLibrary(L.lib_folder(*L._unkey(k), vibsat=True))]
        L._run_workers("--lib-vibsat", libs, "BOLSIG+ libraries (vibsat)")
    with A.xsec_set(A.CR_XSEC):
        tabs = [(k, EEDF) for k in keys if not crf.table_is_current(A.table_cfg(k, EEDF))]
    A.build_all(tabs)


def fit_soft(d, feats, cfg, sweep, P):
    """Penalized CR fit of one condition -> summary dict (best fit, Delta chi^2 <= B^2 bands, lines-only optimum)."""
    tab = qf.table(cfg)
    grid = crf.fine_grid(tab, cfg)
    Te_f = crf.te_eff_fine(tab, grid)
    M = crf.feature_model(tab, feats, cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        cond, _, _, posts = crf.fit_all(d.assign(sweep=sweep), feats, M, grid, Te_f, cfg, verbose=False)
    c = cond.iloc[0]
    fX, fN = grid
    chi2 = cdf.chi2_grid(posts[f"{sweep}|{P:g}"], c.chi2_min, c.birge)     # penalized
    i, j = np.unravel_index(int(np.argmin(chi2)), chi2.shape)
    ok = chi2 <= chi2[i, j] + c.birge ** 2
    T2 = np.broadcast_to(Te_f[:, None], chi2.shape) if np.ndim(Te_f) == 1 else Te_f
    lev = list(tab["levels"])
    n1s5 = np.exp(crf.on_fine_grid(tab, np.log(np.clip(tab["dens"][:, :, lev.index("4s1")], 1e-300, None)), cfg, k=1))
    return dict(n_spec=int(c.n_spec), n_points=int(c.n_points), n_feat=int(c.n_feat), dof=int(c.dof), birge=c.birge,
                EN=float(np.exp(fX[i])), EN_lo=float(np.exp(fX[ok.any(1)].min())),
                EN_hi=float(np.exp(fX[ok.any(1)].max())), Te_eff=float(T2[i, j]),
                Ne=float(np.exp(fN[j])), Ne_lo=float(np.exp(fN[ok.any(0)].min())),
                Ne_hi=float(np.exp(fN[ok.any(0)].max())), chi2_red_lines=float(c.chi2_red),
                penalty_at_fit=float(c.penalty_at_fit), dchi2_lines_at_fit=float(c.dchi2_lines_at_fit),
                EN_lines_only=float(c.x_lines_only), Ne_lines_only=float(c.Ne_lines_only),
                n_1s5=float(n1s5[i, j]), n_1s5_lo=float(n1s5[ok].min()), n_1s5_hi=float(n1s5[ok].max()))


def run_set(instrument, name, ft, feats, cond):
    rows = []
    for c in cond.itertuples():
        d = ft[(ft.sweep == c.sweep) & np.isclose(ft.x, c.P)]
        if d.feature.nunique() < 3:
            continue
        r = fit_soft(d, feats, soft_cfg(c.key), c.sweep, c.P)
        rows.append(dict(instrument=instrument, set=name, sweep=c.sweep, P=c.P, N2=c.N2, Tg=c.Tg, las_1s5=c.las_1s5, **r))
        print(f"  {instrument:<8s} {name:<20s} {c.sweep:<8s} {c.P:3.0f} W: E/N {r['EN']:5.1f} Td "
              f"[{r['EN_lo']:.1f}-{r['EN_hi']:.1f}]  Ne {r['Ne']:.2e} [{r['Ne_lo']:.1e}-{r['Ne_hi']:.1e}]  "
              f"chi2/dof {r['chi2_red_lines']:.2f}  penalty {r['penalty_at_fit']:.2f}  "
              f"(lines only: E/N {r['EN_lines_only']:.1f} Td, Ne {r['Ne_lines_only']:.1e})", flush=True)
    return pd.DataFrame(rows)


def plot_sweep(fits, sweep, path):
    f = fits[(fits.sweep == sweep) & fits.set.isin([QEPRO_SET, ECHELLE_SET])]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    panels = (("EN", "E/N [Td]", False), ("Ne", "N$_e$ [m$^{-3}$]", True))
    for ax, (col, lab, logy) in zip(axs, panels):
        for inst in ("QE Pro", "echelle"):
            g = f[f.instrument == inst].sort_values("P")
            if g.empty:
                continue
            ax.fill_between(g.P, g[f"{col}_lo"], g[f"{col}_hi"], color=COLORS[inst], alpha=0.15, lw=0)
            ax.plot(g.P, g[col], marker=MARKERS[inst], ms=8, lw=2, color=COLORS[inst],
                    mec="white", mew=1.5, label=inst)
        if col == "Ne":
            ax.axhline(qf.N_C, color="0.55", ls=":", lw=1)
            ax.annotate("n$_c$ (2.42 GHz)", (1.0, qf.N_C), xycoords=("axes fraction", "data"), xytext=(-4, 4),
                        textcoords="offset points", ha="right", va="bottom", fontsize=8, color="0.4")
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlabel("Power [W]")
        ax.set_ylabel(lab)
        ax.set_xticks(sorted(f.P.unique()))
        ax.grid(alpha=0.25, which="major")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"CR fit with a soft N$_e$ penalty above n$_c$, {qc.SWEEPS[sweep]['label']}, 1.5 Torr, {EEDF_LABEL} "
                 f"(bands: $\\Delta\\chi^2 \\leq$ B$^2$)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    cond = qf.conditions()
    if EEDF != "ar":
        cond = cond[cond.N2 > 0]
    build_missing(sorted(set(cond.key)))
    ref = qc.reference_lines()
    feats = qf.qepro_features(ref)["qepro_merged"]
    tables, _ = q5.feature_tables(feats)
    fits = [run_set("QE Pro", QEPRO_SET, tables[QEPRO_SET], feats, cond)]
    for sw in [sw for sw in cond.sweep.unique() if qf.has_echelle(sw)]:
        c = cond[cond.sweep == sw]
        e_feats, ft_all, ft_ok = qf.echelle_feature_table(c, sw)
        fits.append(run_set("echelle", ECHELLE_SET, ft_ok, e_feats, c))
        fits.append(run_set("echelle", "echelle_all", ft_all, e_feats, c))
    fits = pd.concat(fits, ignore_index=True)
    fits["ratio_1s5_las"] = fits.n_1s5 / fits.las_1s5
    fits.to_csv(os.path.join(OUTDIR, "fits.csv"), index=False)
    for sw in fits.sweep.unique():
        plot_sweep(fits, sw, os.path.join(FIGDIR, f"EN_Ne_{sw}.png"))
    cols = ["instrument", "set", "P", "n_feat", "EN", "EN_lo", "EN_hi", "Ne", "Ne_lo", "Ne_hi", "chi2_red_lines",
            "penalty_at_fit", "EN_lines_only", "Ne_lines_only", "n_1s5", "ratio_1s5_las"]
    for sw in fits.sweep.unique():
        print(f"\n{qc.SWEEPS[sw]['label']} ({EEDF}):")
        print(fits[fits.sweep == sw][cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nwritten to", OUTDIR)
    return fits


if __name__ == "__main__":
    RES = main()
