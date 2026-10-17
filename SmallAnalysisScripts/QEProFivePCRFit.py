# -*- coding: utf-8 -*-
"""
QEProFivePCRFit.py

CR fit of the QE Pro 1.5 Torr power sweeps (pure Ar and 5 % N2, 10-90 W) WITH the Ar I 5p -> 4s lines, taken from the
QE Pro band + line fit of QEProBandModel (N2 C -> B and N2+ B -> X templates removed, 385-480 nm), together with the
QE Pro 4p -> 4s lines (690-990 nm, LSF fits of QEProCalibration).  QEProCRFit used the band-free 5p groups only within
its 15 % error cut (418.3 and 426.4 nm survive in 5 % N2) and only where the echelle had the same groups; here every
band-free 5p group enters with its band-fit error (dropped only above MAX_REL_ERR_5P, i.e. not detected at 1 sigma).
The CR fit is the one of QEProCRFit (pure-Ar microwave BOLSIG+ EEDF at the LAS-trend Tg, AnalyticFivePFit tables, one
free scale per spectrum, sigma_model per line, no tilt).  The same feature definitions (qepro_merged groups) and the same
band + line method in both sweeps (pure Ar: no band templates), so the pure-Ar sweep checks the method.

Feature sets:
  qp_4p_clean     4p -> 4s groups not contaminated by N2 in the 5 % N2 spectra (below), no 5p   (baseline)
  qp_4p5p_clean   qp_4p_clean + every band-free 5p group
  qp_4p5p_strong  qp_4p_clean + the band-free 5p groups measured to better than STRONG_5P_REL_ERR in 5 % N2 (median
                  over the powers; the others carry N2 / N2+ band residue: 394.9 rises with power next to the N2+ 391.4
                  head, 404.6 / 459.5 / 462.8 / 470.2 sit 0.8-1.8 above the model in N2 but ~0.25 in pure Ar)
  qp_4p5p         every QE Pro 4p -> 4s group + every band-free 5p group
  qp_5p_only      the band-free 5p groups alone
Contaminated 4p groups: on the N2 first positive heads 727.3 / 738.7 / 750.4 nm (FPS_HEADS_NM, QEProCalibration), or
LSF-fit relative error in 5 % N2 > ERR_INFLATION x the pure-Ar one (band flux under the line), or not measured in most
5 % N2 spectra (866.79: N I).  The same 4p list is used in both sweeps.
Reference rows (not refitted) from QEProCRFit's fits.csv: echelle_all, echelle_common_5p, qepro_bandfree_common.

Output in Experimental_Data/Output/QEProComparison/crfit_5p/: fits.csv, residuals.csv, features_<set>.csv,
contamination_4p.csv, figures/fits_<sweep>.png, residuals_<set>_<sweep>.png, spectrum_<sweep>_<P>W.png (band-subtracted
5p window and the 4p window against the CR spectrum at the fit)
Run after QEProCalibration.py, QEProBandModel.py and QEProCRFit.py: python SmallAnalysisScripts/QEProFivePCRFit.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import QEProCalibration as qc                       # noqa: E402
import QEProCRFit as qf                             # noqa: E402
import QEProBandModel as qb                         # noqa: E402

L = qc.L
OUT_ROOT = os.path.join(qc.crf.ROOT_DIR, "Experimental_Data", "Output", "QEProComparison")
OUTDIR = os.path.join(OUT_ROOT, "crfit_5p")
FIGDIR = os.path.join(OUTDIR, "figures")
MAX_REL_ERR_5P = 1.0                                # band-free 5p group dropped above this relative error
STRONG_5P_REL_ERR = 0.5                             # qp_4p5p_strong: median 5 % N2 relative error of a 5p group
ERR_INFLATION = 2.0                                 # 4p group contaminated if rel. error (5 % N2) > this x pure Ar
FPS_HEADS_NM = (727.3, 738.7, 750.4)                # N2 B -> A heads under Ar 727.29 / 738.40 / 750.39
FPS_TOL_NM = 0.4
MIN_N2_FRACTION = 0.5                               # 4p group kept if measured in at least this fraction of N2 spectra
SETS = ("qp_4p_clean", "qp_4p5p_clean", "qp_4p5p_strong", "qp_4p5p", "qp_5p_only")
REF_SETS = ("echelle_all", "echelle_common_5p", "qepro_bandfree_common")
SPECTRUM_SET = "qp_4p5p_clean"
SPECTRUM_P = (10.0, 50.0, 90.0)
qf.STYLE.update({"qp_4p_clean": ("C0", "^", "--"), "qp_4p5p_clean": ("C3", "o", "-"), "qp_4p5p": ("C1", "s", "-."),
                 "qp_4p5p_strong": ("C8", "D", "-"), "qp_5p_only": ("C4", "v", ":")})


# ---- feature tables ---------------------------------------------------------------------------
def red_table():
    """QE Pro 4p -> 4s groups (qepro_merged, > 600 nm) of both sweeps, as QEProCRFit measured them."""
    q = pd.read_csv(os.path.join(qf.OUTDIR, "features_qepro_merged.csv"))
    return q[q.wl > 600].reset_index(drop=True)


def fivep_table():
    """Band-free 5p groups of both sweeps (QEProBandModel features_bandfree.csv)."""
    b = pd.read_csv(os.path.join(qf.qb_outdir(), "features_bandfree.csv"))
    b = b[b.area > 0].assign(rel_err=lambda t: t.err / t.area)
    b = b[b.rel_err <= MAX_REL_ERR_5P]
    return b[["sweep", "file", "x", "rep", "feature", "wl", "area", "rel_err"]].reset_index(drop=True)


def contamination(red, feats):
    """Per 4p group: median relative error in 5 % N2 and pure Ar, their ratio, fraction of N2 spectra measured, FPS head
    coincidence -> contaminated flag."""
    n_spec = red.groupby("sweep").file.nunique()
    e = red.pivot_table(index="feature", columns="sweep", values="rel_err", aggfunc="median")
    n = red.groupby(["feature", "sweep"]).file.nunique().unstack()
    lines = dict(zip(feats.feature, feats.lines))
    t = pd.DataFrame(dict(rel_err_N2=e["N2_5pct"], rel_err_Ar=e["pure_Ar"],
                          frac_N2=n["N2_5pct"].reindex(e.index).fillna(0) / n_spec["N2_5pct"]))
    t["inflation"] = t.rel_err_N2 / t.rel_err_Ar
    t["fps_head"] = [any(abs(l - h) < FPS_TOL_NM for l in lines[f] for h in FPS_HEADS_NM) for f in t.index]
    t["contaminated"] = (t.inflation > ERR_INFLATION) | t.fps_head | (t.frac_N2 < MIN_N2_FRACTION) | t.inflation.isna()
    t["wl"] = [float(f.split()[0]) for f in t.index]
    return t.sort_values("wl")


def feature_tables(feats):
    red, five = red_table(), fivep_table()
    con = contamination(red, feats)
    clean = set(con.index[~con.contaminated])
    red_clean = red[red.feature.isin(clean)]
    e5 = five[five.sweep == "N2_5pct"].groupby("feature").rel_err.median()
    strong = five[five.feature.isin(e5.index[e5 <= STRONG_5P_REL_ERR])]
    cols = five.columns
    return {"qp_4p_clean": red_clean[cols],
            "qp_4p5p_clean": pd.concat([red_clean[cols], five], ignore_index=True),
            "qp_4p5p_strong": pd.concat([red_clean[cols], strong], ignore_index=True),
            "qp_4p5p": pd.concat([red[cols], five], ignore_index=True),
            "qp_5p_only": five}, con


# ---- spectrum figure --------------------------------------------------------------------------
def band_templates_all(wl, p, lines, lev, nplines):
    lo, hi = qb.FIT_RANGE

    def templates(T):
        t = {("CB", v): g for v, g in qb.band_templates(lines, lev, T, p, wl, lo - 2, hi + 2, qc.FINE_STEP).items()}
        if qb.WITH_N2PLUS:
            t.update({("NP", v): g for v, g in qb.n2plus_templates(nplines, T, p, wl, lo - 2, hi + 2).items()})
        return t
    return {T: templates(T) for T in qb.TROT_COLD}, {T: templates(T) for T in qb.TROT_HOT}


def model_spectrum(tab, cfg, EN, Ne, edges, p):
    wl_l, I_l = L.ov.model_lines_at(tab, EN, Ne, cfg)
    spec = np.zeros(len(edges) - 1)
    for w, I in zip(wl_l, I_l):
        if 380 < w < 1000 and I > 0:
            spec += I * qc.line_profile(edges, w, p)
    return wl_l, I_l, spec


def feature_scale(d, feats, wl_l, I_l, cfg):
    """ln scale of the CR line intensities to the measured feature areas (weights 1 / (rel_err^2 + sigma_model^2))."""
    lines = dict(zip(feats.feature, feats.lines))
    mod = np.array([I_l[np.isclose(np.asarray(wl_l)[:, None], lines[f], atol=2e-3).any(1)].sum() for f in d.feature])
    ok = mod > 0
    w = 1 / (d.rel_err.to_numpy()[ok] ** 2 + cfg["sigma_model"] ** 2)
    return float(np.sum(w * (np.log(d.area.to_numpy()[ok]) - np.log(mod[ok]))) / w.sum())


def plot_spectrum(sweep, P, fit, d, feats, cond, cal, band_fit, path):
    wl, p, R = cal
    c = cond[(cond.sweep == sweep) & np.isclose(cond.P, P)].iloc[0]
    cfg = qf.fit_cfg(c.key)
    tab = qf.table(cfg)
    edges = qc.pixel_edges(wl)
    qm = qf.mean_spectra(qc.load_qepro(sweep))
    k = int(np.flatnonzero(np.isclose(qm["files"].P, P))[0])
    y = qm["spec"][k] / R(wl)
    sat = qm["sat"][k]
    r = band_fit
    fig, axs = plt.subplots(2, 1, figsize=(16, 10))
    ax, bx = axs
    ax.plot(r["x"], r["y"] - r["bands"] - r["base"], "o-", color="k", ms=3, lw=0.6,
            label="QE Pro - N$_2$ C-B / N$_2^+$ B-X bands - baseline (QEProBandModel)")
    if len(r["amp"]):
        ax.plot(r["x"], r["bands"] + r["base"], "-", color="0.75", lw=0.8, label="removed bands + baseline")
    m = (wl > 690) & (wl < 990)
    bx.plot(wl[m], np.where(sat[m], np.nan, y[m]), "o", color="k", ms=2.5, label="QE Pro (response-corrected)")
    for EN, Ne, lab, col in ((fit.EN_nc, qf.N_C, f"n$_c$: E/N {fit.EN_nc:.1f} Td, $\\chi^2$/dof {fit.chi2_red_nc:.2f}",
                              "C3"),
                             (fit.EN_free, fit.Ne_free, f"free: E/N {fit.EN_free:.1f} Td, N$_e$ {fit.Ne_free:.1e}, "
                              f"$\\chi^2$/dof {fit.chi2_red_free:.2f}", "C2")):
        wl_l, I_l, spec = model_spectrum(tab, cfg, EN, Ne, edges, p)
        s = np.exp(feature_scale(d, feats, wl_l, I_l, cfg))
        mb = (wl > r["x"].min() - 0.01) & (wl < r["x"].max() + 0.01)
        ax.plot(wl[mb], s * spec[mb], "-", color=col, lw=1.3, label=f"CR model ({lab})")
        bx.plot(wl[m], s * spec[m], "-", color=col, lw=1.0, label=f"CR model ({lab})")
    lines_5p = [l for f in feats.itertuples() if f.wl < 500 and f.feature in set(d.feature) for l in f.lines]
    for l in lines_5p:
        ax.axvline(l, color="C2", lw=0.4, ls=":")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(*qb.FIT_RANGE)
    ax.set_ylabel("signal (echelle units, relative)")
    bx.set_yscale("log")
    top = np.nanmax(y[m])
    bx.set_ylim(top * 3e-4, top * 2)
    bx.set_xlim(690, 990)
    bx.set_xlabel("Wavelength [nm]")
    for a in axs:
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=7)
    ax.set_title(f"QE Pro {qc.SWEEPS[sweep]['label']}, {P:g} W, 1.5 Torr, set {SPECTRUM_SET}: CR spectrum (LSF-convolved, "
                 f"one scale from the fitted 4p + 5p feature areas) against the band-subtracted 5p window and the 4p window",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# ---- driver -----------------------------------------------------------------------------------
def main():
    os.makedirs(FIGDIR, exist_ok=True)
    cond = qf.conditions()
    ref = qc.reference_lines()
    feats = qf.qepro_features(ref)["qepro_merged"]
    tables, con = feature_tables(feats)
    con.to_csv(os.path.join(OUTDIR, "contamination_4p.csv"))
    print("4p groups (5 % N2 contamination check):")
    print(con.round(3).to_string(), flush=True)
    fits, res = [], []
    for name in SETS:
        ft = tables[name]
        ft.to_csv(os.path.join(OUTDIR, f"features_{name}.csv"), index=False)
        print(f"\n{name}: {ft.feature.nunique()} groups "
              f"({', '.join(sorted({f.split()[0] for f in ft.feature}, key=float))})", flush=True)
        f, r = qf.fit_set(name, ft, feats, cond)
        fits.append(f)
        res.append(r)
    fits = pd.concat(fits, ignore_index=True)
    res = pd.concat(res, ignore_index=True)
    old = pd.read_csv(os.path.join(qf.OUTDIR, "fits.csv"))
    fits = pd.concat([fits, old[old.set.isin(REF_SETS)]], ignore_index=True)
    fits["ratio_1s5_nc_las"] = fits.n_1s5_nc / fits.las_1s5
    fits.to_csv(os.path.join(OUTDIR, "fits.csv"), index=False)
    res.to_csv(os.path.join(OUTDIR, "residuals.csv"), index=False)
    for sw in qc.SWEEPS:
        qf.plot_fits(fits, sw, os.path.join(FIGDIR, f"fits_{sw}.png"))
        for name in SETS:
            qf.plot_residuals(res, name, sw, os.path.join(FIGDIR, f"residuals_{name}_{sw}.png"))
    # spectra: band + line fit again at the shown powers (QEProBandModel.fit_spectrum, same settings)
    cal = qc.load_calibration()
    wl, p, R = cal
    lines, lev = qc.n2_cb_lines()
    cold, hot = band_templates_all(wl, p, lines, lev, qb.n2plus_lines())
    f5 = feats[(feats.wl > qb.FIT_RANGE[0] + 1) & (feats.wl < qb.FIT_RANGE[1] - 1)].reset_index(drop=True)
    for sw in qc.SWEEPS:
        qm = qf.mean_spectra(qc.load_qepro(sw))
        no_n2 = qc.SWEEPS[sw]["N2"] == 0
        for P in SPECTRUM_P:
            fit = fits[(fits.set == SPECTRUM_SET) & (fits.sweep == sw) & np.isclose(fits.P, P)]
            if fit.empty:
                continue
            k = int(np.flatnonzero(np.isclose(qm["files"].P, P))[0])
            r = qb.fit_spectrum(wl, qm["spec"][k], qm["sat"][k], R, p, ref, {0.0: {}} if no_n2 else cold,
                                {0.0: {}} if no_n2 else hot, f5)
            ft = tables[SPECTRUM_SET]
            d = ft[(ft.sweep == sw) & np.isclose(ft.x, P)]
            plot_spectrum(sw, P, fit.iloc[0], d, feats, cond, cal, r, os.path.join(FIGDIR, f"spectrum_{sw}_{P:g}W.png"))
    cols = ["set", "P", "n_feat", "EN_free", "Ne_free", "chi2_red_free", "EN_nc", "Te_nc", "chi2_red_nc", "dchi2_nc",
            "n_1s5_nc", "ratio_1s5_nc_las"]
    for sw in qc.SWEEPS:
        print(f"\n{qc.SWEEPS[sw]['label']}:")
        print(fits[fits.sweep == sw][cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nwritten to", OUTDIR)
    return dict(cond=cond, fits=fits, res=res, tables=tables, con=con)


if __name__ == "__main__":
    RES = main()
