# -*- coding: utf-8 -*-
"""
QEProCRFit.py

Can the QE Pro replace the echelle for the CR fit?  The 1.5 Torr power sweeps (pure Ar and 5 % N2, 10-90 W) were
recorded with both spectrometers on the same plasma at the same time.  Both are fitted with the CR model of the
echelle analysis (N2SweepPureArEEDF / Main.py defaults): pure-Ar microwave BOLSIG+ EEDF at the LAS-trend gas
temperature, AnalyticFivePFit CR tables (N2 quenching and dilution in the CR model), one free scale per spectrum,
sigma_model per line, no response tilt (CRFitNeTe.no_tilt).

QE Pro (wavelength axis, LSF and relative response R from QEProCalibration - run it first): the three spectra of a
power agree to 0.03 % and are averaged (their errors are LSF systematics common to all three, which separate fits
would average away).  Every feature (QEProCalibration.features: CR lines closer than MERGE_NM together) is measured
by the LSF fit and divided by R(lambda) -> energy units like the calibrated echelle.  A feature is left out where a
pixel within 1 nm of it exceeds SAT_COUNTS or its relative error exceeds MAX_REL_ERR.  Only 4p -> 4s and 5p -> 4s lines
(as the echelle fit).  Feature sets (QE_SETS):
  qepro_split   every feature, the ~1 nm doublets 750/751, 800/801, 810/811, 840/842 split by the LSF fit
  qepro_merged  features closer than MERGE_SETS_NM summed (no doublet splitting)
  qepro_red     4p -> 4s only (in Ar/N2 the 5p windows are mostly N2 bands at this resolution)
  qepro_9lines  the nine lines of the echelle CR fit (like for like)
  qepro_gaps    4p -> 4s lines the echelle cannot use (order gaps, > 870 nm, or 852.14 at an order edge)
Echelle (both sweeps; 5 spectra per power, export flags 3-1-1-30 accepted):
  echelle_all          CRFitNeTe's nine features measured by the pipeline's group fitter, as the pipeline uses them
  echelle_unsaturated  the same without the features that have clipped raw pixels (EchelleSaturationCheck)
Like for like (same CR-line groups on both instruments, only where both have them at that power; 852.14 left out):
  echelle_common       the qepro_merged groups summed from the echelle line fits (QEProCalibration.echelle_lines;
                       every line of a group fitted, none clipped), repeat scatter added as CRFitNeTe
  qepro_common         the qepro_merged features restricted to the groups echelle_common has at that power
  echelle_common_5p / qepro_bandfree_common   (Ar/N2, needs QEProBandModel) the same with the 5p groups of the QE Pro
                       band + line fit (N2 C -> B and N2+ B -> X removed)
In the 5 % N2 sweep every QE Pro set is restricted to 690-990 nm (5p windows = N2 bands), except qepro_bandfree_common.
Per condition: free fit (E/N, Ne; bands Delta chi^2 <= B^2), fit at n_c (E/N refitted), chi^2/dof, Te_eff, CR 1s5
against the LAS 1s5; residuals per feature at the fit at n_c; the measured QE Pro spectrum against the CR spectrum
(LSF-convolved, response applied) at the fit.

Output in Experimental_Data/Output/QEProComparison/crfit/: fits.csv, residuals.csv, features_<set>.csv,
figures/fits_<sweep>.png, residuals_<set>_<sweep>.png, spectrum_<sweep>_<P>W.png
Run from Spyder (F5) or python SmallAnalysisScripts/QEProCRFit.py
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
import N2SweepPureArEEDF as S                       # noqa: E402

A, L, crf, esc = qc.A, qc.L, qc.crf, qc.esc
cdf = L.mlf.cdf

OUTDIR = os.path.join(crf.ROOT_DIR, "Experimental_Data", "Output", "QEProComparison", "crfit")
FIGDIR = os.path.join(OUTDIR, "figures")
N_C = L.N_C
MAX_REL_ERR = 0.15                                  # QE Pro feature dropped above this relative error
MERGE_SETS_NM = 1.8                                 # qepro_merged: features closer than this summed
ECHELLE_LINES = (415.859, 430.0101, 706.7218, 727.2936, 750.3869, 751.4652, 794.8176, 800.6157, 801.4786)
ECHELLE_USABLE = (706.7218, 727.2936, 750.3869, 751.4652, 772.3761, 772.4207, 794.8176, 800.6157, 801.4786,
                  826.4522)                         # 4p -> 4s lines the echelle measures (not in a gap, < 870 nm,
                                                    # 852.14 excluded: order edge)
QE_SETS = ("qepro_split", "qepro_merged", "qepro_red", "qepro_9lines", "qepro_gaps")
EC_SETS = ("echelle_all", "echelle_unsaturated", "echelle_common", "qepro_common", "echelle_common_5p",
           "qepro_bandfree_common")
STYLE = {"qepro_split": ("C0", "o", "-"), "qepro_merged": ("C9", "s", "--"), "qepro_red": ("C2", "^", "-."),
         "qepro_9lines": ("C1", "D", ":"), "qepro_gaps": ("C4", "v", ":"),
         "echelle_all": ("0.4", "x", "-"), "echelle_unsaturated": ("k", "P", "-"),
         "echelle_common": ("C3", "P", "-"), "qepro_common": ("C3", "o", "--"),
         "echelle_common_5p": ("C5", "P", "-"), "qepro_bandfree_common": ("C5", "o", "--")}
LAS_RUNS = "latest"                                 # as Main.py ([las] runs)


# ---- conditions -------------------------------------------------------------------------------
def conditions():
    """Every (sweep, power): % N2, Tg (LAS trend rounded, as the pipeline), LAS 1s5, CR table key."""
    L.LAS_RUNS = LAS_RUNS
    las = L.load_las()
    m, _, _ = L.fit_tg_trend(las)
    rows = []
    for sweep, d in qc.SWEEPS.items():
        for P in sorted(qc.load_qepro(sweep)["files"].P.unique()):
            Tg = float(L.tg_round(L.tg_trend(m, qc.P_MTORR / 1000, P, d["N2"])))
            r = L.las_1s5_at(las, qc.P_MTORR / 1000, P, d["N2"])
            rows.append(dict(sweep=sweep, P=P, N2=d["N2"], Tg=Tg, key=L._key(d["N2"], qc.P_MTORR, Tg),
                             las_1s5=r["n"], las_1s5_how=r["how"], las_1s5_rel_err=r["rel_err"]))
    return pd.DataFrame(rows)


def fit_cfg(key, sweeps=()):
    cfg = dict(A.table_cfg(key, "ar"), sweeps=list(sweeps), measure_dir=os.path.join(OUTDIR, "echelle_measurements"),
               export_flags="?-?-1-*")
    crf.no_tilt(cfg)
    return cfg


def table(cfg):
    with A.xsec_set(A.CR_XSEC), contextlib.redirect_stdout(io.StringIO()):
        return crf.build_model_table(cfg)


# ---- QE Pro features --------------------------------------------------------------------------
def mean_spectra(qp):
    """One spectrum per power (mean of the repeats; saturated where any repeat is)."""
    Ps = sorted(qp["files"].P.unique())
    k = [np.isclose(qp["files"].P, P) for P in Ps]
    return dict(qp, files=pd.DataFrame(dict(file=[f"mean_{P:g}W" for P in Ps], P=Ps, rep=1)),
                spec=np.array([qp["spec"][i].mean(0) for i in k]), raw=np.array([qp["raw"][i].max(0) for i in k]),
                sat=np.array([qp["sat"][i].any(0) for i in k]))


def _is_4p5p(f):
    return all(u[:2] in ("4p", "5p") and l.startswith("4s") for _, u, l in f.comps)


def qepro_features(ref):
    """Feature definitions of every QE_SETS entry (DataFrames with feature, wl, lines, comps)."""
    base = qc.features(ref)
    base = base[[_is_4p5p(f) for f in base.itertuples()]].reset_index(drop=True)
    near = lambda f, wls: any(np.isclose(l, w, atol=2e-3) for l in f.lines for w in wls)
    red = base[base.wl > 600].reset_index(drop=True)
    return {"qepro_split": base,
            "qepro_merged": qc.merge_features(base, MERGE_SETS_NM),
            "qepro_red": red,
            "qepro_9lines": base[[near(f, ECHELLE_LINES) for f in base.itertuples()]].reset_index(drop=True),
            "qepro_gaps": red[[not near(f, ECHELLE_USABLE) for f in red.itertuples()]].reset_index(drop=True)}


def qepro_feature_table(sweep, feats, ref, cal):
    """Feature table (CRFitNeTe format: sweep, file, x, rep, feature, wl, area, rel_err) of the mean QE Pro spectra
    of the sweep, response-corrected (energy units, relative)."""
    wl_cal, p, R = cal
    qm = mean_spectra(qc.load_qepro(sweep))
    m = qc.measure_sweep(qm, wl_cal, feats, ref, p)
    m["rel_err"] = m.err / m.area
    keep = (m.area > 0) & ~m.saturated & (m.rel_err <= MAX_REL_ERR)
    m = m[keep]
    return pd.DataFrame(dict(sweep=sweep, file=m.file, x=m.P, rep=m.rep, feature=m.feature, wl=m.wl,
                             area=m.area / R(m.wl), rel_err=m.rel_err, purity=m.purity, n_comp=m.n_comp))


# ---- echelle features -------------------------------------------------------------------------
def qb_outdir():
    return os.path.join(crf.ROOT_DIR, "Experimental_Data", "Output", "QEProComparison", "bands")


def has_echelle(sweep):
    return not qc.echelle_table(sweep).empty


def echelle_feature_table(cond, sweep):
    """(features, feature table) of the echelle sweep with the pipeline's nine features, and the table without the
    features that have clipped raw pixels in a spectrum."""
    d = qc.SWEEPS[sweep]
    folder = os.path.join(qc.ECHELLE_DIR, d["echelle"])
    sw = dict(name=sweep, folder=folder, xlabel="Power [W]", note=d["label"], exclude_x=(),
              parse=lambda n: (lambda px: px if px and px[0] != "ref" else None)(qc._parse(n)))
    cfg = dict(fit_cfg(cond.key.iloc[0], [sw]), measure_dir=os.path.join(OUTDIR, f"echelle_measurements_{sweep}"))
    with A.xsec_set(A.CR_XSEC), contextlib.redirect_stdout(io.StringIO()):
        feats, ft = L.measurements(cfg)
    cat, _ = qc.als._catalogue()
    sat = esc.check_folder(os.path.join(qc.ECHELLE_DIR, d.get("echelle_spe", d["echelle"])), cat, spa_folder=folder)
    bad = np.zeros(len(ft), bool)
    comps = dict(zip(feats.feature, feats.comps))
    for k, r in enumerate(ft.itertuples()):
        s = sat[sat.file == r.file].wl.to_numpy() if "wl" in sat else np.array([])
        bad[k] = len(s) > 0 and any(np.min(np.abs(s - w)) < esc.MATCH_NM for w, _, _ in comps[r.feature])
    return feats, ft, ft[~bad].copy()


def echelle_group_table(sweep, feats):
    """Echelle feature table with the QE Pro feature definitions feats (sums of the same CR lines), from the
    echelle line fits of QEProCalibration.echelle_lines (cached): a feature counts in a spectrum when every one of its
    lines is fitted with status ok / weak / blend (not in a gap, at an order edge, or with clipped raw pixels).
    Relative error: line-fit 1/SNR plus the pooled repeat scatter of the feature (as CRFitNeTe)."""
    el = qc.echelle_lines(sweep, [l for ls in feats.lines for l in ls])
    el = el[el.species == "Ar I"].assign(wl=lambda t: t.wl_air.round(3))
    rows = []
    for f in feats.itertuples():
        want = set(np.round(f.lines, 3))
        g = el[el.wl.isin(want)]
        for (file, P, rep), h in g.groupby(["file", "x", "rep"]):
            if set(h.wl) < want or not h.status.isin(qc.VALID_ECHELLE).all():
                continue
            area = h.area.sum()
            if area > 0:
                rows.append(dict(sweep=sweep, file=file, x=P, rep=rep, feature=f.feature, wl=f.wl, area=area,
                                 rel_err=np.sqrt(np.sum((h.area / h.snr.clip(lower=1e-9)) ** 2)) / area))
    ft = pd.DataFrame(rows)
    ft = ft[ft.rel_err <= MAX_REL_ERR]
    rs = crf.repeat_scatter(ft)
    return ft.assign(rel_err=np.sqrt(ft.rel_err ** 2 + ft.feature.map(rs) ** 2))


def common_tables(qe_ft, ec_ft, feats, min_spectra=3):
    """The QE Pro and echelle tables restricted, per sweep and power, to the features both have (the echelle in at
    least min_spectra of its spectra), without the lines the pipeline leaves out (CRFitNeTe exclude_wl: 852.14 at an
    echelle order edge - its echelle calibration is ~1.6x off, QEProCalibration)."""
    bad = {f.feature for f in feats.itertuples()
           if any(np.isclose(l, w, atol=2e-3) for l in f.lines for w in crf.CONFIG["exclude_wl"])}
    ec_ft = ec_ft[~ec_ft.feature.isin(bad)]
    n = ec_ft.groupby(["sweep", "x", "feature"]).file.nunique()
    ok = set(n[n >= min_spectra].index)
    keep_q = [(s, x, f) in ok for s, x, f in zip(qe_ft.sweep, qe_ft.x, qe_ft.feature)]
    q = qe_ft[keep_q]
    have = set(zip(q.sweep, q.x, q.feature))
    return q, ec_ft[[(s, x, f) in have for s, x, f in zip(ec_ft.sweep, ec_ft.x, ec_ft.feature)]]


# ---- fit --------------------------------------------------------------------------------------
def fit_one(d, feats, cfg, sweep, P):
    """CRFitNeTe fit of one condition -> (summary row, residual table at the fit at n_c, best indices)."""
    tab = table(cfg)
    grid = crf.fine_grid(tab, cfg)
    Te_f = crf.te_eff_fine(tab, grid)
    M = crf.feature_model(tab, feats, cfg)
    d = d.assign(sweep=sweep)
    with contextlib.redirect_stdout(io.StringIO()):
        cond, _, _, posts = crf.fit_all(d, feats, M, grid, Te_f, cfg, verbose=False)
    c = cond.iloc[0]
    fX, fN = grid
    chi2 = cdf.chi2_grid(posts[f"{sweep}|{P:g}"], c.chi2_min, c.birge)
    i, j = np.unravel_index(int(np.argmin(chi2)), chi2.shape)
    ok = chi2 <= chi2[i, j] + c.birge ** 2
    jn = int(np.argmin(np.abs(fN - np.log(N_C))))
    i_n = int(np.argmin(chi2[:, jn]))
    ok_n = chi2[:, jn] <= chi2[i_n, jn] + c.birge ** 2
    T2 = np.broadcast_to(Te_f[:, None], chi2.shape) if np.ndim(Te_f) == 1 else Te_f
    lev = list(tab["levels"])
    n1s5 = np.exp(crf.on_fine_grid(tab, np.log(np.clip(tab["dens"][:, :, lev.index("4s1")], 1e-300, None)), cfg, k=1))
    row = dict(n_spec=int(c.n_spec), n_points=int(c.n_points), n_feat=int(c.n_feat), dof=int(c.dof), birge=c.birge,
               EN_free=float(np.exp(fX[i])), EN_free_lo=float(np.exp(fX[ok.any(1)].min())),
               EN_free_hi=float(np.exp(fX[ok.any(1)].max())), Te_free=float(T2[i, j]),
               Ne_free=float(np.exp(fN[j])), Ne_free_lo=float(np.exp(fN[ok.any(0)].min())),
               Ne_free_hi=float(np.exp(fN[ok.any(0)].max())), Ne_at_edge=j >= len(fN) - 2,
               chi2_red_free=float(chi2[i, j]) / c.dof, n_1s5_free=float(n1s5[i, j]),
               n_1s5_free_lo=float(n1s5[ok].min()), n_1s5_free_hi=float(n1s5[ok].max()),
               EN_nc=float(np.exp(fX[i_n])), EN_nc_lo=float(np.exp(fX[ok_n].min())),
               EN_nc_hi=float(np.exp(fX[ok_n].max())), Te_nc=float(T2[i_n, jn]),
               chi2_red_nc=float(chi2[i_n, jn]) / (c.dof + 1), dchi2_nc=float(chi2[i_n, jn] - chi2[i, j]) / c.birge ** 2,
               n_1s5_nc=float(n1s5[i_n, jn]), n_1s5_nc_lo=float(n1s5[ok_n, jn].min()),
               n_1s5_nc_hi=float(n1s5[ok_n, jn].max()))
    res = S.pinned_residuals(d, feats, M, i_n, jn, cfg)
    return row, res, (tab, grid, i, j, i_n, jn)


def fit_set(name, ft, feats, cond):
    rows, res = [], []
    for c in cond.itertuples():
        d = ft[(ft.sweep == c.sweep) & np.isclose(ft.x, c.P)]
        if d.feature.nunique() < 3:
            continue
        row, r, _ = fit_one(d, feats, fit_cfg(c.key), c.sweep, c.P)
        rows.append(dict(set=name, sweep=c.sweep, P=c.P, N2=c.N2, Tg=c.Tg, las_1s5=c.las_1s5, **row))
        res.append(r.assign(set=name, sweep=c.sweep, P=c.P))
        print(f"  {name:<20s} {c.sweep:<8s} {c.P:3.0f} W (Tg {c.Tg:.0f} K): free E/N {row['EN_free']:6.1f} Td "
              f"Ne {row['Ne_free']:.1e} chi2/dof {row['chi2_red_free']:5.2f} | n_c: E/N {row['EN_nc']:6.1f} Td "
              f"Te_eff {row['Te_nc']:.2f} eV chi2/dof {row['chi2_red_nc']:5.2f} dchi2 {row['dchi2_nc']:5.1f} "
              f"1s5 {row['n_1s5_nc']:.1e} (LAS {c.las_1s5:.1e})", flush=True)
    return pd.DataFrame(rows), (pd.concat(res, ignore_index=True) if res else pd.DataFrame())


# ---- plots ------------------------------------------------------------------------------------
def plot_fits(fits, sweep, path):
    f = fits[fits.sweep == sweep]
    fig, axs = plt.subplots(2, 3, figsize=(17, 9.5))
    panels = (("EN_free", "E/N of the free fit [Td]", True), ("Ne_free", "N$_e$ of the free fit [m$^{-3}$]", True),
              ("chi2_red_free", "$\\chi^2$/dof (free fit)", False), ("EN_nc", "E/N at N$_e$ = n$_c$ [Td]", True),
              ("n_1s5_nc", "CR 1s$_5$ at N$_e$ = n$_c$ [m$^{-3}$]", True), ("dchi2_nc",
              "$\\Delta\\chi^2$/B$^2$ of pinning N$_e$ at n$_c$", False))
    for ax, (col, lab, logy) in zip(axs.flat, panels):
        for name, g in f.groupby("set", sort=False):
            g = g.sort_values("P")
            c, mk, ls = STYLE[name]
            lo, hi = (f"{col}_lo", f"{col}_hi") if f"{col}_lo" in g else (None, None)
            if lo:
                ax.fill_between(g.P, g[lo], g[hi], color=c, alpha=0.08, lw=0)
            ax.plot(g.P, g[col], marker=mk, ls=ls, color=c, ms=6, lw=1.4, label=name)
        if col == "Ne_free":
            ax.axhline(N_C, color="k", ls=":", lw=1, label="n$_c$ (2.42 GHz)")
        if col == "n_1s5_nc":
            las = f.drop_duplicates("P").sort_values("P")
            ax.plot(las.P, las.las_1s5, "D", color="C3", ms=8, label="LAS 1s$_5$")
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel(lab)
        ax.set_xlabel("Power [W]")
        ax.grid(alpha=0.3, which="both")
    axs[0, 0].legend(fontsize=7)
    axs[1, 1].legend(fontsize=7)
    fig.suptitle(f"CR fit, {qc.SWEEPS[sweep]['label']}, 1.5 Torr: QE Pro feature sets against the echelle "
                 f"(pure-Ar microwave EEDF, LAS-trend T$_g$, no tilt; bands: $\\Delta\\chi^2 \\leq$ B$^2$)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_residuals(res, name, sweep, path):
    r = res[(res.set == name) & (res.sweep == sweep)]
    if r.empty:
        return
    M = r.groupby(["feature", "P"]).resid.mean().unstack()
    M = M.loc[sorted(M.index, key=lambda s: float(s.split()[0]))]
    fig, ax = plt.subplots(figsize=(0.55 * M.shape[1] + 4.5, 0.32 * M.shape[0] + 2.2))
    v = max(0.3, np.nanmax(np.abs(M.values)))
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M.values[i, j]):
                ax.text(j, i, f"{M.values[i, j]:+.2f}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(M.shape[1]), [f"{P:g}" for P in M.columns])
    ax.set_yticks(range(M.shape[0]), M.index, fontsize=7)
    ax.set_xlabel("Power [W]")
    fig.colorbar(im, ax=ax, label="ln(measured / model) at the fit at n$_c$ (after the scale)")
    ax.set_title(f"{name}, {qc.SWEEPS[sweep]['label']}: line residuals at N$_e$ = n$_c$", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_spectrum(sweep, P, fits, feats, ref, cal, cond, path, set_name="qepro_split"):
    """Measured mean QE Pro spectrum (response-corrected) against the CR spectrum at the fit at n_c and the free fit,
    convolved with the LSF, scaled to the measured feature areas."""
    wl_cal, p, R = cal
    qm = mean_spectra(qc.load_qepro(sweep))
    k = int(np.flatnonzero(np.isclose(qm["files"].P, P))[0])
    y = qm["spec"][k] / R(wl_cal)
    c = cond[(cond.sweep == sweep) & np.isclose(cond.P, P)].iloc[0]
    cfg = fit_cfg(c.key)
    tab = table(cfg)
    f = fits[(fits.set == set_name) & (fits.sweep == sweep) & np.isclose(fits.P, P)]
    if f.empty:
        return
    f = f.iloc[0]
    edges = qc.pixel_edges(wl_cal)
    fig, axs = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw=dict(height_ratios=[1, 1]))
    for (EN, Ne, lab, col) in ((f.EN_nc, N_C, f"n$_c$: E/N {f.EN_nc:.1f} Td", "C3"),
                               (f.EN_free, f.Ne_free, f"free: E/N {f.EN_free:.1f} Td, N$_e$ {f.Ne_free:.1e}", "C2")):
        wl_l, I_l = L.ov.model_lines_at(tab, EN, Ne, cfg)
        model = np.zeros_like(wl_cal)
        for w, I in zip(wl_l, I_l):
            if 380 < w < 1000 and I > 0:
                model += I * qc.line_profile(edges, w, p)
        sel = (wl_cal > 690) & (wl_cal < 990)
        s = np.nansum(y[sel] * model[sel]) / np.nansum(model[sel] ** 2)
        for ax in axs:
            ax.plot(wl_cal, s * model, "-", color=col, lw=1.0, label=f"CR model ({lab})")
    for ax, (a, b) in zip(axs, ((390, 475), (690, 990))):
        m = (wl_cal > a) & (wl_cal < b)
        ax.plot(wl_cal[m], y[m], "o", color="k", ms=2.5, label="QE Pro (response-corrected)")
        ax.set_xlim(a, b)
        ax.set_yscale("log")
        top = np.nanmax(y[m])
        ax.set_ylim(top * 3e-4, top * 2)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    axs[1].set_xlabel("Wavelength [nm]")
    axs[0].set_title(f"QE Pro {qc.SWEEPS[sweep]['label']}, {P:g} W, 1.5 Torr: measured against the CR spectrum "
                     f"(only CR-model lines, LSF-convolved, one scale from 690-990 nm)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# ---- driver -----------------------------------------------------------------------------------
def main():
    os.makedirs(FIGDIR, exist_ok=True)
    cond = conditions()
    print(cond.to_string(index=False), flush=True)
    with A.xsec_set(A.CR_XSEC):
        missing = sorted({k for k in cond.key if not crf.table_is_current(fit_cfg(k))})
    if missing:
        A.build_all([(k, "ar") for k in missing])
    ref = qc.reference_lines()
    cal = qc.load_calibration()
    sets = qepro_features(ref)
    fits, res = [], []
    for name in QE_SETS:
        feats = sets[name]
        ft = pd.concat([qepro_feature_table(sw, feats, ref, cal) for sw in qc.SWEEPS], ignore_index=True)
        ft = ft[(ft.sweep == "pure_Ar") | (ft.wl > 600)]    # 5p windows of the Ar/N2 spectra are N2 bands
        ft.to_csv(os.path.join(OUTDIR, f"features_{name}.csv"), index=False)
        print(f"\n{name}: {ft.feature.nunique()} features", flush=True)
        f, r = fit_set(name, ft, feats, cond)
        fits.append(f)
        res.append(r)
    for sw in [sw for sw in qc.SWEEPS if has_echelle(sw)]:
        c = cond[cond.sweep == sw]
        e_feats, ft_all, ft_ok = echelle_feature_table(c, sw)
        sets["echelle_9lines"] = e_feats
        # like for like: the qepro_merged groups on both instruments, only where both have them
        q_ft = pd.read_csv(os.path.join(OUTDIR, "features_qepro_merged.csv"))
        q_ft = q_ft[q_ft.sweep == sw]
        if qc.SWEEPS[sw]["N2"] > 0:
            q_ft = q_ft[q_ft.wl > 600]
        q_common, e_common = common_tables(q_ft, echelle_group_table(sw, sets["qepro_merged"]).query("sweep == @sw"),
                                           sets["qepro_merged"])
        todo = [("echelle_all", ft_all, e_feats), ("echelle_unsaturated", ft_ok, e_feats),
                ("echelle_common", e_common, sets["qepro_merged"]), ("qepro_common", q_common, sets["qepro_merged"])]
        bf_path = os.path.join(qb_outdir(), "features_bandfree.csv")
        if qc.SWEEPS[sw]["N2"] > 0 and os.path.exists(bf_path):
            # with the N2 bands removed (QEProBandModel): 5p groups from the band + line fit, red groups as above
            bf = pd.read_csv(bf_path)
            bf = bf[(bf.sweep == sw) & (bf.area > 0)].assign(rel_err=lambda t: t.err / t.area)
            bf = bf[bf.rel_err <= MAX_REL_ERR][["sweep", "file", "x", "rep", "feature", "wl", "area", "rel_err"]]
            q_bf = pd.concat([bf, q_ft[q_ft.wl > 600][bf.columns]], ignore_index=True)
            e_all = echelle_group_table(sw, sets["qepro_merged"]).query("sweep == @sw")
            q_bfc, e_bfc = common_tables(q_bf, e_all, sets["qepro_merged"])
            todo += [("echelle_common_5p", e_bfc, sets["qepro_merged"]),
                     ("qepro_bandfree_common", q_bfc, sets["qepro_merged"])]
        for name, ft, feats in todo:
            ft.to_csv(os.path.join(OUTDIR, f"features_{name}_{sw}.csv"), index=False)
            print(f"\n{name} ({sw}): {len(ft)} points, {ft.feature.nunique()} features", flush=True)
            f, r = fit_set(name, ft, feats, c)
            fits.append(f)
            res.append(r)
    fits = pd.concat(fits, ignore_index=True)
    res = pd.concat(res, ignore_index=True)
    fits["ratio_1s5_nc_las"] = fits.n_1s5_nc / fits.las_1s5
    fits.to_csv(os.path.join(OUTDIR, "fits.csv"), index=False)
    res.to_csv(os.path.join(OUTDIR, "residuals.csv"), index=False)
    for sw in qc.SWEEPS:
        plot_fits(fits, sw, os.path.join(FIGDIR, f"fits_{sw}.png"))
        for name in QE_SETS + EC_SETS:
            plot_residuals(res, name, sw, os.path.join(FIGDIR, f"residuals_{name}_{sw}.png"))
        for P in (10.0, 50.0, 90.0):
            plot_spectrum(sw, P, fits, sets["qepro_split"], ref, cal, cond,
                          os.path.join(FIGDIR, f"spectrum_{sw}_{P:g}W.png"),
                          set_name="qepro_split" if sw == "pure_Ar" else "qepro_red")
    print("\nwritten to", OUTDIR)
    return dict(cond=cond, fits=fits, res=res, sets=sets)


if __name__ == "__main__":
    RES = main()
