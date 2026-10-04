# -*- coding: utf-8 -*-
"""
CRFitSpectrumOverlay.py

Synthetic CR-model spectrum at the best-fit (Te_eff, Ne) of one sweep condition,
overlaid on the measured spectrum.

1. CRFitNeTe.run() -> joint Te/Ne fit of every condition (uses its caches).
2. CR-model line intensities of all Ar I lines between model levels at the
   best-fit grid point (spline of ln I over ln x - ln Ne, as in the fit).
3. Same nuisances as the fit, evaluated at that point: one scale per spectrum
   and the linear ln R(lambda) response; the synthetic spectrum uses the mean
   scale of the repeats.
4. Sticks broadened with the measured echelle slit function
   (HelperFunctions.BroadenWithSlit) and plotted over the repeat-averaged,
   baseline-subtracted measurement.

Run from Spyder: edit SWEEP / X, press F5.  Figure in Experimental_Data/Output/CRFit.
"""
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import minimum_filter1d, uniform_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "Experimental_Data", "ExperimentalDataAnalysis"))
import CRFitNeTe as crf                             # noqa: E402

SWEEP = "N2 fraction"           # sweep name in crf.CONFIG["sweeps"]
X = 0                         # condition within the sweep (0 % N2 = pure Ar)
PANELS = [(400.0, 475.0), (690.0, 860.0)]          # 5p -> 4s and 4p -> 4s ranges
BASELINE_NM = 1.5               # window of the rolling-minimum baseline
# Check lines (lines between model levels that are not in the fit) are shown only when the
# measurement is trustworthy, judged from the single-spectrum line analysis (ArN2_out:
# lines_pure.csv for pure Ar, lines_mix.csv for the mixtures) and the manual ratings.
CHECK = dict(
    min_frac=0.005,             # model intensity / strongest model line of its panel
    iso_frac=0.10,              # other model lines in the window: below this x the line
    min_rating=4,               # manual rating (5 clean ... 1 junk); unrated lines are left out
    status=("ok",),             # detected above the quantification SNR, not blended or saturated
    contamination=("clean", "clean?"),   # mixture spectra (lines_mix.csv) only
    width_ratio=(0.8, 1.2),     # free single-line FWHM / group (instrument) FWHM: broadening, hidden blend
    max_resid_peaks=0,          # unexplained residual peaks within 3 FWHM of the line
)


def model_lines_at(tab, x, Ne, cfg):
    """(wavelength nm, intensity) of every model line at (x, Ne), in the fit's units."""
    lx, ln = np.log(tab["x_grid"]), np.log(tab["Ne_grid"])
    energy = cfg["intensity_units"] == "energy"
    I = np.empty(len(tab["line_wl"]))
    for k, wl in enumerate(tab["line_wl"]):
        v = tab["I_obs"][:, :, k] * (1.0 / wl if energy else 1.0)
        if not np.all(v > 0):
            I[k] = 0.0
            continue
        spl = RectBivariateSpline(lx, ln, np.log(v), kx=min(3, len(lx) - 1), ky=3)
        I[k] = np.exp(spl(np.log(x), np.log(Ne))[0, 0])
    return tab["line_wl"], I


def nuisances(d, feats, wl_m, I_m, cfg):
    """Per-spectrum ln scale and response slope at the best fit (generalized LS, as fit_block)."""
    fwl = dict(zip(feats.feature, feats.wl))
    comp_wl = {ft.feature: [w for w, _, _ in ft.comps] for ft in feats.itertuples()}
    lnM = {f: np.log(sum(I_m[np.argmin(np.abs(wl_m - w))] for w in ws)) for f, ws in comp_wl.items()}
    li = pd.factorize(d.feature)[0]
    si, files = pd.factorize(d.file)
    lam = d.feature.map(fwl).to_numpy()
    y = np.log(d.area.to_numpy()) - d.feature.map(lnM).to_numpy()
    X = [np.eye(si.max() + 1)[si]]
    if cfg["response_deg"]:
        X += [((lam - 775.0) / 100.0)[:, None] ** k for k in range(1, cfg["response_deg"] + 1)]
    X = np.hstack(X)
    C = np.diag(d.rel_err.to_numpy() ** 2 + cfg["sigma_fit_floor"] ** 2)
    C += cfg["sigma_model"] ** 2 * (li[:, None] == li[None, :])
    W = np.linalg.inv(C)
    beta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
    n = si.max() + 1
    return dict(zip(files, beta[:n])), beta[n:]


def response(lam, coef):
    return np.exp(sum(c * ((lam - 775.0) / 100.0) ** (k + 1) for k, c in enumerate(coef)))


def measured_spectrum(files, folder):
    """Repeat-averaged spectrum with a rolling-minimum baseline removed; order gaps -> NaN."""
    he = crf._helpers()
    specs = []
    for f in files:
        d = np.loadtxt(os.path.join(folder, f + ".spa"), skiprows=1)
        o = np.argsort(d[:, 0])
        specs.append((d[o, 0], d[o, 1]))
    wl = specs[0][0]
    I = np.mean([np.interp(wl, w, i) for w, i in specs], axis=0)
    gap = I == 0
    step = np.median(np.diff(wl))
    n = max(int(BASELINE_NM / step), 3)
    base = uniform_filter1d(minimum_filter1d(I, n), n)
    I = I - base
    I[gap] = np.nan
    return wl, I


def fit_line_residuals(d, feats, wl_m, I_m, scales, coef):
    """ln(measured / model) of every fitted feature in every repeat, with the
    fit's own nuisances (spectrum scale, response) applied to the model."""
    comp_wl = {ft.feature: [w for w, _, _ in ft.comps] for ft in feats.itertuples()}
    lnM = {f: np.log(sum(I_m[np.argmin(np.abs(wl_m - w))] for w in ws)) for f, ws in comp_wl.items()}
    model = d.feature.map(lnM) + d.file.map(scales) + np.log(response(d.wl.to_numpy(), coef))
    return d.assign(resid=np.log(d.area) - model)


def _half_window(lam):
    return 2.5 * 0.030 * lam / 435.833            # +-2.5 slit FWHM (~30 pm at 435.8 nm, ~ lambda)


def check_line_candidates(wl_m, I_syn, fit_wl, sweep, x, cfg, check=CHECK):
    """Model lines outside the fit, each with the verdict of the line analysis:
    catalogue wavelength, upper level, keep (bool) and the reasons it was dropped."""
    sel = cfg["selection_dir"]
    pure = sweep == "N2 fraction" and x == 0
    cat = pd.read_csv(os.path.join(sel, "lines_pure.csv" if pure else cfg["lines_file"]))
    ratings = pd.read_csv(os.path.join(sel, cfg["ar_ratings"]))
    rmap = dict(zip(ratings.key, ratings.rating))
    det = cat[cat.status.isin(("ok", "weak", "blend", "saturated"))]
    ar = cat[cat.species == "Ar I"].reset_index(drop=True)
    excl = np.asarray(cfg.get("exclude_wl", ()), float)
    rows = []
    for lo, hi in PANELS:
        p = (wl_m >= lo) & (wl_m <= hi) & (I_syn > 0)
        if not p.any():
            continue
        for k in np.flatnonzero(p & (I_syn > check["min_frac"] * I_syn[p].max())):
            lam, hw = wl_m[k], _half_window(wl_m[k])
            if np.min(np.abs(fit_wl - lam)) < hw:
                continue                                  # part of a fitted feature
            if I_syn[np.abs(wl_m - lam) < hw].sum() - I_syn[k] > check["iso_frac"] * I_syn[k]:
                continue                                  # blended with another model line
            r = ar.iloc[int(np.argmin(np.abs(ar.wl_air.to_numpy() - lam)))]
            if abs(r.wl_air - lam) > 0.01:
                rows.append(dict(wl_model=lam, wl_air=np.nan, upper="", keep=False, reason="not in the line catalogue"))
                continue
            rating = rmap.get(crf.als._key("Ar I", r.lower, r.upper, r.wl_air), np.nan)
            wr = r.fwhm_free / r.fwhm_group if np.isfinite(r.fwhm_free) else np.nan
            why = []
            if r.status not in check["status"]:
                why.append(f"status {r.status}")
            if not rating >= check["min_rating"]:
                why.append(f"rating {rating:g}" if np.isfinite(rating) else "not rated")
            if not pure and r.contamination not in check["contamination"]:
                why.append(str(r.contamination))
            if not check["width_ratio"][0] <= wr <= check["width_ratio"][1]:
                why.append(f"width x{wr:.2f}" if np.isfinite(wr) else "no free width fit")
            if r.n_resid_peaks > check["max_resid_peaks"]:
                why.append(f"{int(r.n_resid_peaks)} residual peak(s) nearby")
            nb = det[(np.abs(det.wl_air - r.wl_air) < hw) & (np.abs(det.wl_air - r.wl_air) > 1e-4)]
            if len(nb):
                why.append("detected line in window: " + ", ".join(f"{s} {w:.3f}" for s, w in zip(nb.species, nb.wl_air)))
            if len(excl) and np.min(np.abs(excl - r.wl_air)) < 0.01:
                why.append("excluded by hand (exclude_wl)")
            rows.append(dict(wl_model=lam, wl_air=r.wl_air, upper=r.upper, lower=r.lower, rating=rating,
                             width_ratio=wr, keep=not why, reason="; ".join(why)))
    return pd.DataFrame(rows)


def confident_check_residuals(wl_m, I_m, scales, coef, fit_wl, sweep, x, cfg, check=CHECK):
    """ln(measured / model) of the trustworthy check lines in every repeat.  The areas come
    from the same group fitter as the fitted lines (ActinometryLineSelection.measure_sweeps,
    cached as check_line_measurements.csv); the model gets the fit's nuisances (spectrum
    scale, response).  Returns (residuals, candidate table with the reasons)."""
    s = np.exp(np.mean(list(scales.values())))
    cand = check_line_candidates(wl_m, s * I_m * response(wl_m, coef), fit_wl, sweep, x, cfg, check)
    empty = pd.DataFrame(columns=["wl", "upper", "file", "resid", "rel_err"])
    keep = cand[cand.keep] if len(cand) else cand
    if keep.empty:
        return empty, cand
    mcfg = dict(crf.als.CONFIG, sweeps=[sw for sw in cfg["sweeps"] if sw["name"] == sweep],
                window_nm=cfg["window_nm"], reuse_measurements=cfg["reuse_measurements"],
                outdir=cfg.get("measure_dir") or cfg["outdir"], measure_cache="check_line_measurements.csv",
                export_flags=cfg["export_flags"])
    m = crf.als.filter_export_flags(crf.als.measure_sweeps(keep.assign(species="Ar I")[["species", "wl_air"]], mcfg), mcfg)
    m = m[(m.sweep == sweep) & (m.x == x) & m.status.isin(check["status"]) & m.file.isin(list(scales))]
    m = m[1 / m.snr <= cfg["max_rel_err"]]
    rows = []
    for r in m.itertuples():
        k = int(np.argmin(np.abs(wl_m - r.wl_air)))
        model = np.log(I_m[k]) + scales[r.file] + np.log(response(np.array([r.wl_air]), coef))[0]
        rows.append(dict(wl=r.wl_air, upper=r.upper, file=r.file, resid=np.log(r.area) - model, rel_err=1 / r.snr))
    return (pd.DataFrame(rows) if rows else empty), cand


def check_line_residuals(wl_m, I_syn, wl_meas, I_meas, fit_wl, min_frac=0.005, iso_frac=0.10):
    """Lines NOT in the fit: integral of the measured (repeat-mean) spectrum over
    +-2.5 slit FWHM around each isolated model line vs the model stick area.
    No check of the measurement itself (detection, rating, width, neighbours) -
    see confident_check_residuals; kept for EEDFPhysicsFitComparison's 5p columns."""
    rows = []
    for lo, hi in PANELS:
        p = (wl_m >= lo) & (wl_m <= hi) & (I_syn > 0)
        if not p.any():
            continue
        cut = min_frac * I_syn[p].max()
        for k in np.flatnonzero(p & (I_syn > cut)):
            lam = wl_m[k]
            hw = 2.5 * 0.030 * lam / 435.833            # slit FWHM ~30 pm at 435.8 nm, ~ lambda
            if np.min(np.abs(fit_wl - lam)) < hw:
                continue
            near = np.abs(wl_m - lam) < hw
            if I_syn[near].sum() - I_syn[k] > iso_frac * I_syn[k]:
                continue                                  # blended with another model line
            w = np.abs(wl_meas - lam) < hw
            if w.sum() < 3 or np.isnan(I_meas[w]).any():
                continue                                  # order gap
            area = np.trapezoid(I_meas[w], wl_meas[w])
            if area > 0:
                rows.append(dict(wl=lam, area=area, model=I_syn[k], resid=np.log(area / I_syn[k])))
    return pd.DataFrame(rows)


def plot_line_residuals(res_fit, res_chk, feats, c, cfg, label, path):
    fig, axs = plt.subplots(1, len(PANELS), figsize=(13, 5), sharey=True,
                            gridspec_kw=dict(width_ratios=[hi - lo for lo, hi in PANELS], wspace=0.04))
    sm = cfg["sigma_model"]
    err = np.sqrt(res_fit.rel_err ** 2 + cfg["sigma_fit_floor"] ** 2)
    for ax, (lo, hi) in zip(axs, PANELS):
        ax.axhspan(-sm, sm, color="C3", alpha=0.08, lw=0)
        ax.axhline(0, color="0.3", lw=1)
        if not res_chk.empty:
            r = res_chk[(res_chk.wl >= lo) & (res_chk.wl <= hi)]
            # as for the fitted lines: 1/SNR, the fit floor and the line's repeat-to-repeat scatter
            rs = r.groupby("wl").resid.transform(lambda v: v.std(ddof=1) if len(v) > 1 else 0.0)
            ax.errorbar(r.wl, r.resid, yerr=np.sqrt(r.rel_err ** 2 + cfg["sigma_fit_floor"] ** 2 + rs ** 2), fmt="o",
                        mfc="none", mec="0.45", ecolor="0.6", ms=6, mew=1.1, capsize=2, lw=1)
            for (w, up), g in r.groupby(["wl", "upper"]):
                ax.annotate(up, (w, g.resid.max()), xytext=(4, 6), textcoords="offset points", fontsize=8, color="0.4")
        for (f, g), mk in zip(res_fit.groupby("feature", sort=False), "osD^v<>ph*"):
            g = g[(g.wl >= lo) & (g.wl <= hi)]
            if g.empty:
                continue
            ax.errorbar(g.wl, g.resid, yerr=err[g.index], fmt="o", color="C0", ms=7, capsize=3, lw=1.2)
            ax.annotate(f.split()[1].split("-")[0], (g.wl.iloc[0], g.resid.max()), xytext=(4, 6),
                        textcoords="offset points", fontsize=9, color="C0")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("Wavelength (nm)")
        ax.grid(alpha=0.25)
    for ax in axs[1:]:
        ax.tick_params(left=False)
        ax.spines["left"].set_visible(False)
    for ax in axs[:-1]:
        ax.spines["right"].set_visible(False)
    axs[0].set_ylabel("ln(measured / model)")
    axs[0].errorbar([], [], yerr=[], fmt="o", color="C0", label="Fitted lines (each repeat)")
    axs[0].plot([], [], "o", mfc="none", mec="0.45",
                label=f"Check lines, not fitted (each repeat; rated ≥{CHECK['min_rating']}, unbroadened, isolated)")
    axs[0].fill_between([], [], color="C3", alpha=0.15, label=rf"$\pm\sigma_{{model}}$ = {100 * sm:.0f} %")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9, frameon=False)
    fig.suptitle(rf"{label}: line-integral residuals at $T_{{e,\mathrm{{eff}}}}$ = {c.Te_best:.2f} eV, "
                 rf"$N_e$ = {c.Ne_best:.2e} m$^{{-3}}$   ($\chi^2$/dof = {c.chi2_red:.2f})", fontsize=12)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()


def run(sweep=SWEEP, x=X, cfg=crf.CONFIG):
    FIT = crf.run(cfg)
    tab, feats, ft = FIT["tab"], FIT["feats"], FIT["ft"]
    c = FIT["cond"].query("sweep == @sweep and x == @x").iloc[0]
    d = ft.query("sweep == @sweep and x == @x")
    folder = next(sw["folder"] for sw in cfg["sweeps"] if sw["name"] == sweep)

    wl_m, I_m = model_lines_at(tab, c.x_best, c.Ne_best, cfg)
    scales, coef = nuisances(d, feats, wl_m, I_m, cfg)
    s = np.exp(np.mean(list(scales.values())))
    I_syn = s * I_m * response(wl_m, coef)

    he = crf._helpers()
    wl_meas, I_meas = measured_spectrum(sorted(scales), folder)
    fit_wl = feats.wl.to_numpy()

    fig, axs = plt.subplots(len(PANELS), 1, figsize=(13, 4.2 * len(PANELS)))
    for ax, (lo, hi) in zip(np.atleast_1d(axs), PANELS):
        m = (wl_m > lo - 1) & (wl_m < hi + 1) & (I_syn > 0)
        xs, ys = he.BroadenWithSlit(wl_m[m], I_syn[m])
        mm = (wl_meas >= lo) & (wl_meas <= hi)
        ax.plot(wl_meas[mm], I_meas[mm], color="0.25", lw=0.9, label="Measured (mean of repeats)")
        ax.plot(xs, ys, color="C3", lw=1.1, alpha=0.85, label="CR model, slit-broadened")
        top = np.nanmax(I_meas[mm]) * 1.08
        for w in fit_wl[(fit_wl >= lo) & (fit_wl <= hi)]:
            ax.axvline(w, color="C0", lw=0.8, ls=":", alpha=0.7)
        ax.plot([], [], color="C0", ls=":", label="Lines used in the fit")
        # echelle order gaps (exported as zeros); above ~690 nm they hide 696.5, 738.4, 763.5, 811.5, 842.5 nm
        gap = np.isnan(I_meas) & mm
        edges = np.flatnonzero(np.diff(np.r_[0, gap.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            ax.axvspan(wl_meas[a], wl_meas[b - 1], color="0.85", lw=0, zorder=0)
        ax.fill_between([], [], color="0.85", label="No measured data (echelle order gap)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.03 * top, top)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (calibrated, arb.)")
        ax.grid(alpha=0.25)
    for ax in np.atleast_1d(axs):
        ax.legend(loc="upper left", fontsize=9)
    label = "Pure Ar" if (sweep == "N2 fraction" and x == 0) else f"{sweep} = {x:g}"
    fig.suptitle(
        rf"{label}:  $T_{{e,\mathrm{{eff}}}}$ = {c.Te_best:.2f} eV "
        rf"(68 %: {c.Te_lo:.2f}-{c.Te_hi:.2f}),   "
        rf"$N_e$ = {c.Ne_best:.2e} m$^{{-3}}$ (68 %: {c.Ne_lo:.1e}-{c.Ne_hi:.1e})",
        fontsize=13)
    fig.tight_layout()
    tag = f"{sweep.replace(' ', '_')}_{x:g}"
    path = os.path.join(cfg["outdir"], f"spectrum_overlay_{tag}.png")
    fig.savefig(path, dpi=200)
    plt.show()
    print(f"\nbest fit: Te_eff = {c.Te_best:.2f} eV, Ne = {c.Ne_best:.3e} m^-3, "
          f"chi2/dof = {c.chi2_red:.2f};  response slope {coef}  ->  {path}")

    res_fit = fit_line_residuals(d, feats, wl_m, I_m, scales, coef)
    res_chk, cand = confident_check_residuals(wl_m, I_m, scales, coef, fit_wl, sweep, x, cfg)
    rpath = os.path.join(cfg["outdir"], f"line_residuals_{tag}.png")
    plot_line_residuals(res_fit, res_chk, feats, c, cfg, label, rpath)
    print("\nfitted lines, mean ln(measured/model):")
    print(res_fit.groupby("feature", sort=False).resid.mean().round(3).to_string())
    if len(cand):
        print("\ncheck-line candidates (model lines outside the fit):")
        print(cand.round(3).to_string(index=False))
        cand.to_csv(os.path.join(cfg["outdir"], f"check_lines_{tag}.csv"), index=False)
    if not res_chk.empty:
        g = res_chk.groupby(["wl", "upper"]).resid.agg(["mean", "std", "count"]).round(3)
        print(f"\n{g.shape[0]} confident check lines, mean ln(meas/model) per line:")
        print(g.to_string())
    print(f"-> {rpath}")
    return dict(FIT=FIT, cond=c, wl_model=wl_m, I_model=I_syn, wl_meas=wl_meas, I_meas=I_meas,
                res_fit=res_fit, res_chk=res_chk)


if __name__ == "__main__":
    OUT = run()
