# -*- coding: utf-8 -*-
"""
ArLineCloseups.py

Close-ups of the Ar I lines the CR fit uses (CRFitNeTe.select_features), in pure Ar and in the
Ar/N2 spectra, to check N2 band overlap and the fit windows.

How a line is measured (Argon_Nitrogen_Mix_Analysis_V2.measure_lines, through
ActinometryLineSelection.measure_sweeps): the catalogue lines (Ar I and N I) within
window_nm of a selected line are grouped when closer than cluster_sep_fwhm x FWHM; each group
is fitted over [first line - window_pad_fwhm x FWHM, last line + window_pad_fwhm x FWHM] with
one Gaussian per catalogue line (common shift and width from a grid) plus a linear baseline,
and the line area is the area of its Gaussian.  So a smooth band level under a line goes into
the baseline, but band structure inside the window (N2 rotational lines) is not modelled.

Per feature, for the spectra in SPECTRA (repeat 1 of each condition):
  - the spectrum around the line, normalised to the line's fitted peak, log scale
  - the fit window (shaded), and the fitted model and its baseline for the last spectrum
  - catalogue lines (ticks: red = the feature's lines, grey = other Ar I, purple = N I) and the
    N2 band regions of the band-head catalogue (orange bands, labelled)
  - title: feature, grade (Ar_manual_ratings.csv), automatic flags, repeat-to-repeat scatter of
    the CR fit; box: per spectrum the baseline under the line (pedestal / peak) and the fit
    residual (sigma / peak)

Output in Experimental_Data/Output/ArLineCloseups/: ar_line_closeups.png (all features),
closeup_<wl>.png (one per feature), line_checks.csv (window bounds and metrics).
Run from Spyder (F5) or python SmallAnalysisScripts/ArLineCloseups.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT_DIR, "Experimental_Data", "ExperimentalDataAnalysis"))
import CRFitNeTe as crf                             # noqa: E402
import ActinometryLineSelection as als              # noqa: E402

arn = als.arn

# ---- settings -------------------------------------------------------------------
SPECTRA = [   # label, sweep, file, colour
    ("pure Ar", "N2 fraction", "Percent0_1", "k"),
    ("0.5 % N$_2$", "N2 fraction", "Percent0_5__1", "C0"),
    ("1 % N$_2$", "N2 fraction", "Percent1_0__1", "C2"),
    ("3 % N$_2$", "N2 fraction", "Percent3_0__1", "C3"),
    ("2.4 % N$_2$ (power sweep, 85 W)", "Power", "85W_1", "C1"),
]
HALF_RANGE_NM = 0.45                                # close-up half width
YLIM = (3e-4, 2.0)                                  # normalised intensity (log)
CFG = crf.CONFIG                                    # line selection of the CR fit
OUTDIR = os.path.join(ROOT_DIR, "Experimental_Data", "Output", "ArLineCloseups")


# ---- data -----------------------------------------------------------------------
def fit_features():
    """The CR-fit features with their components, grades and flags, and the repeat scatter."""
    tab = {"line_upper": [], "line_lower": []}
    cr, he, MD, _ = crf.load_cr_model(OUTDIR)
    lines = crf._model_lines(MD)
    tab = dict(line_upper=np.array([l[0] for l in lines]), line_lower=np.array([l[1] for l in lines]))
    feats, comps = crf.select_features(tab, CFG)
    ft = crf.feature_table(crf.measure(comps, CFG), feats, CFG)
    feats["repeat_scatter"] = feats.feature.map(crf.repeat_scatter(ft))
    return feats, comps


def spectrum(sweep, name):
    folder = next(s["folder"] for s in CFG["sweeps"] if s["name"] == sweep)
    return arn.load_spectrum(os.path.join(folder, name + ".spa"), name, arn.CONFIG)


def group_fit(sp, cat, cat_ar, feature_wl):
    """The group fit holding the feature's lines in spectrum sp: (fit dict, results rows of the
    feature's lines, all catalogue lines of the group)."""
    try:
        fwhm_fn, offset_fn, _ = arn.measure_instrument_function(sp, cat_ar, arn.CONFIG)
        off = float(offset_fn(0))
    except RuntimeError:
        fwhm_fn, off = (lambda lam: lam / 20000.0), 0.0
    near = np.zeros(len(cat), bool)
    for w in feature_wl:
        near |= np.abs(cat.wl_air.values - w) < CFG.get("window_nm", 1.5)
    res, fits = arn.measure_lines(sp, cat[near], fwhm_fn, arn.CONFIG, sp.name, off)
    rows = res[(res.species == "Ar I") & np.isin(res.wl_air.round(4), np.round(feature_wl, 4))]
    gid = rows.group.dropna().astype(int).mode()
    if gid.empty or int(gid.iloc[0]) not in fits:
        return None, rows, None
    g = int(gid.iloc[0])
    return fits[g], rows, res[res.group == g]


# ---- plots ----------------------------------------------------------------------
def draw(ax, ft, comps_f, cat, bands, data):
    c = ft.wl
    lo, hi = c - HALF_RANGE_NM, c + HALF_RANGE_NM
    for b in bands[(bands.lo_nm <= hi) & (bands.hi_nm >= lo)].itertuples():
        ax.axvspan(max(b.lo_nm, lo), min(b.hi_nm, hi), color="orange", alpha=0.10, lw=0)
        ax.text(min(max(b.head_nm, lo), hi), YLIM[1] * 0.9, f"{b.system} ({b.vu},{b.vl})", rotation=90,
                fontsize=6, color="darkorange", va="top", ha="center")
    other = cat[(cat.wl_air > lo) & (cat.wl_air < hi)]
    for r in other.itertuples():
        own = np.isclose(r.wl_air, comps_f.wl_air.to_numpy(), atol=5e-4).any()
        col = "C3" if own else ("purple" if r.species == "N I" else "0.6")
        ax.plot([r.wl_air] * 2, [YLIM[0], YLIM[0] * (6 if own else 3)], color=col, lw=1.5 if own else 1)
    box = []
    last = None
    for label, color, d in data:
        if d is None:
            continue
        sp, fit, rows, grp = d
        peak = float(np.nanmax(rows.height)) if len(rows) else np.nan
        if not np.isfinite(peak) or peak <= 0:
            continue
        x, y = sp.window(lo, hi)
        ax.plot(x, np.clip(y / peak, YLIM[0] / 10, None), "-", color=color, lw=1.1, label=label)
        if fit is not None:
            ped = float(np.nanmax(rows.pedestal_frac))
            box.append(f"{label.split(' (')[0]:>10s}: base {ped:+.3f}, resid {fit['sigma'] / peak:.3f}")
            last = (fit, peak, color)
    if last is not None:
        fit, peak, color = last
        ax.axvspan(fit["x"].min(), fit["x"].max(), color="0.5", alpha=0.12, lw=0)
        base = fit["coef"][-2] + fit["coef"][-1] * (fit["x"] - fit["x"].mean())
        ax.plot(fit["x"], fit["model"] / peak, "--", color="k", lw=1)
        ax.plot(fit["x"], np.clip(base / peak, YLIM[0], None), ":", color="k", lw=1)
    grade = comps_f.rating.dropna()
    grade = "/".join(f"{g:g}" for g in grade) if len(grade) else "not graded"
    flags = ", ".join(sorted(set(comps_f.status.astype(str)) | set(comps_f.contamination.astype(str))))
    ax.set_title(f"{ft.feature}  |  grade {grade}  |  {flags}  |  repeat scatter {ft.repeat_scatter:.2f}",
                 fontsize=8)
    ax.text(0.01, 0.02, "\n".join(box), transform=ax.transAxes, fontsize=6, family="monospace", va="bottom",
            bbox=dict(facecolor="white", alpha=0.85, lw=0))
    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("wavelength (air) [nm]", fontsize=8)
    ax.set_ylabel("I / fitted line peak", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, which="both")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    feats, comps = fit_features()
    cat, cat_ar = als._catalogue()
    bands = arn.n2_band_catalogue()
    spectra = {name: spectrum(sweep, name) for _, sweep, name, _ in SPECTRA}
    rows, panels = [], []
    for ft in feats.itertuples():
        comps_f = comps[comps.feature == ft.feature]
        wl = comps_f.wl_air.to_numpy()
        data = []
        for label, sweep, name, color in SPECTRA:
            fit, r, grp = group_fit(spectra[name], cat, cat_ar, wl)
            data.append((label, color, (spectra[name], fit, r, grp)))
            if fit is not None and len(r):
                peak = float(np.nanmax(r.height))
                rows.append(dict(feature=ft.feature, spectrum=name, window_lo_nm=fit["x"].min(),
                                 window_hi_nm=fit["x"].max(), fwhm_nm=fit["fwhm"],
                                 window_width_fwhm=(fit["x"].max() - fit["x"].min()) / fit["fwhm"],
                                 group_lines=";".join(f"{s} {w:.3f}" for s, w in zip(grp.species, grp.wl_air)),
                                 status=";".join(r.status.astype(str)), height=peak, area=float(r.area.sum()),
                                 pedestal_frac=float(np.nanmax(r.pedestal_frac)), resid_frac=fit["sigma"] / peak,
                                 n_resid_peaks=int(np.nansum(r.n_resid_peaks)),
                                 grade=";".join(f"{g:g}" for g in comps_f.rating.dropna()),
                                 bands=";".join(arn.bands_overlapping(bands, ft.wl, 2 * fit["fwhm"]))))
        panels.append((ft, comps_f, data))
    checks = pd.DataFrame(rows)
    checks.to_csv(os.path.join(OUTDIR, "line_checks.csv"), index=False)
    n = len(panels)
    ncol = 3
    fig, axs = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(6.2 * ncol, 4.3 * np.ceil(n / ncol)), squeeze=False)
    for ax, (ft, comps_f, data) in zip(axs.ravel(), panels):
        draw(ax, ft, comps_f, cat, bands, data)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    axs[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Ar I lines of the CR fit: spectra normalised to the fitted line peak (shaded grey: fit window; "
                 "dashed / dotted: fit and baseline of the last spectrum; orange: N$_2$ band regions)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "ar_line_closeups.png"), dpi=150)
    plt.close(fig)
    for ft, comps_f, data in panels:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        draw(ax, ft, comps_f, cat, bands, data)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"closeup_{ft.wl:.2f}.png"), dpi=170)
        plt.close(fig)
    piv = checks.pivot_table(index="feature", columns="spectrum", values=["pedestal_frac", "resid_frac"])
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(checks.drop_duplicates("feature")[["feature", "window_lo_nm", "window_hi_nm", "fwhm_nm",
                                                 "window_width_fwhm", "group_lines", "grade", "bands"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(piv.round(3).to_string())
    print(f"\nfigures in {OUTDIR}")
    return checks


if __name__ == "__main__":
    CHECKS = main()
