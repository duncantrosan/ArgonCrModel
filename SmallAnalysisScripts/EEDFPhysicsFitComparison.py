# -*- coding: utf-8 -*-
"""
EEDFPhysicsFitComparison.py

Does the EEDF physics fix the blue/red mismatch of the CR fit?  One condition
(default: pure Ar, 85 W, 1 Torr) is fitted (CRFitNeTe.fit_block) with each EEDF
library of Scripts/CreateEEDFLibraries.py, all on the same (E/N, Ne) grid, with the
linear response tilt (response_deg = 1) and with the calibration trusted (None):

  MultiBolt 6 terms / 2 terms, BOLSIG+, BOLSIG+ e-e, BOLSIG+ superelastic,
  BOLSIG+ e-e + superelastic

Reported per library and tilt setting:
  - best-fit Te_eff and Ne with 68 % ranges, chi2/dof, fitted tilt
  - 420.07 / 750.39 nm photon ratio (5p2 -> 4s1 over 4p10 -> 4s4) of the model at the
    best fit, against the measured ratio (window integrals of the repeat-mean spectrum)
  - the 5p check lines, not used in the fit: mean ln(measured / model) of 5p6, 5p8, 5p9
    (low before) and of the other 5p levels
  - CR-model 4s1 (1s5) density at the best fit, against the measured 1-8e17 m^-3

Output in Experimental_Data/Output/EEDFPhysics/: summary.csv, summary.png,
residuals_tilt.png, residuals_notilt.png.  The CR tables are cached there per library
(first run: ~3 min per library).  Run from Spyder: edit the settings, press F5.
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT_DIR, "Experimental_Data", "ExperimentalDataAnalysis"))
sys.path.insert(0, os.path.join(ROOT_DIR, "Scripts"))
sys.path.insert(0, HERE)
import CRFitNeTe as crf                             # noqa: E402
import CreateEEDFLibraries as libs                  # noqa: E402
import CRFitSpectrumOverlay as ov                   # noqa: E402

he = crf._helpers()

# ---- settings -------------------------------------------------------------------
SWEEP, X = "N2 fraction", 0.0                       # pure Ar, 85 W, 1 Torr
VARIANTS = [   # label, library folder, colour
    ("MultiBolt 6 terms", he.MULTIBOLT_FOLDER / "Ar_Biagi_lowEN_span12", "0.6"),
    ("MultiBolt 2 terms", he.MULTIBOLT_FOLDER / "Ar_Biagi_lowEN_span12_2term", "k"),
    ("BOLSIG+", he.BOLSIG_FOLDER / "Ar_Biagi_bolsig", "C0"),
    ("BOLSIG+ e-e", he.BOLSIG_FOLDER / "Ar_Biagi_bolsig_ee", "C2"),
    ("BOLSIG+ superelastic", he.BOLSIG_FOLDER / "Ar_Biagi_bolsig_se", "C1"),
    ("BOLSIG+ e-e + superelastic", he.BOLSIG_FOLDER / "Ar_Biagi_bolsig_ee_se", "C3"),
    ("BOLSIG+ microwave", he.BOLSIG_FOLDER / "Ar_Biagi_bolsig_mw", "C4"),
    ("BOLSIG+ microwave e-e", he.BOLSIG_FOLDER / "Ar_Biagi_bolsig_mw_ee", "C5"),
]
LOW_5P = ("5p6", "5p8", "5p9")                      # the 5p levels the MultiBolt fit underpredicted
RATIO_LINES = (420.0674, 750.3869)                  # nm, 5p2 -> 4s1 and 4p10 -> 4s4
MEASURED_N_1S5 = (1e17, 8e17)                       # m^-3, absorption (path length uncertain)
OUTDIR = crf.output_dir("EEDFPhysics")             # + "_RDW4s" with the RDW 4s cross sections
CFG = dict(crf.CONFIG, Ne_grid=libs.NE_GRID, outdir=OUTDIR)


def ready(folder):
    """A finished EEDF library: BOLSIG+ (library.json) or MultiBolt (EEDFs_f0/)."""
    return he.IsBolsigLibrary(folder) or os.path.isdir(os.path.join(folder, "EEDFs_f0"))


def window_integral(wl, I, lam):
    """Integral of the measured spectrum over +-2.5 slit FWHM around lam (NaN in a gap)."""
    hw = 2.5 * 0.030 * lam / 435.833
    w = np.abs(wl - lam) < hw
    return np.trapezoid(I[w], wl[w]) if w.sum() >= 3 and not np.isnan(I[w]).any() else np.nan


def prepare(folder):
    """CR table, features and fine-grid model for one EEDF library (as CRFitNeTe.run)."""
    cfg = dict(CFG, eedf=str(folder))
    tab = crf.build_model_table(cfg)
    feats, comps = crf.select_features(tab, cfg)
    meas = crf.measure(comps, cfg)
    ft = crf.feature_table(meas, feats, cfg)
    rs = crf.repeat_scatter(ft)
    if cfg["use_repeat_scatter"]:
        ft["rel_err"] = np.sqrt(ft.rel_err ** 2 + ft.feature.map(rs) ** 2)
    grid = crf.fine_grid(tab, cfg)
    return cfg, tab, feats, ft, grid, crf.te_eff_fine(tab, grid), crf.feature_model(tab, feats, cfg)


def evaluate(label, folder, wl_meas, I_meas, ratio_meas):
    cfg, tab, feats, ft, grid, Te_f, M = prepare(folder)
    d = ft.query("sweep == @SWEEP and x == @X")
    lev = list(tab["levels"])
    upper_of = lambda w: tab["line_upper"][np.argmin(np.abs(tab["line_wl"] - w))]
    rows, residuals = [], {}
    for deg in (1, None):
        c = dict(cfg, response_deg=deg)
        s, post, _ = crf.fit_block(d, M, feats, grid, Te_f, c)
        wl_m, I_m = ov.model_lines_at(tab, s["x_best"], s["Ne_best"], c)       # energy units
        scales, coef = ov.nuisances(d, feats, wl_m, I_m, c)
        I_syn = np.exp(np.mean(list(scales.values()))) * I_m * ov.response(wl_m, coef)
        res_fit = ov.fit_line_residuals(d, feats, wl_m, I_m, scales, coef)
        res_chk = ov.check_line_residuals(wl_m, I_syn, wl_meas, I_meas, feats.wl.to_numpy())
        res_chk["upper"] = [upper_of(w) for w in res_chk.wl]
        k1, k2 = (int(np.argmin(np.abs(wl_m - w))) for w in RATIO_LINES)
        ratio_model = (I_m[k1] * wl_m[k1]) / (I_m[k2] * wl_m[k2])          # photons
        p5 = res_chk[res_chk.upper.str.startswith("5p")]
        n4s1 = np.exp(crf.posterior_mean(tab, post, np.log(tab["dens"][:, :, lev.index("4s1")]), c))
        rows.append(dict(variant=label, tilt="free" if deg else "none",
                         Te_best=s["Te_best"], Te_lo=s["Te_lo"], Te_hi=s["Te_hi"],
                         Ne_best=s["Ne_best"], Ne_lo=s["Ne_lo"], Ne_hi=s["Ne_hi"],
                         EN_best=s["x_best"], chi2_red=s["chi2_red"],
                         tilt_per_100nm=s.get("response_slope_per_100nm", np.nan),
                         ratio_420_750_model=ratio_model, ratio_420_750_measured=ratio_meas,
                         resid_5p689=p5[p5.upper.isin(LOW_5P)].resid.mean(),
                         resid_5p_other=p5[~p5.upper.isin(LOW_5P)].resid.mean(),
                         rms_fit_lines=np.sqrt(np.mean(res_fit.resid ** 2)),
                         n_4s1_m3=n4s1, edge_EN=s["edge_x"], edge_Ne=s["edge_Ne"]))
        residuals[c["response_deg"]] = (res_fit, res_chk)
    return rows, residuals


def plot_residuals(all_res, deg, path):
    fig, axs = plt.subplots(len(all_res), 2, figsize=(12, 2.1 * len(all_res)), sharey=True,
                            gridspec_kw=dict(width_ratios=[hi - lo for lo, hi in ov.PANELS], wspace=0.04))
    for row, (label, col, res) in zip(np.atleast_2d(axs), all_res):
        res_fit, res_chk = res[deg]
        for ax, (lo, hi) in zip(row, ov.PANELS):
            ax.axhspan(-CFG["sigma_model"], CFG["sigma_model"], color="C3", alpha=0.08, lw=0)
            ax.axhline(0, color="0.3", lw=0.8)
            c = res_chk[(res_chk.wl >= lo) & (res_chk.wl <= hi)]
            low = c.upper.isin(LOW_5P)
            ax.plot(c.wl[~low], c.resid[~low], "o", mfc="none", mec="0.5", ms=5)
            ax.plot(c.wl[low], c.resid[low], "s", mfc="none", mec="C1", ms=6, mew=1.4)
            f = res_fit[(res_fit.wl >= lo) & (res_fit.wl <= hi)]
            ax.plot(f.wl, f.resid, "o", color=col, ms=5)
            ax.set_xlim(lo, hi)
            ax.grid(alpha=0.25)
        row[0].set_ylabel("ln(meas/model)", fontsize=8)
        row[0].text(0.01, 0.95, label, transform=row[0].transAxes, va="top", fontsize=9, weight="bold")
    for ax in np.atleast_2d(axs)[-1]:
        ax.set_xlabel("Wavelength (nm)")
    np.atleast_2d(axs)[0, 0].plot([], [], "o", color="k", label="fitted lines (each repeat)")
    np.atleast_2d(axs)[0, 0].plot([], [], "o", mfc="none", mec="0.5", label="check lines")
    np.atleast_2d(axs)[0, 0].plot([], [], "s", mfc="none", mec="C1", label="check lines from 5p6/5p8/5p9")
    np.atleast_2d(axs)[0, 0].legend(fontsize=7, loc="lower left")
    for ax in np.atleast_2d(axs).ravel():
        ax.set_ylim(-2.5, 2.5)
    fig.suptitle(f"Line residuals at the best fit, response tilt {'free' if deg else 'off (calibration trusted)'}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_summary(df, ratio_meas, path):
    labels = list(dict.fromkeys(df.variant))
    y = np.arange(len(labels))
    fig, axs = plt.subplots(1, 6, figsize=(22, 0.55 * len(labels) + 2.2), sharey=True)
    for tilt, mk, dy in (("free", "o", -0.12), ("none", "s", 0.12)):
        t = df[df.tilt == tilt].set_index("variant").reindex(labels)
        kw = dict(fmt=mk, ms=6, capsize=3, label=f"tilt {tilt}")
        axs[0].errorbar(t.Te_best, y + dy, xerr=np.clip([t.Te_best - t.Te_lo, t.Te_hi - t.Te_best], 0, None), **kw)
        axs[1].errorbar(t.Ne_best, y + dy, xerr=np.clip([t.Ne_best - t.Ne_lo, t.Ne_hi - t.Ne_best], 0, None), **kw)
        axs[2].plot(t.chi2_red, y + dy, mk, ms=7, label=f"tilt {tilt}")
        axs[3].plot(t.ratio_420_750_model, y + dy, mk, ms=7, label=f"model, tilt {tilt}")
        axs[4].plot(t.resid_5p689, y + dy, mk, ms=7, color="C1", label=f"5p6/8/9, tilt {tilt}")
        axs[4].plot(t.resid_5p_other, y + dy, mk, ms=7, mfc="none", color="C0", label=f"other 5p, tilt {tilt}")
        axs[5].plot(t.n_4s1_m3, y + dy, mk, ms=7, label=f"model, tilt {tilt}")
    t = df[df.tilt == "free"].set_index("variant").reindex(labels)
    for yi, v in zip(y, t.tilt_per_100nm):
        axs[2].text(0.98, yi, f"tilt {v:+.2f}/100 nm", transform=axs[2].get_yaxis_transform(),
                    ha="right", va="center", fontsize=7, color="0.35")
    axs[3].axvline(ratio_meas, color="k", ls="--", lw=1.2, label=f"measured {ratio_meas:.4f}")
    axs[4].axvline(0, color="0.4", lw=0.8)
    axs[5].axvspan(*MEASURED_N_1S5, color="0.85", label="measured 1s5 (absorption)")
    axs[5].set_xscale("log")
    axs[1].set_xscale("log")
    axs[3].set_xscale("log")
    titles = [r"$T_{e,\mathrm{eff}}$ (eV), 68 %", r"$N_e$ (m$^{-3}$), 68 %", r"$\chi^2$/dof",
              "420.07 / 750.39 nm (photons)", "5p check lines: mean ln(meas/model)",
              r"$n$(4s1 = 1s5) at the fit (m$^{-3}$)"]
    for ax, ttl in zip(axs, titles):
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    axs[0].set_yticks(y, labels)
    axs[0].invert_yaxis()
    fig.suptitle(f"CR fit of {SWEEP} = {X:g} with each EEDF library", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cache = os.path.join(crf.CONFIG["outdir"], "ar_line_measurements.csv")      # reuse the line fits
    if os.path.exists(cache) and not os.path.exists(os.path.join(OUTDIR, "ar_line_measurements.csv")):
        shutil.copyfile(cache, os.path.join(OUTDIR, "ar_line_measurements.csv"))
    sw = next(s for s in CFG["sweeps"] if s["name"] == SWEEP)
    files = sorted(f for f in os.listdir(sw["folder"]) if f.endswith(".spa") and sw["parse"](f[:-4])
                   and sw["parse"](f[:-4])[0] == X)
    wl_meas, I_meas = ov.measured_spectrum([f[:-4] for f in files], sw["folder"])
    a, b = (window_integral(wl_meas, I_meas, w) for w in RATIO_LINES)
    ratio_meas = (a * RATIO_LINES[0]) / (b * RATIO_LINES[1])
    print(f"measured 420.07/750.39 photon ratio {ratio_meas:.4f} ({len(files)} spectra)")

    rows, all_res = [], []
    for label, folder, col in VARIANTS:
        if not ready(folder):
            print(f"missing {folder} - run Scripts/CreateEEDFLibraries.py")
            continue
        print(f"\n== {label}")
        r, res = evaluate(label, folder, wl_meas, I_meas, ratio_meas)
        rows += r
        all_res.append((label, col, res))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "summary.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df[["variant", "tilt", "Te_best", "Te_lo", "Te_hi", "Ne_best", "chi2_red", "tilt_per_100nm",
                  "ratio_420_750_model", "ratio_420_750_measured", "resid_5p689", "resid_5p_other",
                  "n_4s1_m3", "edge_EN", "edge_Ne"]].round(4).to_string(index=False))
    plot_summary(df, ratio_meas, os.path.join(OUTDIR, "summary.png"))
    plot_residuals(all_res, 1, os.path.join(OUTDIR, "residuals_tilt.png"))
    plot_residuals(all_res, None, os.path.join(OUTDIR, "residuals_notilt.png"))
    print(f"\nwritten to {OUTDIR}")
    return df


if __name__ == "__main__":
    SUMMARY = main()
