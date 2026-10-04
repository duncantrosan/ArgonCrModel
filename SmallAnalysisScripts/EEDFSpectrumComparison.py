# -*- coding: utf-8 -*-
"""
EEDFSpectrumComparison.py

The measured pure-Ar spectrum against the CR model fitted with each EEDF:
MultiBolt (6 terms), Maxwell-Boltzmann, and BOLSIG+ plain, with electron-electron
collisions, with superelastic collisions (4s), with both (DC field), and in the
2.45 GHz microwave field without and with e-e. Each EEDF gets its own best fit (CRFitNeTe.fit_block) of the same
condition, on the same Ne grid, with the response tilt set by RESPONSE_DEG.

  spectrum_comparison_<tilt|notilt>.png   measured spectrum (repeat mean) with the slit-broadened
                            model spectra of the SOURCES marked for it, 5p->4s and 4p->4s ranges
  residual_comparison_<tilt|notilt>.png   ln(measured / model) of the fitted lines (filled) and
                            the check lines (open; rated, unbroadened, isolated -
                            CRFitSpectrumOverlay.CHECK) for each EEDF

Output in Experimental_Data/Output/EEDFSpectra/. CR tables are cached in
Output/EEDFPhysics (shared with EEDFPhysicsFitComparison; the Maxwellian table takes
~5 min the first time). Run from Spyder: edit the settings, press F5.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import EEDFPhysicsFitComparison as pc               # noqa: E402  (paths, CFG, prepare)
import CRFitSpectrumOverlay as ov                   # noqa: E402

crf, he = pc.crf, pc.he

# ---- settings -------------------------------------------------------------------
SWEEP, X = pc.SWEEP, pc.X                           # pure Ar, 85 W, 1 Torr
RESPONSE_DEG = 1                                    # None = calibration trusted, 1 = linear tilt
SOURCES = [   # label, EEDF ('maxwell' or library folder), colour, drawn in the spectrum figure
    ("MultiBolt", he.MULTIBOLT_FOLDER / "Ar_Biagi_lowEN_span12", "0.45", True),
    ("Maxwell-Boltzmann", "maxwell", "C1", True),
    ("BOLSIG+", he.BOLSIG_FOLDER / "Ar_Biagi_bolsig", "C9", False),
    ("BOLSIG+ e-e", he.BOLSIG_FOLDER / "Ar_Biagi_bolsig_ee", "C0", True),
    ("BOLSIG+ superelastic", he.BOLSIG_FOLDER / "Ar_Biagi_bolsig_se", "C8", False),
    ("BOLSIG+ e-e + superelastic", he.BOLSIG_FOLDER / "Ar_Biagi_bolsig_ee_se", "C2", False),
    ("BOLSIG+ microwave", he.BOLSIG_FOLDER / "Ar_Biagi_bolsig_mw", "C4", True),
    ("BOLSIG+ microwave e-e", he.BOLSIG_FOLDER / "Ar_Biagi_bolsig_mw_ee", "C3", False),
]   # the residual figure shows all of them
OUTDIR = os.path.join(pc.ROOT_DIR, "Experimental_Data", "Output", "EEDFSpectra")
TAG = "notilt" if RESPONSE_DEG is None else "tilt"
SLOPES = []


def fit_source(eedf, wl_meas, I_meas):
    """Best fit of the condition with one EEDF -> fit summary, model lines and residuals."""
    cfg, tab, feats, ft, grid, Te_f, M = pc.prepare(eedf)
    cfg = dict(cfg, response_deg=RESPONSE_DEG)
    d = ft.query("sweep == @SWEEP and x == @X")
    s, post, _ = crf.fit_block(d, M, feats, grid, Te_f, cfg)
    wl_m, I_m = ov.model_lines_at(tab, s["x_best"], s["Ne_best"], cfg)
    scales, coef = ov.nuisances(d, feats, wl_m, I_m, cfg)
    I_syn = np.exp(np.mean(list(scales.values()))) * I_m * ov.response(wl_m, coef)
    res_fit = ov.fit_line_residuals(d, feats, wl_m, I_m, scales, coef)
    res_chk, _ = ov.confident_check_residuals(wl_m, I_m, scales, coef, feats.wl.to_numpy(), SWEEP, X, cfg)
    lev = list(tab["levels"])
    s["n_4s1"] = np.exp(crf.posterior_mean(tab, post, np.log(tab["dens"][:, :, lev.index("4s1")]), cfg))
    return s, wl_m, I_syn, res_fit, res_chk, feats.wl.to_numpy()


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    sw = next(x for x in pc.CFG["sweeps"] if x["name"] == SWEEP)
    files = sorted(f[:-4] for f in os.listdir(sw["folder"]) if f.endswith(".spa")
                   and sw["parse"](f[:-4]) and sw["parse"](f[:-4])[0] == X)
    wl_meas, I_meas = ov.measured_spectrum(files, sw["folder"])

    fits = []
    SLOPES.clear()
    for label, eedf, col, show in SOURCES:
        if eedf != "maxwell" and not pc.ready(eedf):
            print(f"missing {eedf} - run Scripts/CreateEEDFLibraries.py")
            continue
        print(f"== {label}")
        s, wl_m, I_syn, res_fit, res_chk, fit_wl = fit_source(str(eedf), wl_meas, I_meas)
        name = (f"{label}: $T_{{e,\\mathrm{{eff}}}}$ = {s['Te_best']:.2f} eV, $N_e$ = {s['Ne_best']:.1e} m$^{{-3}}$, "
                f"$\\chi^2$/dof = {s['chi2_red']:.2f}, n(1s5) = {s['n_4s1']:.1e} m$^{{-3}}$")
        print("  " + name.replace("$", "").replace("\\mathrm", "").replace("{", "").replace("}", ""))
        fits.append((label, name, col, wl_m, I_syn, res_fit, res_chk, show))
        SLOPES.append((label, s.get("response_slope_per_100nm", np.nan)))
    tilt = "calibration trusted (no tilt)" if RESPONSE_DEG is None else f"response tilt deg {RESPONSE_DEG}"
    if RESPONSE_DEG:
        tilt += ": " + ", ".join(f"{lab} {sl:+.2f}" for lab, sl in SLOPES) + " per 100 nm"

    # --- spectra ----------------------------------------------------------------
    fig, axs = plt.subplots(len(ov.PANELS), 1, figsize=(14, 4.4 * len(ov.PANELS)))
    for ax, (lo, hi) in zip(axs, ov.PANELS):
        mm = (wl_meas >= lo) & (wl_meas <= hi)
        ax.plot(wl_meas[mm], I_meas[mm], color="k", lw=1.0, label="Measured (mean of repeats)", zorder=5)
        for label, name, col, wl_m, I_syn, *_, show in fits:
            if not show:
                continue
            m = (wl_m > lo - 1) & (wl_m < hi + 1) & (I_syn > 0)
            xs, ys = he.BroadenWithSlit(wl_m[m], I_syn[m])
            ax.plot(xs, ys, color=col, lw=1.1, alpha=0.85, label=name)
        gap = np.isnan(I_meas) & mm
        edges = np.flatnonzero(np.diff(np.r_[0, gap.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            ax.axvspan(wl_meas[a], wl_meas[b - 1], color="0.9", lw=0, zorder=0)
        top = np.nanmax(I_meas[mm]) * 1.08
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.03 * top, top)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (calibrated, arb.)")
        ax.grid(alpha=0.25)
    axs[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Pure Ar, 1 Torr: measured spectrum vs CR model with each EEDF ({tilt}); "
                 "grey bands = echelle order gaps", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"spectrum_comparison_{TAG}.png"), dpi=200)

    # --- residuals ----------------------------------------------------------------
    fig, axs = plt.subplots(1, len(ov.PANELS), figsize=(14, 5.5), sharey=True,
                            gridspec_kw=dict(width_ratios=[hi - lo for lo, hi in ov.PANELS], wspace=0.04))
    offsets = np.linspace(-0.25, 0.25, max(len(fits), 1))         # nm, so the markers don't overlap
    for (label, name, col, wl_m, I_syn, res_fit, res_chk, _), dx in zip(fits, offsets):
        for ax, (lo, hi) in zip(axs, ov.PANELS):
            f = res_fit[(res_fit.wl >= lo) & (res_fit.wl <= hi)]
            c = res_chk[(res_chk.wl >= lo) & (res_chk.wl <= hi)]
            ax.plot(f.wl + dx, f.resid, "o", color=col, ms=6, label=label if lo == ov.PANELS[0][0] else None)
            ax.plot(c.wl + dx, c.resid, "o", mfc="none", mec=col, ms=6, mew=1.2)
    for ax, (lo, hi) in zip(axs, ov.PANELS):
        ax.axhspan(-pc.CFG["sigma_model"], pc.CFG["sigma_model"], color="C3", alpha=0.08, lw=0)
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("Wavelength (nm)")
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("ln(measured / model)")
    axs[0].plot([], [], "o", color="0.3", label="fitted lines (each repeat)")
    axs[0].plot([], [], "o", mfc="none", mec="0.3", label="check lines (not fitted)")
    axs[0].fill_between([], [], color="C3", alpha=0.15, label=f"$\\pm\\sigma_{{model}}$ = {100 * pc.CFG['sigma_model']:.0f} %")
    axs[0].legend(fontsize=7, loc="lower left", ncol=2)
    fig.suptitle(f"Line residuals at each EEDF's best fit ({tilt})", fontsize=12)
    fig.savefig(os.path.join(OUTDIR, f"residual_comparison_{TAG}.png"), dpi=200, bbox_inches="tight")
    print(f"\nfigures in {OUTDIR}")
    plt.show()


if __name__ == "__main__":
    main()
