# -*- coding: utf-8 -*-
"""
ActinometryWalkthrough.py

Step-by-step figures of what ActinometryTeUncertainty does, for one sweep
condition and one N I / Ar I pair (set below).  Run from Spyder, F5.
Figures go to Experimental_Data/Output/Walkthrough/.

Step 1  raw spectrum + line fits
        Every Ar I line used in the Te/Ne fit, in one spectrum of the chosen
        condition: measured counts (dots), the group fit (line), the fitted
        centre(s).  The area of each fitted Gaussian is the "measured
        intensity" of that line.

Step 2  measured vs CR-model line intensities
        ln(intensity) of each line, normalised per spectrum (mean over the
        lines subtracted, since the absolute scale is unknown), for every repeat
        spectrum, next to the CR model at the best-fit (Te, Ne) after the
        same scale and response correction.  Error bars = the per-point error
        used in the fit (1/SNR, floor, repeat scatter).  The lower panel is
        the residual ln(measured/model) of each line.

Step 3  chi^2 map
        Delta chi^2 over the (Te_eff, Ne) grid (widened by the Birge factor
        when chi^2_min/dof > 1).  Contours: 1-sigma and 2-sigma joint regions.
        Right: the Te marginal = posterior summed over Ne; its 16/50/84 %
        points are the Te_lo / Te_med / Te_hi reported in fit_conditions.csv.

Step 4  rate coefficients at the fitted Te
        k_N and k_Ar (ground-state excitation) and their ratio along the
        EEDF axis, with the 16-84 % Te band shaded and the Te posterior drawn
        underneath.  The k-ratio uncertainty is how much k_N/k_Ar moves across
        that band (weighted by the posterior): rel_err_k_Te.  The dashed
        curves are the Maxwellian at the same Te_eff (EEDF-shape systematic).

Step 5  consensus and rms ln-deviation (one N line, one sweep)
        a) measured I_N/I_Ar for every Ar reference, each divided by its mean
           over the sweep (only the trend is compared)
        b) Te-corrected ratio (I_N/I_Ar)/(k_N/k_Ar), same normalisation - this
           is proportional to n_N/n_Ar.  The consensus (black) is, per
           condition, the median over all Ar references.
        c) deviation of each reference from the consensus, ln(ratio/consensus).
           dev_rms of a pair = sqrt(mean over conditions (and both sweeps) of
           deviation^2).  Small dev_rms = this Ar reference tells the same
           n_N/n_Ar story as the others.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "Experimental_Data", "ExperimentalDataAnalysis"))
import Argon_Nitrogen_Mix_Analysis_V2 as arn        # noqa: E402
import ActinometryLineSelection as als              # noqa: E402
import ActinometryTeUncertainty as atu              # noqa: E402
import CRFitNeTe as crf                             # noqa: E402

# ---- what to show -----------------------------------------------------------------
CONDITION = ("N2 fraction", 1.5)        # (sweep name, x)
SPECTRUM_REP = 0                        # which repeat of that condition for step 1 (0 = first)
PAIR = (744.229, 750.387)               # (N I nm, Ar I nm) for step 4
CONSENSUS = (744.229, "N2 fraction")    # (N I nm, sweep) for step 5
OUTDIR = os.path.join(crf.ROOT_DIR, "Experimental_Data", "Output", "Walkthrough")


def _key(sweep, x):
    return f"{sweep}|{x:g}"


# ----------------------------------------------------------------------------
# Step 1
# ----------------------------------------------------------------------------
def step1_line_fits(R, path):
    fit = R["fit"]
    sweep, x = CONDITION
    rows = fit["ft"][(fit["ft"].sweep == sweep) & (fit["ft"].x == x)]
    fname = sorted(rows.file.unique())[SPECTRUM_REP]
    sw = next(s for s in fit["cfg"]["sweeps"] if s["name"] == sweep)
    acfg = arn.CONFIG
    sp = arn.load_spectrum(os.path.join(sw["folder"], fname + ".spa"), fname, acfg)
    cat, cat_ar = als._catalogue()
    fwhm_fn, offset_fn, _ = arn.measure_instrument_function(sp, cat_ar, acfg)
    feats = fit["feats"]
    near = np.zeros(len(cat), bool)
    for w in feats.wl:
        near |= np.abs(cat.wl_air.values - w) < 1.0
    res, fits = arn.measure_lines(sp, cat[near], fwhm_fn, acfg, fname, float(offset_fn(0)))
    n = len(feats)
    ncol = 5
    fig, axs = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(4 * ncol, 3.4 * np.ceil(n / ncol)), squeeze=False)
    for ax, ft in zip(axs.ravel(), feats.itertuples()):
        g = next((f for f in fits.values() if np.any(np.abs(f["centres"] - ft.wl) < 0.06)), None)
        if g is None:
            ax.set_title(f"{ft.feature}\n(no fit)", fontsize=8)
            continue
        m = np.abs(g["x"] - ft.wl) < 0.25
        ax.plot(g["x"][m], g["y"][m], "o", ms=3, color="0.3", label="measured")
        ax.plot(g["x"][m], g["model"][m], "-", color="C3", lw=1.5, label="group fit")
        for c in g["centres"]:
            if abs(c - ft.wl) < 0.25:
                ax.axvline(c + g["shift"], color="C0", lw=0.8, ls=":")
        r = res[np.isclose(res.wl_air, ft.wl, atol=0.06) & (res.species == "Ar I")]
        snr = r.snr.max() if len(r) else np.nan
        ax.set_title(f"{ft.feature}\nSNR {snr:.0f}", fontsize=8)
        ax.set_xlabel("wavelength [nm]", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(f"Step 1 - line fits in {fname} ({sweep} = {x:g}).  Dots: spectrum, red: fitted Gaussians + "
                 "baseline, dotted: line centres.  Line area = measured intensity.")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Step 2 and 3
# ----------------------------------------------------------------------------
def _condition_fit(R):
    fit = R["fit"]
    sweep, x = CONDITION
    d = fit["ft"][(fit["ft"].sweep == sweep) & (fit["ft"].x == x)].reset_index(drop=True)
    s, post, resid = crf.fit_block(d, fit["M"], fit["feats"], fit["grid"], fit["Te_f"], fit["cfg"])
    return d, s, post, resid


def step2_measured_vs_model(R, path):
    d, s, post, resid = _condition_fit(R)
    y = np.log(d.area.to_numpy())
    model = y - resid                       # model + scale + response at the best fit
    d = d.assign(y=y, model=model)
    # normalise every spectrum by its mean over lines (removes the free scale)
    for col in ("y", "model"):
        d[col] = d[col] - d.groupby("file")[col].transform("mean")
    feats = list(R["fit"]["feats"].feature)
    pos = {f: i for i, f in enumerate(feats)}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw=dict(height_ratios=(2, 1)))
    for k, (f, g) in enumerate(d.groupby("file")):
        xx = g.feature.map(pos) + (k - 1) * 0.12
        a1.errorbar(xx, g.y, yerr=g.rel_err, fmt="o", ms=5, capsize=2, label=f"measured {f}")
        a2.errorbar(xx, g.y - g.model, yerr=g.rel_err, fmt="o", ms=5, capsize=2)
    mod = d.groupby("feature").model.mean().reindex(feats)
    a1.plot(range(len(feats)), mod, "s", ms=11, mfc="none", mec="k", mew=2,
            label=f"CR model at Te_eff = {s['Te_best']:.2f} eV, Ne = {s['Ne_best']:.1e} m$^{{-3}}$")
    a1.set_ylabel("ln(intensity) - spectrum mean")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)
    a2.axhline(0, color="k", lw=0.8)
    a2.axhspan(-R["fit"]["cfg"]["sigma_model"], R["fit"]["cfg"]["sigma_model"], color="0.9",
               label=f"+-sigma_model ({100 * R['fit']['cfg']['sigma_model']:.0f} %)")
    a2.set_ylabel("ln(measured / model)")
    a2.set_xticks(range(len(feats)), feats, rotation=45, ha="right", fontsize=8)
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)
    fig.suptitle(f"Step 2 - measured vs simulated normalised line intensities ({CONDITION[0]} = {CONDITION[1]:g}); "
                 f"chi$^2$/dof = {s['chi2_red']:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def step3_chi2(R, path):
    fit = R["fit"]
    d, s, post, _ = _condition_fit(R)
    Te_f, Ne_f = fit["Te_f"], np.exp(fit["grid"][1])
    dchi2 = -2 * np.log(np.maximum(post / post.max(), 1e-300))      # Birge-widened Delta chi^2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw=dict(width_ratios=(1.4, 1)))
    im = a1.pcolormesh(Te_f, Ne_f, np.minimum(dchi2, 30).T, cmap="viridis_r", shading="auto")
    a1.contour(Te_f, Ne_f, dchi2.T, levels=[2.30, 6.18], colors=["w", "w"], linestyles=["-", "--"])
    a1.plot(s["Te_best"], s["Ne_best"], "r*", ms=14, label="best fit (min chi$^2$)")
    a1.set_yscale("log")
    a1.set_xlabel(crf.te_label(fit["tab"]))
    a1.set_ylabel("$N_e$ [m$^{-3}$]")
    a1.legend(fontsize=8)
    fig.colorbar(im, ax=a1, label=r"$\Delta\chi^2$ (capped at 30); white: 1$\sigma$ / 2$\sigma$")
    a1.set_title(f"chi$^2$ over the model grid (chi$^2_{{min}}$/dof = {s['chi2_red']:.2f}, "
                 f"Birge factor {s['birge']:.2f})", fontsize=10)
    pT = post.sum(1)
    a2.plot(Te_f, pT / pT.max(), "C0-", lw=2)
    for q, ls in ((s["Te_lo"], "--"), (s["Te_med"], "-"), (s["Te_hi"], "--")):
        a2.axvline(q, color="C3", ls=ls)
    a2.axvspan(s["Te_lo"], s["Te_hi"], color="C3", alpha=0.1)
    a2.set_xlabel(crf.te_label(fit["tab"]))
    a2.set_ylabel("posterior (summed over $N_e$), normalised")
    a2.set_title(f"Te_eff = {s['Te_med']:.2f} (+{s['Te_hi'] - s['Te_med']:.2f} / -{s['Te_med'] - s['Te_lo']:.2f}) eV"
                 "\nred: 16 / 50 / 84 % points", fontsize=10)
    a2.grid(alpha=0.3)
    fig.suptitle(f"Step 3 - chi$^2$ and Te posterior ({CONDITION[0]} = {CONDITION[1]:g})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Step 4
# ----------------------------------------------------------------------------
def step4_rates(R, path):
    fit, act = R["fit"], R["act"]
    nl = act["n_lines"][np.isclose(act["n_lines"].wl_air, PAIR[0], atol=1e-3)].iloc[0]
    al = act["a_lines"][np.isclose(act["a_lines"].wl_air, PAIR[1], atol=1e-3)].iloc[0]
    kN, kN_alt = atu.rates_on_axis(fit, act["xs"]["N I"][nl.upper])
    kA, kA_alt = atu.rates_on_axis(fit, act["xs"]["Ar I"][al.upper])
    Te = fit["Te_f"]
    c = fit["cond"][(fit["cond"].sweep == CONDITION[0]) & (fit["cond"].x == CONDITION[1])].iloc[0]
    pT = fit["posts"][_key(*CONDITION)].sum(1)
    kt = R["kt"]
    row = kt[(kt.sweep == CONDITION[0]) & (kt.x == CONDITION[1]) & np.isclose(kt.n_wl, PAIR[0], atol=1e-3)
             & np.isclose(kt.ar_wl, PAIR[1], atol=1e-3)].iloc[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax in (a1, a2):
        ax.axvspan(c.Te_lo, c.Te_hi, color="C3", alpha=0.12, label="Te 16-84 %")
        ax.axvline(c.Te_med, color="C3", lw=1)
        ax.set_xlabel(crf.te_label(fit["tab"]))
        ax.grid(alpha=0.3, which="both")
        tw = ax.twinx()
        tw.fill_between(Te, pT / pT.max(), color="0.85", zorder=0)
        tw.set_ylim(0, 4)
        tw.set_yticks([])
    a1.semilogy(Te, kN, "C0-", lw=2, label=f"k N I {PAIR[0]:.1f} ({nl.upper})")
    a1.semilogy(Te, kA, "C1-", lw=2, label=f"k Ar I {PAIR[1]:.1f} ({al.upper})")
    a1.semilogy(Te, kN_alt, "C0--", lw=1, label="Maxwellian, same Te_eff")
    a1.semilogy(Te, kA_alt, "C1--", lw=1)
    a1.set_ylabel("ground-state excitation rate k [m$^3$ s$^{-1}$]")
    a1.legend(fontsize=8)
    a1.set_title("rate coefficients (grey: Te posterior)")
    a2.semilogy(Te, kN / kA, "k-", lw=2, label="k$_N$/k$_{Ar}$ (fitted EEDF family)")
    a2.semilogy(Te, kN_alt / kA_alt, "k--", lw=1, label="Maxwellian, same Te_eff")
    a2.axhspan(row.k_ratio_lo, row.k_ratio_hi, color="C0", alpha=0.2, label="k-ratio 16-84 % from Te")
    a2.axhline(row.k_ratio, color="C0")
    a2.set_ylabel("k$_N$ / k$_{Ar}$")
    a2.legend(fontsize=8)
    a2.set_title(f"k ratio = {row.k_ratio:.3g}  (x/÷ {np.exp(row.rel_err_k_Te):.2f} from Te), "
                 f"S = {row.S_Te:.1f}", fontsize=10)
    fig.suptitle(f"Step 4 - Te uncertainty -> k-ratio uncertainty ({CONDITION[0]} = {CONDITION[1]:g}); "
                 f"CR model: {100 * row.f_direct_Ar:.2g} % of the Ar {al.upper} production is direct from ground")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Step 5
# ----------------------------------------------------------------------------
def step5_consensus(R, path):
    n_wl, sweep = CONSENSUS
    d = R["d"]
    d = d[np.isclose(d.n_wl, n_wl, atol=1e-3) & (d.sweep == sweep) & np.isfinite(d.corr_norm)].copy()
    d["I_norm"] = d.I_ratio / d.groupby("ar_wl").I_ratio.transform("mean")
    summ = R["summary"]
    refs = sorted(d.ar_wl.unique())
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    for k, aw in enumerate(refs):
        g = d[d.ar_wl == aw].sort_values("x")
        up = g.ar_upper.iloc[0]
        rms = summ[np.isclose(summ.n_wl, n_wl, atol=1e-3) & np.isclose(summ.ar_wl, aw, atol=1e-3)].dev_rms
        lab = f"Ar {aw:.1f} ({up})"
        axs[0].plot(g.x, g.I_norm, "o-", color=cols[k % 10], label=lab)
        axs[1].plot(g.x, g.corr_norm, "o-", color=cols[k % 10], label=lab)
        axs[2].plot(g.x, g.dev, "o-", color=cols[k % 10],
                    label=f"{lab}: dev_rms = {rms.iloc[0]:.3f}" if len(rms) else lab)
    cons = d.groupby("x").consensus.first().sort_index()
    axs[1].plot(cons.index, cons.values, "k-", lw=4, alpha=0.6, label="consensus = median over refs")
    axs[0].set_title("a) measured I$_N$/I$_{Ar}$ / its sweep mean")
    axs[1].set_title("b) Te-corrected (I$_N$/I$_{Ar}$)/(k$_N$/k$_{Ar}$) / mean  $\\propto n_N/n_{Ar}$")
    axs[2].set_title("c) ln(corrected / consensus); dev_rms = rms over conditions\n(dev_rms in legend uses both sweeps)")
    for ax in axs:
        ax.set_xlabel(next(s["xlabel"] for s in R["fit"]["cfg"]["sweeps"] if s["name"] == sweep))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axs[2].axhline(0, color="k", lw=0.8)
    fig.suptitle(f"Step 5 - how the consensus and dev_rms are made (N I {n_wl:.3f} nm, {sweep} sweep)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(R=None):
    """R: result of ActinometryTeUncertainty.run (runs it when None)."""
    R = R or atu.run(atu.CONFIG)
    os.makedirs(OUTDIR, exist_ok=True)
    out = lambda f: os.path.join(OUTDIR, f)
    step1_line_fits(R, out("step1_line_fits.png"))
    step2_measured_vs_model(R, out("step2_measured_vs_model.png"))
    step3_chi2(R, out("step3_chi2_Te_posterior.png"))
    step4_rates(R, out("step4_rates_Te_band.png"))
    step5_consensus(R, out("step5_consensus_dev_rms.png"))
    print("walkthrough figures in", OUTDIR)
    return R


if __name__ == "__main__":
    RESULTS = run(globals().get("RESULTS"))     # reuses RESULTS from ActinometryTeUncertainty in Spyder
