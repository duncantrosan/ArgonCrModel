# -*- coding: utf-8 -*-
"""
SweepTrends.py

Te_eff, Ne and the scale factor s of the CR fit along both sweeps:
  N2 fraction (85 W, 1 Torr)  each condition with the EEDFs and the CR model of its own N2
                              fraction (ActinometryNitrogenContent.mixture_fit; 0 % = pure Ar)
  Power (2.4 % N2, 1 Torr)    the 2.4 % fit
for the microwave and DC BOLSIG+ EEDFs, with the BSR 4s cross sections and, for comparison,
the RDW ones (XSECS).

s is the free scale of each spectrum in the fit, s = ln(measured / model) at 775 nm: absolute
intensity calibration x emitting volume x collection, the same for every spectrum if the optics,
the exposure normalisation and the plasma volume seen by the spectrometer do not change. The
model intensity changes by orders of magnitude along the valley of the fit in (E/N, Ne), so s of
a single condition is poorly determined (quoted as the posterior median and 16-84 %). Whether one
calibration factor fits a whole sweep is tested with CRFitNeTe.common_scale_fit:
  - each condition's likelihood of s alone (scale_posteriors.png): one s fits all where they overlap
  - the joint fit with one s per sweep (condition means may scatter by S_SCATTER): its chi2 above
    the free fits (dchi2, ~ number of conditions - 1 if consistent), the common s, and Te_eff, Ne
    of each condition with it - the absolute intensities then also constrain Ne

Figures:
  n2_fraction_trends.png, power_trends.png  Te_eff, Ne (free s, common s), s, the measured
                       intensity of the fitted lines (no model) and chi2/dof along each sweep
  scale_posteriors.png likelihood of s of every condition and the common-s posterior
  scale_valleys.png    s along the valley of the fit (best E/N at each Ne), main cross sections
trends.csv / trends_spectra.csv: the numbers; trends_results.pkl: everything the plots need
(set REFIT = False to replot from it).

Output in Experimental_Data/Output/SweepTrends/.  Run from Spyder: F5 (~5 min; the CR tables of
ActinometryNitrogenContent are reused, a missing one is built first, ~1 min each).
"""
import contextlib
import io
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT_DIR, "Experimental_Data", "ExperimentalDataAnalysis"))
import ActinometryNitrogenContent as anc            # noqa: E402

crf = anc.crf

# ---- settings -------------------------------------------------------------------
XSECS = ("BSR", "RDW")                              # 4s cross sections; the first is the main result
EEDFS = ("microwave", "dc")
N2_FRACTIONS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # % N2 of the N2 fraction sweep
S_SCATTER = 0.10                                    # ln, allowed scatter of the condition-mean s
                                                    # around the common s (plasma volume, drifts)
REFIT = True                                        # False: replot from trends_results.pkl
OUTDIR = os.path.join(ROOT_DIR, "Experimental_Data", "Output", "SweepTrends")
SWEEPS = {sw["name"]: sw for sw in anc.CONFIG["actinometry"]["sweeps"]}
STYLE = {"BSR": dict(ls="-", mfc=None), "RDW": dict(ls="--", mfc="white")}
ECOL = {"microwave": "C3", "dc": "C0"}


def anc_config(xsec):
    """ActinometryNitrogenContent settings with the 4s cross sections xsec (own output folders)."""
    fit = dict(crf.CONFIG, xsec_4s=xsec, outdir=crf.output_dir("CRFit", {"xsec_4s": xsec}))
    return dict(anc.CONFIG, fit=fit, outdir=crf.output_dir("ActinometryN2Content", {"xsec_4s": xsec}))


_FITS = {}


def fit(xsec, eedf, pct):
    """CRFitNeTe.run result of the fit with the model of pct % N2 (cached in memory)."""
    key = (xsec, eedf, pct)
    if key not in _FITS:
        print(f"  fit: {xsec} 4s, {eedf} EEDF, {pct:g} % N2", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            _FITS[key] = anc.mixture_fit(pct, eedf, anc_config(xsec))
    return _FITS[key]


def conditions(sweep):
    """[(x, % N2 of the model)] of a sweep."""
    if sweep == "N2 fraction":
        return [(p, p) for p in N2_FRACTIONS]
    pct = anc.CONFIG["power_sweep_N2"]
    return [(x, pct) for x in sorted(fit(XSECS[0], EEDFS[0], pct)["cond"].query("sweep == @sweep").x)]


def collect(xsec, eedf, sweep):
    """Free-s and common-s results of one sweep."""
    rows, spec, blocks, valley = [], [], [], {}
    for x, pct in conditions(sweep):
        F = fit(xsec, eedf, pct)
        c = F["cond"].query("sweep == @sweep and x == @x")
        if c.empty:
            continue
        rows.append(dict(c.iloc[0], xsec=xsec, eedf=eedf, model_N2_percent=pct))
        d = F["ft"].query("sweep == @sweep and x == @x")
        tot = d.groupby("file").area.sum()                           # measured, all fitted lines
        sp = F["spec"].query("sweep == @sweep and x == @x").assign(xsec=xsec, eedf=eedf)
        spec.append(sp.assign(area_sum=sp.file.map(tot)))
        blocks.append(dict(key=x, d=d, M=F["M"], feats=F["feats"], grid=F["grid"], Te_f=F["Te_f"], cfg=F["cfg"]))
        g = crf.gls_block(d, F["M"], F["feats"], F["cfg"])           # s along the fit valley
        j = np.argmin(g["chi2"], axis=0)
        cols = np.arange(g["chi2"].shape[1])
        valley[x] = (np.exp(F["grid"][1]), g["s_grid"][j, cols],
                     (g["chi2"][j, cols] - g["chi2"].min()) / c.iloc[0].birge ** 2)
    common, _, s_ax, p_s, dchi2, p_cond = crf.common_scale_fit(blocks, s_scatter=S_SCATTER)
    q = crf._wquantile(s_ax, p_s, [0.16, 0.5, 0.84])
    for r, cm in zip(rows, common):
        r.update({f"{k}_common_s": cm[k] for k in ("Te_lo", "Te_med", "Te_hi", "Ne_lo", "Ne_med", "Ne_hi", "edge_Ne")})
        r.update(s_common_lo=q[0], s_common=q[1], s_common_hi=q[2], dchi2_common_s=dchi2,
                 n_conditions=len(rows))
    return rows, pd.concat(spec, ignore_index=True), dict(valley=valley, s_ax=s_ax, p_s=p_s, p_cond=p_cond)


# ---- plots ------------------------------------------------------------------------
def _range(ax, x, y, lo, hi, **kw):
    """Marker at y with a bar from lo to hi (lo <= y <= hi not required)."""
    ax.vlines(x, lo, hi, color=kw.get("color"), lw=1.2, alpha=0.8)
    ax.plot(x, y, **kw)


def plot_trends(res, spec, sweep, path):
    sw = SWEEPS[sweep]
    r = res[res.sweep == sweep]
    fig, axs = plt.subplots(5, len(EEDFS), figsize=(7.5 * len(EEDFS), 19), sharex=True, squeeze=False)
    span = np.ptp(r.x) if len(r) else 1.0
    for j, eedf in enumerate(EEDFS):
        for k, xsec in enumerate(XSECS):
            g = r[(r.eedf == eedf) & (r.xsec == xsec)].sort_values("x")
            if g.empty:
                continue
            st, col = STYLE[xsec], ECOL[eedf]
            mfc = st["mfc"] or col
            dx = (k - 0.5) * 0.03 * span
            lab = f"{xsec} 4s"
            _range(axs[0, j], g.x + dx, g.Te_med, g.Te_lo, g.Te_hi, marker="o", ls=st["ls"], color=col, mfc=mfc,
                   label=f"{lab}: free s")
            _range(axs[0, j], g.x + dx + 0.012 * span, g.Te_med_common_s, g.Te_lo_common_s, g.Te_hi_common_s,
                   marker="D", ls=":", color="0.25", mfc=st["mfc"] or "0.25", ms=5, label=f"{lab}: common s")
            _range(axs[1, j], g.x + dx, g.Ne_med, g.Ne_lo, g.Ne_hi, marker="o", ls=st["ls"], color=col, mfc=mfc,
                   label=f"{lab}: free s (line ratios)")
            _range(axs[1, j], g.x + dx + 0.012 * span, g.Ne_med_common_s, g.Ne_lo_common_s, g.Ne_hi_common_s,
                   marker="D", ls=":", color="0.25", mfc=st["mfc"] or "0.25", ms=5,
                   label=f"{lab}: common s ($\\pm${S_SCATTER:g}), + absolute intensities")
            edge = g[g.edge_Ne.astype(bool)]
            axs[1, j].plot(edge.x + dx, edge.Ne_med, "x", color="m", ms=9, mew=2,
                           label="free-s posterior at the Ne grid edge" if k == 0 else None)
            _range(axs[2, j], g.x + dx, g.s_med, g.s_lo, g.s_hi, marker="o", ls=st["ls"], color=col, mfc=mfc,
                   label=f"{lab}: free s, median and 16-84 %")
            axs[2, j].axhspan(g.s_common_lo.iloc[0], g.s_common_hi.iloc[0], color=col, alpha=0.25 if k == 0 else 0.1,
                              label=f"{lab}: common s = {g.s_common.iloc[0]:.1f} "
                                    f"($\\Delta\\chi^2$ = {g.dchi2_common_s.iloc[0]:.1f}, {len(g) - 1} constraints)")
            axs[4, j].plot(g.x + dx, g.chi2_red, "o", ls=st["ls"], color=col, mfc=mfc, label=lab)
        sp = spec[(spec.eedf == eedf) & (spec.xsec == XSECS[0]) & (spec.sweep == sweep)]
        ref = sp.area_sum.mean()
        m = sp.groupby("x").area_sum.mean() / ref
        axs[3, j].plot(sp.x, sp.area_sum / ref, ".", color="0.5", label="single spectra")
        axs[3, j].plot(m.index, m.values, "s-", color="k", label="condition mean")
        axs[0, j].set_ylabel(r"$T_{e,\mathrm{eff}}$ [eV] (16-84 %)")
        axs[1, j].set_ylabel("$N_e$ [m$^{-3}$] (16-84 %)")
        axs[1, j].set_yscale("log")
        axs[2, j].set_ylabel("scale $s$ = ln(measured / model) at 775 nm")
        axs[3, j].set_ylabel("measured intensity of the fitted lines\n(sum, / sweep mean; no model)")
        axs[4, j].set_ylabel(r"$\chi^2$/dof (free s)")
        axs[0, j].set_title(f"{eedf} EEDF (BOLSIG+ of each mixture)")
        axs[-1, j].set_xlabel(sw["xlabel"])
        for ax in axs[:, j]:
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=7)
    fig.suptitle(f"CR fit along the {sweep} sweep ({sw['note']}): $T_{{e,\\mathrm{{eff}}}}$, $N_e$ and the scale "
                 f"factor s\n" + ", ".join(f"{x} 4s cross sections: {'filled, solid' if STYLE[x]['mfc'] is None else 'open, dashed'}"
                                           for x in XSECS) + "; diamonds: one s for the whole sweep")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_scale_posteriors(extra, path):
    """Likelihood of s from each condition alone, and the posterior of one common s."""
    keys = [(x, e) for e in EEDFS for x in XSECS]
    fig, axs = plt.subplots(len(SWEEPS), len(keys), figsize=(4.6 * len(keys), 4.6 * len(SWEEPS)), squeeze=False)
    for i, sweep in enumerate(SWEEPS):
        for j, (xsec, eedf) in enumerate(keys):
            ax = axs[i, j]
            e = extra.get((xsec, eedf, sweep))
            if e is None:
                ax.set_visible(False)
                continue
            cols = plt.cm.viridis(np.linspace(0, 0.9, max(len(e["p_cond"]), 1)))
            for col, (x, p) in zip(cols, sorted(e["p_cond"].items())):
                ax.plot(e["s_ax"], p / p.max(), color=col, lw=1.4, label=f"{x:g}")
            ax.plot(e["s_ax"], e["p_s"] / e["p_s"].max(), "k-", lw=2.6, label="common s")
            lo, hi = e["s_ax"][e["p_s"] > 1e-4 * e["p_s"].max()][[0, -1]]
            ax.set_xlim(lo - 6, hi + 6)
            ax.set_title(f"{sweep}: {eedf} EEDF, {xsec} 4s", fontsize=10)
            ax.set_xlabel("s = ln(measured / model) at 775 nm")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("likelihood of s (max = 1)")
            ax.legend(title=SWEEPS[sweep]["xlabel"], fontsize=7, title_fontsize=7)
    fig.suptitle("Scale factor s: each condition alone (marginalised over E/N and Ne) and one s for the whole sweep "
                 f"(condition means within $\\pm${S_SCATTER:g})")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_valleys(extra, path):
    """s along the fit valley (best E/N at each Ne) of every condition: one s picks one Ne per
    condition where a horizontal line crosses its curve."""
    fig, axs = plt.subplots(len(SWEEPS), len(EEDFS), figsize=(7.5 * len(EEDFS), 5.5 * len(SWEEPS)), squeeze=False)
    for i, sweep in enumerate(SWEEPS):
        for j, eedf in enumerate(EEDFS):
            ax = axs[i, j]
            e = extra.get((XSECS[0], eedf, sweep))
            if e is None:
                continue
            cols = plt.cm.viridis(np.linspace(0, 0.9, max(len(e["valley"]), 1)))
            for col, (x, (Ne, s, d)) in zip(cols, sorted(e["valley"].items())):
                ax.semilogx(Ne, s, "-", color=col, lw=0.8, alpha=0.6)
                ok = d < 4
                ax.semilogx(np.where(ok, Ne, np.nan), np.where(ok, s, np.nan), "-", color=col, lw=2.5, label=f"{x:g}")
            q = crf._wquantile(e["s_ax"], e["p_s"], [0.16, 0.84])
            ax.axhspan(*q, color="k", alpha=0.15, label="common s (16-84 %)")
            ax.set_xlabel("$N_e$ [m$^{-3}$]")
            ax.set_ylabel("s at the best E/N for this $N_e$")
            ax.set_title(f"{sweep}: {eedf} EEDF, {XSECS[0]} 4s; thick where $\\Delta\\chi^2/b^2$ < 4", fontsize=10)
            ax.grid(alpha=0.3, which="both")
            ax.legend(title=SWEEPS[sweep]["xlabel"], fontsize=7, title_fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    pkl = os.path.join(OUTDIR, "trends_results.pkl")
    if REFIT or not os.path.isfile(pkl):
        rows, specs, extra = [], [], {}
        for xsec in XSECS:
            for eedf in EEDFS:
                for sweep in SWEEPS:
                    r, sp, e = collect(xsec, eedf, sweep)
                    rows += r
                    specs.append(sp)
                    extra[xsec, eedf, sweep] = e
        res, spec = pd.DataFrame(rows), pd.concat(specs, ignore_index=True)
        res.to_csv(os.path.join(OUTDIR, "trends.csv"), index=False)
        spec.to_csv(os.path.join(OUTDIR, "trends_spectra.csv"), index=False)
        with open(pkl, "wb") as f:
            pickle.dump(dict(res=res, spec=spec, extra=extra), f)
    else:
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        res, spec, extra = d["res"], d["spec"], d["extra"]
    for sweep in SWEEPS:
        tag = "n2_fraction" if sweep == "N2 fraction" else sweep.lower()
        plot_trends(res, spec, sweep, os.path.join(OUTDIR, f"{tag}_trends.png"))
    plot_scale_posteriors(extra, os.path.join(OUTDIR, "scale_posteriors.png"))
    plot_valleys(extra, os.path.join(OUTDIR, "scale_valleys.png"))
    cols = ["xsec", "eedf", "sweep", "x", "Te_med", "Te_lo", "Te_hi", "Ne_med", "Ne_lo", "Ne_hi", "s_med", "s_lo", "s_hi",
            "chi2_red", "Ne_med_common_s", "Ne_lo_common_s", "Ne_hi_common_s", "Te_med_common_s", "s_common",
            "dchi2_common_s"]
    with pd.option_context("display.width", 260, "display.max_columns", 30):
        print(res[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nfigures in {OUTDIR}")
    return res, spec, extra


if __name__ == "__main__":
    RES, SPEC, EXTRA = main()
