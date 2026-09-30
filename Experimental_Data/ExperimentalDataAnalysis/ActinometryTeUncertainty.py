# -*- coding: utf-8 -*-
"""
ActinometryTeUncertainty.py

Actinometry with the plasma state taken from the CR-model fit.

1. CRFitNeTe.run                 -> Te, Ne posterior for every sweep condition
2. ActinometryLineSelection.run  -> selected N I / Ar I pairs and the measured
                                    I_N / I_Ar per condition
3. Per condition and pair the posterior of the EEDF axis (Te for a
   Maxwellian, E/N for MultiBolt EEDFs - the same EEDFs as in the fit) is
   pushed through the ground-state rate coefficients:
       k_N / k_Ar   median and 16-84 % interval, relative 1-sigma from Te
   plus the sensitivity S = dln(k_N/k_Ar)/dln Te_eff at the fitted Te, the
   EEDF-shape systematic as log10 of the k ratio for the fitted EEDF family over
   the alternative at the same Te_eff (Druyvesteyn for a Maxwellian fit,
   Maxwellian for a MultiBolt fit)
   and, from the CR model, the fraction of the Ar reference level's production
   that is direct excitation from the ground state (actinometry assumes 1).
4. Te-corrected ratio  (I_N/I_Ar) / (k_N/k_Ar)  =  n_N/n_Ar * b_N/b_Ar [* lam_Ar/lam_N]
   with the line-ratio and Te errors in quadrature.  For one N I line every Ar
   reference should give the same trend; the deviation of each pair from the
   consensus of its N line is the pair's consistency metric.  Where both
   branching ratios are known, n_N/n_Ar itself is given.

Te is the only plasma parameter that enters k_N/k_Ar; Ne enters through the
direct-excitation fraction only.  The Te posterior already contains the CR-model
error (sigma_model) and the Birge widening of the fit.

Run from Spyder: edit CONFIG, press F5.  Results in Experimental_Data/Output/ActinometryTe.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ActinometryRates as ar                       # noqa: E402
import ActinometryLineSelection as als              # noqa: E402
import CRFitNeTe as crf                             # noqa: E402

CONFIG = dict(
    fit=crf.CONFIG,
    actinometry=als.CONFIG,
    quantiles=(0.16, 0.5, 0.84),
    k_margin_eV=1.0,                    # MultiBolt k valid only where the EEDF export reaches threshold + this
    min_valid_mass=0.9,                 # posterior mass with a valid k needed to quote a k ratio
    outdir=os.path.join(crf.ROOT_DIR, "Experimental_Data", "Output", "ActinometryTe"),
)


# ----------------------------------------------------------------------------
# 1. k ratios with the Te posterior
# ----------------------------------------------------------------------------
def _wq(values, weights, q):
    o = np.argsort(values)
    c = np.cumsum(weights[o])
    return np.interp(q, c / c[-1], values[o])


def rates_on_axis(fit, xs, margin_eV=1.0):
    """k [m^3/s] of cross section xs along the fine EEDF axis of the fit, and
    along the alternative EEDF family with the same Te_eff (Druyvesteyn for a
    Maxwellian fit, Maxwellian for a MultiBolt fit) for the shape systematic.
    MultiBolt EEDFs are exported only down to ~1e-12 of their peak; where the
    export ends less than margin_eV above the threshold, k is set to NaN."""
    tab, fx, Te_f = fit["tab"], fit["grid"][0], fit["Te_f"]
    if str(tab["eedf"]) == "maxwell":
        return (ar.rate_coefficient(xs, Te=Te_f),
                ar.rate_coefficient(xs, Te=Te_f, kind="druyvesteyn"))
    # MultiBolt: rates at the sweep EEDFs, interpolated linearly in ln k vs ln(E/N)
    lx = np.log(tab["x_grid"])
    k = np.array([ar.rate_coefficient(xs, eedf=(e["E"], e["EEDF"])) for e in fit["eedfs"]])
    E_max = np.array([e["E"][e["EEDF"] > 0].max() for e in fit["eedfs"]])
    ok = (E_max >= xs["threshold_eV"] + margin_eV) & (k > 0)
    k_fit = np.exp(np.interp(fx, lx, np.log(np.where(k > 0, k, 1e-300))))
    k_fit[np.interp(fx, lx, ok.astype(float)) < 0.999] = np.nan
    return k_fit, ar.rate_coefficient(xs, Te=Te_f)


def k_ratio_table(fit, act, cfg=CONFIG):
    tab, posts, cond = fit["tab"], fit["posts"], fit["cond"]
    lnTe = np.log(fit["Te_f"])
    xs, n_lines, a_lines = act["xs"], act["n_lines"], act["a_lines"]
    kN = {u: rates_on_axis(fit, xs["N I"][u], cfg["k_margin_eV"]) for u in n_lines.upper.unique()}
    kA = {u: rates_on_axis(fit, xs["Ar I"][u], cfg["k_margin_eV"]) for u in a_lines.upper.unique()}
    lev = list(tab["levels"])
    q = cfg["quantiles"]
    rows = []
    for c in cond.itertuples():
        post = posts[f"{c.sweep}|{c.x:g}"]
        pX = post.sum(1)
        i_med = np.argmin(np.abs(np.cumsum(pX) - 0.5))
        fdir = {u: crf.posterior_mean(tab, post, tab["fdirect"][:, :, lev.index(u)], fit["cfg"])
                for u in a_lines.upper.unique() if u in lev}
        for nl in n_lines.itertuples():
            for al in a_lines.itertuples():
                lr = np.log(kN[nl.upper][0] / kA[al.upper][0])
                lr_alt = np.log(kN[nl.upper][1] / kA[al.upper][1])
                valid = np.isfinite(lr)
                vmass = pX[valid].sum()
                lo = med = hi = S = np.nan
                if vmass >= cfg["min_valid_mass"]:
                    lo, med, hi = _wq(lr[valid], pX[valid], q)
                    S = np.gradient(lr, lnTe)[i_med]
                rows.append(dict(sweep=c.sweep, x=c.x, n_wl=nl.wl_air, n_upper=nl.upper, ar_wl=al.wl_air,
                                 ar_upper=al.upper, Te_lo=c.Te_lo, Te_med=c.Te_med, Te_hi=c.Te_hi,
                                 Ne_med=c.Ne_med, k_ratio=np.exp(med), k_ratio_lo=np.exp(lo),
                                 k_ratio_hi=np.exp(hi), rel_err_k_Te=(hi - lo) / 2, S_Te=S,
                                 k_valid_mass=vmass,
                                 eedf_shape_log10=(lr[i_med] - lr_alt[i_med]) / np.log(10),
                                 f_direct_Ar=fdir.get(al.upper, np.nan)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 2. Te-corrected ratios and consistency between references
# ----------------------------------------------------------------------------
def corrected_ratios(kt, act, cfg=CONFIG):
    key = ["n_wl", "ar_wl", "sweep", "x"]
    c = act["cond"][key + ["mean", "err", "n_rep"]].rename(columns={"mean": "I_ratio", "err": "I_ratio_err"})
    d = kt.merge(c, on=key, how="inner")
    d["ratio_corr"] = d.I_ratio / d.k_ratio
    d["corr_rel_err"] = np.sqrt((d.I_ratio_err / d.I_ratio) ** 2 + d.rel_err_k_Te ** 2)
    d["corr_rel_err_meas"] = d.I_ratio_err / d.I_ratio
    g = d.groupby(["n_wl", "ar_wl", "sweep"])["ratio_corr"].transform("mean")
    d["corr_norm"] = d["ratio_corr"] / g
    # consensus of each N line per condition (median over Ar references) and deviation from it
    d["consensus"] = d.groupby(["n_wl", "sweep", "x"]).corr_norm.transform("median")
    d["dev"] = np.log(d.corr_norm / d.consensus)
    # n_N / n_Ar where both branching ratios are known
    b = pd.concat([act["n_lines"][["wl_air", "branching"]], act["a_lines"][["wl_air", "branching"]]])
    bmap = dict(zip(b.wl_air.round(4), b.branching))
    bN = d.n_wl.round(4).map(bmap)
    bA = d.ar_wl.round(4).map(bmap)
    lam = d.n_wl / d.ar_wl if cfg["fit"]["intensity_units"] == "energy" else 1.0
    d["n_ratio"] = d["ratio_corr"] * bA / bN * lam
    return d


def pair_summary(kt, d):
    k = kt.groupby(["n_wl", "ar_wl"]).agg(ar_upper=("ar_upper", "first"),
                                        rel_err_k_Te_med=("rel_err_k_Te", "median"),
                                        rel_err_k_Te_max=("rel_err_k_Te", "max"),
                                        S_Te_med=("S_Te", "median"),
                                        eedf_shape_log10_med=("eedf_shape_log10", "median"),
                                        k_valid_mass_min=("k_valid_mass", "min"),
                                        f_direct_Ar_med=("f_direct_Ar", "median"),
                                        f_direct_Ar_min=("f_direct_Ar", "min")).reset_index()
    if d.empty:
        return k
    s = d.groupby(["n_wl", "ar_wl"]).agg(dev_rms=("dev", lambda v: np.sqrt(np.mean(v ** 2))),
                                       corr_rel_err_med=("corr_rel_err", "median"),
                                       meas_rel_err_med=("corr_rel_err_meas", "median"),
                                       n_ratio_med=("n_ratio", "median")).reset_index()
    for sw, g in d.groupby("sweep"):
        cv = g.groupby(["n_wl", "ar_wl"])["ratio_corr"].agg(lambda v: v.std(ddof=1) / v.mean()).rename(f"cv_corr_{sw}")
        s = s.merge(cv.reset_index(), on=["n_wl", "ar_wl"], how="left")
    return k.merge(s, on=["n_wl", "ar_wl"], how="left").sort_values(["n_wl", "dev_rms"])


# ----------------------------------------------------------------------------
# 3. Plots
# ----------------------------------------------------------------------------
def _ar_label(al):
    return f"Ar {al.wl_air:.2f} ({al.upper})"


def plot_n_line(nl, a_lines, kt, d, sweeps, path):
    fig, axs = plt.subplots(2, len(sweeps), figsize=(6.5 * len(sweeps), 9), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for j, sw in enumerate(sweeps):
        for k, al in enumerate(a_lines.itertuples()):
            col = colors[k % 10]
            g = kt[(kt.n_wl == nl.wl_air) & (kt.ar_wl == al.wl_air) & (kt.sweep == sw["name"])].sort_values("x")
            if not g.empty:
                m = g.k_ratio.mean()
                axs[0, j].plot(g.x, g.k_ratio / m, "-", color=col, lw=1.5, label=_ar_label(al))
                axs[0, j].fill_between(g.x, g.k_ratio_lo / m, g.k_ratio_hi / m, color=col, alpha=0.15)
            h = d[(d.n_wl == nl.wl_air) & (d.ar_wl == al.wl_air) & (d.sweep == sw["name"])].sort_values("x")
            h = h[np.isfinite(h.corr_norm)]
            if not h.empty:       # relative (ln) errors -> multiplicative error bars
                f = np.exp(h.corr_rel_err)
                axs[1, j].errorbar(h.x, h.corr_norm, yerr=[h.corr_norm - h.corr_norm / f, h.corr_norm * (f - 1)],
                                   fmt="o-", color=col, ms=4, lw=1.2, capsize=2)
        axs[0, j].set_title(f"{sw['name']} sweep ({sw['note']})")
        axs[0, j].set_ylabel(r"$(k_N/k_{Ar})$ / mean   (16-84 % from $T_e$)")
        axs[1, j].set_ylabel(r"$(I_N/I_{Ar})/(k_N/k_{Ar})$ / mean   $\propto n_N/n_{Ar}$")
        for ax in axs[:, j]:
            ax.axhline(1, color="0.5", lw=0.8, ls=":")
            ax.set_xlabel(sw["xlabel"])
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(5, len(l)), fontsize=8, frameon=False)
    fig.suptitle(f"N I {nl.wl_air:.3f} nm ({nl.upper}, {nl.upper_desc}): k-ratio with fitted $T_e$ (top) "
                 f"and $T_e$-corrected line ratio (bottom)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_summary(summ, n_lines, a_lines, path):
    cols = [("rel_err_k_Te_med", r"relative 1$\sigma$ of $k_N/k_{Ar}$ from $T_e$ (median over conditions)", "viridis_r"),
            ("eedf_shape_log10_med", "EEDF-shape systematic: log10 of k-ratio (fitted EEDF / alternative)", "RdBu_r"),
            ("dev_rms", "rms ln-deviation from the N-line consensus", "viridis_r")]
    cols = [c for c in cols if c[0] in summ]
    fig, axs = plt.subplots(1, len(cols) + 1, figsize=(5.5 * (len(cols) + 1), 0.45 * len(n_lines) + 3))
    for ax, (c, title, cmap) in zip(axs, cols):
        M = summ.pivot(index="n_wl", columns="ar_wl", values=c).reindex(index=n_lines.wl_air, columns=a_lines.wl_air)
        v = np.nanmax(np.abs(M.values))
        im = ax.imshow(M.values, cmap=cmap, aspect="auto", vmin=-v if cmap == "RdBu_r" else 0, vmax=v)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M.values[i, j]):
                    ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(M.shape[1]), [f"{w:.1f}" for w in M.columns], rotation=90, fontsize=8)
        ax.set_yticks(range(M.shape[0]), [f"{w:.2f}" for w in M.index], fontsize=8)
        ax.set_xlabel("Ar I reference [nm]"); ax.set_ylabel("N I line [nm]")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axs[-1]
    f = summ.groupby("ar_wl").agg(med=("f_direct_Ar_med", "first"), lo=("f_direct_Ar_min", "first"))
    f = f.reindex(a_lines.wl_air)
    floor = 1e-8
    ax.barh(range(len(f)), f.med.clip(lower=floor), left=floor, color="C0", alpha=0.7,
            label="median over conditions")
    ax.plot(f.lo.clip(lower=floor), range(len(f)), "k|", ms=10, label="minimum")
    for i, v in enumerate(f.med):
        ax.text(1.5 * max(v, floor), i, f"{v:.1e}", va="center", fontsize=7)
    ax.set_yticks(range(len(f)), [_ar_label(a) for a in a_lines.itertuples()], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(floor, 1)
    ax.set_xlabel("direct ground-state fraction of Ar upper-level production (CR model)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 4. Driver
# ----------------------------------------------------------------------------
def run(cfg=CONFIG):
    os.makedirs(cfg["outdir"], exist_ok=True)
    out = lambda f: os.path.join(cfg["outdir"], f)
    print("=== 1. CR-model fit of Te, Ne ===")
    fit = crf.run(cfg["fit"])
    print("\n=== 2. actinometry line selection ===")
    act = als.run(cfg["actinometry"])
    if act is None:
        return None
    print("\n=== 3. k ratios with the Te posterior ===")
    kt = k_ratio_table(fit, act, cfg)
    d = corrected_ratios(kt, act, cfg)
    summ = pair_summary(kt, d)
    kt.to_csv(out("k_ratio_per_condition.csv"), index=False)
    d.to_csv(out("corrected_ratio_per_condition.csv"), index=False)
    summ.to_csv(out("k_ratio_summary.csv"), index=False)
    per_dir = out("per_N_line")
    os.makedirs(per_dir, exist_ok=True)
    for nl in act["n_lines"].itertuples():
        plot_n_line(nl, act["a_lines"], kt, d, cfg["actinometry"]["sweeps"],
                    os.path.join(per_dir, f"N_{nl.wl_air:.3f}_{nl.upper}.png"))
    plot_summary(summ, act["n_lines"], act["a_lines"], out("k_ratio_summary.png"))
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(summ.round(3).to_string(index=False))
    print("\nwritten to", cfg["outdir"])
    return dict(fit=fit, act=act, kt=kt, d=d, summary=summ)


if __name__ == "__main__":
    RESULTS = run(CONFIG)
