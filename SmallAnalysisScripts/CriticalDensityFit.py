# -*- coding: utf-8 -*-
"""
CriticalDensityFit.py

The CR fits with Ne pinned at the critical density of the microwave field, for an
overdense discharge:  n_c = eps0 m_e (2 pi f)^2 / e^2  (7.3e16 m^-3 at 2.42 GHz).

Uses the saved fits of Experimental_Data/ExperimentalDataAnalysis/ActinometryNitrogenContent.py
(one per EEDF and N2 fraction; each condition with the model of its own N2 fraction), so
nothing is refitted: chi^2 on the (E/N, Ne) grid is recovered from the saved posterior,
chi^2 = chi^2_min + 2 s^2 ln(p_max / p) with s the Birge factor of the fit. Then per condition
  profile    chi^2 minimised over E/N at every Ne, in units of the fit's s^2
             (Delta = 1 ~ 1 sigma): does the spectrum allow Ne = n_c?
  pinned     Ne = n_c, only E/N free: Te_eff, chi^2/dof, the CR 1s5 density (against the
             absorption estimate) and the N-atom density of both actinometry variants
  overdense  the free posterior restricted to Ne >= n_c
The pure-Ar condition (0 % N2) is fitted here with the EEDFPhysics tables (BOLSIG+
microwave and DC; profile and pinned fit only, no N lines).

Output in Experimental_Data/Output/CriticalDensity/ (CriticalDensity_RDW4s/ with the RDW 4s cross
sections, CRFitNeTe.CONFIG['xsec_4s']; CriticalDensity_noAT/ without the Ar-atom 2p transfer,
CONFIG['atom_transfer']), incl. chi2_profiles.csv.  Run from Spyder: edit the settings, press F5.
"""
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT_DIR, "Experimental_Data", "ExperimentalDataAnalysis"))
sys.path.insert(0, HERE)
import ActinometryNitrogenContent as anc            # noqa: E402
import ActinometryRates as ar                       # noqa: E402
import ActinometryLineSelection as als              # noqa: E402
import ActinometryTeUncertainty as atu              # noqa: E402
import EEDFPhysicsFitComparison as pc               # noqa: E402

crf, he = anc.crf, anc.he

# ---- settings -------------------------------------------------------------------
FREQ_HZ = 2.42e9                                    # microwave source
N_C = constants.epsilon_0 * constants.m_e * (2 * np.pi * FREQ_HZ) ** 2 / constants.e ** 2
MEASURED_1S5 = (1e17, 8e17)                         # m^-3, absorption estimate (path length uncertain)
PURE_AR = [("microwave", he.BOLSIG_FOLDER / "Ar_Biagi_bolsig_mw"), ("dc", he.BOLSIG_FOLDER / "Ar_Biagi_bolsig")]
ECOL = {"microwave": "C3", "dc": "C0"}
OUTDIR = crf.output_dir("CriticalDensity")          # + crf.model_suffix: "_RDW4s", "_noAT", ... for other models


# ---- fits -----------------------------------------------------------------------
def load_fit(eedf, pct):
    """A saved ActinometryNitrogenContent fit, in the form CRFitNeTe.run returns."""
    cfg = anc.fit_config(pct, eedf)
    with contextlib.redirect_stdout(io.StringIO()):
        tab = crf.build_model_table(cfg)
        eedfs = crf.eedf_axis(cfg)[0]
    grid = crf.fine_grid(tab, cfg)
    z = np.load(os.path.join(cfg["outdir"], "posteriors.npz"))
    return dict(tab=tab, grid=grid, Te_f=crf.te_eff_fine(tab, grid), eedfs=eedfs, cfg=cfg,
                posts={k: z[k] for k in z.files if "|" in k},
                cond=pd.read_csv(os.path.join(cfg["outdir"], "fit_conditions.csv")))


def chi2_grid(post, chi2_min, birge):
    with np.errstate(divide="ignore"):
        return chi2_min + 2 * birge ** 2 * (np.log(post.max()) - np.log(post))


def _q(values, weights):
    return atu._wq(np.ravel(values), np.ravel(weights), np.array([0.16, 0.5, 0.84]))


def analyse(fit, key, chi2_min, dof, birge, post):
    """Profile, pinned and overdense summaries of one condition; returns (row, pinned posterior, profile)."""
    tab, (fX, fN), Te_f = fit["tab"], fit["grid"], fit["Te_f"]
    chi2 = chi2_grid(post, chi2_min, birge)
    profile = (np.min(chi2, axis=0) - chi2_min) / birge ** 2
    j = int(np.argmin(np.abs(fN - np.log(N_C))))
    row = chi2[:, j]
    m = np.min(row)
    s2 = max(1.0, m / (dof + 1)) if dof + 1 > 0 else 1.0
    p = np.exp(-0.5 * (row - m) / s2)
    p[~np.isfinite(p)] = 0.0
    pin = np.zeros_like(post)
    pin[:, j] = p / p.sum()
    T2 = np.broadcast_to(Te_f[:, None], post.shape) if np.ndim(Te_f) == 1 else Te_f
    over = post * (fN[None, :] >= np.log(N_C))
    lev = list(tab["levels"])
    n1s5 = lambda w: np.exp(crf.posterior_mean(tab, w, np.log(tab["dens"][:, :, lev.index("4s1")]), fit["cfg"]))
    Tp, To = _q(T2, pin), _q(T2, over)
    No = np.exp(_q(np.broadcast_to(fN[None, :], post.shape), over))
    return dict(Ne_c=np.exp(fN[j]), delta_chi2_at_nc=profile[j], chi2_red_free=chi2_min / dof,
                chi2_red_pinned=m / (dof + 1), x_pinned=np.exp(fX[np.argmin(row)]),
                Te_pinned=Tp[1], Te_pinned_lo=Tp[0], Te_pinned_hi=Tp[2], n_1s5_pinned=n1s5(pin),
                overdense_mass=over.sum(), Te_overdense=To[1], Ne_overdense=No[1],
                Ne_overdense_lo=No[0], Ne_overdense_hi=No[2]), pin, (np.exp(fN), profile)


def pure_ar():
    """Profile and pinned fit of the pure-Ar condition with the EEDFPhysics tables."""
    out = []
    for eedf, folder in PURE_AR:
        with contextlib.redirect_stdout(io.StringIO()):
            cfg, tab, feats, ft, grid, Te_f, M = pc.prepare(folder)
        d = ft.query("sweep == 'N2 fraction' and x == 0")
        s, post, _ = crf.fit_block(d, M, feats, grid, Te_f, cfg)
        fit = dict(tab=tab, grid=grid, Te_f=Te_f, cfg=cfg)
        r, _, prof = analyse(fit, None, s["chi2_min"], s["dof"], s["birge"], post)
        lev = list(tab["levels"])
        r.update(sweep="N2 fraction", x=0.0, eedf=eedf, Te_free=s["Te_med"], Ne_free=s["Ne_med"],
                 n_1s5_free=np.exp(crf.posterior_mean(tab, post, np.log(tab["dens"][:, :, lev.index("4s1")]), cfg)))
        out.append((r, prof))
    return out


# ---- plots ----------------------------------------------------------------------
def plot_profiles(profiles, path):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, eedf in zip(axs, ("microwave", "dc")):
        ax.axvspan(1e12, N_C, color="0.92", lw=0, label="underdense ($N_e < n_c$)")
        ax.axvline(N_C, color="k", lw=1.5, label=f"$n_c$ = {N_C:.2e} m$^{{-3}}$ ({FREQ_HZ / 1e9:g} GHz)")
        for lev, ls in ((1, ":"), (4, "--")):
            ax.axhline(lev, color="0.4", lw=0.8, ls=ls)
        sel = [p for p in profiles if p[0] == eedf]
        n2 = sorted({x for _, s, x, _ in sel if s == "N2 fraction"})
        cn = dict(zip(n2, plt.cm.viridis(np.linspace(0, 0.85, max(len(n2), 1)))))
        pw = sorted({x for _, s, x, _ in sel if s == "Power"})
        cp = dict(zip(pw, plt.cm.autumn(np.linspace(0, 0.8, max(len(pw), 1)))))
        for _, sweep, x, (Ne, prof) in sel:
            if sweep == "N2 fraction":
                ax.semilogx(Ne, prof, "-", color="k" if x == 0 else cn[x], lw=2.6 if x == 0 else 1.6,
                            label=f"{x:g} % N$_2$" + (" (pure Ar)" if x == 0 else ""))
            else:
                ax.semilogx(Ne, prof, "--", color=cp[x], lw=1.4, label=f"{x:g} W (2.4 % N$_2$)")
        ax.set_xlim(1e13, 3e19)
        ax.set_ylim(0, 30)
        ax.set_xlabel("$N_e$ [m$^{-3}$]")
        ax.set_title(f"{eedf} EEDF: $\\chi^2$ profiled over E/N")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel(r"$\Delta\chi^2 / s^2$ against the best fit (1 and 4 dotted / dashed)")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_summary(res, act, sweeps, path):
    fig, axs = plt.subplots(4, len(sweeps), figsize=(7 * len(sweeps), 16), squeeze=False)
    for j, sw in enumerate(sweeps):
        r = res[res.sweep == sw["name"]]
        a = act[act.sweep == sw["name"]] if len(act) else act
        for eedf, col in ECOL.items():
            g = r[r.eedf == eedf].sort_values("x")
            if g.empty:
                continue
            axs[0, j].plot(g.x, g.Te_free, "o--", color=col, mfc="white", label=f"{eedf}, Ne free")
            axs[0, j].errorbar(g.x, g.Te_pinned, yerr=[g.Te_pinned - g.Te_pinned_lo, g.Te_pinned_hi - g.Te_pinned],
                               fmt="o-", color=col, capsize=3, label=f"{eedf}, Ne = n$_c$")
            axs[1, j].plot(g.x, g.delta_chi2_at_nc, "o-", color=col, label=eedf)
            axs[2, j].semilogy(g.x, g.n_1s5_free, "o--", color=col, mfc="white", label=f"{eedf}, Ne free")
            axs[2, j].semilogy(g.x, g.n_1s5_pinned, "o-", color=col, label=f"{eedf}, Ne = n$_c$")
            h = a[a.eedf == eedf] if len(a) else a
            for v, m in (("direct", "o"), ("stepwise", "s")):
                k = h[h.variant == v].sort_values("x") if len(h) else h
                if len(k):
                    axs[3, j].errorbar(k.x, k.n_N, yerr=[k.n_N - k.n_N_lo, k.n_N_hi - k.n_N], fmt=m + "-",
                                       color=col, mfc=col if v == "direct" else "white", capsize=3,
                                       label=f"{eedf}, {v}")
        axs[1, j].axhline(1, color="0.4", ls=":", lw=0.8)
        axs[1, j].axhline(4, color="0.4", ls="--", lw=0.8)
        axs[2, j].axhspan(*MEASURED_1S5, color="C2", alpha=0.15, label="absorption estimate")
        if sw["name"] == "N2 fraction":
            f = np.linspace(0.4, 3.1, 40) / 100
            N = he.Torr2Volume(anc.CONFIG["fit"]["P_Torr"], anc.CONFIG["fit"]["Tg"])
            axs[3, j].plot(100 * f, 2 * f * N, "k:", lw=1.2, label="complete dissociation")
        axs[3, j].set_yscale("log")
        axs[0, j].set_ylabel(r"$T_{e,\mathrm{eff}}$ [eV]")
        axs[1, j].set_ylabel(r"$\Delta\chi^2/s^2$ for pinning $N_e = n_c$")
        axs[2, j].set_ylabel("CR 1s$_5$ density [m$^{-3}$]")
        axs[3, j].set_ylabel("N-atom density, $N_e = n_c$ [m$^{-3}$]")
        axs[0, j].set_title(f"{sw['name']} sweep ({sw['note']})")
        for ax in axs[:, j]:
            ax.set_xlabel(sw["xlabel"])
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=7)
    fig.suptitle(f"CR fits with $N_e$ pinned at the critical density $n_c$ = {N_C:.2e} m$^{{-3}}$ "
                 f"({FREQ_HZ / 1e9:g} GHz)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"critical density at {FREQ_HZ / 1e9:g} GHz: n_c = {N_C:.3e} m^-3")
    acfg = anc.CONFIG["actinometry"]
    xs = ar.load_all_cross_sections()
    with contextlib.redirect_stdout(io.StringIO()):
        nl, al = als.select_lines(acfg, xs)
        meas = als.filter_export_flags(als.measure_sweeps(pd.concat([nl, al]), acfg), acfg)
        _, cm = als.pair_ratios(meas, nl, al, acfg)
    act_in = dict(xs=xs, n_lines=nl, a_lines=al, cond=cm)
    fractions = sorted({anc.n2_percent(s, x) for s, x in zip(cm.sweep, cm.x)})

    rows, profiles, pinned_fits = [], [], {}
    for eedf in anc.CONFIG["eedfs"]:
        for pct in fractions:
            fit = load_fit(eedf, pct)
            cond = fit["cond"].copy()
            posts = {}
            for c in cond.itertuples():
                if anc.n2_percent(c.sweep, c.x) != pct:
                    continue                    # each condition from the model of its own N2 fraction
                key = f"{c.sweep}|{c.x:g}"
                r, pin, prof = analyse(fit, key, c.chi2_min, c.dof, c.birge, fit["posts"][key])
                r.update(sweep=c.sweep, x=c.x, eedf=eedf, Te_free=c.Te_med, Ne_free=c.Ne_med, n_1s5_free=c.n_4s1)
                rows.append(r)
                profiles.append((eedf, c.sweep, c.x, prof))
                posts[key] = pin
                i = cond.index[(cond.sweep == c.sweep) & (cond.x == c.x)][0]
                cond.loc[i, ["Te_med", "Ne_med", "Ne_lo", "Ne_hi", "x_med", "chi2_red", "edge_Ne", "edge_x"]] = [
                    r["Te_pinned"], N_C, N_C, N_C, r["x_pinned"], r["chi2_red_pinned"], False, False]
            pinned_fits[eedf, pct] = dict(fit, posts=posts, cond=cond)
    for r, prof in pure_ar():
        rows.append(r)
        profiles.append((r["eedf"], r["sweep"], r["x"], prof))
    res = pd.DataFrame(rows)
    with contextlib.redirect_stdout(io.StringIO()):
        _, act = anc.condition_estimates(pinned_fits, act_in, anc.CONFIG)
    res.to_csv(os.path.join(OUTDIR, "pinned_fits.csv"), index=False)
    act.to_csv(os.path.join(OUTDIR, "nN_pinned_nc.csv"), index=False)
    pd.DataFrame([dict(eedf=e, sweep=sw, x=x, Ne=n, delta_chi2_s2=d)
                  for e, sw, x, (Ne, prof) in profiles for n, d in zip(Ne, prof)]
                 ).to_csv(os.path.join(OUTDIR, "chi2_profiles.csv"), index=False)
    plot_profiles(profiles, os.path.join(OUTDIR, "chi2_profile_Ne.png"))
    plot_summary(res, act, acfg["sweeps"], os.path.join(OUTDIR, "pinned_summary.png"))
    cols = ["eedf", "sweep", "x", "Ne_free", "delta_chi2_at_nc", "chi2_red_free", "chi2_red_pinned", "Te_free",
            "Te_pinned", "n_1s5_free", "n_1s5_pinned", "overdense_mass", "Ne_overdense"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(res.sort_values(["eedf", "sweep", "x"])[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        if len(act):
            print(act[["eedf", "sweep", "x", "variant", "Te_eff", "n_N", "ln_err_stat", "birge_pairs",
                       "dissociation"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nfigures in {OUTDIR}")


if __name__ == "__main__":
    main()
