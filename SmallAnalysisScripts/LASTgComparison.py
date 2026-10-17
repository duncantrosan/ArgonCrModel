# -*- coding: utf-8 -*-
"""
LASTgComparison.py

Ar(1s5) density of the CR model with the gas temperature measured by laser absorption (LAS), against the
LAS 1s5 density; microwave EEDF with Ne pinned at the critical density n_c.

LAS data (Experimental_Data/LAS_Data/Master_AllMeasurements.csv, measurements in progress): 11 runs, Jul-Sep
2026, 0.6-1.5 Torr, 30-110 W, 0-12 % N2; gas temperature from the Doppler width (T_gas_K) and the 1s5 density
(N_s_cm3).  Rows within LAS_FREQ_MHZ (the source was retuned to 2418-2432 MHz at the 1 and 3 % N2 points of the
1 Torr run below 85 W - the only low-N2 data at 1 Torr); Tg outside TG_VALID (failed fits, e.g. 108 K, 10831 K)
is dropped.

1. Tg trend   Tg(p, P, x) = T0 + a_p (p - 1 Torr) + a_P (P - 80 W) + a_x x + a_xx x^2      (x = % N2)
     within each run   one offset per run + the power and N2 terms (soft-L1, points beyond 4 sigma dropped):
                       the power and N2 dependence from each run's own sweep
     between runs      robust line through the run offsets (Tg at 80 W, 0 % N2) against pressure - one pressure
                       per run; the offsets scatter by ~100 K and the 22 Sep 0.7 Torr run lies ~350 K above
   Tg of every OES condition from the trend, rounded to TG_STEP.
2. OES conditions (echelle): pure-Ar pressure sweep (80 W, 600-1500 mTorr), N2 sweep (85 W, 1 Torr, 0-3 %) and
   power sweep (2.4 % N2, 1 Torr); spectra without the intensity calibration are dropped (45 W has none left).
3. Per (N2 %, pressure, Tg): BOLSIG+ microwave library (Biagi Ar or Ar/N2, FREQ_HZ, omega/N with N = p / k Tg,
   BOLSIG+ gas temperature Tg) InputData/Bolsig/<Ar_Biagi|ArN2_<x>pct>_bolsig_mw<f>GHz_<Tg>K_<p>mTorr, and the CR
   table at (p, Tg) (CRFitNeTe: ground density, metastable diffusion, trapping, atom transfer, N2 quenching and
   dilution), MicrowaveLowENFit.N_WORKERS processes at a time.
4. Fit of every condition (CRFitNeTe: 9 Ar I features, one scale per spectrum, response tilt, repeat scatter
   pooled per measurement group) with Ne = n_c and E/N refitted: CR 1s densities at the best E/N, band
   Delta chi^2 <= 1 s^2 along E/N; Delta chi^2(n_c) against the free fit for reference.
   The earlier assumptions are fitted the same way for comparison: pressure sweep at Tg = 800 K
   (PressureSweepAnalysis tables), N2 and power sweeps at 300 K (MicrowaveLowENFit lowEN tables, 2.45 GHz).
5. LAS 1s5 at every OES condition (las_1s5_at): LAS points within DP_W of the OES power (median per N2 value);
   per run 1/n linear in % N2 (quenching, 1/n ~ nu_0 + k_Q x N) between the measured N2 values, the nearest
   measured value up to X_EXTRAP outside them (1s5 is flat between 0 and 2 % N2 in the July runs); at one
   pressure the runs with the best match (measured > interpolated > extrapolated) are averaged; ln n linear in
   pressure between the LAS pressures.

Output in Experimental_Data/Output/LASTgComparison/ (one graph per file):
  las_tg_trend.csv (trend coefficients and run offsets), conditions.csv (Tg and LAS 1s5 of every OES condition),
  comparison.csv (fit at n_c and CR 1s densities, LAS Tg and earlier Tg, and the LAS 1s5)
  figures/las_tg_vs_N2.png, las_tg_run_offsets.png   LAS Tg and the trend
  figures/tg_<sweep>.png                               Tg of the CR model against the sweep variable
  figures/1s5_<sweep>.png                              CR 1s5 (LAS Tg, earlier Tg) and LAS 1s5
  figures/1s5_parity.png                               CR against LAS 1s5 for every condition
  figures/chi2_<sweep>.png, Te_eff_<sweep>.png         fit quality and Te_eff at n_c
Run from Spyder (F5) or python SmallAnalysisScripts/LASTgComparison.py (--plan: Tg and the jobs only, nothing
is built).  Libraries and CR tables are cached; delete one to rebuild it.
"""
# The pipeline half of this script (LAS table, Tg trend, LAS 1s5, BOLSIG+ libraries, CR-table settings, the fit of
# one condition) is pipeline/LASAndEEDFLibraries.py; this file keeps the comparison with the earlier assumed Tg and
# its plots, and takes everything else from there.  Main.py does not use this file.

import os
import re
import subprocess
import sys
import time
from fnmatch import fnmatch
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import PressureSweepAnalysis as psa                 # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pipeline"))
import LASAndEEDFLibraries as P                    # noqa: E402 - the pipeline half
from LASAndEEDFLibraries import *                  # noqa: E402,F401,F403
from LASAndEEDFLibraries import (_at_x, _key, _unkey, read_las)   # noqa: E402,F401

ov, crf, mlf, he = psa.ov, psa.crf, psa.mlf, psa.he
als = crf.als

# ---- settings of this analysis (the pipeline settings are those of LASAndEEDFLibraries) ---------------------
FIGDIR = os.path.join(OUTDIR, "figures")
FIGSIZE = psa.FIGSIZE
P.GROUPS["pressure"] = dict(sweeps=[psa.SWEEP], measure_dir=psa.MEASURE_DIR)
GROUPS = P.GROUPS
OES = {   # sweep -> (p [Torr], P [W], % N2) of x, Tg assumed before, labels
    "Pressure": dict(state=lambda x: (x / 1000, 80.0, 0.0), Tg_old=psa.TG, tag="pressure",
                     xlabel="Pressure [mTorr]", note="pure Ar, 80 W"),
    "N2 fraction": dict(state=lambda x: (1.0, 85.0, x), Tg_old=crf.CONFIG["Tg"], tag="N2",
                        xlabel="N$_2$ in the feed [%]", note="85 W, 1 Torr"),
    "Power": dict(state=lambda x: (1.0, x, mlf.anc.CONFIG["power_sweep_N2"]), Tg_old=crf.CONFIG["Tg"], tag="power",
                  xlabel="Power [W]", note=f"{mlf.anc.CONFIG['power_sweep_N2']:g} % N$_2$, 1 Torr"),
}

# ---- 2. OES conditions ----------------------------------------------------------------
def oes_conditions():
    """Every OES condition with a calibrated spectrum: sweep, x, measurement group, p [Torr], P [W], % N2."""
    sweeps = [sw for G in GROUPS.values() for sw in G["sweeps"]]
    fl = als.export_flags(dict(als.CONFIG, sweeps=sweeps))
    bad = set(zip(fl.sweep, fl.file)) - {(r.sweep, r.file) for r in fl.itertuples()
                                         if fnmatch(r.export_flags, als.CONFIG["export_flags"])}
    rows = []
    for g, G in GROUPS.items():
        for sw in G["sweeps"]:
            xs = set()
            for f in glob(os.path.join(sw["folder"], "*.spa")):
                name = os.path.splitext(os.path.basename(f))[0]
                px = sw["parse"](name)
                if px and (sw["name"], name) not in bad:
                    xs.add(px[0])
            for x in sorted(xs):
                p, P, n2 = OES[sw["name"]]["state"](x)
                rows.append(dict(sweep=sw["name"], x=x, group=g, p_Torr=p, P_W=P, N2=n2))
    return pd.DataFrame(rows)


def plan(las=None):
    """LAS Tg trend, every OES condition with its Tg (trend, rounded) and LAS 1s5."""
    las = load_las() if las is None else las
    model, off, tgfit = fit_tg_trend(las)
    cond = oes_conditions()
    cond["p_mTorr"] = (1000 * cond.p_Torr).round(6)
    cond["Tg_trend"] = tg_trend(model, cond.p_Torr, cond.P_W, cond.N2)
    cond["Tg"] = tg_round(cond.Tg_trend)
    cond["Tg_old"] = cond.sweep.map(lambda s: OES[s]["Tg_old"])
    cond["N_gas"] = he.Torr2Volume(cond.p_Torr.to_numpy(), cond.Tg.to_numpy())
    cond["N_gas_old"] = he.Torr2Volume(cond.p_Torr.to_numpy(), cond.Tg_old.to_numpy())
    ref = pd.DataFrame([las_1s5_at(las, r.p_Torr, r.P_W, r.N2) for r in cond.itertuples()])
    cond = pd.concat([cond, ref.add_prefix("las_1s5_").rename(columns={"las_1s5_n": "las_1s5"})], axis=1)
    return las, model, off, tgfit, cond


# ---- 3. CR tables -----------------------------------------------------------------------
def table_cfg(key, group="pressure"):
    """CR table settings of key (LASAndEEDFLibraries.table_cfg; here the default group is the pressure sweep)."""
    return P.table_cfg(key, group)


def _run_workers(flag, keys, what):
    """Library workers run the pipeline file; CR-table workers ('--build', the old-Tg tables) run this file."""
    P._run_workers(flag, keys, what, script=None if flag.startswith("--lib") else os.path.abspath(__file__))


def old_cfg(c):
    """Settings of the earlier assumption for condition c: pressure sweep at Tg = 800 K (PressureSweepAnalysis),
    N2 and power sweeps at 300 K (MicrowaveLowENFit lowEN, 2.45 GHz)."""
    if c.sweep == "Pressure":
        return psa.fit_cfg("microwave", c.x)
    return mlf.job_cfg(f"lowEN:{c.N2:g}", build=False)


def jobs(cond):
    """(missing libraries, CR tables to build) for the conditions; table keys 'new:<key>' (LAS Tg) and
    'old:<sweep>:<x>' (earlier Tg), one job per table file."""
    keys = cond.apply(lambda r: _key(r.N2, r.p_mTorr, r.Tg), axis=1)
    libs = sorted({k for k in keys if not he.IsBolsigLibrary(lib_folder(*_unkey(k)))})
    tabs = sorted({f"new:{k}" for k in keys if not crf.table_is_current(table_cfg(k))})
    olds = sorted({f"old:{r.sweep}:{r.x:g}" for r in cond.itertuples() if not crf.table_is_current(old_cfg(r))},
                  key=lambda k: old_cfg(_old_row(k))["eedf"])
    seen, old_jobs = set(), []
    for k in olds:                                  # one job per table (the power sweep shares 2.4 %)
        path = crf._table_path(old_cfg(_old_row(k)))
        if path not in seen:
            seen.add(path)
            old_jobs.append(k)
    return libs, tabs + old_jobs


def _old_row(key):
    _, sweep, x = key.split(":", 2)
    x = float(x)
    p, P, n2 = OES[sweep]["state"](x)
    return pd.Series(dict(sweep=sweep, x=x, N2=n2))


def build_table(key):
    kind, rest = key.split(":", 1)
    if kind == "new":
        cfg = table_cfg(rest)
    else:
        cfg = old_cfg(_old_row(key))
    tab = crf.build_model_table(cfg)
    print(f"{key}: {len(tab['x_grid'])} E/N rows from {tab['x_grid'][0]:g} Td, "
          f"{(~tab['converged']).sum()} not converged")


# ---- 4. fits ----------------------------------------------------------------------------
measurements = P.measurements


def fit_condition(cfg, sweep, x, feats, ft):
    """LASAndEEDFLibraries.fit_condition with Ne pinned at the n_c of MicrowaveLowENFit (as before)."""
    return P.fit_condition(cfg, sweep, x, feats, ft, mlf.N_C)


def fit_all_conditions(cond):
    """Fit at n_c of every condition with the LAS Tg ('new') and the earlier Tg ('old') -> comparison table."""
    rows = []
    for g, cg in cond.groupby("group", sort=False):
        first = cg.iloc[0]
        feats, ft = measurements(table_cfg(_key(first.N2, first.p_mTorr, first.Tg), g))
        print(f"\n{g}: {len(feats)} features: {', '.join(feats.feature)}", flush=True)
        for c in cg.itertuples():
            new = fit_condition(table_cfg(_key(c.N2, c.p_mTorr, c.Tg), g), c.sweep, c.x, feats, ft)
            old = fit_condition(old_cfg(c), c.sweep, c.x, feats, ft)
            rows.append(dict(c._asdict(), **new, **{f"old_{k}": v for k, v in old.items()}))
            print(f"  {c.sweep:<11s} {c.x:6g}  Tg {c.Tg:4.0f} K: E/N {new['EN_nc']:6.1f} Td  Te_eff {new['Te_nc']:.2f} eV  "
                  f"chi2/dof {new['chi2_red_nc']:5.2f}  1s5 {new['n_1s5']:.2e}   | {c.Tg_old:3.0f} K: chi2/dof "
                  f"{old['chi2_red_nc']:5.2f}  1s5 {old['n_1s5']:.2e}   | LAS 1s5 {c.las_1s5:.2e} ({c.las_1s5_how})",
                  flush=True)
    comp = pd.DataFrame(rows).drop(columns="Index")
    comp["ratio_cr_las"] = comp.n_1s5 / comp.las_1s5
    comp["old_ratio_cr_las"] = comp.old_n_1s5 / comp.las_1s5
    return comp


# ---- 5. plots -------------------------------------------------------------------------
def _las_points(las, sweep):
    """Raw LAS 1s5 / Tg points at the conditions of a sweep, as (label, rows, x column) series."""
    v = las[(las.P - {"Pressure": 80.0, "N2 fraction": 85.0}.get(sweep, 0)).abs() <= DP_W + 1e-9] \
        if sweep != "Power" else las
    if sweep == "Pressure":
        g = v[v.x == 0]
        return [("LAS points (0 % N$_2$, 75-85 W)", g.assign(X=1000 * g.p))]
    if sweep == "N2 fraction":
        return [(f"LAS points ({p:g} Torr, 80-90 W)", v[(v.p == p) & (v.x <= 4)].assign(X=lambda h: h.x))
                for p in (0.9, 1.0)]
    n2 = mlf.anc.CONFIG["power_sweep_N2"]               # the 1 Torr N2 values that bracket it (1 and 3 %)
    v = las[las.p == 1.0]
    xs = np.sort(v.x.unique())
    return [(f"LAS points (1 Torr, {x:g} % N$_2$)", v[v.x == x].assign(X=lambda h: h.P))
            for x in (xs[xs <= n2].max(), xs[xs >= n2].min())]


def plot_las_tg(model, off, tgfit, outdir):
    """LAS Tg against % N2 (75-85 W) with the trend at each LAS pressure, and the run offsets against pressure."""
    ps = sorted(tgfit.p.unique())
    col = {p: plt.cm.viridis(i / max(len(ps) - 1, 1)) for i, p in enumerate(ps)}
    fig, ax = plt.subplots(figsize=FIGSIZE)
    v = tgfit[(tgfit.P - 80).abs() <= DP_W + 1e-9]
    for (run, date, p), g in v.groupby(["run", "date", "p"]):
        mk = "o" if date < "2026-09-01" else "^"
        ax.plot(g.x[~g.outlier], g.Tg[~g.outlier], mk, color=col[p], ms=4, alpha=0.8,
                label=f"{date} {run[:4]} ({p:g} Torr)")
        ax.plot(g.x[g.outlier], g.Tg[g.outlier], "x", color=col[p], ms=6)
    xx = np.linspace(0, 12, 50)
    for p in ps:
        ax.plot(xx, tg_trend(model, p, 80.0, xx), "-", color=col[p], lw=1.5)
    ax.plot([], [], "k-", lw=1.5, label="trend at 80 W (one line per pressure)")
    ax.plot([], [], "kx", label="outlier (> 4 $\\sigma$ within the run)")
    ax.set_ylabel("$T_g$ [K] (LAS, Doppler width)")
    ax.legend(fontsize=7, ncol=2)
    psa._save(fig, ax, os.path.join(outdir, "las_tg_vs_N2.png"),
              f"LAS gas temperature, 75-85 W (o July, $\\triangle$ September)\ntrend: "
              f"{model['a_x']:.0f} K/% N$_2$ ({model['a_xx']:+.2f} K/%$^2$), {10 * model['a_P']:.0f} K/10 W, "
              f"{model['a_p']:.0f} K/Torr", xlabel="N$_2$ in the feed [%]", legend=False)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for r in off.itertuples():
        mk = "o" if r.date < "2026-09-01" else "^"
        ax.plot(r.p, r.T_80W_0pct, mk, color=col[r.p], ms=8, mec="k", mew=0.5)
        ax.annotate(f"{r.date[5:]} ({r.run[:4]})", (r.p, r.T_80W_0pct), xytext=(5, 0), textcoords="offset points",
                    fontsize=7, va="center")
    pp = np.linspace(0.55, 1.55, 20)
    ax.plot(pp, model["T0"] + model["a_p"] * (pp - 1), "k-", label=f"robust line: {model['T0']:.0f} K at 1 Torr, "
                                                                   f"{model['a_p']:.0f} K/Torr")
    ax.set_ylabel("run offset: $T_g$ at 80 W, 0 % N$_2$ [K]")
    psa._save(fig, ax, os.path.join(outdir, "las_tg_run_offsets.png"),
              "LAS gas temperature of each run (power and N$_2$ dependence removed)", xlabel="Pressure [Torr]")


def plot_tg(cond, las, sweep, outdir):
    o = OES[sweep]
    c = cond[cond.sweep == sweep].sort_values("x")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for k, (lab, g) in enumerate(_las_points(las[las.valid_Tg], sweep)):
        ax.plot(g.X, g.Tg, "o^"[k % 2], color="0.6", ms=4, alpha=0.7, label=lab)
    ax.plot(c.x, c.Tg_trend, "-", color="C3", lw=1.5, label="LAS trend")
    ax.plot(c.x, c.Tg, "s", color="C3", ms=6, label=f"CR model / library (rounded to {TG_STEP:g} K)")
    ax.axhline(o["Tg_old"], color="0.4", ls="--", lw=1.2, label=f"earlier assumption ({o['Tg_old']:g} K)")
    ax.set_ylabel("$T_g$ [K]")
    psa._save(fig, ax, os.path.join(outdir, f"tg_{o['tag']}.png"), f"Gas temperature of the CR model: {o['note']}",
              xlabel=o["xlabel"])


def plot_1s5(comp, las, sweep, outdir):
    o = OES[sweep]
    c = comp[comp.sweep == sweep].sort_values("x")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for k, (lab, g) in enumerate(_las_points(las[las.valid_n], sweep)):
        ax.plot(g.X, g.n1s5, "o^"[k % 2], color="0.65", ms=4, alpha=0.7, label=lab)
    ax.fill_between(c.x, c.n_1s5_lo, c.n_1s5_hi, color="C3", alpha=0.15, lw=0)
    ax.plot(c.x, c.n_1s5, "s-", color="C3", ms=6, label="CR, LAS $T_g$ (band: $\\Delta\\chi^2 \\leq 1\\,s^2$ along E/N)")
    ax.fill_between(c.x, c.old_n_1s5_lo, c.old_n_1s5_hi, color="0.5", alpha=0.12, lw=0)
    ax.plot(c.x, c.old_n_1s5, "o--", color="0.35", mfc="white", ms=5, label=f"CR, $T_g$ = {o['Tg_old']:g} K (earlier)")
    meas = c.las_1s5_how == "measured"
    yerr = c.las_1s5 * c.las_1s5_rel_err
    ax.errorbar(c.x[meas], c.las_1s5[meas], yerr=yerr[meas], fmt="D", color="k", ms=6, capsize=3,
                label="LAS at the condition (measured)")
    ax.errorbar(c.x[~meas], c.las_1s5[~meas], yerr=yerr[~meas], fmt="D", color="k", mfc="white", ms=6, capsize=3,
                label="LAS at the condition (interpolated / extrapolated)")
    ax.set_yscale("log")
    ax.set_ylabel("Ar(1s$_5$) density [m$^{-3}$]")
    ax.legend(fontsize=7)
    psa._save(fig, ax, os.path.join(outdir, f"1s5_{o['tag']}.png"),
              f"Ar(1s$_5$): CR model (microwave EEDF, N$_e$ = n$_c$) and LAS\n{o['note']}",
              xlabel=o["xlabel"], legend=False)


def plot_parity(comp, outdir):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    cols = dict(zip(OES, ("C0", "C1", "C2")))
    for s, g in comp.groupby("sweep", sort=False):
        meas = g.las_1s5_how == "measured"
        for sel, mfc, lab in ((meas, cols[s], "measured LAS"), (~meas, "white", "interp. LAS")):
            h = g[sel]
            ax.errorbar(h.las_1s5, h.n_1s5, xerr=h.las_1s5 * h.las_1s5_rel_err,
                        yerr=[h.n_1s5 - h.n_1s5_lo, h.n_1s5_hi - h.n_1s5], fmt="s", color=cols[s], mfc=mfc, ms=6,
                        capsize=2, lw=0.8, label=f"{s} ({OES[s]['note']}), {lab}, LAS $T_g$")
        ax.plot(g.las_1s5, g.old_n_1s5, "o", color=cols[s], mfc="none", ms=5, alpha=0.5,
                label=f"{s}, earlier $T_g$ ({OES[s]['Tg_old']:g} K)")
    allv = np.r_[comp.las_1s5, comp.n_1s5, comp.old_n_1s5]
    allv = allv[np.isfinite(allv) & (allv > 0)]
    lim = np.array([allv.min() / 1.5, allv.max() * 1.5])
    ax.plot(lim, lim, "k-", lw=1, label="1 : 1")
    ax.fill_between(lim, lim / 2, lim * 2, color="0.5", alpha=0.1, lw=0, label="factor 2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_ylabel("CR model 1s$_5$ [m$^{-3}$] (microwave, N$_e$ = n$_c$)")
    ax.legend(fontsize=6.5, loc="lower right")
    psa._save(fig, ax, os.path.join(outdir, "1s5_parity.png"), "Ar(1s$_5$): CR model against LAS",
              xlabel="LAS 1s$_5$ [m$^{-3}$]", legend=False)


def plot_fit(comp, sweep, outdir):
    o = OES[sweep]
    c = comp[comp.sweep == sweep].sort_values("x")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(c.x, c.chi2_red_nc, "s-", color="C3", ms=6, label="LAS $T_g$")
    ax.plot(c.x, c.old_chi2_red_nc, "o--", color="0.35", mfc="white", ms=5, label=f"$T_g$ = {o['Tg_old']:g} K")
    ax.axhline(1, color="0.6", lw=0.8)
    ax.set_ylabel("$\\chi^2$/dof with N$_e$ = n$_c$")
    psa._save(fig, ax, os.path.join(outdir, f"chi2_{o['tag']}.png"),
              f"Fit quality at n$_c$ (microwave EEDF): {o['note']}", xlabel=o["xlabel"])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(c.x, c.Te_nc, yerr=psa._yerr(c.Te_nc, c.Te_nc_lo, c.Te_nc_hi), fmt="s-", color="C3", ms=6,
                capsize=2, label="LAS $T_g$")
    ax.errorbar(c.x, c.old_Te_nc, yerr=psa._yerr(c.old_Te_nc, c.old_Te_nc_lo, c.old_Te_nc_hi), fmt="o--",
                color="0.35", mfc="white", ms=5, capsize=2, label=f"$T_g$ = {o['Tg_old']:g} K")
    ax.set_ylabel(r"$T_{e,\mathrm{eff}} = \frac{2}{3}\langle\varepsilon\rangle$ at n$_c$ [eV]")
    psa._save(fig, ax, os.path.join(outdir, f"Te_eff_{o['tag']}.png"),
              f"Effective electron temperature at n$_c$ (microwave EEDF): {o['note']}", xlabel=o["xlabel"])


# ---- driver -------------------------------------------------------------------------
def main(plan_only=False):
    os.makedirs(FIGDIR, exist_ok=True)
    las, model, off, tgfit, cond = plan()
    print(f"LAS: {len(las)} rows ({LAS_FREQ_MHZ[0]}-{LAS_FREQ_MHZ[1]} MHz), {las.valid_Tg.sum()} with Tg in {TG_VALID} K "
          f"({model['n_outliers']} more dropped as outliers within their run)")
    print(f"Tg trend: {model['T0']:.0f} K at 1 Torr, 80 W, 0 % N2; {model['a_p']:+.0f} K/Torr, "
          f"{10 * model['a_P']:+.1f} K/10 W, {model['a_x']:+.1f} K/% N2 {model['a_xx']:+.2f} K/%^2;  scatter "
          f"within runs {model['sigma_within']:.0f} K, between runs {model['sigma_runs']:.0f} K")
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(off.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    pd.DataFrame([model]).to_csv(os.path.join(OUTDIR, "las_tg_trend.csv"), index=False)
    off.to_csv(os.path.join(OUTDIR, "las_tg_run_offsets.csv"), index=False)
    cond.to_csv(os.path.join(OUTDIR, "conditions.csv"), index=False)
    cols = ["sweep", "x", "p_Torr", "P_W", "N2", "Tg_trend", "Tg", "Tg_old", "N_gas", "las_1s5", "las_1s5_how"]
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(cond[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    plot_las_tg(model, off, tgfit, FIGDIR)
    for s in OES:
        plot_tg(cond, las, s, FIGDIR)
    libs, tabs = jobs(cond)
    print(f"\nto build: {len(libs)} BOLSIG+ libraries, {len(tabs)} CR tables")
    for k in libs + tabs:
        print("  ", k)
    if plan_only:
        return las, model, cond, None
    _run_workers("--lib", libs, "BOLSIG+ libraries")
    _run_workers("--build", tabs, "CR tables")
    comp = fit_all_conditions(cond)
    comp.to_csv(os.path.join(OUTDIR, "comparison.csv"), index=False)
    for s in OES:
        plot_1s5(comp, las, s, FIGDIR)
        plot_fit(comp, s, FIGDIR)
    plot_parity(comp, FIGDIR)
    cols = ["sweep", "x", "Tg", "EN_nc", "Te_nc", "chi2_red_nc", "dchi2_nc", "n_1s5", "las_1s5", "ratio_cr_las",
            "old_n_1s5", "old_ratio_cr_las", "old_chi2_red_nc"]
    with pd.option_context("display.width", 250, "display.max_rows", 100):
        print(comp[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nfigures in {FIGDIR}")
    return las, model, cond, comp


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--build":
        build_table(sys.argv[2])
    elif len(sys.argv) > 2 and sys.argv[1].startswith("--lib"):
        sys.exit(f"library workers run pipeline/LASAndEEDFLibraries.py, not this file ({sys.argv[1]})")
    else:
        LAS, MODEL, COND, COMP = main(plan_only="--plan" in sys.argv)
