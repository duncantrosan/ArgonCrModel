# -*- coding: utf-8 -*-
"""
PressureSweepAnalysis.py

CR-model fit of the pure-Ar pressure sweep (80 W, 600-1500 mTorr in 50 mTorr steps, 5 spectra per
pressure; Experimental_Data/EchelleData/PressureSweep_80W_PureArgon/SPAFiles - ref_* are plasma-off
backgrounds and are not used) for two EEDF families, everything at the gas temperature TG:

  microwave  BOLSIG+ (Biagi Ar), FREQ_HZ field.  omega/N = 2 pi f / N depends on N = p / k Tg, so
             one library per pressure: InputData/Bolsig/Ar_Biagi_bolsig_mw<f>GHz_<Tg>K_<p>mTorr
  dc         BOLSIG+ DC field.  The EEDF depends on E/N only (no e-e, no superelastics), so one
             library serves every pressure: InputData/Bolsig/Ar_Biagi_bolsig_dc_<Tg>K
CR model at each pressure with P = p and Tg = TG (ground density, metastable diffusion, trapping):
one table per pressure and family (CRFitNeTe.build_model_table, MicrowaveLowENFit.N_WORKERS at a
time).  Lines, nuisances and errors as CRFitNeTe (its 9 Ar I features, one scale per spectrum and the
linear response tilt, sigma_model, repeat scatter pooled over the sweep).

Per pressure and family
  free  (E/N, Ne) free: Te_eff and Ne (16-50-84 %), chi^2/dof, CR 1s5 (posterior mean)
  n_c   Ne pinned at the critical density of FREQ_HZ, E/N refitted (CRFitSpectrumOverlay.pinned_at_nc):
        Te_eff, chi^2/dof, Delta chi^2 / s^2 against the free fit, CR 1s5
Output in Experimental_Data/Output/PressureSweep/:
  Te_Ne_vs_pressure.png       Te_eff, E/N, Ne, chi^2/dof, Delta chi^2(n_c) and CR 1s5 against pressure
  residuals_vs_pressure.png   mean ln(measured / model) of every fitted line and of the trustworthy
                              check lines (not fitted) against pressure, per family and Ne mode
  pressure_sweep_fits.csv, line_residuals.csv, check_line_residuals.csv
  overlays/<family>_<free|nc>/  spectrum overlay and line residuals of every pressure
                                (CRFitSpectrumOverlay.plot_condition)
Run from Spyder (F5) or python SmallAnalysisScripts/PressureSweepAnalysis.py.  Libraries, CR tables
and line measurements are cached; delete one to rebuild it.
"""
import os
import re
import subprocess
import sys
import time
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import CRFitSpectrumOverlay as ov                   # noqa: E402

crf, mlf = ov.crf, ov.mlf
he = crf._helpers()


def _pressure_x(name):
    """600mTorr_3 -> (600.0, 3);  ref_1 (background) -> None"""
    m = re.match(r"(\d+)mTorr_(\d+)$", name)
    return (float(m.group(1)), int(m.group(2))) if m else None


# ---- settings -------------------------------------------------------------------
TG = 800.0                                          # K, gas temperature (libraries and CR model)
FREQ_HZ = 2.42e9                                    # microwave source
N_C = constants.epsilon_0 * constants.m_e * (2 * np.pi * FREQ_HZ) ** 2 / constants.e ** 2
FAMILIES = ("microwave", "dc")
NE_MODES = ("free", "nc")                           # overlays and residuals at these points
SWEEP = dict(name="Pressure",
             folder=os.path.join(crf.ROOT_DIR, "Experimental_Data", "EchelleData", "PressureSweep_80W_PureArgon",
                                 "SPAFiles"),
             parse=_pressure_x, xlabel="Pressure [mTorr]", note="pure Ar, 80 W", exclude_x=(), pure_ar=True)
EN_MW = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 35, 40, 45, 50, 55, 60,
         70, 80, 90, 100, 115, 130, 150, 170, 210, 250, 300, 350, 400, 500, 600, 700, 850, 1000]   # Td
EN_DC = [1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50, 75, 100, 150, 200]
PREC_MAX_TD = 150                                   # precision 1e-30 up to here (tails reach the
                                                    # thresholds at low E/N), 1e-25 above
BOLSIG_TIMEOUT = 1200                               # s per library; then retried at 1e-25
NE_GRID = crf.CONFIG["Ne_grid"]                     # 1e15-3e19 m^-3
OUTDIR = os.path.join(crf.ROOT_DIR, "Experimental_Data", "Output", "PressureSweep")
COLORS = {"microwave": "C3", "dc": "C0"}


def pressures():
    names = [os.path.splitext(os.path.basename(f))[0] for f in glob(os.path.join(SWEEP["folder"], "*.spa"))]
    return sorted({px[0] for px in map(_pressure_x, names) if px})


# ---- 1. libraries and CR tables -----------------------------------------------------
def lib_folder(family, p=None):
    if family == "dc":
        return he.BOLSIG_FOLDER / f"Ar_Biagi_bolsig_dc_{TG:g}K"
    return he.BOLSIG_FOLDER / f"Ar_Biagi_bolsig_mw{FREQ_HZ / 1e9:g}GHz_{TG:g}K_{p:g}mTorr"


def library(family, p=None):
    """BOLSIG+ library of the family (and pressure, microwave), run if missing."""
    folder = lib_folder(family, p)
    if he.IsBolsigLibrary(folder):
        return folder
    en = EN_DC if family == "dc" else EN_MW
    prec = [1e-30 if v <= PREC_MAX_TD else 1e-25 for v in en]
    kw = dict(mlf.BOLSIG_SETTINGS, Tg=TG, overwrite=True, verbose=False, timeout=BOLSIG_TIMEOUT)
    if family == "microwave":
        kw["omega_N"] = 2 * np.pi * FREQ_HZ / he.Torr2Volume(p / 1000, TG)
    xsec = he.BOLSIG_XSEC_FOLDER / "Biagi_Ar.txt"
    run = lambda pr: he.RunBolsig(xsec, folder.name, en, species=["Ar"], fractions=[1.0], **dict(kw, precision=pr))
    try:
        run(prec)
    except (subprocess.TimeoutExpired, RuntimeError) as err:
        print(f"{folder.name}: {type(err).__name__} at precision 1e-30 - retrying at 1e-25", flush=True)
        run(1e-25)
        return folder
    lib = he.ImportBolsigLibrary(folder, verbose=False)
    bad = [i for i, e in enumerate(lib["EEDFs"]) if e["E"][e["EEPF"] > 0].max() >= 1000]
    if bad:                                           # 1e-30 can give runaway tails to 9990 eV
        print(f"{folder.name}: runaway tails at {[en[i] for i in bad]} Td - those rows at 1e-25", flush=True)
        run([1e-25 if i in bad else v for i, v in enumerate(prec)])
    return folder


def base_cfg():
    return dict(crf.CONFIG, sweeps=[SWEEP], Tg=TG, Ne_grid=NE_GRID, N2_percent=0.0, outdir=OUTDIR,
                measure_dir=os.path.join(OUTDIR, "measurements"))


def fit_cfg(family, p):
    return dict(base_cfg(), eedf=str(lib_folder(family, p)), P_Torr=p / 1000,
                outdir=os.path.join(OUTDIR, f"fit_{family}", f"{p:g}mTorr"))


def _run_workers(flag, keys, what):
    """This script with flag for each key, mlf.N_WORKERS processes at a time (one BLAS thread each)."""
    if not keys:
        return
    logs = os.path.join(OUTDIR, "logs")
    os.makedirs(logs, exist_ok=True)
    print(f"{what}: {len(keys)} jobs ({mlf.N_WORKERS} at a time, logs in {logs}) ...", flush=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0, procs = time.time(), {}
    for k in keys:
        while sum(q.poll() is None for q in procs.values()) >= mlf.N_WORKERS:
            time.sleep(5)
        log = open(os.path.join(logs, f"{flag.strip('-')}_{k.replace(':', '_')}.log"), "w")
        procs[k] = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), flag, k],
                                    stdout=log, stderr=subprocess.STDOUT, cwd=crf.ROOT_DIR, env=env)
    failed = [k for k, q in procs.items() if q.wait() != 0]
    print(f"{what} done ({time.time() - t0:.0f} s)", flush=True)
    if failed:
        raise RuntimeError(f"{what} failed for {failed} - see {logs}")


def _split(key):
    family, p = key.split(":")
    return family, (float(p) if p else None)


def build_all(ps):
    libs = [f"dc:"] + [f"microwave:{p:g}" for p in ps]
    _run_workers("--lib", [k for k in libs if not he.IsBolsigLibrary(lib_folder(*_split(k)))], "BOLSIG+ libraries")
    tabs = [f"{fam}:{p:g}" for fam in FAMILIES for p in ps]
    _run_workers("--build", [k for k in tabs if not os.path.exists(crf._table_path(fit_cfg(*_split(k))))],
                 "CR tables")


# ---- 2. fits ----------------------------------------------------------------------
def measurements(ps):
    """Features, measured feature table (rel_err incl. the pooled repeat scatter) of the sweep."""
    cfg = fit_cfg("dc", ps[0])
    tab = crf.build_model_table(cfg)
    feats, comps = crf.select_features(tab, cfg)
    ft = crf.feature_table(crf.measure(comps, cfg), feats, cfg)
    rs = crf.repeat_scatter(ft)
    feats["repeat_scatter"] = feats.feature.map(rs)
    ft["rel_err_fit"] = ft.rel_err
    if cfg["use_repeat_scatter"]:
        ft["rel_err"] = np.sqrt(ft.rel_err ** 2 + ft.feature.map(rs) ** 2)
    return feats, ft


def fit_condition(family, p, feats, ft):
    """Free and n_c-pinned fit at one pressure -> (FIT dict for the overlays, summary row, pinned cond row)."""
    cfg = fit_cfg(family, p)
    tab = crf.build_model_table(cfg)
    grid = crf.fine_grid(tab, cfg)
    Te_f = crf.te_eff_fine(tab, grid)
    M = crf.feature_model(tab, feats, cfg)
    cond, _, _, posts = crf.fit_all(ft[ft.x == p], feats, M, grid, Te_f, cfg, verbose=False)
    c = cond.iloc[0]
    post = posts[f"Pressure|{p:g}"]
    FIT = dict(tab=tab, feats=feats, ft=ft, grid=grid, Te_f=Te_f, posts=posts, cond=cond, cfg=cfg)
    lev = list(tab["levels"])
    ln1s5 = crf.on_fine_grid(tab, np.log(np.clip(tab["dens"][:, :, lev.index("4s1")], 1e-300, None)), cfg, k=1)
    fX, fN = grid
    j = int(np.argmin(np.abs(fN - np.log(N_C))))
    chi2 = mlf.cdf.chi2_grid(post, c.chi2_min, c.birge)
    i = int(np.argmin(chi2[:, j]))
    pin = ov.pinned_at_nc(FIT, c)
    row = dict(family=family, p_mTorr=p, n_spec=c.n_spec, Te_best=c.Te_best, Te_lo=c.Te_lo, Te_med=c.Te_med,
               Te_hi=c.Te_hi, EN_best=c.x_best, EN_lo=c.x_lo, EN_hi=c.x_hi, Ne_best=c.Ne_best, Ne_lo=c.Ne_lo,
               Ne_med=c.Ne_med, Ne_hi=c.Ne_hi, chi2_red=c.chi2_red, birge=c.birge, edge_EN=c.edge_x,
               edge_Ne=c.edge_Ne, response_slope=c.get("response_slope_per_100nm", np.nan),
               n_1s5=float(np.exp((post * ln1s5).sum())),
               EN_nc=pin.x_best, Te_nc=pin.Te_best, Te_nc_lo=pin.Te_lo, Te_nc_hi=pin.Te_hi, chi2_red_nc=pin.chi2_red,
               dchi2_nc=(chi2[i, j] - c.chi2_min) / c.birge ** 2, n_1s5_nc=float(np.exp(ln1s5[i, j])))
    return FIT, row, pin


def premeasure_check_lines(fits, feats, ft):
    """Measure the union of the trustworthy check lines of every overlay once, so that
    CRFitSpectrumOverlay.confident_check_residuals reuses one cache for all pressures."""
    keep = set()
    for (family, p, mode), (FIT, c) in fits.items():
        cfg = FIT["cfg"]
        d = ft[ft.x == p]
        wl_m, I_m = ov.model_lines_at(FIT["tab"], c.x_best, c.Ne_best, cfg)
        scales, coef = ov.nuisances(d, feats, wl_m, I_m, cfg)
        I_syn = np.exp(np.mean(list(scales.values()))) * I_m * ov.response(wl_m, coef)
        cand = ov.check_line_candidates(wl_m, I_syn, feats.wl.to_numpy(), SWEEP["name"], p, cfg)
        if len(cand):
            keep |= set(cand[cand.keep].wl_air.round(4))
    if not keep:
        return
    cfg = base_cfg()
    mcfg = dict(crf.als.CONFIG, sweeps=[SWEEP], window_nm=cfg["window_nm"], reuse_measurements=cfg["reuse_measurements"],
                outdir=cfg["measure_dir"], measure_cache="check_line_measurements.csv", export_flags=cfg["export_flags"])
    crf.als.measure_sweeps(pd.DataFrame(dict(species="Ar I", wl_air=sorted(keep))), mcfg)


# ---- 3. plots ---------------------------------------------------------------------
def _yerr(best, lo, hi):
    """Error bars from the 16-84 % interval; the best grid point can lie outside it (bar 0 there)."""
    return [np.clip(best - lo, 0, None), np.clip(hi - best, 0, None)]


def plot_vs_pressure(res, path):
    fig, axs = plt.subplots(3, 2, figsize=(14, 13), sharex=True)
    (aT, aE), (aN, aC), (aD, aS) = axs
    for fam in FAMILIES:
        g = res[res.family == fam].sort_values("p_mTorr")
        if g.empty:
            continue
        c, p = COLORS[fam], g.p_mTorr
        aT.errorbar(p, g.Te_best, yerr=_yerr(g.Te_best, g.Te_lo, g.Te_hi), fmt="o-", color=c, ms=5,
                    capsize=2, label=f"{fam}, Ne free")
        aT.errorbar(p, g.Te_nc, yerr=_yerr(g.Te_nc, g.Te_nc_lo, g.Te_nc_hi), fmt="s--", color=c, mfc="white",
                    ms=5, capsize=2, label=f"{fam}, Ne = n$_c$")
        aE.plot(p, g.EN_best, "o-", color=c, ms=5, label=f"{fam}, Ne free")
        aE.plot(p, g.EN_nc, "s--", color=c, mfc="white", ms=5, label=f"{fam}, Ne = n$_c$")
        aN.errorbar(p, g.Ne_best, yerr=_yerr(g.Ne_best, g.Ne_lo, g.Ne_hi),
                    fmt="o-", color=c, ms=5, capsize=2, label=f"{fam} (best, 16-84 %)")
        edge = g[g.edge_Ne.astype(bool)]
        aN.plot(edge.p_mTorr, edge.Ne_best, "x", color=c, ms=10, mew=2)
        aC.plot(p, g.chi2_red, "o-", color=c, ms=5, label=f"{fam}, Ne free")
        aC.plot(p, g.chi2_red_nc, "s--", color=c, mfc="white", ms=5, label=f"{fam}, Ne = n$_c$")
        aD.plot(p, g.dchi2_nc, "o-", color=c, ms=5, label=fam)
        aS.plot(p, g.n_1s5, "o-", color=c, ms=5, label=f"{fam}, Ne free")
        aS.plot(p, g.n_1s5_nc, "s--", color=c, mfc="white", ms=5, label=f"{fam}, Ne = n$_c$")
    aN.axhline(N_C, color="k", lw=1, ls=":", label=f"n$_c$ = {N_C:.2e} m$^{{-3}}$ ({FREQ_HZ / 1e9:g} GHz)")
    aN.plot([], [], "kx", ms=9, mew=2, label="posterior at the Ne grid edge")
    aC.axhline(1, color="0.6", lw=0.8)
    for v in (1, 4):
        aD.axhline(v, color="0.6", lw=0.8, ls=":")
    aT.set_ylabel(r"$T_{e,\mathrm{eff}} = \frac{2}{3}\langle\varepsilon\rangle$ [eV]")
    aE.set_ylabel("E/N [Td]")
    aN.set_ylabel("$N_e$ [m$^{-3}$]")
    aC.set_ylabel(r"$\chi^2$/dof")
    aD.set_ylabel(r"$\Delta\chi^2/s^2$ for $N_e = n_c$")
    aS.set_ylabel("CR 1s$_5$ density [m$^{-3}$]")
    for ax in (aE, aN, aS):
        ax.set_yscale("log")
    for ax in axs.ravel():
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel(SWEEP["xlabel"])
    fig.suptitle(f"Pure-Ar pressure sweep (80 W): CR fit, BOLSIG+ EEDFs and CR model at $T_g$ = {TG:g} K",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_residuals(lines, checks, sigma_model, path):
    combos = [(f, m) for f in FAMILIES for m in NE_MODES]
    fig, axs = plt.subplots(len(combos), 1, figsize=(12, 3.6 * len(combos)), sharex=True, squeeze=False)
    feats = list(dict.fromkeys(lines.feature))
    cols = dict(zip(feats, plt.cm.tab10(np.linspace(0, 1, 10))))
    for ax, (fam, mode) in zip(axs[:, 0], combos):
        ax.axhspan(-sigma_model, sigma_model, color="C3", alpha=0.08, lw=0)
        ax.axhline(0, color="0.3", lw=1)
        g = lines[(lines.family == fam) & (lines.ne_mode == mode)]
        for f in feats:
            h = g[g.feature == f].sort_values("p_mTorr")
            ax.errorbar(h.p_mTorr, h.resid, yerr=h.resid_sd, fmt="o-", ms=4, lw=1.2, capsize=2, color=cols[f],
                        label=f)
        h = checks[(checks.family == fam) & (checks.ne_mode == mode)] if len(checks) else checks
        for (w, up), k in (h.groupby(["wl", "upper"]) if len(h) else []):
            k = k.sort_values("p_mTorr")
            ax.plot(k.p_mTorr, k.resid, "x:", color="0.45", ms=5, lw=0.9)
            ax.annotate(f"{w:.2f} {up}", (k.p_mTorr.iloc[-1], k.resid.iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=7, color="0.4", va="center")
        ax.set_ylabel("ln(measured / model)")
        ax.set_title(f"{fam} EEDF, Ne {'free' if mode == 'free' else 'pinned at n$_c$'}", fontsize=10)
        ax.grid(alpha=0.3)
    axs[0, 0].plot([], [], "x:", color="0.45", label="check lines (not fitted)")
    axs[0, 0].legend(fontsize=7, ncol=3, loc="best")
    axs[-1, 0].set_xlabel(SWEEP["xlabel"])
    fig.suptitle(f"Line residuals against pressure (mean of the repeats; band: $\\pm\\sigma_{{model}}$ = "
                 f"{100 * sigma_model:.0f} %)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---- driver -------------------------------------------------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ps = pressures()
    print(f"pressures [mTorr]: {[f'{p:g}' for p in ps]};  Tg = {TG:g} K, f = {FREQ_HZ / 1e9:g} GHz, "
          f"n_c = {N_C:.3e} m^-3")
    build_all(ps)
    feats, ft = measurements(ps)
    print(f"\n{len(feats)} features: {', '.join(feats.feature)}")
    rows, fits = [], {}
    for fam in FAMILIES:
        for p in ps:
            FIT, row, pin = fit_condition(fam, p, feats, ft)
            rows.append(row)
            fits[(fam, p, "free")] = (FIT, FIT["cond"].iloc[0])
            fits[(fam, p, "nc")] = (FIT, pin)
            print(f"  {fam:<9s} {p:5.0f} mTorr  Te_eff = {row['Te_best']:.2f} eV  Ne = {row['Ne_best']:.2e}  "
                  f"chi2/dof = {row['chi2_red']:.2f}   |  n_c: Te_eff = {row['Te_nc']:.2f}  chi2/dof = "
                  f"{row['chi2_red_nc']:.2f}  dchi2 = {row['dchi2_nc']:.1f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUTDIR, "pressure_sweep_fits.csv"), index=False)
    plot_vs_pressure(res, os.path.join(OUTDIR, "Te_Ne_vs_pressure.png"))

    print("\nspectrum overlays and line residuals ...", flush=True)
    premeasure_check_lines(fits, feats, ft)
    lines, checks = [], []
    for (fam, p, mode), (FIT, c) in fits.items():
        label = f"Pure Ar {p:g} mTorr, {fam} EEDF ({TG:g} K)"
        out = ov.plot_condition(FIT, c, SWEEP["name"], p, FIT["cfg"], label, f"{p:g}mTorr_{fam}_Ne{mode}",
                                outdir=os.path.join(OUTDIR, "overlays", f"{fam}_{mode}"), pinned=mode == "nc",
                                show=False, verbose=False)
        g = out["res_fit"].groupby("feature", sort=False).resid
        lines.append(pd.DataFrame(dict(family=fam, ne_mode=mode, p_mTorr=p, resid=g.mean(), resid_sd=g.std(ddof=1)))
                     .rename_axis("feature").reset_index())
        if not out["res_chk"].empty:
            h = out["res_chk"].groupby(["wl", "upper"]).resid.mean().reset_index()
            checks.append(h.assign(family=fam, ne_mode=mode, p_mTorr=p))
    lines = pd.concat(lines, ignore_index=True)
    checks = pd.concat(checks, ignore_index=True) if checks else pd.DataFrame(columns=["wl", "upper", "resid"])
    lines.to_csv(os.path.join(OUTDIR, "line_residuals.csv"), index=False)
    checks.to_csv(os.path.join(OUTDIR, "check_line_residuals.csv"), index=False)
    plot_residuals(lines, checks, crf.CONFIG["sigma_model"], os.path.join(OUTDIR, "residuals_vs_pressure.png"))

    cols = ["family", "p_mTorr", "Te_best", "Te_lo", "Te_hi", "EN_best", "Ne_best", "Ne_lo", "Ne_hi", "chi2_red",
            "edge_Ne", "n_1s5", "EN_nc", "Te_nc", "chi2_red_nc", "dchi2_nc", "n_1s5_nc"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 100):
        print(res[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nfigures in {OUTDIR}")
    return res, lines, checks


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--lib":
        library(*_split(sys.argv[2]))
    elif len(sys.argv) > 2 and sys.argv[1] == "--build":
        tab = crf.build_model_table(fit_cfg(*_split(sys.argv[2])))
        print(f"{sys.argv[2]}: {len(tab['x_grid'])} E/N rows, {(~tab['converged']).sum()} not converged")
    else:
        RES, LINES, CHECKS = main()
