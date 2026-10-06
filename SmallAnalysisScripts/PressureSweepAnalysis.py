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
  free  (E/N, Ne) free: Te_eff and Ne (16-50-84 %), chi^2/dof, CR 1s5 / 1s3 metastables (posterior mean)
  n_c   Ne pinned at the critical density of FREQ_HZ, E/N refitted (CRFitSpectrumOverlay.pinned_at_nc):
        Te_eff, chi^2/dof, Delta chi^2 / s^2 against the free fit, CR 1s5 / 1s3 metastables (best E/N and
        the range with Delta chi^2 <= 1 s^2 along E/N)
Output in Experimental_Data/Output/PressureSweep/ (one graph per file):
  figures/<quantity>_vs_pressure.png   Te_eff, EN, Ne, chi2_dof, dchi2_nc; the four 1s densities on one
                                       graph (microwave, Ne = n_c only)
                                       against pressure, both families, Ne free and Ne = n_c
  figures/chi2_vs_Ne_<dchi2|chi2_red|Te_eff>_<family>.png
                              chi^2 profiled over E/N at each Ne (Delta chi^2 / s^2 and chi^2/dof with Ne
                              fixed), with Te_eff along the profile, one curve per pressure; local minima
                              marked (chi2_vs_Ne.csv, chi2_local_minima.csv)
  figures/residuals_<family>_Ne<free|nc>.png
                              mean ln(measured / model) of every fitted line and of the trustworthy
                              check lines (not fitted) against pressure
  microwave_nc/               the microwave EEDF with Ne = n_c on its own: Te_eff, E/N, chi^2/dof,
                              1s densities (metastable 1s5, 1s3; resonant 1s4, 1s2) and line residuals
                              against pressure; 1s_densities_microwave_nc.csv (pressure vs 1s densities, Excel)
  pressure_sweep_fits.csv, line_residuals.csv, check_line_residuals.csv
  overlays/<family>_<free|nc>/  spectrum overlay and line residuals of every pressure
                                (CRFitSpectrumOverlay.plot_condition)
Run from Spyder (F5) or python SmallAnalysisScripts/PressureSweepAnalysis.py (--profiles: only the
chi^2-vs-Ne figures, from the cached tables; --fits: no overlays or line residuals).  Libraries, CR tables
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
ESCAPE_MODE = os.environ.get("CR_ESCAPE_MODE", "table")   # 'walsh': Holstein-Walsh escape factors (systematic)
TRAP_LINES = os.environ.get("CR_TRAP_LINES", "all")      # 'ground': lines to excited levels optically thin
TRAP_REF = os.environ.get("CR_TRAP_REF", "")             # 'nc': escape factors frozen at the Ne = n_c solution
OUTDIR = os.path.join(crf.ROOT_DIR, "Experimental_Data", "Output",
                      "PressureSweep" + ("" if ESCAPE_MODE == "table" else f"_{ESCAPE_MODE}")
                      + ("" if TRAP_LINES == "all" else f"_trap{TRAP_LINES}")
                      + (f"_trapfrozen{TRAP_REF}" if TRAP_REF else ""))
MEASURE_DIR = os.path.join(crf.ROOT_DIR, "Experimental_Data", "Output", "PressureSweep", "measurements")
FIGDIR = os.path.join(OUTDIR, "figures")              # one file per graph
MW_NC_DIR = os.path.join(OUTDIR, "microwave_nc")      # microwave EEDF with Ne = n_c on its own
LEVELS_1S = {"1s5": "4s1", "1s4": "4s2", "1s3": "4s3", "1s2": "4s4"}   # Paschen -> CR-model label
KIND_1S = {"1s5": "metastable", "1s4": "resonant", "1s3": "metastable", "1s2": "resonant"}
COLORS = {"microwave": "C3", "dc": "C0"}
FIGSIZE = (8, 5.5)


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
                escape_mode=ESCAPE_MODE, trap_lines=TRAP_LINES,
                trap_ref_Ne={"": None, "nc": N_C}[TRAP_REF],
                measure_dir=MEASURE_DIR)          # the line areas do not depend on the model


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
    _run_workers("--build", [k for k in tabs if not crf.table_is_current(fit_cfg(*_split(k)))],
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
    dens = {m: tab["dens"][:, :, lev.index(lbl)] for m, lbl in LEVELS_1S.items()}
    dens["meta"] = dens["1s5"] + dens["1s3"]
    ln_n = {m: crf.on_fine_grid(tab, np.log(np.clip(d, 1e-300, None)), cfg, k=1) for m, d in dens.items()}
    fX, fN = grid
    j = int(np.argmin(np.abs(fN - np.log(N_C))))
    chi2 = mlf.cdf.chi2_grid(post, c.chi2_min, c.birge)
    i = int(np.argmin(chi2[:, j]))
    ok = chi2[:, j] <= chi2[i, j] + c.birge ** 2       # Delta chi^2 = 1 s^2 along E/N at n_c (as pinned_at_nc)
    pin = ov.pinned_at_nc(FIT, c)
    k = np.argmin(chi2, axis=0)                       # best E/N at each Ne
    T2 = np.broadcast_to(Te_f[:, None], chi2.shape) if np.ndim(Te_f) == 1 else Te_f
    prof = pd.DataFrame(dict(family=family, p_mTorr=p, Ne=np.exp(fN),
                             dchi2=(chi2[k, np.arange(len(fN))] - c.chi2_min) / c.birge ** 2,
                             chi2_red=chi2[k, np.arange(len(fN))] / (c.dof + 1), EN=np.exp(fX[k]),
                             Te_eff=T2[k, np.arange(len(fN))]))
    row = dict(family=family, p_mTorr=p, n_spec=c.n_spec, Te_best=c.Te_best, Te_lo=c.Te_lo, Te_med=c.Te_med,
               Te_hi=c.Te_hi, EN_best=c.x_best, EN_lo=c.x_lo, EN_hi=c.x_hi, Ne_best=c.Ne_best, Ne_lo=c.Ne_lo,
               Ne_med=c.Ne_med, Ne_hi=c.Ne_hi, chi2_red=c.chi2_red, birge=c.birge, edge_EN=c.edge_x,
               edge_Ne=c.edge_Ne, response_slope=c.get("response_slope_per_100nm", np.nan),
               EN_nc=pin.x_best, Te_nc=pin.Te_best, Te_nc_lo=pin.Te_lo, Te_nc_hi=pin.Te_hi, chi2_red_nc=pin.chi2_red,
               dchi2_nc=(chi2[i, j] - c.chi2_min) / c.birge ** 2)
    for m, ln in ln_n.items():                         # free: posterior mean; n_c: best E/N and its range
        n_j = np.exp(ln[:, j])
        row.update({f"n_{m}": float(np.exp((post * ln).sum())), f"n_{m}_nc": float(n_j[i]),
                    f"n_{m}_nc_lo": float(n_j[ok].min()), f"n_{m}_nc_hi": float(n_j[ok].max())})
    return FIT, row, pin, prof


def local_minima(prof, min_depth=1.0):
    """Interior local minima of one Delta chi^2 / s^2 profile: points below both neighbours whose barrier
    on each side (highest point between them and the next lower point, or the grid edge) is >= min_depth.
    Grid-edge points are never minima (the curve just runs off the grid there)."""
    y = prof.dchi2.to_numpy()
    out = []
    for i in range(1, len(y) - 1):
        if y[i - 1] < y[i] or y[i + 1] < y[i]:
            continue
        lower_l = np.flatnonzero(y[:i] < y[i])
        lower_r = np.flatnonzero(y[i + 1:] < y[i]) + i + 1
        left = y[(lower_l[-1] if len(lower_l) else 0):i].max() - y[i]
        right = y[i + 1:(lower_r[0] + 1 if len(lower_r) else len(y))].max() - y[i]
        if min(left, right) >= min_depth:
            out.append(prof.iloc[i])
    return pd.DataFrame(out, columns=prof.columns)


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


def _save(fig, ax, path, title, xlabel=SWEEP["xlabel"], legend=True):
    """Finish one single-graph figure and write it."""
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3, which="both")
    if legend:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_vs_pressure(res, outdir):
    """Te_eff, E/N, Ne, chi^2/dof, Delta chi^2(n_c) and the CR 1s5 and 1s3 metastable densities against
    pressure, both EEDF families, Ne free and Ne = n_c: one file per graph in outdir."""
    os.makedirs(outdir, exist_ok=True)
    names = ["Te_eff", "EN", "Ne", "chi2_dof", "dchi2_nc"]
    F = {n: plt.subplots(figsize=FIGSIZE) for n in names}
    aT, aE, aN, aC, aD = (F[n][1] for n in names)
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
    aN.axhline(N_C, color="k", lw=1, ls=":", label=f"n$_c$ = {N_C:.2e} m$^{{-3}}$ ({FREQ_HZ / 1e9:g} GHz)")
    aN.plot([], [], "kx", ms=9, mew=2, label="posterior at the Ne grid edge")
    aC.axhline(1, color="0.6", lw=0.8)
    for v in (1, 4):
        aD.axhline(v, color="0.6", lw=0.8, ls=":")
    labels = {"Te_eff": (r"$T_{e,\mathrm{eff}} = \frac{2}{3}\langle\varepsilon\rangle$ [eV]", "Effective electron temperature"),
              "EN": ("E/N [Td]", "Reduced field of the fitted EEDF"),
              "Ne": ("$N_e$ [m$^{-3}$]", "Electron density (Ne free)"),
              "chi2_dof": (r"$\chi^2$/dof", "Fit quality"),
              "dchi2_nc": (r"$\Delta\chi^2/s^2$ for $N_e = n_c$", "Cost of pinning Ne at the critical density")}
    for n in names:
        fig, ax = F[n]
        ylab, title = labels[n]
        ax.set_ylabel(ylab)
        if n in ("EN", "Ne"):
            ax.set_yscale("log")
        _save(fig, ax, os.path.join(outdir, f"{n}_vs_pressure.png"),
              f"{title}: pure Ar, 80 W, $T_g$ = {TG:g} K")
    plot_1s_densities(res, os.path.join(outdir, "1s_densities_vs_pressure_microwave_nc.png"))


def plot_1s_densities(res, path):
    """CR densities of the four 1s levels (metastable 1s5, 1s3; resonant 1s4, 1s2) against pressure on
    one graph, microwave EEDF with Ne = n_c (band: Delta chi^2 <= 1 s^2 along E/N)."""
    g = res[res.family == "microwave"].sort_values("p_mTorr")
    p = g.p_mTorr
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for m, mk, col in (("1s5", "s-", "C3"), ("1s4", "D--", "C0"), ("1s3", "^-", "C1"), ("1s2", "v--", "C2")):
        ax.fill_between(p, g[f"n_{m}_nc_lo"], g[f"n_{m}_nc_hi"], color=col, alpha=0.15, lw=0)
        ax.plot(p, g[f"n_{m}_nc"], mk, color=col, ms=5, label=f"Ar({m[:2]}$_{m[2]}$), {KIND_1S[m]}")
    ax.set_yscale("log")
    ax.set_ylabel("CR density [m$^{-3}$]")
    _save(fig, ax, path, "Ar(1s) metastable and resonant densities (CR model; band: $\\Delta\\chi^2 \\leq 1\\,s^2$ "
          f"along E/N)\nmicrowave EEDF, Ne = n$_c$ = {N_C:.2e} m$^{{-3}}$, pure Ar, $T_g$ = {TG:g} K")


def plot_residuals(lines, checks, sigma_model, outdir, combos=None):
    """Mean ln(measured / model) of the fitted lines (and the check lines) against pressure: one file
    per (EEDF family, Ne mode), residuals_<family>_Ne<mode>.png."""
    os.makedirs(outdir, exist_ok=True)
    combos = combos or [(f, m) for f in FAMILIES for m in NE_MODES]
    feats = list(dict.fromkeys(lines.feature))
    cols = dict(zip(feats, plt.cm.tab10(np.linspace(0, 1, 10))))
    for fam, mode in combos:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.axhspan(-sigma_model, sigma_model, color="C3", alpha=0.08, lw=0,
                   label=f"$\\pm\\sigma_{{model}}$ = {100 * sigma_model:.0f} %")
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
        if len(h):
            ax.plot([], [], "x:", color="0.45", label="check lines (not fitted)")
        ax.set_ylabel("ln(measured / model)")
        ax.legend(fontsize=7, ncol=3, loc="best")
        _save(fig, ax, os.path.join(outdir, f"residuals_{fam}_Ne{mode}.png"),
              f"Line residuals (mean of the repeats): {fam} EEDF, Ne "
              f"{'free' if mode == 'free' else 'pinned at n$_c$'}", legend=False)


def save_profiles(profs):
    profs.to_csv(os.path.join(OUTDIR, "chi2_vs_Ne.csv"), index=False)
    mins = pd.concat([local_minima(g) for _, g in profs.groupby(["family", "p_mTorr"])], ignore_index=True)
    mins.to_csv(os.path.join(OUTDIR, "chi2_local_minima.csv"), index=False)
    plot_chi2_vs_ne(profs, mins, FIGDIR)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(f"\ninterior local minima of chi^2(Ne) (E/N profiled; depth >= 1 s^2): {len(mins)}")
        if len(mins):
            print(mins[["family", "p_mTorr", "Ne", "dchi2", "chi2_red", "EN", "Te_eff"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return mins


def plot_chi2_vs_ne(profs, mins, outdir):
    """chi^2 profiled over E/N against Ne, one curve per pressure: one file per quantity and EEDF family,
    chi2_vs_Ne_<dchi2|chi2_red|Te_eff>_<family>.png."""
    os.makedirs(outdir, exist_ok=True)
    ps = sorted(profs.p_mTorr.unique())
    cmap = plt.cm.viridis
    col = {p: cmap(i / max(len(ps) - 1, 1)) for i, p in enumerate(ps)}
    keys = [("dchi2", r"$\Delta\chi^2/s^2$ (E/N profiled)", "log", r"$\chi^2$ against $N_e$"),
            ("chi2_red", r"$\chi^2$/dof, Ne fixed", "linear", r"$\chi^2$/dof against $N_e$"),
            ("Te_eff", r"$T_{e,\mathrm{eff}}$ along the profile [eV]", "linear", r"$T_{e,\mathrm{eff}}$ along the $\chi^2$ profile")]
    off = lambda k: 0.1 if k == "dchi2" else 0          # +0.1 so the best point shows on the log axis
    for fam in FAMILIES:
        for k, ylab, scale, title in keys:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            for p in ps:
                g = profs[(profs.family == fam) & (profs.p_mTorr == p)]
                m = mins[(mins.family == fam) & (mins.p_mTorr == p)] if len(mins) else mins
                ax.plot(g.Ne, g[k] + off(k), color=col[p], lw=1.2)
                if len(m):
                    ax.plot(m.Ne, m[k] + off(k), "o", color=col[p], ms=6, mec="k", mew=0.6)
                b = g.loc[g.dchi2.idxmin()]
                ax.plot([b.Ne], [b[k] + off(k)], "*", color=col[p], ms=9, mec="k", mew=0.5)
            ax.axvline(N_C, color="k", ls=":", lw=1.2, label=f"n$_c$ = {N_C:.2e} m$^{{-3}}$")
            if k == "dchi2":
                for v in (1, 4):
                    ax.axhline(v + 0.1, color="0.6", lw=0.8, ls="--")
            elif k == "chi2_red":
                ax.axhline(1, color="0.6", lw=0.8)
            ax.plot([], [], "k*", ms=9, label="best fit")
            if len(mins):
                ax.plot([], [], "ko", ms=6, label="interior local minimum (depth >= 1 s$^2$)")
            ax.set_xscale("log")
            ax.set_yscale(scale)
            ax.set_ylabel(ylab)
            fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(ps), max(ps))), ax=ax,
                         label=SWEEP["xlabel"])
            _save(fig, ax, os.path.join(outdir, f"chi2_vs_Ne_{k}_{fam}.png"),
                  f"{title}: {fam} EEDF, pure Ar, $T_g$ = {TG:g} K", xlabel="$N_e$ [m$^{-3}$]")


def plot_microwave_nc(res, lines, checks, outdir):
    """The microwave EEDF with Ne pinned at n_c on its own: Te_eff, E/N, chi^2/dof and the CR metastable
    densities (1s5, 1s3, sum; bars/bands = Delta chi^2 <= 1 s^2 along E/N) against pressure, the line
    residuals, and metastable_density_microwave_nc.csv (pressure vs metastable density, for Excel)."""
    os.makedirs(outdir, exist_ok=True)
    g = res[res.family == "microwave"].sort_values("p_mTorr")
    p, c = g.p_mTorr, COLORS["microwave"]
    tag = f"microwave EEDF, Ne = n$_c$ = {N_C:.2e} m$^{{-3}}$, pure Ar, $T_g$ = {TG:g} K"
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(p, g.Te_nc, yerr=_yerr(g.Te_nc, g.Te_nc_lo, g.Te_nc_hi), fmt="s-", color=c, ms=5, capsize=2)
    ax.set_ylabel(r"$T_{e,\mathrm{eff}} = \frac{2}{3}\langle\varepsilon\rangle$ [eV]")
    _save(fig, ax, os.path.join(outdir, "Te_eff_vs_pressure_microwave_nc.png"), f"Effective electron temperature\n{tag}",
          legend=False)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(p, g.EN_nc, "s-", color=c, ms=5)
    ax.set_ylabel("E/N [Td]")
    _save(fig, ax, os.path.join(outdir, "EN_vs_pressure_microwave_nc.png"), f"Reduced field of the fitted EEDF\n{tag}",
          legend=False)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(p, g.chi2_red_nc, "s-", color=c, ms=5)
    ax.axhline(1, color="0.6", lw=0.8)
    ax.set_ylabel(r"$\chi^2$/dof")
    _save(fig, ax, os.path.join(outdir, "chi2_dof_vs_pressure_microwave_nc.png"), f"Fit quality\n{tag}", legend=False)
    plot_1s_densities(res, os.path.join(outdir, "1s_densities_vs_pressure_microwave_nc.png"))
    if lines is not None:
        plot_residuals(lines, checks, crf.CONFIG["sigma_model"], outdir, combos=[("microwave", "nc")])
    pd.DataFrame({"Pressure (mTorr)": p.round().astype(int).to_numpy(),
                  **{f"Ar({m}) {KIND_1S[m]} (m^-3)": g[f"n_{m}_nc"].to_numpy() for m in LEVELS_1S},
                  "Total metastable 1s5+1s3 (m^-3)": g.n_meta_nc.to_numpy()}).to_csv(
        os.path.join(outdir, "1s_densities_microwave_nc.csv"), index=False, float_format="%.3e")


def profiles_only():
    """chi^2-vs-Ne figure from the cached tables and measurements (no overlays)."""
    ps = pressures()
    feats, ft = measurements(ps)
    return save_profiles(pd.concat([fit_condition(f, p, feats, ft)[3] for f in FAMILIES for p in ps],
                                   ignore_index=True))


# ---- driver -------------------------------------------------------------------------
def main(overlays=True):
    os.makedirs(OUTDIR, exist_ok=True)
    ps = pressures()
    print(f"pressures [mTorr]: {[f'{p:g}' for p in ps]};  Tg = {TG:g} K, f = {FREQ_HZ / 1e9:g} GHz, "
          f"n_c = {N_C:.3e} m^-3")
    build_all(ps)
    feats, ft = measurements(ps)
    print(f"\n{len(feats)} features: {', '.join(feats.feature)}")
    rows, fits, profs = [], {}, []
    for fam in FAMILIES:
        for p in ps:
            FIT, row, pin, prof = fit_condition(fam, p, feats, ft)
            rows.append(row)
            profs.append(prof)
            fits[(fam, p, "free")] = (FIT, FIT["cond"].iloc[0])
            fits[(fam, p, "nc")] = (FIT, pin)
            print(f"  {fam:<9s} {p:5.0f} mTorr  Te_eff = {row['Te_best']:.2f} eV  Ne = {row['Ne_best']:.2e}  "
                  f"chi2/dof = {row['chi2_red']:.2f}   |  n_c: Te_eff = {row['Te_nc']:.2f}  chi2/dof = "
                  f"{row['chi2_red_nc']:.2f}  dchi2 = {row['dchi2_nc']:.1f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUTDIR, "pressure_sweep_fits.csv"), index=False)
    plot_vs_pressure(res, FIGDIR)
    save_profiles(pd.concat(profs, ignore_index=True))
    if not overlays:
        plot_microwave_nc(res, None, None, MW_NC_DIR)
        return res, None, None

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
    plot_residuals(lines, checks, crf.CONFIG["sigma_model"], FIGDIR)
    plot_microwave_nc(res, lines, checks, MW_NC_DIR)

    cols = ["family", "p_mTorr", "Te_best", "Te_lo", "Te_hi", "EN_best", "Ne_best", "Ne_lo", "Ne_hi", "chi2_red",
            "edge_Ne", "n_1s5", "EN_nc", "Te_nc", "chi2_red_nc", "dchi2_nc", "n_1s5_nc", "n_1s3_nc"]
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
    elif "--profiles" in sys.argv:
        MINIMA = profiles_only()
    else:
        RES, LINES, CHECKS = main(overlays="--fits" not in sys.argv)

