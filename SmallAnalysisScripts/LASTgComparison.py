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
import os
import subprocess
import sys
import time
from fnmatch import fnmatch
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import PressureSweepAnalysis as psa                 # noqa: E402

ov, crf, mlf, he = psa.ov, psa.crf, psa.mlf, psa.he
als = crf.als

# ---- settings -------------------------------------------------------------------
LAS_FILE = os.path.join(crf.ROOT_DIR, "Experimental_Data", "LAS_Data", "Master_AllMeasurements.csv")
LAS_FREQ_MHZ = (2400, 2450)                         # source frequency range of the rows used
TG_VALID = (300.0, 2000.0)                          # K
TG_STEP = 50.0                                      # K, Tg of the libraries and CR tables rounded to this
FREQ_HZ = psa.FREQ_HZ                               # 2.42 GHz: libraries and n_c
N_C = psa.N_C
EN_MW = psa.EN_MW                                   # 3-1000 Td
PREC_MAX_TD_N2 = 30                                 # Ar/N2: precision 1e-30 only up to here (1e-30 hangs at
                                                    # 40 Td with >= 1.5 % N2, MicrowaveLowENFit); pure Ar as
                                                    # PressureSweepAnalysis (150 Td)
NE_GRID = psa.NE_GRID                               # 1e15-3e19 m^-3
DP_W = 5.0                                          # W, LAS points this close to the OES power are used
X_EXTRAP = 1.5                                      # % N2, nearest LAS N2 value held at most this far
OUTDIR = os.path.join(crf.ROOT_DIR, "Experimental_Data", "Output", "LASTgComparison")
FIGDIR = os.path.join(OUTDIR, "figures")
FIGSIZE = psa.FIGSIZE
GROUPS = {   # measured together, repeat scatter pooled (as PressureSweepAnalysis / CRFitNeTe)
    "pressure": dict(sweeps=[psa.SWEEP], measure_dir=psa.MEASURE_DIR),
    "n2": dict(sweeps=als.CONFIG["sweeps"], measure_dir=crf.CONFIG["outdir"]),
}
OES = {   # sweep -> (p [Torr], P [W], % N2) of x, Tg assumed before, labels
    "Pressure": dict(state=lambda x: (x / 1000, 80.0, 0.0), Tg_old=psa.TG, tag="pressure",
                     xlabel="Pressure [mTorr]", note="pure Ar, 80 W"),
    "N2 fraction": dict(state=lambda x: (1.0, 85.0, x), Tg_old=crf.CONFIG["Tg"], tag="N2",
                        xlabel="N$_2$ in the feed [%]", note="85 W, 1 Torr"),
    "Power": dict(state=lambda x: (1.0, x, mlf.anc.CONFIG["power_sweep_N2"]), Tg_old=crf.CONFIG["Tg"], tag="power",
                  xlabel="Power [W]", note=f"{mlf.anc.CONFIG['power_sweep_N2']:g} % N$_2$, 1 Torr"),
}
RANK = {"measured": 0, "interpolated": 1, "extrapolated": 2}


# ---- 1. LAS data and the Tg trend ------------------------------------------------------
def load_las(path=LAS_FILE):
    """LAS rows within LAS_FREQ_MHZ: run, date, nominal pressure p [Torr], power P [W], x [% N2], Tg [K],
    1s5 density n1s5 [m^-3] and their errors; valid_Tg / valid_n flag usable values."""
    d = pd.read_csv(path)
    d = d[d.frequency.between(*LAS_FREQ_MHZ)]
    las = pd.DataFrame(dict(run=d.run_id, date=d.date, p=d.pressure, P=d.power.astype(float),
                            x=d.Nitrogen_Percent.round(2), Tg=d.T_gas_K, Tg_err=d.T_gas_K_err,
                            n1s5=d.N_s_cm3 * 1e6, n1s5_err=d.N_s_cm3_err * 1e6)).reset_index(drop=True)
    las["valid_Tg"] = las.Tg.between(*TG_VALID)
    las["valid_n"] = np.isfinite(las.n1s5) & (las.n1s5 > 0)
    return las


def fit_tg_trend(las):
    """Tg trend (module docstring, 1.) -> (coefficients, run offsets, LAS Tg rows with residual / outlier)."""
    d = las[las.valid_Tg].reset_index(drop=True)
    runs, ri = np.unique(d.run, return_inverse=True)
    x, P, y = d.x.to_numpy(float), d.P.to_numpy(float), d.Tg.to_numpy(float)
    X = np.column_stack([np.eye(len(runs))[ri], P - 80.0, x, x ** 2])
    b = np.r_[np.full(len(runs), np.median(y)), 0.0, 0.0, 0.0]
    keep = np.ones(len(y), bool)
    for _ in range(3):
        b = least_squares(lambda c: (X @ c - y)[keep], b, loss="soft_l1", f_scale=50.0).x
        res = y - X @ b
        sigma = 1.4826 * np.median(np.abs(res[keep]))
        keep = np.abs(res) <= 4 * sigma
    off = (d.groupby("run").agg(date=("date", "first"), p=("p", "first"), n_points=("Tg", "size"))
           .loc[runs].assign(T_80W_0pct=b[:len(runs)]).reset_index())
    pr, T80 = off.p.to_numpy(float), off.T_80W_0pct.to_numpy(float)
    c = least_squares(lambda c: c[0] + c[1] * (pr - 1.0) - T80, [np.median(T80), 0.0], loss="soft_l1", f_scale=50.0).x
    off["resid"] = off.T_80W_0pct - (c[0] + c[1] * (off.p - 1.0))
    model = dict(T0=c[0], a_p=c[1], a_P=b[-3], a_x=b[-2], a_xx=b[-1], sigma_within=sigma,
                 n_points=int(keep.sum()), n_outliers=int((~keep).sum()), sigma_runs=1.4826 * np.median(np.abs(off.resid)))
    return model, off, d.assign(resid=res, outlier=~keep)


def tg_trend(m, p, P, x):
    """Tg [K] of the trend at pressure p [Torr], power P [W] and x % N2."""
    x = np.asarray(x, float)
    return m["T0"] + m["a_p"] * (np.asarray(p) - 1.0) + m["a_P"] * (np.asarray(P) - 80.0) + m["a_x"] * x + m["a_xx"] * x ** 2


def tg_round(T):
    return TG_STEP * np.floor(np.asarray(T, float) / TG_STEP + 0.5)


def _at_x(s, x):
    """1s5 of one LAS run (median per N2 value, Series indexed by % N2) at x % N2 -> (n, how) or None;
    1/n linear in x between measured N2 values; outside them (up to X_EXTRAP) the nearest measured value is
    held - the runs with 0 and 2 % N2 (0.6-0.9 Torr) give the same 1s5 within a few %, and at 1.5 Torr 1s5
    first rises with N2, so extrapolating 1/n from 1 and 3 % to 0 % would overestimate pure Ar by ~20 %."""
    xs, inv = s.index.to_numpy(float), 1.0 / s.to_numpy(float)
    hit = np.flatnonzero(np.isclose(xs, x, atol=0.02))
    if len(hit):
        return float(s.iloc[hit[0]]), "measured"
    if len(xs) >= 2 and xs.min() < x < xs.max():
        return 1.0 / float(np.interp(x, xs, inv)), "interpolated"
    j = int(np.argmin(np.abs(xs - x)))
    return (float(s.iloc[j]), "extrapolated") if np.abs(xs[j] - x) <= X_EXTRAP else None


def las_1s5_at(las, p, P, x):
    """LAS 1s5 [m^-3] at (p Torr, P W, x % N2) -> dict(n, how, rel_err, runs) (module docstring, 5.)."""
    d = las[las.valid_n & ((las.P - P).abs() <= DP_W + 1e-9)]
    per_p = {}
    for pr, g in d.groupby("p"):
        cand = []
        for run, h in g.groupby("run"):
            r = _at_x(h.groupby("x").n1s5.median(), x)
            if r:
                cand.append((*r, run, float(np.median(h.n1s5_err / h.n1s5))))
        if cand:
            best = min(RANK[c[1]] for c in cand)
            use = [c for c in cand if RANK[c[1]] == best]
            per_p[pr] = dict(n=1.0 / np.mean([1.0 / c[0] for c in use]), how=use[0][1],
                             rel_err=float(np.median([c[3] for c in use])), runs="+".join(c[2] for c in use))
    none = dict(n=np.nan, how="none", rel_err=np.nan, runs="")
    if not per_p:
        return none
    ps = np.array(sorted(per_p))
    if np.any(np.isclose(ps, p)):
        return per_p[ps[np.isclose(ps, p)][0]]
    lo, hi = ps[ps < p], ps[ps > p]
    if not len(lo) or not len(hi):
        return none
    a, b = per_p[lo.max()], per_p[hi.min()]
    w = (p - lo.max()) / (hi.min() - lo.max())
    return dict(n=float(np.exp((1 - w) * np.log(a["n"]) + w * np.log(b["n"]))),
                how=max(a["how"], b["how"], key=RANK.get) + ", p-interpolated",
                rel_err=max(a["rel_err"], b["rel_err"]), runs=a["runs"] + "|" + b["runs"])


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


# ---- 3. libraries and CR tables ----------------------------------------------------------
VIBSAT_XSEC = he.BOLSIG_XSEC_FOLDER / "Biagi_ArN2_novib.txt"   # Biagi Ar/N2 without the N2 vibrational excitations


def vibsat_xsec():
    """VIBSAT_XSEC (written from Biagi_ArN2.txt if missing): the N2 vibrational manifold saturated, i.e. in balance
    with the electrons - no net e-V energy loss; N2 elastic, electronic excitation and ionization kept."""
    if not VIBSAT_XSEC.is_file():
        lines = (he.BOLSIG_XSEC_FOLDER / "Biagi_ArN2.txt").read_text(errors="replace").splitlines()
        out, i = [], 0
        while i < len(lines):
            if lines[i].strip() == "EXCITATION" and i + 1 < len(lines) and "VIB" in lines[i + 1] \
                    and lines[i + 1].strip().startswith("N2"):
                j = i
                while not lines[j].startswith("-----"):
                    j += 1
                j += 1
                while not lines[j].startswith("-----"):
                    j += 1
                i = j + 1
                continue
            out.append(lines[i])
            i += 1
        VIBSAT_XSEC.write_text("\n".join(out) + "\n")
    return VIBSAT_XSEC


def lib_folder(n2, p_mTorr, Tg, vibsat=False):
    """vibsat: Ar/N2 library with saturated N2(v) (VIBSAT_XSEC); pure Ar (n2 = 0) is the same either way."""
    base = "Ar_Biagi_bolsig" if n2 == 0 else f"ArN2_{n2:g}pct_bolsig"
    tag = "_vibsat" if vibsat and n2 > 0 else ""
    return he.BOLSIG_FOLDER / f"{base}_mw{FREQ_HZ / 1e9:g}GHz_{Tg:g}K_{p_mTorr:g}mTorr{tag}"


def library(n2, p_mTorr, Tg, vibsat=False):
    """Microwave BOLSIG+ library of (n2 % N2, p_mTorr, Tg), run if missing (as PressureSweepAnalysis.library)."""
    folder = lib_folder(n2, p_mTorr, Tg, vibsat)
    if he.IsBolsigLibrary(folder):
        return folder
    prec_max = psa.PREC_MAX_TD if n2 == 0 else PREC_MAX_TD_N2
    prec = [1e-30 if v <= prec_max else 1e-25 for v in EN_MW]
    kw = dict(mlf.BOLSIG_SETTINGS, Tg=Tg, omega_N=2 * np.pi * FREQ_HZ / he.Torr2Volume(p_mTorr / 1000, Tg),
              overwrite=True, verbose=False, timeout=psa.BOLSIG_TIMEOUT)
    if n2 == 0:
        xsec, species, fractions = he.BOLSIG_XSEC_FOLDER / "Biagi_Ar.txt", ["Ar"], [1.0]
    else:
        xsec = vibsat_xsec() if vibsat else he.BOLSIG_XSEC_FOLDER / "Biagi_ArN2.txt"
        species, fractions = ["Ar", "N2"], [1 - n2 / 100, n2 / 100]
    run = lambda pr: he.RunBolsig(xsec, folder.name, EN_MW, species=species, fractions=fractions,
                                  **dict(kw, precision=pr))
    try:
        run(prec)
    except (subprocess.TimeoutExpired, RuntimeError) as err:
        print(f"{folder.name}: {type(err).__name__} at precision 1e-30 - retrying at 1e-25", flush=True)
        run(1e-25)
        return folder
    lib = he.ImportBolsigLibrary(folder, verbose=False)
    bad = [i for i, e in enumerate(lib["EEDFs"]) if e["E"][e["EEPF"] > 0].max() >= 1000]
    if bad:                                           # 1e-30 can give runaway tails to 9990 eV
        print(f"{folder.name}: runaway tails at {[EN_MW[i] for i in bad]} Td - those rows at 1e-25", flush=True)
        run([1e-25 if i in bad else v for i, v in enumerate(prec)])
    return folder


def _key(n2, p_mTorr, Tg):
    return f"{n2:g}:{p_mTorr:g}:{Tg:g}"


def _unkey(key):
    n2, p, Tg = key.split(":")
    return float(n2), float(p), float(Tg)


def table_cfg(key, group="pressure"):
    """CRFitNeTe settings of the CR table of key = '<% N2>:<p mTorr>:<Tg K>' (LAS Tg); one table folder per
    library, shared by the conditions (and sweeps) that round to the same (N2, p, Tg) - the group only sets
    the sweeps and measurement cache of the fit, not the table."""
    n2, p, Tg = _unkey(key)
    lib = lib_folder(n2, p, Tg)
    G = GROUPS[group]
    return dict(crf.CONFIG, sweeps=G["sweeps"], eedf=str(lib), P_Torr=p / 1000, Tg=Tg, N2_percent=n2,
                Ne_grid=NE_GRID, outdir=os.path.join(OUTDIR, "tables", lib.name), measure_dir=G["measure_dir"])


def old_cfg(c):
    """Settings of the earlier assumption for condition c: pressure sweep at Tg = 800 K (PressureSweepAnalysis),
    N2 and power sweeps at 300 K (MicrowaveLowENFit lowEN, 2.45 GHz)."""
    if c.sweep == "Pressure":
        return psa.fit_cfg("microwave", c.x)
    return mlf.job_cfg(f"lowEN:{c.N2:g}", build=False)


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


# ---- 4. fits --------------------------------------------------------------------------
def measurements(cfg):
    """Features and measured feature table of cfg's sweeps (rel_err incl. the pooled repeat scatter)."""
    tab = crf.build_model_table(cfg)
    feats, comps = crf.select_features(tab, cfg)
    ft = crf.feature_table(crf.measure(comps, cfg), feats, cfg)
    rs = crf.repeat_scatter(ft)
    feats["repeat_scatter"] = feats.feature.map(rs)
    ft["rel_err_fit"] = ft.rel_err
    if cfg["use_repeat_scatter"]:
        ft["rel_err"] = np.sqrt(ft.rel_err ** 2 + ft.feature.map(rs) ** 2)
    return feats, ft


def fit_condition(cfg, sweep, x, feats, ft):
    """Fit of one condition with the table of cfg; Ne pinned at n_c -> summary dict (CR 1s densities at the
    best E/N along n_c and their range over Delta chi^2 <= 1 s^2)."""
    tab = crf.build_model_table(cfg)
    grid = crf.fine_grid(tab, cfg)
    Te_f = crf.te_eff_fine(tab, grid)
    M = crf.feature_model(tab, feats, cfg)
    d = ft[(ft.sweep == sweep) & np.isclose(ft.x, x)]
    cond, _, _, posts = crf.fit_all(d, feats, M, grid, Te_f, cfg, verbose=False)
    c = cond.iloc[0]
    post = posts[f"{sweep}|{x:g}"]
    FIT = dict(tab=tab, feats=feats, ft=ft, grid=grid, Te_f=Te_f, posts=posts, cond=cond, cfg=cfg)
    pin = ov.pinned_at_nc(FIT, c)
    fX, fN = grid
    j = int(np.argmin(np.abs(fN - np.log(N_C))))
    chi2 = mlf.cdf.chi2_grid(post, c.chi2_min, c.birge)
    i = int(np.argmin(chi2[:, j]))
    ok = chi2[:, j] <= chi2[i, j] + c.birge ** 2
    row = dict(n_spec=c.n_spec, EN_nc=pin.x_best, Te_nc=pin.Te_best, Te_nc_lo=pin.Te_lo, Te_nc_hi=pin.Te_hi,
               chi2_red_nc=pin.chi2_red, dchi2_nc=(chi2[i, j] - c.chi2_min) / c.birge ** 2,
               EN_nc_at_edge=i in (0, len(fX) - 1), Ne_free=c.Ne_best, chi2_red_free=c.chi2_red)
    lev = list(tab["levels"])
    for m, lbl in psa.LEVELS_1S.items():
        ln = crf.on_fine_grid(tab, np.log(np.clip(tab["dens"][:, :, lev.index(lbl)], 1e-300, None)), cfg, k=1)
        n_j = np.exp(ln[:, j])
        row.update({f"n_{m}": float(n_j[i]), f"n_{m}_lo": float(n_j[ok].min()), f"n_{m}_hi": float(n_j[ok].max())})
    return row


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
    if len(sys.argv) > 2 and sys.argv[1] == "--lib":
        library(*_unkey(sys.argv[2]))
    elif len(sys.argv) > 2 and sys.argv[1] == "--lib-vibsat":
        library(*_unkey(sys.argv[2]), vibsat=True)
    elif len(sys.argv) > 2 and sys.argv[1] == "--build":
        build_table(sys.argv[2])
    else:
        LAS, MODEL, COND, COMP = main(plan_only="--plan" in sys.argv)
