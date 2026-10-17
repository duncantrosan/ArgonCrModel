# -*- coding: utf-8 -*-
"""
LASAndEEDFLibraries.py  -  gas temperature and 1s5 from the LAS table, BOLSIG+ EEDF libraries, the fit of one condition
=======================================================================================================================
TAKES IN   the LAS table (LAS_FILE, set by Main.py from [las] file: Master_AllMeasurements or Master_Summary layout),
           conditions as keys '<% admixture>:<p mTorr>:<Tg K>', the EEDF choice (vibsat, tv, an EEDF scheme)
GIVES OUT  load_las -> LAS rows; fit_tg_trend / tg_trend / tg_round -> the Tg of every OES condition;
           las_1s5_at -> the LAS 1s5 density at a condition;
           library / lib_folder -> a BOLSIG+ microwave library folder in InputData/Bolsig (built if missing, also
           in parallel worker processes: _run_workers, this file run with --lib... flags);
           table_cfg -> CRTableFit settings of the CR table of a condition;
           measurements / fit_condition -> the measured line features and the CR fit of one condition with Ne
           pinned (CR 1s densities at the best E/N, chi^2, band)
USED BY    CRTableBuilder (table settings), RunDatasets (Tg, LAS 1s5, libraries, fits), LatexReport, Main.py (sets
           TG_STEP, LAS_FILE, LAS_FREQ_MHZ, DP_W, LAS_RUNS, LAS_EXCLUDE_RUNS and the Ne pin N_C)

This is the pipeline half of SmallAnalysisScripts/LASTgComparison.py (that script keeps its own analysis: the LAS Tg
against the earlier assumed Tg, and imports everything below from here).  Logs of the library workers still go to
Experimental_Data/Output/LASTgComparison/logs.

The Tg trend:  Tg(p, P, x) = T0 + a_p (p - 1 Torr) + a_P (P - 80 W) + a_x x + a_xx x^2   (x = % N2), one offset per
LAS run (soft-L1, points beyond 4 sigma dropped), a robust line through the run offsets against pressure; Tg of a
condition = trend rounded to TG_STEP.  LAS 1s5 at a condition: LAS points within DP_W of the OES power, per run 1/n
linear in % N2 between the measured values (nearest measured value up to X_EXTRAP outside them), runs combined by
LAS_RUNS, ln n linear in pressure between the LAS pressures.
Libraries: Biagi Ar or Ar/N2 (vibsat: N2 vibrations removed; tv: reversible N2(v) ladder at Tv; scheme: an EEDF
scheme of CreateEEDFScheme.py), microwave field FREQ_HZ with omega/N at N = p / k Tg, BOLSIG+ gas temperature Tg,
precision 1e-30 at low E/N (1e-25 above, and for rows with runaway tails).
=======================================================================================================================
"""
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import constants
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # pipeline/: its modules import each other by name
import Paths                                        # noqa: E402 - the folder of Main.py
import PipelineSettings as PS                       # noqa: E402
import CRTableFit as crf                            # noqa: E402
import NitrogenActinometry as anc                   # noqa: E402 (BOLSIG+ settings)
from LASTableReader import read_las                 # noqa: E402

he = crf._helpers()
als = crf.als

# ---- settings -------------------------------------------------------------------
LAS_FILE = os.path.join(Paths.LAS_DATA, "Master_AllMeasurements.csv")
LAS_FREQ_MHZ = (2400, 2450)                         # source frequency range of the rows used
TG_VALID = (300.0, 2000.0)                          # K
TG_STEP = 50.0                                      # K, Tg of the libraries and CR tables rounded to this
FREQ_HZ = PS.FREQ_HZ                                # 2.42 GHz: libraries and n_c
N_C = PS.N_C                                        # Ne of the pinned fits: n_c, or the [fit] Ne_pin of Main.py
                                                    # (n_c itself stays PipelineSettings.N_C)
EN_MW = PS.EN_MW                                    # 3-1000 Td
PREC_MAX_TD_N2 = 30                                 # Ar/N2: precision 1e-30 only up to here (1e-30 hangs at
                                                    # 40 Td with >= 1.5 % N2, MicrowaveLowENFit); pure Ar
                                                    # PipelineSettings.PREC_MAX_TD (150 Td)
NE_GRID = crf.CONFIG["Ne_grid"]                     # 1e15-3e19 m^-3
DP_W = 5.0                                          # W, LAS points this close to the OES power are used
X_EXTRAP = 1.5                                      # % N2, nearest LAS N2 value held at most this far
LAS_RUNS = "all"                                    # las_1s5_at at one pressure: "all" = mean of the runs with the
                                                    # best value (measured > interpolated > extrapolated), "latest" =
                                                    # only the most recent of them
LAS_EXCLUDE_RUNS = ()                               # LAS run ids left out of everything (Main.py [las] exclude_runs)
OUTDIR = os.path.join(Paths.OUTPUT, "LASTgComparison")   # (name kept) logs of the library workers
GROUPS = {   # measured together, repeat scatter pooled (as CRTableFit); LASTgComparison.py adds its "pressure"
    "n2": dict(sweeps=als.CONFIG["sweeps"], measure_dir=crf.CONFIG["outdir"]),
}
RANK = {"measured": 0, "interpolated": 1, "extrapolated": 2}


# ---- 1. LAS data and the Tg trend ------------------------------------------------------
def load_las(path=LAS_FILE):
    """LAS rows within LAS_FREQ_MHZ: run, date, nominal pressure p [Torr], power P [W], x [% N2], Tg [K],
    1s5 density n1s5 [m^-3] and their errors; valid_Tg / valid_n flag usable values.  Rows of the runs
    LAS_EXCLUDE_RUNS are dropped (a Master_Summary row only when every run it combines is excluded)."""
    d = read_las(path)
    d = d[d.frequency.between(*LAS_FREQ_MHZ)]
    if LAS_EXCLUDE_RUNS:
        d = d[~d.run_id.astype(str).str.split(";").map(lambda r: set(r) <= set(LAS_EXCLUDE_RUNS))]
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
                cand.append((*r, run, float(np.median(h.n1s5_err / h.n1s5)), str(h.date.max())))
        if cand:
            best = min(RANK[c[1]] for c in cand)
            use = [c for c in cand if RANK[c[1]] == best]
            if LAS_RUNS == "latest":
                use = [max(use, key=lambda c: c[4])]
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


LADDER_XSEC = he.BOLSIG_XSEC_FOLDER / "Biagi_ArN2_vibladder.txt"   # Biagi Ar/N2 with a reversible N2(v) ladder


def ladder_xsec():
    """LADDER_XSEC (written from Biagi_ArN2.txt if missing) -> {v: level energy [eV]}.  The N2 v = 0 -> D vibrational
    excitations are replaced by reversible 'N2(vv) <-> N2(vw)' for every v < w (cross section of 0 -> D, D = w - v,
    shifted to the anharmonic threshold E_w - E_v; statistical weight ratio 1), so BOLSIG+ adds the superelastic
    N2(vw) -> N2(vv) collisions weighted by the N2(vw) mole fraction (ladder_composition).  'N2' keeps the elastic,
    electronic and ionization cross sections of all N2 molecules."""
    lines = (he.BOLSIG_XSEC_FOLDER / "Biagi_ArN2.txt").read_text(errors="replace").splitlines()
    keep, vib, i = [], {}, 0
    while i < len(lines):
        if lines[i].strip() in ("ELASTIC", "EFFECTIVE", "EXCITATION", "IONIZATION", "ATTACHMENT"):
            j = i
            while not lines[j].startswith("-----"):
                j += 1
            k = j + 1
            while not lines[k].startswith("-----"):
                k += 1
            m = re.fullmatch(r"\s*N2 -> N2\( VIB (\d*)V1\)\s*", lines[i + 1])
            if lines[i].strip() == "EXCITATION" and m:
                vib[int(m.group(1) or 1)] = (float(lines[i + 2].split()[0]),
                                             np.array([[float(a) for a in s.split()[:2]] for s in lines[j + 1:k]]))
            else:
                keep += lines[i:k + 1]
            i = k + 1
            continue
        keep.append(lines[i])
        i += 1
    E = {0: 0.0, **{D: e for D, (e, _) in vib.items()}}
    if not LADDER_XSEC.is_file():
        out, vmax = list(keep), max(vib)
        for v in range(vmax):
            for w in range(v + 1, vmax + 1):
                thr0, tab = vib[w - v]
                thr = E[w] - E[v]
                out += ["EXCITATION", f"N2(v{v}) <-> N2(v{w})", f"{thr:.6e}  1", "SPECIES: e / N2",
                        f"PROCESS: N2(v{v}) <-> N2(v{w}), cross section of v = 0 -> {w - v} shifted to the threshold",
                        "COLUMNS: Energy (eV) | Cross section (m2)", "-----------------------------"]
                out += [f"{max(a - thr0 + thr, 0.0):.6e}\t{b:.6e}" for a, b in tab]
                out += ["-----------------------------", ""]
        LADDER_XSEC.write_text("\n".join(out) + "\n")
    return E


def ladder_composition(n2, Tv):
    """(cross sections, species, mole fractions) of n2 % N2 with N2(v) in a Boltzmann distribution at Tv [K]: 'N2'
    (fraction x, elastic / electronic / ionization) and N2(vk), k = 0..vmax (fractions x p_k, the ladder only).  The
    fractions add up to 1 + x (BOLSIG+ takes them as given: a 1 + x error on the Ar density, 2 % at 2 % N2)."""
    E = ladder_xsec()
    lev = np.array([E[k] for k in sorted(E)])
    p = np.exp(-lev * constants.e / (constants.k * Tv))
    p /= p.sum()
    x = n2 / 100
    return LADDER_XSEC, ["Ar", "N2"] + [f"N2(v{k})" for k in sorted(E)], [1 - x, x] + list(x * p)


def lib_folder(n2, p_mTorr, Tg, vibsat=False, tv=None):
    """vibsat: Ar/N2 library with saturated N2(v) (VIBSAT_XSEC); tv: with N2(v) in a Boltzmann distribution at tv [K]
    (ladder_composition); pure Ar (n2 = 0) is the same either way."""
    base = "Ar_Biagi_bolsig" if n2 == 0 else f"ArN2_{n2:g}pct_bolsig"
    tag = "" if n2 == 0 else ("_vibsat" if vibsat else (f"_Tv{tv:g}K" if tv else ""))
    return he.BOLSIG_FOLDER / f"{base}_mw{FREQ_HZ / 1e9:g}GHz_{Tg:g}K_{p_mTorr:g}mTorr{tag}"


def eedf_scheme(name):
    """EEDF scheme <name> made with CreateEEDFScheme.py (InputData/EEDFSchemes/<name>.toml)."""
    import EEDFSchemes                              # noqa: E402 (pipeline/EEDFSchemes.py)
    return EEDFSchemes.load_scheme(name)


def library(n2, p_mTorr, Tg, vibsat=False, tv=None, scheme=None):
    """Microwave BOLSIG+ library of (n2 % N2, p_mTorr, Tg), run if missing (as PressureSweepAnalysis.library).
    scheme: name of an EEDF scheme (CreateEEDFScheme.py), which gives the cross sections, species, field and E/N;
    (n2, p_mTorr, Tg) are then the library's own (Scheme.eedf_conditions)."""
    S = eedf_scheme(scheme) if scheme else None
    folder = S.lib_folder(n2, p_mTorr, Tg) if S else lib_folder(n2, p_mTorr, Tg, vibsat, tv)
    if he.IsBolsigLibrary(folder):
        return folder
    en, freq = (S.EN_Td, S.freq_Hz) if S else (EN_MW, FREQ_HZ)
    prec_max = PS.PREC_MAX_TD if n2 == 0 else PREC_MAX_TD_N2
    prec = [1e-30 if v <= prec_max else 1e-25 for v in en]
    kw = dict(anc.CONFIG["bolsig_settings"], Tg=Tg, omega_N=2 * np.pi * freq / he.Torr2Volume(p_mTorr / 1000, Tg),
              overwrite=True, verbose=False, timeout=PS.BOLSIG_TIMEOUT)
    if S:
        xsec, (species, fractions) = S.xsec_file(), S.composition(n2)
    elif n2 == 0:
        xsec, species, fractions = he.BOLSIG_XSEC_FOLDER / "Biagi_Ar.txt", ["Ar"], [1.0]
    else:
        xsec = vibsat_xsec() if vibsat else he.BOLSIG_XSEC_FOLDER / "Biagi_ArN2.txt"
        species, fractions = ["Ar", "N2"], [1 - n2 / 100, n2 / 100]
        if tv and not vibsat:
            xsec, species, fractions = ladder_composition(n2, tv)
    run = lambda pr: he.RunBolsig(xsec, folder.name, en, species=species, fractions=fractions,
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
        print(f"{folder.name}: runaway tails at {[en[i] for i in bad]} Td - those rows at 1e-25", flush=True)
        run([1e-25 if i in bad else v for i, v in enumerate(prec)])
    return folder


def _key(n2, p_mTorr, Tg):
    return f"{n2:g}:{p_mTorr:g}:{Tg:g}"


def _unkey(key):
    n2, p, Tg = key.split(":")
    return float(n2), float(p), float(Tg)


def table_cfg(key, group="n2"):
    """CRFitNeTe settings of the CR table of key = '<% N2>:<p mTorr>:<Tg K>' (LAS Tg); one table folder per
    library, shared by the conditions (and sweeps) that round to the same (N2, p, Tg) - the group only sets
    the sweeps and measurement cache of the fit, not the table."""
    n2, p, Tg = _unkey(key)
    lib = lib_folder(n2, p, Tg)
    G = GROUPS[group]
    return dict(crf.CONFIG, sweeps=G["sweeps"], eedf=str(lib), P_Torr=p / 1000, Tg=Tg, N2_percent=n2,
                Ne_grid=NE_GRID, outdir=os.path.join(OUTDIR, "tables", lib.name), measure_dir=G["measure_dir"])


def _run_workers(flag, keys, what, script=None):
    """script (default this file) with flag for each key, PipelineSettings.N_WORKERS processes at a time
    (one BLAS thread each)."""
    if not keys:
        return
    logs = os.path.join(OUTDIR, "logs")
    os.makedirs(logs, exist_ok=True)
    print(f"{what}: {len(keys)} jobs ({PS.N_WORKERS} at a time, logs in {logs}) ...", flush=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0, procs = time.time(), {}
    for k in keys:
        while sum(q.poll() is None for q in procs.values()) >= PS.N_WORKERS:
            time.sleep(5)
        log = open(os.path.join(logs, f"{flag.strip('-')}_{k.replace(':', '_')}.log"), "w")
        procs[k] = subprocess.Popen([sys.executable, "-u", script or os.path.abspath(__file__), flag, k],
                                    stdout=log, stderr=subprocess.STDOUT, cwd=crf.ROOT_DIR, env=env)
    failed = [k for k, q in procs.items() if q.wait() != 0]
    print(f"{what} done ({time.time() - t0:.0f} s)", flush=True)
    if failed:
        raise RuntimeError(f"{what} failed for {failed} - see {logs}")


# ---- 4. fit of one condition ------------------------------------------------------
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


def fit_condition(cfg, sweep, x, feats, ft, ne=None):
    """Fit of one condition with the table of cfg; Ne pinned at ne (default N_C) -> summary dict (CR 1s densities at the
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
    ne = N_C if ne is None else ne
    pin = crf.pinned_at(FIT, c, ne)
    fX, fN = grid
    j = int(np.argmin(np.abs(fN - np.log(ne))))
    chi2 = crf.chi2_grid(post, c.chi2_min, c.birge)
    i = int(np.argmin(chi2[:, j]))
    ok = chi2[:, j] <= chi2[i, j] + c.birge ** 2
    row = dict(n_spec=c.n_spec, EN_nc=pin.x_best, Te_nc=pin.Te_best, Te_nc_lo=pin.Te_lo, Te_nc_hi=pin.Te_hi,
               chi2_red_nc=pin.chi2_red, dchi2_nc=(chi2[i, j] - c.chi2_min) / c.birge ** 2,
               EN_nc_at_edge=i in (0, len(fX) - 1), Ne_free=c.Ne_best, chi2_red_free=c.chi2_red)
    lev = list(tab["levels"])
    for m, lbl in PS.LEVELS_1S.items():
        ln = crf.on_fine_grid(tab, np.log(np.clip(tab["dens"][:, :, lev.index(lbl)], 1e-300, None)), cfg, k=1)
        n_j = np.exp(ln[:, j])
        row.update({f"n_{m}": float(n_j[i]), f"n_{m}_lo": float(n_j[ok].min()), f"n_{m}_hi": float(n_j[ok].max())})
    return row


if __name__ == "__main__":                          # library workers (_run_workers)
    if len(sys.argv) > 2 and sys.argv[1] == "--lib":
        library(*_unkey(sys.argv[2]))
    elif len(sys.argv) > 2 and sys.argv[1] == "--lib-vibsat":
        library(*_unkey(sys.argv[2]), vibsat=True)
    elif len(sys.argv) > 2 and sys.argv[1].startswith("--lib-scheme="):    # --lib-scheme=<EEDF scheme name>
        library(*_unkey(sys.argv[2]), scheme=sys.argv[1].split("=", 1)[1])
    elif len(sys.argv) > 2 and sys.argv[1].startswith("--lib-tv"):    # --lib-tv<Tv in K>
        library(*_unkey(sys.argv[2]), tv=float(sys.argv[1][len("--lib-tv"):]))
    else:
        print(__doc__)
