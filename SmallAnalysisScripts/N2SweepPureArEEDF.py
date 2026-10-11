# -*- coding: utf-8 -*-
"""
N2SweepPureArEEDF.py

CR fit and N I 744.2 nm / Ar I actinometry of the echelle sweeps listed in N2SweepPureArEEDF_input.toml (first
GsaSweep_85W_1_torr: 85 W, 1 Torr, 0-3 % N2) with the CR fit that works for the N2 sweep:
  - pure-Ar microwave BOLSIG+ EEDF (2.42 GHz, LAS gas temperature) - N2 left out of the Boltzmann equation only;
    the CR model keeps the N2 quenching of the Ar(4s, 4p) levels and the dilution of the Ar ground state
  - cross sections ArgonCrossSections_BSR_analytic5p.json: BSR-500, ground/4s -> 5p from the model's analytic
    (Drawin / Bogaerts) formulae instead of the NGFSRDW fallback (AnalyticFivePFit)
  - Ne pinned at the critical density n_c, E/N refitted; no response tilt (CRFitNeTe.no_tilt)
CR tables are AnalyticFivePFit's.  EEDF = 'vibsat' (Main.py, [eedf] model = "ArN2_vib_saturated") uses instead the
Ar/N2 microwave EEDF with saturated N2(v) (no net e-V loss; LASTgComparison.lib_folder(vibsat=True)), one library per
(% N2, p, Tg).

0. Data sets: the input file (N2SweepPureArEEDF_input.toml next to this script, or the file given on the command
   line) gives per set the folder, the file-name pattern, the swept quantity (% N2, pressure or power) and the
   fixed plasma condition.  Per condition Tg from the LAS trend rounded to 50 K (or the set's Tg_K) and the LAS
   1s5 (LASTgComparison.tg_trend, las_1s5_at).  Missing pure-Ar microwave libraries and CR tables of every set
   are built first, in parallel.  Line fits are cached per set and redone when the spectra in the folder change.
1. Ar metastables: CR 1s5 and 1s3 at the fit (band: Delta chi^2 <= 1 s^2 along E/N at n_c) against the LAS 1s5.
2. N I / Ar I actinometry (ActinometryNitrogenContent: per pair, Tukey fences, weighted combination) with the
   fitted EEDF and E/N posterior at n_c.  N I lines: 744.23 (ActinometryLineSelection) and EXTRA_N_WL (746.83, same
   upper level 3p 4S*3/2, so the same k_N; branching ratios 0.32 and 0.53); per condition one estimate from all
   pairs and one per N I line.  Two models of the Ar reference line:
     direct    no metastables: excitation from the Ar ground state only, every excitation emits with the branching
               ratio b:  n_N/n_Ar = (I_N/I_Ar) (k_Ar b_Ar)/(k_N b_N) (lam_N/lam_Ar)
     cr        with metastables: the CR model's emission of the line per ground-state Ar atom and electron,
               I_model/(Ne n_Ar) = n_u A eta/(Ne n_Ar), replaces k_Ar b_Ar - stepwise excitation from the 4s
               (metastable / resonant) levels, cascades, collisional losses of the upper level and trapping.
               (ActinometryNitrogenContent's 'stepwise' k_Ar b_Ar / f_direct agrees with it within ~20 % for the
               4p levels but overestimates the 5p lines 4-6x: at n_c the 5p levels lose most of their population
               in collisions, not by radiation.)
   The N I lines are excited from ground-state N only (k_N, BSR).  Ar ground-state cross sections are the CR model's
   own (BSR 4p, analytic 5p).  The Ar reference lines are ActinometryLineSelection's (4p and 5p) without
   EXCLUDE_AR_WL (left out of the per-pair results, the Tukey fences and the combination).  Conditions without N2
   are left out (pure-Ar sets: metastables only).
3. Per Ar reference line (with metastables): n_N/n_Ar of each N I / Ar pair, to check that the lines agree.
4. Checks of the data: Te_eff and E/N of the fit at n_c against the sweep; the measured N I / Ar I line ratios
   (median of the repeats, every Ar reference, the EXCLUDE_AR_WL ones dashed); I(746.83)/I(744.23) against the A/lambda
   ratio (same upper level: off it means response calibration, a blend or an N2 band); the raw spectra around each
   N I line (the tracked ones and the other lines from their upper level) and the Ar I reference ZOOM_AR_WL - every
   spectrum of the set on one graph, coloured by the sweep value, with the number of spectra in which the line fit
   detected the line.

Output in Experimental_Data/Output/N2SweepPureArEEDF/<data set>/ (<x> = N2, pressure or power):
  fits.csv, actinometry_per_condition.csv (N_line: 'combined' or one N I line), actinometry_per_pair.csv,
  line_ratios.csv, NI_same_upper_ratios.csv, measurements/ (line fits)
  figures/metastables_vs_<x>.png, actinometry_vs_<x>.png, dissociation_vs_<x>.png, actinometry_by_N_line_vs_<x>.png,
          actinometry_per_line_NI<wl>_vs_<x>.png, Te_eff_vs_<x>.png, EN_vs_<x>.png, line_ratios_NI<wl>_vs_<x>.png,
          NI_746.83_over_744.23_vs_<x>.png, raw_lines/raw_<NI|ArI>_<wl>nm.png
and for all data sets of the run in Experimental_Data/Output/N2SweepPureArEEDF/:
  fits_all.csv, actinometry_per_condition_all.csv
  overview_<x>_<1s5|Te_eff|EN|actinometry|dissociation>.png (sweep types with >= 2 data sets)
Run from Spyder (F5) or python SmallAnalysisScripts/N2SweepPureArEEDF.py [input file] [--plan]
(--plan: conditions, Tg, LAS 1s5 and the libraries / CR tables to build; nothing is built or fitted).
"""
import contextlib
import functools
import io
import os
import re
import sys
import tomllib
import warnings
from collections import Counter
from fnmatch import fnmatch
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import AnalyticFivePFit as A                        # noqa: E402

L, crf, he = A.L, A.crf, A.he
anc = L.mlf.anc                                     # ActinometryNitrogenContent
ar, als = anc.ar, anc.als

INPUT_FILE = os.path.join(HERE, "N2SweepPureArEEDF_input.toml")
ECHELLE_DIR = os.path.join(crf.ROOT_DIR, "Experimental_Data", "EchelleData")   # input-file folders relative to this
OUTDIR = os.path.join(crf.ROOT_DIR, "Experimental_Data", "Output", "N2SweepPureArEEDF")
N_C = L.N_C
NE_MODE = "nc"                                      # 'free': Ne fitted with E/N (2D posterior); 'nc': pinned at n_c;
                                                    # 'value': pinned at N_C set by Main.py ([fit] Ne_pin)
EEDF = "ar"                                         # AnalyticFivePFit table set: 'ar' pure-Ar microwave EEDF,
                                                    # 'vibsat' Ar/N2 microwave EEDF with saturated N2(v), 'tv<Tv>'
                                                    # Ar/N2 with N2(v) Boltzmann at Tv K, e.g. 'tv8000' (Main.py)
PLOTS = dict(metastables=True, fit_params=True, eedf=True, residuals=True, fit_overlay=False, chi2_maps=False,
             actinometry=True, actinometry_per_line=True, line_ratios=True, same_upper_ratios=True, raw_lines=True,
             overviews=True, dpi=200)                # which figures to make (Main.py [plots])


def _ne(pre=", ", post=""):
    """'<pre>N_e = n_c<post>' for plot titles when Ne is pinned, '' when it is fitted."""
    if NE_MODE == "free":
        return ""
    return f"{pre}N$_e$ = " + ("n$_c$" if NE_MODE == "nc" else f"{N_C:.2g} m$^{{-3}}$") + post


def _band():
    """Where the uncertainty bands come from: along E/N at the pinned Ne, or over (E/N, Ne)."""
    return "over (E/N, N$_e$)" if NE_MODE == "free" else "along E/N"


def eedf_label():
    """The EEDF family of the run, for plot titles."""
    if EEDF == "vibsat":
        return "Ar/N$_2$ microwave EEDF, saturated N$_2$(v)"
    if EEDF.startswith("tv"):
        return f"Ar/N$_2$ microwave EEDF, N$_2$(v) at T$_v$ = {float(EEDF[2:]):g} K"
    return "pure-Ar microwave EEDF"
ACT = dict(anc.CONFIG, pin_Ne=N_C)                  # actinometry settings (fences, margins, xsec errors)
EXTRA_N_WL = (746.831,)                             # N I lines tracked besides ActinometryLineSelection's (744.229):
                                                    # 746.83 is from the same upper level (3p 4S*3/2, A 1.96e7 vs
                                                    # 1.19e7 s^-1) but was not detected in the selection's test
                                                    # spectrum; 742.36 (third line) lies in an echelle gap
EXCLUDE_AR_WL = (706.7218, 415.859, 794.8176)       # Ar references left out: 4p8, 5p5, 4p7 (systematically off the
                                                    # other pairs by 1.5-2.5x at every N2 % of the 85 W sweep)
VARIANTS = {"direct": dict(color="C0", marker="o", label="no metastables (ground-state excitation of the Ar level)"),
            "cr": dict(color="C3", marker="s", label="with metastables (CR-model emission of the Ar line)")}
FIT_PARAMS = (   # fit results at n_c plotted against the sweep: column (band <col>_lo/_hi), axis label, file name
    ("Te_nc", r"$T_{e,\mathrm{eff}} = \frac{2}{3}\langle\varepsilon\rangle$ of the fitted EEDF [eV]", "Te_eff"),
    ("EN_nc", "E/N of the fitted EEDF [Td]", "EN"))
ZOOM_AR_WL = 750.3869                               # Ar I reference of the raw-spectrum zooms (4p10, mostly direct)
ZOOM_HALF_NM = 0.4                                  # half width of the zooms
SWEPT = {   # 'sweep' of the input file: input key of the swept quantity, axis label, unit
    "N2": dict(key="N2_percent", xlabel="N$_2$ in the feed [%]", unit="% N2"),
    "pressure": dict(key="pressure_mTorr", xlabel="Pressure [mTorr]", unit="mTorr"),
    "power": dict(key="power_W", xlabel="Power [W]", unit="W"),
}
FIXED = {"power_W": lambda v: f"{v:g} W", "pressure_mTorr": lambda v: f"{v / 1000:g} Torr",
         "N2_percent": lambda v: f"{v:g} % N$_2$"}  # plasma condition of a data set (plot titles)


# ---- 0. data sets --------------------------------------------------------------------------
def file_parser(patterns, exclude_x=()):
    """Spectrum name (without .spa) -> (x, repeat) or None from the input file's patterns: {x} the sweep value
    ('_' or '.' as decimal point), {rep} the repeat number (1 if absent); x in exclude_x -> None."""
    regs = []
    for p in [patterns] if isinstance(patterns, str) else patterns:
        if "{x}" not in p:
            raise ValueError(f"file pattern {p!r} has no {{x}}")
        regs.append(re.compile(re.escape(p).replace(re.escape("{x}"), r"(?P<x>\d+(?:[_.]\d+)?)")
                               .replace(re.escape("{rep}"), r"(?P<rep>\d+)")))
    exclude_x = np.asarray(exclude_x, float)

    def parse(name):
        for r in regs:
            m = r.fullmatch(name)
            if m:
                x = float(m["x"].replace("_", "."))
                return None if np.any(np.isclose(x, exclude_x)) else (x, int(m.groupdict().get("rep") or 1))
        return None
    return parse


LIST_COLUMNS = ("file", "pressure_mTorr", "gas1", "gas2", "gas2_percent", "power_W")   # file list of a data set
LIST_OPTIONAL = ("flow_sccm", "repeat", "Tg_K", "note")
LIST_STATE = {"pressure_mTorr": "pressure_mTorr", "power_W": "power_W", "N2_percent": "gas2_percent"}


def read_file_list(path, kind, folder, exclude_x, where):
    """Spectra of a data set listed in a CSV file (one row per spectrum, columns LIST_COLUMNS + LIST_OPTIONAL; header
    case and spaces do not matter, '#' starts a comment) -> (parse, state, {x: Tg} or {}, fixed {key: value} of the
    quantities that are the same in every row).  The swept quantity of `kind` is read from its column; every other
    condition must be the same in all rows of one sweep value.  Files listed but not in the folder are an error."""
    t = pd.read_csv(path, dtype=str, comment="#", skipinitialspace=True).dropna(how="all")
    low = {c.strip().lower(): c for c in t.columns}
    if missing := [c for c in LIST_COLUMNS if c.lower() not in low]:
        raise ValueError(f"{where}: {path} has no column {', '.join(missing)} (needed: {', '.join(LIST_COLUMNS)}; "
                         f"optional: {', '.join(LIST_OPTIONAL)})")
    t = t.rename(columns={low[c.lower()]: c for c in LIST_COLUMNS + LIST_OPTIONAL if c.lower() in low})
    t["file"] = t.file.str.strip().str.replace(r"\.spa$", "", case=False, regex=True)
    for c in ("pressure_mTorr", "gas2_percent", "power_W", "flow_sccm", "repeat", "Tg_K"):
        if c in t:
            t[c] = pd.to_numeric(t[c].str.strip(), errors="raise")
    t["gas2_percent"] = t.gas2_percent.fillna(0.0)
    t["gas1"], t["gas2"] = t.gas1.fillna("").str.strip(), t.gas2.fillna("").str.strip()
    if bad := sorted(set(t.gas1[t.gas1.str.lower() != "ar"])):
        raise ValueError(f"{where}: {path}: gas1 must be Ar (the CR model is argon), not {bad}")
    if bad := sorted(set(t.gas2[(t.gas2_percent > 0) & (t.gas2.str.upper() != "N2")])):
        raise ValueError(f"{where}: {path}: gas2 must be N2 when gas2_percent > 0, not {bad}")
    if dup := sorted(t.file[t.file.str.lower().duplicated()]):
        raise ValueError(f"{where}: {path}: files listed twice: {', '.join(dup[:10])}")
    have = {os.path.splitext(os.path.basename(f))[0].lower() for f in glob(os.path.join(folder, "*.spa"))}
    if missing := [f for f in t.file if f.lower() not in have]:
        raise FileNotFoundError(f"{where}: {path} lists {len(missing)} file(s) not in {folder}: "
                                f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    x_col = LIST_STATE[SWEPT[kind]["key"]]
    t["x"] = t[x_col].astype(float)
    t = t[~np.isclose(t.x.to_numpy()[:, None], np.asarray(exclude_x, float)[None, :]).any(axis=1)] \
        if len(exclude_x) else t
    if "repeat" not in t or t.repeat.isna().any():
        t["repeat"] = t.groupby("x").cumcount() + 1
    states, tgs = {}, {}
    for x, g in t.groupby("x"):
        v = g[list(LIST_STATE.values())].drop_duplicates()
        if len(v) > 1:
            raise ValueError(f"{where}: {path}: the rows with {x_col} = {x:g} differ in their other conditions:\n"
                             f"{v.to_string(index=False)}")
        states[x] = tuple(float(v.iloc[0][c]) for c in LIST_STATE.values())
        if "Tg_K" in g and g.Tg_K.notna().any():
            if g.Tg_K.nunique() > 1:
                raise ValueError(f"{where}: {path}: Tg_K differs within {x_col} = {x:g}")
            tgs[x] = float(g.Tg_K.dropna().iloc[0])
    lookup = {f.lower(): (float(x), int(r)) for f, x, r in zip(t.file, t.x, t.repeat)}
    fixed = {q: float(t[LIST_STATE[q]].iloc[0]) for q in FIXED                     # in FIXED order (titles)
             if LIST_STATE[q] != x_col and t[LIST_STATE[q]].nunique() == 1}
    return (lambda name: lookup.get(name.lower())), (lambda x: states[min(states, key=lambda v: abs(v - x))]), tgs, fixed


def load_input(path=INPUT_FILE):
    """Data sets of the input file with run = true (default): dict(name, kind, note, Tg_K, sweep = the
    ActinometryLineSelection / CRFitNeTe sweep entry, state = x -> (p mTorr, P W, % N2), outdir, measure_dir).
    The spectra of a set are found by the file-name pattern 'files' or listed in the CSV 'file_list' (read_file_list;
    relative to the input file's folder)."""
    with open(path, "rb") as f:
        entries = tomllib.load(f).get("dataset", [])
    sets, names = [], set()
    for k, d in enumerate(entries):
        name = str(d.get("name", f"dataset_{k + 1}"))
        where = f"{os.path.basename(path)}, data set {name!r}"
        if name in names:
            raise ValueError(f"{where}: the name is used twice")
        names.add(name)
        if not d.get("run", True):
            continue
        if missing := [q for q in ("folder", "sweep") if q not in d]:
            raise ValueError(f"{where}: {', '.join(missing)} missing")
        if ("files" in d) == ("file_list" in d):
            raise ValueError(f"{where}: give either files (file-name pattern) or file_list (CSV), not "
                             f"{'both' if 'files' in d else 'neither'}")
        kind = d["sweep"]
        if kind not in SWEPT:
            raise ValueError(f"{where}: sweep = {kind!r} - use one of {', '.join(map(repr, SWEPT))}")
        folder = os.path.join(ECHELLE_DIR, d["folder"])
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"{where}: folder {folder} not found")
        tg_x = {}
        if "file_list" in d:
            lst = d["file_list"] if os.path.isabs(d["file_list"]) else os.path.join(os.path.dirname(os.path.abspath(path)),
                                                                                  d["file_list"])
            if not os.path.isfile(lst):
                raise FileNotFoundError(f"{where}: file_list {lst} not found")
            parse, state, tg_x, fixed = read_file_list(lst, kind, folder, d.get("exclude_x", ()), where)
        else:
            need = [q for q in FIXED if q != SWEPT[kind]["key"]]
            if missing := [q for q in need if q not in d]:
                raise ValueError(f"{where}: {' and '.join(missing)} missing (a {kind} sweep needs {' and '.join(need)})")
            fixed = {q: float(d[q]) for q in need}

            def state(x, kind=kind, fixed=fixed):
                v = dict(fixed, **{SWEPT[kind]["key"]: float(x)})
                return v["pressure_mTorr"], v["power_W"], v["N2_percent"]

            parse = file_parser(d["files"], d.get("exclude_x", ()))
        note = ", ".join(FIXED[q](v) for q, v in fixed.items())
        out = os.path.join(OUTDIR, name)
        sets.append(dict(name=name, kind=kind, note=note, state=state, Tg_x=tg_x,
                         Tg_K=float(d["Tg_K"]) if "Tg_K" in d else None,
                         sweep=dict(name=name, folder=folder, parse=parse,
                                    xlabel=SWEPT[kind]["xlabel"], note=note, exclude_x=()),
                         outdir=out, measure_dir=os.path.join(out, "measurements")))
    return sets


def _plain(s):
    return s.replace("N$_2$", "N2").replace("T$_v$", "Tv")


def _tg_note(ds):
    return "LAS T$_g$" if ds["Tg_K"] is None else f"T$_g$ = {ds['Tg_K']:g} K"


def conditions(ds, las, trend):
    """Conditions of a data set with a calibrated spectrum: x, p, P, % N2, number of spectra, Tg (LAS trend rounded
    to LASTgComparison.TG_STEP, or the set's Tg_K), LAS 1s5 and the key of the CR table."""
    sw = ds["sweep"]
    fl = als.export_flags(dict(als.CONFIG, sweeps=[sw]))
    if fl.empty:
        return pd.DataFrame()
    fl["px"] = fl.file.map(sw["parse"])
    fl = fl[fl.px.notna()]
    cal = fl.export_flags.map(lambda f: fnmatch(f, als.CONFIG["export_flags"])).astype(bool)
    if (~cal).any():
        print(f"{ds['name']}: {(~cal).sum()} spectra without the intensity calibration left out "
              f"({', '.join(fl.file[~cal])})")
    n_files = Counter(px[0] for px in fl.px[cal])
    rows = []
    for x in sorted(n_files):
        p, P, n2 = ds["state"](x)
        Tg = ds["Tg_K"] if ds["Tg_K"] is not None else ds.get("Tg_x", {}).get(x)
        if Tg is None:
            Tg = float(L.tg_round(L.tg_trend(trend, p / 1000, P, n2)))
        r = L.las_1s5_at(las, p / 1000, P, n2)
        rows.append(dict(sweep=sw["name"], x=x, p_mTorr=p, p_Torr=p / 1000, P_W=P, N2=n2, n_files=n_files[x], Tg=Tg,
                         las_1s5=r["n"], las_1s5_how=r["how"], las_1s5_rel_err=r["rel_err"], key=L._key(n2, p, Tg)))
    return pd.DataFrame(rows)


def to_build(keys):
    """(microwave libraries as (worker flag, key), CR tables) still missing for the table keys: pure-Ar libraries,
    and with EEDF 'vibsat' / 'tv<Tv>' the Ar/N2 library of every condition with N2."""
    libs = sorted({("--lib", L._key(0.0, p, Tg)) for n2, p, Tg in map(L._unkey, keys)
                   if (EEDF == "ar" or n2 == 0) and not he.IsBolsigLibrary(L.lib_folder(0.0, p, Tg))})
    if EEDF != "ar":
        flag = "--lib-vibsat" if EEDF == "vibsat" else f"--lib-tv{float(EEDF[2:]):g}"
        kw = dict(vibsat=True) if EEDF == "vibsat" else dict(tv=float(EEDF[2:]))
        libs += sorted({(flag, k) for k in keys if L._unkey(k)[0] > 0
                        and not he.IsBolsigLibrary(L.lib_folder(*L._unkey(k), **kw))})
    with A.xsec_set(A.CR_XSEC):
        tabs = [k for k in keys if not crf.table_is_current(A.table_cfg(k, EEDF))]
    return libs, tabs


def fit_cfg(c, ds):
    cfg = dict(A.table_cfg(c.key, EEDF), sweeps=[ds["sweep"]], measure_dir=ds["measure_dir"])
    crf.no_tilt(cfg)
    return cfg


def measure_note(ds, cache, wl=None):
    """Delete a line-fit cache of the data set that does not hold exactly the spectra now in its folder
    (ActinometryLineSelection.measure_sweeps only checks the lines); say so when the spectra are fitted (also when
    the cache lacks one of the lines wl, which measure_sweeps then fits)."""
    path, sw = os.path.join(ds["measure_dir"], cache), ds["sweep"]
    if os.path.exists(path):
        m = pd.read_csv(path, usecols=["file", "wl_air"])
        names = (os.path.splitext(os.path.basename(f))[0] for f in glob(os.path.join(sw["folder"], "*.spa")))
        if set(m.file.astype(str)) == {n for n in names if sw["parse"](n)}:
            if wl is None or set(np.round(wl, 4)) <= set(m.wl_air.round(4)):
                return
            print(f"  {cache}: new lines", flush=True)
        else:
            os.remove(path)
            print(f"  {cache}: the spectra in the folder changed", flush=True)
    print(f"  fitting the lines of every spectrum (cached in {ds['measure_dir']}) ...", flush=True)


# ---- 1. CR fits at n_c ---------------------------------------------------------------------
def fit_conditions(ds, cond):
    """Fit at n_c of every condition -> (summary table, {x: fit dict for the actinometry})."""
    unit = SWEPT[ds["kind"]]["unit"]
    measure_note(ds, "ar_line_measurements.csv")
    with A.xsec_set(A.CR_XSEC), contextlib.redirect_stdout(io.StringIO()):
        feats, ft = L.measurements(fit_cfg(next(cond.itertuples()), ds))
    rows, fits = [], {}
    for c in cond.itertuples():
        d = ft[(ft.sweep == c.sweep) & np.isclose(ft.x, c.x)]
        if d.empty:
            print(f"  {c.x:6g} {unit}: no usable Ar I feature in any spectrum - skipped", flush=True)
            continue
        cfg = fit_cfg(c, ds)
        with A.xsec_set(A.CR_XSEC), contextlib.redirect_stdout(io.StringIO()):
            row = L.fit_condition(cfg, c.sweep, c.x, feats, ft)
            tab = crf.build_model_table(cfg)
            grid = crf.fine_grid(tab, cfg)
            Te_f = crf.te_eff_fine(tab, grid)
            M = crf.feature_model(tab, feats, cfg)
            fc, _, _, posts = crf.fit_all(d, feats, M, grid, Te_f, cfg, verbose=False)
            eedfs = crf.eedf_axis(cfg)[0]
        f0, (fX, fN) = fc.iloc[0], grid                 # E/N band along n_c, as the Te and 1s5 bands
        j = int(np.argmin(np.abs(fN - np.log(N_C))))
        chi2 = L.mlf.cdf.chi2_grid(posts[f"{c.sweep}|{c.x:g}"], f0.chi2_min, f0.birge)[:, j]
        band = fX[chi2 <= chi2.min() + f0.birge ** 2]
        row.update(EN_nc_lo=float(np.exp(band.min())), EN_nc_hi=float(np.exp(band.max())), Ne_fit=N_C,
                   Ne_fit_lo=N_C, Ne_fit_hi=N_C)
        i = int(np.argmin(chi2))
        if NE_MODE == "free":
            i, j = free_fit_row(row, tab, grid, Te_f, posts[f"{c.sweep}|{c.x:g}"], f0, cfg)
        fits[c.x] = dict(tab=tab, grid=grid, Te_f=Te_f, posts=posts, cond=fc, cfg=cfg, eedfs=eedfs, i=i, j=j,
                         EN_best=float(np.exp(fX[i])), resid=pinned_residuals(d, feats, M, i, j, cfg))
        rows.append(dict(x=c.x, N2=c.N2, p_Torr=c.p_Torr, P_W=c.P_W, Tg=c.Tg, las_1s5=c.las_1s5,
                         las_1s5_how=c.las_1s5_how, las_1s5_rel_err=c.las_1s5_rel_err, **row))
        print(f"  {c.x:6g} {unit} (Tg {c.Tg:.0f} K): E/N {row['EN_nc']:5.1f} Td  Te_eff {row['Te_nc']:.2f} eV  "
              f"chi2/dof {row['chi2_red_nc']:.2f}  Ne {row['Ne_fit']:.2e}  1s5 {row['n_1s5']:.2e} (LAS {c.las_1s5:.2e})",
              flush=True)
    t = pd.DataFrame(rows)
    if len(t):
        t["ratio_cr_las"] = t.n_1s5 / t.las_1s5
    return t, fits


def free_fit_row(row, tab, grid, Te_f, post, f0, cfg):
    """With NE_MODE 'free': the summary row of LASTgComparison.fit_condition (columns named *_nc for the pinned fit)
    replaced in place by the global best fit over (E/N, Ne), bands over Delta chi^2 <= B^2 in both; dchi2_nc keeps
    the cost of pinning Ne at n_c.  -> (i, j) of the best fine-grid point."""
    fX, fN = grid
    chi2 = L.mlf.cdf.chi2_grid(post, f0.chi2_min, f0.birge)
    i, j = np.unravel_index(int(np.argmin(chi2)), chi2.shape)
    ok = chi2 <= chi2[i, j] + f0.birge ** 2
    T2 = np.broadcast_to(Te_f[:, None], chi2.shape) if np.ndim(Te_f) == 1 else Te_f
    rows_ok, cols_ok = np.any(ok, axis=1), np.any(ok, axis=0)
    row.update(EN_nc=float(np.exp(fX[i])), EN_nc_lo=float(np.exp(fX[rows_ok].min())),
               EN_nc_hi=float(np.exp(fX[rows_ok].max())), Te_nc=float(T2[i, j]), Te_nc_lo=float(T2[ok].min()),
               Te_nc_hi=float(T2[ok].max()), chi2_red_nc=float(f0.chi2_red), EN_nc_at_edge=i in (0, len(fX) - 1),
               Ne_fit=float(np.exp(fN[j])), Ne_fit_lo=float(np.exp(fN[cols_ok].min())),
               Ne_fit_hi=float(np.exp(fN[cols_ok].max())))
    lev = list(tab["levels"])
    for m, lbl in L.psa.LEVELS_1S.items():
        n2 = np.exp(crf.on_fine_grid(tab, np.log(np.clip(tab["dens"][:, :, lev.index(lbl)], 1e-300, None)), cfg, k=1))
        row.update({f"n_{m}": float(n2[i, j]), f"n_{m}_lo": float(n2[ok].min()), f"n_{m}_hi": float(n2[ok].max())})
    return int(i), int(j)


def pinned_residuals(d, feats, M, i, j, cfg):
    """Per measured point of one condition at the fit (E/N row i, Ne column j of the fine grid): ln I_meas, the model
    ln I_model, the spectrum's profiled scale s (as CRFitNeTe.fit_block), the residual and its error."""
    li = d.feature.map({f: k for k, f in enumerate(feats.feature)}).to_numpy()
    si, files = pd.factorize(d.file)
    y, m = np.log(d.area.to_numpy()), M[i, j, li]
    X = np.eye(si.max() + 1)[si]
    sig2 = d.rel_err.to_numpy() ** 2 + cfg["sigma_fit_floor"] ** 2
    W = np.linalg.inv(np.diag(sig2) + cfg["sigma_model"] ** 2 * (li[:, None] == li[None, :]))
    sc = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ (y - m)
    return pd.DataFrame(dict(file=d.file.to_numpy(), feature=d.feature.to_numpy(), wl=d.wl.to_numpy(), ln_meas=y,
                             ln_model=m, s=sc[si], resid=y - sc[si] - m, err=np.sqrt(sig2)))


# ---- 2. actinometry ----------------------------------------------------------------------
def ar_cross_sections_of_model():
    """Ground-state excitation cross sections of every Ar level as the CR model uses them (analytic-5p set),
    in the ActinometryRates format."""
    with A.xsec_set(A.CR_XSEC), contextlib.redirect_stdout(io.StringIO()):
        MD, _ = he.GetData()
    g = next(v for v in MD.values() if v["kind"] == "ground")
    out = {}
    for r in g["Electron Impact CrossSections"]["Reactants"]:
        lab = r["partner_label"]
        out[lab] = dict(species="Ar I", label=lab, energy_eV=np.asarray(r["energy_eV"], float),
                        cross_section_m2=np.asarray(r["cross_section"], float), threshold_eV=float(r["threshold_eV"]),
                        level_energy_eV=float(MD[lab]["energy_eV"]), desc=MD[lab].get("term", ""),
                        source=r.get("source", ""))
    return out


class CRTerms(anc.FitTerms):
    """FitTerms whose Ar 'rate' is the CR-model emission of the reference line along E/N at Ne:
    ln(I_obs / (Ne n_Ar b_Ar)), so that k_Ar b_Ar in ActinometryNitrogenContent becomes I_model / (Ne n_Ar)."""

    def __init__(self, fit, xs, cfg, lines, Ne):
        super().__init__(fit, xs, cfg)
        self.lines, self.Ne = lines, Ne                    # {Ar upper: (wl, b_Ar)}; Ne None: every Ne of the grid
        self.j = None if Ne is None else int(np.argmin(np.abs(fit["grid"][1] - np.log(Ne))))
        c = fit["cfg"]
        self.n_Ar = he.Torr2Volume(crf.gas_fractions(c)[0] * c["P_Torr"], c["Tg"])

    def lnk(self, species, upper):
        if species != "Ar I":
            return super().lnk(species, upper)
        key = ("CR", upper)
        if key not in self._k:
            tab, (wl, b) = self.fit["tab"], self.lines[upper]
            k = int(np.argmin(np.abs(tab["line_wl"] - wl)))
            lnI = crf.on_fine_grid(tab, np.log(np.clip(tab["I_obs"][:, :, k], 1e-300, None)), self.fit["cfg"], k=1)
            if self.j is None:                             # (E/N, Ne) field for the 2D posterior
                self._k[key] = lnI - np.log(np.exp(self.fit["grid"][1])[None, :] * self.n_Ar * b)
            else:
                self._k[key] = lnI[:, self.j] - np.log(self.Ne * self.n_Ar * b)
        return self._k[key]


def tracked_lines(xs):
    """(N I, Ar I) lines of the actinometry: ActinometryLineSelection's selection and the N I lines EXTRA_N_WL from
    its line list (A, upper level, branching ratio from the N I A sums, as select_lines)."""
    n_lines, a_lines = als.select_lines(als.CONFIG, xs)
    res = pd.read_csv(os.path.join(als.CONFIG["selection_dir"], als.CONFIG["lines_file"]))
    extra = res[(res.species == "N I") & res.wl_air.round(3).isin(np.round(EXTRA_N_WL, 3))
                & ~res.wl_air.round(3).isin(n_lines.wl_air.round(3))]
    have = pd.concat([n_lines.wl_air, extra.wl_air])
    if missing := [w for w in EXTRA_N_WL if not np.isclose(w, have, atol=1e-3).any()]:
        raise ValueError(f"EXTRA_N_WL {missing}: not an N I line of {als.CONFIG['lines_file']}")
    if bad := sorted(set(extra.upper) - set(xs["N I"])):
        raise ValueError(f"EXTRA_N_WL: no electron-impact cross section for the N I upper level(s) {bad}")
    A_sums = ar.load_n_A_sums()
    extra = extra.assign(threshold_eV=[xs["N I"][u]["threshold_eV"] for u in extra.upper],
                         branching=[ar.branching_ratio(A, A_sums.get(u)) for A, u in zip(extra.Aki, extra.upper)])
    return pd.concat([n_lines, extra]).sort_values("wl_air").reset_index(drop=True), a_lines


def same_upper_ratios(meas, n_lines, exclude_x):
    """Per condition: measured intensity ratio of every two tracked N I lines from the same upper level (median of
    the spectra with both detected) and the ratio it must have, A/lambda (energy units) or A (counts)."""
    m = meas[(meas.species == "N I") & meas.status.isin(als.CONFIG["sweep_status"])
             & ~meas.x.round(6).isin(np.round(exclude_x, 6))]
    rows = []
    for _, g in n_lines.groupby("upper"):
        g = g.sort_values("wl_air")
        for i, a in enumerate(g.itertuples()):
            for b in list(g.itertuples())[i + 1:]:
                ma, mb = (m[np.isclose(m.wl_air, r.wl_air, atol=5e-5)].set_index("file") for r in (a, b))
                j = ma.join(mb, lsuffix="_a", rsuffix="_b", how="inner")
                if j.empty:
                    continue
                expect = (b.Aki / a.Aki) * ((a.wl_air / b.wl_air) if crf.CONFIG["intensity_units"] == "energy" else 1.0)
                r = j.area_b / j.area_a
                for x, k in r.groupby(j.x_a):
                    if len(k) >= als.CONFIG["min_reps"]:
                        rows.append(dict(wl_a=a.wl_air, wl_b=b.wl_air, x=x, ratio=k.median(), n_rep=len(k),
                                         err=1.2533 * k.std(ddof=1) / np.sqrt(len(k)), expected=expect,
                                         spectra=";".join(f"{v:.4g}" for v in k)))
    return pd.DataFrame(rows)


def measured_pairs(ds, cond):
    """Measured N I / Ar I line ratios of the conditions with N2 (with branching ratios, upper levels and ref_used =
    Ar reference not in EXCLUDE_AR_WL), the cross sections (Ar I: the CR model's) and the same-upper-level N I
    ratios (same_upper_ratios)."""
    xs = ar.load_all_cross_sections()
    sw = dict(ds["sweep"], exclude_x=tuple(cond.x[cond.N2 <= 0]))      # no N I without N2
    acfg = dict(als.CONFIG, sweeps=[sw], outdir=ds["measure_dir"], measure_cache="actinometry_line_measurements.csv")
    n_lines, a_lines = tracked_lines(xs)
    xs["Ar I"] = ar_cross_sections_of_model()
    measure_note(ds, acfg["measure_cache"], pd.concat([n_lines.wl_air, a_lines.wl_air]))
    with contextlib.redirect_stdout(io.StringIO()):
        meas = als.filter_export_flags(als.measure_sweeps(pd.concat([n_lines, a_lines]), acfg), acfg)
    same = same_upper_ratios(meas, n_lines, sw["exclude_x"])
    _, cond = als.pair_ratios(meas, n_lines, a_lines, acfg)
    if cond.empty:
        return cond, xs, same
    wl = pd.concat([n_lines.wl_air, a_lines.wl_air]).round(4)
    bmap = dict(zip(wl, pd.concat([n_lines.branching, a_lines.branching])))
    umap = dict(zip(wl, pd.concat([n_lines.upper, a_lines.upper])))
    c = cond.rename(columns={"mean": "I_ratio", "err": "I_err"})
    c = c.assign(bN=c.n_wl.round(4).map(bmap), bA=c.ar_wl.round(4).map(bmap),
                 n_upper=c.n_wl.round(4).map(umap), ar_upper=c.ar_wl.round(4).map(umap),
                 lam=c.n_wl / c.ar_wl if crf.CONFIG["intensity_units"] == "energy" else 1.0)
    c = c[np.isfinite(c.bN) & np.isfinite(c.bA)]
    return c.assign(ref_used=~c.ar_wl.round(3).isin(np.round(EXCLUDE_AR_WL, 3))), xs, same


def actinometry(ds, cond, fits):
    """-> (per-pair estimates, per-condition estimates, measured line ratios of every pair, same-upper-level N I
    ratios).  Per condition one estimate from every pair (N_line 'combined') and one per N I line."""
    ratios, xs, same = measured_pairs(ds, cond)
    if ratios.empty:
        return pd.DataFrame(), pd.DataFrame(), ratios, same
    pairs = ratios[ratios.ref_used]
    pair_rows, cond_rows = [], []
    for x, p in pairs.groupby("x"):
        if x not in fits:
            continue
        fit = fits[x]
        c = cond[cond.x == x].iloc[0]
        if NE_MODE == "free":                           # the full (E/N, Ne) posterior
            post = fit["posts"][f"{ds['name']}|{x:g}"] / fit["posts"][f"{ds['name']}|{x:g}"].sum()
            T2 = np.broadcast_to(fit["Te_f"][:, None], post.shape) if np.ndim(fit["Te_f"]) == 1 else fit["Te_f"]
            te, en = float((post * T2).sum()), float(np.exp((post.sum(axis=1) * fit["grid"][0]).sum()))
        else:
            post, te, en = anc.pinned_posterior(fit, fit["posts"][f"{ds['name']}|{x:g}"], N_C)
        lines = {r.ar_upper: (r.ar_wl, r.bA) for r in p.drop_duplicates("ar_upper").itertuples()}
        terms = {"direct": anc.FitTerms(fit, xs, ACT),
                 "cr": CRTerms(fit, xs, ACT, lines, None if NE_MODE == "free" else N_C)}
        f = c.N2 / 100
        N = he.Torr2Volume(fit["cfg"]["P_Torr"], fit["cfg"]["Tg"])
        for v in VARIANTS:
            per, est = anc.estimate(terms[v], p, post, "direct", ACT)   # 'direct' fields: no f_direct factor
            pair_rows += [dict(x=x, variant=v, **r) for r in per]
            ests = [("combined", est)] + [(f"{w:.2f}", anc.estimate(terms[v], q, post, "direct", ACT)[1])
                                          for w, q in p.groupby("n_wl")]
            for line, e in ests:
                if e is None:
                    continue
                n, s = e["nN_nAr"], e["ln_err_stat"]
                cond_rows.append(dict(x=x, N2=c.N2, p_Torr=c.p_Torr, P_W=c.P_W, variant=v, N_line=line, Te_eff=te,
                                      EN_Td=en, **e, nN_nAr_lo=n * np.exp(-s), nN_nAr_hi=n * np.exp(s),
                                      n_N=n * (1 - f) * N, dissociation=n * (1 - f) / (2 * f)))
    return pd.DataFrame(pair_rows), pd.DataFrame(cond_rows), ratios, same


def _combined(act):
    return act[act.N_line == "combined"] if len(act) else act


# ---- plots ---------------------------------------------------------------------------------
def _yerr(y, s):
    return [y - y * np.exp(-s), y * np.exp(s) - y]


def _savefig(fig, path):
    """Save at 200 dpi; a file Windows will not open for writing (open in a viewer) is reported, not fatal."""
    try:
        fig.savefig(path, dpi=PLOTS["dpi"])
    except OSError as err:
        print(f"could not write {path} ({err.strerror}) - is it open in a viewer?", flush=True)


def _dx(x, frac):
    """frac of the sweep range: horizontal offset that keeps overlapping series apart."""
    span = float(np.ptp(np.asarray(x, float))) if len(x) > 1 else 0.0
    return frac * (span or 1.0)


def _fit_eepf(fit, E):
    """EEPF of the fit's E/N (between the two neighbouring library rows, linear in ln EEPF and ln E/N) on E [eV]."""
    xg, ee = np.asarray(fit["tab"]["x_grid"], float), fit["eedfs"]
    k = int(np.clip(np.searchsorted(xg, fit["EN_best"]) - 1, 0, len(xg) - 2))
    w = np.clip(np.log(fit["EN_best"] / xg[k]) / np.log(xg[k + 1] / xg[k]), 0.0, 1.0)
    lf = [np.log(np.clip(np.interp(E, e["E"], e["EEPF"], right=0.0), 1e-300, None)) for e in (ee[k], ee[k + 1])]
    return np.exp((1 - w) * lf[0] + w * lf[1])


def plot_eedfs(t, fits, ds, outdir):
    """EEPF of the fit at n_c of every condition, and relative to the first condition."""
    E = np.linspace(0.01, 30.0, 1500)
    xs = sorted(fits)
    cols = plt.cm.viridis(np.linspace(0, 0.92, len(xs)))
    unit = SWEPT[ds["kind"]]["unit"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    ref = _fit_eepf(fits[xs[0]], E)
    for c, x in zip(cols, xs):
        f = _fit_eepf(fits[x], E)
        r = t[np.isclose(t.x, x)].iloc[0]
        axs[0].plot(E, f, color=c, lw=1.8, label=f"{x:g} {unit}: {r.EN_nc:.0f} Td, T$_{{e,eff}}$ {r.Te_nc:.2f} eV")
        axs[1].plot(E, f / ref, color=c, lw=1.8)
    for ax in axs:
        for th in (11.55, 12.9, 14.5):
            ax.axvline(th, color="0.6", ls=":", lw=1)
        ax.set(yscale="log", xlim=(0, 25), xlabel="electron energy [eV]")
        ax.grid(True, which="both", alpha=0.25)
    axs[0].set(ylim=(1e-14, 3), ylabel="EEPF [eV$^{-3/2}$]",
               title=f"EEDF of the fit{_ne(' at ')}, {ds['note']}\n{eedf_label()}; dotted: 1s, 4p, 5p thresholds")
    axs[1].set(ylim=(1e-3, 1e3), ylabel=f"EEPF / EEPF({xs[0]:g} {unit})", title="relative to the first condition")
    axs[0].legend(fontsize=7, ncol=2 if len(xs) > 10 else 1)
    fig.tight_layout()
    _savefig(fig, os.path.join(outdir, f"eedf_vs_{ds['kind']}.png"))


def plot_residuals(fits, ds, outdir):
    """Residual ln I_meas - s - ln I_model of every line feature at the fit (n_c) against the sweep, and measured
    (scale removed) against model intensities of all conditions."""
    r = pd.concat([f["resid"].assign(x=x) for x, f in fits.items()], ignore_index=True)
    g = r.groupby(["wl", "x"]).agg(resid=("resid", "mean"), sd=("resid", "std")).reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
    for k, (wl, h) in enumerate(g.groupby("wl")):
        h = h.sort_values("x")
        axs[0].errorbar(h.x, h.resid, yerr=h.sd.fillna(0), fmt="o-", ms=4, capsize=2, color=plt.cm.tab10(k % 10),
                        label=f"{wl:.2f} nm")
    axs[0].axhline(0, color="k", lw=1)
    sm = next(iter(fits.values()))["cfg"]["sigma_model"]
    axs[0].axhspan(-sm, sm, color="0.88", zorder=0, label=f"$\\sigma_{{model}}$ = {sm:.2f}")
    axs[0].set(xlabel=SWEPT[ds["kind"]]["xlabel"], ylabel="ln I$_{meas}$ - s - ln I$_{model}$ (mean of the repeats)",
               title=f"Line residuals at the fit{_ne(' (', ')')}, {ds['note']}\n{eedf_label()}")
    sc = axs[1].scatter(r.ln_model, r.ln_meas - r.s, c=r.x, s=12, cmap="viridis")
    lim = np.array([r.ln_model.min(), r.ln_model.max()])
    axs[1].plot(lim, lim, "k--", lw=1)
    fig.colorbar(sc, ax=axs[1], label=SWEPT[ds["kind"]]["xlabel"])
    axs[1].set(xlabel="ln I$_{model}$", ylabel="ln I$_{meas}$ - s",
               title=f"measured (per-spectrum scale removed) vs model, {len(fits)} conditions")
    for ax in axs:
        ax.grid(True, alpha=0.25)
    axs[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _savefig(fig, os.path.join(outdir, f"residuals_vs_{ds['kind']}.png"))


def _x_tag(x, ds):
    return f"{x:g}{SWEPT[ds['kind']]['unit'].split()[0]}".replace("%", "pct")


def plot_fit_overlay(fits, ds, outdir):
    """One figure per condition: measured line features of every spectrum (scale removed) and the model at the fit."""
    os.makedirs(outdir, exist_ok=True)
    for x, f in sorted(fits.items()):
        r = f["resid"]
        fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
        mod = r.drop_duplicates("feature")
        axs[0].plot(mod.wl, np.exp(mod.ln_model), "s", color="C3", ms=8, mfc="none", mew=2, label="model")
        for k, (fl, h) in enumerate(r.groupby("file")):
            axs[0].plot(h.wl, np.exp(h.ln_meas - h.s), "o", ms=4, color="0.4", alpha=0.7,
                        label="measured spectra (scale removed)" if k == 0 else None)
            axs[1].errorbar(h.wl, h.resid, yerr=h.err, fmt="o", ms=3, color="0.4", alpha=0.7, capsize=2)
        axs[1].axhline(0, color="k", lw=1)
        axs[0].set(yscale="log", ylabel="line intensity [model units]",
                   title=f"{ds['name']}, {x:g} {SWEPT[ds['kind']]['unit']}: fit{_ne(' at ')}, "
                         f"E/N {f['EN_best']:.1f} Td ({eedf_label()})")
        axs[1].set(xlabel="wavelength [nm]", ylabel="residual (ln)")
        for ax in axs:
            ax.grid(True, which="both", alpha=0.25)
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        _savefig(fig, os.path.join(outdir, f"fit_{_x_tag(x, ds)}.png"))
        plt.close(fig)


def plot_chi2_maps(fits, ds, outdir):
    """One figure per condition: Delta chi^2 / B^2 over (E/N, Ne) with the free and the n_c fit, and its profile over
    Ne (B = Birge ratio)."""
    os.makedirs(outdir, exist_ok=True)
    for x, f in sorted(fits.items()):
        f0, (fX, fN) = f["cond"].iloc[0], f["grid"]
        chi2 = L.mlf.cdf.chi2_grid(f["posts"][f"{ds['name']}|{x:g}"], f0.chi2_min, f0.birge)
        d = (chi2 - chi2.min()) / f0.birge ** 2
        fig, axs = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(width_ratios=[1.25, 1]))
        pc = axs[0].pcolormesh(np.exp(fX), np.exp(fN), np.minimum(d, 25).T, cmap="viridis_r", shading="auto")
        axs[0].contour(np.exp(fX), np.exp(fN), d.T, levels=[1, 4, 9], colors="w", linewidths=[1.5, 1, 0.7])
        fig.colorbar(pc, ax=axs[0], label="$\\Delta\\chi^2/B^2$ (clipped at 25)")
        axs[0].axhline(N_C, color="C3", ls="--", lw=1.5)
        axs[0].plot(f0.x_best, f0.Ne_best, "*", color="gold", mec="k", ms=16, label="free fit")
        jn = int(np.argmin(np.abs(fN - np.log(N_C))))
        axs[0].plot(np.exp(fX[int(np.argmin(chi2[:, jn]))]), N_C, "o", color="C3", mec="k", ms=9, label="best at n$_c$")
        axs[0].set(xscale="log", yscale="log", xlabel="E/N [Td]", ylabel="N$_e$ [m$^{-3}$]",
                   title=f"{ds['name']}, {x:g} {SWEPT[ds['kind']]['unit']} ({eedf_label()})")
        axs[0].legend(fontsize=8, loc="lower left")
        prof = d.min(axis=0)
        axs[1].plot(np.exp(fN), prof, "k-", lw=2)
        axs[1].axvline(N_C, color="C3", ls="--", lw=1.5, label="n$_c$")
        axs[1].axhline(1, color="0.5", ls=":", lw=1.2)
        axs[1].set(xscale="log", ylim=(-0.5, max(12.0, 1.1 * prof[jn])), xlabel="N$_e$ [m$^{-3}$]",
                   ylabel="min over E/N of $\\Delta\\chi^2/B^2$", title=f"profile over N$_e$ (B = {f0.birge:.2f})")
        axs[1].grid(True, which="both", alpha=0.25)
        axs[1].legend(fontsize=8)
        fig.tight_layout()
        _savefig(fig, os.path.join(outdir, f"chi2_{_x_tag(x, ds)}.png"))
        plt.close(fig)


def plot_metastables(t, ds, outdir):
    t = t.sort_values("x")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m, col, mk, lab in (("1s5", "C3", "s", "CR 1s$_5$ (metastable, J=2)"),
                            ("1s3", "C1", "^", "CR 1s$_3$ (metastable, J=0)")):
        ax.fill_between(t.x, t[f"n_{m}_lo"], t[f"n_{m}_hi"], color=col, alpha=0.15, lw=0)
        ax.plot(t.x, t[f"n_{m}"], mk + "-", color=col, ms=7, lw=2, label=lab)
    meas = t.las_1s5_how == "measured"
    for sel, mfc, lab in ((meas, "k", "LAS 1s$_5$ (measured at the condition)"),
                          (~meas, "white", "LAS 1s$_5$ (interpolated / extrapolated)")):
        g = t[sel & np.isfinite(t.las_1s5)]
        if len(g):
            ax.errorbar(g.x, g.las_1s5, yerr=g.las_1s5 * g.las_1s5_rel_err, fmt="D", color="k", mfc=mfc, ms=7,
                        capsize=3, label=lab)
    ax.set_ylim(0, 1.25 * np.nanmax(np.r_[t.n_1s5_hi, t.las_1s5 * (1 + t.las_1s5_rel_err)]))   # room for the legend
    ax.set_xlabel(SWEPT[ds["kind"]]["xlabel"])
    ax.set_ylabel("density [m$^{-3}$]")
    ax.set_title(f"Ar metastables, {ds['note']}: CR fit ({eedf_label()}{_ne()}, {_tg_note(ds)}, "
                 f"no tilt)\nE/N {t.EN_nc.min():.0f}-{t.EN_nc.max():.0f} Td, $\\chi^2$/dof {t.chi2_red_nc.min():.1f}-"
                 f"{t.chi2_red_nc.max():.1f}; band: $\\Delta\\chi^2 \\leq 1\\,s^2$ {_band()}", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, os.path.join(outdir, f"metastables_vs_{ds['kind']}.png"))


def _n_label(cond):
    """'N I 744.23 + 746.83 nm' for the N I lines of the per-condition estimates."""
    return "N I " + " + ".join(sorted(set(cond.N_line) - {"combined"})) + " nm"


def plot_actinometry(cond, ds, outdir):
    lab, cond = _n_label(cond), _combined(cond)
    for key, ylab, fname in (("nN_nAr", "n$_N$ / n$_{Ar}$", "actinometry"),
                             ("dissociation", "dissociation degree n$_N$ / (2 n$_{N_2}$)  (feed N$_2$)",
                              "dissociation")):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for k, (v, st) in enumerate(VARIANTS.items()):
            g = cond[cond.variant == v].sort_values("x")
            dx = (k - 0.5) * _dx(cond.x, 0.01)
            y = g[key].to_numpy()
            ax.errorbar(g.x + dx, y, yerr=_yerr(y, g.ln_err_total.to_numpy()), fmt="none", ecolor=st["color"],
                        alpha=0.3, elinewidth=4)
            ax.errorbar(g.x + dx, y, yerr=_yerr(y, g.ln_err_stat.to_numpy()), fmt=st["marker"] + "-",
                        color=st["color"], ms=7, capsize=3, label=st["label"])
        ax.set_ylim(bottom=0)
        ax.set_xlabel(SWEPT[ds["kind"]]["xlabel"])
        ax.set_ylabel(ylab)
        ax.set_title(f"{lab} / Ar I actinometry, {ds['note']} ({eedf_label()}{_ne()}, "
                     "no tilt)\nbars: line ratios + E/N posterior + pair scatter; faint: + cross sections",
                     fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        _savefig(fig, os.path.join(outdir, f"{fname}_vs_{ds['kind']}.png"))


def plot_by_n_line(cond, ds, outdir, variant="cr"):
    """n_N/n_Ar from each N I line on its own and from all pairs together."""
    c = cond[cond.variant == variant]
    lines = sorted(set(c.N_line) - {"combined"})
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for k, (line, col, mk) in enumerate([("combined", "k", "s")] + [(w, f"C{i}", "o") for i, w in enumerate(lines)]):
        g = c[c.N_line == line].sort_values("x")
        y = g.nN_nAr.to_numpy()
        ax.errorbar(g.x + (k - len(lines) / 2) * _dx(c.x, 0.008), y, yerr=_yerr(y, g.ln_err_stat.to_numpy()),
                    fmt=mk + "-", color=col, ms=7 if line == "combined" else 6, lw=2.2 if line == "combined" else 1.4,
                    capsize=3, label="all pairs" if line == "combined" else f"N I {line} nm pairs only")
    ax.set_ylim(bottom=0)
    ax.set_xlabel(SWEPT[ds["kind"]]["xlabel"])
    ax.set_ylabel("n$_N$ / n$_{Ar}$")
    ax.set_title(f"Actinometry per N I line, {VARIANTS[variant]['label']}\n{ds['note']}, {eedf_label()}, "
                 f"{_ne('', ', ')}no tilt (bars: line ratios + E/N posterior + pair scatter)", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, os.path.join(outdir, f"actinometry_by_N_line_vs_{ds['kind']}.png"))


def plot_same_upper(same, ds, outdir):
    """Measured ratio of two N I lines from the same upper level against A/lambda (calibration, blends, N2 bands)."""
    for (wa, wb), g in same.groupby(["wl_a", "wl_b"]):
        g = g.sort_values("x")
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for r in g.itertuples():
            v = np.array(r.spectra.split(";"), float)
            ax.plot(np.full(len(v), r.x), v, ".", color="0.6", ms=5)
        ax.plot([], [], ".", color="0.6", ms=5, label="single spectra (both lines detected)")
        ax.errorbar(g.x, g.ratio, yerr=g.err, fmt="s-", color="C3", ms=7, capsize=3, label="median of the spectra")
        ax.axhline(g.expected.iloc[0], color="k", ls="--", lw=1.2,
                   label=f"expected {g.expected.iloc[0]:.3f} (" +
                         ("A ratio x $\\lambda$ ratio, calibrated)" if crf.CONFIG["intensity_units"] == "energy"
                          else "A ratio)"))
        ax.set_ylim(bottom=0)
        ax.set_xlabel(SWEPT[ds["kind"]]["xlabel"])
        ax.set_ylabel(f"I(N I {wb:.2f}) / I(N I {wa:.2f})")
        ax.set_title(f"N I {wb:.2f} / {wa:.2f} nm (same upper level), {ds['note']}\n"
                     "off the expected value: response calibration, blend or N$_2$ band under one line", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        _savefig(fig, os.path.join(outdir, f"NI_{wb:.2f}_over_{wa:.2f}_vs_{ds['kind']}.png"))


def plot_per_line(pairs, cond, ds, outdir, variant="cr"):
    """n_N/n_Ar of every N I / Ar pair, one graph per N I line, with the combined estimate of all pairs."""
    c = _combined(cond)
    c = c[c.variant == variant].sort_values("x")
    p = pairs[(pairs.variant == variant) & (pairs.valid_mass >= ACT["min_valid_mass"])]
    refs = p.drop_duplicates("ar_wl").sort_values("ar_wl")
    cols = dict(zip(refs.ar_wl, plt.cm.tab10(np.arange(len(refs)) % 10)))
    for n_wl, pn in p.groupby("n_wl"):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.fill_between(c.x, c.nN_nAr_lo, c.nN_nAr_hi, color="0.5", alpha=0.2, lw=0)
        ax.plot(c.x, c.nN_nAr, "k-", lw=2.5, label=f"combined, {_n_label(cond)} (outliers left out)")
        for k, r in enumerate(refs.itertuples()):
            g = pn[pn.ar_wl == r.ar_wl].sort_values("x")
            dx = (k - len(refs) / 2) * _dx(p.x, 0.004)
            ax.errorbar(g.x + dx, g.nN_nAr, yerr=[g.nN_nAr - g.nN_nAr_lo, g.nN_nAr_hi - g.nN_nAr], fmt="o-", ms=5,
                        lw=1.3, capsize=2, color=cols[r.ar_wl], label=f"Ar {r.ar_wl:.2f} ({r.ar_upper})")
            o = g[g.outlier]
            ax.plot(o.x + dx, o.nN_nAr, "x", color="k", ms=10, mew=2)
        ax.plot([], [], "x", color="k", ms=10, mew=2, label="outlier (Tukey fence)")
        ax.set_ylim(bottom=0)
        ax.set_xlabel(SWEPT[ds["kind"]]["xlabel"])
        ax.set_ylabel(f"n$_N$ / n$_{{Ar}}$ from N I {n_wl:.2f} nm / Ar line")
        ax.set_title(f"Actinometry per Ar reference line, N I {n_wl:.2f} nm, {VARIANTS[variant]['label']}\n"
                     f"{ds['note']}, {eedf_label()}{_ne()}, no tilt (bars: 16-84 % from E/N)",
                     fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        _savefig(fig, os.path.join(outdir, f"actinometry_per_line_NI{n_wl:.2f}_vs_{ds['kind']}"
                                           f"{'' if variant == 'cr' else '_' + variant}.png"))


def plot_fit_params(t, ds, outdir):
    """Te_eff and E/N of the fit at n_c against the sweep (bars: Delta chi^2 <= 1 s^2 along E/N)."""
    t = t.sort_values("x")
    for col, ylab, fname in FIT_PARAMS:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        y = t[col]
        ax.errorbar(t.x, y, yerr=[y - t[f"{col}_lo"], t[f"{col}_hi"] - y], fmt="s-", color="C3", ms=7, lw=2, capsize=3)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(SWEPT[ds["kind"]]["xlabel"])
        ax.set_ylabel(ylab)
        ax.set_title(f"CR fit{_ne(' at ')}, {ds['note']} ({eedf_label()}, {_tg_note(ds)}, no tilt)\n"
                     f"bars: $\\Delta\\chi^2 \\leq 1\\,s^2$ {_band()}", fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        _savefig(fig, os.path.join(outdir, f"{fname}_vs_{ds['kind']}.png"))


def plot_ratios(ratios, ds, outdir):
    """Measured N I / Ar I line ratios (median of the repeats) against the sweep, one curve per Ar reference."""
    for n_wl, r in ratios.groupby("n_wl"):
        refs = r.drop_duplicates("ar_wl").sort_values("ar_wl")
        cols = dict(zip(refs.ar_wl, plt.cm.tab10(np.arange(len(refs)) % 10)))
        fig, ax = plt.subplots(figsize=(9, 6))
        for k, a in enumerate(refs.itertuples()):
            g = r[r.ar_wl == a.ar_wl].sort_values("x")
            dx = (k - len(refs) / 2) * _dx(r.x, 0.004)
            ax.errorbar(g.x + dx, g.I_ratio, yerr=g.I_err, fmt="o-" if a.ref_used else "o--", ms=5, lw=1.3, capsize=2,
                        color=cols[a.ar_wl], mfc=None if a.ref_used else "white",
                        label=f"Ar {a.ar_wl:.2f} ({a.ar_upper})" + ("" if a.ref_used else ", not used"))
        ax.set_yscale("log")
        ax.set_xlabel(SWEPT[ds["kind"]]["xlabel"])
        ax.set_ylabel(f"I(N I {n_wl:.2f}) / I(Ar I)  (line areas, calibrated spectra)")
        ax.set_title(f"Measured line ratios N I {n_wl:.2f} nm / Ar I, {ds['note']}\nmedian of the repeats "
                     f"(at least {als.CONFIG['min_reps']} with both lines detected); bars: repeat scatter or fit error",
                     fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        _savefig(fig, os.path.join(outdir, f"line_ratios_NI{n_wl:.2f}_vs_{ds['kind']}.png"))


@functools.lru_cache(maxsize=None)
def zoom_lines():
    """Lines of the raw-spectrum zooms -> ([dict(species, wl, role)], line catalogue): the tracked N I lines
    (tracked_lines), the other catalogue lines from the upper level of the N I actinometry line(s), and ZOOM_AR_WL."""
    with contextlib.redirect_stdout(io.StringIO()):
        n_lines, _ = tracked_lines(ar.load_all_cross_sections())
        cat, _ = als._catalogue()
    act = n_lines[np.isfinite(n_lines.branching)]
    out = {}
    for r in n_lines.itertuples():
        out[round(r.wl_air, 4)] = dict(species="N I", wl=r.wl_air, role="actinometry line" if r.wl_air in set(act.wl_air)
                                       else "selected N I line, not used (no branching ratio)")
    for a in act.itertuples():
        for r in cat[(cat.species == "N I") & (cat.upper == a.upper)].itertuples():
            out.setdefault(round(r.wl_air, 4), dict(species="N I", wl=r.wl_air,
                                                    role=f"same upper level as N I {a.wl_air:.2f}"))
    up = cat.upper[(cat.species == "Ar I") & np.isclose(cat.wl_air, ZOOM_AR_WL, atol=1e-3)]
    out[round(ZOOM_AR_WL, 4)] = dict(species="Ar I", wl=ZOOM_AR_WL,
                                     role=f"Ar I reference{', ' + up.iloc[0] if len(up) else ''}")
    return sorted(out.values(), key=lambda d: (d["species"] != "N I", d["wl"])), cat


def plot_raw_lines(ds, cond, outdir):
    """Raw spectra around each zoom line (zoom_lines): every calibrated spectrum of the set (thin) and the mean of
    the repeats (thick), coloured by the sweep value; legend: spectra in which the line fit detected the line."""
    lines, cat = zoom_lines()
    sw, unit = ds["sweep"], SWEPT[ds["kind"]]["unit"]
    os.makedirs(outdir, exist_ok=True)
    fl = als.export_flags(dict(als.CONFIG, sweeps=[sw]))
    cal = set(fl.file[[fnmatch(f, als.CONFIG["export_flags"]) for f in fl.export_flags]])
    spectra = []                                        # (x, name, wl, I) with the echelle gaps as NaN
    for f in sorted(glob(os.path.join(sw["folder"], "*.spa"))):
        name = os.path.splitext(os.path.basename(f))[0]
        px = sw["parse"](name)
        if px is None or name not in cal or not np.any(np.isclose(px[0], cond.x)):
            continue
        wl, I = als.arn.read_spa(f)
        I = np.where(als.arn.find_gaps(I, als.arn.CONFIG["gap_min_run"])[0], np.nan, np.asarray(I, float))
        spectra.append((px[0], name, np.asarray(wl, float), I))
    status = {}                                         # (file, wl) -> line-fit status, from the cached line fits
    for cache in ("ar_line_measurements.csv", "actinometry_line_measurements.csv"):
        path = os.path.join(ds["measure_dir"], cache)
        if os.path.exists(path):
            m = pd.read_csv(path)
            status.update(zip(zip(m.file.astype(str), m.wl_air.round(4)), m.status))
    xs = sorted({s[0] for s in spectra})
    cols = plt.cm.viridis(np.linspace(0, 0.92, len(xs)))
    for ln in lines:
        w0, key = ln["wl"], float(np.round(ln["wl"], 4))
        fig, ax = plt.subplots(figsize=(9, 5.5))
        near, low = [], []
        for col, x in zip(cols, xs):
            grp = [(n, W[m], I[m]) for xx, n, W, I in spectra if xx == x for m in [np.abs(W - w0) <= ZOOM_HALF_NM]]
            for _, W, I in grp:
                ax.plot(W, I, "-", color=col, lw=0.6, alpha=0.35)
                near.append(np.nanmax(I[np.abs(W - w0) <= 0.06], initial=np.nan))
                low.append(np.nanpercentile(I, 1) if np.isfinite(I).any() else np.nan)
            W0 = grp[0][1]
            ax.plot(W0, np.nanmean([np.interp(W0, W, I) for _, W, I in grp], axis=0), "-", color=col, lw=2)
            st = [status.get((n, key)) for n, _, _ in grp]
            det = f": detected in {sum(s in als.CONFIG['sweep_status'] for s in st)}/{len(st)}" \
                if all(s is not None for s in st) else f" ({len(st)} spectra)"
            ax.plot([], [], "-", color=col, lw=2, label=f"{x:g} {unit}{det}")
        for r in cat[(cat.wl_air - w0).abs() <= ZOOM_HALF_NM].itertuples():
            ax.axvline(r.wl_air, color="k" if np.isclose(r.wl_air, w0, atol=1e-3) else "0.55",
                       ls="-" if np.isclose(r.wl_air, w0, atol=1e-3) else ":", lw=0.8, zorder=0)
            ax.text(r.wl_air, 0.99, f" {r.species} {r.wl_air:.2f}", transform=ax.get_xaxis_transform(), rotation=90,
                    va="top", ha="right", fontsize=7, color="0.3")
        near, low = (np.asarray(v, float) for v in (near, low))
        near, low = near[np.isfinite(near)], low[np.isfinite(low)]
        top, bot = (near.max() if len(near) else np.nan), (low.min() if len(low) else np.nan)
        if np.isfinite(top) and np.isfinite(bot) and top > bot:
            ax.set_ylim(bot - 0.05 * (top - bot), top + 0.3 * (top - bot))   # neighbours may be clipped
        else:
            ax.text(0.5, 0.5, "no data (echelle gap?)", transform=ax.transAxes, ha="center")
        ax.set_xlim(w0 - ZOOM_HALF_NM, w0 + ZOOM_HALF_NM)
        ax.set_xlabel("wavelength [nm] (as exported)")
        ax.set_ylabel("intensity, calibrated export [a.u.]")
        ax.set_title(f"{ln['species']} {w0:.2f} nm ({ln['role']}): raw spectra, {ds['name']} ({ds['note']})\n"
                     "thin: every spectrum; thick: mean of the repeats; vertical lines: catalogue lines", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2 if len(xs) > 10 else 1, loc="upper left")
        fig.tight_layout()
        _savefig(fig, os.path.join(outdir, f"raw_{ln['species'].replace(' ', '')}_{w0:.2f}nm.png"))


def plot_overview(kind, sets, res, outdir):
    """Data sets of one sweep type together: CR and LAS 1s5, and (with metastables) n_N/n_Ar and dissociation."""
    sets = [d for d in sets if d["kind"] == kind and d["name"] in res]
    if len(sets) < 2:
        return
    cols = dict(zip((d["name"] for d in sets), plt.cm.tab10(np.arange(len(sets)) % 10)))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for d in sets:
        t, c = res[d["name"]][0].sort_values("x"), cols[d["name"]]
        ax.fill_between(t.x, t.n_1s5_lo, t.n_1s5_hi, color=c, alpha=0.12, lw=0)
        ax.plot(t.x, t.n_1s5, "s-", color=c, ms=6, lw=2, label=f"CR, {d['name']}")
        ax.plot(t.x, t.las_1s5, "D--", color=c, mfc="white", ms=6, lw=1, label=f"LAS, {d['name']}")
    ax.set_ylim(bottom=0)
    ax.set_xlabel(SWEPT[kind]["xlabel"])
    ax.set_ylabel("Ar(1s$_5$) density [m$^{-3}$]")
    ax.set_title(f"Ar(1s$_5$): CR fit ({eedf_label()}{_ne()}, no tilt) and LAS\n"
                 f"band: $\\Delta\\chi^2 \\leq 1\\,s^2$ {_band()}; open diamonds: LAS (measured or interpolated)",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _savefig(fig, os.path.join(outdir, f"overview_{kind}_1s5.png"))
    for col, ylab, fname in FIT_PARAMS:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for d in sets:
            t = res[d["name"]][0].sort_values("x")
            y = t[col]
            ax.errorbar(t.x, y, yerr=[y - t[f"{col}_lo"], t[f"{col}_hi"] - y], fmt="s-", color=cols[d["name"]], ms=6,
                        capsize=3, label=d["name"])
        ax.set_ylim(bottom=0)
        ax.set_xlabel(SWEPT[kind]["xlabel"])
        ax.set_ylabel(ylab)
        ax.set_title(f"CR fit{_ne(' at ')} ({eedf_label()}, no tilt)\n"
                     f"bars: $\\Delta\\chi^2 \\leq 1\\,s^2$ {_band()}", fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        _savefig(fig, os.path.join(outdir, f"overview_{kind}_{fname}.png"))
    acts = {d["name"]: res[d["name"]][1] for d in sets if len(res[d["name"]][1])}
    if not acts:
        return
    for key, ylab, fname in (("nN_nAr", "n$_N$ / n$_{Ar}$", "actinometry"),
                             ("dissociation", "dissociation degree n$_N$ / (2 n$_{N_2}$)  (feed N$_2$)",
                              "dissociation")):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for name, a in acts.items():
            g = _combined(a)
            g = g[g.variant == "cr"].sort_values("x")
            y = g[key].to_numpy()
            ax.errorbar(g.x, y, yerr=_yerr(y, g.ln_err_stat.to_numpy()), fmt="s-", color=cols[name], ms=6, capsize=3,
                        label=name)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(SWEPT[kind]["xlabel"])
        ax.set_ylabel(ylab)
        ax.set_title(f"{_n_label(next(iter(acts.values())))} / Ar I actinometry, {VARIANTS['cr']['label']}\n"
                     f"{eedf_label()}{_ne()}, no tilt (bars: line ratios + E/N posterior + pair scatter)",
                     fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        _savefig(fig, os.path.join(outdir, f"overview_{kind}_{fname}.png"))


# ---- driver --------------------------------------------------------------------------------
def run_dataset(ds, cond):
    """CR fits, metastables and actinometry of one data set -> (fits table, actinometry per condition)."""
    figdir = os.path.join(ds["outdir"], "figures")
    os.makedirs(figdir, exist_ok=True)
    print(f"\n=== {ds['name']} ({_plain(ds['note'])}): CR fits at n_c ({_plain(eedf_label())}, analytic 5p, no tilt)",
          flush=True)
    t, fits = fit_conditions(ds, cond)
    if t.empty:
        return t, pd.DataFrame()
    t.to_csv(os.path.join(ds["outdir"], "fits.csv"), index=False)
    pd.concat([f["resid"].assign(x=x) for x, f in fits.items()]).to_csv(os.path.join(ds["outdir"], "residuals.csv"),
                                                                         index=False)
    if PLOTS["metastables"]:
        plot_metastables(t, ds, figdir)
    if PLOTS["fit_params"]:
        plot_fit_params(t, ds, figdir)
    if PLOTS["eedf"]:
        plot_eedfs(t, fits, ds, figdir)
    if PLOTS["residuals"]:
        plot_residuals(fits, ds, figdir)
    if PLOTS["fit_overlay"]:
        plot_fit_overlay(fits, ds, os.path.join(figdir, "fit_overlay"))
    if PLOTS["chi2_maps"]:
        plot_chi2_maps(fits, ds, os.path.join(figdir, "chi2_maps"))
    act = pd.DataFrame()
    if not (cond.N2 > 0).any():
        print("pure Ar: no actinometry")
    else:
        act = actinometry_outputs(ds, cond, t, fits, figdir)
    if PLOTS["raw_lines"]:
        with warnings.catch_warnings():                 # all-NaN slices in the echelle gaps
            warnings.simplefilter("ignore", RuntimeWarning)
            plot_raw_lines(ds, cond, os.path.join(figdir, "raw_lines"))
    return t, act


def actinometry_outputs(ds, cond, t, fits, figdir):
    """Actinometry of one data set: tables, plots and printout -> actinometry per condition."""
    pairs, act, ratios, same = actinometry(ds, cond, fits)
    if len(ratios):
        ratios.to_csv(os.path.join(ds["outdir"], "line_ratios.csv"), index=False)
        if PLOTS["line_ratios"]:
            plot_ratios(ratios, ds, figdir)
    if len(same):
        same.to_csv(os.path.join(ds["outdir"], "NI_same_upper_ratios.csv"), index=False)
        if PLOTS["same_upper_ratios"]:
            plot_same_upper(same, ds, figdir)
        print("\nN I lines from the same upper level, measured / expected ratio:")
        print(same.assign(meas_over_expected=same.ratio / same.expected)
              .pivot_table(index=["wl_b", "wl_a"], columns="x", values="meas_over_expected")
              .to_string(float_format=lambda v: f"{v:.2f}"))
    if act.empty:
        print("no N I / Ar I pair with a valid result: no actinometry")
        return act
    if gap := sorted(set(t.x[t.N2 > 0]) - set(act.x)):
        print(f"no actinometry at {', '.join(f'{x:g}' for x in gap)} {SWEPT[ds['kind']]['unit']}: no N I line detected "
              f"in {als.CONFIG['min_reps']} spectra (or no valid pair)")
    pairs.to_csv(os.path.join(ds["outdir"], "actinometry_per_pair.csv"), index=False)
    act.to_csv(os.path.join(ds["outdir"], "actinometry_per_condition.csv"), index=False)
    if PLOTS["actinometry"]:
        plot_actinometry(act, ds, figdir)
        plot_by_n_line(act, ds, figdir)
    if PLOTS["actinometry_per_line"]:
        plot_per_line(pairs, act, ds, figdir, "cr")
        plot_per_line(pairs, act, ds, figdir, "direct")
    pd.set_option("display.width", 250)
    print("\nactinometry per condition (all pairs):")
    print(_combined(act)[["x", "variant", "n_pairs", "nN_nAr", "ln_err_stat", "birge_pairs", "dissociation", "n_N"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nn_N/n_Ar with metastables per N I line:")
    print(act[act.variant == "cr"].pivot_table(index="N_line", columns="x", values="nN_nAr")
          .to_string(float_format=lambda v: f"{v:.3g}"))
    for v in VARIANTS:
        print(f"\nper pair ({v}), n_N/n_Ar:")
        print(pairs[pairs.variant == v].pivot_table(index=["n_wl", "ar_wl", "ar_upper"], columns="x", values="nN_nAr")
              .to_string(float_format=lambda v: f"{v:.3g}"))
    print("\nf_direct of the Ar reference levels:")
    print(pairs[pairs.variant == "cr"].pivot_table(index=["ar_wl", "ar_upper"], columns="x", values="f_direct_Ar")
          .to_string(float_format=lambda v: f"{v:.3g}"))
    print("\noutliers:", pairs[pairs.outlier][["x", "variant", "n_wl", "ar_wl"]].to_string(index=False))
    return act


def main(input_file=INPUT_FILE, plan_only=False):
    sets = load_input(input_file)
    if not sets:
        print(f"{input_file}: no data set with run = true")
        return {}
    las = L.load_las()
    trend = L.fit_tg_trend(las)[0]
    conds = {}
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        for d in sets:
            c = conditions(d, las, trend)
            print(f"\n{d['name']} ({_plain(d['note'])}): {c.n_files.sum() if len(c) else 0} spectra, {len(c)} conditions "
                  f"- {d['sweep']['folder']}")
            if c.empty:
                print("  no calibrated spectrum matches the file pattern - skipped")
                continue
            print(c[["x", "p_mTorr", "P_W", "N2", "n_files", "Tg", "las_1s5", "las_1s5_how"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
            conds[d["name"]] = c
    if not conds:
        return {}
    libs, tabs = to_build(sorted(set(pd.concat(conds.values()).key)))
    print(f"\nto build: {len(libs)} BOLSIG+ libraries ({_plain(eedf_label())}), {len(tabs)} CR tables "
          f"(AnalyticFivePFit, ~2-3 min each, {A.N_WORKERS} at a time)")
    if plan_only:
        for what, keys in (("libraries", libs), ("CR tables", tabs)):
            if keys:
                print(f"  {what} (% N2:p mTorr:Tg K): {', '.join(k if isinstance(k, str) else k[1] for k in keys)}")
        return conds
    if EEDF == "vibsat":
        L.vibsat_xsec()                             # written once here, not by the parallel library workers
    elif EEDF.startswith("tv"):
        L.ladder_xsec()
    for flag in sorted({f for f, _ in libs}):
        L._run_workers(flag, [k for f, k in libs if f == flag], f"BOLSIG+ libraries ({flag[2:]})")
    A.build_all([(k, EEDF) for k in tabs])
    res = {}
    for d in sets:
        if d["name"] in conds:
            res[d["name"]] = run_dataset(d, conds[d["name"]])
            if len(conds) > 1:
                plt.close("all")                    # per-set figures are in <set>/figures; the overviews stay open
    res = {n: r for n, r in res.items() if len(r[0])}
    for i, fname in enumerate(("fits_all.csv", "actinometry_per_condition_all.csv")):
        frames = {n: r[i] for n, r in res.items() if len(r[i])}
        if frames:
            pd.concat(frames, names=["dataset"]).reset_index(level=0).to_csv(os.path.join(OUTDIR, fname), index=False)
    if PLOTS["overviews"]:
        for kind in SWEPT:
            plot_overview(kind, sets, res, OUTDIR)
    print(f"\noutput in {OUTDIR}")
    plt.show()
    return res


if __name__ == "__main__":
    ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
    RES = main(ARGS[0] if ARGS else INPUT_FILE, plan_only="--plan" in sys.argv)
