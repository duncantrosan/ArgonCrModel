"""N2(C-B) second positive system: rotational and vibrational temperatures of the echelle data sets.

Data sets from N2SweepPureArEEDF_input.toml (same [[dataset]] blocks, run = false skipped, spectra without the
intensity calibration dropped, ref*/ref_* spectra of the folder averaged and subtracted as background).
Fitter: MolecularFitting/SingleSpectraFit.py + Fit_One_Temp_V2.MolecularFitter, calibrated 14N2 library
(T_rot x T_vib grid, echelle slit 09_04_2026).

Per spectrum
1. T_vib (+ T_rot) from the Delta v = +1 (310-316 nm), -1 (346-358 nm) and -2 (366-381 nm) sequences.
2. T_rot of every single band (v' = 0, 1, 2), T_vib fixed at that spectrum's Delta v = -2 value.
Every fit is sigma-clipped (points > 3.5 robust sigma off the fit masked, refitted): atomic lines and the
stitching dropouts of the echelle spectra.

Output: Experimental_Data/Output/N2RotVibTemperatures/<name>/
    fit_summary.csv            every spectrum x band
    temperatures.csv           median / spread of the repeats per sweep value
    T_vib.png, T_rot.png       sweep graphs (median of the repeats, error bar = min-max of the repeats)
    representative_bands.png   single-band fits of the representative spectrum (one panel per band)
    representative_sequences.png  Delta v = -1 / -2 fits of the same spectrum
    representative_system.png  measured 310-382 nm vs simulation at the fitted T_rot / T_vib
    fits/                      single fit plots of the representative spectrum
plus overview_<sweep>.png (T_vib and T_rot of all data sets of one sweep type).

    python SmallAnalysisScripts/N2RotVibTemperatures.py [--workers N] [--only NAME]
"""
import argparse
import glob
import os
import re
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "MolecularFitting"))
from SingleSpectraFit import Band, read_spa, plot_fit, _sanitize  # noqa: E402
from Fit_One_Temp_V2 import MolecularFitter  # noqa: E402

INPUT_FILE = os.path.join(HERE, "N2SweepPureArEEDF_input.toml")
ECHELLE_DIR = os.path.join(ROOT, "Experimental_Data", "EchelleData")
OUTDIR = os.path.join(ROOT, "Experimental_Data", "Output", "N2RotVibTemperatures")
KEY = "14N2"

SWEPT = {"N2": ("N2_percent", "N$_2$ in the feed [%]"), "pressure": ("pressure_mTorr", "Pressure [mTorr]"),
         "power": ("power_W", "Power [W]")}

# sequences for T_vib (several v' in the window)
SEQUENCES = [Band(KEY, (310.2, 316.2), label="dv+1 sequence", fit_tvib=True),
             Band(KEY, (346.0, 358.0), label="dv-1 sequence", fit_tvib=True),
             Band(KEY, (366.0, 380.8), label="dv-2 sequence", fit_tvib=True)]
TVIB_SEQ = "dv-2 sequence"          # T_vib used for the single-band T_rot fits
# single bands: (v', v'') -> band head [nm, air]; window head - width .. head + 0.25 (bands shade to violet).
# 1-1, 2-2 (in the 0-0 tail) and 2-3 (atomic lines of the dv-1 sequence) are too weak to fit on their own.
HEADS = {(0, 0): 337.13, (0, 1): 357.69, (0, 2): 380.49, (1, 0): 315.93, (1, 2): 353.67, (1, 3): 375.54,
         (2, 1): 313.60, (2, 4): 371.05}
WIDTH = {(1, 0): 1.9, (2, 1): 1.6, (2, 4): 1.8}
BANDS = [Band(KEY, (h - WIDTH.get(vv, 2.6), h + 0.25), label=f"{vv[0]}-{vv[1]}") for vv, h in HEADS.items()]
CLIP_SIGMA, CLIP_ITER = 3.5, 3       # atomic lines / stitching dropouts: points this far off the fit are masked
HEAD_PROTECT = 0.25                  # ... except where the model is above this fraction of its peak (band heads)
VCOL = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c"}
SYSTEM_RANGE = (310.0, 382.0)


def calibrated(path):
    with open(path) as f:
        tok = f.readline().split()
    flags = tok[1].split("-") if len(tok) > 1 else []
    return len(flags) > 2 and flags[2] == "1"


def file_parser(patterns):
    regs = [re.compile(re.escape(p).replace(re.escape("{x}"), r"(?P<x>\d+(?:[_.]\d+)?)")
                       .replace(re.escape("{rep}"), r"(?P<rep>\d+)"))
            for p in ([patterns] if isinstance(patterns, str) else patterns)]

    def parse(name):
        for r in regs:
            m = r.fullmatch(name)
            if m:
                return float(m["x"].replace("_", ".")), int(m.groupdict().get("rep") or 1)
        return None
    return parse


def load_sets():
    with open(INPUT_FILE, "rb") as f:
        entries = tomllib.load(f).get("dataset", [])
    sets = []
    for d in entries:
        if not d.get("run", True):
            continue
        folder = d["folder"] if os.path.isabs(d["folder"]) else os.path.join(ECHELLE_DIR, d["folder"])
        parse, excl = file_parser(d["files"]), np.asarray(d.get("exclude_x", []), float)
        spectra, dropped = [], []
        for p in sorted(glob.glob(os.path.join(folder, "*.spa"))):
            xr = parse(os.path.splitext(os.path.basename(p))[0])
            if xr is None or np.any(np.isclose(xr[0], excl)):
                continue
            if not calibrated(p):
                dropped.append(os.path.basename(p))
                continue
            spectra.append({"file": p, "x": xr[0], "rep": xr[1]})
        refs = [p for p in glob.glob(os.path.join(folder, "ref*.spa")) if calibrated(p)]
        key, xlabel = SWEPT[d["sweep"]]
        sets.append(dict(name=d["name"], kind=d["sweep"], xlabel=xlabel, spectra=spectra, refs=refs,
                         dropped=dropped, outdir=os.path.join(OUTDIR, d["name"])))
    return sets


_FITTER = None
LIB_NPZ = os.path.join(ROOT, "MolecularFitting", "Calibrated_Library", "09_04_2026", KEY, f"{KEY}_calibrated_library.npz")
LIB_CACHE = os.path.join(OUTDIR, "_library_cache")
LIB_RANGE = (309.0, 382.0)          # the library is cropped to this range (float32, memory-mapped by every worker)


def build_library_cache():
    """Crop the 14N2 library (T_vib x T_rot x wl, 3.9 GB as float64) to LIB_RANGE once, as float32 .npy files
    that the worker processes memory-map (shared, instead of one full copy per process)."""
    os.makedirs(LIB_CACHE, exist_ok=True)
    out = os.path.join(LIB_CACHE, "intensity.npy")
    if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(LIB_NPZ):
        return
    print("cropping the 14N2 library ...", flush=True)
    with np.load(LIB_NPZ) as d:
        wl = d["wavelength_nm"]
        m = (wl >= LIB_RANGE[0]) & (wl <= LIB_RANGE[1])
        for k in ("wavelength_nm", "T_rot", "T_vib"):
            np.save(os.path.join(LIB_CACHE, f"{k}.npy"), d[k][m] if k == "wavelength_nm" else d[k])
        np.save(out + ".tmp.npy", d["convolved_rot_matrix"][:, :, m].astype(np.float32))
    os.replace(out + ".tmp.npy", out)


def cropped_fitter():
    """MolecularFitter (its _interpolate_spectrum) on the memory-mapped cropped 14N2 library only."""
    f = MolecularFitter.__new__(MolecularFitter)
    ld = lambda k: np.load(os.path.join(LIB_CACHE, f"{k}.npy"))
    f.loaded_libraries = {KEY: dict(wl_grid=ld("wavelength_nm").astype(float), T_rot_grid=ld("T_rot").astype(float),
                                    T_vib_grid=ld("T_vib").astype(float), is_3d=True,
                                    intensity=np.load(os.path.join(LIB_CACHE, "intensity.npy"), mmap_mode="r"))}
    return f


def _shape(fitter, wl, T_rot, T_vib, shift):
    sh = fitter._interpolate_spectrum(wl + shift, T_rot, T_vib, KEY)
    return sh / sh.max() if sh.max() > 0 else sh


def _linear(sh, wl, y):
    A = np.column_stack([sh, wl - wl.mean(), np.ones_like(sh)])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c, A @ c


def _model(fitter, wl, f):
    sh = _shape(fitter, wl, f["T_rot"], f["T_vib_used"], f["shift_nm"])
    return f["amplitude"] * sh + f["baseline_slope"] * (wl - f["wl_mean"]) + f["baseline_offset"]


def _de_fit(fitter, wl, y, fit_tvib, tvib):
    """T_rot (+ T_vib) and shift by differential evolution with the library interpolation of MolecularFitter, amplitude
    and linear baseline by linear least squares, UNWEIGHTED residuals.  (MolecularFitter.fit_spectrum weights by the
    model shape itself, so a model with no intensity where the data is and amplitude 0 can win.)"""
    lib = fitter.loaded_libraries[KEY]
    bounds = [tuple(lib["T_rot_grid"][[0, -1]])] + ([tuple(lib["T_vib_grid"][[0, -1]])] if fit_tvib else [])         + [(-0.2, 0.2)]

    def unpack(p):
        return (p[0], p[1], p[2]) if fit_tvib else (p[0], tvib, p[1])

    def obj(p):
        tr, tv, sh = unpack(p)
        _, m = _linear(_shape(fitter, wl, tr, tv, sh), wl, y)
        return np.mean((y - m) ** 2)

    res = differential_evolution(obj, bounds, maxiter=150, popsize=20, tol=1e-6, seed=42, polish=True)
    tr, tv, sh = unpack(res.x)
    c, _ = _linear(_shape(fitter, wl, tr, tv, sh), wl, y)
    return dict(T_rot=float(tr), T_vib=float(tv) if fit_tvib else np.nan, T_vib_used=float(tv), shift_nm=float(sh),
                amplitude=float(c[0]), baseline_slope=float(c[1]), baseline_offset=float(c[2]),
                wl_mean=float(wl.mean()), rmse=float(np.sqrt(res.fun)))


def fit_band(fitter, wl, I, band, tvib):
    """Fit one window with sigma clipping: crop, 5th-percentile floor, normalise to the peak, fit, mask the points
    more than CLIP_SIGMA robust sigmas off the fit (atomic lines, dropouts; band heads protected), refit on the rest."""
    m = (wl >= band.roi[0]) & (wl <= band.roi[1])
    wl_c, y = wl[m], I[m] - np.percentile(I[m], 5)
    y = y / y.max()
    keep = np.ones(len(y), bool)
    for _ in range(CLIP_ITER):
        f = _de_fit(fitter, wl_c[keep], y[keep], band.fit_tvib, tvib)
        model = _model(fitter, wl_c, f)
        r = y - model
        sig = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        new = (np.abs(r) < CLIP_SIGMA * sig) | (model > HEAD_PROTECT * model.max())
        if np.array_equal(new, keep):
            break
        keep = new
    yk, mk = y[keep], model[keep]
    r2 = 1 - np.sum((yk - mk) ** 2) / np.sum((yk - yk.mean()) ** 2)
    lo, hi = fitter.loaded_libraries[KEY]["T_rot_grid"][[0, -1]]
    reasons = [s for s, bad in [(f"Low R2 ({r2:.2f})", r2 < 0.5),
                                ("T_rot at grid edge", not lo + 0.02 * (hi - lo) < f["T_rot"] < hi - 0.02 * (hi - lo)),
                                ("Shift at bound", abs(f["shift_nm"]) > 0.196), ("no band (amplitude <= 0)", f["amplitude"] <= 0)] if bad]
    return dict(f, wl_crop=wl_c, I_exp_norm=y, I_fit_norm=model, keep=keep, r2=r2, band=band.label,
                status="VALID" if not reasons else "FIT_REJECTED", reason="; ".join(reasons),
                clipped_frac=1 - keep.mean())


def _fit_one(args):
    """Fit one spectrum: sequences first, then the single bands at that spectrum's dv-2 T_vib."""
    entry, refs = args
    global _FITTER
    if _FITTER is None:
        _FITTER = cropped_fitter()
    wl, I = read_spa(entry["file"])
    if refs:
        I = I - np.mean([np.interp(wl, *read_spa(r)) for r in refs], axis=0)
    fits = [fit_band(_FITTER, wl, I, b, 4500.0) for b in SEQUENCES]
    tv = next(f for f in fits if f["band"] == TVIB_SEQ)["T_vib"]
    fits += [fit_band(_FITTER, wl, I, b, tv) for b in BANDS]
    rows = []
    for f in fits:
        f.update(label=os.path.basename(entry["file"]), file=entry["file"])
        rows.append(dict(file=entry["file"], x=entry["x"], rep=entry["rep"], band=f["band"], status=f["status"],
                         reason=f["reason"], T_rot=f["T_rot"], T_vib=f["T_vib"], r2=f["r2"],
                         shift_nm=f["shift_nm"], clipped_frac=f["clipped_frac"], rmse=f["rmse"]))
    return pd.DataFrame(rows), fits


def representative(df):
    """Largest sweep value (most N2 / highest p), the repeat whose 0-0 T_rot is closest to the median of the repeats."""
    x = df.x.max() if df.x.max() > 0 else df.x.min()
    g = df[(df.x == x) & (df.band == "0-0") & (df.status == "VALID")]
    if g.empty:
        return df[df.x == x].rep.min(), x
    return int(g.rep.iloc[np.argmin(np.abs(g.T_rot - g.T_rot.median()))]), x


def agg(df):
    ok = df[df.status == "VALID"]
    rows = []
    for (band, x), g in ok.groupby(["band", "x"]):
        for col in ("T_rot", "T_vib"):
            v = g[col].dropna()
            if len(v):
                rows.append(dict(band=band, x=x, quantity=col, median=v.median(), lo=v.min(), hi=v.max(),
                                 std=v.std(ddof=1) if len(v) > 1 else np.nan, n=len(v)))
    return pd.DataFrame(rows)


def _err(t):
    return np.vstack([t["median"] - t.lo, t.hi - t["median"]])


def plot_sweeps(t, ds):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for lab, c in zip([s.label for s in SEQUENCES], ["#8c564b", "#9467bd", "#ff7f0e"]):
        g = t[(t.band == lab) & (t.quantity == "T_vib")].sort_values("x")
        ax.errorbar(g.x, g["median"], yerr=_err(g), fmt="o-", color=c, capsize=3, label=f"T$_{{vib}}$, {lab}")
    ax.set(xlabel=ds["xlabel"], ylabel="T$_{vib}$ of N$_2$(C) [K]", title=f"N$_2$(C-B) vibrational temperature, {ds['name']}")
    ax.grid(ls=":", alpha=0.6); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(ds["outdir"], "T_vib.png"), dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for vp, ax in zip((0, 1, 2), axs):
        labs = [f"{a}-{b}" for (a, b) in HEADS if a == vp]
        for k, lab in enumerate(labs):
            g = t[(t.band == lab) & (t.quantity == "T_rot")].sort_values("x")
            if len(g):
                ax.errorbar(g.x, g["median"], yerr=_err(g), fmt="os^Dv"[k] + "-", capsize=3, ms=5,
                            color=plt.cm.tab10(k), label=f"({lab})")
        for lab, c in zip([s.label for s in SEQUENCES], ["#8c564b", "#9467bd", "#ff7f0e"]):
            g = t[(t.band == lab) & (t.quantity == "T_rot")].sort_values("x")
            ax.plot(g.x, g["median"], "--", color=c, lw=1, label=f"{lab}" if vp == 0 else None)
        ax.set(xlabel=ds["xlabel"], title=f"v' = {vp}")
        ax.grid(ls=":", alpha=0.6); ax.legend(fontsize=8)
    axs[0].set_ylabel("T$_{rot}$ of N$_2$(C) [K]")
    fig.suptitle(f"N$_2$(C-B) rotational temperature per band, {ds['name']}")
    fig.tight_layout(); fig.savefig(os.path.join(ds["outdir"], "T_rot.png"), dpi=150); plt.close(fig)


def _panel(ax, f):
    ax.plot(f["wl_crop"], f["I_exp_norm"], color="0.3", lw=1, label="measured")
    ax.plot(f["wl_crop"], f["I_fit_norm"], color="#d62728", lw=1, ls="-" if f["status"] == "VALID" else ":",
            label="fit")
    if "keep" in f and (~f["keep"]).any():
        ax.plot(f["wl_crop"][~f["keep"]], f["I_exp_norm"][~f["keep"]], "x", color="#1f77b4", ms=3, label="clipped")
    tv = f", T$_{{vib}}$={f['T_vib']:.0f} K" if np.isfinite(f.get("T_vib", np.nan)) else ""
    ax.set_title(f"{f['band']}: T$_{{rot}}$={f['T_rot']:.0f} K{tv}, R²={f['r2']:.3f}"
                 + ("" if f["status"] == "VALID" else " (rejected)"), fontsize=9)
    ax.grid(ls=":", alpha=0.5)


def plot_representative(fits, ds, x, rep, refs):
    out = ds["outdir"]
    tag = f"{ds['name']}, x = {x:g}, repeat {rep}"
    fb = [f for f in fits if f["band"] in {b.label for b in BANDS}]
    fb.sort(key=lambda f: (int(f["band"][0]), int(f["band"][2])))
    fig, axs = plt.subplots(3, 4, figsize=(17, 10))
    for ax, f in zip(axs.flat, fb):
        _panel(ax, f)
    for ax in axs.flat[len(fb):]:
        ax.axis("off")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"N$_2$(C-B) single-band T$_{{rot}}$ fits, {tag}")
    fig.supxlabel("Wavelength [nm]"); fig.supylabel("Normalised intensity")
    fig.tight_layout(); fig.savefig(os.path.join(out, "representative_bands.png"), dpi=140); plt.close(fig)

    fs = [f for f in fits if f["band"] in {s.label for s in SEQUENCES}]
    fig, axs = plt.subplots(len(fs), 1, figsize=(11, 4 * len(fs)))
    for ax, f in zip(np.atleast_1d(axs), fs):
        _panel(ax, f); ax.legend(fontsize=8)
    fig.suptitle(f"N$_2$(C-B) sequence fits (T$_{{rot}}$, T$_{{vib}}$), {tag}")
    fig.supxlabel("Wavelength [nm]")
    fig.tight_layout(); fig.savefig(os.path.join(out, "representative_sequences.png"), dpi=140); plt.close(fig)
    for f in fits:
        plot_fit(f, os.path.join(out, "fits", f"{_sanitize(f['band'])}.png"))

    # measured over the whole system vs simulation at the dv-2 T_rot / T_vib
    f = next(f for f in fs if f["band"] == TVIB_SEQ)
    wl, I = read_spa(f["file"])
    if refs:
        I = I - np.mean([np.interp(wl, *read_spa(r)) for r in refs], axis=0)
    m = (wl >= SYSTEM_RANGE[0]) & (wl <= SYSTEM_RANGE[1])
    wl, I = wl[m], I[m]
    sim = cropped_fitter()._interpolate_spectrum(wl + f["shift_nm"], f["T_rot"], f["T_vib"], KEY)
    A = np.column_stack([sim, np.ones_like(sim)])
    w = np.sqrt(sim / sim.max() + 0.01)
    (a, b), *_ = np.linalg.lstsq(A * w[:, None], I * w, rcond=None)
    I, simf = (I - b) / (a * sim.max()), sim / sim.max()
    fig, ax = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax[0].plot(wl, I, color="0.3", lw=0.8, label="measured (background subtracted)")
    ax[0].plot(wl, simf, color="#d62728", lw=0.8,
               label=f"simulation: T$_{{rot}}$ = {f['T_rot']:.0f} K, T$_{{vib}}$ = {f['T_vib']:.0f} K (dv-2 fit)")
    for (vu, vl), h in HEADS.items():
        ax[0].annotate(f"{vu}-{vl}", (h, 1.02), ha="center", fontsize=8, color=VCOL[vu])
    ax[0].set(ylabel="Normalised intensity", ylim=(-0.05, 1.12),
              title=f"N$_2$(C-B) second positive system, measured vs simulated, {tag}")
    ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.42, 0.97)); ax[0].grid(ls=":", alpha=0.5)
    ax[1].plot(wl, I - simf, color="#1f77b4", lw=0.7); ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set(xlabel="Wavelength [nm]", ylabel="Residual"); ax[1].grid(ls=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, "representative_system.png"), dpi=150); plt.close(fig)


def plot_overview(tabs, sets):
    for kind in {d["kind"] for d in sets}:
        ds_k = [d for d in sets if d["kind"] == kind and d["name"] in tabs]
        if not ds_k:
            continue
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
        for k, d in enumerate(ds_k):
            t, c = tabs[d["name"]], plt.cm.tab10(k)
            g = t[(t.band == TVIB_SEQ) & (t.quantity == "T_vib")].sort_values("x")
            axs[0].errorbar(g.x, g["median"], yerr=_err(g), fmt="o-", color=c, capsize=3, label=d["name"])
            g = t[(t.band == "0-0") & (t.quantity == "T_rot")].sort_values("x")
            axs[1].errorbar(g.x, g["median"], yerr=_err(g), fmt="o-", color=c, capsize=3, label=f"{d['name']} (0-0)")
            g = t[(t.band == "0-2") & (t.quantity == "T_rot")].sort_values("x")
            axs[1].plot(g.x, g["median"], "s--", color=c, ms=4, label=f"{d['name']} (0-2)")
        axs[0].set(xlabel=ds_k[0]["xlabel"], ylabel="T$_{vib}$ [K]", title="T$_{vib}$ (dv-2 sequence)")
        axs[1].set(xlabel=ds_k[0]["xlabel"], ylabel="T$_{rot}$ [K]", title="T$_{rot}$ (0-0 and 0-2 bands)")
        for ax in axs:
            ax.grid(ls=":", alpha=0.6); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, f"overview_{kind}.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--only")
    a = ap.parse_args()
    sets = [d for d in load_sets() if a.only in (None, d["name"])]
    build_library_cache()
    tabs = {}
    for ds in sets:
        os.makedirs(ds["outdir"], exist_ok=True)
        print(f"\n== {ds['name']}: {len(ds['spectra'])} spectra, {len(ds['refs'])} background(s)"
              + (f", dropped (not calibrated): {ds['dropped']}" if ds["dropped"] else ""), flush=True)
        if not ds["spectra"]:
            continue
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(_fit_one, [(e, ds["refs"]) for e in ds["spectra"]]))
        df = pd.concat([r[0] for r in res], ignore_index=True)
        allfits = [f for r in res for f in r[1]]
        df.to_csv(os.path.join(ds["outdir"], "fit_summary.csv"), index=False)
        t = agg(df)
        t.to_csv(os.path.join(ds["outdir"], "temperatures.csv"), index=False)
        tabs[ds["name"]] = t
        print(df.groupby("band").status.value_counts().unstack(fill_value=0).to_string())
        plot_sweeps(t, ds)
        rep, x = representative(df)
        f_rep = next(e["file"] for e in ds["spectra"] if e["x"] == x and e["rep"] == rep)
        plot_representative([f for f in allfits if f["file"] == f_rep], ds, x, rep, ds["refs"])
    plot_overview(tabs, sets)
    print(f"\nResults in {OUTDIR}")


if __name__ == "__main__":
    main()
