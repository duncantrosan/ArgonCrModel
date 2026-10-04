

"""
fit_temperatures.py
===================

Batch rotational / vibrational temperature fitting of OES spectra with the
calibrated molecular library. A general version of
Fit1TemperatureDistributions_V2.py: the file collection is left to you, so it
works with any folder layout or naming scheme.

Place this file next to Fit_One_Temp_V2.py (the folder that contains
Calibrated_Library/).

Quick start
-----------
    from pathlib import Path
    from fit_temperatures import Band, fit_temperature_distributions

    bands = [
        Band("16O-1H", (304.0, 308.0), label="OH(A-X) (0-0)"),
        Band("14N2",   (335.0, 337.5), label="N2(C-B) (0-0)"),
        Band("14N2",   (330.0, 385.0), label="N2(C-B) system", fit_tvib=True),
    ]
    spectra = sorted(Path("my_data").glob("*.spa"))

    summary, fits = fit_temperature_distributions(
        spectra, bands,
        background="my_data/background/plasma_off.spa",   # optional
        out_dir="fit_results",
    )

Ways to pass `spectra`
----------------------
1. A list of paths                    [Path("a.spa"), Path("b.spa")]
2. A dict  {label: path}              {"8 kV": "a.spa", "10 kV": "b.spa"}
3. A list of dicts with metadata      [{"file": "a.spa", "voltage_kV": 8, "background": "bg1.spa"}, ...]
   Every extra key becomes a column in the summary table; an optional
   "background" key overrides the global background for that spectrum.
Optionally, `parse_name=lambda path: {...}` builds metadata columns from each
file name (e.g. with a regex), which is how the original script read flow rate
and gas mixture from its file names.

Background / reference spectra
------------------------------
`background` is subtracted from every spectrum before fitting (after
interpolation onto the spectrum's wavelength axis if the grids differ):
    None                      no subtraction
    path                      one background for all spectra
    [path, path, ...]         averaged, then used for all spectra
    {spectrum_path: path}     per-spectrum background (or give it per entry, form 3)
`background_scale` multiplies it first (e.g. an exposure-time ratio).

Optional relative spectral response
-----------------------------------
`response` = path to a two-column file (wavelength nm, relative sensitivity),
or a (wl, sens) tuple. The background-subtracted spectrum is divided by it.
This matters most for fits spanning a wide range (e.g. T_vib from several bands).

Outputs (in out_dir)
--------------------
    fit_summary.csv                  one row per spectrum x band, incl. skipped ones
    fits/<spectrum>__<band>.png      data, fit and residual for every attempted fit
    trends/<band>.png                T_rot (and T_vib) across spectra
Returned: (summary DataFrame, list of per-fit dicts holding the curves).

Deliberately left out: the "T_boltzmann" analysis of Boltzmann_Fitter.py (its
energies are photon-energy offsets between spectral peaks, not assigned
upper-state energies), and the 3x3 grids tied to the original experiment.
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from Fit_One_Temp_V2 import MolecularFitter  # noqa: E402

PathLike = Union[str, Path]


# =============================================================================
# Band definition
# =============================================================================
@dataclass
class Band:
    """One fit target.

    key          : library key = species folder name in Calibrated_Library/<slit_stem>/
                   (e.g. "16O-1H", "14N2", "OH(A-X)(0-0)"). Close matches are accepted.
    roi          : (min_nm, max_nm) fit window in measured wavelength.
    label        : display name used in tables, plots and file names (defaults to key).
    fit_tvib     : fit T_vib too. Only meaningful if the window spans several
                   vibrational bands of an ExoMol library.
    default_tvib : T_vib used when fit_tvib is False.
    max_shift_nm : per-band override of the allowed wavelength shift.
    """
    key: str
    roi: Tuple[float, float]
    label: Optional[str] = None
    fit_tvib: bool = False
    default_tvib: float = 4500.0
    max_shift_nm: Optional[float] = None

    def __post_init__(self):
        if self.label is None:
            self.label = self.key
        lo, hi = self.roi
        if not lo < hi:
            raise ValueError(f"Band {self.label}: roi must be (min, max), got {self.roi}")


# =============================================================================
# File reading
# =============================================================================
def read_spa(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """Two whitespace-separated columns (wavelength nm, counts) with one header line.
    Returns arrays sorted by wavelength. Pass your own reader to support other formats."""
    path = Path(path)
    try:
        df = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None, usecols=[0, 1], engine="c")
        wl, I = df[0].to_numpy(float), df[1].to_numpy(float)
    except Exception:
        data = np.genfromtxt(path, skip_header=1)
        wl, I = data[:, 0], data[:, 1]
    order = np.argsort(wl)
    return wl[order], I[order]


def _sanitize(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name).strip())
    return re.sub(r"_+", "_", cleaned).strip("_")


# =============================================================================
# Input normalisation
# =============================================================================
def _normalise_spectra(spectra, background, parse_name) -> List[Dict[str, Any]]:
    """Turn any accepted `spectra` form into a list of entries
    {file, label, background, meta}."""
    entries: List[Dict[str, Any]] = []

    if isinstance(spectra, (str, Path)):
        spectra = [spectra]

    if isinstance(spectra, dict):
        items = [{"file": p, "label": str(lbl)} for lbl, p in spectra.items()]
    else:
        items = []
        for s in spectra:
            if isinstance(s, dict):
                if "file" not in s:
                    raise ValueError(f"Spectrum entry {s} needs a 'file' key")
                items.append(dict(s))
            else:
                items.append({"file": s})

    for it in items:
        f = Path(it.pop("file"))
        label = str(it.pop("label", f.stem))
        bg = it.pop("background", None)
        if bg is None and isinstance(background, dict):
            bg = background.get(str(f), background.get(f, background.get(f.name)))
        elif bg is None and background is not None and not isinstance(background, dict):
            bg = background
        meta = {}
        if parse_name is not None:
            meta.update(parse_name(f) or {})
        meta.update(it)  # explicit metadata wins over parsed
        entries.append({"file": f, "label": label, "background": bg, "meta": meta})

    labels = [e["label"] for e in entries]
    dupes = {l for l in labels if labels.count(l) > 1}
    if dupes:  # make labels unique so plots do not overwrite each other
        for e in entries:
            if e["label"] in dupes:
                e["label"] = f"{e['label']}_{e['file'].parent.name}"
    return entries


def _load_background(bg, wl: np.ndarray, reader, cache: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """Background on the spectrum's wavelength axis (averaged if several files)."""
    if bg is None:
        return None
    if isinstance(bg, tuple) and len(bg) == 2 and np.ndim(bg[0]) == 1:
        paths_or_arrays = [bg]
    elif isinstance(bg, (list, tuple)):
        paths_or_arrays = list(bg)
    else:
        paths_or_arrays = [bg]

    curves = []
    for b in paths_or_arrays:
        if isinstance(b, tuple):
            wl_b, I_b = np.asarray(b[0], float), np.asarray(b[1], float)
        else:
            key = str(Path(b))
            if key not in cache:
                cache[key] = reader(b)
            wl_b, I_b = cache[key]
        if len(wl_b) == len(wl) and np.allclose(wl_b, wl):
            curves.append(I_b)
        else:
            if wl.min() < wl_b.min() - 1e-6 or wl.max() > wl_b.max() + 1e-6:
                raise ValueError(f"Background {b} does not cover {wl.min():.2f}-{wl.max():.2f} nm")
            curves.append(np.interp(wl, wl_b, I_b))
    return np.mean(curves, axis=0)


def _load_response(response, reader) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    if response is None:
        return None
    if isinstance(response, tuple):
        wl_r, s_r = np.asarray(response[0], float), np.asarray(response[1], float)
    else:
        wl_r, s_r = reader(response)
    order = np.argsort(wl_r)
    wl_r, s_r = wl_r[order], s_r[order]
    if np.any(s_r <= 0):
        raise ValueError("Spectral response must be positive everywhere it is defined")

    def resp(wl: np.ndarray) -> np.ndarray:
        if wl.min() < wl_r.min() or wl.max() > wl_r.max():
            raise ValueError(f"Response covers {wl_r.min():.1f}-{wl_r.max():.1f} nm only")
        return np.interp(wl, wl_r, s_r)

    return resp


def _resolve_band_key(key: str, loaded: Sequence[str]) -> Optional[str]:
    """Exact match first, then the tolerant matching of the original script."""
    if key in loaded:
        return key
    clean_t = re.sub(r"[^a-zA-Z0-9]", "", key).lower()
    for k in loaded:
        clean_k = re.sub(r"[^a-zA-Z0-9]", "", k).lower()
        if clean_t and (clean_t in clean_k or clean_k in clean_t):
            return k
    synonyms = {"16o1h": ["oh"], "14n16o": ["no"], "14n14n": ["14n2", "n2"], "n2+bx": ["n2+", "14n2+"]}
    for k in loaded:
        ck = re.sub(r"[^a-zA-Z0-9]", "", k).lower()
        if any(s in ck for s in synonyms.get(clean_t, [])):
            return k
    return None


# =============================================================================
# Quality checks (same criteria as Fit1TemperatureDistributions_V2)
# =============================================================================
def evaluate_signal_quality(wl, I, roi, min_peak_counts, min_snr, floor_percentile):
    """Crop to roi, subtract a percentile floor, normalise to peak, estimate SNR.
    Returns (ok, reason, peak, snr, wl_crop, I_norm)."""
    m = (wl >= roi[0]) & (wl <= roi[1])
    if np.count_nonzero(m) < 10:
        return False, "Too few ROI points", 0.0, 0.0, wl[m], np.zeros(np.count_nonzero(m))
    wl_c, I_c = wl[m], I[m]
    floor = np.percentile(I_c, floor_percentile) if floor_percentile is not None else 0.0
    I_sub = np.maximum(I_c - floor, 0.0)
    peak = float(I_sub.max())

    d = np.diff(I_c)  # robust noise from point-to-point differences (MAD)
    noise = max(float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)), 1e-12)
    snr = peak / noise

    if peak < min_peak_counts:
        return False, "Low intensity", peak, snr, wl_c, np.zeros_like(wl_c)
    if snr < min_snr:
        return False, "Low SNR", peak, snr, wl_c, np.zeros_like(wl_c)
    return True, "PRESENT", peak, snr, wl_c, I_sub / peak


def evaluate_fit_quality(fit, trot_bounds, max_shift_nm, r2_min, edge_tol_frac):
    y, yf = fit["I_exp_norm"], fit["I_fit_norm"]
    ss_res, ss_tot = np.sum((y - yf) ** 2), np.sum((y - y.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    reasons = []
    if r2 < r2_min:
        reasons.append(f"Low R2 ({r2:.2f})")
    lo, hi = trot_bounds
    if fit["T_rot"] <= lo + edge_tol_frac * (hi - lo):
        reasons.append("T_rot at grid floor")
    elif fit["T_rot"] >= hi - edge_tol_frac * (hi - lo):
        reasons.append("T_rot at grid ceiling")
    if abs(fit["shift_nm"]) >= (1.0 - edge_tol_frac) * max_shift_nm:
        reasons.append("Shift at bound")
    return len(reasons) == 0, r2, reasons


# =============================================================================
# Main entry point
# =============================================================================
def fit_temperature_distributions(
    spectra,
    bands: Iterable[Band],
    *,
    background=None,
    background_scale: float = 1.0,
    response=None,
    slit_stem: str = "09_04_2026",
    calib_lib_dir: Optional[PathLike] = None,
    fitter: Optional[MolecularFitter] = None,
    parse_name: Optional[Callable[[Path], Dict[str, Any]]] = None,
    reader: Callable[[PathLike], Tuple[np.ndarray, np.ndarray]] = read_spa,
    out_dir: Optional[PathLike] = "fit_results",
    save_plots: bool = True,
    trend_x: Optional[str] = None,
    max_shift_nm: float = 0.20,
    min_peak_counts: float = 1500.0,
    min_snr: float = 8.0,
    r2_min: float = 0.50,
    edge_tol_frac: float = 0.02,
    floor_percentile: Optional[float] = 5.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Fit every band in every spectrum. See the module docstring for input forms.

    Returns
    -------
    summary : DataFrame, one row per (spectrum, band), including skipped fits.
              Status is VALID, FIT_REJECTED (fit ran but failed a quality check),
              SKIP (signal too weak / missing) or ERROR.
    fits    : list of dicts with the curves (wl, I_exp_norm, I_fit_norm, ...) for
              every fit that ran, for your own plotting.
    """
    bands = list(bands)
    if not bands:
        raise ValueError("No bands given")

    t0 = time.time()
    if fitter is None:
        lib_dir = Path(calib_lib_dir) if calib_lib_dir is not None else HERE / "Calibrated_Library"
        fitter = MolecularFitter(calib_lib_dir=lib_dir, slit_stem=slit_stem)
    loaded = list(fitter.loaded_libraries)

    # --- resolve band keys up front so typos fail immediately ---------------------------
    resolved: Dict[str, str] = {}
    for b in bands:
        k = _resolve_band_key(b.key, loaded)
        if k is None:
            raise KeyError(f"Band '{b.label}': no library matches key '{b.key}'. Available: {loaded}")
        resolved[b.label] = k
        g = fitter.loaded_libraries[k]["wl_grid"]
        if b.roi[0] < g[0] or b.roi[1] > g[-1]:
            print(f"  WARNING: {b.label} ROI {b.roi} extends beyond library '{k}' "
                  f"({g[0]:.1f}-{g[-1]:.1f} nm); the model is zero outside it.")
    labels = [b.label for b in bands]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Band labels must be unique: {labels}")

    entries = _normalise_spectra(spectra, background, parse_name)
    if not entries:
        raise ValueError("No spectra given")
    resp = _load_response(response, reader)

    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Libraries: {loaded}")
        print(f"{len(entries)} spectra x {len(bands)} bands"
              f"{' | background subtracted' if any(e['background'] is not None for e in entries) else ''}"
              f"{' | response-corrected' if resp is not None else ''}")

    bg_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    rows: List[Dict[str, Any]] = []
    fits: List[Dict[str, Any]] = []

    for n, e in enumerate(entries, 1):
        if verbose:
            print(f"\n[{n}/{len(entries)}] {e['label']}  ({e['file'].name})")
        try:
            wl, I = reader(e["file"])
            bg = _load_background(e["background"], wl, reader, bg_cache)
            if bg is not None:
                I = I - background_scale * bg
            if resp is not None:
                I = I / resp(wl)
        except Exception as err:  # unreadable file, bad background, ...
            for b in bands:
                rows.append({**_base_row(e, b), "status": "ERROR", "reason": str(err)})
            if verbose:
                print(f"  ERROR reading/preprocessing: {err}")
            continue

        for b in bands:
            row = _base_row(e, b)
            key = resolved[b.label]
            shift_max = b.max_shift_nm if b.max_shift_nm is not None else max_shift_nm

            ok, reason, peak, snr, wl_c, I_norm = evaluate_signal_quality(
                wl, I, b.roi, min_peak_counts, min_snr, floor_percentile)
            row.update(peak_counts=peak, snr=snr)
            if not ok:
                row.update(status="SKIP", reason=reason)
                rows.append(row)
                if verbose:
                    print(f"  [SKIP] {b.label:<22} {reason} (peak {peak:.0f}, SNR {snr:.1f})")
                continue

            try:
                fit = fitter.fit_spectrum(
                    wl_crop=wl_c, I_crop_norm=I_norm, target_band=key,
                    fit_baseline=True, max_shift_nm=shift_max,
                    fit_tvib=b.fit_tvib, default_tvib=b.default_tvib)
            except Exception as err:
                row.update(status="ERROR", reason=str(err))
                rows.append(row)
                if verbose:
                    print(f"  [ERROR] {b.label:<21} {err}")
                continue

            lib = fitter.loaded_libraries[key]
            valid, r2, reasons = evaluate_fit_quality(
                fit, (lib["T_rot_grid"][0], lib["T_rot_grid"][-1]), shift_max, r2_min, edge_tol_frac)
            status = "VALID" if valid else "FIT_REJECTED"
            tvib_fixed = not np.isfinite(fit["T_vib"])
            row.update(
                status=status, reason="; ".join(reasons),
                T_rot=fit["T_rot"],
                T_vib=np.nan if tvib_fixed else fit["T_vib"],
                T_vib_fixed=b.default_tvib if tvib_fixed else np.nan,
                shift_nm=fit["shift_nm"], amplitude=fit["amplitude"],
                baseline_offset=fit["baseline_offset"], baseline_slope=fit["baseline_slope"],
                rmse_weighted=fit["rmse"], r2=r2)
            rows.append(row)
            fits.append({"label": e["label"], "file": str(e["file"]), "band": b.label,
                         "status": status, "r2": r2, **fit})

            if verbose:
                tv = f" | T_vib={fit['T_vib']:.0f} K" if not tvib_fixed else ""
                extra = f" | {', '.join(reasons)}" if reasons else ""
                tag = "PASS" if valid else "FAIL"
                print(f"  [{tag}] {b.label:<22} T_rot={fit['T_rot']:.0f} K{tv} | "
                      f"shift={fit['shift_nm']:+.3f} nm | R2={r2:.3f}{extra}")

            if save_plots and out is not None:
                plot_fit(fits[-1], out / "fits" / f"{_sanitize(e['label'])}__{_sanitize(b.label)}.png")

    summary = pd.DataFrame(rows)
    summary = summary[_column_order(summary)]
    if out is not None:
        summary.to_csv(out / "fit_summary.csv", index=False)
        if save_plots:
            plot_temperature_trends(summary, out / "trends", x=trend_x)

    if verbose:
        counts = summary["status"].value_counts().to_dict()
        print(f"\nDone in {time.time() - t0:.1f} s: {counts}")
        if out is not None:
            print(f"Results in {out.resolve()}")
    return summary, fits


def _base_row(e, b: Band) -> Dict[str, Any]:
    return {"label": e["label"], "file": str(e["file"]), **e["meta"],
            "band": b.label, "library": b.key, "roi_min_nm": b.roi[0], "roi_max_nm": b.roi[1],
            "status": "", "reason": "", "peak_counts": np.nan, "snr": np.nan,
            "T_rot": np.nan, "T_vib": np.nan, "T_vib_fixed": np.nan, "shift_nm": np.nan,
            "amplitude": np.nan, "baseline_offset": np.nan, "baseline_slope": np.nan,
            "rmse_weighted": np.nan, "r2": np.nan}


def _column_order(df: pd.DataFrame) -> List[str]:
    first = ["label", "band", "status", "T_rot", "T_vib", "T_vib_fixed", "r2", "shift_nm", "reason"]
    return [c for c in first if c in df.columns] + [c for c in df.columns if c not in first]


# =============================================================================
# Plotting
# =============================================================================
def plot_fit(fit: Dict[str, Any], path: PathLike) -> None:
    """Data, fit and residual for one fit (a dict from the returned `fits` list)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wl, y, yf = fit["wl_crop"], fit["I_exp_norm"], fit["I_fit_norm"]
    tv = f", T_vib = {fit['T_vib']:.0f} K" if np.isfinite(fit.get("T_vib", np.nan)) else ""
    fig, ax = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(wl, y, color="0.35", lw=1.1, label="measured (normalised)")
    ax[0].plot(wl, yf, color="#d62728", lw=1.1, ls="-" if fit["status"] == "VALID" else ":",
               label=f"fit: T_rot = {fit['T_rot']:.0f} K{tv}")
    ax[0].set_ylabel("Normalised intensity")
    ax[0].set_title(f"{fit['label']}  |  {fit['band']}  |  shift {fit['shift_nm']:+.3f} nm  |  "
                    f"R² = {fit['r2']:.3f}  |  {fit['status']}", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[0].grid(True, ls=":", alpha=0.6)
    ax[1].plot(wl, y - yf, color="#1f77b4", lw=0.8)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_ylabel("Residual")
    ax[1].set_xlabel("Measured wavelength (nm)")
    ax[1].grid(True, ls=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_temperature_trends(summary: pd.DataFrame, out_dir: PathLike, x: Optional[str] = None) -> None:
    """One figure per band: T_rot (and T_vib if fitted) across spectra.
    x = a metadata column to plot against (e.g. "voltage_kV"); default is spectrum order.
    Valid fits are filled markers, rejected fits hollow; skipped spectra are left out."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for band, g in summary.groupby("band", sort=False):
        g = g[g["status"].isin(["VALID", "FIT_REJECTED"])]
        if g.empty:
            continue
        numeric_x = x is not None and x in g.columns and pd.api.types.is_numeric_dtype(g[x])
        if numeric_x:
            g = g.sort_values(x)
            xv = g[x].to_numpy(float)
        else:
            xv = np.arange(len(g))
        has_vib = g["T_vib"].notna().any()
        fig, axes = plt.subplots(2 if has_vib else 1, 1, figsize=(8, 6 if has_vib else 3.8),
                                 sharex=True, squeeze=False)
        for ax, col, color in zip(axes[:, 0], ["T_rot", "T_vib"] if has_vib else ["T_rot"],
                                  ["#d62728", "#1f77b4"]):
            v = g["status"].to_numpy() == "VALID"
            yv = g[col].to_numpy(float)
            ax.plot(xv[v], yv[v], "o-", color=color, ms=6, label="valid")
            if (~v).any():
                ax.plot(xv[~v], yv[~v], "o", mfc="none", color=color, ms=6, label="rejected")
            ax.set_ylabel(f"{col} (K)")
            ax.grid(True, ls=":", alpha=0.6)
            ax.legend(fontsize=8)
        axes[-1, 0].set_xlabel(x if numeric_x else "spectrum")
        if not numeric_x:
            axes[-1, 0].set_xticks(xv)
            axes[-1, 0].set_xticklabels(g["label"], rotation=45, ha="right", fontsize=8)
        axes[0, 0].set_title(f"{band}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{_sanitize(band)}.png", dpi=140)
        plt.close(fig)


# =============================================================================
# Example (edit and run this file directly, or import the function)
# =============================================================================
if __name__ == "__main__":
    DATA = Path(r"C:\Users\dptro\Documents\Work\Python\ArgonCrModel\Experimental_Data\EchelleData\PrelijminaryNitrogenAdmixtureMini\GsaSweep_85W_1_torr\SPA_Files")

    BANDS = [
        Band("14N2", (335.0, 337.5), label="N2(C-B) (0-0)"),
        Band("14N2", (330.0, 385.0), label="N2(C-B) system", fit_tvib=True),
    ]
    
    from pathlib import Path
    
    # Matches "Percent1_5__1" -> whole=1, frac=5, rep=1  (also tolerates the "Precent" typo)
    _PAT = re.compile(r"P(?:er|re)cent(\d+)(?:_(\d+))?__(\d+)", re.IGNORECASE)
    
    
    def parse_name(path):
        """Ar/N2 mixture filename -> {'n2_pct', 'ar_pct', 'replicate'}.
    
        'Percent2_5__3.spa' -> {'n2_pct': 2.5, 'ar_pct': 97.5, 'replicate': 3}
        Returns {} if the name doesn't match, so unmatched files still load.
        """
        m = _PAT.search(Path(path).stem)
        if not m:
            return {}
        whole, frac, rep = m.groups()
        n2 = float(f"{whole}.{frac or 0}")
        return {"n2_pct": n2, "ar_pct": 100.0 - n2, "replicate": int(rep)}

    summary, fits = fit_temperature_distributions(
        sorted(DATA.glob("*.spa")),
        BANDS,
        background=DATA / "background" / "ref1.spa",
        parse_name=parse_name,
        trend_x="voltage_kV",
        out_dir=DATA / "fit_results",
    )
    print(summary[["label", "band", "status", "T_rot", "r2"]])