
# -*- coding: utf-8 -*-
"""
ArN2_LineAnalysis.py
====================

Line identification / intensity extraction for high-resolution (echelle)
OES of Ar and Ar/N2 plasmas, built on the CR-model level lists.

Pipeline
--------
1. Catalogue: every Ar I line (NIST csv) and N I line (NIST json + Ritz
   supplement from the level list) is placed at its *air* wavelength and
   mapped onto the CR-model level labels (ArgonLevelList.json,
   AtomicNitrogen_levels.json).  N2 / N2+ band heads are generated from
   Dunham constants (SPS, FNS) or literature-anchored constants (FPS).
2. Measurement: catalogue lines are clustered into blend groups and each
   group is fitted with fixed-centre Gaussians (common shift + common
   instrument width, amplitudes >= 0 by bounded linear least squares) on a
   linear baseline.  Strong lines get an additional free Gaussian fit for
   shape metrics.  Detection is decided on local SNR.
3. Contamination (mixture vs pure-Ar reference): for every Ar line the
   pure-Ar spectrum is scaled onto the mixture in a +-3 FWHM window; what
   the scaled reference cannot explain is "excess".  Excess area, residual
   peaks (labelled N I / N2 band / unidentified), width ratio, centre
   shift and pedestal change are combined into clean / suspect /
   contaminated.
4. Output: line tables (csv), lower->upper intensity matrices (csv + png),
   overview spectrum (png), per-line diagnostic panels (pdf), actinometry
   candidate table (csv).

Run from Spyder: edit CONFIG, press F5.  Everything is a plain function so
individual steps can be re-run interactively.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HC_EV_NM = 1239.841984  # eV*nm



# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
# Project root (ArgonCrModel folder), found relative to this file
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FOLDER = os.path.join(ROOT_DIR, "Experimental_Data", "EchelleData", "TestData")



CONFIG = dict(
    # --- inputs -------------------------------------------------------------
    folder = FOLDER,
    pure="Percent0_1.spa",              # pure argon spectrum (reference)
    mix="Percent3_0__1.spa",            # Ar / N2 mixture
    background="ref1.spa",              # plasma-off frame (noise floor); None to skip
    ar_levels="ArgonLevelList.json",
    ar_lines="ArgonReactionList.csv",
    n_levels="AtomicNitrogen_levels.json",
    n_lines="AtomicNitrogenReactionList.json",
    outdir=os.path.join(ROOT_DIR, "Experimental_Data", "Output", "ArN2_out"),
    # --- spectral range -----------------------------------------------------
    wl_min=250.0,                       # below ~250 nm the echelle is noise only
    wl_max=870.0,
    # --- fitting ------------------------------------------------------------
    resolving_power=None,               # FWHM = lambda / R ; None -> measured from pure spectrum
    cluster_sep_fwhm=4.0,               # lines closer than this (in FWHM) are fitted jointly
    window_pad_fwhm=4.0,                # baseline flank on each side of a group
    shift_scan_fwhm=0.5,                # common shift searched per group, +- this x FWHM (after global calibration offset)
    width_scan=(0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.8),   # width factors searched (x instrument FWHM)
    gap_min_run=5,                      # >= this many consecutive zeros = order gap
    edge_margin_nm=0.15,                # lines closer than this to a gap edge are flagged
    # --- detection thresholds -----------------------------------------------
    snr_detect=3.0,                     # peak height / local sigma to call a line present
    snr_quant=10.0,                     # ... to call its intensity usable
    free_fit_snr=15.0,                  # free single-Gaussian shape fit above this SNR
    blend_frac=0.2,                     # neighbour within 0.6 FWHM carrying > this fraction -> blend
    saturation_level=None,              # counts; None -> flat-top test only
    # --- contamination thresholds -------------------------------------------
    excess_suspect=0.05,                # signed excess area / line area (needs z > 3)
    excess_contam=0.10,
    resid_peak_sigma=4.0,               # residual peak must exceed this x local sigma ...
    resid_peak_frac=0.10,               # ... AND this fraction of the line height
    ref_shift_fwhm=0.5,                 # shift allowed for the pure-Ar reference (x FWHM)
    extra_amp_frac=0.20,                # hidden-line test: Gaussian fitted to the residual, amplitude / line height
    extra_gain_min=0.5,                 # ... and it must remove this fraction of the residual SSE
    branching_A_dev=0.5,                # |I_i/I_j / (A_i/A_j) - 1| above this for same-upper-level pairs (< 15 nm apart)
    branching_dev=0.30,                 # mix/pure ratio deviating > this from other lines of same upper level
    width_ratio_contam=1.30,            # second-moment width mix/pure
    pedestal_suspect=0.05,              # pedestal/height change mix - pure
    # --- plotting -----------------------------------------------------------
    make_pdf=True,
    pdf_model_levels_only=True,         # PDF panels only for Ar lines between CR-model levels (+ all N I)
    matrix_value="area",                # 'area' or 'height'
)
for _k in ("pure", "mix", "background", "ar_levels", "ar_lines", "n_levels", "n_lines"):
    if CONFIG[_k]:
        CONFIG[_k] = os.path.join(CONFIG["folder"], CONFIG[_k])
if not os.path.isabs(CONFIG["outdir"]):
    CONFIG["outdir"] = os.path.join(CONFIG["folder"], CONFIG["outdir"])

# ----------------------------------------------------------------------------
# 1. IO helpers
# ----------------------------------------------------------------------------
def read_spa(path):
    """Two-column ASCII echelle export (header line, then 'wl  I')."""
    d = np.loadtxt(path, skiprows=1)
    wl, I = d[:, 0], d[:, 1]
    order = np.argsort(wl)
    return wl[order], I[order]


def find_gaps(I, min_run=5):
    """Order gaps are exported as runs of exact zeros. Returns bool mask + list."""
    z = (I == 0)
    mask = np.zeros_like(z)
    gaps = []
    i = 0
    n = len(I)
    while i < n:
        if z[i]:
            j = i
            while j < n and z[j]:
                j += 1
            if j - i >= min_run:
                mask[i:j] = True
                gaps.append((i, j - 1))
            i = j
        else:
            i += 1
    return mask, gaps


def n_air(lam_nm):
    """Refractive index of standard air (NIST/Peck & Reeder), lam in nm."""
    s2 = (1e3 / np.asarray(lam_nm, float)) ** 2
    return 1 + 0.05792105 / (238.0185 - s2) + 0.00167917 / (57.362 - s2)


def vac_to_air(lam_vac_nm):
    return lam_vac_nm / n_air(lam_vac_nm)


def air_to_vac(lam_air_nm):
    lam = np.asarray(lam_air_nm, float)
    for _ in range(3):
        lam = lam_air_nm * n_air(lam)
    return lam


def _clean(s):
    """Strip NIST csv decoration: ="...", quotes, spaces."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return re.sub(r'^=?"?|"?$', "", str(s)).strip()


def level_key(conf, term, J):
    conf = _clean(conf).replace(" ", "")
    term = _clean(term).replace(" ", "")
    try:
        J = float(_clean(J))
    except ValueError:
        J = np.nan
    return (conf, term, J)


# ----------------------------------------------------------------------------
# 2. Levels
# ----------------------------------------------------------------------------
_PASCHEN_4S = {"4s1": "1s5", "4s2": "1s4", "4s3": "1s3", "4s4": "1s2"}


def paschen(label):
    """Paschen name for the Ar 4s/4p labels used in the CR model."""
    if label in _PASCHEN_4S:
        return _PASCHEN_4S[label]
    m = re.match(r"^4p(\d+)$", label)
    if m:
        return f"2p{11 - int(m.group(1))}"
    return ""


def short_level_name(conf, term, J):
    """Compact human label for levels not in the CR-model list, e.g. 5d'[5/2]2."""
    conf, term, J = level_key(conf, term, J)
    m = re.search(r"\((2P\*<(\d)/2>)\)\.(\d+[spdfgh])$", conf)
    if m:
        core = "'" if m.group(2) == "1" else ""
        nl = m.group(3)
        k = re.search(r"\[(\d+/\d+)\]", term)
        kk = f"[{k.group(1)}]" if k else term
        return f"{nl}{core}{kk}{J:g}"
    # nitrogen / generic: 'conf term J'
    conf_s = conf.replace("2s2.2p2.", "").replace("2s2.", "")
    return f"{conf_s} {term} {J:g}"


def load_levels(path):
    """Level json -> DataFrame (index = label) + key lookup dict."""
    d = json.load(open(path))
    rows = []
    for lab, v in d.items():
        rows.append(dict(label=lab, id=v["id"], configuration=v["configuration"],
                         term=v["term"], J=float(v["J"]), g=v["g"],
                         energy_eV=v["energy_eV"], manifold=v["manifold"], kind=v["kind"],
                         paschen=paschen(lab),
                         desc=short_level_name(v["configuration"], v["term"], v["J"])))
    df = pd.DataFrame(rows).set_index("label")
    lookup = {level_key(r.configuration, r.term, r.J): lab for lab, r in df.iterrows()}
    return df, lookup


# ----------------------------------------------------------------------------
# 3. Line catalogues
# ----------------------------------------------------------------------------
_LINE_COLS = ["species", "wl_air", "wl_vac", "Aki", "acc", "Ei", "Ek", "gi", "gk",
              "lower", "upper", "lower_in_model", "upper_in_model",
              "lower_desc", "upper_desc", "source"]


def _medium_from_row(wl_listed, Ei, Ek):
    """Decide whether a listed wavelength is air or vacuum from the Ritz value."""
    if not (np.isfinite(wl_listed) and np.isfinite(Ei) and np.isfinite(Ek)) or Ek <= Ei:
        return "unknown"
    lam_vac = HC_EV_NM / (Ek - Ei)
    if abs(wl_listed - lam_vac) < abs(wl_listed - vac_to_air(lam_vac)):
        return "vac"
    return "air"


def load_ar_lines(csv_path, levels, lookup):
    """NIST ASD csv (Ar I) -> catalogue DataFrame. Handles ="..." quoting and
    repeated header rows. Wavelength medium is auto-detected per row."""
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    raw = raw.map(_clean)
    raw = raw[raw["conf_i"] != "conf_i"]                      # repeated headers
    num = lambda c: pd.to_numeric(raw[c].str.replace(r"[^0-9.eE+\-]", "", regex=True), errors="coerce")
    obs, ritz = num("obs_wl_vac(nm)"), num("ritz_wl_vac(nm)")
    wl = obs.where(obs.notna(), ritz)
    Aki, Ei, Ek = num("Aki(s^-1)"), num("Ei(eV)"), num("Ek(eV)")
    Ji, Jk = num("J_i"), num("J_k")
    rows = []
    for idx in raw.index:
        if not np.isfinite(wl[idx]) or not np.isfinite(Aki[idx]) or Aki[idx] <= 0:
            continue
        kl = level_key(raw.at[idx, "conf_i"], raw.at[idx, "term_i"], Ji[idx])
        ku = level_key(raw.at[idx, "conf_k"], raw.at[idx, "term_k"], Jk[idx])
        med = _medium_from_row(wl[idx], Ei[idx], Ek[idx])
        if med == "vac":
            wl_vac, wl_air = wl[idx], vac_to_air(wl[idx])
        else:                                                    # air or unknown(>200nm)
            wl_air, wl_vac = wl[idx], air_to_vac(wl[idx])
        lo = lookup.get(kl); up = lookup.get(ku)
        rows.append(dict(species="Ar I", wl_air=float(wl_air), wl_vac=float(wl_vac),
                         Aki=Aki[idx], acc=raw.at[idx, "Acc"], Ei=Ei[idx], Ek=Ek[idx],
                         gi=2 * Ji[idx] + 1 if np.isfinite(Ji[idx]) else np.nan,
                         gk=2 * Jk[idx] + 1 if np.isfinite(Jk[idx]) else np.nan,
                         lower=lo if lo else short_level_name(*kl),
                         upper=up if up else short_level_name(*ku),
                         lower_in_model=lo is not None, upper_in_model=up is not None,
                         lower_desc=short_level_name(*kl), upper_desc=short_level_name(*ku),
                         source="NIST-csv"))
    return pd.DataFrame(rows, columns=_LINE_COLS).sort_values("wl_air").reset_index(drop=True)


def _parity(term):
    return "odd" if term.strip().endswith("*") else "even"


def _multiplicity(term):
    m = re.match(r"^(\d+)", term.strip())
    return int(m.group(1)) if m else None


def ritz_supplement(levels, wl_lo, wl_hi, same_multiplicity=True):
    """E1-allowed level pairs (parity change, dJ=0,+-1, no 0-0) in [wl_lo, wl_hi]
    computed from level energies. Used where the NIST export is truncated."""
    rows = []
    L = levels.reset_index()
    for a, b in itertools.combinations(L.itertuples(), 2):
        lo, up = (a, b) if a.energy_eV < b.energy_eV else (b, a)
        dE = up.energy_eV - lo.energy_eV
        if dE <= 0:
            continue
        lam_vac = HC_EV_NM / dE
        lam_air = vac_to_air(lam_vac)
        if not (wl_lo <= lam_air <= wl_hi):
            continue
        if _parity(lo.term) == _parity(up.term):
            continue
        if abs(lo.J - up.J) > 1 or (lo.J == 0 and up.J == 0):
            continue
        if same_multiplicity and _multiplicity(lo.term) != _multiplicity(up.term):
            continue
        rows.append(dict(species="N I", wl_air=float(lam_air), wl_vac=float(lam_vac),
                         Aki=np.nan, acc="", Ei=lo.energy_eV, Ek=up.energy_eV,
                         gi=lo.g, gk=up.g, lower=lo.label, upper=up.label,
                         lower_in_model=True, upper_in_model=True,
                         lower_desc=lo.desc, upper_desc=up.desc, source="ritz"))
    return pd.DataFrame(rows, columns=_LINE_COLS)


def load_n_lines(json_path, levels, lookup, wl_max=870.0, supplement=True):
    """NIST json export (N I, E1 only) + Ritz supplement beyond its last line."""
    d = json.load(open(json_path))
    medium = d.get("wl_medium", "air")
    rows = []
    for t in d["transitions"]:
        if t.get("type", ""):                       # skip M1/E2
            continue
        kl = level_key(t["lower"]["config"], t["lower"]["term"], t["lower"]["J"])
        ku = level_key(t["upper"]["config"], t["upper"]["term"], t["upper"]["J"])
        wl = float(t["wl_nm"])
        if medium == "vac":
            wl_vac, wl_air = wl, vac_to_air(wl)
        else:
            wl_air, wl_vac = wl, air_to_vac(wl)
        lo, up = lookup.get(kl), lookup.get(ku)
        rows.append(dict(species="N I", wl_air=wl_air, wl_vac=float(wl_vac), Aki=t["Aki"],
                         acc=t.get("acc", ""), Ei=t["lower"]["E_eV"], Ek=t["upper"]["E_eV"],
                         gi=t["lower"]["g"], gk=t["upper"]["g"],
                         lower=lo if lo else short_level_name(*kl),
                         upper=up if up else short_level_name(*ku),
                         lower_in_model=lo is not None, upper_in_model=up is not None,
                         lower_desc=short_level_name(*kl), upper_desc=short_level_name(*ku),
                         source="NIST-json"))
    df = pd.DataFrame(rows, columns=_LINE_COLS)
    if supplement and len(df):
        last = df["wl_air"].max()
        if last < wl_max:
            sup = ritz_supplement(levels, last + 0.01, wl_max)
            df = pd.concat([df, sup], ignore_index=True)
    return df.sort_values("wl_air").reset_index(drop=True)


# ----------------------------------------------------------------------------
# 4. Molecular band heads
# ----------------------------------------------------------------------------
# Dunham constants (cm^-1). SPS/FNS heads reproduce Pearse & Gaydon to <0.1 nm.
_STATES = {
    "N2 C3Pu":  dict(Te=89136.88, we=2047.178, wexe=28.445, weye=2.08833, weze=-0.5350, Be=1.82473, ae=0.01868),
    "N2 B3Pg":  dict(Te=59619.35, we=1733.391, wexe=14.1221, weye=-0.0569, weze=-0.00361, Be=1.63745, ae=0.01791),
    "N2 A3Su":  dict(Te=50203.66, we=1460.941, wexe=13.980, weye=0.0244, weze=-0.00256, Be=1.45460, ae=0.01801),
    "N2+ B2Su": dict(Te=25461.46, we=2419.84, wexe=23.19, weye=-0.5375, weze=-0.0495, Be=2.07456, ae=0.0240),
    "N2+ X2Sg": dict(Te=0.0, we=2207.00, wexe=16.10, weye=-0.040, weze=0.0, Be=1.93176, ae=0.01881),
}


def _G(s, v):
    x = v + 0.5
    return s["we"] * x - s["wexe"] * x ** 2 + s["weye"] * x ** 3 + s["weze"] * x ** 4


def _band_head(up, lo, vu, vl):
    nu0 = up["Te"] - lo["Te"] + _G(up, vu) - _G(lo, vl)
    Bu, Bl = up["Be"] - up["ae"] * (vu + 0.5), lo["Be"] - lo["ae"] * (vl + 0.5)
    dB = Bu - Bl
    m = -(Bu + Bl) / (2 * dB)
    nu_head = nu0 + (Bu + Bl) * m + dB * m * m
    lam = 1e7 / nu_head
    return vac_to_air(lam), ("violet" if dB > 0 else "red")


def n2_band_catalogue(wl_min=250.0, wl_max=870.0):
    """Band heads of the N2 systems relevant to Ar/N2 OES plus common
    impurity bands. 'extent_nm' is the approximate width of the resolved
    rotational structure on the degraded side of the head."""
    rows = []
    spec = [("SPS C-B", "N2 C3Pu", "N2 B3Pg", range(0, 5), range(0, 9), 3.5, 0.0),
            ("FNS B-X", "N2+ B2Su", "N2+ X2Sg", range(0, 4), range(0, 5), 4.0, 0.0),
            # FPS: triplet sub-band heads spread ~1-2 nm; +1.5 nm brings the
            # Dunham head onto the Pearse & Gaydon (reddest) head to ~+-1 nm
            ("FPS B-A", "N2 B3Pg", "N2 A3Su", range(0, 13), range(0, 11), 8.0, 1.5)]
    for sys_name, up, lo, vus, vls, ext, corr in spec:
        for vu in vus:
            for vl in vls:
                lam, deg = _band_head(_STATES[up], _STATES[lo], vu, vl)
                lam += corr
                if wl_min <= lam <= wl_max:
                    rows.append(dict(system=sys_name, vu=vu, vl=vl, head_nm=round(lam, 2),
                                     extent_nm=ext, degraded=deg,
                                     note="Dunham" + (" +1.5nm (approx +-1nm)" if corr else "")))
    # impurity markers (literature heads)
    for sysn, vu, vl, lam, ext in [("OH A-X", 0, 0, 306.36, 4.0), ("OH A-X", 0, 0, 308.9, 3.0),
                                   ("CN B-X", 0, 0, 388.34, 2.5), ("CN B-X", 1, 1, 387.14, 1.5),
                                   ("NH A-X", 0, 0, 336.0, 2.0)]:
        if wl_min <= lam <= wl_max:
            rows.append(dict(system=sysn, vu=vu, vl=vl, head_nm=lam, extent_nm=ext,
                             degraded="violet", note="literature"))
    df = pd.DataFrame(rows).sort_values("head_nm").reset_index(drop=True)
    df["lo_nm"] = np.where(df.degraded == "violet", df.head_nm - df.extent_nm, df.head_nm - 0.3)
    df["hi_nm"] = np.where(df.degraded == "violet", df.head_nm + 0.3, df.head_nm + df.extent_nm)
    return df


def bands_overlapping(bands, wl, halfwidth=0.0):
    m = (bands.lo_nm <= wl + halfwidth) & (bands.hi_nm >= wl - halfwidth)
    return [f"{r.system} ({r.vu},{r.vl})" for r in bands[m].itertuples()]


# ----------------------------------------------------------------------------
# 5. Spectrum container, noise, instrument function
# ----------------------------------------------------------------------------
@dataclass
class Spectrum:
    name: str
    wl: np.ndarray
    I: np.ndarray
    gapmask: np.ndarray
    gaps: list
    sigma_floor: np.ndarray = None      # per-point noise floor (from background frame)

    def in_gap(self, wl0, margin=0.0):
        """True if wl0 (+- margin) touches an order gap or lies outside the data."""
        if wl0 - margin < self.wl[0] or wl0 + margin > self.wl[-1]:
            return True
        i0, i1 = np.searchsorted(self.wl, [wl0 - margin, wl0 + margin])
        return bool(self.gapmask[max(i0 - 1, 0):i1 + 1].any())

    def window(self, lo, hi):
        i0, i1 = np.searchsorted(self.wl, [lo, hi])
        sl = slice(i0, i1)
        keep = ~self.gapmask[sl]
        return self.wl[sl][keep], self.I[sl][keep]


def load_spectrum(path, name, cfg, background=None):
    wl, I = read_spa(path)
    gapmask, gaps = find_gaps(I, cfg["gap_min_run"])
    sp = Spectrum(name, wl, I, gapmask, [(wl[a], wl[b]) for a, b in gaps])
    if background is not None:
        sp.sigma_floor = np.interp(wl, background[0], background[1])
    return sp


def noise_floor_from_background(path, cfg, win_nm=2.0):
    """Robust local sigma of a plasma-off frame as a function of wavelength."""
    wl, I = read_spa(path)
    gapmask, _ = find_gaps(I, cfg["gap_min_run"])
    resid = I - median_filter(I, 31)
    resid[gapmask] = np.nan
    centres = np.arange(wl[0], wl[-1], win_nm / 2)
    sig = []
    for c in centres:
        m = (wl >= c - win_nm) & (wl < c + win_nm) & np.isfinite(resid)
        sig.append(1.4826 * np.nanmedian(np.abs(resid[m])) if m.sum() > 20 else np.nan)
    sig = pd.Series(sig).interpolate(limit_direction="both").to_numpy()
    return centres, sig


def gauss(x, A, x0, fwhm):
    s = fwhm / 2.3548200450309493
    return A * np.exp(-0.5 * ((x - x0) / s) ** 2)


def gauss_area(A, fwhm):
    return A * fwhm * np.sqrt(np.pi / (4 * np.log(2)))


def _free_fit(wl, I, x0, fwhm0):
    """Free Gaussian + linear baseline. Returns dict or None."""
    A0 = max(I.max() - np.median(I), 1e-9)
    p0 = [A0, x0, fwhm0, np.median(I), 0.0]
    lb = [0, x0 - fwhm0, 0.4 * fwhm0, -np.inf, -np.inf]
    ub = [np.inf, x0 + fwhm0, 3.0 * fwhm0, np.inf, np.inf]
    f = lambda x, A, c, w, b0, b1: gauss(x, A, c, w) + b0 + b1 * (x - x0)
    try:
        p, _ = curve_fit(f, wl, I, p0=p0, bounds=(lb, ub), maxfev=4000)
    except Exception:
        return None
    resid = I - f(wl, *p)
    # asymmetry: left vs right half width at half maximum of the data
    core = I - (p[3] + p[4] * (wl - x0))
    half = 0.5 * p[0]
    ic = np.argmin(abs(wl - p[1]))
    try:
        il = ic - np.argmax(core[ic::-1] < half)
        ir = ic + np.argmax(core[ic:] < half)
        asym = ((wl[ir] - p[1]) - (p[1] - wl[il])) / p[2]
    except Exception:
        asym = np.nan
    return dict(A=p[0], centre=p[1], fwhm=p[2], base=p[3], slope=p[4],
                rms=np.std(resid), asym=asym)


def measure_instrument_function(sp: Spectrum, cat: pd.DataFrame, cfg, n_lines=25):
    """Fit the strongest isolated catalogue lines with free Gaussians and
    return (fwhm_fn(lambda), calibration offset fn, table)."""
    rows = []
    cand = cat[(cat.wl_air > cfg["wl_min"]) & (cat.wl_air < cfg["wl_max"])].copy()
    # isolated: no other catalogue line within 0.5 nm
    wls = cand.wl_air.values
    iso = np.array([(np.abs(wls - w) < 0.5).sum() == 1 for w in wls])
    cand = cand[iso]
    # rank by observed peak height
    heights = []
    for w in cand.wl_air:
        if sp.in_gap(w, 0.3):
            heights.append(-1); continue
        x, y = sp.window(w - 0.3, w + 0.3)
        heights.append(y.max() - np.median(y) if len(y) > 10 else -1)
    cand = cand.assign(h=heights).sort_values("h", ascending=False).head(n_lines)
    fwhm_guess = 0.06
    for r in cand.itertuples():
        x, y = sp.window(r.wl_air - 0.4, r.wl_air + 0.4)
        ff = _free_fit(x, y, r.wl_air, fwhm_guess)
        if ff is None or ff["A"] < 20 * ff["rms"] or ff["fwhm"] < 0.015:
            continue
        rows.append(dict(wl_cat=r.wl_air, species=r.species, centre=ff["centre"],
                         fwhm=ff["fwhm"], height=ff["A"], offset=ff["centre"] - r.wl_air))
    tab = pd.DataFrame(rows)
    if len(tab) < 3:
        raise RuntimeError("Could not measure instrument function (too few strong isolated lines)")
    # FWHM = a*lambda + b  (constant resolving power -> a = 1/R)
    a, b = np.polyfit(tab.wl_cat, tab.fwhm, 1)
    fwhm_fn = lambda lam: np.maximum(a * lam + b, 0.02)
    off = np.median(tab.offset)
    offset_fn = lambda lam: off
    tab.attrs["fit"] = dict(a=a, b=b, R_eff=float(np.median(tab.wl_cat / tab.fwhm)), offset_median=off)
    return fwhm_fn, offset_fn, tab


# ----------------------------------------------------------------------------
# 6. Group fitting (fixed centres, common shift/width, NNLS amplitudes)
# ----------------------------------------------------------------------------
def group_lines(cat, fwhm_fn, sep_fwhm):
    """Cluster catalogue lines (sorted by wl) whose neighbours are within sep_fwhm*FWHM."""
    groups, cur = [], [cat.index[0]]
    for i0, i1 in zip(cat.index[:-1], cat.index[1:]):
        if cat.at[i1, "wl_air"] - cat.at[i0, "wl_air"] <= sep_fwhm * fwhm_fn(cat.at[i0, "wl_air"]):
            cur.append(i1)
        else:
            groups.append(cur); cur = [i1]
    groups.append(cur)
    return groups


def _design(x, centres, fwhm, shift):
    cols = [gauss(x, 1.0, c + shift, fwhm) for c in centres]
    cols += [np.ones_like(x), (x - x.mean())]
    return np.column_stack(cols)


def fit_group(x, y, centres, fwhm0, cfg, offset0=0.0):
    """Grid over (shift, width factor); amplitudes >=0 by bounded linear LSQ.
    The shift is searched around the global calibration offset offset0."""
    nl = len(centres)
    shifts = offset0 + np.linspace(-cfg["shift_scan_fwhm"], cfg["shift_scan_fwhm"], 15) * fwhm0
    best = None
    lb = np.r_[np.zeros(nl), -np.inf, -np.inf]
    ub = np.r_[np.full(nl, np.inf), np.inf, np.inf]
    for wf in cfg["width_scan"]:
        for sh in shifts:
            X = _design(x, centres, wf * fwhm0, sh)
            res = lsq_linear(X, y, bounds=(lb, ub), method="bvls", max_iter=200)
            cost = np.sum((y - X @ res.x) ** 2)
            if best is None or cost < best["cost"]:
                best = dict(cost=cost, shift=sh, fwhm=wf * fwhm0, coef=res.x, X=X)
    model = best["X"] @ best["coef"]
    resid = y - model
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    best.update(model=model, resid=resid, sigma=max(sigma, 1e-12))
    return best


def measure_lines(sp: Spectrum, cat: pd.DataFrame, fwhm_fn, cfg, tag="", offset0=0.0):
    """Measure every catalogue line in one spectrum. Returns results DataFrame
    (same index as cat) and a dict of group fits keyed by group id."""
    cat = cat[(cat.wl_air >= cfg["wl_min"]) & (cat.wl_air <= cfg["wl_max"])].sort_values("wl_air")
    groups = group_lines(cat, fwhm_fn, cfg["cluster_sep_fwhm"])
    out = {}
    fits = {}
    for gid, idx in enumerate(groups):
        centres = cat.loc[idx, "wl_air"].to_numpy()
        fw = float(fwhm_fn(centres.mean()))
        lo, hi = centres.min() - cfg["window_pad_fwhm"] * fw, centres.max() + cfg["window_pad_fwhm"] * fw
        # gap handling: lines inside/at the edge of a gap are not fitted
        status = {i: "ok" for i in idx}
        for i, c in zip(idx, centres):
            if sp.in_gap(c, 0.0):
                status[i] = "gap"
            elif sp.in_gap(c, cfg["edge_margin_nm"] + fw):
                status[i] = "edge"
        fit_idx = [i for i in idx if status[i] == "ok"]
        x, y = sp.window(lo, hi)
        if len(fit_idx) == 0 or len(x) < 8:
            for i in idx:
                out[i] = dict(group=gid, status=status[i] if status[i] != "ok" else "no_data")
            continue
        fc = cat.loc[fit_idx, "wl_air"].to_numpy()
        gf = fit_group(x, y, fc, fw, cfg, offset0)
        fits[gid] = dict(x=x, y=y, centres=fc, idx=fit_idx, **{k: gf[k] for k in ("shift", "fwhm", "coef", "model", "resid", "sigma")})
        base = gf["coef"][-2:]
        # unexplained residual peaks in the window
        hmin = max(cfg["resid_peak_sigma"] * gf["sigma"], cfg["resid_peak_frac"] * gf["coef"][:len(fc)].max())
        pk, props = find_peaks(gf["resid"], height=hmin, distance=3)
        for j, i in enumerate(fit_idx):
            A = gf["coef"][j]
            c = fc[j] + gf["shift"]
            height = A
            snr = height / gf["sigma"]
            area = gauss_area(A, gf["fwhm"])
            # blend check
            others = [(k, gf["coef"][k]) for k in range(len(fc)) if k != j and abs(fc[k] - fc[j]) < 0.6 * gf["fwhm"]]
            blended = any(a2 > cfg["blend_frac"] * max(A, 1e-30) for _, a2 in others)
            # saturation: flat top in raw data
            core = (x > c - 0.6 * gf["fwhm"]) & (x < c + 0.6 * gf["fwhm"])
            sat = False
            if core.sum() >= 3:
                yc = y[core]
                sat = ((yc > 0.99 * yc.max()).sum() >= 3) and (snr > 50)
                if cfg["saturation_level"] is not None:
                    sat = sat or (yc.max() > cfg["saturation_level"])
            st = "ok"
            if snr < cfg["snr_detect"]:
                st = "not_detected"
            elif snr < cfg["snr_quant"]:
                st = "weak"
            if blended and st == "ok":
                st = "blend"
            if sat:
                st = "saturated"
            pedestal = (base[0] + base[1] * (c - x.mean())) / max(height, 1e-30)
            near_pk = [x[p] for p in pk if abs(x[p] - c) < 3 * gf["fwhm"]]
            rec = dict(group=gid, status=st, centre_fit=c, shift=gf["shift"], fwhm_group=gf["fwhm"],
                       height=height, area=area, sigma=gf["sigma"], snr=snr, pedestal_frac=pedestal,
                       blended=blended, n_resid_peaks=len(near_pk),
                       resid_peaks_nm=";".join(f"{v:.3f}" for v in near_pk))
            # free single-line fit for shape metrics
            if snr >= cfg["free_fit_snr"] and not blended:
                m = (x > c - 2.5 * gf["fwhm"]) & (x < c + 2.5 * gf["fwhm"])
                y_other = gf["model"] - gauss(x, A, c, gf["fwhm"]) - (base[0] + base[1] * (x - x.mean()))
                ff = _free_fit(x[m], y[m] - y_other[m], c, gf["fwhm"])
                if ff is not None:
                    rec.update(fwhm_free=ff["fwhm"], centre_free=ff["centre"], asym=ff["asym"],
                               height_free=ff["A"], area_free=gauss_area(ff["A"], ff["fwhm"]))
            out[i] = rec
        for i in idx:
            if i not in out:
                out[i] = dict(group=gid, status=status[i])
    res = pd.DataFrame.from_dict(out, orient="index").reindex(cat.index)
    for col in ["centre_fit", "shift", "fwhm_group", "height", "area", "sigma", "snr", "pedestal_frac",
                "fwhm_free", "centre_free", "asym", "height_free", "area_free", "n_resid_peaks"]:
        if col not in res:
            res[col] = np.nan
    res["blended"] = res.get("blended", False)
    res = pd.concat([cat, res], axis=1)
    res.attrs["tag"] = tag
    return res, fits


# ----------------------------------------------------------------------------
# 7. N2 contamination assessment (mixture vs pure-Ar reference)
# ----------------------------------------------------------------------------
def _label_feature(wl_feat, cat_all, bands, tol_nm):
    """Name a residual feature: N I / Ar I catalogue line, N2 band region, or unidentified."""
    d = np.abs(cat_all.wl_air.values - wl_feat)
    if d.min() < tol_nm:
        r = cat_all.iloc[int(np.argmin(d))]
        return f"{r.species} {r.wl_air:.2f}"
    b = bands_overlapping(bands, wl_feat)
    if b:
        return "N2? " + "/".join(b[:2])
    return "unidentified"


def scaled_reference_fit(xm, ym, sp_ref: Spectrum, c, fw, cfg, extra=True):
    """mix ~ a * ref(lambda - d) + b0 + b1 (lambda - c) [+ A_x * G(lambda - x_x)]

    a >= 0; d scanned over +-ref_shift_fwhm x FWHM.  With extra=True an
    additional instrument-width Gaussian (A_x >= 0) is scanned over
    +-1.5 FWHM: a contaminating rotational line under the Ar line shows up
    as a significant A_x, even where the line is only 2-3 pixels wide and
    raw residual peaks are dominated by sampling noise.  Returns dict."""
    ok = ~sp_ref.gapmask
    wr, Ir = sp_ref.wl[ok], sp_ref.I[ok]
    shifts = np.linspace(-cfg["ref_shift_fwhm"], cfg["ref_shift_fwhm"], 9) * fw
    best = None
    for d in shifts:
        yp = np.interp(xm - d, wr, Ir)
        X = np.column_stack([yp, np.ones_like(xm), xm - c])
        sol = lsq_linear(X, ym, bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), method="bvls")
        resid = ym - X @ sol.x
        cost = np.sum(resid ** 2)
        if best is None or cost < best["cost"]:
            best = dict(cost=cost, resid=resid, scale=sol.x[0], shift=d, yp=yp, base=sol.x[1:])
    flank = np.abs(xm - c) > 1.5 * fw
    r0 = best["resid"]
    sig = 1.4826 * np.median(np.abs(r0[flank] - np.median(r0[flank]))) if flank.sum() >= 4 else 1.4826 * np.median(np.abs(r0 - np.median(r0)))
    best["sigma"] = max(sig, 1e-12)
    best["at_shift_limit"] = abs(abs(best["shift"]) - shifts.max()) < 1e-12
    best.update(extra_amp=0.0, extra_pos=np.nan, extra_gain=0.0)
    if extra:
        # fit an instrument-width Gaussian (+ linear baseline) to the RESIDUAL of
        # the scaled-reference fit; the reference scale is kept fixed so the
        # Gaussian can only explain what the pure-Ar line shape cannot.
        r0 = best["resid"]
        base_cost = np.sum(r0 ** 2)
        bx = None
        for xx in np.arange(c - 1.5 * fw, c + 1.5 * fw + 1e-9, fw / 6):
            X = np.column_stack([np.ones_like(xm), xm - c, gauss(xm, 1.0, xx, fw)])
            sol = lsq_linear(X, r0, bounds=([-np.inf, -np.inf, 0], [np.inf] * 3), method="bvls")
            cost = np.sum((r0 - X @ sol.x) ** 2)
            if bx is None or cost < bx[0]:
                bx = (cost, sol.x[2], xx)
        best.update(extra_amp=bx[1], extra_pos=bx[2], extra_gain=1 - bx[0] / max(base_cost, 1e-30))
    return best


def _moment_width(x, y, c, fw):
    """Second-moment width (in FWHM units of a Gaussian) over +-1.5 FWHM after
    removing a linear baseline through the window ends."""
    m = np.abs(x - c) < 1.5 * fw
    if m.sum() < 4:
        return np.nan
    xx, yy = x[m], y[m]
    b = np.polyfit([xx[0], xx[-1]], [yy[0], yy[-1]], 1)
    yy = np.clip(yy - np.polyval(b, xx), 0, None)
    if yy.sum() <= 0:
        return np.nan
    cbar = np.sum(xx * yy) / yy.sum()
    var = np.sum((xx - cbar) ** 2 * yy) / yy.sum()
    return 2.3548 * np.sqrt(max(var, 0))


def contamination_check(res_mix, res_pure, sp_mix: Spectrum, sp_pure: Spectrum,
                        cat_all, bands, cfg):
    """Adds contamination metrics + flag to res_mix.

    Ar I rows (reference = pure Ar spectrum):
      * excess_frac / excess_z : signed residual area in +-1.5 FWHM after
        scaling+shifting the pure-Ar line onto the mixture, / line area
      * extra_amp_frac         : amplitude of an added instrument-width
        Gaussian under the line (scanned +-1.5 FWHM), / line height
      * n_extra_near           : resolved residual peaks 1.5-3 FWHM away
      * width_ratio            : second-moment width mix / pure
      * ratio_mix_pure, branching_dev : intensity ratio and its deviation
        from other lines sharing the upper level (catches coincident blends)
      * absent in pure Ar      : the 'Ar' line only exists with N2 present
    N I rows (no N-free reference): width vs instrument, unexplained
    features in the fit window, presence in the pure-Ar spectrum.
    band_overlap is informational (N2 band region by position) and does not
    by itself change the flag."""
    cols = dict(ref_scale=np.nan, ref_shift=np.nan, excess_frac=np.nan, excess_z=np.nan,
                resid_core_frac=np.nan, extra_amp_frac=np.nan, extra_pos=np.nan, extra_gain=np.nan,
                extra_peaks="", n_extra_near=0, width_ratio=np.nan, pedestal_change=np.nan,
                ratio_mix_pure=np.nan, branching_dev=np.nan, band_overlap="",
                contamination="n/a", contam_reason="")
    for k, v in cols.items():
        res_mix[k] = v
    detected = ("ok", "weak", "blend", "saturated")
    quant = ("ok", "blend", "saturated")
    for i, r in res_mix.iterrows():
        if r.status in ("gap", "edge", "no_data"):
            continue
        fw, c = r.fwhm_group, r.centre_fit
        bo = bands_overlapping(bands, c, 1.5 * fw)
        res_mix.at[i, "band_overlap"] = "; ".join(bo)
        hard, soft, info = [], [], []
        rp = res_pure.loc[i] if i in res_pure.index else None
        if r.status not in detected:
            continue
        # ---- presence / ratio in pure Ar -------------------------------------
        if rp is not None:
            if rp.status in detected and np.isfinite(rp.area) and rp.area > 0 and np.isfinite(r.area):
                res_mix.at[i, "ratio_mix_pure"] = r.area / rp.area
            if r.species == "Ar I" and rp.status == "not_detected" and r.status in quant:
                hard.append("absent in pure Ar")
            if r.species == "N I" and rp.status in quant and r.status in quant and rp.area > 0.3 * r.area:
                soft.append(f"present in pure Ar (pure/mix = {rp.area / r.area:.2f})")
        # ---- reference-scaled residual (Ar I) --------------------------------
        if r.species == "Ar I":
            xm, ym = sp_mix.window(c - 3 * fw, c + 3 * fw)
            if len(xm) >= 8:
                sf = scaled_reference_fit(xm, ym, sp_pure, c, fw, cfg)
                resid, sig = sf["resid"], sf["sigma"]
                core = np.abs(xm - c) < 1.5 * fw
                dx = np.gradient(xm)
                excess = np.sum(resid[core] * dx[core])
                z = np.sum(resid[core]) / (sig * np.sqrt(core.sum()))
                area = r.area if np.isfinite(r.area) and r.area > 0 else np.nan
                ef = excess / area if np.isfinite(area) else np.nan
                hmin = max(cfg["resid_peak_sigma"] * sig, cfg["resid_peak_frac"] * r.height)
                pk, _ = find_peaks(resid, height=hmin, distance=2)
                labels, n_near = [], 0
                for p in pk:
                    dd = abs(xm[p] - c)
                    if 1.5 * fw <= dd < 3 * fw:
                        n_near += 1
                        labels.append(f"{xm[p]:.3f}:{_label_feature(xm[p], cat_all, bands, 1.0 * fw)}")
                xa = sf["extra_amp"] / max(r.height, 1e-30)
                res_mix.at[i, "ref_scale"] = sf["scale"]
                res_mix.at[i, "ref_shift"] = sf["shift"]
                res_mix.at[i, "excess_frac"] = ef
                res_mix.at[i, "excess_z"] = z
                res_mix.at[i, "resid_core_frac"] = np.sqrt(np.mean(resid[core] ** 2)) / max(r.height, 1e-30)
                res_mix.at[i, "extra_amp_frac"] = xa
                res_mix.at[i, "extra_pos"] = sf["extra_pos"]
                res_mix.at[i, "extra_gain"] = sf["extra_gain"]
                res_mix.at[i, "extra_peaks"] = "; ".join(labels)
                res_mix.at[i, "n_extra_near"] = n_near
                if np.isfinite(ef) and z > 3:
                    if ef > cfg["excess_contam"]:
                        hard.append(f"excess {ef:.2f} ({z:.1f} sigma)")
                    elif ef > cfg["excess_suspect"]:
                        soft.append(f"excess {ef:.2f} ({z:.1f} sigma)")
                if (xa > cfg["extra_amp_frac"] and sf["extra_amp"] > cfg["resid_peak_sigma"] * sig
                        and sf["extra_gain"] > cfg["extra_gain_min"]):
                    lab = _label_feature(sf["extra_pos"], cat_all, bands, 0.5 * fw)
                    hard.append(f"hidden line {xa:.2f} x height at {sf['extra_pos']:.3f} ({lab})")
                if n_near:
                    soft.append(f"{n_near} resolved feature(s) 1.5-3 FWHM away")
                if sf["at_shift_limit"]:
                    soft.append("reference shift at scan limit")
            # width: second moment mix vs pure (same grid, both aligned by fit)
            if rp is not None and r.status in quant and rp.status in quant:
                xp_, yp_ = sp_pure.window(c - 3 * fw, c + 3 * fw)
                wm = _moment_width(xm, ym, c, fw)
                wp = _moment_width(xp_, yp_, rp.centre_fit, fw)
                if np.isfinite(wm) and np.isfinite(wp) and wp > 0:
                    res_mix.at[i, "width_ratio"] = wm / wp
                    if wm / wp > cfg["width_ratio_contam"]:
                        hard.append(f"width x{wm / wp:.2f} vs pure")
                dp = r.pedestal_frac - rp.pedestal_frac
                res_mix.at[i, "pedestal_change"] = dp
                if np.isfinite(dp) and dp > cfg["pedestal_suspect"]:
                    soft.append(f"pedestal +{dp:.2f} of height")
        # ---- N I: shape vs instrument, unexplained features ------------------
        if r.species == "N I":
            xm, ym = sp_mix.window(c - 3 * fw, c + 3 * fw)
            wm = _moment_width(xm, ym, c, fw)
            if np.isfinite(wm) and r.status in quant:
                res_mix.at[i, "width_ratio"] = wm / fw
                if wm / fw > cfg["width_ratio_contam"] * 1.15:      # moment width of an isolated line ~1.0-1.15
                    hard.append(f"width x{wm / fw:.2f} vs instrument")
            if r.n_resid_peaks:
                soft.append(f"{int(r.n_resid_peaks)} unexplained feature(s) in fit window")
        if bo:
            info.append("N2 band region: " + "/".join(bo[:2]))
        # ---- classify --------------------------------------------------------
        if hard:
            flag = "contaminated"
        elif soft:
            flag = "suspect"
        else:
            flag = "clean"
        if r.status == "weak":
            flag += "?"                        # low SNR: tests have little power
        res_mix.at[i, "contamination"] = flag
        res_mix.at[i, "contam_reason"] = "; ".join(hard + soft + info)
    # ---- branching-ratio consistency (same upper level, Ar I) ----------------
    ar = res_mix[(res_mix.species == "Ar I") & res_mix.ratio_mix_pure.notna() & res_mix.status.isin(quant)]
    for up, grp in ar.groupby("upper"):
        if len(grp) < 2:
            continue
        med = np.median(grp.ratio_mix_pure)
        for i, rr in grp.iterrows():
            dev = rr.ratio_mix_pure / med - 1
            res_mix.at[i, "branching_dev"] = dev
            if abs(dev) > cfg["branching_dev"]:
                reason = f"mix/pure ratio {rr.ratio_mix_pure:.2f} vs {med:.2f} for other {up} lines"
                res_mix.at[i, "contam_reason"] = "; ".join(filter(None, [res_mix.at[i, "contam_reason"], reason]))
                if res_mix.at[i, "contamination"].startswith("clean"):
                    res_mix.at[i, "contamination"] = "suspect" + ("?" if rr.status == "weak" else "")
    return res_mix


def branching_ratio_check(res, max_sep_nm=15.0, statuses=("ok", "blend", "saturated"), dev_max=0.5):
    """Lines sharing an upper level must have I_i / I_j = A_i / A_j (same N_k,
    optically thin, similar instrument response -> pairs closer than
    max_sep_nm only).  Adds columns A_branch_dev / A_branch_partner and a
    soft reason to lines that deviate by more than dev_max."""
    res["A_branch_dev"] = np.nan
    res["A_branch_partner"] = ""
    sub = res[res.status.isin(statuses) & res.Aki.notna() & (res.Aki > 0) & res.area.notna() & (res.area > 0)]
    for up, grp in sub.groupby(["species", "upper"]):
        if len(grp) < 2:
            continue
        for i, ri in grp.iterrows():
            devs = []
            for j, rj in grp.iterrows():
                if i == j or abs(ri.wl_air - rj.wl_air) > max_sep_nm:
                    continue
                devs.append(((ri.area / rj.area) / (ri.Aki / rj.Aki) - 1, rj.wl_air))
            if devs:
                dev, partner = min(devs, key=lambda t: abs(t[0]))     # most consistent partner
                res.at[i, "A_branch_dev"] = dev
                res.at[i, "A_branch_partner"] = f"{partner:.2f}"
                if abs(dev) > dev_max and "contamination" in res:
                    reason = f"I/A branching vs {partner:.2f}: {dev:+.0%}"
                    res.at[i, "contam_reason"] = "; ".join(filter(None, [res.at[i, "contam_reason"], reason]))
                    if str(res.at[i, "contamination"]).startswith("clean"):
                        res.at[i, "contamination"] = "suspect" + ("?" if ri.status == "weak" else "")
    return res


def unidentified_features(sp: Spectrum, cat_all, bands, fwhm_fn, cfg, snr_min=8.0, other: Spectrum = None):
    """Peaks in a spectrum that are not within 1 FWHM of any catalogue line.
    If `other` (e.g. the pure-Ar spectrum) is given, the peak height there is
    reported too, so N-related features stand out."""
    resid = sp.I - median_filter(sp.I, 41)
    ok = ~sp.gapmask & (sp.wl > cfg["wl_min"]) & (sp.wl < cfg["wl_max"])
    # local sigma in 5 nm bins
    sig = np.full_like(sp.I, np.nan)
    edges = np.arange(cfg["wl_min"], cfg["wl_max"] + 5, 5.0)
    for a, b in zip(edges[:-1], edges[1:]):
        m = ok & (sp.wl >= a) & (sp.wl < b)
        if m.sum() > 20:
            sig[m] = 1.4826 * np.median(np.abs(resid[m] - np.median(resid[m])))
    pk, props = find_peaks(np.where(ok, resid, 0), height=1, distance=2)
    rows = []
    cw = cat_all.wl_air.values
    for p in pk:
        if not np.isfinite(sig[p]) or resid[p] < snr_min * sig[p]:
            continue
        fw = fwhm_fn(sp.wl[p])
        d = np.abs(cw - sp.wl[p])
        if d.min() < 1.0 * fw:
            continue
        rows.append(dict(wl=sp.wl[p], height=resid[p], snr=resid[p] / sig[p],
                         nearest_catalogue=f"{cat_all.iloc[int(np.argmin(d))].species} {cw[np.argmin(d)]:.3f}",
                         nearest_dist_nm=d.min(),
                         band_overlap="; ".join(bands_overlapping(bands, sp.wl[p])),
                         height_in_other=(np.interp(sp.wl[p], other.wl, other.I) - np.interp(sp.wl[p], other.wl, median_filter(other.I, 41))) if other is not None else np.nan))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 8. Matrices
# ----------------------------------------------------------------------------
def build_matrix(res, levels, value="area", statuses=("ok", "weak", "blend", "saturated"),
                 only_model_levels=True, order="energy", drop_empty=False):
    """lower (rows) x upper (cols) matrix of line intensity; NaN where no line
    is in the catalogue, 0 where the line is in the catalogue but not usable
    (gap / edge / not detected).  drop_empty=True also removes rows/columns
    with no *usable* line (used for the large, sparse N I matrix)."""
    labs = levels.sort_values("energy_eV" if order == "energy" else "id").index.tolist()
    sub = res[res.lower_in_model & res.upper_in_model] if only_model_levels else res
    M = pd.DataFrame(np.nan, index=labs, columns=labs)
    W = pd.DataFrame("", index=labs, columns=labs)
    F = pd.DataFrame("", index=labs, columns=labs)
    for r in sub.itertuples():
        if r.lower not in M.index or r.upper not in M.columns:
            continue
        v = getattr(r, value) if r.status in statuses else 0.0
        if not np.isfinite(v):
            v = 0.0
        if np.isnan(M.at[r.lower, r.upper]) or v > M.at[r.lower, r.upper]:
            M.at[r.lower, r.upper] = v
            W.at[r.lower, r.upper] = f"{r.wl_air:.2f}"
            F.at[r.lower, r.upper] = f"{r.status}/{getattr(r, 'contamination', '')}"
    keep_r = M.notna().any(axis=1); keep_c = M.notna().any(axis=0)
    if drop_empty:
        keep_r &= (M > 0).any(axis=1); keep_c &= (M > 0).any(axis=0)
    return M.loc[keep_r, keep_c], W.loc[keep_r, keep_c], F.loc[keep_r, keep_c]


def plot_matrix(M, W, F, levels, title, path, ratio=False):
    fig_w = max(8, 0.55 * M.shape[1] + 3); fig_h = max(6, 0.5 * M.shape[0] + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vals = M.to_numpy(float)
    data = np.log10(np.where(vals > 0, vals, np.nan))
    if ratio:
        vmax = max(0.3, np.nanpercentile(np.abs(data), 90)) if np.isfinite(data).any() else 1   # robust: one outlier should not flatten the rest
        im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        cb = fig.colorbar(im, ax=ax, fraction=0.03); cb.set_label("log10(I_mix / I_pure)")
    else:
        im = ax.imshow(data, cmap="viridis", aspect="auto")
        cb = fig.colorbar(im, ax=ax, fraction=0.03); cb.set_label(f"log10({title.split()[2] if len(title.split())>2 else 'I'})")
    yy, xx = np.where(vals == 0)
    ax.scatter(xx, yy, marker="x", s=25, c="0.6", lw=0.8)
    norm = im.norm
    for (yi, xi) in zip(*np.where(np.isfinite(data))):
        flag = F.iat[yi, xi]
        v = norm(data[yi, xi])
        col = ("k" if abs(v - 0.5) < 0.35 else "w") if ratio else ("w" if v < 0.6 else "k")   # readable on pale / bright cells
        if "contaminated" in flag:
            col = "red"
        elif "suspect" in flag:
            col = "orange"
        elif "weak" in flag:
            col = "0.85" if v < 0.6 else "0.3"
        ax.text(xi, yi, W.iat[yi, xi], ha="center", va="center", fontsize=6, color=col,
                fontweight="bold" if col in ("red", "orange") else "normal")
    tag = lambda c: levels.at[c, "paschen"] or levels.at[c, "desc"]
    xl = [f"{c}\n{tag(c)}" for c in M.columns]
    yl = [f"{c} {tag(c)}" for c in M.index]
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(xl, rotation=90, fontsize=7)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(yl, fontsize=7)
    ax.set_xlabel("upper level"); ax.set_ylabel("lower level")
    ax.set_title(title + "\ncell text = air wavelength; x = catalogue line not usable (gap/edge/undetected); "
                 "grey = weak; orange = suspect; red = contaminated", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------
# 9. Plots
# ----------------------------------------------------------------------------
def plot_overview(sp_pure, sp_mix, res_mix, bands, path,
                  panels=((250, 400), (400, 560), (560, 720), (720, 870)), label_snr=20):
    fig, axs = plt.subplots(len(panels), 1, figsize=(18, 4.0 * len(panels)))
    for ax, (a, b) in zip(axs, panels):
        floor = None
        for sp, col, lab, z in ((sp_pure, "0.3", sp_pure.name, 2), (sp_mix, "crimson", sp_mix.name, 1)):
            m = (sp.wl >= a) & (sp.wl <= b)
            y = np.where(sp.gapmask[m], np.nan, sp.I[m])
            if floor is None:   # ~ +1 sigma of the (background-subtracted) pure-Ar noise in this panel
                floor = max(np.nanpercentile(y, 84), 1.0)
            ax.plot(sp.wl[m], np.clip(y, floor, None), color=col, lw=0.5, label=lab, zorder=z, alpha=0.85)
        for g0, g1 in sp_mix.gaps:
            if g1 > a and g0 < b:
                ax.axvspan(max(g0, a), min(g1, b), color="0.88", zorder=0)
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor * 0.9)
        ymax = ax.get_ylim()[1]
        det = res_mix[(res_mix.wl_air >= a) & (res_mix.wl_air <= b) & res_mix.status.isin(["ok", "blend", "saturated"])]
        k = 0
        for r in det.sort_values("wl_air").itertuples():
            col = "tab:blue" if r.species == "Ar I" else "tab:green"
            ax.axvline(r.wl_air, ymin=0.90, ymax=0.94, color=col, lw=0.7)
            if r.snr > label_snr:
                ax.text(r.wl_air, ymax * (1.6 if k % 2 else 1.1), f"{r.wl_air:.1f}", rotation=90, fontsize=5,
                        color=col, ha="center", va="bottom", clip_on=False)
                k += 1
        for bd in bands[(bands.head_nm >= a) & (bands.head_nm <= b)].itertuples():
            ax.axvline(bd.head_nm, ymin=0.95, ymax=0.99, color="magenta", lw=0.7)
        ax.set_xlim(a, b); ax.set_ylabel("intensity")
        ax.legend(loc="upper left", fontsize=8)
    axs[0].set_title("ticks: blue = Ar I detected, green = N I detected, magenta = N2/N2+ band heads; grey = order gaps "
                     f"(labels for SNR > {label_snr})", fontsize=9)
    axs[-1].set_xlabel("air wavelength (nm)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_line_diagnostics(res_mix, fits_mix, sp_pure, sp_mix, cat_all, bands, path,
                          statuses=("ok", "weak", "blend", "saturated"), per_page=12, model_only=True):
    """One panel per detected line: pure (grey), mixture (red), mixture group fit
    (blue dashed) with residual below, N2 band regions (magenta shading),
    catalogue positions (green = N I, blue = other Ar I)."""
    sel = res_mix[res_mix.status.isin(statuses)]
    if model_only:
        sel = sel[(sel.species != "Ar I") | (sel.lower_in_model & sel.upper_in_model)]
    sel = sel.sort_values("wl_air")
    with PdfPages(path) as pdf:
        for p0 in range(0, len(sel), per_page):
            chunk = sel.iloc[p0:p0 + per_page]
            fig, axs = plt.subplots(3, 4, figsize=(17, 11))
            for ax, (i, r) in zip(axs.flat, chunk.iterrows()):
                fw = r.fwhm_group; c = r.centre_fit
                lo, hi = c - 5 * fw, c + 5 * fw
                xm, ym = sp_mix.window(lo, hi); xp, yp = sp_pure.window(lo, hi)
                ax.plot(xp, yp, color="0.45", lw=0.9, label="pure Ar")
                ax.plot(xm, ym, color="crimson", lw=0.9, label="Ar/N2")
                gf = fits_mix.get(r.group)
                top = max(ym.max(), 1)
                if gf is not None:
                    m = (gf["x"] >= lo) & (gf["x"] <= hi)
                    ax.plot(gf["x"][m], gf["model"][m], "b--", lw=0.8, label="fit (mix)")
                    off = -0.25 * top
                    ax.plot(gf["x"][m], gf["resid"][m] + off, color="tab:blue", lw=0.6, alpha=0.7)
                    ax.axhline(off, color="tab:blue", lw=0.3)
                # band regions: shade the *union* once (stacked alphas would saturate), list names once
                ivs = sorted((max(bd.lo_nm, lo), min(bd.hi_nm, hi), f"{bd.system}({bd.vu},{bd.vl})")
                             for bd in bands.itertuples() if bd.hi_nm >= lo and bd.lo_nm <= hi)
                merged = []
                for a, b, nm in ivs:
                    if merged and a <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], b)
                    else:
                        merged.append([a, b])
                for a, b in merged:
                    ax.axvspan(a, b, color="magenta", alpha=0.08, lw=0)
                if ivs:
                    ax.text(0.01, 0.97, "N2: " + ", ".join(nm for _, _, nm in ivs[:4]) + (" ..." if len(ivs) > 4 else ""),
                            transform=ax.transAxes, fontsize=5, color="magenta", va="top")
                for nl in cat_all[(cat_all.wl_air >= lo) & (cat_all.wl_air <= hi)].itertuples():
                    if abs(nl.wl_air - r.wl_air) < 1e-6:
                        continue
                    ax.axvline(nl.wl_air, color="tab:green" if nl.species == "N I" else "tab:blue", lw=0.5, alpha=0.5)
                ax.axvline(c, color="k", lw=0.5, ls=":")
                flag = getattr(r, "contamination", "")
                ttl = f"{r.species} {r.wl_air:.2f}  {r.lower} -> {r.upper}\n{r.status}, SNR {r.snr:.0f}, {flag}"
                reason = getattr(r, "contam_reason", "")
                if isinstance(reason, str) and reason:
                    ttl += f"\n{reason[:70]}"
                ax.set_title(ttl, fontsize=7, color={"contaminated": "red", "suspect": "darkorange"}.get(flag, "k"))
                ax.tick_params(labelsize=6); ax.set_xlim(lo, hi)
            for ax in axs.flat[len(chunk):]:
                ax.axis("off")
            axs.flat[0].legend(fontsize=6)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


# ----------------------------------------------------------------------------
# 10. Actinometry helper
# ----------------------------------------------------------------------------
ACTINOMETRY_N = [746.83, 744.23, 742.36, 821.63, 821.07, 818.49, 820.04, 822.31, 824.24,
                 868.03, 868.34, 862.92, 859.40, 856.77]
ACTINOMETRY_AR = [750.39, 751.47, 811.53, 763.51, 772.38, 772.42, 794.82, 800.62, 801.48, 826.45,
                  840.82, 842.46, 852.14, 866.79, 696.54, 706.72, 738.40, 727.29, 714.70, 667.73]


def actinometry_table(res_mix, tol=0.05):
    rows = []
    for wl0 in ACTINOMETRY_N + ACTINOMETRY_AR:
        d = np.abs(res_mix.wl_air - wl0)
        if d.min() > tol:
            continue
        r = res_mix.loc[d.idxmin()]
        rows.append(dict(species=r.species, wl_air=r.wl_air, lower=r.lower, upper=r.upper,
                         lower_desc=r.lower_desc, upper_desc=r.upper_desc, Aki=r.Aki, Ek=r.Ek,
                         status=r.status, snr=r.snr, height=r.height, area=r.area,
                         contamination=r.contamination, contam_reason=r.contam_reason,
                         band_overlap=r.band_overlap))
    tab = pd.DataFrame(rows)
    for ref in (750.39, 751.47):
        m = np.abs(tab.wl_air - ref) < tol
        if m.any() and tab.loc[m, "status"].iloc[0] in ("ok", "blend", "saturated"):
            aref = tab.loc[m, "area"].iloc[0]
            tab[f"ratio_to_Ar{ref:.0f}"] = np.where(tab.species == "N I", tab.area / aref, np.nan)
    return tab


# ----------------------------------------------------------------------------
# 11. Persistence  (RESULTS <-> disk, for post-processing scripts)
# ----------------------------------------------------------------------------
_SPECTRUM_KEYS = ("sp_pure", "sp_mix")

def save_results(results, path):
    """Pickle the RESULTS dict for post-processing.

    Two things are handled so the pickle stays portable:
      * fwhm_fn is a closure (not picklable) -> dropped; its parameters are kept
        under 'ifn_fit' so a reader can rebuild it.
      * Spectrum objects are stored as plain dicts, so unpickling does not depend
        on whether this module was run as __main__ or imported."""
    from dataclasses import asdict, is_dataclass
    r = dict(results)
    r.pop("fwhm_fn", None)
    ifn = r.get("ifn")
    if ifn is not None and hasattr(ifn, "attrs"):
        r["ifn_fit"] = dict(ifn.attrs.get("fit", {}))
    for k in _SPECTRUM_KEYS:
        if k in r and is_dataclass(r[k]):
            r[k] = {"__spectrum__": True, **asdict(r[k])}
    with open(path, "wb") as f:
        pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_results(path):
    """Load a pickled RESULTS dict: rebuild the Spectrum objects and re-attach
    fwhm_fn from its stored parameters."""
    with open(path, "rb") as f:
        r = pickle.load(f)
    for k in _SPECTRUM_KEYS:
        v = r.get(k)
        if isinstance(v, dict) and v.get("__spectrum__"):
            v = {kk: vv for kk, vv in v.items() if kk != "__spectrum__"}
            r[k] = Spectrum(**v)
    p = r.get("ifn_fit")
    if p and "a" in p and "b" in p:
        r["fwhm_fn"] = lambda lam, a=p["a"], b=p["b"]: a * lam + b
    return r


# ----------------------------------------------------------------------------
# 12. Driver
# ----------------------------------------------------------------------------
def run(cfg=CONFIG):
    os.makedirs(cfg["outdir"], exist_ok=True)
    out = lambda f: os.path.join(cfg["outdir"], f)
    # levels + catalogues
    ar_lv, ar_lookup = load_levels(cfg["ar_levels"])
    n_lv, n_lookup = load_levels(cfg["n_levels"])
    cat_ar = load_ar_lines(cfg["ar_lines"], ar_lv, ar_lookup)
    cat_n = load_n_lines(cfg["n_lines"], n_lv, n_lookup, wl_max=cfg["wl_max"])
    cat = pd.concat([cat_ar, cat_n], ignore_index=True).sort_values("wl_air").reset_index(drop=True)
    bands = n2_band_catalogue(cfg["wl_min"], cfg["wl_max"])
    cat.to_csv(out("catalogue_lines.csv"), index=False)
    bands.to_csv(out("n2_band_heads.csv"), index=False)
    # spectra
    bg = noise_floor_from_background(cfg["background"], cfg) if cfg.get("background") else None
    sp_pure = load_spectrum(cfg["pure"], "pure Ar", cfg, bg)
    sp_mix = load_spectrum(cfg["mix"], "Ar/N2", cfg, bg)
    # instrument function + wavelength offset from the pure spectrum
    if cfg["resolving_power"]:
        fwhm_fn = lambda lam: lam / cfg["resolving_power"]
        offset0, ifn_tab = 0.0, pd.DataFrame()
    else:
        fwhm_fn, offset_fn, ifn_tab = measure_instrument_function(sp_pure, cat_ar, cfg)
        offset0 = float(offset_fn(0))
        ifn_tab.to_csv(out("instrument_function_lines.csv"), index=False)
        print("Instrument function:", {k: round(float(v), 5) for k, v in ifn_tab.attrs["fit"].items()})
    # measurement
    res_pure, fits_pure = measure_lines(sp_pure, cat, fwhm_fn, cfg, "pure", offset0)
    res_mix, fits_mix = measure_lines(sp_mix, cat, fwhm_fn, cfg, "mix", offset0)
    res_mix = contamination_check(res_mix, res_pure, sp_mix, sp_pure, cat, bands, cfg)
    res_mix = branching_ratio_check(res_mix, dev_max=cfg["branching_A_dev"])
    res_pure = branching_ratio_check(res_pure, dev_max=cfg["branching_A_dev"])
    res_pure.to_csv(out("lines_pure.csv"), index=False)
    res_mix.to_csv(out("lines_mix.csv"), index=False)
    unid = unidentified_features(sp_mix, cat, bands, fwhm_fn, cfg, other=sp_pure)
    unid.to_csv(out("unidentified_features_mix.csv"), index=False)
    # matrices
    val = cfg["matrix_value"]
    Mp, Wp, Fp = build_matrix(res_pure[res_pure.species == "Ar I"], ar_lv, val)
    Mm, Wm, Fm = build_matrix(res_mix[res_mix.species == "Ar I"], ar_lv, val)
    Mp.to_csv(out("matrix_Ar_pure.csv")); Mm.to_csv(out("matrix_Ar_mix.csv"))
    Mp_al = Mp.reindex_like(Mm)
    R = Mm / Mp_al
    R = R.where((Mm > 0) & (Mp_al > 0))
    R.to_csv(out("matrix_Ar_ratio_mix_over_pure.csv"))
    plot_matrix(Mp, Wp, Fp, ar_lv, f"Ar I {val} - pure Ar", out("matrix_Ar_pure.png"))
    plot_matrix(Mm, Wm, Fm, ar_lv, f"Ar I {val} - Ar/N2", out("matrix_Ar_mix.png"))
    plot_matrix(R, Wm, Fm, ar_lv, f"Ar I {val} ratio Ar/N2 over pure", out("matrix_Ar_ratio.png"), ratio=True)
    Mn, Wn, Fn = build_matrix(res_mix[res_mix.species == "N I"], n_lv, val, drop_empty=True)
    Mn.to_csv(out("matrix_N_mix.csv"))
    if Mn.size:
        plot_matrix(Mn, Wn, Fn, n_lv, f"N I {val} - Ar/N2", out("matrix_N_mix.png"))
    # plots
    plot_overview(sp_pure, sp_mix, res_mix, bands, out("overview.png"))
    if cfg["make_pdf"]:
        plot_line_diagnostics(res_mix, fits_mix, sp_pure, sp_mix, cat, bands, out("line_diagnostics.pdf"),
                              model_only=cfg["pdf_model_levels_only"])
    act = actinometry_table(res_mix)
    act.to_csv(out("actinometry_candidates.csv"), index=False)
    # summary
    print("\n=== status counts (mixture) ===")
    print(res_mix.groupby(["species", "status"]).size().unstack(fill_value=0))
    print("\n=== contamination (mixture, detected lines) ===")
    print(res_mix[res_mix.status.isin(["ok", "weak", "blend", "saturated"])].groupby(["species", "contamination"]).size().unstack(fill_value=0))
    results = dict(cat=cat, bands=bands, sp_pure=sp_pure, sp_mix=sp_mix, res_pure=res_pure, res_mix=res_mix,
                fits_pure=fits_pure, fits_mix=fits_mix, matrices=dict(pure=Mp, mix=Mm, ratio=R, N=Mn),
                actinometry=act, unidentified=unid, levels=dict(Ar=ar_lv, N=n_lv), fwhm_fn=fwhm_fn,
                offset0=offset0, ifn=ifn_tab)
    save_results(results, out("results.pkl"))
    print("saved", out("results.pkl"))
    return results


if __name__ == "__main__":
    RESULTS = run(CONFIG)