# -*- coding: utf-8 -*-
"""
QEProCalibration.py

Wavelength calibration, instrument function and relative intensity calibration of the Ocean Insight QE Pro
spectra (1044 px, 195-993 nm, 0.76 nm/px, no intensity calibration) from the Ar I lines.  Data: the 1.5 Torr
power sweeps recorded with the QE Pro and the echelle on the same plasma at the same time (the two PCs' clocks
are ~11 min apart): pure Ar and 5 % N2, 10-90 W, 3 QE Pro spectra (5 echelle) per power, plasma-off frames ref*.

0. Spectra: the plasma-off frames of a sweep are averaged and subtracted.  The QE Pro spectra are scan averages:
   read noise 0.3 counts, repeats agree to 0.03 % - statistics never limit; LSF and blends do.  Pixels whose raw
   count exceeds SAT_COUNTS are left out of every fit: at 80 W, 5 % N2 two exposures (ratio 1.984) agree within
   0.5 % up to 1e5 counts, the peak pixel is 1 % low at 1.45e5 and 4 % low at 1.9e5 (811 nm).
1. Instrument function (LSF): split Gaussian (ln sigma linear between LSF_KNOTS, one asymmetry) + Lorentzian
   wings, integrated over each pixel.  The position of a line is the centroid of its split Gaussian, so the fitted line
   positions do not depend on how the shape is parametrised (Gaussian instead of Lorentzian wings moves them by
   < 0.03 nm).  Fitted jointly to CAL_LINES in the mean pure-Ar spectrum (all powers), one shift per line, every
   CR-model line around it a free component (reference CR spectrum below), linear baseline per window, and to the N2
   band windows BAND_LSF_WINDOWS (305-545 nm) of a 5 % N2 spectrum against the echelle spectrum of the same plasma
   convolved with the LSF (one shift per window): the Ar I lines below 480 nm are too weak and blended to fix the
   width there (alone they gave a sigma 3x too small in the UV / blue).
2. Wavelength calibration: shift of each line (position in the file's wavelength axis - catalogue air wavelength)
   fitted with a polynomial in the pixel index (WL_DEG) -> corrected wavelength of every pixel.  Below the Ar I lines
   the shifts come from the N2 band windows of the LSF fit (echelle wavelength scale; the 315.9, 337.1, 353.7/357.7,
   371.0-380.5, 394.3, 399.8/405.9 nm heads of the second positive system and more).  The LSF is then refitted on the
   corrected axis.  Check: N2 C -> B templates from the ExoMol line positions (absolute) fitted in N2_BAND_WINDOWS
   (n2_band_wavelength_points: shift and rotational temperature, n2_band_check.csv).
3. Line areas (measure_spectrum): every feature (one CR-model line, or several closer than MERGE_NM) is fitted in
   its window (+- WINDOW_NM around its lines) by linear least squares: one LSF component per CR-model line of the
   window above REL_MIN of the reference spectrum (lines closer than MERGE_COMP_NM are one component, weighted by
   the reference spectrum) and a linear baseline -> area [counts] (sum over its lines) and its error.
   Reference spectrum: CR model (AnalyticFivePFit table, pure Ar, 1.5 Torr, REF_TG, E/N = REF_EN, Ne = n_c),
   observed intensities in energy units; only used to choose and weight the components.
4. Relative response R(lambda) = QE Pro counts / echelle units (ln R, cubic spline on RESP_KNOTS):
   a) echelle transfer (both sweeps): QE Pro area / echelle area of the same CR lines (groups of lines closer than
      MERGE_TRANSFER_NM summed: blends the LSF cannot split; echelle: CRFitNeTe's group fitter, every catalogue line
      of the window), per power; groups whose echelle lines are all measured (status ok/weak/blend: not in an order
      gap, at an order edge or saturated) and that hold >= MIN_PURITY of the reference flux of their QE Pro window.
      One free scale per sweep and power.
   b) branching ratios: same-upper-level pairs (BR_PAIRS; escape factors of the reference CR spectrum, pairs with
      a line below BR_MIN_ETA left out, BR_ETA_ERR of the trapping correction as error):
      ln(Q1/Q2) - ln(I1/I2)_model = ln R(lambda1) - ln R(lambda2).  Needs no echelle, reaches beyond 870 nm (922.45,
      935.42, 965.78, 978.45) and is done for both sweeps.
   Echelle lines of the 5 % N2 sweep are transferred only above LINE_TRANSFER_MIN_NM (there the QE Pro windows of
   the 5p lines hold mostly N2 bands): the blue comes from the 5p line groups of the pure-Ar sweep.  The UV (no Ar
   I lines) comes from windows of the whole spectrum (echelle convolved with the QE Pro LSF, SPEC_RANGE).  With the
   windows up to 545 nm (they dominate by number) the 5p groups came out 0.15-0.6 too bright against the red lines.
   The response used later (response.csv) is the joint fit of a) and b) (b) only gives differences).

Output in Experimental_Data/Output/QEProComparison/calibration/:
  wavelength_calibration.csv (pixel, wl_file, wl_cal), wavelength_lines.csv (line shifts and residuals),
  lsf.json (LSF parameters), response.csv (wl, ln_R, R, err), response_points.csv (transfer and branching data),
  echelle_lines_<sweep>.csv (echelle line fits, cached), figures/*.png
Run from Spyder (F5) or python SmallAnalysisScripts/QEProCalibration.py
"""
import contextlib
import io
import json
import os
import re
import sys
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import AnalyticFivePFit as A                        # noqa: E402
import EchelleSaturationCheck as esc                # noqa: E402

L, crf, he = A.L, A.crf, A.he
als = crf.als
arn = als.arn

# ---- settings ---------------------------------------------------------------------------
QEPRO_DIR = os.path.join(crf.ROOT_DIR, "Experimental_Data", "QEProData", "QEProEchelleComparision_QEProData")
ECHELLE_DIR = os.path.join(crf.ROOT_DIR, "Experimental_Data", "EchelleData", "EchelleQEProComparisionData")
SWEEPS = {   # sweep -> QE Pro folder, echelle folder of the .spa exports (and of the raw .spe frames, if elsewhere),
             # % N2 (both at 1500 mTorr)
    "pure_Ar": dict(qepro="PowerSweep1_5__Torr_PureArgon", echelle="1500mTorr_PureArgon/SPAFiles",
                    echelle_spe="1500mTorr_PureArgon", N2=0.0, label="pure Ar"),
    "N2_5pct": dict(qepro="PowerSweep1_5__torr_5_PercentNitrogen", echelle="1500mTorr_5_Percent_Nitrogen", N2=5.0,
                    label="5 % N$_2$"),
}
P_MTORR = 1500.0
OUTDIR = os.path.join(crf.ROOT_DIR, "Experimental_Data", "Output", "QEProComparison", "calibration")
FIGDIR = os.path.join(OUTDIR, "figures")
SAT_COUNTS = 1.2e5                                  # raw counts: pixels above are left out (nonlinear)
NOISE = dict(read2=0.1, gain=0.002, lsf=0.01, lsf_peak=0.005)   # pixel variance read2 + gain C + (lsf C)^2
                                                    # + (lsf_peak x window maximum)^2 [counts^2]: the measured scatter
                                                    # (0.3 counts, 0.002 C) and the LSF error (1 % of the pixel, 0.5 %
                                                    # of the strongest pixel of the window - the wings)
REF_TG, REF_EN = 650.0, 35.0                        # reference CR spectrum: pure Ar, 1.5 Torr, Tg [K], E/N [Td]
REL_MIN = 3e-3                                      # components: CR lines above this x the strongest of the window
MERGE_COMP_NM = 0.25                                # CR lines closer than this are one LSF component
WINDOW_NM = 4.0                                     # fit window: +- this around the lines of a feature
CAL_LINES = (   # (lines of one window [nm, air]): strong Ar I lines for the LSF and the wavelength calibration
    (394.8979,), (404.4418,), (415.859, 416.418), (419.8317, 420.0674), (425.9362, 426.6286, 427.2169), (430.0101,),
    (433.3561, 433.5338, 434.5168), (451.0733,), (459.6097,), (470.2316,), (641.6307,), (667.7282,), (675.2834,),
    (696.5431,), (706.7218,), (714.7042,), (727.2936,), (738.398,), (750.3869, 751.4652), (763.5106,),
    (772.3761, 772.4207), (794.8176,), (800.6157, 801.4786), (810.3693, 811.5311), (826.4522,),
    (840.821, 842.4648), (852.1442,), (866.7944,), (912.2967,), (922.4499,), (965.7786,))
WL_DEG = 4                                          # polynomial (pixel index) of the wavelength correction
WL_FLOOR = 0.02                                     # nm, systematic error added to every point (3-sigma clipping)
BAND_LSF = dict(sweep="N2_5pct", P=50.0)           # N2 band windows of the LSF / wavelength fit: QE Pro spectrum of
BAND_LSF_WINDOWS = ((305.0, 318.5), (329.0, 338.5), (346.0, 359.0), (365.0, 382.0), (386.0, 398.0), (398.0, 410.0),
                    (412.0, 424.0), (430.0, 448.0), (500.0, 520.0), (520.0, 545.0))   # this power against the echelle
                                                    # spectrum convolved with the LSF (no echelle order gaps < 540 nm)
EXOMOL_N2 = os.path.join(crf.ROOT_DIR, "MolecularFitting", "EXOMOL", "14N2")   # ExoMol 14N2 (WCCRMT): C <-> B only
N2_CB_A_MIN = 1e2                                   # s^-1, C -> B lines kept
N2_BAND_SWEEP = "N2_5pct"                           # spectra of the N2 C-B wavelength points
N2_BAND_WINDOWS = ((309.5, 318.5), (329.0, 338.5), (346.0, 359.0), (365.0, 382.0))   # nm: 3-2/2-1/1-0, 2-2/1-1/0-0,
                                                    # 2-3/1-2/0-1, 3-5/2-4/1-3/0-2 heads (no Ar I line of note): check of
                                                    # the calibration with the ExoMol C-B line positions
N2_TROT_GRID = np.arange(250.0, 1501.0, 50.0)       # K
N2_SHIFTS = np.arange(-0.25, 0.251, 0.01)          # nm, searched per window
N2_BAND_ERR_MIN = 0.01                              # nm, error floor of a band window shift
FINE_STEP = 0.01                                    # nm, fine grid of the convolutions (fine_convolve)
CONV_BLOCK_NM = 10.0                                # ... LSF taken constant within blocks of this width
LSF_KNOTS = (310.0, 420.0, 560.0, 700.0, 830.0, 970.0)   # nm: ln sigma linear in lambda between these
LSF_START = dict(knots=list(LSF_KNOTS), ls=[float(np.log(0.45))] * len(LSF_KNOTS), a0=0.3, a1=0.0, f=0.4, g=0.4)
                                                    # ln sigma at the knots [nm], asymmetry A_MAX tanh(a0 + a1 u)
                                                    # (u = (lambda - 800) / 100), Lorentz fraction and HWHM [nm]
A_MAX = 0.8                                         # |asymmetry| < A_MAX at every wavelength
LSF_FIT = ("ls", "a0", "f", "g")                   # LSF parameters fitted; a1 = 0 (one asymmetry): with a slope
                                                    # the asymmetry runs to +-0.7 at the ends, traded against the
                                                    # line shifts.  sigma free at the knots: a sigma exponential in
                                                    # lambda fitted to the Ar I lines alone came out 3x too narrow in
                                                    # the UV / blue (isolated lines and the N2 bands: FWHM ~1.1 nm)
LSF_BOUNDS = dict(ls=(np.log(0.05), np.log(2.0)), a0=(-2.0, 2.0), a1=(-1.0, 1.0), f=(0.0, 0.9), g=(0.03, 10.0))
MERGE_NM = 0.30                                     # 'split' features: CR lines closer than this are one feature
FEATURE_RANGES = ((390.0, 475.0), (690.0, 990.0))   # 5p -> 4s and 4p -> 4s (+ 5s/6s/4d -> 4p) lines
FEATURE_MIN = 0.001                                 # feature kept if its reference flux >= this x the strongest
                                                    # line of its range
VALID_ECHELLE = ("ok", "weak", "blend")             # echelle line status usable for the transfer
MIN_PURITY = 0.90                                   # transfer: feature flux / reference flux of its QE Pro window
RESP_KNOTS = (300.0, 340.0, 380.0, 420.0, 460.0, 500.0, 560.0, 640.0, 700.0, 740.0, 780.0, 820.0, 860.0, 900.0,
              940.0, 1000.0)
RESP_SMOOTH = 3.0                                   # weight of the second differences of the ln R spline coefficients
SPEC_RANGE = (300.0, 385.0)                         # window transfer: range (UV: N2 bands, no Ar I lines; above,
                                                    # the Ar I line groups fix the response)
SPEC_WINDOW_NM = 6.0                                # ... window width
SPEC_FLOOR = 0.08                                   # ... error floor per window (LSF / wavelength mismatch at band heads)
BR_FLOOR = 0.04                                     # error floor of a branching point (ln)
BR_PAIRS = ((714.7042, 852.1442), (727.2936, 826.4522), (696.5431, 826.4522), (696.5431, 727.2936),
            (800.6157, 922.4499), (866.7944, 935.422), (738.398, 840.821), (706.7218, 840.821),
            (965.7786, 912.2967), (978.4503, 842.4648))   # same upper level: 4p7, 4p9 (x3), 4p5, 4p4, 4p8 (x2),
                                                          # 4p1, 4p3
BR_MIN_ETA = 0.5                                    # branching pair used for R only if both escape factors >= this;
BR_ETA_ERR = 0.3                                    # ... error of ln(eta1 / eta2) of the reference CR spectrum, relative
MERGE_TRANSFER_NM = 1.8                             # echelle transfer: CR lines closer than this summed (blends the
                                                    # LSF cannot split, e.g. 415.86/416.42, 425-427 nm)
LINE_TRANSFER_MIN_NM = 690.0                        # echelle line transfer of the Ar/N2 sweep only above this: in the
                                                    # blue its QE Pro line windows are N2 bands (windows used)


# ---- 0. spectra ---------------------------------------------------------------------------
def read_qepro(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1]


def _parse(name):
    """'Power_10_W_2', '10_W__3', 'Power_10_W' -> (10.0, rep); 'ref_1', 'ref__2', 'ref3' -> ('ref', rep); else None.
    Echelle names 'Power_10W_1', 'Power_10_W__1' are read the same way."""
    m = re.fullmatch(r"(?:Power_)?(\d+(?:\.\d+)?)_?W(?:_+(\d+))?", name)
    if m:
        return float(m.group(1)), int(m.group(2) or 1)
    m = re.fullmatch(r"ref_*(\d+)", name)
    return ("ref", int(m.group(1))) if m else None


def file_table(folder, ext):
    rows = []
    for f in sorted(glob(os.path.join(folder, "*" + ext))):
        name = os.path.splitext(os.path.basename(f))[0]
        px = _parse(name)
        if px:
            rows.append(dict(file=name, path=f, P=px[0], rep=px[1]))
    return pd.DataFrame(rows, columns=["file", "path", "P", "rep"])


def load_qepro(sweep):
    """QE Pro spectra of a sweep: dict(wl (file axis), files (file, P, rep), raw, spec = raw - mean plasma-off
    frame, sat = raw > SAT_COUNTS)."""
    t = file_table(os.path.join(QEPRO_DIR, SWEEPS[sweep]["qepro"]), ".csv")
    refs = t[t.P == "ref"]
    t = t[t.P != "ref"].astype(dict(P=float)).sort_values(["P", "rep"]).reset_index(drop=True)
    wl = read_qepro(t.path.iloc[0])[0]
    raw = np.array([read_qepro(p)[1] for p in t.path])
    dark = np.mean([read_qepro(p)[1] for p in refs.path], axis=0)
    return dict(sweep=sweep, wl=wl, files=t[["file", "P", "rep"]], raw=raw, spec=raw - dark, sat=raw > SAT_COUNTS,
                dark=dark)


def echelle_table(sweep):
    """Calibrated echelle spectra (.spa) of a sweep (P, rep); empty if not exported (the pure-Ar sweep has only the
    raw .spe frames so far)."""
    t = file_table(os.path.join(ECHELLE_DIR, SWEEPS[sweep]["echelle"]), ".spa")
    return t[t.P != "ref"].astype(dict(P=float)).sort_values(["P", "rep"]).reset_index(drop=True)


def pixel_variance(c):
    c = np.clip(c, 0, None)
    return NOISE["read2"] + NOISE["gain"] * c + (NOISE["lsf"] * c) ** 2


# ---- 1. instrument function ------------------------------------------------------------------
def pixel_edges(wl):
    mid = (wl[1:] + wl[:-1]) / 2
    return np.r_[2 * wl[0] - mid[0], mid, 2 * wl[-1] - mid[-1]]


def lsf_shape(lam, p):
    """(sigma, asymmetry, Lorentz fraction, Lorentz HWHM) of the LSF at lam [nm]: ln sigma linear between the knots
    (older parameter sets: sigma = s0 exp(s1 u))."""
    lam = np.asarray(lam, float)
    u = (lam - 800.0) / 100.0
    if "ls" in p:
        s = np.exp(np.interp(lam, p.get("knots", LSF_KNOTS), p["ls"]))
    else:
        s = p["s0"] * np.exp(p["s1"] * u)
    return s, A_MAX * np.tanh(p["a0"] + p.get("a1", 0.0) * u), p["f"], p["g"]


def lsf_cdf(x, lam, p):
    """Cumulative LSF at x = lambda - line position [nm] for a line at lam: split Gaussian (sigma (1 -+ a)) centred
    on its centroid + Lorentzian (fraction f)."""
    s, a, f, g = lsf_shape(lam, p)
    x = x + (1 - f) * np.sqrt(2 / np.pi) * 2 * a * s
    sl, sr = s * (1 - a), s * (1 + a)
    G = np.where(x < 0, (1 - a) * (1 + erf(x / (np.sqrt(2) * sl))), (1 - a) + (1 + a) * erf(x / (np.sqrt(2) * sr))) / 2
    return (1 - f) * G + f * (0.5 + np.arctan(x / g) / np.pi)


def line_profile(edges, lam0, p):
    """Fraction of a line at lam0 falling on each pixel (edges: pixel edges, n + 1)."""
    return np.diff(lsf_cdf(edges - lam0, lam0, p))


def fwhm(lam, p, n=4001):
    x = np.linspace(-5, 5, n)
    y = np.gradient(lsf_cdf(x, lam, p), x)
    above = x[y >= y.max() / 2]
    return above[-1] - above[0]


# ---- reference CR spectrum and components ------------------------------------------------------
def reference_table(n2=0.0, Tg=REF_TG):
    """CR table (AnalyticFivePFit, pure-Ar microwave EEDF) at 1.5 Torr, n2 % N2, Tg."""
    cfg = A.table_cfg(L._key(n2, P_MTORR, Tg), "ar")
    with A.xsec_set(A.CR_XSEC), contextlib.redirect_stdout(io.StringIO()):
        return crf.build_model_table(cfg)


def reference_lines(tab=None, EN=REF_EN, Ne=L.N_C):
    """CR-model lines with their observed intensity in energy units (I_obs / lambda, normalised to the strongest)
    and escape factor at (EN, Ne) of tab (default: the pure-Ar reference table)."""
    tab = reference_table() if tab is None else tab
    i = int(np.argmin(np.abs(np.log(tab["x_grid"] / EN))))
    j = int(np.argmin(np.abs(np.log(tab["Ne_grid"] / Ne))))
    I = tab["I_obs"][i, j] / tab["line_wl"]
    d = pd.DataFrame(dict(wl=tab["line_wl"], upper=tab["line_upper"], lower=tab["line_lower"], A=tab["line_A"],
                          I_ref=I / I.max(), eta=tab["I_obs"][i, j] / tab["I_thin"][i, j]))
    return d.sort_values("wl").reset_index(drop=True)


def components(ref, lo, hi, rel_min=REL_MIN, merge=MERGE_COMP_NM):
    """LSF components of the window [lo, hi]: CR lines above rel_min x the strongest line in it, lines closer than
    merge in one component -> list of dict(wl (flux-weighted), lines (their wl), w (flux fractions))."""
    d = ref[(ref.wl >= lo) & (ref.wl <= hi)]
    d = d[d.I_ref >= rel_min * d.I_ref.max()] if len(d) else d
    out, cur = [], []
    for r in d.itertuples():
        if cur and r.wl - cur[-1].wl > merge:
            out.append(cur)
            cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    comps = []
    for c in out:
        w = np.array([r.I_ref for r in c])
        comps.append(dict(wl=float(np.dot(w, [r.wl for r in c]) / w.sum()), lines=[r.wl for r in c], w=w / w.sum()))
    return comps


def comp_profile(edges, comp, p):
    return sum(w * line_profile(edges, l, p) for w, l in zip(comp["w"], comp["lines"]))


def fit_window(wl, y, var, mask, comps, p, shift=0.0, deg=1):
    """Linear least squares of y (pixels wl, variance var, used where mask) with one LSF component per comps entry
    (shifted by shift) and a polynomial baseline of degree deg -> (amplitudes = areas [counts], covariance, model,
    residuals / sigma of the used pixels)."""
    edges = pixel_edges(wl)
    c0 = wl.mean()
    X = np.column_stack([comp_profile(edges + 0.0, dict(c, lines=[l + shift for l in c["lines"]]), p) for c in comps]
                        + [((wl - c0) / WINDOW_NM) ** k for k in range(deg + 1)])
    w = 1 / np.sqrt(var + (NOISE["lsf_peak"] * np.max(y[mask])) ** 2)
    Xw, yw = X[mask] * w[mask, None], y[mask] * w[mask]
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    cov = np.linalg.pinv(Xw.T @ Xw)
    r = yw - Xw @ coef
    dof = max(len(yw) - X.shape[1], 1)
    cov *= max(1.0, float(r @ r) / dof)                 # widened when the LSF model does not fit to the noise
    n = len(comps)
    return coef[:n], cov[:n, :n], X @ coef, r


# ---- 1-2. LSF and wavelength calibration -----------------------------------------------------
def _cal_windows(wl, ref):
    wins = []
    for lines in CAL_LINES:
        lo, hi = min(lines) - WINDOW_NM, max(lines) + WINDOW_NM
        idx = np.flatnonzero((wl > lo) & (wl < hi))
        comps = components(ref, lo - 0.5, hi + 0.5)
        if len(idx) >= 6 and comps:
            wins.append(dict(lines=lines, idx=idx, comps=comps, wl=float(np.mean(lines))))
    return wins


def band_lsf_data(cfg=BAND_LSF):
    """QE Pro spectrum (dark-subtracted mean of cfg's power, saturation mask) and echelle spectrum (mean of the repeats,
    per point x its wavelength step) of the band windows of the LSF fit, or None without echelle exports."""
    et = echelle_table(cfg["sweep"])
    if et.empty:
        return None
    qp = load_qepro(cfg["sweep"])
    k = np.isclose(qp["files"].P, cfg["P"])
    g = et[np.isclose(et.P, cfg["P"])]
    lam = arn.read_spa(g.path.iloc[0])[0]
    E = np.mean([arn.read_spa(f)[1] for f in g.path], axis=0) * np.gradient(lam)
    return dict(y=qp["spec"][k].mean(0), sat=qp["sat"][k].any(0), lam=lam, E=E)


def _pack(p, keys):
    return np.concatenate([np.atleast_1d(np.asarray(p[k], float)) for k in keys])


def _unpack(q, p0, keys):
    p, i = dict(p0), 0
    for k in keys:
        n = len(np.atleast_1d(p0[k]))
        p[k] = list(map(float, q[i:i + n])) if n > 1 else float(q[i])
        i += n
    return p, q[i:]


def fit_lsf(wl, y, sat, ref, p0=LSF_START, per_line=True, bands=None):
    """Joint fit of the LSF and a shift per window to the CAL_LINES windows of the spectrum y (axis wl; every CR-model
    line a free component, linear baseline) and, with bands (band_lsf_data), to the BAND_LSF_WINDOWS of the N2 spectrum
    against the echelle convolved with the LSF (free scale and linear baseline) -> (LSF dict, table of the windows:
    shift, error, source)."""
    wins = _cal_windows(wl, ref)
    bwins = [w for w in BAND_LSF_WINDOWS if bands is not None and ((wl > w[0]) & (wl < w[1])).sum() >= 8]
    keys = list(LSF_FIT)
    var = pixel_variance(y)

    cache = {}

    def band_conv(p):
        """Echelle convolved with the LSF p in every band window (cached: the shifts do not change it)."""
        key = tuple(np.round(_pack(p, keys), 12))
        if key not in cache:
            cache.clear()
            cache[key] = [fine_convolve(bands["lam"], bands["E"], p, wl, lo - 2.0, hi + 2.0) for lo, hi in bwins]
        return cache[key]

    def resid(q):
        p, sh = _unpack(q, p0, keys)
        out = []
        conv = band_conv(p) if bwins else []
        for k, wn in enumerate(wins):
            i = wn["idx"]
            out.append(fit_window(wl[i], y[i], var[i], ~sat[i], wn["comps"], p, sh[k] if per_line else 0.0)[3])
        for k, (lo, hi) in enumerate(bwins):
            m = (wl > lo) & (wl < hi) & ~bands["sat"]
            g, c = conv[k]
            e = np.interp(wl[m] - sh[len(wins) + k], g, c)
            yy = bands["y"][m]
            w = 1 / np.sqrt(pixel_variance(yy) + (NOISE["lsf_peak"] * yy.max()) ** 2)
            X = np.column_stack([e, np.ones(m.sum()), (wl[m] - wl[m].mean()) / 10.0]) * w[:, None]
            cc, *_ = np.linalg.lstsq(X, yy * w, rcond=None)
            out.append(yy * w - X @ cc)
        return np.concatenate(out)

    n_sh = len(wins) + len(bwins)
    q0 = np.r_[_pack(p0, keys), np.zeros(n_sh)]
    lb = np.r_[np.concatenate([np.full(len(np.atleast_1d(p0[k])), LSF_BOUNDS[k][0]) for k in keys]), np.full(n_sh, -1.0)]
    ub = np.r_[np.concatenate([np.full(len(np.atleast_1d(p0[k])), LSF_BOUNDS[k][1]) for k in keys]), np.full(n_sh, 1.0)]
    r = least_squares(resid, np.clip(q0, lb + 1e-9, ub - 1e-9), bounds=(lb, ub), x_scale="jac")
    J = r.jac
    cov = np.linalg.pinv(J.T @ J) * max(1.0, np.sum(r.fun ** 2) / max(len(r.fun) - len(r.x), 1))
    p, sh = _unpack(r.x, p0, keys)
    err = np.sqrt(np.diag(cov))[len(r.x) - n_sh:]
    tab = pd.DataFrame(dict(lines=["+".join(f"{l:.2f}" for l in wn["lines"]) for wn in wins]
                            + [f"N2 bands {lo:.0f}-{hi:.0f}" for lo, hi in bwins],
                            wl=[wn["wl"] for wn in wins] + [(lo + hi) / 2 for lo, hi in bwins],
                            n_lines=[len(wn["lines"]) for wn in wins] + [1] * len(bwins),
                            source=["Ar I"] * len(wins) + ["N2 bands (echelle)"] * len(bwins),
                            shift=sh, shift_err=err))
    return p, tab


def wavelength_correction(wl, tab, deg=WL_DEG):
    """Polynomial (pixel index) through the line shifts of the single-line windows (weighted, 3-sigma clipped on the
    errors + WL_FLOOR) ->
    (corrected axis, coefficients, table with the fit and residuals)."""
    pix = np.arange(len(wl))
    tab = tab.copy()
    tab["pixel"] = np.interp(tab.wl + tab["shift"], wl, pix)
    single = (tab.n_lines == 1).to_numpy()
    use = single.copy()
    err = np.sqrt(tab.shift_err ** 2 + WL_FLOOR ** 2).to_numpy()
    for _ in range(4):
        c = np.polyfit(tab.pixel[use], tab["shift"][use], deg, w=1 / err[use])
        res = tab["shift"] - np.polyval(c, tab.pixel)
        use = single & (np.abs(res) < 3 * err).to_numpy()
    tab["fit"] = np.polyval(c, tab.pixel)
    tab["resid"] = tab["shift"] - tab.fit
    tab["used"] = use
    return wl - np.polyval(c, pix), c, tab


def fine_convolve(x, w, p, wl_axis, lo, hi, step=FINE_STEP, block=CONV_BLOCK_NM):
    """Lines (or a finely sampled spectrum) at x [nm] with weights w, seen by the QE Pro: on a fine grid in [lo, hi],
    the signal of a pixel centred there (LSF of p integrated over the local pixel width of wl_axis), the LSF taken
    constant within blocks of block nm (it changes by < 2 % over 10 nm) -> (grid, signal)."""
    from scipy.signal import fftconvolve
    grid = np.arange(lo, hi + step / 2, step)
    pad = 15.0
    big = np.arange(lo - pad, hi + pad + step / 2, step)
    k = np.clip(np.round((np.asarray(x, float) - big[0]) / step).astype(int), -1, len(big))
    ok = (k >= 0) & (k < len(big))
    F = np.bincount(k[ok], weights=np.asarray(w, float)[ok], minlength=len(big))
    dpix = np.gradient(wl_axis)
    d = np.arange(-pad, pad + step / 2, step)
    out = np.zeros_like(grid)
    for a in np.arange(lo, hi, block):
        c = a + block / 2
        D = float(np.interp(c, wl_axis, dpix))
        kern = lsf_cdf(d + D / 2, c, p) - lsf_cdf(d - D / 2, c, p)
        conv = fftconvolve(F, kern[::-1], mode="same")
        sel = (grid >= a - 1e-9) & (grid < a + block + 1e-9)
        out[sel] = np.interp(grid[sel], big, conv)
    return grid, out


def n2_cb_lines(lo=280.0, hi=500.0):
    """N2 C3Pi_u -> B3Pi_g (second positive) lines of ExoMol 14N2 between lo and hi [nm, air] -> DataFrame(wl, A, g,
    Erot, vu, vl): Erot = upper energy - lowest energy of its v' [cm^-1]; also returns the C-state levels (for the
    rotational partition functions).  Cached in OUTDIR."""
    cache = os.path.join(OUTDIR, "n2_cb_lines.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        d = pd.DataFrame({k: z[k] for k in ("wl", "A", "g", "Erot", "vu", "vl")})
        lev = pd.DataFrame({k: z["lev_" + k] for k in ("E", "g", "v")})
    else:
        st = pd.read_csv(os.path.join(EXOMOL_N2, "14N2.states"), sep=r"\s+", header=None, comment="#",
                         usecols=[0, 1, 2, 3, 8, 9], names=["id", "E", "g", "J", "state", "v"])
        tr = pd.read_csv(os.path.join(EXOMOL_N2, "14N2.trans"), sep=r"\s+", header=None, usecols=[0, 1, 2],
                         names=["u", "l", "A"])
        si = st.set_index("id")
        u, l = si.reindex(tr.u), si.reindex(tr.l)
        k = ((u.state.values == "C3Piu") & (l.state.values == "B3Pig") & (tr.A.values >= N2_CB_A_MIN))
        lev = st[st.state == "C3Piu"][["E", "g", "v"]].reset_index(drop=True)
        e0 = lev.groupby("v").E.min()
        d = pd.DataFrame(dict(wl=arn.vac_to_air(1e7 / (u.E.values[k] - l.E.values[k])), A=tr.A.values[k],
                              g=u.g.values[k], Erot=u.E.values[k] - u.v.values[k] * 0, vu=u.v.values[k].astype(int),
                              vl=l.v.values[k].astype(int)))
        d["Erot"] = d.Erot - d.vu.map(e0)
        os.makedirs(OUTDIR, exist_ok=True)
        np.savez_compressed(cache, **{c: d[c].to_numpy() for c in d.columns},
                            **{"lev_" + c: lev[c].to_numpy() for c in lev.columns})
    lev["Erot"] = lev.E - lev.v.map(lev.groupby("v").E.min())
    return d[(d.wl >= lo) & (d.wl <= hi)].reset_index(drop=True), lev


def n2_cb_templates(lines, lev, Trot, p, wl_axis, lo, hi, step=FINE_STEP):
    """Per upper vibrational level v' of N2(C): the C -> B emission of one molecule in v' at the rotational
    temperature Trot, in energy units (sum of g A exp(-Erot/kT) / Q_rot(v', T) hc/lambda), seen by the QE Pro ->
    {v': (fine grid, signal)} (fine_convolve; evaluate at a pixel centre with np.interp)."""
    out = {}
    for v, g in lines.groupby("vu"):
        Q = float((lev.g[lev.v == v] * np.exp(-1.4387769 * lev.Erot[lev.v == v] / Trot)).sum())
        w = g.A * g.g * np.exp(-1.4387769 * g.Erot / Trot) / Q / g.wl
        out[int(v)] = fine_convolve(g.wl.to_numpy(), w.to_numpy(), p, wl_axis, lo, hi, step=step)
    return out


def n2_band_wavelength_points(wl, p, windows=N2_BAND_WINDOWS):
    """Wavelength shifts in the UV from the N2 C-B band heads (ExoMol line positions, absolute; no echelle needed):
    in each window of every spectrum of N2_BAND_SWEEP (mean of the repeats per power), the v' = 0-4 templates
    (populations >= 0) + a linear baseline, fitted on a grid of Trot and of the shift d of the file axis
    (template evaluated at wl - d) -> rows like fit_lsf's table: shift = median d over the powers, error = their
    scatter (>= N2_BAND_ERR_MIN)."""
    from scipy.optimize import lsq_linear
    try:
        qp = load_qepro(N2_BAND_SWEEP)
    except (FileNotFoundError, IndexError, ValueError):
        return pd.DataFrame()
    lines, lev = n2_cb_lines()
    rows = []
    for lo, hi in windows:
        m = (wl > lo) & (wl < hi)
        tm = {T: n2_cb_templates(lines, lev, T, p, wl, lo - 1.0, hi + 1.0) for T in N2_TROT_GRID}
        best = []
        for P in sorted(qp["files"].P.unique()):
            k = np.isclose(qp["files"].P, P)
            y, ok = qp["spec"][k].mean(0)[m], ~qp["sat"][k].any(0)[m]
            sig = np.sqrt(pixel_variance(y) + (NOISE["lsf_peak"] * y.max()) ** 2)
            b = None
            for T, tv in tm.items():
                for d in N2_SHIFTS:
                    X = np.column_stack([np.interp(wl[m] - d, *tv[v]) for v in sorted(tv)]
                                        + [np.ones(m.sum()), wl[m] - wl[m].mean()])
                    nb = len(tv)
                    r = lsq_linear((X / sig[:, None])[ok], (y / sig)[ok],
                                   bounds=(np.r_[np.zeros(nb), -np.inf, -np.inf], np.r_[np.full(nb, np.inf), np.inf, np.inf]))
                    c2 = float(np.sum(r.fun ** 2))
                    if b is None or c2 < b[0]:
                        b = (c2, d, T)
            best.append(b)
        d = np.array([b[1] for b in best])
        rows.append(dict(lines=f"N2 C-B {lo:.0f}-{hi:.0f}", wl=float(np.mean([lo, hi])), n_lines=1,
                         shift=float(np.median(d)), shift_err=max(N2_BAND_ERR_MIN, float(np.std(d))),
                         source="N2 C-B", Trot=float(np.median([b[2] for b in best]))))
    return pd.DataFrame(rows)


def calibrate_wavelength(qp, ref):
    """LSF and wavelength calibration from the mean pure-Ar spectrum and the N2 band windows (module docstring 1-2)."""
    y, sat = qp["spec"].mean(0), qp["sat"].any(0)
    bands = band_lsf_data()
    p, tab = fit_lsf(qp["wl"], y, sat, ref, bands=bands)
    wl_cal, coef, tab = wavelength_correction(qp["wl"], tab)
    p2, tab2 = fit_lsf(wl_cal, y, sat, ref, p0=p, bands=bands)    # on the corrected axis; shifts now ~0
    tab["shift_after"], tab["shift_after_err"] = tab2["shift"].to_numpy(), tab2.shift_err.to_numpy()
    return wl_cal, coef, p2, tab


# ---- 3. features and line areas --------------------------------------------------------------
def features(ref, merge=MERGE_NM, ranges=FEATURE_RANGES, min_rel=FEATURE_MIN):
    """QE Pro features: CR lines in ranges, those closer than merge together -> DataFrame(feature, wl, lines, I_ref);
    kept if the feature holds >= min_rel x the strongest line of its range in the reference spectrum."""
    rows = []
    for lo, hi in ranges:
        d = ref[(ref.wl >= lo) & (ref.wl <= hi)]
        top = d.I_ref.max()
        groups, cur = [], []
        for r in d[d.I_ref >= 1e-3 * top].itertuples():
            if cur and r.wl - cur[-1].wl > merge:
                groups.append(cur)
                cur = []
            cur.append(r)
        if cur:
            groups.append(cur)
        for g in groups:
            I = sum(r.I_ref for r in g)
            if I < min_rel * top:
                continue
            wl = sum(r.wl * r.I_ref for r in g) / I
            name = f"{wl:.2f} " + "+".join(f"{r.upper}-{r.lower}" for r in g)
            rows.append(dict(feature=name, wl=wl, lines=[r.wl for r in g], comps=[(r.wl, r.upper, r.lower) for r in g],
                             I_ref=I / top, n_lines=len(g)))
    return pd.DataFrame(rows).sort_values("wl").reset_index(drop=True)


def merge_features(feats, merge_nm):
    """Features closer than merge_nm (between neighbouring lines) summed into one."""
    rows, cur = [], []
    for r in feats.itertuples():
        if cur and min(r.lines) - max(cur[-1].lines) > merge_nm:
            rows.append(cur)
            cur = []
        cur.append(r)
    if cur:
        rows.append(cur)
    out = []
    for g in rows:
        lines = [l for r in g for l in r.lines]
        comps = [c for r in g for c in r.comps]
        I = sum(r.I_ref for r in g)
        wl = sum(r.wl * r.I_ref for r in g) / I
        out.append(dict(feature=f"{wl:.2f} " + "+".join(f"{u}-{l}" for _, u, l in comps), wl=wl, lines=lines,
                        comps=comps, I_ref=I, n_lines=len(lines)))
    return pd.DataFrame(out)


def measure_spectrum(wl, y, sat, feats, ref, p, deg=1):
    """Area [counts] of every feature in one spectrum (module docstring 3) -> DataFrame(feature, wl, area, err,
    purity = reference flux of the feature / of every component of its window, chi2 = window residual / pixel)."""
    var = pixel_variance(y)
    rows = []
    for f in feats.itertuples():
        lo, hi = min(f.lines) - WINDOW_NM, max(f.lines) + WINDOW_NM
        idx = np.flatnonzero((wl > lo) & (wl < hi))
        comps = components(ref, lo - 1.0, hi + 1.0)
        own = [k for k, c in enumerate(comps) if any(np.isclose(c["lines"], l, atol=1e-3).any() for l in f.lines)]
        if len(idx) < 6 or not own:
            continue
        a, cov, m, r = fit_window(wl[idx], y[idx], var[idx], ~sat[idx], comps, p, deg=deg)
        # a component may hold lines of the feature and of a neighbour (closer than MERGE_COMP_NM): its share
        share = np.array([sum(w for w, l in zip(comps[k]["w"], comps[k]["lines"])
                              if np.isclose(f.lines, l, atol=1e-3).any()) for k in own])
        area = float(share @ a[own])
        err = float(np.sqrt(share @ cov[np.ix_(own, own)] @ share))
        tot = sum(ref.I_ref[np.isclose(ref.wl, l, atol=1e-3)].sum() for c in comps for l in c["lines"])
        peak_sat = bool(sat[idx][np.abs(wl[idx] - f.wl) < 1.0].any())
        rows.append(dict(feature=f.feature, wl=f.wl, area=area, err=err, purity=f.I_ref_abs / tot if tot else np.nan,
                         chi2=float(r @ r) / max(len(r), 1), saturated=peak_sat, n_comp=len(comps)))
    return pd.DataFrame(rows)


def measure_sweep(qp, wl_cal, feats, ref, p):
    """measure_spectrum for every spectrum of a loaded QE Pro sweep -> long table (file, P, rep, feature, ...)."""
    feats = feats.assign(I_ref_abs=[ref.I_ref[np.isin(np.round(ref.wl, 3), np.round(f, 3))].sum() for f in feats.lines])
    out = []
    for k, r in enumerate(qp["files"].itertuples()):
        m = measure_spectrum(wl_cal, qp["spec"][k], qp["sat"][k], feats, ref, p)
        out.append(m.assign(sweep=qp["sweep"], file=r.file, P=r.P, rep=r.rep))
    return pd.concat(out, ignore_index=True)


# ---- 4a. echelle transfer -----------------------------------------------------------------------
def echelle_lines(sweep, ref_lines, reuse=True):
    """Every catalogue line within +-WINDOW_NM of the QE Pro features in the calibrated echelle spectra of the sweep,
    fitted with CRFitNeTe's group fitter (ActinometryLineSelection.measure_sweeps; instrument function per
    spectrum) -> long table (file, x = P, rep, species, wl_air, status, area, snr); cached.  Lines with pixels at the
    ADC ceiling in the raw frame (EchelleSaturationCheck, within MATCH_NM) get status 'saturated'."""
    t = echelle_table(sweep)
    if t.empty:
        return pd.DataFrame()
    cat, _ = als._catalogue()
    near = np.zeros(len(cat), bool)
    for w in ref_lines:
        near |= np.abs(cat.wl_air.to_numpy() - w) < WINDOW_NM
    sel = cat[near & (cat.wl_air < 870.0)][["species", "wl_air"]]
    cache = os.path.join(OUTDIR, f"echelle_lines_{sweep}.csv")
    if reuse and os.path.exists(cache):
        m = pd.read_csv(cache)
        have = set(zip(m.species, m.wl_air.round(4)))
        if set(m.file) == set(t.file) and set(zip(sel.species, sel.wl_air.round(4))) <= have:
            return mark_saturated(m, sweep)
        if set(m.file) == set(t.file):                  # new lines: fit them together with the cached ones
            sel = pd.concat([sel, m[["species", "wl_air"]]]).drop_duplicates()
    sw = dict(name=sweep, folder=os.path.join(ECHELLE_DIR, SWEEPS[sweep]["echelle"]),
              parse=lambda n: (lambda px: px if px and px[0] != "ref" else None)(_parse(n)), exclude_x=())
    cfg = dict(als.CONFIG, sweeps=[sw], window_nm=0.5, reuse_measurements=False, outdir=OUTDIR,
               measure_cache=f"echelle_lines_{sweep}.csv")
    os.makedirs(OUTDIR, exist_ok=True)
    return mark_saturated(als.measure_sweeps(sel, cfg), sweep)


def mark_saturated(m, sweep):
    """Status 'saturated' for the echelle lines with clipped pixels in their raw frame (.spe)."""
    d = SWEEPS[sweep]
    cat, _ = als._catalogue()
    sat = esc.check_folder(os.path.join(ECHELLE_DIR, d.get("echelle_spe", d["echelle"])), cat,
                           spa_folder=os.path.join(ECHELLE_DIR, d["echelle"]))
    if "wl" not in sat:
        return m.assign(sat_raw=False)
    hit = np.zeros(len(m), bool)
    for f, g in sat.groupby("file"):
        k = (m.file == f).to_numpy()
        hit[k] |= (np.abs(m.wl_air.to_numpy()[k, None] - g.wl.to_numpy()[None, :]) < esc.MATCH_NM).any(axis=1)
    return m.assign(sat_raw=hit, status=np.where(hit, "saturated", m.status))


def transfer_points(qf, el, feats):
    """Per feature and power: mean QE Pro area / mean echelle area of the feature's lines (module docstring 4a)."""
    if el.empty:
        return pd.DataFrame()
    el = el.assign(wl=el.wl_air.round(3))
    rows = []
    for f in feats.itertuples():
        want = np.round(f.lines, 3)
        for P, g in el[np.isin(el.wl, want) & (el.species == "Ar I")].groupby("x"):
            per = g.groupby("file").apply(lambda h: pd.Series(dict(ok=h.status.isin(VALID_ECHELLE).all()
                                                                 and set(h.wl) >= set(want), area=h.area.sum())),
                                          include_groups=False)
            per = per[per.ok.astype(bool)]
            q = qf[(qf.feature == f.feature) & np.isclose(qf.P, P) & ~qf.saturated]
            if len(per) < 2 or q.empty:
                continue
            E, Q = per.area.astype(float), q.area
            if not (Q.mean() > 0 and E.mean() > 0):
                continue
            rows.append(dict(kind="line", feature=f.feature, wl=f.wl, P=P, Q=Q.mean(), E=E.mean(), n_E=len(E),
                             n_Q=len(Q), ln_R=np.log(Q.mean() / E.mean()),
                             err=np.sqrt((E.std(ddof=1) / E.mean()) ** 2 / len(E) + (Q.std(ddof=1) / Q.mean()) ** 2
                                         / max(len(Q), 1) + (q.err.mean() / Q.mean()) ** 2 + 0.02 ** 2),
                             purity=float(q.purity.mean())))
    return pd.DataFrame(rows)


def convolution_matrix(lam, wl, p, reach=12.0):
    """Sparse (QE Pro pixels x echelle points): fraction of a line at each echelle wavelength falling on each pixel."""
    from scipy import sparse
    edges = pixel_edges(wl)
    lo = np.clip(np.searchsorted(edges, lam - reach) - 1, 0, len(edges) - 1)
    hi = np.clip(np.searchsorted(edges, lam + reach), 0, len(edges) - 1)
    rows, cols, vals = [], [], []
    for j, (l, a, b) in enumerate(zip(lam, lo, hi)):
        if b > a:
            rows.append(np.arange(a, b))
            cols.append(np.full(b - a, j))
            vals.append(np.diff(lsf_cdf(edges[a:b + 1] - l, l, p)))
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(wl), len(lam)))


def window_transfer(qp, wl, p, sweep, lo=None, hi=None, width=None):
    """Echelle spectrum (lines, bands and continuum) convolved with the QE Pro LSF against the QE Pro spectrum, summed
    over windows of the given width in [lo, hi] (module docstring 4a) -> one row per window and power.  Windows with
    an echelle order gap within the LSF reach of any of their pixels (> 1e-4 of a line's flux) are left out."""
    lo, hi = SPEC_RANGE if lo is None else (lo, hi)
    width = SPEC_WINDOW_NM if width is None else width
    et = echelle_table(sweep)
    if et.empty:
        return pd.DataFrame()
    lam, I0 = arn.read_spa(et.path.iloc[0])
    K = convolution_matrix(lam, wl, p)
    gap_pix = (K[:, I0 == 0] > 1e-4).sum(axis=1).A1 > 0
    dlam = np.gradient(lam)
    rows = []
    for P, g in et.groupby("P"):
        E = np.array([arn.read_spa(f)[1] for f in g.path])
        Ec = np.array([K @ (e * dlam) for e in E])                    # per echelle repeat
        k = np.isclose(qp["files"].P, P)
        if not k.any():
            continue
        Q, sat = qp["spec"][k].mean(0), qp["sat"][k].any(0)
        for a in np.arange(lo, hi - width + 1e-9, width):
            w = (wl >= a) & (wl < a + width)
            if not w.any() or gap_pix[w].any() or sat[w].any():
                continue
            e = Ec[:, w].sum(axis=1)
            if not (e > 0).all():
                continue
            rows.append(dict(kind="window", feature=f"window {a:.0f}-{a + width:.0f}", wl=a + width / 2, P=P,
                             Q=Q[w].sum(), E=e.mean(), n_E=len(e), n_Q=int(k.sum()), ln_R=np.log(Q[w].sum() / e.mean()),
                             err=np.hypot(e.std(ddof=1) / e.mean() / np.sqrt(len(e)), SPEC_FLOOR), purity=1.0))
    return pd.DataFrame(rows)


# ---- 4b. branching ratios -----------------------------------------------------------------------
def branching_points(qf, ref, pairs=BR_PAIRS, min_eta=BR_MIN_ETA):
    """Per pair and spectrum: ln(Q1/Q2) - ln(I1/I2) of the reference CR spectrum (same upper level: A ratio and the
    escape factors) = ln R(lambda1) - ln R(lambda2)."""
    rows = []
    for l1, l2 in pairs:
        r1, r2 = (ref.iloc[int(np.argmin(np.abs(ref.wl - l)))] for l in (l1, l2))
        if r1.upper != r2.upper:
            raise ValueError(f"{l1} and {l2} do not share an upper level")
        f1, f2 = (qf[qf.feature.str.startswith(f"{l:.2f} ")] for l in (l1, l2))
        if f1.empty or f2.empty:
            continue
        j = f1.merge(f2, on=["sweep", "file", "P", "rep"], suffixes=("_1", "_2"))
        j = j[~j.saturated_1 & ~j.saturated_2 & (j.area_1 > 0) & (j.area_2 > 0)]
        model = np.log(r1.I_ref / r2.I_ref)
        for r in j.itertuples():
            rows.append(dict(sweep=r.sweep, file=r.file, P=r.P, wl_1=l1, wl_2=l2, upper=r1.upper, eta_1=r1.eta,
                             eta_2=r2.eta, d_ln_R=np.log(r.area_1 / r.area_2) - model,
                             err=np.sqrt((r.err_1 / r.area_1) ** 2 + (r.err_2 / r.area_2) ** 2
                                         + (BR_ETA_ERR * np.log(r1.eta / r2.eta)) ** 2),
                             n_comp=max(r.n_comp_1, r.n_comp_2), use=min(r1.eta, r2.eta) >= min_eta
                             and r.n_lines_1 == 1 and r.n_lines_2 == 1))
    return pd.DataFrame(rows)


# ---- 4. response fit --------------------------------------------------------------------------
def spline_basis(wl, knots=None, k=3):
    """Cubic B-spline basis of ln R on wl (knots [nm], clamped)."""
    from scipy.interpolate import BSpline
    knots = RESP_KNOTS if knots is None else knots
    t = np.r_[[knots[0]] * k, knots, [knots[-1]] * k]
    x = np.clip(np.asarray(wl, float), knots[0], knots[-1] - 1e-9)
    return BSpline.design_matrix(x, t, k).toarray()


def fit_response(tp, bp, min_purity=MIN_PURITY, smooth=None, wl_norm=750.0):
    """ln R(lambda) = B-spline (RESP_KNOTS) + one constant per (sweep, power) of the transfer points; the branching
    points constrain differences; second differences of the spline coefficients penalised (weight smooth) and ln R
    pinned to 0 at wl_norm.  Weighted least squares, 3-sigma clipped -> (coefficients, covariance, tp and bp with
    use / resid, list of (sweep, power) constants)."""
    smooth = RESP_SMOOTH if smooth is None else smooth
    nb = spline_basis([wl_norm]).shape[1]
    if len(tp):
        blue_ok = lambda t: (t.kind == "window") | (t.wl >= LINE_TRANSFER_MIN_NM) | (t.sweep.map(
            lambda sw: SWEEPS[sw]["N2"]) == 0)        # 5p lines of the pure-Ar sweep are clean
        tp = tp.assign(use=(tp.purity >= min_purity) & blue_ok(tp))
    bp = bp.copy()
    keys = sorted(set(zip(tp.sweep, tp.P))) if len(tp) else []
    D = np.diff(np.eye(nb), 2, axis=0)
    for _ in range(4):
        X, y, w = [], [], []
        if len(tp):
            t = tp[tp.use]
            C = np.array([[float(k == kk) for kk in keys] for k in zip(t.sweep, t.P)]).reshape(len(t), len(keys))
            X.append(np.column_stack([spline_basis(t.wl), C]))
            y.append(t.ln_R.to_numpy())
            w.append(1 / t.err.to_numpy())
        b = bp[bp.use] if len(bp) else bp
        if len(b):
            X.append(np.column_stack([spline_basis(b.wl_1) - spline_basis(b.wl_2), np.zeros((len(b), len(keys)))]))
            y.append(b.d_ln_R.to_numpy())
            w.append(1 / np.hypot(b.err.to_numpy(), BR_FLOOR))
        X.append(np.column_stack([np.sqrt(smooth) * D, np.zeros((len(D), len(keys)))]))   # smoothness
        y.append(np.zeros(len(D)))
        w.append(np.ones(len(D)))
        X.append(np.column_stack([spline_basis([wl_norm]), np.zeros((1, len(keys)))]))      # ln R(wl_norm) = 0
        y.append([0.0])
        w.append([1e3])
        X, y, w = np.vstack(X), np.concatenate(y), np.concatenate(w)
        Xw = X * w[:, None]
        c, *_ = np.linalg.lstsq(Xw, y * w, rcond=None)
        n_data = len(y) - len(D) - 1
        res = (y - X @ c) * w
        s2 = max(1.0, float(res[:n_data] @ res[:n_data]) / max(n_data - len(c), 1))
        cov = np.linalg.pinv(Xw.T @ Xw) * s2
        if len(tp):
            const = np.array([c[nb + keys.index(k)] for k in zip(tp.sweep, tp.P)])
            tp["resid"] = tp.ln_R - spline_basis(tp.wl) @ c[:nb] - const
            tp["use"] = (tp.purity >= min_purity) & blue_ok(tp) & (np.abs(tp.resid) < 3 * np.sqrt(s2) * tp.err)
        if len(bp):
            bp["resid"] = bp.d_ln_R - (spline_basis(bp.wl_1) - spline_basis(bp.wl_2)) @ c[:nb]
    return c, cov, tp, bp, keys


def response_curve(c, cov, wl):
    """ln R on wl (0 at the normalisation wavelength of fit_response) and its 1-sigma error."""
    G = spline_basis(wl)
    nb = G.shape[1]
    return G @ c[:nb], np.sqrt(np.einsum("ij,jk,ik->i", G, cov[:nb, :nb], G))


def load_calibration(folder=OUTDIR):
    """(corrected wavelength axis, LSF dict, response function lam -> R) written by main()."""
    wlc = pd.read_csv(os.path.join(folder, "wavelength_calibration.csv"))
    p = json.load(open(os.path.join(folder, "lsf.json")))
    r = pd.read_csv(os.path.join(folder, "response.csv"))
    return wlc.wl_cal.to_numpy(), p, (lambda lam: np.exp(np.interp(lam, r.wl, r.ln_R)))


# ---- plots ------------------------------------------------------------------------------------
def plot_wavelength(tab, wl, wl_cal, coef, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    u = tab[tab.used & (tab.source == "Ar I")]
    ax.errorbar(u.wl, u["shift"], yerr=u.shift_err, fmt="o", color="C0", ms=5, capsize=2, label="Ar I line (pure Ar)")
    u = tab[tab.used & (tab.source != "Ar I")]
    ax.errorbar(u.wl, u["shift"], yerr=u.shift_err, fmt="D", color="C1", ms=5, capsize=2,
                label="N2 band window (5 % N$_2$ vs echelle)")
    o = tab[~tab.used]
    ax.errorbar(o.wl, o["shift"], yerr=o.shift_err, fmt="s", mfc="white", color="0.5", ms=5, capsize=2,
                label="multi-line window / clipped (not fitted)")
    ax.plot(wl, wl - wl_cal, "-", color="C3", lw=1.5, label=f"correction, polynomial deg {len(coef) - 1} in pixel")
    ax.errorbar(tab.wl, tab.shift_after, yerr=tab.shift_after_err, fmt=".", color="C2", ms=4, capsize=1,
                label="after the correction (LSF refitted)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("line position - catalogue (air) [nm]")
    rms = np.sqrt(np.mean(tab.resid[tab.used] ** 2))
    ax.set_title(f"QE Pro wavelength calibration from Ar I lines (pure Ar, mean of all spectra)\n"
                 f"rms residual {rms:.3f} nm of {tab.used.sum()} lines", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_lsf(qp, wl_cal, p, ref, path, lines=(696.5431, 738.398, 763.5106, 826.4522, 852.1442, 912.2967, 965.7786,
                                              430.0101)):
    y, sat = qp["spec"].mean(0), qp["sat"].any(0)
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.5))
    for ax, l in zip(axs.flat, lines):
        lo, hi = l - WINDOW_NM, l + WINDOW_NM
        i = np.flatnonzero((wl_cal > lo) & (wl_cal < hi))
        comps = components(ref, lo - 0.5, hi + 0.5)
        a, cov, m, r = fit_window(wl_cal[i], y[i], pixel_variance(y[i]), ~sat[i], comps, p)
        x = np.linspace(lo, hi, 600)
        ed = pixel_edges(wl_cal[i])
        ax.semilogy(wl_cal[i], np.clip(y[i], 0.3, None), "o", color="C0", ms=4, label="mean pure-Ar spectrum")
        ax.semilogy(wl_cal[i], np.clip(m, 0.3, None), "-", color="C3", lw=1.2, label="LSF fit (pixel-integrated)")
        if sat[i].any():
            ax.semilogy(wl_cal[i][sat[i]], y[i][sat[i]], "x", color="k", ms=7, label="> SAT_COUNTS in a spectrum")
        bx = ax.twinx()
        bx.plot(wl_cal[i], (y[i] - m) / y[i].max(), ".-", color="C2", lw=0.7, ms=3)
        bx.set_ylim(-0.06, 0.06)
        bx.axhline(0, color="C2", lw=0.4)
        if ax in axs[:, -1]:
            bx.set_ylabel("residual / peak", color="C2")
        ax.set_title(f"{l:.2f} nm: FWHM {fwhm(l, p):.2f} nm", fontsize=9)
        ax.grid(alpha=0.3, which="both")
    axs.flat[0].legend(fontsize=7)
    for ax in axs[-1]:
        ax.set_xlabel("Wavelength (calibrated) [nm]")
    fig.suptitle("QE Pro instrument function: split Gaussian + Lorentzian wings, fitted jointly to the Ar I lines",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_response(tp, bp, c, cov, keys, path):
    wl = np.linspace(390, 990, 601)
    lnR, err = response_curve(c, cov, wl)
    nb = spline_basis([750.0]).shape[1]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 9.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
    ax.fill_between(wl, np.exp(lnR - err), np.exp(lnR + err), color="C0", alpha=0.25, lw=0)
    ax.plot(wl, np.exp(lnR), "-", color="C0", lw=2, label="fit (ln R cubic spline), 1 sigma")
    if len(tp):
        cmap = plt.get_cmap("viridis")
        Ps = sorted({P for _, P in keys})
        for (sweep, P) in keys:
            t = tp[(tp.sweep == sweep) & np.isclose(tp.P, P)]
            k = c[nb + keys.index((sweep, P))]
            col = cmap(Ps.index(P) / max(len(Ps) - 1, 1))
            for kind, mk in (("line", "o"), ("window", "s")):
                u = t[(t.kind == kind) & t.use]
                ax.errorbar(u.wl, np.exp(u.ln_R - k), yerr=np.exp(u.ln_R - k) * u.err, fmt=mk, color=col,
                            ms=4 if kind == "line" else 3, capsize=1.5, alpha=0.8,
                            label=f"{P:g} W" if kind == "line" else None)
                x = t[(t.kind == kind) & ~t.use]
                ax.plot(x.wl, np.exp(x.ln_R - k), "x", color=col, ms=6)
    ax.set_yscale("log")
    ax.set_ylabel("QE Pro counts / echelle units (relative, 1 at 750 nm)")
    ax.set_title("QE Pro relative response (5 % N$_2$ sweep: echelle transfer - circles: Ar I lines, squares: 6 nm\n"
                 "windows of the whole spectrum below the echelle order gaps; x: left out) + branching ratios (below)",
                 fontsize=9)
    ax.legend(fontsize=7, ncol=5, title="power", title_fontsize=7)
    ax.grid(alpha=0.3, which="both")
    if len(bp):
        G = lambda a: spline_basis(a) @ c[:nb]
        for sweep, mk in (("pure_Ar", "o"), ("N2_5pct", "s")):
            b = bp[bp.sweep == sweep]
            if b.empty:
                continue
            g = b.groupby(["wl_1", "wl_2"]).agg(d=("d_ln_R", "median"), s=("d_ln_R", "std"), use=("use", "first"),
                                                 eta1=("eta_1", "first"), eta2=("eta_2", "first")).reset_index()
            pred = G(g.wl_1) - G(g.wl_2)
            for r, pr in zip(g.itertuples(), pred):
                col = "C3" if r.use else "0.6"
                bx.errorbar(r.wl_1, r.d - pr, yerr=r.s, fmt=mk, color=col, mfc=col if sweep == "pure_Ar" else "white",
                            ms=6, capsize=2)
                bx.annotate(f"{r.wl_1:.0f}/{r.wl_2:.0f}", (r.wl_1, r.d - pr), fontsize=6, xytext=(3, 3),
                            textcoords="offset points")
    bx.axhline(0, color="k", lw=0.7)
    bx.set_ylabel("ln(Q$_1$/Q$_2$) measured - CR model\n- [ln R($\\lambda_1$) - ln R($\\lambda_2$)] of the fit")
    bx.set_xlabel("Wavelength [nm] (first line of the same-upper-level pair)")
    bx.set_title("Branching-ratio check (filled: pure Ar, open: 5 % N$_2$; red: used in the fit, grey: trapped "
                 f"line, escape factor < {BR_MIN_ETA:g}, or a split doublet)", fontsize=9)
    bx.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---- driver -----------------------------------------------------------------------------------
def main():
    os.makedirs(FIGDIR, exist_ok=True)
    ref = reference_lines()
    qs = {s: load_qepro(s) for s in SWEEPS}
    for s, q in qs.items():
        print(f"{s}: {len(q['files'])} QE Pro spectra, {q['sat'].any(1).sum()} with pixels > {SAT_COUNTS:g} counts",
              flush=True)
    # 1-2. LSF and wavelength calibration (pure Ar)
    wl_cal, coef, p, tab = calibrate_wavelength(qs["pure_Ar"], ref)
    wl = qs["pure_Ar"]["wl"]
    pd.DataFrame(dict(pixel=np.arange(len(wl)), wl_file=wl, wl_cal=wl_cal)).to_csv(
        os.path.join(OUTDIR, "wavelength_calibration.csv"), index=False)
    tab.to_csv(os.path.join(OUTDIR, "wavelength_lines.csv"), index=False)
    json.dump(dict(p, wl_coef_pixel=list(coef), sat_counts=SAT_COUNTS), open(os.path.join(OUTDIR, "lsf.json"), "w"),
              indent=1)
    print(f"\nwavelength correction (file - calibrated) {np.min(wl - wl_cal):+.3f} to {np.max(wl - wl_cal):+.3f} nm; "
          f"rms residual {np.sqrt(np.mean(tab.resid[tab.used] ** 2)):.3f} nm ({tab.used.sum()} lines)")
    print("LSF:", {k: np.round(np.asarray(v, float), 4).tolist() for k, v in p.items()},
          "FWHM", {l: round(fwhm(l, p), 2) for l in (420, 700, 800, 900, 965)})
    plot_wavelength(tab, wl, wl_cal, coef, os.path.join(FIGDIR, "wavelength_calibration.png"))
    chk = n2_band_wavelength_points(wl_cal, p)
    if len(chk):
        chk.to_csv(os.path.join(OUTDIR, "n2_band_check.csv"), index=False)
        print("ExoMol N2 C-B check on the calibrated axis (shift should be ~0, Trot ~ Tg):")
        print(chk[["lines", "shift", "shift_err", "Trot"]].to_string(index=False))
    plot_lsf(qs["pure_Ar"], wl_cal, p, ref, os.path.join(FIGDIR, "lsf_fits.png"))
    # 3. line areas
    feats = features(ref)
    qf = pd.concat([measure_sweep(q, wl_cal, feats, ref, p) for q in qs.values()], ignore_index=True)
    qf = qf.merge(feats[["feature", "n_lines"]], on="feature")
    qf.to_csv(os.path.join(OUTDIR, "qepro_features.csv"), index=False)
    feats_m = merge_features(feats, MERGE_TRANSFER_NM)            # line groups of the echelle transfer
    qf_m = pd.concat([measure_sweep(q, wl_cal, feats_m, ref, p) for q in qs.values()], ignore_index=True)
    # 4. response
    tps = []
    for sw in SWEEPS:
        el = echelle_lines(sw, [l for ls in feats.lines for l in ls])
        if el.empty:
            print(f"{sw}: no echelle .spa exports - no transfer points")
            continue
        n_sat = el[el.status == "saturated"].groupby("wl_air").file.nunique()
        print(f"{sw}: echelle lines with clipped raw pixels (spectra): "
              + ", ".join(f"{w:.2f} ({n})" for w, n in n_sat.items()))
        tps.append(transfer_points(qf_m[qf_m.sweep == sw], el, feats_m).assign(sweep=sw))
        tps.append(window_transfer(qs[sw], wl_cal, p, sw).assign(sweep=sw))
    tp = pd.concat(tps, ignore_index=True) if tps else pd.DataFrame()
    bp = branching_points(qf, ref)
    c, cov, tp, bp, keys = fit_response(tp, bp)
    wlr = np.round(np.arange(300.0, 990.01, 0.5), 2)
    lnR, err = response_curve(c, cov, wlr)
    pd.DataFrame(dict(wl=wlr, ln_R=lnR, R=np.exp(lnR), err=err)).to_csv(os.path.join(OUTDIR, "response.csv"),
                                                                       index=False)
    pd.concat([tp.assign(source="transfer"), bp.assign(source="branching")], ignore_index=True).to_csv(
        os.path.join(OUTDIR, "response_points.csv"), index=False)
    plot_response(tp, bp, c, cov, keys, os.path.join(FIGDIR, "response.png"))
    u = tp[tp.use] if len(tp) else tp
    print(f"\nresponse: {len(u)} transfer points ({(u.kind == 'line').sum()} lines, {(u.kind == 'window').sum()} "
          f"windows), {int(bp.use.sum())} branching points; rms of the transfer points "
          f"{np.sqrt(np.mean(u.resid ** 2)) if len(u) else np.nan:.3f}, of the branching points "
          f"{np.sqrt(np.mean(bp.resid[bp.use] ** 2)):.3f} (ln)")
    print("written to", OUTDIR)
    return dict(ref=ref, qs=qs, wl_cal=wl_cal, p=p, tab=tab, feats=feats, qf=qf, tp=tp, bp=bp, c=c, cov=cov)


if __name__ == "__main__":
    RES = main()
