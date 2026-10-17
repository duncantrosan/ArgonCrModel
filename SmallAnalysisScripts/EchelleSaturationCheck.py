# -*- coding: utf-8 -*-
"""
EchelleSaturationCheck.py

Which echelle lines are saturated?  The calibrated .spa exports hide it (the CRFitNeTe group fitter's flat-top test
- 3 points within 1 % of the maximum - catches none of them), but the raw frame (.spe, saved next to each .spa)
shows it: pixels at the ADC ceiling.

.spe layout (ESAWIN, read here without the vendor software): 4100-byte header (WinSpec style: x dim at byte 42,
y dim at 656, data type at 108: 1 = int32 - the 2026-10 sweeps, offset-subtracted -, 3 = uint16 - the older pressure /
GSA / power sweeps), 1024 x 1024 frame, then a calibration trailer: header (byte 20: highest
order, 24: lowest order m0), the reference line name (Zn-213.856), one cubic per order for the order trace
y(x) = c0 + c1 x + c2 x^2 + c3 x^3 (float32, from m0 = 28 at the top of the frame, y ~ 889, down to m = 120), then
per order 5 float32: x(dl) = d0 + d1 dl + d2 dl^2 + d3 dl^3 with dl = lambda - lambda_c [um] and lambda_c = 24.002 um / m
(the blaze wavelength of order m).  The slit image of an order lies in the ~4 rows above its trace (checked on the
750.39 nm line, rows 887-890 above the m = 32 trace at 886.6 at x = 500).

Per frame: pixels >= SAT_FRACTION x the clip level are grouped into blobs; each blob is given the order whose trace
lies within (-1, +5) rows below it and the wavelength of its x in that order (two candidate orders: the one whose
wavelength has the brighter point in the .spa export wins), then the nearest Ar I / N I catalogue line within
MATCH_NM.  Clip level: the frame's ADC ceiling, 65535 x the number of accumulations minus the offset; in practice the
largest value of the folder when at least two frames reach it exactly (hard clip), else 65535 for uint16 frames and
CLIP_PER_ACC x accumulations for int32 frames (65535 - the ~2893 count offset; accumulations = 4th header flag of the
.spa export: 1 or 30).

Output in Experimental_Data/Output/EchelleSaturation/: saturated_pixels.csv (every blob), saturated_lines.csv (per
folder and line: frames saturated / frames), summary printed per folder with the CR-fit lines (CRFitNeTe) marked.
Run from Spyder (F5) or python SmallAnalysisScripts/EchelleSaturationCheck.py [folder under EchelleData ...]
"""
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "Experimental_Data", "ExperimentalDataAnalysis"))
import ActinometryLineSelection as als              # noqa: E402

ECHELLE_DIR = os.path.join(ROOT, "Experimental_Data", "EchelleData")
OUTDIR = os.path.join(ROOT, "Experimental_Data", "Output", "EchelleSaturation")
SAT_FRACTION = 0.995                                # pixel saturated above this x the clip level
CLIP_PER_ACC = 62642                                # clip level of one accumulation seen in the 1 Torr sweeps
MATCH_NM = 0.15                                     # blob -> catalogue line
FIT_LINES = (415.859, 430.0101, 706.7218, 727.2936, 750.3869, 751.4652, 794.8176, 800.6157, 801.4786)  # CR fit


SPE_DTYPES = {0: "<f4", 1: "<i4", 2: "<i2", 3: "<u2"}


def read_spe(path):
    """(frame 1024 x 1024, order traces (n, 4), wavelength records (n, 5), lowest order m0); ValueError if the file
    is empty or truncated."""
    b = open(path, "rb").read()
    if len(b) < 4100:
        raise ValueError(f"{path}: {len(b)} bytes")
    nx, ny = int(np.frombuffer(b, "<u2", 1, 42)[0]), int(np.frombuffer(b, "<u2", 1, 656)[0])
    dt = np.dtype(SPE_DTYPES[int(np.frombuffer(b, "<i2", 1, 108)[0])])
    a = np.frombuffer(b, dt, nx * ny, 4100).reshape(ny, nx)
    t = b[4100 + dt.itemsize * nx * ny:]
    m_hi, m0 = np.frombuffer(t, "<i4", 2, 20)
    n = int(m_hi - m0 + 1)
    tr = np.frombuffer(t, "<f4", 4 * n, 96).reshape(n, 4).astype(float)
    wr = np.frombuffer(t, "<f4", 5 * n, 96 + 16 * n).reshape(n, 5).astype(float)
    return a, tr, wr, int(m0)


def wavelength(x, rec):
    """Wavelength [nm] at column x of the order with wavelength record rec."""
    d0, d1, d2, d3, lc = rec
    r = np.roots([d3, d2, d1, d0 - x])
    r = r[np.isreal(r)].real
    return 1000.0 * (lc + r[np.argmin(np.abs(r))])


def n_accumulations(spa):
    """4th flag of the .spa header 'Wavelength a-b-c-d-7617' (1 or 30 in our data), 1 if absent."""
    if not os.path.exists(spa):
        return 1
    with open(spa, encoding="latin-1") as f:
        tok = f.readline().split()
    try:
        return int(tok[1].split("-")[3])
    except (IndexError, ValueError):
        return 1


def clip_level(frames_max, n_acc, dtype):
    top = max(frames_max)
    if sum(m == top for m in frames_max) >= 2 and top > 0.9 * CLIP_PER_ACC * n_acc:
        return float(top)
    return 65535.0 if dtype == np.uint16 else float(CLIP_PER_ACC * n_acc)


def blobs(path, clip, spa_wl=None, spa_I=None):
    """Saturated blobs of one frame -> list of dict(y, x, n_pix, order, wl)."""
    a, tr, wr, m0 = read_spe(path)
    lab, n = ndimage.label(a >= SAT_FRACTION * clip)
    out = []
    for k in range(n):
        yy, xx = np.nonzero(lab == k + 1)
        y, x = yy.mean(), xx.mean()
        ytr = np.array([np.polyval(r[::-1], x) for r in tr])
        cand = [o for o in np.flatnonzero((y - ytr > -1.0) & (y - ytr < 5.0))]
        best = None
        for o in cand:
            wl = wavelength(x, wr[o])
            bright = -np.inf
            if spa_wl is not None:
                near = np.abs(spa_wl - wl) < 0.05
                bright = spa_I[near].max() if near.any() else -np.inf
            if best is None or bright > best[2]:
                best = (o, wl, bright)
        if best:
            out.append(dict(y=y, x=x, n_pix=len(yy), order=m0 + best[0], wl=best[1]))
    return out


def check_folder(folder, cat, spa_folder=None):
    """Saturated blobs of every frame (.spe) in folder; the .spa exports of the same names in spa_folder (default:
    folder) pick between candidate orders."""
    spa_folder = spa_folder or folder
    rows = []
    spes, maxes, dtype = [], [], None
    for f in sorted(glob(os.path.join(folder, "*.spe"))):
        if "ref" in os.path.basename(f).lower():
            continue
        try:
            a = read_spe(f)[0]
        except ValueError as err:
            print(f"skipped {err}")
            continue
        spes.append(f)
        maxes.append(int(a.max()))
        dtype = a.dtype
    if not spes:
        return pd.DataFrame()
    spa_of = lambda f: os.path.join(spa_folder, os.path.splitext(os.path.basename(f))[0] + ".spa")
    n_acc = n_accumulations(spa_of(spes[0]))
    clip = clip_level(maxes, n_acc, dtype)
    for f, mx in zip(spes, maxes):
        name = os.path.splitext(os.path.basename(f))[0]
        spa = spa_of(f)
        sw = si = None
        if os.path.exists(spa):
            sw, si = als.arn.read_spa(spa)
        for b in blobs(f, clip, sw, si) if mx >= SAT_FRACTION * clip else []:
            d = np.abs(cat.wl_air.to_numpy() - b["wl"])
            j = int(np.argmin(d))
            line = cat.iloc[j] if d[j] < MATCH_NM else None
            rows.append(dict(folder=os.path.relpath(folder, ECHELLE_DIR), file=name, n_acc=n_acc, clip=clip, **b,
                             species=line.species if line is not None else "", line_nm=line.wl_air if line is not None
                             else np.nan, upper=line.upper if line is not None else ""))
    t = pd.DataFrame(rows)
    return t.assign(n_frames=len(spes)) if len(t) else pd.DataFrame([dict(folder=os.path.relpath(folder, ECHELLE_DIR),
                                                                          n_frames=len(spes), clip=clip, n_acc=n_acc)])


def main(folders=None):
    os.makedirs(OUTDIR, exist_ok=True)
    cat, _ = als._catalogue()
    if not folders:
        folders = sorted({os.path.dirname(f) for f in glob(os.path.join(ECHELLE_DIR, "**", "*.spe"), recursive=True)})
    else:
        folders = [os.path.join(ECHELLE_DIR, f) for f in folders]
    spa_dir = lambda f: (os.path.join(f, "SPAFiles") if not glob(os.path.join(f, "*.spa"))
                         and os.path.isdir(os.path.join(f, "SPAFiles")) else f)   # exports kept in a subfolder
    t = pd.concat([check_folder(f, cat, spa_dir(f)) for f in folders], ignore_index=True)
    t.to_csv(os.path.join(OUTDIR, "saturated_pixels.csv"), index=False)
    s = t[t.line_nm.notna()] if "line_nm" in t else t.iloc[:0]
    lines = (s.groupby(["folder", "species", "line_nm", "upper"]).agg(frames=("file", "nunique"), n_frames=("n_frames", "first"),
                                                                    max_pix=("n_pix", "max")).reset_index())
    lines["in_cr_fit"] = lines.line_nm.round(3).isin(np.round(FIT_LINES, 3))
    lines.to_csv(os.path.join(OUTDIR, "saturated_lines.csv"), index=False)
    for f, g in t.groupby("folder"):
        n = int(g.n_frames.iloc[0])
        sat = lines[lines.folder == f]
        unid = int(g.line_nm.isna().sum()) if "line_nm" in g else 0
        desc = ", ".join(f"{r.line_nm:.2f}{'*' if r.in_cr_fit else ''} ({r.frames}/{n})" for r in sat.itertuples())
        print(f"{f:<55s} clip {g['clip'].iloc[0]:>9.0f} ({int(g.n_acc.iloc[0])} acc.): "
              f"{desc or 'no saturated line'}" + (f"; {unid} unidentified blobs" if unid else ""))
    print("* = line of the CR fit (CRFitNeTe features)")
    print("written to", OUTDIR)
    return t, lines


if __name__ == "__main__":
    PIX, LINES = main(sys.argv[1:])
