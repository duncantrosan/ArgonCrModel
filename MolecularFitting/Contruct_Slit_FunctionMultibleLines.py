# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:48:08 2026

@author: dptro
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf

# =========================================================
# CONFIGURATION
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CALIBRATION_FILENAME = "09_04_2026.spa"
CALIBRATION_FILE = os.path.join(SCRIPT_DIR, "Slit_Functions", "SPA", CALIBRATION_FILENAME)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Slit_Functions")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "Slit_Functions", "Figures")

# Spectral extraction region
CENTRAL_WAVELENGTH = 435.833  # Target emission line (nm)
DELTA_WAVELENGTH = 1.0        # Range (+/- nm)

# Resolving-power check: every clean Hg + Ar line in the lamp spectrum
# (air wavelengths, nm).  Lines in order gaps / blends / below SNR are skipped automatically.
HG_LINES = [296.728, 302.150, 312.567, 334.148, 365.015, 365.484, 366.328, 404.656,
            407.783, 435.833, 546.074, 576.960, 579.066]
AR_LINES = [696.543, 706.722, 714.704, 727.294, 738.398, 750.387, 751.465, 763.511,
            794.818, 800.616, 801.479, 810.369, 811.531, 826.452, 840.821, 842.465,
            852.144, 866.794]
# Optional plasma spectrum for comparison (same Ar lines, measured the same way); None to skip
COMPARE_FILE = None   # e.g. r"...\EchelleData\TestData\Percent0_1.spa"
COMPARE_LABEL = "plasma Ar"
MIN_SNR = 20.0
# =========================================================


def extract_central_line(file_path, output_dir, figures_dir, center_wl, delta_wl):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load 2-column ASCII data, skipping header row
    data = np.loadtxt(file_path, skiprows=1)
    raw_wavelength = data[:, 0]
    raw_intensity = data[:, 1]

    # Crop to range [center_wl - delta_wl, center_wl + delta_wl]
    min_wl = center_wl - delta_wl
    max_wl = center_wl + delta_wl
    mask = (raw_wavelength >= min_wl) & (raw_wavelength <= max_wl)

    wavelength = raw_wavelength[mask]
    intensity = raw_intensity[mask]

    if len(wavelength) == 0:
        raise ValueError(f"No spectral data found within range {min_wl:.3f} nm to {max_wl:.3f} nm.")

    # Locate peak intensity within cropped range
    peak_idx = np.argmax(intensity)
    peak_wl = wavelength[peak_idx]

    # Center wavelength scale around the peak (peak = 0.0 nm)
    relative_wavelength = wavelength - peak_wl

    # Normalize peak intensity to 1.0
    norm_intensity = intensity / intensity[peak_idx] if intensity[peak_idx] != 0 else intensity

    # Base filename without extension
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # ---------------------------------------------------------
    # 1. Save TXT Data
    # ---------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}.txt")

    output_data = np.column_stack((relative_wavelength, norm_intensity))
    np.savetxt(
        output_path,
        output_data,
        fmt="%.6f",
        delimiter="\t",
        header=(
            f"Target_Line: {center_wl} nm (+/- {delta_wl} nm)\n"
            f"Peak_Wavelength: {peak_wl:.4f} nm\n"
            f"Relative_Wavelength\tNormalized_Intensity"
        ),
        comments="#"
    )

    # ---------------------------------------------------------
    # 2. Plot & Save Figure
    # ---------------------------------------------------------
    os.makedirs(figures_dir, exist_ok=True)
    figure_path = os.path.join(figures_dir, f"{base_name}.png")

    plt.figure(figsize=(8, 5))
    plt.plot(relative_wavelength, norm_intensity, color="darkblue", linewidth=1.5, label="Slit Function")
    plt.axvline(0, color="red", linestyle="--", alpha=0.7, label=f"Peak ({peak_wl:.3f} nm)")
    
    plt.title(f"Slit Function Profile: {base_name}\nTarget: {center_wl} nm (±{delta_wl} nm)")
    plt.xlabel("Relative Wavelength (nm)")
    plt.ylabel("Normalized Intensity")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(figure_path, dpi=300)
    print(f"Extracted range: {min_wl:.3f} nm to {max_wl:.3f} nm")
    print(f"Peak line located at: {peak_wl:.4f} nm")
    print(f"Saved text data: {output_path}")
    print(f"Saved figure: {figure_path}")
    
    plt.show()


# =========================================================
# RESOLVING-POWER CHECK (all clean Hg + Ar lines)
# =========================================================
def _read_spa(path):
    d = np.loadtxt(path, skiprows=1)
    o = np.argsort(d[:, 0])
    return d[o, 0], d[o, 1]


def _longest_zero_run(v):
    best = run = 0
    for z in (v == 0):
        run = run + 1 if z else 0
        best = max(best, run)
    return best


def _pix_gauss(dx):
    """Gaussian (FWHM w) integrated over pixels of width dx: what the detector records.
    Fitting this instead of a point-sampled Gaussian removes most of the sampling-phase
    bias at ~2-3 px per FWHM."""
    def f(x, A, c, w, b):
        s, e = w / 2.3548200450309493, dx / 2
        return A * (erf((x + e - c) / (np.sqrt(2) * s)) - erf((x - e - c) / (np.sqrt(2) * s))) / 2 + b
    return f


def measure_line(wl, I, lam, min_snr=MIN_SNR):
    """Pixel-integrated Gaussian width of one line. None if in an order gap, weak or blended.
    Isolated exact zeros are dark-subtracted counts, not gaps (gap = >= 5 zeros in a row)."""
    i = np.searchsorted(wl, lam)
    if i < 25 or i > len(wl) - 25:
        return None
    k = i - 3 + np.argmax(I[i - 3:i + 4])
    if abs(wl[k] - lam) > 0.03 or _longest_zero_run(I[k - 12:k + 13]) >= 5:
        return None
    flank = np.r_[I[k - 20:k - 8], I[k + 8:k + 20]]
    b = np.median(flank)
    sig = max(1.4826 * np.median(np.abs(flank - b)), 1.0)
    h = I[k] - b
    if h / sig < min_snr or np.r_[I[k - 9:k - 4], I[k + 5:k + 10]].max() - b > 0.2 * h:
        return None
    dx = np.median(np.diff(wl[k - 10:k + 10]))
    x, y = wl[k - 6:k + 7], I[k - 6:k + 7]
    try:
        p, cov = curve_fit(_pix_gauss(dx), x, y, p0=[1.2 * h, wl[k], 1.8 * dx, b],
                           bounds=([0, wl[k] - dx, 0.2 * dx, -np.inf], [np.inf, wl[k] + dx, 8 * dx, np.inf]))
    except Exception:
        return None
    w, we = abs(p[2]), np.sqrt(cov[2, 2])
    if not np.isfinite(we) or we > 0.25 * w:
        return None
    return dict(lam=lam, centre=p[1], snr=h / sig, dx=dx, fwhm=w, fwhm_err=we,
                fwhm_px=w / dx, R=lam / w, R_err=lam * we / w ** 2)


def measure_lines(path, lines, label):
    wl, I = _read_spa(path)
    rows = []
    for lam in lines:
        r = measure_line(wl, I, lam)
        if r is not None:
            rows.append(dict(source=label, **r))
    return rows


def fit_power_law(lam, fwhm, clip=3.0, n_iter=10):
    """FWHM = C * lam^k, unweighted in log space with sigma clipping (line-to-line scatter,
    not photon noise, dominates).  k = 1 -> constant R;  k = 0 -> constant FWHM."""
    x, y = np.log(lam), np.log(fwhm)
    keep = np.ones(len(x), bool)
    for _ in range(n_iter):
        k, c = np.polyfit(x[keep], y[keep], 1)
        r = y - (k * x + c)
        mad = max(1.4826 * np.median(np.abs(r[keep] - np.median(r[keep]))), 1e-3)
        new = np.abs(r) < clip * mad
        if new.sum() < 3 or (new == keep).all():
            break
        keep = new
    k, c = np.polyfit(x[keep], y[keep], 1)
    r = y - (k * x + c)
    A = np.column_stack([x[keep], np.ones(keep.sum())])
    cov = np.linalg.inv(A.T @ A) * np.sum(r[keep] ** 2) / max(keep.sum() - 2, 1)
    return dict(k=k, k_err=np.sqrt(cov[0, 0]), c=c, keep=keep, scatter=np.std(r[keep]))


def check_resolving_power(cal_file, output_dir, figures_dir, compare_file=None):
    """Is R = lambda/FWHM constant?  Measures every clean Hg and Ar line in the lamp
    spectrum (plus the same Ar lines in a plasma spectrum, if given), fits FWHM ~ lambda^k
    and plots R and FWHM vs wavelength.  Writes a csv and a png."""
    import pandas as pd
    rows = measure_lines(cal_file, HG_LINES, "lamp Hg") + measure_lines(cal_file, AR_LINES, "lamp Ar")
    if compare_file:
        rows += measure_lines(compare_file, AR_LINES, COMPARE_LABEL)
    df = pd.DataFrame(rows)
    groups = {"lamp (Hg + Ar)": df.source.str.startswith("lamp")}
    if compare_file:
        groups[COMPARE_LABEL] = df.source == COMPARE_LABEL
    fits = {}
    print("\n=== resolving power:  FWHM ~ lambda^k  (k = 1 -> constant R, k = 0 -> constant FWHM) ===")
    for name, m in groups.items():
        sub = df[m]
        if len(sub) < 3:
            print(f"{name}: only {len(sub)} usable lines"); continue
        f = fit_power_law(sub.lam.to_numpy(), sub.fwhm.to_numpy())
        f["span"] = (sub.lam.min(), sub.lam.max())
        fits[name] = f
        df.loc[sub.index, "clipped"] = ~f["keep"]
        Rk = sub.R[f["keep"]]
        ktxt = (f"k = {f['k']:.2f} +- {f['k_err']:.2f}" if sub.lam.max() / sub.lam.min() > 1.5
                else "k undetermined (span < 1.5x in lambda)")
        print(f"{name:16s} n={len(sub):2d} ({(~f['keep']).sum()} clipped)   {ktxt}   "
              f"R median {Rk.median():.0f} (16-84%: {Rk.quantile(.16):.0f}-{Rk.quantile(.84):.0f})   "
              f"line-to-line scatter {100 * f['scatter']:.0f}%")
    base = os.path.splitext(os.path.basename(cal_file))[0]
    df.to_csv(os.path.join(output_dir, f"resolving_power_{base}.csv"), index=False)

    # ---- plot: (a) R vs lambda, (b) FWHM vs lambda --------------------------------
    style = {"lamp Hg": dict(fmt="D", color="tab:blue", mfc="tab:blue"),
             "lamp Ar": dict(fmt="o", color="0.35", mfc="0.35"),
             COMPARE_LABEL: dict(fmt="o", color="0.35", mfc="none")}
    lam = np.linspace(df.lam.min() - 20, df.lam.max() + 20, 300)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
    for src, g in df.groupby("source"):
        st = style.get(src, dict(fmt="s", color="k", mfc="none"))
        a1.errorbar(g.lam, g.R, yerr=g.R_err, ms=6, lw=0.7, label=src, **st)
        a2.errorbar(g.lam, 1e3 * g.fwhm, yerr=1e3 * g.fwhm_err, ms=6, lw=0.7, label=src, **st)
        cl = g[g.get("clipped", False) == True]
        a1.plot(cl.lam, cl.R, "x", color="tab:red", ms=9, mew=1.2)
        a2.plot(cl.lam, 1e3 * cl.fwhm, "x", color="tab:red", ms=9, mew=1.2)
    for (name, f), ls in zip(fits.items(), ("-", "--")):
        lo, hi = f["span"]
        lf = np.linspace(lo - 10, hi + 10, 200)           # draw each fit only over its own lines
        fw = np.exp(f["c"]) * lf ** f["k"]
        R0 = np.median(df[groups[name]].R[f["keep"]])
        if hi / lo > 1.5:
            lab = f"{name}: k = {f['k']:.2f} ± {f['k_err']:.2f}"
            a1.plot(lf, lf / fw, color="k", ls=ls, lw=1.2, label=lab)
            a2.plot(lf, 1e3 * fw, color="k", ls=ls, lw=1.2, label=lab)
        a1.plot(lf, np.full_like(lf, R0), color="0.55", ls=ls, lw=1.0, label=f"{name}: constant R = {R0:.0f}")
        a2.plot(lam, 1e3 * lam / R0, color="0.55", ls=ls, lw=1.0, label=f"{name}: constant R = {R0:.0f}")
    a1.plot([], [], "x", color="tab:red", label="clipped from fit")
    a1.set_ylabel(r"resolving power  $\lambda$ / FWHM"); a2.set_ylabel("FWHM (pm)")
    for ax in (a1, a2):
        ax.set_xlabel("air wavelength (nm)"); ax.grid(alpha=0.25); ax.legend(fontsize=7)
    a1.set_title("(a) R per line (pixel-integrated Gaussian fits); black = power-law fit, grey = constant R",
                 fontsize=9)
    a2.set_title("(b) FWHM per line; black = power-law fit, grey = constant R", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, f"resolving_power_{base}.png"), dpi=200)
    return df, fits


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    RP, RP_FITS = check_resolving_power(CALIBRATION_FILE, OUTPUT_DIR, FIGURES_DIR, COMPARE_FILE)
    extract_central_line(
        CALIBRATION_FILE, 
        OUTPUT_DIR, 
        FIGURES_DIR,
        center_wl=CENTRAL_WAVELENGTH, 
        delta_wl=DELTA_WAVELENGTH
    )