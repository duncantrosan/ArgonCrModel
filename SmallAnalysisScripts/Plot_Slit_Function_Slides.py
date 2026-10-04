"""
Slide figures for the instrument-broadening model.

  (1) slit_function_vs_delta.png  measured Hg 435.833 nm profile vs the ideal delta
                                   line the analysis assumes the lamp line to be
  (2) fwhm_vs_wavelength.png      measured FWHM of every clean lamp line vs the
                                   constant-R scaling used in Calibrate_Library_Echelle_V2
                                   (S is stretched by lambda / lambda_ref -> FWHM ~ lambda)

Equations are left off the plots so they can be typeset on the slide.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent / "MolecularFitting"
SLIT_DIR = BASE_DIR / "Slit_Functions"
FIG_DIR = SLIT_DIR / "Figures" / "Slides"
SLIT_STEM = "09_04_2026"
LAMBDA_REF = 435.833  # nm, Hg line the slit function was taken from
X_HALF = 0.10         # nm window shown around the line

plt.rcParams.update({
    "font.size": 16, "axes.labelsize": 18, "legend.fontsize": 14,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "axes.linewidth": 1.2,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
C_MEAS, C_MODEL = "#1f4e9c", "#c0392b"


def fwhm_from_profile(x, y):
    """FWHM by linear interpolation of the half-maximum crossings."""
    k = np.argmax(y)
    half = y[k] / 2
    i = k
    while y[i] > half:
        i -= 1
    left = np.interp(half, [y[i], y[i + 1]], [x[i], x[i + 1]])
    j = k
    while y[j] > half:
        j += 1
    right = np.interp(half, [y[j], y[j - 1]], [x[j], x[j - 1]])
    return right - left, left, right


def plot_slit_vs_delta():
    x, y = np.loadtxt(SLIT_DIR / f"{SLIT_STEM}.txt", unpack=True)
    x = x - x[np.argmax(y)]
    m = np.abs(x) <= X_HALF
    x, y = x[m] * 1e3, y[m]  # pm
    fwhm, lo, hi = fwhm_from_profile(x, y)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot([0, 0], [0, 1.0], color=C_MODEL, lw=2.5, label=r"Ideal line  $\delta(\lambda-\lambda_0)$")
    ax.plot(x, y, "-o", color=C_MEAS, lw=2.2, ms=6, label=r"Measured slit function  $S(\Delta\lambda)$")
    ax.annotate("", xy=(lo, 0.5), xytext=(hi, 0.5),
                arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.5))
    ax.text(hi + 4, 0.5, f"FWHM = {fwhm:.0f} pm", va="center", fontsize=15, color="0.2")

    ax.set_xlim(-X_HALF * 1e3, X_HALF * 1e3)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel(r"$\lambda - \lambda_0$  (pm)")
    ax.set_ylabel("Normalized intensity")
    ax.text(0.98, 0.95, f"Hg I {LAMBDA_REF:.3f} nm", transform=ax.transAxes, ha="right", va="top", fontsize=16)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=14)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(FIG_DIR / "slit_function_vs_delta.png", transparent=True)
    plt.show()
    return fwhm


def plot_fwhm_vs_wavelength(fwhm_ref_pm):
    df = pd.read_csv(SLIT_DIR / f"resolving_power_{SLIT_STEM}.csv")
    df = df[df.source.str.startswith("lamp")]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for src, mk in (("lamp Hg", "D"), ("lamp Ar", "o")):
        if src == "lamp Hg":
            g = df[df.source == src]
            ax.errorbar(g.lam, 1e3 * g.fwhm, yerr=1e3 * g.fwhm_err, fmt=mk, ms=8, color=C_MEAS,
                    label="Measured Hg lines")
    lam = np.linspace(280, 820, 200)
    ax.plot(lam, fwhm_ref_pm * lam / LAMBDA_REF, color=C_MODEL, lw=2.5,
            label=r"Model: FWHM $\propto \lambda$ (constant $R$)")
    ax.plot(LAMBDA_REF, fwhm_ref_pm, ".", ms=20, color='red',
            label=r"Reference $\lambda_{\rm ref}$ = 435.8 nm")

    ax.set_xlim(280, 600)
    ax.set_ylim(0, None)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("FWHM (pm)")
    ax.legend(loc="upper left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(FIG_DIR / "fwhm_vs_wavelength.png", transparent=True)
    plt.show()



if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fwhm = plot_slit_vs_delta()
    plot_fwhm_vs_wavelength(fwhm)
    print(f"Reference FWHM {fwhm:.1f} pm  ->  R = {LAMBDA_REF / (fwhm * 1e-3):.0f}")
    print(f"Saved to {FIG_DIR}")
