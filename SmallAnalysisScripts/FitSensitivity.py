# -*- coding: utf-8 -*-
"""
FitSensitivity.py

How strongly do the fitted Ar I lines respond to Ne and to Te_eff?  From the cached
CR-model tables of the fits (CRFitNeTe), for the lines the fit uses:

  shape change = RMS over the lines of d ln I_line, after removing what the fit removes
                 for free - one overall scale per spectrum and (response_deg = 1) a linear
                 response tilt in wavelength - per decade of Ne (at the fitted E/N) and per
                 0.1 eV of Te_eff (at the fitted Ne).

The lines carry ~20 % model error each (sigma_model) plus their repeat scatter, so with
n lines a shape change below ~sigma/sqrt(n) is invisible to the fit (chi^2 changes by
less than 1); that level is drawn as the detection limit.

Figure in Experimental_Data/Output/FitSensitivity/.  Run from Spyder: edit the settings, press F5.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT_DIR, "Experimental_Data", "Output")

# ---- settings -------------------------------------------------------------------
CASES = [   # label, CR table, fit folder (features + fitted E/N, Ne), sweep, x, colour
    ("Ar, BOLSIG+ microwave", "EEDFPhysics/cr_model_table_Ar_Biagi_bolsig_mw.npz", None, None, None, "C4"),
    ("Ar, BOLSIG+ DC", "EEDFPhysics/cr_model_table_Ar_Biagi_bolsig.npz", None, None, None, "C0"),
    ("Ar + 2.4 % N$_2$, BOLSIG+ microwave",
     "ActinometryN2Content/fit_microwave_2.4pct_N2/cr_model_table_ArN2_2.4pct_bolsig_mw_N2_2.4pct.npz",
     "ActinometryN2Content/fit_microwave_2.4pct_N2", "Power", 85.0, "C3"),
    ("Ar + 2.4 % N$_2$, BOLSIG+ DC",
     "ActinometryN2Content/fit_dc_2.4pct_N2/cr_model_table_ArN2_2.4pct_bolsig_N2_2.4pct.npz",
     "ActinometryN2Content/fit_dc_2.4pct_N2", "Power", 85.0, "C1"),
]
PURE_AR_FIT = {"Ar, BOLSIG+ microwave": (6.9, 1e15), "Ar, BOLSIG+ DC": (1.63, 2.6e16)}   # E/N [Td], Ne:
                                        # pure-Ar best fits, calibration trusted (EEDFPhysics summary)
FEATURES = "ActinometryN2Content/fit_microwave_2.4pct_N2/features.csv"   # the fitted lines
MERGE_NM = 0.10
SIGMA_LINE = 0.22                       # ln, per line: sigma_model 0.2 + typical repeat scatter
OUTDIR = os.path.join(OUT, "FitSensitivity")


def feature_logI(tab, wl_feat):
    """ln(energy intensity) of each fitted line on the (E/N, Ne) grid, shape (nx, nNe, nfeat)."""
    I = tab["I_obs"] / tab["line_wl"]
    cols = [I[:, :, np.abs(tab["line_wl"] - w) < MERGE_NM].sum(-1) for w in wl_feat]
    return np.log(np.stack(cols, -1)), np.asarray(wl_feat)


def shape(dlnI, lam, tilt):
    """Remove an overall scale (and a linear tilt in wavelength) from line changes (..., nfeat)."""
    X = np.stack([np.ones_like(lam), (lam - 775) / 100], 1) if tilt else np.ones((len(lam), 1))
    P = np.eye(len(lam)) - X @ np.linalg.pinv(X)
    return dlnI @ P.T


def rms(a):
    return np.sqrt(np.mean(a ** 2, axis=-1))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    wl = pd.read_csv(os.path.join(OUT, FEATURES)).wl.to_numpy()
    floor = SIGMA_LINE / np.sqrt(len(wl))
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    rows = []
    for label, tpath, fdir, sweep, x, col in CASES:
        tab = dict(np.load(os.path.join(OUT, tpath), allow_pickle=False))
        G, lam = feature_logI(tab, wl)
        lx, lN = np.log(tab["x_grid"]), np.log10(tab["Ne_grid"])
        if fdir:
            c = pd.read_csv(os.path.join(OUT, fdir, "fit_conditions.csv")).set_index(["sweep", "x"]).loc[(sweep, x)]
            EN, Ne = c.x_med, c.Ne_med
        else:
            EN, Ne = PURE_AR_FIT[label]
        i = int(np.argmin(np.abs(lx - np.log(EN))))
        j = int(np.argmin(np.abs(lN - np.log10(Ne))))
        Te = np.asarray(tab["Te_eff"], float)
        for tilt, ls in ((False, ":"), (True, "-")):
            # per decade of Ne along the fitted E/N row
            dN = shape(np.gradient(G[i], lN, axis=0), lam, tilt)
            axs[0].semilogx(10 ** lN, rms(dN), ls, color=col, lw=2,
                            label=f"{label} (E/N = {tab['x_grid'][i]:g} Td)" if tilt else None)
            # per 0.1 eV of Te_eff along the EEDF axis at the fitted Ne
            order = np.argsort(Te)
            Ts = Te[order]
            ok = np.diff(Ts) > 1e-3
            keep = np.r_[True, ok]
            dT = shape(np.gradient(G[order][keep][:, j], Ts[keep], axis=0), lam, tilt) * 0.1
            axs[1].plot(Ts[keep], rms(dT), ls, color=col, lw=2,
                        label=f"{label} ($N_e$ = {tab['Ne_grid'][j]:.1e})" if tilt else None)
            if tilt:
                rows.append(dict(case=label, EN=tab["x_grid"][i], Ne=tab["Ne_grid"][j],
                                 per_decade_Ne_at_fit=rms(dN[j]),
                                 per_0p1eV_at_fit=float(np.interp(Te[i], Ts[keep], rms(dT)))))
        axs[0].plot(Ne, rms(shape(np.gradient(G[i], lN, axis=0), lam, True))[j], "o", color=col, ms=9, mec="k")
    for ax in axs:
        ax.axhline(floor, color="k", ls="--", lw=1.2,
                   label=f"detection limit ≈ {SIGMA_LINE}/√{len(wl)} lines = {floor:.2f}")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 3)
        ax.grid(alpha=0.3, which="both")
    axs[0].set_xlabel("$N_e$ [m$^{-3}$]")
    axs[0].set_ylabel("RMS change of the line pattern per decade of $N_e$ (ln)")
    axs[0].set_title("Sensitivity to $N_e$ at the fitted E/N (dot = fitted $N_e$)")
    axs[1].set_xlabel(r"$T_{e,\mathrm{eff}}$ [eV]")
    axs[1].set_ylabel(r"RMS change of the line pattern per 0.1 eV of $T_{e,\mathrm{eff}}$ (ln)")
    axs[1].set_title("Sensitivity to $T_{e,\\mathrm{eff}}$ at the fitted $N_e$")
    axs[0].legend(fontsize=8, loc="lower left")
    axs[1].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"How much the {len(wl)} fitted Ar I lines change - solid: after removing scale and response tilt "
                 "(as the fit does), dotted: scale only", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fit_sensitivity.png"), dpi=170)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUTDIR, "fit_sensitivity.csv"), index=False)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"detection limit {floor:.3f}; figure in {OUTDIR}")
    plt.show()


if __name__ == "__main__":
    main()
