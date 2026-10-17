# -*- coding: utf-8 -*-
"""
QEProBandModel.py

The N2 second positive system (C3Pi_u -> B3Pi_g) at QE Pro resolution, built with the MolecularFitting tools, to take
the N2 bands off the Ar I 5p -> 4s lines (390-475 nm) of the Ar/N2 spectra.

1. Slit function in the MolecularFitting format: the QE Pro LSF of QEProCalibration (split Gaussian + Lorentzian,
   sigma free at knots; fitted to the Ar I lines of the pure-Ar spectra and to N2 band windows of a 5 % N2 spectrum
   against the echelle) integrated over the local pixel width ->
     MolecularFitting/Slit_Functions/<SLIT_STEM>.txt       profile at SLIT_REF_NM (relative wavelength, normalised)
     MolecularFitting/Slit_Functions/<SLIT_STEM>_lsf.json  the whole lambda-dependent model
   The echelle convention of Calibrate_Library_Echelle_V2 (one profile scaled with lambda, constant R) does not hold
   for the QE Pro (FWHM 1.3 nm at 420 nm, 0.9 at 700, 1.5 at 965): use the library of 2., not that script.
2. Calibrated library in the MolecularFitting format (Fit_One_Temp_V2.MolecularFitter(slit_stem=SLIT_STEM)): ExoMol 14N2
   C -> B lines (QEProCalibration.n2_cb_lines: air wavelengths, rotational energy above the lowest level of each v'),
   Boltzmann in v' (Tvib, energies of the ExoMol C levels) and in rotation (Trot), energy units, seen by the QE Pro
   (a pixel centred on each point of a LIB_STEP grid) -> MolecularFitting/Calibrated_Library/<SLIT_STEM>/14N2_CB/.
   The raw 14N2 library of MolecularFitting is not used: its wavelengths are vacuum and it stops at 450 nm.
3. Band + line fit of every QE Pro spectrum (both sweeps, mean of the repeats per power, response-corrected) in
   FIT_RANGE: C -> B templates per upper level v' = 0-4 at two rotational temperatures (TROT_COLD grid ~ Tg and a
   hot one,
     TROT_HOT: N2(C) made by Ar metastables radiates (37 ns) before it thermalises at 1.5 Torr; one Trot fits
   the band envelopes at 950-1500 K), and N2+ first negative B -> X templates per v' = 0-3 (not in ExoMol 14N2: band
   origins, P / R branches and Hoenl-London factors from the molecular constants, Franck-Condon factors of Morse
   potentials; heads 391.4, 427.8 / 423.6 / 419.9, 470.9 / 465.2 nm), all with amplitudes >= 0 and a common shift (+-BAND_SHIFT: the LSF asymmetry moves
   band heads against line centroids by up to ~0.1 nm), + every CR-model Ar I line above REL_MIN (LSF components,
   >= 0) + polynomial baseline (BASE_DEG) -> Ar I feature areas with the bands removed (the qepro_merged groups of
   QEProCRFit), v' populations, Tvib (Boltzmann line through the cold populations), residuals.
4. Check against the echelle (same plasma, same time): ln(QE Pro / echelle) of every feature (echelle: sum of its
   lines, none clipped or in a gap), minus the median of the red reference features of that spectrum - for 5 % N2
   with the bands removed (independent: the 5p lines of the Ar/N2 sweep were not used for the response) and pure Ar.
5. MolecularFitter demo: Fit_One_Temp_V2.MolecularFitter with the library of 2. on the 0-0 band window of the 5 % N2
   spectra (Trot, Tvib, shift).

Output in Experimental_Data/Output/QEProComparison/bands/: band_fits.csv, features_bandfree.csv,
check_vs_echelle.csv, molecularfitter_demo.csv, figures/fit_<sweep>_<P>W.png, check_vs_echelle.png; the files of
1-2 in MolecularFitting/.
Run after QEProCalibration.py: python SmallAnalysisScripts/QEProBandModel.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import QEProCalibration as qc                       # noqa: E402
import QEProCRFit as qf                             # noqa: E402

MF_DIR = os.path.join(qc.crf.ROOT_DIR, "MolecularFitting")
OUTDIR = os.path.join(qc.crf.ROOT_DIR, "Experimental_Data", "Output", "QEProComparison", "bands")
FIGDIR = os.path.join(OUTDIR, "figures")
SLIT_STEM = "QEPro_2026_10_13"
SLIT_REF_NM = 435.833                               # as the echelle slit function (Hg line)
LIB_RANGE, LIB_STEP = (280.0, 500.0), 0.05          # nm
LIB_TROT = np.arange(200.0, 3001.0, 50.0)           # K
LIB_TVIB = np.arange(1000.0, 15001.0, 500.0)        # K
FIT_RANGE = (385.0, 480.0)                          # nm, band + line fit (Delta v = -3 ... -5 bands, 5p -> 4s lines)
TROT_COLD = (400.0, 500.0, 600.0, 700.0)            # K, grid of the cold rotational component
TROT_HOT = (1500.0, 2500.0, 4000.0)                 # K, grid of the hot one
BAND_SHIFT = np.arange(-0.20, 0.201, 0.02)          # nm, common shift of the band templates
BASE_DEG = 2
WITH_N2PLUS = True                                  # N2+ first negative bands (B -> X) in the model
REL_MIN = 1e-3                                      # Ar I components: CR lines above this x the strongest of the window
RED_REF = (727.2936, 794.8176, 800.6157, 826.4522)  # echelle check: reference lines (never clipped, no bands)
DEMO_WINDOW = (333.0, 338.6)                        # MolecularFitter demo: 0-0 band (head 337.13 nm)


# ---- 1. slit function --------------------------------------------------------------------------
def write_slit_function(wl_cal, p):
    d = np.arange(-4.0, 4.0001, 0.01)
    D = float(np.interp(SLIT_REF_NM, wl_cal, np.gradient(wl_cal)))
    prof = qc.lsf_cdf(d + D / 2, SLIT_REF_NM, p) - qc.lsf_cdf(d - D / 2, SLIT_REF_NM, p)
    prof /= prof.max()
    out = os.path.join(MF_DIR, "Slit_Functions", f"{SLIT_STEM}.txt")
    np.savetxt(out, np.column_stack([d, prof]), fmt="%.6f", delimiter="\t", comments="#", header=(
        f"QE Pro instrument function at {SLIT_REF_NM} nm (QEProCalibration: split Gaussian + Lorentzian fitted to Ar I "
        f"lines and N2 bands, integrated over the {D:.3f} nm pixel)\n"
        f"Not a constant-R profile: use {SLIT_STEM}_lsf.json / the library built by SmallAnalysisScripts/QEProBandModel.py\n"
        "Relative_Wavelength\tNormalized_Intensity"))
    json.dump(dict(p, ref_nm=SLIT_REF_NM, pixel_width_ref_nm=D,
                   fwhm_nm={f"{l:g}": float(qc.fwhm(l, p)) for l in (310, 350, 400, 450, 500, 600, 700, 800, 900, 965)},
                   note="ln sigma linear between knots [nm]; asymmetry 0.8 tanh(a0); Lorentz fraction f, HWHM g; "
                        "pixel widths from Experimental_Data/Output/QEProComparison/calibration/wavelength_calibration.csv"),
              open(os.path.join(MF_DIR, "Slit_Functions", f"{SLIT_STEM}_lsf.json"), "w"), indent=1)
    return out


# ---- 2. calibrated library ---------------------------------------------------------------------
band_templates = qc.n2_cb_templates


# ---- N2+ first negative system (B2Sigma_u+ -> X2Sigma_g+) from molecular constants ----------------
N2P = dict(X=dict(Te=0.0, we=2207.00, wexe=16.10, Be=1.93176, ae=0.01881, De=6.10e-6, re=1.11642),
           B=dict(Te=25461.5, we=2419.84, wexe=23.18, Be=2.07456, ae=0.0240, De=6.17e-6, re=1.07493))   # cm^-1, A
N2P_VU, N2P_VL, N2P_NMAX = 4, 9, 80                 # v' = 0-3, v'' = 0-8, N' up to this


def morse_fcf(nu, nl):
    """Franck-Condon factors q(v', v'') of N2+ B-X from Morse potentials (numerical wavefunctions); they agree with
    the published ones within a few % (q00 0.67, q01 0.25, q12 0.29, q21 0.40)."""
    from scipy.linalg import eigh_tridiagonal
    hb = 16.857629 / (14.003074 / 2)                # hbar^2 / 2 mu [cm^-1 A^2]
    r = np.linspace(0.75, 1.75, 4000)
    h = r[1] - r[0]
    psi = {}
    for k in ("X", "B"):
        c = N2P[k]
        a, De = np.sqrt(c["wexe"] / hb), c["we"] ** 2 / (4 * c["wexe"])
        V = De * (1 - np.exp(-a * (r - c["re"]))) ** 2
        _, v = eigh_tridiagonal(2 * hb / h ** 2 + V, np.full(len(r) - 1, -hb / h ** 2), select="i",
                                select_range=(0, max(nu, nl)))
        psi[k] = v / np.sqrt(h * (v ** 2).sum(0))
    return (h * psi["B"].T @ psi["X"])[:nu, :nl] ** 2


def n2plus_lines():
    """N2+ B -> X rotational lines (P and R branches, spin splitting neglected; Hund b): air wavelength, v', N',
    upper rotational energy, statistical weight (2N'+1) x nuclear spin (2:1 alternation) and q nu^3 S/(2N'+1)."""
    q = morse_fcf(N2P_VU, N2P_VL)
    B, X = N2P["B"], N2P["X"]
    G = lambda c, v: c["we"] * (v + 0.5) - c["wexe"] * (v + 0.5) ** 2
    F = lambda c, v, N: (c["Be"] - c["ae"] * (v + 0.5)) * N * (N + 1) - c["De"] * (N * (N + 1)) ** 2
    rows = []
    Nu = np.arange(0, N2P_NMAX + 1)
    for vu in range(N2P_VU):
        Fu = F(B, vu, Nu)
        for vl in range(N2P_VL):
            nu0 = B["Te"] + G(B, vu) - G(X, vl)
            for dN, S in ((-1, Nu / (2 * Nu + 1)), (1, (Nu + 1) / (2 * Nu + 1))):     # R (N'' = N'-1), P (N'' = N'+1)
                Nl = Nu + dN
                ok = Nl >= 0
                nu = nu0 + Fu[ok] - F(X, vl, Nl[ok])
                rows.append(pd.DataFrame(dict(wl=qc.arn.vac_to_air(1e7 / nu), vu=vu, Nu=Nu[ok], Erot=Fu[ok],
                                              gN=(2 * Nu[ok] + 1) * np.where(Nu[ok] % 2 == 0, 2.0, 1.0),
                                              S=q[vu, vl] * nu ** 3 * S[ok])))
    return pd.concat(rows, ignore_index=True)


def n2plus_templates(lines, Trot, p, wl_axis, lo, hi, step=qc.FINE_STEP):
    """{v': (grid, signal)}: N2+ B -> X emission of one ion in v' at Trot (energy units, relative), seen by the QE Pro."""
    out = {}
    for v, g in lines.groupby("vu"):
        Q = float((g.drop_duplicates("Nu").gN * np.exp(-1.4387769 * g.drop_duplicates("Nu").Erot / Trot)).sum())
        w = g.S * g.gN * np.exp(-1.4387769 * g.Erot / Trot) / Q / g.wl
        out[int(v)] = qc.fine_convolve(g.wl.to_numpy(), w.to_numpy(), p, wl_axis, lo, hi, step=step)
    return out


def vib_energies(lev):
    e = lev.groupby("v").E.min()
    return (e - e.min()).to_dict()


def write_library(wl_cal, p, lines, lev):
    G = vib_energies(lev)
    grid = None
    per_v = []
    for T in LIB_TROT:
        tv = band_templates(lines, lev, T, p, wl_cal, *LIB_RANGE, step=LIB_STEP)
        grid = tv[0][0]
        per_v.append(np.array([tv[v][1] for v in sorted(tv)]))
    per_v = np.array(per_v)                                            # (Trot, v', wl)
    vs = sorted(G)
    boltz = np.exp(-1.4387769 * np.array([G[v] for v in vs])[None, :] / LIB_TVIB[:, None])   # (Tvib, v')
    cube = np.einsum("tv,rvw->trw", boltz, per_v).astype(np.float32)   # (Tvib, Trot, wl)
    d = Path(MF_DIR) / "Calibrated_Library" / SLIT_STEM / "14N2_CB"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "14N2_CB_calibrated_library.npz", wavelength_nm=grid, T_rot=LIB_TROT, T_vib=LIB_TVIB,
                        convolved_rot_matrix=cube, convolved_3d_matrix=cube, grid_step_nm=LIB_STEP,
                        echelle_scaling=False, ref_wavelength_nm=SLIT_REF_NM,
                        note="ExoMol 14N2 C->B, air wavelengths, energy units, QE Pro LSF (QEProBandModel.py)")
    return d


# ---- 3. band + line fit -------------------------------------------------------------------------
def fit_spectrum(wl, y, sat, R, p, ref, tmpl_cold, tmpl_hot, feats, lo=FIT_RANGE[0], hi=FIT_RANGE[1]):
    """One spectrum (counts) in [lo, hi]: bands (two Trot components per v', common shift) + Ar I components +
    polynomial baseline, on the response-corrected spectrum -> dict (best fit, feature areas and errors)."""
    m = (wl > lo) & (wl < hi)
    x, yc = wl[m], y[m] / R(wl[m])
    var = qc.pixel_variance(y[m]) / R(wl[m]) ** 2
    w = 1 / np.sqrt(var + (qc.NOISE["lsf_peak"] * yc.max()) ** 2)
    ok = ~sat[m]
    edges = qc.pixel_edges(x)
    comps = qc.components(ref, lo - 1, hi + 1, rel_min=REL_MIN)
    A_ar = np.column_stack([qc.comp_profile(edges, c, p) for c in comps])
    base = np.column_stack([((x - x.mean()) / (hi - lo)) ** k for k in range(BASE_DEG + 1)])
    best = None
    for Tc, tc in tmpl_cold.items():
        for Th, th in tmpl_hot.items():
            keys = [("cold",) + k for k in sorted(tc)] + [("hot",) + k for k in sorted(th)]
            for d in BAND_SHIFT:
                B = np.column_stack([np.interp(x - d, *tc[k]) for k in sorted(tc)]
                                    + [np.interp(x - d, *th[k]) for k in sorted(th)]) if tc else np.zeros((len(x), 0))
                X = np.column_stack([B, A_ar, base])
                nb, na = B.shape[1], A_ar.shape[1]
                sc = np.abs(X).max(axis=0)                  # columns to unit maximum (the N2+ templates carry nu^3)
                sc[sc == 0] = 1.0
                lb = np.r_[np.zeros(nb + na), np.full(base.shape[1], -np.inf)]
                r = lsq_linear((X / sc * w[:, None])[ok], (yc * w)[ok], bounds=(lb, np.full(X.shape[1], np.inf)),
                               method="bvls")
                c2 = float(np.sum(r.fun ** 2))
                if best is None or c2 < best["chi2"]:
                    best = dict(chi2=c2, Tc=Tc, Th=Th, shift=d, coef=r.x / sc, X=X, nb=nb, na=na, keys=keys, sc=sc)
    X, coef, nb, na, sc = best["X"], best["coef"], best["nb"], best["na"], best["sc"]
    Xw = (X / sc * w[:, None])[ok]
    cov = (np.linalg.pinv(Xw.T @ Xw) * max(1.0, best["chi2"] / max(ok.sum() - X.shape[1], 1))) / np.outer(sc, sc)
    ia = slice(nb, nb + na)
    areas, rows = coef[ia], []
    for f in feats.itertuples():
        share = np.array([sum(wc for wc, l in zip(c["w"], c["lines"]) if np.isclose(f.lines, l, atol=1e-3).any())
                          for c in comps])
        if share.sum() == 0:
            continue
        a = float(share @ areas)
        e = float(np.sqrt(share @ cov[ia, ia] @ share))
        rows.append(dict(feature=f.feature, wl=f.wl, area=a, err=e))
    amp = dict(zip(best["keys"], coef[:nb]))           # (cold | hot, system, v') -> population (relative)
    sysflux = {sy: X[:, [i for i, k in enumerate(best["keys"]) if k[1] == sy]] @ coef[[i for i, k in
                                                                                     enumerate(best["keys"]) if k[1] == sy]]
               for sy in {k[1] for k in best["keys"]}}
    return dict(chi2_red=best["chi2"] / max(ok.sum() - X.shape[1], 1), Trot_cold=best["Tc"], Trot_hot=best["Th"],
                shift=best["shift"], amp=amp, x=x, y=yc, model=X @ coef, bands=X[:, :nb] @ coef[:nb],
                sysflux=sysflux, ar=X[:, ia] @ areas, base=X[:, nb + na:] @ coef[nb + na:],
                feats=pd.DataFrame(rows), ok=ok)


def tvib(pops, G):
    """Tvib [K] of a Boltzmann line through the populations of v' (> 0) against G(v') [cm^-1]; nan if < 2."""
    v = np.array(sorted(G))[:len(pops)]
    k = pops > 0
    if k.sum() < 2:
        return np.nan
    s = np.polyfit(np.array([G[i] for i in v[k]]), np.log(pops[k]), 1)[0]
    return -1.4387769 / s if s < 0 else np.inf


# ---- 4. check against the echelle -------------------------------------------------------------
def echelle_check(ft_q, sweep, feats):
    """ln(QE Pro / echelle) per feature and power, minus the median of the RED_REF features of that power."""
    e = qf.echelle_group_table(sweep, feats)
    if e.empty:
        return pd.DataFrame()
    E = e.groupby(["x", "feature"]).area.mean().rename("E")
    Q = ft_q.groupby(["x", "feature"]).area.mean().rename("Q")
    t = pd.concat([E, Q], axis=1).dropna().reset_index()
    t = t[(t.Q > 0) & (t.E > 0)]
    t["ln_ratio"] = np.log(t.Q / t.E)
    wl = dict(zip(feats.feature, feats.wl))
    t["wl"] = t.feature.map(wl)
    red = {f.feature for f in feats.itertuples() if any(np.isclose(l, r, atol=2e-3) for l in f.lines for r in RED_REF)}
    norm = t[t.feature.isin(red)].groupby("x").ln_ratio.median()
    t["rel"] = t.ln_ratio - t.x.map(norm)
    return t.assign(sweep=sweep)


# ---- 5. MolecularFitter demo ------------------------------------------------------------------
def molecularfitter_demo(wl, R):
    sys.path.insert(0, MF_DIR)
    from Fit_One_Temp_V2 import MolecularFitter
    mf = MolecularFitter(calib_lib_dir=Path(MF_DIR) / "Calibrated_Library", slit_stem=SLIT_STEM)
    q = qf.mean_spectra(qc.load_qepro("N2_5pct"))
    rows = []
    m = (wl > DEMO_WINDOW[0]) & (wl < DEMO_WINDOW[1])
    for k, P in enumerate(q["files"].P):
        y = q["spec"][k][m] / R(wl[m])
        r = mf.fit_spectrum(wl[m], y / y.max(), "14N2_CB", fit_baseline=True, max_shift_nm=0.3, fit_tvib=False,
                            default_tvib=5000.0)
        rows.append(dict(P=P, T_rot=r["T_rot"], shift_nm=r["shift_nm"], rmse=r["rmse"]))
        print(f"  MolecularFitter, 0-0 band, 5 % N2 {P:g} W: T_rot {r['T_rot']:.0f} K, shift {r['shift_nm']:+.3f} nm, "
              f"rmse {r['rmse']:.3f}", flush=True)
    return pd.DataFrame(rows)


# ---- plots ------------------------------------------------------------------------------------
def plot_fit(r, title, path, lines_5p):
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3]))
    ax.plot(r["x"], r["y"], "o", color="k", ms=3, label="QE Pro (response-corrected)")
    ax.plot(r["x"], r["model"], "-", color="C3", lw=1.2, label="fit")
    ax.plot(r["x"], r["bands"] + r["base"], "-", color="C0", lw=1.0, label="N$_2$ C$\\rightarrow$B bands + baseline")
    ax.plot(r["x"], r["ar"], "-", color="C2", lw=1.0, label="Ar I lines (fit)")
    ax.plot(r["x"], r["y"] - r["bands"] - r["base"], ".-", color="C1", ms=3, lw=0.6,
            label="measured - bands - baseline")
    for l in lines_5p:
        ax.axvline(l, color="C2", lw=0.4, ls=":")
    ax.set_ylabel("signal (echelle units, relative)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10)
    bx.plot(r["x"], (r["y"] - r["model"]) / r["y"].max(), ".-", color="C3", ms=3, lw=0.6)
    bx.axhline(0, color="k", lw=0.5)
    bx.set_ylabel("residual / max")
    bx.set_xlabel("Wavelength [nm]")
    bx.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_check(chk, path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.get_cmap("viridis")
    for (sweep, mk) in (("pure_Ar", "o"), ("N2_5pct", "s")):
        c = chk[chk.sweep == sweep]
        Ps = sorted(c.x.unique())
        for i, P in enumerate(Ps):
            g = c[np.isclose(c.x, P)]
            ax.plot(g.wl + (0.6 if sweep == "N2_5pct" else -0.6), g.rel, mk, color=cmap(i / max(len(Ps) - 1, 1)),
                    mfc="none" if sweep == "N2_5pct" else None, ms=5,
                    label=f"{qc.SWEEPS[sweep]['label']}" if i == len(Ps) // 2 else None)
    ax.axhline(0, color="k", lw=0.7)
    for v in (-0.2, 0.2):
        ax.axhline(v, color="0.6", lw=0.6, ls="--")
    ax.set_xlabel("feature wavelength [nm]")
    ax.set_ylabel("ln(QE Pro / echelle), relative to\n727.3/794.8/800.6/826.5 of the same spectrum")
    ax.set_title("QE Pro feature areas with the N$_2$ bands removed against the echelle (open: 5 % N$_2$, filled: "
                 "pure Ar; colour: power)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    wl, p, R = qc.load_calibration()
    print("slit function:", write_slit_function(wl, p), flush=True)
    lines, lev = qc.n2_cb_lines()
    print("library:", write_library(wl, p, lines, lev), flush=True)
    G = vib_energies(lev)
    ref = qc.reference_lines()
    feats = qf.qepro_features(ref)["qepro_merged"]
    feats = feats[(feats.wl > FIT_RANGE[0] + 1) & (feats.wl < FIT_RANGE[1] - 1)].reset_index(drop=True)
    lo, hi = FIT_RANGE
    nplines = n2plus_lines()

    def templates(T):
        t = {("CB", v): g for v, g in band_templates(lines, lev, T, p, wl, lo - 2, hi + 2, qc.FINE_STEP).items()}
        if WITH_N2PLUS:
            t.update({("NP", v): g for v, g in n2plus_templates(nplines, T, p, wl, lo - 2, hi + 2).items()})
        return t
    tmpl_cold = {T: templates(T) for T in TROT_COLD}
    tmpl_hot = {T: templates(T) for T in TROT_HOT}
    lines_5p = [l for ls in feats.lines for l in ls]
    rows, ft, checks = [], [], []
    for sweep in qc.SWEEPS:
        q = qf.mean_spectra(qc.load_qepro(sweep))
        no_n2 = qc.SWEEPS[sweep]["N2"] == 0             # pure Ar: no band templates (else they fit wings / continuum)
        for k, P in enumerate(q["files"].P):
            r = fit_spectrum(wl, q["spec"][k], q["sat"][k], R, p, ref, {0.0: {}} if no_n2 else tmpl_cold,
                             {0.0: {}} if no_n2 else tmpl_hot, feats)
            tot = np.sum(r["y"][r["ok"]])
            band_frac = float(np.sum(r["bands"][r["ok"]]) / tot) if len(r["amp"]) else 0.0
            cb_c = np.array([r["amp"].get(("cold", "CB", v), 0.0) for v in sorted(G)])
            cb_h = np.array([r["amp"].get(("hot", "CB", v), 0.0) for v in sorted(G)])
            rows.append(dict(sweep=sweep, P=P, chi2_red=r["chi2_red"], Trot_cold=r["Trot_cold"],
                             Trot_hot=r["Trot_hot"], shift=r["shift"], band_fraction=band_frac,
                             n2plus_fraction=float(np.sum(r["sysflux"].get("NP", np.zeros(1))[r["ok"]]) / tot)
                             if "NP" in r["sysflux"] else 0.0,
                             Tvib_cold=tvib(cb_c, G), hot_fraction=float(cb_h.sum() / max(cb_c.sum() + cb_h.sum(), 1e-300)),
                             **{f"pop_v{v}": float(a + b) for v, a, b in zip(sorted(G), cb_c, cb_h)}))
            f = r["feats"].assign(sweep=sweep, file=f"mean_{P:g}W", x=P, rep=1)
            ft.append(f)
            print(f"  {sweep:<8s} {P:3.0f} W: chi2/dof {r['chi2_red']:.2f}, bands {100 * band_frac:.0f} % "
                  f"(N2+ {100 * rows[-1]['n2plus_fraction']:.0f} %) of the "
                  f"{lo:g}-{hi:g} nm signal, Trot {r['Trot_cold']:.0f} + {r['Trot_hot']:.0f} K, shift {r['shift']:+.2f} nm",
                  flush=True)
            if P in (10.0, 50.0, 90.0):
                plot_fit(r, f"QE Pro {qc.SWEEPS[sweep]['label']}, {P:g} W: N$_2$ C$\\rightarrow$B bands (v' = 0-4, "
                            f"Trot {r['Trot_cold']:.0f} K + {r['Trot_hot']:.0f} K) + Ar I lines + baseline",
                         os.path.join(FIGDIR, f"fit_{sweep}_{P:g}W.png"), lines_5p)
    bf = pd.DataFrame(rows)
    bf.to_csv(os.path.join(OUTDIR, "band_fits.csv"), index=False)
    ft = pd.concat(ft, ignore_index=True)
    ft["rel_err"] = ft.err / ft.area.where(ft.area > 0)
    ft.to_csv(os.path.join(OUTDIR, "features_bandfree.csv"), index=False)
    # echelle check (features of the fit range + the red reference features from the QE Pro line fits)
    allf = qf.qepro_features(ref)["qepro_merged"]
    red_f = allf[[any(np.isclose(l, r_, atol=2e-3) for l in f.lines for r_ in RED_REF) for f in allf.itertuples()]]
    for sweep in qc.SWEEPS:
        if not qf.has_echelle(sweep):
            continue
        red_q = qf.qepro_feature_table(sweep, red_f, ref, (wl, p, R))     # the reference lines, same calibration
        q_all = pd.concat([ft[ft.sweep == sweep][["x", "feature", "area"]], red_q[["x", "feature", "area"]]])
        checks.append(echelle_check(q_all, sweep, allf))
    chk = pd.concat(checks, ignore_index=True)
    chk.to_csv(os.path.join(OUTDIR, "check_vs_echelle.csv"), index=False)
    plot_check(chk[chk.wl < 690], os.path.join(FIGDIR, "check_vs_echelle.png"))
    print("\nln(QE Pro / echelle) of the 5p features, relative to the red reference lines (median over the powers):")
    print(chk[chk.wl < 690].groupby(["feature", "sweep"]).rel.median().unstack().round(2).to_string())
    demo = molecularfitter_demo(wl, R)
    demo.to_csv(os.path.join(OUTDIR, "molecularfitter_demo.csv"), index=False)
    print("written to", OUTDIR)
    return dict(bf=bf, ft=ft, chk=chk, demo=demo)


if __name__ == "__main__":
    RES = main()
