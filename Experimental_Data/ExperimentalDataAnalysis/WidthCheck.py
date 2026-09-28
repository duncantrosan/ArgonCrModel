# -*- coding: utf-8 -*-
"""
N I line-width check (screen for dissociative excitation) for the Ar/N2 analysis.

Fast N atoms from dissociative excitation of N2 (e + N2 -> N* + N + fragments
with ~eV kinetic energy) show up as Doppler broadening of N I lines well beyond
the instrument function.  Thermal N (300-1000 K) is ~2-5 pm FWHM in the near-IR,
far below the echelle width, so any measurable excess points at fast fragments.

Per N I line:
  1. isolate it (subtract the other components of its group fit),
  2. free single-Gaussian fit (width bound 8x instrument; the pipeline's
     fwhm_free is capped at 3x, so it is NOT reused here),
  3. instrument width from nearby clean Ar I lines (default), the pipeline's
     linear IFN fit, or a user-supplied echelle slit function,
  4. excess width in quadrature -> Doppler T and 1.5kT equivalent energy,
  5. optional narrow (instrument) + broad Gaussian fit -> fast-atom fraction.

Ar I lines are run through the same fit, so they set the instrument width and
also the empirical scatter.  Pixel-sampling bias largely cancels because both
species are fitted identically.

Run from Spyder: press F5.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
import Argon_Nitrogen_Mix_Analysis_V2 as arn        # needed so Spectrum unpickles
from Argon_Nitrogen_Mix_Analysis_V2 import load_results, gauss, gauss_area

KB, C, AMU, EV = 1.380649e-23, 2.99792458e8, 1.66053907e-27, 1.602176634e-19
MASS_AMU = {"N I": 14.007, "Ar I": 39.948}

# Project root (ArgonCrModel folder), found relative to this file
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MainPath = os.path.join(ROOT_DIR, "Experimental_Data", "Output", "ArN2_out")
CONFIG = dict(
    results=os.path.join(MainPath, "results.pkl"),
    out_csv=os.path.join(MainPath, "width_check.csv"),
    out_png=os.path.join(MainPath, "width_check.png"),
    out_pdf=os.path.join(MainPath, "width_check_NI.pdf"),   # None to skip per-line panels
    spectrum="mix",
    statuses=("ok", "blend"),       # 'weak' widths are mostly noise; 'saturated' flat-tops read wide
    snr_min=15.0,
    win_fwhm=5.0,                   # fit window +- this x instrument FWHM
    inst_ref="ar_local",            # 'ar_local' | 'ifn' | 'slit'
    ar_local_nm=40.0,               # Ar lines within +- this set the local resolving power
    ar_min_lines=3,
    slit_fn=None,                   # callable lam_nm -> instrument FWHM (nm); built from slit_file if None
    slit_file=os.path.join(ROOT_DIR, "Experimental_Data", "EchelleData", "TestData", "Slit_Functions", "09_04_2026.txt"),   # Hg 435.83 profile; None to skip
    slit_lam0=435.833,
    two_component=True,
    broad_max=12.0,                 # broad-component FWHM upper bound (x instrument)
    sig_flag=3.0,                   # excess-width significance to flag a line as broadened
    # --- slit / Doppler diagnostic plots ---
    slit_plots=True,
    out_res=os.path.join(MainPath, "width_check_resolution.png"),
    out_conv=os.path.join(MainPath, "width_check_convolution.png"),
    T_gas=700.0,                    # gas temperature for the thermal Doppler reference (K)
    E_fast_eV=(1.0, 5.0),           # fast-fragment 'temperatures' shown as 1.5kT = E
    n_ar_conv=3,                    # strongest Ar lines in the N I range added to the convolution panels
)
# plot colours: N I green, Hg slit blue, Doppler expectations one purple ramp (+ line styles), Ar grey
C_N, C_HG, C_AR = "tab:green", "tab:blue", "0.45"
C_DOP = ("#c2a5cf", "#9970ab", "#40004b")        # light -> dark = cold -> fast


# --- Doppler helpers ----------------------------------------------------------
def doppler_T(w_d, lam, m_amu):
    """Gaussian Doppler FWHM (nm) at lam (nm) -> temperature (K)."""
    return (np.asarray(w_d) / lam) ** 2 * m_amu * AMU * C ** 2 / (8 * KB * np.log(2))


def doppler_fwhm(T, lam, m_amu):
    return lam * np.sqrt(8 * KB * T * np.log(2) / (m_amu * AMU * C ** 2))


def quad_excess(w, w_inst):
    return np.sqrt(np.maximum(np.asarray(w) ** 2 - w_inst ** 2, 0.0))


def T_to_eV(T):
    return 1.5 * KB * np.asarray(T) / EV


def eV_to_T(E):
    return np.asarray(E) * EV / (1.5 * KB)


# --- fitting --------------------------------------------------------------------
def _isolate(i, r, fits, sp, hw):
    """Window around line i with the *other* components of its group fit subtracted."""
    c = r.centre_fit
    x, y = sp.window(c - hw, c + hw)
    gf = fits.get(int(r.group)) if np.isfinite(r.group) else None
    if gf is not None and i in list(gf["idx"]) and len(gf["idx"]) > 1:
        j = list(gf["idx"]).index(i)
        for k in range(len(gf["centres"])):
            if k != j:
                y = y - gauss(x, gf["coef"][k], gf["centres"][k] + gf["shift"], gf["fwhm"])
    return x, y


def fit_single(x, y, c0, w0):
    f = lambda x, A, c, w, b0, b1: gauss(x, A, c, w) + b0 + b1 * (x - c0)
    A0 = max(y.max() - np.median(y), 1e-9)
    p0 = [A0, c0, w0, np.median(y), 0.0]
    lb = [0, c0 - w0, 0.3 * w0, -np.inf, -np.inf]
    ub = [np.inf, c0 + w0, 8.0 * w0, np.inf, np.inf]
    try:
        p, cov = curve_fit(f, x, y, p0=p0, bounds=(lb, ub), maxfev=5000)
    except Exception:
        return None
    e = np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov)) else np.full(5, np.nan)
    return dict(A=p[0], centre=p[1], fwhm=p[2], fwhm_err=e[2], at_bound=p[2] > 7.9 * w0,
                model=f(x, *p))


def fit_two(x, y, c0, w_inst, broad_max):
    """Narrow Gaussian fixed at instrument width + free broad Gaussian, common centre."""
    f = lambda x, An, Ab, c, wb, b0, b1: (gauss(x, An, c, w_inst) + gauss(x, Ab, c, wb)
                                          + b0 + b1 * (x - c0))
    A0 = max(y.max() - np.median(y), 1e-9)
    p0 = [0.8 * A0, 0.2 * A0, c0, 3.0 * w_inst, np.median(y), 0.0]
    lb = [0, 0, c0 - w_inst, 1.5 * w_inst, -np.inf, -np.inf]
    ub = [np.inf, np.inf, c0 + w_inst, broad_max * w_inst, np.inf, np.inf]
    try:
        p, cov = curve_fit(f, x, y, p0=p0, bounds=(lb, ub), maxfev=8000)
    except Exception:
        return None
    e = np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov)) else np.full(6, np.nan)
    an, ab = gauss_area(p[0], w_inst), gauss_area(p[1], p[3])
    base = p[4] + p[5] * (x - c0)
    return dict(fwhm_b=p[3], fwhm_b_err=e[3], broad_frac=ab / (an + ab) if an + ab > 0 else np.nan,
                broad_sig=p[1] / e[1] if e[1] > 0 else np.nan, model=f(x, *p),
                narrow=gauss(x, p[0], p[2], w_inst) + base, broad=gauss(x, p[1], p[2], p[3]) + base)


# --- instrument width -----------------------------------------------------------
def hg_slit_fn(path, lam0):
    """Hg calibration line -> FWHM(lambda), scaled at constant resolving power.
    Same point-sampled Gaussian estimator as fit_single, so ratios are like-for-like.
    (The export grid is constant dlambda/lambda, so this is also constant width in pixels.)"""
    x, y = np.loadtxt(path, comments="#").T
    f = lambda x, A, c, w, b: gauss(x, A, c, w) + b
    p, _ = curve_fit(f, x, y, p0=[1.0, x[np.argmax(y)], 0.025, 0.0])
    w0 = abs(p[2])
    fn = lambda lam: float(w0 * lam / lam0)
    fn.fwhm0, fn.R = w0, lam0 / w0
    return fn


def ar_reference_widths(res, fits, sp, fwhm_fn, cfg):
    """Same free fit on clean, isolated, strong Ar I lines -> table of widths."""
    sel = res[(res.species == "Ar I") & (res.status == "ok") & (res.snr >= cfg["snr_min"])]
    if "contamination" in sel:
        sel = sel[sel.contamination.astype(str).str.startswith("clean")]
    rows = []
    for i, r in sel.iterrows():
        w0 = float(fwhm_fn(r.centre_fit))
        x, y = _isolate(i, r, fits, sp, cfg["win_fwhm"] * w0)
        if len(x) < 8:
            continue
        ff = fit_single(x, y, r.centre_fit, w0)
        if ff is None or ff["at_bound"] or not np.isfinite(ff["fwhm_err"]):
            continue
        rows.append(dict(species="Ar I", idx=i, wl_air=r.wl_air, lower=r.lower, upper=r.upper,
                         status=r.status, snr=r.snr, contamination=r.get("contamination", ""),
                         fwhm_meas=ff["fwhm"], fwhm_err=ff["fwhm_err"]))
    return pd.DataFrame(rows)


def make_inst_fn(cfg, ar_tab, fwhm_fn):
    if cfg["inst_ref"] == "slit":
        if cfg["slit_fn"] is None:
            raise ValueError("inst_ref='slit' needs CONFIG['slit_fn']")
        return cfg["slit_fn"]
    if cfg["inst_ref"] == "ifn":
        return lambda lam: float(fwhm_fn(lam))
    if len(ar_tab) < cfg["ar_min_lines"]:
        raise RuntimeError("too few clean Ar I reference lines; use inst_ref='ifn'")
    # local resolving power (FWHM/lambda) from nearby Ar lines; global median as fallback
    q = (ar_tab.fwhm_meas / ar_tab.wl_air).to_numpy()
    wl = ar_tab.wl_air.to_numpy()
    q_all = np.median(q)

    def fn(lam):
        m = np.abs(wl - lam) < cfg["ar_local_nm"]
        return float(lam * (np.median(q[m]) if m.sum() >= cfg["ar_min_lines"] else q_all))
    return fn


# --- driver ---------------------------------------------------------------------
def width_check(cfg=CONFIG):
    R = load_results(cfg["results"])
    key = cfg["spectrum"]
    res, sp, fits, fwhm_fn = R[f"res_{key}"], R[f"sp_{key}"], R[f"fits_{key}"], R["fwhm_fn"]

    ar_tab = ar_reference_widths(res, fits, sp, fwhm_fn, cfg)
    if cfg["slit_fn"] is None and cfg.get("slit_file") and os.path.exists(cfg["slit_file"]):
        cfg = dict(cfg, slit_fn=hg_slit_fn(cfg["slit_file"], cfg["slit_lam0"]))
        print(f"Hg slit: FWHM {1e3 * cfg['slit_fn'].fwhm0:.1f} pm at {cfg['slit_lam0']} nm (R = {cfg['slit_fn'].R:.0f})")
    inst = make_inst_fn(cfg, ar_tab, fwhm_fn)
    ar_tab["fwhm_inst"] = [inst(w) for w in ar_tab.wl_air]
    ar_tab["ratio"] = ar_tab.fwhm_meas / ar_tab.fwhm_inst
    # empirical width-ratio scatter of Ar lines (robust); floor at 2 %
    scat = max(1.4826 * np.median(np.abs(ar_tab.ratio - np.median(ar_tab.ratio))), 0.02) if len(ar_tab) else 0.05

    sel = res[(res.species == "N I") & res.status.isin(cfg["statuses"]) & (res.snr >= cfg["snr_min"])]
    rows, panels = [], []
    for i, r in sel.sort_values("wl_air").iterrows():
        c = r.centre_fit
        wi = inst(c)
        x, y = _isolate(i, r, fits, sp, cfg["win_fwhm"] * wi)
        if len(x) < 8:
            continue
        ff = fit_single(x, y, c, wi)
        if ff is None:
            continue
        rec = dict(species="N I", wl_air=r.wl_air, lower=r.lower, upper=r.upper, status=r.status,
                   snr=r.snr, contamination=r.get("contamination", ""),
                   fwhm_meas=ff["fwhm"], fwhm_err=ff["fwhm_err"], fwhm_inst=wi,
                   ratio=ff["fwhm"] / wi, at_bound=ff["at_bound"])
        tf = fit_two(x, y, c, wi, cfg["broad_max"]) if cfg["two_component"] else None
        if tf is not None:
            wb = quad_excess(tf["fwhm_b"], wi)
            rec.update(broad_frac=tf["broad_frac"], broad_sig=tf["broad_sig"],
                       fwhm_broad=tf["fwhm_b"], T_broad_K=doppler_T(wb, c, MASS_AMU["N I"]))
            rec["E_broad_eV"] = T_to_eV(rec["T_broad_K"])
        rows.append(rec)
        panels.append((rec, x, y, ff, tf))

    out = pd.concat([pd.DataFrame(rows), ar_tab], ignore_index=True)
    # excess width -> Doppler T, both species; significance against fit error + Ar scatter
    m_amu = out.species.map(MASS_AMU)
    out["ratio"] = out.fwhm_meas / out.fwhm_inst
    out["fwhm_excess"] = quad_excess(out.fwhm_meas, out.fwhm_inst)
    out["T_dopp_K"] = doppler_T(out.fwhm_excess, out.wl_air, m_amu)
    out["E_eq_eV"] = T_to_eV(out.T_dopp_K)
    out["excess_sig"] = (out.fwhm_meas - out.fwhm_inst) / np.sqrt(out.fwhm_err ** 2 + (scat * out.fwhm_inst) ** 2)
    out["broadened"] = out.excess_sig > cfg["sig_flag"]
    if cfg["slit_fn"] is not None:            # Hg-slit comparison, whatever inst_ref is
        out["fwhm_slit"] = [cfg["slit_fn"](l) for l in out.wl_air]
        out["ratio_slit"] = out.fwhm_meas / out.fwhm_slit
        for s in ("Ar I", "N I"):
            v = out.loc[out.species == s, "ratio_slit"]
            if len(v):
                print(f"{s}: FWHM / Hg slit  median {v.median():.2f}  (16-84% {v.quantile(.16):.2f}-{v.quantile(.84):.2f}, n={len(v)})")
    out.to_csv(cfg["out_csv"], index=False)

    # sensitivity: smallest excess we can call = sig_flag x Ar scatter
    lam0 = 746.83
    w0 = inst(lam0)
    Tmin = doppler_T(quad_excess(w0 * (1 + cfg["sig_flag"] * scat), w0), lam0, MASS_AMU["N I"])
    print(f"instrument ref '{cfg['inst_ref']}': {len(ar_tab)} Ar lines, width-ratio scatter {scat:.3f}")
    print(f"at {lam0} nm: instrument {1e3 * w0:.1f} pm, thermal N 300 K "
          f"{1e3 * doppler_fwhm(300, lam0, MASS_AMU['N I']):.1f} pm; "
          f"min detectable T ~ {Tmin:.0f} K ({T_to_eV(Tmin):.2f} eV)")
    cols = ["wl_air", "lower", "upper", "snr", "ratio", "excess_sig", "E_eq_eV",
            "broad_frac", "E_broad_eV", "contamination"]
    nt = out[out.species == "N I"]
    print(nt[[c for c in cols if c in nt]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    _summary_plot(out, scat, cfg)
    if cfg["out_pdf"]:
        _line_pdf(panels, cfg["out_pdf"])
    if cfg.get("slit_plots"):
        ar_pure = (ar_reference_widths(R["res_pure"], R["fits_pure"], R["sp_pure"], fwhm_fn, cfg)
                   if "res_pure" in R else pd.DataFrame())
        _resolution_plot(out, ar_pure, inst, fwhm_fn, scat, cfg)
        if cfg.get("slit_file") and os.path.exists(cfg["slit_file"]):
            _convolution_plot(panels, out, res, fits, sp, inst, cfg)
    plt.show()
    return out


# --- plots ----------------------------------------------------------------------
def _summary_plot(out, scat, cfg):
    ar, nt = out[out.species == "Ar I"], out[out.species == "N I"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    a1.axhspan(1 - scat, 1 + scat, color="0.85", lw=0)
    a1.axhline(1, color="k", lw=0.6)
    a1.errorbar(ar.wl_air, ar.ratio, yerr=ar.fwhm_err / ar.fwhm_inst, fmt="o", ms=3,
                color="0.5", lw=0.6, label="Ar I (reference)")
    contam = nt.contamination.astype(str).str.startswith("contaminated")
    for m, mfc, lab in ((~contam, "tab:green", "N I"), (contam, "none", "N I contaminated")):
        a1.errorbar(nt.wl_air[m], nt.ratio[m], yerr=(nt.fwhm_err / nt.fwhm_inst)[m], fmt="s", ms=5,
                    color="tab:green", mfc=mfc, lw=0.8, label=lab)
    a1.set_ylabel("FWHM / instrument"); a1.legend(fontsize=8)
    a1.set_title(f"width vs instrument ({cfg['inst_ref']}); grey band = Ar scatter", fontsize=9)
    for col, mk, lab in (("E_eq_eV", "s", "single-Gaussian excess"), ("E_broad_eV", "^", "broad component")):
        if col in nt:
            v = nt[col]
            ok = np.isfinite(v) & (v > 0)
            a2.plot(nt.wl_air[ok], v[ok], mk, color="tab:green" if mk == "s" else "tab:purple",
                    mfc="none" if mk == "^" else None, label=lab)
    a2.set_yscale("log"); a2.set_ylabel("equivalent 1.5kT (eV)")
    a2.set_xlabel("air wavelength (nm)"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(cfg["out_png"], dpi=150)


def _line_pdf(panels, path, per_page=12):
    with PdfPages(path) as pdf:
        for p0 in range(0, len(panels), per_page):
            chunk = panels[p0:p0 + per_page]
            fig, axs = plt.subplots(3, 4, figsize=(17, 11))
            for ax, (rec, x, y, ff, tf) in zip(axs.flat, chunk):
                ax.plot(x, y, ".-", color="crimson", lw=0.7, ms=3, label="data (neighbours removed)")
                ax.plot(x, ff["model"], "k-", lw=0.8, label=f"single, x{rec['ratio']:.2f}")
                ax.plot(x, gauss(x, ff["A"], ff["centre"], rec["fwhm_inst"]) + (ff["model"] - gauss(x, ff["A"], ff["centre"], ff["fwhm"])),
                        color="0.6", ls=":", lw=0.8, label="instrument width")
                if tf is not None:
                    ax.plot(x, tf["narrow"], color="tab:blue", ls="--", lw=0.7, label="narrow")
                    ax.plot(x, tf["broad"], color="tab:purple", ls="--", lw=0.7,
                            label=f"broad {tf['broad_frac']:.0%}")
                ax.set_title(f"N I {rec['wl_air']:.2f}  {rec['lower']}->{rec['upper']}\n"
                             f"SNR {rec['snr']:.0f}, {rec['contamination']}", fontsize=7)
                ax.tick_params(labelsize=6); ax.legend(fontsize=5)
            for ax in axs.flat[len(chunk):]:
                ax.axis("off")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


# --- slit function / Doppler diagnostics ----------------------------------------
def hg_template(path):
    """Hg calibration profile with zero baseline, centred on its centroid
    (the file's 0 is the peak pixel, which sits ~half a pixel off the true centre)."""
    x, y = np.loadtxt(path, comments="#").T
    y = y - np.median(y[np.abs(x) > 0.3])
    core = np.abs(x) < 0.05
    return x - np.sum(x[core] * y[core]) / np.sum(y[core]), y


def slit_doppler(hx, hy, lam0, lam, fwhm_d, n_sub=20):
    """Unit-area profile(dlam): Hg slit scaled to lam at constant R (the export grid is
    constant dlam/lam, so the pixel integration scales with it), convolved with a
    Gaussian Doppler profile of FWHM fwhm_d (nm) on a fine grid."""
    xs = hx * lam / lam0
    step = np.median(np.diff(xs)) / n_sub
    xf = np.arange(xs[0], xs[-1], step)
    yf = np.interp(xf, xs, hy)
    if fwhm_d > step:
        n = np.ceil(4 * fwhm_d / step)
        k = np.arange(-n, n + 1) * step
        g = gauss(k, 1.0, 0.0, fwhm_d)
        yf = np.convolve(yf, g / g.sum(), mode="same")
    yf = yf / (yf.sum() * step)
    return lambda d: np.interp(d, xf, yf, left=0.0, right=0.0)


def fit_template(x, y, P, c0, half, n_shift=61):
    """y ~ A P(x - c) + b0 + b1 (x - c0): c scanned, A/b0/b1 linear. Returns (chi2, model fn)."""
    best = None
    for c in np.linspace(c0 - half, c0 + half, n_shift):
        X = np.column_stack([P(x - c), np.ones_like(x), x - c0])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        chi = np.sum((y - X @ coef) ** 2)
        if best is None or chi < best[0]:
            best = (chi, c, coef)
    chi, c, coef = best
    return chi, (lambda xx: coef[0] * P(xx - c) + coef[1] + coef[2] * (xx - c0))


def _resolution_plot(out, ar_pure, inst, fwhm_fn, scat, cfg):
    """(a) resolving power lambda/FWHM measured from the lines themselves;
    (b) N I FWHM against instrument (+) N Doppler at T_gas and at fast-fragment energies.
    Fits with relative width error > 50 % are drawn hollow without error bars."""
    ar, nt = out[out.species == "Ar I"], out[out.species == "N I"]
    good = lambda d: (d.fwhm_err / d.fwhm_meas < 0.5) & np.isfinite(d.fwhm_err)
    slit = cfg["slit_fn"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # (a) resolving power vs wavelength
    lam = np.linspace(min(out.wl_air.min(), cfg["slit_lam0"]) - 10, out.wl_air.max() + 10, 400)
    for d, mfc, lab in ((ar_pure, C_AR, "Ar I, pure"), (ar, "none", f"Ar I, {cfg['spectrum']}")):
        if len(d):
            d = d[good(d)]
            Rm = d.wl_air / d.fwhm_meas
            a1.errorbar(d.wl_air, Rm, yerr=Rm * d.fwhm_err / d.fwhm_meas, fmt="o", ms=5, mfc=mfc,
                        color=C_AR, lw=0.6, label=lab)
    if len(nt):
        g = good(nt)
        Rm = nt.wl_air / nt.fwhm_meas
        a1.errorbar(nt.wl_air[g], Rm[g], yerr=(Rm * nt.fwhm_err / nt.fwhm_meas)[g], fmt="s", ms=6,
                    color=C_N, lw=0.8, label="N I")
        if (~g).any():
            a1.plot(nt.wl_air[~g], Rm[~g], "s", ms=6, mfc="none", color=C_N, label="N I, poorly constrained")
    a1.plot(lam, lam / np.array([inst(l) for l in lam]), color="k", lw=1.2,
            label=f"instrument ref ({cfg['inst_ref']})")
    a1.plot(lam, lam / np.asarray(fwhm_fn(lam)), color="k", ls=":", lw=1, label="pipeline IFN fit")
    if slit is not None:
        R_hg = cfg["slit_lam0"] / slit(cfg["slit_lam0"])
        a1.axhline(R_hg, color=C_HG, ls="--", lw=1)
        a1.plot(cfg["slit_lam0"], R_hg, "*", ms=15, color=C_HG, mec="w",
                label=f"Hg {cfg['slit_lam0']:.2f} nm, R = {R_hg:.0f} (dashed: constant R)")
    a1.set_xlabel("air wavelength (nm)"); a1.set_ylabel(r"resolving power  $\lambda$ / FWHM")
    a1.set_title("(a) resolving power measured from the lines", fontsize=10)
    a1.grid(alpha=0.25); a1.legend(fontsize=8)

    # (b) N I widths vs instrument (+) Doppler
    lo, hi = (nt.wl_air.min() - 15, nt.wl_air.max() + 15) if len(nt) else (730.0, 875.0)
    lam = np.linspace(lo, hi, 300)
    wi = np.array([inst(l) for l in lam])
    a2.fill_between(lam, 1e3 * wi * (1 - scat), 1e3 * wi * (1 + scat), color="0.9", lw=0,
                    label=f"instrument ±{scat:.0%} (Ar line-to-line scatter)")
    a2.plot(lam, 1e3 * wi, color="k", lw=1.2, label="instrument ref")
    if slit is not None:
        a2.plot(lam, 1e3 * np.array([slit(l) for l in lam]), color=C_HG, ls="--", lw=1, label="Hg slit, scaled")
    curves = [(cfg["T_gas"], f"{cfg['T_gas']:.0f} K")] + \
             [(float(eV_to_T(E)), f"{E:g} eV ({float(eV_to_T(E)):.0f} K)") for E in cfg["E_fast_eV"]]
    for (T, lab), col, ls in zip(curves, C_DOP, ("-", "--", "-.")):
        a2.plot(lam, 1e3 * np.hypot(wi, doppler_fwhm(T, lam, MASS_AMU["N I"])), color=col, ls=ls, lw=1.5,
                label=f"instrument ⊕ N Doppler {lab}")
    arw = ar[(ar.wl_air > lo) & (ar.wl_air < hi)]
    arw = arw[good(arw)]
    a2.errorbar(arw.wl_air, 1e3 * arw.fwhm_meas, yerr=1e3 * arw.fwhm_err, fmt="o", ms=5, mfc="none",
                color=C_AR, lw=0.6, label="Ar I")
    if len(nt):
        g = good(nt)
        a2.errorbar(nt.wl_air[g], 1e3 * nt.fwhm_meas[g], yerr=1e3 * nt.fwhm_err[g], fmt="s", ms=6,
                    color=C_N, lw=0.8, label="N I")
        if (~g).any():
            a2.plot(nt.wl_air[~g], 1e3 * nt.fwhm_meas[~g], "s", ms=6, mfc="none", color=C_N,
                    label="N I, poorly constrained")
        for r in nt.itertuples():
            a2.annotate(f"{r.wl_air:.1f}", (r.wl_air, 1e3 * r.fwhm_meas), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="0.25")
    lm = 0.5 * (lo + hi)
    dtxt = ", ".join(f"{1e3 * doppler_fwhm(T, lm, MASS_AMU['N I']):.1f}" for T, _ in curves[:len(C_DOP)])
    a2.set_title(f"(b) N I widths vs instrument ⊕ Doppler   (N Doppler FWHM at {lm:.0f} nm: {dtxt} pm)",
                 fontsize=10)
    a2.set_xlim(lo, hi); a2.set_xlabel("air wavelength (nm)"); a2.set_ylabel("FWHM (pm)")
    a2.grid(alpha=0.25); a2.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(cfg["out_res"], dpi=150)


def _convolution_plot(panels, out, res, fits, sp, inst, cfg):
    """Each fitted N I line (plus the strongest Ar lines in the same range) against the Hg
    slit scaled to its wavelength: slit alone and slit convolved with Doppler profiles for
    the line's own mass.  Amplitude, sub-pixel shift and linear baseline are fitted per curve;
    chi2 is quoted relative to slit-only."""
    hx, hy = hg_template(cfg["slit_file"])
    items = [(rec, x, y) for rec, x, y, _, _ in panels]
    nt = out[out.species == "N I"]
    if len(nt) and "idx" in out:
        ar = out[(out.species == "Ar I") & out.wl_air.between(nt.wl_air.min() - 20, nt.wl_air.max() + 20)]
        for _, a in ar.sort_values("snr", ascending=False).head(cfg["n_ar_conv"]).iterrows():
            i = int(a.idx)
            r = res.loc[i]
            x, y = _isolate(i, r, fits, sp, cfg["win_fwhm"] * inst(r.centre_fit))
            items.append((a.to_dict(), x, y))
    items = items[:12]
    if not items:
        return
    ncols = 4
    nrows = int(np.ceil(len(items) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.3 * nrows), squeeze=False)
    temps = [0.0, cfg["T_gas"]] + [float(eV_to_T(E)) for E in cfg["E_fast_eV"]][:len(C_DOP) - 1]
    labs = ["slit only", f"⊗ {cfg['T_gas']:.0f} K"] + [f"⊗ {E:g} eV" for E in cfg["E_fast_eV"]][:len(C_DOP) - 1]
    styles = [(C_HG, "-")] + list(zip(C_DOP, ("-", "--", "-.")))
    for ax, (rec, x, y) in zip(axs.flat, items):
        lam, m = rec["wl_air"], MASS_AMU[rec["species"]]
        wi = inst(lam)
        c0 = x[np.argmax(y)]
        xx = np.linspace(x.min(), x.max(), 800)
        ax.plot(x, y, drawstyle="steps-mid", color="0.15", lw=1.0, label="data (pixels)")
        chi0 = None
        for T, lab, (col, ls) in zip(temps, labs, styles):
            P = slit_doppler(hx, hy, cfg["slit_lam0"], lam, doppler_fwhm(T, lam, m))
            chi, mod = fit_template(x, y, P, c0, 0.5 * wi)
            chi0 = chi if chi0 is None else chi0
            ax.plot(xx, mod(xx), color=col, ls=ls, lw=1.3, label=f"{lab}   χ²/χ²₀ = {chi / chi0:.2f}")
        ax.set_xlim(c0 - 3 * wi, c0 + 3 * wi)
        ax.set_title(f"{rec['species']} {lam:.2f}  {rec['lower']}->{rec['upper']}   SNR {rec['snr']:.0f}",
                     fontsize=8)
        ax.tick_params(labelsize=6); ax.legend(fontsize=6, loc="upper right")
    for ax in axs.flat[len(items):]:
        ax.axis("off")
    fig.suptitle("lines vs Hg 435.83 slit scaled to each wavelength, convolved with Doppler "
                 "(line's own mass; amplitude, shift and baseline fitted per curve)", fontsize=10)
    fig.tight_layout(); fig.savefig(cfg["out_conv"], dpi=150)


if __name__ == "__main__":
    WIDTHS = width_check(CONFIG)