# -*- coding: utf-8 -*-
"""
CRFitNeTe.py

Fit Te and Ne of every sweep condition by comparing measured Ar I line
intensities with the argon CR model (Scripts/MainFileV2.py).  The N2 in the
discharge is ignored for now (pure-Ar CR model at the total pressure).

1. Lines: Ar I lines between CR-model levels, picked from the single-spectrum
   analysis (TestData/ArN2_out/lines_mix.csv + manual ratings): detected,
   clean, inside the fit wavelength ranges.  Lines closer than merge_nm are
   merged into one feature (measured and modelled intensities summed), since
   the group fitter cannot split them reliably.
2. Measurement: every spectrum of both sweeps is fitted with the group fitter
   of Argon_Nitrogen_Mix_Analysis_V2 (through ActinometryLineSelection).
   Spectra exported without the intensity calibration (header flags, see
   ActinometryLineSelection.filter_export_flags) are dropped.  Cached.
3. Model: CR model on an (EEDF, Ne) grid at the sweep pressure.  The EEDF axis
   is either a Maxwellian Te sweep (eedf='maxwell') or the E/N sweep of a
   MultiBolt export (eedf=<run folder>; Boltzmann EEDFs with the inelastic
   depletion of the tail).  Everything is reported against Te_eff = 2/3 <E>
   (= Te for a Maxwellian).  Stored per grid point: observed line intensities
   n_u A eta (photons; eta is the escape factor of the line, same as in the
   balance equations), level densities, and the fraction of each level's
   production that is direct excitation from the ground state.  Cached.
4. Fit, per condition (all repeats jointly) and per spectrum (as a check):
       ln I_meas = ln I_model(x, Ne) + s_spectrum + ln R(lambda) + noise
   x          : Te or E/N (the EEDF axis)
   s_spectrum : free scale per spectrum (exposure, absolute calibration)
   ln R       : residual spectral response, polynomial in lambda of degree
                response_deg (None = trust the intensity calibration)
   covariance : per point the line-fit error (1/SNR plus a floor) and the
                measured repeat-to-repeat scatter of that line (10-40 %, far
                above 1/SNR); plus a CR-model error sigma_model per line that
                is fully correlated between repeats - repeats do not average
                down model errors.
   The linear nuisances (s, R) are profiled out analytically and chi^2 is
   evaluated on a fine ln x - ln Ne grid -> posterior (log-uniform prior).  If
   chi^2_min/dof > 1 the posterior is widened by that factor (Birge ratio).

The 5p -> 4s lines (415-470 nm) carry the most EEDF information (thresholds
~1.3 eV above the 4p).  The intensity calibration is checked within each range
by same-upper-level branching ratios (within ~0.2 in ln for the clean 5p and
4p9 pairs), but nothing ties the two ranges together, hence response_deg=1.

A Maxwellian EEDF fits these spectra only at Te ~0.6-1 eV (its tail above the
11.5 eV thresholds is far too full at 1 Torr); use the MultiBolt EEDFs.

Run from Spyder: edit CONFIG, press F5.  Results in Experimental_Data/Output/CRFit.
"""
import ast
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ActinometryLineSelection as als              # noqa: E402

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "Scripts")
# Biagi Ar, E/N = 2-50 Td, 12-decade EEDF span:
#   he.RunMultiBolt(<MultiBolt>/cross-sections/Biagi_Ar.txt, 'Ar_Biagi_lowEN_span12',
#                   [2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8.5, 10, 12, 15, 20, 30, 50], remap_span=12)
MULTIBOLT_RUN = os.path.join(ROOT_DIR, "InputData", "MultiBolt", "Ar_Biagi_lowEN_span12")

CONFIG = dict(
    # --- line selection (single-spectrum analysis + manual ratings) ----------
    selection_dir=als.CONFIG["selection_dir"],
    lines_file="lines_mix.csv",
    ar_ratings="Ar_manual_ratings.csv",
    status=("ok",),
    contamination=("clean", "clean?"),
    min_rating=4,                       # unrated lines pass on the automatic flags
    wl_ranges=[(400.0, 475.0), (690.0, 860.0)],     # 5p -> 4s and 4p -> 4s lines
    merge_nm=0.10,                      # model lines closer than this form one feature
    exclude_wl=(),                      # air wavelengths [nm] to drop by hand
    # --- measurement ---------------------------------------------------------
    sweeps=als.CONFIG["sweeps"],
    export_flags=als.CONFIG["export_flags"],    # spectra with other header flags are dropped
    window_nm=1.5,                      # catalogue lines within this of a fitted line are fitted too
    reuse_measurements=True,
    valid_status=("ok", "weak", "blend", "not_detected"),   # allowed component statuses in a feature
    max_rel_err=0.15,                   # feature dropped from a spectrum above this relative error
    # --- CR model ------------------------------------------------------------
    P_Torr=1.0, Tg=300.0, R=0.04,       # both sweeps are at 1 Torr
    trap_lines="all",                   # see SolveDirect in MainFileV2
    eedf=MULTIBOLT_RUN,                 # 'maxwell', or a MultiBolt export folder (E/N sweep)
    Te_grid=np.geomspace(0.5, 6.0, 32), # eV, eedf='maxwell' only
    EN_range=None,                      # (min, max) Td of the MultiBolt sweep to use; None = all
    min_eedf_Emax=15.0,                 # eV; drop EEDFs exported only below this (no 4p/5p excitation)
    Ne_grid=np.geomspace(1e15, 3e19, 28),   # m^-3
    reuse_model=True,
    # --- fit -----------------------------------------------------------------
    intensity_units="energy",           # calibrated spectra: 'energy' (radiance, model x 1/lambda) or 'counts'
    response_file=None,                 # 2 columns: wavelength [nm], relative response (measured / true)
    response_deg=1,                     # None = trust the calibration, 1 = ln R linear in lambda
                                        # (ties the 400-475 and 690-860 nm ranges, which the
                                        # branching-ratio checks only verify separately)
    sigma_fit_floor=0.02,               # relative, added in quadrature to 1/SNR
    use_repeat_scatter=True,            # add the measured repeat-to-repeat scatter of each line
    sigma_model=0.20,                   # relative CR-model error per line
    birge=True,                         # widen the posterior by chi^2_min/dof when > 1
    fine_n=(240, 220),                  # fine (EEDF axis, Ne) grid for the posterior
    # --- output --------------------------------------------------------------
    outdir=os.path.join(ROOT_DIR, "Experimental_Data", "Output", "CRFit"),
)


# ----------------------------------------------------------------------------
# 0. CR model (functions of MainFileV2.py, loaded without running its sweep)
# ----------------------------------------------------------------------------
def _helpers():
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    import HelperFunctions as he
    return he


def load_cr_model(outdir):
    he = _helpers()
    path = os.path.join(SCRIPTS_DIR, "MainFileV2.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    defs = ast.Module([n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))], [])
    cr = {"__file__": path, "OUTPUT_DIR": outdir, "plt": plt}
    exec(compile(defs, path, "exec"), cr)
    with contextlib.redirect_stdout(io.StringIO()):
        ModelData, RTM = he.GetData()
    tau = np.unique([d["Tau_R"] for d in RTM])
    shape = np.unique([d["Shape"] for d in RTM])
    eta = np.array([d["EscapeFactor"][0] for d in RTM]).reshape(len(tau), len(shape))
    interp = RegularGridInterpolator((np.log10(tau), shape), np.log10(eta), bounds_error=False, fill_value=None)
    return cr, he, ModelData, interp


def eedf_axis(cfg=CONFIG):
    """(EEDF inputs for CRModel, axis values x, axis name, Te_eff per x)."""
    if cfg["eedf"] == "maxwell":
        Te = np.asarray(cfg["Te_grid"], float)
        return [float(t) for t in Te], Te, "Te [eV]", Te
    he = _helpers()
    with contextlib.redirect_stdout(io.StringIO()):
        E = he.ImportMultiBoltEEDFs(cfg["eedf"], verbose=False)
    x = np.array([e["sweep"]["value"] for e in E], float)
    lo, hi = cfg["EN_range"] or (-np.inf, np.inf)
    E_max = np.array([e["E"][e["EEDF"] > 0].max() for e in E])
    keep = (x >= lo) & (x <= hi) & (E_max >= cfg["min_eedf_Emax"])
    E = [e for e, k in zip(E, keep) if k]
    Te_eff = np.array([e["Te_eff"] for e in E])
    if np.any(np.diff(Te_eff) <= 0):
        raise ValueError("Te_eff of the MultiBolt EEDFs is not increasing with E/N - restrict EN_range")
    return E, x[keep], f"E/N [{E[0]['sweep']['unit']}]", Te_eff


def te_label(tab):
    return (r"$T_e$ [eV] (Maxwellian)" if str(tab["eedf"]) == "maxwell"
            else r"$T_{e,\mathrm{eff}} = \frac{2}{3}\langle\varepsilon\rangle$ [eV] (MultiBolt, E/N sweep)")


def _model_lines(MD):
    """Every radiative line between model levels except those to the ground state."""
    return [(lbl, rad["partner"], rad["wavelength_nm"], rad["coeff"])
            for lbl, s in MD.items() for rad in s["RadiativeDecay"]
            if rad["direction"] == "loss" and rad["partner"] != "ground"]


def _table_path(cfg):
    tag = "maxwell" if cfg["eedf"] == "maxwell" else os.path.basename(os.path.normpath(cfg["eedf"]))
    return os.path.join(cfg["outdir"], f"cr_model_table_{tag}.npz")


_PER_X = ("x_grid", "Te_eff", "I_obs", "I_thin", "dens", "fdirect", "converged")


def _cached_subset(tab, cfg, x):
    """The cached table restricted to the axis values x, or None if it does not hold them."""
    try:
        same = (np.allclose(tab["Ne_grid"], cfg["Ne_grid"])
                and np.allclose([tab["P_Torr"], tab["Tg"], tab["R"]], [cfg["P_Torr"], cfg["Tg"], cfg["R"]])
                and str(tab["trap_lines"]) == cfg["trap_lines"] and str(tab["eedf"]) == str(cfg["eedf"]))
    except (KeyError, ValueError):
        return None
    rows = [np.flatnonzero(np.isclose(tab["x_grid"], v, rtol=1e-9)) for v in x]
    if not same or any(len(r) != 1 for r in rows):
        return None
    rows = np.concatenate(rows)
    return {k: (v[rows] if k in _PER_X else v) for k, v in tab.items()}


def build_model_table(cfg=CONFIG):
    """CR model on the (EEDF axis, Ne) grid -> dict of arrays (cached as npz)."""
    path = _table_path(cfg)
    eedfs, x, x_name, Te_eff = eedf_axis(cfg)
    if cfg["reuse_model"] and os.path.exists(path):
        tab = _cached_subset(dict(np.load(path, allow_pickle=False)), cfg, x)
        if tab is not None:
            print(f"reusing {path}")
            return tab
    cr, he, MD, interp = load_cr_model(cfg["outdir"])
    P, Tg, R = cfg["P_Torr"], cfg["Tg"], cfg["R"]
    MD = he.AddDiffusionLoss(MD, P, Tg, R)
    levels = [l for l, s in MD.items() if s["kind"] != "ground"]
    lines = _model_lines(MD)
    Ne = cfg["Ne_grid"]
    shp = (len(x), len(Ne))
    I_obs = np.zeros(shp + (len(lines),))
    I_thin = np.zeros_like(I_obs)
    dens = np.zeros(shp + (len(levels),))
    fdir = np.zeros_like(dens)
    conv = np.zeros(shp, bool)

    def escape(up, lo, rad):          # same escape factor as SolveDirect
        if cfg["trap_lines"] == "ground" and lo["kind"] != "ground":
            return 1.0
        if lo["density_m^-3"] == 0:
            return 1.0
        a, tau = he.FindTauInModel(up, lo, rad, he.Volume2Torr(lo["density_m^-3"], Tg), Tg, R)
        return float(np.squeeze(cr["GetEta"](interp, tau, a)))

    rad_of = {(u, l): next(r for r in MD[u]["RadiativeDecay"] if r["direction"] == "loss" and r["partner"] == l)
              for u, l, _, _ in lines}
    for i, eedf in enumerate(eedfs):
        for j, ne in enumerate(Ne):
            with contextlib.redirect_stdout(io.StringIO()):
                D, _, _, solver = cr["CRModel"](MD, eedf, float(ne), P, Tg, R, interp,
                                                trap_lines=cfg["trap_lines"])
            conv[i, j] = solver["converged"]
            ng = D["ground"]["density_m^-3"]
            for k, lbl in enumerate(levels):
                s = D[lbl]
                dens[i, j, k] = s["density_m^-3"]
                direct = ne * ng * sum(r["Rate"] for r in s["Electron Impact CrossSections"]["Products"]
                                       if r["partner_label"] == "ground")
                prod = s.get("Production_m^-3s^-1", 0.0)
                fdir[i, j, k] = direct / prod if prod > 0 else np.nan
            for k, (u, l, wl, A) in enumerate(lines):
                I_thin[i, j, k] = D[u]["density_m^-3"] * A
                I_obs[i, j, k] = I_thin[i, j, k] * escape(D[u], D[l], rad_of[(u, l)])
        print(f"  CR grid: {x_name} = {x[i]:.3g} (Te_eff {Te_eff[i]:.2f} eV) done ({i + 1}/{len(x)}), "
              f"{(~conv[i]).sum()} not converged")
    tab = dict(x_grid=x, x_name=x_name, Te_eff=Te_eff, eedf=str(cfg["eedf"]), Ne_grid=Ne,
               P_Torr=P, Tg=Tg, R=R, trap_lines=cfg["trap_lines"],
               levels=np.array(levels), line_upper=np.array([l[0] for l in lines]),
               line_lower=np.array([l[1] for l in lines]), line_wl=np.array([l[2] for l in lines], float),
               line_A=np.array([l[3] for l in lines], float),
               I_obs=I_obs, I_thin=I_thin, dens=dens, fdirect=fdir, converged=conv)
    os.makedirs(cfg["outdir"], exist_ok=True)
    np.savez(path, **tab)
    return tab


# ----------------------------------------------------------------------------
# 1. Line selection -> features
# ----------------------------------------------------------------------------
def select_features(tab, cfg=CONFIG):
    """Selected lines grouped into features.  Returns (features, component lines)."""
    res = pd.read_csv(os.path.join(cfg["selection_dir"], cfg["lines_file"]))
    model_pairs = set(zip(tab["line_upper"], tab["line_lower"]))
    ar = res[(res.species == "Ar I").to_numpy() & np.array([(u, l) in model_pairs for u, l in zip(res.upper, res.lower)])]
    ar = als._attach_ratings(ar, os.path.join(cfg["selection_dir"], cfg["ar_ratings"]))
    in_range = np.zeros(len(ar), bool)
    for lo, hi in cfg["wl_ranges"]:
        in_range |= ((ar.wl_air >= lo) & (ar.wl_air <= hi)).to_numpy()
    ok = (ar.status.isin(cfg["status"]) & ar.contamination.isin(cfg["contamination"])
          & (ar.rating.isna() | (ar.rating >= cfg["min_rating"])) & in_range
          & ~ar.wl_air.round(3).isin(np.round(cfg["exclude_wl"], 3)))
    sel = ar[ok].sort_values("wl_air")
    # cluster selected lines, then add every other model line within merge_nm
    feats, cur = [], []
    for r in sel.itertuples():
        if cur and r.wl_air - cur[-1].wl_air > cfg["merge_nm"]:
            feats.append(cur); cur = []
        cur.append(r)
    if cur:
        feats.append(cur)
    out, comps = [], []
    for f in feats:
        lo, hi = f[0].wl_air - cfg["merge_nm"], f[-1].wl_air + cfg["merge_nm"]
        c = ar[(ar.wl_air >= lo) & (ar.wl_air <= hi)].sort_values("wl_air")
        name = f"{c.wl_air.mean():.2f} " + "+".join(f"{u}-{l}" for u, l in zip(c.upper, c.lower))
        out.append(dict(feature=name, wl=float(c.wl_air.mean()), n_comp=len(c),
                        uppers="+".join(c.upper), comps=list(zip(c.wl_air, c.upper, c.lower))))
        comps.append(c.assign(feature=name))
    return pd.DataFrame(out), pd.concat(comps, ignore_index=True)


# ----------------------------------------------------------------------------
# 2. Measurement
# ----------------------------------------------------------------------------
def measure(comps, cfg=CONFIG):
    mcfg = dict(als.CONFIG, sweeps=cfg["sweeps"], window_nm=cfg["window_nm"],
                reuse_measurements=cfg["reuse_measurements"], outdir=cfg["outdir"],
                measure_cache="ar_line_measurements.csv", export_flags=cfg["export_flags"])
    return als.filter_export_flags(als.measure_sweeps(comps[["species", "wl_air"]].drop_duplicates(), mcfg), mcfg)


def _response(cfg):
    if not cfg["response_file"]:
        return None
    d = np.loadtxt(cfg["response_file"])
    return lambda lam: np.interp(lam, d[:, 0], d[:, 1])


def feature_table(meas, feats, cfg=CONFIG):
    """Per spectrum and feature: summed area and its relative error."""
    m = meas[meas.species == "Ar I"].assign(wl=lambda d: d.wl_air.round(4))
    resp = _response(cfg)
    rows = []
    for (sweep, f), g in m.groupby(["sweep", "file"]):
        gi = g.drop_duplicates("wl").set_index("wl")
        for ft in feats.itertuples():
            c = gi.reindex([round(w, 4) for w, _, _ in ft.comps])
            if c.status.isna().any() or not c.status.isin(cfg["valid_status"]).all():
                continue
            area = c.area.sum()
            if not area > 0:
                continue
            err = np.sqrt(np.nansum((c.area / c.snr.where(c.snr > 0)) ** 2)) / area
            if not err <= cfg["max_rel_err"]:
                continue
            if resp is not None:
                area /= resp(ft.wl)
            rows.append(dict(sweep=sweep, file=f, x=g.x.iloc[0], rep=g.rep.iloc[0], feature=ft.feature,
                             wl=ft.wl, area=area, rel_err=err))
    return pd.DataFrame(rows)


def repeat_scatter(ft):
    """Per feature: pooled scatter of ln(area) between repeats of the same condition,
    after removing each spectrum's overall scale (plasma / exposure fluctuations
    that change the line ratios, invisible in the 1/SNR of a single spectrum)."""
    acc = {}
    for _, g in ft.groupby(["sweep", "x"]):
        p = g.pivot_table(index="file", columns="feature", values="area").dropna(axis=1)
        if len(p) < 2 or p.shape[1] < 2:
            continue
        p = np.log(p)
        p = p.sub(p.mean(axis=1), axis=0)
        for f in p:
            acc.setdefault(f, []).append((p[f].var(ddof=1) * (len(p) - 1), len(p) - 1))
    s = pd.Series({f: np.sqrt(sum(v for v, _ in l) / sum(d for _, d in l)) for f, l in acc.items()})
    return s.reindex(ft.feature.unique()).fillna(s.median() if len(s) else 0.0)


# ----------------------------------------------------------------------------
# 3. Fit
# ----------------------------------------------------------------------------
def fine_grid(tab, cfg=CONFIG):
    """Fine ln x and ln Ne axes of the posterior."""
    nX, nN = cfg["fine_n"]
    lx, ln = np.log(tab["x_grid"]), np.log(tab["Ne_grid"])
    return np.linspace(lx[0], lx[-1], nX), np.linspace(ln[0], ln[-1], nN)


def te_eff_fine(tab, grid):
    """Te_eff [eV] on the fine EEDF axis (Te itself for a Maxwellian)."""
    return np.interp(grid[0], np.log(tab["x_grid"]), tab["Te_eff"])


def on_fine_grid(tab, values, cfg=CONFIG, k=3):
    """Spline a (x, Ne) grid quantity onto the fine ln x - ln Ne grid."""
    lx, ln = np.log(tab["x_grid"]), np.log(tab["Ne_grid"])
    fX, fN = fine_grid(tab, cfg)
    k_x = min(k, len(lx) - 1)
    return RectBivariateSpline(lx, ln, values, kx=k_x, ky=k)(fX, fN)


def feature_model(tab, feats, cfg=CONFIG):
    """ln(modelled feature intensity) on the fine grid, shape (nX, nNe, nFeat)."""
    idx = {(u, l): k for k, (u, l) in enumerate(zip(tab["line_upper"], tab["line_lower"]))}
    photon_energy = (lambda k: 1.0 / tab["line_wl"][k]) if cfg["intensity_units"] == "energy" else (lambda k: 1.0)
    out = []
    for ft in feats.itertuples():
        I = sum(tab["I_obs"][:, :, idx[(u, l)]] * photon_energy(idx[(u, l)]) for _, u, l in ft.comps)
        out.append(on_fine_grid(tab, np.log(I), cfg))
    return np.stack(out, axis=-1)


def _wquantile(x, p, q):
    c = np.cumsum(p)
    return np.interp(q, c / c[-1], x)


def hpd_level(post, mass=0.68):
    p = np.sort(post.ravel())[::-1]
    c = np.cumsum(p) / p.sum()
    return p[np.searchsorted(c, mass)]


def fit_block(d, M, feats, grid, Te_f, cfg=CONFIG):
    """Joint fit of the rows in d (one or more spectra).  Returns summary dict,
    posterior (nX, nNe) and per-row best-fit residuals."""
    fX, fN = grid
    fidx = {f: k for k, f in enumerate(feats.feature)}
    li = d.feature.map(fidx).to_numpy()
    si = pd.factorize(d.file)[0]
    y = np.log(d.area.to_numpy())
    lam = d.wl.to_numpy()
    n, n_spec = len(d), si.max() + 1
    X = [np.eye(n_spec)[si]]
    if cfg["response_deg"]:
        X += [((lam - 775.0) / 100.0)[:, None] ** k for k in range(1, cfg["response_deg"] + 1)]
    X = np.hstack(X)
    C = np.diag(d.rel_err.to_numpy() ** 2 + cfg["sigma_fit_floor"] ** 2)
    C += cfg["sigma_model"] ** 2 * (li[:, None] == li[None, :])
    W = np.linalg.inv(C)
    G = np.linalg.pinv(X.T @ W @ X) @ X.T @ W          # beta = G @ resid
    Pm = W - W @ X @ G
    D = y[None, None, :] - M[:, :, li]
    chi2 = np.einsum("abi,ij,abj->ab", D, Pm, D)
    dof = n - np.linalg.matrix_rank(X) - 2
    ib = np.unravel_index(np.argmin(chi2), chi2.shape)
    chi2_min = float(chi2[ib])
    s2 = max(1.0, chi2_min / dof) if (cfg["birge"] and dof > 0) else 1.0
    post = np.exp(-0.5 * (chi2 - chi2_min) / s2)
    post /= post.sum()
    pX, pN = post.sum(1), post.sum(0)
    qX = _wquantile(fX, pX, [0.16, 0.5, 0.84])
    qN = np.exp(_wquantile(fN, pN, [0.16, 0.5, 0.84]))
    qT = np.interp(qX, fX, Te_f)                       # Te_eff is monotonic in x
    mX, mN = (pX * fX).sum(), (pN * fN).sum()
    cov = (post * (fX[:, None] - mX) * (fN[None, :] - mN)).sum()
    corr = cov / np.sqrt((pX * (fX - mX) ** 2).sum() * (pN * (fN - mN) ** 2).sum())
    beta = G @ D[ib]
    resid = D[ib] - X @ beta
    edge = lambda p: p[:3].sum() + p[-3:].sum()
    out = dict(n_spec=n_spec, n_points=n, n_feat=len(np.unique(li)),
               Te_best=float(Te_f[ib[0]]), Te_lo=qT[0], Te_med=qT[1], Te_hi=qT[2],
               x_best=float(np.exp(fX[ib[0]])), x_lo=np.exp(qX[0]), x_med=np.exp(qX[1]), x_hi=np.exp(qX[2]),
               Ne_best=float(np.exp(fN[ib[1]])), Ne_lo=qN[0], Ne_med=qN[1], Ne_hi=qN[2],
               corr_lnx_lnNe=corr, chi2_min=chi2_min, dof=dof,
               chi2_red=chi2_min / dof if dof > 0 else np.nan, birge=np.sqrt(s2),
               edge_x=edge(pX) > 0.05, edge_Ne=edge(pN) > 0.05)
    if cfg["response_deg"]:
        out["response_slope_per_100nm"] = float(beta[n_spec])
    return out, post, resid


def fit_all(ft, feats, M, grid, Te_f, cfg=CONFIG, verbose=True):
    cond_rows, spec_rows, res_rows, posts = [], [], [], {}
    for (sweep, x), d in ft.groupby(["sweep", "x"]):
        s, post, r = fit_block(d, M, feats, grid, Te_f, cfg)
        cond_rows.append(dict(sweep=sweep, x=x, **s))
        posts[f"{sweep}|{x:g}"] = post
        for f, g in d.assign(r=r).groupby("feature"):
            res_rows.append(dict(sweep=sweep, x=x, feature=f, resid=g.r.mean()))
        if verbose:
            print(f"  {sweep:<12s} x={x:<6g} Te = {s['Te_med']:.2f} (+{s['Te_hi'] - s['Te_med']:.2f} "
                  f"-{s['Te_med'] - s['Te_lo']:.2f}) eV   Ne = {s['Ne_med']:.2e} m^-3   "
                  f"chi2/dof = {s['chi2_red']:.2f}  ({s['n_spec']} spectra, {s['n_feat']} lines)")
    for (sweep, f), d in ft.groupby(["sweep", "file"]):
        s, _, _ = fit_block(d, M, feats, grid, Te_f, cfg)
        spec_rows.append(dict(sweep=sweep, file=f, x=d.x.iloc[0], rep=d.rep.iloc[0], **s))
    return pd.DataFrame(cond_rows), pd.DataFrame(spec_rows), pd.DataFrame(res_rows), posts


def posterior_mean(tab, post, values, cfg=CONFIG):
    """<values> over the posterior; values is a (x, Ne) grid quantity of the model table."""
    return float((post * on_fine_grid(tab, values, cfg, k=1)).sum())


# ----------------------------------------------------------------------------
# 4. Plots
# ----------------------------------------------------------------------------
def plot_sweeps(cond, spec, tab, cfg, path):
    sweeps = cfg["sweeps"]
    fig, axs = plt.subplots(2, len(sweeps), figsize=(6.5 * len(sweeps), 8), squeeze=False)
    for j, sw in enumerate(sweeps):
        c = cond[cond.sweep == sw["name"]].sort_values("x")
        s = spec[spec.sweep == sw["name"]]
        for i, (q, lab) in enumerate((("Te", te_label(tab)), ("Ne", "$N_e$ [m$^{-3}$]"))):
            ax = axs[i, j]
            ax.plot(s.x, s[f"{q}_med"], "o", color="0.7", ms=4, label="single spectrum")
            ax.errorbar(c.x, c[f"{q}_med"], yerr=[c[f"{q}_med"] - c[f"{q}_lo"], c[f"{q}_hi"] - c[f"{q}_med"]],
                        fmt="s-", color="C0" if q == "Te" else "C3", ms=6, capsize=3,
                        label="condition (all repeats), 16-84 %")
            if q == "Ne":
                ax.set_yscale("log")
            ax.set_xlabel(sw["xlabel"])
            ax.set_ylabel(lab, fontsize=9)
            ax.grid(alpha=0.3, which="both")
            ax.set_title(f"{sw['name']} sweep ({sw['note']})")
        axs[0, j].legend(fontsize=8)
    resp = "calibrated response" if not cfg["response_deg"] else f"ln R poly deg {cfg['response_deg']}"
    eedf = "Maxwellian" if cfg["eedf"] == "maxwell" else f"MultiBolt {os.path.basename(os.path.normpath(cfg['eedf']))}"
    fig.suptitle(f"CR-model fit of Ar I lines ({eedf}, {resp}, "
                 rf"$\sigma_{{model}}$ = {100 * cfg['sigma_model']:.0f} %)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_posteriors(cond, posts, tab, grid, cfg, path):
    Te_f, Ne_f = te_eff_fine(tab, grid), np.exp(grid[1])
    sweeps = cfg["sweeps"]
    fig, axs = plt.subplots(1, len(sweeps), figsize=(6.5 * len(sweeps), 5.5), squeeze=False)
    for ax, sw in zip(axs[0], sweeps):
        c = cond[cond.sweep == sw["name"]].sort_values("x")
        cols = plt.cm.viridis(np.linspace(0, 0.9, max(len(c), 1)))
        for col, r in zip(cols, c.itertuples()):
            post = posts[f"{sw['name']}|{r.x:g}"]
            ax.contour(Te_f, Ne_f, post.T, levels=[hpd_level(post)], colors=[col])
            ax.plot(r.Te_best, r.Ne_best, "o", color=col, ms=5, label=f"{r.x:g}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(te_label(tab), fontsize=9); ax.set_ylabel("$N_e$ [m$^{-3}$]")
        ax.set_title(f"{sw['name']}: 68 % posterior regions")
        ax.legend(title=sw["xlabel"], fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(res, feats, path):
    if res.empty:
        return
    res = res.assign(cond=res.sweep + " " + res.x.map("{:g}".format))
    M = res.pivot(index="feature", columns="cond", values="resid").reindex(feats.feature)
    fig, ax = plt.subplots(figsize=(0.45 * M.shape[1] + 4, 0.4 * M.shape[0] + 2.5))
    v = np.nanmax(np.abs(M.values))
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M.values[i, j]):
                ax.text(j, i, f"{M.values[i, j]:+.2f}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(M.shape[0]), M.index, fontsize=8)
    fig.colorbar(im, ax=ax, label="ln(measured / model) at best fit")
    ax.set_title("Line residuals (after the per-spectrum scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. Driver
# ----------------------------------------------------------------------------
def run(cfg=CONFIG):
    os.makedirs(cfg["outdir"], exist_ok=True)
    out = lambda f: os.path.join(cfg["outdir"], f)
    tab = build_model_table(cfg)
    if not tab["converged"].all():
        print(f"WARNING: {(~tab['converged']).sum()} CR grid points did not converge")
    feats, comps = select_features(tab, cfg)
    print(f"\n{len(feats)} features from {len(comps)} Ar I lines:")
    print(feats[["feature", "wl", "n_comp"]].to_string(index=False))
    meas = measure(comps, cfg)
    ft = feature_table(meas, feats, cfg)
    rs = repeat_scatter(ft)
    feats["repeat_scatter"] = feats.feature.map(rs)
    ft["rel_err_fit"] = ft.rel_err
    if cfg["use_repeat_scatter"]:
        ft["rel_err"] = np.sqrt(ft.rel_err_fit ** 2 + ft.feature.map(rs) ** 2)
    print("\nrepeat scatter per line (ln):")
    print(rs.round(3).to_string())
    ft.to_csv(out("feature_intensities.csv"), index=False)
    feats.drop(columns="comps").to_csv(out("features.csv"), index=False)
    grid = fine_grid(tab, cfg)
    Te_f = te_eff_fine(tab, grid)
    M = feature_model(tab, feats, cfg)
    print(f"\nfitting (EEDF axis {tab['x_name']}) ...")
    cond, spec, res, posts = fit_all(ft, feats, M, grid, Te_f, cfg)
    # CR-model context at the fit: metastable densities (1s5 = 4s1, 1s3 = 4s3)
    lev = list(tab["levels"])
    for lbl in ("4s1", "4s3"):
        cond[f"n_{lbl}"] = [np.exp(posterior_mean(tab, posts[f"{s}|{x:g}"], np.log(tab["dens"][:, :, lev.index(lbl)]), cfg))
                            for s, x in zip(cond.sweep, cond.x)]
    cond.to_csv(out("fit_conditions.csv"), index=False)
    spec.to_csv(out("fit_spectra.csv"), index=False)
    res.to_csv(out("fit_residuals.csv"), index=False)
    np.savez(out("posteriors.npz"), lnx=grid[0], lnNe=grid[1], Te_eff=Te_f, **posts)
    plot_sweeps(cond, spec, tab, cfg, out("Te_Ne_vs_sweep.png"))
    plot_posteriors(cond, posts, tab, grid, cfg, out("posteriors.png"))
    plot_residuals(res, feats, out("residuals.png"))
    for c in cond.itertuples():
        if c.edge_x or c.edge_Ne:
            print(f"WARNING: {c.sweep} x={c.x:g}: posterior reaches the grid edge "
                  f"({'EEDF axis' if c.edge_x else ''}{' Ne' if c.edge_Ne else ''})")
    print("\nwritten to", cfg["outdir"])
    eedfs = eedf_axis(cfg)[0]
    return dict(tab=tab, feats=feats, comps=comps, meas=meas, ft=ft, grid=grid, Te_f=Te_f, M=M,
                cond=cond, spec=spec, res=res, posts=posts, eedfs=eedfs, cfg=cfg)


if __name__ == "__main__":
    FIT = run(CONFIG)
