# -*- coding: utf-8 -*-
"""
MicrowaveLowENFit.py

The CR fit with the microwave (2.45 GHz) BOLSIG+ EEDFs extended to lower E/N, and the
electron density walked from the free-fit value down to the critical density n_c.

The microwave libraries of ActinometryNitrogenContent start at 6 Td at BOLSIG+ precision
1e-25.  In Ar/N2 their rows below ~10 Td end below the 4s threshold (CRFitNeTe drops them,
min_bolsig_Emax), and the free microwave fits sit on that low-E/N edge (Te_eff ~0.5 eV)
with Ne on the 3e19 grid edge.  Per N2 fraction of the data (0-3 %):

1. BOLSIG+ microwave E/N sweep from EN_MW[0] Td, at precision 1e-30 up to 30 Td: BOLSIG+
   then follows the tail 1-4 eV further, so rows ~2 Td lower reach the 4s threshold (1e-35
   hangs; with >= 1.5 % N2 1e-30 also hangs at 40 Td, where 1e-25 already reaches > 20 eV).
   -> InputData/Bolsig/<Ar_Biagi|ArN2_<f>pct>_bolsig_mw_lowEN.  Rows whose tail still ends
   below 11.6 eV give no excitation at all and are dropped by the fit as before.
   Also the same field with e-e collisions at the ionization degree of n_c (n_c / N ~ 2e-6),
   where e-e matter: <...>_bolsig_mw_ee_nc, precision 1e-25 (e-e with 1e-30 hangs).  These
   EEDFs hold only at Ne = n_c, so their CR table spans n_c/4-4 n_c and only the fit at n_c
   is reported.
   And (Ar/N2 only) with superelastic collisions from vibrationally excited N2 at a vibrational
   temperature Tv (VIB_TV): <...>_bolsig_mw_vib<Tv>K.  With all N2 in v = 0 every electron
   crossing 2-4 eV loses energy to vibration, and in a microwave field (heating ~ nu_m, small
   at low energy) that empties the 2-4 eV range and the tail behind it; in an Ar-dominated
   1 Torr discharge Ar barely relaxes N2(v), so Tv of thousands of K is expected (no measured
   Tv yet - MolecularFitting/Joint_Fitting.py fits one from the N2 bands).  Fitted at n_c
   only, like the e-e EEDFs, to compare with them.  (Two-term vs multi-term is not the
   issue: MultiBolt 2 vs 8 terms in Ar + 5 % N2, DC, differ by <= 2 % in mean energy and
   <= 30 % in the tail at 3-7 Td, less above.)
2. CR table on the Ne grid of ActinometryNitrogenContent (1e13-3e19), N_WORKERS tables at a
   time in separate processes (~15 min each).  Also the old pure-Ar microwave and DC
   libraries on the same grid, for the comparison.
3. Every condition of that fraction (N2 sweep at x = f, the power sweep at 2.4 %):
     free       (E/N, Ne) free, as CRFitNeTe; also with the table cut at E/N >= 6 Td
     Ne path    chi^2 minimised over E/N at each Ne, from the free optimum down through n_c:
                Delta chi^2 / s^2, E/N, Te_eff, CR 1s5 density and the ionization frequency
                per electron nu_iz (ground state from the BOLSIG+ rates, excited levels from
                the CR model's cross sections and densities) against the electron loss
     pinned     Ne = n_c (CriticalDensityFit.analyse)
   with the same for the saved microwave (6 Td, 1e-25) and DC fits of ActinometryNitrogenContent.

Output in Experimental_Data/Output/MicrowaveLowEN/: summary.csv, ne_paths.csv, nc_rows.csv and
  pure_Ar_chi2_vs_Ne.png   pure Ar: chi^2 (E/N profiled) against Ne, microwave vs DC
  ne_path_families.png     the Ne path of a few conditions for every EEDF family
  nc_row_vs_EN.png         at Ne = n_c: chi^2/dof, Te_eff and CR 1s5 along E/N
  eepf_at_nc.png           the EEPF each family uses at n_c, against pure-Ar microwave
  summary.png, ne_path_<family>.png
Run from Spyder (F5) or python SmallAnalysisScripts/MicrowaveLowENFit.py.  Libraries and
tables are cached; delete a library folder or a table (.npz) to rebuild it.
"""
import contextlib
import io
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT_DIR, "Experimental_Data", "ExperimentalDataAnalysis"))
sys.path.insert(0, HERE)
import ActinometryNitrogenContent as anc            # noqa: E402
import CriticalDensityFit as cdf                    # noqa: E402

crf, he = anc.crf, anc.he

# ---- settings -------------------------------------------------------------------
FRACTIONS = [0, 0.5, 1, 1.5, 2, 2.4, 2.5, 3]        # % N2 of the conditions (power sweep at 2.4 %)
EN_MW = [3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10, 11, 12, 13.5, 15, 17.5, 20, 25, 30, 40, 50, 60,
         80, 100, 130, 170, 210, 250, 300, 350, 400, 500, 600, 700, 850, 1000]     # Td
OLD_FLOOR_TD = 6.0                                  # lower E/N limit of the old microwave libraries
FREQ_HZ = anc.CONFIG["eedfs"]["microwave"]["freq_Hz"]   # 2.45 GHz, as the old libraries
BOLSIG_SETTINGS = anc.CONFIG["bolsig_settings"]
PRECISION = [1e-30 if en <= 30 else 1e-25 for en in EN_MW]   # per E/N row, see 1. above
SUFFIX = "_mw_lowEN"
REF_PURE_AR = {"microwave": "Ar_Biagi_bolsig_mw", "dc": "Ar_Biagi_bolsig"}   # old pure-Ar libraries
NE_GRID = anc.CONFIG["Ne_grid"]
N_C = cdf.N_C
NE_GRID_NC = np.geomspace(N_C / 4, 4 * N_C, 9)      # CR tables of the EEDFs fitted at n_c only
VIB_TV = [3000, 5000, 8000]                         # K, N2 vibrational temperatures (see 1.)
VIB_XSEC = he.BOLSIG_XSEC_FOLDER / "Biagi_ArN2_vibSE.txt"
A_XSEC = he.BOLSIG_XSEC_FOLDER / "Biagi_ArN2_vibSE_ASE.txt"   # + N2(A) superelastics (MicrowaveN2AFit)
MEASURED_1S5 = cdf.MEASURED_1S5
# electron loss per electron at 1 Torr, order of magnitude: ambipolar diffusion to the
# R = 4 cm wall ~1e2-1e3 s^-1, Ar2+ conversion ~2e2 s^-1, Ar+ -> N2+ charge transfer and
# dissociative recombination up to ~1e4 s^-1 with N2
NU_LOSS = (1e2, 1e4)
N_WORKERS = 6
OUTDIR = os.path.join(ROOT_DIR, "Experimental_Data", "Output", "MicrowaveLowEN")
FAMILIES = {   # label, colour, line style
    "mw_lowEN": ("microwave, from 3 Td (new)", "C3", "-"),
    "mw_lowEN_6Td": ("microwave, new table cut at 6 Td", "C1", ":"),
    "mw_old": ("microwave, old library (6 Td, 1e-25)", "C4", "--"),
    "mw_ee_nc": ("microwave + e-e at n$_c$ (only Ne = n$_c$)", "C2", "-"),
    **{f"mw_vib{t}": (f"microwave + N$_2$(v) superelastics, T$_v$ = {t} K (only Ne = n$_c$)", c, "-")
       for t, c in zip(VIB_TV, ("C5", "C6", "C8"))},
    "dc": ("DC", "C0", "-."),
}
NC_ONLY = {"mw_ee_nc"} | {f"mw_vib{t}" for t in VIB_TV}    # EEDFs that hold at Ne = n_c only


# ---- libraries and CR tables ----------------------------------------------------
def vib_xsec():
    """Biagi Ar/N2 set with the N2 vibrational excitations v = 0 -> k written reversible
    ('N2 <-> N2(vk)', g ratio 1), so BOLSIG+ adds the superelastic N2(vk) -> N2 collisions
    weighted by the N2(vk) mole fraction (written to VIB_XSEC if missing).
    Returns {k: level energy [eV]}."""
    lines = (he.BOLSIG_XSEC_FOLDER / "Biagi_ArN2.txt").read_text(errors="replace").splitlines()
    out, levels, i = [], {}, 0
    while i < len(lines):
        out.append(lines[i])
        m = (re.fullmatch(r"\s*N2 -> N2\( VIB (\d*)V1\)\s*", lines[i + 1])
             if lines[i].strip() == "EXCITATION" and i + 2 < len(lines) else None)
        if m:
            k, E = int(m.group(1) or 1), float(lines[i + 2].split()[0])
            out += [f"N2 <-> N2(v{k})", f"{E:.6e}  1"]
            levels[k] = E
            i += 3
            continue
        i += 1
    if not VIB_XSEC.is_file():
        VIB_XSEC.write_text("\n".join(out) + "\n")
    return levels


def vib_composition(pct, Tv):
    """Cross sections, species and mole fractions of pct % N2 with N2(v) in a Boltzmann
    distribution at Tv.  Every N2 molecule is excited with the v = 0 cross sections (fraction
    x) and the N2(vk) fractions x p_k de-excite, so the electron-vibration exchange is in
    detailed balance at k Tv (no ladder climbing from v > 0, no anharmonic shifts)."""
    levels = vib_xsec()
    k = sorted(levels)
    p = np.exp(-np.array([levels[j] for j in k]) * constants.e / (constants.k * Tv))
    p /= 1 + p.sum()
    x = pct / 100
    return VIB_XSEC, ["Ar", "N2"] + [f"N2(v{j})" for j in k], [1 - x, x] + list(x * p)


def a_xsec():
    """VIB_XSEC with the N2(A3Sigma, v = 0-4) excitation also written reversible ('N2 <-> N2(A)',
    g ratio 3), so BOLSIG+ adds N2(A) -> N2 superelastics weighted by the N2(A) mole fraction
    (written to A_XSEC if missing)."""
    vib_xsec()
    if not A_XSEC.is_file():
        text = VIB_XSEC.read_text(errors="replace")
        old = re.search(r"N2 -> N2\( A3SIG V=0-4\)\n\s*(\S+)", text)
        if old is None:
            raise ValueError(f"{VIB_XSEC.name}: no N2(A3Sigma, v = 0-4) excitation")
        A_XSEC.write_text(text.replace(old.group(0), f"N2 <-> N2(A)\n{float(old.group(1)):.6e}  3"))
    return A_XSEC


def a_composition(pct, fA):
    """Cross sections, species and mole fractions of pct % N2 with N2(A) at n_A = fA n_N2 (all
    N2(v) populations 0: N2(A) superelastics only)."""
    levels = vib_xsec()
    x = pct / 100
    return (a_xsec(), ["Ar", "N2"] + [f"N2(v{j})" for j in sorted(levels)] + ["N2(A)"],
            [1 - x, x] + [0.0] * len(levels) + [fA * x])


def lib_folder(pct, variant="lowEN"):
    base = "Ar_Biagi_bolsig" if pct == 0 else f"ArN2_{pct:g}pct_bolsig"
    tail = {"lowEN": SUFFIX, "ee": "_mw_ee_nc"}.get(variant)
    if tail is None:
        tail = f"_mw_{variant}K" if variant.startswith("vib") else f"_mw_{variant}"
    return he.BOLSIG_FOLDER / (base + tail)


def library(pct, variant="lowEN"):
    """The microwave library of pct % N2: 'lowEN' (extended E/N range), 'ee' (+ e-e collisions at
    the ionization degree of n_c), 'vib<Tv>' (+ N2(v) superelastics) or 'A<fA>' (+ N2(A)
    superelastics at n_A = fA n_N2); BOLSIG+ run if missing."""
    folder = lib_folder(pct, variant)
    if not he.IsBolsigLibrary(folder):
        N = he.Torr2Volume(anc.CONFIG["fit"]["P_Torr"], anc.CONFIG["fit"]["Tg"])
        kw = dict(BOLSIG_SETTINGS, precision=PRECISION, omega_N=2 * np.pi * FREQ_HZ / N,
                  overwrite=True, verbose=False)
        if variant == "ee":
            kw.update(precision=1e-25, ionization_degree=N_C / N, plasma_density=N_C)
        if pct == 0:
            xsec, species, fractions = he.BOLSIG_XSEC_FOLDER / "Biagi_Ar.txt", ["Ar"], [1.0]
        elif variant.startswith("vib"):
            xsec, species, fractions = vib_composition(pct, float(variant[3:]))
        elif variant.startswith("A"):
            xsec, species, fractions = a_composition(pct, float(variant[1:]))
        else:
            xsec, species, fractions = he.BOLSIG_XSEC_FOLDER / "Biagi_ArN2.txt", ["Ar", "N2"], [1 - pct / 100, pct / 100]
        he.RunBolsig(xsec, folder.name, EN_MW, species=species, fractions=fractions, **kw)
    return folder


def fit_cfg(pct, lib, outdir, **extra):
    """CRFitNeTe settings of one table (as ActinometryNitrogenContent.mixture_fit)."""
    fit = anc.CONFIG["fit"]
    return {**fit, "eedf": str(lib), "N2_percent": pct, "Ne_grid": NE_GRID, "outdir": outdir,
            "measure_dir": fit.get("measure_dir") or fit["outdir"], **extra}


def job_cfg(key, build=True):
    """'<variant>:<pct>' (variant as library()) or 'ref:<library name>' (old pure-Ar library)
    -> fit settings; the variants fitted at n_c only get the narrow Ne grid."""
    kind, val = key.split(":")
    if kind == "ref":
        return fit_cfg(0.0, he.BOLSIG_FOLDER / val, os.path.join(OUTDIR, f"ref_{val}"))
    pct = float(val)
    lib = library(pct, kind) if build else lib_folder(pct, kind)
    if kind == "lowEN":
        return fit_cfg(pct, lib, os.path.join(OUTDIR, f"fit_{pct:g}pct_N2"))
    return fit_cfg(pct, lib, os.path.join(OUTDIR, f"fit_{kind}_nc_{pct:g}pct_N2"), Ne_grid=NE_GRID_NC)


def build_job(key):
    t0 = time.time()
    cfg = job_cfg(key)
    tab = crf.build_model_table(cfg)
    print(f"{key}: {len(tab['x_grid'])} E/N rows from {tab['x_grid'][0]:g} Td, "
          f"{(~tab['converged']).sum()} not converged ({time.time() - t0:.0f} s)", flush=True)


def table_current(cfg):
    """A cached CR table exists for cfg and was made with the current N2 physics (quench_4p)."""
    path = crf._table_path(cfg)
    if not os.path.exists(path):
        return False
    with np.load(path) as t:
        return (bool(t["quench_4p"]) if "quench_4p" in t.files else False) == crf.quench_4p_applied(cfg)


def build_tables(jobs=None):
    """Missing or outdated libraries and CR tables of jobs (default: this script's), N_WORKERS
    processes at a time (one table each)."""
    if jobs is None:
        jobs = ([f"lowEN:{p:g}" for p in FRACTIONS] + [f"ee:{p:g}" for p in FRACTIONS]
                + [f"vib{t}:{p:g}" for t in VIB_TV for p in FRACTIONS if p > 0]
                + [f"ref:{n}" for n in REF_PURE_AR.values()])
    todo = [k for k in jobs if not table_current(job_cfg(k, build=False))]
    if not todo:
        return
    vib_xsec()                     # written once here, before the workers read them
    a_xsec()
    logs = os.path.join(OUTDIR, "logs")
    os.makedirs(logs, exist_ok=True)
    print(f"building {len(todo)} CR tables ({N_WORKERS} at a time, logs in {logs}) ...")
    # one BLAS thread per worker: the default thread pools commit ~1 GB per process
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0, procs = time.time(), {}
    for k in todo:
        while sum(p.poll() is None for p in procs.values()) >= N_WORKERS:
            time.sleep(5)
        log = open(os.path.join(logs, k.replace(":", "_") + ".log"), "w")
        procs[k] = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--build", k],
                                    stdout=log, stderr=subprocess.STDOUT, cwd=ROOT_DIR, env=env)
    failed = [k for k, p in procs.items() if p.wait() != 0]
    print(f"tables done ({time.time() - t0:.0f} s)")
    if failed:
        raise RuntimeError(f"table build failed for {failed} - see {logs}")


# ---- fits -----------------------------------------------------------------------
def nu_iz(fit):
    """Ionization frequency per electron [s^-1] on the model grid: ground-state Ar and N2
    (BOLSIG+ rate coefficients of the library) + every excited level (CR-model cross
    sections x CR-model densities)."""
    cfg, tab = fit["cfg"], fit["tab"]
    with contextlib.redirect_stdout(io.StringIO()):
        eedfs = crf.eedf_axis(cfg)[0]
        cr, _, MD, _ = crf.load_cr_model(cfg["outdir"])
    N = he.Torr2Volume(cfg["P_Torr"], cfg["Tg"])
    f = cfg.get("N2_percent", 0.0) / 100
    lev = list(tab["levels"])
    nu = np.zeros(tab["dens"].shape[:2])
    for i, e in enumerate(eedfs):
        k0 = {}
        for r in e["rates"]:
            if r["process"] == "Ionization":
                k0[r["species"]] = k0.get(r["species"], 0.0) + r["rate_m3s"]
        with contextlib.redirect_stdout(io.StringIO()):
            cr["CreateIonizationReactionRates"](MD, e)
        k = np.array([MD[lbl]["Ionization Data"].get("Rate_cm^3", 0.0) for lbl in lev])   # m^3/s
        nu[i] = (1 - f) * N * k0.get("Ar", 0.0) + f * N * k0.get("N2", 0.0) + np.clip(tab["dens"][i], 0, None) @ k
    return nu


def add_fields(fit, with_nu=True):
    """ln(CR 1s5 density) and ln(nu_iz) (with_nu) on the fine grid of the fit."""
    tab, cfg = fit["tab"], fit["cfg"]
    lev = list(tab["levels"])
    ln = lambda v: crf.on_fine_grid(tab, np.log(np.clip(v, 1e-300, None)), cfg, k=1)
    fit["fields"] = {"n_1s5": ln(tab["dens"][:, :, lev.index("4s1")])}
    if with_nu:
        fit["fields"]["nu_iz"] = ln(nu_iz(fit))
    return fit


def prepare(cfg, with_nu=True):
    """CR table, features, measured feature table and fine-grid model (as CRFitNeTe.run)."""
    with contextlib.redirect_stdout(io.StringIO()):
        tab = crf.build_model_table(cfg)
        feats, comps = crf.select_features(tab, cfg)
        ft = crf.feature_table(crf.measure(comps, cfg), feats, cfg)
    rs = crf.repeat_scatter(ft)
    if cfg["use_repeat_scatter"]:
        ft["rel_err"] = np.sqrt(ft.rel_err ** 2 + ft.feature.map(rs) ** 2)
    grid = crf.fine_grid(tab, cfg)
    return add_fields(dict(cfg=cfg, tab=tab, feats=feats, ft=ft, grid=grid, Te_f=crf.te_eff_fine(tab, grid),
                           M=crf.feature_model(tab, feats, cfg)), with_nu)


def ne_path(fit, post, chi2_min, birge):
    """Along Ne: chi^2 minimised over E/N -> Delta chi^2 / s^2, E/N, Te_eff, 1s5, nu_iz there."""
    (fX, fN), Te_f = fit["grid"], fit["Te_f"]
    chi2 = cdf.chi2_grid(post, chi2_min, birge)
    i, j = np.argmin(chi2, axis=0), np.arange(len(fN))
    T2 = np.broadcast_to(Te_f[:, None], chi2.shape) if np.ndim(Te_f) == 1 else Te_f
    out = pd.DataFrame(dict(Ne=np.exp(fN), dchi2=(chi2[i, j] - chi2_min) / birge ** 2,
                            EN=np.exp(fX[i]), Te_eff=T2[i, j]))
    for name, v in fit["fields"].items():
        out[name] = np.exp(v[i, j])
    return out


def nc_row(fit, post, s):
    """Along E/N at Ne = n_c: chi^2/dof (Ne fixed: dof + 1), Te_eff and the CR 1s5 density."""
    (fX, fN), Te_f = fit["grid"], fit["Te_f"]
    j = int(np.argmin(np.abs(fN - np.log(N_C))))
    chi2 = cdf.chi2_grid(post, s["chi2_min"], s["birge"])[:, j]
    Te = Te_f if np.ndim(Te_f) == 1 else Te_f[:, j]
    return pd.DataFrame(dict(EN=np.exp(fX), Te_eff=Te, chi2_red=chi2 / (s["dof"] + 1),
                             n_1s5=np.exp(fit["fields"]["n_1s5"][:, j])))


def condition(fit, family, sweep, x, s, post, out):
    """Summary row, Ne path and n_c row of one condition (appended to out); s holds chi2_min,
    dof, birge and the free fit."""
    r, _, _ = cdf.analyse(fit, None, s["chi2_min"], s["dof"], s["birge"], post)
    path = ne_path(fit, post, s["chi2_min"], s["birge"])
    ib = np.unravel_index(np.argmax(post), post.shape)
    at_nc = path.iloc[int(np.argmin(np.abs(np.log(path.Ne / N_C))))]
    row = dict(family=family, sweep=sweep, x=x, N2_percent=anc.n2_percent(sweep, x),
               EN_free=s["x_best"] if "x_best" in s else np.exp(fit["grid"][0][ib[0]]),
               Te_free=s["Te_med"], Ne_free=s["Ne_med"], Ne_free_lo=s["Ne_lo"], Ne_free_hi=s["Ne_hi"],
               chi2_red_free=s["chi2_red"], edge_EN=bool(s["edge_x"]), edge_Ne=bool(s["edge_Ne"]),
               n_1s5_free=float(np.exp(fit["fields"]["n_1s5"][ib])),
               nu_iz_free=float(np.exp(fit["fields"]["nu_iz"][ib])),
               delta_chi2_at_nc=r["delta_chi2_at_nc"], chi2_red_nc=r["chi2_red_pinned"],
               EN_nc=r["x_pinned"], Te_nc=r["Te_pinned"], n_1s5_nc=at_nc.n_1s5, nu_iz_nc=at_nc.nu_iz,
               chi2_min=s["chi2_min"], dof=s["dof"], birge=s["birge"])
    if family in NC_ONLY:          # EEDFs valid at Ne = n_c only: no free fit, no profile
        row.update({k: np.nan for k in row if k.endswith("_free") or k.startswith("Ne_free")},
                   edge_EN=False, edge_Ne=False, delta_chi2_at_nc=np.nan)
        path = path.iloc[[int(np.argmin(np.abs(np.log(path.Ne / N_C))))]].assign(dchi2=np.nan)
    out["rows"].append(row)
    out["paths"].append(path.assign(family=family, sweep=sweep, x=x))
    out["nc"].append(nc_row(fit, post, s).assign(family=family, sweep=sweep, x=x))


def conditions_of(ft, pct):
    return [(sw, x) for (sw, x), _ in ft.groupby(["sweep", "x"]) if anc.n2_percent(sw, x) == pct]


def fit_new(pct, out):
    """The new libraries of pct % N2: full E/N range, cut at the old 6 Td floor, e-e at n_c and
    N2(v) superelastics."""
    cfg = job_cfg(f"lowEN:{pct:g}", build=False)
    variants = [("mw_lowEN", cfg), ("mw_lowEN_6Td", dict(cfg, EN_range=(OLD_FLOOR_TD, np.inf))),
                ("mw_ee_nc", job_cfg(f"ee:{pct:g}", build=False))]
    variants += [(f"mw_vib{t}", job_cfg(f"vib{t}:{pct:g}", build=False)) for t in VIB_TV if pct > 0]
    for family, c in variants:
        fit = prepare(c)
        for sw, x in conditions_of(fit["ft"], pct):
            d = fit["ft"][(fit["ft"].sweep == sw) & (fit["ft"].x == x)]
            s, post, _ = crf.fit_block(d, fit["M"], fit["feats"], fit["grid"], fit["Te_f"], c)
            condition(fit, family, sw, x, s, post, out)
        if family == "mw_lowEN":
            tab = fit["tab"]
            print(f"  {pct:g} % N2: {len(tab['x_grid'])} E/N rows kept, {tab['x_grid'][0]:g}-{tab['x_grid'][-1]:g} Td, "
                  f"Te_eff {np.min(tab['Te_eff']):.2f}-{np.max(tab['Te_eff']):.2f} eV")


def fit_old(pct, out):
    """The saved ActinometryNitrogenContent fits (pct > 0) or the old pure-Ar libraries refitted (pct = 0)."""
    for family, eedf in (("mw_old", "microwave"), ("dc", "dc")):
        if pct == 0:
            fit = prepare(job_cfg(f"ref:{REF_PURE_AR[eedf]}"))
            for sw, x in conditions_of(fit["ft"], 0.0):
                d = fit["ft"][(fit["ft"].sweep == sw) & (fit["ft"].x == x)]
                s, post, _ = crf.fit_block(d, fit["M"], fit["feats"], fit["grid"], fit["Te_f"], fit["cfg"])
                condition(fit, family, sw, x, s, post, out)
            continue
        fit = add_fields(cdf.load_fit(eedf, pct))
        for c in fit["cond"].itertuples():
            if anc.n2_percent(c.sweep, c.x) != pct:
                continue
            condition(fit, family, c.sweep, c.x, dict(c._asdict()), fit["posts"][f"{c.sweep}|{c.x:g}"], out)


# ---- plots ----------------------------------------------------------------------
PATH_ROWS = [("dchi2", r"$\Delta\chi^2/s^2$ (E/N profiled)", "linear"),
             ("Te_eff", r"$T_{e,\mathrm{eff}}$ at best E/N [eV]", "linear"),
             ("n_1s5", "CR 1s$_5$ density [m$^{-3}$]", "log"),
             ("nu_iz", r"ionization frequency $\nu_{iz}$ [s$^{-1}$]", "log")]


def _decorate(ax, key):
    ax.axvline(N_C, color="k", lw=1.2)
    if key == "dchi2":
        ax.axhline(1, color="0.5", ls=":", lw=0.8)
        ax.axhline(4, color="0.5", ls="--", lw=0.8)
        ax.set_ylim(0, 40)
    elif key == "n_1s5":
        ax.axhspan(*MEASURED_1S5, color="C2", alpha=0.15, lw=0)
        ax.set_ylim(1e6, 1e21)
    elif key == "nu_iz":
        ax.axhspan(*NU_LOSS, color="C2", alpha=0.15, lw=0)
        ax.set_ylim(1e-10, 1e8)
    ax.set_xscale("log")
    ax.set_xlim(1e14, 3e19)
    ax.grid(alpha=0.3, which="both")


def plot_paths(paths, family, sweeps, path):
    """The Ne path of every condition with one EEDF family, one column per sweep."""
    fig, axs = plt.subplots(len(PATH_ROWS), len(sweeps), figsize=(7 * len(sweeps), 3.3 * len(PATH_ROWS)),
                            sharex=True, squeeze=False)
    P = paths[paths.family == family]
    for j, sw in enumerate(sweeps):
        g = P[P.sweep == sw["name"]]
        xs = sorted(g.x.unique())
        cols = plt.cm.viridis(np.linspace(0, 0.9, max(len(xs), 1)))
        for col, x in zip(cols, xs):
            h = g[g.x == x]
            for i, (key, _, scale) in enumerate(PATH_ROWS):
                axs[i, j].plot(h.Ne, h[key], "*-" if len(h) == 1 else "-", color=col, lw=1.5, ms=10,
                               label=f"{x:g}")
        for i, (key, lab, scale) in enumerate(PATH_ROWS):
            axs[i, j].set_yscale(scale)
            _decorate(axs[i, j], key)
            axs[i, 0].set_ylabel(lab, fontsize=9)
        axs[0, j].set_title(f"{sw['name']} sweep ({sw['note']})")
        axs[0, j].legend(title=sw["xlabel"], fontsize=7, ncol=2)
        axs[-1, j].set_xlabel("$N_e$ [m$^{-3}$] (fixed; E/N refitted at each $N_e$)")
    fig.suptitle(f"{FAMILIES[family][0]}: the fit walked in $N_e$ (black: $n_c$; green: measured 1s$_5$ "
                 f"and the electron loss rate)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_families(paths, picks, path):
    """The Ne path of a few conditions, every EEDF family."""
    fig, axs = plt.subplots(len(PATH_ROWS), len(picks), figsize=(4.6 * len(picks), 3.3 * len(PATH_ROWS)),
                            sharex=True, squeeze=False)
    for j, (sw, x) in enumerate(picks):
        for fam, (lab, col, ls) in FAMILIES.items():
            h = paths[(paths.family == fam) & (paths.sweep == sw) & (paths.x == x)]
            for i, (key, _, scale) in enumerate(PATH_ROWS):
                axs[i, j].plot(h.Ne, h[key], "*" if len(h) == 1 else ls, color=col, lw=1.6, ms=12, label=lab)
        for i, (key, ylab, scale) in enumerate(PATH_ROWS):
            axs[i, j].set_yscale(scale)
            _decorate(axs[i, j], key)
            axs[i, 0].set_ylabel(ylab, fontsize=9)
        axs[0, j].set_title(f"{sw} = {x:g} ({anc.n2_percent(sw, x):g} % N$_2$)", fontsize=10)
        axs[-1, j].set_xlabel("$N_e$ [m$^{-3}$]")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def family_library(family, pct):
    """BOLSIG+ library folder behind a family's fit at pct % N2."""
    if family in ("mw_old", "dc"):
        eedf = "microwave" if family == "mw_old" else "dc"
        return anc.mixture_library(pct, eedf) if pct > 0 else he.BOLSIG_FOLDER / REF_PURE_AR[eedf]
    variant = "ee" if family == "mw_ee_nc" else family[3:] if family.startswith("mw_vib") else "lowEN"
    return lib_folder(pct, variant)


def plot_eepf_at_nc(res, picks, path):
    """The EEPF each family uses at Ne = n_c (library row nearest the pinned E/N), against the
    pure-Ar microwave fit at n_c and the Maxwellian of its mean energy."""
    fams = [f for f in FAMILIES if f not in ("mw_lowEN_6Td", "mw_old")]
    ar = res[(res.family == "mw_lowEN") & (res.N2_percent == 0)].iloc[0]
    lib_ar = he.ImportBolsigLibrary(family_library("mw_lowEN", 0.0), verbose=False)
    e_ar = lib_ar["EEDFs"][int(np.argmin(np.abs(np.log(lib_ar["EN_Td"] / ar.EN_nc))))]
    fig, axs = plt.subplots(1, len(picks), figsize=(6.5 * len(picks), 5.2), sharey=True, squeeze=False)
    for ax, (sw, x) in zip(axs[0], picks):
        ax.axvspan(1.5, 4.0, color="0.9", lw=0, label="N$_2$ vibrational losses")
        for E0, lab in ((11.55, "4s"), (13.1, "4p"), (14.5, "5p")):
            ax.axvline(E0, color="0.6", lw=0.8, ls=":")
            ax.text(E0, 2e-1, lab, fontsize=7, ha="center", color="0.4")
        ax.semilogy(e_ar["E"], e_ar["EEPF"], color="k", lw=2.4,
                    label=f"pure Ar, microwave at n$_c$: {ar.EN_nc:.3g} Td, T$_{{e,eff}}$ {e_ar['Te_eff']:.2f} eV")
        M = he.MaxwellianEEDF(e_ar["Te_eff"], e_ar["E"])
        ax.semilogy(M["E"], M["EEPF"], color="k", lw=1, ls=":", label="Maxwellian, same mean energy")
        pct = anc.n2_percent(sw, x)
        for fam in fams:
            r = res[(res.family == fam) & (res.sweep == sw) & (res.x == x)]
            if r.empty or not np.isfinite(r.EN_nc.iloc[0]):
                continue
            lib = he.ImportBolsigLibrary(family_library(fam, pct), verbose=False)
            e = lib["EEDFs"][int(np.argmin(np.abs(np.log(lib["EN_Td"] / r.EN_nc.iloc[0]))))]
            lab, col, ls = FAMILIES[fam]
            ax.semilogy(e["E"], e["EEPF"], ls, color=col, lw=1.5,
                        label=f"{lab.split(' (')[0]}: {r.EN_nc.iloc[0]:.3g} Td, T$_{{e,eff}}$ {e['Te_eff']:.2f} eV, "
                              f"$\\chi^2$/dof {r.chi2_red_nc.iloc[0]:.2f}")
        ax.set_xlim(0, 25)
        ax.set_ylim(1e-14, 3)
        ax.set_xlabel("electron energy [eV]")
        ax.set_title(f"{sw} = {x:g} ({pct:g} % N$_2$): EEPF of the fit at $N_e = n_c$", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6.5, loc="lower left")
    axs[0, 0].set_ylabel("EEPF [eV$^{-3/2}$]")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_summary(res, sweeps, path):
    """Free fit and Ne = n_c per condition, every EEDF family."""
    rows = [("Te_free", "Te_nc", r"$T_{e,\mathrm{eff}}$ [eV] (open: free, filled: $N_e = n_c$)", "linear"),
            ("Ne_free", None, "free-fit $N_e$ [m$^{-3}$]", "log"),
            ("delta_chi2_at_nc", None, r"$\Delta\chi^2/s^2$ for $N_e = n_c$", "linear"),
            ("chi2_red_free", "chi2_red_nc", r"$\chi^2$/dof (open: free, filled: $n_c$)", "linear"),
            ("n_1s5_free", "n_1s5_nc", "CR 1s$_5$ [m$^{-3}$] (open: free, filled: $n_c$)", "log")]
    fig, axs = plt.subplots(len(rows), len(sweeps), figsize=(7 * len(sweeps), 3.2 * len(rows)), squeeze=False)
    for j, sw in enumerate(sweeps):
        r = res[res.sweep == sw["name"]]
        for fam, (lab, col, ls) in FAMILIES.items():
            g = r[r.family == fam].sort_values("x")
            for i, (free, nc, _, _) in enumerate(rows):
                axs[i, j].plot(g.x, g[free], "o" + ls, color=col, mfc="white", ms=5, lw=1, label=lab)
                if nc:
                    axs[i, j].plot(g.x, g[nc], "o" + ls, color=col, ms=5, lw=1)
        for i, (free, nc, ylab, scale) in enumerate(rows):
            ax = axs[i, j]
            ax.set_yscale(scale)
            ax.grid(alpha=0.3, which="both")
            ax.set_ylabel(ylab, fontsize=9)
            ax.set_xlabel(sw["xlabel"])
            if free == "Ne_free":
                ax.axhline(N_C, color="k", lw=1.2)
            if free == "n_1s5_free":
                ax.axhspan(*MEASURED_1S5, color="C2", alpha=0.15, lw=0)
            if free == "delta_chi2_at_nc":
                ax.axhline(4, color="0.5", ls="--", lw=0.8)
        axs[0, j].set_title(f"{sw['name']} sweep ({sw['note']})")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_nc_rows(ncs, picks, path):
    """At Ne = n_c: chi^2/dof, Te_eff and the CR 1s5 density along E/N; thick where the CR 1s5
    lies in the measured range."""
    fams = [f for f in FAMILIES if f not in ("mw_lowEN_6Td", "mw_old")]
    fig, axs = plt.subplots(3, len(picks), figsize=(5.6 * len(picks), 11), sharex=True, squeeze=False)
    for j, (sw, x) in enumerate(picks):
        for fam in fams:
            h = ncs[(ncs.family == fam) & (ncs.sweep == sw) & (ncs.x == x)].sort_values("EN")
            if h.empty:
                continue
            lab, col, ls = FAMILIES[fam]
            ok = ((h.n_1s5 >= MEASURED_1S5[0]) & (h.n_1s5 <= MEASURED_1S5[1])).to_numpy()
            for i, key in enumerate(("chi2_red", "Te_eff", "n_1s5")):
                axs[i, j].plot(h.EN, h[key], ls, color=col, lw=1.2, label=lab.split(" (")[0])
                axs[i, j].plot(h.EN.where(ok), h[key].where(ok), "-", color=col, lw=4, alpha=0.6)
        axs[0, j].axhline(1, color="0.5", lw=0.8, ls=":")
        axs[0, j].set_ylim(0, 8)
        axs[1, j].axhline(0.95, color="0.5", lw=0.8, ls=":")
        axs[2, j].axhspan(*MEASURED_1S5, color="C2", alpha=0.15, lw=0)
        axs[2, j].set_yscale("log")
        axs[2, j].set_ylim(1e12, 1e21)
        for ax in axs[:, j]:
            ax.set_xscale("log")
            ax.grid(alpha=0.3, which="both")
        axs[0, j].set_title(f"{sw} = {x:g} ({anc.n2_percent(sw, x):g} % N$_2$), $N_e = n_c$", fontsize=10)
        axs[-1, j].set_xlabel("E/N [Td]")
    axs[0, 0].set_ylabel(r"$\chi^2$/dof at $N_e = n_c$")
    axs[1, 0].set_ylabel(r"$T_{e,\mathrm{eff}}$ [eV] (dotted: 0.95 eV)")
    axs[2, 0].set_ylabel("CR 1s$_5$ [m$^{-3}$] (thick: in the measured range)")
    axs[0, -1].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_pure_ar_profile(res, paths, path):
    """Pure Ar: chi^2 profiled over E/N against Ne for the microwave EEDFs (and DC for reference)."""
    fams = [("mw_lowEN", 2.4), ("mw_old", 1.3), ("dc", 1.3)]
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for fam, lw in fams:
        r = res[(res.family == fam) & (res.N2_percent == 0)].iloc[0]
        h = paths[(paths.family == fam) & (paths.sweep == r.sweep) & (paths.x == r.x)]
        lab, col, ls = FAMILIES[fam]
        chi2 = r.chi2_min + h.dchi2 * r.birge ** 2
        lab = f"{lab} (free: $N_e$ {r.Ne_free:.2g}, $\\chi^2$/dof {r.chi2_red_free:.2f})"
        axs[0, 0].plot(h.Ne, chi2 / (r.dof + 1), ls, color=col, lw=lw, label=lab)
        axs[0, 1].plot(h.Ne, h.dchi2, ls, color=col, lw=lw)
        axs[1, 0].plot(h.Ne, h.Te_eff, ls, color=col, lw=lw)
        axs[1, 1].plot(h.Ne, h.n_1s5, ls, color=col, lw=lw)
    ee = res[(res.family == "mw_ee_nc") & (res.N2_percent == 0)].iloc[0]
    axs[0, 0].plot(N_C, ee.chi2_red_nc, "*", color=FAMILIES["mw_ee_nc"][1], ms=13,
                   label=f"microwave + e-e at n$_c$ ($\\chi^2$/dof {ee.chi2_red_nc:.2f})")
    axs[1, 0].plot(N_C, ee.Te_nc, "*", color=FAMILIES["mw_ee_nc"][1], ms=13)
    axs[1, 1].plot(N_C, ee.n_1s5_nc, "*", color=FAMILIES["mw_ee_nc"][1], ms=13)
    axs[0, 0].axhline(1, color="0.5", ls=":", lw=0.8)
    axs[0, 0].set_ylim(0, 2.5)
    axs[0, 1].axhline(1, color="0.5", ls=":", lw=0.8)
    axs[0, 1].axhline(4, color="0.5", ls="--", lw=0.8)
    axs[0, 1].set_ylim(0, 30)
    axs[1, 1].axhspan(*MEASURED_1S5, color="C2", alpha=0.15, lw=0, label="measured (absorption)")
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_ylim(1e10, 1e19)
    for ax in axs.ravel():
        ax.axvline(N_C, color="k", lw=1.2)
        ax.set_xscale("log")
        ax.set_xlim(1e13, 3e19)
        ax.grid(alpha=0.3, which="both")
    axs[0, 0].set_ylabel(r"$\chi^2$/dof, E/N profiled ($N_e$ fixed)")
    axs[0, 1].set_ylabel(r"$\Delta\chi^2/s^2$ against the best fit (1 and 4)")
    axs[1, 0].set_ylabel(r"$T_{e,\mathrm{eff}}$ at the best E/N [eV]")
    axs[1, 1].set_ylabel("CR 1s$_5$ density [m$^{-3}$]")
    for ax in axs[1]:
        ax.set_xlabel("$N_e$ [m$^{-3}$] (black: $n_c$)")
    axs[0, 0].legend(fontsize=7.5, loc="upper left")
    axs[1, 1].legend(fontsize=7.5)
    fig.suptitle("Pure Ar (85 W, 1 Torr): CR-fit $\\chi^2$ against $N_e$", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    build_tables()
    out = dict(rows=[], paths=[], nc=[])
    print("fitting ...")
    for pct in FRACTIONS:
        fit_new(pct, out)
        fit_old(pct, out)
    res = pd.DataFrame(out["rows"])
    paths, ncs = (pd.concat(out[k], ignore_index=True) for k in ("paths", "nc"))
    res.to_csv(os.path.join(OUTDIR, "summary.csv"), index=False)
    paths.to_csv(os.path.join(OUTDIR, "ne_paths.csv"), index=False)
    ncs.to_csv(os.path.join(OUTDIR, "nc_rows.csv"), index=False)
    plot_pure_ar_profile(res, paths, os.path.join(OUTDIR, "pure_Ar_chi2_vs_Ne.png"))
    plot_nc_rows(ncs, [("N2 fraction", 0.0), ("N2 fraction", 1.0), ("Power", 85.0)],
                 os.path.join(OUTDIR, "nc_row_vs_EN.png"))
    sweeps = crf.CONFIG["sweeps"]
    for fam in FAMILIES:
        plot_paths(paths, fam, sweeps, os.path.join(OUTDIR, f"ne_path_{fam}.png"))
    picks = [("N2 fraction", 0.0), ("N2 fraction", 1.0), ("N2 fraction", 3.0), ("Power", 85.0)]
    plot_families(paths, picks, os.path.join(OUTDIR, "ne_path_families.png"))
    plot_summary(res, sweeps, os.path.join(OUTDIR, "summary.png"))
    plot_eepf_at_nc(res, [("N2 fraction", 1.0), ("Power", 85.0)], os.path.join(OUTDIR, "eepf_at_nc.png"))
    cols = ["family", "sweep", "x", "EN_free", "Te_free", "Ne_free", "chi2_red_free", "edge_EN", "edge_Ne",
            "n_1s5_free", "nu_iz_free", "delta_chi2_at_nc", "chi2_red_nc", "EN_nc", "Te_nc", "n_1s5_nc", "nu_iz_nc"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(res.sort_values(["sweep", "x", "family"])[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nfigures in {OUTDIR}")
    return res, paths, ncs


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--build":
        build_job(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "--build-all":
        build_tables()
    else:
        RES, PATHS, NC_ROWS = main()
