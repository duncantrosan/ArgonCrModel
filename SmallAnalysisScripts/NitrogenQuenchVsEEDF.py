# -*- coding: utf-8 -*-
"""
NitrogenQuenchVsEEDF.py

Why the CR fit fails for the Ar/N2 spectra at n_c (CR 1s5 50-66x the LAS value, LASTgComparison): too much
quenching of the Ar metastables by N2, or the microwave EEDF of the Ar/N2 mixture?  Everything at the LAS gas
temperature of LASTgComparison and Ne = n_c (2.42 GHz).

1. LAS only: 1/n(1s5) against % N2 per run (LAS_P_W).  With the production P and the other losses nu_0 independent
   of N2, 1/n = (nu_0 + k_Q x N) / P is linear in x, and intercept / slope gives nu_0 = k_Q N a / b, the 1s5 loss
   frequency without N2 (k_Q of 1s5 + N2 from he.ImportArQuenchingData, N from the run's LAS Tg).  P also falls
   with N2 (dilution, hotter gas), so this nu_0 is rather a lower limit.  Compared with the CR model's 1s5 loss
   frequency at n_c for pure Ar (LASTgComparison fits of the pressure sweep): electron collisions dominate it and
   scale with Ne.
2. Measured line pattern against % N2 (no model): per spectrum ln(feature) minus its mean over the features,
   mean over the repeats, relative to 0 % N2; bars = standard error of the mean of the repeats.
3. CR fits at n_c of every Ar/N2 condition (N2 sweep 0.5-3 % at 85 W, power sweep at 2.4 %) with
     A  Ar/N2 microwave EEDF + N2 quenching          (LASTgComparison, the baseline)
     B  Ar/N2 microwave EEDF, no N2 quenching        (CRFitNeTe N2_quench_scale = 0, Ar dilution kept)
     C  pure-Ar microwave EEDF + N2 quenching        (same p and Tg: N2 left out of the Boltzmann equation only)
     D  pure-Ar microwave EEDF, no N2 quenching
     E  Ar/N2 DC EEDF + N2 quenching                 (BOLSIG+ DC field at the LAS Tg)
   -> chi^2/dof, E/N, Te_eff and CR 1s5 against the LAS 1s5.

Output in Experimental_Data/Output/NitrogenQuenchVsEEDF/ (one graph per file):
  las_quench_fits.csv, las_quench_inverse_n.png, las_quench_nu0.png   1.
  line_pattern_vs_N2.png                                               2.
  variants.csv, 1s5_<sweep>.png, chi2_<sweep>.png, EN_<sweep>.png      3.
  nc_profiles.csv, chi2_profile_<cond>.png, 1s5_profile_<cond>.png    chi^2/dof and CR 1s5 along E/N at n_c
Run from Spyder (F5) or python SmallAnalysisScripts/NitrogenQuenchVsEEDF.py.  Libraries and CR tables are cached.
"""
import contextlib
import io
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import LASTgComparison as L                         # noqa: E402

crf, he, mlf, psa = L.crf, L.he, L.mlf, L.psa

OUTDIR = os.path.join(crf.ROOT_DIR, "Experimental_Data", "Output", "NitrogenQuenchVsEEDF")
LAS_P_W = (75.0, 85.0)                              # LAS powers of the quenching curves (median per % N2)
EN_DC = mlf.anc.CONFIG["eedfs"]["dc"]["EN_Td"]      # 3-200 Td, as the DC Ar/N2 libraries of
                                                    # ActinometryNitrogenContent (precision 1e-25: at 1e-30 BOLSIG+
                                                    # fails at 1 Td and then hangs with N2)
VARIANTS = {
    "A": dict(label="Ar/N$_2$ microwave EEDF + N$_2$ quenching (baseline)", eedf="mw_mix", Q=1.0, color="C3", marker="s"),
    "B": dict(label="Ar/N$_2$ microwave EEDF, no N$_2$ quenching", eedf="mw_mix", Q=0.0, color="C1", marker="v"),
    "C": dict(label="pure-Ar microwave EEDF + N$_2$ quenching", eedf="mw_ar", Q=1.0, color="C0", marker="o"),
    "D": dict(label="pure-Ar microwave EEDF, no N$_2$ quenching", eedf="mw_ar", Q=0.0, color="C9", marker="^"),
    "E": dict(label="Ar/N$_2$ DC EEDF + N$_2$ quenching", eedf="dc_mix", Q=1.0, color="C2", marker="D"),
}
FIGSIZE = psa.FIGSIZE


# ---- libraries and CR tables -------------------------------------------------------------
def dc_lib_folder(n2, Tg):
    return he.BOLSIG_FOLDER / f"ArN2_{n2:g}pct_bolsig_dc_{Tg:g}K"


def dc_library(n2, Tg):
    """BOLSIG+ DC E/N sweep of Ar + n2 % N2 at Tg (settings of ActinometryNitrogenContent's DC libraries), run if
    missing."""
    folder = dc_lib_folder(n2, Tg)
    if not he.IsBolsigLibrary(folder):
        he.RunBolsig(he.BOLSIG_XSEC_FOLDER / "Biagi_ArN2.txt", folder.name, EN_DC, species=["Ar", "N2"],
                     fractions=[1 - n2 / 100, n2 / 100], **dict(mlf.BOLSIG_SETTINGS, Tg=Tg, overwrite=True,
                                                                 verbose=False, timeout=psa.BOLSIG_TIMEOUT))
    return folder


def lib_of(eedf, n2, p_mTorr, Tg):
    if eedf == "mw_mix":
        return L.lib_folder(n2, p_mTorr, Tg)
    if eedf == "mw_ar":
        return L.lib_folder(0.0, p_mTorr, Tg)
    return dc_lib_folder(n2, Tg)


def build_lib(eedf, n2, p_mTorr, Tg):
    if eedf == "dc_mix":
        return dc_library(n2, Tg)
    return L.library(n2 if eedf == "mw_mix" else 0.0, p_mTorr, Tg)


def variant_cfg(v, n2, p_mTorr, Tg):
    """CRFitNeTe settings of variant v at (n2 % N2, p_mTorr, Tg); A is LASTgComparison's table."""
    if v == "A":
        return L.table_cfg(L._key(n2, p_mTorr, Tg), "n2")
    V, G = VARIANTS[v], L.GROUPS["n2"]
    lib = lib_of(V["eedf"], n2, p_mTorr, Tg)
    return dict(crf.CONFIG, sweeps=G["sweeps"], eedf=str(lib), P_Torr=p_mTorr / 1000, Tg=Tg, N2_percent=n2,
                N2_quench_scale=V["Q"], Ne_grid=L.NE_GRID, outdir=os.path.join(OUTDIR, "tables", v, lib.name),
                measure_dir=G["measure_dir"])


def _parse(key):
    a, n2, p, Tg = key.split(":")
    return a, float(n2), float(p), float(Tg)


def _lib_key(eedf, n2, p, Tg):
    """One key per library folder: the pure-Ar library has no N2, the DC library no pressure."""
    return f"{eedf}:{0 if eedf == 'mw_ar' else n2:g}:{0 if eedf == 'dc_mix' else p:g}:{Tg:g}"


def jobs(cond):
    states = {(r.N2, r.p_mTorr, r.Tg) for r in cond.itertuples()}
    libs = sorted({_lib_key(V["eedf"], n2, p, Tg) for V in VARIANTS.values() for n2, p, Tg in states
                   if not he.IsBolsigLibrary(lib_of(V["eedf"], n2, p, Tg))})
    tabs = sorted({f"{v}:{n2:g}:{p:g}:{Tg:g}" for v in VARIANTS for n2, p, Tg in states
                   if not crf.table_is_current(variant_cfg(v, n2, p, Tg))})
    return libs, tabs


def _run_workers(flag, keys, what):
    """This script with flag for each key, mlf.N_WORKERS processes at a time (one BLAS thread each)."""
    if not keys:
        return
    logs = os.path.join(OUTDIR, "logs")
    os.makedirs(logs, exist_ok=True)
    print(f"{what}: {len(keys)} jobs ({mlf.N_WORKERS} at a time, logs in {logs}) ...", flush=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0, procs = time.time(), {}
    for k in keys:
        while sum(q.poll() is None for q in procs.values()) >= mlf.N_WORKERS:
            time.sleep(3)
        log = open(os.path.join(logs, f"{flag.strip('-')}_{k.replace(':', '_')}.log"), "w")
        procs[k] = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), flag, k],
                                    stdout=log, stderr=subprocess.STDOUT, cwd=crf.ROOT_DIR, env=env)
    failed = [k for k, q in procs.items() if q.wait() != 0]
    print(f"{what} done ({time.time() - t0:.0f} s)", flush=True)
    if failed:
        raise RuntimeError(f"{what} failed for {failed} - see {logs}")


# ---- 1. LAS quenching curves -----------------------------------------------------------------
def las_quench(las, model):
    """Per run: 1/n(1s5) = a + b x (x = % N2, LAS_P_W) -> nu_0 = k_Q N a / b."""
    kq = he.ImportArQuenchingData("N2", verbose=False)["4s1"]["kQ"]
    d = las[las.valid_n & las.P.between(*LAS_P_W)]
    rows = []
    for (run, p), g in d.groupby(["run", "p"]):
        s = g.groupby("x").n1s5.median()
        if len(s) < 3 or np.ptp(s.index) < 2:
            continue
        x, inv = s.index.to_numpy(float), 1.0 / s.to_numpy(float)
        b, a = np.polyfit(x, inv, 1)
        tg = g.Tg[g.valid_Tg]
        Tg = float(tg.median()) if len(tg) else float(L.tg_trend(model, p, 80.0, np.median(x)))
        N = he.Torr2Volume(p, Tg)
        rows.append(dict(run=run, date=g.date.iloc[0], p=p, x_min=x.min(), x_max=x.max(), n_x=len(x), a=a, b=b,
                         n1s5_at_0pct=1 / a, Tg=Tg, N=N, kQ=kq, nu0=kq * N * (a / b) / 100))
    return pd.DataFrame(rows).sort_values(["p", "date"]).reset_index(drop=True), d


def model_loss(cfg, EN, Ne=L.N_C, label="4s1"):
    """CR model at (EEDF row nearest E/N, Ne): 1s5 density and loss frequency split into electron collisions
    and the rest [s^-1]."""
    cr, _, MD, interp = crf.load_cr_model(cfg["outdir"], cfg.get("escape_mode", "table"))
    MD = he.AddDiffusionLoss(MD, cfg["P_Torr"], cfg["Tg"], cfg["R"])
    P_Ar = crf.apply_nitrogen(MD, cfg, he)
    with contextlib.redirect_stdout(io.StringIO()):
        eedfs, x, _, _ = crf.eedf_axis(cfg)
    i = int(np.argmin(np.abs(np.log(x / EN))))
    with contextlib.redirect_stdout(io.StringIO()):
        D, _, _, _ = cr["CRModel"](MD, eedfs[i], float(Ne), P_Ar, cfg["Tg"], cfg["R"], interp,
                                   trap_lines=cfg["trap_lines"], atom_transfer=cfg.get("atom_transfer", True))
    US = D[label]
    nu_e = Ne * (sum(r["Rate"] for r in US["Electron Impact CrossSections"]["Reactants"])
                 + sum(s["coeff"] for s in US.get("Superelastic", []) if s["direction"] == "loss")
                 + US["Ionization Data"]["Rate_cm^3"])
    return dict(EN_row=x[i], n1s5=US["density_m^-3"], nu_total=US["Loss_s^-1"], nu_e=nu_e,
                nu_other=US["Loss_s^-1"] - nu_e)


def plot_las_quench(q, d, nu_model, outdir):
    ps = sorted(q.p.unique())
    col = {p: plt.cm.viridis(i / max(len(ps) - 1, 1)) for i, p in enumerate(ps)}
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for r in q.itertuples():
        g = d[(d.run == r.run)].groupby("x").n1s5.median()
        mk = "o" if r.date < "2026-09-01" else "^"
        ax.plot(g.index, 1e16 / g.to_numpy(), mk, color=col[r.p], ms=5, label=f"{r.date} {r.run[:4]} ({r.p:g} Torr)")
        xx = np.linspace(0, r.x_max, 20)
        ax.plot(xx, 1e16 * (r.a + r.b * xx), "-", color=col[r.p], lw=1)
    ax.set_ylabel("10$^{16}$ / n(1s$_5$)  [m$^3$]  (LAS)")
    ax.legend(fontsize=7, ncol=2)
    psa._save(fig, ax, os.path.join(outdir, "las_quench_inverse_n.png"),
              f"LAS 1s$_5$ against N$_2$ ({LAS_P_W[0]:g}-{LAS_P_W[1]:g} W): "
              "$1/n = (\\nu_0 + k_Q\\,x\\,N)\\,/\\,P$",
              xlabel="N$_2$ in the feed [%]", legend=False)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for r in q.itertuples():
        mk = "o" if r.date < "2026-09-01" else "^"
        ax.plot(r.p, r.nu0, mk, color=col[r.p], ms=8, mec="k", mew=0.5)
        ax.annotate(f"{r.date[5:]}", (r.p, r.nu0), xytext=(5, 0), textcoords="offset points", fontsize=7, va="center")
    m = nu_model.sort_values("p")
    ax.plot(m.p, m.nu_total, "s-", color="C3", ms=5, label="CR model at n$_c$: total 1s$_5$ loss (pure Ar, LAS T$_g$)")
    ax.plot(m.p, m.nu_e, "--", color="C3", lw=1, label="CR model at n$_c$: electron collisions only")
    ax.plot(m.p, m.nu_other, ":", color="0.4", lw=1.2, label="CR model: diffusion + Ar quenching + pooling")
    ax.set_yscale("log")
    ax.set_ylabel("1s$_5$ loss frequency without N$_2$, $\\nu_0$ [s$^{-1}$]")
    psa._save(fig, ax, os.path.join(outdir, "las_quench_nu0.png"),
              "1s$_5$ loss frequency: from the LAS quenching curves (points, $k_Q$ = "
              f"{q.kQ.iloc[0]:.2g} m$^3$/s) and the CR model", xlabel="Pressure [Torr]")


# ---- 2. measured line pattern -------------------------------------------------------------
def line_pattern(ft, sweep="N2 fraction"):
    """Mean over the repeats of ln(feature) - spectrum mean, relative to x = 0 -> long table."""
    P = ft[ft.sweep == sweep].pivot_table(index=["x", "file"], columns="feature", values="area").dropna()
    y = np.log(P)
    y = y.sub(y.mean(axis=1), axis=0)
    g = y.groupby(level="x")
    mean, sem = g.mean(), g.std(ddof=1) / np.sqrt(g.size().to_numpy()[:, None])
    rel = mean - mean.loc[0.0]
    err = np.sqrt(sem ** 2 + sem.loc[0.0] ** 2)
    return (rel.stack().rename("dln").to_frame().join(err.stack().rename("err")).reset_index())


def plot_line_pattern(pat, outdir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    feats = list(dict.fromkeys(pat.feature))
    cols = dict(zip(feats, plt.cm.tab10(np.linspace(0, 1, 10))))
    for k, f in enumerate(feats):
        h = pat[pat.feature == f].sort_values("x")
        ax.errorbar(h.x + 0.02 * (k - len(feats) / 2), h.dln, yerr=h.err, fmt="o-", ms=4, lw=1.2, capsize=2,
                    color=cols[f], label=f)
    ax.axhline(0, color="0.3", lw=1)
    ax.set_ylabel("$\\Delta$ ln(line / mean of the 9 lines) from 0 % N$_2$")
    ax.legend(fontsize=7, ncol=3)
    psa._save(fig, ax, os.path.join(outdir, "line_pattern_vs_N2.png"),
              "Measured Ar I line pattern against N$_2$ (85 W, 1 Torr; mean of the repeats, no model)",
              xlabel="N$_2$ in the feed [%]", legend=False)


# ---- 3. variant fits and plots -------------------------------------------------------------
def plot_variants(res, las, ref0, sweep, outdir):
    o = L.OES[sweep]
    r = res[res.sweep == sweep]
    for key, ylab, log, fname in (("n_1s5", "Ar(1s$_5$) density [m$^{-3}$]", True, "1s5"),
                                  ("chi2_red_nc", "$\\chi^2$/dof with N$_e$ = n$_c$", False, "chi2"),
                                  ("EN_nc", "E/N of the fit at n$_c$ [Td]", True, "EN")):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        if key == "n_1s5":
            for k, (lab, g) in enumerate(L._las_points(las[las.valid_n], sweep)):
                ax.plot(g.X, g.n1s5, "o^"[k % 2], color="0.7", ms=4, alpha=0.7, label=lab)
            c = r[r.variant == "A"].sort_values("x")
            ax.errorbar(c.x, c.las_1s5, yerr=c.las_1s5 * c.las_1s5_rel_err, fmt="D", color="k", ms=6, capsize=3,
                        mfc="white", label="LAS at the condition")
        for v, V in VARIANTS.items():
            g = r[r.variant == v].sort_values("x")
            if key == "n_1s5":
                ax.fill_between(g.x, g.n_1s5_lo, g.n_1s5_hi, color=V["color"], alpha=0.10, lw=0)
            ax.plot(g.x, g[key], V["marker"] + "-", color=V["color"], ms=5, label=f"{v}: {V['label']}")
        if ref0 is not None and sweep == "N2 fraction":
            ax.plot([0.0], [ref0[key]], "*", color="k", ms=12, label="0 % N$_2$ (pure Ar, microwave)")
        if key == "chi2_red_nc":
            ax.axhline(1, color="0.6", lw=0.8)
        if log:
            ax.set_yscale("log")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=6.5)
        psa._save(fig, ax, os.path.join(outdir, f"{fname}_{o['tag']}.png"),
                  f"Quenching or EEDF? CR fits at n$_c$, LAS T$_g$: {o['note']}", xlabel=o["xlabel"], legend=False)


def nc_profile(cfg, sweep, x, feats, ft):
    """chi^2/dof, Te_eff and CR 1s5 along E/N with Ne = n_c (fine grid of the fit)."""
    tab = crf.build_model_table(cfg)
    grid = crf.fine_grid(tab, cfg)
    Te_f = crf.te_eff_fine(tab, grid)
    M = crf.feature_model(tab, feats, cfg)
    d = ft[(ft.sweep == sweep) & np.isclose(ft.x, x)]
    cond, _, _, posts = crf.fit_all(d, feats, M, grid, Te_f, cfg, verbose=False)
    c = cond.iloc[0]
    fX, fN = grid
    j = int(np.argmin(np.abs(fN - np.log(L.N_C))))
    chi2 = mlf.cdf.chi2_grid(posts[f"{sweep}|{x:g}"], c.chi2_min, c.birge)[:, j]
    lev = list(tab["levels"])
    ln = crf.on_fine_grid(tab, np.log(np.clip(tab["dens"][:, :, lev.index("4s1")], 1e-300, None)), cfg, k=1)[:, j]
    return pd.DataFrame(dict(EN=np.exp(fX), chi2_red=chi2 / (c.dof + 1),
                             Te_eff=Te_f if np.ndim(Te_f) == 1 else Te_f[:, j], n_1s5=np.exp(ln)))


PROFILE_CONDITIONS = [("N2 fraction", 1.0), ("N2 fraction", 3.0), ("Power", 55.0)]


def plot_profiles(prof, las_n, sweep, x, outdir):
    """chi^2/dof and CR 1s5 along E/N at n_c for every variant: one file each."""
    o = L.OES[sweep]
    tag = f"{o['tag']}_{x:g}"
    title = f"N$_e$ = n$_c$, LAS T$_g$: {o['note'].split(',')[0] if sweep == 'Power' else '85 W, 1 Torr'}, " \
            + (f"{x:g} % N$_2$" if sweep == "N2 fraction" else f"{x:g} W, 2.4 % N$_2$")
    for key, ylab, log, fname in (("chi2_red", "$\\chi^2$/dof along E/N (N$_e$ = n$_c$)", False, "chi2_profile"),
                                  ("n_1s5", "CR Ar(1s$_5$) [m$^{-3}$] along E/N (N$_e$ = n$_c$)", True, "1s5_profile")):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for v, V in VARIANTS.items():
            g = prof[prof.variant == v]
            ax.plot(g.EN, g[key], "-", color=V["color"], lw=1.6, label=f"{v}: {V['label']}")
            b = g.loc[g.chi2_red.idxmin()]
            ax.plot([b.EN], [b[key]], V["marker"], color=V["color"], ms=8, mec="k", mew=0.6)
        if key == "n_1s5":
            ax.axhline(las_n, color="k", lw=1.5, ls="--", label=f"LAS 1s$_5$ ({las_n:.2g} m$^{{-3}}$)")
        else:
            ax.set_ylim(0, max(8.0, 1.1 * prof.chi2_red[np.isfinite(prof.chi2_red)].min() + 6))
        ax.plot([], [], "ko", mec="k", mfc="none", label="best E/N of each variant")
        ax.set_xscale("log")
        if log:
            ax.set_yscale("log")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=6.5)
        psa._save(fig, ax, os.path.join(outdir, f"{fname}_{tag}.png"), title, xlabel="E/N [Td]", legend=False)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    las, model, off, tgfit, cond = L.plan()
    comp = pd.read_csv(os.path.join(L.OUTDIR, "comparison.csv"))      # LASTgComparison fits (pure Ar E/N at n_c)

    # 1. LAS quenching curves against the model's loss frequency without N2
    q, dq = las_quench(las, model)
    q.to_csv(os.path.join(OUTDIR, "las_quench_fits.csv"), index=False)
    ps = comp[comp.sweep == "Pressure"]
    nu = []
    for p in sorted(q.p.unique()):
        c = ps.iloc[int(np.argmin(np.abs(ps.p_Torr - p)))]
        m = model_loss(L.table_cfg(L._key(0.0, c.p_mTorr, c.Tg)), c.EN_nc)
        nu.append(dict(p=c.p_Torr, Tg=c.Tg, EN_nc=c.EN_nc, **m))
    nu = pd.DataFrame(nu)
    nu.to_csv(os.path.join(OUTDIR, "model_loss_pure_ar_nc.csv"), index=False)
    plot_las_quench(q, dq, nu, OUTDIR)
    with pd.option_context("display.width", 200):
        print("LAS quenching curves (1/n = a + b x):")
        print(q[["run", "date", "p", "x_min", "x_max", "Tg", "n1s5_at_0pct", "nu0"]].to_string(
            index=False, float_format=lambda v: f"{v:.3g}"))
        print("CR model 1s5 loss at n_c, pure Ar (LAS Tg):")
        print(nu.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # libraries and tables of the variants
    c2 = cond[(cond.group == "n2") & (cond.N2 > 0)]
    libs, tabs = jobs(c2)
    print(f"\nto build: {len(libs)} libraries, {len(tabs)} CR tables")
    _run_workers("--lib", libs, "BOLSIG+ libraries")
    _run_workers("--build", tabs, "CR tables")

    # 2. measured line pattern
    first = c2.iloc[0]
    feats, ft = L.measurements(variant_cfg("A", first.N2, first.p_mTorr, first.Tg))
    pat = line_pattern(ft)
    pat.to_csv(os.path.join(OUTDIR, "line_pattern_vs_N2.csv"), index=False)
    plot_line_pattern(pat, OUTDIR)

    # 3. variant fits
    rows = []
    for c in c2.itertuples():
        for v in VARIANTS:
            f = L.fit_condition(variant_cfg(v, c.N2, c.p_mTorr, c.Tg), c.sweep, c.x, feats, ft)
            rows.append(dict(variant=v, sweep=c.sweep, x=c.x, N2=c.N2, Tg=c.Tg, las_1s5=c.las_1s5,
                             las_1s5_rel_err=c.las_1s5_rel_err, **f))
        print(f"  {c.sweep:<11s} {c.x:5g}: " + "   ".join(
            f"{r['variant']} chi2 {r['chi2_red_nc']:4.2f} E/N {r['EN_nc']:5.0f} 1s5/LAS {r['n_1s5'] / c.las_1s5:5.2f}"
            for r in rows[-len(VARIANTS):]), flush=True)
    res = pd.DataFrame(rows)
    res["ratio_cr_las"] = res.n_1s5 / res.las_1s5
    res.to_csv(os.path.join(OUTDIR, "variants.csv"), index=False)
    ref0 = comp[(comp.sweep == "N2 fraction") & (comp.x == 0)].iloc[0]
    for s in ("N2 fraction", "Power"):
        plot_variants(res, las, ref0, s, OUTDIR)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(res.pivot_table(index=["sweep", "x"], columns="variant",
                              values=["chi2_red_nc", "EN_nc", "Te_nc", "ratio_cr_las"]).round(2).to_string())

    # 4. chi^2 and CR 1s5 along E/N at n_c for a few conditions
    profs = []
    for sweep, x in PROFILE_CONDITIONS:
        c = c2[(c2.sweep == sweep) & np.isclose(c2.x, x)].iloc[0]
        prof = pd.concat([nc_profile(variant_cfg(v, c.N2, c.p_mTorr, c.Tg), sweep, x, feats, ft).assign(variant=v)
                          for v in VARIANTS], ignore_index=True)
        plot_profiles(prof, c.las_1s5, sweep, x, OUTDIR)
        profs.append(prof.assign(sweep=sweep, x=x))
    pd.concat(profs, ignore_index=True).to_csv(os.path.join(OUTDIR, "nc_profiles.csv"), index=False)
    print(f"\nwritten to {OUTDIR}")
    return q, nu, pat, res


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--lib":
        eedf, n2, p, Tg = _parse(sys.argv[2])
        build_lib(eedf, n2, p, Tg)
    elif len(sys.argv) > 2 and sys.argv[1] == "--build":
        v, n2, p, Tg = _parse(sys.argv[2])
        tab = crf.build_model_table(variant_cfg(v, n2, p, Tg))
        print(f"{sys.argv[2]}: {len(tab['x_grid'])} E/N rows from {tab['x_grid'][0]:g} Td, "
              f"{(~tab['converged']).sum()} not converged")
    else:
        Q, NU, PAT, RES = main()
