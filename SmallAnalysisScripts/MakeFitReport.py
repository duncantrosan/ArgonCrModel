# -*- coding: utf-8 -*-
"""
MakeFitReport.py

Figures and tables of the Overleaf report LatexReport/main.tex (repo root) on the N2SweepPureArEEDF analysis:
CR fit of the echelle sweeps (per-spectrum scale s, Ne free or pinned at n_c), the BOLSIG+ libraries behind it and
the N I / Ar I actinometry.  main.tex is written by hand; this script only refreshes LatexReport/figures and
LatexReport/tables from the outputs of the last N2SweepPureArEEDF run (run that first):

  figures/  copies of the overview and per-set figures (file names without extra dots, for graphicx), and
            new figures of the example conditions EXAMPLE_SET at EXAMPLE_X / S_EXAMPLE_X (refitted here):
              fit_free_vs_nc.png   Delta chi^2 / B^2 over (E/N, Ne) with the free and the pinned minimum,
                                   and the profile over Ne (minimum over E/N per Ne)
              s_scale.png          line intensities of every spectrum of one condition against the model at the
                                   pinned fit, before and after the per-spectrum scale s, and the residuals
              pin_cost.png         cost of the pin (Delta chi^2 / B^2) and the free-fit Ne of every condition
  tables/   libraries.tex (BOLSIG+ libraries / CR tables per condition), fits.tex, actinometry.tex (longtables),
            numbers.tex (\newcommand macros used in the text: n_c, Tg trend, counts)

Run from Spyder (F5) or python SmallAnalysisScripts/MakeFitReport.py
"""
import contextlib
import io
import os
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import N2SweepPureArEEDF as S                       # noqa: E402

L, crf, A, he = S.L, S.crf, S.A, S.he
REPORT = os.path.join(crf.ROOT_DIR, "LatexReport")
FIGDIR, TABDIR = os.path.join(REPORT, "figures"), os.path.join(REPORT, "tables")
EXAMPLE_SET = "80W_1Torr"                           # data set of the example figures
EXAMPLE_X = (0.33, 2.0)                             # % N2: free fit at the Ne grid edge / close to n_c
S_EXAMPLE_X = 2.0                                   # % N2 of the s figure
COPY = [   # (path in the N2SweepPureArEEDF output folder, name in figures/)
    ("overview_N2_1s5.png", "ov_N2_1s5.png"),
    ("overview_N2_EN.png", "ov_N2_EN.png"),
    ("overview_N2_Te_eff.png", "ov_N2_Te_eff.png"),
    ("overview_N2_actinometry.png", "ov_N2_actinometry.png"),
    ("overview_N2_dissociation.png", "ov_N2_dissociation.png"),
    ("80W_3pctN2_pressure/figures/metastables_vs_pressure.png", "p_metastables.png"),
    ("80W_3pctN2_pressure/figures/EN_vs_pressure.png", "p_EN.png"),
    ("80W_3pctN2_pressure/figures/actinometry_vs_pressure.png", "p_actinometry.png"),
    ("80W_3pctN2_pressure/figures/dissociation_vs_pressure.png", "p_dissociation.png"),
    ("80W_1Torr/figures/actinometry_vs_N2.png", "80W_actinometry_variants.png"),
    ("80W_1Torr/figures/actinometry_per_line_NI746.83_vs_N2.png", "80W_per_line_NI746p83.png"),
    ("80W_1Torr/figures/actinometry_by_N_line_vs_N2.png", "80W_by_N_line.png"),
    ("80W_1Torr/figures/NI_746.83_over_744.23_vs_N2.png", "80W_NI_same_upper.png"),
    ("80W_1Torr/figures/line_ratios_NI746.83_vs_N2.png", "80W_line_ratios_NI746p83.png"),
    ("80W_1Torr/figures/raw_lines/raw_NI_746.83nm.png", "80W_raw_NI746p83.png"),
    ("80W_1Torr/figures/raw_lines/raw_ArI_750.39nm.png", "80W_raw_ArI750p39.png"),
]


def tex(s):
    """Text for LaTeX (underscores, percent signs)."""
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def sci(v, d=2):
    """3.1e+16 -> 3.1\\times10^{16} (math mode)."""
    if not np.isfinite(v):
        return "--"
    m, e = f"{v:.{d - 1}e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def copy_figures():
    for src, dst in COPY:
        path = os.path.join(S.OUTDIR, src)
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(FIGDIR, dst))
        else:
            print(f"missing (run N2SweepPureArEEDF first?): {path}")


# ---- example fits ------------------------------------------------------------------------
def fit_one(ds, c):
    """CR fit of condition c (row of N2SweepPureArEEDF.conditions) as N2SweepPureArEEDF does it -> dict."""
    cfg = S.fit_cfg(c, ds)
    with A.xsec_set(A.CR_XSEC), contextlib.redirect_stdout(io.StringIO()):
        feats, ft = L.measurements(cfg)
        tab = crf.build_model_table(cfg)
        grid = crf.fine_grid(tab, cfg)
        Te_f = crf.te_eff_fine(tab, grid)
        M = crf.feature_model(tab, feats, cfg)
    d = ft[(ft.sweep == c.sweep) & np.isclose(ft.x, c.x)]
    fc, _, _, posts = crf.fit_all(d, feats, M, grid, Te_f, cfg, verbose=False)
    f0 = fc.iloc[0]
    chi2 = L.mlf.cdf.chi2_grid(posts[f"{c.sweep}|{c.x:g}"], f0.chi2_min, f0.birge)
    j = int(np.argmin(np.abs(grid[1] - np.log(S.N_C))))
    return dict(cfg=cfg, feats=feats, d=d, M=M, grid=grid, fit=f0, chi2=chi2, j=j, i=int(np.argmin(chi2[:, j])))


def plot_free_vs_nc(fits, path):
    """Delta chi^2 / B^2 over (E/N, Ne) and its profile over Ne, one row per example condition."""
    fig, axs = plt.subplots(len(fits), 2, figsize=(12, 4.6 * len(fits)), gridspec_kw=dict(width_ratios=[1.25, 1]))
    for ax, (x, F) in zip(np.atleast_2d(axs), fits.items()):
        f, (fX, fN), s2 = F["fit"], F["grid"], F["fit"].birge ** 2
        dchi = (F["chi2"] - F["chi2"].min()) / s2
        EN, Ne = np.exp(fX), np.exp(fN)
        pc = ax[0].pcolormesh(EN, Ne, np.minimum(dchi, 25).T, cmap="viridis_r", shading="auto")
        ax[0].contour(EN, Ne, dchi.T, levels=[1, 4, 9], colors="w", linewidths=[1.5, 1, 0.7])
        fig.colorbar(pc, ax=ax[0], label=r"$\Delta\chi^2 / B^2$ (clipped at 25)")
        ax[0].axhline(S.N_C, color="C3", ls="--", lw=1.5, label=f"n$_c$ = {S.N_C:.2e} m$^{{-3}}$ (2.42 GHz)")
        band = EN[F["chi2"][:, F["j"]] <= F["chi2"][F["i"], F["j"]] + s2]
        ax[0].plot([band.min(), band.max()], [S.N_C, S.N_C], "-", color="C3", lw=5, alpha=0.6)
        ax[0].plot(f.x_best, f.Ne_best, "*", color="gold", mec="k", ms=18, label="free fit (global minimum)")
        ax[0].plot(EN[F["i"]], S.N_C, "o", color="C3", mec="k", ms=10, label=r"pinned fit, band $\Delta\chi^2 \leq B^2$")
        ax[0].set(xscale="log", yscale="log", xlabel="E/N of the EEDF [Td]", ylabel="N$_e$ [m$^{-3}$]")
        ax[0].set_title(f"{EXAMPLE_SET}, {x:g} % N$_2$: contours $\\Delta\\chi^2/B^2$ = 1, 4, 9 "
                        f"(B = Birge ratio {f.birge:.2f})", fontsize=10)
        ax[0].legend(fontsize=8, loc="lower left")
        prof = dchi.min(axis=0)
        ax[1].plot(Ne, prof, "k-", lw=2)
        ax[1].axvline(S.N_C, color="C3", ls="--", lw=1.5, label="n$_c$")
        ax[1].axhline(1, color="0.5", ls=":", lw=1.2, label=r"$\Delta\chi^2 = B^2$")
        ax[1].plot(S.N_C, prof[F["j"]], "o", color="C3", mec="k", ms=10,
                   label=f"cost of the pin: {prof[F['j']]:.1f} B$^2$")
        ax[1].plot(f.Ne_best, 0, "*", color="gold", mec="k", ms=16, label=f"free fit, N$_e$ = {f.Ne_best:.1e}")
        ax[1].set(xscale="log", xlabel="N$_e$ [m$^{-3}$]", ylabel=r"min over E/N of $\Delta\chi^2/B^2$",
                  ylim=(-0.5, max(12, 1.1 * prof[F["j"]])))
        ax[1].set_title(f"profile over N$_e$: $\\chi^2$/dof {f.chi2_red:.2f} free, "
                        f"{F['chi2'][F['i'], F['j']] / (f.dof + 1):.2f} at n$_c$", fontsize=10)
        ax[1].grid(True, which="both", alpha=0.25)
        ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_s(F, x, path):
    """Line intensities of each spectrum against the model at the pinned fit, before / after its scale s."""
    d, cfg, feats = F["d"], F["cfg"], F["feats"]
    li = d.feature.map({f: k for k, f in enumerate(feats.feature)}).to_numpy()
    si, files = pd.factorize(d.file)
    y, m = np.log(d.area.to_numpy()), F["M"][F["i"], F["j"], li]
    X = np.eye(si.max() + 1)[si]
    sig2 = d.rel_err.to_numpy() ** 2 + cfg["sigma_fit_floor"] ** 2
    C = np.diag(sig2) + cfg["sigma_model"] ** 2 * (li[:, None] == li[None, :])
    W = np.linalg.inv(C)
    s = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ (y - m)            # the profiled scales, as in fit_block
    cols = plt.cm.tab10(np.arange(len(files)) % 10)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for k, f in enumerate(files):
        sel = si == k
        axs[0].plot(m[sel], y[sel], "o", color=cols[k], ms=5, label=f"{f}: s = {s[k]:.2f}")
        t = np.linspace(m.min(), m.max(), 2)
        axs[0].plot(t, t + s[k], "-", color=cols[k], lw=0.8, alpha=0.7)
        axs[1].errorbar(m[sel], y[sel] - s[k], yerr=np.sqrt(sig2[sel]), fmt="o", color=cols[k], ms=5, capsize=2)
        axs[2].errorbar(d.wl.to_numpy()[sel] + 0.6 * (k - len(files) / 2), y[sel] - s[k] - m[sel],
                        yerr=np.sqrt(sig2[sel]), fmt="o", color=cols[k], ms=4, capsize=2)
    t = np.linspace(m.min(), m.max(), 2)
    axs[1].plot(t, t, "k--", lw=1.2, label="measured = model")
    axs[2].axhline(0, color="k", lw=1)
    axs[2].axhspan(-cfg["sigma_model"], cfg["sigma_model"], color="0.85", zorder=0,
                   label=f"model error $\\sigma_{{model}}$ = {cfg['sigma_model']:.2f}")
    axs[0].set(xlabel=r"ln $I_{model}$ (pinned fit)", ylabel=r"ln $I_{meas}$ (calibrated line area)",
               title="before: one offset s per spectrum (lines of slope 1)")
    axs[1].set(xlabel=r"ln $I_{model}$ (pinned fit)", ylabel=r"ln $I_{meas}$ $-$ s",
               title="after removing s: every spectrum on one line")
    axs[2].set(xlabel="wavelength of the feature [nm]", ylabel=r"ln $I_{meas}$ $-$ s $-$ ln $I_{model}$",
               title="residuals (bars: line fit + floor + repeat scatter)")
    for ax in axs:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle(f"{EXAMPLE_SET}, {x:g} % N$_2$, N$_e$ = n$_c$, E/N = {np.exp(F['grid'][0][F['i']]):.1f} Td: "
                 f"{len(files)} spectra, {len(np.unique(li))} line features", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return pd.DataFrame(dict(file=files, s=s))


def plot_pin_cost(fits, kinds, path):
    """Delta chi^2 / B^2 of the pin and the free-fit Ne of every condition, N2 sweeps and pressure sweep."""
    groups = [("N2", [n for n, k in kinds.items() if k == "N2"]), ("pressure", [n for n, k in kinds.items()
                                                                                if k == "pressure"])]
    groups = [(k, n) for k, n in groups if n]
    fig, axs = plt.subplots(2, len(groups), figsize=(7 * len(groups), 8), squeeze=False)
    for col, (kind, names) in enumerate(groups):
        for i, n in enumerate(names):
            t = fits[fits.dataset == n].sort_values("x")
            axs[0, col].plot(t.x, t.dchi2_nc, "s-", color=f"C{i}", ms=6, label=n)
            edge = t.Ne_free >= 0.99 * crf.CONFIG["Ne_grid"][-1]
            axs[1, col].plot(t.x, t.Ne_free, "-", color=f"C{i}", lw=1)
            axs[1, col].plot(t.x[~edge], t.Ne_free[~edge], "s", color=f"C{i}", ms=6, label=n)
            axs[1, col].plot(t.x[edge], t.Ne_free[edge], "^", color=f"C{i}", mfc="white", ms=8)
        axs[1, col].plot([], [], "^", color="k", mfc="white", ms=8, label="at the N$_e$ grid edge")
        axs[0, col].axhline(1, color="0.5", ls=":", lw=1.2)
        axs[1, col].axhline(S.N_C, color="C3", ls="--", lw=1.5, label="n$_c$")
        axs[0, col].set(ylabel=r"cost of the pin $(\chi^2_{n_c} - \chi^2_{min}) / B^2$", ylim=(0, None))
        axs[1, col].set(yscale="log", ylabel="N$_e$ of the free fit [m$^{-3}$]", xlabel=S.SWEPT[kind]["xlabel"])
        for ax in axs[:, col]:
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle("Free fit against the fit with N$_e$ pinned at n$_c$ (pure-Ar microwave EEDF, no tilt)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---- tables ------------------------------------------------------------------------------
def write(name, text):
    with open(os.path.join(TABDIR, name), "w", encoding="utf-8") as f:
        f.write(text)


def libraries_table(conds):
    """One row per BOLSIG+ library (p, Tg): the conditions that use it and their CR tables (% N2)."""
    rows = []
    for (p, Tg), g in conds.groupby(["p_mTorr", "Tg"]):
        use = "; ".join(f"{tex(n)}: " + ", ".join(f"{x:g}" for x in h.x) for n, h in g.groupby("set", sort=False))
        rows.append(rf"\texttt{{{tex(L.lib_folder(0.0, p, Tg).name)}}} & {p:g} & {Tg:g} & "
                    rf"{g.N2.nunique()} & {use} \\")
    write("libraries.tex", "\n".join([
        r"\begingroup\footnotesize", r"\begin{longtable}{p{0.42\linewidth}rrrp{0.30\linewidth}}",
        r"\caption{BOLSIG+ libraries (pure Ar, 2.42\,GHz, Biagi cross sections, " + f"{len(L.EN_MW)}"
        + r" E/N rows from 3 to 1000\,Td) and the conditions that use them. One CR table per library and N$_2$"
        r" fraction (column N$_2$: number of tables).}\label{tab:libraries}\\",
        r"\toprule library & $p$ [mTorr] & $T_g$ [K] & N$_2$ & conditions (sweep values) \\ \midrule\endfirsthead",
        r"\toprule library & $p$ [mTorr] & $T_g$ [K] & N$_2$ & conditions (sweep values) \\ \midrule\endhead",
        *rows, r"\bottomrule", r"\end{longtable}", r"\endgroup", ""]))


def fits_table(fits):
    rows = []
    for n, g in fits.groupby("dataset", sort=False):
        rows.append(rf"\midrule\multicolumn{{9}}{{l}}{{\textbf{{{tex(n)}}}}}\\")
        for r in g.sort_values("x").itertuples():
            rows.append(rf"{r.x:g} & {r.Tg:g} & {r.EN_nc:.1f} [{r.EN_nc_lo:.1f}, {r.EN_nc_hi:.1f}] & {r.Te_nc:.3f} & "
                        rf"{r.chi2_red_nc:.2f} & {r.dchi2_nc:.1f} & ${sci(r.Ne_free)}$ & {r.chi2_red_free:.2f} & "
                        rf"{r.ratio_cr_las:.2f} \\")
    head = (r"$x$ & $T_g$ [K] & E/N at $n_c$ [Td] & $T_{e,\mathrm{eff}}$ [eV] & $\chi^2_\nu(n_c)$ & "
            r"$\Delta\chi^2/B^2$ & $N_e$ free [m$^{-3}$] & $\chi^2_\nu$ free & 1s$_5$ CR/LAS \\")
    write("fits.tex", "\n".join([
        r"\begin{longtable}{rrlrrrrrr}",
        r"\caption{CR fit of every condition: pinned at $n_c$ (E/N with its $\Delta\chi^2\le B^2$ band, "
        r"$T_{e,\mathrm{eff}}$, $\chi^2$ per degree of freedom), the cost of the pin and the free fit. $x$: % N$_2$ "
        r"or pressure in mTorr.}\label{tab:fits}\\",
        r"\toprule " + head + r" \endfirsthead", r"\toprule " + head + r" \endhead",
        *rows, r"\bottomrule", r"\end{longtable}", ""]).replace("% N$_2$", r"\% N$_2$"))


def actinometry_table(act):
    a = act[act.N_line == "combined"]
    rows = []
    for n, g in a.groupby("dataset", sort=False):
        rows.append(rf"\midrule\multicolumn{{7}}{{l}}{{\textbf{{{tex(n)}}}}}\\")
        for x, h in g.groupby("x"):
            v = {r.variant: r for r in h.itertuples()}
            cr, di = v.get("cr"), v.get("direct")
            rows.append(rf"{x:g} & {cr.n_pairs:d} & ${sci(cr.nN_nAr)}$ & {cr.ln_err_stat:.2f} & {cr.birge_pairs:.2f} & "
                        rf"{100 * cr.dissociation:.2f} & ${sci(di.nN_nAr) if di else '--'}$ \\")
    head = (r"$x$ & pairs & $n_N/n_{Ar}$ (CR) & $\sigma_{\ln}$ stat & Birge & dissociation [\%] & "
            r"$n_N/n_{Ar}$ (direct) \\")
    write("actinometry.tex", "\n".join([
        r"\begin{longtable}{rrrrrrr}",
        r"\caption{N I / Ar I actinometry at $n_c$, all pairs combined: with metastables (CR-model emission of the "
        r"Ar line) and, last column, without (ground-state excitation only). $\sigma_{\ln}$: statistical error of "
        r"$\ln(n_N/n_{Ar})$ (line ratios, E/N posterior, pair scatter); the cross sections add 0.25.}"
        r"\label{tab:actinometry}\\",
        r"\toprule " + head + r" \endfirsthead", r"\toprule " + head + r" \endhead",
        *rows, r"\bottomrule", r"\end{longtable}", ""]))


def numbers(trend, conds, fits):
    nlib = conds.groupby(["p_mTorr", "Tg"]).ngroups
    ntab = conds.key.nunique()
    m = {"Nc": sci(S.N_C, 3), "TgZero": f"{trend['T0']:.0f}", "TgAp": f"{trend['a_p']:.0f}",
         "TgAP": f"{trend['a_P']:.2f}", "TgAx": f"{trend['a_x']:.1f}", "TgAxx": f"{abs(trend['a_xx']):.2f}", "TgAxxSign": "-" if trend["a_xx"] < 0 else "+",
         "TgSigma": f"{trend['sigma_within']:.0f}", "NLib": str(nlib), "NTab": str(ntab),
         "NCond": str(len(fits)), "NSets": str(fits.dataset.nunique()), "NEN": str(len(L.EN_MW)),
         "NNe": str(len(L.NE_GRID))}
    write("numbers.tex", "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in m.items()))


def summary(fits, act):
    """Ranges quoted in the text."""
    pd.set_option("display.width", 200)
    g = fits.groupby("dataset", sort=False)
    print(g.agg(EN_min=("EN_nc", "min"), EN_max=("EN_nc", "max"), Te_min=("Te_nc", "min"), Te_max=("Te_nc", "max"),
                chi2_min=("chi2_red_nc", "min"), chi2_max=("chi2_red_nc", "max"), pin_max=("dchi2_nc", "max"),
                r_first=("ratio_cr_las", "first"), r_last=("ratio_cr_las", "last")).round(3).to_string())
    a = act[(act.N_line == "combined")].groupby(["dataset", "variant"], sort=False)
    print(a.agg(nN_min=("nN_nAr", "min"), nN_max=("nN_nAr", "max"), diss_min=("dissociation", "min"),
                diss_max=("dissociation", "max"), nN_abs_min=("n_N", "min"), nN_abs_max=("n_N", "max"),
                err_med=("ln_err_stat", "median"), birge_med=("birge_pairs", "median")).to_string(
        float_format=lambda v: f"{v:.3g}"))


def main(input_file=None):
    for d in (FIGDIR, TABDIR):
        os.makedirs(d, exist_ok=True)
    copy_figures()
    fits = pd.read_csv(os.path.join(S.OUTDIR, "fits_all.csv"))
    act = pd.read_csv(os.path.join(S.OUTDIR, "actinometry_per_condition_all.csv"))
    sets = {d["name"]: d for d in S.load_input(input_file or S.INPUT_FILE)}
    las = L.load_las()
    trend = L.fit_tg_trend(las)[0]
    conds = pd.concat([S.conditions(d, las, trend).assign(set=n) for n, d in sets.items()], ignore_index=True)
    libraries_table(conds)
    fits_table(fits)
    actinometry_table(act)
    numbers(trend, conds, fits)
    summary(fits, act)
    print("Tg trend:", {k: round(v, 3) for k, v in trend.items()})
    c = conds[conds.set == EXAMPLE_SET]
    xs = sorted(set(EXAMPLE_X) | {S_EXAMPLE_X})
    if EXAMPLE_SET not in sets or not all(np.isclose(c.x, x).any() for x in xs):
        print(f"example figures skipped: {EXAMPLE_SET} at {xs} is not in this run")
        print(f"figures and tables in {REPORT}")
        return
    ds = sets[EXAMPLE_SET]
    ex = {x: fit_one(ds, next(c[np.isclose(c.x, x)].itertuples())) for x in xs}
    plot_free_vs_nc({x: ex[x] for x in EXAMPLE_X}, os.path.join(FIGDIR, "fit_free_vs_nc.png"))
    print(plot_s(ex[S_EXAMPLE_X], S_EXAMPLE_X, os.path.join(FIGDIR, "s_scale.png")).to_string(index=False))
    plot_pin_cost(fits, {n: d["kind"] for n, d in sets.items() if n in set(fits.dataset)},
                  os.path.join(FIGDIR, "pin_cost.png"))
    print(f"figures and tables in {REPORT}")


if __name__ == "__main__":
    main()
