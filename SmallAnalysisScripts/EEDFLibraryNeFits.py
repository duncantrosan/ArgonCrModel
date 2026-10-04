# -*- coding: utf-8 -*-
"""
EEDFLibraryNeFits.py

Does the EEDF representation decide the electron density the CR fit returns? The pure-Ar
condition (N2 fraction 0 %, 1 Torr) is fitted (CRFitNeTe.fit_block) with each BOLSIG+ library
of Scripts/CreateEEDFLibraries.py, and for each the chi^2 profile over Ne (minimised over E/N)
is compared, as in CriticalDensityFit:

  microwave, microwave + e-e, microwave + e-e down to 1 Td (_lowEN),
  DC, DC + e-e, DC + e-e + superelastic, DC + e-e down to 0.3 Td (_lowEN)

With e-e collisions the EEDF depends on Ne (E/N x Ne libraries), so the bulk temperature and
Ne are no longer independent. Libraries that are not built yet are skipped (BOLSIG+ runs locally:
python Scripts/CreateEEDFLibraries.py builds the missing ones).

Also eedf_shapes.png: the EEDFs of the microwave and DC fits of the same spectrum (pure Ar, and
2.4 % N2 at 85 W from ActinometryNitrogenContent), and Te_eff = 2/3 <E> along each library -
Te_eff mostly measures the slow electrons, which neither the line excitation (> 11.5 eV) nor
the stepwise excitation out of the 1s levels (> ~1.5 eV) uses.

Output: Experimental_Data/Output/EEDFLibraryNe/ (summary.csv, chi2_profiles.png, eedf_shapes.png).
The CR tables are cached with those of EEDFPhysicsFitComparison (~10 min per E/N x Ne library
on the first run; N_WORKERS run in parallel). Run from Spyder (F5) or python.
"""
import contextlib
import io
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT_DIR, "Experimental_Data", "ExperimentalDataAnalysis"))
sys.path.insert(0, os.path.join(ROOT_DIR, "Scripts"))
sys.path.insert(0, HERE)
import CRFitNeTe as crf                             # noqa: E402

he = crf._helpers()

# ---- settings -------------------------------------------------------------------
LIBRARIES = [   # label, BOLSIG+ library, colour, line style
    ("microwave", "Ar_Biagi_bolsig_mw", "#2a78d6", "-"),
    ("microwave + e-e", "Ar_Biagi_bolsig_mw_ee", "#2a78d6", "--"),
    ("microwave + e-e, low E/N", "Ar_Biagi_bolsig_mw_ee_lowEN", "#2a78d6", "-."),
    ("DC", "Ar_Biagi_bolsig", "#eb6834", "-"),
    ("DC + e-e", "Ar_Biagi_bolsig_ee", "#eb6834", "--"),
    ("DC + e-e + superelastic", "Ar_Biagi_bolsig_ee_se", "#eb6834", ":"),
    ("DC + e-e, low E/N", "Ar_Biagi_bolsig_ee_lowEN", "#eb6834", "-."),
]
SWEEP, X = "N2 fraction", 0.0                       # pure Ar, 85 W, 1 Torr
NE_CHECK = (1e17, 3e17, 1e18)                       # m^-3, chi^2 profile reported at these Ne
MIXTURE_FIT = ("2.4 % N2, 85 W", "Power", 85.0, 2.4)  # for eedf_shapes.png (ActinometryNitrogenContent fits)
N_WORKERS = 4
OUTDIR = os.path.join(ROOT_DIR, "Experimental_Data", "Output", "EEDFLibraryNe")


def config(lib):
    """Fit settings for one library: EEDFPhysics settings and table folder; an E/N x Ne
    library brings its own Ne grid."""
    import CreateEEDFLibraries as libs
    cfg = dict(crf.CONFIG, Ne_grid=libs.NE_GRID, eedf=str(he.BOLSIG_FOLDER / lib))
    cfg["outdir"] = crf.output_dir("EEDFPhysics", cfg)
    info = he.ImportBolsigLibrary(cfg["eedf"], verbose=False)
    if info["kind"] == "2D":
        cfg["Ne_grid"] = np.asarray(info["Ne"], float)
    return cfg


def fit(lib):
    """Free and Ne = n_c fit of the pure-Ar condition with one library."""
    with contextlib.redirect_stdout(io.StringIO()):
        import CriticalDensityFit as cdf
        cfg = config(lib)
        tab = crf.build_model_table(cfg)
        feats, comps = crf.select_features(tab, cfg)
        ft = crf.feature_table(crf.measure(comps, cfg), feats, cfg)
        rs = crf.repeat_scatter(ft)
        if cfg["use_repeat_scatter"]:
            ft["rel_err"] = np.sqrt(ft.rel_err ** 2 + ft.feature.map(rs) ** 2)
        grid = crf.fine_grid(tab, cfg)
        Te_f = crf.te_eff_fine(tab, grid)
        d = ft.query("sweep == @SWEEP and x == @X")
        s, post, _ = crf.fit_block(d, crf.feature_model(tab, feats, cfg), feats, grid, Te_f, cfg)
        r, pin, (Ne, prof) = cdf.analyse(dict(tab=tab, grid=grid, Te_f=Te_f, cfg=cfg), None,
                                         s["chi2_min"], s["dof"], s["birge"], post)
    lev = list(tab["levels"])
    ln1s5 = np.log(tab["dens"][:, :, lev.index("4s1")])
    n1s5 = lambda w: float(np.exp(crf.posterior_mean(tab, w, ln1s5, cfg)))
    out = dict(library=lib, kind="E/N x Ne" if np.ndim(tab["Te_eff"]) == 2 else "E/N",
               EN_min=float(tab["x_grid"][0]), Ne_min=float(tab["Ne_grid"][0]), Ne_max=float(tab["Ne_grid"][-1]),
               EN_med=float(s["x_med"]), Te_med=float(s["Te_med"]), Ne_med=float(s["Ne_med"]), Ne_lo=float(s["Ne_lo"]),
               Ne_hi=float(s["Ne_hi"]), edge_EN=bool(s["edge_x"]), edge_Ne=bool(s["edge_Ne"]),
               chi2_red=float(s["chi2_red"]), dchi2_nc=float(r["delta_chi2_at_nc"]),
               chi2_red_nc=float(r["chi2_red_pinned"]), EN_nc=float(r["x_pinned"]), Te_nc=float(r["Te_pinned"]),
               n1s5_free=n1s5(post), n1s5_nc=n1s5(pin))
    for n in NE_CHECK:
        out[f"dchi2_{n:.0e}"] = float(np.interp(np.log(n), np.log(Ne), prof))
    return out, (np.asarray(Ne), np.asarray(prof))


# ---- EEDF shapes ----------------------------------------------------------------
def eedf_at(lib, EN, Ne=None):
    """EEPF of a library at E/N (interpolated in ln E/N; for an E/N x Ne library the Ne column
    nearest to Ne), normalised to int sqrt(E) f dE = 1."""
    info = he.ImportBolsigLibrary(he.BOLSIG_FOLDER / lib, verbose=False)
    x = np.asarray(info["EN_Td"], float)
    rows = info["EEDFs"]
    if info["kind"] == "2D":
        j = int(np.argmin(np.abs(np.log(np.asarray(info["Ne"], float) / Ne))))
        rows = [r[j] for r in rows]
    i = int(np.clip(np.searchsorted(x, EN), 1, len(x) - 1))
    w = float(np.clip((np.log(EN) - np.log(x[i - 1])) / (np.log(x[i]) - np.log(x[i - 1])), 0, 1))
    E = np.linspace(0.01, 20, 2000)
    lnf = lambda e: np.log(np.maximum(np.interp(E, e["E"], e["EEPF"], right=0.0), 1e-300))
    f = np.exp((1 - w) * lnf(rows[i - 1]) + w * lnf(rows[i]))
    return E, f / np.trapezoid(f * np.sqrt(E), E)


def slope_T(E, f, lo, hi):
    """Slope temperature -1/(d ln f/dE) over [lo, hi] eV; inf for a flat or rising EEPF."""
    m = (E >= lo) & (E <= hi) & (f > 1e-250)
    s = np.polyfit(E[m], np.log(f[m]), 1)[0]
    return -1 / s if s < 0 else np.inf


def mixture_fits():
    """E/N, Te_eff and Ne of the microwave and DC ActinometryNitrogenContent fits of MIXTURE_FIT."""
    label, sweep, x, pct = MIXTURE_FIT
    out = {}
    for eedf, lib in (("microwave", f"ArN2_{pct:g}pct_bolsig_mw"), ("DC", f"ArN2_{pct:g}pct_bolsig")):
        path = os.path.join(crf.output_dir("ActinometryN2Content"), f"fit_{eedf.lower()}_{pct:g}pct_N2",
                            "fit_conditions.csv")
        if not os.path.isfile(path):
            return None
        d = pd.read_csv(path)
        r = d[(d.sweep == sweep) & (d.x == x)].iloc[0]
        out[eedf] = dict(lib=lib, EN=float(r.x_med), Te=float(r.Te_med), Ne=float(r.Ne_med))
    return out


def plot_shapes(res, path):
    pure = {e: res.set_index("library").loc[lib] for e, lib in (("microwave", "Ar_Biagi_bolsig_mw"),
                                                                ("DC", "Ar_Biagi_bolsig"))
            if lib in set(res.library)}
    panels = [("pure Ar", {e: dict(lib=r.name, EN=r.EN_med, Te=r.Te_med, Ne=r.Ne_med) for e, r in pure.items()})]
    mix = mixture_fits()
    if mix:
        panels.append((MIXTURE_FIT[0], mix))
    col = {"microwave": "#2a78d6", "DC": "#eb6834"}
    fig, axs = plt.subplots(1, len(panels) + 1, figsize=(6.3 * (len(panels) + 1), 5.6))
    for ax, (title, fits) in zip(axs, panels):
        for lo, hi, text in ((1, 4, "1s → 2p stepwise,\n2p ↔ 2p mixing"), (11.55, 15, "excitation from\nthe ground state")):
            ax.axvspan(lo, hi, color="0.93", lw=0)
            ax.text((lo + hi) / 2, 0.97, text, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=9, color="0.35")
        for eedf, F in fits.items():
            E, f = eedf_at(F["lib"], F["EN"])
            Tb, Tt = slope_T(E, f, 1, 4), slope_T(E, f, 12, 15)
            tb = f"{Tb:.2f} eV" if Tb < 5 else "flat"
            ax.semilogy(E, f, color=col[eedf], lw=2,
                        label=f"{eedf} fit: {F['EN']:.3g} Td, Te_eff = {F['Te']:.2f} eV, Ne = {F['Ne']:.1e}\n"
                              f"   slope temperature {tb} (1-4 eV), {Tt:.2f} eV (12-15 eV)")
        Te = fits["microwave"]["Te"] if "microwave" in fits else None
        if Te:
            ax.semilogy(E, 2 / np.sqrt(np.pi) * Te ** -1.5 * np.exp(-E / Te), "--", color="0.45", lw=1.4,
                        label=f"Maxwellian at the microwave Te_eff ({Te:.2f} eV)")
        ax.set_ylim(1e-14, 10)
        ax.set_xlim(0, 18)
        ax.set_xlabel("electron energy [eV]")
        ax.set_ylabel("EEPF [eV$^{-3/2}$]")
        ax.set_title(f"{title}: the EEDFs of the microwave and the DC fit of the same spectrum", fontsize=10.5)
        ax.legend(fontsize=7.5, loc="lower left", frameon=False)
        ax.grid(alpha=0.3)
    ax = axs[-1]
    for lib, colr, ls, lab in (("Ar_Biagi_bolsig_mw", col["microwave"], "-", "pure Ar, microwave"),
                               (f"ArN2_{MIXTURE_FIT[3]:g}pct_bolsig_mw", col["microwave"], ":", "Ar/N$_2$, microwave"),
                               ("Ar_Biagi_bolsig", col["DC"], "-", "pure Ar, DC"),
                               (f"ArN2_{MIXTURE_FIT[3]:g}pct_bolsig", col["DC"], ":", "Ar/N$_2$, DC")):
        if not he.IsBolsigLibrary(he.BOLSIG_FOLDER / lib):
            continue
        info = he.ImportBolsigLibrary(he.BOLSIG_FOLDER / lib, verbose=False)
        keep = [(x, e["Te_eff"]) for x, e in zip(info["EN_Td"], info["EEDFs"]) if e["E"][e["EEPF"] > 0].max() > 12]
        ax.semilogx(*zip(*keep), ls, color=colr, lw=2, marker="o", ms=4, label=lab)
    ax.set_xlabel("E/N of the BOLSIG+ library [Td] (rows with electrons above 12 eV)")
    ax.set_ylabel("Te_eff = 2/3 <E> [eV]")
    ax.set_title("Te_eff along each EEDF library", fontsize=10.5)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_profiles(res, profiles, path):
    from scipy import constants
    n_c = constants.epsilon_0 * constants.m_e * (2 * np.pi * 2.42e9) ** 2 / constants.e ** 2
    style = {lib: (c, ls, lab) for lab, lib, c, ls in LIBRARIES}
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.axvspan(1e12, n_c, color="0.93", lw=0)
    ax.axvline(n_c, color="k", lw=1.2)
    for lev, ls in ((1, ":"), (4, "--")):
        ax.axhline(lev, color="0.5", lw=0.8, ls=ls)
    for r in res.itertuples():
        c, ls, lab = style[r.library]
        Ne, prof = profiles[r.library]
        ax.semilogx(Ne, prof, ls, color=c, lw=2,
                    label=f"{lab}: Ne = {r.Ne_med:.1e}{' (grid edge)' if r.edge_Ne else ''}, chi2/dof {r.chi2_red:.2f}, "
                          f"1s5 = {r.n1s5_free:.0e}")
    ax.set_xlim(1e15, 3e18)
    ax.set_ylim(0, 60)
    ax.set_xlabel("$N_e$ [m$^{-3}$] (the E/N x Ne libraries end at 3e18)")
    ax.set_ylabel(r"$\Delta\chi^2/s^2$ (profiled over E/N)")
    ax.set_title("Pure Ar, 1 Torr: chi$^2$ vs $N_e$ with each BOLSIG+ EEDF library (absorption: 1s5 = 1-8e17 m$^{-3}$)",
                 fontsize=10.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    libs = [lib for _, lib, _, _ in LIBRARIES if he.IsBolsigLibrary(he.BOLSIG_FOLDER / lib)]
    missing = [lib for _, lib, _, _ in LIBRARIES if lib not in libs]
    if missing:
        print("not built yet (python Scripts/CreateEEDFLibraries.py):", ", ".join(missing))
    if N_WORKERS > 1:
        with Pool(min(N_WORKERS, len(libs))) as p:
            out = p.map(fit, libs)
    else:
        out = [fit(lib) for lib in libs]
    res = pd.DataFrame([o for o, _ in out])
    profiles = {o["library"]: prof for o, prof in out}
    res.to_csv(os.path.join(OUTDIR, "summary.csv"), index=False)
    plot_profiles(res, profiles, os.path.join(OUTDIR, "chi2_profiles.png"))
    plot_shapes(res, os.path.join(OUTDIR, "eedf_shapes.png"))
    cols = ["library", "kind", "EN_min", "EN_med", "edge_EN", "Te_med", "Ne_med", "edge_Ne", "chi2_red", "dchi2_nc"] \
        + [f"dchi2_{n:.0e}" for n in NE_CHECK] + ["n1s5_free", "n1s5_nc"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(res[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nmeasured 1s5 (absorption): 1-8e17 m^-3. Figures and table in {OUTDIR}")
    return res


if __name__ == "__main__":
    RES = main()
