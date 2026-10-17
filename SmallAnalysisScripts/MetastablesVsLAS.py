# -*- coding: utf-8 -*-
"""
MetastablesVsLAS.py

The CR fits of the 80 W N2 admixture sweeps at 0.6, 1 and 1.5 Torr (fits.csv of the Main.py runs RUNS) against the
LAS measurements (LAS_FILE without the runs EXCLUDE_RUNS), one column per pressure:
  row 1  Ar(1s5): every LAS measurement at that pressure and power (points), their mean per % N2 (diamonds), and the
         CR 1s5 with its band of each run
  row 2  CR / mean LAS (log scale)
  row 3  E/N of the fit (band: Delta chi^2 <= B^2)
  row 4  Ne of the fit (band, linear), n_c dashed
  row 5  Tv of the runs with Tv from the LAS ([eedf] Tv_K = "LAS"): Tv where the CR 1s5 crosses the LAS 1s5 (bars: Tv
         range where the CR 1s5 band overlaps LAS +- its error); a condition that no Tv of the grid brings to the LAS
         value is drawn as an open triangle at the grid Tv closest to it
Conditions whose Ne is pinned at n_c (Ne_mode 'pinned ...', e.g. Ne_pin = "free_else_nc" where the free fit runs to the
grid edge) are drawn with open symbols; any condition still at the Ne grid edge (free fit on the cold, dense branch)
is marked at the bottom of row 1 instead of drawn.

Output: Experimental_Data/Output/<OUT_NAME>
Run from Spyder (F5) or python SmallAnalysisScripts/MetastablesVsLAS.py (after python Main.py with the input files
InputFiles/input_N2sweep_80W_EchelleLAS_pureAr.toml, _vibsat.toml and _TvLAS.toml)
"""
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from LASColumns import read_las

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "Experimental_Data", "Output")
LAS_FILE = os.path.join(ROOT, "Experimental_Data", "LAS_Data", "Master_AllMeasurements.csv")
EXCLUDE_RUNS = ("707ea594",)                         # LAS runs left out (707ea594: bad first run, redone as bc16bda5)
RUNS = {"pure-Ar EEDF": ("N2Sweep80W_EchelleLAS_pureArEEDF", "C0"),
        "saturated N$_2$(v) EEDF": ("N2Sweep80W_EchelleLAS_vibsatEEDF", "C3"),
        "N$_2$(v) at T$_v$ from LAS": ("N2Sweep80W_EchelleLAS_TvLAS", "C2")}
SETS = {0.6: "80W_0.6Torr_0-10pctN2", 1.0: "80W_1Torr_0-10pctN2", 1.5: "80W_1.5Torr_0-10pctN2"}
OUT_NAME = "N2Sweep80W_EchelleLAS_metastables_vs_LAS.png"
POWER_W, FREQ_MHZ = 80.0, (2400, 2450)
EDGE_NE = 2.9e19                                     # m^-3, top of the Ne grid
N_C = 7.26e16                                        # m^-3, critical density of 2.42 GHz


def _series(ax, x, y, lo, hi, pinned, c, lab, band=True):
    """Line through all points, band, filled markers for fitted Ne and open ones for pinned Ne."""
    if band:
        ax.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0)
    ax.plot(x, y, "-", color=c, lw=2)
    ax.plot(x[~pinned], y[~pinned], "s", color=c, ms=6, label=lab)
    if pinned.any():
        ax.plot(x[pinned], y[pinned], "s", color=c, mfc="white", ms=7, mew=1.5, label=f"{lab}, N$_e$ pinned at n$_c$")


def _tv(tx, g, c, grid):
    """Row 5: Tv from the LAS with its range; no crossing: open triangle at the closest grid Tv (pointing out of the
    grid when that is a grid end)."""
    hit = g[np.isfinite(g.Tv_LAS)]
    tx.errorbar(hit.x, hit.Tv_LAS, yerr=[hit.Tv_LAS - hit.Tv_LAS_lo, hit.Tv_LAS_hi - hit.Tv_LAS], fmt="o-", color=c,
                ms=7, capsize=3, label="T$_v$ where CR 1s$_5$ = LAS (bars: bands overlap)")
    miss = g[~np.isfinite(g.Tv_LAS) & np.isfinite(g.Tv_used)]
    for v in miss.itertuples():
        mk = "^" if np.isclose(v.Tv_used, grid.max()) else "v" if np.isclose(v.Tv_used, grid.min()) else "x"
        tx.plot(v.x, v.Tv_used, mk, color=c, mfc="white", mew=1.5, ms=9)
    if len(miss):
        tx.plot([], [], "^", color=c, mfc="white", mew=1.5, ms=9, label="no T$_v$ of the grid reaches LAS: closest")
    for v in grid:
        tx.axhline(v, color="0.85", lw=0.8, zorder=0)


def main():
    las = read_las(LAS_FILE)
    las = las[np.isclose(las.power, POWER_W) & las.frequency.between(*FREQ_MHZ) & (las.N_s_cm3 > 0)
              & ~las.run_id.astype(str).isin(EXCLUDE_RUNS)]
    fig, axs = plt.subplots(5, len(SETS), figsize=(6 * len(SETS), 18.5), sharex=True,
                            gridspec_kw=dict(height_ratios=[2, 1, 1.2, 1.2, 1.2]))
    for col, (p, name) in enumerate(SETS.items()):
        ax, rx, ex, nx, tx = axs[:, col]
        m = las[np.isclose(las.pressure, p)].assign(x=lambda d: d.Nitrogen_Percent.round(2), n=lambda d: d.N_s_cm3 * 1e6)
        mean = m.groupby("x").n.agg(["mean", "std", "size"]).reset_index()
        ax.plot(m.x, m.n, ".", color="0.6", ms=5, label="LAS, single measurements")
        ax.errorbar(mean.x, mean["mean"], yerr=mean["std"].fillna(0), fmt="D", color="k", mfc="white", ms=7, capsize=3,
                    label="LAS, mean per % N$_2$")
        top = float(np.nanmax(mean["mean"] + mean["std"].fillna(0))) if len(mean) else 0.0
        edges, grids = [], []
        for lab, (folder, c) in RUNS.items():
            f = os.path.join(OUT, folder, name, "fits.csv")
            if not os.path.exists(f):
                print(f"no {f} - run left out")
                continue
            t = pd.read_csv(f).sort_values("x")
            if "Ne_fit" not in t:                         # pinned-only runs
                t["Ne_fit"] = t["Ne_fit_lo"] = t["Ne_fit_hi"] = N_C
            mode = t["Ne_mode"] if "Ne_mode" in t else pd.Series("pinned", index=t.index)
            pin = mode.astype(str).str.startswith("pinned").to_numpy()
            edge = (t.Ne_fit >= EDGE_NE).to_numpy()
            if edge.any():
                edges.append((t.x[edge], c, lab))
            g, pg = t[~edge], pin[~edge]
            x = g.x.to_numpy()
            _series(ax, x, g.n_1s5.to_numpy(), g.n_1s5_lo, g.n_1s5_hi, pg, c, f"CR fit, {lab}")
            top = max(top, float(np.nanmax(g.n_1s5_hi)) if len(g) else 0.0)
            r = g.merge(mean, on="x")
            rp = np.isin(r.x, x[pg])
            _series(rx, r.x.to_numpy(), (r.n_1s5 / r["mean"]).to_numpy(), None, None, rp, c, lab, band=False)
            _series(ex, x, g.EN_nc.to_numpy(), g.EN_nc_lo, g.EN_nc_hi, pg, c, lab)
            _series(nx, x, g.Ne_fit.to_numpy(), g.Ne_fit_lo, g.Ne_fit_hi, pg, c, lab)
            if "Tv_LAS" in g:
                scan = os.path.join(OUT, folder, name, "tv_scan.csv")
                grid = np.unique(pd.read_csv(scan).Tv) if os.path.exists(scan) else g.Tv_used.dropna().unique()
                _tv(tx, g[g.N2 > 0], c, grid)
                grids.append(grid)
        for k, (exs, c, lab) in enumerate(edges):           # dense-branch fits (CR 1s5 ~ 0) at the bottom
            ax.plot(exs, np.full(len(exs), (0.02 + 0.03 * k) * 1.15 * top), "v", color=c, mfc="white", ms=9,
                    label=f"{lab}: free fit at the N$_e$ grid edge (CR 1s$_5$ ~ 0)")
        ax.set(ylim=(0, 1.15 * top), ylabel="Ar(1s$_5$) [m$^{-3}$]", title=f"{POWER_W:g} W, {p:g} Torr: CR fit vs LAS")
        rx.axhline(1, color="k", lw=1)
        rx.set(yscale="log", ylim=(0.05, 20), ylabel="CR / LAS (mean)")
        ex.set(ylim=(0, None), ylabel="E/N of the fit [Td]")
        nx.axhline(N_C, color="k", ls="--", lw=1, label="n$_c$ (2.42 GHz)")
        nx.set(ylim=(0, 6e17), ylabel="N$_e$ of the fit [m$^{-3}$]")
        if grids:
            gr = np.concatenate(grids)
            tx.set(yscale="log", ylim=(0.7 * gr.min(), 1.4 * gr.max()))
            tx.legend(fontsize=7, loc="best")
        tx.set(ylabel="T$_v$ of N$_2$(X) [K]\n(grey: T$_v$ grid)", xlabel="N$_2$ in the feed [%]")
        for a in axs[:, col]:
            a.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
        nx.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    path = os.path.join(OUT, OUT_NAME)
    fig.savefig(path, dpi=170)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
