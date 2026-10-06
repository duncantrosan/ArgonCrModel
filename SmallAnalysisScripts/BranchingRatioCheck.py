# -*- coding: utf-8 -*-
"""
BranchingRatioCheck.py

Same-upper-level check of the calibrated line intensities of the pure-Ar pressure sweep (80 W,
600-1500 mTorr): for lines from one upper level, I lambda / (A eta) must be equal (photon rate per
upper-state atom), whatever populates the level. eta = escape factor of the line in the CR model
(current model, DC EEDF) at the fit point, Ne = n_c or the free fit; 'thin' = eta 1.
ln of each line's value relative to the mean of its group shows a calibration (or trapping) error of
that line, independent of the CR excitation physics.

Groups: every model line (< 870 nm) of the level outside the echelle order gaps:
  2p1: 750.39, 667.73     2p2: 727.29, 772.42, 826.45     2p4: 794.82, 852.14, 747.12
The other lines of these levels and of 2p3, 2p6, 2p7, 2p8 are in order gaps in every spectrum
(696.54, 714.70, 738.40, 763.51, 810.37, 840.82, 842.47, 866.79 nm) or beyond the data (922.45 nm),
so 706.72 (2p3), 800.62 (2p6) and 801.48 nm (2p8) cannot be checked this way.
The distance of each line to the nearest order gap is taken from one spectrum (gap_distance).

772.38 (2p7 -> 1s5) and 772.42 nm (2p2 -> 1s3) are 0.045 nm apart, about one instrumental FWHM; the
line fit splits them with fixed centres. doublet_check compares that split with the CR model ratio.

Output: Experimental_Data/Output/LiteratureComparison/branching_check.png, branching_check.csv,
        branching_doublet_772.csv
"""
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import PressureSweepAnalysis as ps                   # noqa: E402

crf, he = ps.crf, ps.he
GROUPS = {"2p1": [750.387, 667.728], "2p2": [727.294, 772.421, 826.452], "2p4": [794.818, 852.144, 747.117],
          "2p7": [772.376]}                         # 2p7: doublet partner only (not plotted)
DOUBLET = {"2p7": 772.376, "2p2": 772.421}          # one feature at the instrumental resolution
FAMILY = "dc"
OUTDIR = os.path.join(crf.ROOT_DIR, "Experimental_Data", "Output", "LiteratureComparison")


def gap_distance(wls):
    """{wl: text} distance of each line to the nearest order gap (or data edge), from one sweep spectrum."""
    f = sorted(glob(os.path.join(ps.SWEEP["folder"], "1000mTorr_*.spa")))[0]
    sp = crf.als.arn.load_spectrum(f, "gaps", crf.als.arn.CONFIG)
    edges = [(a, "before") for a, _ in sp.gaps] + [(b, "after") for _, b in sp.gaps]
    out = {}
    for w in wls:
        e, side = min(edges, key=lambda t: abs(t[0] - w))
        out[w] = f"{abs(w - e):.1f} nm {'after' if w > e else 'before'} a gap"
    return out


def measure_lines():
    cfg = ps.base_cfg()
    cat, _ = crf.als._catalogue()
    ar = cat[cat.species == "Ar I"].wl_air.to_numpy()
    lines = sorted({float(ar[np.argmin(np.abs(ar - w))]) for ws in GROUPS.values() for w in ws})   # catalogue values
    mcfg = dict(crf.als.CONFIG, sweeps=[ps.SWEEP], window_nm=cfg["window_nm"], reuse_measurements=True,
                outdir=cfg["measure_dir"], measure_cache="branching_measurements.csv", export_flags=cfg["export_flags"])
    m = crf.als.filter_export_flags(crf.als.measure_sweeps(pd.DataFrame(dict(species="Ar I", wl_air=lines)), mcfg), mcfg)
    return m[m.status.isin(("ok", "weak", "blend"))]


def doublet_check(R, model_ratio):
    """772.38 / 772.42 nm as split by the line fit, per spectrum, against the CR model ratio at the same
    fit point (model_ratio: {(p, mode): photon-rate ratio}) -> ln(measured / model).  The other lines of
    2p7 (810.37, 866.79 nm) lie in order gaps, so the split can only be compared with the model."""
    a, b = DOUBLET["2p7"], DOUBLET["2p2"]
    rows = []
    for (p, mode, f), g in R.groupby(["p_mTorr", "mode", "file"]):
        ia, ib = g[np.abs(g.wl - a) < 1e-3], g[np.abs(g.wl - b) < 1e-3]
        if ia.empty or ib.empty:
            continue
        meas = np.exp(ia.thin.iloc[0] - ib.thin.iloc[0]) * ia.A.iloc[0] / ib.A.iloc[0]   # photon-rate ratio
        rows.append(dict(p_mTorr=p, mode=mode, file=f, ratio_meas=meas, ratio_model=model_ratio[(p, mode)],
                         ln_meas_over_model=np.log(meas / model_ratio[(p, mode)])))
    return pd.DataFrame(rows)


def main():
    m = measure_lines()
    feats, ft = ps.measurements(ps.pressures())
    rows, model_ratio = [], {}
    for p in ps.pressures():
        FIT, row, pin, _ = ps.fit_condition(FAMILY, p, feats, ft)
        tab = FIT["tab"]
        lx, ln = np.log(tab["x_grid"]), np.log(tab["Ne_grid"])
        ka, kb = (int(np.argmin(np.abs(tab["line_wl"] - DOUBLET[u]))) for u in ("2p7", "2p2"))
        f_ratio = RectBivariateSpline(lx, ln, np.log(tab["I_obs"][:, :, ka] / tab["I_obs"][:, :, kb]), kx=1, ky=1)
        for mode, c in (("free", FIT["cond"].iloc[0]), ("nc", pin)):
            model_ratio[(p, mode)] = float(np.exp(f_ratio(np.log(c.x_best), np.log(c.Ne_best))[0, 0]))
            for up, wls in GROUPS.items():
                for wl in wls:
                    k = int(np.argmin(np.abs(tab["line_wl"] - wl)))
                    if abs(tab["line_wl"][k] - wl) > 0.05:
                        continue
                    f = RectBivariateSpline(lx, ln, np.log(tab["I_obs"][:, :, k] / tab["I_thin"][:, :, k]), kx=1, ky=1)
                    eta = float(np.exp(f(np.log(c.x_best), np.log(c.Ne_best))[0, 0]))
                    d = m[(m.x == p) & (np.abs(m.wl_air - wl) < 0.01)]
                    for r in d.itertuples():
                        rows.append(dict(p_mTorr=p, mode=mode, upper=up, wl=wl, file=r.file, eta=eta, A=tab["line_A"][k],
                                         thin=np.log(r.area * wl / tab["line_A"][k]),
                                         val=np.log(r.area * wl / (tab["line_A"][k] * eta))))
    R = pd.DataFrame(rows)
    for col in ("val", "thin"):
        R[f"{col}_rel"] = R[col] - R.groupby(["p_mTorr", "mode", "upper", "file"])[col].transform("mean")
    S = R.groupby(["mode", "upper", "wl"]).agg(ln_rel=("val_rel", "mean"), ln_rel_sd=("val_rel", "std"),
                                               ln_rel_thin=("thin_rel", "mean"), eta=("eta", "median"),
                                               n=("val_rel", "size")).reset_index()
    EDGE = gap_distance(sorted({w for ws in GROUPS.values() for w in ws}))
    S["gap"] = S.wl.map(EDGE)
    S.to_csv(os.path.join(OUTDIR, "branching_check.csv"), index=False)
    Dd = doublet_check(R, model_ratio)
    Dd.to_csv(os.path.join(OUTDIR, "branching_doublet_772.csv"), index=False)
    S = S[S.upper.map(lambda u: len(GROUPS[u]) > 1)]
    plot_groups = {u: w for u, w in GROUPS.items() if len(w) > 1}
    fig, axs = plt.subplots(1, len(plot_groups), figsize=(4.2 * len(plot_groups), 4.6), sharey=True)
    for ax, (up, wls) in zip(axs, plot_groups.items()):
        for k, (mode, mk, col) in enumerate((("nc", "D", "#1baf7a"), ("free", "s", "#8a3ffc"))):
            g = S[(S["mode"] == mode) & (S.upper == up)].set_index("wl").reindex(wls)
            x = np.arange(len(wls)) + (k - 0.5) * 0.2
            ax.errorbar(x, g.ln_rel, yerr=g.ln_rel_sd, fmt=mk, color=col, capsize=3, ms=7,
                        label=f"eta of the model at {'n$_c$' if mode == 'nc' else 'the free fit'}")
            if mode == "nc":
                ax.plot(x, g.ln_rel_thin, "_", color="0.4", ms=14, mew=2, label="eta = 1")
        ax.axhline(0, color="0.5", lw=1)
        ax.set_xticks(np.arange(len(wls)), [f"{w:.1f}\n{EDGE.get(w, '')}" for w in wls], fontsize=8)
        ax.set_title(f"upper level {up}", fontsize=10)
        ax.grid(axis="y", color="0.9")
    axs[0].set_ylabel(r"ln($I\lambda/(A\eta)$) - group mean")
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Same-upper-level check, pure-Ar pressure sweep (all pressures and repeats): "
                 "lines from one level must agree", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(OUTDIR, "branching_check.png"), dpi=140)
    plt.close(fig)
    with pd.option_context("display.width", 200):
        print(S.round(3).to_string(index=False))
        print("\n772.38 / 772.42 nm photon-rate ratio from the line fit vs the CR model at the fit point:")
        print(Dd.groupby("mode")[["ratio_meas", "ratio_model", "ln_meas_over_model"]]
              .agg(["mean", "std"]).round(3).to_string())
        print(Dd.groupby(["mode", "p_mTorr"]).ln_meas_over_model.mean().unstack(0).round(2).to_string())
    return S, Dd


if __name__ == "__main__":
    S, DOUBLET_772 = main()
