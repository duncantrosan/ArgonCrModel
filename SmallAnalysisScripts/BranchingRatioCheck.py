# -*- coding: utf-8 -*-
"""
BranchingRatioCheck.py

Same-upper-level check of the calibrated line intensities of the pure-Ar pressure sweep (80 W,
600-1500 mTorr): for lines from one upper level, I lambda / (A eta) must be equal (photon rate per
upper-state atom), whatever populates the level. eta = escape factor of the line in the CR model
(current model, DC EEDF) at the fit point, Ne = n_c or the free fit; 'thin' = eta 1.
ln of each line's value relative to the mean of its group shows a calibration (or trapping) error of
that line, independent of the CR excitation physics.

Groups (lines in the echelle data, i.e. outside the order gaps):
  2p1: 750.39, 667.73     2p2: 727.29, 772.42, 826.45     2p4: 794.82, 852.14, 747.12
  2p6: 800.62, 763.51
794.82 and 852.14 nm sit 2.8 and 3.4 nm after an order gap, 763.51 nm 2.9 nm before one.

Output: Experimental_Data/Output/LiteratureComparison/branching_check.png, branching_check.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import PressureSweepAnalysis as ps                   # noqa: E402

crf, he = ps.crf, ps.he
GROUPS = {"2p1": [750.387, 667.728], "2p2": [727.294, 772.421, 826.452], "2p4": [794.818, 852.144, 747.117],
          "2p6": [800.616, 763.511]}
EDGE = {794.818: "2.8 nm after a gap", 852.144: "3.4 nm after a gap", 763.511: "2.9 nm before a gap"}
FAMILY = "dc"
OUTDIR = os.path.join(crf.ROOT_DIR, "Experimental_Data", "Output", "LiteratureComparison")


def measure_lines():
    cfg = ps.base_cfg()
    cat, _ = crf.als._catalogue()
    ar = cat[cat.species == "Ar I"].wl_air.to_numpy()
    lines = sorted({float(ar[np.argmin(np.abs(ar - w))]) for ws in GROUPS.values() for w in ws})   # catalogue values
    mcfg = dict(crf.als.CONFIG, sweeps=[ps.SWEEP], window_nm=cfg["window_nm"], reuse_measurements=True,
                outdir=cfg["measure_dir"], measure_cache="branching_measurements.csv", export_flags=cfg["export_flags"])
    m = crf.als.filter_export_flags(crf.als.measure_sweeps(pd.DataFrame(dict(species="Ar I", wl_air=lines)), mcfg), mcfg)
    return m[m.status.isin(("ok", "weak", "blend"))]


def main():
    m = measure_lines()
    feats, ft = ps.measurements(ps.pressures())
    rows = []
    for p in ps.pressures():
        FIT, row, pin, _ = ps.fit_condition(FAMILY, p, feats, ft)
        tab = FIT["tab"]
        lx, ln = np.log(tab["x_grid"]), np.log(tab["Ne_grid"])
        for mode, c in (("free", FIT["cond"].iloc[0]), ("nc", pin)):
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
    S.to_csv(os.path.join(OUTDIR, "branching_check.csv"), index=False)
    fig, axs = plt.subplots(1, len(GROUPS), figsize=(4.2 * len(GROUPS), 4.6), sharey=True)
    for ax, (up, wls) in zip(axs, GROUPS.items()):
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
    return S


if __name__ == "__main__":
    S = main()
