# -*- coding: utf-8 -*-
"""
Measured2pVsZhuPu.py

Our measured 2p distribution next to the one Zhu & Pu measured in their 100 Pa CCP (their Fig. 2c, OES)
and to our CR model, for the pure-Ar pressure sweep (80 W, Tg 800 K) at P_SHOW pressures.

n/g of the seven 2p levels we fit (2p1 750.4, 2p2 727.3, 2p3 706.7, 2p4 794.8, 2p5 751.5, 2p6 800.6,
2p8 801.5 nm) from the calibrated line areas: n/g ~ I lambda / (A g eta), mean over the repeat spectra,
normalised to a sum of 100 over the seven levels. eta = escape factor of the line in the CR model at the
fit point (Ne = n_c, DC EEDF; the 'eta = 1' variant is the optically thin bound). Their CCP OES and CRM
values are renormalised over the same seven levels.

Output: Experimental_Data/Output/LiteratureComparison/measured_2p_vs_zhupu.png, measured_2p_vs_zhupu.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import PressureSweepAnalysis as ps                   # noqa: E402

crf, he = ps.crf, ps.he
P_SHOW = [600.0, 1000.0, 1500.0]
FAMILY = "dc"
LEVELS = {"2p1": 750.387, "2p5": 751.465, "2p3": 706.722, "2p4": 794.818, "2p2": 727.294, "2p6": 800.616,
          "2p8": 801.479}
ORDER = ["2p1", "2p5", "2p3", "2p8", "2p4", "2p2", "2p6"]
OUTDIR = os.path.join(crf.ROOT_DIR, "Experimental_Data", "Output", "LiteratureComparison")


def main():
    Z = pd.read_csv(os.path.join(crf.ROOT_DIR, "InputData", "References", "ZhuPu2010_digitized.csv"), comment="#")
    zz = {(r.source, r.level): r.value for r in Z[(Z.figure == 2) & (Z.discharge == "CCP 100 Pa")].itertuples()}
    norm = lambda v: 100 * np.asarray(v, float) / np.nansum(v)
    ps_list = ps.pressures()
    feats, ft = ps.measurements(ps_list)
    rows = []
    for p in P_SHOW:
        FIT, row, pin, _ = ps.fit_condition(FAMILY, p, feats, ft)
        tab, cfg = FIT["tab"], FIT["cfg"]
        lev = list(tab["levels"])
        lx, ln = np.log(tab["x_grid"]), np.log(tab["Ne_grid"])
        from scipy.interpolate import RectBivariateSpline
        for mode, c in (("free", FIT["cond"].iloc[0]), ("nc", pin)):
            for name, wl in LEVELS.items():
                k = int(np.argmin(np.abs(tab["line_wl"] - wl)))
                up = tab["line_upper"][k]
                f_eta = RectBivariateSpline(lx, ln, np.log(tab["I_obs"][:, :, k] / tab["I_thin"][:, :, k]), kx=1, ky=1)
                f_n = RectBivariateSpline(lx, ln, np.log(tab["dens"][:, :, lev.index(up)]), kx=1, ky=1)
                eta = float(np.exp(f_eta(np.log(c.x_best), np.log(c.Ne_best))[0, 0]))
                n_mod = float(np.exp(f_n(np.log(c.x_best), np.log(c.Ne_best))[0, 0]))
                feat = [f for f in feats.feature if f.startswith(f"{wl:.2f}")]
                d = ft[(ft.x == p) & ft.feature.isin(feat)]
                area = d.area.mean()
                A = tab["line_A"][k]
                gl = {"2p1": 1, "2p2": 3, "2p3": 5, "2p4": 3, "2p5": 1, "2p6": 5, "2p8": 5}[name]
                rows.append(dict(p_mTorr=p, mode=mode, level=name, wl=wl, eta=eta, meas_thin=area * wl / (A * gl),
                                 meas=area * wl / (A * gl * eta), model=n_mod / gl))
    R = pd.DataFrame(rows)
    out = []
    fig, axs = plt.subplots(1, len(P_SHOW), figsize=(6 * len(P_SHOW), 5), sharey=True, squeeze=False)
    x = np.arange(len(ORDER))
    for ax, p in zip(axs[0], P_SHOW):
        oes = norm([zz.get(("OES", l), np.nan) for l in ORDER])
        crm = norm([zz.get(("CRM", l), np.nan) for l in ORDER])
        ax.bar(x, oes, width=0.62, color="0.82", label="Zhu & Pu CCP 100 Pa, OES")
        ax.plot(x, crm, "_", ms=22, mew=2, color="0.45", label="Zhu & Pu CCP 100 Pa, CRM")
        for mode, mk, col in (("nc", "D", "#1baf7a"), ("free", "s", "#8a3ffc")):
            g = R[(R.p_mTorr == p) & (R["mode"] == mode)].set_index("level").loc[ORDER]
            meas, thin, mod = norm(g.meas), norm(g.meas_thin), norm(g.model)
            if mode == "nc":
                ax.vlines(x - 0.15, np.minimum(meas, thin), np.maximum(meas, thin), color=col, lw=6, alpha=0.3)
                ax.plot(x - 0.15, meas, mk, color=col, ms=8, label="measured (eta of the model at n$_c$; band: eta = 1)")
            ax.plot(x + 0.15, mod, "o", color=col, mfc="white" if mode == "free" else col, ms=7, mew=1.6,
                    label=f"our CR model, {'Ne = n$_c$' if mode == 'nc' else 'free fit'} ({FAMILY} EEDF)")
            for l, a, b, c in zip(ORDER, meas, thin, mod):
                out.append(dict(p_mTorr=p, mode=mode, level=l, measured=a, measured_eta1=b, model=c))
        ax.set_xticks(x, [f"2p$_{{{l[2:]}}}$\n{LEVELS[l]:.1f}" for l in ORDER], fontsize=8)
        ax.set_title(f"pure Ar {p:g} mTorr, 80 W", fontsize=10)
        ax.grid(axis="y", color="0.9")
    axs[0, 0].set_ylabel("$n/g$ (sum of the seven levels = 100)")
    axs[0, 0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Measured 2p distribution (seven fitted levels) vs Zhu & Pu's 100 Pa CCP and our CR model "
                 "(primed core: 2p1-2p4)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUTDIR, "measured_2p_vs_zhupu.png"), dpi=140)
    plt.close(fig)
    O = pd.DataFrame(out)
    O.to_csv(os.path.join(OUTDIR, "measured_2p_vs_zhupu.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(O.pivot_table(index=["p_mTorr", "level"], columns="mode", values=["measured", "model"]).round(1).to_string())
        print("Zhu & Pu CCP OES (7 levels, sum 100):", dict(zip(ORDER, norm([zz.get(('OES', l)) for l in ORDER]).round(1))))
    return O


if __name__ == "__main__":
    O = main()
