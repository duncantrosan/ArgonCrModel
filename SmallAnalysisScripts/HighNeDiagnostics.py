# -*- coding: utf-8 -*-
"""
HighNeDiagnostics.py

Why does the CR fit put Ne at ~1e19 m^-3 (BSR 4s cross sections)? For pure Ar (85 W) and
Ar + 2.4 % N2 (85 W), microwave and DC EEDF, the chi^2 profile over Ne (profiled over E/N) and the
fit results for

  model variants   baseline (CRFitNeTe.CONFIG), no trapping of the 4p -> 4s lines (escape factor
                   only on lines to the ground state), Tg = 700 K, R = 1 cm, and the ground-state
                   excitation from BSR (he.XSEC_GROUND)
  line subsets     the baseline fitted without the primed-core 2p lines (2p1-2p4: 750.4, 727.3,
                   706.7, 794.8 nm), without 750.4 nm (2p1) alone, without 2p2-2p4 (keeping
                   750.4 nm), without the 5p lines, without 800.6 / 801.5 nm

Each variant has its own CR tables (Output/HighNeDiagnostics/<variant>/; ~1 min each, 20 in all
on the first run). Results: high_ne_diagnostics.png and .csv in Output/HighNeDiagnostics/.
Run from Spyder: F5.
"""
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT_DIR, "Experimental_Data", "ExperimentalDataAnalysis"))
import ActinometryNitrogenContent as anc            # noqa: E402

crf = anc.crf

# ---- settings -------------------------------------------------------------------
CASES = [   # EEDF, % N2 of the model, sweep, condition
    ("microwave", 0.0, "N2 fraction", 0.0), ("microwave", 2.4, "Power", 85.0),
    ("dc", 0.0, "N2 fraction", 0.0), ("dc", 2.4, "Power", 85.0),
]
VARIANTS = {   # name: CRFitNeTe settings that differ from the baseline
    "baseline": {},
    "no 4p-4s trapping": dict(trap_lines="ground"),
    "Tg = 700 K": dict(Tg=700.0),
    "R = 1 cm": dict(R=0.01),
    "BSR ground state": dict(xsec_ground="BSR"),
}
PRIMED_2P = ("4p7", "4p8", "4p9", "4p10")           # 2p4, 2p3, 2p2, 2p1 (2P1/2 ion core)
SUBSETS = {    # name: feature -> keep?
    "all 9 lines": lambda f: True,
    "without primed 2p": lambda f: not any(u in PRIMED_2P for u in f.uppers.split("+")),
    "without 750.4 (2p1)": lambda f: "4p10" not in f.uppers.split("+"),
    "without 2p2-2p4": lambda f: not any(u in PRIMED_2P[:3] for u in f.uppers.split("+")),
    "without 5p": lambda f: not f.uppers.startswith("5p"),
    "without 800.6/801.5": lambda f: not 800.0 < f.wl < 802.0,
}
N_C = constants.epsilon_0 * constants.m_e * (2 * np.pi * 2.42e9) ** 2 / constants.e ** 2
MEASURED_1S5 = (1e17, 8e17)                         # m^-3, absorption estimate
OUTDIR = os.path.join(ROOT_DIR, "Experimental_Data", "Output", "HighNeDiagnostics")


def variant_config(variant, eedf, pct):
    """CRFitNeTe settings of one variant (as ActinometryNitrogenContent.fit_config) in its own folder."""
    tag = variant.replace(" ", "_").replace("=", "").replace("-", "").replace("/", "")
    return dict(anc.fit_config(pct, eedf), **VARIANTS[variant],
                outdir=os.path.join(OUTDIR, tag, f"fit_{eedf}_{pct:g}pct_N2"))


def analyse(R, sweep, x, subset):
    """Fit of one condition with a line subset -> (summary, Ne axis, chi^2 profile / b^2)."""
    feats = R["feats"]
    keep = set(feats[[SUBSETS[subset](f) for f in feats.itertuples()]].feature)
    d = R["ft"].query("sweep == @sweep and x == @x")
    d = d[d.feature.isin(keep)]
    s, post, _ = crf.fit_block(d, R["M"], feats, R["grid"], R["Te_f"], R["cfg"])
    g = crf.gls_block(d, R["M"], feats, R["cfg"])
    Ne = np.exp(R["grid"][1])
    prof = (np.min(g["chi2"], axis=0) - g["chi2"].min()) / s["birge"] ** 2
    lev = list(R["tab"]["levels"])
    n1s5 = np.exp(crf.posterior_mean(R["tab"], post, np.log(R["tab"]["dens"][:, :, lev.index("4s1")]), R["cfg"]))
    out = dict(n_lines=len(keep), Te_med=s["Te_med"], Ne_med=s["Ne_med"], Ne_lo=s["Ne_lo"], Ne_hi=s["Ne_hi"],
               chi2_red=s["chi2_red"], dchi2_nc=np.interp(np.log(N_C), np.log(Ne), prof),
               dchi2_1e18=np.interp(np.log(1e18), np.log(Ne), prof), n_1s5=n1s5)
    return out, Ne, prof


def plot(profiles, path):
    fig, axs = plt.subplots(2, len(CASES), figsize=(5.2 * len(CASES), 9), sharex=True, sharey=True, squeeze=False)
    for j, case in enumerate(CASES):
        for i, (kind, names) in enumerate((("variant", VARIANTS), ("subset", SUBSETS))):
            ax = axs[i, j]
            for k, name in enumerate(names):
                Ne, prof = profiles[case, kind, name]
                ax.semilogx(Ne, prof, lw=2.4 if k == 0 else 1.5, color="k" if k == 0 else f"C{k}", label=name)
            ax.axvline(N_C, color="k", lw=1)
            ax.axvspan(1e12, N_C, color="0.93", lw=0)
            for lev, ls in ((1, ":"), (4, "--")):
                ax.axhline(lev, color="0.5", lw=0.8, ls=ls)
            ax.set_ylim(0, 40)
            ax.set_xlim(1e13, 3e19)
            ax.grid(alpha=0.3, which="both")
            if i == 0:
                ax.set_title(f"{case[0]} EEDF, {case[2]} {case[3]:g}" + (" % N$_2$" if case[2] == "N2 fraction" else " W"),
                             fontsize=10)
            else:
                ax.set_xlabel("$N_e$ [m$^{-3}$]")
            if j == 0:
                ax.set_ylabel(("model variants" if i == 0 else "line subsets (baseline model)")
                              + "\n" + r"$\Delta\chi^2/s^2$ (profiled over E/N)")
            if j == len(CASES) - 1:
                ax.legend(fontsize=8)
    fig.suptitle("What drives the CR fit to high $N_e$? (BSR 4s cross sections; vertical line: $n_c$ at 2.42 GHz)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows, profiles = [], {}
    for case in CASES:
        eedf, pct, sweep, x = case
        for variant in VARIANTS:
            print(f"  {variant}: {eedf} EEDF, {pct:g} % N2", flush=True)
            with contextlib.redirect_stdout(io.StringIO()):
                R = crf.run(variant_config(variant, eedf, pct))
            subsets = SUBSETS if variant == "baseline" else ("all 9 lines",)
            for subset in subsets:
                s, Ne, prof = analyse(R, sweep, x, subset)
                rows.append(dict(eedf=eedf, sweep=sweep, x=x, variant=variant, subset=subset, **s))
                if subset == "all 9 lines":
                    profiles[case, "variant", variant] = (Ne, prof)
                if variant == "baseline":
                    profiles[case, "subset", subset] = (Ne, prof)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUTDIR, "high_ne_diagnostics.csv"), index=False)
    plot(profiles, os.path.join(OUTDIR, "high_ne_diagnostics.png"))
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(res.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nmeasured 1s5 (absorption): {MEASURED_1S5[0]:.0e}-{MEASURED_1S5[1]:.0e} m^-3;  n_c = {N_C:.2e} m^-3")
    print(f"figure and table in {OUTDIR}")
    return res


if __name__ == "__main__":
    RES = main()
