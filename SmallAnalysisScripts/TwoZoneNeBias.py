# -*- coding: utf-8 -*-
"""
TwoZoneNeBias.py

Does light from the underdense plasma outside the critical layer (Ne < n_c) push the CR-fit
Ne up or down?  Synthetic two-zone spectra are fitted with the one-zone CR fit:

  side-on line of sight through an inner zone at (E/N_C, Ne_C) - the critical layer or the
  overdense core, path L_C - and an outer shell at (E/N_E, Ne_E < n_c), path L_E (both sides):
      I_line = L_C eps_C t + L_E eps_E
  eps = n_u A eta of the CR table (photon emissivity with the fit model's own trapping, a
        uniform R = 4 cm column), interpolated in ln E/N - ln Ne exactly as
        CRFitNeTe.feature_model.  t = what the near-side shell (L_E / 2) does to the inner
        zone's lines through its 4s atoms (CR lower-level densities of the shell), from a
        Doppler slab on slab: the emergent inner profile 1 - exp(-tau_C g) through
        exp(-tau g), g = exp(-x^2).  Three geometries:
  emission only     t = 1: the shell only adds its own light
  outer part of     the modelled column is inner zone + shell (chord 2R split L_C + L_E):
  the column        the shell absorbs less than the inner-zone gas the one-zone model puts
                    there, t = T(shell absorber) / T(inner-zone absorber) >= 1
  outside the       the modelled column (chord 2R) is all inner zone and the shell lies
  column            outside it (path ratio L_E / L_C as above): t = T(shell absorber) <= 1
The synthetic features replace the measured ones of a real condition (same lines, spectra,
error model and table) and crf.fit_block fits them.  Without the shell the fit returns
(E/N_C, Ne_C) exactly (checked).  With the fit's error model the Ne posterior of even a
perfect one-zone spectrum runs from ~n_c up to the 3e19 grid edge (the high-Ne branch), so
the best fit and the 16 % quantile are reported, not the median.  Scans: Ne_E / n_c, the
shell path fraction L_E / (L_C + L_E) and the shell E/N relative to the inner zone (the
microwave field is strongest outside the critical layer, so the shell can be hotter).

Cases: the DC EEDFs at 2.4 % N2 (power sweep, 85 W), which fit well at n_c, and the
extended microwave EEDFs (MicrowaveLowENFit) for pure Ar.  The inner zone sits at the E/N of
the fit with Ne pinned at n_c.

Output in Experimental_Data/Output/TwoZoneNeBias/.  Run from Spyder (F5) or
python SmallAnalysisScripts/TwoZoneNeBias.py (needs the CR tables of
ActinometryNitrogenContent and MicrowaveLowENFit).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import MicrowaveLowENFit as mlf                     # noqa: E402

anc, cdf, crf, he = mlf.anc, mlf.cdf, mlf.crf, mlf.he

# ---- settings -------------------------------------------------------------------
N_C = cdf.N_C
CASES = [   # name, fit settings, condition (sweep, x)
    ("DC EEDF, 2.4 % N2, 85 W",
     dict(anc.CONFIG["fit"], eedf=str(anc.mixture_library(2.4, "dc")), N2_percent=2.4,
          Ne_grid=anc.CONFIG["Ne_grid"], outdir=os.path.join(anc.CONFIG["outdir"], "fit_dc_2.4pct_N2"),
          measure_dir=anc.CONFIG["fit"].get("measure_dir") or anc.CONFIG["fit"]["outdir"]),
     ("Power", 85.0)),
    ("microwave EEDF (from 3 Td), pure Ar, 85 W", mlf.job_cfg("lowEN:0", build=False), ("N2 fraction", 0.0)),
]
INNER_NE = [1.0, 3.0]                               # Ne_C / n_c: the critical layer, an overdense core
EDGE_NE = np.geomspace(0.01, 1.0, 13)               # Ne_E / n_c
EDGE_FRAC = [0.25, 0.5, 0.75]                       # L_E / (L_C + L_E)
EDGE_EN = [0.7, 1.0, 1.5, 2.0]                      # (E/N)_E / (E/N)_C
CHORD_M = 2 * anc.CONFIG["fit"]["R"]                # side-on chord through the axis
OUTDIR = os.path.join(mlf.ROOT_DIR, "Experimental_Data", "Output", "TwoZoneNeBias")


# ---- zone emission --------------------------------------------------------------
class Zones:
    """Per-line emissivities and lower-level densities of one CR table at any (E/N, Ne)."""

    def __init__(self, fit):
        self.fit, tab = fit, fit["tab"]
        self.lx, self.ln = np.log(tab["x_grid"]), np.log(tab["Ne_grid"])
        self.lev = list(tab["levels"])
        idx = {(u, l): k for k, (u, l) in enumerate(zip(tab["line_upper"], tab["line_lower"]))}
        energy = fit["cfg"]["intensity_units"] == "energy"
        kx = min(3, len(self.lx) - 1)
        spl = lambda v: RectBivariateSpline(self.lx, self.ln, np.log(np.clip(v, 1e-300, None)), kx=kx, ky=3)
        # features as CRFitNeTe.feature_model: spline of ln(sum of the components)
        self.comps = [[idx[(u, l)] for _, u, l in ft.comps] for ft in fit["feats"].itertuples()]
        w = lambda k: 1.0 / tab["line_wl"][k] if energy else 1.0
        self.feat = [spl(sum(tab["I_obs"][:, :, k] * w(k) for k in c)) for c in self.comps]
        self.line = {k: spl(tab["I_obs"][:, :, k]) for c in self.comps for k in c}
        self.dens = {l: spl(tab["dens"][:, :, self.lev.index(l)]) for l in set(tab["line_lower"])}
        cr, _, self.MD, _ = crf.load_cr_model(fit["cfg"]["outdir"])
        self.rad = {k: next(r for r in self.MD[tab["line_upper"][k]]["RadiativeDecay"]
                            if r["direction"] == "loss" and r["partner"] == tab["line_lower"][k])
                    for k in self.line}

    def features(self, x, Ne):
        """Feature emissivities (energy units as the fit) at (E/N, Ne)."""
        return np.array([np.exp(s(np.log(x), np.log(Ne))[0, 0]) for s in self.feat])

    def tau(self, k, x, Ne, L):
        """Line-centre optical depth of line k over L [m] with the lower-level density at (x, Ne)."""
        tab = self.fit["tab"]
        lo = tab["line_lower"][k]
        n = np.exp(self.dens[lo](np.log(x), np.log(Ne))[0, 0])
        Tg = self.fit["cfg"]["Tg"]
        return he.FindTauInModel(self.MD[tab["line_upper"][k]], self.MD[lo], self.rad[k],
                                 he.Volume2Torr(n, Tg), Tg, L)[1]

    def transmission(self, xC, NeC, LC, xA, NeA, LA):
        """Per feature: emission-weighted transmission of the lines of a slab at (xC, NeC), path
        LC, through an absorbing layer with the lower-level densities at (xA, NeA), path LA."""
        g = np.exp(-np.linspace(-6, 6, 2401) ** 2)
        T = []
        for c in self.comps:
            I = np.array([np.exp(self.line[k](np.log(xC), np.log(NeC))[0, 0]) for k in c])
            t = []
            for k in c:
                tC, tA = self.tau(k, xC, NeC, LC), self.tau(k, xA, NeA, LA)
                prof = -np.expm1(-tC * g) if tC > 1e-6 else tC * g
                t.append(np.sum(prof * np.exp(-tA * g)) / np.sum(prof))
            T.append(np.sum(I * np.array(t)) / I.sum())
        return np.array(T)


def fit_synthetic(fit, d, I_feat):
    """One-zone CR fit of the synthetic feature intensities on the rows of condition d."""
    fidx = {f: k for k, f in enumerate(fit["feats"].feature)}
    syn = d.assign(area=[I_feat[fidx[f]] for f in d.feature])
    s, post, _ = crf.fit_block(syn, fit["M"], fit["feats"], fit["grid"], fit["Te_f"], fit["cfg"])
    return s


def scan(name, cfg, cond):
    fit = mlf.prepare(cfg)
    z = Zones(fit)
    d = fit["ft"][(fit["ft"].sweep == cond[0]) & (fit["ft"].x == cond[1])]
    s, post, _ = crf.fit_block(d, fit["M"], fit["feats"], fit["grid"], fit["Te_f"], cfg)
    pin, _, _ = cdf.analyse(fit, None, s["chi2_min"], s["dof"], s["birge"], post)
    xC = pin["x_pinned"]                             # inner zone: E/N of the Ne = n_c fit
    TeC = float(np.interp(np.log(xC), z.lx, fit["tab"]["Te_eff"]))
    print(f"\n== {name}: inner zone E/N = {xC:.3g} Td (Te_eff {TeC:.2f} eV), measured fit at n_c: "
          f"chi2/dof {pin['chi2_red_pinned']:.2f}")
    rows = []
    for rC in INNER_NE:
        NeC = rC * N_C
        IC = z.features(xC, NeC)
        ref = fit_synthetic(fit, d, IC)
        print(f"  one zone at Ne = {rC:g} n_c: fit Ne = {ref['Ne_best'] / N_C:.3g} n_c, E/N = {ref['x_best']:.3g} Td")
        for rE in EDGE_EN:
            xE = xC * rE
            if not z.lx[0] <= np.log(xE) <= z.lx[-1]:
                continue
            TeE = float(np.interp(np.log(xE), z.lx, fit["tab"]["Te_eff"]))
            for fE in EDGE_FRAC:
                LC, LE = (1 - fE) * CHORD_M, fE * CHORD_M          # chord split (first two modes)
                LCo, LEo = CHORD_M, fE / (1 - fE) * CHORD_M        # shell outside the column
                for nE in EDGE_NE:
                    NeE = nE * N_C
                    IE = z.features(xE, NeE)
                    in_col = (z.transmission(xC, NeC, LC, xE, NeE, LE / 2)
                              / z.transmission(xC, NeC, LC, xC, NeC, LE / 2))
                    outside = z.transmission(xC, NeC, LCo, xE, NeE, LEo / 2)
                    for mode, t, lc, le in (("emission only", np.ones(len(IC)), LC, LE),
                                            ("outer part of the column", in_col, LC, LE),
                                            ("outside the column", outside, LCo, LEo)):
                        I = lc * IC * t + le * IE
                        r = fit_synthetic(fit, d, I)
                        rows.append(dict(case=name, mode=mode, inner_Ne_nc=rC, edge_EN_ratio=rE, Te_inner=TeC,
                                         Te_edge=TeE, edge_frac=fE, edge_Ne_nc=nE,
                                         edge_light=float((le * IE).sum() / I.sum()),
                                         t_min=float(np.min(t)), t_max=float(np.max(t)),
                                         Ne_fit_nc=r["Ne_best"] / N_C, Ne_med_nc=r["Ne_med"] / N_C,
                                         Ne_lo_nc=r["Ne_lo"] / N_C, Ne_hi_nc=r["Ne_hi"] / N_C,
                                         EN_fit=r["x_best"], Te_fit=r["Te_best"], chi2_red=r["chi2_red"]))
    return pd.DataFrame(rows)


# ---- plots ----------------------------------------------------------------------
MODES = ["emission only", "outer part of the column", "outside the column"]


def plot(res, path):
    cases, modes = list(res.case.unique()), MODES
    fig, axs = plt.subplots(len(cases) * len(INNER_NE), len(modes), figsize=(18, 4.2 * len(cases) * len(INNER_NE)),
                            sharex=True, sharey=True, squeeze=False)
    ls = dict(zip(EDGE_FRAC, (":", "--", "-")))
    cols = dict(zip(EDGE_EN, ("C0", "k", "C1", "C3")))
    for a, (case, rC) in enumerate([(c, r) for c in cases for r in INNER_NE]):
        for b, mode in enumerate(modes):
            ax = axs[a, b]
            g = res[(res.case == case) & (res.inner_Ne_nc == rC) & (res["mode"] == mode)]
            for (rE, fE), h in g.groupby(["edge_EN_ratio", "edge_frac"]):
                h = h.sort_values("edge_Ne_nc")
                ax.plot(h.edge_Ne_nc, h.Ne_fit_nc, ls[fE], color=cols[rE], lw=1.5,
                        label=f"shell E/N x{rE:g} (Te {h.Te_edge.iloc[0]:.2f} eV), path {100 * fE:.0f} %")
            ax.axhline(rC, color="0.4", lw=0.8)
            ax.axhline(1, color="k", lw=1.2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1e-2, 1e3)
            ax.grid(alpha=0.3, which="both")
            ax.set_title(f"{case}; inner zone Ne = {rC:g} n$_c$ ({mode})", fontsize=9)
            ax.set_ylabel("fitted $N_e$ / $n_c$")
        axs[a, 0].legend(fontsize=6, ncol=2)
    for ax in axs[-1]:
        ax.set_xlabel("shell $N_e$ / $n_c$")
    fig.suptitle("One-zone CR fit of a two-zone line of sight: inner zone at the critical layer "
                 "(or overdense) + underdense shell", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    res = pd.concat([scan(*c) for c in CASES], ignore_index=True)
    res.to_csv(os.path.join(OUTDIR, "two_zone_fits.csv"), index=False)
    plot(res, os.path.join(OUTDIR, "two_zone_Ne_bias.png"))
    sel = res[res.edge_Ne_nc.round(4).isin([0.01, 0.1, 0.3162, 1.0]) & (res.edge_frac == 0.5)]
    cols = ["case", "mode", "inner_Ne_nc", "edge_EN_ratio", "Te_edge", "edge_Ne_nc", "edge_light", "t_min",
            "t_max", "Ne_fit_nc", "Ne_lo_nc", "Ne_hi_nc", "Te_fit", "chi2_red"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 500):
        print(sel[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nfigure in {OUTDIR}")
    return res


if __name__ == "__main__":
    RES = main()
