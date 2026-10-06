# -*- coding: utf-8 -*-
"""
LiteratureComparison.py

Our CR model (BSR-500 cross sections + Ar-atom 2p transfer: the current model) at the conditions of
two published argon CR models, a level-by-level audit of its channels, and the process budget of
the 2p levels next to the published one.

Zhu & Pu, J. Phys. D 43, 015204 (2010)
    Their four discharges (Table 7): EBP 1 Pa, ICP 1 Pa, CCP 100 Pa, SRR 1e5 Pa. Our model with a
    Maxwellian EEDF at their Te, their ne and Tg, R = half the smallest plasma dimension.
    Fig. 1 (1s n/g/n_g, ppm) and Fig. 2 (2p n/g, sum 100; their CRM and OES) are digitised in
    InputData/References/ZhuPu2010_digitized.csv; Fig. 4 (production and loss rates of 2p1, 2p3,
    2p6, 2p9) is transcribed below (ZHU_FIG4; 'rad' = spontaneous emission A n, 'trap' = the
    re-absorbed part, so the net radiative loss is rad - trap).
Bogaerts, Gijbels & Vlcek, J. Appl. Phys. 84, 121 (1998)
    1000 V, 1 Torr, 2 mA dc glow, Tg = 450 K, cell radius 2 cm. Their excitation comes from fast
    electrons, ions and atoms (Monte Carlo models), which a Maxwellian cannot reproduce, so only the
    pattern within each manifold is compared: Fig. 2 (4s at z = 1 cm, negative glow), Fig. 4 (4p at
    the cathode-glow maximum), Fig. 6 (3d, 5s at the start of the negative glow), read off the
    scanned figures (BOGAERTS, about +-5 % of full scale). Our model: Maxwellian Te = 1-4 eV,
    ne = 2e17 m^-3.

Model variants: BSR + atom transfer (current), BSR without it, the old NGFSRDW set + atom transfer.

Audit, at every Zhu & Pu discharge and at our 1 Torr conditions (Ne = n_c, Maxwellian 1.5 eV):
channels of every level (radiative decay to tracked / untracked levels, electron-impact excitation
in and out with the cross-section source, de-excitation, ionization, diffusion, atom transfer) and
its production / loss budget (terms as MainFileV2.SolveDirect). Flags: no radiative decay
(non-metastable), mostly radiating to untracked levels, no electron-impact production, ground-state
excitation missing / analytic / tabulated only from > 0.5 eV above threshold, production or
electron-impact loss mostly from analytic (Drawin) cross sections, no ionization, budget not closed,
n/g above the Boltzmann value at Te.

Output: Experimental_Data/Output/LiteratureComparison/
    zhupu_comparison.png, bogaerts_comparison.png, budget_2p_vs_zhupu.png, audit_flags.md,
    audit_channels.csv, audit_budget.csv, zhupu_levels.csv, bogaerts_levels.csv
Run from Spyder (F5) or python SmallAnalysisScripts/LiteratureComparison.py (~1 min).
"""
import contextlib
import copy
import io
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT_DIR, "Experimental_Data", "ExperimentalDataAnalysis"))
import CRFitNeTe as crf                             # noqa: E402

he = crf._helpers()

# ---- settings -------------------------------------------------------------------
PA_PER_TORR = 133.322
N_C = constants.epsilon_0 * constants.m_e * (2 * np.pi * 2.42e9) ** 2 / constants.e ** 2
ZHU = {   # their Table 7; R = half the smallest plasma dimension [m] (trapping); box = dimensions [m]
    "EBP 1 Pa": dict(p_Pa=1.0, Te=10.0, ne=3e15, Tg=300.0, R=0.005, box=(0.40, 0.04, 0.01)),
    "ICP 1 Pa": dict(p_Pa=1.0, Te=3.0, ne=3e16, Tg=400.0, R=0.075, box=(0.40, 0.15, 0.15)),
    "CCP 100 Pa": dict(p_Pa=100.0, Te=1.5, ne=3e17, Tg=400.0, R=0.02, box=(0.20, 0.20, 0.04)),
    "SRR 1e5 Pa": dict(p_Pa=1e5, Te=1.0, ne=1e20, Tg=500.0, R=5e-5, box=(1e-3, 1e-4, 1e-4)),
}
OURS = {"ours 1 Torr, n_c": dict(p_Pa=PA_PER_TORR, Te=1.5, ne=N_C, Tg=300.0, R=0.04)}
BOG_COND = dict(p_Pa=PA_PER_TORR, Tg=450.0, R=0.02, ne=2e17)
BOG_TE = [1.0, 2.0, 4.0]
# xsec: cross-section file; at: Ar-atom 2p transfer; trap: escape factors from the 'old' table (a >= 0.01,
# extrapolated below), the extended 'v2' table (default since 6 Oct 2026) or 'walsh' (Holstein-Walsh
# cylinder); geom: metastable diffusion length of a hemisphere of radius R ('R', the model default) or
# of the discharge's box ('box', Zhu & Pu discharges only)
VARIANTS = {
    "morning model (NGFSRDW, no AT, old table)": dict(xsec="ArgonCrossSections.json", at=False, trap="old", geom="R"),
    "BSR + AT, old table": dict(xsec="ArgonCrossSections_BSR.json", at=True, trap="old", geom="R"),
    "BSR + AT, v2 table (current)": dict(xsec="ArgonCrossSections_BSR.json", at=True, trap="v2", geom="R"),
    "BSR + AT, v2 table, box diffusion": dict(xsec="ArgonCrossSections_BSR.json", at=True, trap="v2", geom="box"),
    "BSR + AT, Walsh trapping, box diffusion": dict(xsec="ArgonCrossSections_BSR.json", at=True, trap="walsh",
                                                    geom="box"),
}
CURRENT = "BSR + AT, v2 table (current)"
VSTYLE = {"morning model (NGFSRDW, no AT, old table)": dict(color="#eb6834", marker="s", mfc="white"),
          "BSR + AT, old table": dict(color="#9aa7b8", marker="o", mfc="white"),
          "BSR + AT, v2 table (current)": dict(color="#2a78d6", marker="o", mfc="#2a78d6"),
          "BSR + AT, v2 table, box diffusion": dict(color="#1baf7a", marker="D", mfc="#1baf7a"),
          "BSR + AT, Walsh trapping, box diffusion": dict(color="#8a3ffc", marker="^", mfc="white")}
ZHU_FIG4 = {   # (level, process): rate in their panel units; unit [m^-3 s^-1] per panel
    "EBP 1 Pa": (1e19, {"2p1": dict(gs=51, cas=7.0, rad=58), "2p3": dict(gs=30, cas=23, rad=53),
                        "2p6": dict(gs=36, cas=23, trap=3.8, rad=64),
                        "2p9": dict(gs=45, cas=47, s1=3.9, trap=22, rad=117)}),
    "ICP 1 Pa": (1e19, {"2p1": dict(gs=21, cas=2.9, trap=2.4, rad=27),
                        "2p3": dict(gs=13, cas=9.6, s1=8.3, trap=9.1, rad=40),
                        "2p6": dict(gs=16, cas=9.6, s1=44, trap=87, rad=157),
                        "2p9": dict(gs=21, cas=19, s1=127, trap=1220, rad=1390)}),
    "CCP 100 Pa": (1e20, {"2p1": dict(gs=7.0, cas=1.7, s1=15, trap=41, rad=64),
                          "2p3": dict(gs=4.8, cas=5.6, s1=104, trap=295, rad=395),
                          "2p6": dict(gs=6.4, cas=5.6, s1=137, trap=486, rad=630),
                          "2p9": dict(gs=8.1, cas=11, s1=280, trap=2220, rad=2410)}),
    "SRR 1e5 Pa": (1e24, {"2p1": dict(gs=10.5, s1=70, rad=6.8, at=84), "2p3": dict(gs=7.5, s1=434, rad=18, at=298),
                          "2p6": dict(gs=10.3, s1=452, rad=52), "2p9": dict(gs=13.1, s1=834, rad=39, at=639)}),
}
FIG4_PROC = [("gs", "ground-state excitation"), ("s1", "excitation from 1s"), ("cas", "cascade"),
             ("rad", "emission A n"), ("trap", "re-absorbed"), ("at", "Ar-atom transfer to 1s")]
BOGAERTS = {   # n [cm^-3], read off the scanned figures
    # Fig. 2 at z = 1 cm (negative glow), 10^12 cm^-3: n=2 1s5, n=3 1s4, n=4 1s3, n=5 1s2 (1s4, 1s2 near zero)
    "4s": dict(scale=1e12, values={"4s[3/2]2": 0.62, "4s[3/2]1": 0.06, "4s'[1/2]0": 0.10, "4s'[1/2]1": 0.02}),
    # Fig. 4, cathode-glow maximum, 10^9 cm^-3
    "4p": dict(scale=1e9, values={"4p[1/2]1": 2.52, "4p[5/2]3": 2.78, "4p[5/2]2": 1.98, "4p[3/2]1": 1.20,
                                   "4p[3/2]2": 1.97, "4p[1/2]0": 0.26, "4p'[3/2]1": 0.36, "4p'[3/2]2": 0.60,
                                   "4p'[1/2]1": 0.37, "4p'[1/2]0": 0.12}),
    # Fig. 6, start of the negative glow, 10^7 cm^-3
    "3d+5s": dict(scale=1e7, values={"3d[1/2]0": 2.15, "3d[1/2]1": 6.45, "3d[3/2]2": 10.8, "3d[7/2]4": 7.1,
                                      "3d[7/2]3": 5.5, "3d[5/2]2": 6.65, "5s[3/2]2": 6.65, "5s[3/2]1": 3.9,
                                      "3d[5/2]3": 9.35, "3d[3/2]1": 3.95, "3d'[5/2]2": 4.6, "3d'[3/2]2": 4.6,
                                      "3d'[5/2]3": 6.4, "5s'[1/2]0": 1.25, "5s'[1/2]1": 3.9, "3d'[3/2]1": 6.3}),
}
DIGITIZED = os.path.join(ROOT_DIR, "InputData", "References", "ZhuPu2010_digitized.csv")
ORDER_2P = ["2p1", "2p5", "2p3", "2p7", "2p8", "2p4", "2p9", "2p2", "2p6", "2p10"]   # their Fig. 2 order
GROUP_2P = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]
ORDER_1S = ["1s5", "1s3", "1s4", "1s2"]
PASCHEN = {name: lbl for name, (lbl, _) in he.PASCHEN_LABELS.items()}       # Paschen -> CR label
K_MET = 6.4e-16                                     # m^3/s, as MainFileV2.SolveDirect
OUTDIR = os.path.join(ROOT_DIR, "Experimental_Data", "Output", "LiteratureComparison")
_JK = re.compile(r"(\d[spd])\s*(['!]?)\s*\[(\d+/\d+)\](\d+)")


# ---- model --------------------------------------------------------------------------
_MODELS = {}


def model(xsec):
    """(cr functions, ModelData, escape-factor interpolator) with the cross-section file xsec."""
    if xsec not in _MODELS:
        keep = he.CROSS_SECTION_FILE
        he.CROSS_SECTION_FILE = xsec
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cr, _, MD, interp = crf.load_cr_model(OUTDIR)
        finally:
            he.CROSS_SECTION_FILE = keep
        _MODELS[xsec] = (cr, MD, interp)
    return _MODELS[xsec]


_INTERP = {}


def escape_interp(trap, default):
    """Escape-factor interpolator: 'v2' (the model's), 'walsh', or the 'old' table as the model used it
    (linear in a, extrapolated below a = 0.01)."""
    if trap == "v2":
        return default
    if trap not in _INTERP:
        if trap == "walsh":
            _INTERP[trap] = he.EscapeFactorInterpolator(mode="walsh")
        else:
            from scipy.interpolate import RegularGridInterpolator
            RTM = he.ImportRadiationTrappingMatrix("RadiationTrappingLookupTable.json")
            tau = np.unique([d["Tau_R"] for d in RTM])
            shape = np.unique([d["Shape"] for d in RTM])
            eta = np.array([d["EscapeFactor"][0] for d in RTM]).reshape(len(tau), len(shape))
            _INTERP[trap] = RegularGridInterpolator((np.log10(tau), shape), np.log10(eta), bounds_error=False,
                                                    fill_value=None)
    return _INTERP[trap]


def box_lambda(dims):
    return 1.0 / np.sqrt(sum((np.pi / L) ** 2 for L in dims))


def solve(variant, c):
    """Our CR model at the conditions c (Maxwellian EEDF at c['Te']) -> (level dict, converged)."""
    v = VARIANTS[variant]
    cr, MD, interp0 = model(v["xsec"])
    at, interp = v["at"], escape_interp(v["trap"], interp0)
    lam = box_lambda(c["box"]) if (v["geom"] == "box" and "box" in c) else None
    P = c["p_Pa"] / PA_PER_TORR
    D = he.AddDiffusionLoss(copy.deepcopy(MD), P, c["Tg"], c["R"], lam)
    with contextlib.redirect_stdout(io.StringIO()):
        D, _, _, s = cr["CRModel"](D, c["Te"], c["ne"], P, c["Tg"], c["R"], interp, atom_transfer=at)
    return D, bool(s["converged"])


def manifold(label):
    return "ground" if label == "ground" else label.rstrip("0123456789")


def jk_label(name, levels):
    """CR label of a jK name like "4p'[1/2]0" (prime = 2P1/2 core)."""
    nl, prime, K, J = _JK.search(name).groups()
    core = "<1/2>" if prime else "<3/2>"
    cand = [l for l, v in levels.items() if v.get("manifold") == nl and core in v["configuration"]
            and f"[{K}]" in v["term"] and float(v["J"]) == float(J)]
    if len(cand) != 1:
        raise ValueError(f"{name}: {cand}")
    return cand[0]


# ---- audit --------------------------------------------------------------------------
def escape_factors(D, c, interp):
    """Escape factor of every line at the solved densities (as MainFileV2.SolveDirect)."""
    eta = {}
    for U in D.values():
        for Rad in U["RadiativeDecay"]:
            if Rad["direction"] != "loss":
                continue
            Lo = D[Rad["partner"]]
            key = (U["label"], Lo["label"], Rad["coeff"])
            if Lo["density_m^-3"] == 0:
                eta[key] = 1.0
                continue
            a, tau = he.FindTauInModel(U, Lo, Rad, he.Volume2Torr(Lo["density_m^-3"], c["Tg"]), c["Tg"], c["R"])
            f = getattr(interp, "resonance", interp) if Lo["kind"] == "ground" else interp
            eta[key] = float(np.squeeze(10 ** f(np.array([[np.log10(float(np.squeeze(tau))),
                                                            float(np.squeeze(a))]]))))
    return eta


def channels(D):
    """Condition-independent channel inventory of every excited level."""
    rows = []
    for lbl, U in D.items():
        if U["kind"] == "ground":
            continue
        rad_out = [r for r in U["RadiativeDecay"] if r["direction"] == "loss"]
        rad_in = [r for r in U["RadiativeDecay"] if r["direction"] == "gain"]
        e_in, e_out = U["Electron Impact CrossSections"]["Products"], U["Electron Impact CrossSections"]["Reactants"]
        src = lambda LS: "analytic" if LS.get("analytic") else LS.get("database", "LXCat")
        g = [LS for LS in e_in if LS["partner_label"] == "ground"]
        gap = (min(np.asarray(LS["energy_eV"], float)[np.asarray(LS["cross_section"], float) > 0][0]
                   - LS["threshold_eV"] for LS in g) if g else np.nan)
        rows.append(dict(level=lbl, manifold=manifold(lbl), kind=U["kind"], g=U["g"], E_eV=U["energy_eV"],
                         n_rad_out=len(rad_out), A_out_s=sum(r["coeff"] for r in rad_out),
                         A_untracked_s=U.get("A_untracked_loss", 0.0), n_cascade_in=len(rad_in),
                         n_e_in=len(e_in), n_e_in_analytic=sum(src(LS) == "analytic" for LS in e_in),
                         n_e_out=len(e_out), n_e_out_analytic=sum(src(LS) == "analytic" for LS in e_out),
                         ground_exc=("none" if not g else "+".join(sorted({src(LS) for LS in g}))),
                         ground_exc_gap_eV=gap, n_atom_transfer=len(U.get("AtomTransfer", [])),
                         ionization=U["Ionization Data"].get("Rate_cm^3", 0.0) > 0))
    return pd.DataFrame(rows)


def budget(D, c, cr, interp):
    """Production [m^-3 s^-1] and loss [s^-1] terms of every excited level (MainFileV2.SolveDirect)."""
    Ne = c["ne"]
    Ng = D["ground"]["density_m^-3"]
    n = {l: d["density_m^-3"] for l, d in D.items()}
    n_meta = sum(d["density_m^-3"] for d in D.values() if d["kind"] == "metastable")
    eta = escape_factors(D, c, interp)
    kTe = c["Te"]
    rows = []
    for lbl, U in D.items():
        if U["kind"] == "ground":
            continue
        P, L = defaultdict(float), defaultdict(float)
        for LS in U["Electron Impact CrossSections"]["Products"]:
            j = LS["partner_label"]
            r = Ne * LS.get("Rate", 0.0) * n[j]
            P["e-exc from ground" if j == "ground" else f"e-exc from {manifold(j)}"] += r
            if LS.get("analytic"):
                P["_analytic"] += r
        for SE in U.get("Superelastic", []):
            if SE["direction"] == "gain":
                P["e-deexc from above"] += Ne * SE["coeff"] * n[SE["partner"]]
        for Rad in U["RadiativeDecay"]:
            if Rad["direction"] == "gain":
                P["cascade"] += Rad["coeff"] * eta[(Rad["partner"], lbl, Rad["coeff"])] * n[Rad["partner"]]
        for AT in U.get("AtomTransfer", []):
            if AT["direction"] == "gain":
                P["atom transfer in"] += Ng * AT["coeff"] * n[AT["partner"]]
        rad_out = [r for r in U["RadiativeDecay"] if r["direction"] == "loss"]
        emitted = sum(r["coeff"] for r in rad_out)
        if U["kind"] != "metastable":
            L["radiation"] = (sum(r["coeff"] * eta[(lbl, r["partner"], r["coeff"])] for r in rad_out)
                              + U.get("A_untracked_loss", 0.0))
        e_out = U["Electron Impact CrossSections"]["Reactants"]
        L["e-exc out"] = Ne * sum(LS.get("Rate", 0.0) for LS in e_out)
        L["_e-exc out analytic"] = Ne * sum(LS.get("Rate", 0.0) for LS in e_out if LS.get("analytic"))
        L["e-deexc down"] = Ne * sum(SE["coeff"] for SE in U.get("Superelastic", []) if SE["direction"] == "loss")
        L["ionization"] = Ne * U["Ionization Data"].get("Rate_cm^3", 0.0)
        if U["kind"] == "metastable":
            L["diffusion"] = 1 / U["DiffusionLoss"]
            L["Ar quenching"] = cr["GroundQuenchingLoss"](Ng)
        L["metastable collisions"] = K_MET * n_meta
        L["atom transfer out"] = Ng * sum(AT["coeff"] for AT in U.get("AtomTransfer", []) if AT["direction"] == "loss")
        prod = sum(v for k, v in P.items() if not k.startswith("_"))
        loss = sum(v for k, v in L.items() if not k.startswith("_"))
        boltz = Ng / D["ground"]["g"] * np.exp(-U["energy_eV"] / kTe)
        rows.append(dict(level=lbl, manifold=manifold(lbl), kind=U["kind"], density=n[lbl], n_over_g=n[lbl] / U["g"],
                         boltzmann_ratio=(n[lbl] / U["g"]) / boltz, production=prod, loss_rate=loss,
                         lifetime_s=1 / loss if loss > 0 else np.inf, closure=prod / (loss * n[lbl]) - 1 if loss * n[lbl] > 0 else np.nan,
                         emitted=emitted * n[lbl], reabsorbed=sum(r["coeff"] * (1 - eta[(lbl, r["partner"], r["coeff"])])
                                                                 for r in rad_out) * n[lbl],
                         analytic_prod_frac=P["_analytic"] / prod if prod > 0 else np.nan,
                         analytic_eloss_frac=L["_e-exc out analytic"] / L["e-exc out"] if L["e-exc out"] > 0 else np.nan,
                         **{f"P: {k}": v for k, v in P.items() if not k.startswith("_")},
                         **{f"L: {k}": v * n[lbl] for k, v in L.items() if not k.startswith("_")}))
    return pd.DataFrame(rows).fillna({c: 0.0 for c in []})


def flags(ch, bu):
    """Audit flags per level (ch: channels(), bu: budget() of one condition)."""
    out = []
    b = bu.set_index("level")
    for r in ch.itertuples():
        f = []
        if r.kind not in ("metastable",) and r.A_out_s + r.A_untracked_s == 0:
            f.append("NO RADIATIVE DECAY")
        if r.A_untracked_s > 0.5 * (r.A_out_s + r.A_untracked_s) and r.A_untracked_s > 0:
            f.append(f"radiates mostly to untracked levels ({r.A_untracked_s / (r.A_out_s + r.A_untracked_s):.0%})")
        if r.n_e_in == 0:
            f.append("NO ELECTRON-IMPACT PRODUCTION")
        if r.ground_exc == "none":
            f.append("no ground-state excitation")
        elif "analytic" in r.ground_exc:
            f.append("ground-state excitation analytic (Drawin)")
        if np.isfinite(r.ground_exc_gap_eV) and r.ground_exc_gap_eV > 0.5:
            f.append(f"ground-state cross section starts {r.ground_exc_gap_eV:.1f} eV above threshold")
        if not r.ionization:
            f.append("no ionization")
        x = b.loc[r.level]
        if x.analytic_prod_frac > 0.5:
            f.append(f"production {x.analytic_prod_frac:.0%} from analytic cross sections")
        if x.analytic_eloss_frac > 0.5 and x["L: e-exc out"] > 0.2 * x.loss_rate * x.density:
            f.append(f"electron-impact loss {x.analytic_eloss_frac:.0%} analytic (and >20 % of all loss)")
        if np.isfinite(x.closure) and abs(x.closure) > 1e-3:
            f.append(f"budget not closed ({x.closure:+.1e})")
        if x.boltzmann_ratio > 1.0:
            f.append(f"n/g above Boltzmann at Te (x{x.boltzmann_ratio:.2g})")
        pcols = [k for k in x.index if k.startswith("P: ")]
        lcols = [k for k in x.index if k.startswith("L: ")]
        top_p = max(pcols, key=lambda k: x[k] if np.isfinite(x[k]) else -1)
        top_l = max(lcols, key=lambda k: x[k] if np.isfinite(x[k]) else -1)
        out.append(dict(level=r.level, flag_text="; ".join(f),
                        top_production=f"{top_p[3:]} ({x[top_p] / x.production:.0%})" if x.production > 0 else "-",
                        top_loss=f"{top_l[3:]} ({x[top_l] / (x.loss_rate * x.density):.0%})" if x.loss_rate * x.density > 0 else "-"))
    return pd.DataFrame(out)


# ---- Zhu & Pu -------------------------------------------------------------------------
def zhu_data():
    d = pd.read_csv(DIGITIZED, comment="#")
    return {(r.figure, r.discharge, r.level, r.source): r.value for r in d.itertuples()}


def xpos(groups, gap=0.6):
    return np.arange(len(groups)) + gap * np.asarray(groups)


def style(ax, ylabel=None):
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)


def zhu_comparison(sol, budgets):
    Z = zhu_data()
    rows = []
    fig, axs = plt.subplots(2, len(ZHU), figsize=(5.2 * len(ZHU), 9.5), squeeze=False)
    x2 = xpos(GROUP_2P)
    for j, disc in enumerate(ZHU):
        ax = axs[0, j]
        crm = np.array([Z.get((2, disc, l, "CRM"), np.nan) for l in ORDER_2P])
        oes = np.array([Z.get((2, disc, l, "OES"), np.nan) for l in ORDER_2P])
        ax.bar(x2, crm, width=0.62, color="0.8", label="Zhu & Pu CRM")
        ax.plot(x2, oes, "*", ms=12, mfc="white", mec="0.15", mew=1.2, label="Zhu & Pu OES")
        for k, (v, st) in enumerate(VSTYLE.items()):
            D = sol[v, disc][0]
            ng = np.array([D[PASCHEN[l]]["density_m^-3"] / D[PASCHEN[l]]["g"] for l in ORDER_2P])
            val = 100 * ng / ng.sum()
            ax.plot(x2 + 0.13 * (k - 2), val, st["marker"], ms=7, color=st["color"], mfc=st["mfc"], mew=1.5, label=v)
            for l, a, b, c in zip(ORDER_2P, val, crm, oes):
                rows.append(dict(discharge=disc, quantity="2p n/g, sum 100", level=l, variant=v, ours=a,
                                 zhupu_crm=b, zhupu_oes=c))
        cnd = ZHU[disc]
        ax.set_title(f"{disc}: Te = {cnd['Te']:g} eV, ne = {cnd['ne']:.0e} m$^{{-3}}$, Tg = {cnd['Tg']:g} K",
                     fontsize=10)
        ax.set_xticks(x2, [f"2p$_{{{l[2:]}}}$" for l in ORDER_2P], fontsize=8)
        style(ax, "$n/g$ (sum of the ten 2p = 100)" if j == 0 else None)
        ax = axs[1, j]
        x1 = np.arange(4)
        crm = np.array([Z.get((1, disc, l, "CRM"), np.nan) for l in ORDER_1S])
        oes = np.array([Z.get((1, disc, l, "OES"), np.nan) for l in ORDER_1S])
        ax.bar(x1, crm, width=0.62, color="0.8")
        ax.plot(x1, oes, "*", ms=12, mfc="white", mec="0.15", mew=1.2)
        for k, (v, st) in enumerate(VSTYLE.items()):
            D = sol[v, disc][0]
            ngr = D["ground"]["density_m^-3"]
            val = np.array([1e6 * D[PASCHEN[l]]["density_m^-3"] / D[PASCHEN[l]]["g"] / ngr for l in ORDER_1S])
            ax.plot(x1 + 0.13 * (k - 2), val, st["marker"], ms=7, color=st["color"], mfc=st["mfc"], mew=1.5)
            for l, a, b, c in zip(ORDER_1S, val, crm, oes):
                rows.append(dict(discharge=disc, quantity="1s n/g/n_g [ppm]", level=l, variant=v, ours=a,
                                 zhupu_crm=b, zhupu_oes=c))
        ax.set_yscale("log")
        ax.set_xticks(x1, [f"1s$_{{{l[2:]}}}$" for l in ORDER_1S])
        style(ax, "$n/(g\\,n_g)$ [ppm]" if j == 0 else None)
    axs[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Our CR model at the four discharges of Zhu & Pu, J. Phys. D 43, 015204 (2010) "
                 "(Maxwellian at their Te; their Figs. 1-2 digitised)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUTDIR, "zhupu_comparison.png"), dpi=140)
    plt.close(fig)

    # Fig. 4: process rates of 2p1, 2p3, 2p6, 2p9
    fig, axs = plt.subplots(1, len(ZHU), figsize=(5.4 * len(ZHU), 5.2), squeeze=False)
    procs = [p for p, _ in FIG4_PROC]
    for j, disc in enumerate(ZHU):
        ax = axs[0, j]
        unit, theirs = ZHU_FIG4[disc]
        b = budgets[CURRENT, disc].set_index("level")
        levels = list(theirs)
        w = 0.8 / len(procs)
        cols = plt.cm.tab10(np.arange(len(procs)))
        for k, p in enumerate(procs):
            xs = np.arange(len(levels)) + (k - (len(procs) - 1) / 2) * w
            th = [theirs[l].get(p, np.nan) * unit for l in levels]
            ou = []
            for l in levels:
                x = b.loc[PASCHEN[l]]
                ou.append({"gs": x.get("P: e-exc from ground", 0.0), "s1": x.get("P: e-exc from 4s", 0.0),
                           "cas": x.get("P: cascade", 0.0), "rad": x.emitted, "trap": x.reabsorbed,
                           "at": x.get("L: atom transfer out", 0.0)}[p])
            ax.bar(xs, th, width=w * 0.9, color=cols[k], alpha=0.35, edgecolor=cols[k], hatch="//",
                   label=f"Zhu & Pu: {dict(FIG4_PROC)[p]}" if j == 0 else None)
            ax.plot(xs, ou, "o", color=cols[k], mec="k", mew=0.5, ms=6,
                    label=f"ours: {dict(FIG4_PROC)[p]}" if j == 0 else None)
            for l, t, o in zip(levels, th, ou):
                rows.append(dict(discharge=disc, quantity=f"2p rate: {dict(FIG4_PROC)[p]} [m^-3 s^-1]", level=l,
                                 variant=CURRENT, ours=o, zhupu_crm=t))
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(levels)), [f"2p$_{{{l[2:]}}}$" for l in levels])
        ax.set_title(disc, fontsize=10)
        style(ax, "rate [m$^{-3}$ s$^{-1}$]" if j == 0 else None)
    axs[0, 0].legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
    fig.suptitle(f"Production and loss of 2p1, 2p3, 2p6, 2p9: Zhu & Pu Fig. 4 (hatched bars) vs our model "
                 f"({CURRENT}, dots); our cascade = all radiative feeding", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUTDIR, "budget_2p_vs_zhupu.png"), dpi=140)
    plt.close(fig)
    return pd.DataFrame(rows)


# ---- Bogaerts -----------------------------------------------------------------------------
def bogaerts_comparison():
    _, MD, _ = model(VARIANTS[CURRENT]["xsec"])
    levels = {l: dict(manifold=d.get("manifold"), configuration=d.get("configuration", ""), term=d.get("term", ""),
                      J=d.get("J"), g=d["g"]) for l, d in MD.items() if l != "ground"}
    sols = {(v, te): solve(v, dict(BOG_COND, Te=te))[0] for v in (CURRENT, next(iter(VARIANTS))) for te in BOG_TE}
    rows = []
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw=dict(width_ratios=[0.7, 1.1, 1.6]))
    for ax, (grp, spec) in zip(axs, BOGAERTS.items()):
        names = list(spec["values"])
        labs = [jk_label(nm, levels) for nm in names]
        g = np.array([levels[l]["g"] for l in labs], float)
        theirs = np.array([spec["values"][nm] for nm in names]) / g
        theirs = 100 * theirs / theirs.sum()
        x = np.arange(len(names))
        ax.bar(x, theirs, width=0.62, color="0.8", label="Bogaerts et al. 1998")
        for k, te in enumerate(BOG_TE):
            for v, st in VSTYLE.items():
                if v != CURRENT and (te != 2.0 or not v.startswith("morning")):
                    continue
                D = sols[v, te]
                ours = np.array([D[l]["density_m^-3"] for l in labs]) / g
                ours = 100 * ours / ours.sum()
                a = 1.0 if v == CURRENT else 0.8
                ax.plot(x + 0.12 * (k - 1), ours, st["marker"], ms=6, color=plt.cm.viridis(k / 2) if v == CURRENT else st["color"],
                        mfc=plt.cm.viridis(k / 2) if v == CURRENT else st["mfc"], mew=1.3, alpha=a,
                        label=f"ours {v}, Te = {te:g} eV")
                for nm, l, t, o in zip(names, labs, theirs, ours):
                    rows.append(dict(group=grp, level=nm, label=l, variant=v, Te=te, ours=o, bogaerts=t))
        ax.set_xticks(x, [nm.replace("[", "\n[") for nm in names], fontsize=7)
        ax.set_title(f"{grp}: n/g within the group (sum 100)", fontsize=10)
        style(ax, "$n/g$ (sum = 100)")
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Bogaerts, Gijbels & Vlcek, J. Appl. Phys. 84, 121 (1998), 1 Torr dc glow (4s: negative glow z = 1 cm; "
                 "4p: cathode-glow maximum; 3d+5s: negative glow) vs our model at 1 Torr, Tg 450 K, ne 2e17 m$^{-3}$, Maxwellian",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUTDIR, "bogaerts_comparison.png"), dpi=140)
    plt.close(fig)
    return pd.DataFrame(rows)


# ---- driver -------------------------------------------------------------------------------
def write_flags_md(path, flag_tables, ch):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# CR-model audit (BSR + atom transfer)\n\n")
        for cond, ft in flag_tables.items():
            f.write(f"## {cond}\n\n| level | flags | top production | top loss |\n|---|---|---|---|\n")
            for r in ft.itertuples():
                f.write(f"| {r.level} | {r.flag_text or '-'} | {r.top_production} | {r.top_loss} |\n")
            f.write("\n")
        f.write("## Channel inventory\n\n" + ch.to_markdown(index=False, floatfmt=".3g") + "\n")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    conds = {**ZHU, **OURS}
    sol = {(v, d): solve(v, c) for v in VARIANTS for d, c in conds.items()}
    for (v, d), (_, ok) in sol.items():
        if not ok:
            print(f"warning: CR model not converged for {v}, {d}")
    budgets = {(v, d): budget(sol[v, d][0], conds[d], model(VARIANTS[v]["xsec"])[0],
                              escape_interp(VARIANTS[v]["trap"], model(VARIANTS[v]["xsec"])[2]))
               for v in VARIANTS for d in conds}
    ch = channels(sol[CURRENT, "CCP 100 Pa"][0])
    ch.to_csv(os.path.join(OUTDIR, "audit_channels.csv"), index=False)
    pd.concat([b.assign(variant=v, condition=d) for (v, d), b in budgets.items()]).to_csv(
        os.path.join(OUTDIR, "audit_budget.csv"), index=False)
    flag_tables = {d: flags(ch, budgets[CURRENT, d]) for d in conds}
    write_flags_md(os.path.join(OUTDIR, "audit_flags.md"), flag_tables, ch)
    zl = zhu_comparison(sol, budgets)
    zl.to_csv(os.path.join(OUTDIR, "zhupu_levels.csv"), index=False)
    bl = bogaerts_comparison()
    bl.to_csv(os.path.join(OUTDIR, "bogaerts_levels.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 120, "display.max_rows", 300):
        print(ch.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        for d, ft in flag_tables.items():
            print(f"\n=== {d}")
            print(ft[ft["flag_text"] != ""].to_string(index=False))
    print(f"\noutput in {OUTDIR}")
    return dict(sol=sol, budgets=budgets, channels=ch, flags=flag_tables, zhu=zl, bogaerts=bl)


if __name__ == "__main__":
    OUT = main()
