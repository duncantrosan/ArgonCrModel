# -*- coding: utf-8 -*-
"""
ZhuPuComparison.py

Our CR model and our 1 Torr spectra next to X.-M. Zhu and Y.-K. Pu, J. Phys. D 43, 015204 (2010),
the source of the 2p <-> 2p atom-collision rates the model uses (InputData/Ar_2p_atom_transfer.csv).
Their Figs. 1 and 2 are digitised in InputData/References/ZhuPu2010_digitized.csv.

  (a, b) 2p distribution n/g (sum of the ten levels = 100) of their CCP (100 Pa) and ICP (1 Pa):
         their CRM and OES, and our CR model at their conditions (their Table 7: Te, ne, p, Tg;
         Maxwellian EEDF; R = half the smallest plasma dimension), with and without the Ar-atom transfer
  (c)    1s densities n/g/n_g [ppm] of the same two discharges
  (d)    our pure-Ar spectra (1 Torr, N2 fraction 0 %), the seven 2p levels we measure: n/g from the
         calibrated line intensities, I lambda / (A eta g) with eta the escape factor of the CR model
         at the Ne = n_c fit, microwave EEDF (sum of the seven = 100, mean over the repeat spectra);
         the band spans eta = 1 to eta of the DC-EEDF fit at n_c, whose 1s densities are close to the
         absorption estimate; our CR model at Ne = n_c and at the free fit (microwave EEDF); their
         CCP OES on the same seven levels
  (e)    ln(measured / CR model) per level at Ne = n_c, with and without the Ar-atom transfer, both
         EEDFs (mean over the seven levels removed: the absolute scale is free in the fit)
  (f)    1s densities [ppm] of our discharge: absorption estimate of 1s5, CR model at Ne = n_c and
         at the free fit (microwave EEDF) and at Ne = n_c with the DC EEDF; their CCP for reference

The fits are those of CriticalDensityFit.pure_ar (EEDFPhysics tables, BOLSIG+ microwave and DC).
Output: Experimental_Data/Output/ZhuPuComparison/zhupu_comparison.png and zhupu_comparison.csv.
Run from Spyder: F5.
"""
import contextlib
import copy
import io
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT_DIR, "Experimental_Data", "ExperimentalDataAnalysis"))
sys.path.insert(0, HERE)
import CriticalDensityFit as cdf                    # noqa: E402

crf, he, pc = cdf.crf, cdf.he, cdf.pc

# ---- settings -------------------------------------------------------------------
DIGITIZED = os.path.join(ROOT_DIR, "InputData", "References", "ZhuPu2010_digitized.csv")
THEIR = {   # their Table 7; R [m] = half the smallest plasma dimension (CCP: 4 cm gap, ICP: 15 cm)
    "CCP 100 Pa": dict(p_Pa=100.0, Te=1.5, ne=3e17, Tg=400.0, R=0.02),
    "ICP 1 Pa": dict(p_Pa=1.0, Te=3.0, ne=3e16, Tg=400.0, R=0.075),
}
ORDER_2P = ["2p1", "2p5", "2p3", "2p7", "2p8", "2p4", "2p9", "2p2", "2p6", "2p10"]   # their Fig. 2 groups
GROUP_2P = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]
ORDER_1S = ["1s5", "1s3", "1s4", "1s2"]
PASCHEN = {lbl: name for name, (lbl, _) in he.PASCHEN_LABELS.items()}     # CR label -> Paschen
LABEL = {name: lbl for lbl, name in PASCHEN.items()}
PA_PER_TORR = 133.322
COL = dict(at="#2a78d6", noat="#eb6834", meas="#1baf7a", theirs="0.78", ink="0.12")
OUTDIR = os.path.join(ROOT_DIR, "Experimental_Data", "Output", "ZhuPuComparison")


def their_data():
    d = pd.read_csv(DIGITIZED, comment="#")
    return {(r.figure, r.discharge, r.level, r.source): r.value for r in d.itertuples()}


# ---- our CR model at their conditions --------------------------------------------
def model_at(cr, MD, interp, cond, at):
    """Our CR model at one of their discharges -> ({Paschen: n/g [m^-3]}, n_g [m^-3])."""
    P = cond["p_Pa"] / PA_PER_TORR
    D = he.AddDiffusionLoss(copy.deepcopy(MD), P, cond["Tg"], cond["R"])
    with contextlib.redirect_stdout(io.StringIO()):
        D, _, _, s = cr["CRModel"](D, cond["Te"], cond["ne"], P, cond["Tg"], cond["R"], interp,
                                   atom_transfer=at)
    if not s["converged"]:
        print(f"warning: CR model not converged at {cond}")
    return {name: D[lbl]["density_m^-3"] / D[lbl]["g"] for name, lbl in LABEL.items()}, D["ground"]["density_m^-3"]


def norm(values, levels):
    """n/g of levels scaled to a sum of 100."""
    v = np.array([values[l] for l in levels], float)
    return 100 * v / v.sum()


# ---- our pure-Ar spectra and fits -------------------------------------------------
def prepare(eedf, at):
    """CR table, measured features and the free and Ne = n_c fits of the pure-Ar condition
    (as CriticalDensityFit.pure_ar), with or without the Ar-atom transfer."""
    cfg = dict(pc.CFG, eedf=str(dict(cdf.PURE_AR)[eedf]), atom_transfer=at)
    cfg["outdir"] = crf.output_dir("EEDFPhysics", cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        tab = crf.build_model_table(cfg)
        feats, comps = crf.select_features(tab, cfg)
        ft = crf.feature_table(crf.measure(comps, cfg), feats, cfg)
    rs = crf.repeat_scatter(ft)
    if cfg["use_repeat_scatter"]:
        ft["rel_err"] = np.sqrt(ft.rel_err ** 2 + ft.feature.map(rs) ** 2)
    grid = crf.fine_grid(tab, cfg)
    Te_f = crf.te_eff_fine(tab, grid)
    d = ft.query("sweep == 'N2 fraction' and x == 0")
    s, post, _ = crf.fit_block(d, crf.feature_model(tab, feats, cfg), feats, grid, Te_f, cfg)
    r, pin, _ = cdf.analyse(dict(tab=tab, grid=grid, Te_f=Te_f, cfg=cfg), None,
                            s["chi2_min"], s["dof"], s["birge"], post)
    return dict(cfg=cfg, tab=tab, feats=feats, d=d, s=s, r=r, post=post, pin=pin)


def our_levels(F, MD):
    """Per measured 2p level: ln(n/g) measured (eta at Ne = n_c) and of the CR model at Ne = n_c
    and at the free fit; mean over the repeat spectra (each spectrum has its own scale)."""
    tab, cfg, d = F["tab"], F["cfg"], F["d"]
    lev = list(tab["levels"])
    pmean = lambda w, values: crf.posterior_mean(tab, w, values, cfg)
    rows = []
    for f in F["feats"].itertuples():
        u = f.uppers
        if "+" in u or not u.startswith("4p"):
            continue
        k = int(np.flatnonzero((tab["line_upper"] == u) & (np.abs(tab["line_wl"] - f.wl) < 0.05))[0])
        ln_eta = np.log(tab["I_obs"][:, :, k] / tab["I_thin"][:, :, k])
        ln_n = np.log(tab["dens"][:, :, lev.index(u)])
        lng = np.log(MD[u]["g"])
        rows.append(dict(level=PASCHEN[u], label=u, feature=f.feature, wl=f.wl, A=tab["line_A"][k],
                         eta_nc=np.exp(pmean(F["pin"], ln_eta)), eta_free=np.exp(pmean(F["post"], ln_eta)),
                         ln_model_nc=pmean(F["pin"], ln_n) - lng, ln_model_free=pmean(F["post"], ln_n) - lng))
    L = pd.DataFrame(rows)
    # measured ln(n/g) per spectrum (photons: area x lambda), centred within each spectrum
    m = d[d.feature.isin(L.feature)].pivot_table(index="file", columns="feature", values="area")
    m = m.dropna()                                  # spectra with all seven levels
    lnI = np.log(m * L.set_index("feature").wl)
    lnI -= np.log(L.set_index("feature").A * L.set_index("feature").eta_nc * [MD[u]["g"] for u in L.label])
    lnI = lnI.sub(lnI.mean(axis=1), axis=0)
    L["ln_meas"] = L.feature.map(lnI.mean())
    L["ln_meas_sd"] = L.feature.map(lnI.std(ddof=1))
    L["n_spectra"] = len(lnI)
    for c in ("ln_model_nc", "ln_model_free"):
        L[c] -= L[c].mean()
    L["resid_nc"] = (L.ln_meas - L.ln_model_nc) - (L.ln_meas - L.ln_model_nc).mean()
    return L


def our_1s(F, MD):
    """n/g/n_g [ppm] of the four 1s levels of the CR model at Ne = n_c and at the free fit."""
    tab, cfg = F["tab"], F["cfg"]
    lev = list(tab["levels"])
    ng = he.Torr2Volume(cfg["P_Torr"], cfg["Tg"])
    out = {}
    for name in ORDER_1S:
        u = LABEL[name]
        ln_n = np.log(tab["dens"][:, :, lev.index(u)])
        for key, w in (("nc", F["pin"]), ("free", F["post"])):
            out[name, key] = 1e6 * np.exp(crf.posterior_mean(tab, w, ln_n, cfg)) / MD[u]["g"] / ng
    return out, ng


# ---- figure -----------------------------------------------------------------------
def xpos(groups, gap=0.6):
    return np.arange(len(groups)) + gap * np.asarray(groups)


def style(ax, ylabel=None):
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    Z = their_data()
    cr, _, MD, interp = crf.load_cr_model(OUTDIR)
    rows = []

    # (a-c) our model at their conditions
    ours = {(disc, at): model_at(cr, MD, interp, cond, at) for disc, cond in THEIR.items() for at in (True, False)}
    # (d-f) our discharge
    fits = {(eedf, at): prepare(eedf, at) for eedf in ("microwave", "dc") for at in (True, False)}
    levs = {k: our_levels(F, MD) for k, F in fits.items()}
    s1s = {k: our_1s(F, MD) for k, F in fits.items()}

    fig, axs = plt.subplots(2, 3, figsize=(18, 10.5), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
    x2 = xpos(GROUP_2P)
    for ax, disc in zip(axs[0, :2], THEIR):
        crm = np.array([Z[2, disc, l, "CRM"] for l in ORDER_2P])
        oes = np.array([Z[2, disc, l, "OES"] for l in ORDER_2P])
        ax.bar(x2, crm, width=0.62, color=COL["theirs"], label="Zhu & Pu CRM")
        ax.plot(x2, oes, "*", ms=13, mfc="white", mec=COL["ink"], mew=1.3, label="Zhu & Pu OES")
        for at, key, mk in ((True, "at", "o"), (False, "noat", "o")):
            v = norm(ours[disc, at][0], ORDER_2P)
            ax.plot(x2 + (0.16 if at else -0.16), v, mk, ms=8, color=COL[key],
                    mfc=COL[key] if at else "white", mew=1.8,
                    label="our CR model, with Ar-atom transfer" if at else "our CR model, without")
            for l, a, b, c in zip(ORDER_2P, v, crm, oes):
                rows.append(dict(panel=disc, quantity="n/g, sum 100", level=l, zhupu_crm=b, zhupu_oes=c,
                                 source="ours with AT" if at else "ours without AT", value=a))
        c = THEIR[disc]
        ax.set_title(f"{disc}: our CR model at their conditions\n"
                     f"(Te = {c['Te']:g} eV Maxwellian, ne = {c['ne']:.0e} m$^{{-3}}$, Tg = {c['Tg']:g} K)",
                     fontsize=10.5)
        ax.set_xticks(x2, [f"2p$_{{{l[2:]}}}$" for l in ORDER_2P])
        style(ax, "$n/g$ (sum of the ten 2p levels = 100)")
    axs[0, 0].legend(fontsize=8.5, loc="upper left", frameon=False)

    # (c) 1s of their discharges
    ax = axs[0, 2]
    x1 = np.arange(4)
    for j, disc in enumerate(THEIR):
        xs = x1 + j * 5
        crm = np.array([Z[1, disc, l, "CRM"] for l in ORDER_1S])
        oes = np.array([Z[1, disc, l, "OES"] for l in ORDER_1S])
        ax.bar(xs, crm, width=0.62, color=COL["theirs"], label="Zhu & Pu CRM" if j == 0 else None)
        ax.plot(xs, oes, "*", ms=13, mfc="white", mec=COL["ink"], mew=1.3, label="Zhu & Pu OES" if j == 0 else None)
        for at, key in ((True, "at"), (False, "noat")):
            n, ng = ours[disc, at]
            v = np.array([1e6 * n[l] / ng for l in ORDER_1S])
            ax.plot(xs + (0.16 if at else -0.16), v, "o", ms=8, color=COL[key], mfc=COL[key] if at else "white",
                    mew=1.8, label=(("our CR model, with Ar-atom transfer" if at else "our CR model, without")
                                    if j == 0 else None))
            for l, a, b, c in zip(ORDER_1S, v, crm, oes):
                rows.append(dict(panel=disc, quantity="n/g/n_g [ppm]", level=l, zhupu_crm=b, zhupu_oes=c,
                                 source="ours with AT" if at else "ours without AT", value=a))
        ax.text(xs.mean(), 0.96, disc, ha="center", va="top", fontsize=10, transform=ax.get_xaxis_transform())
    ax.set_yscale("log")
    ax.set_ylim(0.3, 3000)
    ax.set_xticks(np.r_[x1, x1 + 5], [f"1s$_{{{l[2:]}}}$" for l in ORDER_1S] * 2)
    ax.set_title("1s levels of their discharges\n(symbols as in the first panel)", fontsize=10.5)
    style(ax, "$n/(g\\,n_g)$ [ppm]")

    # (d) our measured 2p distribution
    ax = axs[1, 0]
    L = levs["microwave", True]
    order = [l for l in ORDER_2P if l in set(L.level)]
    grp = [GROUP_2P[ORDER_2P.index(l)] for l in order]
    xo = xpos(grp)
    Li = L.set_index("level").loc[order]
    Ld = levs["dc", True].set_index("level").loc[order]
    to100 = lambda ln: 100 * np.exp(ln - ln.max()) / np.exp(ln - ln.max()).sum()
    meas = to100(Li.ln_meas)                                        # eta of the microwave fit at n_c
    thin = to100(Li.ln_meas + np.log(Li.eta_nc))                    # eta = 1
    dc_eta = to100(Li.ln_meas + np.log(Li.eta_nc) - np.log(Ld.eta_nc))
    lo, hi = meas * np.exp(-Li.ln_meas_sd), meas * np.exp(Li.ln_meas_sd)
    ccp = norm({l: Z[2, "CCP 100 Pa", l, "OES"] for l in order}, order)
    ax.bar(xo, ccp, width=0.62, color=COL["theirs"], label="Zhu & Pu CCP 100 Pa, OES")
    ax.vlines(xo - 0.18, np.minimum(thin, dc_eta), np.maximum(thin, dc_eta), color=COL["meas"], lw=7, alpha=0.3,
              label="measured, $\\eta$ = 1 to $\\eta$ of the DC-EEDF fit at $n_c$")
    ax.vlines(xo - 0.18, lo, hi, color=COL["meas"], lw=2)
    ax.plot(xo - 0.18, meas, "D", ms=8, color=COL["meas"],
            label=f"measured, $\\eta$ of the microwave fit at $n_c$ ({int(Li.n_spectra.iloc[0])} spectra, $\\pm$1 SD)")
    for key, mk, lab in (("ln_model_nc", "o", "our CR model, $N_e = n_c$"),
                         ("ln_model_free", "s", "our CR model, free fit")):
        v = np.exp(Li[key])
        ax.plot(xo + 0.18, 100 * v / v.sum(), mk, ms=8, color=COL["at"], mfc=COL["at"] if mk == "o" else "white",
                mew=1.8, label=lab)
        for l, a in zip(order, 100 * v / v.sum()):
            rows.append(dict(panel="our discharge (microwave EEDF, AT)", quantity="n/g, sum of the 7 = 100", level=l,
                             source=lab.replace("$N_e = n_c$", "Ne = n_c"), value=a))
    for l, a, t, e, b in zip(order, meas, thin, dc_eta, ccp):
        rows.append(dict(panel="our discharge (microwave EEDF, AT)", quantity="n/g, sum of the 7 = 100", level=l,
                         source="measured", value=a, value_eta1=t, value_eta_dc=e, zhupu_oes=b,
                         eta_nc=Li.loc[l, "eta_nc"], eta_nc_dc=Ld.loc[l, "eta_nc"], ln_sd=Li.loc[l, "ln_meas_sd"]))
    ax.set_xticks(xo, [f"2p$_{{{l[2:]}}}$\n{Li.loc[l, 'wl']:.1f}" for l in order], fontsize=9)
    F = fits["microwave", True]
    ax.set_title(f"Our discharge (1 Torr Ar, microwave EEDF, with Ar-atom transfer)\n"
                 f"free fit $N_e$ = {F['s']['Ne_med']:.1e} m$^{{-3}}$; $n_c$ = {cdf.N_C:.1e} m$^{{-3}}$; "
                 f"measured with $\\eta$ at $n_c$", fontsize=10.5)
    style(ax, "$n/g$ (sum of these seven levels = 100)")
    ax.set_ylim(0, 52)
    ax.legend(fontsize=8, loc="upper left", ncol=2, frameon=False)

    # (e) residuals at n_c with / without the transfer
    ax = axs[1, 1]
    ax.axhline(0, color="0.5", lw=1)
    for (eedf, at), L in levs.items():
        Li = L.set_index("level").loc[order]
        dx = (0.09 if at else -0.09) + (0.0 if eedf == "microwave" else 0.27) - 0.135
        key = "at" if at else "noat"
        ax.plot(xo + dx, Li.resid_nc, "o" if eedf == "microwave" else "^", ms=8, color=COL[key],
                mfc=COL[key] if eedf == "microwave" else "white", mew=1.8,
                label=f"{eedf} EEDF, {'with' if at else 'without'} Ar-atom transfer")
        for l, v in Li.resid_nc.items():
            rows.append(dict(panel="residual at n_c", quantity="ln(measured/model), mean removed", level=l,
                             source=f"{eedf}, {'with' if at else 'without'} AT", value=v))
    ax.set_xticks(xo, [f"2p$_{{{l[2:]}}}$\n{l in ('2p1', '2p2', '2p3', '2p4') and 'primed' or ''}" for l in order],
                  fontsize=9)
    ax.set_title("Our discharge at $N_e = n_c$: ln(measured / CR model) per level\n"
                 "(mean over the seven removed; > 0: model too weak)", fontsize=10.5)
    style(ax, "ln(measured / model)")
    ax.set_ylim(-1.1, 1.5)
    ax.legend(fontsize=8, loc="upper left", ncol=2, frameon=False)

    # (f) 1s of our discharge
    ax = axs[1, 2]
    one, ng = s1s["microwave", True]
    ccp_crm = np.array([Z[1, "CCP 100 Pa", l, "CRM"] for l in ORDER_1S])
    ccp_oes = np.array([Z[1, "CCP 100 Pa", l, "OES"] for l in ORDER_1S])
    ax.bar(x1, ccp_crm, width=0.62, color=COL["theirs"], label="Zhu & Pu CCP 100 Pa, CRM")
    ax.plot(x1, ccp_oes, "*", ms=13, mfc="white", mec=COL["ink"], mew=1.3, label="Zhu & Pu CCP 100 Pa, OES")
    a_lo, a_hi = [1e6 * n / 5 / ng for n in cdf.MEASURED_1S5]
    ax.vlines(-0.18, a_lo, a_hi, color=COL["meas"], lw=6, label="our 1s$_5$, absorption estimate")
    for key, mk, lab in (("nc", "o", "our CR model, $N_e = n_c$"), ("free", "s", "our CR model, free fit")):
        v = np.array([one[l, key] for l in ORDER_1S])
        ax.plot(x1 + 0.18, v, mk, ms=8, color=COL["at"], mfc=COL["at"] if key == "nc" else "white", mew=1.8, label=lab)
        for l, a in zip(ORDER_1S, v):
            rows.append(dict(panel="our discharge (microwave EEDF, AT)", quantity="n/g/n_g [ppm]", level=l,
                             source=f"CR model {key}", value=a))
    one_dc, _ = s1s["dc", True]
    v = np.array([one_dc[l, "nc"] for l in ORDER_1S])
    ax.plot(x1 + 0.18, v, "^", ms=9, color=COL["at"], mfc="white", mew=1.8, label="our CR model, DC EEDF, $N_e = n_c$")
    for l, a in zip(ORDER_1S, v):
        rows.append(dict(panel="our discharge (DC EEDF, AT)", quantity="n/g/n_g [ppm]", level=l,
                         source="CR model nc", value=a))
    rows.append(dict(panel="our discharge", quantity="n/g/n_g [ppm]", level="1s5", source="absorption lo-hi",
                     value=a_lo, value_hi=a_hi))
    ax.set_yscale("log")
    ax.set_ylim(3e-4, 3e4)
    ax.set_xticks(x1, [f"1s$_{{{l[2:]}}}$" for l in ORDER_1S])
    ax.set_title("1s levels of our discharge (with Ar-atom transfer)", fontsize=10.5)
    style(ax, "$n/(g\\,n_g)$ [ppm]")
    ax.legend(fontsize=7.5, loc="upper right", ncol=2, frameon=False, columnspacing=0.8, handletextpad=0.4)

    fig.suptitle("Zhu & Pu, J. Phys. D 43, 015204 (2010) and this work: Ar 1s and 2p populations "
                 "(their Figs. 1-2 digitised)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUTDIR, "zhupu_comparison.png"), dpi=140)
    plt.close(fig)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUTDIR, "zhupu_comparison.csv"), index=False)
    summary = pd.DataFrame([dict(eedf=e, atom_transfer=at, Ne_free=F["s"]["Ne_med"], Te_free=F["s"]["Te_med"],
                                 chi2_red_free=F["s"]["chi2_red"], Te_nc=F["r"]["Te_pinned"],
                                 chi2_red_nc=F["r"]["chi2_red_pinned"], dchi2_nc=F["r"]["delta_chi2_at_nc"],
                                 rms_resid_nc=np.sqrt(np.mean(levs[e, at].resid_nc ** 2)))
                            for (e, at), F in fits.items()])
    summary.to_csv(os.path.join(OUTDIR, "our_fits.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        for k, L in levs.items():
            print(k)
            print(L[["level", "wl", "A", "eta_nc", "eta_free", "ln_meas", "ln_meas_sd", "ln_model_nc", "ln_model_free",
                     "resid_nc"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"figure and tables in {OUTDIR}")
    return res, summary, levs


if __name__ == "__main__":
    RES = main()
