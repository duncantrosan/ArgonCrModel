# -*- coding: utf-8 -*-
"""
ActinometryNitrogenContent.py

N-atom content n_N/n_Ar of every sweep condition from N I / Ar I actinometry, with the
plasma state from the CR-model fit and the EEDF of each Ar/N2 mixture (BOLSIG+, two-term:
MultiBolt with 2-8 terms gives the same EEDF in Ar + 10 % N2), for two EEDFs
  microwave  2.45 GHz field (the discharge) - main result
  dc         DC field - for the EEDF systematic
and two variants for the population of the Ar reference level:

  direct    excitation from the Ar ground state only (classic actinometry):
                n_N/n_Ar = (I_N/I_Ar) (k_Ar b_Ar)/(k_N b_N) (lam_N/lam_Ar)
  stepwise  the Ar upper level is also populated from the 4s levels (metastable and
            resonant densities of the CR fit) and by cascades, i.e. the CR model's total
            production: k_Ar -> k_Ar / f_direct, with f_direct the direct share of the
            production at the fitted (Te, Ne):
                n_N/n_Ar = (direct value) / f_direct

The N I upper level is excited from ground-state N atoms only: no dissociative
excitation (e + N2 -> N* + N + e) and no stepwise excitation through N(2D, 2P).
(lam_N/lam_Ar converts the calibrated energy intensities to photon rates.)

Steps
1. Mixture EEDFs: a BOLSIG+ E/N sweep (Biagi Ar + N2, no e-e / superelastics) for each
   N2 fraction of the data and each EEDF, InputData/Bolsig/ArN2_<f>pct_bolsig[_mw]
   (built on first use, ~1 min each).
2. CR fit (CRFitNeTe) of each condition with the EEDFs and the CR model (N2 quenching of
   the Ar(4s) levels + Ar dilution) of its own N2 fraction -> posterior over (E/N, Ne).
3. Per condition, pair and variant the posterior is pushed through
       ln(n_N/n_Ar)(x, Ne) = ln(I_N/I_Ar b_Ar/b_N lam_N/lam_Ar) - ln(k_N/k_Ar) [- ln f_direct]
   -> median and 16-84 % per pair.
4. The pairs of a condition are combined at every (x, Ne) point (weights from their
   line-ratio errors) before the posterior quantiles are taken, so the Te/Ne error they
   share is not averaged down; the line-ratio error of the weighted mean is added, scaled
   by the Birge ratio when the pairs disagree beyond their errors.  The cross-section
   errors (N 20 %, Ar 15 %, as in ActinometryRates) are common to all conditions and are
   given separately as a systematic.
Only N I lines with a known branching ratio give n_N/n_Ar (744.2 nm with the current
line list).  With the N2 feed fraction f, n_N2/n_Ar = f/(1-f) if little N2 is
dissociated, so the dissociation degree is n_N/(2 n_N2) = (n_N/n_Ar)(1-f)/(2f), and the
N-atom density is n_N = (n_N/n_Ar)(1-f) N, N from P_Torr and Tg of the fit (300 K).

Run from Spyder: edit CONFIG, press F5.  Results in Experimental_Data/Output/ActinometryN2Content
(ActinometryN2Content_RDW4s with the RDW 4s cross sections, CRFitNeTe.CONFIG['xsec_4s']).
"""
import os
import sys
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ActinometryRates as ar                       # noqa: E402
import ActinometryLineSelection as als              # noqa: E402
import ActinometryTeUncertainty as atu              # noqa: E402
import CRFitNeTe as crf                             # noqa: E402

he = crf._helpers()

CONFIG = dict(
    fit=crf.CONFIG,
    actinometry=als.CONFIG,
    variants=("direct", "stepwise"),
    power_sweep_N2=2.4,                 # % N2 of the power sweep (PowerSweepN2_2.4_Ar77_6_1Torr)
    eedfs=dict(                         # name: library suffix, field frequency [Hz], E/N grid [Td]
        microwave=dict(suffix="_mw", freq_Hz=2.45e9,
                       EN_Td=[6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 130, 170, 210, 250,
                              300, 350, 400, 500, 600, 700, 850, 1000]),
        dc=dict(suffix="", freq_Hz=0.0,     # as Scripts/MainFileWithNitrogen.py (its libraries are reused)
                EN_Td=[3, 4, 5, 6, 7, 8.5, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]),
    ),
    Ne_grid=np.geomspace(1e13, 3e19, 34),   # m^-3; the microwave fits reach 1e15
    bolsig_settings=dict(n_grid=500, precision=1e-25, max_iter=20000),
    rel_err_xs_N=0.20, rel_err_xs_Ar=0.15,   # cross sections (BSR N I, Ar I), systematic
    k_margin_eV=1.0,                    # k valid only where the EEDF reaches threshold + this
    min_valid_mass=0.9,                 # posterior mass with a valid result needed per pair
    outdir=crf.output_dir("ActinometryN2Content"),     # + crf.model_suffix: "_RDW4s", "_noAT", ... for other models
)


# ----------------------------------------------------------------------------
# 1. Mixture EEDFs and CR fits
# ----------------------------------------------------------------------------
def n2_percent(sweep, x, cfg=CONFIG):
    return float(x) if sweep == "N2 fraction" else cfg["power_sweep_N2"]


def mixture_library(pct, eedf, cfg=CONFIG):
    """Folder of the BOLSIG+ E/N sweep for pct % N2 in Ar with EEDF `eedf` (built if missing);
    0 % is pure Ar, the Biagi Ar libraries of Scripts/CreateEEDFLibraries.py."""
    e = cfg["eedfs"][eedf]
    if pct == 0:
        return he.BOLSIG_FOLDER / f"Ar_Biagi_bolsig{e['suffix']}"
    name = f"ArN2_{pct:g}pct_bolsig{e['suffix']}"
    folder = he.BOLSIG_FOLDER / name
    if not he.IsBolsigLibrary(folder):
        xsec = he.BOLSIG_XSEC_FOLDER / "Biagi_ArN2.txt"
        if not xsec.is_file():
            raise FileNotFoundError(f"{xsec} missing - run Scripts/MainFileWithNitrogen.py with "
                                    "EEDF_MODE = 'bolsig' once to make it")
        N = he.Torr2Volume(cfg["fit"]["P_Torr"], cfg["fit"]["Tg"])
        extra = dict(omega_N=2 * np.pi * e["freq_Hz"] / N) if e["freq_Hz"] else {}
        print(f"BOLSIG+ ({eedf}) for {pct:g} % N2 -> {folder}")
        he.RunBolsig(xsec, name, e["EN_Td"], species=["Ar", "N2"], fractions=[1 - pct / 100, pct / 100],
                     verbose=False, **cfg["bolsig_settings"], **extra)
    return folder


def fit_config(pct, eedf, cfg=CONFIG):
    """CRFitNeTe settings of the fit with the EEDFs and the CR model of pct % N2."""
    return dict(cfg["fit"], eedf=str(mixture_library(pct, eedf, cfg)), N2_percent=pct, Ne_grid=cfg["Ne_grid"],
                outdir=os.path.join(cfg["outdir"], f"fit_{eedf}_{pct:g}pct_N2"),
                measure_dir=cfg["fit"].get("measure_dir") or cfg["fit"]["outdir"])


def mixture_fit(pct, eedf, cfg=CONFIG):
    """CR fit of all conditions with the EEDFs and the CR model of pct % N2."""
    fcfg = fit_config(pct, eedf, cfg)
    print(f"\n=== CR fit, {eedf} EEDF, {pct:g} % N2 ({os.path.basename(fcfg['eedf'])}) ===")
    return crf.run(fcfg)


# ----------------------------------------------------------------------------
# 2. n_N / n_Ar per pair and per condition
# ----------------------------------------------------------------------------
class FitTerms:
    """ln k on the fine EEDF axis and ln f_direct on the fine (x, Ne) grid of one fit (cached)."""

    def __init__(self, fit, xs, cfg):
        self.fit, self.xs, self.cfg = fit, xs, cfg
        self.lev = list(fit["tab"]["levels"])
        self._k, self._f = {}, {}

    def lnk(self, species, upper):
        key = (species, upper)
        if key not in self._k:
            k = atu.rates_on_axis(self.fit, self.xs[species][upper], self.cfg["k_margin_eV"])[0]
            self._k[key] = np.log(np.where(k > 0, k, np.nan))
        return self._k[key]

    def ln_fdirect(self, upper):
        if upper not in self._f:
            tab = self.fit["tab"]
            f = np.nan_to_num(tab["fdirect"][:, :, self.lev.index(upper)], nan=1.0)
            self._f[upper] = np.minimum(crf.on_fine_grid(tab, np.log(np.clip(f, 1e-300, 1.0)), self.fit["cfg"], k=1), 0.0)
        return self._f[upper]


def pair_fields(terms, pairs, variant):
    """ln(n_N/n_Ar) on the fine (x, Ne) grid for each pair (rows of the measured ratios)."""
    out = []
    for p in pairs.itertuples():
        L = (terms.lnk("N I", p.n_upper) - terms.lnk("Ar I", p.ar_upper))[:, None]
        if variant == "stepwise":
            L = L + terms.ln_fdirect(p.ar_upper)
        out.append(np.log(p.I_ratio * p.bA / p.bN * p.lam) - L)
    return out


def _quantiles(field, post, q=(0.16, 0.5, 0.84)):
    field = np.broadcast_to(field, post.shape)       # the 'direct' fields do not depend on Ne
    ok = np.isfinite(field)
    return atu._wq(field[ok], post[ok], np.asarray(q)), post[ok].sum()


def estimate(terms, pairs, post, variant, cfg=CONFIG):
    """Per-pair rows and the combined ln(n_N/n_Ar) estimate of one condition (None if no valid pair)."""
    rows, keep = [], []
    for p, F in zip(pairs.itertuples(), pair_fields(terms, pairs, variant)):
        (lo, med, hi), mass = _quantiles(F, post)
        fd = np.exp(_quantiles(np.broadcast_to(terms.ln_fdirect(p.ar_upper), post.shape), post)[0][1])
        ok = mass >= cfg["min_valid_mass"]
        rows.append(dict(n_wl=p.n_wl, ar_wl=p.ar_wl, ar_upper=p.ar_upper, I_ratio=p.I_ratio,
                         I_rel_err=p.I_err / p.I_ratio, n_rep=p.n_rep, f_direct_Ar=fd, valid_mass=mass,
                         used=ok, nN_nAr=np.exp(med), nN_nAr_lo=np.exp(lo), nN_nAr_hi=np.exp(hi)))
        if ok:
            keep.append((F, (p.I_err / p.I_ratio) ** 2, med))
    if not keep:
        return rows, None
    w = np.array([1 / v for _, v, _ in keep])
    comb = sum(wi * F for wi, (F, _, _) in zip(w, keep)) / w.sum()   # NaN where any pair is invalid
    (lo, med, hi), _ = _quantiles(comb, post)
    meds = np.array([m for _, _, m in keep])
    M = (w * meds).sum() / w.sum()
    birge = np.sqrt((w * (meds - M) ** 2).sum() / (len(meds) - 1)) if len(meds) > 1 else 1.0
    s_meas = max(1.0, birge) / np.sqrt(w.sum())
    s_TeNe = (hi - lo) / 2
    s_stat = np.hypot(s_TeNe, s_meas)
    s_xs = np.hypot(cfg["rel_err_xs_N"], cfg["rel_err_xs_Ar"])
    return rows, dict(n_pairs=len(keep), nN_nAr=np.exp(med), ln_err_TeNe=s_TeNe, ln_err_meas=s_meas,
                      birge_pairs=birge, ln_err_stat=s_stat, ln_err_xs=s_xs,
                      ln_err_total=np.hypot(s_stat, s_xs))


def condition_estimates(fits, act, cfg=CONFIG):
    xs, nl, al = act["xs"], act["n_lines"], act["a_lines"]
    bmap = dict(zip(pd.concat([nl.wl_air, al.wl_air]).round(4), pd.concat([nl.branching, al.branching])))
    umap = dict(zip(pd.concat([nl.wl_air, al.wl_air]).round(4), pd.concat([nl.upper, al.upper])))
    c = act["cond"].rename(columns={"mean": "I_ratio", "err": "I_err"})
    c = c.assign(bN=c.n_wl.round(4).map(bmap), bA=c.ar_wl.round(4).map(bmap),
                 n_upper=c.n_wl.round(4).map(umap), ar_upper=c.ar_wl.round(4).map(umap),
                 lam=c.n_wl / c.ar_wl if cfg["fit"]["intensity_units"] == "energy" else 1.0)
    skipped = sorted(c[~np.isfinite(c.bN)].n_wl.unique())
    if skipped:
        print(f"N I lines without a branching ratio (trend only, not used): {skipped}")
    c = c[np.isfinite(c.bN) & np.isfinite(c.bA)]
    terms = {key: FitTerms(fit, xs, cfg) for key, fit in fits.items()}
    N = he.Torr2Volume(cfg["fit"]["P_Torr"], cfg["fit"]["Tg"])
    pair_rows, cond_rows = [], []
    for (sweep, x), pairs in c.groupby(["sweep", "x"]):
        pct = n2_percent(sweep, x, cfg)
        f = pct / 100
        for eedf, variant in product(cfg["eedfs"], cfg["variants"]):
            fit = fits[eedf, pct]
            post = fit["posts"].get(f"{sweep}|{x:g}")
            if post is None:
                print(f"no CR fit for {sweep} {x:g} ({eedf}) - skipped")
                continue
            meta = dict(sweep=sweep, x=x, N2_percent=pct, eedf=eedf, variant=variant)
            per_pair, est = estimate(terms[eedf, pct], pairs, post, variant, cfg)
            pair_rows += [dict(meta, **r) for r in per_pair]
            if est is None:
                continue
            fc = fit["cond"].set_index(["sweep", "x"]).loc[(sweep, x)]
            n, s = est["nN_nAr"], est["ln_err_stat"]
            cond_rows.append(dict(meta, Te_eff=fc.Te_med, Ne=fc.Ne_med, Ne_lo=fc.Ne_lo, Ne_hi=fc.Ne_hi,
                                  EN_Td=fc.x_med, chi2_red_fit=fc.chi2_red, edge_Ne=fc.edge_Ne,
                                  edge_EN=fc.edge_x, **est,
                                  nN_nAr_lo=n * np.exp(-s), nN_nAr_hi=n * np.exp(s),
                                  n_N=n * (1 - f) * N, n_N_lo=n * (1 - f) * N * np.exp(-s),
                                  n_N_hi=n * (1 - f) * N * np.exp(s),
                                  dissociation=n * (1 - f) / (2 * f)))
    return pd.DataFrame(pair_rows), pd.DataFrame(cond_rows)


# ----------------------------------------------------------------------------
# 3. Plots
# ----------------------------------------------------------------------------
VSTYLE = {"direct": dict(color="C0", marker="o", label="direct excitation only"),
          "stepwise": dict(color="C3", marker="s", label="+ stepwise (CR metastables, cascades)")}
ESTYLE = {"microwave": dict(ls="-", mfc=None, label="microwave EEDF"),
          "dc": dict(ls="--", mfc="white", label="DC EEDF")}


def _err(ax, x, y, s, v, e, dx=0.0, alpha=1.0, label=True):
    st, es = VSTYLE[v], ESTYLE.get(e, ESTYLE["microwave"])
    ax.errorbar(x + dx, y, yerr=[y - y * np.exp(-s), y * np.exp(s) - y], fmt=st["marker"], ls=es["ls"],
                color=st["color"], mfc=es["mfc"] or st["color"], ms=5, capsize=3, alpha=alpha,
                label=f"{es['label']}, {st['label']}" if label else None)


def plot_content(cond, sweeps, path):
    fig, axs = plt.subplots(3, len(sweeps), figsize=(7.2 * len(sweeps), 13), squeeze=False)
    combos = list(dict.fromkeys(zip(cond.eedf, cond.variant)))
    for j, sw in enumerate(sweeps):
        c = cond[cond.sweep == sw["name"]]
        span = np.ptp(c.x) if len(c) else 1
        for k, (e, v) in enumerate(combos):
            g = c[(c.eedf == e) & (c.variant == v)].sort_values("x")
            if g.empty:
                continue
            dx = (k - (len(combos) - 1) / 2) * 0.012 * span
            _err(axs[0, j], g.x, g.n_N, g.ln_err_total, v, e, dx, alpha=0.3, label=False)   # + cross sections
            _err(axs[0, j], g.x, g.n_N, g.ln_err_stat, v, e, dx)                             # statistical
            _err(axs[1, j], g.x, g.dissociation, g.ln_err_stat, v, e, dx, label=False)
        for e, es in ESTYLE.items():         # fitted Ne (same for both excitation variants)
            g = c[(c.eedf == e) & (c.variant == c.variant.iloc[0])].sort_values("x") if len(c) else c
            if g.empty:
                continue
            axs[2, j].errorbar(g.x, g.Ne, yerr=[g.Ne - g.Ne_lo, g.Ne_hi - g.Ne], fmt="o", ls=es["ls"], color="k",
                               mfc=es["mfc"] or "k", ms=5, capsize=3, label=f"{es['label']} (16-84 %)")
            edge = g[g.edge_Ne.astype(bool)]
            axs[2, j].plot(edge.x, edge.Ne, "rx", ms=10, mew=2, label="posterior at the Ne grid edge" if e == "microwave" else None)
        if sw["name"] == "N2 fraction":
            f = np.linspace(0.4, 3.1, 50) / 100
            axs[1, j].plot(100 * f, np.ones_like(f), "k:", lw=1, label="complete dissociation")
            axs[1, j].legend(fontsize=8)
        axs[0, j].set_ylabel("N-atom density $n_N$ [m$^{-3}$]")
        axs[1, j].set_ylabel(r"dissociation degree $n_N / (2 n_{N_2})$ (feed $N_2$)")
        axs[2, j].set_ylabel("$N_e$ of the CR fit [m$^{-3}$]")
        axs[0, j].set_title(f"{sw['name']} sweep ({sw['note']})")
        for ax in axs[:, j]:
            ax.set_yscale("log")
            ax.set_xlabel(sw["xlabel"])
            ax.grid(alpha=0.3, which="both")
        axs[0, j].legend(fontsize=7)
        axs[2, j].legend(fontsize=7)
    fig.suptitle("Atomic N from N I 744.2 nm actinometry (BOLSIG+ EEDF of each Ar/N$_2$ mixture, CR fit of Te, Ne; "
                 "$T_g$ = 300 K)\nerror bars: statistical (line ratios, Te/Ne posterior, pair scatter); "
                 "faint: + cross sections")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pairs(pairs, sweeps, eedf, path):
    pairs = pairs[pairs.used & (pairs.eedf == eedf)]
    refs = sorted(pairs.ar_wl.unique())
    cols = dict(zip(refs, plt.cm.tab10(np.linspace(0, 1, 10))))
    fig, axs = plt.subplots(3, len(sweeps), figsize=(6.8 * len(sweeps), 12.5), squeeze=False)
    for j, sw in enumerate(sweeps):
        p = pairs[pairs.sweep == sw["name"]]
        for i, v in enumerate(VSTYLE):
            for r in refs:
                g = p[(p.variant == v) & (p.ar_wl == r)].sort_values("x")
                if g.empty:
                    continue
                axs[i, j].errorbar(g.x, g.nN_nAr, yerr=[g.nN_nAr - g.nN_nAr_lo, g.nN_nAr_hi - g.nN_nAr],
                                   fmt="o-", ms=3, lw=1, capsize=2, color=cols[r],
                                   label=f"Ar {r:.1f} ({g.ar_upper.iloc[0]})")
                if i == 0:
                    axs[2, j].plot(g.x, g.f_direct_Ar, "o-", ms=3, color=cols[r])
            axs[i, j].set_ylabel(f"$n_N/n_{{Ar}}$, {VSTYLE[v]['label']}", fontsize=9)
        axs[2, j].set_ylabel("$f_{direct}$ of the Ar reference level (CR fit)", fontsize=9)
        axs[0, j].set_title(f"{sw['name']} sweep, {eedf} EEDF: each Ar reference (16-84 % from Te/Ne)")
        for ax in axs[:, j]:
            ax.set_yscale("log")
            ax.set_xlabel(sw["xlabel"])
            ax.grid(alpha=0.3, which="both")
    axs[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 4. Driver
# ----------------------------------------------------------------------------
def run(cfg=CONFIG):
    os.makedirs(cfg["outdir"], exist_ok=True)
    out = lambda f: os.path.join(cfg["outdir"], f)
    acfg = cfg["actinometry"]
    print("=== actinometry lines and measured ratios ===")
    xs = ar.load_all_cross_sections()
    n_lines, a_lines = als.select_lines(acfg, xs)
    meas = als.filter_export_flags(als.measure_sweeps(pd.concat([n_lines, a_lines]), acfg), acfg)
    _, cond_meas = als.pair_ratios(meas, n_lines, a_lines, acfg)
    act = dict(xs=xs, n_lines=n_lines, a_lines=a_lines, cond=cond_meas)
    fractions = sorted({n2_percent(s, x, cfg) for s, x in zip(cond_meas.sweep, cond_meas.x)})
    print(f"N2 fractions of the actinometry conditions: {fractions} %")
    fits = {(eedf, pct): mixture_fit(pct, eedf, cfg) for eedf in cfg["eedfs"] for pct in fractions}
    print("\n=== n_N / n_Ar ===")
    pairs, cond = condition_estimates(fits, act, cfg)
    pairs.to_csv(out("nN_nAr_per_pair.csv"), index=False)
    cond.to_csv(out("nN_nAr_per_condition.csv"), index=False)
    plot_content(cond, acfg["sweeps"], out("nitrogen_content.png"))
    for eedf in cfg["eedfs"]:
        plot_pairs(pairs, acfg["sweeps"], eedf, out(f"nitrogen_content_per_pair_{eedf}.png"))
    cols = ["sweep", "x", "eedf", "variant", "n_pairs", "Te_eff", "Ne", "chi2_red_fit", "edge_Ne", "nN_nAr",
            "n_N", "ln_err_stat", "ln_err_total", "birge_pairs", "dissociation"]
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(cond[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nwritten to", cfg["outdir"])
    return dict(fits=fits, act=act, pairs=pairs, cond=cond)


if __name__ == "__main__":
    RESULTS = run(CONFIG)
