# -*- coding: utf-8 -*-
"""
NitrogenTrendsNearNc.py

Te and the N-atom density of the N2-admixture and power sweeps from the CR fit with the
microwave EEDFs, three Ar I lines and the intensity calibration trusted, at the local chi^2
minimum in Ne nearest the critical density n_c.

CR fit: CRFitNeTe with the 2.45 GHz BOLSIG+ EEDFs of MicrowaveLowENFit (from 3 Td) and the CR
model of each N2 fraction, Ar I lines LINES only (415.86 5p5, 706.72 4p8, 750.39 4p10: clean,
lowest repeat scatter, ArLineCloseups), response_deg = None (no tilt between the blue and red
ranges).  Per condition the fit is walked in Ne (chi^2 minimised over E/N at each Ne); within
n_c / NC_WINDOW - n_c NC_WINDOW the interior local minima are found and the one nearest n_c
whose Delta chi^2 is within 1 of the lowest of them is taken (none: the window minimum, flagged
'edge').  At that Ne: Te_eff with E/N refitted (16-84 % over E/N), E/N and the CR 1s5 density;
the Ne bar is the range where Delta chi^2 stays within 1 of the minimum.

With three lines the chi^2 alone hardly separates E/N branches at fixed Ne (pure Ar near n_c:
~18 Td with CR 1s5 ~6e17 and ~84 Td with 1s5 ~2e19 fit alike), so a second version adds the
absorption 1s5 as a chi^2 term, ((ln n_1s5,CR - ln n_meas) / sigma)^2 with n_meas the geometric
mean of MEASURED_1S5 and its range as +-1 sigma, before the same search near n_c.

N atoms: N I 744.2 nm actinometry (ActinometryNitrogenContent.condition_estimates) against the
same Ar lines, at that Ne and the E/N posterior there, for the Ar reference
  direct    excited from the ground state only (classic actinometry)
  stepwise  produced as in the CR model (k_Ar / f_direct of the CR model)
n_N = (n_N / n_Ar)(1 - f) N; complete dissociation would be n_N = 2 f N.

Output in Experimental_Data/Output/NitrogenTrendsNearNc/: trends_<version>.png (N density and
Te_eff against N2 % and against power; versions 'lines' and 'lines_1s5'), one figure per panel,
trends.csv, nN_near_nc.csv, nN_per_pair.csv.
Run from Spyder (F5) or python SmallAnalysisScripts/NitrogenTrendsNearNc.py (needs the CR
tables of MicrowaveLowENFit, built if missing).
"""
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import MicrowaveLowENFit as mlf                     # noqa: E402

crf, anc, he = mlf.crf, mlf.anc, mlf.he
als, ar = anc.als, anc.ar

# ---- settings -------------------------------------------------------------------
LINES = (415.859, 706.722, 750.387)                 # nm (air): CR fit and actinometry references
RESPONSE_DEG = None                                 # trust the intensity calibration
NC_WINDOW = 10.0                                    # search the local minimum in n_c / this - n_c x this
FRACTIONS = mlf.FRACTIONS                           # 0 (pure Ar, Te only) and the N2 fractions of the data
VERSIONS = {"lines": "Ar I lines only", "lines_1s5": "Ar I lines + measured 1s$_5$"}
LN_1S5 = 0.5 * np.log(mlf.MEASURED_1S5[0] * mlf.MEASURED_1S5[1])    # ln of the geometric mean
SIGMA_LN_1S5 = 0.5 * np.log(mlf.MEASURED_1S5[1] / mlf.MEASURED_1S5[0])   # range = +-1 sigma
VARIANTS = ("direct", "stepwise")
VSTYLE = {"direct": dict(color="C0", marker="o", ls="-", label="direct excitation of the Ar reference"),
          "stepwise": dict(color="C3", marker="s", ls="--", label="+ stepwise (CR model) Ar reference")}
FONT = 12
OUTDIR = os.path.join(mlf.ROOT_DIR, "Experimental_Data", "Output", "NitrogenTrendsNearNc")


# ---- CR fit near n_c --------------------------------------------------------------
def fit_cfg(pct):
    """Microwave CR table of pct % N2 with only LINES and the calibration trusted."""
    cfg = dict(mlf.job_cfg(f"lowEN:{pct:g}"), response_deg=RESPONSE_DEG)
    with contextlib.redirect_stdout(io.StringIO()):
        tab = crf.build_model_table(cfg)
    _, comps = crf.select_features(tab, cfg)
    drop = [w for w in comps.wl_air.round(3) if not np.isclose(LINES, w, atol=1e-3).any()]
    cfg["exclude_wl"] = tuple(cfg["exclude_wl"]) + tuple(drop)
    return cfg


def near_nc(p):
    """Index (fine Ne grid) of the local minimum nearest n_c, and whether it is only a window edge."""
    d2 = p.dchi2.to_numpy()
    w = np.flatnonzero(np.abs(np.log(p.Ne.to_numpy() / mlf.N_C)) <= np.log(NC_WINDOW))
    mins = [j for j in w[1:-1] if d2[j] <= d2[j - 1] and d2[j] <= d2[j + 1] and d2[j] < max(d2[j - 1], d2[j + 1])]
    if not mins:
        return int(w[np.argmin(d2[w])]), True
    best = min(d2[j] for j in mins)
    ok = [j for j in mins if d2[j] <= best + 1.0]
    return int(min(ok, key=lambda j: abs(np.log(p.Ne[j] / mlf.N_C)))), False


def fit_condition(fit, sw, x, with_1s5):
    """Free fit, Ne walk and the fit at the local minimum near n_c of one condition; with_1s5
    adds the absorption 1s5 to chi^2."""
    cfg = fit["cfg"]
    d = fit["ft"][(fit["ft"].sweep == sw) & (fit["ft"].x == x)]
    s, post, _ = crf.fit_block(d, fit["M"], fit["feats"], fit["grid"], fit["Te_f"], cfg)
    chi2_lines = mlf.cdf.chi2_grid(post, s["chi2_min"], s["birge"])
    pull = ((fit["fields"]["n_1s5"] - LN_1S5) / SIGMA_LN_1S5) ** 2
    chi2 = chi2_lines + pull if with_1s5 else chi2_lines
    (fX, fN), b2 = fit["grid"], s["birge"] ** 2
    ix, iN = np.argmin(chi2, axis=0), np.arange(len(fN))
    T2 = np.broadcast_to(fit["Te_f"][:, None], chi2.shape) if np.ndim(fit["Te_f"]) == 1 else fit["Te_f"]
    p = pd.DataFrame(dict(Ne=np.exp(fN), dchi2=(chi2[ix, iN] - np.min(chi2)) / b2, EN=np.exp(fX[ix]),
                          Te_eff=T2[ix, iN], n_1s5=np.exp(fit["fields"]["n_1s5"][ix, iN])))
    j, edge = near_nc(p)
    col = chi2[:, j]
    m = np.min(col)
    s2 = max(1.0, chi2_lines[np.argmin(col), j] / (s["dof"] + 1))
    w = np.exp(-0.5 * (col - m) / s2)
    pin = np.zeros_like(post)
    pin[:, j] = w / w.sum()
    o = np.argsort(T2[:, j])
    q = mlf.crf._wquantile(T2[o, j], pin[o, j], [0.16, 0.5, 0.84])
    d2 = p.dchi2.to_numpy()
    lo = hi = j                                      # contiguous Delta chi^2 <= +1 around the minimum
    while lo > 0 and d2[lo - 1] <= d2[j] + 1:
        lo -= 1
    while hi < len(d2) - 1 and d2[hi + 1] <= d2[j] + 1:
        hi += 1
    i = int(np.argmin(col))
    row = dict(sweep=sw, x=x, N2_percent=anc.n2_percent(sw, x), Ne=p.Ne[j], Ne_lo=p.Ne[lo], Ne_hi=p.Ne[hi],
               edge=edge, Te=q[1], Te_lo=q[0], Te_hi=q[2], EN=p.EN[j], n_1s5=p.n_1s5[j],
               chi2_red_lines=chi2_lines[i, j] / (s["dof"] + 1), pull_1s5=np.sqrt(pull[i, j]), dchi2=d2[j],
               Ne_free=s["Ne_best"], Te_free=s["Te_best"], chi2_red_free=s["chi2_red"], n_lines=int(d.feature.nunique()))
    cond = dict(sweep=sw, x=x, Te_med=q[1], Ne_med=p.Ne[j], Ne_lo=p.Ne[lo], Ne_hi=p.Ne[hi], x_med=p.EN[j],
                chi2_red=row["chi2_red_lines"], edge_Ne=edge, edge_x=False)
    return row, pin, cond


def fit_all():
    """Both versions for every condition -> (rows, {version: {('microwave', pct): fit}})."""
    rows, fits = [], {v: {} for v in VERSIONS}
    for pct in FRACTIONS:
        cfg = fit_cfg(pct)
        fit = mlf.prepare(cfg, with_nu=False)
        with contextlib.redirect_stdout(io.StringIO()):
            fit["eedfs"] = crf.eedf_axis(cfg)[0]
        for version in VERSIONS:
            posts, conds = {}, []
            for sw, x in mlf.conditions_of(fit["ft"], pct):
                row, pin, cond = fit_condition(fit, sw, x, version == "lines_1s5")
                rows.append(dict(version=version, **row))
                posts[f"{sw}|{x:g}"] = pin
                conds.append(cond)
                print(f"  {version:<9s} {sw:<12s} {x:<5g} Ne {row['Ne']:.2e} ({row['Ne_lo']:.1e}-{row['Ne_hi']:.1e})"
                      f"{' edge' if row['edge'] else ''}  Te_eff {row['Te']:.2f} eV  E/N {row['EN']:.3g} Td  "
                      f"1s5 {row['n_1s5']:.1e}  chi2/dof(lines) {row['chi2_red_lines']:.2f}")
            fits[version]["microwave", pct] = dict(fit, posts=posts, cond=pd.DataFrame(conds))
    return pd.DataFrame(rows), fits


# ---- actinometry --------------------------------------------------------------------
def nitrogen(fits):
    acfg = anc.CONFIG["actinometry"]
    xs = ar.load_all_cross_sections()
    with contextlib.redirect_stdout(io.StringIO()):
        nl, al = als.select_lines(acfg, xs)
        meas = als.filter_export_flags(als.measure_sweeps(pd.concat([nl, al]), acfg), acfg)
        _, cm = als.pair_ratios(meas, nl, al, acfg)
    keep = lambda wl: np.isclose(np.asarray(wl)[:, None], LINES, atol=1e-3).any(axis=1)
    fitted = np.array([("microwave", anc.n2_percent(s, x)) in fits for s, x in zip(cm.sweep, cm.x)])
    act = dict(xs=xs, n_lines=nl, a_lines=al[keep(al.wl_air)], cond=cm[keep(cm.ar_wl) & fitted])
    cfg = dict(anc.CONFIG, eedfs={"microwave": anc.CONFIG["eedfs"]["microwave"]}, variants=VARIANTS)
    with contextlib.redirect_stdout(io.StringIO()):
        pairs, cond = anc.condition_estimates({k: v for k, v in fits.items() if k[1] > 0}, act, cfg)
    return pairs, cond


# ---- plots ----------------------------------------------------------------------
def panel(ax, res, nn, sweep, key, xlabel):
    N = he.Torr2Volume(anc.CONFIG["fit"]["P_Torr"], anc.CONFIG["fit"]["Tg"])
    if key == "n_N":
        for v in VARIANTS:
            g = nn[(nn.sweep == sweep) & (nn.variant == v)].sort_values("x")
            st = VSTYLE[v]
            ax.errorbar(g.x, g.n_N, yerr=[g.n_N - g.n_N_lo, g.n_N_hi - g.n_N], color=st["color"], marker=st["marker"],
                        ls=st["ls"], capsize=3, lw=1.6, ms=6, label=st["label"])
        if sweep == "N2 fraction":
            f = np.linspace(0.4, 3.1, 50) / 100
            ax.plot(100 * f, 2 * f * N, "k:", lw=1.2, label="complete dissociation (2 f N)")
        else:
            ax.axhline(2 * anc.CONFIG["power_sweep_N2"] / 100 * N, color="k", ls=":", lw=1.2,
                       label="complete dissociation (2 f N)")
        ax.set_yscale("log")
        ax.set_ylabel("N-atom density [m$^{-3}$]", fontsize=FONT)
    else:
        g = res[res.sweep == sweep].sort_values("x")
        ax.errorbar(g.x, g.Te, yerr=[g.Te - g.Te_lo, g.Te_hi - g.Te], color="C3", marker="o", capsize=3, lw=1.8, ms=6,
                    label=r"$N_e$ at the local minimum near $n_c$")
        e = g[g.edge]
        if len(e):
            ax.plot(e.x, e.Te, "o", mfc="white", mec="C3", ms=9, label="no local minimum within 10x of n$_c$ (window edge)")
        ar0 = res[(res.N2_percent == 0)]
        if sweep == "N2 fraction" and len(ar0):
            ax.axhline(ar0.Te.iloc[0], color="0.5", ls="--", lw=1.2, label=f"pure Ar ({ar0.Te.iloc[0]:.2f} eV)")
        ax.set_ylim(0, None)
        ax.set_ylabel(r"$T_{e,\mathrm{eff}} = \frac{2}{3}\langle\varepsilon\rangle$ [eV]", fontsize=FONT)
    ax.set_xlabel(xlabel, fontsize=FONT)
    ax.tick_params(labelsize=FONT - 2)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=FONT - 4)


def plot(res, nn, version):
    sweeps = [(s["name"], s["xlabel"]) for s in crf.CONFIG["sweeps"]]
    title = (f"Microwave EEDF, {VERSIONS[version]} (415.86 / 706.72 / 750.39 nm, calibration trusted), "
             r"$N_e$ at the local $\chi^2$ minimum near $n_c$")
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
    for j, (sw, xl) in enumerate(sweeps):
        panel(axs[0, j], res, nn, sw, "n_N", xl)
        panel(axs[1, j], res, nn, sw, "Te", xl)
        axs[0, j].set_title(f"{sw} sweep", fontsize=FONT + 1)
    fig.suptitle(title, fontsize=FONT - 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"trends_{version}.png"), dpi=170)
    plt.close(fig)
    for sw, xl in sweeps:
        for key, name in (("n_N", "N_density"), ("Te", "Te")):
            fig, ax = plt.subplots(figsize=(7.5, 5.2))
            panel(ax, res, nn, sw, key, xl)
            ax.set_title(f"{sw} sweep, {VERSIONS[version]}", fontsize=FONT - 1)
            fig.tight_layout()
            fig.savefig(os.path.join(OUTDIR, f"{name}_vs_{sw.split()[0]}_{version}.png"), dpi=170)
            plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    mlf.build_tables([f"lowEN:{p:g}" for p in FRACTIONS])
    print("CR fit, Ne at the local minimum near n_c ...")
    res, fits = fit_all()
    nns, pairs_all = [], []
    for version in VERSIONS:
        pairs, cond = nitrogen(fits[version])
        r = res[res.version == version]
        nn = cond.merge(r[["sweep", "x", "Ne", "Te"]], on=["sweep", "x"], suffixes=("_act", "")).assign(version=version)
        plot(r, nn, version)
        nns.append(nn)
        pairs_all.append(pairs.assign(version=version))
    nn, pairs = pd.concat(nns, ignore_index=True), pd.concat(pairs_all, ignore_index=True)
    res.to_csv(os.path.join(OUTDIR, "trends.csv"), index=False)
    nn.to_csv(os.path.join(OUTDIR, "nN_near_nc.csv"), index=False)
    pairs.to_csv(os.path.join(OUTDIR, "nN_per_pair.csv"), index=False)
    cols = ["version", "sweep", "x", "variant", "n_pairs", "n_N", "n_N_lo", "n_N_hi", "dissociation"]
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(nn[cols].sort_values(["version", "sweep", "x", "variant"]).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nfigures in {OUTDIR}")
    return res, nn


if __name__ == "__main__":
    RES, NN = main()
