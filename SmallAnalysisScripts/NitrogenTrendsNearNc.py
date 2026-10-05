# -*- coding: utf-8 -*-
"""
NitrogenTrendsNearNc.py

Te, the N-atom density and the N2 dissociation of the N2-admixture and power sweeps from the CR
fit with three Ar I lines and the intensity calibration trusted, at the local chi^2 minimum in
Ne nearest the critical density n_c.

CR fit: CRFitNeTe with, per fraction, EEDF_PURE_AR for pure Ar (2.45 GHz BOLSIG+ EEDFs of
MicrowaveLowENFit) and EEDF_N2 for Ar/N2 (DC BOLSIG+ EEDFs of ActinometryNitrogenContent: with N2
the microwave EEDFs cannot match the lines and the measured 1s5 together, the DC shape can), the
CR model of each N2 fraction, Ar I lines LINES only (415.86 5p5, 706.72 4p8, 750.39 4p10: clean,
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
n_N = (n_N / n_Ar)(1 - f) N; complete dissociation would be n_N = 2 f N, dissociation n_N / (2 f N).

N2 check: the N2 second positive (0,0) band at 337.1 nm (integral over N2C_BAND, baseline from
N2C_FLANK, the pure-Ar blank subtracted) against the same Ar lines, for N2(C, v=0)
  direct  made only by electron impact on N2(X, v=0): Biagi C3Pi (v=0-4) cross section x the
          Franck-Condon share N2C_FC00 of v'=0, branching N2C_B00 to B(v''=0); Ar reference direct
  CR      + Ar(4s) + N2 -> Ar + N2(C) (CR 4s densities x kQ of he.ImportArQuenchingData, a share
          N2C_TRANSFER_V0 into v'=0); Ar reference as in the CR model (k_Ar / f_direct)
  n_N2 / n_Ar = (I_337 / I_Ar)(lam_337 / lam_Ar) (N_e k_Ar b_Ar [/ f_direct]) / (b_00 P_C / n_N2)
and dissociation = 1 - n_N2 / (f N) (quenching of N2(C) at 1 Torr is < 0.1 % of its decay).

Output in Experimental_Data/Output/NitrogenTrendsNearNc/: trends_<version>.png (N density, Te_eff
and dissociation against N2 % and against power; versions 'lines' and 'lines_1s5'), one figure per
panel, trends.csv, nN_near_nc.csv, nN_per_pair.csv, n2c_near_nc.csv.
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
EEDF_PURE_AR = "lowEN"                              # MicrowaveLowENFit job kind for 0 % N2 (microwave)
EEDF_N2 = "dc"                                      # ... for the Ar/N2 fractions ('dc' or 'lowEN' = microwave)
EEDF_LABEL = {"lowEN": "microwave", "dc": "DC"}
N2C_BAND = (333.95, 337.25)                         # nm: SPS (0,0), degraded to the violet, above the (1,1) head
N2C_FLANK = (337.35, 338.6)                         # nm: baseline
N2C_LAMBDA = 337.13                                 # nm
N2C_FC00 = 0.55                                     # X(v=0) -> C(v'=0) Franck-Condon share of C(v'=0-4)
N2C_B00 = 0.46                                      # C(v'=0) -> B(v''=0) branching ratio
N2C_TRANSFER_V0 = 0.5                               # share of Ar(4s) + N2 -> N2(C) landing in v'=0
N2C_STYLE = {"direct": dict(color="C2", marker="^", ls="-", label="N$_2$(C) 337 nm, electron impact only"),
             "CR": dict(color="C4", marker="v", ls="--", label="N$_2$(C) 337 nm, + Ar(4s) transfer, CR Ar reference")}
VERSIONS = {"lines": "Ar I lines only", "lines_1s5": "Ar I lines + measured 1s$_5$"}
LN_1S5 = 0.5 * np.log(mlf.MEASURED_1S5[0] * mlf.MEASURED_1S5[1])    # ln of the geometric mean
SIGMA_LN_1S5 = 0.5 * np.log(mlf.MEASURED_1S5[1] / mlf.MEASURED_1S5[0])   # range = +-1 sigma
VARIANTS = ("direct", "stepwise")
VSTYLE = {"direct": dict(color="C0", marker="o", ls="-", label="direct excitation of the Ar reference"),
          "stepwise": dict(color="C3", marker="s", ls="--", label="+ stepwise (CR model) Ar reference")}
FONT = 12
OUTDIR = os.path.join(mlf.ROOT_DIR, "Experimental_Data", "Output", "NitrogenTrendsNearNc")


# ---- CR fit near n_c --------------------------------------------------------------
def eedf_kind(pct):
    return EEDF_PURE_AR if pct == 0 else EEDF_N2


def fit_cfg(pct):
    """CR table of pct % N2 (EEDF of eedf_kind) with only LINES and the calibration trusted."""
    cfg = dict(mlf.job_cfg(f"{eedf_kind(pct)}:{pct:g}"), response_deg=RESPONSE_DEG)
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
    """Both versions for every condition -> (rows, {version: {(eedf kind, pct): fit}})."""
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
            fits[version][eedf_kind(pct), pct] = dict(fit, posts=posts, cond=pd.DataFrame(conds))
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
    fitted = np.array([(EEDF_N2, anc.n2_percent(s, x)) in fits for s, x in zip(cm.sweep, cm.x)])
    act = dict(xs=xs, n_lines=nl, a_lines=al[keep(al.wl_air)], cond=cm[keep(cm.ar_wl) & fitted])
    eedfs = {EEDF_N2: anc.CONFIG["eedfs"]["dc" if EEDF_N2 == "dc" else "microwave"]}
    cfg = dict(anc.CONFIG, eedfs=eedfs, variants=VARIANTS)
    with contextlib.redirect_stdout(io.StringIO()):
        pairs, cond = anc.condition_estimates({k: v for k, v in fits.items() if k[1] > 0}, act, cfg)
    return pairs, cond


def n2c_cross_section():
    """Biagi e + N2(X) -> N2(C3Pi_u, v=0-4) from the BOLSIG+ Ar/N2 file, as an ActinometryRates xs."""
    lines = (he.BOLSIG_XSEC_FOLDER / "Biagi_ArN2.txt").read_text(errors="replace").splitlines()
    i = next(k for k, l in enumerate(lines) if l.replace(" ", "") == "N2->N2(C3PIV=0-4)")
    a = next(k for k in range(i, len(lines)) if lines[k].startswith("-----"))
    b = next(k for k in range(a + 1, len(lines)) if lines[k].startswith("-----"))
    E, s = np.array([[float(v) for v in l.split()[:2]] for l in lines[a + 1:b]]).T
    return dict(species="N2", label="C3Pu(v=0-4)", energy_eV=E, cross_section_m2=s,
                threshold_eV=float(lines[i + 1].split()[0]))


def band_ratios(ft):
    """Per spectrum and Ar line: 337 nm band integral / Ar line area (energy units), the mean
    pure-Ar value (blank) subtracted."""
    folders = {s["name"]: s["folder"] for s in crf.CONFIG["sweeps"]}
    arn = als.arn
    rows = []
    for (sw, f), g in ft.groupby(["sweep", "file"]):
        sp = arn.load_spectrum(os.path.join(folders[sw], f + ".spa"), f, arn.CONFIG)
        x, y = sp.window(*N2C_BAND)
        _, yf = sp.window(*N2C_FLANK)
        band = np.trapezoid(y - np.median(yf), x)
        rows += [dict(sweep=sw, file=f, x=r.x, ar_wl=r.wl, R=band / r.area) for r in g.itertuples()]
    R = pd.DataFrame(rows)
    blank = R[(R.sweep == "N2 fraction") & (R.x == 0)].groupby("ar_wl").R.mean()
    return R.assign(R=R.R - R.ar_wl.map(blank), blank=R.ar_wl.map(blank))


def nitrogen_c(fits, R):
    """n_N2 / n_Ar and the dissociation from the 337 nm band at the fit near n_c (both variants)."""
    xs = ar.load_all_cross_sections()["Ar I"]
    with contextlib.redirect_stdout(io.StringIO()):
        _, al = als.select_lines(anc.CONFIG["actinometry"], ar.load_all_cross_sections())
    refs = al[np.isclose(al.wl_air.to_numpy()[:, None], LINES, atol=1e-3).any(axis=1)]
    xsC = n2c_cross_section()
    Q = he.ImportArQuenchingData("N2", verbose=False)
    N = he.Torr2Volume(anc.CONFIG["fit"]["P_Torr"], anc.CONFIG["fit"]["Tg"])
    rows = []
    for (kind, pct), fit in fits.items():
        if pct == 0:
            continue
        f = pct / 100
        tab, cfg = fit["tab"], fit["cfg"]
        lev = list(tab["levels"])
        kC0 = N2C_FC00 * mlf.cdf.atu.rates_on_axis(fit, xsC, 0.0)[0]
        fine = lambda v: np.exp(crf.on_fine_grid(tab, np.log(np.clip(v, 1e-300, None)), cfg, k=1))
        transfer = sum((Q[l]["kQ"] + Q[l]["kQM"] * (1 - f) * N) * fine(tab["dens"][:, :, lev.index(l)]) for l in Q)
        for c in fit["cond"].itertuples():
            pin = fit["posts"][f"{c.sweep}|{c.x:g}"]
            j = int(np.argmax(pin.sum(axis=0)))
            w, Ne = pin[:, j], np.exp(fit["grid"][1][j])
            meas = R[(R.sweep == c.sweep) & (R.x == c.x)]
            for variant in ("direct", "CR"):
                vals = []
                for r in refs.itertuples():
                    m = meas[np.isclose(meas.ar_wl, r.wl_air, atol=0.06)].R
                    if m.empty or not np.median(m) > 0:
                        continue
                    kAr = mlf.cdf.atu.rates_on_axis(fit, xs[r.upper], 0.0)[0]
                    ratio = np.median(m) * N2C_LAMBDA / r.wl_air * r.branching * Ne * kAr
                    if variant == "direct":
                        field = ratio / (N2C_B00 * Ne * kC0)
                    else:
                        fdir = fine(np.nan_to_num(tab["fdirect"][:, :, lev.index(r.upper)], nan=1.0))[:, j]
                        field = ratio / fdir / (N2C_B00 * (Ne * kC0 + N2C_TRANSFER_V0 * transfer[:, j]))
                    ok = np.isfinite(field) & (field > 0)
                    if ok.sum() and w[ok].sum() > 0.5:
                        o = np.argsort(field[ok])
                        vals.append(np.exp(crf._wquantile(np.log(field[ok][o]), w[ok][o], [0.5])[0]))
                if not vals:
                    continue
                g = np.exp(np.mean(np.log(vals)))
                to_d = lambda v: 1 - v * (1 - f) / f
                rows.append(dict(sweep=c.sweep, x=c.x, N2_percent=pct, variant=variant, n_refs=len(vals),
                                 nN2_nAr=g, nN2_nAr_lo=min(vals), nN2_nAr_hi=max(vals), n_N2=g * (1 - f) * N,
                                 dissociation=to_d(g), dissociation_lo=to_d(max(vals)), dissociation_hi=to_d(min(vals)),
                                 transfer_over_e=float(np.sum(w * N2C_TRANSFER_V0 * transfer[:, j] / (Ne * kC0)))))
    return pd.DataFrame(rows)


# ---- plots ----------------------------------------------------------------------
def panel(ax, res, nn, sweep, key, xlabel, n2c=None):
    N = he.Torr2Volume(anc.CONFIG["fit"]["P_Torr"], anc.CONFIG["fit"]["Tg"])
    if key == "diss":
        for v in VARIANTS:
            g = nn[(nn.sweep == sweep) & (nn.variant == v)].sort_values("x")
            st = VSTYLE[v]
            ax.plot(g.x, g.dissociation, color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.6, ms=6,
                    label="N I 744 nm, " + st["label"])
        if n2c is not None and len(n2c):
            for v, st in N2C_STYLE.items():
                g = n2c[(n2c.sweep == sweep) & (n2c.variant == v)].sort_values("x")
                ax.errorbar(g.x, g.dissociation, yerr=[g.dissociation - g.dissociation_lo,
                                                       g.dissociation_hi - g.dissociation],
                            capsize=3, lw=1.6, ms=6, **st)
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(1, color="k", ls=":", lw=1.2, label="complete dissociation")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_ylabel("N$_2$ dissociation fraction", fontsize=FONT)
    elif key == "n_N":
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


def plot(res, nn, n2c, version):
    sweeps = [(s["name"], s["xlabel"]) for s in crf.CONFIG["sweeps"]]
    title = (f"{EEDF_LABEL[EEDF_N2]} EEDF (Ar/N$_2$), {EEDF_LABEL[EEDF_PURE_AR]} (pure Ar), {VERSIONS[version]} "
             r"(415.86 / 706.72 / 750.39 nm, calibration trusted), $N_e$ at the local $\chi^2$ minimum near $n_c$")
    keys = (("n_N", "N_density"), ("Te", "Te"), ("diss", "dissociation"))
    fig, axs = plt.subplots(len(keys), 2, figsize=(13, 4.6 * len(keys)))
    for j, (sw, xl) in enumerate(sweeps):
        for i, (key, _) in enumerate(keys):
            panel(axs[i, j], res, nn, sw, key, xl, n2c)
        axs[0, j].set_title(f"{sw} sweep", fontsize=FONT + 1)
    fig.suptitle(title, fontsize=FONT - 2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"trends_{version}.png"), dpi=170)
    plt.close(fig)
    for sw, xl in sweeps:
        for key, name in keys:
            fig, ax = plt.subplots(figsize=(7.5, 5.2))
            panel(ax, res, nn, sw, key, xl, n2c)
            ax.set_title(f"{sw} sweep, {VERSIONS[version]}", fontsize=FONT - 1)
            fig.tight_layout()
            fig.savefig(os.path.join(OUTDIR, f"{name}_vs_{sw.split()[0]}_{version}.png"), dpi=170)
            plt.show()


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    mlf.build_tables([f"{eedf_kind(p)}:{p:g}" for p in FRACTIONS])
    print("CR fit, Ne at the local minimum near n_c ...")
    res, fits = fit_all()
    R = band_ratios(fits["lines"][EEDF_PURE_AR, 0.0]["ft"])
    nns, pairs_all, n2cs = [], [], []
    for version in VERSIONS:
        pairs, cond = nitrogen(fits[version])
        n2c = nitrogen_c(fits[version], R).assign(version=version)
        r = res[res.version == version]
        nn = cond.merge(r[["sweep", "x", "Ne", "Te"]], on=["sweep", "x"], suffixes=("_act", "")).assign(version=version)
        plot(r, nn, n2c, version)
        nns.append(nn)
        n2cs.append(n2c)
        pairs_all.append(pairs.assign(version=version))
    nn, pairs, n2c = (pd.concat(v, ignore_index=True) for v in (nns, pairs_all, n2cs))
    res.to_csv(os.path.join(OUTDIR, "trends.csv"), index=False)
    nn.to_csv(os.path.join(OUTDIR, "nN_near_nc.csv"), index=False)
    pairs.to_csv(os.path.join(OUTDIR, "nN_per_pair.csv"), index=False)
    n2c.to_csv(os.path.join(OUTDIR, "n2c_near_nc.csv"), index=False)
    cols = ["version", "sweep", "x", "variant", "n_pairs", "n_N", "n_N_lo", "n_N_hi", "dissociation"]
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(nn[cols].sort_values(["version", "sweep", "x", "variant"]).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        print(n2c[["version", "sweep", "x", "variant", "n_refs", "n_N2", "dissociation", "dissociation_lo",
                   "dissociation_hi", "transfer_over_e"]].sort_values(["version", "sweep", "x", "variant"])
              .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nfigures in {OUTDIR}")
    return res, nn


if __name__ == "__main__":
    RES, NN = main()
