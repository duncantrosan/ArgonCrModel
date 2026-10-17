# -*- coding: utf-8 -*-
"""
FivePLinesVsN2.py

Ar I 5p -> 4s line intensities against % N2 (echelle N2 percent sweep, GsaSweep_85W_1_torr: 85 W, 1 Torr, 0-3 %),
and why the CR model's 1s5 jumps ~60x as soon as N2 is added (LASTgComparison, LAS1s5vsN2).

1. Measurement: every 5p -> 4s line of the single-spectrum analysis (lines_mix.csv, status ok / weak) and the clean
   4p -> 4s lines as a reference, fitted in every intensity-calibrated spectrum of the sweep (group fitter of
   Argon_Nitrogen_Mix_Analysis_V2 through ActinometryLineSelection.measure_sweeps).  Per line and % N2: median
   area over the repeats; bars = scatter of the repeats / sqrt(n).  The spectra are calibrated and taken one after
   the other, so the absolute areas are compared across the sweep (I / I at 0 %).
2. CR model at the fits of LASTgComparison (microwave EEDF, Ne = n_c, LAS Tg, E/N of the best fit at n_c): the
   emitted intensity n_u A eta of the same lines and the 1s5 density, also relative to 0 %; fraction of the 5p
   production that is direct excitation from the ground state.
3. Ar(1s5) budget of the CR model at those fits: production by channel and loss frequency by channel.

Output in Experimental_Data/Output/FivePLinesVsN2/:
  ar_5p_4p_measurements.csv (cache of the line fits), measured_lines.csv, model_lines.csv, budget_1s5.csv
  figures/5p_intensity_vs_N2.png      measured 5p lines (absolute, calibrated) against % N2
  figures/5p_relative_vs_N2.png       measured 5p lines relative to 0 % N2, with the 4p lines for reference
  figures/model_vs_measured_vs_N2.png measured and CR-model change of the 5p and 4p lines and the 1s5 density
  figures/eepf_at_nc.png              EEPFs of the fits at n_c
Run from Spyder (F5) or python SmallAnalysisScripts/FivePLinesVsN2.py.  Needs LASTgComparison's comparison.csv
and CR tables; the line fits are cached (delete the csv to refit).
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
import LASTgComparison as L                         # noqa: E402

crf, he, als = L.crf, L.he, L.als

OUTDIR = os.path.join(crf.ROOT_DIR, "Experimental_Data", "Output", "FivePLinesVsN2")
FIGDIR = os.path.join(OUTDIR, "figures")
SWEEP = "N2 fraction"
STATUS_5P = ("ok", "weak")                          # lines_mix status of the 5p lines measured
CLEAN = ("clean", "clean?")                         # contamination flags drawn as clean lines
MIN_SNR = 5.0                                       # line fits below this SNR in a spectrum are dropped
N_C = L.N_C


# ---- 1. measurement ---------------------------------------------------------------------
def line_list():
    """5p -> 4s lines (status ok / weak) and the clean 4p -> 4s lines of lines_mix.csv."""
    res = pd.read_csv(os.path.join(als.CONFIG["selection_dir"], als.CONFIG["lines_file"]))
    ar = res[(res.species == "Ar I") & res.lower.str.startswith("4s")]
    p5 = ar[ar.upper.str.startswith("5p") & ar.status.isin(STATUS_5P)].assign(kind="5p")
    p4 = ar[ar.upper.str.startswith("4p") & (ar.status == "ok") & ar.contamination.isin(CLEAN)
            & ~ar.wl_air.round(3).isin(np.round(crf.CONFIG["exclude_wl"], 3))].assign(kind="4p")
    cols = ["species", "wl_air", "upper", "lower", "Aki", "status", "contamination", "kind"]
    return pd.concat([p5, p4])[cols].sort_values("wl_air").reset_index(drop=True)


def measure(lines):
    sweeps = [sw for sw in als.CONFIG["sweeps"] if sw["name"] == SWEEP]
    mcfg = dict(als.CONFIG, sweeps=sweeps, outdir=OUTDIR, measure_cache="ar_5p_4p_measurements.csv")
    return als.filter_export_flags(als.measure_sweeps(lines[["species", "wl_air"]], mcfg), mcfg)


def measured_table(m, lines):
    """Per line and % N2: median area over the repeats, its error in ln (scatter / sqrt(n)), number of repeats."""
    m = m.assign(wl=m.wl_air.round(4))
    m = m[(m.species == "Ar I") & m.wl.isin(lines.wl_air.round(4)) & m.status.isin(("ok", "weak"))
          & (m.snr >= MIN_SNR) & (m.area > 0)]
    rows = []
    for (wl, x), g in m.groupby(["wl", "x"]):
        ln = np.log(g.area.to_numpy())
        err = np.std(ln, ddof=1) / np.sqrt(len(ln)) if len(ln) > 1 else np.nan
        rows.append(dict(wl=wl, x=x, area=float(np.median(g.area)), dln=err, n_rep=len(ln),
                         snr=float(np.median(g.snr))))
    t = pd.DataFrame(rows)
    info = lines.assign(wl=lines.wl_air.round(4)).set_index("wl")[["upper", "lower", "Aki", "contamination", "kind"]]
    t = t.join(info, on="wl")
    ref = t[t.x == 0].set_index("wl").area
    t["rel"] = t.area / t.wl.map(ref)
    t["dln_rel"] = np.sqrt(t.dln ** 2 + t.wl.map(t[t.x == 0].set_index("wl").dln) ** 2)
    return t[t.rel.notna()].reset_index(drop=True)


# ---- 2. CR model at the fits ------------------------------------------------------------
def conditions():
    comp = pd.read_csv(os.path.join(L.OUTDIR, "comparison.csv"))
    return comp[comp.sweep == SWEEP].sort_values("x").reset_index(drop=True)


def _cfg(c):
    return L.table_cfg(L._key(c.N2, c.p_mTorr, c.Tg), "n2")


def model_lines(cond, wls):
    """CR model at (E/N of the fit at n_c, n_c) of every condition: n_u A eta of the lines wls, level densities and
    the direct-excitation fraction of the 5p levels."""
    rows = []
    for c in cond.itertuples():
        cfg = _cfg(c)
        with contextlib.redirect_stdout(io.StringIO()):
            tab = crf.build_model_table(cfg)
        fX, fN = crf.fine_grid(tab, cfg)
        i, j = int(np.argmin(np.abs(fX - np.log(c.EN_nc)))), int(np.argmin(np.abs(fN - np.log(N_C))))

        def at(v):
            return float(np.exp(crf.on_fine_grid(tab, np.log(np.clip(v, 1e-300, None)), cfg, k=1)[i, j]))
        lev = list(tab["levels"])
        for w in wls:
            k = int(np.argmin(np.abs(tab["line_wl"] - w)))
            if abs(tab["line_wl"][k] - w) > 0.005:
                continue
            u = tab["line_upper"][k]
            rows.append(dict(x=c.x, wl=round(w, 4), upper=u, lower=tab["line_lower"][k], I=at(tab["I_obs"][:, :, k]),
                             n_upper=at(tab["dens"][:, :, lev.index(u)]),
                             fdirect=at(tab["fdirect"][:, :, lev.index(u)]),
                             n_1s5=at(tab["dens"][:, :, lev.index("4s1")]), EN=c.EN_nc, Te_eff=c.Te_nc))
    t = pd.DataFrame(rows)
    t["rel"] = t.I / t.wl.map(t[t.x == 0].set_index("wl").I)
    return t


# ---- 3. Ar(1s5) budget ------------------------------------------------------------------
def solve(cfg, EN, Ne=N_C):
    cr, _, MD, interp = crf.load_cr_model(cfg["outdir"], cfg.get("escape_mode", "table"))
    MD = he.AddDiffusionLoss(MD, cfg["P_Torr"], cfg["Tg"], cfg["R"])
    P_Ar = crf.apply_nitrogen(MD, cfg, he)
    with contextlib.redirect_stdout(io.StringIO()):
        eedfs, x, _, _ = crf.eedf_axis(cfg)
    i = int(np.argmin(np.abs(np.log(x / EN))))
    with contextlib.redirect_stdout(io.StringIO()):
        D, _, _, _ = cr["CRModel"](MD, eedfs[i], float(Ne), P_Ar, cfg["Tg"], cfg["R"], interp,
                                   trap_lines=cfg["trap_lines"], atom_transfer=cfg.get("atom_transfer", True))
    return D, eedfs[i], x[i]


def budget_1s5(D, Ne=N_C, label="4s1"):
    """Production [m^-3 s^-1] and loss frequency [s^-1] of 1s5 by channel."""
    US = D[label]
    n = {k: v["density_m^-3"] for k, v in D.items()}
    is4s = lambda s: s.startswith("4s")
    P = dict(ground=0.0, mix_4s=0.0, from_above=0.0)
    for r in US["Electron Impact CrossSections"]["Products"]:
        key = "ground" if r["partner_label"] == "ground" else "mix_4s" if is4s(r["partner_label"]) else "from_above"
        P[key] += Ne * r["Rate"] * n[r["partner_label"]]
    for s in US.get("Superelastic", []):
        if s["direction"] == "gain":
            P["mix_4s" if is4s(s["partner"]) else "from_above"] += Ne * s["coeff"] * n[s["partner"]]
    Ng = n["ground"]
    for a in US.get("AtomTransfer", []):
        if a["direction"] == "gain":
            P["from_above"] += Ng * a["coeff"] * n[a["partner"]]
    P_tot = US["Production_m^-3s^-1"]
    P["cascade"] = P_tot - sum(P.values())
    nu = dict(e_to_4p=0.0, e_mix_4s=0.0, e_to_higher=0.0)
    for r in US["Electron Impact CrossSections"]["Reactants"]:
        p = r["partner_label"]
        nu["e_to_4p" if p.startswith("4p") else "e_mix_4s" if is4s(p) else "e_to_higher"] += Ne * r["Rate"]
    nu["e_superelastic_ground"] = Ne * sum(s["coeff"] for s in US.get("Superelastic", []) if s["direction"] == "loss")
    nu["e_ionization"] = Ne * US["Ionization Data"]["Rate_cm^3"]
    nu["N2_quench"] = US.get("GasQuenching_s^-1", 0.0)
    nu["diffusion"] = 1.0 / US["DiffusionLoss"]
    nu["other"] = US["Loss_s^-1"] - sum(nu.values())     # metastable pooling, 2-/3-body Ar, atom transfer
    k_ground = P["ground"] / (Ne * Ng)
    return dict(n_1s5=US["density_m^-3"], P_total=P_tot, **{f"P_{k}": v for k, v in P.items()},
                nu_total=US["Loss_s^-1"], **{f"nu_{k}": v for k, v in nu.items()}, k_ground_1s5=k_ground, N_gas=Ng)


def budgets(cond):
    rows, eepf = [], []
    for c in cond.itertuples():
        D, e, EN = solve(_cfg(c), c.EN_nc)
        rows.append(dict(x=c.x, Tg=c.Tg, EN_row=EN, Te_eff=e["Te_eff"], **budget_1s5(D)))
        eepf.append((c.x, EN, e))
    return pd.DataFrame(rows), eepf


# ---- plots -------------------------------------------------------------------------------
def _label(r):
    return f"{r.wl:.2f} ({r.upper}-{r.lower})"


def plot_5p(t, outdir):
    p5 = t[t.kind == "5p"]
    order = p5[p5.x == 0].sort_values("area", ascending=False).wl.tolist()
    clean = [w for w in order if p5[p5.wl == w].contamination.iloc[0] in CLEAN]
    col = dict(zip(clean, plt.cm.tab10(np.arange(len(clean)) % 10)))
    # absolute
    fig, ax = plt.subplots(figsize=(9, 6))
    for k, w in enumerate(order):
        g = p5[p5.wl == w].sort_values("x")
        if w in col:
            ax.errorbar(g.x + 0.012 * (k - len(order) / 2), g.area, yerr=g.area * g.dln.fillna(0), fmt="o-", ms=5,
                        lw=1.6, capsize=2, color=col[w], label=_label(g.iloc[0]), zorder=3)
        else:
            ax.plot(g.x, g.area, "o--", ms=3, lw=0.8, color="0.7", zorder=1)
    ax.plot([], [], "o--", ms=3, lw=0.8, color="0.7",
            label=f"{len(order) - len(clean)} lines flagged as possibly blended (N$_2$ bands)")
    ax.set_yscale("log")
    ax.set_xlabel("N$_2$ in the feed [%]")
    ax.set_ylabel("line area (calibrated, median of the repeats)")
    ax.set_title("Ar I 5p $\\rightarrow$ 4s lines, 85 W, 1 Torr (echelle N$_2$ sweep)", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "5p_intensity_vs_N2.png"), dpi=200)
    # relative to 0 %
    fig, ax = plt.subplots(figsize=(9, 6))
    p4 = t[t.kind == "4p"]
    for w, g in p4.groupby("wl"):
        ax.plot(g.x, g.rel, "-", color="0.75", lw=1, zorder=1)
    ax.plot([], [], "-", color="0.75", label="4p $\\rightarrow$ 4s lines (clean), for reference")
    for k, w in enumerate(order):
        g = p5[p5.wl == w].sort_values("x")
        if w not in col:
            continue
        ax.errorbar(g.x + 0.012 * (k - len(order) / 2), g.rel, yerr=g.rel * g.dln_rel.fillna(0), fmt="o-", ms=5,
                    lw=1.6, capsize=2, color=col[w], label=_label(g.iloc[0]), zorder=3)
    ax.axhline(1, color="0.3", lw=1)
    ax.set_xlabel("N$_2$ in the feed [%]")
    ax.set_ylabel("line area / line area at 0 % N$_2$")
    ax.set_title("Ar I 5p $\\rightarrow$ 4s lines relative to pure Ar (clean lines), 85 W, 1 Torr", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "5p_relative_vs_N2.png"), dpi=200)


def plot_model(t, mod, cond, outdir):
    kind = dict(zip(t.wl, t.kind))
    clean = dict(zip(t.wl, t.contamination.isin(CLEAN)))
    mod = mod[mod.wl.isin([w for w in clean if clean[w]])].assign(kind=lambda d: d.wl.map(kind))
    tc = t[t.contamination.isin(CLEAN)]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for kd, col, mk in (("5p", "C0", "o"), ("4p", "C2", "^")):
        meas = tc[tc.kind == kd].groupby("x").rel.agg(lambda r: np.exp(np.median(np.log(r))))
        lo = tc[tc.kind == kd].groupby("x").rel.min()
        hi = tc[tc.kind == kd].groupby("x").rel.max()
        ax.fill_between(meas.index, lo, hi, color=col, alpha=0.12, lw=0)
        ax.plot(meas.index, meas, mk + "-", color=col, ms=7, lw=2,
                label=f"measured {kd} $\\rightarrow$ 4s (median of {tc[tc.kind == kd].wl.nunique()} lines; band = range)")
        m = mod[mod.kind == kd].groupby("x").rel.agg(lambda r: np.exp(np.median(np.log(r))))
        ax.plot(m.index, m, mk + "--", color=col, mfc="white", ms=7, lw=1.5, label=f"CR model {kd} $\\rightarrow$ 4s (same lines)")
    c = cond.set_index("x")
    ax.plot(c.index, c.n_1s5 / c.n_1s5.loc[0.0], "s:", color="C3", ms=6, label="CR model 1s$_5$ density")
    ax.plot(c.index, c.las_1s5 / c.las_1s5.loc[0.0], "D:", color="k", ms=6, label="LAS 1s$_5$ density")
    ax.axhline(1, color="0.3", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("N$_2$ in the feed [%]")
    ax.set_ylabel("relative to 0 % N$_2$")
    ax.set_title("Change with N$_2$: measured lines vs CR model (microwave EEDF, N$_e$ = n$_c$, LAS T$_g$)\n"
                 "85 W, 1 Torr", fontsize=10)
    ax.set_ylim(0.2, 1e5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "model_vs_measured_vs_N2.png"), dpi=200)


def plot_eepf(eepf, outdir):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(eepf)))
    for (x, EN, e), col in zip(eepf, cols):
        ax.semilogy(e["E"], e["EEPF"], color=col, lw=1.8,
                    label=f"{x:g} % N$_2$: {EN:.0f} Td, T$_{{e,eff}}$ = {e['Te_eff']:.2f} eV")
    for thr, lab in ((11.55, "1s$_5$"), (13.1, "4p"), (14.5, "5p")):
        ax.axvline(thr, color="0.5", ls=":", lw=1)
        ax.text(thr, 1.0, lab, rotation=90, va="top", ha="right", fontsize=8, color="0.4", transform=ax.get_xaxis_transform())
    ax.set_xlim(0, 25)
    ax.set_ylim(1e-12, 3)
    ax.set_xlabel("Electron energy [eV]")
    ax.set_ylabel("EEPF [eV$^{-3/2}$]")
    ax.set_title("EEPF of the CR fits at n$_c$ (BOLSIG+ microwave, Ar or Ar/N$_2$, LAS T$_g$), 85 W, 1 Torr",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "eepf_at_nc.png"), dpi=200)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    lines = line_list()
    print(f"{(lines.kind == '5p').sum()} 5p lines, {(lines.kind == '4p').sum()} 4p lines")
    t = measured_table(measure(lines), lines)
    t.to_csv(os.path.join(OUTDIR, "measured_lines.csv"), index=False)
    cond = conditions()
    mod = model_lines(cond, sorted(t.wl.unique()))
    mod.to_csv(os.path.join(OUTDIR, "model_lines.csv"), index=False)
    bud, eepf = budgets(cond)
    bud.to_csv(os.path.join(OUTDIR, "budget_1s5.csv"), index=False)
    plot_5p(t, FIGDIR)
    plot_model(t, mod, cond, FIGDIR)
    plot_eepf(eepf, FIGDIR)
    pd.set_option("display.width", 250)
    print("\nmeasured 5p lines, area / area at 0 %:")
    print(t[t.kind == "5p"].pivot_table(index=["wl", "upper", "lower", "contamination"], columns="x", values="rel")
          .round(2).to_string())
    print("\nCR model, n_u A eta / at 0 % (clean lines):")
    print(mod.pivot_table(index=["wl", "upper"], columns="x", values="rel").round(2).to_string())
    print("\nCR model, direct-excitation fraction of the 5p levels:")
    print(mod[mod.upper.str.startswith("5p")].drop_duplicates(["x", "upper"])
          .pivot_table(index="upper", columns="x", values="fdirect").round(2).to_string())
    print("\n1s5 budget at n_c:")
    print(bud.T.to_string(float_format=lambda v: f"{v:.3g}"))
    plt.show()


if __name__ == "__main__":
    main()
