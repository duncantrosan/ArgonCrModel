# -*- coding: utf-8 -*-
"""
ActinometryLineSelection.py

Cross-reference clean N I lines against clean Ar I lines and look for
actinometry pairs whose ratio is flat.

1. Selection (from the single-spectrum analysis in ArN2_out + manual ratings)
   N I  : detected (ok/weak), automatic contamination call clean/clean?,
          manual rating >= n_min_rating (if rated), and the upper level has a
          ground-state LXCat cross section.
   Ar I : detected (ok), clean/clean?, rating >= ar_min_rating (if rated),
          isolated (no other detected line within ar_isolation_nm), upper
          level has a ground-state cross section.  One line per Ar upper level
          (highest SNR) so each reference carries distinct excitation physics.

2. Measurement: every spectrum of the N2-fraction sweep and the power sweep is
   fitted with the same group fitter as Argon_Nitrogen_Mix_Analysis_V2
   (instrument function measured per spectrum).  Cached to CSV.

3. Per N I line one figure with three panels, one curve per Ar reference:
     a) I_N / I_Ar vs N2 fraction   (normalised to its mean)
     b) I_N / I_Ar vs power         (normalised to its mean)
     c) k_N / k_Ar vs Te            (Maxwellian, normalised at Te_ref)
   Panels a/b contain the change of n_N/n_Ar AND of the rate ratio; since
   n_N/n_Ar is common to every Ar reference, curves that coincide mean the
   references behave alike.  Panel c isolates the Te sensitivity.

4. Summary: pair table (CSV) with flatness metrics and a heat map.

Run from Spyder: edit CONFIG, press F5.
"""
import os
import re
import sys
from fnmatch import fnmatch
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Argon_Nitrogen_Mix_Analysis_V2 as arn      # noqa: E402
import ActinometryRates as ar                      # noqa: E402

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_DIR = os.path.join(ROOT_DIR, "Experimental_Data", "EchelleData", "TestData")
MINI_DIR = os.path.join(ROOT_DIR, "Experimental_Data", "EchelleData", "PrelijminaryNitrogenAdmixtureMini")


def _gsa_x(name):
    """Percent0_1 -> (0.0, 1);  Percent1_5__2 -> (1.5, 2)"""
    m = re.match(r"Percent(\d+)_(\d+)__(\d+)$", name)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}"), int(m.group(3))
    m = re.match(r"Percent0_(\d+)$", name)
    if m:
        return 0.0, int(m.group(1))
    return None


def _power_x(name):
    """35W_2 -> (35.0, 2)"""
    m = re.match(r"(\d+)W_(\d+)$", name)
    return (float(m.group(1)), int(m.group(2))) if m else None


CONFIG = dict(
    # --- selection inputs (single-spectrum analysis + manual ratings) --------
    selection_dir=os.path.join(TEST_DIR, "ArN2_out"),
    lines_file="lines_mix.csv",
    n_ratings="N2_manual_ratings.csv",
    ar_ratings="Ar_manual_ratings.csv",
    # --- N I selection -------------------------------------------------------
    n_status=("ok", "weak"),
    n_contamination=("clean", "clean?"),
    n_min_rating=3,                     # 5 clean ... 1 junk; unrated lines pass on the auto flags
    # --- Ar I selection ------------------------------------------------------
    ar_status=("ok",),
    ar_contamination=("clean", "clean?"),
    ar_min_rating=4,
    ar_isolation_nm=0.10,               # no other detected catalogue line closer than this
    one_ar_line_per_level=True,
    max_ar_refs=10,
    # --- sweeps --------------------------------------------------------------
    sweeps=[
        dict(name="N2 fraction", folder=os.path.join(MINI_DIR, "GsaSweep_85W_1_torr", "SPA_Files"),
             parse=_gsa_x, xlabel="N$_2$ fraction [%]", note="85 W, 1 Torr", exclude_x=(0.0,)),
        dict(name="Power", folder=os.path.join(MINI_DIR, "PowerSweepN2_2.4_Ar77_6_1Torr"),
             parse=_power_x, xlabel="Power [W]", note="2.4 % N$_2$, 1 Torr", exclude_x=()),
    ],
    export_flags="?-?-1-1",             # header flags to keep (fnmatch); 3rd = intensity calibration
    sweep_status=("ok", "weak"),        # both lines must have this status in a spectrum
    min_reps=2,                         # condition kept only if >= this many repeats are valid
    aggregate="median",                 # 'median' (robust to single-frame spikes) or 'mean'
    blank=dict(sweep="N2 fraction", x=0.0),   # pure-Ar spectra used as the N I blank
    window_nm=1.5,                      # catalogue lines within this of a selected line are fitted
    reuse_measurements=True,            # reuse cached sweep_measurements.csv if present
    # --- model ---------------------------------------------------------------
    Te_grid=np.linspace(0.8, 8.0, 73),
    Te_ref=3.0,                         # normalisation / sensitivity point [eV]
    Te_band=(1.5, 5.0),                 # range for the max/min flatness metric
    # --- output --------------------------------------------------------------
    outdir=os.path.join(ROOT_DIR, "Experimental_Data", "Output", "Actinometry"),
)


# ----------------------------------------------------------------------------
# 1. Selection
# ----------------------------------------------------------------------------
def _key(species, lower, upper, wl):
    return f"{species}|{lower}->{upper}|{wl:.3f}"


def _attach_ratings(res, path):
    res = res.copy()
    res["rating"] = np.nan
    if path and os.path.exists(path):
        r = pd.read_csv(path)
        rmap = dict(zip(r["key"], r["rating"]))
        keys = [_key(s, lo, up, w) for s, lo, up, w in zip(res.species, res.lower, res.upper, res.wl_air)]
        res["rating"] = [rmap.get(k, np.nan) for k in keys]
    return res


def _isolated(res, idx, tol, detected=("ok", "weak", "blend", "saturated")):
    det = res[res.status.isin(detected)]
    w = res.at[idx, "wl_air"]
    near = det[(np.abs(det.wl_air - w) < tol) & (det.index != idx)]
    return near.empty


def select_lines(cfg=CONFIG, xs=None):
    """Return (N I lines, Ar I reference lines) as DataFrames."""
    xs = xs or ar.load_all_cross_sections()
    res = pd.read_csv(os.path.join(cfg["selection_dir"], cfg["lines_file"]))
    n = _attach_ratings(res[res.species == "N I"], os.path.join(cfg["selection_dir"], cfg["n_ratings"]))
    a = _attach_ratings(res[res.species == "Ar I"], os.path.join(cfg["selection_dir"], cfg["ar_ratings"]))

    def passes(df, status, contam, min_rating):
        ok = df.status.isin(status) & df.contamination.isin(contam)
        return ok & (df.rating.isna() | (df.rating >= min_rating))

    n = n[passes(n, cfg["n_status"], cfg["n_contamination"], cfg["n_min_rating"]) & n.upper.isin(xs["N I"])]
    a = a[passes(a, cfg["ar_status"], cfg["ar_contamination"], cfg["ar_min_rating"]) & a.upper.isin(xs["Ar I"])]
    a = a[[_isolated(res, i, cfg["ar_isolation_nm"]) for i in a.index]]
    if cfg["one_ar_line_per_level"]:
        a = a.sort_values("snr", ascending=False).drop_duplicates("upper")
    a = a.sort_values("snr", ascending=False).head(cfg["max_ar_refs"])

    A_sums = {"N I": ar.load_n_A_sums(), "Ar I": ar.load_ar_A_sums()}
    for df, sp in ((n, "N I"), (a, "Ar I")):
        df["threshold_eV"] = [xs[sp][u]["threshold_eV"] for u in df.upper]
        df["branching"] = [ar.branching_ratio(A, A_sums[sp].get(u)) for A, u in zip(df.Aki, df.upper)]
    n = n.sort_values("wl_air").reset_index(drop=True)
    a = a.sort_values("threshold_eV").reset_index(drop=True)
    return n, a


# ----------------------------------------------------------------------------
# 2. Sweep measurement
# ----------------------------------------------------------------------------
def _catalogue():
    c = arn.CONFIG
    ar_lv, ar_lookup = arn.load_levels(c["ar_levels"])
    n_lv, n_lookup = arn.load_levels(c["n_levels"])
    cat_ar = arn.load_ar_lines(c["ar_lines"], ar_lv, ar_lookup)
    cat_n = arn.load_n_lines(c["n_lines"], n_lv, n_lookup, wl_max=c["wl_max"])
    cat = pd.concat([cat_ar, cat_n], ignore_index=True).sort_values("wl_air").reset_index(drop=True)
    return cat, cat_ar


def measure_sweeps(sel_lines, cfg=CONFIG):
    """Fit the selected lines in every spectrum of every sweep -> long table."""
    cache = os.path.join(cfg["outdir"], cfg.get("measure_cache", "sweep_measurements.csv"))
    if cfg["reuse_measurements"] and os.path.exists(cache):
        m = pd.read_csv(cache)
        have = set(zip(m.species, m.wl_air.round(4)))
        want = set(zip(sel_lines.species, sel_lines.wl_air.round(4)))
        if want <= have:
            print(f"reusing {cache}")
            return m
    cat, cat_ar = _catalogue()
    near = np.zeros(len(cat), bool)
    for w in sel_lines.wl_air:
        near |= np.abs(cat.wl_air.values - w) < cfg["window_nm"]
    sub = cat[near]
    want = set(zip(sel_lines.species, sel_lines.wl_air.round(4)))
    acfg = arn.CONFIG
    rows = []
    for sw in cfg["sweeps"]:
        files = sorted(glob(os.path.join(sw["folder"], "*.spa")))
        for f in files:
            name = os.path.splitext(os.path.basename(f))[0]
            px = sw["parse"](name)
            if px is None:
                continue
            x, rep = px
            sp = arn.load_spectrum(f, name, acfg)
            try:
                fwhm_fn, offset_fn, _ = arn.measure_instrument_function(sp, cat_ar, acfg)
                off = float(offset_fn(0))
            except RuntimeError:
                fwhm_fn, off = (lambda lam: lam / 20000.0), 0.0
                print(f"  {name}: instrument function failed, using R=20000")
            res, _ = arn.measure_lines(sp, sub, fwhm_fn, acfg, name, off)
            res = res[[(s, round(w, 4)) in want for s, w in zip(res.species, res.wl_air)]]
            for r in res.itertuples():
                rows.append(dict(sweep=sw["name"], file=name, x=x, rep=rep, species=r.species,
                                 wl_air=r.wl_air, upper=r.upper, status=r.status,
                                 area=r.area, height=r.height, snr=r.snr))
            print(f"  {sw['name']:<12s} {name:<16s} measured")
    m = pd.DataFrame(rows)
    os.makedirs(cfg["outdir"], exist_ok=True)
    m.to_csv(cache, index=False)
    return m


def export_flags(cfg=CONFIG):
    """Header flags of every sweep spectrum ('Wavelength 1-0-1-1-7617' -> '1-0-1-1')."""
    rows = []
    for sw in cfg["sweeps"]:
        for f in sorted(glob(os.path.join(sw["folder"], "*.spa"))):
            with open(f, encoding="latin-1") as fh:
                head = fh.readline().split()
            flags = "-".join(head[1].split("-")[:-1]) if len(head) > 1 else ""
            rows.append(dict(sweep=sw["name"], file=os.path.splitext(os.path.basename(f))[0], export_flags=flags))
    return pd.DataFrame(rows)


def filter_export_flags(meas, cfg=CONFIG):
    """Drop spectra exported with other processing flags than cfg['export_flags']
    (the 3rd flag switches the intensity calibration: raw counts are ~1e6 lower
    and have a different spectral response)."""
    pattern = cfg.get("export_flags")
    if not pattern:
        return meas
    fl = export_flags(cfg)
    bad = fl[[not fnmatch(f, pattern) for f in fl.export_flags]]
    if len(bad):
        print(f"dropping {len(bad)} spectra with export flags != {pattern}: "
              + ", ".join(f"{r.file} ({r.export_flags})" for r in bad.itertuples()))
    keep = ~pd.MultiIndex.from_frame(meas[["sweep", "file"]]).isin(pd.MultiIndex.from_frame(bad[["sweep", "file"]]))
    return meas[keep]


# ----------------------------------------------------------------------------
# 3. Ratios and flatness metrics
# ----------------------------------------------------------------------------
def pair_ratios(meas, n_lines, a_lines, cfg=CONFIG):
    """Per spectrum I_N/I_Ar, then mean +- error per sweep condition."""
    ok = meas.status.isin(cfg["sweep_status"])
    m = meas.assign(ok=ok, wl=meas.wl_air.round(4))
    rows = []
    for nl in n_lines.itertuples():
        mn = m[(m.species == "N I") & (m.wl == round(nl.wl_air, 4))].set_index(["sweep", "file"])
        for al in a_lines.itertuples():
            ma = m[(m.species == "Ar I") & (m.wl == round(al.wl_air, 4))].set_index(["sweep", "file"])
            j = mn.join(ma, lsuffix="_N", rsuffix="_Ar", how="inner")
            j = j[j.ok_N & j.ok_Ar]
            for (sweep, f), r in j.iterrows():
                rows.append(dict(n_wl=nl.wl_air, ar_wl=al.wl_air, sweep=sweep, file=f, x=r.x_N, rep=r.rep_N,
                                 ratio=r.area_N / r.area_Ar,
                                 rel_err=np.sqrt(r.snr_N ** -2 + r.snr_Ar ** -2)))
    per = pd.DataFrame(rows)
    if per.empty:
        return per, per
    excl = {sw["name"]: set(sw["exclude_x"]) for sw in cfg["sweeps"]}
    per = per[[x not in excl.get(s, ()) for s, x in zip(per.sweep, per.x)]]

    def agg(g):
        n = len(g)
        med = cfg["aggregate"] == "median"
        mu = g.ratio.median() if med else g.ratio.mean()
        # standard error of the median ~ 1.2533 x that of the mean
        err_rep = (1.2533 if med else 1.0) * g.ratio.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        err_fit = mu * np.sqrt((g.rel_err ** 2).sum()) / n
        return pd.Series(dict(mean=mu, err=np.nanmax([err_rep, err_fit]), n_rep=n))

    cond = per.groupby(["n_wl", "ar_wl", "sweep", "x"]).apply(agg, include_groups=False).reset_index()
    cond = cond[cond.n_rep >= cfg["min_reps"]]
    cond["norm"] = cond["mean"] / cond.groupby(["n_wl", "ar_wl", "sweep"])["mean"].transform("mean")
    cond["norm_err"] = cond["err"] / cond.groupby(["n_wl", "ar_wl", "sweep"])["mean"].transform("mean")
    return per, cond


def blank_check(meas, n_lines, a_lines, cond, cfg=CONFIG):
    """Signal-to-blank per pair: typical I_N/I_Ar in the sweeps divided by the
    mean I_N/I_Ar fitted in the pure-Ar spectra (whatever the fitter returns
    there, detected or not).  < ~3 means the N I 'line' is at the noise level."""
    b = cfg.get("blank")
    if not b:
        return pd.DataFrame(columns=["n_wl", "ar_wl"])
    m = meas[(meas.sweep == b["sweep"]) & (meas.x == b["x"])].assign(wl=lambda d: d.wl_air.round(4))
    rows = []
    for nl in n_lines.itertuples():
        mn = m[(m.species == "N I") & (m.wl == round(nl.wl_air, 4))].set_index("file")
        for al in a_lines.itertuples():
            ma = m[(m.species == "Ar I") & (m.wl == round(al.wl_air, 4))].set_index("file")
            r = (mn.area.fillna(0) / ma.area).dropna()
            blank = r.mean() if len(r) else np.nan
            sig = cond[(cond.n_wl == nl.wl_air) & (cond.ar_wl == al.wl_air)]["mean"].median()
            rows.append(dict(n_wl=nl.wl_air, ar_wl=al.wl_air, blank_ratio=blank,
                             signal_to_blank=sig / blank if blank > 0 else np.inf))
    return pd.DataFrame(rows)


def repeat_scatter(per):
    """Median over conditions of the relative repeat scatter (MAD/median) per pair."""
    def mad_rel(g):
        return 1.4826 * np.median(np.abs(g - g.median())) / g.median() if len(g) > 1 else np.nan
    s = per.groupby(["n_wl", "ar_wl", "sweep", "x"]).ratio.apply(mad_rel)
    return s.groupby(["n_wl", "ar_wl"]).median().rename("repeat_scatter").reset_index()


def scale_anomalies(meas, factor=100.0):
    """Spectra whose Ar I line areas are > factor away from their sweep median
    (e.g. different exposure or units).  Ratios are unaffected, but worth knowing."""
    a = meas[meas.species == "Ar I"].groupby(["sweep", "file"]).area.median()
    med = a.groupby("sweep").transform("median")
    return a[(a > factor * med) | (a < med / factor)]


def flatness(cond):
    """CV of the condition means and fractional change across the sweep (linear fit)."""
    out = []
    for (nw, aw, sw), g in cond.groupby(["n_wl", "ar_wl", "sweep"]):
        g = g.sort_values("x")
        cv = g["mean"].std(ddof=1) / g["mean"].mean() if len(g) > 1 else np.nan
        if len(g) > 1:
            slope = np.polyfit(g.x, g.norm, 1)[0]
            change = slope * (g.x.max() - g.x.min())
        else:
            change = np.nan
        out.append(dict(n_wl=nw, ar_wl=aw, sweep=sw, n_cond=len(g), cv=cv, frac_change=change))
    return pd.DataFrame(out)


def model_ratios(n_lines, a_lines, xs, cfg=CONFIG):
    Te = cfg["Te_grid"]
    kN = {u: ar.rate_coefficient(xs["N I"][u], Te=Te) for u in n_lines.upper.unique()}
    kA = {u: ar.rate_coefficient(xs["Ar I"][u], Te=Te) for u in a_lines.upper.unique()}
    lo, hi = cfg["Te_band"]
    band = (Te >= lo) & (Te <= hi)
    curves, rows = {}, []
    for nl in n_lines.itertuples():
        for al in a_lines.itertuples():
            r = kN[nl.upper] / kA[al.upper]
            curves[(nl.wl_air, al.wl_air)] = r / np.interp(cfg["Te_ref"], Te, r)
            S = ar.te_sensitivity(xs["N I"][nl.upper], xs["Ar I"][al.upper], cfg["Te_ref"])
            rows.append(dict(n_wl=nl.wl_air, ar_wl=al.wl_air,
                             k_ratio_Te_ref=float(np.interp(cfg["Te_ref"], Te, r)),
                             S_Te_ref=S, model_max_over_min=r[band].max() / r[band].min()))
    return curves, pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 4. Plots
# ----------------------------------------------------------------------------
def _ar_label(al):
    return f"Ar {al.wl_air:.2f} ({al.upper_desc}, {al.threshold_eV:.2f} eV)"


def plot_n_line(nl, a_lines, cond, curves, cfg, path, extra=""):
    sweeps = cfg["sweeps"]
    fig, axs = plt.subplots(1, len(sweeps) + 1, figsize=(6 * (len(sweeps) + 1), 5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax, sw in zip(axs, sweeps):
        for k, al in enumerate(a_lines.itertuples()):
            g = cond[(cond.n_wl == nl.wl_air) & (cond.ar_wl == al.wl_air) & (cond.sweep == sw["name"])].sort_values("x")
            if g.empty:
                continue
            ax.errorbar(g.x, g.norm, yerr=g.norm_err, fmt="o-", color=colors[k % 10], ms=4, lw=1.3, capsize=2,
                        label=_ar_label(al))
        ax.axhline(1, color="0.5", lw=0.8, ls=":")
        ax.set_xlabel(sw["xlabel"])
        ax.set_ylabel(r"$(I_N/I_{Ar})\,/\,\langle I_N/I_{Ar}\rangle$")
        ax.set_title(f"{sw['name']} sweep ({sw['note']})")
        ax.grid(alpha=0.3)
    ax = axs[-1]
    Te = cfg["Te_grid"]
    for k, al in enumerate(a_lines.itertuples()):
        c = curves.get((nl.wl_air, al.wl_air))
        if c is not None:
            ax.plot(Te, c, color=colors[k % 10], lw=1.6, label=_ar_label(al))
    ax.axvspan(*cfg["Te_band"], color="0.9", zorder=0)
    ax.axhline(1, color="0.5", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T_e$ [eV]  (Maxwellian)")
    ax.set_ylabel(rf"$(k_N/k_{{Ar}})\,/\,(k_N/k_{{Ar}})_{{T_e={cfg['Te_ref']:g}\,eV}}$")
    ax.set_title("Model rate-coefficient ratio")
    ax.grid(alpha=0.3, which="both")
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8, frameon=False)
    rating = "" if np.isnan(nl.rating) else f", rating {nl.rating:.0f}"
    fig.suptitle(f"N I {nl.wl_air:.3f} nm   upper {nl.upper} = {nl.upper_desc}   "
                 f"E$_{{th}}$ = {nl.threshold_eV:.2f} eV   ({nl.status}, SNR {nl.snr:.0f}{rating})\n"
                 f"sweeps: {cfg['aggregate']} of >= {cfg['min_reps']} valid repeats;  {extra}")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_summary(summary, n_lines, a_lines, cfg, path):
    cols = [c for c in ("cv_N2 fraction", "cv_Power") if c in summary] + ["S_Te_ref"]
    titles = {"cv_N2 fraction": "CV across N$_2$ sweep", "cv_Power": "CV across power sweep",
              "S_Te_ref": rf"model $d\ln(k_N/k_{{Ar}})/d\ln T_e$ at {cfg['Te_ref']:g} eV"}
    fig, axs = plt.subplots(1, len(cols), figsize=(5.5 * len(cols), 0.45 * len(n_lines) + 2.5))
    axs = np.atleast_1d(axs)
    for ax, c in zip(axs, cols):
        M = summary.pivot(index="n_wl", columns="ar_wl", values=c).reindex(index=n_lines.wl_air, columns=a_lines.wl_air)
        V = np.abs(M.values) if c == "S_Te_ref" else M.values
        im = ax.imshow(V, cmap="viridis_r", aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="w" if abs(v) > np.nanmax(V) * 0.5 else "k")
        ax.set_xticks(range(M.shape[1]), [f"{w:.1f}" for w in M.columns], rotation=90, fontsize=8)
        ax.set_yticks(range(M.shape[0]), [f"{w:.2f}" for w in M.index], fontsize=8)
        ax.set_xlabel("Ar I reference [nm]")
        ax.set_ylabel("N I line [nm]")
        ax.set_title(titles[c], fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Actinometry pair flatness (lower = flatter)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. Driver
# ----------------------------------------------------------------------------
def run(cfg=CONFIG):
    os.makedirs(cfg["outdir"], exist_ok=True)
    per_dir = os.path.join(cfg["outdir"], "per_N_line")
    os.makedirs(per_dir, exist_ok=True)
    out = lambda f: os.path.join(cfg["outdir"], f)
    xs = ar.load_all_cross_sections()

    n_lines, a_lines = select_lines(cfg, xs)
    cols = ["species", "wl_air", "upper", "upper_desc", "Ek", "threshold_eV", "Aki", "branching",
            "status", "snr", "contamination", "rating"]
    xref = pd.concat([n_lines[cols], a_lines[cols]], ignore_index=True)
    xref.to_csv(out("selected_lines.csv"), index=False)
    print(f"\n{len(n_lines)} N I lines x {len(a_lines)} Ar I references")
    print(xref.to_string(index=False))
    if n_lines.empty or a_lines.empty:
        print("nothing to pair - loosen the selection thresholds in CONFIG")
        return None

    meas = filter_export_flags(measure_sweeps(pd.concat([n_lines, a_lines]), cfg), cfg)
    per, cond = pair_ratios(meas, n_lines, a_lines, cfg)
    per.to_csv(out("pair_ratios_per_spectrum.csv"), index=False)
    cond.to_csv(out("pair_ratios_per_condition.csv"), index=False)
    flat = flatness(cond) if not cond.empty else pd.DataFrame(columns=["n_wl", "ar_wl", "sweep"])
    curves, model = model_ratios(n_lines, a_lines, xs, cfg)

    wide = flat.pivot_table(index=["n_wl", "ar_wl"], columns="sweep", values=["cv", "frac_change", "n_cond"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    summary = model.merge(wide.reset_index(), on=["n_wl", "ar_wl"], how="left")
    summary["dE_threshold"] = [
        n_lines.set_index("wl_air").at[nw, "threshold_eV"] - a_lines.set_index("wl_air").at[aw, "threshold_eV"]
        for nw, aw in zip(summary.n_wl, summary.ar_wl)]
    if not per.empty:
        summary = summary.merge(repeat_scatter(per), on=["n_wl", "ar_wl"], how="left")
    blank = blank_check(meas, n_lines, a_lines, cond, cfg)
    if not blank.empty:
        summary = summary.merge(blank, on=["n_wl", "ar_wl"], how="left")
    summary = summary.sort_values(["n_wl", "S_Te_ref"], key=lambda s: s.abs() if s.name == "S_Te_ref" else s)
    summary.to_csv(out("pair_summary.csv"), index=False)

    stb = summary.groupby("n_wl").signal_to_blank.median() if "signal_to_blank" in summary else {}
    rsc = summary.groupby("n_wl").repeat_scatter.median() if "repeat_scatter" in summary else {}
    for nl in n_lines.itertuples():
        extra = f"signal/blank {stb.get(nl.wl_air, np.nan):.1f}, repeat scatter {100 * rsc.get(nl.wl_air, np.nan):.0f} %"
        plot_n_line(nl, a_lines, cond, curves, cfg, os.path.join(per_dir, f"N_{nl.wl_air:.3f}_{nl.upper}.png"), extra)
    plot_summary(summary, n_lines, a_lines, cfg, out("pair_summary.png"))

    anom = scale_anomalies(meas)
    if len(anom):
        print("\nWARNING: spectra with Ar line areas >100x off their sweep median "
              "(exposure/units?) - ratios unaffected:")
        print(anom.to_string())

    print("\n=== pair summary (sorted by |S| within each N line) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))
    print("\nwritten to", cfg["outdir"])
    return dict(n_lines=n_lines, a_lines=a_lines, meas=meas, per=per, cond=cond, summary=summary, xs=xs)


if __name__ == "__main__":
    RESULTS = run(CONFIG)
