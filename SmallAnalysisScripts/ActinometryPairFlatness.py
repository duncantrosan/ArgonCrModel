# -*- coding: utf-8 -*-
"""
ActinometryPairFlatness.py

How much k_N/k_Ar of each N I / Ar I pair changes over a Te window, for a
Maxwellian EEDF and for the MultiBolt EEDFs used in the CR fit (plotted against
their Te_eff = 2/3 <eps>).  The swing factor  max/min  of the ratio over TE_WIN is
the actinometry error from not knowing Te inside that window; 1 = ideal pair.
The MultiBolt Te_eff barely moves while the >12 eV tail changes by orders of
magnitude, so the MultiBolt swings are much larger than the Maxwellian ones.

Left : heat maps of the swing factor (Maxwellian | MultiBolt).
Right: k ratio / its value at TE_NORM vs Te for the BEST_N best pairs (by the
       Maxwellian swing), solid = Maxwellian, dashed = MultiBolt.
Run from Spyder, F5.  Figure + csv go to Experimental_Data/Output/.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "Experimental_Data", "ExperimentalDataAnalysis"))
import ActinometryRates as ar                      # noqa: E402
import ActinometryLineSelection as als             # noqa: E402
import CRFitNeTe as crf                            # noqa: E402

TE_WIN = (1.0, 4.0)                                # eV, window the swing is taken over
TE_NORM = 2.0                                      # eV, curves normalised to 1 here
BEST_N = 8                                         # pairs drawn on the right
TE = np.linspace(0.5, 6.0, 111)                    # Maxwellian Te axis
MARGIN_EV = 1.0                                    # MultiBolt EEDF must reach threshold + this
OUTDIR = os.path.join(als.ROOT_DIR, "Experimental_Data", "Output")


def multibolt_k(xs, eedfs):
    """k along the MultiBolt sweep; NaN where the EEDF export ends too close to threshold."""
    k = np.array([ar.rate_coefficient(xs, eedf=(e["E"], e["EEDF"])) for e in eedfs])
    E_max = np.array([e["E"][e["EEDF"] > 0].max() for e in eedfs])
    return np.where((E_max >= xs["threshold_eV"] + MARGIN_EV) & (k > 0), k, np.nan)


def swing(Te, r):
    m = (Te >= TE_WIN[0]) & (Te <= TE_WIN[1]) & np.isfinite(r)
    return np.nanmax(r[m]) / np.nanmin(r[m]) if m.sum() > 1 else np.nan


def run():
    xs = ar.load_all_cross_sections()
    n_lines, a_lines = als.select_lines(als.CONFIG, xs)
    n_up, a_up = n_lines.drop_duplicates("upper"), a_lines.drop_duplicates("upper")
    eedfs, _, _, Te_mb = crf.eedf_axis(crf.CONFIG)
    print(f"MultiBolt Te_eff range {Te_mb.min():.2f}-{Te_mb.max():.2f} eV")

    kM = {("N I", u): ar.rate_coefficient(xs["N I"][u], Te=TE) for u in n_up.upper}
    kM |= {("Ar I", u): ar.rate_coefficient(xs["Ar I"][u], Te=TE) for u in a_up.upper}
    kB = {("N I", u): multibolt_k(xs["N I"][u], eedfs) for u in n_up.upper}
    kB |= {("Ar I", u): multibolt_k(xs["Ar I"][u], eedfs) for u in a_up.upper}

    rows, curves = [], {}
    for nl in n_up.itertuples():
        for al in a_up.itertuples():
            rM = kM["N I", nl.upper] / kM["Ar I", al.upper]
            rB = kB["N I", nl.upper] / kB["Ar I", al.upper]
            lab = f"{nl.wl_air:.1f}/{al.wl_air:.1f}"
            curves[lab] = (rM, rB)
            rows.append(dict(pair=lab, n_wl=nl.wl_air, n_upper=nl.upper, ar_wl=al.wl_air, ar_upper=al.upper,
                             swing_maxwell=swing(TE, rM), swing_multibolt=swing(Te_mb, rB)))
    df = pd.DataFrame(rows).sort_values("swing_maxwell").reset_index(drop=True)
    df.to_csv(os.path.join(OUTDIR, "actinometry_pair_flatness.csv"), index=False)
    print(df.to_string(formatters={c: "x{:.3g}".format for c in ("swing_maxwell", "swing_multibolt")}))

    fig = plt.figure(figsize=(18, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.4])
    from matplotlib.colors import LogNorm
    norm = LogNorm(1.0 + 1e-3, np.nanmax(df[["swing_maxwell", "swing_multibolt"]].values))
    for i, (col, name) in enumerate((("swing_maxwell", "Maxwellian"), ("swing_multibolt", "MultiBolt"))):
        ax = fig.add_subplot(gs[i])
        M = df.pivot(index="n_wl", columns="ar_wl", values=col).reindex(
            index=n_up.wl_air, columns=a_up.wl_air)
        im = ax.imshow(M.values, cmap="viridis_r", norm=norm, aspect="auto")
        for (a, b), v in np.ndenumerate(M.values):
            if np.isfinite(v):
                ax.text(b, a, f"{v:.2g}" if v < 1e3 else f"{v:.0e}", ha="center", va="center", fontsize=7,
                        color="w" if np.log(v) > 0.6 * np.log(norm.vmax) else "k")
        ax.set_xticks(range(len(a_up)), [f"{a.wl_air:.1f}\n{a.upper}" for a in a_up.itertuples()], fontsize=7)
        ax.set_yticks(range(len(n_up)), [f"N {n.wl_air:.1f}\n{n.upper}" for n in n_up.itertuples()], fontsize=8)
        ax.set_title(f"{name}: max/min of k$_N$/k$_{{Ar}}$, Te {TE_WIN[0]:g}-{TE_WIN[1]:g} eV")
        ax.set_xlabel("Ar I line [nm]")
    fig.colorbar(im, ax=ax, fraction=0.04)

    ax = fig.add_subplot(gs[2])
    ax.axvspan(*TE_WIN, color="0.9")
    for k, lab in enumerate(df.pair.head(BEST_N)):
        c = plt.cm.tab10(k % 10)
        rM, rB = curves[lab]
        ax.plot(TE, rM / np.interp(TE_NORM, TE, rM), color=c, label=lab)
        ok = np.isfinite(rB)
        if ok.sum() > 1:
            ax.plot(Te_mb[ok], rB[ok] / np.interp(TE_NORM, Te_mb[ok], rB[ok]), color=c, ls="--")
    ax.axhline(1, color="k", lw=0.8)
    ax.set(xlabel="Te  (Maxwellian)  /  Te$_{eff}$  (MultiBolt, dashed)  [eV]",
           ylabel=f"k$_N$/k$_{{Ar}}$ / value at {TE_NORM:g} eV", yscale="log",
           title=f"{BEST_N} flattest pairs (Maxwellian swing)")
    ax.legend(fontsize=8, title="N / Ar [nm]")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(OUTDIR, "actinometry_pair_flatness.png")
    fig.savefig(out, dpi=150)
    print("saved", out)
    plt.show()
    return df


if __name__ == "__main__":
    DF = run()
