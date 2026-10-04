# -*- coding: utf-8 -*-
"""
CrossSectionRatios.py

sigma_N(E) / sigma_Ar(E) for every N I / Ar I actinometry pair selected by
ActinometryLineSelection, on one plot.  An ideal actinometer pair has a flat
ratio (same energy dependence), so k_N/k_Ar would not depend on Te or the
EEDF shape.  Each ratio is normalised to its value at E_NORM so the shapes can
be compared on one axis; the ratio is only drawn above the higher of the two
thresholds.  Run from Spyder, F5.  Figure goes to Experimental_Data/Output/.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "Experimental_Data", "ExperimentalDataAnalysis"))
import ActinometryRates as ar                      # noqa: E402
import ActinometryLineSelection as als             # noqa: E402

E = np.linspace(10.0, 60.0, 2001)                  # eV
E_NORM = 25.0                                      # eV, ratio normalised to 1 here
X_LAB = 10.0                                       # eV, right edge of the ratio-label column
OUT = os.path.join(als.ROOT_DIR, "Experimental_Data", "Output", "cross_section_ratios.png")


def sigma(x):
    s = np.interp(E, x["energy_eV"], x["cross_section_m2"], left=0.0, right=0.0)
    s[E < x["threshold_eV"]] = 0.0
    return s


def run():
    xs = ar.load_all_cross_sections()
    n_lines, a_lines = als.select_lines(als.CONFIG, xs)
    n_up = n_lines.drop_duplicates("upper")
    a_up = a_lines.drop_duplicates("upper")
    cols = plt.cm.tab10(np.arange(len(a_up)) % 10)
    styles = ["-", "--", ":", "-.", (0, (5, 1, 1, 1)), (0, (1, 3))]

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 6))
    for k, al in enumerate(a_up.itertuples()):
        a0.plot(E, sigma(xs["Ar I"][al.upper]), color=cols[k], label=f"Ar {al.wl_air:.2f} ({al.upper})")
    labels = []                                    # (log10 y, x, text, colour) of each ratio start
    for j, nl in enumerate(n_up.itertuples()):
        a0.plot(E, sigma(xs["N I"][nl.upper]), "k", ls=styles[j % len(styles)], label=f"N {nl.wl_air:.2f} ({nl.upper})")
        sN = sigma(xs["N I"][nl.upper])
        for k, al in enumerate(a_up.itertuples()):
            sA = sigma(xs["Ar I"][al.upper])
            ok = (sA > 0) & (sN > 0)
            r = np.where(ok, sN / np.where(ok, sA, 1), np.nan)
            r /= np.interp(E_NORM, E[ok], r[ok])
            a1.plot(E, r, color=cols[k], ls=styles[j % len(styles)], lw=1.3)
            i0 = np.argmax(np.isfinite(r) & (r > 0.1) & (r < 10))    # first point shown
            labels.append((np.log10(r[i0]), E[i0], f"{nl.wl_air:.1f}/{al.wl_air:.1f}", cols[k]))
    # label column left of the curves, spread so labels don't overlap, leader line to each start
    labels.sort()
    y = np.array([l[0] for l in labels])
    gap = 1.1 / len(y)                             # decades per label over the 0.1-10 axis
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i - 1] + gap)
    y -= max(0.0, y[-1] - 0.97)                    # keep the top label inside the axis
    for i in range(len(y) - 2, -1, -1):
        y[i] = min(y[i], y[i + 1] - gap)
    for yl, (y0, x0, txt, c) in zip(y, labels):
        a1.plot([X_LAB + 0.2, x0], [10 ** yl, 10 ** y0], color=c, lw=0.4, alpha=0.6)
        a1.text(X_LAB, 10 ** yl, txt, color=c, fontsize=6.5, ha="right", va="center")
    a0.set(xlabel="electron energy [eV]", ylabel="cross section [m$^2$]", yscale="log",
           title="excitation cross sections (N: black, Ar: colour)")
    a0.legend(fontsize=7)
    a1.axhline(1, color="k", lw=0.8)
    a1.set(xlabel="electron energy [eV]", ylabel=f"$\sigma_N/\sigma_{{Ar}}$ / value at {E_NORM:g} eV",
           yscale="log", ylim=(0.1, 10), xlim=(X_LAB - 5.5, 61),
           title="cross-section ratio (colour = Ar line, style = N line); flat = ideal")
    for ax in (a0, a1):
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("saved", OUT)
    plt.show()


if __name__ == "__main__":
    run()
