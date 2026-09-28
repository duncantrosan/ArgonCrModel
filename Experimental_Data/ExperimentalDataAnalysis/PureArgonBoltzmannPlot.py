"""
SpectrumLineRatios.py

Experimental counterpart to the CR-model emission output: takes a measured
argon spectrum plus the same ArgonLevelList.json / ArgonReactionList.json the
CR model uses, and produces

  * a line table (one row per NIST transition that lands in the spectrum)
  * I_matrix[N_state, N_state]  – integrated intensity of upper -> lower, NaN elsewhere
  * R_matrix[N_state, N_state]  – I_matrix normalised to a reference line
  * a Boltzmann plot  ln( I*lambda / (g_k A_ki) )  vs  E_k

Indexing convention matches the level JSON: row/col index == level["id"],
row = upper level, col = lower level.

Usage (see __main__ at the bottom):
    levels, labels = LoadLevels("ArgonLevelList.json")
    trans          = LoadTransitions("ArgonReactionList.json", levels)
    wl, I          = LoadSpectrum("Percent_1_0__1.spa")
    lines          = MeasureLines(wl, I, trans)
    I_mat          = BuildIntensityMatrix(lines, len(levels))
    R_mat, ref     = BuildRatioMatrix(I_mat, lines, ref_wl=750.387)
    BoltzmannPlot(lines, levels)
"""

import json
import csv
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

kB_eV = 8.617333262e-5  # eV/K


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def LoadLevels(path):
    """Return (levels dict keyed by label, labels array indexed by id)."""
    with open(path) as f:
        levels = json.load(f)
    N = max(v["id"] for v in levels.values()) + 1
    labels = np.empty(N, dtype=object)
    for lab, v in levels.items():
        labels[v["id"]] = lab
    return levels, labels


def _LevelKey(config, term, J):
    return (config, term, float(J))


def LoadTransitions(path, levels, wl_min=None, wl_max=None):
    """
    Read the NIST transition list and attach upper/lower level labels + ids.
    Transitions whose upper OR lower level is not in the level list are dropped
    (the CR model doesn't track them anyway). Aki == None also dropped.
    """
    with open(path) as f:
        raw = json.load(f)["transitions"]

    lut = {_LevelKey(v["configuration"], v["term"], v["J"]): lab
           for lab, v in levels.items()}

    trans = []
    for t in raw:
        u = lut.get(_LevelKey(t["upper"]["config"], t["upper"]["term"], t["upper"]["J"]))
        l = lut.get(_LevelKey(t["lower"]["config"], t["lower"]["term"], t["lower"]["J"]))
        if u is None or l is None or t["Aki"] is None:
            continue
        if wl_min is not None and t["wl_nm"] < wl_min:
            continue
        if wl_max is not None and t["wl_nm"] > wl_max:
            continue
        trans.append({
            "wl_nm": t["wl_nm"],
            "Aki": t["Aki"],
            "acc": t["acc"],
            "upper": u, "lower": l,
            "u_id": levels[u]["id"], "l_id": levels[l]["id"],
            "g_u": levels[u]["g"], "E_u": levels[u]["energy_eV"],
            "g_l": levels[l]["g"], "E_l": levels[l]["energy_eV"],
        })
    trans.sort(key=lambda t: t["wl_nm"])
    return trans


def LoadSpectrum(path, skiprows=1):
    """Two-column (wavelength nm, intensity) text file with one header line."""
    d = np.loadtxt(path, skiprows=skiprows)
    order = np.argsort(d[:, 0])
    return d[order, 0], d[order, 1]


def FindGaps(wl, I, min_width_nm=0.3):
    """
    Echelle-style stitched spectra have zero-filled order gaps. Return a list
    of (wl_start, wl_end) for every exactly-zero run wider than min_width_nm.
    """
    z = (I == 0)
    gaps = []
    if not z.any():
        return gaps
    edges = np.flatnonzero(np.diff(z.astype(int)))
    start = 0 if z[0] else None
    for e in edges:
        if z[e + 1]:
            start = e + 1
        elif start is not None:
            if wl[e] - wl[start] >= min_width_nm:
                gaps.append((wl[start], wl[e]))
            start = None
    if z[-1] and start is not None and wl[-1] - wl[start] >= min_width_nm:
        gaps.append((wl[start], wl[-1]))
    return gaps


def InGap(w, gaps, pad=0.05):
    return any(a - pad <= w <= b + pad for a, b in gaps)


# --------------------------------------------------------------------------
# Line fitting
# --------------------------------------------------------------------------
def _ClusterTransitions(trans, width):
    """Group transitions whose NIST wavelengths are within `width` of a neighbour."""
    clusters, cur = [], [trans[0]]
    for t in trans[1:]:
        if t["wl_nm"] - cur[-1]["wl_nm"] <= width:
            cur.append(t)
        else:
            clusters.append(cur)
            cur = [t]
    clusters.append(cur)
    return clusters


def _MultiGauss(x, *p):
    """
    p = [b0, b1, shift, sigma, A_1 ... A_n]; centres are NIST wl + common shift.
    Centres are passed through the closure `_MultiGauss.centres`.
    """
    b0, b1, shift, sigma = p[:4]
    amps = p[4:]
    y = b0 + b1 * (x - x.mean())
    for c, A in zip(_MultiGauss.centres, amps):
        y = y + A * np.exp(-0.5 * ((x - c - shift) / sigma) ** 2)
    return y


def MeasureLines(wl, I, trans,
                 fwhm_guess=0.04,        # instrument FWHM, nm
                 cluster_width=0.3,      # transitions closer than this are fit together
                 window=0.35,            # half-width of fit window beyond cluster edges, nm
                 max_shift=0.06,         # allowed common wavelength offset, nm
                 snr_min=3.0,            # detection threshold on amplitude / residual noise
                 max_rel_err=0.5,        # I_err/I above this -> 'suspect'
                 response_fn=None,       # callable wl_nm -> relative sensitivity (divide out)
                 gaps=None):
    """
    Fit every NIST transition (that lies inside the spectrum) with a sum of
    Gaussians on a linear baseline. Returns a list of dicts, one per transition,
    with integrated intensity, uncertainty, SNR and status flags.

    status: 'ok' | 'weak' (below snr_min) | 'suspect' (shift/width pinned at a
            fit bound, i.e. probably a misidentification, or I_err/I > max_rel_err) |
            'gap' (in an order gap) | 'out_of_range' | 'fit_failed'
    """
    if gaps is None:
        gaps = FindGaps(wl, I)
    sig0 = fwhm_guess / 2.3548
    lines = []

    in_range = [t for t in trans if wl[0] + window < t["wl_nm"] < wl[-1] - window]
    for t in trans:
        if t not in in_range:
            lines.append({**t, "I": np.nan, "I_err": np.nan, "snr": np.nan,
                          "shift": np.nan, "fwhm": np.nan, "status": "out_of_range"})

    for cl in _ClusterTransitions(in_range, cluster_width):
        centres = np.array([t["wl_nm"] for t in cl])
        # gap check per transition
        gap_flags = [InGap(c, gaps) for c in centres]
        if all(gap_flags):
            for t in cl:
                lines.append({**t, "I": np.nan, "I_err": np.nan, "snr": np.nan,
                              "shift": np.nan, "fwhm": np.nan, "status": "gap"})
            continue

        lo, hi = centres.min() - window, centres.max() + window
        m = (wl >= lo) & (wl <= hi) & ~np.isnan(I)
        x, y = wl[m], I[m]
        if len(x) < 8 + len(cl):
            for t in cl:
                lines.append({**t, "I": np.nan, "I_err": np.nan, "snr": np.nan,
                              "shift": np.nan, "fwhm": np.nan, "status": "fit_failed"})
            continue

        # initial guesses
        base0 = np.median(y)
        amp0 = [max(y[np.abs(x - c) < 2 * sig0].max() - base0, 0) if (np.abs(x - c) < 2 * sig0).any() else 0
                for c in centres]
        p0 = [base0, 0.0, 0.0, sig0] + amp0
        lower = [-np.inf, -np.inf, -max_shift, 0.3 * sig0] + [0.0] * len(cl)
        upper = [np.inf, np.inf, max_shift, 4.0 * sig0] + [np.inf] * len(cl)
        _MultiGauss.centres = centres

        try:
            popt, pcov = curve_fit(_MultiGauss, x, y, p0=p0, bounds=(lower, upper), maxfev=20000)
            perr = np.sqrt(np.abs(np.diag(pcov)))
        except (RuntimeError, ValueError):
            for t in cl:
                lines.append({**t, "I": np.nan, "I_err": np.nan, "snr": np.nan,
                              "shift": np.nan, "fwhm": np.nan, "status": "fit_failed"})
            continue

        resid = y - _MultiGauss(x, *popt)
        noise = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-30  # robust sigma
        shift, sigma = popt[2], popt[3]
        for k, t in enumerate(cl):
            A, Aerr = popt[4 + k], perr[4 + k]
            Iint = A * sigma * np.sqrt(2 * np.pi)
            # propagate amp + sigma errors (ignoring covariance between them)
            Ierr = Iint * np.sqrt((Aerr / A) ** 2 + (perr[3] / sigma) ** 2) if A > 0 else np.nan
            if response_fn is not None:
                s = response_fn(t["wl_nm"])
                Iint, Ierr = Iint / s, Ierr / s
            snr = A / noise
            at_bound = (abs(abs(shift) - max_shift) < 0.02 * max_shift
                        or abs(sigma - lower[3]) < 0.02 * sig0
                        or abs(sigma - upper[3]) < 0.02 * sig0)
            if gap_flags[k]:
                status, Iint, Ierr = "gap", np.nan, np.nan
            elif snr < snr_min:
                status = "weak"
            elif at_bound or not np.isfinite(Ierr) or Ierr > max_rel_err * Iint:
                status = "suspect"   # fit parameter pinned at a bound, or error > max_rel_err
            else:
                status = "ok"
            lines.append({**t, "I": Iint, "I_err": Ierr, "snr": snr, "shift": shift,
                          "fwhm": 2.3548 * sigma, "status": status,
                          "blend": [u["wl_nm"] for u in cl if u is not t]})

    lines.sort(key=lambda d: d["wl_nm"])
    return lines


# --------------------------------------------------------------------------
# Matrices
# --------------------------------------------------------------------------
def BuildIntensityMatrix(lines, N, use=("ok",)):
    """
    I_mat[u_id, l_id] = integrated intensity of upper->lower. NaN where not measured.
    If two transitions map to the same (u,l) pair (shouldn't happen for E1) they're summed.
    """
    I_mat = np.full((N, N), np.nan)
    for ln in lines:
        if ln["status"] in use and np.isfinite(ln["I"]):
            i, j = ln["u_id"], ln["l_id"]
            I_mat[i, j] = ln["I"] if np.isnan(I_mat[i, j]) else I_mat[i, j] + ln["I"]
    return I_mat


def FindLine(lines, ref_wl=None, ref_pair=None, tol=0.05):
    """Locate a line by wavelength (nm) or by (upper_label, lower_label)."""
    for ln in lines:
        if ref_pair is not None and (ln["upper"], ln["lower"]) == tuple(ref_pair):
            return ln
        if ref_wl is not None and abs(ln["wl_nm"] - ref_wl) < tol:
            return ln
    raise KeyError(f"reference line not found: wl={ref_wl} pair={ref_pair}")


def BuildRatioMatrix(I_mat, lines, ref_wl=750.387, ref_pair=None):
    """R_mat[u,l] = I(u->l) / I(ref). Default ref is 4p10->4s4 (2p1->1s2) at 750.387 nm."""
    ref = FindLine(lines, ref_wl=ref_wl, ref_pair=ref_pair)
    if not np.isfinite(ref["I"]) or ref["status"] != "ok":
        raise ValueError(f"reference line {ref['wl_nm']} nm not usable (status={ref['status']})")
    return I_mat / ref["I"], ref


def PairwiseRatioMatrix(lines, use=("ok",)):
    """
    Small n_lines x n_lines matrix P[a,b] = I_a / I_b over the usable lines,
    plus the list of lines defining the row/col order.
    """
    good = [ln for ln in lines if ln["status"] in use and np.isfinite(ln["I"])]
    Iv = np.array([ln["I"] for ln in good])
    return Iv[:, None] / Iv[None, :], good


def LineRatio(lines, a, b):
    """Ratio of two lines; a, b are wavelengths (nm) or (upper, lower) label tuples."""
    la = FindLine(lines, ref_wl=a) if np.isscalar(a) else FindLine(lines, ref_pair=a)
    lb = FindLine(lines, ref_wl=b) if np.isscalar(b) else FindLine(lines, ref_pair=b)
    return la["I"] / lb["I"]


def SaveLineTable(lines, path):
    cols = ["wl_nm", "upper", "lower", "u_id", "l_id", "E_u", "g_u", "Aki", "acc",
            "I", "I_err", "snr", "shift", "fwhm", "status"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for ln in lines:
            w.writerow([ln.get(c, "") for c in cols])


def SaveMatrix(M, labels, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["upper\\lower"] + list(labels))
        for i, row in enumerate(M):
            w.writerow([labels[i]] + ["" if np.isnan(v) else f"{v:.6g}" for v in row])


# --------------------------------------------------------------------------
# Boltzmann plot
# --------------------------------------------------------------------------
def BoltzmannData(lines, intensity_units="energy", use=("ok",), min_acc=None):
    """
    y = ln( I * lambda / (g_k A_ki) )   for energy-calibrated (W-like) intensities
    y = ln( I          / (g_k A_ki) )   for photon-counting intensities
    Returns arrays (E_u, y, y_err, lines_used).
    """
    rank = {"AA": 0, "A+": 1, "A": 2, "B+": 3, "B": 4, "C+": 5, "C": 6, "D+": 7, "D": 8, "E": 9}
    used = []
    for ln in lines:
        if ln["status"] not in use or not np.isfinite(ln["I"]) or ln["I"] <= 0:
            continue
        if min_acc is not None and rank.get(ln["acc"], 99) > rank[min_acc]:
            continue
        used.append(ln)
    E = np.array([ln["E_u"] for ln in used])
    Iv = np.array([ln["I"] for ln in used])
    Ie = np.array([ln["I_err"] for ln in used])
    gA = np.array([ln["g_u"] * ln["Aki"] for ln in used])
    lam = np.array([ln["wl_nm"] for ln in used])
    y = np.log(Iv / gA) + (np.log(lam) if intensity_units == "energy" else 0.0)
    yerr = Ie / Iv
    return E, y, yerr, used


def FitBoltzmann(E, y, yerr=None):
    """Weighted linear fit; returns (T_exc_K, T_err_K, slope, intercept)."""
    w = None if yerr is None else 1.0 / np.maximum(yerr, 1e-3)
    p, cov = np.polyfit(E, y, 1, w=w, cov=True)
    slope, inter = p
    T = -1.0 / (kB_eV * slope)
    T_err = T * np.sqrt(cov[0, 0]) / abs(slope)
    return T, T_err, slope, inter


def BoltzmannPlot(lines, levels, intensity_units="energy", use=("ok",), min_acc=None,
                  ax=None, title=None, annotate=True):
    import matplotlib.pyplot as plt
    E, y, yerr, used = BoltzmannData(lines, intensity_units, use, min_acc)
    T, T_err, slope, inter = FitBoltzmann(E, y, yerr)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
    manifolds = sorted({ln["upper"][:2] for ln in used}, key=lambda m: (m[1], m[0]))
    colors = dict(zip(manifolds, plt.cm.tab10.colors))
    good_acc = {"AA", "A+", "A", "B+", "B", "C+"}
    for ln, e, yy, ye in zip(used, E, y, yerr):
        mf = ln["upper"][:2]
        mk = "o" if ln["acc"] in good_acc else "x"
        ax.errorbar(e, yy, yerr=ye, fmt=mk, color=colors[mf], ms=6, capsize=2,
                    mfc="none" if mk == "o" else None)
        if annotate:
            ax.annotate(f"{ln['wl_nm']:.1f}", (e, yy), textcoords="offset points",
                        xytext=(3, 3), fontsize=6, alpha=0.8)
    xx = np.array([E.min() - 0.1, E.max() + 0.1])
    ax.plot(xx, slope * xx + inter, "k--", lw=1,
            label=f"fit: T_exc = {T:.0f} ± {T_err:.0f} K ({kB_eV*T:.2f} eV)")
    for mf in manifolds:
        ax.plot([], [], "o", color=colors[mf], label=f"upper {mf}")
    ax.plot([], [], "kx", label="A_ki accuracy ≤ C")
    ax.set_xlabel("Upper level energy E_k (eV)")
    ax.set_ylabel("ln( I λ / g_k A_ki )" if intensity_units == "energy" else "ln( I / g_k A_ki )")
    ax.set_title(title or "Ar I Boltzmann plot")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax, (T, T_err, slope, inter), used


def PerLevelConsistency(lines, intensity_units="energy", use=("ok",)):
    """
    Lines sharing an upper level must give the same Boltzmann ordinate
    regardless of T_exc. Spread within a level == calibration / A-value / blend error.
    Returns {upper_label: (E_u, mean_y, std_y, n_lines, [wl...])}.
    """
    E, y, yerr, used = BoltzmannData(lines, intensity_units, use)
    out = {}
    for lab in sorted({ln["upper"] for ln in used}):
        idx = [i for i, ln in enumerate(used) if ln["upper"] == lab]
        out[lab] = (E[idx[0]], y[idx].mean(), y[idx].std() if len(idx) > 1 else 0.0,
                    len(idx), [used[i]["wl_nm"] for i in idx])
    return out


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def PlotSpectrumWithLines(wl, I, lines, gaps=None, ax=None, wl_range=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4.5))
    m = np.ones_like(wl, bool) if wl_range is None else (wl >= wl_range[0]) & (wl <= wl_range[1])
    ax.plot(wl[m], I[m], lw=0.6, color="0.2")
    if gaps:
        for a, b in gaps:
            ax.axvspan(a, b, color="0.85", lw=0)
    ymax = np.nanmax(I[m])
    for ln in lines:
        if wl_range and not (wl_range[0] <= ln["wl_nm"] <= wl_range[1]):
            continue
        c = {"ok": "C3", "weak": "C1", "suspect": "C4", "gap": "0.6"}.get(ln["status"], "C7")
        ax.axvline(ln["wl_nm"], color=c, lw=0.5, alpha=0.6)
        if ln["status"] == "ok":
            ax.annotate(f"{ln['upper']}→{ln['lower']}\n{ln['wl_nm']:.2f}",
                        (ln["wl_nm"], min(ln["I"] / (ln["fwhm"] / 2.3548 * 2.5066), ymax)),
                        rotation=90, fontsize=5, ha="center", va="bottom")
    if wl_range:
        ax.set_xlim(*wl_range)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (a.u.)")
    return ax


def PlotMatrix(M, labels, ax=None, title="", log=True, only_nonempty=True):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    rows = np.arange(M.shape[0]); cols = np.arange(M.shape[1])
    if only_nonempty:
        rows = rows[np.isfinite(M).any(axis=1)]
        cols = cols[np.isfinite(M).any(axis=0)]
    sub = M[np.ix_(rows, cols)]
    norm = LogNorm(vmin=np.nanmin(sub[sub > 0]), vmax=np.nanmax(sub)) if log else None
    im = ax.imshow(sub, cmap="viridis", norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels[cols], rotation=90, fontsize=7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels[rows], fontsize=7)
    ax.set_xlabel("lower level"); ax.set_ylabel("upper level"); ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return ax


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    level_file = sys.argv[1] if len(sys.argv) > 1 else "ArgonLevelList.json"
    trans_file = sys.argv[2] if len(sys.argv) > 2 else "ArgonReactionList.json"
    spec_file  = sys.argv[3] if len(sys.argv) > 3 else "Percent_1_0__1.spa"
    outdir     = sys.argv[4] if len(sys.argv) > 4 else "."
    os.makedirs(outdir, exist_ok=True)

    levels, labels = LoadLevels(level_file)
    trans = LoadTransitions(trans_file, levels)
    wl, I = LoadSpectrum(spec_file)
    gaps = FindGaps(wl, I)
    print(f"{len(trans)} transitions map onto the {len(labels)}-level list; "
          f"spectrum {wl[0]:.1f}–{wl[-1]:.1f} nm with {len(gaps)} order gaps")

    lines = MeasureLines(wl, I, trans, gaps=gaps)
    from collections import Counter
    print("line status:", dict(Counter(ln["status"] for ln in lines)))

    print(f"\n{'wl':>8} {'upper':>5}->{'lower':<5} {'I':>10} {'±':>8} {'SNR':>7} {'shift':>6} {'FWHM':>5}  status")
    for ln in lines:
        if ln["status"] in ("ok", "weak", "suspect"):
            print(f"{ln['wl_nm']:8.3f} {ln['upper']:>5}->{ln['lower']:<5} {ln['I']:10.3e} "
                  f"{ln['I_err']:8.1e} {ln['snr']:7.1f} {ln['shift']:6.3f} {ln['fwhm']:5.3f}  {ln['status']}")

    N = len(labels)
    I_mat = BuildIntensityMatrix(lines, N)
    R_mat, ref = BuildRatioMatrix(I_mat, lines, ref_wl=750.387)
    print(f"\nreference: {ref['upper']}->{ref['lower']} {ref['wl_nm']} nm, I = {ref['I']:.3e}")

    SaveLineTable(lines, os.path.join(outdir, "ar_lines.csv"))
    SaveMatrix(I_mat, labels, os.path.join(outdir, "I_matrix.csv"))
    SaveMatrix(R_mat, labels, os.path.join(outdir, "R_matrix.csv"))
    np.save(os.path.join(outdir, "I_matrix.npy"), I_mat)
    np.save(os.path.join(outdir, "R_matrix.npy"), R_mat)

    # ---- figures
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    PlotSpectrumWithLines(wl, I, lines, gaps, ax=axes[0], wl_range=(390, 480))
    PlotSpectrumWithLines(wl, I, lines, gaps, ax=axes[1], wl_range=(660, 870))
    axes[0].set_title("Ar I line identification (grey = order gaps, red = ok, orange = below SNR, purple = suspect)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "spectrum_lines.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(8, 7))
    PlotMatrix(R_mat, labels, ax=ax, title=f"I(u→l) / I({ref['wl_nm']} nm)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "ratio_matrix.png"), dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    BoltzmannPlot(lines, levels, ax=axes[0], title="All fitted lines")
    BoltzmannPlot(lines, levels, ax=axes[1], min_acc="C+", title="A_ki accuracy ≥ C+ only")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "boltzmann.png"), dpi=150)

    print("\nPer-upper-level consistency (std of ln(Iλ/gA) among lines sharing an upper level):")
    for lab, (E, m, s, n, wls) in PerLevelConsistency(lines).items():
        if n > 1:
            print(f"  {lab:>5}  E={E:.3f}  n={n}  std={s:.2f}   lines={[round(w,1) for w in wls]}")
