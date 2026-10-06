# -*- coding: utf-8 -*-
"""
ActinometryRates.py

Electron-impact excitation rate coefficients for the N I / Ar I actinometry
pairs, from the LXCat ground-state cross sections, for either a supplied EEDF
or a Maxwellian at Te.  Also the first pieces of the actinometry error budget.

Actinometric relation (direct excitation from the ground state, radiative
de-excitation only, same optical path / response for both lines):

    I_N / I_Ar = (n_N / n_Ar) * (k_N * b_N) / (k_Ar * b_Ar) * (lam_Ar / lam_N)**p

    k   = sqrt(2e/m_e) * int sqrt(E) sigma(E) f(E) dE      [m^3 s^-1]
    b   = A_ul / sum_l A_ul    branching ratio of the observed line
    p   = 1 if intensities are energy (W), 0 if photon counts
    f   = EEDF normalised so int f(E) dE = 1  [eV^-1]

so  n_N / n_Ar = (I_N / I_Ar) * (k_Ar b_Ar) / (k_N b_N) * (lam_N / lam_Ar)**p

Cross-section sources
    N I : AtomicNitrogen_LevelData.json  (BSR term-resolved, statistically
          J-split by Build_Nitrogen_Level_Data.py)
    Ar I: InputData/ArgonCrossSections_BSR.json (the CR model's set), ground-state 'excitation' entries.
          The 5p entries carry no model label there, so they are matched to
          ArgonLevelList.json by threshold energy and J.

Usage (Spyder / console):
    import ActinometryRates as ar
    XS = ar.load_all_cross_sections()
    ar.actinometry_rates("other21", "4p10", Te=3.0, xs=XS)          # 746.8/744.2 N vs 750.4 Ar
    ar.actinometry_rates("other21", "4p10", eedf=(E, f), xs=XS)      # any EEDF
    ar.actinometry_rates("other21", "4p10", Te=np.linspace(1, 6, 50), xs=XS)
"""
import json
import os
import re

import numpy as np
from scipy import constants

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PATHS = dict(
    n_level_data=os.path.join(ROOT_DIR, "Experimental_Data", "EchelleData", "TestData", "AtomicNitrogen_LevelData.json"),
    ar_cross_sections=os.path.join(ROOT_DIR, "InputData", "ArgonCrossSections_BSR.json"),
    ar_levels=os.path.join(ROOT_DIR, "InputData", "ArgonLevelList.json"),
    ar_transitions=os.path.join(ROOT_DIR, "InputData", "ArgonReactionList.json"),
)

E_CHARGE = constants.e
M_E = constants.m_e
V_FACTOR = np.sqrt(2 * E_CHARGE / M_E)          # v = V_FACTOR * sqrt(E[eV])  [m/s]
E_GRID = np.linspace(0.0, 150.0, 15001)          # default integration grid [eV]


# ----------------------------------------------------------------------------
# 1. EEDFs  (all normalised so int f dE = 1, f in eV^-1)
# ----------------------------------------------------------------------------
def maxwellian_eedf(E, Te):
    """Maxwellian EEDF, <E> = 3/2 Te."""
    E = np.asarray(E, float)
    return 2.0 / np.sqrt(np.pi) * Te ** -1.5 * np.sqrt(E) * np.exp(-E / Te)


def druyvesteyn_eedf(E, Te):
    """Druyvesteyn EEDF with the same mean energy as a Maxwellian at Te
    (<E> = 3/2 Te).  Useful as an EEDF-shape systematic."""
    E = np.asarray(E, float)
    Em = 1.5 * Te
    f = np.sqrt(E) * np.exp(-0.55 * (E / Em) ** 2)   # shape; normalised below
    return _normalise(E, f)


def _normalise(E, f):
    norm = np.trapezoid(f, E)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("EEDF has zero or non-finite integral")
    return f / norm


def _eedf_on_grid(E, eedf=None, Te=None, kind="maxwellian"):
    """Return f(E) on grid E from either an EEDF or Te.

    eedf : (E_data, f_data) arrays  -> interpolated onto E, zero outside
           callable f(E)            -> evaluated on E
    Te   : float [eV]               -> analytic EEDF of the given kind
    """
    if eedf is not None:
        if callable(eedf):
            f = np.asarray(eedf(E), float)
        else:
            E_d, f_d = (np.asarray(a, float) for a in eedf)
            f = np.interp(E, E_d, f_d, left=0.0, right=0.0)
        return _normalise(E, np.clip(f, 0, None))
    if Te is None:
        raise ValueError("give either eedf or Te")
    if kind == "maxwellian":
        return maxwellian_eedf(E, Te)
    if kind == "druyvesteyn":
        return druyvesteyn_eedf(E, Te)
    raise ValueError(f"unknown EEDF kind {kind!r}")


# ----------------------------------------------------------------------------
# 2. Cross sections
# ----------------------------------------------------------------------------
def load_n_cross_sections(path=PATHS["n_level_data"]):
    """{label: xs} for N I levels with a ground-state LXCat cross section."""
    with open(path, encoding="utf-8") as f:
        levels = json.load(f)
    out = {}
    for lab, lev in levels.items():
        exc = lev.get("electron_impact_excitation") or []
        if not exc:
            continue
        e = exc[0]
        out[lab] = dict(species="N I", label=lab, energy_eV=np.asarray(e["energy_eV"], float),
                        cross_section_m2=np.asarray(e["cross_section_m2"], float),
                        threshold_eV=float(e["threshold_eV_lxcat"]),
                        level_energy_eV=float(lev["energy_eV"]),
                        desc=f'{lev["configuration"].split(".")[-1]} {lev["term"]} {lev["J"]}',
                        source=f'{e["database"]} ({e["lxcat_key"]}, J-split x{e["branching_factor"]:.3f})')
    return out


def load_ar_cross_sections(cs_path=PATHS["ar_cross_sections"], levels_path=PATHS["ar_levels"], tol_eV=5e-3):
    """{label: xs} for Ar I levels with a ground-state cross section.

    Entries with an upper_label use it directly; unlabelled ones (the 5p
    manifold) are matched to the level list by threshold energy and J."""
    with open(cs_path, encoding="utf-8") as f:
        cs = json.load(f)["cross_sections"]
    with open(levels_path, encoding="utf-8") as f:
        levels = json.load(f)
    out = {}
    for c in cs:
        if c.get("type") != "excitation" or c.get("lower_label") != "ground":
            continue
        lab = c.get("upper_label")
        if lab is None:
            m = re.search(r"J\s*=\s*(\d+)", c["lxcat_upper"])
            J = float(m.group(1)) if m else None
            cand = [l for l, v in levels.items()
                    if abs(v["energy_eV"] - c["threshold_eV"]) < tol_eV and (J is None or v.get("J") == J)]
            if len(cand) != 1:
                continue
            lab = cand[0]
        lev = levels.get(lab, {})
        out[lab] = dict(species="Ar I", label=lab, energy_eV=np.asarray(c["energy_eV"], float),
                        cross_section_m2=np.asarray(c["cross_section"], float),
                        threshold_eV=float(c["threshold_eV"]),
                        level_energy_eV=float(lev.get("energy_eV", c["threshold_eV"])),
                        desc=lev.get("term", ""), source=c["lxcat_reaction"])
    return out


def load_all_cross_sections():
    return {"N I": load_n_cross_sections(), "Ar I": load_ar_cross_sections()}


# ----------------------------------------------------------------------------
# 3. Branching ratios  b = A_line / sum A(upper)
# ----------------------------------------------------------------------------
HC_EV_NM = 1239.841984
_L_OF = {c: i for i, c in enumerate("SPDFGHIK")}


def _term_parts(term):
    """'4S*' -> (mult 4, L 0, odd True); None if not LS-coupled."""
    m = re.match(r"(\d)([SPDFGHIK])(\*?)$", term.strip())
    return (int(m.group(1)), _L_OF[m.group(2)], m.group(3) == "*") if m else None


def _missing_ls_decays(levels, lab, wl_max):
    """Strong decays of level `lab` whose wavelength lies beyond the transition
    list (wl > wl_max).  Strong = LS-allowed E1 single-electron jump (same core,
    only the outer orbital changes).  Non-empty -> the listed A-sum is incomplete."""
    up = levels[lab]
    tu = _term_parts(up["term"])
    if tu is None:
        return []
    core_u = up["configuration"].split(".")[:-1]
    miss = []
    for l2, lo in levels.items():
        dE = up["energy_eV"] - lo["energy_eV"]
        tl = _term_parts(lo["term"])
        if dE <= 0 or tl is None or lo["configuration"].split(".")[:-1] != core_u:
            continue
        if tl[0] != tu[0] or tl[2] == tu[2] or abs(tl[1] - tu[1]) > 1 or (tl[1] == tu[1] == 0):
            continue
        if abs(lo["J"] - up["J"]) > 1 or (lo["J"] == up["J"] == 0):
            continue
        wl = HC_EV_NM / dE
        if wl_max < wl < 3000:
            miss.append((l2, round(wl, 1)))
    return miss


def load_n_A_sums(path=PATHS["n_level_data"], trans_path=None):
    """{label: sum of listed A out of the level}.  Set to NaN when an LS-allowed
    decay lies beyond the long-wavelength end of the NIST export (791.5 nm for
    the current file), since the sum would then be too small."""
    with open(path, encoding="utf-8") as f:
        levels = json.load(f)
    wl_max = max((t["wl_nm"] for lev in levels.values() for t in lev.get("radiative_out", [])), default=np.inf)
    out = {}
    for lab, lev in levels.items():
        s = sum(t["Aki"] for t in lev.get("radiative_out", []) if t.get("Aki"))
        out[lab] = np.nan if _missing_ls_decays(levels, lab, wl_max) else s
    return out


def load_ar_A_sums(levels_path=PATHS["ar_levels"], trans_path=PATHS["ar_transitions"]):
    with open(levels_path, encoding="utf-8") as f:
        levels = json.load(f)
    with open(trans_path, encoding="utf-8") as f:
        trans = json.load(f)["transitions"]
    key = {(v["configuration"], v["term"], float(v["J"])): l for l, v in levels.items()}
    sums = {l: 0.0 for l in levels}
    for t in trans:
        u = t["upper"]
        lab = key.get((u["config"], u["term"], float(u["J"])))
        if lab is not None and t.get("Aki"):
            sums[lab] += t["Aki"]
    return sums


def branching_ratio(Aki, A_sum):
    if Aki is None or not np.isfinite(Aki) or not A_sum:
        return np.nan
    return Aki / A_sum


# ----------------------------------------------------------------------------
# 4. Rate coefficients
# ----------------------------------------------------------------------------
def rate_coefficient(xs, eedf=None, Te=None, kind="maxwellian", E=E_GRID):
    """Excitation rate coefficient k [m^3/s] for one cross section.

    xs   : dict with 'energy_eV', 'cross_section_m2', 'threshold_eV'
    eedf : (E, f) arrays or callable f(E); normalised internally
    Te   : float or array [eV] (used when eedf is None) -> Maxwellian / Druyvesteyn
    Returns a float, or an array matching Te.
    """
    sig = np.interp(E, xs["energy_eV"], xs["cross_section_m2"], left=0.0, right=0.0)
    sig[E < xs["threshold_eV"]] = 0.0
    w = V_FACTOR * np.sqrt(E) * sig
    if eedf is None and np.ndim(Te) > 0:
        return np.array([np.trapezoid(w * _eedf_on_grid(E, Te=t, kind=kind), E) for t in Te])
    return float(np.trapezoid(w * _eedf_on_grid(E, eedf, Te, kind), E))


def actinometry_rates(n_label, ar_label, eedf=None, Te=None, kind="maxwellian", xs=None,
                      n_line=None, ar_line=None, A_sums=None):
    """Rate coefficients of an N I / Ar I actinometry pair.

    n_label / ar_label : upper-level labels (e.g. 'other21', '4p10')
    eedf / Te / kind    : see rate_coefficient
    n_line / ar_line    : optional dicts with 'Aki' and 'wl_air' of the observed
                          lines -> adds branching ratios and the effective
                          emission rate coefficients k*b
    Returns dict with k_N, k_Ar, k_ratio (= k_N/k_Ar) [, b_N, b_Ar, keff_ratio].
    """
    xs = xs or load_all_cross_sections()
    xn, xa = xs["N I"][n_label], xs["Ar I"][ar_label]
    kN = rate_coefficient(xn, eedf, Te, kind)
    kA = rate_coefficient(xa, eedf, Te, kind)
    out = dict(n_label=n_label, ar_label=ar_label, Te=Te, k_N=kN, k_Ar=kA,
               k_ratio=np.asarray(kN) / np.asarray(kA),
               threshold_N=xn["threshold_eV"], threshold_Ar=xa["threshold_eV"])
    if n_line is not None and ar_line is not None:
        A_sums = A_sums or {"N I": load_n_A_sums(), "Ar I": load_ar_A_sums()}
        bN = branching_ratio(n_line.get("Aki"), A_sums["N I"].get(n_label))
        bA = branching_ratio(ar_line.get("Aki"), A_sums["Ar I"].get(ar_label))
        out.update(b_N=bN, b_Ar=bA, keff_ratio=out["k_ratio"] * bN / bA)
    return out


def te_sensitivity(xs_N, xs_Ar, Te, kind="maxwellian", dlnT=0.02):
    """S = d ln(k_N/k_Ar) / d ln Te (central difference).  A relative error
    dTe/Te in Te gives a relative error S*dTe/Te in the derived n_N/n_Ar."""
    Te = np.atleast_1d(np.asarray(Te, float))
    up, dn = Te * np.exp(dlnT), Te * np.exp(-dlnT)
    r_up = rate_coefficient(xs_N, Te=up, kind=kind) / rate_coefficient(xs_Ar, Te=up, kind=kind)
    r_dn = rate_coefficient(xs_N, Te=dn, kind=kind) / rate_coefficient(xs_Ar, Te=dn, kind=kind)
    S = (np.log(r_up) - np.log(r_dn)) / (2 * dlnT)
    return S if S.size > 1 else float(S[0])


# ----------------------------------------------------------------------------
# 5. Error budget (first version)
# ----------------------------------------------------------------------------
def actinometry_density_ratio(I_ratio, rel_err_I, n_label, ar_label, Te, rel_err_Te=0.2,
                              xs=None, n_line=None, ar_line=None, intensity_units="counts",
                              rel_err_xs_N=0.20, rel_err_xs_Ar=0.15, rel_err_response=0.0):
    """n_N / n_Ar from a measured line ratio plus a first-order error budget.

    I_ratio          : measured I_N / I_Ar (response-corrected, or ratio of
                       uncorrected areas if rel_err_response covers it)
    rel_err_I        : relative 1-sigma error of I_ratio (fit + repeat scatter)
    rel_err_Te       : relative 1-sigma error of Te
    rel_err_xs_*     : assumed relative cross-section uncertainties (BSR N, Ar)
    rel_err_response : relative error of the relative spectral response
    intensity_units  : 'counts' (photon) or 'energy'

    All terms are added in quadrature (independent, small errors).  The
    Maxwellian/Druyvesteyn difference is reported separately as an EEDF-shape
    systematic because it is not a random error.
    """
    xs = xs or load_all_cross_sections()
    r = actinometry_rates(n_label, ar_label, Te=Te, xs=xs, n_line=n_line, ar_line=ar_line)
    keff = r.get("keff_ratio", r["k_ratio"])
    if n_line is None or ar_line is None or not np.isfinite(keff):
        keff = r["k_ratio"]
    lam = 1.0
    if intensity_units == "energy" and n_line and ar_line:
        lam = n_line["wl_air"] / ar_line["wl_air"]
    nratio = I_ratio / keff * lam

    S = te_sensitivity(xs["N I"][n_label], xs["Ar I"][ar_label], Te)
    budget = {
        "line ratio": rel_err_I,
        "Te (S*dTe/Te)": abs(S) * rel_err_Te,
        "cross section N": rel_err_xs_N,
        "cross section Ar": rel_err_xs_Ar,
        "spectral response": rel_err_response,
    }
    rel_total = float(np.sqrt(sum(v ** 2 for v in budget.values())))
    rD = actinometry_rates(n_label, ar_label, Te=Te, kind="druyvesteyn", xs=xs)["k_ratio"]
    eedf_sys = float(r["k_ratio"] / rD - 1.0)
    return dict(n_ratio=nratio, rel_err=rel_total, abs_err=rel_total * nratio, budget=budget,
                te_sensitivity=S, eedf_shape_systematic=eedf_sys, rates=r)


def print_budget(res):
    print(f"n_N/n_Ar = {res['n_ratio']:.4g} +- {res['abs_err']:.2g} ({100 * res['rel_err']:.1f} %)")
    for k, v in res["budget"].items():
        print(f"   {k:<20s} {100 * v:6.1f} %")
    print(f"   Te sensitivity S = {res['te_sensitivity']:+.3f}")
    print(f"   EEDF shape (Maxwell vs Druyvesteyn) systematic {100 * res['eedf_shape_systematic']:+.1f} %")


if __name__ == "__main__":
    XS = load_all_cross_sections()
    print(f"{len(XS['N I'])} N I and {len(XS['Ar I'])} Ar I levels with ground-state cross sections")
    for Te in (1.0, 2.0, 3.0, 5.0):
        r = actinometry_rates("other21", "4p10", Te=Te, xs=XS)
        print(f"Te={Te:3.1f} eV  k(N 3p4S)={r['k_N']:.3e}  k(Ar 2p1)={r['k_Ar']:.3e}  ratio={r['k_ratio']:.3f}")
    res = actinometry_density_ratio(0.05, 0.1, "other21", "4p10", Te=3.0, xs=XS)
    print_budget(res)
