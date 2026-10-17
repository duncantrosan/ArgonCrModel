# -*- coding: utf-8 -*-
"""
CRTableFit.py  -  CR tables on an (E/N, Ne) grid and the fit of the Ar I line intensities with them
=======================================================================================================================
TAKES IN   a settings dict cfg (CONFIG, plus per condition: the EEDF library folder, P_Torr, Tg, R, the admixture
           admixture_gas / N2_percent, the Ne grid, the output folder of the table); CONFIG can be changed from
           outside through the environment variable CR_CONFIG_OVERRIDES (Main.py); measured line features
GIVES OUT  build_model_table(cfg)   the CR model on the (EEDF axis, Ne) grid: densities of all levels and the line
                                    intensities (cached as .npz, rebuilt only when the settings differ)
           select_features / measure / feature_table / repeat_scatter   the measured line features of the spectra
           fine_grid / feature_model / fit_all   the fit: posterior over (E/N, Ne) of every condition, one free
                                    scale per spectrum, model error sigma_model, Birge widening
           chi2_grid, pinned_at     chi^2 from a posterior; the fit with Ne pinned
           cr_state_solver(cfg)     the CR model at any (EEDF, Ne): the full level dict (for the results dict)
           admixture_quenching / check_resonant_quenching / apply_nitrogen / quenching_check   quenching of the Ar
                                    levels by the admixed gas (rates from the two quenching tables) and dilution
           load_cr_model            the functions of CRModelSolver.py and the model data
USED BY    LASAndEEDFLibraries, CRTableBuilder, RunDatasets, the actinometry modules, and most analysis scripts
           (through the old name CRFitNeTe)
=======================================================================================================================
(Before the pipeline was collected in pipeline/ this file was Experimental_Data/ExperimentalDataAnalysis/CRFitNeTe.py.)

Details:

Fit Te and Ne of every sweep condition by comparing measured Ar I line
intensities with the argon CR model (Scripts/MainFileV2.py).  The N2 in the
discharge is ignored for now (pure-Ar CR model at the total pressure).

1. Lines: Ar I lines between CR-model levels, picked from the single-spectrum
   analysis (TestData/ArN2_out/lines_mix.csv + manual ratings): detected,
   clean, inside the fit wavelength ranges.  Lines closer than merge_nm are
   merged into one feature (measured and modelled intensities summed), since
   the group fitter cannot split them reliably.
2. Measurement: every spectrum of both sweeps is fitted with the group fitter
   of Argon_Nitrogen_Mix_Analysis_V2 (through ActinometryLineSelection).
   Spectra exported without the intensity calibration (header flags, see
   ActinometryLineSelection.filter_export_flags) are dropped.  Cached.
3. Model: CR model on an (EEDF, Ne) grid at the sweep pressure.  The EEDF axis
   is either a Maxwellian Te sweep (eedf='maxwell') or the E/N sweep of a
   MultiBolt export (eedf=<run folder>; Boltzmann EEDFs with the inelastic
   depletion of the tail) or of a BOLSIG+ library (he.RunBolsig).  A BOLSIG+
   library built with e-e and/or superelastic collisions (he.BuildBolsigLibrary)
   has one EEDF per (E/N, Ne) grid point.  Everything is reported against
   Te_eff = 2/3 <E> (= Te for a Maxwellian).  Stored per grid point: observed line intensities
   n_u A eta (photons; eta is the escape factor of the line, same as in the
   balance equations), level densities, and the fraction of each level's
   production that is direct excitation from the ground state.  Cached.
4. Fit, per condition (all repeats jointly) and per spectrum (as a check):
       ln I_meas = ln I_model(x, Ne) + s_spectrum + ln R(lambda) + noise
   x          : Te or E/N (the EEDF axis)
   s_spectrum : free scale per spectrum (exposure, absolute calibration)
   ln R       : residual spectral response - switched OFF: response_deg must be
                None (the intensity calibration is trusted; fit_block raises otherwise)
   covariance : per point the line-fit error (1/SNR plus a floor) and the
                measured repeat-to-repeat scatter of that line (10-40 %, far
                above 1/SNR); plus a CR-model error sigma_model per line that
                is fully correlated between repeats - repeats do not average
                down model errors.
   The linear nuisances (s, R) are profiled out analytically and chi^2 is
   evaluated on a fine ln x - ln Ne grid -> posterior (log-uniform prior).  If
   chi^2_min/dof > 1 the posterior is widened by that factor (Birge ratio).
   Optional (Ne_penalty_above, off by default): a soft one-sided penalty on Ne
   above a density (Main.py: the critical density) is added to Delta chi^2/B^2,
   a prior that pulls a fit far above it back unless the lines insist.

The 5p -> 4s lines (415-470 nm) carry the most EEDF information (thresholds
~1.3 eV above the 4p).  The intensity calibration is checked within each range
by same-upper-level branching ratios (within ~0.2 in ln for the clean 5p and
4p9 pairs).  A free linear tilt tying the two ranges together was used until
2026-10-07; it only absorbed the 5p/4p mismatch of the NGFSRDW 5p cross sections
(SmallAnalysisScripts/AnalyticFivePFit.py) and is now disabled (no_tilt).

A Maxwellian EEDF fits these spectra only at Te ~0.6-1 eV (its tail above the
11.5 eV thresholds is far too full at 1 Torr); use the MultiBolt EEDFs.

Run from Spyder: edit CONFIG, press F5.  Results in Experimental_Data/Output/CRFit.
"""
import ast
import contextlib
import json
import io
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # pipeline/: its modules import each other by name
import Paths                                        # noqa: E402 - the folder of Main.py
import LineSelection as als              # noqa: E402

ROOT_DIR = Paths.ROOT                               # the folder of Main.py (pipeline/Paths.py)
SCRIPTS_DIR = os.path.join(ROOT_DIR, "Scripts")
# Biagi Ar, E/N = 2-50 Td, 12-decade EEDF span:
#   he.RunMultiBolt(<MultiBolt>/cross-sections/Biagi_Ar.txt, 'Ar_Biagi_lowEN_span12',
#                   [2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8.5, 10, 12, 15, 20, 30, 50], remap_span=12)
MULTIBOLT_RUN = os.path.join(ROOT_DIR, "InputData", "MultiBolt", "Ar_Biagi_lowEN_span12")

CONFIG = dict(
    # --- line selection (single-spectrum analysis + manual ratings) ----------
    selection_dir=als.CONFIG["selection_dir"],
    lines_file="lines_mix.csv",
    ar_ratings="Ar_manual_ratings.csv",
    status=("ok",),
    contamination=("clean", "clean?"),
    min_rating=4,                       # unrated lines pass on the automatic flags
    wl_ranges=[(400.0, 475.0), (690.0, 860.0)],     # 5p -> 4s and 4p -> 4s lines
    merge_nm=0.10,                      # model lines closer than this form one feature
    exclude_wl=(852.144,),              # air wavelengths [nm] to drop by hand; 852.14 (4p7-4s4)
                                        # sits at the edge of an echelle order (unreliable response)
    # --- measurement ---------------------------------------------------------
    sweeps=als.CONFIG["sweeps"],
    export_flags=als.CONFIG["export_flags"],    # spectra with other header flags are dropped
    window_nm=1.5,                      # catalogue lines within this of a fitted line are fitted too
    reuse_measurements=True,
    measure_dir=None,                   # folder of the measurement cache; None = outdir
    valid_status=("ok", "weak", "blend", "not_detected"),   # allowed component statuses in a feature
    max_rel_err=0.15,                   # feature dropped from a spectrum above this relative error
    # --- CR model ------------------------------------------------------------
    P_Torr=1.0, Tg=300.0, R=0.04,       # both sweeps are at 1 Torr
    trap_lines="all",                   # see SolveDirect in MainFileV2
    trap_ref_Ne=None,                   # m^-3: escape factors of the lines to excited levels frozen at
                                        # the absorber densities of the solution at this Ne (per EEDF
                                        # row), so trapping does not change with Ne; None = self-consistent
    N2_percent=0.0,                     # N2 admixture of the CR model [%]: quenching of the Ar(4s)
                                        # levels by N2 and dilution of the Ar ground state, as in
                                        # Scripts/MainFileWithNitrogen.py (0 = pure Ar).  With
                                        # admixture_gas: the % of that gas instead
    admixture_gas="N2",                 # the admixed gas (name as in the quenching tables): only its
                                        # quenching rates of the Ar levels change (admixture_quenching);
                                        # dilution and everything else are the same for any gas
    quench_4p=True,                     # with N2: also quench the 4p levels measured by Sadeghi 2001
                                        # (4p10, 4p6, 4p5, 4p3; he.ImportAr2pQuenchingData)
    N2_quench_scale=1.0,                # factor on every N2 quenching rate (test of the quenching; 0 = only
                                        # the dilution of the Ar ground state is left)
    escape_mode="table",                # escape factors: 'table' (Monte Carlo hemisphere, v2 table) or
                                        # 'walsh' (Holstein-Walsh cylinder for tau > 10), he.EscapeFactorInterpolator
    atom_transfer=True,                 # 2p <-> 2p and 2p -> 1s transfer by collisions with ground-state Ar
                                        # (Zhu & Pu 2010 rates, MainFileV2.AttachAtomTransfer)
    N2_dissociation=0.0,                # fraction of the N2 feed dissociated to N atoms: only the rest
                                        # quenches, and N2 -> 2N dilutes the Ar ground state further
                                        # (gas_fractions; no quenching of Ar levels by N atoms)
    eedf=MULTIBOLT_RUN,                 # 'maxwell', a MultiBolt export folder (E/N sweep), or a
                                        # BOLSIG+ library folder (he.RunBolsig / he.BuildBolsigLibrary;
                                        # E/N x Ne libraries must use the same Ne_grid)
    Te_grid=np.geomspace(0.5, 6.0, 32), # eV, eedf='maxwell' only
    EN_range=None,                      # (min, max) Td of the MultiBolt sweep to use; None = all
    min_eedf_Emax=15.0,                 # eV; drop EEDFs exported only below this (no 4p/5p excitation)
    min_bolsig_Emax=11.6,               # eV; BOLSIG+ E/N rows whose tail ends below the 4s threshold
                                        # (f < precision x f(0) there) give no excitation at all
    Ne_grid=np.geomspace(1e15, 3e19, 28),   # m^-3
    reuse_model=True,
    # --- fit -----------------------------------------------------------------
    intensity_units="energy",           # calibrated spectra: 'energy' (radiance, model x 1/lambda) or 'counts'
    response_file=None,                 # 2 columns: wavelength [nm], relative response (measured / true)
    response_deg=None,                  # response tilt: must stay None (calibration trusted) - the
                                        # fit raises for any other value (no_tilt)
    sigma_fit_floor=0.02,               # relative, added in quadrature to 1/SNR
    use_repeat_scatter=True,            # add the measured repeat-to-repeat scatter of each line
    sigma_model=0.20,                   # relative CR-model error per line
    birge=True,                         # widen the posterior by chi^2_min/dof when > 1
    Ne_penalty_above=None,              # m^-3: soft one-sided penalty on Ne above this density (Main.py: a multiple
                                        # of the critical density): (ln(Ne / Ne_penalty_above) / ln(Ne_penalty_width))^2
                                        # is added to Delta chi^2/B^2 above it, nothing below; None = off
    Ne_penalty_width=3.0,               # factor in Ne above Ne_penalty_above that costs Delta chi^2/B^2 = 1
    fine_n=(240, 220),                  # fine (EEDF axis, Ne) grid for the posterior
    # --- output --------------------------------------------------------------
    outdir=os.path.join(ROOT_DIR, "Experimental_Data", "Output", "CRFit"),
)
# Settings from the input file of Main.py (repo root), as JSON in an environment variable so that the worker
# processes building the CR tables get them too; empty outside Main.py
CONFIG.update(json.loads(os.environ.get("CR_CONFIG_OVERRIDES", "{}")))


# ----------------------------------------------------------------------------
# 0. CR model (functions of MainFileV2.py, loaded without running its sweep)
# ----------------------------------------------------------------------------
def _helpers():
    if Paths.PIPELINE_DIR not in sys.path:
        sys.path.insert(0, Paths.PIPELINE_DIR)
    import PhysicsHelpers as he
    return he


def load_cr_model(outdir, escape_mode="table"):
    he = _helpers()
    path = os.path.join(Paths.PIPELINE_DIR, "CRModelSolver.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    defs = ast.Module([n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))], [])
    cr = {"__file__": path, "OUTPUT_DIR": outdir, "plt": plt}
    exec(compile(defs, path, "exec"), cr)
    with contextlib.redirect_stdout(io.StringIO()):
        ModelData, RTM = he.GetData()
    interp = he.EscapeFactorInterpolator(RTM, mode=escape_mode)   # escape factors on (log10 tau, log10 a)
    return cr, he, ModelData, interp


def quench_4p_applied(cfg):
    """True when the CR model of cfg quenches the 4p levels by the admixture (stored with the CR table)."""
    return bool(cfg.get("quench_4p", True)) and cfg.get("N2_percent", 0.0) > 0


def gas_fractions(cfg):
    """Mole fractions (Ar, N2, N) of the N2 feed cfg['N2_percent'] with the fraction
    cfg['N2_dissociation'] of it dissociated, N2 -> 2 N at constant total pressure.  With another
    admixture_gas: (Ar, that gas, its atoms) - the dissociation assumes a diatomic gas X2 -> 2 X."""
    f, d = cfg.get("N2_percent", 0.0) / 100, cfg.get("N2_dissociation", 0.0)
    tot = 1 + f * d
    return (1 - f) / tot, f * (1 - d) / tot, 2 * f * d / tot


# ---- quenching by the admixed gas ---------------------------------------------------------------
METASTABLES, RESONANT = ("4s1", "4s3"), ("4s2", "4s4")
RESONANT_QUENCH_TOL = 0.05          # a resonant 4s level without quenching data is accepted (warning) when
                                    # its estimated quenching frequency is below this fraction of its trapped
                                    # radiative decay (escape factor x A); above it the CR model stops


def _quench_stop(msg):
    print(f"WARNING: {msg}", flush=True)
    raise ValueError(msg)


def admixture_quenching(cfg, he):
    """Quenching rate coefficients of the Ar levels by cfg['admixture_gas'] (default 'N2'), read from
        InputData/Ar_1s_quenching_data.csv        4s levels: kQ and three-body kQM, median of the
                                                   measurements (N2, H2, O2)    he.ImportArQuenchingData
        InputData/Ar_2p_quenching_Sadeghi2001.csv  4p10, 4p6, 4p5, 4p3: kQ (~20 gases), and for the gases
                                                   of its Table III a 1s5/1s3 average (Velazco 1978)
                                                   he.ImportAr2pQuenchingData
    The 4s file is used where it has the gas; otherwise the 1s5/1s3 average goes on the two metastables
    (4s1, 4s3) and the resonant 4s2, 4s4 have no rate (then check_resonant_quenching decides).
    -> ({CR label: {'kQ' [m^3/s], 'kQM' [m^6/s], 'source'}}, [resonant levels without a rate]).
    Stops (printed warning + ValueError) when the gas has no metastable rates, or no 4p rates while
    quench_4p is on."""
    gas = cfg.get("admixture_gas", "N2")
    try:
        q1 = he.ImportArQuenchingData(gas, verbose=False)
    except ValueError:                                # gas not in the 4s table
        q1 = {}
    try:
        q2 = he.ImportAr2pQuenchingData(gas, verbose=False)
    except ValueError:                                # gas not in the 4p table
        q2 = {}
    Q = {lbl: dict(kQ=q["kQ"], kQM=q["kQM"], source=f"Ar_1s_quenching_data.csv: {q['source_kQ']}")
         for lbl, q in q1.items() if q["kQ"] > 0}
    avg = q2.get("4s1;4s3")
    for lbl in METASTABLES:
        if lbl not in Q and avg:
            Q[lbl] = dict(kQ=avg["kQ"], kQM=0.0,
                          source=f"Ar_2p_quenching_Sadeghi2001.csv: 1s5/1s3 average, {avg['source']}")
    if miss := [lbl for lbl in METASTABLES if lbl not in Q]:
        _quench_stop(f"no quenching rate of the Ar metastables {', '.join(miss)} by admixture_gas = {gas!r} in "
                     f"InputData/Ar_1s_quenching_data.csv or Ar_2p_quenching_Sadeghi2001.csv: the CR model "
                     f"cannot run with {gas}.  Add the rates to the 4s table, or check the name (4s rates: N2, "
                     f"H2, O2; 1s5/1s3 averages: the Table III gases of the 4p table, e.g. CO2, NO, CF4)")
    if quench_4p_applied(cfg):
        p4 = {lbl: q for lbl, q in q2.items() if lbl.startswith("4p")}
        if not p4:
            _quench_stop(f"no 4p quenching rate by admixture_gas = {gas!r} in Ar_2p_quenching_Sadeghi2001.csv: "
                         f"add them, or switch the 4p quenching off ([model] quench_4p = false)")
        Q.update({lbl: dict(kQ=q["kQ"], kQM=0.0, source=f"Ar_2p_quenching_Sadeghi2001.csv: {q['source']}")
                  for lbl, q in p4.items()})
    return Q, [lbl for lbl in RESONANT if lbl not in Q]


def check_resonant_quenching(MD, cfg, he, Q, missing, n_gas, P_Ar, eta=None):
    """Does a missing quenching rate of a resonant 4s level matter?  A resonant level only holds population
    when its resonance line to the ground state is trapped: its radiative loss is eta A (eta = escape
    factor at the Ar ground density p_Ar, as the CR model computes it).  The missing quenching is estimated
    with the larger metastable rate of the same gas (the 4s rates of one gas agree within ~2-3x):
        ratio = kQ_metastable n_gas / (eta A)
    ratio < RESONANT_QUENCH_TOL (weak trapping: low pressure, or little admixture): printed warning, the
    level is left unquenched.  Otherwise the missing rate would change the model: warning + ValueError.
    eta(tau, a): escape factor (the CR model's); None = Holstein-Walsh, 1 for tau < 3.  -> {level: ratio}"""
    gas, out = cfg.get("admixture_gas", "N2"), {}
    if not missing or n_gas == 0:
        return out
    if eta is None:
        eta = lambda tau, a: 1.0 if tau < 3 else min(1.0, float(he.EscapeFactorWalsh(tau, a)))
    kq = cfg.get("N2_quench_scale", 1.0) * max(Q[lbl]["kQ"] for lbl in METASTABLES)
    for lbl in missing:
        rad = next(r for r in MD[lbl]["RadiativeDecay"] if r["direction"] == "loss" and r["partner"] == "ground")
        a, tau = he.FindTauInModel(MD[lbl], MD["ground"], rad, P_Ar, cfg["Tg"], cfg["R"])
        g = float(np.squeeze(eta(tau, a)))
        out[lbl] = ratio = kq * n_gas / (g * rad["coeff"])
        msg = (f"no quenching rate of the resonant level {lbl} by {gas} (p_Ar = {P_Ar:.3g} Torr, Tg = "
               f"{cfg['Tg']:g} K, {cfg.get('N2_percent', 0.0):g} % {gas}): trapped radiative loss eta A = "
               f"{g:.2e} x {rad['coeff']:.2e} = {g * rad['coeff']:.2e} s^-1, estimated quenching (metastable "
               f"rate) {kq * n_gas:.2e} s^-1 = {100 * ratio:.2g} % of it")
        if ratio >= RESONANT_QUENCH_TOL:
            _quench_stop(f"{msg} - above {RESONANT_QUENCH_TOL:.0%}: the line is too strongly trapped to leave the "
                         f"quenching out.  Add the rate of {lbl} to InputData/Ar_1s_quenching_data.csv")
        print(f"WARNING: {msg} - below {RESONANT_QUENCH_TOL:.0%}, {lbl} is left unquenched", flush=True)
    return out


_CHECKED = {}


def quenching_check(cfg):
    """admixture_quenching and check_resonant_quenching of the CR table cfg without building anything, so the
    warnings and stops come before the tables are built (and with --plan); once per condition.
    -> (quenching rates, {resonant level without a rate: estimated quenching / trapped radiative loss})."""
    x_ar, x, _ = gas_fractions(cfg)
    if x == 0:
        return {}, {}
    key = tuple(cfg.get(k) for k in ("P_Torr", "Tg", "R", "N2_percent", "N2_dissociation", "N2_quench_scale",
                                     "admixture_gas", "escape_mode")) + (quench_4p_applied(cfg),)
    if key not in _CHECKED:
        mode = cfg.get("escape_mode", "table")
        if ("model", mode) not in _CHECKED:
            _CHECKED[("model", mode)] = load_cr_model(cfg["outdir"], mode)
        cr, he, MD, interp = _CHECKED[("model", mode)]
        Q, missing = admixture_quenching(cfg, he)
        N = he.Torr2Volume(cfg["P_Torr"], cfg["Tg"])
        _CHECKED[key] = Q, check_resonant_quenching(MD, cfg, he, Q, missing, x * N, x_ar * cfg["P_Torr"],
                                                    lambda tau, a: cr["GetEta"](interp, tau, a))
    return _CHECKED[key]


def apply_nitrogen(MD, cfg, he, eta=None):
    """Admixture of cfg['N2_percent'] % cfg['admixture_gas'] (N2 by default; Scripts/MainFileWithNitrogen.py):
    sets the loss of the Ar(4s) levels - and with cfg['quench_4p'] of the 4p levels Sadeghi 2001 measured -
    by quenching on MD and returns the Ar partial pressure [Torr] to give CRModel (dilution of the Ar ground
    state). Metastable diffusion keeps the total pressure.
    Rate coefficients: admixture_quenching (InputData/Ar_1s_quenching_data.csv for 4s,
    InputData/Ar_2p_quenching_Sadeghi2001.csv for 4p; quenching to products outside the model, for N2
    mostly N2(C, B) + Ar ground); a resonant level without a rate: check_resonant_quenching (eta = escape
    factor function of the CR model).  With cfg['N2_dissociation'] only the undissociated gas quenches
    (gas_fractions); cfg['N2_quench_scale'] multiplies every quenching rate.
    Every quenched level, and a resonant level left without a rate, gets MD[level]['GasQuenching']: what was
    read from the tables (for debugging) - gas, kQ_m3_s, kQM_m6_s, source, n_gas_m3, scale, nu_s^-1."""
    x_ar, x, _ = gas_fractions(cfg)
    N = he.Torr2Volume(cfg["P_Torr"], cfg["Tg"])
    gas, scale = cfg.get("admixture_gas", "N2"), cfg.get("N2_quench_scale", 1.0)
    Q, missing = admixture_quenching(cfg, he) if x > 0 else ({}, [])
    ratios = check_resonant_quenching(MD, cfg, he, Q, missing, x * N, x_ar * cfg["P_Torr"], eta)
    for lbl, s in MD.items():
        q = Q.get(lbl)
        s["GasQuenching_s^-1"] = scale * (q["kQ"] * x * N + q["kQM"] * x * N * x_ar * N) if q else 0.0
        s.pop("GasQuenching", None)
        if q:
            s["GasQuenching"] = {"gas": gas, "kQ_m3_s": q["kQ"], "kQM_m6_s": q["kQM"], "source": q["source"],
                                 "n_gas_m3": x * N, "scale": scale, "nu_s^-1": s["GasQuenching_s^-1"]}
        elif lbl in ratios:
            s["GasQuenching"] = {"gas": gas, "kQ_m3_s": None, "kQM_m6_s": None, "n_gas_m3": x * N,
                                 "scale": scale, "nu_s^-1": 0.0,
                                 "source": f"no data: not quenched (estimated quenching / trapped radiative "
                                           f"loss = {100 * ratios[lbl]:.2g} % < {RESONANT_QUENCH_TOL:.0%})"}
    return x_ar * cfg["P_Torr"]


def cr_density_solver(cfg=CONFIG):
    """Function (eedf, Ne) -> {level: density [m^-3]} of the CR model with the
    settings of cfg, e.g. for the superelastic populations of he.BuildBolsigLibrary."""
    cr, he, MD, interp = load_cr_model(cfg["outdir"], cfg.get("escape_mode", "table"))
    MD = he.AddDiffusionLoss(MD, cfg["P_Torr"], cfg["Tg"], cfg["R"])
    P_Ar = apply_nitrogen(MD, cfg, he, eta=lambda tau, a: cr["GetEta"](interp, tau, a))

    def solve(eedf, Ne):
        with contextlib.redirect_stdout(io.StringIO()):
            D, _, _, _ = cr["CRModel"](MD, eedf, float(Ne), P_Ar, cfg["Tg"], cfg["R"],
                                       interp, trap_lines=cfg["trap_lines"],
                                       atom_transfer=cfg.get("atom_transfer", True))
        return {lbl: s["density_m^-3"] for lbl, s in D.items()}
    return solve


_BASE_MODEL = {}


def cr_state_solver(cfg=CONFIG):
    """Function (eedf, Ne) -> (level dict, solver info) of the CR model with the settings of cfg - set up as for
    the CR tables (diffusion, admixture quenching and dilution, trapping, atom transfer).  The level dict is the
    one CRModelSolver.CRModel fills: every level with density_m^-3, Loss_s^-1, Production_m^-3s^-1, rates, ...
    The model data are read once per escape mode and copied for each cfg."""
    import copy
    mode = cfg.get("escape_mode", "table")
    if mode not in _BASE_MODEL:
        _BASE_MODEL[mode] = load_cr_model(cfg["outdir"], mode)
    cr, he, MD, interp = _BASE_MODEL[mode]
    MD = he.AddDiffusionLoss(copy.deepcopy(MD), cfg["P_Torr"], cfg["Tg"], cfg["R"])
    with contextlib.redirect_stdout(io.StringIO()):
        P_Ar = apply_nitrogen(MD, cfg, he, eta=lambda tau, a: cr["GetEta"](interp, tau, a))

    def solve(eedf, Ne):
        with contextlib.redirect_stdout(io.StringIO()):
            D, _, _, info = cr["CRModel"](MD, eedf, float(Ne), P_Ar, cfg["Tg"], cfg["R"], interp,
                                          trap_lines=cfg["trap_lines"], atom_transfer=cfg.get("atom_transfer", True),
                                          compute_rates=True)
        return D, info
    return solve


def eedf_axis(cfg=CONFIG):
    """(EEDF inputs for CRModel, axis values x, axis name, Te_eff).

    One EEDF per x, or - for a BOLSIG+ library on an E/N x Ne grid (EEDF depends on
    Ne through e-e / superelastic collisions) - rows eedfs[i][j] on cfg["Ne_grid"],
    with Te_eff of shape (nx, nNe).  BOLSIG+ tails are computed down to the solver
    precision (he.BOLSIG_DEFAULTS); rows ending below min_bolsig_Emax (no excitation at all)
    are dropped, and min_eedf_Emax only applies to MultiBolt."""
    if cfg["eedf"] == "maxwell":
        Te = np.asarray(cfg["Te_grid"], float)
        return [float(t) for t in Te], Te, "Te [eV]", Te
    he = _helpers()
    if he.IsBolsigLibrary(cfg["eedf"]):
        lib = he.ImportBolsigLibrary(cfg["eedf"], verbose=False)
        x = lib["EN_Td"]
        lo, hi = cfg["EN_range"] or (-np.inf, np.inf)
        rows = [r if isinstance(r, list) else [r] for r in lib["EEDFs"]]
        E_max = np.array([min(e["E"][e["EEPF"] > 0].max() for e in r) for r in rows])
        keep = np.flatnonzero((x >= lo) & (x <= hi) & (E_max >= cfg["min_bolsig_Emax"]))
        E = [lib["EEDFs"][i] for i in keep]
        if lib["kind"] == "2D":
            if len(lib["Ne"]) != len(cfg["Ne_grid"]) or not np.allclose(lib["Ne"], cfg["Ne_grid"], rtol=1e-6):
                raise ValueError(f"{cfg['eedf']}: library Ne grid differs from cfg['Ne_grid'] - "
                                 "build the library on the same Ne grid")
            Te_eff = np.array([[e["Te_eff"] for e in row] for row in E])
        else:
            Te_eff = np.array([e["Te_eff"] for e in E])
        if np.any(np.diff(Te_eff, axis=0) <= 0):     # e.g. microwave EEDFs (Ramsauer minimum)
            print(f"note: Te_eff of {os.path.basename(str(cfg['eedf']))} is not monotonic in E/N; "
                  "Te is reported from the posterior")
        return E, x[keep], "E/N [Td]", Te_eff
    with contextlib.redirect_stdout(io.StringIO()):
        E = he.ImportMultiBoltEEDFs(cfg["eedf"], verbose=False)
    x = np.array([e["sweep"]["value"] for e in E], float)
    lo, hi = cfg["EN_range"] or (-np.inf, np.inf)
    E_max = np.array([e["E"][e["EEDF"] > 0].max() for e in E])
    keep = (x >= lo) & (x <= hi) & (E_max >= cfg["min_eedf_Emax"])
    E = [e for e, k in zip(E, keep) if k]
    Te_eff = np.array([e["Te_eff"] for e in E])
    if np.any(np.diff(Te_eff) <= 0):
        raise ValueError("Te_eff of the MultiBolt EEDFs is not increasing with E/N - restrict EN_range")
    return E, x[keep], f"E/N [{E[0]['sweep']['unit']}]", Te_eff


def te_label(tab):
    return (r"$T_e$ [eV] (Maxwellian)" if str(tab["eedf"]) == "maxwell"
            else r"$T_{e,\mathrm{eff}} = \frac{2}{3}\langle\varepsilon\rangle$ [eV] (MultiBolt, E/N sweep)")


def _model_lines(MD):
    """Every radiative line between model levels except those to the ground state."""
    return [(lbl, rad["partner"], rad["wavelength_nm"], rad["coeff"])
            for lbl, s in MD.items() for rad in s["RadiativeDecay"]
            if rad["direction"] == "loss" and rad["partner"] != "ground"]


def _table_path(cfg):
    tag = "maxwell" if cfg["eedf"] == "maxwell" else os.path.basename(os.path.normpath(cfg["eedf"]))
    if cfg.get("N2_percent", 0.0):
        tag += f"_{cfg.get('admixture_gas', 'N2')}_{cfg['N2_percent']:g}pct"
        if cfg.get("N2_dissociation", 0.0):
            tag += f"_diss{100 * cfg['N2_dissociation']:g}"
        if cfg.get("N2_quench_scale", 1.0) != 1.0:
            tag += f"_Q{cfg['N2_quench_scale']:g}"
    return os.path.join(cfg["outdir"], f"cr_model_table_{tag}.npz")


_PER_X = ("x_grid", "Te_eff", "I_obs", "I_thin", "dens", "fdirect", "converged")


def _xsec_set():
    """Name of the electron-impact cross-section file of the CR model (stored with each table)."""
    return _helpers().CROSS_SECTION_FILE


def _same_settings(tab, cfg):
    """A cached table was made with the settings of cfg (tables without an xsec entry predate the
    BSR set and were made with ArgonCrossSections.json)."""
    try:
        return (np.allclose(tab["Ne_grid"], cfg["Ne_grid"])
                and np.allclose([tab["P_Torr"], tab["Tg"], tab["R"]], [cfg["P_Torr"], cfg["Tg"], cfg["R"]])
                and str(tab["trap_lines"]) == cfg["trap_lines"] and str(tab["eedf"]) == str(cfg["eedf"])
                and float(tab["N2_percent"] if "N2_percent" in tab else 0.0) == float(cfg.get("N2_percent", 0.0))
                and str(tab["admixture_gas"] if "admixture_gas" in tab else "N2") == cfg.get("admixture_gas", "N2")
                and bool(tab["quench_4p"] if "quench_4p" in tab else False) == quench_4p_applied(cfg)
                and float(tab["N2_dissociation"] if "N2_dissociation" in tab else 0.0)
                == float(cfg.get("N2_dissociation", 0.0))
                and float(tab["N2_quench_scale"] if "N2_quench_scale" in tab else 1.0)
                == float(cfg.get("N2_quench_scale", 1.0))
                and str(tab["xsec"] if "xsec" in tab else "ArgonCrossSections.json") == _xsec_set()
                and bool(tab["atom_transfer"] if "atom_transfer" in tab else False)
                == bool(cfg.get("atom_transfer", True))
                and str(tab["trap_table"] if "trap_table" in tab else "RadiationTrappingLookupTable.json")
                == _helpers().TRAPPING_TABLE_FILE
                and str(tab["escape_mode"] if "escape_mode" in tab else "table") == cfg.get("escape_mode", "table")
                and float(tab["trap_ref_Ne"] if "trap_ref_Ne" in tab else 0.0) == float(cfg.get("trap_ref_Ne") or 0.0))
    except (KeyError, ValueError):
        return False


def table_is_current(cfg):
    """A cached CR table exists for cfg and was made with its settings and the current cross sections."""
    path = _table_path(cfg)
    if not os.path.exists(path):
        return False
    with np.load(path, allow_pickle=False) as t:
        return _same_settings(t, cfg)


def _cached_subset(tab, cfg, x):
    """The cached table restricted to the axis values x, or None if it does not hold them."""
    rows = [np.flatnonzero(np.isclose(tab["x_grid"], v, rtol=1e-9)) for v in x]
    if not _same_settings(tab, cfg) or any(len(r) != 1 for r in rows):
        return None
    rows = np.concatenate(rows)
    return {k: (v[rows] if k in _PER_X else v) for k, v in tab.items()}


def build_model_table(cfg=CONFIG):
    """CR model on the (EEDF axis, Ne) grid -> dict of arrays (cached as npz)."""
    path = _table_path(cfg)
    eedfs, x, x_name, Te_eff = eedf_axis(cfg)
    if cfg["reuse_model"] and os.path.exists(path):
        tab = _cached_subset(dict(np.load(path, allow_pickle=False)), cfg, x)
        if tab is not None:
            print(f"reusing {path}")
            return tab
    cr, he, MD, interp = load_cr_model(cfg["outdir"], cfg.get("escape_mode", "table"))
    P, Tg, R = cfg["P_Torr"], cfg["Tg"], cfg["R"]
    MD = he.AddDiffusionLoss(MD, P, Tg, R)
    P_Ar = apply_nitrogen(MD, cfg, he, eta=lambda tau, a: cr["GetEta"](interp, tau, a))
    levels = [l for l, s in MD.items() if s["kind"] != "ground"]
    lines = _model_lines(MD)
    Ne = cfg["Ne_grid"]
    shp = (len(x), len(Ne))
    I_obs = np.zeros(shp + (len(lines),))
    I_thin = np.zeros_like(I_obs)
    dens = np.zeros(shp + (len(levels),))
    fdir = np.zeros_like(dens)
    conv = np.zeros(shp, bool)

    def escape(up, lo, rad, trap=None):   # same escape factor as SolveDirect
        if cfg["trap_lines"] == "ground" and lo["kind"] != "ground":
            return 1.0
        n_lo = trap.get(lo["label"], lo["density_m^-3"]) if trap is not None and lo["kind"] != "ground" \
            else lo["density_m^-3"]
        if n_lo == 0:
            return 1.0
        a, tau = he.FindTauInModel(up, lo, rad, he.Volume2Torr(n_lo, Tg), Tg, R)
        return float(np.squeeze(cr["GetEta"](interp, tau, a)))

    def solve(eedf, ne, compute_rates, trap=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return cr["CRModel"](MD, eedf, float(ne), P_Ar, Tg, R, interp, trap_lines=cfg["trap_lines"],
                                 atom_transfer=cfg.get("atom_transfer", True), compute_rates=compute_rates,
                                 trap_density=trap)

    ref_Ne = cfg.get("trap_ref_Ne")
    rad_of = {(u, l): next(r for r in MD[u]["RadiativeDecay"] if r["direction"] == "loss" and r["partner"] == l)
              for u, l, _, _ in lines}
    for i, eedf_row in enumerate(eedfs):
        trap = None
        if ref_Ne:                    # absorber densities of the solution at ref_Ne, frozen for this row
            j_ref = int(np.argmin(np.abs(np.log(Ne) - np.log(ref_Ne))))
            e_ref = eedf_row[j_ref] if isinstance(eedf_row, list) else eedf_row
            D_ref = solve(e_ref, ref_Ne, True)[0]
            trap = {lbl: s["density_m^-3"] for lbl, s in D_ref.items() if s["kind"] != "ground"}
        for j, ne in enumerate(Ne):
            eedf = eedf_row[j] if isinstance(eedf_row, list) else eedf_row   # E/N x Ne library
            # rates depend on the EEDF only: once per row of a 1D library
            D, _, _, solver = solve(eedf, ne, j == 0 or isinstance(eedf_row, list), trap)
            conv[i, j] = solver["converged"]
            ng = D["ground"]["density_m^-3"]
            for k, lbl in enumerate(levels):
                s = D[lbl]
                dens[i, j, k] = s["density_m^-3"]
                direct = ne * ng * sum(r["Rate"] for r in s["Electron Impact CrossSections"]["Products"]
                                       if r["partner_label"] == "ground")
                prod = s.get("Production_m^-3s^-1", 0.0)
                fdir[i, j, k] = direct / prod if prod > 0 else np.nan
            for k, (u, l, wl, A) in enumerate(lines):
                I_thin[i, j, k] = D[u]["density_m^-3"] * A
                I_obs[i, j, k] = I_thin[i, j, k] * escape(D[u], D[l], rad_of[(u, l)], trap)
        te = (f"{Te_eff[i]:.2f}" if np.ndim(Te_eff) == 1
              else f"{np.min(Te_eff[i]):.2f}-{np.max(Te_eff[i]):.2f}")
        print(f"  CR grid: {x_name} = {x[i]:.3g} (Te_eff {te} eV) done ({i + 1}/{len(x)}), "
              f"{(~conv[i]).sum()} not converged")
    tab = dict(x_grid=x, x_name=x_name, Te_eff=Te_eff, eedf=str(cfg["eedf"]), Ne_grid=Ne,
               P_Torr=P, Tg=Tg, R=R, trap_lines=cfg["trap_lines"], N2_percent=float(cfg.get("N2_percent", 0.0)),
               admixture_gas=str(cfg.get("admixture_gas", "N2")),
               quench_4p=quench_4p_applied(cfg), N2_dissociation=float(cfg.get("N2_dissociation", 0.0)),
               N2_quench_scale=float(cfg.get("N2_quench_scale", 1.0)), xsec=_xsec_set(), atom_transfer=bool(cfg.get("atom_transfer", True)),
               trap_table=_helpers().TRAPPING_TABLE_FILE, escape_mode=cfg.get("escape_mode", "table"),
               trap_ref_Ne=float(ref_Ne or 0.0),
               levels=np.array(levels), line_upper=np.array([l[0] for l in lines]),
               line_lower=np.array([l[1] for l in lines]), line_wl=np.array([l[2] for l in lines], float),
               line_A=np.array([l[3] for l in lines], float),
               I_obs=I_obs, I_thin=I_thin, dens=dens, fdirect=fdir, converged=conv)
    os.makedirs(cfg["outdir"], exist_ok=True)
    np.savez(path, **tab)
    return tab


# ----------------------------------------------------------------------------
# 1. Line selection -> features
# ----------------------------------------------------------------------------
def select_features(tab, cfg=CONFIG):
    """Selected lines grouped into features.  Returns (features, component lines)."""
    res = pd.read_csv(os.path.join(cfg["selection_dir"], cfg["lines_file"]))
    model_pairs = set(zip(tab["line_upper"], tab["line_lower"]))
    ar = res[(res.species == "Ar I").to_numpy() & np.array([(u, l) in model_pairs for u, l in zip(res.upper, res.lower)])]
    ar = als._attach_ratings(ar, os.path.join(cfg["selection_dir"], cfg["ar_ratings"]))
    in_range = np.zeros(len(ar), bool)
    for lo, hi in cfg["wl_ranges"]:
        in_range |= ((ar.wl_air >= lo) & (ar.wl_air <= hi)).to_numpy()
    ok = (ar.status.isin(cfg["status"]) & ar.contamination.isin(cfg["contamination"])
          & (ar.rating.isna() | (ar.rating >= cfg["min_rating"])) & in_range
          & ~ar.wl_air.round(3).isin(np.round(cfg["exclude_wl"], 3)))
    sel = ar[ok].sort_values("wl_air")
    # cluster selected lines, then add every other model line within merge_nm
    feats, cur = [], []
    for r in sel.itertuples():
        if cur and r.wl_air - cur[-1].wl_air > cfg["merge_nm"]:
            feats.append(cur); cur = []
        cur.append(r)
    if cur:
        feats.append(cur)
    out, comps = [], []
    for f in feats:
        lo, hi = f[0].wl_air - cfg["merge_nm"], f[-1].wl_air + cfg["merge_nm"]
        c = ar[(ar.wl_air >= lo) & (ar.wl_air <= hi)].sort_values("wl_air")
        name = f"{c.wl_air.mean():.2f} " + "+".join(f"{u}-{l}" for u, l in zip(c.upper, c.lower))
        out.append(dict(feature=name, wl=float(c.wl_air.mean()), n_comp=len(c),
                        uppers="+".join(c.upper), comps=list(zip(c.wl_air, c.upper, c.lower))))
        comps.append(c.assign(feature=name))
    return pd.DataFrame(out), pd.concat(comps, ignore_index=True)


# ----------------------------------------------------------------------------
# 2. Measurement
# ----------------------------------------------------------------------------
def measure(comps, cfg=CONFIG):
    mcfg = dict(als.CONFIG, sweeps=cfg["sweeps"], window_nm=cfg["window_nm"],
                reuse_measurements=cfg["reuse_measurements"], outdir=cfg.get("measure_dir") or cfg["outdir"],
                measure_cache="ar_line_measurements.csv", export_flags=cfg["export_flags"])
    return als.filter_export_flags(als.measure_sweeps(comps[["species", "wl_air"]].drop_duplicates(), mcfg), mcfg)


def _response(cfg):
    if not cfg["response_file"]:
        return None
    d = np.loadtxt(cfg["response_file"])
    return lambda lam: np.interp(lam, d[:, 0], d[:, 1])


def feature_table(meas, feats, cfg=CONFIG):
    """Per spectrum and feature: summed area and its relative error."""
    m = meas[meas.species == "Ar I"].assign(wl=lambda d: d.wl_air.round(4))
    resp = _response(cfg)
    rows = []
    for (sweep, f), g in m.groupby(["sweep", "file"]):
        gi = g.drop_duplicates("wl").set_index("wl")
        for ft in feats.itertuples():
            c = gi.reindex([round(w, 4) for w, _, _ in ft.comps])
            if c.status.isna().any() or not c.status.isin(cfg["valid_status"]).all():
                continue
            area = c.area.sum()
            if not area > 0:
                continue
            err = np.sqrt(np.nansum((c.area / c.snr.where(c.snr > 0)) ** 2)) / area
            if not err <= cfg["max_rel_err"]:
                continue
            if resp is not None:
                area /= resp(ft.wl)
            rows.append(dict(sweep=sweep, file=f, x=g.x.iloc[0], rep=g.rep.iloc[0], feature=ft.feature,
                             wl=ft.wl, area=area, rel_err=err))
    return pd.DataFrame(rows)


def repeat_scatter(ft):
    """Per feature: pooled scatter of ln(area) between repeats of the same condition,
    after removing each spectrum's overall scale (plasma / exposure fluctuations
    that change the line ratios, invisible in the 1/SNR of a single spectrum)."""
    acc = {}
    for _, g in ft.groupby(["sweep", "x"]):
        p = g.pivot_table(index="file", columns="feature", values="area").dropna(axis=1)
        if len(p) < 2 or p.shape[1] < 2:
            continue
        p = np.log(p)
        p = p.sub(p.mean(axis=1), axis=0)
        for f in p:
            acc.setdefault(f, []).append((p[f].var(ddof=1) * (len(p) - 1), len(p) - 1))
    s = pd.Series({f: np.sqrt(sum(v for v, _ in l) / sum(d for _, d in l)) for f, l in acc.items()})
    return s.reindex(ft.feature.unique()).fillna(s.median() if len(s) else 0.0)


# ----------------------------------------------------------------------------
# 3. Fit
# ----------------------------------------------------------------------------
def fine_grid(tab, cfg=CONFIG):
    """Fine ln x and ln Ne axes of the posterior."""
    nX, nN = cfg["fine_n"]
    lx, ln = np.log(tab["x_grid"]), np.log(tab["Ne_grid"])
    return np.linspace(lx[0], lx[-1], nX), np.linspace(ln[0], ln[-1], nN)


def te_eff_fine(tab, grid):
    """Te_eff [eV] on the fine EEDF axis (Te itself for a Maxwellian); on the fine
    (x, Ne) grid, shape (nX, nNe), when the EEDF depends on Ne (BOLSIG+ E/N x Ne library)."""
    lx = np.log(tab["x_grid"])
    if np.ndim(tab["Te_eff"]) == 2:
        return RectBivariateSpline(lx, np.log(tab["Ne_grid"]), tab["Te_eff"], kx=1, ky=1)(*grid)
    return np.interp(grid[0], lx, tab["Te_eff"])


def on_fine_grid(tab, values, cfg=CONFIG, k=3):
    """Spline a (x, Ne) grid quantity onto the fine ln x - ln Ne grid."""
    lx, ln = np.log(tab["x_grid"]), np.log(tab["Ne_grid"])
    fX, fN = fine_grid(tab, cfg)
    k_x = min(k, len(lx) - 1)
    return RectBivariateSpline(lx, ln, values, kx=k_x, ky=k)(fX, fN)


def feature_model(tab, feats, cfg=CONFIG):
    """ln(modelled feature intensity) on the fine grid, shape (nX, nNe, nFeat)."""
    idx = {(u, l): k for k, (u, l) in enumerate(zip(tab["line_upper"], tab["line_lower"]))}
    photon_energy = (lambda k: 1.0 / tab["line_wl"][k]) if cfg["intensity_units"] == "energy" else (lambda k: 1.0)
    out = []
    for ft in feats.itertuples():
        I = sum(tab["I_obs"][:, :, idx[(u, l)]] * photon_energy(idx[(u, l)]) for _, u, l in ft.comps)
        out.append(on_fine_grid(tab, np.log(I), cfg))
    return np.stack(out, axis=-1)


def _wquantile(x, p, q):
    c = np.cumsum(p)
    return np.interp(q, c / c[-1], x)


def hpd_level(post, mass=0.68):
    p = np.sort(post.ravel())[::-1]
    c = np.cumsum(p) / p.sum()
    return p[np.searchsorted(c, mass)]


def no_tilt(cfg):
    """The response tilt is disabled: raise if cfg asks for one."""
    if cfg.get("response_deg"):
        raise ValueError(f"response_deg = {cfg['response_deg']!r}: the response tilt is disabled - "
                         "set response_deg=None (intensity calibration trusted)")


def ne_penalty(lnNe, cfg):
    """Soft one-sided penalty on Ne in units of Delta chi^2/B^2: (ln(Ne / Ne_0) / ln(width))^2 above
    Ne_0 = cfg['Ne_penalty_above'], 0 below it and everywhere when Ne_0 is None (off)."""
    lnNe = np.asarray(lnNe, float)
    ne0 = cfg.get("Ne_penalty_above")
    if not ne0:
        return np.zeros_like(lnNe)
    return (np.maximum(lnNe - np.log(ne0), 0.0) / np.log(cfg.get("Ne_penalty_width", 3.0))) ** 2


def fit_block(d, M, feats, grid, Te_f, cfg=CONFIG):
    """Joint fit of the rows in d (one or more spectra).  Returns summary dict,
    posterior (nX, nNe) and per-row best-fit residuals."""
    no_tilt(cfg)
    fX, fN = grid
    fidx = {f: k for k, f in enumerate(feats.feature)}
    li = d.feature.map(fidx).to_numpy()
    si = pd.factorize(d.file)[0]
    y = np.log(d.area.to_numpy())
    lam = d.wl.to_numpy()
    n, n_spec = len(d), si.max() + 1
    X = [np.eye(n_spec)[si]]
    if cfg["response_deg"]:
        X += [((lam - 775.0) / 100.0)[:, None] ** k for k in range(1, cfg["response_deg"] + 1)]
    X = np.hstack(X)
    C = np.diag(d.rel_err.to_numpy() ** 2 + cfg["sigma_fit_floor"] ** 2)
    C += cfg["sigma_model"] ** 2 * (li[:, None] == li[None, :])
    W = np.linalg.inv(C)
    G = np.linalg.pinv(X.T @ W @ X) @ X.T @ W          # beta = G @ resid
    Pm = W - W @ X @ G
    D = y[None, None, :] - M[:, :, li]
    chi2 = np.einsum("abi,ij,abj->ab", D, Pm, D)       # the lines alone
    dof = n - np.linalg.matrix_rank(X) - 2
    i0 = np.unravel_index(np.argmin(chi2), chi2.shape)
    s2 = max(1.0, float(chi2[i0]) / dof) if (cfg["birge"] and dof > 0) else 1.0
    pen = ne_penalty(fN, cfg)                           # optional prior on Ne, not widened by the Birge factor
    chi2p = chi2 + s2 * pen[None, :]                    # = chi2 with the penalty off
    ib = np.unravel_index(np.argmin(chi2p), chi2p.shape)
    chi2_min = float(chi2p[ib])                         # chi2_grid of the saved posterior gives chi2p back
    post = np.exp(-0.5 * (chi2p - chi2_min) / s2)
    post /= post.sum()
    pX, pN = post.sum(1), post.sum(0)
    qX = _wquantile(fX, pX, [0.16, 0.5, 0.84])
    qN = np.exp(_wquantile(fN, pN, [0.16, 0.5, 0.84]))
    if np.ndim(Te_f) == 1 and np.all(np.diff(Te_f) > 0):
        qT = np.interp(qX, fX, Te_f)                   # Te_eff is monotonic in x
        Te_best = float(Te_f[ib[0]])
    else:                                              # Te_eff(x, Ne) or not monotonic in x:
        T2 = np.broadcast_to(Te_f[:, None], post.shape) if np.ndim(Te_f) == 1 else Te_f
        o = np.argsort(T2, axis=None)                  # quantiles over the posterior
        qT = _wquantile(T2.ravel()[o], post.ravel()[o], [0.16, 0.5, 0.84])
        Te_best = float(T2[ib])
    mX, mN = (pX * fX).sum(), (pN * fN).sum()
    cov = (post * (fX[:, None] - mX) * (fN[None, :] - mN)).sum()
    corr = cov / np.sqrt((pX * (fX - mX) ** 2).sum() * (pN * (fN - mN) ** 2).sum())
    beta = G @ D[ib]
    resid = D[ib] - X @ beta
    edge = lambda p: p[:3].sum() + p[-3:].sum()
    out = dict(n_spec=n_spec, n_points=n, n_feat=len(np.unique(li)),
               Te_best=Te_best, Te_lo=qT[0], Te_med=qT[1], Te_hi=qT[2],
               x_best=float(np.exp(fX[ib[0]])), x_lo=np.exp(qX[0]), x_med=np.exp(qX[1]), x_hi=np.exp(qX[2]),
               Ne_best=float(np.exp(fN[ib[1]])), Ne_lo=qN[0], Ne_med=qN[1], Ne_hi=qN[2],
               corr_lnx_lnNe=corr, chi2_min=chi2_min, dof=dof,
               chi2_red=float(chi2[ib]) / dof if dof > 0 else np.nan, birge=np.sqrt(s2),
               edge_x=edge(pX) > 0.05, edge_Ne=edge(pN) > 0.05)
    if cfg.get("Ne_penalty_above"):                     # where the lines alone put the fit, and what the pull cost
        out.update(x_lines_only=float(np.exp(fX[i0[0]])), Ne_lines_only=float(np.exp(fN[i0[1]])),
                   penalty_at_fit=float(pen[ib[1]]), dchi2_lines_at_fit=float(chi2[ib] - chi2[i0]) / s2)
    if cfg["response_deg"]:
        out["response_slope_per_100nm"] = float(beta[n_spec])
    return out, post, resid


def fit_all(ft, feats, M, grid, Te_f, cfg=CONFIG, verbose=True):
    cond_rows, spec_rows, res_rows, posts = [], [], [], {}
    for (sweep, x), d in ft.groupby(["sweep", "x"]):
        s, post, r = fit_block(d, M, feats, grid, Te_f, cfg)
        cond_rows.append(dict(sweep=sweep, x=x, **s))
        posts[f"{sweep}|{x:g}"] = post
        for f, g in d.assign(r=r).groupby("feature"):
            res_rows.append(dict(sweep=sweep, x=x, feature=f, resid=g.r.mean()))
        if verbose:
            print(f"  {sweep:<12s} x={x:<6g} Te = {s['Te_med']:.2f} (+{s['Te_hi'] - s['Te_med']:.2f} "
                  f"-{s['Te_med'] - s['Te_lo']:.2f}) eV   Ne = {s['Ne_med']:.2e} m^-3   "
                  f"chi2/dof = {s['chi2_red']:.2f}  ({s['n_spec']} spectra, {s['n_feat']} lines)")
    for (sweep, f), d in ft.groupby(["sweep", "file"]):
        s, _, _ = fit_block(d, M, feats, grid, Te_f, cfg)
        spec_rows.append(dict(sweep=sweep, file=f, x=d.x.iloc[0], rep=d.rep.iloc[0], **s))
    return pd.DataFrame(cond_rows), pd.DataFrame(spec_rows), pd.DataFrame(res_rows), posts


def chi2_grid(post, chi2_min, birge):
    """chi^2 on the fine (E/N, Ne) grid from a saved posterior (its minimum chi2_min, Birge ratio birge)."""
    with np.errstate(divide="ignore"):
        return chi2_min + 2 * birge ** 2 * (np.log(post.max()) - np.log(post))


def pinned_at(FIT, c, ne):
    """The condition's fit with Ne pinned at ne [m^-3] and E/N refitted (chi^2 from the saved posterior): a copy of
    the cond row c with x/Te/Ne/chi2 replaced; band Delta chi^2 = 1 s^2 along E/N.  FIT: dict with grid, Te_f, posts.
    (Was CRFitSpectrumOverlay.pinned_at_nc, which read the pin from MicrowaveLowENFit.N_C.)"""
    (fX, fN), Te_f = FIT["grid"], FIT["Te_f"]
    post = FIT["posts"][f"{c.sweep}|{c.x:g}"]
    j = int(np.argmin(np.abs(fN - np.log(ne))))
    row = chi2_grid(post, c.chi2_min, c.birge)[:, j]
    i = int(np.argmin(row))
    ok = row <= row[i] + c.birge ** 2
    Te = Te_f if np.ndim(Te_f) == 1 else Te_f[:, j]
    c = c.copy()
    c["x_best"], c["Te_best"], c["Te_lo"], c["Te_hi"] = np.exp(fX[i]), Te[i], Te[ok].min(), Te[ok].max()
    c["Ne_best"] = c["Ne_lo"] = c["Ne_hi"] = ne
    c["chi2_red"] = row[i] / (c.dof + 1)
    return c


def posterior_mean(tab, post, values, cfg=CONFIG):
    """<values> over the posterior; values is a (x, Ne) grid quantity of the model table."""
    return float((post * on_fine_grid(tab, values, cfg, k=1)).sum())


# ----------------------------------------------------------------------------
# 4. Plots
# ----------------------------------------------------------------------------
def plot_sweeps(cond, spec, tab, cfg, path):
    sweeps = cfg["sweeps"]
    fig, axs = plt.subplots(2, len(sweeps), figsize=(6.5 * len(sweeps), 8), squeeze=False)
    for j, sw in enumerate(sweeps):
        c = cond[cond.sweep == sw["name"]].sort_values("x")
        s = spec[spec.sweep == sw["name"]]
        for i, (q, lab) in enumerate((("Te", te_label(tab)), ("Ne", "$N_e$ [m$^{-3}$]"))):
            ax = axs[i, j]
            ax.plot(s.x, s[f"{q}_med"], "o", color="0.7", ms=4, label="single spectrum")
            ax.errorbar(c.x, c[f"{q}_med"], yerr=[c[f"{q}_med"] - c[f"{q}_lo"], c[f"{q}_hi"] - c[f"{q}_med"]],
                        fmt="s-", color="C0" if q == "Te" else "C3", ms=6, capsize=3,
                        label="condition (all repeats), 16-84 %")
            if q == "Ne":
                ax.set_yscale("log")
            ax.set_xlabel(sw["xlabel"])
            ax.set_ylabel(lab, fontsize=9)
            ax.grid(alpha=0.3, which="both")
            ax.set_title(f"{sw['name']} sweep ({sw['note']})")
        axs[0, j].legend(fontsize=8)
    resp = "calibrated response" if not cfg["response_deg"] else f"ln R poly deg {cfg['response_deg']}"
    eedf = "Maxwellian" if cfg["eedf"] == "maxwell" else f"MultiBolt {os.path.basename(os.path.normpath(cfg['eedf']))}"
    fig.suptitle(f"CR-model fit of Ar I lines ({eedf}, {resp}, "
                 rf"$\sigma_{{model}}$ = {100 * cfg['sigma_model']:.0f} %)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_posteriors(cond, posts, tab, grid, cfg, path):
    Te_f, Ne_f = te_eff_fine(tab, grid), np.exp(grid[1])
    if np.ndim(Te_f) == 2:                     # Te_eff(x, Ne): contour on 2D coordinates
        X, Y, T = Te_f, np.broadcast_to(Ne_f, Te_f.shape), lambda p: p
    else:
        X, Y, T = Te_f, Ne_f, lambda p: p.T
    sweeps = cfg["sweeps"]
    fig, axs = plt.subplots(1, len(sweeps), figsize=(6.5 * len(sweeps), 5.5), squeeze=False)
    for ax, sw in zip(axs[0], sweeps):
        c = cond[cond.sweep == sw["name"]].sort_values("x")
        cols = plt.cm.viridis(np.linspace(0, 0.9, max(len(c), 1)))
        for col, r in zip(cols, c.itertuples()):
            post = posts[f"{sw['name']}|{r.x:g}"]
            ax.contour(X, Y, T(post), levels=[hpd_level(post)], colors=[col])
            ax.plot(r.Te_best, r.Ne_best, "o", color=col, ms=5, label=f"{r.x:g}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(te_label(tab), fontsize=9); ax.set_ylabel("$N_e$ [m$^{-3}$]")
        ax.set_title(f"{sw['name']}: 68 % posterior regions")
        ax.legend(title=sw["xlabel"], fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(res, feats, path):
    if res.empty:
        return
    res = res.assign(cond=res.sweep + " " + res.x.map("{:g}".format))
    M = res.pivot(index="feature", columns="cond", values="resid").reindex(feats.feature)
    fig, ax = plt.subplots(figsize=(0.45 * M.shape[1] + 4, 0.4 * M.shape[0] + 2.5))
    v = np.nanmax(np.abs(M.values))
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M.values[i, j]):
                ax.text(j, i, f"{M.values[i, j]:+.2f}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(M.shape[0]), M.index, fontsize=8)
    fig.colorbar(im, ax=ax, label="ln(measured / model) at best fit")
    ax.set_title("Line residuals (after the per-spectrum scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. Driver
# ----------------------------------------------------------------------------
def run(cfg=CONFIG):
    os.makedirs(cfg["outdir"], exist_ok=True)
    out = lambda f: os.path.join(cfg["outdir"], f)
    tab = build_model_table(cfg)
    if not tab["converged"].all():
        print(f"WARNING: {(~tab['converged']).sum()} CR grid points did not converge")
    feats, comps = select_features(tab, cfg)
    print(f"\n{len(feats)} features from {len(comps)} Ar I lines:")
    print(feats[["feature", "wl", "n_comp"]].to_string(index=False))
    meas = measure(comps, cfg)
    ft = feature_table(meas, feats, cfg)
    rs = repeat_scatter(ft)
    feats["repeat_scatter"] = feats.feature.map(rs)
    ft["rel_err_fit"] = ft.rel_err
    if cfg["use_repeat_scatter"]:
        ft["rel_err"] = np.sqrt(ft.rel_err_fit ** 2 + ft.feature.map(rs) ** 2)
    print("\nrepeat scatter per line (ln):")
    print(rs.round(3).to_string())
    ft.to_csv(out("feature_intensities.csv"), index=False)
    feats.drop(columns="comps").to_csv(out("features.csv"), index=False)
    grid = fine_grid(tab, cfg)
    Te_f = te_eff_fine(tab, grid)
    M = feature_model(tab, feats, cfg)
    print(f"\nfitting (EEDF axis {tab['x_name']}) ...")
    cond, spec, res, posts = fit_all(ft, feats, M, grid, Te_f, cfg)
    # CR-model context at the fit: metastable densities (1s5 = 4s1, 1s3 = 4s3)
    lev = list(tab["levels"])
    for lbl in ("4s1", "4s3"):
        cond[f"n_{lbl}"] = [np.exp(posterior_mean(tab, posts[f"{s}|{x:g}"], np.log(tab["dens"][:, :, lev.index(lbl)]), cfg))
                            for s, x in zip(cond.sweep, cond.x)]
    cond.to_csv(out("fit_conditions.csv"), index=False)
    spec.to_csv(out("fit_spectra.csv"), index=False)
    res.to_csv(out("fit_residuals.csv"), index=False)
    np.savez(out("posteriors.npz"), lnx=grid[0], lnNe=grid[1], Te_eff=Te_f, **posts)
    plot_sweeps(cond, spec, tab, cfg, out("Te_Ne_vs_sweep.png"))
    plot_posteriors(cond, posts, tab, grid, cfg, out("posteriors.png"))
    plot_residuals(res, feats, out("residuals.png"))
    for c in cond.itertuples():
        if c.edge_x or c.edge_Ne:
            print(f"WARNING: {c.sweep} x={c.x:g}: posterior reaches the grid edge "
                  f"({'EEDF axis' if c.edge_x else ''}{' Ne' if c.edge_Ne else ''})")
    print("\nwritten to", cfg["outdir"])
    eedfs = eedf_axis(cfg)[0]
    return dict(tab=tab, feats=feats, comps=comps, meas=meas, ft=ft, grid=grid, Te_f=Te_f, M=M,
                cond=cond, spec=spec, res=res, posts=posts, eedfs=eedfs, cfg=cfg)


if __name__ == "__main__":
    FIT = run(CONFIG)
