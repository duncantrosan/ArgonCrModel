# -*- coding: utf-8 -*-
"""
Main.py

Runs the echelle analysis from one input file (TOML): CR fit of every spectrum set (pure-Ar microwave BOLSIG+ EEDF,
Ne pinned at the critical density), Ar metastables against LAS and N I / Ar I actinometry, optionally followed by
the LaTeX report.  The work is done by SmallAnalysisScripts/N2SweepPureArEEDF.py (and MakeFitReport.py); this file
reads the input file, checks it, hands the settings to those scripts and starts them.

    python Main.py input_example.toml            full run
    python Main.py input_example.toml --plan     conditions, Tg and the libraries / CR tables still to build only
    python Main.py                               same as python Main.py input_example.toml

From Spyder: Run > Configuration per file > Command line options: input_example.toml (or input_example.toml --plan).

Every setting is listed, with its default and what it does, in input_example.toml; a setting left out of an input
file keeps its default (the defaults reproduce N2SweepPureArEEDF.py).  The [[dataset]] blocks are read by
N2SweepPureArEEDF.load_input.  The settings of each run are written to <output_folder>/settings_used.json.

How the settings reach the code: the CR-model and fit settings go into CRFitNeTe.CONFIG through the environment
variable CR_CONFIG_OVERRIDES (read when CRFitNeTe is imported, so the worker processes that build the CR tables get
them too), the cross sections through CR_XSEC_FILE, and CR tables made with non-default model settings are kept in
their own folder (CR_TABLE_TAG, AnalyticFivePFit.TABLE_DIR).  Everything else is set on the modules directly.
"""
import hashlib
import json
import os
import sys
import tomllib

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(ROOT, "input_example.toml")

DEFAULTS = {   # section -> setting -> default
    "run": dict(output_folder="Experimental_Data/Output/N2SweepPureArEEDF", echelle_folder="Experimental_Data/EchelleData",
                plan_only=False, show_plots=False, overwrite=True),
    "model": dict(cross_sections="ArgonCrossSections_BSR_analytic5p.json", plasma_radius_m=0.04, trap_lines="all",
                  escape_mode="table", atom_transfer=True, quench_4p=True, N2_quench_scale=1.0, N2_dissociation=0.0),
    "eedf": dict(model="pure_Ar", Tv_K=8000.0, Tg_step_K=50.0),
    "fit": dict(Ne_pin="free", sigma_model=0.20, sigma_fit_floor=0.02, use_repeat_scatter=True, birge=True,
                max_rel_err=0.15, merge_nm=0.10, wavelength_ranges_nm=[[400.0, 475.0], [690.0, 860.0]],
                exclude_lines_nm=[852.144], min_rating=4),
    "spectra": dict(export_flags="?-?-1-1", min_repeats=2, line_status=["ok", "weak"]),
    "actinometry": dict(extra_N_lines_nm=[746.831], exclude_Ar_references_nm=[706.7218, 415.859, 794.8176],
                        tukey_fence=1.5, tukey_min_pairs=5, min_valid_mass=0.9, xsec_error_N=0.20, xsec_error_Ar=0.15,
                        k_margin_eV=1.0),
    "plots": dict(metastables=True, fit_params=True, eedf=True, residuals=True, fit_overlay=False, chi2_maps=False,
                  actinometry=True, actinometry_per_line=True, line_ratios=True, same_upper_ratios=True,
                  raw_lines=True, overviews=True, dpi=200, zoom_Ar_line_nm=750.3869, zoom_half_width_nm=0.4),
    "las": dict(file="Experimental_Data/LAS_Data/Master_AllMeasurements.csv", frequency_MHz=[2400.0, 2450.0],
                match_power_W=5.0, write_summary=True, summary_base="", summary_file="", summary_in_place=False),
    "report": dict(make=False, folder="LatexReport", example_set="80W_1Torr", example_x=[0.33, 2.0], s_example_x=2.0),
}
NE_MODES = ("free", "free_else_nc", "nc")
CHOICES = {("eedf", "model"): ("pure_Ar", "ArN2_vib_saturated", "ArN2_vib_Tv"),
           ("model", "trap_lines"): ("all", "ground"), ("model", "escape_mode"): ("table", "walsh"),
           ("spectra", "line_status"): ("ok", "weak", "blend", "not_detected")}
MODEL_KEYS = dict(plasma_radius_m="R", trap_lines="trap_lines", escape_mode="escape_mode",   # -> CRFitNeTe.CONFIG;
                  atom_transfer="atom_transfer", quench_4p="quench_4p",                      # these define the CR tables
                  N2_quench_scale="N2_quench_scale", N2_dissociation="N2_dissociation")
FIT_KEYS = dict(sigma_model="sigma_model", sigma_fit_floor="sigma_fit_floor", use_repeat_scatter="use_repeat_scatter",
                birge="birge", max_rel_err="max_rel_err", merge_nm="merge_nm", wavelength_ranges_nm="wl_ranges",
                exclude_lines_nm="exclude_wl", min_rating="min_rating")


def _checked(where, value, default, choices=None):
    """value converted to the type of default (bool, int, float, str, list), or ValueError."""
    if isinstance(default, bool):
        ok = isinstance(value, bool)
    elif isinstance(default, (int, float)):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        value = type(default)(value) if ok and (isinstance(default, float) or float(value).is_integer()) else value
        ok = ok and isinstance(value, type(default))
    else:
        ok = isinstance(value, type(default))
    if not ok:
        raise ValueError(f"{where} = {value!r}: expected {type(default).__name__} like {default!r}")
    for v in (value if isinstance(value, list) else [value]):
        if choices and v not in choices:
            raise ValueError(f"{where} = {value!r}: use {' or '.join(map(repr, choices))}")
    return value


def read_input(path):
    """Settings of the input file on top of DEFAULTS; unknown sections or settings and wrong types are errors."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    raw.pop("dataset", None)
    out = {sec: dict(d) for sec, d in DEFAULTS.items()}
    name = os.path.basename(path)
    for sec, d in raw.items():
        if sec not in DEFAULTS or not isinstance(d, dict):
            raise ValueError(f"{name}: unknown section [{sec}] - sections are {', '.join(DEFAULTS)} and [[dataset]]")
        for k, v in d.items():
            where = f"{name}: [{sec}] {k}"
            if k not in DEFAULTS[sec]:
                raise ValueError(f"{name}: [{sec}] has no setting {k!r} - settings are {', '.join(DEFAULTS[sec])}")
            if (sec, k) == ("fit", "Ne_pin"):
                if v not in NE_MODES and not (isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0):
                    raise ValueError(f"{where} = {v!r}: {', '.join(map(repr, NE_MODES))} or a density in m^-3, e.g. 1.5e17")
                out[sec][k] = v if v in NE_MODES else float(v)
            else:
                out[sec][k] = _checked(where, v, DEFAULTS[sec][k], CHOICES.get((sec, k)))
    xsec = os.path.join(ROOT, "InputData", out["model"]["cross_sections"])
    if not os.path.isfile(xsec):
        raise FileNotFoundError(f"{name}: [model] cross_sections - {xsec} not found")
    return out


def in_repo(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def table_tag(s):
    """'' for the default CR-model settings, else '_<hash>': the folder suffix of their CR tables."""
    model = s["model"]
    if model == DEFAULTS["model"]:
        return ""
    return "_" + hashlib.sha1(json.dumps(model, sort_keys=True).encode()).hexdigest()[:8]


def changed(s):
    return [f"[{sec}] {k} = {v!r}" for sec, d in s.items() for k, v in d.items() if v != DEFAULTS[sec][k]]


def configure(S, s):
    """Settings of the main process on the pipeline's modules (the CR-table workers read theirs from the
    environment, set in main before the import)."""
    L, als = S.L, S.als
    S.OUTDIR = in_repo(s["run"]["output_folder"])
    S.ECHELLE_DIR = in_repo(s["run"]["echelle_folder"])
    L.TG_STEP = s["eedf"]["Tg_step_K"]
    e = s["eedf"]
    S.EEDF = {"pure_Ar": "ar", "ArN2_vib_saturated": "vibsat", "ArN2_vib_Tv": f"tv{e['Tv_K']:g}"}[e["model"]]
    pin = s["fit"]["Ne_pin"]
    S.NE_MODE = pin if pin in NE_MODES else "value"
    ne = pin if S.NE_MODE == "value" else S.N_C       # free: N_C stays n_c (cost of the pin, chi^2 maps)
    S.N_C = L.N_C = L.mlf.N_C = ne                    # every module that pins Ne reads one of these
    lc = s["las"]
    S.LAS_FILE = in_repo(lc["file"])
    if not os.path.isfile(S.LAS_FILE):
        raise FileNotFoundError(f"[las] file: {S.LAS_FILE} not found")
    L.LAS_FREQ_MHZ, L.DP_W = tuple(lc["frequency_MHz"]), lc["match_power_W"]
    S.LAS_SUMMARY = dict(write=lc["write_summary"], path=in_repo(lc["summary_file"]) if lc["summary_file"] else None,
                         base=in_repo(lc["summary_base"]) if lc["summary_base"] else None,
                         in_place=lc["summary_in_place"])
    if S.LAS_SUMMARY["base"] and not os.path.isfile(S.LAS_SUMMARY["base"]):
        raise FileNotFoundError(f"[las] summary_base: {S.LAS_SUMMARY['base']} not found")
    S.OVERWRITE = s["run"]["overwrite"]
    sp, a, pl = s["spectra"], s["actinometry"], s["plots"]
    als.CONFIG.update(export_flags=sp["export_flags"], min_reps=sp["min_repeats"], sweep_status=tuple(sp["line_status"]))
    S.EXTRA_N_WL = tuple(a["extra_N_lines_nm"])
    S.EXCLUDE_AR_WL = tuple(a["exclude_Ar_references_nm"])
    S.ACT.update(iqr_fence=a["tukey_fence"], iqr_min_pairs=a["tukey_min_pairs"], min_valid_mass=a["min_valid_mass"],
                 rel_err_xs_N=a["xsec_error_N"], rel_err_xs_Ar=a["xsec_error_Ar"], k_margin_eV=a["k_margin_eV"],
                 pin_Ne=ne)
    S.ZOOM_AR_WL, S.ZOOM_HALF_NM = pl["zoom_Ar_line_nm"], pl["zoom_half_width_nm"]
    S.PLOTS.update({k: pl[k] for k in S.PLOTS})
    S.zoom_lines.cache_clear()


def make_report(s, path):
    import MakeFitReport as R                       # noqa: E402 (SmallAnalysisScripts is on the path)
    r = s["report"]
    R.REPORT = in_repo(r["folder"])
    R.FIGDIR, R.TABDIR = os.path.join(R.REPORT, "figures"), os.path.join(R.REPORT, "tables")
    R.EXAMPLE_SET, R.EXAMPLE_X, R.S_EXAMPLE_X = r["example_set"], tuple(r["example_x"]), r["s_example_x"]
    R.main(path)


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    path = os.path.abspath(args[0] if args else DEFAULT_INPUT)
    s = read_input(path)
    plan = s["run"]["plan_only"] or "--plan" in argv
    print(f"input file {path}")
    print("settings different from the defaults:", ("\n  " + "\n  ".join(changed(s))) if changed(s) else "none")
    over = {MODEL_KEYS[k]: s["model"][k] for k in MODEL_KEYS}
    over.update({FIT_KEYS[k]: s["fit"][k] for k in FIT_KEYS}, export_flags=s["spectra"]["export_flags"])
    os.environ["CR_CONFIG_OVERRIDES"] = json.dumps(over)
    os.environ["CR_XSEC_FILE"] = s["model"]["cross_sections"]
    os.environ["CR_TABLE_TAG"] = table_tag(s)
    if not s["run"]["show_plots"]:
        os.environ["MPLBACKEND"] = "Agg"            # figures are only saved; plt.show() does not wait
    sys.path.insert(0, os.path.join(ROOT, "SmallAnalysisScripts"))
    import N2SweepPureArEEDF as S                   # noqa: E402 - after the environment is set
    configure(S, s)
    print(f"CR tables in {S.A.TABLE_DIR}" + (" (own folder: model settings differ from the defaults)"
                                              if table_tag(s) else ""))
    os.makedirs(S.OUTDIR, exist_ok=True)
    with open(os.path.join(S.OUTDIR, "settings_used.json"), "w", encoding="utf-8") as f:
        json.dump(dict(input_file=path, plan_only=plan, table_folder=S.A.TABLE_DIR, settings=s), f, indent=2)
    res = S.main(path, plan_only=plan)
    if s["report"]["make"] and not plan:
        make_report(s, path)
    return res


if __name__ == "__main__":
    RES = main(sys.argv[1:])
