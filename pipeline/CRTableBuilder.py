# -*- coding: utf-8 -*-
"""
CRTableBuilder.py  -  which CR table (and EEDF library) each condition uses, and building the missing tables
=======================================================================================================================
TAKES IN   a condition key '<% admixture>:<p mTorr>:<Tg K>' and the EEDF choice of the run ('ar' pure-Ar EEDF, 'mix'
           Ar/N2, 'vibsat', 'tv<Tv>', 'scheme:<name>'); the environment of Main.py: CR_XSEC_FILE (cross sections of
           the CR model), CR_TABLE_TAG (own table folder for non-default [model] settings), CR_CONFIG_OVERRIDES
GIVES OUT  table_cfg(key, eedf) -> the CRTableFit settings of that condition's CR table (EEDF library folder, p, Tg,
           admixture, output folder); build_all([(key, eedf), ...]) builds the missing CR tables, N_WORKERS at a time
           in worker processes (this file run with --build); xsec_set(name) selects the CR cross sections
USED BY    RunDatasets (to_build, fit_cfg), LatexReport; the analysis scripts through
           SmallAnalysisScripts/AnalyticFivePFit.py

This is the pipeline half of SmallAnalysisScripts/AnalyticFivePFit.py (which keeps its cross-section comparison of the
analytic 5p set).  The CR tables are in Experimental_Data/Output/AnalyticFivePFit/tables[<tag>] (folder name kept),
one subfolder per EEDF library: ArEEDF_<library> (pure-Ar EEDF), VibSat_..., VibTv_..., Scheme_..., and one table
file per admixture % (CRTableFit._table_path).  A table is rebuilt only when its stored settings differ from the
current ones (CRTableFit.table_is_current).
=======================================================================================================================
"""
import contextlib
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # pipeline/: its modules import each other by name
import Paths                                        # noqa: E402 - the folder of Main.py
import LASAndEEDFLibraries as L                     # noqa: E402

crf, he = L.crf, L.he

XSEC = "ArgonCrossSections_BSR_analytic5p.json"     # BSR-500 without the NGFSRDW 5p fallbacks (InputData/)
CR_XSEC = os.environ.get("CR_XSEC_FILE", XSEC)          # cross sections of the CR tables (Main.py may choose)
OUTDIR = os.path.join(Paths.OUTPUT, "AnalyticFivePFit")   # (name kept: the cached CR tables are here)
TABLE_DIR = os.path.join(OUTDIR, "tables" + os.environ.get("CR_TABLE_TAG", ""))   # Main.py: other model settings
                                                    # get their own table folder
N_WORKERS = 6                                       # CR tables built at the same time
N_C = L.N_C


@contextlib.contextmanager
def xsec_set(name):
    keep = he.CROSS_SECTION_FILE
    he.CROSS_SECTION_FILE = name
    try:
        yield
    finally:
        he.CROSS_SECTION_FILE = keep


def table_cfg(key, eedf="mix"):
    """CR table settings of key with the analytic-5p set; eedf 'mix' = Ar/N2 microwave library (LASTgComparison),
    'ar' = pure-Ar microwave library at the same p and Tg (N2 quenching and dilution kept in the CR model),
    'vibsat' = Ar/N2 microwave library with saturated N2(v) (LASTgComparison.lib_folder(vibsat=True); 'ar' in pure Ar),
    'tv<Tv>' (e.g. 'tv8000') = Ar/N2 microwave library with N2(v) in a Boltzmann distribution at Tv K (lib_folder(tv=Tv)),
    'scheme:<name>' = library of the EEDF scheme <name> (CreateEEDFScheme.py; with a fixed scheme Tg the folder also
    carries the CR-model Tg of the condition)."""
    cfg = L.table_cfg(key, "n2")
    if eedf.startswith("scheme:"):
        S = L.eedf_scheme(eedf[len("scheme:"):])
        n2, p, Tg = L._unkey(key)
        lib = S.lib_folder(*S.eedf_conditions(n2, p, Tg))
        sub = f"Scheme_{lib.name}" + (f"_CRTg{Tg:g}K" if S.Tg_fixed else "")
        return dict(cfg, eedf=str(lib), outdir=os.path.join(TABLE_DIR, sub))
    if eedf == "vibsat" or eedf.startswith("tv"):
        n2, p, Tg = L._unkey(key)
        if n2 == 0:
            return table_cfg(key, "ar")
        if eedf == "vibsat":
            lib, tag = L.lib_folder(n2, p, Tg, vibsat=True), "VibSat_"
        else:
            lib, tag = L.lib_folder(n2, p, Tg, tv=float(eedf[2:])), "VibTv_"
        return dict(cfg, eedf=str(lib), outdir=os.path.join(TABLE_DIR, tag + lib.name))
    if eedf == "ar":
        n2, p, Tg = L._unkey(key)
        lib = L.lib_folder(0.0, p, Tg)
        return dict(cfg, eedf=str(lib), outdir=os.path.join(TABLE_DIR, "ArEEDF_" + lib.name))
    return dict(cfg, outdir=os.path.join(TABLE_DIR, os.path.basename(cfg["outdir"])))


def build_table(key, eedf="mix"):
    with xsec_set(CR_XSEC):
        tab = crf.build_model_table(table_cfg(key, eedf))
    print(f"{key}: {len(tab['x_grid'])} E/N rows, {(~tab['converged']).sum()} not converged", flush=True)


def build_all(keys):
    """keys: (table key, eedf)"""
    with xsec_set(CR_XSEC):
        todo = [k for k in keys if not crf.table_is_current(table_cfg(*k))]
    if not todo:
        return
    logs = os.path.join(OUTDIR, "logs")
    os.makedirs(logs, exist_ok=True)
    print(f"CR tables: {len(todo)} jobs ({N_WORKERS} at a time) ...", flush=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0, procs = time.time(), {}
    for k in todo:
        while sum(q.poll() is None for q in procs.values()) >= N_WORKERS:
            time.sleep(5)
        log = open(os.path.join(logs, f"build_{k[1].replace(':', '-')}_{k[0].replace(':', '_')}.log"), "w")
        procs[k] = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--build", *k],
                                    stdout=log, stderr=subprocess.STDOUT, cwd=crf.ROOT_DIR, env=env)
    failed = [k for k, q in procs.items() if q.wait() != 0]
    print(f"CR tables done ({time.time() - t0:.0f} s)", flush=True)
    if failed:
        raise RuntimeError(f"CR tables failed for {failed} - see {logs}")


if __name__ == "__main__":                          # CR-table workers (build_all)
    if len(sys.argv) > 2 and sys.argv[1] == "--build":
        build_table(*sys.argv[2:4])
    else:
        print(__doc__)
