# -*- coding: utf-8 -*-
"""
Paths.py  -  every folder the pipeline uses, built from the folder of Main.py
=======================================================================================================================
TAKES IN   nothing: only where this file is (pipeline/, inside the repo folder that holds Main.py)
GIVES OUT  ROOT, the folder of Main.py, and the fixed folders below it (strings), and in_root(...)
USED BY    every module of the pipeline - no pipeline file builds a path from its own location any more, so the
           files can be moved around inside pipeline/ without changing where anything is read or written

The only paths that do not come from here are the ones in the input file: [run] echelle_folder may be absolute;
every other path in an input file is relative to ROOT (Main.py, in_repo), a data set's file_list relative to the
input file.

The output folders keep the names they had before the pipeline was collected in pipeline/ (AnalyticFivePFit,
LASTgComparison, N2SweepPureArEEDF), so the cached BOLSIG+ libraries, CR tables and line fits are reused.
=======================================================================================================================
"""
import os

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PIPELINE_DIR)                        # the folder of Main.py (the repo folder)


def in_root(*parts):
    """Path below ROOT."""
    return os.path.join(ROOT, *parts)


INPUT_DATA = in_root("InputData")                           # cross sections, level lists, quenching tables, ...
BOLSIG_LIBRARIES = in_root("InputData", "Bolsig")           # BOLSIG+ libraries (one folder each)
EEDF_SCHEMES = in_root("InputData", "EEDFSchemes")          # EEDF schemes of CreateEEDFScheme.py
BOLSIG_PROGRAM = in_root("Bolsig")                          # bolsigminus.exe, its template, Cross-Sections/
BOLSIG_CROSS_SECTIONS = in_root("Bolsig", "Cross-Sections")
EXPERIMENTAL_DATA = in_root("Experimental_Data")
ECHELLE_DATA = in_root("Experimental_Data", "EchelleData")  # default [run] echelle_folder
LAS_DATA = in_root("Experimental_Data", "LAS_Data")
OUTPUT = in_root("Experimental_Data", "Output")             # every result folder of the pipeline
CR_SOLVER_OUTPUT = in_root("Scripts", "Output")             # where the CR solver's own plots went (OUTPUT_DIR)
