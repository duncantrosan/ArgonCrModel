# -*- coding: utf-8 -*-
"""
PipelineSettings.py  -  fixed numbers of the pipeline that several modules share
=======================================================================================================================
TAKES IN   nothing
GIVES OUT  the constants below (microwave frequency, critical density, the E/N grid and BOLSIG+ precision of the
           libraries, worker count, labels)
USED BY    LASAndEEDFLibraries (libraries), RunDatasets (labels, n_c), and - so the numbers exist only once - the
           analysis scripts PressureSweepAnalysis.py and MicrowaveLowENFit.py, which defined them before

These used to live in those two analysis scripts, and the pipeline imported the whole scripts to get them.
Changing a value here changes the libraries the pipeline builds (their folder names do not record it), so do not
change them for a test: make an EEDF scheme instead (CreateEEDFScheme.py).
=======================================================================================================================
"""
import numpy as np
from scipy import constants

FREQ_HZ = 2.42e9                                    # microwave source
N_C = constants.epsilon_0 * constants.m_e * (2 * np.pi * FREQ_HZ) ** 2 / constants.e ** 2   # critical density [m^-3]
EN_MW = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 35, 40, 45, 50, 55, 60,
         70, 80, 90, 100, 115, 130, 150, 170, 210, 250, 300, 350, 400, 500, 600, 700, 850, 1000]   # Td, libraries
PREC_MAX_TD = 150                                   # pure Ar: BOLSIG+ precision 1e-30 up to here (tails reach the
                                                    # thresholds at low E/N), 1e-25 above
BOLSIG_TIMEOUT = 1200                               # s per library; then retried at 1e-25
N_WORKERS = 6                                       # BOLSIG+ libraries built at the same time
LEVELS_1S = {"1s5": "4s1", "1s4": "4s2", "1s3": "4s3", "1s2": "4s4"}   # Paschen -> CR-model label
FIGSIZE = (8, 5.5)                                  # one graph per figure
