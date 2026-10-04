# -*- coding: utf-8 -*-
"""
CreateEEDFLibraries.py

EEDF libraries for the CR-model fit (Experimental_Data/ExperimentalDataAnalysis/
CRFitNeTe.py), all from the Biagi Ar cross sections that the MultiBolt runs use:

  InputData/MultiBolt/Ar_Biagi_lowEN_span12_2term
        MultiBolt with 2 Legendre terms (same order as BOLSIG+), else as
        Ar_Biagi_lowEN_span12 (6 terms)
  InputData/Bolsig/Ar_Biagi_bolsig        BOLSIG+, no e-e, no superelastics   (E/N sweep)
  InputData/Bolsig/Ar_Biagi_bolsig_ee     + electron-electron collisions at Ne/N  (E/N x Ne)
  InputData/Bolsig/Ar_Biagi_bolsig_se     + superelastic collisions with the four 4s
                                            levels, populations from the CR model,
                                            iterated to self-consistency       (E/N x Ne)
  InputData/Bolsig/Ar_Biagi_bolsig_ee_se  both                                  (E/N x Ne)
  InputData/Bolsig/Ar_Biagi_bolsig_mw     2.45 GHz microwave field, no e-e      (E/N sweep)
  InputData/Bolsig/Ar_Biagi_bolsig_mw_ee  2.45 GHz microwave field + e-e        (E/N x Ne)
  InputData/Bolsig/Ar_Biagi_bolsig_ee_lowEN, Ar_Biagi_bolsig_mw_ee_lowEN
        the two e-e libraries extended to lower E/N, on Ne >= 1e16 m^-3 (the pure-Ar fits
        with e-e sit at the lowest E/N of the libraries above; at Ne = 1e15 the low-E/N
        tails do not reach the 4s threshold, which drops those rows for every Ne)

The E/N x Ne libraries are built on NE_GRID, which the fit must then use as its
Ne_grid (CRFitNeTe checks). Existing folders are kept - delete one to rebuild it.

Run from Spyder (F5) or:  python Scripts/CreateEEDFLibraries.py
Takes ~1 h on 12 cores, mostly the self-consistent superelastic + e-e library:
BOLSIG+ converges slowly once e-e collisions and superelastics are both on at
ionization degrees above ~1e-4, hence max_iter = 20000 and Ne <= 3e18 m^-3.
"""
import os
import sys
import time

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "Experimental_Data", "ExperimentalDataAnalysis"))
import HelperFunctions as he                        # noqa: E402
import CRFitNeTe as crf                             # noqa: E402

# ---- settings ---------------------------------------------------------------------
MULTIBOLT_XSEC = os.path.join(os.path.dirname(os.path.dirname(str(he.MULTIBOLT_EXE))),
                              "cross-sections", "Biagi_Ar.txt")
EN_MULTIBOLT = [2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8.5, 10, 12, 15, 20, 30, 50]   # Td, as span12
EN_BOLSIG = [0.75, 1.0, 1.25, 1.5] + EN_MULTIBOLT  # BOLSIG+ tails reach 14.5 eV down to ~2.5 Td;
                                                  # the no-tilt fits sit at ~1.5 Td, so go below it
NE_GRID = np.geomspace(1e15, 3e18, 22)             # m^-3, same spacing as CRFitNeTe's Ne_grid
CR_CFG = dict(crf.CONFIG, Ne_grid=NE_GRID)         # P, Tg, R and trapping of the fit
# 2.45 GHz microwave field (planar applicator): BOLSIG+ angular frequency / N. In argon the
# Ramsauer minimum leaves slow electrons nearly unheated (nu << omega), so a microwave field
# gives a cold bulk with a fuller tail, and needs a much higher E/N than DC.
MW_FREQ_HZ = 2.45e9
OMEGA_N = 2 * np.pi * MW_FREQ_HZ / he.Torr2Volume(CR_CFG["P_Torr"], CR_CFG["Tg"])   # m^3/s
EN_MICROWAVE = [6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 130, 170, 250, 400, 700]   # Td
EN_BOLSIG_LOW = [0.3, 0.4, 0.5, 0.6] + EN_BOLSIG          # Td, e-e libraries down to lower E/N
EN_MICROWAVE_LOW = [1, 1.5, 2, 3, 4, 5] + EN_MICROWAVE     # Td
NE_GRID_LOW = NE_GRID[NE_GRID >= 0.95e16]                  # m^-3, 1e16-3e18 (subset of NE_GRID)
SUPERELASTIC = {k: lbl for k, (lbl, _) in he.AR_PASCHEN_4S.items()}    # Ar(1S5) -> 4s1, ...
BOLSIG_SETTINGS = dict(n_grid=500, precision=1e-25, max_iter=20000)
N_WORKERS = 12


def biagi_files():
    """Biagi Ar set for BOLSIG+ (copy of MultiBolt's) and its superelastic version."""
    plain = he.BOLSIG_XSEC_FOLDER / "Biagi_Ar.txt"
    if not plain.is_file():
        import shutil
        shutil.copyfile(MULTIBOLT_XSEC, plain)
    se = he.BOLSIG_XSEC_FOLDER / "Biagi_Ar_superelastic.txt"
    if not se.is_file():
        he.MakeSuperelasticXsecFile(plain, se, {k: g for k, (_, g) in he.AR_PASCHEN_4S.items()})
    return plain, se


def exists(folder):
    return os.path.isdir(folder) and (he.IsBolsigLibrary(folder)
                                      or os.path.isdir(os.path.join(folder, "EEDFs_f0")))


def main():
    t0 = time.time()
    plain, se_file = biagi_files()

    name = "Ar_Biagi_lowEN_span12_2term"
    if not exists(os.path.join(he.MULTIBOLT_FOLDER, name)):
        print(f"\n== MultiBolt, 2 terms: {name}")
        he.RunMultiBolt(MULTIBOLT_XSEC, name, EN_MULTIBOLT, N_terms=2, remap_span=12, verbose=True)

    for name, EN, extra in (("Ar_Biagi_bolsig", EN_BOLSIG, {}),
                            ("Ar_Biagi_bolsig_mw", EN_MICROWAVE, dict(omega_N=OMEGA_N))):
        if not exists(os.path.join(he.BOLSIG_FOLDER, name)):
            print(f"\n== BOLSIG+: {name}")
            he.RunBolsig(plain, name, EN, **BOLSIG_SETTINGS, **extra)

    builds = [   # name, cross sections, physics, E/N [Td], Ne [m^-3]
        ("Ar_Biagi_bolsig_ee", plain, dict(electron_electron=True), EN_BOLSIG, NE_GRID),
        ("Ar_Biagi_bolsig_mw_ee", plain, dict(electron_electron=True, omega_N=OMEGA_N), EN_MICROWAVE, NE_GRID),
        ("Ar_Biagi_bolsig_ee_lowEN", plain, dict(electron_electron=True), EN_BOLSIG_LOW, NE_GRID_LOW),
        ("Ar_Biagi_bolsig_mw_ee_lowEN", plain, dict(electron_electron=True, omega_N=OMEGA_N), EN_MICROWAVE_LOW,
         NE_GRID_LOW),
        ("Ar_Biagi_bolsig_se", se_file, dict(electron_electron=False, superelastic=SUPERELASTIC), EN_BOLSIG, NE_GRID),
        ("Ar_Biagi_bolsig_ee_se", se_file, dict(electron_electron=True, superelastic=SUPERELASTIC), EN_BOLSIG,
         NE_GRID)]
    cr_solve = None
    for name, xsec, physics, EN, Ne in builds:
        if exists(os.path.join(he.BOLSIG_FOLDER, name)):
            continue
        print(f"\n== BOLSIG+ E/N x Ne library: {name} ({time.time() - t0:.0f} s)")
        if physics.get("superelastic") and cr_solve is None:
            cr_solve = crf.cr_density_solver(CR_CFG)
        he.BuildBolsigLibrary(xsec, name, EN, Ne, P_Torr=CR_CFG["P_Torr"],
                              Tg=CR_CFG["Tg"], cr_solve=cr_solve, n_iter=8, tol=0.03,
                              n_workers=N_WORKERS, **physics, **BOLSIG_SETTINGS)
    print(f"\nall libraries done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
