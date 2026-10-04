# -*- coding: utf-8 -*-
"""
CR model of argon with a nitrogen admixture, swept over the N2 fraction.

Nitrogen enters in three ways:
  - quenching of the four Ar(4s) levels by N2, from InputData/Ar_1s_quenching_data.csv
    (he.ImportArQuenchingData): loss frequency kQ n_N2 + kQM n_N2 n_Ar on each 4s
    level (third body taken as ground-state Ar; kQM only where measured)
  - dilution: at total pressure Pressure the Ar ground density is (1 - x) N, which
    lowers excitation from the ground state and the trapping of the resonance lines.
    Metastable diffusion still uses the total pressure (D of Ar* in Ar).
  - the EEDF, for EEDF_MODE 'bolsig' or 'multibolt': an E/N sweep for each Ar/N2
    mixture (Biagi cross sections), run on first use and reused afterwards
    (InputData/Bolsig/ArN2_<x>pct_bolsig, InputData/MultiBolt/ArN2_<x>pct_6terms).
    With 'maxwell' the EEDF is set by Te and does not see the N2.
Not included: quenching of the 4p levels by N2, the N2 emission itself, and
superelastic / e-e effects on the EEDF.

Uses the functions of MainFileV2.py (loaded without running its own sweep); the
quenching enters its solvers through each level's 'GasQuenching_s^-1'.
    python MainFileWithNitrogen.py  ->  figures + NitrogenSweep.csv in Scripts/Output
"""
import ast
import os
import csv
import contextlib
import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import HelperFunctions as he

# ---- settings -----------------------------------------------------------------
Pressure = 1                             # total pressure [Torr]
Tg = 300                                 # gas temperature [K]
R = 2/100                                # radius [m]
N2_PERCENT = [0, 1, 2, 5, 10]            # % N2 in the Ar/N2 mixture, swept
N2_MAX_PERCENT = 10                      # the mixtures studied so far go up to 10 %
QUENCH_CHOICE = 'median'                 # 'median' of the measurements, or a reference, e.g. 'Velazco1978'
Ne = [1e16, 1e17, 1e18]                  # electron density [m^-3]
NE_SHOW = 1e17                           # Ne of the figures (all Ne go to the csv)
TRAP_LINES = 'all'                       # 'all' or 'ground' (Bogaerts), see SolveDirect
# Electron energy distribution, as in MainFileV2:
#   'maxwell'   : Maxwellian at each Te below
#   'bolsig'    : BOLSIG+ E/N sweep of each mixture (seconds to run)
#   'multibolt' : MultiBolt E/N sweep (6 terms) of each mixture (minutes to run)
# For the numerical EEDFs the x-axis is Te_eff = 2/3 <E>, which depends on the mixture.
EEDF_MODE = 'maxwell'
Te = np.linspace(0.5, 3, 12)             # eV, 'maxwell' only
EN_TD = [3, 4, 5, 6, 7, 8.5, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]   # Td, new numerical runs
LINES = [750.387, 751.465, 811.531, 763.511, 420.067, 415.859]                   # nm (air), to the csv
RATIOS = [(750.387, 751.465), (811.531, 750.387), (420.067, 750.387)]
LEVELS_4S = ['4s1', '4s2', '4s3', '4s4']
NAMES_4S = {'4s1': '1s$_5$ (metastable)', '4s2': '1s$_4$ (resonant)',
            '4s3': '1s$_3$ (metastable)', '4s4': '1s$_2$ (resonant)'}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MB_XSEC_DIR = os.path.join(os.path.dirname(os.path.dirname(str(he.MULTIBOLT_EXE))), "cross-sections")
BOLSIG_SETTINGS = dict(n_grid=500, precision=1e-25, max_iter=20000)   # as Scripts/CreateEEDFLibraries.py

if any(not 0 <= p <= N2_MAX_PERCENT for p in N2_PERCENT):
    raise ValueError(f'N2_PERCENT must lie in 0-{N2_MAX_PERCENT} % (raise N2_MAX_PERCENT to go higher)')

# ---- load the CR-model functions from MainFileV2 (definitions only) -------------
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MainFileV2.py')
_tree = ast.parse(open(_path, encoding='utf-8').read())
_defs = ast.Module([n for n in _tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))], [])
cr = {'__file__': _path, 'OUTPUT_DIR': OUTPUT_DIR, 'plt': plt}
exec(compile(_defs, _path, 'exec'), cr)

with contextlib.redirect_stdout(io.StringIO()):
    ModelData, RTM = he.GetData()
tau_grid = np.unique([d['Tau_R'] for d in RTM])
shape_grid = np.unique([d['Shape'] for d in RTM])
eta = np.array([d['EscapeFactor'][0] for d in RTM]).reshape(len(tau_grid), len(shape_grid))
interp = RegularGridInterpolator((np.log10(tau_grid), shape_grid), np.log10(eta),
                                 bounds_error=False, fill_value=None)


def line(EI, wl):
    """Intensity (n*A) of the modelled line closest to wl (within 0.05 nm)."""
    best = min(EI, key=lambda x: abs(x['Wavelength'] - wl))
    return best['intensity'] if abs(best['Wavelength'] - wl) < 0.05 else np.nan


def MixtureXsecFile():
    """Biagi Ar + N2 cross sections in one file for BOLSIG+ (made on first use)."""
    path = he.BOLSIG_XSEC_FOLDER / 'Biagi_ArN2.txt'
    if not path.is_file():
        parts = [open(os.path.join(MB_XSEC_DIR, f'Biagi_{s}.txt'), errors='replace').read().rstrip()
                 for s in ('Ar', 'N2')]
        path.write_text('\n\n'.join(parts) + '\n')
    return path


def MixtureEEDFs(pct):
    """EEDFs for the mixture with pct % N2 (EEDF_MODE); numerical runs are made on first use."""
    if EEDF_MODE == 'maxwell':
        return [he.MaxwellianEEDF(t) for t in Te]
    x = pct / 100
    if EEDF_MODE == 'bolsig':
        name = 'Ar_Biagi_bolsig' if x == 0 else f'ArN2_{pct:g}pct_bolsig'
        folder = he.BOLSIG_FOLDER / name
        if not he.IsBolsigLibrary(folder):
            print(f'  running BOLSIG+ for {pct:g} % N2 -> {folder}')
            if x == 0:
                he.RunBolsig(he.BOLSIG_XSEC_FOLDER / 'Biagi_Ar.txt', name, EN_TD, verbose=False,
                             **BOLSIG_SETTINGS)
            else:
                he.RunBolsig(MixtureXsecFile(), name, EN_TD, species=['Ar', 'N2'], fractions=[1 - x, x],
                             verbose=False, **BOLSIG_SETTINGS)
        return he.ImportBolsigEEDFs(folder, verbose=False)
    if EEDF_MODE == 'multibolt':
        name = 'Ar_Biagi_lowEN_span12' if x == 0 else f'ArN2_{pct:g}pct_6terms'
        folder = he.MULTIBOLT_FOLDER / name
        if not (folder / 'EEDFs_f0').is_dir():
            print(f'  running MultiBolt for {pct:g} % N2 -> {folder}')
            species = {'Ar': 1.0} if x == 0 else {'Ar': 1 - x, 'N2': x}
            he.RunMultiBolt([os.path.join(MB_XSEC_DIR, f'Biagi_{s}.txt') for s in species], name, EN_TD,
                            species=species, N_terms=6, remap_span=12, verbose=False)
        return he.ImportMultiBoltEEDFs(folder, verbose=False)
    raise ValueError(f"EEDF_MODE must be 'maxwell', 'bolsig' or 'multibolt', not {EEDF_MODE!r}")


def SetNitrogenQuenching(ModelData, Q, n_N2, n_Ar):
    """Loss frequency [s^-1] by N2 on the 4s levels (0 on all others); returns {label: frequency}."""
    nu = {lbl: q['kQ'] * n_N2 + q['kQM'] * n_N2 * n_Ar for lbl, q in Q.items()}
    for lbl, MD in ModelData.items():
        MD['GasQuenching_s^-1'] = nu.get(lbl, 0.0)
    return nu


#%% Sweep
Q = he.ImportArQuenchingData('N2', QUENCH_CHOICE)
ModelData = he.AddDiffusionLoss(ModelData, Pressure, Tg, R)    # Ar* diffusing in the total gas
N_total = he.Torr2Volume(Pressure, Tg)
rows = []
for pct in N2_PERCENT:
    x = pct / 100
    n_N2, n_Ar = x * N_total, (1 - x) * N_total
    nu = SetNitrogenQuenching(ModelData, Q, n_N2, n_Ar)
    print(f"\n{pct:g} % N2: n_Ar = {n_Ar:.2e} m^-3, n_N2 = {n_N2:.2e} m^-3, N2 quenching "
          + ', '.join(f'{lbl} {v:.2e}/s' for lbl, v in nu.items()))
    for eedf in MixtureEEDFs(pct):
        for ne in Ne:
            with contextlib.redirect_stdout(io.StringIO()):
                Data, SD, EI, solver = cr['CRModel'](ModelData, eedf, ne, (1 - x) * Pressure, Tg, R,
                                                     interp, trap_lines=TRAP_LINES)
            row = {'N2_percent': pct, 'eedf': eedf['label'], 'Te_eff_eV': eedf['Te_eff'],
                   'EN_Td': (eedf.get('sweep') or {}).get('value', np.nan), 'Ne_m3': ne,
                   'converged': solver['converged']}
            for lbl in LEVELS_4S:
                row[f'n_{lbl}'] = Data[lbl]['density_m^-3']
                row[f'N2share_{lbl}'] = nu[lbl] / Data[lbl]['Loss_s^-1']   # fraction of the loss due to N2
            row.update({f'I_{wl}': line(EI, wl) for wl in LINES})
            rows.append(row)
    n_conv = sum(not r['converged'] for r in rows if r['N2_percent'] == pct)
    print(f'  done, {n_conv} not converged')

with open(os.path.join(OUTPUT_DIR, 'NitrogenSweep.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


#%% Plots (at NE_SHOW)
np.seterr(divide='ignore', invalid='ignore')     # line ratios are 0/0 where the EEDF has no 4p electrons
ne_show =Ne[int(np.argmin(np.abs(np.log(np.array(Ne) / NE_SHOW))))]
colors = plt.cm.viridis(np.linspace(0, 0.85, len(N2_PERCENT)))
TeLabel = r'$T_e$ [eV]' if EEDF_MODE == 'maxwell' else r'$T_{e,\mathrm{eff}} = \frac{2}{3}\langle\varepsilon\rangle$ [eV]'


def series(pct, key):
    sel = sorted((r for r in rows if r['N2_percent'] == pct and r['Ne_m3'] == ne_show),
                 key=lambda r: r['Te_eff_eV'])
    return np.array([r['Te_eff_eV'] for r in sel]), np.array([r[key] for r in sel], float)


# 4s densities and the share of their loss due to N2
fig, ax = plt.subplots(2, 4, figsize=(20, 9), sharex=True)
for pct, c in zip(N2_PERCENT, colors):
    for k, lbl in enumerate(LEVELS_4S):
        ax[0, k].semilogy(*series(pct, f'n_{lbl}'), 'o-', ms=3, color=c, label=f'{pct:g} % N$_2$')
        ax[1, k].plot(*series(pct, f'N2share_{lbl}'), 'o-', ms=3, color=c)
for k, lbl in enumerate(LEVELS_4S):
    ax[0, k].set_title(f'{lbl} = {NAMES_4S[lbl]}')
    ax[1, k].set_xlabel(TeLabel)
    ax[1, k].set_ylim(0, 1)
    for a in ax[:, k]:
        a.grid(True, alpha=0.3, which='both')
ax[0, 0].set_ylabel('Density [m$^{-3}$]')
ax[1, 0].set_ylabel('Fraction of the loss due to N$_2$ quenching')
ax[0, 0].legend(fontsize=9)
fig.suptitle(f'Ar/N$_2$ at {Pressure:g} Torr, $N_e$ = {ne_show:.0e} m$^{{-3}}$ ({EEDF_MODE} EEDF, '
             f'N$_2$ quenching: {QUENCH_CHOICE})', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'nitrogen_4s_{EEDF_MODE}.png'), dpi=300)
plt.show()

# line ratios
fig, ax = plt.subplots(1, len(RATIOS), figsize=(6 * len(RATIOS), 5))
for pct, c in zip(N2_PERCENT, colors):
    for k, (a, b) in enumerate(RATIOS):
        T, Ia = series(pct, f'I_{a}')
        _, Ib = series(pct, f'I_{b}')
        ax[k].plot(T, Ia / Ib, 'o-', ms=3, color=c, label=f'{pct:g} % N$_2$')
for k, (a, b) in enumerate(RATIOS):
    ax[k].set_title(f'{a:.1f} / {b:.1f} nm')
    ax[k].set_xlabel(TeLabel)
    ax[k].grid(True, alpha=0.3)
ax[0].set_ylabel('Line ratio')
ax[0].legend(fontsize=9)
fig.suptitle(f'Ar/N$_2$ at {Pressure:g} Torr, $N_e$ = {ne_show:.0e} m$^{{-3}}$ ({EEDF_MODE} EEDF)',
             fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'nitrogen_ratios_{EEDF_MODE}.png'), dpi=300)
plt.show()
