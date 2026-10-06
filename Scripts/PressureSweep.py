# -*- coding: utf-8 -*-
"""
Pressure sweep of the CR model: emission lines and metastable densities vs P.

Per pressure the ground-state density, metastable diffusion (D ~ 1/P) and
radiation trapping are recomputed; Te follows Te = Te_ref (P_ref/P)^TE_EXPONENT
(TE_EXPONENT = 1 -> Te ~ 1/P, 0 -> fixed Te). Several Ne values are swept too.

Uses the functions of MainFileV2.py (loaded without running its own sweep).
    python PressureSweep.py   ->  figures + PressureSweep.csv in Scripts/Output
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
P_LIST = np.geomspace(0.5, 1.5, 9)       # Torr (a factor 3)
TE_REF, P_REF = 1.0, 1.0                 # eV at Torr
TE_EXPONENT = 1.0                        # Te ~ P^-exponent
NE_LIST = [3e17, 1e18, 1e19]             # m^-3
TG, R = 300, 4 / 100                     # K, m
TRAP_LINES = 'all'                       # 'all' or 'ground' (Bogaerts)
LINES = [750.387, 751.465, 811.531, 763.511, 794.818, 801.479, 420.067, 415.859]   # nm (air)
RATIOS = [(750.387, 751.465), (811.531, 750.387), (420.067, 750.387)]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- load the CR-model functions from MainFileV2 (definitions only) -------------
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MainFileV2.py')
_tree = ast.parse(open(_path, encoding='utf-8').read())
_defs = ast.Module([n for n in _tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))], [])
cr = {'__file__': _path, 'OUTPUT_DIR': OUTPUT_DIR, 'plt': plt}
exec(compile(_defs, _path, 'exec'), cr)

with contextlib.redirect_stdout(io.StringIO()):
    ModelData, RTM = he.GetData()
interp = he.EscapeFactorInterpolator(RTM)          # log10(eta) on (log10 tau, log10 a)


def line(EI, wl):
    """Intensity (n*A) of the modelled line closest to wl (within 0.05 nm)."""
    best = min(EI, key=lambda x: abs(x['Wavelength'] - wl))
    return best['intensity'] if abs(best['Wavelength'] - wl) < 0.05 else np.nan


rows = []
for Ne in NE_LIST:
    for P in P_LIST:
        Te = TE_REF * (P_REF / P) ** TE_EXPONENT
        ModelData = he.AddDiffusionLoss(ModelData, P, TG, R)
        with contextlib.redirect_stdout(io.StringIO()):
            Data, SD, EI, solver = cr['CRModel'](ModelData, Te, Ne, P, TG, R, interp, trap_lines=TRAP_LINES)
        row = {'P_Torr': P, 'Te_eV': Te, 'Ne_m3': Ne, 'converged': solver['converged'],
               'n_1s5': Data['4s1']['density_m^-3'], 'n_1s4': Data['4s2']['density_m^-3'],
               'n_1s3': Data['4s3']['density_m^-3'], 'n_1s2': Data['4s4']['density_m^-3']}
        row.update({f'I_{wl}': line(EI, wl) for wl in LINES})
        rows.append(row)
        print(f"Ne={Ne:.0e}  P={P:.3f} Torr  Te={Te:.2f} eV  n1s5={row['n_1s5']:.2e}  conv={solver['converged']}")

with open(os.path.join(OUTPUT_DIR, 'PressureSweep.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# ---- plots ------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
colors = plt.cm.Blues(np.linspace(0.45, 1, len(NE_LIST)))
for Ne, c in zip(NE_LIST, colors):
    sel = [r for r in rows if r['Ne_m3'] == Ne]
    P = [r['P_Torr'] for r in sel]
    ax[0].semilogy(P, [r['n_1s5'] for r in sel], 'o-', color=c, label=f'1s$_5$, $N_e$={Ne:.0e}')
    ax[0].semilogy(P, [r['n_1s3'] for r in sel], 's--', color=c, label=f'1s$_3$, $N_e$={Ne:.0e}')
    for (a, b), m in zip(RATIOS, 'o^s'):
        ax[1].plot(P, [r[f'I_{a}'] / r[f'I_{b}'] for r in sel], m + '-', color=c,
                   label=f'{a:.1f}/{b:.1f}' if Ne == NE_LIST[-1] else None)
Ne0 = NE_LIST[len(NE_LIST) // 2]
sel = [r for r in rows if r['Ne_m3'] == Ne0]
for wl in LINES:
    ax[2].semilogy([r['P_Torr'] for r in sel], [r[f'I_{wl}'] for r in sel], 'o-', label=f'{wl:.1f} nm')
ax[0].set_ylabel('Metastable density [m$^{-3}$]')
ax[1].set_ylabel('Line ratio (dark = high $N_e$)')
ax[2].set_ylabel(f'Line intensity, $N_e$={Ne0:.0e} [arb.]')
for a in ax:
    a.set_xlabel('Pressure [Torr]')
    a.grid(True, alpha=0.3, which='both')
    a.legend(fontsize=8)
fig.suptitle(f'Pressure sweep: $T_e$ = {TE_REF} eV $\\times$ ({P_REF}/P)$^{{{TE_EXPONENT}}}$', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pressure_sweep.png'), dpi=300)
plt.show()
