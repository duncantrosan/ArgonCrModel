# -*- coding: utf-8 -*-
"""
MainFileV2.py  -  demo sweep of the argon CR model over Te and Ne (standalone)

The CR model itself (all functions) is pipeline/CRModelSolver.py; this file executes it into its own namespace
(so the functions below see the settings here, as before) and then runs the sweep and the plots of the old
MainFileV2.  Output in Scripts/Output.  Created on Wed Jul 15 15:30:28 2026, @author: dptro
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:37:07 2026

@author: dptro
"""

# Create Reaction Rates and solve equations 
import numpy as np
import HelperFunctions as he
import os

# CR-model outputs (figures etc.) go to Scripts/Output
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


Pressure = 1 # Pressure in Torr 
Tg = 300 # Temperature in K
Te = np.linspace(0.5,3,20) # Electron Temperature in ev
Ne = np.linspace(1,100,10)*10**17 # electron Density in m^-3
R = 2/100 # Radius in meters

# Electron energy distribution for all electron-impact rates
#   'maxwell'   : Maxwell-Boltzmann EEDF at each Te above
#   'multibolt' : numerical EEDFs from a MultiBolt export folder (the one holding
#                 RUN_DETAILS.txt and EEDFs_f0/), one per sweep point (e.g. E/N).
#                 Te is then replaced by Te_eff = 2/3 <E> of each EEDF, which is
#                 what the plots use as their x-axis.
#                 New run with your own cross sections (lands in InputData/MultiBolt):
#                 he.RunMultiBolt(r'path\to\Ar_set.txt', 'Ar_MySet_EN_sweep', [5, 10, 30, 100])
#   'bolsig'    : numerical EEDFs from a BOLSIG+ E/N sweep (InputData/Bolsig/<run>), used
#                 like 'multibolt'. New run: he.RunBolsig(he.BOLSIG_XSEC_FOLDER / 'Biagi_Ar.txt',
#                 'Ar_MySweep', [2, 5, 10, 50]); with e-e collisions add ionization_degree=1e-6.
#                 (E/N x Ne libraries from he.BuildBolsigLibrary are read by CRFitNeTe.)
EEDF_MODE = 'maxwell'
MULTIBOLT_RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                             'InputData', 'MultiBolt', 'Ar_Biagi_EN_sweep')
BOLSIG_RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                          'InputData', 'Bolsig', 'Ar_Biagi_bolsig')


## Import preliminary structured data 
ModelData,RadiationTrappingMatrix = he.GetData()

# ---- the CR-model functions (pipeline/CRModelSolver.py) ------------------------------------------------
_SOLVER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pipeline', 'CRModelSolver.py')
exec(compile(open(_SOLVER, encoding='utf-8').read(), _SOLVER, 'exec'))


#%% Main Code 

# Set up Model 
ModelData = he.AddDiffusionLoss(ModelData, Pressure, Tg, R)
# ModelData = AttachRadiationTrapping(ModelData,Pressure,Tg,R)
Ng = he.Torr2Volume(Pressure)
StateDensities = []
EmissionIntensities = []
# build once, reuse many times
interp = he.EscapeFactorInterpolator(RadiationTrappingMatrix)   # log10(eta) on (log10 tau, log10 a)

# Electron energy distributions to sweep over (see EEDF_MODE at the top)
if EEDF_MODE == 'maxwell':
    EEDFs = [he.MaxwellianEEDF(t) for t in Te]
    TeLabel = r'$T_e$ [eV]'
elif EEDF_MODE in ('multibolt', 'bolsig'):
    EEDFs = (he.ImportMultiBoltEEDFs(MULTIBOLT_RUN) if EEDF_MODE == 'multibolt'
             else he.ImportBolsigEEDFs(BOLSIG_RUN))
    Te = np.array([eedf['Te_eff'] for eedf in EEDFs])   # x-axis for the plots below
    TeLabel = r'$T_{e,\mathrm{eff}} = \frac{2}{3}\langle\varepsilon\rangle$ [eV]'
else:
    raise ValueError(f"EEDF_MODE must be 'maxwell', 'multibolt' or 'bolsig', not {EEDF_MODE!r}")
PlotEEDFs(EEDFs)

Ratio_a = np.zeros((len(Ne), len(Te)))   # 4p10 / 4p6
Ratio_b = np.zeros((len(Ne), len(Te)))   # 4d8 / 4p6
Ratio_c = np.zeros((len(Ne), len(Te)))   # 4d3 / 4p6
Metastable_4s1 = np.zeros((len(Ne), len(Te)))  # ADD THIS
Metastable_4s3 = np.zeros((len(Ne), len(Te)))  # ADD THIS
Resonant_4s2  = np.zeros((len(Ne), len(Te))) 
Resonant_4s4  = np.zeros((len(Ne), len(Te))) 


for j,eedf in enumerate(EEDFs) :
    for i,n in enumerate(Ne):
        print(eedf['label'])
        print(n)
        Data, SD, EI, solver = CRModel(ModelData, eedf, n, Pressure, Tg, R, interp)
        # StateDensities.append(SD)
        # EmissionIntensities.append(EI)
        # for Reac in EI.values():
        #     I750 = 
        #     I811
        #     I
        
        I_4p6  = GetLineIntensity(EI, '4p6')    # 4p[1/2]0   (unprimed)
        I_4p10 = GetLineIntensity(EI, '4p10')   # 4p'[1/2]0  (primed)
        I_4d   = GetLineIntensity(EI, '4d8')  # once you add 4d states
        I_4d3   = GetLineIntensity(EI, '4d3')  # once you add 4d states

        Ratio_a[i,j] = I_4p10 / I_4p6 if I_4p6 != 0 else np.nan
        Ratio_b[i, j] = I_4d / I_4p6 if I_4p6 != 0 else np.nan
        Ratio_c[i, j] = I_4d3 / I_4p6 if I_4p6 != 0 else np.nan
        # Metastables (ADD THESE LINES)
        Metastable_4s1[i, j] = Data['4s1']['density_m^-3']
        Resonant_4s2[i, j] = Data['4s2']['density_m^-3']
        Metastable_4s3[i, j] = Data['4s3']['density_m^-3']
        Resonant_4s4[i, j] = Data['4s4']['density_m^-3']
PlotEmissionIntensities(EI,1000)
PlotStateDensities(SD)
    


# ------------------------------------------------------------------
# Plot (a): 4p10 / 4p6
# ------------------------------------------------------------------
fig, ax = plt.subplots(1,2,figsize=(16, 5))

TeGrid, NeGrid = np.meshgrid(Te, Ne)  # match paper's axis units
CS = ax[0].contour(TeGrid, NeGrid, Ratio_a, levels=8, colors='black')
ax[0].clabel(CS, inline=True, fontsize=9)
ax[0].set_xlabel(TeLabel)
ax[0].set_ylabel(r'$N_e$ [m$^{-3}$]')
ax[0].set_title('4p10 / 4p6  (750.4 / 751.5 nm)')
CS = ax[1].contour(TeGrid, NeGrid, Ratio_b, levels=8, colors='black')
ax[1].clabel(CS, inline=True, fontsize=9)
ax[1].set_xlabel(TeLabel)
ax[1].set_ylabel(r'$N_e$ [ m$^{-3}$]')
ax[1].set_title('4d8 / 4p6')


plt.tight_layout()


plt.show()





eedf_test = 4.0 if EEDF_MODE == 'maxwell' else EEDFs[len(EEDFs) // 2]
n_test = 1e15

UpdatedModel = CreateExcitationReactionRates(ModelData, eedf_test)
UpdatedModel = CreateIonizationReactionRates(UpdatedModel, eedf_test)
UpdatedModel = CreateSuperelasticRates(UpdatedModel, eedf_test)

Data, SD, EI,solver = CRModel(UpdatedModel, eedf_test, n_test, Pressure, Tg, R, interp)

print("Loss terms for 4s3 (metastable):")
print(f"  IonLoss: {Data['4s3'].get('IonLoss', 'N/A')}")
print(f"  CollisionalLoss (Ne·k): {n_test * sum(r['Rate'] for r in Data['4s3']['Electron Impact CrossSections']['Reactants']):.3e}")
print(f"  DiffusionLoss: {Data['4s3']['DiffusionLoss']:.3e}")
print(f"  Total Loss: {Data['4s3']['Loss_s^-1']:.3e}")

print("\nMetastable density saturation:")
print(f"  Is DiffusionLoss > Ne·k? {Data['4s3']['DiffusionLoss'] > n_test * sum(r['Rate'] for r in Data['4s3']['Electron Impact CrossSections']['Reactants'])}")





#=================================================================
# ASSUMING you've already collected:
#   Te, Ne_cm3, Ratio_a, Metastable_4s3, Metastable_4s4
# from your main loop (with metastables saved as shown above)
# ====================================================================
 
# ====================================================================
# PLOT 1: Ratio vs Te (multiple lines, one per Ne)
# ====================================================================
fig, ax = plt.subplots(figsize=(12, 7))
 
# Plot every Nth line to avoid overcrowding (adjust step as needed)
step = max(1, len(Ne) // 6)  # Show ~6 Ne values
 
for i in range(0, len(Ne), step):
    ax.plot(Te, Ratio_a[i, :], 'o-', 
            label=f'$N_e$ = {Ne[i]:.1e} m$^{{-3}}$', 
            linewidth=2.5, markersize=6)
 
ax.set_xlabel(TeLabel, fontsize=13)
ax.set_ylabel(r'Intensity ratio: $I_{4p10} / I_{4p6}$', fontsize=13)
ax.set_title('Line ratio vs electron temperature', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='best')
 
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ratio_vs_Te_multiNe.png'), dpi=300)
plt.show()
 
# ====================================================================
# PLOT 2: Metastable density vs Te (multiple lines, one per Ne)
# ====================================================================
fig, ax = plt.subplots(figsize=(12, 7))
 
for i in range(0, len(Ne), step):
    n_meta_total = Metastable_4s1[i, :] + Metastable_4s3[i, :]
    ax.semilogy(Te, n_meta_total, 's-',
                label=f'$N_e$ = {Ne[i]:.1e} m$^{{-3}}$',
                linewidth=2.5, markersize=6)
 
ax.set_xlabel(TeLabel, fontsize=13)
ax.set_ylabel(r'Metastable density: $n_{4s1} + n_{4s3}$ [m$^{-3}$]', fontsize=13)
ax.set_title('Metastable population vs electron temperature', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=11, loc='best')
 
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'metastable_vs_Te_multiNe.png'), dpi=300)
plt.show()

 
 
# ====================================================================
# PLOT 2: Resonant density vs Te (multiple lines, one per Ne)
# ====================================================================
 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
 
 
# 

for i in range(0, len(Ne), step):
    # Left axis 1 Resonant 4s2 
    ax1.semilogy(Te, Resonant_4s2[i,:], 's-',
                label=f'$N_e$ = {Ne[i]:.1e} m$^{{-3}}$',
                linewidth=2.5, markersize=6)
    
    
    ax2.semilogy(Te, Resonant_4s4[i,:], 's-',
                label=f'$N_e$ = {Ne[i]:.1e} m$^{{-3}}$',
                linewidth=2.5, markersize=6)
 
ax1.set_xlabel(TeLabel, fontsize=13)
ax2.set_xlabel(TeLabel, fontsize=13)
ax1.set_ylabel(r'Resonant density $n_{4s2}$ [m$^{-3}$]', fontsize=13)
ax2.set_ylabel(r'Resonant density $n_{4s4}$ [m$^{-3}$]', fontsize=13)
ax1.set_title('4s2 vs electron temperature', fontsize=14, fontweight='bold')
ax2.set_title('4s4 vs electron temperature', fontsize=14, fontweight='bold')
for a in (ax1, ax2):
    a.grid(True, alpha=0.3, which='both')
    a.legend(fontsize=11, loc='best')
 
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'resonant_vs_Te_multiNe.png'), dpi=300)
plt.show()



# ====================================================================
# PLOT 3: Side-by-side comparison at one representative Ne
# ====================================================================
Ne_idx = len(Ne) // 2  # pick middle Ne
Ne_val = Ne[Ne_idx]
 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
 
# Left: Ratio
ax1.plot(Te, Ratio_a[Ne_idx, :], 'o-', color='darkred', linewidth=3, markersize=8)
ax1.set_xlabel(TeLabel, fontsize=12)
ax1.set_ylabel(r'Intensity ratio', fontsize=12, color='darkred')
ax1.set_title(f'Ratio vs $T_e$ (at $N_e = {Ne_val:.1e}$ m$^{{-3}}$)', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='y', labelcolor='darkred')
 
# Right: Metastable
ax2.semilogy(Te, Metastable_4s3[Ne_idx, :], 's-', label='4s3', color='steelblue', linewidth=2.5, markersize=7)
ax2.semilogy(Te, Metastable_4s1[Ne_idx, :], 's-', label='4s1', color='navy', linewidth=2.5, markersize=7, linestyle='--')
ax2.set_xlabel(TeLabel, fontsize=12)
ax2.set_ylabel(r'Metastable density [m$^{-3}$]', fontsize=12, color='navy')
ax2.set_title(f'Metastables vs $T_e$ (at $N_e = {Ne_val:.1e}$ m$^{{-3}}$)', fontsize=13)
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(fontsize=11, loc='best')
ax2.tick_params(axis='y', labelcolor='navy')
 
fig.suptitle('Ratio and metastable saturation at representative density', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ratio_metastable_detailed.png'), dpi=300)
plt.show()
 


# Plot spectrum 
x_spec, y_spec = PlotSpectrum(EI)

# ====================================================================
# STATISTICS
# ====================================================================
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)
 
for i in range(0, len(Ne), step):
    ratio_range = (np.nanmin(Ratio_a[i, :]), np.nanmax(Ratio_a[i, :]))
    ratio_change = (ratio_range[1] - ratio_range[0]) / ratio_range[0] * 100
    
    meta_range = (np.nanmin(Metastable_4s3[i, :] + Metastable_4s1[i, :]),
                  np.nanmax(Metastable_4s3[i, :] + Metastable_4s3[i, :]))
    meta_change = (meta_range[1] - meta_range[0]) / meta_range[0] * 100
    
    print(f"\nNe = {Ne[i]:.1e} cm^-3:")
    print(f"  Ratio variation: {ratio_range[0]:.4f} → {ratio_range[1]:.4f} ({ratio_change:+.1f}%)")
    print(f"  Metastable range: {meta_range[0]:.2e} → {meta_range[1]:.2e} ({meta_change:+.1f}%)")
 
print("\n" + "="*70)