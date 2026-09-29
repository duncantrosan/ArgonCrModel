# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:40:24 2026

@author: dptro
"""

import json 
import re
from pathlib import Path
import numpy as np
import math
from scipy import constants
from scipy.special import wofz

#%% Conversion and Simple Math Functions
#### Convertions Functions 
def eV2nm(eV):
    Joules = eV * constants.e
    Wavelength = (constants.h * constants.c) / Joules * 10**9
    return Wavelength

def Volume2Torr(N,T=300):
    k_B = 1.380649e-23   # Boltzmann constant, J/K
    P = N*k_B*T # p = (n/V)kT pressure in pascal
    Torr = P / 133.322368  # Pascal -> Torr
    return Torr

def Torr2Volume(P,T = 300): # Converts partial pressure of a gas to molecules per m^3
    """Converts partial pressure of a gas (in Torr) to number density (molecules per m^3).
    
    P : pressure in Torr
    T : temperature in K (default 293.15 K = 20 °C)
    """
    k_B = 1.380649e-23   # Boltzmann constant, J/K
    Pa = P * 133.322368  # Torr -> Pascal
    return Pa / (k_B * T)  # n = P / (k_B T), molecules per m^3

def Te2EEPF(Te = 2,E = None):
    # Returns the probability density function of maxweillian Te 
    # EEPF so int(E,EEPF) = 1
    # or in python test: Test = np.trapezoid(EEPF,E) should equal near 1
    pi = math.pi 
    
    E_True = E is not None
    if E is None:
        E = np.linspace(0, 40, 1000)
    
    EEPF = 2*np.sqrt(1/pi)* Te**(-3/2)*np.sqrt(E)*np.exp(-E/Te)
    if E_True:
        return EEPF
    else:
        return EEPF, E

    

#%% Electron Energy Distributions (Maxwellian or numerical, e.g. MultiBolt)
# All electron-impact rates in the CR model come from an EEDF dict:
#   'E'      : energy grid [eV]
#   'EEDF'   : F(E) [eV^-1], int F dE = 1   (what Te2EEPF returns)
#   'EEPF'   : f(E) = F(E)/sqrt(E) [eV^-3/2], int sqrt(E) f dE = 1   (MultiBolt's f0)
#   'Te'     : Maxwellian temperature [eV], or None for a numerical EEDF
#   'Te_eff' : 2/3 <E> [eV], temperature of the Maxwellian with the same mean energy
#   'label'  : text for print-outs and plot legends
# Rate coefficient of a cross section sigma(E) [m^2]:
#   k = sqrt(2e/m_e) * int sqrt(E) sigma(E) F(E) dE   [m^3/s]

def MaxwellianEEDF(Te, E=None):
    """Maxwell-Boltzmann EEDF at Te [eV], on the Te2EEPF grid (0-40 eV) by default."""
    if E is None:
        E = np.linspace(0, 40, 1000)
    E = np.asarray(E, dtype=float)
    EEPF = 2*np.sqrt(1/math.pi) * Te**(-3/2) * np.exp(-E/Te)
    return {'E': E, 'EEDF': Te2EEPF(Te, E), 'EEPF': EEPF,
            'Te': float(Te), 'Te_eff': float(Te),
            'label': f'Maxwellian, Te = {Te:.2f} eV'}


def TabulatedEEDF(E_data, f_data, form='EEPF', E=None, dE=0.02, label='numerical EEDF'):
    """
    EEDF dict from a numerical distribution, e.g. Boltzmann-solver output.

    form : 'EEPF' -> f_data is f(E) [eV^-3/2] (MultiBolt f0; BOLSIG+ uses the
                     same normalisation)
           'EEDF' -> f_data is F(E) [eV^-1]
    E    : integration grid [eV]. Default: 0 to the last data point in steps of dE.

    f is interpolated linearly in log(f), which follows exponential tails
    between data points, held constant below the first point (f is flat as
    E -> 0) and set to zero above the last point, then renormalised on E.
    """
    E_data = np.asarray(E_data, dtype=float)
    f_data = np.asarray(f_data, dtype=float)
    if form == 'EEDF':          # f(0) is not defined by F(0) = 0, so drop E = 0
        keep = E_data > 0
        E_data, f_data = E_data[keep], f_data[keep] / np.sqrt(E_data[keep])
    elif form != 'EEPF':
        raise ValueError("form must be 'EEPF' or 'EEDF'")
    order = np.argsort(E_data)
    E_data, f_data = E_data[order], f_data[order]

    if E is None:
        E = np.linspace(0.0, E_data[-1], int(round(E_data[-1] / dE)) + 1)
    E = np.asarray(E, dtype=float)

    floor = f_data.max() * 1e-100    # zeros / negative round-off -> effectively zero
    logf = np.interp(E, E_data, np.log(np.clip(f_data, floor, None)))
    EEPF = np.where(E <= E_data[-1], np.exp(logf), 0.0)
    EEDF = np.sqrt(E) * EEPF

    norm = np.trapezoid(EEDF, E)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f'{label}: EEDF has zero or non-finite integral on the grid')
    EEPF, EEDF = EEPF / norm, EEDF / norm
    Te_eff = 2/3 * np.trapezoid(E * EEDF, E)
    return {'E': E, 'EEDF': EEDF, 'EEPF': EEPF,
            'Te': None, 'Te_eff': float(Te_eff), 'label': label}


def AsEEDF(eedf):
    """EEDF dict as is; a number is taken as Te [eV] of a Maxwellian."""
    return eedf if isinstance(eedf, dict) else MaxwellianEEDF(eedf)


_NUMBER = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'

def ReadMultiBoltEEDF(path):
    """
    Read one MultiBolt EEDF file, <run>/EEDFs_f0/f0_<i>.txt.

    Returns E [eV], f0 [eV^-3/2] (int sqrt(E) f0 dE = 1), and the swept value
    from the header as {'name', 'unit', 'value'}, e.g.
    {'name': 'E_N', 'unit': 'Td', 'value': 100.0} (None if not in the header).
    """
    E, f0, sweep = [], [], None
    with open(path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                # e.g. '# E_N [Td]\t1.00000e+02'
                m = re.match(r'#\s*(.+?)\s*\[([^\]]*)\]\s*(' + _NUMBER + r')\s*$', line)
                if m and sweep is None:
                    sweep = {'name': m.group(1), 'unit': m.group(2),
                             'value': float(m.group(3))}
                continue
            parts = line.split()
            try:
                e, f = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                continue        # column-name line 'eV  f0', or blank
            E.append(e)
            f0.append(f)
    if not E:
        raise ValueError(f'No EEDF data found in {path}')
    return np.array(E), np.array(f0), sweep


def ImportMultiBoltEEDFs(RunFolder, E=None, dE=0.02, verbose=True):
    """
    EEDF dicts for every sweep point of a MultiBolt export, in sweep order.

    RunFolder : the export folder that holds RUN_DETAILS.txt and EEDFs_f0/.
                MultiBolt must be run without --LIMIT_EXPORT / --NO_EXPORT so
                the f0 files are written.
    E, dE     : integration grid, see TabulatedEEDF.
    Each dict also carries 'sweep' ({'name','unit','value'}) and 'source'.
    """
    f0_dir = Path(RunFolder) / 'EEDFs_f0'
    files = sorted((p for p in f0_dir.glob('f0_*.txt') if p.stem[3:].isdigit()),
                   key=lambda p: int(p.stem[3:]))
    if not files:
        raise FileNotFoundError(f'No EEDFs_f0/f0_<i>.txt files in {RunFolder} - point '
                                'MULTIBOLT_RUN at a MultiBolt export folder')
    EEDFs = []
    for path in files:
        E_data, f0, sweep = ReadMultiBoltEEDF(path)
        label = (f"MultiBolt {sweep['name']} = {sweep['value']:g} {sweep['unit']}"
                 if sweep else f'MultiBolt {path.stem}')
        eedf = TabulatedEEDF(E_data, f0, form='EEPF', E=E, dE=dE, label=label)
        eedf['sweep'] = sweep
        eedf['source'] = str(path)
        EEDFs.append(eedf)

    if verbose:
        print(f'Imported {len(EEDFs)} MultiBolt EEDFs from {RunFolder}')
        for eedf in EEDFs:
            print(f"  {eedf['label']:<34} <E> = {1.5*eedf['Te_eff']:7.3f} eV   "
                  f"Te_eff = {eedf['Te_eff']:6.3f} eV   E_max = {eedf['E'][-1]:6.1f} eV")
    return EEDFs


# MultiBolt command-line binary (lives outside this repo - edit if it moves)
MULTIBOLT_EXE = Path(r'C:\Users\dptro\Documents\Work\Python\Multibolt\MultiBolt-master'
                     r'\MultiBolt-master\bin\multibolt_win64.exe')
MULTIBOLT_FOLDER = Path(__file__).resolve().parent.parent / 'InputData' / 'MultiBolt'

def RunMultiBolt(XsecFiles, Name, EN_Td, species='Ar', P_Torr=1, T_K=300,
                 Nu=1000, N_terms=6, model='HD+GE', export_xsecs=False,
                 overwrite=False, ExportFolder=MULTIBOLT_FOLDER, exe=MULTIBOLT_EXE,
                 verbose=True):
    """
    Run MultiBolt for an E/N sweep and return the EEDFs (ImportMultiBoltEEDFs).

    XsecFiles : LXCat cross-section file, or a list of them. Must hold a complete
                set for each species (elastic/effective + excitation + ionization).
    Name      : run folder, written to ExportFolder/Name (InputData/MultiBolt by
                default) - use it as MULTIBOLT_RUN in MainFileV2.py afterwards
    EN_Td     : E/N values [Td], one EEDF each
    species   : 'Ar' (fraction 1), or {'Ar': 0.9, 'N2': 0.1}; names as in the files
    export_xsecs : also save the cross sections MultiBolt used in the run folder
    overwrite : MultiBolt silently replaces an existing run folder, so by default
                an existing Name is refused

    The command and MultiBolt's console output are saved in the run folder as
    multibolt_log.txt, which records the cross-section files used.
    """
    import subprocess
    exe = Path(exe)
    if not exe.is_file():
        raise FileNotFoundError(f'MultiBolt binary not found: {exe} (set he.MULTIBOLT_EXE)')
    XsecFiles = [XsecFiles] if isinstance(XsecFiles, (str, Path)) else list(XsecFiles)
    XsecFiles = [Path(f).resolve() for f in XsecFiles]
    for f in XsecFiles:
        if not f.is_file():
            raise FileNotFoundError(f'Cross-section file not found: {f}')
    RunFolder = Path(ExportFolder).resolve() / Name
    if RunFolder.exists() and not overwrite:
        raise FileExistsError(f'{RunFolder} already exists - pick another Name or pass overwrite=True')
    RunFolder.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(species, str):
        species = {species: 1.0}

    cmd = [str(exe)]
    for f in XsecFiles:                      # cross sections before species
        cmd += ['--LXCat_Xsec_fid', str(f)]
    for name, frac in species.items():
        cmd += ['--species', name, str(frac)]
    cmd += ['--export_location', str(RunFolder.parent), '--export_name', Name,
            '--sweep_option', 'EN_Td', '--sweep_style', 'def', *[f'{x:g}' for x in EN_Td],
            '--model', model, '--N_terms', str(N_terms), '--Nu', str(Nu),
            '--p_Torr', f'{P_Torr:g}', '--T_K', f'{T_K:g}', '--EN_Td', f'{EN_Td[0]:g}',
            '--initial_eV_max', '100', '--USE_ENERGY_REMAP',
            '--remap_target_order_span', '10', '--remap_grid_trial_max', '10',
            '--conv_err', '1e-6', '--weight_f0', '1.0', '--iter_max', '100', '--iter_min', '4']
    if export_xsecs:
        cmd.append('--EXPORT_XSECS')

    if verbose:
        print(f'Running MultiBolt: {len(EN_Td)} E/N points, Nu = {Nu} -> {RunFolder}')
    result = subprocess.run(cmd, cwd=exe.parent, capture_output=True, text=True, errors='replace')
    log = (subprocess.list2cmdline(cmd) + '\n\n' + result.stdout + result.stderr)
    if result.returncode != 0 or not (RunFolder / 'EEDFs_f0').is_dir():
        tail = '\n'.join(log.strip().splitlines()[-15:])
        raise RuntimeError(f'MultiBolt failed (exit code {result.returncode}):\n{tail}')
    with open(RunFolder / 'multibolt_log.txt', 'w') as file:
        file.write(log)
    return ImportMultiBoltEEDFs(RunFolder, verbose=verbose)


#%% Helpers for Radiation Trapping solve   
    
############
def Einstein2Oscilator(A_ul, lam, g_u, g_l):
    e, eps0, me, c = constants.e, constants.epsilon_0, constants.m_e, constants.c
    nu = c/lam
    f = A_ul*(g_u/g_l)*(eps0*me*c**3)/(2*np.pi*e**2*nu**2)
    return f

def FindK0(UpperLevel, LowerLevel, Reaction, P_torr, Tg,
                m_Ar=6.63e-26):
    """Convert (pressure, temperature) -> (a, tau0) for a resonance line."""
    
    # Define Constants
    kB, c, e, eps0, me = (constants.k, constants.c, constants.e,
                          constants.epsilon_0, constants.m_e)
    g_u, g_l = UpperLevel['g'], LowerLevel['g']
    lam = eV2nm(UpperLevel['energy_eV'] - LowerLevel['energy_eV'])/1e9
    A_ul = Reaction['Aki']
    nu0 = c/lam

    n_g  = (P_torr*133.322)/(kB*Tg)               # ground density m^-3
    dnu_D = (nu0/c)*np.sqrt(2*kB*Tg/m_Ar)         # 1/e half-width
    f_lu = Einstein2Oscilator(A_ul, lam, g_u, g_l) # Oscillator strength

    # resonance (self) broadening FWHM, Hz
    K = e**2/(4*np.pi*eps0*me)
    Gamma_L = 1.61*K*np.sqrt(g_l/g_u)*f_lu*n_g/(2*np.pi*nu0)

    a    = (Gamma_L/2.0)/dnu_D
    phi0 = np.real(wofz(1j*a))/(dnu_D*np.sqrt(np.pi))
    k0_over_n = (lam**2/(8*np.pi))*(g_u/g_l)*A_ul*phi0
    # Returns the Lorenztian over Guassian ratio and k0 over the lower state density
    return a, k0_over_n

def FindTau(UpperLevel, LowerLevel, Reaction, P_torr, Tg, R_m,
                m_Ar=6.63e-26):
    """Convert (pressure, temperature) -> (a, tau0) for a resonance line."""
    kB, c, e, eps0, me = (constants.k, constants.c, constants.e,
                          constants.epsilon_0, constants.m_e)
    g_u, g_l = UpperLevel['g'], LowerLevel['g']
    lam = eV2nm(UpperLevel['energy_eV'] - LowerLevel['energy_eV'])/1e9
    A_ul = Reaction['Aki']
    nu0 = c/lam

    n_g  = (P_torr*133.322)/(kB*Tg)               # ground density m^-3
    dnu_D = (nu0/c)*np.sqrt(2*kB*Tg/m_Ar)         # 1/e half-width
    f_lu = Einstein2Oscilator(A_ul, lam, g_u, g_l) # Oscillator strength

    # resonance (self) broadening FWHM, Hz
    K = e**2/(4*np.pi*eps0*me)
    Gamma_L = 1.61*K*np.sqrt(g_l/g_u)*f_lu*n_g/(2*np.pi*nu0)

    a    = (Gamma_L/2.0)/dnu_D
    phi0 = np.real(wofz(1j*a))/(dnu_D*np.sqrt(np.pi))
    k0_over_n = (lam**2/(8*np.pi))*(g_u/g_l)*A_ul*phi0
    tau0 = k0_over_n*n_g*R_m
    return a, tau0


def FindTauInModel(UpperLevel, LowerLevel, Reaction, P_torr, Tg, R_m,
                m_Ar=6.63e-26):
    """Convert (pressure, temperature) -> (a, tau0) for a resonance line."""
    kB, c, e, eps0, me = (constants.k, constants.c, constants.e,
                          constants.epsilon_0, constants.m_e)
    g_u, g_l = UpperLevel['g'], LowerLevel['g']
    lam = eV2nm(UpperLevel['energy_eV'] - LowerLevel['energy_eV'])/1e9
    A_ul = Reaction['coeff']
    nu0 = c/lam

    n_g  = (P_torr*133.322)/(kB*Tg)               # ground density m^-3
    dnu_D = (nu0/c)*np.sqrt(2*kB*Tg/m_Ar)         # 1/e half-width
    f_lu = Einstein2Oscilator(A_ul, lam, g_u, g_l) # Oscillator strength

    # resonance (self) broadening FWHM, Hz
    K = e**2/(4*np.pi*eps0*me)
    Gamma_L = 1.61*K*np.sqrt(g_l/g_u)*f_lu*n_g/(2*np.pi*nu0)

    a    = (Gamma_L/2.0)/dnu_D
    phi0 = np.real(wofz(1j*a))/(dnu_D*np.sqrt(np.pi))
    k0_over_n = (lam**2/(8*np.pi))*(g_u/g_l)*A_ul*phi0
    tau0 = k0_over_n*n_g*R_m
    return a, tau0



#############
# Find k_0 
def FindKOverN(UpperLevel,LowerLevel,Reaction,Tg = 300):
    # Function assumes a Voight profile of light emission 
    # All guassian broadening is Doppler all Lorenzian broadening is Van Der Waals
    
    m_Ar = 6.63 *10**-26
    gRatio = UpperLevel['g'] / LowerLevel['g']
    LambdaTerm = (eV2nm(UpperLevel['energy_eV'] - LowerLevel['energy_eV'])/10**9   )**2 / (8*math.pi) 
    Ein = Reaction['Aki']/(8*math.pi)
    Vel = np.sqrt(m_Ar/(constants.k * Tg))
    KoN = gRatio * LambdaTerm * Vel * Ein
    return KoN


#%% Formulas for cross sections /rates

def FindDiffusionCoeff(P=1,Tg = 300):
    D_s3 = 1.9 * 10**18 * 100# m^-1 s^-1 From Spectrochimica Acta Part B 62 (2007) 344 – 356
    D_s5 = 1.8 * 10**18 * 100# m^-1 s^-1 From Spectrochimica Acta Part B 62 (2007) 344 – 356
    n = Torr2Volume(P,Tg)
    Deff_s3 = D_s3 * 1/n * np.sqrt(Tg/300) # m^2 /s
    Deff_s5 = D_s5 * 1/n * np.sqrt(Tg/300) # m^2 /s
    return Deff_s3 , Deff_s5

def FindDiffusionTime(P,Tg,R):
    # P in torr Tg in K R in m
    s3,s5 = FindDiffusionCoeff(P,Tg)
    Lam = 4.493 # Derived from transport geomettry in hemispherical Coord. 
    T_s3 = (R/Lam)**2 / s3
    T_s5 = (R/Lam)**2 / s5
    return T_s3, T_s5


def CreateIonizationCrossSections(LevelList, Ee=None):
    pi = math.pi
    a0 = constants.physical_constants['Bohr radius'][0]           # cm
    R  = constants.physical_constants['Rydberg constant times hc in eV'][0]  # eV
    e4 = (2 * a0 * R)**2   # cm^2 * eV^2

    E_ion = 15.7596  # eV, Ar first IP
    alpha = 3.25

    E_True = Ee is not None
    if Ee is None:
        Ee = np.linspace(0.01, 100, 2000)   # avoid Ee=0
    Ee = np.atleast_1d(Ee).astype(float)
    
    
    
    
    for label, level in LevelList.items():
        level['Ionization Data'] = {
            'Energy_eV': [],
            'CrossSection_m^2': [],
            'Threshold_eV':[],
            'Rate_cm^3':[]
        }
        
        if level.get('kind') == 'ground':
            continue   # handle ground state separately

        Ek = E_ion - level['energy_eV']   # E_pi from this level

        with np.errstate(divide='ignore', invalid='ignore'):
            Cross = (pi * e4 / (Ee + alpha * Ek)
                      * (5/(3*Ek) - 1/Ee - 2*Ek/(3*Ee**2)))

        Cross = np.where(Ee > Ek, Cross, 0.0)
        Cross = np.nan_to_num(Cross, nan=0.0, posinf=0.0, neginf=0.0)

        level['Ionization Data']['CrossSection_m^2'] = Cross
        level['Ionization Data']['Energy_eV'] = Ee
        level['Ionization Data']['Threshold_eV'] = Ek
    return LevelList

def CreateIonizationRate(): 
    a =1 
    return a 

def MaxweillianReactionRates(CrossSections,Te=2):
    m_e = 9.10938e-31 # electron mass in kg
    e_charge = 1.60217e-19 # C / J per eV
    # Create Energy Vector 
    E = np.linspace(0,20,1000)
    
    EEPF = Te2EEPF(Te,E)
    Results = []
    for CS in CrossSections:
        Threshold  = CS['threshold_eV']
        DataEnergy = CS['energy_eV']
        DataCS     = CS['cross_section']

        InterpolatedCS = np.interp(E, DataEnergy, DataCS, left=0.0, right=DataCS[-1])
        InterpolatedCS[E <= Threshold] = 0.0     # enforce threshold, same length as E

        Integrand = np.sqrt(E) * EEPF * InterpolatedCS
        Rate = np.sqrt(2 * e_charge / m_e) * np.trapezoid(Integrand, E)

        NewCS = dict(CS)          # shallow copy, so original CrossSections is untouched
        NewCS['Rate'] = Rate
        Results.append(NewCS)

    return Results
        


def FindRadiationTrapping(RadiationTrappingMatrix,Tau,a):
    
    a=1
    return 
        
    

#%%  Level List
###############################################################################
## Import JSON Files and Build Dicts
def ImportCrossSections():
    # 1. Get the directory
    MainDir = Path(__file__).resolve().parent.parent
    
    # 2. Join using the / operator
    DataFolder = MainDir / 'InputData'
    CSPath = DataFolder / 'ArgonCrossSections.json'
    
    # 3. Open the file directly using the Path object
    with open(CSPath, 'r') as file:
        CrossSectionList = json.load(file)
        return CrossSectionList

def ImportLevelList():
    # 1. Get the directory
    MainDir = Path(__file__).resolve().parent.parent
    
    # 2. Join using the / operator
    DataFolder = MainDir / 'InputData'
    LevelListPath = DataFolder / 'ArgonLevelList.json'
    
    # 3. Open the file directly using the Path object
    with open(LevelListPath, 'r') as file:
        LevelList = json.load(file)
        return LevelList

def ImportReactionList(IncludeSupplement=True):
    # IncludeSupplement=False returns the NIST export alone (used to build the supplement)
    # 1. Get the directory
    MainDir = Path(__file__).resolve().parent.parent
    
    # 2. Join using the / operator
    DataFolder = MainDir / 'InputData'
    ReactionListPath = DataFolder / 'ArgonReactionList.json'
    
    # 3. Open the file directly using the Path object
    with open(ReactionListPath, 'r') as file:
        ReactionList = json.load(file)
    for t in ReactionList['transitions']:
        t.setdefault('source', 'NIST ASD')

    # 4. Add transitions NIST has no A for (ArgonReactionListSupplement.json);
    #    a line that is also in the NIST file keeps the NIST value
    SupplementPath = DataFolder / 'ArgonReactionListSupplement.json'
    if IncludeSupplement and SupplementPath.exists():
        with open(SupplementPath, 'r') as file:
            Supplement = json.load(file)
        in_nist = {(_key_of(t['upper']), _key_of(t['lower'])) for t in ReactionList['transitions']}
        for t in Supplement['transitions']:
            if (_key_of(t['upper']), _key_of(t['lower'])) in in_nist:
                print(f"Supplement line {t['wl_nm']} nm is also in the NIST list - keeping NIST")
                continue
            ReactionList['transitions'].append(t)
    return ReactionList

def BuildRateModel(LevelList, LevelList_Update):
    Model = {label: dict(info) for label, info in LevelList.items()}
    for lvl in Model.values():
        lvl['RadiativeDecay'] = []

    # --- Radiative decay: upper -> lower + photon ---
    for t in LevelList_Update['transitions_in_set']:
        up, lo, Aki, WL = t['upper_label'], t['lower_label'], t['Aki'], t['wl_nm']
        if Aki is None or up not in Model or lo not in Model:
            continue
        Model[up]['RadiativeDecay'].append(
            {'partner': lo, 'coeff': Aki, 'process': 'radiative',
             'direction': 'loss', 'wavelength_nm': WL})
        Model[lo]['RadiativeDecay'].append(
            {'partner': up, 'coeff': Aki, 'process': 'radiative',
             'direction': 'gain', 'wavelength_nm': WL})

    # --- Cascades from untracked upper levels: informational only ---
    for t in LevelList_Update['transitions_cascade_in']:
        lo, Aki = t['lower_label'], t['Aki']
        if Aki is None or lo not in Model:
            continue
        Model[lo].setdefault('untracked_cascade_in', []).append(
            {'Aki': Aki, 'wl_nm': t['wl_nm']})

    # --- Radiative loss to UNTRACKED lower levels ---
    A_tot = LevelList_Update['A_total_per_upper']
    for label, lvl in Model.items():
        A_all = A_tot.get(label, 0.0)
        A_tracked = sum(r['coeff'] for r in lvl['RadiativeDecay']
                        if r['direction'] == 'loss')
        A_missing = A_all - A_tracked
        lvl['A_untracked_loss'] = A_missing if A_missing > 1e-3 * A_all else 0.0

    return Model

#%% Electron Excitation Functions
def AddElectronExcitation(ModelData, CrossSectionList):
    CrossSections = CrossSectionList['cross_sections']

    for label, level in ModelData.items():
        level['Electron Impact CrossSections'] = {
            'Reactants': [],
            'Products': [],
        }
        for CS in CrossSections:
            LowerLevel = CS['lower_label']
            UpperLevel = CS['upper_label']

            # Drop reactions pointing to/from untracked levels (e.g. 5p),
            # now that ground state has already been correctly resolved above.
            if LowerLevel is None or UpperLevel is None:
                continue

            if LowerLevel == label:
                level['Electron Impact CrossSections']['Reactants'].append({
                    'partner_label': UpperLevel,
                    'threshold_eV': CS['threshold_eV'],
                    'energy_eV': CS['energy_eV'],
                    'cross_section': CS['cross_section'],
                })
            if UpperLevel == label:
                level['Electron Impact CrossSections']['Products'].append({
                    'partner_label': LowerLevel,
                    'threshold_eV': CS['threshold_eV'],
                    'energy_eV': CS['energy_eV'],
                    'cross_section': CS['cross_section'],
                })
    return ModelData

# Bug fix to help parse the ground state in LXcat reactions 
def resolve_label(lxcat_name, crosswalk):
    if lxcat_name is None:
        return None
    cleaned = lxcat_name.strip().rstrip('<').strip()
    if cleaned.startswith('Ar(3p6'):
        return 'ground'
    return crosswalk.get(cleaned)

def ImportRadiationTrappingMatrix():
    # 1. Get the directory
    MainDir = Path(__file__).resolve().parent.parent
    
    # 2. Join using the / operator
    DataFolder = MainDir / 'InputData'
    CSPath = DataFolder / 'RadiationTrappingLookupTable.json'
    
    # 3. Open the file directly using the Path object
    with open(CSPath, 'r') as file:
        TrappingMatrix = json.load(file)
        return TrappingMatrix

def AddDiffusionLoss(ModelData,P,T,R):
    T_s3 , T_s5 = FindDiffusionTime(P, T, R)
    for MD in ModelData.values() :
        if MD['label'] == '4s1': # 4s1 is my notation and corresponds to 1s3 in Paschen
            MD['DiffusionLoss'] = T_s3
        if MD['label'] == '4s3': # 4s3 is 1s5 in Paschen Notation
            MD['DiffusionLoss'] = T_s5
    return ModelData
    

#%% Parsing data an helper lookups 

def _level_lookup(levels):
    """Build (config, term, J) -> label map from the levels dict."""
    lookup = {}
    for label, v in levels.items():
        key = (v['configuration'], v['term'], float(v['J']))
        lookup[key] = label
    return lookup
 
 
def _key_of(side):
    """(config, term, J) tuple from a transition's 'upper'/'lower' subdict."""
    J = side['J']
    return (side['config'], side['term'], float(J) if J is not None else None)
 
 
def combine(levels, transitions):
    """Pure combination: match transitions to levels, classify, total.
 
    Parameters
    ----------
    levels : dict   label -> level record (must have configuration, term, J)
    transitions : list of dicts with 'upper','lower' (each config/term/J),
                  plus 'Aki','wl_nm','acc'
 
    Returns
    -------
    dict with transitions_in_set, transitions_cascade_in, A_total_per_upper
    """
    from collections import defaultdict
    lookup = _level_lookup(levels)
 
    in_set       = []
    cascade_in   = []
    A_tot_upper  = defaultdict(float)
 
    for t in transitions:
        up = lookup.get(_key_of(t['upper']))
        lo = lookup.get(_key_of(t['lower']))
 
        # accumulate the upper level's total radiative rate over every channel
        if up is not None and t.get('Aki') is not None:
            A_tot_upper[up] += t['Aki']
 
        rec = {
            'label_pair':  f"{up} -> {lo}" if (up and lo) else None,
            'upper_label': up,
            'lower_label': lo,
            'wl_nm':       t.get('wl_nm'),
            'Aki':         t.get('Aki'),
            'acc':         t.get('acc'),
            'source':      t.get('source'),
        }
 
        if up is not None and lo is not None:
            in_set.append(rec)
        elif lo is not None and up is None:
            cascade_in.append(rec)
        # up-only or neither: not part of the modeled population balance
        # (already counted in A_tot if the upper is modeled)
 
    return {
        'transitions_in_set':     in_set,
        'transitions_cascade_in': cascade_in,
        'A_total_per_upper':      dict(A_tot_upper),
    }


def AddTerm(Model, level, kind, coeff, process,
                    partner=None, depends_on=None):
    """
    kind: 'gain' or 'loss'
    level: label of the level being affected
    coeff: numeric rate coefficient/constant
    process: string tag for bookkeeping/debugging
    partner: optional single partner label (for simple two-body terms)
    depends_on: optional explicit list of (species_label, power) tuples.
                If not given, inferred from partner (or [] if partner is None).
    """
    if level not in Model:
        raise KeyError(f"Level '{level}' not found in Model")
    if kind not in ('gain', 'loss'):
        raise ValueError("kind must be 'gain' or 'loss'")

    if depends_on is None:
        depends_on = [(partner, 1)] if partner is not None else []

    entry = {
        'partner': partner,
        'coeff': coeff,
        'process': process,
        'depends_on': depends_on,
    }
    Model[level][kind].append(entry)
    return entry



#%% Create Analytical Excitation rates 
# -*- coding: utf-8 -*-
"""
Analytic (Drawin / Bogaerts) electron-impact excitation cross sections.

Paste this whole block into HelperFunctions.py, e.g. under a new cell
    #%% Analytic cross sections (Bogaerts 1998)
placed AFTER Einstein2Oscilator and eV2nm are defined.

Requires (already in HelperFunctions.py):
    numpy as np, math, scipy.constants as constants
    eV2nm(), Einstein2Oscilator()

Entry point:
    ModelData = AddAnalyticExcitationCrossSections(ModelData)

Produces cross sections in m^2 on a shared energy grid in eV, appended to
    level['Electron Impact CrossSections']['Reactants']   (level = LOWER)
    level['Electron Impact CrossSections']['Products']    (level = UPPER)
with exactly the same keys the LXCat importer uses, plus a 'source' tag.
"""




# ---------------------------------------------------------------------------
# Constants / provenance
# ---------------------------------------------------------------------------

BOGAERTS_SOURCE = ("Bogaerts, Gijbels & Vlcek, J. Appl. Phys. 84, 121 (1998), "
                   "Sec. III A - Drawin semi-empirical formulae "
                   "(orig. Drawin 1967; cf. Vlcek, J. Phys. D 22, 623 (1989))")

_A0_M       = constants.physical_constants['Bohr radius'][0]                      # m
_E_H        = constants.physical_constants['Rydberg constant times hc in eV'][0]  # 13.6057 eV
_FOURPI_A0SQ = 4.0 * np.pi * _A0_M**2                                             # 3.5195e-20 m^2
_CM2_TO_M2  = 1.0e-4

# Bogaerts Eq. for optically-forbidden transitions among the four 4s levels.
# Coefficients are quoted in cm^2 in the paper -> converted to m^2 here.
_Q_4S = {(2, 3): 1.0,
         (2, 4): 0.1,
         (2, 5): 0.1,
         (3, 4): 0.1,
         (3, 5): 0.1}
_C_4S_Q   = 5.797e-15 * _CM2_TO_M2   # m^2, prefactor for the Q_nm formula
_P_4S_Q   = -0.54                    # exponent on (E - E_mn)
_C_4S_45  = 8.111e-16 * _CM2_TO_M2   # m^2, prefactor for the n=4 -> m=5 formula
_P_4S_45  = -1.04


# ---------------------------------------------------------------------------
# Energy grid
# ---------------------------------------------------------------------------

def DefaultCrossSectionGrid(E_max=200.0):
    """
    Energy grid (eV) for analytic cross sections.

    Dense below 2 eV so the near-degenerate 4s-4s transitions (dE ~ 0.08 eV)
    are resolved, dense through the 10-20 eV region where the EEPF grid in
    Te2EEPF lives, coarser out to E_max.
    """
    return np.unique(np.concatenate([
        np.linspace(0.0,  2.0,  401),
        np.linspace(2.0,  20.0, 901),
        np.linspace(20.0, E_max, 401),
    ])).astype(float)


# ---------------------------------------------------------------------------
# The three cross-section shapes
# ---------------------------------------------------------------------------

def SigmaDrawinAllowed(E, dE, f_lu, alpha=1.0, beta=1.0):
    """
    Optically allowed (dipole) transitions, Drawin form:

        sigma = 4 pi a0^2 (E_H/dE)^2 f_nm alpha (u-1)/u^2 ln(1.25 beta u)

    with u = E/dE. E [eV] array, dE [eV] scalar, returns m^2.
    """
    E = np.asarray(E, dtype=float)
    sigma = np.zeros_like(E)
    if dE <= 0 or f_lu <= 0:
        return sigma
    u = E / dE
    m = u > 1.0
    if not np.any(m):
        return sigma
    ln_term = np.clip(np.log(1.25 * beta * u[m]), 0.0, None)
    sigma[m] = (_FOURPI_A0SQ * (_E_H / dE)**2 * f_lu * alpha
                * (u[m] - 1.0) / u[m]**2 * ln_term)
    return sigma


def SigmaDrawinForbidden(E, dE, alpha=1.0, p=1.0, q=2.0):
    """
    Optically forbidden transitions, generalised Drawin form:

        sigma = 4 pi a0^2 alpha (u-1)^p / u^q,     u = E/dE

    Defaults (p, q) = (1, 2) correspond to the parity-forbidden branch.
    Use (p, q) = (2, 3) for the spin-forbidden branch.

    NOTE: the exponents in the scanned paper are ambiguous in the OCR text.
    They are exposed as parameters so you can set them once you have the
    original typeset equations (or Vlcek 1989) in front of you.
    """
    E = np.asarray(E, dtype=float)
    sigma = np.zeros_like(E)
    if dE <= 0:
        return sigma
    u = E / dE
    m = u > 1.0
    if not np.any(m):
        return sigma
    sigma[m] = _FOURPI_A0SQ * alpha * (u[m] - 1.0)**p / u[m]**q
    return sigma


def Sigma4sManifold(E, dE, g_lower, g_upper, Q=None, mode='Q'):
    """
    Bogaerts' special-cased cross sections for the optically forbidden
    transitions among the four 4s / 4s' levels (their n = 2..5):

      mode='Q'  :  sigma = (g_m/g_n) ((E-dE)/E) * 5.797e-15 * Q_nm * (E-dE)^-0.54
      mode='45' :  sigma = (g_m/g_n) ((E-dE)/E) * 8.111e-16 * (E-dE)^-1.04

    Prefactors converted from cm^2 to m^2. Returns m^2.
    """
    E = np.asarray(E, dtype=float)
    sigma = np.zeros_like(E)
    if dE <= 0:
        return sigma
    # strict inequality + small offset: the exponents are negative, so
    # (E - dE) -> 0 would diverge.
    m = E > (dE + 1e-6)
    if not np.any(m):
        return sigma
    dEE = E[m] - dE
    gratio = float(g_upper) / float(g_lower)
    if mode == 'Q':
        if Q is None:
            Q = 1.0
        sigma[m] = gratio * (dEE / E[m]) * _C_4S_Q * Q * dEE**_P_4S_Q
    else:
        sigma[m] = gratio * (dEE / E[m]) * _C_4S_45 * dEE**_P_4S_45
    return sigma


# ---------------------------------------------------------------------------
# Small structural helpers
# ---------------------------------------------------------------------------

def _ExistingExcitationPairs(ModelData):
    """Set of (lower_label, upper_label) pairs that already carry a cross section."""
    pairs = set()
    for label, lvl in ModelData.items():
        cs = lvl.get('Electron Impact CrossSections')
        if not cs:
            continue
        for r in cs.get('Reactants', []):          # this level is the LOWER one
            pairs.add((lvl['label'], r['partner_label']))
        for p in cs.get('Products', []):           # this level is the UPPER one
            pairs.add((p['partner_label'], lvl['label']))
    return pairs


def _FindAki(ModelData, upper_label, lower_label):
    """Einstein A (s^-1) for upper -> lower, or None if the transition is absent."""
    up = ModelData.get(upper_label)
    if up is None:
        return None
    for rad in up.get('RadiativeDecay', []):
        if rad.get('direction') == 'loss' and rad.get('partner') == lower_label:
            A = rad.get('coeff')
            if A:
                return float(A)
    return None


def _Is4sLevel(lvl):
    """Heuristic identification of the 3p5.4s manifold (the four 1s levels)."""
    cfg = str(lvl.get('configuration', ''))
    Ee = lvl.get('energy_eV')
    if Ee is None:
        return False
    return ('4s' in cfg) and (11.0 < float(Ee) < 12.0)


def _CoreAndL(lvl):
    """(ion core '3/2' or '1/2', outer-electron l) of an excited level; (None, None) for the ground state."""
    parts = str(lvl.get('configuration', '')).split('.')
    if lvl.get('kind') == 'ground' or len(parts) < 2 or '<' not in parts[-2]:
        return None, None
    return ('1/2' if '<1/2>' in parts[-2] else '3/2'), 'spdfg'.index(parts[-1][-1])


def _Build4sIndexMap(ModelData):
    """
    Map your labels onto Bogaerts' effective level numbers n = 2,3,4,5 for the
    4s manifold. Bogaerts orders them by energy:
        n=2  4s[3/2]_2  11.548 eV   (metastable, Paschen 1s5)
        n=3  4s[3/2]_1  11.624 eV   (resonant,   Paschen 1s4)
        n=4  4s'[1/2]_0 11.723 eV   (metastable, Paschen 1s3)
        n=5  4s'[1/2]_1 11.828 eV   (resonant,   Paschen 1s2)
    so sorting by energy reproduces his numbering regardless of your labelling.
    Returns {} unless exactly four such levels are found.
    """
    found = [(lvl['label'], float(lvl['energy_eV']))
             for lvl in ModelData.values() if _Is4sLevel(lvl)]
    if len(found) != 4:
        return {}
    found.sort(key=lambda t: t[1])
    return {lbl: n for n, (lbl, _) in enumerate(found, start=2)}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def AddAnalyticExcitationCrossSections(ModelData,
                                       E=None,
                                       alpha_allowed=1.0,
                                       beta_allowed=1.0,
                                       alpha_forbidden=0.01,
                                       forbidden_powers=(1.0, 2.0),
                                       use_4s_special=True,
                                       only_missing=True,
                                       sigma_cap_m2=1.0e-18,
                                       exclude_labels=(),
                                       neglect_intercombination_forbidden=True,
                                       verbose=True):
    """
    Fill in every lower -> upper electron-impact excitation channel that does
    not already have a measured/LXCat cross section, using the Drawin
    semi-empirical formulae as implemented by Bogaerts et al. (1998).

    Classification of each pair:
      * an Einstein A exists for upper -> lower   -> optically allowed,
        f_lu computed from that A via Einstein2Oscilator
      * both levels are in the 4s manifold        -> Bogaerts' special-cased
        4s-4s formulae (only if use_4s_special)
      * otherwise                                 -> forbidden branch

    Parameters
    ----------
    E : array or None
        Energy grid in eV. Defaults to DefaultCrossSectionGrid().
    alpha_allowed, beta_allowed : float
        Drawin's transition-dependent parameters alpha_nm, beta_nm. The paper
        takes these from Vlcek (1989) per transition; with no table available
        the standard fallback is 1.0 for both. Override per run if you fit them.
    alpha_forbidden : float
        alpha_P (or alpha_S) for the forbidden branch.
    forbidden_powers : (p, q)
        Exponents in (u-1)^p / u^q. (1,2) = parity-forbidden,
        (2,3) = spin-forbidden.
    only_missing : bool
        If True (default) never touch a pair that already has data. If False,
        analytic channels are added alongside existing ones - which would
        DOUBLE-COUNT in the rate sums, so leave this True unless you are
        deliberately comparing.
    sigma_cap_m2 : float or None
        Hard ceiling on sigma. The (E_H/dE)^2 prefactor diverges for
        near-degenerate pairs, so this guards against nonsense for closely
        spaced high-lying levels. 1e-18 m^2 = 1e-14 cm^2 is far above any
        real atomic excitation cross section. Set None to disable.
    exclude_labels : iterable
        Level labels to leave out entirely.
    neglect_intercombination_forbidden : bool
        Leave out parity-forbidden channels (outer electron dl != +-1) between
        the primed (2P1/2 core) and unprimed (2P3/2 core) systems, except among
        the 4s levels, as Bogaerts, Gijbels & Vlcek, J. Appl. Phys. 84, 121
        (1998) do: these break two selection rules and are expected to be
        very small.

    Returns
    -------
    ModelData (modified in place) with new entries carrying:
        'partner_label', 'threshold_eV', 'energy_eV', 'cross_section',
        'source', 'method', 'analytic'
    """
    if E is None:
        E = DefaultCrossSectionGrid()
    E = np.asarray(E, dtype=float)

    p_forb, q_forb = forbidden_powers
    exclude = set(exclude_labels)

    # levels usable for excitation bookkeeping
    levels = []
    for lvl in ModelData.values():
        if lvl['label'] in exclude:
            continue
        if lvl.get('energy_eV') is None or lvl.get('g') is None:
            continue
        lvl.setdefault('Electron Impact CrossSections',
                       {'Reactants': [], 'Products': []})
        levels.append(lvl)
    levels.sort(key=lambda l: float(l['energy_eV']))

    existing = _ExistingExcitationPairs(ModelData) if only_missing else set()
    map4s = _Build4sIndexMap(ModelData) if use_4s_special else {}

    added, skipped, tally = 0, 0, {'allowed': 0, 'forbidden': 0, '4s': 0, 'neglected': 0}

    for i, low in enumerate(levels):
        for up in levels[i + 1:]:
            lo_lbl, up_lbl = low['label'], up['label']
            dE = float(up['energy_eV']) - float(low['energy_eV'])
            if dE <= 0:
                continue
            if (lo_lbl, up_lbl) in existing:
                skipped += 1
                continue

            g_lo, g_up = float(low['g']), float(up['g'])
            Aki = _FindAki(ModelData, up_lbl, lo_lbl)

            if neglect_intercombination_forbidden and Aki is None:
                (core_lo, l_lo), (core_up, l_up) = _CoreAndL(low), _CoreAndL(up)
                if (core_lo is not None and core_up is not None and core_lo != core_up
                        and abs(l_lo - l_up) != 1
                        and not (lo_lbl in map4s and up_lbl in map4s)):
                    tally['neglected'] += 1
                    continue

            # --- pick the branch -------------------------------------------
            if lo_lbl in map4s and up_lbl in map4s and Aki is None:
                n_lo, n_up = map4s[lo_lbl], map4s[up_lbl]
                if (n_lo, n_up) == (4, 5):
                    sigma = Sigma4sManifold(E, dE, g_lo, g_up, mode='45')
                    method = "4s manifold, n=4->5 empirical fit"
                else:
                    Q = _Q_4S.get((n_lo, n_up))
                    if Q is None:
                        sigma = SigmaDrawinForbidden(E, dE, alpha_forbidden,
                                                     p_forb, q_forb)
                        method = "Drawin forbidden (no Q_nm tabulated)"
                    else:
                        sigma = Sigma4sManifold(E, dE, g_lo, g_up, Q=Q, mode='Q')
                        method = f"4s manifold, Q_nm={Q} (n={n_lo}->{n_up})"
                tally['4s'] += 1

            elif Aki is not None:
                lam_m = eV2nm(dE) / 1.0e9
                f_lu = Einstein2Oscilator(Aki, lam_m, g_up, g_lo)
                sigma = SigmaDrawinAllowed(E, dE, f_lu,
                                           alpha=alpha_allowed,
                                           beta=beta_allowed)
                method = (f"Drawin allowed, f_lu={f_lu:.4g} from A={Aki:.4g} s^-1, "
                          f"alpha={alpha_allowed}, beta={beta_allowed}")
                tally['allowed'] += 1

            else:
                sigma = SigmaDrawinForbidden(E, dE, alpha_forbidden,
                                             p_forb, q_forb)
                method = (f"Drawin forbidden, alpha={alpha_forbidden}, "
                          f"(p,q)=({p_forb},{q_forb})")
                tally['forbidden'] += 1

            sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
            if sigma_cap_m2 is not None:
                if sigma.max() > sigma_cap_m2 and verbose:
                    print(f"  [cap] {lo_lbl} -> {up_lbl} (dE={dE:.4f} eV): "
                          f"peak {sigma.max():.3e} m^2 clipped to {sigma_cap_m2:.1e}")
                sigma = np.minimum(sigma, sigma_cap_m2)
            if sigma.max() <= 0:
                continue

            # --- append, mirroring the LXCat entry format -------------------
            common = dict(threshold_eV=dE,
                          energy_eV=E,
                          cross_section=sigma,
                          source=BOGAERTS_SOURCE,
                          method=method,
                          analytic=True)

            low['Electron Impact CrossSections']['Reactants'].append(
                dict(partner_label=up_lbl, **common))
            up['Electron Impact CrossSections']['Products'].append(
                dict(partner_label=lo_lbl, **common))

            existing.add((lo_lbl, up_lbl))
            added += 1

    if verbose:
        print("\n" + "=" * 68)
        print("ANALYTIC EXCITATION CROSS SECTIONS (Bogaerts / Drawin)")
        print("=" * 68)
        print(f"  channels added        : {added}")
        print(f"    optically allowed   : {tally['allowed']}")
        print(f"    forbidden           : {tally['forbidden']}")
        print(f"    4s manifold special : {tally['4s']}")
        print(f"  neglected (parity-forbidden between the primed and unprimed systems): {tally['neglected']}")
        print(f"  channels already present (left alone): {skipped}")
        print(f"  grid: {E[0]:.2f} - {E[-1]:.1f} eV, {len(E)} points, sigma in m^2")
        print("=" * 68 + "\n")

    return ModelData


# ---------------------------------------------------------------------------
# Convenience: tag the imported LXCat data so provenance is uniform
# ---------------------------------------------------------------------------

def TagExistingCrossSectionSources(ModelData, source='LXCat import'):
    """Give every cross-section entry without a 'source' key one."""
    n = 0
    for lvl in ModelData.values():
        cs = lvl.get('Electron Impact CrossSections')
        if not cs:
            continue
        for direction in ('Reactants', 'Products'):
            for entry in cs.get(direction, []):
                if 'source' not in entry:
                    entry['source'] = source
                    entry['analytic'] = False
                    n += 1
    return ModelData


def ListAnalyticChannels(ModelData):
    """Return a flat list of the analytic channels, for inspection/plotting."""
    out = []
    for lvl in ModelData.values():
        cs = lvl.get('Electron Impact CrossSections', {})
        for entry in cs.get('Reactants', []):     # lvl is the lower state
            if entry.get('analytic'):
                out.append({'lower': lvl['label'],
                            'upper': entry['partner_label'],
                            'threshold_eV': entry['threshold_eV'],
                            'sigma_peak_m^2': float(np.max(entry['cross_section'])),
                            'method': entry['method']})
    out.sort(key=lambda d: d['threshold_eV'])
    return out


#%% Create Final Data 
def GetData():
    #import all levels taken into consideration JSON file includes type of level
    # Leveltypes -- ground, resonant, metastable,normal
    LevelList = ImportLevelList()
    print('Importing Level List...')
    # Import Radiative Reaction list from Nistdatabase
    RadiationReactions = ImportReactionList()
    print('Importing Reaction List...')
    # Combine level list to make radiative loss 
    LevelList_Update = combine(LevelList,RadiationReactions['transitions'])
    # Import JSON cross section from LXcat 
    CrossSectionList = ImportCrossSections()
    print('Importing Reaction List...')
    CrossSectionRates = MaxweillianReactionRates(CrossSectionList['cross_sections'])
    #Combines Radiative transition into ModelData
    ModelData = BuildRateModel(LevelList, LevelList_Update)
    # Adds in cross-section data into the Model List
    ModelData = AddElectronExcitation(ModelData,CrossSectionList)
    print('Combining Data...')
    # Adds ionization Rates to the data (adds to loss terms)
    ModelData = CreateIonizationCrossSections(ModelData)
    # Tags exisiting Cross sections and marks Lxcat data 
    ModelData = TagExistingCrossSectionSources(ModelData)      # optional, marks LXCat data
    # Fills and left over excitation rates with analytic estimation
    ModelData = AddAnalyticExcitationCrossSections(ModelData)  # fills the gaps
    # Imports Radiation Trapping Matrix
    print('Importing Radiation Trapping Lookup...')
    RadiationTrappingMatrix = ImportRadiationTrappingMatrix()
    print('Adding In Diffusion Reactions...')
    # ModelData = he.AddDiffusionLoss(ModelData)
    return ModelData,RadiationTrappingMatrix
    
#%% Main File

if __name__ == "__main__":
    # Testing Functions
    LevelList = ImportLevelList()
    A = LevelList['4p10']
    A1 = LevelList['4s2']
    B = A['configuration']
    RadiationReactions = ImportReactionList()
    LevelList_Update = combine(LevelList,RadiationReactions['transitions'])
    ModelData = BuildRateModel(LevelList, LevelList_Update)
    C1 = A['energy_eV']
    C2 = A1['energy_eV']
    
    Ediff = C1 - C2 
    Wl = eV2nm(Ediff)
    print(f'The wavelength transition is {Wl} nm')
    ModelData,RadiationTrappingMatrix = GetData()
    print('Model data imported')

