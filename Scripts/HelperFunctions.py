# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:40:24 2026

@author: dptro
"""

import json 
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
        E = np.linspace(0, 20, 1000)
    
    EEPF = 2*np.sqrt(1/pi)* Te**(-3/2)*np.sqrt(E)*np.exp(-E/Te)
    if E_True:
        return EEPF
    else:
        return EEPF, E

    
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
    T_s5 = (R/Lam)**2 / s3
    return T_s3, T_s5


def CreateIonizationCrossSections(LevelList, Ee=None):
    pi = math.pi
    a0 = constants.physical_constants['Bohr radius'][0] * 100          # cm
    R  = constants.physical_constants['Rydberg constant times hc in eV'][0]  # eV
    e4 = (2 * a0 * R)**2   # cm^2 * eV^2

    E_ion = 15.7596  # eV, Ar first IP
    alpha = 3.25

    E_True = Ee is not None
    if Ee is None:
        Ee = np.linspace(0.01, 100, 2000)   # avoid Ee=0
    Ee = np.atleast_1d(Ee).astype(float)

    for label, level in LevelList.items():
        if level.get('kind') == 'ground':
            continue   # handle ground state separately

        Ek = E_ion - level['energy_eV']   # E_pi from this level

        with np.errstate(divide='ignore', invalid='ignore'):
            Cross = (pi * e4 / (Ee + alpha * Ek)
                      * (5/(3*Ek) - 1/Ee - 2*Ek/(3*Ee**2)))

        Cross = np.where(Ee > Ek, Cross, 0.0)
        Cross = np.nan_to_num(Cross, nan=0.0, posinf=0.0, neginf=0.0)

        level['IonizationCrossSection'] = Cross
        level['IonizationCrossSectionEnergy'] = Ee

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
        Rate = np.sqrt(2 * e_charge**3 / m_e) * np.trapezoid(Integrand, E)

        NewCS = dict(CS)          # shallow copy, so original CrossSections is untouched
        NewCS['Rate'] = Rate
        Results.append(NewCS)

    return Results
        

        
       
        
    

#%%  Importing Data
###############################################################################
## Import JSON Files and Build Dicts
def ImportCrossSections():
    # 1. Get the directory
    MainDir = Path(__file__).resolve().parent.parent
    
    # 2. Join using the / operator
    DataFolder = MainDir / 'InputData'
    CSPath = DataFolder / 'ArgonCross_Sections.json'
    
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

def ImportReactionList():
    # 1. Get the directory
    MainDir = Path(__file__).resolve().parent.parent
    
    # 2. Join using the / operator
    DataFolder = MainDir / 'InputData'
    ReactionListPath = DataFolder / 'ArgonReactionList.json'
    
    # 3. Open the file directly using the Path object
    with open(ReactionListPath, 'r') as file:
        ReactionList = json.load(file)
        return ReactionList

def BuildRateModel(LevelList, LevelList_Update, CrossSectionRates):
    """
    Merge radiative (Aki) and electron-impact (Rate) data into one
    per-level structure of production/loss channels.

    LevelList        : original dict keyed by label, with g/energy_eV/etc.
    LevelList_Update  : output of combine(), with transitions_in_set,
                        transitions_cascade_in, A_total_per_upper
    CrossSectionRates : output of MaxwellianReactionRates(), list of dicts
                        with lower_label/upper_label/Rate
    """
    Model = {label: dict(info) for label, info in LevelList.items()}
    for lvl in Model.values():
        lvl['gain'] = []   # each entry: {'partner', 'coeff', 'process'}
        lvl['loss'] = []

    # --- Radiative decay: upper -> lower + photon ---
    for t in LevelList_Update['transitions_in_set']:
        up, lo, Aki = t['upper_label'], t['lower_label'], t['Aki']
        if Aki is None or up not in Model or lo not in Model:
            continue
        Model[up]['loss'].append({'partner': lo, 'coeff': Aki, 'process': 'radiative'})
        Model[lo]['gain'].append({'partner': up, 'coeff': Aki, 'process': 'radiative'})

    # --- Cascades from untracked upper levels: informational only ---
    # These add population to 'lower_label' but the source population isn't
    # modeled here. Store separately so you can decide how to handle them.
    for t in LevelList_Update['transitions_cascade_in']:
        lo, Aki = t['lower_label'], t['Aki']
        if Aki is None or lo not in Model:
            continue
        Model[lo].setdefault('untracked_cascade_in', []).append(
            {'Aki': Aki, 'wl_nm': t['wl_nm']})

    # --- Electron-impact excitation/de-excitation ---
    g_lookup = {label: info['g'] for label, info in LevelList.items()}
    E_lookup = {label: info['energy_eV'] for label, info in LevelList.items()}

    for CS in CrossSectionRates:
        lo, up, k_exc = CS.get('lower_label'), CS.get('upper_label'), CS.get('Rate')
        if lo is None or up is None or k_exc is None:
            continue
        if lo not in Model or up not in Model:
            continue

        # Excitation: lower -> upper
        Model[lo]['loss'].append({'partner': up, 'coeff': k_exc, 'process': 'e_excitation'})
        Model[up]['gain'].append({'partner': lo, 'coeff': k_exc, 'process': 'e_excitation'})

        # De-excitation via detailed balance (Klein-Rosseland), same k_exc
        # already integrated -> need separate integral in general, but if
        # you computed k_exc via a Maxwellian rate, the standard shortcut
        # k_deexc = (g_lo/g_up) * k_exc * exp(threshold/Te) holds ONLY for
        # a Maxwellian EEDF specifically (not general EEDFs).
        g_lo, g_up = g_lookup.get(lo), g_lookup.get(up)
        threshold = CS.get('threshold_eV')
        if g_lo and g_up and threshold is not None:
            # NOTE: Te must be passed in or stored on CS; shown here for clarity
            pass  # see below

        Model[up]['loss'].append({'partner': lo, 'coeff': None, 'process': 'e_deexcitation_TODO'})
        Model[lo]['gain'].append({'partner': up, 'coeff': None, 'process': 'e_deexcitation_TODO'})

    return Model

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
    ModelData = BuildRateModel(LevelList, LevelList_Update, CrossSectionRates)
    # Adds in cross-section data into the Model List
    ModelData = AddElectronExcitation(ModelData,CrossSectionList)
    print('Combining Data...')
    # Adds ionization Rates to the data (adds to loss terms)
    ModelData = CreateIonizationCrossSections(ModelData)
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
    print(A)
    B = A['configuration']
    
    C1 = A['energy_eV']
    C2 = A1['energy_eV']
    
    Ediff = C1 - C2 
    Wl = eV2nm(Ediff)
    print(f'The wavelength transition is {Wl} nm')
    ModelData,RadiationTrappingMatrix = GetData()
    print('Model data imported')

