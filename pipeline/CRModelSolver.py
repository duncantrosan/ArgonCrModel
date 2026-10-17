# -*- coding: utf-8 -*-
"""
CRModelSolver.py  -  the argon collisional-radiative (CR) model: balance equations of all levels
=======================================================================================================================
TAKES IN   ModelData (PhysicsHelpers.GetData: the 51 Ar levels with energies, g, radiative rates, electron-impact
           cross sections, ionization data; plus the diffusion loss, admixture quenching and atom transfer that
           CRTableFit.apply_nitrogen / PhysicsHelpers.AddDiffusionLoss attach), an EEDF (Maxwellian Te or a BOLSIG+ /
           MultiBolt EEDF), Ne [m^-3], the Ar pressure [Torr], Tg [K], plasma radius R [m] and the escape-factor
           interpolator (PhysicsHelpers.EscapeFactorInterpolator)
GIVES OUT  CRModel(...) -> (Data, SD, EI, solver):
             Data    the same level dict with every level's density_m^-3, Loss_s^-1, Production_m^-3s^-1 and the
                     electron-impact rates of this EEDF
             SD, EI  state densities and emission intensities as lists (for the plotting functions here)
             solver  {'iterations', 'residual', 'converged'}
           plus helpers: rate coefficients (CreateExcitationReactionRates, ...), GetEta (escape factor), the
           balance-equation solvers (SolveLabelEquation, SolveDirect), GetLineIntensity, plotting functions
USED BY    CRTableFit (load_cr_model: the functions of this file are executed into a dict 'cr', so cr["CRModel"]
           etc.), and the standalone scripts Scripts/MainFileV2.py (demo sweep), MainFileWithNitrogen.py,
           PressureSweep.py, which load it the same way

This file only defines functions (it used to be the top part of Scripts/MainFileV2.py, whose own demo sweep over Te
and Ne stays there).  Physics: electron-impact excitation / de-excitation / ionization from every level, radiative
decay with escape factors (radiation trapping), metastable diffusion, 4s energy pooling, three-body quenching by Ar,
quenching by an admixed gas (GasQuenching_s^-1, set by CRTableFit.apply_nitrogen), 2p atom transfer (Zhu & Pu 2010).
=======================================================================================================================
"""
import os
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # pipeline/: its modules import each other by name
import Paths                                        # noqa: E402 - the folder of Main.py
import PhysicsHelpers as he                         # noqa: E402

# CR-model figures of the plotting functions below (as before: Scripts/Output)
OUTPUT_DIR = Paths.CR_SOLVER_OUTPUT
os.makedirs(OUTPUT_DIR, exist_ok=True)


def GetEta(interp,tau,a):
    return 10**interp(((np.log10(tau), a)))

def GroundQuenchingLoss(Ng):
    """
    Loss frequency [s^-1] of a metastable by collisions with ground-state Ar
    (density Ng [m^-3]), K. Tachibana, Phys. Rev. A 34, 1007 (1986):
        three-body  Ar(1s5,1s3) + 2Ar -> Ar2* + Ar   k3 = 1.4e-32 cm^6/s
    Tachibana's two-body term (k2 = 2.3e-15 cm^3/s, Ar* + Ar -> 2Ar) is left out: a thermal
    Ar* + Ar collision cannot take up 11.5 eV, and the measured k2 is the transfer of the
    metastables to the nearby resonance levels, which radiate - in the model that path is the
    4s mixing followed by the trapped resonance decay, not a sink.
    """
    k3 = 1.4e-44   # m^6/s
    return k3 * Ng**2

def CreateSuperelasticRates(ModelData, eedf):
    """
    Compute electron-impact de-excitation (superelastic) rate coefficients.
 
    MUST be called AFTER CreateExcitationReactionRates(ModelData, eedf) in the
    same iteration, since it reads the 'Rate' field that function fills in.
 
    Adds a 'Superelastic' list to every level, mirroring the structure of
    'RadiativeDecay':
        {'partner': label, 'coeff': k_deexc (m^3/s), 'process': 'superelastic',
         'direction': 'gain'|'loss'}
 
    - 'loss' entries go on the UPPER state (it de-excites away, needs *Ne
       and the upper state's own density when used in the loss sum)
    - 'gain' entries go on the LOWER state (it gets repopulated, needs
       *Ne and the UPPER state's density when used in the production sum)
 
    eedf : Te [eV] (Maxwellian) or an EEDF dict (see he.MaxwellianEEDF,
           he.ImportMultiBoltEEDFs).
    Maxwellian EEDF: detailed balance of the rates,
        k_deexc = k_exc (g_lower/g_upper) exp(dE/Te)
    Numerical EEDF: the rates are not in detailed balance, so k_deexc is
    integrated from the Klein-Rosseland (detailed balance) cross section
        sigma_sup(E) = (g_lower/g_upper) (E + dE)/E sigma_exc(E + dE)
    which, with u = E + dE the electron energy after the collision, gives
        k_deexc = sqrt(2e/m_e) (g_lower/g_upper) int_dE u sigma_exc(u) f(u - dE) du
    with f the EEPF. This is integrated on the same grid (same sigma samples) as
    the excitation rate, so a Maxwellian reproduces exp(dE/Te) detailed balance
    exactly instead of to within the quadrature error of the threshold step.
    """
    eedf = he.AsEEDF(eedf)
    Te = eedf['Te']                      # None for a numerical EEDF
    E = eedf['E']
    logEEPF = np.log(np.maximum(eedf['EEPF'], 1e-300))    # f(u - dE) interpolated in log(f)
    prefactor = np.sqrt(2 * 1.60217e-19 / 9.10938e-31)   # sqrt(2e/m_e)
    for MD in ModelData.values():
        MD['Superelastic'] = []
 
    for US in ModelData.values():
        upper_label = US['label']
        g_upper = US.get('g')
        E_upper = US.get('energy_eV')
        if g_upper is None or E_upper is None:
            continue
 
        for Reac in US['Electron Impact CrossSections']['Products']:
            lower_label = Reac['partner_label']
            k_exc = Reac.get('Rate')
            # Maxwellian k_deexc is proportional to k_exc; a numerical EEDF with
            # no electrons above threshold still de-excites, so no skip there
            if Te is not None and not k_exc:
                continue
 
            LowerState = next((p for p in ModelData.values()
                                if p['label'] == lower_label), None)
            if LowerState is None:
                continue
 
            g_lower = LowerState.get('g')
            E_lower = LowerState.get('energy_eV')
            if g_lower is None or E_lower is None:
                continue
 
            DeltaE = E_upper - E_lower  # eV; should be > 0 for upper > lower
            if DeltaE <= 0:
                continue
 
            if Te is not None:
                k_deexc = k_exc * (g_lower / g_upper) * np.exp(DeltaE / Te)
            else:
                Sigma = np.interp(E, Reac['energy_eV'], Reac['cross_section'],
                                  left=0.0, right=0.0)
                Sigma[(E <= Reac['threshold_eV']) | (E < DeltaE)] = 0.0
                EEPF_in = np.exp(np.interp(E - DeltaE, E, logEEPF))   # f(u - dE)
                k_deexc = prefactor * (g_lower / g_upper) * np.trapezoid(E * Sigma * EEPF_in, E)
                if not k_deexc:
                    continue
 
            # Loss for the upper state (it de-excites away)
            US['Superelastic'].append({
                'partner': lower_label,
                'coeff': k_deexc,
                'process': 'superelastic',
                'direction': 'loss',
            })
            # Gain for the lower state (repopulated by the collision)
            LowerState['Superelastic'].append({
                'partner': upper_label,
                'coeff': k_deexc,
                'process': 'superelastic',
                'direction': 'gain',
            })
 
    return ModelData

def CreateExcitationReactionRates(ModelData, eedf):
    # eedf: Te [eV] (Maxwellian) or an EEDF dict, e.g. from he.ImportMultiBoltEEDFs
    eedf = he.AsEEDF(eedf)
    E, EEDF = eedf['E'], eedf['EEDF']
    m_e = 9.10938e-31       # kg
    e_charge = 1.60217e-19  # J per eV

    prefactor = np.sqrt(2 * e_charge / m_e)

    for MD in ModelData.values():
        for direction in ('Reactants', 'Products'):
            for Reac in MD['Electron Impact CrossSections'][direction]:
                Threshold  = Reac['threshold_eV']
                DataEnergy = Reac['energy_eV']
                DataCS     = Reac['cross_section']

                InterpolatedCS = np.interp(E, DataEnergy, DataCS, left=0.0, right=0.0)
                InterpolatedCS[E <= Threshold] = 0.0

                Integrand = np.sqrt(E) * EEDF * InterpolatedCS
                Rate = prefactor * np.trapezoid(Integrand, E)

                Reac['Rate'] = Rate   # overwritten fresh each call to this function

    return ModelData

def CreateIonizationReactionRates(ModelData, eedf):
    # eedf: Te [eV] (Maxwellian) or an EEDF dict, e.g. from he.ImportMultiBoltEEDFs
    eedf = he.AsEEDF(eedf)
    E, EEDF = eedf['E'], eedf['EEDF']
    m_e = 9.10938e-31       # kg
    e_charge = 1.60217e-19  # J per eV
    prefactor = np.sqrt(2 * e_charge / m_e)
    
    # Go through every level and convert ionization cross-section to rate
    for MD in ModelData.values():
        label = MD['label']
        
        # Skip ground state (no ionization from ground)
        if label == 'ground':
            continue 
        
        # Access ionization data directly (it's a dict, not a list)
        IonData = MD['Ionization Data']
        
        # Check if there's valid data
        if len(IonData['Energy_eV']) == 0:
            print(f"  No ionization data for {label}")
            continue
            
        Threshold = IonData['Threshold_eV']
        DataEnergy = IonData['Energy_eV']
        DataCS = IonData['CrossSection_m^2']
        
        # Interpolate cross-section to EEDF energy grid
        InterpolatedCS = np.interp(E, DataEnergy, DataCS, left=0.0, right=0.0)
        InterpolatedCS[E <= Threshold] = 0.0
        
        # Calculate rate coefficient
        Integrand = np.sqrt(E) * EEDF * InterpolatedCS
        Rate = prefactor * np.trapezoid(Integrand, E)
        
        # Store the rate
        IonData['Rate_cm^3'] = Rate

    
    return ModelData

def AttachAtomTransfer(ModelData, Tg, enabled=True):
    """
    Population transfer between excited levels by collisions with ground-state Ar atoms,
    Ar(x) + Ar -> Ar(y) + Ar (he.AtomTransferRates: the 2p <-> 2p and 2p -> 1s rates of
    Zhu & Pu, J. Phys. D 43, 015204 (2010), uphill rates by detailed balance at Tg).
    Adds an 'AtomTransfer' list to every level, like 'Superelastic':
        {'partner': label, 'coeff': k (m^3/s), 'direction': 'loss'|'gain'}
    'loss' on the level the population leaves, 'gain' on the level it enters; both are
    multiplied by the ground-state Ar density when used. enabled=False: empty lists.
    """
    for MD in ModelData.values():
        MD['AtomTransfer'] = []
    if enabled:
        for x, y, k in he.AtomTransferRates(ModelData, Tg):
            ModelData[x]['AtomTransfer'].append({'partner': y, 'coeff': k, 'direction': 'loss'})
            ModelData[y]['AtomTransfer'].append({'partner': x, 'coeff': k, 'direction': 'gain'})
    return ModelData


def AttachRadiationTrapping(ModelData,P,Tg,R,interp):
    # Radiation trapping is only applied to 4s resonant states 
    LowerLevel = next(MD for MD in ModelData.values() if MD['label'] == 'ground')
    for MD in ModelData.values():
        if MD['kind'] == 'resonant':
            UpperLevel = MD
            for Reac in MD['RadiativeDecay']:
                if Reac['partner'] == LowerLevel:
                    a,Tau = he.FindTauInModel(UpperLevel, LowerLevel, Reac, P, Tg, R)
                    eta = GetEta(interp, Tau, a)
                    Reac['RadiationTrappng'] = eta
    return ModelData

#%%
##########################################################################
##########################################################################
#### Solve Equations! 


def SolveLabelEquation(InputModel, Ng, Ne, P, T, R, interp):
    """
    Unified balance-equation solver for 'normal', 'resonant', and
    'metastable' states. Replaces NormalLabelEquation, MetaStableLabelEquation,
    and ResonantLabelEquation, which were near-duplicates.

    Kind-dependent differences (everything else is shared):
      - RadiativeLossSum : computed for 'normal' and 'resonant' (radiative
                            decay); skipped for 'metastable' (forbidden
                            transitions -> no radiative loss channel).
      - DiffusionLoss     : only 'metastable' states carry a DiffusionLoss
                            entry (radiation-trapped resonant lines lose
                            population via the trapping factor instead of
                            diffusion).
      - CollisionalLoss   : now applied to ALL three kinds (previously
                            missing from the 'normal' states).

    Iterates in the same order as the original three separate calls
    (metastable -> resonant -> normal) so the Gauss-Seidel-style coupling
    behaves identically to before.

    Bug fixes applied during merge:
      1. CollisionalLoss previously summed `LS['Rate'] * LowerStateDensity`,
         multiplying a loss FREQUENCY (s^-1) by the density of the state
         being excited INTO -- dimensionally wrong, and it silently
         suppressed the Ne-dependence this term is supposed to provide.
         Fixed to `Ne * sum(LS['Rate'] for LS in Reactants)`.
      2. The local variable `P` (Torr) was shadowing the function's
         `P` (Pressure) argument inside the radiative-gain loop, which
         then corrupted the Pressure value used later in the same pass
         for the loss-side radiative trapping calculation. Renamed to `Pt`.
    """

    def update_state(US):
        Label = US['label']
        kind = US['kind']
        US['GainTerms'] = []
        US['LossTerms'] = []

        # ---- Collisional production (excitation from lower partners) ----
        Sum = 0
        for LS in US['Electron Impact CrossSections']['Products']:
            partner = LS['partner_label']
            for p in InputModel.values():
                if p['label'] == partner:
                    LowerStateDensity = p['density_m^-3']
            Sum = Sum + LS['Rate'] * LowerStateDensity
        CollisionalSum = Sum * Ne

        # ---- Superelastic production (de-excitation from states above) ----
        SuperelasticGain = 0
        for SE in US.get('Superelastic', []):
            if SE['direction'] == 'gain':
                partner = SE['partner']
                for p in InputModel.values():
                    if p['label'] == partner:
                        UpperStateDensity = p['density_m^-3']
                SuperelasticGain += SE['coeff'] * UpperStateDensity
        SuperelasticGain *= Ne

        # ---- Radiative gain (cascades from above, with trapping) ----
        Sum = 0
        LevelDensity = US['density_m^-3']
        for Rad in US['RadiativeDecay']:
            if Rad['direction'] == 'gain':
                partner = Rad['partner']
                A = Rad['coeff']
                for p in InputModel.values():
                    if p['label'] == partner:
                        UpperStateDensity = p['density_m^-3']
                        UpperState = p
                if UpperStateDensity == 0:
                    Eta = 1
                else:
                    Pt = he.Volume2Torr(LevelDensity, T)
                    a, tau = he.FindTauInModel(UpperState, US, Rad, Pt, T, R)
                    Eta = GetEta(interp, tau, a)
                Sum = A * Eta * UpperStateDensity + Sum
        RadiativeGainSum = Sum

        # ---- Transfer from other levels by collisions with ground-state Ar ----
        AtomGain = Ng * sum(AT['coeff'] * InputModel[AT['partner']]['density_m^-3']
                            for AT in US.get('AtomTransfer', []) if AT['direction'] == 'gain')

        Prod = CollisionalSum + RadiativeGainSum + SuperelasticGain + AtomGain
        US['Production_m^-3s^-1'] = Prod

        # ---- Ionization loss ----
        if Label == 'ground':
            IonLoss = 0
        else:
            IonLossRate = US['Ionization Data']['Rate_cm^3']
            IonLoss = IonLossRate * Ne

        # ---- Superelastic loss (US de-excites down to states below) ----
        SuperelasticLoss = sum(SE['coeff'] for SE in US.get('Superelastic', [])
                               if SE['direction'] == 'loss')
        SuperelasticLoss *= Ne

        # ---- Collisional loss (excitation OUT to higher partners) ----
        # Fixed: no longer multiplied by the destination state's density.
        CollisionalLoss = Ne * sum(
            LS['Rate'] for LS in US['Electron Impact CrossSections']['Reactants']
        )

        # ---- Radiative loss (metastables have none: forbidden transitions) ----
        RadiativeLossSum = 0
        if kind != 'metastable':
            Sum = 0
            for Rad in US['RadiativeDecay']:
                if Rad['direction'] == 'loss':
                    partner = Rad['partner']
                    A = Rad['coeff']
                    for p in InputModel.values():
                        if p['label'] == partner:
                            LowerStateDensity = p['density_m^-3']
                            LowerState = p
                    if LowerStateDensity == 0:
                        Eta = 1
                    else:
                        Pt = he.Volume2Torr(LowerStateDensity, T)
                        a, tau = he.FindTauInModel(US, LowerState, Rad, Pt, T, R)
                        Eta = GetEta(interp, tau, a)
                    Sum = A * Eta + Sum
            RadiativeLossSum = Sum + US.get('A_untracked_loss', 0.0)

        # ---- Diffusion loss (metastables only) ----
        DiffusionLoss = 1/(US['DiffusionLoss']) if kind == 'metastable' else 0
        # Energy pooling between pairs of 4s levels, Ar(1s_j) + Ar(1s_l) -> Ar + Ar+ + e (or Ar2+ + e):
        # loss k_p * n_4s on each of the four 4s levels, n_4s the summed density of all four,
        # Ferreira, Loureiro & Ricard, J. Appl. Phys. 57, 82 (1985), Eq. (8)
        k_met = 6.4e-16   # m^3/s, Ferreira et al. (Ref 31 in Bogaerts)
        n_4s = sum(p['density_m^-3'] for p in InputModel.values() if p['kind'] in ('metastable', 'resonant'))
        MetLoss = k_met * n_4s if kind in ('metastable', 'resonant') else 0
        # Two- and three-body quenching by ground-state Ar (metastables only)
        QuenchLoss = GroundQuenchingLoss(Ng) if kind == 'metastable' else 0
        # Quenching by an admixed gas, set by the caller (MainFileWithNitrogen); 0 in pure Ar
        QuenchLoss += US.get('GasQuenching_s^-1', 0.0)
        # Transfer to other levels by collisions with ground-state Ar (AttachAtomTransfer)
        AtomLoss = Ng * sum(AT['coeff'] for AT in US.get('AtomTransfer', []) if AT['direction'] == 'loss')
        Loss = (IonLoss + RadiativeLossSum + SuperelasticLoss + CollisionalLoss + DiffusionLoss
                + MetLoss + QuenchLoss + AtomLoss)
        US['Loss_s^-1'] = Loss
        US['density_m^-3'] = Prod / Loss

    # Preserve original update order: metastable -> resonant -> normal
    for US in InputModel.values():
        if US['kind'] == 'metastable':
            update_state(US)
    for US in InputModel.values():
        if US['kind'] == 'resonant':
            update_state(US)
    for US in InputModel.values():
        if US['kind'] == 'normal':
            update_state(US)

    return InputModel


def SolveDirect(InputModel, Ne, T, R, interp, trap_lines='all', trap_density=None):
    """
    The balance equations of all excited levels as one linear system,
        sum_j G_ij n_j - L_i n_i = -S_i ,
    with the same gain and loss terms as SolveLabelEquation (S_i is the
    production from the fixed ground-state density). Solved directly, so it
    does not slow down when the levels are strongly coupled (high Ne), where
    the level-by-level iteration converges very slowly.

    The escape factors are taken from the current densities; CRModel repeats
    the solve until they are consistent. The 4s energy-pooling loss
    k_met * n_4s * n_i (4s levels only) is quadratic in the densities, so it enters as a
    Newton step (its derivative is included in the matrix) - a plain
    fixed-point update of n_meta oscillates slowly when this loss dominates.

    trap_lines : 'all'    - escape factor on every line, lower level as absorber
                 'ground' - escape factor only on lines to the ground state, as
                            in Bogaerts, Gijbels & Vlcek, J. Appl. Phys. 84, 121 (1998)
    trap_density : {label: density [m^-3]} of the absorbing (lower) excited levels used for the
                   escape factors instead of the current densities, i.e. trapping frozen at a
                   reference solution (None = self-consistent)
    """
    Levels = [lbl for lbl, s in InputModel.items() if s['kind'] != 'ground']
    idx = {lbl: i for i, lbl in enumerate(Levels)}
    M = np.zeros((len(Levels), len(Levels)))
    S = np.zeros(len(Levels))
    k_met = 6.4e-16   # m^3/s, 4s energy pooling, Ferreira et al. (Ref 31 in Bogaerts)
    n_4s = sum(p['density_m^-3'] for p in InputModel.values() if p['kind'] in ('metastable', 'resonant'))
    Ng =next(p['density_m^-3'] for p in InputModel.values() if p['kind'] == 'ground')

    # Escape factors of every line at the current densities, with one interpolator call for
    # all lines (GetEta line by line took most of the solve time). A line's escape factor
    # depends on its upper and lower level and A, so the gain entry of the lower level gets
    # the value of the loss entry of the upper level.
    EtaOf, keys, tau_a, keys_g, tau_a_g = {}, [], [], [], []
    resonance = getattr(interp, 'resonance', interp)   # lines to the ground state (EscapeFactorInterpolator)
    for Upper in InputModel.values():
        for Rad in Upper['RadiativeDecay']:
            if Rad['direction'] != 'loss':
                continue
            Lower = InputModel[Rad['partner']]
            key = (Upper['label'], Lower['label'], Rad['coeff'])
            n_low = Lower['density_m^-3']
            if trap_density is not None and Lower['kind'] != 'ground':
                n_low = trap_density.get(Lower['label'], n_low)
            if (trap_lines == 'ground' and Lower['kind'] != 'ground') or n_low == 0:
                EtaOf[key] = 1.0
                continue
            Pt = he.Volume2Torr(n_low, T)
            a, tau = he.FindTauInModel(Upper, Lower, Rad, Pt, T, R)
            if Lower['kind'] == 'ground':
                keys_g.append(key)
                tau_a_g.append((np.log10(tau), a))
            else:
                keys.append(key)
                tau_a.append((np.log10(tau), a))
    if keys:
        EtaOf.update(zip(keys, 10**interp(np.array(tau_a, dtype=float))))
    if keys_g:
        EtaOf.update(zip(keys_g, 10**resonance(np.array(tau_a_g, dtype=float))))

    def Eta(Upper, Lower, Rad):
        return float(EtaOf[Upper['label'], Lower['label'], Rad['coeff']])

    def Gain(i, partner, rate):
        if partner in idx:
            M[i, idx[partner]] += rate
        else:                                  # ground state, density fixed
            S[i] += rate * InputModel[partner]['density_m^-3']

    for lbl in Levels:
        US, i = InputModel[lbl], idx[lbl]
        # ---- production: excitation from below, superelastic from above, cascades ----
        for LS in US['Electron Impact CrossSections']['Products']:
            Gain(i, LS['partner_label'], Ne * LS['Rate'])
        for SE in US.get('Superelastic', []):
            if SE['direction'] == 'gain':
                Gain(i, SE['partner'], Ne * SE['coeff'])
        for Rad in US['RadiativeDecay']:
            if Rad['direction'] == 'gain':
                Gain(i, Rad['partner'], Rad['coeff'] * Eta(InputModel[Rad['partner']], US, Rad))
        for AT in US.get('AtomTransfer', []):      # collisions with ground-state Ar
            if AT['direction'] == 'gain':
                Gain(i, AT['partner'], Ng * AT['coeff'])

        # ---- loss frequency (same terms as SolveLabelEquation) ----
        IonLoss = US['Ionization Data']['Rate_cm^3'] * Ne
        SuperelasticLoss = Ne * sum(SE['coeff'] for SE in US.get('Superelastic', [])
                                    if SE['direction'] == 'loss')
        CollisionalLoss = Ne * sum(LS['Rate'] for LS in US['Electron Impact CrossSections']['Reactants'])
        RadiativeLossSum = 0
        if US['kind'] != 'metastable':
            RadiativeLossSum = sum(Rad['coeff'] * Eta(US, InputModel[Rad['partner']], Rad)
                                   for Rad in US['RadiativeDecay'] if Rad['direction'] == 'loss')
            RadiativeLossSum += US.get('A_untracked_loss', 0.0)
        DiffusionLoss = 1/(US['DiffusionLoss']) if US['kind'] == 'metastable' else 0
        MetLoss = k_met * n_4s if US['kind'] in ('metastable', 'resonant') else 0
        QuenchLoss = GroundQuenchingLoss(Ng) if US['kind'] == 'metastable' else 0
        QuenchLoss += US.get('GasQuenching_s^-1', 0.0)     # admixed gas (MainFileWithNitrogen)
        AtomLoss = Ng * sum(AT['coeff'] for AT in US.get('AtomTransfer', []) if AT['direction'] == 'loss')
        US['Loss_s^-1'] =(IonLoss + RadiativeLossSum + SuperelasticLoss + CollisionalLoss
                           + DiffusionLoss + MetLoss + QuenchLoss + AtomLoss)
        M[i, i] -= US['Loss_s^-1']

    # Newton step on F(n) = M n + S, where d/dn_m of -k_met n_4s n_i adds -k_met n_i to
    # column m for each 4s level m, on the 4s rows i only (M already holds -k_met n_4s)
    n_old = np.array([InputModel[lbl]['density_m^-3'] for lbl in Levels], dtype=float)
    pooled = np.array([InputModel[lbl]['kind'] in ('metastable', 'resonant') for lbl in Levels])
    J = M.copy()
    for lbl in Levels:
        if InputModel[lbl]['kind'] in ('metastable', 'resonant'):
            J[:, idx[lbl]] -= k_met * n_old * pooled
    n = n_old - np.linalg.solve(J, M @ n_old + S)
    for lbl in Levels:
        US = InputModel[lbl]
        US['density_m^-3'] = n[idx[lbl]]
        US['Production_m^-3s^-1'] = US['Loss_s^-1'] * n[idx[lbl]]
    return InputModel


#%% Initilize State Densities and Extract Data
def InitializeStateDensities(ModelData,P,T):
    for US in ModelData.values():
        Label = US['label']
        if Label == 'ground':
            US['density_m^-3'] = he.Torr2Volume(P,T)
        else:
            US['density_m^-3'] = 0
    return ModelData


# Define state densities and emission intensities after CR model 

def ExtractStateDensities(ModelData):
    """Extract final state densities as a simple array"""
    SD = []
    for state_label, state_data in ModelData.items():
        density = state_data['density_m^-3']
        SD.append({
            'label': state_data['label'],
            'density_m^-3': density
        })
    return SD
def ExtractEmissionIntensities(ModelData):
    """Calculate emission intensities: density * radiative rate"""
    EI = []
    
    for state_label, state_data in ModelData.items():
        density = state_data['density_m^-3']
        label = state_data['label']
        
        # Skip if no radiative decay
        if 'RadiativeDecay' not in state_data:
            continue
        
        # Skip resonant states
        elif state_data['kind'] == 'resonant':  # ← Changed from 'else if' to 'elif'
            continue 
        
        for rad_decay in state_data['RadiativeDecay']:
            # Only count loss processes (emission FROM this state)
            if rad_decay.get('direction') != 'loss':
                continue
            if rad_decay['partner'] == 'ground':
                continue          # VUV, trapped, not observed
            A_coeff = rad_decay['coeff']  # Einstein A coefficient (s^-1)
            partner = rad_decay['partner']
            wavelength = rad_decay['wavelength_nm']  # Now works!
            
            # Emission intensity = population * transition rate
            intensity = density * A_coeff
            
            EI.append({
                'from_state': label,
                'to_state': partner,
                'A_coeff': A_coeff,
                'density': density,
                'intensity': intensity,
                'Wavelength': wavelength
            })
    
    return EI
def PrintStateTable(SD):
    """Print state densities in a nice table"""
    print("\n" + "="*60)
    print("FINAL STATE DENSITIES")
    print("="*60)
    print(f"{'State':<15} {'Density (m^-3)':<25}")
    print("-"*60)
    for item in SD:
        print(f"{item['label']:<15} {item['density_m^-3']:.6e}")
    print("="*60 + "\n")

def PrintEmissionTable(EI, top_n=20):
    """Print emission intensities sorted by intensity"""
    print("\n" + "="*80)
    print("TOP EMISSION LINES")
    print("="*80)
    print(f"{'From':<12} {'To':<12} {'A coeff (s^-1)':<18} {'Intensity (arb.)':<20}")
    print("-"*80)
    
    # Sort by intensity
    EI_sorted = sorted(EI, key=lambda x: x['intensity'], reverse=True)
    
    for i, item in enumerate(EI_sorted[:top_n]):
        print(f"{item['from_state']:<12} {item['to_state']:<12} "
              f"{item['A_coeff']:.6e}          {item['intensity']:.6e}")
    print("="*80 + "\n")

def PlotStateDensities(SD):
    """Bar plot of state densities"""
    import matplotlib.pyplot as plt
    
    labels = [item['label'] for item in SD]
    densities = [item['density_m^-3'] for item in SD]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(labels, densities, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Atomic State', fontsize=12)
    ax.set_ylabel('Density (m$^{-3}$)', fontsize=12)
    ax.set_title('Final State Densities', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'state_densities.png'), dpi=300)
    plt.show()

def PlotEmissionIntensities(EI, top_n=30):
    """Stem plot of top emission intensities"""
    import matplotlib.pyplot as plt
    
    # Sort and take top N
    EI_sorted = sorted(EI, key=lambda x: x['intensity'], reverse=True)
    EI_top = EI_sorted[:top_n]
    
    # Create labels for x-axis
    labels = [f"{item['from_state']}\n→{item['to_state']}" for item in EI_top]
    Wavelength = [item['Wavelength'] for item in EI_top]
    intensities = [item['intensity'] for item in EI_top]
    x_pos = range(len(labels))
    
    fig, ax = plt.subplots(figsize=(16, 6))
    markerline, stemlines, baseline = ax.stem(Wavelength, intensities, basefmt=' ')
    markerline.set_markerfacecolor('red')
    markerline.set_markeredgecolor('darkred')
    markerline.set_markersize(8)
    stemlines.set_linewidth(2)
    stemlines.set_color('darkred')
    
    ax.set_xlabel('Wavlength / nm', fontsize=12)
    ax.set_ylabel('Emission Intensity (arb. units)', fontsize=12)
    ax.set_title(f'Top {top_n} Emission Lines', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'emission_intensities.png'), dpi=300)
    plt.show()

def PlotEEDFs(EEDFs):
    """EEPF of every EEDF in the sweep; numerical ones get the Maxwellian with the same <E> dashed"""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Blues(np.linspace(0.35, 1.0, len(EEDFs)))   # ordered sweep: light -> dark
    for eedf, color in zip(EEDFs, colors):
        ax.semilogy(eedf['E'], eedf['EEPF'], color=color, linewidth=1.5, label=eedf['label'])
        if eedf['Te'] is None:
            Maxwell = he.MaxwellianEEDF(eedf['Te_eff'], eedf['E'])
            ax.semilogy(eedf['E'], Maxwell['EEPF'], '--', color=color,
                        linewidth=1, alpha=0.6)
    if any(eedf['Te'] is None for eedf in EEDFs):
        ax.plot([], [], 'k--', linewidth=1, label=r'Maxwellian, same $\langle\varepsilon\rangle$')
    peak = max(eedf['EEPF'].max() for eedf in EEDFs)
    ax.set_ylim(peak * 1e-10, peak * 2)
    ax.set_xlabel('Electron energy [eV]', fontsize=12)
    ax.set_ylabel(r'EEPF [eV$^{-3/2}$]', fontsize=12)
    ax.set_title('Electron energy probability functions', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eedfs.png'), dpi=300)
    plt.show()


#%% Cr Model
# Simple CR model that does not test for convergence
# def CRModel(ModelData, Te, Ne, P, T, R, interp):
#     # Calculate reaction rates
#     ModelData = CreateExcitationReactionRates(ModelData, Te)
#     ModelData = CreateIonizationReactionRates(ModelData, Te)
#     ModelData = CreateSuperelasticRates(ModelData, t) 
#     # Get ground state density
#     Ng = he.Torr2Volume(P,T)
    
#     # Initialize densities
#     Data = InitializeStateDensities(ModelData, P, T)
    
#     # Iterative loop
#     for iteration in range(20):
#         Data = SolveLabelEquation(Data, Ng, Ne, P, T, R, interp)
        
#         # TODO: Add residual check here
#         # if convergence_criterion < tolerance:
#         #     break

     
#     SD = ExtractStateDensities(Data)
#     EI = ExtractEmissionIntensities(Data)
#     PrintEmissionTable(EI)
#     return Data,SD,EI  # Return only Data, not undefined variables



def CRModel(ModelData, eedf, Ne, P, T, R, interp,
            max_iter=500, tol=1e-6, relax=1.0, verbose=False,
            solver='direct', trap_lines='all', compute_rates=True, atom_transfer=True,
            trap_density=None):
    """
    Solve the CR balance to self-consistency.

    eedf   : electron energy distribution for all electron-impact rates -
             Te [eV] for a Maxwell-Boltzmann EEDF, or an EEDF dict
             (he.MaxwellianEEDF, he.TabulatedEEDF, he.ImportMultiBoltEEDFs)
    tol    : convergence on max relative change in density between sweeps
    relax  : under-relaxation factor. 1.0 = no relaxation. Drop to ~0.5 if
             the metastable-metastable term makes it oscillate.
    solver : 'direct'       - linear solve of all levels at once (SolveDirect),
                              repeated only to update escape factors and the
                              metastable-collision loss
             'gauss-seidel' - level-by-level iteration (SolveLabelEquation);
                              slow to converge at high Ne
    trap_lines : 'all' or 'ground', see SolveDirect ('direct' solver only)
    compute_rates : False reuses the electron-impact rates already in ModelData, i.e. from a
             previous call with the same EEDF (a sweep over Ne): they depend on the EEDF only
    atom_transfer : population transfer 2p <-> 2p and 2p -> 1s by collisions with ground-state
             Ar (AttachAtomTransfer, Zhu & Pu 2010); False leaves it out
    trap_density : fixed absorber densities for the escape factors, see SolveDirect
    """
    if solver == 'gauss-seidel' and (trap_lines != 'all' or trap_density is not None):
        raise ValueError("trap_lines='ground' and trap_density need solver='direct'")
    eedf = he.AsEEDF(eedf)
    if compute_rates:
        ModelData = CreateExcitationReactionRates(ModelData, eedf)
        ModelData = CreateIonizationReactionRates(ModelData, eedf)
        ModelData = CreateSuperelasticRates(ModelData, eedf)   # was `t` - global leak
    ModelData = AttachAtomTransfer(ModelData, T, atom_transfer)

    Ng = he.Torr2Volume(P, T)
    Data = InitializeStateDensities(ModelData, P, T)

    labels = [lbl for lbl, s in Data.items() if s['kind'] != 'ground']
    converged, residual = False, np.inf

    for iteration in range(1, max_iter + 1):
        old = {lbl: Data[lbl]['density_m^-3'] for lbl in labels}

        if solver == 'direct':
            Data = SolveDirect(Data, Ne, T, R, interp, trap_lines, trap_density)
        else:
            Data = SolveLabelEquation(Data, Ng, Ne, P, T, R, interp)

        if relax < 1.0:
            for lbl in labels:
                Data[lbl]['density_m^-3'] = (
                    relax * Data[lbl]['density_m^-3'] + (1 - relax) * old[lbl])

        # max relative change, ignoring levels that are still ~zero
        residual = 0.0
        worst = None
        for lbl in labels:
            new = Data[lbl]['density_m^-3']
            ref = max(abs(new), abs(old[lbl]))
            if ref <= 0:
                continue
            r = abs(new - old[lbl]) / ref
            if r > residual:
                residual, worst = r, lbl

        if verbose:
            print(f"  iter {iteration:3d}  residual {residual:.3e}  ({worst})")

        if residual < tol:
            converged = True
            break

    if not converged:
        print(f"WARNING: CRModel did not converge in {max_iter} sweeps "
              f"(residual {residual:.3e}, worst level {worst}) "
              f"at {eedf['label']}, Ne={Ne:.2e}")

    SD = ExtractStateDensities(Data)
    EI = ExtractEmissionIntensities(Data)
    solver = {'iterations': iteration, 'residual': residual,
              'converged': converged}
    return Data, SD, EI, solver



#%% Post Processing Functions 



def GetLineIntensity(EI, from_label, to_label=None):
    """
    Pull intensity for a specific line (or all lines from a state) out of EI.
 
    from_label : upper state label, e.g. '4p6'
    to_label   : lower state label, e.g. '4s2'. If None, sums ALL
                 decay channels out of from_label (total state intensity).
    """
    candidates = [item for item in EI if item['from_state'] == from_label]
    if to_label is not None:
        candidates = [c for c in candidates if c['to_state'] == to_label]
    if not candidates:
        return 0.0
    return sum(c['intensity'] for c in candidates)


from scipy.special import wofz

def voigt_broaden(wl, intensity, delG, delL, npts=4000, pad=5.0):
    """
    Broaden a stick spectrum with a Voigt profile.

    wl, intensity : 1-D arrays of line centers (nm) and stick areas (arb.)
    delG, delL    : Gaussian and Lorentzian FWHM (nm)
    Returns x_out (nm), y_out (intensity per nm)
    """
    wl = np.asarray(wl, float)
    intensity = np.asarray(intensity, float)

    sigma = delG / (2*np.sqrt(2*np.log(2)))   # Gaussian std dev
    gamma = delL / 2.0                        # Lorentzian HWHM

    w = pad * max(delG, delL)
    x_out = np.linspace(wl.min() - w, wl.max() + w, npts)

    # (npts, nlines) offset matrix -> unit-area Voigt for each line
    dx = x_out[:, None] - wl[None, :]
    z = (dx + 1j*gamma) / (sigma*np.sqrt(2))
    V = np.real(wofz(z)) / (sigma*np.sqrt(2*np.pi))   # already area-normalized

    y_out = V @ intensity
    return x_out, y_out


def PlotSpectrum(EI, slit_file=he.SLIT_FUNCTION_FILE, show_sticks=True):
    wl = np.array([item['Wavelength'] for item in EI])
    I  = np.array([item['intensity']  for item in EI])

    good = np.isfinite(wl) & np.isfinite(I) & (I > 0)
    wl, I = wl[good], I[good]

    x, y = he.BroadenWithSlit(wl, I, slit_file)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, y, color='darkred', linewidth=1.2, label='Slit-function broadened')
    if show_sticks:
        # scale sticks to peak height for visual comparison only
        ax.vlines(wl, 0, I * y.max() / I.max(), color='steelblue',
                  alpha=0.4, linewidth=1, label='stick spectrum')
    ax.set_xlabel('Wavelength / nm', fontsize=12)
    ax.set_ylabel('Intensity (arb. units nm$^{-1}$)', fontsize=12)
    ax.set_title('Synthetic argon emission spectrum', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'synthetic_spectrum.png'), dpi=300)
    plt.show()
    return x, y
