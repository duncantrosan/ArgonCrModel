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
Pressure = 1 # Pressure in Torr 
Tg = 300 # Temperature in K
Te = np.linspace(0.5,6,10) # Electron Temperature in ev
Ne = np.logspace(6,18,10) # electron Density in m^-3
R = 4/100 # Radius in meters


## Import preliminary structured data 
ModelData,RadiationTrappingMatrix = he.GetData()
for label in ['4s2', '4s4', '4p6', '4p10']:
    md = ModelData[label]
    prods = [r['partner_label'] for r in md['Electron Impact CrossSections']['Products']]
    reacs = [r['partner_label'] for r in md['Electron Impact CrossSections']['Reactants']]
    print(f"{label:6s} | excited FROM: {prods} | excites TO: {reacs}")

def GetEta(interp,tau,a):
    return 10**interp(((np.log10(tau), a)))

def CreateSuperelasticRates(ModelData, Te):
    """
    Compute electron-impact de-excitation (superelastic) rate coefficients
    via detailed balance from the already-computed excitation rates.
 
    MUST be called AFTER CreateExcitationReactionRates(ModelData, Te) in the
    same iteration, since it reads the 'Rate' field that function fills in.
 
    Adds a 'Superelastic' list to every level, mirroring the structure of
    'RadiativeDecay':
        {'partner': label, 'coeff': k_deexc (m^3/s), 'process': 'superelastic',
         'direction': 'gain'|'loss'}
 
    - 'loss' entries go on the UPPER state (it de-excites away, needs *Ne
       and the upper state's own density when used in the loss sum)
    - 'gain' entries go on the LOWER state (it gets repopulated, needs
       *Ne and the UPPER state's density when used in the production sum)
 
    Note: this uses the standard Maxwellian detailed-balance relation.
    If your EEPF (Te2EEPF) is non-Maxwellian, this is an approximation -
    strictly correct detailed balance requires balancing against the same
    EEPF used for the forward rate, but this is the standard first-order
    treatment used in most CR models.
    """
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
            if not k_exc:
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
 
            k_deexc = k_exc * (g_lower / g_upper) * np.exp(DeltaE / Te)
 
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

def CreateExcitationReactionRates(ModelData, Te):
    EEPF, E = he.Te2EEPF(Te)
    m_e = 9.10938e-31       # kg
    e_charge = 1.60217e-19  # J per eV

    prefactor = np.sqrt(2 * e_charge**3 / m_e)

    for MD in ModelData.values():
        for direction in ('Reactants', 'Products'):
            for Reac in MD['Electron Impact CrossSections'][direction]:
                Threshold  = Reac['threshold_eV']
                DataEnergy = Reac['energy_eV']
                DataCS     = Reac['cross_section']

                InterpolatedCS = np.interp(E, DataEnergy, DataCS, left=0.0, right=0.0)
                InterpolatedCS[E <= Threshold] = 0.0

                Integrand = np.sqrt(E) * EEPF * InterpolatedCS
                Rate = prefactor * np.trapezoid(Integrand, E)

                Reac['Rate'] = Rate   # overwritten fresh each call to this function

    return ModelData

def CreateIonizationReactionRates(ModelData, Te):
    EEPF, E = he.Te2EEPF(Te)
    m_e = 9.10938e-31       # kg
    e_charge = 1.60217e-19  # J per eV
    prefactor = np.sqrt(2 * e_charge**3 / m_e)
    
    # Go through every level and convert ionization cross-section to rate
    for MD in ModelData.values():
        label = MD['label']
        print(f"Processing ionization for {label}")
        
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
        
        # Interpolate cross-section to EEPF energy grid
        InterpolatedCS = np.interp(E, DataEnergy, DataCS, left=0.0, right=0.0)
        InterpolatedCS[E <= Threshold] = 0.0
        
        # Calculate rate coefficient
        Integrand = np.sqrt(E) * EEPF * InterpolatedCS
        Rate = prefactor * np.trapezoid(Integrand, E)
        
        # Store the rate
        IonData['Rate_cm^3'] = Rate
        print(f"  Ionization rate for {label}: {Rate:.6e} cm³/s")
    
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

##########################################################################
##########################################################################
#### Solve Equations! 


def NormalLabelEquation(InputModel,Ng,Ne,P,T,R,interp):
    
    # Production of 4p, 3d, and 5s States
    for US in InputModel.values(): # Go throught Input model for the upper states
        Label = US['label']
        kind = US['kind']
        if kind == 'normal':
            US['GainTerms'] = []
            US['LossTerms'] = []
            # Create collisional production
            Sum = 0
            for LS in US['Electron Impact CrossSections']['Products']:
                partner = LS['partner_label']
                for p in InputModel.values():
                    if p['label'] == partner:
                        LowerStateDensity = p['density_m^-3']
                Sum = Sum + LS['Rate']*LowerStateDensity
            CollisionalSum = Sum *Ne
            
            # --- Superelastic production (gain from states above de-exciting into US) ---
            SuperelasticGain = 0
            for SE in US.get('Superelastic', []):
                if SE['direction'] == 'gain':
                    partner = SE['partner']
                    for p in InputModel.values():
                        if p['label'] == partner:
                            UpperStateDensity = p['density_m^-3']
                    SuperelasticGain += SE['coeff'] * UpperStateDensity
            SuperelasticGain *= Ne
            
            # Find Radiation Gain
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
                    if UpperStateDensity == 0 :
                        Eta = 1
                    else:
                        P = he.Volume2Torr(LevelDensity,T)
                        a,tau = he.FindTauInModel(UpperState,US, Rad, P, T, R)
                        Eta = GetEta(interp, tau, a)
                    Sum = A*Eta*UpperStateDensity + Sum
            RadiativeGainSum = Sum
            Prod = CollisionalSum + RadiativeGainSum + SuperelasticGain
                    
            US['Production_m^-3s^-1'] = Prod
            #### Create Loss Mechanisms Loss terms are without updated density terms
            # Ionization Loss 
            if Label == 'ground':
                IonLoss = 0
            else:
                IonLossRate = US['Ionization Data']['Rate_cm^3']


                IonLoss = IonLossRate*Ne
            
            # --- Superelastic loss (US de-excites down to states below) ---
            SuperelasticLoss = sum(SE['coeff'] for SE in US.get('Superelastic', [])
                                   if SE['direction'] == 'loss')
            SuperelasticLoss *= Ne
            # Loss radiative Terms 
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
                        P = he.Volume2Torr(LowerStateDensity,T)
                        a,tau = he.FindTauInModel(US,LowerState, Rad, P, T, R)
                        Eta = GetEta(interp, tau, a)
                    Sum = A*Eta + Sum
            RadiativeLossSum = Sum
            Loss = IonLoss + RadiativeLossSum + SuperelasticLoss
            US['Loss_s^-1'] = Loss
            StateDensity = Prod/Loss
            US['density_m^-3'] = StateDensity
        
    return InputModel


def MetaStableLabelEquation(InputModel,Ng,Ne,P,T,R,interp):
    
    # Production of 4s2 and 4s4 States
    for US in InputModel.values(): # Go throught Input model for the upper states
        Label = US['label']
        kind = US['kind']
        if kind == 'metastable':
            US['GainTerms'] = []
            US['LossTerms'] = []
            # Create collisional production
            Sum = 0
            for LS in US['Electron Impact CrossSections']['Products']:
                partner = LS['partner_label']
                for p in InputModel.values():
                    if p['label'] == partner:
                        LowerStateDensity = p['density_m^-3']
                Sum = Sum + LS['Rate']*LowerStateDensity
            CollisionalSum = Sum *Ne
            # --- Superelastic production (gain from states above de-exciting into US) ---
            SuperelasticGain = 0
            for SE in US.get('Superelastic', []):
                if SE['direction'] == 'gain':
                    partner = SE['partner']
                    for p in InputModel.values():
                        if p['label'] == partner:
                            UpperStateDensity = p['density_m^-3']
                    SuperelasticGain += SE['coeff'] * UpperStateDensity
            SuperelasticGain *= Ne
            # Find Radiation Gain
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
                    if UpperStateDensity == 0 :
                        Eta = 1
                    else:
                        P = he.Volume2Torr(LevelDensity,T)
                        a,tau = he.FindTauInModel(UpperState,US, Rad, P, T, R)
                        Eta = GetEta(interp, tau, a)
                    Sum = A*Eta*UpperStateDensity + Sum
            RadiativeGainSum = Sum
            Prod = CollisionalSum + RadiativeGainSum + SuperelasticGain
                    
            US['Production_m^-3s^-1'] = Prod
            #### Create Loss Mechanisms Loss terms are without updated density terms
            # Ionization Loss 
            if Label == 'ground':
                IonLoss = 0
            else:
                IonLossRate = US['Ionization Data']['Rate_cm^3']


                IonLoss = IonLossRate*Ne
                
            # --- Superelastic loss (US de-excites down to states below) ---
            SuperelasticLoss = sum(SE['coeff'] for SE in US.get('Superelastic', [])
                                   if SE['direction'] == 'loss')
            SuperelasticLoss *= Ne
            DiffusionLoss = US['DiffusionLoss']
            # Collisional Loss
            # Create collisional Loss Term
            Sum = 0
            for LS in US['Electron Impact CrossSections']['Reactants']:
                partner = LS['partner_label']
                for p in InputModel.values():
                    if p['label'] == partner:
                        LowerStateDensity = p['density_m^-3']
                Sum = Sum + LS['Rate']
            CollisionalLoss = Sum *Ne
            Loss = IonLoss  + DiffusionLoss + CollisionalLoss + SuperelasticLoss
            US['Loss_s^-1'] = Loss
            StateDensity = Prod/Loss
            US['density_m^-3'] = StateDensity
        
    return InputModel


def ResonantLabelEquation(InputModel,Ng,Ne,P,T,R,interp):
    
    # Production of 4s2 and 4s4 States
    for US in InputModel.values(): # Go throught Input model for the upper states
        Label = US['label']
        kind = US['kind']
        if kind == 'resonant':
            US['GainTerms'] = []
            US['LossTerms'] = []
            # Create collisional production
            Sum = 0
            for LS in US['Electron Impact CrossSections']['Products']:
                partner = LS['partner_label']
                for p in InputModel.values():
                    if p['label'] == partner:
                        LowerStateDensity = p['density_m^-3']
                Sum = Sum + LS['Rate']*LowerStateDensity
            CollisionalSum = Sum *Ne
            # --- Superelastic production (gain from states above de-exciting into US) ---
            SuperelasticGain = 0
            for SE in US.get('Superelastic', []):
                if SE['direction'] == 'gain':
                    partner = SE['partner']
                    for p in InputModel.values():
                        if p['label'] == partner:
                            UpperStateDensity = p['density_m^-3']
                    SuperelasticGain += SE['coeff'] * UpperStateDensity
            SuperelasticGain *= Ne
            
            
            # Find Radiation Gain
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
                    if UpperStateDensity == 0 :
                        Eta = 1
                    else:
                        P = he.Volume2Torr(LevelDensity,T)
                        a,tau = he.FindTauInModel(UpperState,US, Rad, P, T, R)
                        Eta = GetEta(interp, tau, a)
                    Sum = A*Eta*UpperStateDensity + Sum
            RadiativeGainSum = Sum
            Prod = CollisionalSum + RadiativeGainSum + SuperelasticGain
                    
            US['Production_m^-3s^-1'] = Prod
            #### Create Loss Mechanisms Loss terms are without updated density terms
            # Ionization Loss 
            if Label == 'ground':
                IonLoss = 0
            else:
                IonLossRate = US['Ionization Data']['Rate_cm^3']
                print(f'The Current level is {US}')
                print(IonLossRate)
                IonLoss = IonLossRate*Ne
            
            
            
            # --- Superelastic loss (US de-excites down to states below) ---
            SuperelasticLoss = sum(SE['coeff'] for SE in US.get('Superelastic', [])
                                   if SE['direction'] == 'loss')
            SuperelasticLoss *= Ne

            # Loss radiative with Trapping
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
                        P = he.Volume2Torr(LowerStateDensity,T)
                        a,tau = he.FindTauInModel(US,LowerState, Rad, P, T, R)
                        Eta = GetEta(interp, tau, a)
                    Sum = A*Eta + Sum
            RadiativeLossSum = Sum   

            # Collisional Loss
            # Create collisional Loss Term
            Sum = 0
            for LS in US['Electron Impact CrossSections']['Reactants']:
                partner = LS['partner_label']
                for p in InputModel.values():
                    if p['label'] == partner:
                        LowerStateDensity = p['density_m^-3']
                Sum = Sum + LS['Rate']
            CollisionalLoss = Sum *Ne
            Loss = IonLoss   + CollisionalLoss + RadiativeLossSum + SuperelasticLoss
            US['Loss_s^-1'] = Loss
            StateDensity = Prod/Loss
            US['density_m^-3'] = StateDensity
        
    return InputModel


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
            if rad_decay.get('direction') == 'loss':
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




def CRModel(ModelData, Te, Ne, P, T, R, interp):
    # Calculate reaction rates
    ModelData = CreateExcitationReactionRates(ModelData, Te)
    ModelData = CreateIonizationReactionRates(ModelData, Te)
    ModelData = CreateSuperelasticRates(ModelData, t) 
    # Get ground state density
    Ng = he.Torr2Volume(P,T)
    
    # Initialize densities
    Data = InitializeStateDensities(ModelData, P, T)
    
    # Iterative loop
    for iteration in range(20):
        Data = MetaStableLabelEquation(Data, Ng, Ne, P, T, R, interp)
        Data = ResonantLabelEquation(Data, Ng, Ne, P, T, R, interp)  # ← Fixed
        Data = NormalLabelEquation(Data, Ng, Ne, P, T, R, interp)
        
        # TODO: Add residual check here
        # if convergence_criterion < tolerance:
        #     break

     
    SD = ExtractStateDensities(Data)
    EI = ExtractEmissionIntensities(Data)
    PrintEmissionTable(EI)
    return Data,SD,EI  # Return only Data, not undefined variables


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




# Set up Model 
ModelData = he.AddDiffusionLoss(ModelData, Pressure, Tg, R)
# ModelData = AttachRadiationTrapping(ModelData,Pressure,Tg,R)
Ng = he.Torr2Volume(Pressure)
StateDensities = []
EmissionIntensities = []
# build once, reuse many times
tau_grid   = np.unique([d['Tau_R'] for d in RadiationTrappingMatrix])
shape_grid = np.unique([d['Shape'] for d in RadiationTrappingMatrix])
eta = np.array([d['EscapeFactor'][0] for d in RadiationTrappingMatrix]).reshape(len(tau_grid), len(shape_grid))

interp = RegularGridInterpolator(
    (np.log10(tau_grid), shape_grid),
    np.log10(eta),                    # eta spans ~4 decades, so interpolate its log
    bounds_error=False, fill_value=None
)

Ratio_a = np.zeros((len(Ne), len(Te)))   # 4p'[1/2]0 / 4p[1/2]0
Ratio_b = np.zeros((len(Ne), len(Te)))   # 4d[3/2]°  / 4p[1/2]0  (needs 4d added)


for j,t in enumerate(Te) :
    for i,n in enumerate(Ne):
        print(t)
        print(n)
        UpdatedModel = CreateExcitationReactionRates(ModelData, t)
        UpdatedModel = CreateIonizationReactionRates(ModelData, t)
        Data,SD,EI = CRModel(UpdatedModel,t,n,Pressure,Tg,R,interp)
        # StateDensities.append(SD)
        # EmissionIntensities.append(EI)
        # for Reac in EI.values():
        #     I750 = 
        #     I811
        #     I
        
        I_4p6  = GetLineIntensity(EI, '4p6')    # 4p[1/2]0   (unprimed)
        I_4p10 = GetLineIntensity(EI, '4p10')   # 4p'[1/2]0  (primed)
        # I_4d   = GetLineIntensity(EI, '4d_LABEL')  # once you add 4d states

        Ratio_a[i,j] = I_4p10 / I_4p6 if I_4p6 != 0 else np.nan
        # Ratio_b[i, j] = I_4d / I_4p6 if I_4p6 != 0 else np.nan
        
    PlotEmissionIntensities(EI,1000)
    PlotStateDensities(SD)
    


import matplotlib.pyplot as plt
# ------------------------------------------------------------------
# Plot (a): 4p'[1/2]0 / 4p[1/2]0
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
TeGrid, NeGrid = np.meshgrid(Te, Ne / 1e11)  # match paper's axis units
CS = ax.contour(TeGrid, NeGrid, Ratio_a, levels=12, colors='black')
ax.clabel(CS, inline=True, fontsize=9)
ax.set_xlabel(r'$T_e$ [eV]')
ax.set_ylabel(r'$N_e$ [$10^{11}$ cm$^{-3}$]')
ax.set_title("4p'[1/2]$_0$ / 4p[1/2]$_0$")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ratio_4pprime_4p.png'), dpi=300)
plt.show()


