# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:30:28 2026

@author: dptro
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
Te = np.linspace(2,8,20) # Electron Temperature in ev
Ne = np.linspace(1,100,10)*10**17 # electron Density in m^-3
R = 4/100 # Radius in meters


## Import preliminary structured data 
ModelData,RadiationTrappingMatrix = he.GetData()

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

    prefactor = np.sqrt(2 * e_charge / m_e)

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
        
        # Interpolate cross-section to EEPF energy grid
        InterpolatedCS = np.interp(E, DataEnergy, DataCS, left=0.0, right=0.0)
        InterpolatedCS[E <= Threshold] = 0.0
        
        # Calculate rate coefficient
        Integrand = np.sqrt(E) * EEPF * InterpolatedCS
        Rate = prefactor * np.trapezoid(Integrand, E)
        
        # Store the rate
        IonData['Rate_cm^3'] = Rate

    
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

        Prod = CollisionalSum + RadiativeGainSum + SuperelasticGain
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
        # Collisional Mixing between metastables 
        k_met = 6.4e-16   # m^3/s, Ferreira et al. (Ref 31 in Bogaerts)
        n_meta = sum(p['density_m^-3'] for p in InputModel.values()
                     if p['kind'] == 'metastable')
        MetLoss = k_met * n_meta
        Loss = IonLoss + RadiativeLossSum + SuperelasticLoss + CollisionalLoss + DiffusionLoss + MetLoss
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



def CRModel(ModelData, Te, Ne, P, T, R, interp,
            max_iter=500, tol=1e-6, relax=1.0, verbose=False):
    """
    Solve the CR balance to self-consistency.

    tol    : convergence on max relative change in density between sweeps
    relax  : under-relaxation factor. 1.0 = plain Gauss-Seidel. Drop to
             ~0.5 if the metastable-metastable term makes it oscillate.
    """
    ModelData = CreateExcitationReactionRates(ModelData, Te)
    ModelData = CreateIonizationReactionRates(ModelData, Te)
    ModelData = CreateSuperelasticRates(ModelData, Te)   # was `t` - global leak

    Ng = he.Torr2Volume(P, T)
    Data = InitializeStateDensities(ModelData, P, T)

    labels = [lbl for lbl, s in Data.items() if s['kind'] != 'ground']
    converged, residual = False, np.inf

    for iteration in range(1, max_iter + 1):
        old = {lbl: Data[lbl]['density_m^-3'] for lbl in labels}

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
              f"at Te={Te:.2f} eV, Ne={Ne:.2e}")

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


def PlotSpectrum(EI, delG=0.02, delL=0.005, npts=8000, show_sticks=True):
    wl = np.array([item['Wavelength'] for item in EI])
    I  = np.array([item['intensity']  for item in EI])

    good = np.isfinite(wl) & np.isfinite(I) & (I > 0)
    wl, I = wl[good], I[good]

    x, y = voigt_broaden(wl, I, delG, delL, npts)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, y, color='darkred', linewidth=1.2, label='Voigt-broadened')
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





#%% Main Code 

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
Ratio_c = np.zeros((len(Ne), len(Te)))   # 4d[3/2]°  / 4p[1/2]0  (needs 4d added)
Metastable_4s3 = np.zeros((len(Ne), len(Te)))  # ADD THIS
Metastable_4s4 = np.zeros((len(Ne), len(Te)))  # ADD THIS

for j,t in enumerate(Te) :
    for i,n in enumerate(Ne):
        print(t)
        print(n)
        UpdatedModel = CreateExcitationReactionRates(ModelData, t)
        UpdatedModel = CreateIonizationReactionRates(ModelData, t)
        Data, SD, EI, solver = CRModel(UpdatedModel, t, n, Pressure, Tg, R, interp)
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
        Metastable_4s3[i, j] = Data['4s3']['density_m^-3']
        Metastable_4s4[i, j] = Data['4s4']['density_m^-3']
PlotEmissionIntensities(EI,1000)
PlotStateDensities(SD)
    


# ------------------------------------------------------------------
# Plot (a): 4p'[1/2]0 / 4p[1/2]0
# ------------------------------------------------------------------
fig, ax = plt.subplots(1,2,figsize=(16, 5))

TeGrid, NeGrid = np.meshgrid(Te, Ne)  # match paper's axis units
CS = ax[0].contour(TeGrid, NeGrid, Ratio_a, levels=8, colors='black')
ax[0].clabel(CS, inline=True, fontsize=9)
ax[0].set_xlabel(r'$T_e$ [eV]')
ax[0].set_ylabel(r'$N_e$ [m$^{-3}$]')
ax[0].set_title("4p'[1/2]$_0$ / 4p[1/2]$_0$")
CS = ax[1].contour(TeGrid, NeGrid, Ratio_b, levels=8, colors='black')
ax[1].clabel(CS, inline=True, fontsize=9)
ax[1].set_xlabel(r'$T_e$ [eV]')
ax[1].set_ylabel(r'$N_e$ [ m$^{-3}$]')
ax[1].set_title( r'$4d[3/2]^0$/ 4p[1/2]$_0$')


plt.tight_layout()


plt.show()





t_test, n_test = 4.0, 1e15

UpdatedModel = CreateExcitationReactionRates(ModelData, t_test)
UpdatedModel = CreateIonizationReactionRates(UpdatedModel, t_test)
UpdatedModel = CreateSuperelasticRates(UpdatedModel, t_test)

Data, SD, EI,solver = CRModel(UpdatedModel, t_test, n_test, Pressure, Tg, R, interp)

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
 
ax.set_xlabel(r'$T_e$ [eV]', fontsize=13)
ax.set_ylabel(r'Intensity ratio: $I_{4p^{\prime}[1/2]_0} / I_{4p[1/2]_0}$', fontsize=13)
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
    n_meta_total = Metastable_4s3[i, :] + Metastable_4s4[i, :]
    ax.semilogy(Te, n_meta_total, 's-',
                label=f'$N_e$ = {Ne[i]:.1e} cm$^{{-3}}$',
                linewidth=2.5, markersize=6)
 
ax.set_xlabel(r'$T_e$ [eV]', fontsize=13)
ax.set_ylabel(r'Metastable density: $n_{4s3} + n_{4s4}$ [m$^{-3}$]', fontsize=13)
ax.set_title('Metastable population vs electron temperature', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=11, loc='best')
 
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'metastable_vs_Te_multiNe.png'), dpi=300)
plt.show()
 
# ====================================================================
# PLOT 3: Side-by-side comparison at one representative Ne
# ====================================================================
Ne_idx = len(Ne) // 2  # pick middle Ne
Ne_val = Ne[Ne_idx]
 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
 
# Left: Ratio
ax1.plot(Te, Ratio_a[Ne_idx, :], 'o-', color='darkred', linewidth=3, markersize=8)
ax1.set_xlabel(r'$T_e$ [eV]', fontsize=12)
ax1.set_ylabel(r'Intensity ratio', fontsize=12, color='darkred')
ax1.set_title(f'Ratio vs $T_e$ (at $N_e = {Ne_val:.1e}$ m$^{{-3}}$)', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='y', labelcolor='darkred')
 
# Right: Metastable
ax2.semilogy(Te, Metastable_4s3[Ne_idx, :], 's-', label='4s3', color='steelblue', linewidth=2.5, markersize=7)
ax2.semilogy(Te, Metastable_4s4[Ne_idx, :], 's-', label='4s4', color='navy', linewidth=2.5, markersize=7, linestyle='--')
ax2.set_xlabel(r'$T_e$ [eV]', fontsize=12)
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
x_spec, y_spec = PlotSpectrum(EI, delG=2, delL=0.005)

# ====================================================================
# STATISTICS
# ====================================================================
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)
 
for i in range(0, len(Ne), step):
    ratio_range = (np.nanmin(Ratio_a[i, :]), np.nanmax(Ratio_a[i, :]))
    ratio_change = (ratio_range[1] - ratio_range[0]) / ratio_range[0] * 100
    
    meta_range = (np.nanmin(Metastable_4s3[i, :] + Metastable_4s4[i, :]),
                  np.nanmax(Metastable_4s3[i, :] + Metastable_4s4[i, :]))
    meta_change = (meta_range[1] - meta_range[0]) / meta_range[0] * 100
    
    print(f"\nNe = {Ne[i]:.1e} cm^-3:")
    print(f"  Ratio variation: {ratio_range[0]:.4f} → {ratio_range[1]:.4f} ({ratio_change:+.1f}%)")
    print(f"  Metastable range: {meta_range[0]:.2e} → {meta_range[1]:.2e} ({meta_change:+.1f}%)")
 
print("\n" + "="*70)