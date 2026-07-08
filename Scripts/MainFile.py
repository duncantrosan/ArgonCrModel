# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:37:07 2026

@author: dptro
"""

# Create Reaction Rates and solve equations 
import numpy as np
import HelperFunctions as he
Pressure = 1 # Pressure in Torr 
Tg = 300 # Temperature in K
Te = np.linspace(0.001,6,5) # Electron Temperature in ev
Ne = np.logspace(12,18,6) # electron Density in m^-3
ModelData,RadiationTrapping = he.GetData()
R = 2/100 # Radius in meters


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
    print(Reac)
    return ModelData

def CRModel(ModelData,Te,ne):
    
    
    
    SD = 1
    EI = 1
    
    return SD ,EI


# Set up Model 
ModelData = he.AddDiffusionLoss(ModelData, Pressure, Tg, R)
Ng = he.Torr2Volume(Pressure)
StateDensities = []
EmissionIntensities = []
for t in Te :
    for n in Ne:
        UpdatedModel = CreateExcitationReactionRates(ModelData, t)
        SD,EI = CRModel(ModelData,t,n)
        StateDensities.append(SD)
        EmissionIntensities.append(EI)
        

    



