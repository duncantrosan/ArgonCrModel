# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:40:41 2026

@author: dptro
"""

# Creates a look up table to use for Radiation Trapping of the 4p and 4s states
# Looks through a varity of temperatures and Pressures and combines everything into 
# tau and Psi to reduce dimionsionality of the lookup
# Creates a sperate table for the two 4s resonant states as well as all the 4p states

# Outputs 12 tables (10 for all the p states and 2 for 4s2 and 4s4 states) 

# Implemnetation into main script find tau and psi in main script pass it to this function 
# along with the specified de-excitation
# returns an eta for the specified conditions




# Dependendicies -- HelperFunctions.py and MonteCarloEscapeFactorSolve.py 
# MainDir/InputData/LevelList.json


# List of reactions 
# CreateTable = {'Ar(4s2) -> ground',
#               'Ar(4s4) -> ground'}

import json 
from pathlib import Path
import numpy as np
import HelperFunctions as he
import MonteCarloEscapeFactorSolve as MC
import time

#%%  List of reactions to create table for 
CreateTable = {'4s2 -> ground'}


#%% Define Functions 


#%% MainFile 
MainDir = Path(__file__).resolve().parent.parent
LevelList = he.ImportLevelList() # Dict of all states taken into account
RadiativeReactionList = he.ImportReactionList() # Dict of all the Radiative De-excitations
UpdatedLevelList = he.combine(LevelList, RadiativeReactionList['transitions'])




# Create list of tau and Psi to run over the monte Carlo solve 
# Tau = k_0 * n_ls * R 
# Tau lower = 500 mTorr R = 1 cm Tg = 300 
# Tau higher = 5 Torr R = 5 cm Tg = 1200
PressureLow = 0.5 # Low pressure in Torr 
TempLow = 300 # K
RadLow = 1/100 # m 

PressureHigh = 5 # Torr
TempHigh = 1200 
RadHigh = 5/100

EstimatedTimePer = 152
NTerms = 1
TotalTime = 152*NTerms**2/60
print(f'The Estimated Time to run this code is around {TotalTime} minutes')
StartTime = time.perf_counter()
LowerTau = []
UpperTau = []
UpperShape = []
LowerShape = []
Data= []
TotalTerms = NTerms**2
Done = 1
for Reac in CreateTable:
    for i,t in enumerate(UpdatedLevelList['transitions_in_set']):
        if Reac in t['label_pair']:
            A = t
            left, right = Reac.split(' -> ')
            
            UpperLevel = LevelList[left]
            LowerLevel = LevelList[right]
            Shape, tau = he.FindTau(UpperLevel, LowerLevel, t, PressureLow, TempLow, RadLow)
            LowerTau.append(tau)
            LowerShape.append(Shape)
            Shape, tau = he.FindTau(UpperLevel, LowerLevel, t, PressureHigh, TempHigh, RadHigh)
            UpperTau.append(tau)
            UpperShape.append(Shape)
            
            # Create Search Matrix 
            TauSearch = np.logspace(-3, 5, NTerms)
            ShapeSearch = np.linspace(0.01,0.2,NTerms)
            print(f'The Tau search space is {TauSearch[0]} to {TauSearch[-1]} with {NTerms*len(Reac)} points')
            print(f'The Space search space is {ShapeSearch[0]} to {ShapeSearch[-1]} with {NTerms} points')
            for j in TauSearch:
                for k in ShapeSearch:
                    eta = MC.escape_factor_fast(j,k)
                    DataRow = {'Tau_R':j,
                            'Shape':k,
                            'EscapeFactor':eta}
                    Data.append(DataRow)
                    print(f'Tau = {j}, Shape = {k} Finished -- Completed {Done} out of {TotalTerms} Runs')
                    
                    NewTime = time.perf_counter()
                    UpdatedTime = NewTime  - StartTime
                    UpdatedTimePer = UpdatedTime/Done
                    EstimatedTimeRemaining = UpdatedTimePer * (NTerms**2 - Done)/60
                    Done = Done +1
                    print(f'Time Remaining Estimate is {EstimatedTimeRemaining} minutes')


EndTime = time.perf_counter()
FinishTime = EndTime - StartTime 
print(f'The Total Time taken was {FinishTime}')         

Response = input('Save Data? Y or N? ')
if Response == 'Y':
    with open('RadiationTrappingLookupTable.json', 'w') as f:
        json.dump(Data, f)
        print('Data Saved')
else:
    print('Data Not Saved')


# Limits for the 4s2 state _> n2(X)
# ni = he.Torr2Volume(0.5)
# k_0 = ni * ( LevelList['4s2']['g'] / LevelList['ground'][]    )




