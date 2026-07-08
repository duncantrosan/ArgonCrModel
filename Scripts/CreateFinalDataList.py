# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:05:31 2026

@author: dptro
"""

# Create Data and save to JSON format

import HelperFunctions as he 



#import all levels taken into consideration JSON file includes type of level
# Leveltypes -- ground, resonant, metastable,normal
LevelList = he.ImportLevelList()
print('Importing Level List...')
# Import Radiative Reaction list from Nistdatabase
RadiationReactions = he.ImportReactionList()
print('Importing Reaction List...')
# Combine level list to make radiative loss 
LevelList_Update = he.combine(LevelList,RadiationReactions['transitions'])
# Import JSON cross section from LXcat 
CrossSectionList = he.ImportCrossSections()
print('Importing Reaction List...')
CrossSectionRates = he.MaxweillianReactionRates(CrossSectionList['cross_sections'])
#Combines Radiative transition into ModelData
ModelData = he.BuildRateModel(LevelList, LevelList_Update, CrossSectionRates)
# Adds in cross-section data into the Model List
ModelData = he.AddElectronExcitation(ModelData,CrossSectionList)
print('Combining Data...')
# Adds ionization Rates to the data (adds to loss terms)
ModelData = he.CreateIonizationCrossSections(ModelData)
# Imports Radiation Trapping Matrix
print('Importing Radiation Trapping Lookup...')
RadiationTrappingMatrix = he.ImportRadiationTrappingMatrix()
print('Adding In Diffusion Reactions...')
# ModelData = he.AddDiffusionLoss(ModelData)





