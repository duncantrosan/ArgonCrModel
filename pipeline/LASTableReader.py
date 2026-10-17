"""
LASTableReader.py  -  reading the LAS table in either of its layouts
=======================================================================================================================
TAKES IN   the path of an LAS table: Master_AllMeasurements.csv (one row per measurement) or Master_Summary.csv (one
           row per condition)
GIVES OUT  read_las(path) -> DataFrame with the Master_AllMeasurements column names (SUMMARY_TO_ALL renames the Summary
           columns), so the rest of the pipeline works on both
USED BY    LASAndEEDFLibraries (load_las), RunDatasets (the LAS summary with the OES results)
=======================================================================================================================
(Before the pipeline was collected in pipeline/ this file was SmallAnalysisScripts/LASColumns.py.)
"""
import pandas as pd

SUMMARY_TO_ALL = {"Pressure_Input_Torr": "pressure", "Power_Input_W": "power", "N2_Percent_Input": "Nitrogen_Percent",
                  "Frequency_Input_MHz": "frequency", "run_ids": "run_id", "first_measured": "date"}


def read_las(path):
    """LAS table at path with the Summary columns renamed to the AllMeasurements names (SUMMARY_TO_ALL)."""
    d = pd.read_csv(path)
    return d.rename(columns={k: v for k, v in SUMMARY_TO_ALL.items() if k in d.columns and v not in d.columns})
