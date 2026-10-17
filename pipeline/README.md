# pipeline/

Everything `Main.py` needs to run the echelle analysis lives in this folder. Start a run with
`python Main.py <input file>` in the repo folder. Every setting of the input file is explained in
`input_example.toml`.

Paths: every file and folder is located from the folder of `Main.py` (`Paths.py`). The only exception is a full
path given in the input file, such as `[run] echelle_folder`.

## How a run flows

```
Main.py                       reads + checks the input file, sets the modules, returns RES (results dict)
 └─ RunDatasets.py            per data set: conditions, fits, results dict, metastables, actinometry, plots
     ├─ LASAndEEDFLibraries   LAS table -> Tg of each condition, LAS 1s5; BOLSIG+ libraries; fit of one condition
     │   ├─ LASTableReader    reads either LAS table layout
     │   └─ EEDFSchemes       EEDF schemes of CreateEEDFScheme.py ([eedf] model = "<scheme>")
     ├─ CRTableBuilder        which CR table / EEDF library a condition uses; builds missing tables in parallel
     ├─ CRTableFit            CR tables on (E/N, Ne), the line-intensity fit, admixture quenching, CR state solver
     │   ├─ CRModelSolver     the CR model: balance equations of all 51 Ar levels
     │   ├─ PhysicsHelpers    atomic data, cross sections, EEDFs, BOLSIG+, escape factors, quenching tables
     │   └─ LineSelection     which lines; line areas of every spectrum
     │       └─ SpectrumLineFitter   reading .spa spectra, fitting their lines
     ├─ NitrogenActinometry   n_N / n_Ar from N I / Ar I pairs
     │   ├─ ActinometryRateCoefficients   rate coefficients of the actinometry lines
     │   └─ ActinometryUncertainty        posterior / quantile helpers
     └─ LatexReport           figures and tables of LatexReport/ ([report] make = true)
PipelineSettings.py           shared constants (2.42 GHz, n_c, E/N grid and BOLSIG+ precision of the libraries)
Paths.py                      all folders, from the folder of Main.py
```

## Files and where they came from

The header of each file says what it takes in, what it gives out, and which files use it.

| file | does | was |
|---|---|---|
| `RunDatasets.py` | the analysis of every data set; the results dict | `SmallAnalysisScripts/N2SweepPureArEEDF.py` |
| `CRTableFit.py` | CR tables, the line fit, admixture quenching, CR state solver | `Experimental_Data/ExperimentalDataAnalysis/CRFitNeTe.py` |
| `CRModelSolver.py` | the CR model (functions only) | top part of `Scripts/MainFileV2.py` |
| `PhysicsHelpers.py` | atomic data, EEDFs, BOLSIG+, trapping, quenching tables | `Scripts/HelperFunctions.py` |
| `LASAndEEDFLibraries.py` | LAS Tg and 1s5, BOLSIG+ libraries, fit of one condition | pipeline half of `SmallAnalysisScripts/LASTgComparison.py` |
| `CRTableBuilder.py` | CR-table settings per EEDF option, parallel builds | pipeline half of `SmallAnalysisScripts/AnalyticFivePFit.py` |
| `EEDFSchemes.py` | reading / checking EEDF schemes | library half of `CreateEEDFScheme.py` |
| `LineSelection.py` | line selection, line areas of the spectra | `.../ActinometryLineSelection.py` |
| `SpectrumLineFitter.py` | `.spa` reading, line catalogue, group line fits | `.../Argon_Nitrogen_Mix_Analysis_V2.py` |
| `NitrogenActinometry.py` | N I / Ar I actinometry | `.../ActinometryNitrogenContent.py` |
| `ActinometryRateCoefficients.py` | actinometry rate coefficients | `.../ActinometryRates.py` |
| `ActinometryUncertainty.py` | actinometry uncertainty helpers | `.../ActinometryTeUncertainty.py` |
| `LASTableReader.py` | LAS table in either layout | `SmallAnalysisScripts/LASColumns.py` |
| `LatexReport.py` | report figures / tables | `SmallAnalysisScripts/MakeFitReport.py` |
| `PipelineSettings.py` | shared constants | from `PressureSweepAnalysis.py` / `MicrowaveLowENFit.py` |
| `Paths.py` | all folders | new |

## The old file names still work

Each old file is still there as a small stand-in, so the analysis scripts in `SmallAnalysisScripts/` and
`Scripts/` keep running unchanged:

- Importing an old name (e.g. `import CRFitNeTe`) gives you the pipeline module itself, the same object.
  A setting you change through either name is therefore seen by both.
- Running an old file runs the new one.

`LASTgComparison.py`, `AnalyticFivePFit.py` and `MainFileV2.py` keep their own analysis or demo and take the
pipeline part from here. New code should import the pipeline modules directly.

## Caches

The output folder names were kept, so the cached libraries, CR tables and line fits from before the move are
reused:

- BOLSIG+ libraries in `InputData/Bolsig`
- CR tables in `Experimental_Data/Output/AnalyticFivePFit/tables*`
- library-worker logs in `Experimental_Data/Output/LASTgComparison/logs`
- line fits in `<output_folder>/<data set>/measurements`

Checked on 2026-10-16: a run of the 80 W N2 sweep and the 2.4 % N2 power sweep reproduces the results from
before the move exactly, with 0 libraries and 0 CR tables rebuilt.
