# SmallAnalysisScripts

Stand-alone scripts that each answer one question or make one set of figures.
They import the analysis modules from `Experimental_Data/ExperimentalDataAnalysis`
(CRFitNeTe, ActinometryLineSelection, ActinometryRates, ...) and `Scripts`
(HelperFunctions), and write to `Experimental_Data/Output/<...>` unless noted.
Run any of them from Spyder (F5) or `python SmallAnalysisScripts/<name>.py`.

| Script | What it shows | Output |
|---|---|---|
| `ActinometryPairFlatness.py` | How much k_N/k_Ar of each actinometry pair swings over a Te window (Maxwellian vs MultiBolt) | `Output/` |
| `ActinometryWalkthrough.py` | Step-by-step figures of what ActinometryTeUncertainty does for one condition and pair | `Output/Walkthrough/` |
| `CriticalDensityFit.py` | The CR fits (saved by ActinometryNitrogenContent) with Ne pinned at the 2.42 GHz critical density: chi^2 profile over Ne, pinned Te_eff, CR 1s5 density vs absorption, N-atom density | `Output/CriticalDensity/` |
| `CrossSection4sComparison.py` | Excitation out of the 4s levels: BSR (used now) vs RDW and the analytic Drawin fill-ins (used before) - rate coefficients, cross sections, and the CR-model density change over Ne | `Output/CrossSections4s/` |
| `CrossSectionRatios.py` | sigma_N(E)/sigma_Ar(E) for every selected actinometry pair (flat = ideal) | `Output/cross_section_ratios.png` |
| `CRFitSpectrumOverlay.py` | CR-model spectrum at the best-fit (Te_eff, Ne), slit-broadened, over the measured spectrum, plus line-integral residuals vs wavelength | `Output/CRFit/` |
| `EEDFComparison.py` | Ground-state rate coefficients for MultiBolt EEDFs with 2-8 Legendre terms (Ar/N2) vs a Maxwellian | `Output/EEDFComparison.png` |
| `EEDFSolverComparison.py` | EEDFs from MultiBolt (6, 2 terms), BOLSIG+ (plain, e-e, superelastic, both) and Maxwell-Boltzmann, and the CR-model rates they give | `Output/EEDFSolvers/` |
| `EEDFComparisonArN2.py` | Pure Ar and Ar + 10 % N2 at matched mean energy: MultiBolt 2-8 terms vs BOLSIG+; Maxwell vs BOLSIG+ (plain, e-e, superelastic); BOLSIG+ DC vs microwave with e-e + superelastic. Builds the libraries it needs | `Output/EEDFComparisonArN2/` |
| `EEDFPhysicsFitComparison.py` | The pure-Ar CR fit with each EEDF library, with and without the response tilt | `Output/EEDFPhysics/` |
| `FitSensitivity.py` | How much the fitted Ar I line pattern changes per decade of Ne and per 0.1 eV of Te_eff (after the free scale and response tilt), against the detection limit of the fit | `Output/FitSensitivity/` |
| `HighNeDiagnostics.py` | Why the CR fit puts Ne at ~1e19: chi^2 vs Ne for model variants (no 4p-4s trapping, Tg, R, BSR ground state) and line subsets (without the primed-core 2p lines, the 5p lines, 800.6/801.5 nm) | `Output/HighNeDiagnostics/` |
| `NitrogenAdmixture.py` | 1s5 / 1s3 metastable densities and Ar I line ratios vs N2 admixture (0-10 %), at fixed Te (Maxwellian) or fixed E/N (BOLSIG+ per mixture), with the N2 physics of Scripts/MainFileWithNitrogen.py | `Output/NitrogenAdmixture/` |
| `Plot_Slit_Function_Slides.py` | Slide figures of the echelle slit function (Hg 435.8 nm vs a delta line) and FWHM vs wavelength | `MolecularFitting/Slit_Functions/Figures/Slides/` |
| `SweepTrends.py` | Te_eff, Ne and the scale factor s of the CR fit along the N2-fraction and power sweeps (each N2 fraction with its own mixture EEDF and CR model; microwave and DC), Ne also with one common s per sweep (absolute intensities), BSR vs RDW 4s cross sections | `Output/SweepTrends/` |

The EEDF libraries the two EEDF scripts use are made by `Scripts/CreateEEDFLibraries.py`.

Cross sections out of the 4s levels: the CR model uses the BSR set (`he.XSEC_4S = 'BSR'`,
`CRFitNeTe.CONFIG['xsec_4s']`). Set `xsec_4s = 'RDW'` in `CRFitNeTe.CONFIG` for the cross
sections used before October 2026; the CR tables are cached per set, and the output folders of
the CR-fit scripts get the suffix `_RDW4s`. `xsec_ground = 'BSR'` (`he.XSEC_GROUND`) takes the
ground-state excitation (4s, 4p, 3d, 5s) from BSR too (suffix `_BSRgnd`); the default is still RDW.
