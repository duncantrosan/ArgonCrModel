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
| `CrossSectionRatios.py` | sigma_N(E)/sigma_Ar(E) for every selected actinometry pair (flat = ideal) | `Output/cross_section_ratios.png` |
| `CRFitSpectrumOverlay.py` | CR-model spectrum at the best-fit (Te_eff, Ne), slit-broadened, over the measured spectrum, plus line-integral residuals vs wavelength | `Output/CRFit/` |
| `EEDFComparison.py` | Ground-state rate coefficients for MultiBolt EEDFs with 2-8 Legendre terms (Ar/N2) vs a Maxwellian | `Output/EEDFComparison.png` |
| `EEDFSolverComparison.py` | EEDFs from MultiBolt (6, 2 terms), BOLSIG+ (plain, e-e, superelastic, both) and Maxwell-Boltzmann, and the CR-model rates they give | `Output/EEDFSolvers/` |
| `EEDFComparisonArN2.py` | Pure Ar and Ar + 10 % N2 at matched mean energy: MultiBolt 2-8 terms vs BOLSIG+; Maxwell vs BOLSIG+ (plain, e-e, superelastic); BOLSIG+ DC vs microwave with e-e + superelastic. Builds the libraries it needs | `Output/EEDFComparisonArN2/` |
| `EEDFPhysicsFitComparison.py` | The pure-Ar CR fit with each EEDF library, with and without the response tilt | `Output/EEDFPhysics/` |
| `FitSensitivity.py` | How much the fitted Ar I line pattern changes per decade of Ne and per 0.1 eV of Te_eff (after the free scale and response tilt), against the detection limit of the fit | `Output/FitSensitivity/` |
| `NitrogenAdmixture.py` | 1s5 / 1s3 metastable densities and Ar I line ratios vs N2 admixture (0-10 %), at fixed Te (Maxwellian) or fixed E/N (BOLSIG+ per mixture), with the N2 physics of Scripts/MainFileWithNitrogen.py | `Output/NitrogenAdmixture/` |
| `Plot_Slit_Function_Slides.py` | Slide figures of the echelle slit function (Hg 435.8 nm vs a delta line) and FWHM vs wavelength | `MolecularFitting/Slit_Functions/Figures/Slides/` |

The EEDF libraries the two EEDF scripts use are made by `Scripts/CreateEEDFLibraries.py`.
