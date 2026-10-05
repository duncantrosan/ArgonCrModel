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
| `ArLineCloseups.py` | Close-ups of the Ar I lines of the CR fit in pure Ar and the Ar/N2 spectra (normalised to the fitted peak, log), with the group-fit window, model and baseline, catalogue lines, N2 band regions, the manual grades and flags; table of window bounds, baseline under the line and residual per spectrum | `Output/ArLineCloseups/` |
| `CriticalDensityFit.py` | The CR fits (saved by ActinometryNitrogenContent) with Ne pinned at the 2.42 GHz critical density: chi^2 profile over Ne, pinned Te_eff, CR 1s5 density vs absorption, N-atom density | `Output/CriticalDensity/` |
| `CrossSectionRatios.py` | sigma_N(E)/sigma_Ar(E) for every selected actinometry pair (flat = ideal) | `Output/cross_section_ratios.png` |
| `CRFitSpectrumOverlay.py` | CR-model spectrum at the best-fit (Te_eff, Ne), slit-broadened, over the measured spectrum, plus line-integral residuals vs wavelength | `Output/CRFit/` |
| `EEDFComparison.py` | Ground-state rate coefficients for MultiBolt EEDFs with 2-8 Legendre terms (Ar/N2) vs a Maxwellian | `Output/EEDFComparison.png` |
| `EEDFSolverComparison.py` | EEDFs from MultiBolt (6, 2 terms), BOLSIG+ (plain, e-e, superelastic, both) and Maxwell-Boltzmann, and the CR-model rates they give | `Output/EEDFSolvers/` |
| `EEDFComparisonArN2.py` | Pure Ar and Ar + 10 % N2 at matched mean energy: MultiBolt 2-8 terms vs BOLSIG+; Maxwell vs BOLSIG+ (plain, e-e, superelastic); BOLSIG+ DC vs microwave with e-e + superelastic. Builds the libraries it needs | `Output/EEDFComparisonArN2/` |
| `EEDFPhysicsFitComparison.py` | The pure-Ar CR fit with each EEDF library, with and without the response tilt | `Output/EEDFPhysics/` |
| `FitSensitivity.py` | How much the fitted Ar I line pattern changes per decade of Ne and per 0.1 eV of Te_eff (after the free scale and response tilt), against the detection limit of the fit | `Output/FitSensitivity/` |
| `MicrowaveLowENFit.py` | The CR fit with the 2.45 GHz BOLSIG+ EEDFs extended below 6 Td (precision 1e-30 at <= 30 Td) and with e-e collisions at the ionization degree of n_c or N2(v) superelastics (Tv 3000-8000 K); pure-Ar chi^2 vs Ne; chi^2, Te_eff and 1s5 along E/N at Ne = n_c; the fit walked in Ne from the free optimum down through n_c (Delta chi^2, Te_eff, CR 1s5, ionization frequency), against the old microwave and DC fits. Builds its libraries and CR tables in parallel processes | `Output/MicrowaveLowEN/` |
| `TwoZoneNeBias.py` | Synthetic two-zone line of sight (critical layer or overdense core + underdense shell, with and without the shell's 4s absorption) fitted with the one-zone CR fit: which way the fitted Ne moves | `Output/TwoZoneNeBias/` |
| `MicrowaveN2AFit.py` | Microwave EEDFs with superelastic collisions from N2(A3Sigma) at an estimated N2(A) density (production/loss balance, scanned n_A/n_N2 = 1e-3 to 1e-2): chi^2/dof, Te_eff and CR 1s5 at Ne = n_c for every Ar/N2 condition, against a no-N2(A) baseline and the DC fits | `Output/MicrowaveN2A/` |
| `NitrogenTrendsNearNc.py` | N-atom density (N I 744.2 actinometry, direct and stepwise Ar reference) and Te_eff against N2 % and power from the microwave CR fit with Ar I 415.86 / 706.72 / 750.39 nm and the calibration trusted, at the local chi^2 minimum in Ne nearest n_c; lines only and lines + measured 1s5 | `Output/NitrogenTrendsNearNc/` |
| `NitrogenAdmixture.py` | 1s5 / 1s3 metastable densities and Ar I line ratios vs N2 admixture (0-10 %), at fixed Te (Maxwellian) or fixed E/N (BOLSIG+ per mixture), with the N2 physics of Scripts/MainFileWithNitrogen.py | `Output/NitrogenAdmixture/` |
| `PureArgonMicrowaveNe.py` | Presentation figures: the pure-Ar CR fit with the microwave EEDFs walked in Ne (chi^2/dof, Delta chi^2, Te_eff and CR 1s5 against Ne, n_c marked), all panels together and one per figure; the same profile with fewer lines and with / without the response tilt (line_sets_vs_Ne.png) | `Output/PureArgonMicrowave/` |
| `Plot_Slit_Function_Slides.py` | Slide figures of the echelle slit function (Hg 435.8 nm vs a delta line) and FWHM vs wavelength | `MolecularFitting/Slit_Functions/Figures/Slides/` |

The EEDF libraries the two EEDF scripts use are made by `Scripts/CreateEEDFLibraries.py`.
