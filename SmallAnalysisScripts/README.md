# SmallAnalysisScripts

Stand-alone scripts that each answer one question or make one set of figures.
They import the analysis modules - since 2026-10-16 in `pipeline/` (see `pipeline/README.md`), still importable
under their old names (CRFitNeTe, ActinometryLineSelection, ActinometryRates, HelperFunctions, ...) - and write to
`Experimental_Data/Output/<...>` unless noted.
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
| `EchelleSaturationCheck.py` | Which echelle lines are saturated: reads the raw ESAWIN frames (.spe: frame, order traces, per-order wavelength polynomials) and maps pixels at the ADC ceiling to orders, wavelengths and catalogue lines; per folder the lines clipped and in how many spectra (the .spa flat-top test misses them) | `Output/EchelleSaturation/` |
| `FitSensitivity.py` | How much the fitted Ar I line pattern changes per decade of Ne and per 0.1 eV of Te_eff (after the free scale and response tilt), against the detection limit of the fit | `Output/FitSensitivity/` |
| `MakeFitReport.py` (now `pipeline/LatexReport.py`) | Figures and tables of the Overleaf report `LatexReport/main.tex` from the last N2SweepPureArEEDF run: copies of its overview / per-set figures, the chi^2 map with the free and n_c-pinned fit, the per-spectrum scale s, the cost of the pin, and the libraries / fits / actinometry tables (run from `Main.py` with `[report] make = true`) | `LatexReport/` |
| `MicrowaveLowENFit.py` | The CR fit with the 2.45 GHz BOLSIG+ EEDFs extended below 6 Td (precision 1e-30 at <= 30 Td) and with e-e collisions at the ionization degree of n_c or N2(v) superelastics (Tv 3000-8000 K); pure-Ar chi^2 vs Ne; chi^2, Te_eff and 1s5 along E/N at Ne = n_c; the fit walked in Ne from the free optimum down through n_c (Delta chi^2, Te_eff, CR 1s5, ionization frequency), against the old microwave and DC fits. Builds its libraries and CR tables in parallel processes | `Output/MicrowaveLowEN/` |
| `TwoZoneNeBias.py` | Synthetic two-zone line of sight (critical layer or overdense core + underdense shell, with and without the shell's 4s absorption) fitted with the one-zone CR fit: which way the fitted Ne moves | `Output/TwoZoneNeBias/` |
| `MicrowaveN2AFit.py` | Microwave EEDFs with superelastic collisions from N2(A3Sigma) at an estimated N2(A) density (production/loss balance, scanned n_A/n_N2 = 1e-3 to 1e-2): chi^2/dof, Te_eff and CR 1s5 at Ne = n_c for every Ar/N2 condition, against a no-N2(A) baseline and the DC fits | `Output/MicrowaveN2A/` |
| `NitrogenTrendsNearNc.py` | N-atom density (N I 744.2 actinometry, direct and stepwise Ar reference) and Te_eff against N2 % and power from the microwave CR fit with Ar I 415.86 / 706.72 / 750.39 nm and the calibration trusted, at the local chi^2 minimum in Ne nearest n_c; lines only and lines + measured 1s5 | `Output/NitrogenTrendsNearNc/` |
| `NitrogenAdmixture.py` | 1s5 / 1s3 metastable densities and Ar I line ratios vs N2 admixture (0-10 %), at fixed Te (Maxwellian) or fixed E/N (BOLSIG+ per mixture), with the N2 physics of Scripts/MainFileWithNitrogen.py | `Output/NitrogenAdmixture/` |
| `QEProCalibration.py` | QE Pro (1044 px, 0.76 nm/px) spectra of the 1.5 Torr power sweeps recorded with the echelle: instrument function (split Gaussian + Lorentzian, pixel-integrated), wavelength calibration from the Ar I lines, line areas by LSF fits, relative response from the echelle (Ar I lines; whole-spectrum windows below the order gaps) and same-upper-level branching ratios | `Output/QEProComparison/calibration/` |
| `QEProBandModel.py` | N2 bands at QE Pro resolution with the MolecularFitting tools: QE Pro slit function (`MolecularFitting/Slit_Functions/QEPro_2026_10_13.txt` + `_lsf.json`), ExoMol 14N2 C-B library calibrated to the QE Pro (`Calibrated_Library/QEPro_2026_10_13/14N2_CB`, air wavelengths) for `Fit_One_Temp_V2.MolecularFitter`, N2+ B-X templates from molecular constants, and the band + Ar-line fit of 385-480 nm that takes the bands off the 5p lines; check of the band-free 5p areas against the echelle | `Output/QEProComparison/bands/` |
| `QEProCRFit.py` | The CR fit (pipeline settings) of the QE Pro spectra with several line sets against the echelle fits of the same plasma (all nine lines, without clipped lines, and like for like: the same summed line groups on both); residuals per line, measured vs CR spectrum | `Output/QEProComparison/crfit/` |
| `QEProFivePCRFit.py` | The same CR fit with the band-free 5p groups of `QEProBandModel.py` (every group, band-fit errors) added to the QE Pro 4p lines; 4p groups contaminated by N2 in the 5 % N2 spectra left out (FPS heads, inflated errors); sets 4p only / 4p + all 5p / 4p + well-measured 5p / 5p only; band-subtracted 5p window and 4p window against the CR spectrum | `Output/QEProComparison/crfit_5p/` |
| `QEProEchelleSoftNc.py` | QE Pro (clean 4p + well-measured band-free 5p) and echelle (pipeline lines without clipped ones; with them in the CSV) CR fits with the soft Ne penalty above n_c (`Ne_penalty_above`, Main.py `nc_penalty` defaults); E/N and Ne vs power, one line per instrument; lines-only optimum kept in the CSV; `--eedf vibsat` = Ar/N2 EEDF with saturated N2(v) (builds missing libraries / tables) | `Output/QEProComparison/soft_nc[_vibsat]/` |
| `PureArgonMicrowaveNe.py` | Presentation figures: the pure-Ar CR fit with the microwave EEDFs walked in Ne (chi^2/dof, Delta chi^2, Te_eff and CR 1s5 against Ne, n_c marked), all panels together and one per figure; the same profile with fewer lines and with / without the response tilt (line_sets_vs_Ne.png) | `Output/PureArgonMicrowave/` |
| `Plot_Slit_Function_Slides.py` | Slide figures of the echelle slit function (Hg 435.8 nm vs a delta line) and FWHM vs wavelength | `MolecularFitting/Slit_Functions/Figures/Slides/` |

The EEDF libraries the two EEDF scripts use are made by `Scripts/CreateEEDFLibraries.py`.

The main analysis (CR fit at n_c, metastables, actinometry of the echelle sweeps) is `pipeline/RunDatasets.py`
(formerly `N2SweepPureArEEDF.py`, which still works as a stand-in); run it through `Main.py` in the repo folder with an input file (`python Main.py input_example.toml --plan`, then without
`--plan`). `input_example.toml` lists every setting with its default and explanation.
New EEDF options for `[eedf] model` (gases, cross sections and processes left out, cold / saturated / Boltzmann-Tv
vibrations, field, E/N, LAS or fixed Tg) are made with `python CreateEEDFScheme.py` in the repo folder; the list of
them, with what each is for, is `InputData/EEDFSchemes/EEDF_schemes.md`.
