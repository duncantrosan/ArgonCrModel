# BSR cross sections out of the 4s levels (4 Oct 2026)

Snapshot of the figures and tables from switching the CR model's electron-impact excitation
**out of the four 4s levels** from the RDW (NGFSRDW) set to the BSR (B-spline R-matrix,
Zatsarinny & Bartschat) set in `InputData/ArgonLxCatPureArgonUpdated.txt`. Transitions starting
from higher levels (4p, 3d, 5s) in that file are not used yet. The ground state stays RDW, and
4s → 5p (not in the BSR set) also stays RDW.

To regenerate (outputs go to `Experimental_Data/Output/...`, which git ignores):

| Step | Command | Time |
|---|---|---|
| BSR JSON (already committed) | `python Scripts/ParserforLXCatData.py` | s |
| CR fits, N-atom content | `python Experimental_Data/ExperimentalDataAnalysis/ActinometryNitrogenContent.py` | ~15 min (builds 14 CR tables) |
| chi^2 vs Ne, Ne pinned at n_c | `python SmallAnalysisScripts/CriticalDensityFit.py` | ~2 min |
| Te_eff, Ne, s along the sweeps | `python SmallAnalysisScripts/SweepTrends.py` | ~10 min (+ the RDW tables if missing) |
| What changed (needs CriticalDensityFit with both sets) | `python SmallAnalysisScripts/CrossSection4sComparison.py` | ~1 min |

For the RDW comparison, set `xsec_4s="RDW"` in `CRFitNeTe.CONFIG` and rerun the first two; outputs
then go to folders ending in `_RDW4s`.

## Files

| File | Content |
|---|---|
| `rates_4s.png`, `sigma_4s.png` | Rate coefficients (Maxwellian, 1 eV) and cross sections out of each 4s level: BSR vs the RDW and analytic (Drawin) data used before |
| `densities_4s.png` | CR-model densities with BSR / with RDW over Ne (pure-Ar microwave EEDF) |
| `chi2_profile_Ne_BSR.png` | The chi^2-vs-Ne graph (CriticalDensityFit) with the BSR cross sections |
| `chi2_profiles_RDW_vs_BSR.png` | The same graph, before (RDW) and now (BSR) side by side |
| `pinned_summary_BSR.png` | CriticalDensityFit summary: Te_eff free vs Ne pinned at n_c, the cost of pinning, CR 1s5 density vs the absorption estimate, N-atom density |
| `n2_fraction_trends.png`, `power_trends.png` | Te_eff, Ne (free s and one common s), the scale factor s, the measured intensity of the fitted lines (no model) and chi^2/dof along each sweep, BSR vs RDW |
| `scale_posteriors.png` | Likelihood of s from each condition alone and the posterior of one common s per sweep |
| `trends.csv` | Numbers behind the trend figures |
| `fit_changes.csv`, `pinned_fits_BSR.csv`, `pinned_fits_RDW.csv` | Free and pinned fit results per condition, RDW vs BSR |
| `high_ne_diagnostics.png`, `.csv` | chi^2 vs Ne for model variants and line subsets: what drives the fit to Ne ~ 1e19 |

## Results

**Cross sections.** Out of the metastable 1s5 the two sets roughly agree for the strong 4p
channels (rates within ~30 %; 1s5 → 4p7/4p8 are ~3x lower in BSR). Out of the resonant 1s4 and
1s2 levels the RDW rates to the 4p levels are 5-100x larger than BSR, and the 4s ↔ 4s mixing
rates 30-50x larger (RDW at 1 eV: 1s5 → 1s4 = 2.6e-12 m^3/s). BSR also replaces the analytic
4s → 3d / 5s estimates (BSR is 10-500x larger) and adds 32 channels the model neglected. BSR
thresholds are BSR's own level energies (4p 0.11-0.17 eV, 3d up to 0.22 eV above NIST); the cross
sections are shifted to the NIST thresholds the model uses.

**CR model.** At Ne = 1e16 m^-3 (Te_eff 0.9 eV) BSR gives ~5x more 1s5 and ~6x more 1s3 (less
electron-impact mixing out of the metastables); at 1e17 the 4p densities drop to 0.4-0.6x (less
stepwise excitation through the resonant levels); above ~1e18 the two sets agree within ~20-30 %.

**Fits.** chi^2/dof is lower with BSR for all 24 condition x EEDF fits (median -0.11). The chi^2
profile changes qualitatively: with RDW the DC-EEDF fits had their minimum near Ne ~ 3e16 m^-3 and
pure Ar was at n_c = 7.3e16 m^-3 (Delta chi^2/s^2 at n_c = 0.02-5.9); with BSR every condition has
a single minimum at Ne ~ 3e18-1e19 and pinning Ne = n_c costs Delta chi^2/s^2 = 14-62 (both EEDFs).
But the BSR best fits predict a 1s5 density of 1e10-3e16 m^-3, far below the absorption estimate
(1-8e17); with Ne pinned at n_c the DC-EEDF fits give 1.4e17-1.8e18, consistent with absorption.
Line ratios and absorption therefore disagree in the model (see the caveat below).

**Te_eff vs N2 fraction** (BSR, free s, 0-3 % N2): DC EEDF 2.47, 1.93, 1.41, 1.17, 1.08, 0.91,
0.87 eV; microwave EEDF 1.08 eV for pure Ar and 0.52-0.54 eV, flat, from 0.5 to 3 %. The 0 % point
uses the pure-Ar EEDFs, whose bulk-vs-tail relation differs from the Ar/N2 ones (N2 vibrational
losses), so Te_eff (a mean energy) is not directly comparable across that step.

**Scale factor s** (= ln(measured / model) at 775 nm). Each condition alone hardly fixes s, since E/N
and Ne trade off along the fit valley. But one s fits every condition of a sweep: DC EEDF
s = -22.4 for the N2 sweep (Delta chi^2 = 1.4 for 6 constraints) and -22.2 for the power sweep
(1.9 for 4); microwave EEDF -16.8 and -16.8 (5.5 and 0.7). The same s for both sweeps is what one
calibration and fixed optics imply. With RDW the microwave N2 sweep needed s = -8.4 against
-19.3 for the power sweep (Delta chi^2 = 8.0 for 6 constraints).

**Ne vs power** (2.4 % N2, BSR): no significant change. DC EEDF Ne = 5.4, 6.2, 8.4, 6.5,
6.7 x 1e18 m^-3 at 35/55/65/75/85 W; microwave 11, 6.3, 9.2, 7.4, 11 x 1e18. The 16-84 % ranges are
~2.5x down and reach the top of the Ne grid (3e19). The fitted lines get stronger with power
(relative to 85 W: 0.68 at 55 W, 0.74, 0.87, 1.00). With one common s, the model explains that rise
with a small Te_eff increase (DC: 1.09 → 1.16 eV) at constant Ne, not with Ne. A larger emitting
volume at higher power (which a common s cannot tell apart) would also fit. 45 W has no
intensity-calibrated spectra, and the two 35 W spectra differ by 2.2x in intensity (35W_1 was
exported with header flags 1-1-1-1, 35W_2 with 1-0-1-1).

## Why the fit wants Ne ~ 1e19 (`SmallAnalysisScripts/HighNeDiagnostics.py`)

Ne ~ 1e19 m^-3 is not a credible density for this discharge, and the absorption 1s5 density
(1-8e17 m^-3) is reproduced by the model only near Ne ~ n_c - 1e18. `high_ne_diagnostics.png`
shows what drives the fit there (pure Ar and 2.4 % N2 at 85 W, both EEDFs):

- **The four primed-core 2p lines** (2p1-2p4: 750.4, 727.3, 706.7, 794.8 nm). Without them the
  chi^2 profile over Ne is nearly flat: Delta chi^2/s^2 at n_c drops from 30-59 to 2-7, and
  1e18 is fully consistent. At realistic Ne the CR model underpopulates the 2p levels of the
  2P1/2 core relative to the 2P3/2 ones (750.4 nm ~e^1 too weak, 800.6/801.5 nm ~e^0.8 too
  strong at n_c, also without any trapping). It reaches the observed, strongly mixed 4p
  distribution only through electron-collision mixing at ~1e19.
- **Not the cause** (minimum stays at Ne >= 2e18): radiation trapping of the 4p -> 4s lines
  (switched off), Tg = 700 K, R = 1 cm, the 5p lines (dropped), 800.6/801.5 nm (dropped).
- **Partly** (Ne down ~1.5-3x, Delta chi^2 at n_c down 30-45 %): ground-state excitation from
  BSR (`xsec_ground = "BSR"`). The RDW ground -> 4s/4p cross sections are 10-100x above both
  BSR and the Biagi set (which the BOLSIG+ EEDFs use) within ~8 eV of threshold.
- **Worse** (Ne to the grid top): BSR excitation out of the 4p levels (weaker 4p <-> 4p mixing
  than RDW), tested but not added to the code.

So the electron density from the line ratios is set by a model shortfall in how the primed-core
2p levels are populated (candidates: cascades from 2s/3d' levels, heavy-particle 2p mixing, the
1s3/1s2 densities), not by the plasma. Until that is resolved, Ne should come from elsewhere
(critical density, absorption, power balance), with Te_eff fitted at that Ne.

Separate known gap: the RDW ground -> 3d / 5s / 5p tables start at 20 eV and 4s -> 5p at 5-6 eV
(sigma = 0 below), so the 5p upper levels of 415.9/430.0 nm get ~0 % direct excitation at
Te_eff <= 1.3 eV. Dropping the 5p lines does not lower Ne, so this is not the Ne driver.
