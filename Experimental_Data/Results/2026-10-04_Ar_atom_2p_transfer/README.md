# Ar-atom 2p transfer in the CR model (4 Oct 2026)

**Question:** is missing heavy-particle mixing between the 4p (Paschen 2p) levels the reason the
line-ratio fit puts Ne at ~1e19 m^-3? **Answer: no.** With measured transfer rates the fit and its
chi^2-vs-Ne profile barely change.

## What was added

Population transfer by collisions with ground-state Ar, Ar(2p_i) + Ar -> Ar(2p_j or 1s) + Ar. The
18 channels are from Zhu & Pu, J. Phys. D 43, 015204 (2010), Table 3, who compiled measurements at
300 K by Chang & Setser (1978), Nguyen & Sadeghi (1978), Inoue, Setser & Sadeghi (1982) and
Sadeghi et al. (2001).

- **Data:** `InputData/Ar_2p_atom_transfer.csv`.
- **Rates:** k = (1-4e-11 cm^3/s) x (Tg/300)^0.5, listed downhill. The uphill rates follow from
  detailed balance at Tg.
- **2p -> 1s:** the paper does not say which 1s level, so the rate is shared among the four 1s
  levels by statistical weight.
- **Code:**
  - `he.AtomTransferRates` reads the file and applies detailed balance.
  - `AttachAtomTransfer` and the two solvers in `Scripts/MainFileV2.py` add the terms.
  - `CRModel(..., atom_transfer=True)` is now the default.
  - `CRFitNeTe.CONFIG['atom_transfer']` is the switch for the fits. Output folders and table names
    get `_noAT` without the transfer. Tables built before this change are treated as without.
- **Check:** with `atom_transfer=False` the densities are bit-identical to the model before this
  change.
- **Reaction tables:** `Scripts/MakeReactionTables.py` lists the process (`tab:ar_atom_transfer`).

**Size of the effect at 1 Torr, 300 K** (N_g = 3.2e22 m^-3):
- Each channel moves 0.3-1.3e6 s^-1. A 2p level loses at most ~2e6 s^-1 to Ar collisions in total.
- Its radiative loss A x eta is 1-4e7 s^-1 at realistic Ne. So the transfer is a 5-20 % effect,
  and largest where the 4p -> 4s lines are strongly trapped.

## To regenerate

| Step | Command |
|---|---|
| CR fits with the transfer (builds 16 tables, ~10 min with 4 workers) | `python Experimental_Data/ExperimentalDataAnalysis/ActinometryNitrogenContent.py` |
| chi^2 vs Ne, Ne pinned at n_c | `python SmallAnalysisScripts/CriticalDensityFit.py` |
| The same without the transfer | set `atom_transfer=False` in `CRFitNeTe.CONFIG`, rerun both (outputs go to `..._noAT`) |
| Without vs with | `python SmallAnalysisScripts/AtomTransferComparison.py` |
| What drives the fit to high Ne (now with a "no atom transfer" variant) | `python SmallAnalysisScripts/HighNeDiagnostics.py` |
| Comparison with Zhu & Pu | `python SmallAnalysisScripts/ZhuPuComparison.py` |

## Files

| File | Content |
|---|---|
| `chi2_profile_Ne_AT.png` | The chi^2-vs-Ne graph (CriticalDensityFit) with the transfer |
| `chi2_profiles_noAT_vs_AT.png` | The same graph without (before) and with (now) the transfer, side by side |
| `fit_changes.csv` | Free and Ne = n_c fits per condition, without vs with |
| `pinned_fits_AT.csv` | CriticalDensityFit results with the transfer |
| `high_ne_diagnostics.png`, `.csv` | Model variants (incl. "no atom transfer") and line subsets, chi^2 vs Ne |
| `zhupu_comparison.png`, `.csv` | Our model and our data next to Zhu & Pu's Figs. 1-2 (digitised, `InputData/References/ZhuPu2010_digitized.csv`) |
| `zhupu_our_pure_ar_fits.csv` | The pure-Ar fits used in that figure |

## Results

**chi^2 vs Ne.** The profiles with and without the transfer are almost the same
(`chi2_profiles_noAT_vs_AT.png`; in `high_ne_diagnostics.png` the black and orange curves overlap).

- **Best-fit Ne:** changes by -2 to +6 %. It stays at 2e18-1.5e19 m^-3 (DC 2.2-8.8e18, microwave
  2.2e18-1.5e19).
- **chi^2/dof of the free fits:** changes by at most 0.01.
- **Cost of pinning Ne = n_c (Delta chi^2/s^2), DC EEDF:** 14-57 before, 14-49 now (down by up to 7).
- **Same, microwave EEDF:** 28-62 before, 38-88 now.
  - The profiles are steep at n_c, so a small shift changes these numbers a lot.
  - For several Ar/N2 conditions the n_c-pinned fit also jumps between two E/N solutions (Te_eff
    2.6-2.9 eV before, 0.3-0.7 eV now). Both are poor fits.
- **Per-level mismatch at n_c (pure Ar):** unchanged by the transfer, for both EEDFs
  (`zhupu_comparison.png`, panel e).
  - Too weak: 2p1 / 750.4 nm by e^0.3-0.6 and 2p4 / 794.8 nm by e^0.6-0.8 (both EEDFs);
    2p3 / 706.7 nm by e^0.6 (microwave EEDF only).
  - Too strong: 2p6 / 800.6 nm by e^0.65-0.75 and 2p2 / 727.3 nm by e^0.3-0.8.

**Our model at their conditions** (`zhupu_comparison.png` a-c: Te, ne, p and Tg from their
Table 7, Maxwellian EEDF):
- **2p distribution:** reproduced to rms 0.35-0.45 in ln (CCP 100 Pa and ICP 1 Pa), against both
  their model and their OES.
- **The transfer itself behaves as expected.** At 1 Pa it changes nothing. At 100 Pa it shifts
  2p9 -> 2p10 by ~15 %: 2p10 moves toward their measurement (19.5 -> 22.4, OES 30.7) and 2p9 away
  from it (22.4 -> 19.0, OES 25.1). This fits their finding that at 100 Pa radiation still
  dominates the 2p losses.
- **Two differences from their model remain, with or without the transfer:**
  - 2p1 is 2.4-2.7x too high in both discharges.
  - The 1s densities are 1.2-7x higher than theirs (mostly 3-5x).

**Our data next to theirs** (panels d and f):
- **2p distribution:** our measured 1 Torr pure-Ar 2p distribution is close to their CCP
  (100 Pa, ne = 3e17 m^-3) OES. Over the seven levels we measure it agrees within 0.75-1.4x, except
  2p4 / 794.8 nm, which is 1.7x higher in ours. Their model reproduces that discharge at a
  realistic ne.
- **Trapping correction:** the measured populations depend on the escape factors. The band in (d)
  spans eta = 1 to eta of the DC-EEDF fit at n_c. Across that band the normalised populations
  change by up to 25 % (2p8, 2p4) and 45 % (2p2).
- **1s densities:** our absorption estimate for 1s5 is 0.6-5 ppm (n/g/n_g). Their CCP has 6-11 ppm.
  - Our microwave-EEDF model at n_c gives ~100x less (0.02 ppm).
  - The DC-EEDF model at n_c reaches the bottom of the absorption range (0.6 ppm).

## So what drives the high Ne?

Not radiation trapping, Tg, R or the 5p lines (earlier diagnostics), and not Ar-atom 2p transfer.
The fit raises Ne to fix the relative excitation of individual 2p levels: 2p4, 2p1 and (microwave)
2p3 too weak, 2p6 and 2p2 too strong at n_c. These levels differ in how strongly they are fed out
of the 1s levels (e.g. 1s3 -> 2p4 has the largest 1s -> 2p rate in Zhu & Pu's Table 2,
1s5 -> 2p6) and, for 2p1, from the ground state through the EEDF tail.

Candidates, in order of how cheap they are to test:
1. **1s densities.** Constrain them with the absorption measurement: add the measured 1s5 density
   to the chi^2, or fix the 1s densities. The microwave-EEDF fits give 1s densities ~100x below
   absorption, so stepwise excitation is underweighted at n_c.
2. **The 1s -> 2p cross sections (BSR).** Compare with the Boffard et al. measurements that
   Zhu & Pu use.
3. **The EEDF tail.** It sets 2p1 relative to 2p5.
