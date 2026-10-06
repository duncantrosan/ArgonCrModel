# CR model vs the literature: BSR cross sections, Ar-atom 2p transfer, escape-factor fix (6-7 Oct 2026)

Overnight work. The question: with the new cross sections, is the CR model far off the published argon CR
models (Zhu & Pu 2010, Bogaerts et al. 1998), and does any level lack decay or production channels?

## What changed in the model

| Change | Where | Effect |
|---|---|---|
| BSR-500 electron-impact cross sections (from threshold) for every channel BSR has; NGFSRDW only for 5p | `Scripts/ParserforBSRData.py` -> `InputData/ArgonCrossSections_BSR.json`, `he.CROSS_SECTION_FILE` | ground -> 4s/4p were 10-30x too large in NGFSRDW (BSR and Biagi agree) |
| Ar-atom 2p <-> 2p and 2p -> 1s transfer (Zhu & Pu 2010, Table 3, measured rates; detailed balance at Tg) | `MainFileV2.AttachAtomTransfer`, `he.AtomTransferRates`, `InputData/Ar_2p_atom_transfer.csv`; `CRModel(atom_transfer=True)`, `CRFitNeTe.CONFIG['atom_transfer']` | ported from the 4 Oct cloud branch (`origin/claude/amazing-heisenberg-uqihdz`); small at 1 Torr |
| **Escape-factor table extended to small Voigt a** | `Scripts/ExtendTrappingLookup.py` -> `InputData/RadiationTrappingLookupTable_v2.json`, `he.EscapeFactorInterpolator` | **bug fix**, see below |
| Optional Holstein-Walsh escape factors for the resonance lines | `he.EscapeFactorWalsh`, `EscapeFactorInterpolator(mode='walsh').resonance`, `CRFitNeTe.CONFIG['escape_mode']` | systematic (default stays the Monte Carlo table) |
| Optional diffusion length | `he.AddDiffusionLoss(..., Lambda=...)` | default unchanged (hemisphere, R/4.493) |
| Faster CR tables | escape factors in one interpolator call (cloud branch); `build_model_table` reuses the rates of a 1D-library row across Ne | ~10x faster table builds |

CR tables record the cross-section set, atom transfer, escape table and escape mode and rebuild when any
of them differs (`crf.table_is_current`).

### The escape-factor bug

`RadiationTrappingLookupTable.json` (Monte Carlo, uniformly emitting hemisphere of radius R) was computed
for Voigt a = 0.01-0.2. The Ar resonance lines (4s2, 4s4 -> ground) have a ~ 1e-3-5e-3 at 0.5-1.5 Torr and
~2e-5 at 1 Pa (resonance broadening, `he.FindTauInModel`), so the model extrapolated log10(eta) linearly in a
below the table. In the Lorentz-wing regime eta grows like sqrt(a), so the extrapolation let the resonance
radiation escape too easily: eta too high by 1.47x (100 Pa, 400 K, R 2 cm), 1.23x (1 Torr, 300 K, R 4 cm),
1.64x (600 mTorr, 800 K). The new table uses the same method and geometry (it reproduces the old table within
1 % where both exist) for a = 1e-6-0.2 and tau0 = 1e-3-1e6, interpolated in (log10 tau, log10 a).

The Holstein-Walsh fundamental-mode escape factor (cylinder; what Bogaerts used) is another 1.4-3.4x lower
than the Monte Carlo first-flight value at these conditions. The resonance escape factor is the main
uncertainty of the 1s densities (below).

## Audit of every level (`audit_flags.md`, `audit_channels.csv`)

At the four Zhu & Pu discharges and at our 1 Torr conditions (Ne = n_c, Maxwellian 1.5 eV):
- **No level is without radiative decay** (all non-metastable levels radiate to tracked levels; no radiation
  to untracked levels), every level has electron-impact production and ionization, every budget closes
  (production = loss to < 1e-3), no level is above its Boltzmann population at Te.
- **5p (10 levels):** BSR has no 5p; the NGFSRDW ground -> 5p tables start 5.3-5.5 eV above threshold
  (sigma = 0 below) and 4s -> 5p starts at 5-6 eV (threshold ~2.8 eV). At our conditions 5p is fed mostly
  by cascades and 4s -> 5p, so 415.9/430.0 nm (fitted) carry this uncertainty.
- **4d, 6s (10 levels):** all electron-impact data are analytic (Drawin); they are populated 93-100 % by
  ground-state excitation from these estimates.
- At atmospheric pressure (SRR) the electron-impact losses of 3d, 5s, 5p are 70-100 % analytic.
- Minor: the metastable-metastable collision loss k_met n_meta is added to every level's loss in
  `SolveDirect`/`SolveLabelEquation`, not only the metastables (negligible for radiating levels).

## Zhu & Pu, J. Phys. D 43, 015204 (2010) (`zhupu_comparison.png`, `budget_2p_vs_zhupu.png`, `zhupu_levels.csv`)

Our model at their four discharges (their Table 7: EBP 1 Pa 10 eV, ICP 1 Pa 3 eV, CCP 100 Pa 1.5 eV,
SRR 1e5 Pa 1 eV; Maxwellian at their Te, their ne and Tg; trapping radius = half the smallest dimension).
Their Figs. 1-2 are digitised (`InputData/References/ZhuPu2010_digitized.csv`), their Figs. 3-4 (process
rates of 1s5, 1s4 and 2p1, 2p3, 2p6, 2p9) are transcribed in `SmallAnalysisScripts/LiteratureComparison.py`.

**2p distribution** (n/g, sum of the ten levels = 100; mean |ours - their CRM|):

| | EBP 1 Pa | ICP 1 Pa | CCP 100 Pa | SRR 1e5 Pa |
|---|---|---|---|---|
| model of 6 Oct morning (NGFSRDW, no atom transfer, old table) | 6.1 | 4.5 | 4.8 | 5.9 |
| current (BSR + atom transfer + v2 table) | 2.7 | 4.2 | 1.3 | 1.0 |
| current, box diffusion | 2.6 | 3.2 | 1.3 | 1.0 |

The BSR model reproduces their CRM and OES 2p distributions closely (CCP: 2p9 26 vs 25 OES, 2p10 27 vs
28-31, 2p8 11 vs 11). The old set overpopulated the primed levels by 2-3x.

**1s densities** (n/(g n_g) in ppm; their CRM / OES):

| | 1s5 | 1s3 | 1s4 | 1s2 |
|---|---|---|---|---|
| EBP 1 Pa, Zhu & Pu | 26 / 30 | 22 / 21 | 0.84 / 2.9 | 0.93 / 1.6 |
| ours, current | 1.9 | 2.1 | 0.18 | 0.38 |
| ours, box diffusion + Walsh | 14.8 | 18.5 | 0.84 | 2.3 |
| ICP 1 Pa, Zhu & Pu | 85 / 110 | 68 / 80 | 8.1 / 3.9 | 7.2 / 3.0 |
| ours, current | 47 | 33 | 1.0 | 1.2 |
| ours, box diffusion + Walsh | 69 | 49 | 8.4 | 10 |
| CCP 100 Pa, Zhu & Pu | 6.1 / 10.9 | 6.5 / 5.5 | 5.8 / 6.0 | 5.4 / 1.7 |
| ours, current | 2.6 | 1.6 | 1.4 | 0.59 |
| ours, box diffusion + Walsh | 3.3 | 2.1 | 2.2 | 0.90 |
| ours, morning model | 24 | 20 | 19 | 12 |

- At 1 Pa the 1s densities are set by diffusion to the walls and resonance escape. With their geometry
  (box diffusion length instead of our hemisphere) and Holstein-Walsh escape, the BSR model agrees with
  their CRM within ~0.6-1.4x (1s2 at EBP 2.5x). The current-model metastables are low there because the
  model's default geometry (hemisphere of R = half the smallest dimension) diffuses them 4x (ICP) to 8x
  (EBP) faster than their boxes; the resonant 1s4/1s2 need the lower (Walsh) escape factors as well.
- At 100 Pa (closest to our discharge) the BSR model is 2-5x low. Term by term (their Fig. 3 vs our
  budget, CCP): ground-state excitation of 1s5/1s4 agrees (36.8 vs 36, 32.2 vs 35 x 1e14 cm^-3 s^-1), wall
  loss agrees (1.2 vs 1.3), 1s -> 2p excitation per 1s atom is 0.7-0.9x theirs, and the BSR 1s <-> 1s
  electron-mixing rate coefficients are within 0.7-1.4x of the lumped ones they use (Maxwellian 1.5 eV:
  1s5->1s4 5.2e-8 vs 7.8e-8, 1s5->1s2 1.6e-8 vs 1.6e-8, 1s4->1s3 1.1e-8 vs 1.6e-8, 1s3->1s2 1.1e-7 vs
  7.8e-8 cm^3/s; NGFSRDW was 10-30x larger); but the **resonance-radiation loss per 1s4 atom is 5.9x
  theirs with the old table**
  (escape factor 1.2e-3 vs ~2e-4 implied by their numbers), ~4x with the v2 table and ~2x with Walsh. The
  missing 1s population then also lowers the stepwise 1s -> 2p excitation (4-6x below their Fig. 4) and the
  re-absorption of the 2p -> 1s lines.
- The morning model was 2-4x **high** at 100 Pa (NGFSRDW ground -> 1s 10-30x too large).
- SRR (1 atm): 10x low; our model has no Ar2+/Ar2* chemistry or recombination, which dominate there.

## Bogaerts, Gijbels & Vlcek, J. Appl. Phys. 84, 121 (1998) (`bogaerts_comparison.png`, `bogaerts_levels.csv`)

1000 V, 1 Torr, 2 mA dc glow. Their populations come from fast electrons, ions and atoms (Monte Carlo), so
only the pattern within each manifold is compared (n/g, sum 100): their Fig. 2 (4s at z = 1 cm, negative
glow), Fig. 4 (4p at the cathode-glow maximum), Fig. 6 (3d + 5s, negative glow), read off the scanned
figures; our model at 1 Torr, Tg 450 K, R 2 cm, ne 2e17 m^-3, Maxwellian 1-4 eV.
- 4p: same ordering (2p10 largest, 2p9/2p8/2p7/2p6 next, the primed 2p4-2p1 lowest), within ~2x level by
  level.
- 3d + 5s: within ~2x; BSR removed the old model's outlier at 3d[7/2]4 (25 vs their 3.5).
- 4s: our resonant levels (1s4, 1s2) are relatively more populated than in their negative glow, where the
  slow electrons (and their weaker 1s <-> 1s mixing) leave the resonant levels strongly depleted.

## Our discharge

### Pressure sweep (80 W, 600-1500 mTorr, Tg 800 K; `pressure_sweep_*`, `chi2_vs_Ne*.png`)

Rebuilt with the corrected model (BSR + atom transfer + v2 escape table):

| | DC EEDF | microwave EEDF |
|---|---|---|
| free fit: Ne | 1.6-3e19 (grid edge) | 3e19 (grid edge) |
| free fit: Te_eff, chi^2/dof | 3.5-3.7 eV, 0.4-1.2 | 0.9-1.1 eV, 0.5-1.4 |
| Ne = n_c: Delta chi^2/s^2 (BSR only, before AT + escape fix) | 26-33 (39-49) | 31-43 (34-47) |
| Ne = n_c: chi^2/dof | 1.2-2.0 | 1.6-2.5 |
| CR 1s5 at n_c | 6.8e17 | 2.5e16 |

No interior local minimum of chi^2(Ne) at any pressure (`chi2_vs_Ne_pressure_sweep.png`); the atom
transfer and the escape-factor fix lower the cost of n_c only modestly. With Holstein-Walsh escape factors
for the resonance lines (`chi2_vs_Ne_pressure_sweep_walsh.png`, `pressure_sweep_fits_walsh.csv`) the free
fits are unchanged (Ne at the grid edge, same Te_eff and chi^2/dof), the cost of n_c drops a little
further (DC 17-27, chi^2/dof at n_c 1.0-1.9; microwave 29-41) and the CR 1s5 at n_c rises (DC 8.5e17,
microwave 4.6e16); still no interior minimum.

### 1 Torr pure Ar (85 W, Tg 300 K; `pure_ar_1torr_profiles.png`, `pure_ar_1torr_summary.csv`)

| EEDF, escape | free Ne | chi^2/dof | Delta chi^2/s^2 at n_c | CR 1s5 at n_c [m^-3] (absorption: 1-8e17) |
|---|---|---|---|---|
| microwave, MC table | 3e19 | 0.71 | 46 | 8.5e16 |
| microwave, Walsh (resonance lines) | 3e19 | 0.71 | 48 | 1.2e17 |
| DC, MC table | 3e19 | 0.43 | 30 | 6.4e17 |
| DC, Walsh (resonance lines) | 3e19 | 0.43 | 29 | 7.3e17 |

With the corrected model the CR 1s5 at n_c is inside the absorption range for both EEDFs (microwave needs
the Walsh escape factors), and along the chi^2 profile it stays in that range from ~1e16 (DC) or ~1e17
(microwave) up to 3e19 m^-3, so the absorption is consistent with n_c but does not pin Ne by itself. The
escape-factor treatment does not change the line-ratio preference for high Ne.

### What drives the high Ne (`measured_2p_vs_zhupu.png`, `line_subset_tests.csv`)

Our measured 2p distribution (seven fitted levels, n/g from the calibrated line areas with the model's
escape factors at n_c) next to the distribution Zhu & Pu **measured** in their 100 Pa CCP and to our model
at n_c (DC EEDF), sum of the seven = 100, 1000 mTorr:

| level (line) | 2p1 (750.4) | 2p5 (751.5) | 2p3 (706.7) | 2p8 (801.5) | 2p4 (794.8) | 2p2 (727.3) | 2p6 (800.6) |
|---|---|---|---|---|---|---|---|
| our measurement | 5.6 | 9.2 | 7.5 | 37.9 | **20.0** | 5.6 | **14.3** |
| our CR model, Ne = n_c | 6.1 | 9.4 | 6.3 | 32.6 | 10.1 | 9.2 | 26.3 |
| Zhu & Pu CCP, measured | 9.2 | 11.7 | 7.8 | 30.9 | 9.8 | 6.9 | 23.6 |

At n_c our model reproduces the literature measurement; our measurement differs from both mainly in
2p4 (794.8 nm, 2x high) and 2p6 (800.6 nm, ~0.6x), the same at 600 and 1500 mTorr. Refitting with line
subsets (Delta chi^2/s^2 at n_c, 1000 mTorr):

| lines | DC | microwave |
|---|---|---|
| all 9 | 31 | 42 |
| without 794.8 | 20 | 40 |
| without 800.6 | 11 | 30 |
| without 794.8 and 800.6 | 8.5 | 28 |
| without the primed-core lines 706.7, 727.3, 750.4, 794.8 | 4.5 | **0.4** |

So the preference for Ne ~ 1e19 comes entirely from the relative populations of the primed-core 2p
levels (2p1-2p4) and 800.6 nm; without the primed lines the microwave fit accepts n_c at every pressure
tested and the DC fit nearly so. At high Ne the model mixes the 2p levels by electron collisions, which
is how it mimics the measured primed-core populations. Candidates:
- measurement: **794.8 nm sits 2.8 nm after an echelle order gap (782.0-792.0 nm), at the very start of
  an order - the same position as 852.1 nm (3.4 nm after the 835.7-848.8 nm gap), which is already
  excluded for its unreliable response at the order edge.** The fitted lines otherwise sit 6-7.5 nm from
  the gaps. The same-upper-level calibration check of 30 Sep found 794.8/852.1 nm (both from 2p4) off by
  +0.6 in ln and attributed it to self-absorption of 852.1 nm; the check below (this sweep, current model)
  gives 794.8/852.1 = 1.45x (+0.37 in ln) with thin lines or the free-fit escape factors. An order-edge
  response error at 794.8 nm would explain part of the 2p4 excess; the rest comes from the model's strong
  trapping of 794.8 nm at n_c (next section). Worth checking with the lamp calibration around 792-797 nm,
  or excluding 794.8 nm like 852.1 nm;
- physics of this discharge that feeds the primed core more than in a 100 Pa CCP (higher 1s3/1s2
  densities, e.g. through weaker resonance escape of 104.8 nm; cascades from the primed 3d'/5s' levels).
  In the CR model the 1s3/1s5 ratio is 0.6 (n/g) where Zhu & Pu's model gives 1.07 and Bogaerts' 0.8.

**Same-upper-level check** (`branching_check.png`, `branching_check.csv`; all 95 spectra of the sweep):
lines from one upper level must give the same I lambda / (A eta), whatever populates the level.
ln(value / group mean), optically thin (eta = 1) / with the model's eta at the free fit / at n_c (DC):

| upper | line [nm] | thin | eta free fit | eta at n_c | model eta at n_c |
|---|---|---|---|---|---|
| 2p2 | 727.3 | -0.08 | -0.24 | -0.41 | 0.91 |
| 2p2 | 772.4 | +0.03 | +0.05 | +0.37 | 0.46 |
| 2p2 | 826.5 | +0.05 | +0.19 | +0.04 | 0.66 |
| 2p4 | 794.8 | **+0.16** | **+0.27** | **+0.73** | 0.33 |
| 2p4 | 852.1 | -0.22 | -0.12 | -0.33 | 0.66 |
| 2p4 | 747.1 | +0.07 (noisy, sd 0.55) | -0.17 | -0.44 | 1.00 |
| 2p1 | 750.4 | +0.16 | +0.29 | +0.32 | 0.72 |
| 2p1 | 667.7 | -0.16 | -0.29 | -0.32 | 1.00 |

- The 2p2 lines agree within ~8 % if optically thin; the model's trapping at the DC n_c point (1s
  densities high there) spreads them by e^0.8. The data favour weaker trapping of the 4p -> 4s lines than
  the DC fit at n_c implies.
- 794.8 nm is 1.45x high relative to 852.1 nm (thin or free-fit eta); 852.1 nm is also at an order start,
  so the pair cannot tell which of the two is off. At n_c the model's escape factor for 794.8 nm (0.33,
  strong trapping on 1s3) inflates the inferred 2p4 population further: part of the "2p4 excess" at n_c is
  the model's trapping of 794.8 nm, not only the population.

## Open issues

1. **794.8 nm (and 852.1 nm) at the start of an echelle order**: check the response there (lamp
   calibration around 792-797 and 849-855 nm); refit without 794.8 nm.
2. **Trapping of the 4p -> 4s lines**: the same-upper-level check favours weaker trapping than the CR model
   gives at the DC n_c point. The 1s densities that set it depend on the resonance escape factor:
   Monte Carlo first-flight (uniform hemisphere) vs Holstein fundamental mode differ by ~2x
   (`escape_mode='walsh'`: Holstein-Walsh for the lines to the ground state only, as Bogaerts et al. do,
   blended with the table between tau = 3 and 30; `EscapeFactorInterpolator.resonance`, used by
   `SolveDirect` for those lines). A first version applied Walsh to every line with tau > 10 through a hard
   switch, which left ~12 % of the grid unconverged at E/N >= 100 Td; those results were discarded.
3. 5p levels: no BSR data; the NGFSRDW ground -> 5p and 4s -> 5p tables start well above threshold.
4. 4d, 6s: analytic cross sections only.
5. Ne from the line ratios stays at the grid edge until the primed-core 2p populations are understood;
   until then Ne should come from n_c / absorption / power balance, with Te_eff fitted at that Ne.

## Regenerate

| Step | Command | Time |
|---|---|---|
| BSR cross-section JSON | `python Scripts/ParserforBSRData.py` | s |
| Escape-factor table v2 | `python Scripts/ExtendTrappingLookup.py` | ~3 min |
| Literature comparison + audit | `python SmallAnalysisScripts/LiteratureComparison.py` | ~1 min |
| Pressure sweep | `python SmallAnalysisScripts/PressureSweepAnalysis.py` (Holstein-Walsh: set `CR_ESCAPE_MODE=walsh`) | ~30 min with table builds |
| 1 Torr pure Ar | `python SmallAnalysisScripts/PureArgon1TorrCheck.py` | ~15 min |

Outputs go to `Experimental_Data/Output/...` (git-ignored); the figures and tables here are copies.
