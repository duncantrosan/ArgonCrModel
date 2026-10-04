# EEDF shape and the fitted Ne (4 Oct 2026)

**Question:** is Te a poor description of the microwave EEDFs, and should the libraries go to
lower E/N?

## Te_eff is a poor label for these EEDFs (`eedf_shapes.png`)

Te_eff = 2/3 <E> is set by the slow electrons. The line excitation uses electrons above 11.5 eV.
Stepwise excitation out of the 1s levels uses electrons above ~1.5 eV.

The microwave and DC fits of the same spectrum have very different EEDFs:

| Spectrum | Fit | E/N | Te_eff | Slope T, 1-4 eV | Slope T, 12-15 eV |
|---|---|---|---|---|---|
| 2.4 % N2, 85 W | microwave | 19 Td | 0.55 eV | 0.14 eV | 0.37 eV |
| 2.4 % N2, 85 W | DC | 4.4 Td | 0.91 eV | 0.52 eV | 0.15 eV |
| pure Ar | microwave | 7.5 Td | 1.08 eV | 1.0 eV | 0.16 eV |
| pure Ar | DC | 2.9 Td | 2.61 eV | flat | 0.10 eV |

- **N2 vibrational barrier:** in the Ar/N2 microwave EEDF the EEPF drops ~7 decades between 2 and
  4 eV.
- **Te_eff is not monotonic:** along the pure-Ar microwave library it goes 0.70 -> 1.18 -> 0.86 eV
  between 6 and 20 Td.
- **Consequence:** Te_eff values are not comparable between EEDF families. Quote E/N, or the
  rates that matter.

## Where the fits sit on the library grids

| Fit | Position on the grid | Lower E/N needed? |
|---|---|---|
| Microwave Ar/N2 | 10-25 Td, inside the 6-1000 Td library | No: below ~8 Td these EEDFs have no electrons above 10-11 eV |
| Pure-Ar microwave | 7.5 Td, at the 6 Td floor (edge); n_c fit at 6.8 Td | Yes |
| DC Ar/N2 | 3.6-5 Td, at the 3 Td floor (edge) | Yes |
| Pure-Ar e-e | at their lowest E/N (DC + e-e 1 Td, microwave + e-e 6 Td) | Yes |

The pure-Ar fits below also stop at Ne = 3e18 m^-3, because the e-e libraries end there. The plain
EEDFs reach that cap.

## With e-e collisions the fitted Ne changes (`chi2_profiles.png`, `summary.csv`)

Pure-Ar spectrum, chi^2 profile over Ne (minimised over E/N):

| EEDF library | Best Ne [m^-3] | chi^2/dof | Delta chi^2/s^2 at n_c | CR 1s5 at the fit [m^-3] |
|---|---|---|---|---|
| microwave | >= 2e18 (grid top) | 0.49 | 51 | 7e15 |
| microwave + e-e | >= 2e18 (grid top) | 0.50 | 50 | 1e16 |
| DC | >= 2e18 (grid top) | 0.52 | 49 | 2e13 |
| DC + e-e | 2.8e17 | 1.43 | 22 | 5.8e17 |
| DC + e-e + superelastic | 2.6e17 | 1.42 | 21 | 5.6e17 |

The absorption estimate of 1s5 is 1-8e17 m^-3.

- **DC + e-e:** with an EEDF that depends on Ne (e-e collisions at ionization degree Ne/N), the
  fit moves to Ne ~ 3e17 m^-3. Its 1s5 density then matches the absorption measurement.
- **But:** the line fit is worse (chi^2/dof 1.4 vs 0.5), and it sits at the lowest E/N of the
  library (1 Td).
- **Microwave + e-e:** sits at its 6 Td floor.

## Next step (needs BOLSIG+, which runs on the local machine)

`python Scripts/CreateEEDFLibraries.py` builds the two missing libraries (a few minutes):
- `Ar_Biagi_bolsig_ee_lowEN`: DC + e-e down to 0.3 Td.
- `Ar_Biagi_bolsig_mw_ee_lowEN`: microwave + e-e down to 1 Td.

Both use Ne = 1e16-3e18.

Then run `python SmallAnalysisScripts/EEDFLibraryNeFits.py`. The CR tables take ~10 min per new
library.

- **If the low-E/N e-e fits reach chi^2/dof ~0.5 at Ne ~1e17-1e18:** the high Ne was an artifact
  of EEDFs without e-e collisions. The Ar/N2 fits should then also get E/N x Ne (e-e) libraries.
