# Zhu & Pu, J. Phys. D: Appl. Phys. 43, 015204 (2010): notes

"A simple collisional-radiative model for low-temperature argon discharges with pressure ranging
from 1 Pa to atmospheric pressure: kinetics of Paschen 1s and 2p levels". Individual 1s and 2p
levels plus lumped 2s3d, 3p and hl levels, Ar2*, Ar+ and Ar2+. Tested against OES (1s densities
by self-absorption, 2p by line ratios) for four cases: EBP 1 Pa, ICP 1 Pa, CCP 100 Pa and
microwave SRR at 1e5 Pa. The PDF is not in the repo (copyright).

## Table 3: atom-collision population transfer (Ar + Ar* -> Ar + Ar*)

All rate coefficients scale as k * (Tg/300)^0.5 cm^3/s. Inverse rates come from detailed balance.
The listed direction is downhill in energy.

| # | Process | k(300 K) [cm^3/s] | Source |
|---|---|---|---|
| 9 | 2p1 -> 1s | 3e-11 | [36,37] |
| 10 | 2p2 -> 1s | 1e-11 | [23] (from experiment) |
| 11 | 2p3 -> 1s | 3e-11 | [38] |
| 12 | 2p4 -> 1s | 3e-11 | [38] |
| 13 | 2p7 -> 1s | 4e-11 | [38] |
| 14 | 2p8 -> 1s | 4e-11 | [38] |
| 15 | 2p9 -> 1s | 3e-11 | [38] |
| 16 | 2p10 -> 1s | 1.5e-11 | [38] |
| 17 | 2p3 -> 2p4 | 2e-11 | [38,39] |
| 18 | 2p3 -> 2p6 | 2e-11 | [38,39] |
| 19 | 2p5 -> 2p6 | 2.5e-11 | [36,37,39] |
| 20 | 2p5 -> 2p8 | 1.5e-11 | [36,37,39] |
| 21 | 2p6 -> 2p7 | 2e-11 | [37-39] |
| 22 | 2p6 -> 2p8 | 1e-11 | [37-39] |
| 23 | 2p7 -> 2p8 | 1e-11 | [38,39] |
| 24 | 2p7 -> 2p9 | 2e-11 | [38,39] |
| 25 | 2p8 -> 2p9 | 2.5e-11 | [38,39] |
| 26 | 2p9 -> 2p10 | 3e-11 | [38,39] |
| 27 | 3p -> 2p | 1e-10 | [36] |
| 28 | 3p -> 2s3d | 1e-10 | [36] |
| 29, 30 | 2s3d -> 2p, hl -> 2s3d/3p | = row 28 | assumed |

- No 2p -> 1s entry for 2p5 or 2p6.
- The table does not say how "1s" is split among 1s2-1s5.
- No 2p -> 2p channel is listed out of 2p1 or 2p2.

Sources:
- [36] Inoue, Setser & Sadeghi, J. Chem. Phys. 76, 977 (1982)
- [37] Sadeghi et al., J. Chem. Phys. 115, 3144 (2001)
- [38] Chang & Setser, J. Chem. Phys. 69, 3885 (1978)
- [39] Nguyen & Sadeghi, Phys. Rev. A 18, 1388 (1978)
- [23] Zhu et al., J. Phys. D 42, 142003 (2009)

**Used in the CR model** since 4 Oct 2026: rows 9-26 are in `InputData/Ar_2p_atom_transfer.csv`.
The 2p -> 1s rate is shared among 1s2-1s5 by g, and uphill rates come from detailed balance at Tg
(he.AtomTransferRates, AttachAtomTransfer in MainFileV2; switch: CRModel(atom_transfer=...)).
Rows 27-30 (lumped 3p / 2s3d / hl levels) are not used.
Results: `Experimental_Data/Results/2026-10-04_Ar_atom_2p_transfer/`.

## Digitised figures

`InputData/References/ZhuPu2010_digitized.csv` has their Fig. 2 and Fig. 1, from pixel analysis of
the page images:
- Fig. 2: n_2p/g_2p of the ten 2p levels, normalised to a sum of 100; read to about +-0.3.
- Fig. 1: n_1s/g_1s/n_g in ppm; read to about +-3 %.

Both figures have CRM and OES values for EBP 1 Pa, ICP 1 Pa, CCP 100 Pa and SRR 1e5 Pa (SRR has no
1s OES). `SmallAnalysisScripts/ZhuPuComparison.py` uses them.

## Other rates useful here (Table 3, electron rows)

- e + Ar(2p) <-> e + Ar(2p), lumped: 4e-7 Te^-0.6 cm^3/s (Pokrzywka 2002).
- 1s5 <-> 1s4 and 1s3 <-> 1s2: 1e-7 Te^-0.6 cm^3/s.
- 1s3 <-> 1s4 and 1s5 <-> 1s2: 2e-8 Te^-0.6 cm^3/s (Bartschat & Zeman 1999).

## Relevance to our 1 Torr data

- Their CCP case (100 Pa, Te = 1.5 eV, ne = 3e11 cm^-3 = 3e17 m^-3, Tg = 400 K) is close to our
  conditions. With these processes and that ne, their model reproduces the measured 2p
  distribution (Fig. 2c).
- At 100 Pa they put the 2p levels in the "M-regime": spontaneous radiation still dominates 2p
  loss, so atom-collision transfer is a correction, not the main loss.
