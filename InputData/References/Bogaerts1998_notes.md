# Bogaerts, Gijbels & Vlcek, J. Appl. Phys. 84, 121 (1998): notes

"Collisional-radiative model for an argon glow discharge". 65 levels (4s and 4p individual,
n = 2-5 and 6-15; higher levels lumped into effective levels). Coupled to MC/fluid models of a
1000 V, 1 Torr, 2 mA dc glow (Tg = 450 K). The PDF is not in the repo (copyright).

## Heavy-particle excitation / de-excitation between excited levels

All of it is **empirical/analytic**. None of it is measured state-to-state data.

**Fast Ar+ and Ar atoms (Sec. III.B).** These use the Drawin-type ionization formula (Vlcek, ref. 21;
Drawin & Emard, Phys. Lett. A 43, 333 (1973), ref. 50) with E_ioniz replaced by E_nm and
multiplied by the oscillator strength f_nm, so the rates are nonzero only for optically allowed
transitions. Excitation from the ground state uses Phelps' data instead.

**Thermal Ar atoms (Sec. III.C).** Allowed and forbidden transitions both use

    sigma_exc(n -> m, E) = b_nm (E - E_nm),   E = E_thermal ~ 0.06 eV (Tg = 450 K)
    b_nm = 8.69e-18 * E_nm^-2.26   [cm^2/eV, E_nm in eV]      (Vlcek, ref. 21; fitted to experiments)

De-excitation comes from detailed balance. Excitation is nonzero only for E_nm < 0.06 eV, i.e.
between closely spaced levels.

- 4s <-> 4s uses its own b_nm (from Tachibana, Phys. Rev. A 34, 1007 (1986), ref. 53):
  - b_23 = b_45 = 1.79e-20 E_nm^-2.26 (1s5 -> 1s4, 1s3 -> 1s2)
  - b_24 = b_25 = b_34 = b_35 = 4.8e-22 E_nm^-2.26
  - "No further intercombination transitions between primed and unprimed system were considered."
- Thermal-atom ionization and three-body recombination use the same formula as for fast atoms.
  Ionization is zero below level 57.
- The paper's results show this matters mostly for the highly excited levels (Table II).

## What this formula gives for the 4p (2p) levels

Evaluated in this repo at E = 0.06 eV with v = sqrt(2E/mu), upward rate k = sigma*v:

| pair | dE [eV] | sigma [cm^2] | k_up [cm^3/s] |
|---|---|---|---|
| 2p5 -> 2p4 | 0.010 | 1.4e-14 | 1.1e-9 |
| 2p9 -> 2p8, 2p7 -> 2p6, 2p4 -> 2p3 | 0.019 | 2.8e-15 | 2.1e-10 |
| 2p3 -> 2p2 | 0.026 | 1.1e-15 | 8.6e-11 |
| 2p5 -> 2p3 | 0.029 | 8.0e-16 | 6.1e-11 |
| 2p4 -> 2p2 | 0.045 | 1.4e-16 | 1.1e-11 |
| 2p8 -> 2p7, 2p5 -> 2p2 | ~0.056 | ~1-3e-17 | ~1-2e-12 |

- **2p1 (750.4 nm) is not coupled at all**, because it lies 0.15 eV above 2p2.
- 2p10 is not coupled either.
- The power law blows up as dE -> 0, and the threshold depends on a single fixed atom energy
  rather than a thermal average. For 2p mixing, measured state-to-state rate coefficients for
  Ar(4p) + Ar are the better choice (Setser/Sadeghi group, as used in later low-pressure Ar 2p
  CR models).

## Other heavy-particle processes they included (4s metastables)

- Two- and three-body quenching by Ar (Ar2* formation).
- Metastable-metastable collisions, k = 6.4e-10 cm^3/s (Ferreira et al., ref. 31).
- Penning ionization of sputtered Cu.
- Metastable diffusion coefficient D = 74.6 cm^2/s at 1 Torr, 300 K.

## Radiation trapping

- Escape factors are applied only to lines to the ground state (Holstein/Walsh-type formula for a
  cylinder, combined Doppler and collisional broadening).
- Lines to excited levels get escape factor 1.
- A values: Vlcek (ref. 21); the 4s-4p and 4s-5p values are from Wiese et al.
