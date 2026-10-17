# EEDF options

Written by `CreateEEDFScheme.py` - do not edit by hand: edit the scheme files (`<name>.toml` in this folder) and run `python CreateEEDFScheme.py --list`.

Use one in a Main.py input file:

```toml
[eedf]
model = "<name>"
```

## Built into Main.py

- **pure_Ar** - Pure-Ar microwave BOLSIG+ EEDF (Biagi Ar, 2.42 GHz, 42 E/N 3-1000 Td, Tg of the condition); the N2 only acts in the CR model (quenching, dilution).  Libraries Ar_Biagi_bolsig_mw2.42GHz_<Tg>K_<p>mTorr.
- **ArN2_vib_saturated** - Ar/N2 microwave BOLSIG+ EEDF (Biagi Ar/N2) with the N2 vibrational excitations left out (saturated N2(v), no net e-V loss).  Libraries ArN2_<x>pct_bolsig_mw..._vibsat.
- **ArN2_vib_Tv** - Ar/N2 microwave BOLSIG+ EEDF with a reversible N2(v) ladder in a Boltzmann distribution at [eedf] Tv_K (or Tv per condition from the LAS 1s5 with Tv_K = "LAS", scanned over Tv_grid_K).  Libraries ArN2_<x>pct_bolsig_mw..._Tv<Tv>K.

## Schemes made with CreateEEDFScheme.py

(none yet)
