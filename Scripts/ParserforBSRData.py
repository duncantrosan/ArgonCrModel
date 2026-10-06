"""
ParserforBSRData.py

Builds InputData/ArgonCrossSections_BSR.json, the electron-impact excitation set the CR model
reads (HelperFunctions.CROSS_SECTION_FILE), from

  BSR      InputData/ArgonLxCatPureArgonUpdated.txt (LXCat BSR database: BSR-500 B-spline R-matrix,
           Zatsarinny & Bartschat). Ground -> 4s, 4p, 3d, 5s and all channels among 4s, 4p, 3d, 5s,
           every table tabulated from threshold.
  NGFSRDW  InputData/ArgonCrossSections.json (ParserforLXCatData.py, distorted wave) for the pairs
           BSR does not have - everything into 5p, 4d, 6s. These still start above threshold
           (e.g. ground -> 5p from 20 eV, 4s -> 5p from 5-6 eV; sigma = 0 below in the model).

BSR states are named in jK coupling, e.g. 'Ar(4p'[1/2]0)' or 'Ar(4p![1/2]0, 1S0, p1)' ('!' = prime,
core 2P1/2); the label is the level of ArgonLevelList.json with the same manifold, core j, K and J.
'Ar(Rydberg)' and 'Ar(nl>5s)' are lumps of untracked levels (labels None, ignored by the model).

BSR thresholds are BSR's own level energies, 0.07-0.22 eV above NIST for 4p, 3d and 5s. Each BSR
table is shifted in energy so that its threshold equals the model's level spacing
(E_upper - E_lower of the level list, which the detailed-balance de-excitation also uses); the
shift is stored per entry ('energy_shift_eV'). BSR pairs whose model spacing is <= 0 (order of
two near-degenerate levels differs between BSR and NIST) are left out and listed.
The file holds the BSR-500 ground-state set twice (BSR-500 block and the PRA 89, 022706 (2014)
block, equal within a few %); the first block of each pair is kept.

Run: python Scripts/ParserforBSRData.py
"""
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'InputData'
BSR_TXT = DATA / 'ArgonLxCatPureArgonUpdated.txt'
NGFSRDW_JSON = DATA / 'ArgonCrossSections.json'
LEVELS_JSON = DATA / 'ArgonLevelList.json'
OUT_JSON = DATA / 'ArgonCrossSections_BSR.json'

_JK = re.compile(r"Ar\((\d[spd])\s*(['!]?)\s*\[(\d+/\d+)\](\d+)")


def bsr_label(name, levels):
    """Model label of a BSR state name ('ground' for 'Ar'); None for lumps and unknown states."""
    name = name.strip()
    if name == 'Ar':
        return 'ground'
    m = _JK.search(name)
    if m is None:
        return None
    nl, prime, K, J = m.groups()
    core = '<1/2>' if prime else '<3/2>'
    cand = [lab for lab, v in levels.items()
            if v.get('manifold') == nl and core in v['configuration'] and f'[{K}]' in v['term']
            and float(v['J']) == float(J)]
    if len(cand) > 1:
        raise ValueError(f'{name}: matches {cand}')
    return cand[0] if cand else None


def parse_bsr(path, levels):
    """Excitation blocks of the BSR file -> list of entries in the ArgonCrossSections.json format."""
    lines = Path(path).read_text(errors='replace').splitlines()
    out, seen, skipped = [], set(), []
    i = 0
    while i < len(lines):
        if lines[i].strip() != 'EXCITATION':
            i += 1
            continue
        reaction = lines[i + 1].strip()
        lo_raw, up_raw = [s.strip() for s in re.split(r'<?->', reaction, maxsplit=1)]
        nums = [float(x) for x in lines[i + 2].split()]
        j = i + 3
        while not lines[j].strip().startswith('-----'):
            j += 1
        j += 1
        energy, sigma = [], []
        while not lines[j].strip().startswith('-----'):
            parts = lines[j].split()
            if len(parts) >= 2:
                energy.append(float(parts[0]))
                sigma.append(float(parts[1]))
            j += 1
        i = j + 1

        lo, up = bsr_label(lo_raw, levels), bsr_label(up_raw, levels)
        if (lo, up) in seen and lo is not None and up is not None:
            continue                                   # second copy of the ground-state set
        seen.add((lo, up))
        thr_bsr = nums[0]
        entry = dict(type='excitation', source='BSR-500', lxcat_reaction=reaction, lxcat_lower=lo_raw,
                     lxcat_upper=up_raw, lower_label=lo, upper_label=up, threshold_eV=thr_bsr,
                     threshold_bsr_eV=thr_bsr, energy_shift_eV=0.0,
                     g_ratio=nums[1] if len(nums) > 1 else None, energy_eV=energy, cross_section=sigma)
        if lo is not None and up is not None:
            dE = levels[up]['energy_eV'] - (0.0 if lo == 'ground' else levels[lo]['energy_eV'])
            if dE <= 0:
                skipped.append(f'{reaction}  (model spacing {dE:+.4f} eV)')
                continue
            shift = dE - thr_bsr
            entry.update(threshold_eV=dE, energy_shift_eV=shift,
                         energy_eV=[max(e + shift, dE) for e in energy])
        out.append(entry)
    return out, skipped


def build(out_json=OUT_JSON):
    levels = json.loads(LEVELS_JSON.read_text())
    bsr, skipped = parse_bsr(BSR_TXT, levels)
    have = {(c['lower_label'], c['upper_label']) for c in bsr}
    old = json.loads(NGFSRDW_JSON.read_text())['cross_sections']
    fallback = [dict(c, source='NGFSRDW') for c in old
                if c['lower_label'] is not None and c['upper_label'] is not None
                and (c['lower_label'], c['upper_label']) not in have]
    data = dict(description=__doc__.strip().split('\n\n')[1],
                sources=dict(BSR=BSR_TXT.name, NGFSRDW=NGFSRDW_JSON.name),
                cross_sections=bsr + fallback, bsr_skipped_reversed_order=skipped)
    Path(out_json).write_text(json.dumps(data, indent=1))
    return data


if __name__ == '__main__':
    d = build()
    cs = d['cross_sections']
    tracked = [c for c in cs if c['lower_label'] is not None and c['upper_label'] is not None]
    n_bsr = sum(c['source'] == 'BSR-500' for c in tracked)
    gap = [c for c in tracked if c['energy_eV'][0] - c['threshold_eV'] > 0.5]
    shifts = [c['energy_shift_eV'] for c in tracked if c['source'] == 'BSR-500']
    print(f"{len(tracked)} tracked channels: {n_bsr} BSR-500, {len(tracked) - n_bsr} NGFSRDW fallback "
          f"({len(cs) - len(tracked)} BSR lumps of untracked levels) -> {OUT_JSON}")
    print(f"BSR threshold shifts to the model level energies: {min(shifts):+.3f} to {max(shifts):+.3f} eV")
    print(f"channels whose table starts > 0.5 eV above threshold (all NGFSRDW): {len(gap)}")
    for r in d['bsr_skipped_reversed_order']:
        print(f"  BSR pair left out, reversed level order in the model: {r}")
