import re, json
from pathlib import Path

# Paschen designation -> model level label (energy-ordered within manifold)
PASCHEN_TO_LABEL = {
    # 4s manifold (Paschen 1s)
    '1s5': '4s1', '1s4': '4s2', '1s3': '4s3', '1s2': '4s4',
    # 4p manifold (Paschen 2p, 2p10 = lowest energy)
    '2p10': '4p1', '2p9': '4p2', '2p8': '4p3', '2p7': '4p4', '2p6': '4p5',
    '2p5':  '4p6', '2p4': '4p7', '2p3': '4p8', '2p2': '4p9', '2p1': '4p10',
    # 3d manifold (Paschen 3d, 3d12 = lowest energy)
    '3d12': '3d1', '3d11': '3d2', '3d10': '3d3', '3d9': '3d4',
    '3d8':  '3d5', '3d7':  '3d6', '3d6':  '3d7', '3d5': '3d8',
    '3d4':  '3d9', '3d3': '3d10', '3d2': '3d11', '3d1': '3d12',
    # 5s manifold (Paschen 2s)
    '2s5': '5s1', '2s4': '5s2', '2s3': '5s3', '2s2': '5s4',
}

# The 5p names in the NGFSRDW file are not reliable: 3p2/3p3 swap J between the
# ground-state and 4s blocks, one ground-state target is labelled '3d8', and one
# '5p 3p9' target is really 2p9 (threshold 0.1687 eV from 2p10). These targets are
# identified by J and energy (lower-level energy + threshold) instead of the name.
ENERGY_MATCHED_MANIFOLDS = {'5p'}


def resolve_label(lxcat_name):
    """
    Map an lxcat state string like 'Ar(3p5 4p J = 1  2p10)' to a model label.
    Handles the ground state ('Ar(3p6 J = 0)'), stray '<' characters left
    over from splitting '<->', inconsistent spacing, and lowercase 'j'.
    Returns None for 5p states (see resolve_by_energy) and unknown names.
    """
    if lxcat_name is None:
        return None
    name = lxcat_name.strip().rstrip('<').strip()

    if name.startswith('Ar(3p6'):
        return 'ground'

    # pull out the contents of the parentheses
    try:
        inner = name[name.index('(') + 1: name.rindex(')')]
    except ValueError:
        return None
    tokens = inner.split()
    if len(tokens) < 3:
        return None

    manifold = tokens[1]          # '4s', '4p', '3d', '5s', '5p'
    paschen  = tokens[-1]         # '1s5', '2p10', '3d12', '2s5', '3p9', ...

    if manifold in ENERGY_MATCHED_MANIFOLDS:
        return None               # resolved by resolve_by_energy in parse_lxcat
    return PASCHEN_TO_LABEL.get(paschen)


def resolve_by_energy(lxcat_name, E_target, levels, tol_eV=5e-3):
    """Model label with the J in lxcat_name and energy within tol_eV of E_target."""
    m = re.search(r'J\s*=\s*(\d+)', lxcat_name)
    if m is None:
        return None
    cand = [lab for lab, v in levels.items()
            if abs(v['energy_eV'] - E_target) < tol_eV and float(v['J']) == float(m.group(1))]
    return cand[0] if len(cand) == 1 else None


def parse_lxcat(path, levels=None):
    """
    Parse an LXCat cross-section text file into a CrossSectionList dict.

    levels : the model level list (ArgonLevelList.json); needed to resolve 5p
             targets by energy. Exact duplicate channels are kept once.
    """
    with open(path) as f:
        lines = f.read().splitlines()

    cross_sections = []
    crosswalk = {}
    seen, duplicates = set(), []
    i = 0
    while i < len(lines):
        if lines[i].strip() != 'EXCITATION':
            i += 1
            continue

        # --- reaction line ---
        reaction = lines[i + 1].strip()
        if '<->' in reaction:
            lo_raw, up_raw = reaction.split('<->')
        else:
            lo_raw, up_raw = reaction.split('->')
        lo_raw, up_raw = lo_raw.strip(), up_raw.strip()

        # --- threshold line: one number (->) or two (<->; second is g_up/g_lo) ---
        nums = [float(x) for x in lines[i + 2].split()]
        threshold_eV = nums[0]
        g_ratio = nums[1] if len(nums) > 1 else None

        # --- skip metadata until the first dashed line ---
        j = i + 3
        while not lines[j].strip().startswith('-----'):
            j += 1
        j += 1

        # --- numeric table until the closing dashed line ---
        energy, sigma = [], []
        while not lines[j].strip().startswith('-----'):
            parts = lines[j].split()
            if len(parts) >= 2:
                energy.append(float(parts[0]))
                sigma.append(float(parts[1]))
            j += 1

        lower_label = resolve_label(lo_raw)
        upper_label = resolve_label(up_raw)
        if (upper_label is None and levels is not None and lower_label is not None
                and ' 5p ' in up_raw):
            upper_label = resolve_by_energy(up_raw, levels[lower_label]['energy_eV'] + threshold_eV,
                                            levels)
        crosswalk.setdefault(lo_raw, lower_label)
        crosswalk.setdefault(up_raw, upper_label)

        pair = (lower_label, upper_label)
        if lower_label is not None and upper_label is not None and pair in seen:
            duplicates.append(reaction)
            i = j + 1
            continue
        seen.add(pair)

        cross_sections.append({
            'type':           'excitation',
            'lxcat_reaction': reaction,
            'lxcat_lower':    lo_raw,
            'lxcat_upper':    up_raw,
            'lower_label':    lower_label,
            'upper_label':    upper_label,
            'threshold_eV':   threshold_eV,
            'g_ratio':        g_ratio,        # g_upper/g_lower, only on <-> blocks
            'energy_eV':      energy,
            'cross_section':  sigma,
        })
        i = j + 1

    return {'cross_sections': cross_sections, 'crosswalk': crosswalk,
            'duplicates_skipped': duplicates}


def build_cross_sections_json(in_txt, out_json, levels_json=None):
    levels = None
    if levels_json is not None:
        with open(levels_json) as f:
            levels = json.load(f)
    data = parse_lxcat(in_txt, levels)
    with open(out_json, 'w') as f:
        json.dump(data, f, indent=2)
    return data


# --- BSR database (B-spline R-matrix, Zatsarinny & Bartschat; www.lxcat.net/BSR) ---------
# States are named in jK coupling: 'Ar(4s[3/2]2)', "Ar(4p'[1/2]0)" (prime = 2P1/2 ion core),
# and in the BSR-500 ground-state set also 'Ar(4s![1/2]0, 3P0, 1s3)' ('!' = prime) or
# 'Ar(4p [1/2]0, 3P0, p5)'. They are matched to the model levels on (core, nl, K, J).
# The tabulated thresholds are BSR's own level energies: the 4s levels agree with NIST, the
# 4p levels lie 0.11-0.17 eV and the 3d levels up to 0.22 eV above NIST. The model uses the
# NIST energies (ArgonLevelList.json), so each cross section is shifted in energy to start
# at the model threshold (same cross section at the same energy above threshold).
_BSR_STATE = re.compile(r"^Ar\((\d+[spdfg])\s*(['!]?)\s*\[(\d+/2)\]\s*(\d+)")


def bsr_level_lookup(levels):
    """(ion core '3/2' or '1/2', nl, K, J) -> model label, for the excited levels."""
    lookup = {}
    for label, v in levels.items():
        parts = v['configuration'].split('.')
        if v.get('kind') == 'ground' or len(parts) < 2 or '<' not in parts[-2]:
            continue
        core = '1/2' if '<1/2>' in parts[-2] else '3/2'
        K = re.search(r'\[(\d+/2)\]', v['term']).group(1)
        lookup[(core, parts[-1], K, int(round(float(v['J']))))] = label
    return lookup


def resolve_bsr_label(name, lookup):
    """Model label of a BSR state name; 'ground' for 'Ar', None for 'Ar(nl>5s)',
    'Ar(Rydberg)' and levels the model does not have."""
    name = name.strip()
    if name == 'Ar':
        return 'ground'
    m = _BSR_STATE.match(name)
    if m is None:
        return None
    nl, prime, K, J = m.groups()
    return lookup.get(('1/2' if prime else '3/2', nl, K, int(J)))


def parse_bsr_lxcat(path, levels, align_thresholds=True):
    """
    Parse an LXCat download of the BSR database into a CrossSectionList dict, like
    parse_lxcat. The file holds two data sets (top-level COMMENT lines): BSR-500 from the
    ground state (2013), and the ground state again plus excitation out of the 4s, 4p, 3d
    and 5s levels (Zatsarinny, Wang & Bartschat 2014). A channel in both is kept once
    (first occurrence, listed in 'duplicates_skipped').

    align_thresholds : shift each cross section by (model dE - BSR threshold) so it starts
                       at the model threshold; 'threshold_lxcat_eV' and 'energy_shift_eV'
                       keep the original.
    """
    with open(path, encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    lookup = bsr_level_lookup(levels)

    cross_sections, crosswalk, seen, duplicates = [], {}, set(), []
    dataset, i, prev = None, 0, ''
    while i < len(lines):
        s = lines[i].strip()
        if s:
            prev, last = s, prev                     # last: the previous non-empty line
        if lines[i].startswith('COMMENT:') and last.startswith('*****'):   # data-set comment
            text = [s[len('COMMENT:'):].strip()]
            while i + 1 < len(lines) and lines[i + 1].startswith(' ') and lines[i + 1].strip():
                i += 1
                text.append(lines[i].strip())
            dataset = ' '.join(text)
            i += 1
            continue
        if s != 'EXCITATION':
            i += 1
            continue

        reaction = lines[i + 1].strip()
        lo_raw, up_raw = (x.strip() for x in re.split(r'\s*<?->\s*', reaction))
        nums = [float(x) for x in lines[i + 2].split()]
        threshold_eV = nums[0]
        g_ratio = nums[1] if len(nums) > 1 else None

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

        lower_label = resolve_bsr_label(lo_raw, lookup)
        upper_label = resolve_bsr_label(up_raw, lookup)
        crosswalk.setdefault(lo_raw, lower_label)
        crosswalk.setdefault(up_raw, upper_label)
        pair = (lower_label, upper_label)
        if lower_label is not None and upper_label is not None and pair in seen:
            duplicates.append(reaction)
            continue
        seen.add(pair)

        shift = 0.0
        if align_thresholds and lower_label is not None and upper_label is not None:
            dE = levels[upper_label]['energy_eV'] - (0.0 if lower_label == 'ground'
                                                     else levels[lower_label]['energy_eV'])
            shift = dE - threshold_eV
            energy = [e + shift for e in energy]
        cross_sections.append({
            'type':              'excitation',
            'database':          'BSR',
            'dataset':           dataset,
            'lxcat_reaction':    reaction,
            'lxcat_lower':       lo_raw,
            'lxcat_upper':       up_raw,
            'lower_label':       lower_label,
            'upper_label':       upper_label,
            'threshold_eV':      threshold_eV + shift,
            'threshold_lxcat_eV': threshold_eV,
            'energy_shift_eV':   shift,
            'g_ratio':           g_ratio,
            'energy_eV':         energy,
            'cross_section':     sigma,
        })
    return {'cross_sections': cross_sections, 'crosswalk': crosswalk,
            'duplicates_skipped': duplicates}


def build_bsr_cross_sections_json(in_txt, out_json, levels_json, align_thresholds=True):
    with open(levels_json) as f:
        levels = json.load(f)
    data = parse_bsr_lxcat(in_txt, levels, align_thresholds)
    with open(out_json, 'w') as f:
        json.dump(data, f, indent=1)
    return data


if __name__ == '__main__':
    MainDir = Path(__file__).resolve().parent.parent
    DataFolder = MainDir / 'InputData'
    InTXT   = DataFolder / 'LXCatPureArgon.txt'    # <- your LXCat file
    OutJSON = DataFolder / 'ArgonCrossSections.json'
    d = build_cross_sections_json(InTXT, OutJSON, DataFolder / 'ArgonLevelList.json')
    n_total  = len(d['cross_sections'])
    n_ground = sum(1 for c in d['cross_sections'] if c['lower_label'] == 'ground')
    n_none   = sum(1 for c in d['cross_sections']
                   if c['lower_label'] is None or c['upper_label'] is None)
    print(f"Parsed {n_total} cross sections "
          f"({n_ground} from ground, {n_none} involving untracked levels) "
          f"-> {OutJSON}")
    for r in d['duplicates_skipped']:
        print(f"  duplicate block skipped: {r}")

    # BSR (B-spline R-matrix) set, incl. excitation out of the 4s levels
    InBSR   = DataFolder / 'ArgonLxCatPureArgonUpdated.txt'
    OutBSR  = DataFolder / 'ArgonCrossSectionsBSR.json'
    d = build_bsr_cross_sections_json(InBSR, OutBSR, DataFolder / 'ArgonLevelList.json')
    from collections import Counter
    n_from = Counter(c['lower_label'] for c in d['cross_sections']
                     if c['lower_label'] is not None and c['upper_label'] is not None)
    n_none = sum(1 for c in d['cross_sections']
                 if c['lower_label'] is None or c['upper_label'] is None)
    print(f"Parsed {len(d['cross_sections'])} BSR cross sections ({n_none} involving untracked "
          f"levels) -> {OutBSR}")
    print('  by lower level: ' + ', '.join(f'{k} {v}' for k, v in sorted(n_from.items())))
    print(f"  {len(d['duplicates_skipped'])} ground-state blocks in both BSR data sets kept once")