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

UNTRACKED_MANIFOLDS = {'5p'}   # states in the file we deliberately don't model


def resolve_label(lxcat_name):
    """
    Map an lxcat state string like 'Ar(3p5 4p J = 1  2p10)' to a model label.
    Handles the ground state ('Ar(3p6 J = 0)'), stray '<' characters left
    over from splitting '<->', inconsistent spacing, and lowercase 'j'.
    Returns None for untracked states (e.g. 5p).
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

    if manifold in UNTRACKED_MANIFOLDS:
        return None               # e.g. all 5p targets, incl. the '3d8' typo block
    return PASCHEN_TO_LABEL.get(paschen)


def parse_lxcat(path):
    """Parse an LXCat cross-section text file into a CrossSectionList dict."""
    with open(path) as f:
        lines = f.read().splitlines()

    cross_sections = []
    crosswalk = {}
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
        crosswalk[lo_raw] = lower_label
        crosswalk[up_raw] = upper_label

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

    return {'cross_sections': cross_sections, 'crosswalk': crosswalk}


def build_cross_sections_json(in_txt, out_json):
    data = parse_lxcat(in_txt)
    with open(out_json, 'w') as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == '__main__':
    MainDir = Path(__file__).resolve().parent.parent
    DataFolder = MainDir / 'InputData'
    InTXT   = DataFolder / 'LXCatPureArgon.txt'    # <- your LXCat file
    OutJSON = DataFolder / 'ArgonCrossSections.json'
    d = build_cross_sections_json(InTXT, OutJSON)
    n_total  = len(d['cross_sections'])
    n_ground = sum(1 for c in d['cross_sections'] if c['lower_label'] == 'ground')
    n_none   = sum(1 for c in d['cross_sections']
                   if c['lower_label'] is None or c['upper_label'] is None)
    print(f"Parsed {n_total} cross sections "
          f"({n_ground} from ground, {n_none} involving untracked levels) "
          f"-> {OutJSON}")