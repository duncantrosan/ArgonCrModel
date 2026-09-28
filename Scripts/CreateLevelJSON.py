# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:50:12 2026

@author: dptro
"""

"""
Parse NIST ASD Ar I level export -> CR-model species dictionary.

Adds Paschen notation. Notes on Paschen labels:
  * 4s  -> 1s5,1s4,1s3,1s2   (energy ASCENDING; 1s5 lowest, 1s2 highest)
  * 4p  -> 2p10 ... 2p1       (energy ASCENDING; 2p10 lowest, 2p1 highest)
  * 3d & 5s -> Paschen's historic 3d / 2s labels, assigned by ENERGY ORDER
    across the merged 3d+5s block (this is the conventional, if ugly, scheme).

Also flags the 4s sublevels:
  * 1s5 (J=2) and 1s3 (J=0) are METASTABLE (no dipole decay to ground)
  * 1s4 (J=1) and 1s2 (J=1) are RESONANT  (decay to ground -> escape factor)
"""

import csv, re, json, os


# ----------------------------------------------------------------------
# 0. Path to the raw NIST export
# ----------------------------------------------------------------------
# Project root (ArgonCrModel folder), found relative to this file
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FILE_PATH = os.path.join(ROOT_DIR, "InputData", "ArgonLevelListUpdated.txt")

# ----------------------------------------------------------------------
# 1. Strip NIST's ="..." wrapper
# ----------------------------------------------------------------------
def clean(cell):
    s = cell.strip()
    m = re.match(r'^="(.*)"$', s)
    if m:
        s = m.group(1)
    return s.strip().strip('"')


# ----------------------------------------------------------------------
# 2. Identify manifold (4s,4p,5s,3d,ground) from configuration
# ----------------------------------------------------------------------
def manifold_of(config):
    if config.endswith('3p6'):
        return 'ground'
    for tag in ('4s', '4p', '5s', '5p', '3d', '4d', '6s', '6p'):
        if config.endswith('.' + tag) or config.endswith(tag):
            return tag
    return 'other'


# ----------------------------------------------------------------------
# 3. Parse rows
#
# IMPORTANT: `filepath` is a path to a file on disk. This function OPENS
# that file and reads its contents. (The previous bug: this used to do
# `csv.reader(io.StringIO(raw))` where `raw` was the FILE_PATH string
# itself — that wraps the path text as if it WERE the CSV data, instead
# of reading the file it points to. Since the path has no commas/newlines,
# csv.reader saw it as a single one-column header row and nothing else,
# so every run silently produced 0 rows with no error.)
# ----------------------------------------------------------------------
def parse(filepath, debug=True):
    """
    Column-name-driven parser: reads the header row and looks up each
    field by name instead of a hardcoded position. This makes it robust
    to different NIST ASD export variants, e.g.:
      - Variant A: Configuration,Term,J,g,Prefix,Level (eV),Suffix,Uncertainty (eV),Lande,Reference
      - Variant B: Configuration,Term,J,Prefix,Level (cm-1),Suffix,Uncertainty (cm-1),Lande,Reference
    If there's no 'g' column, g is computed as 2J+1.
    If energy is in cm-1 (no '(eV)' column present), it's converted to eV
    (1 eV = 8065.544 cm^-1).
    """
    CM1_PER_EV = 8065.544

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]

        if debug:
            print(f"DEBUG: header ({len(header)} cols) = {header}")

        # Locate columns by exact header name (case-insensitive, stripped).
        # Exact matching matters here: substring matching would let a
        # single-letter target like 'g' match inside 'Configuration'.
        header_lower = [h.strip().lower() for h in header]

        def find_exact(*name_options):
            for name in name_options:
                if name.lower() in header_lower:
                    return header_lower.index(name.lower())
            return None

        def find_contains(*name_options):
            for name in name_options:
                for i, h in enumerate(header_lower):
                    if name.lower() in h:
                        return i
            return None

        idx_config = find_exact('Configuration')
        idx_term   = find_exact('Term')
        idx_J      = find_exact('J')
        idx_g      = find_exact('g')                       # None if not present
        idx_ev     = find_contains('Level (eV)', 'Level (ev)')
        idx_cm1    = find_contains('Level (cm-1)', 'Level (cm')

        if idx_config is None or idx_term is None or idx_J is None:
            raise ValueError(
                f"Could not find Configuration/Term/J columns in header: {header}"
            )
        if idx_ev is None and idx_cm1 is None:
            raise ValueError(
                f"Could not find a Level (eV) or Level (cm-1) column in header: {header}"
            )

        if debug:
            print(f"DEBUG: column map -> config={idx_config}, term={idx_term}, "
                  f"J={idx_J}, g={idx_g}, Level(eV)={idx_ev}, Level(cm-1)={idx_cm1}")

        rows = []
        skipped = 0
        raw_row_count = 0

        for r in reader:
            raw_row_count += 1
            if debug and raw_row_count <= 3:
                print(f"DEBUG: raw row {raw_row_count} ({len(r)} cols) = {r}")

            if not r or all(c.strip() == '' for c in r):
                continue
            try:
                config = clean(r[idx_config])
                term   = clean(r[idx_term])
                J_raw  = clean(r[idx_J])

                if not J_raw:
                    raise ValueError("empty J field")

                # J may be fraction "3/2" — handle anyway
                if '/' in J_raw:
                    num, den = J_raw.split('/')
                    J = float(num) / float(den)
                else:
                    J = float(J_raw)

                # Energy: prefer eV column if present, else convert cm-1 -> eV
                if idx_ev is not None:
                    E_str = clean(r[idx_ev])
                    if not E_str:
                        raise ValueError("empty energy (eV) field")
                    E = float(E_str)
                else:
                    E_str = clean(r[idx_cm1])
                    if not E_str:
                        raise ValueError("empty energy (cm-1) field")
                    E = float(E_str) / CM1_PER_EV

                # g: use file's value if the column exists and is non-empty,
                # otherwise compute from J
                if idx_g is not None and clean(r[idx_g]).strip():
                    g = int(round(float(clean(r[idx_g]))))
                else:
                    g = int(round(2 * J + 1))

                rows.append({
                    'config': config, 'term': term, 'J': J, 'g': g,
                    'energy_eV': E, 'manifold': manifold_of(config),
                })
            except (IndexError, ValueError) as e:
                skipped += 1
                if skipped <= 5:
                    print(f"⚠ Skipped row: {r} — {e}")
                continue

        print(f"DEBUG: total raw rows = {raw_row_count}, parsed = {len(rows)}, skipped = {skipped}")

    return rows


# ----------------------------------------------------------------------
# 4. Private index labels (NOT real Paschen). Energy-ranked within manifold:
#    ground
#    4s1..4s4   (4s1 lowest energy)
#    4p1..4p10
#    3d1..3dN
#    5s1..5s4
#    These are bookkeeping handles only; report the full term symbol.
# ----------------------------------------------------------------------
def assign_labels(rows):
    def idx_sorted(man):
        idxs = [i for i,r in enumerate(rows) if rows[i]['manifold']==man]
        return sorted(idxs, key=lambda i: rows[i]['energy_eV'])

    for man, prefix in (('4s','4s'), ('4p','4p'), ('3d','3d'), ('5s','5s'),
                        ('6s','6s'), ('4d','4d'), ('5p','5p'), ('6p','6p')):
        for rank, i in enumerate(idx_sorted(man)):
            rows[i]['label'] = f'{prefix}{rank+1}'   # 1-based, energy-ascending

    for i, r in enumerate(rows):
        if r['manifold'] == 'ground':
            rows[i]['label'] = 'ground'

    # Safety net: any row that still has no label (e.g. an unexpected
    # manifold not listed above) gets a fallback so build() never KeyErrors.
    for i, r in enumerate(rows):
        if 'label' not in r:
            print(f"⚠ Unrecognized manifold '{r['manifold']}' for config "
                  f"'{r['config']}' — assigning fallback label")
            rows[i]['label'] = f"other{i}"
    return rows


# ----------------------------------------------------------------------
# 5. Metastable / resonant flag for 4s (by J, robust to labeling)
#    4s J=2 and J=0 -> metastable (no dipole decay to ground 1S0)
#    4s J=1         -> resonant   (decays to ground -> escape factor)
# ----------------------------------------------------------------------
def flag_4s(rows):
    for r in rows:
        r['kind'] = 'normal'
        if r['manifold'] == '4s':
            if r['J'] in (0.0, 2.0):
                r['kind'] = 'metastable'
            elif r['J'] == 1.0:
                r['kind'] = 'resonant'
        elif r['manifold'] == 'ground':
            r['kind'] = 'ground'
    return rows


# ----------------------------------------------------------------------
# 6. Build dictionary keyed by a clean ID
# ----------------------------------------------------------------------
def build(rows):
    species = {}
    for idx, r in enumerate(rows):
        key = r['label']
        species[key] = {
            'id': idx,
            'label': r['label'],         # private energy-ranked handle
            'configuration': r['config'],
            'term': r['term'],           # full term symbol -> report THIS
            'J': r['J'],
            'g': r['g'],                 # statistical weight 2J+1
            'energy_eV': r['energy_eV'],
            'manifold': r['manifold'],
            'kind': r['kind'],           # ground/metastable/resonant/normal
        }
    return species



if __name__ == '__main__':
    try:
        print(f"Reading levels from: {FILE_PATH}")
        rows = parse(FILE_PATH)
        print(f"✓ Parsed {len(rows)} levels")

        rows = assign_labels(rows)
        rows = flag_4s(rows)
        species = build(rows)

        # pretty print
        print(f"\n{'label':<8} {'manifold':<8} {'term':<12} {'J':>4} {'g':>3} "
              f"{'E(eV)':>11}  kind")
        print('-'*70)
        for k, v in sorted(species.items(), key=lambda kv: kv[1]['energy_eV']):
            print(f"{v['label']:<8} {v['manifold']:<8} {v['term']:<12} "
                  f"{v['J']:>4.1f} {v['g']:>3} {v['energy_eV']:>11.5f}  {v['kind']}")

        # Write to JSON
        output_file = 'ar_levels.json'
        with open(output_file, 'w') as f:
            json.dump(species, f, indent=2)
        print(f"\n✓ Wrote {len(species)} levels to {output_file}")

    except FileNotFoundError:
        print(f"✗ File not found: {FILE_PATH}")
        print(f"  Update FILE_PATH in the script")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()