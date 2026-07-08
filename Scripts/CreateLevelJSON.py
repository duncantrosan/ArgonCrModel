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

import csv, re, io, json

# ----------------------------------------------------------------------
# 0. The raw CSV (paste/replace with file read if you prefer)
# ----------------------------------------------------------------------
RAW = r'''Configuration,Term,J,g,Prefix,Level (eV),Suffix,Uncertainty (eV),Reference
"=""3s2.3p6""","=""1S""","=""0""",1,"=""""","=""0.00000000""","=""""","=""0""","=""L2131"""
"=""3s2.3p5.(2P*<3/2>).4s""","=""2[3/2]*""","=""2""",5,"=""""","=""11.54835442""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<3/2>).4s""","=""2[3/2]*""","=""1""",3,"=""""","=""11.62359272""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<1/2>).4s""","=""2[1/2]*""","=""0""",1,"=""""","=""11.72316039""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<1/2>).4s""","=""2[1/2]*""","=""1""",3,"=""""","=""11.82807116""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<3/2>).4p""","=""2[1/2]""","=""1""",3,"=""""","=""12.90701530""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<3/2>).4p""","=""2[1/2]""","=""0""",1,"=""""","=""13.27303810""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<3/2>).4p""","=""2[5/2]""","=""3""",7,"=""""","=""13.07571571""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<3/2>).4p""","=""2[5/2]""","=""2""",5,"=""""","=""13.09487256""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<3/2>).4p""","=""2[3/2]""","=""1""",3,"=""""","=""13.15314387""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<3/2>).4p""","=""2[3/2]""","=""2""",5,"=""""","=""13.17177770""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<1/2>).4p""","=""2[3/2]""","=""1""",3,"=""""","=""13.28263902""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<1/2>).4p""","=""2[3/2]""","=""2""",5,"=""""","=""13.30222747""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<1/2>).4p""","=""2[1/2]""","=""1""",3,"=""""","=""13.32785705""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<1/2>).4p""","=""2[1/2]""","=""0""",1,"=""""","=""13.47988682""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<3/2>).3d""","=""2[1/2]*""","=""0""",1,"=""""","=""13.8450385""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<3/2>).3d""","=""2[1/2]*""","=""1""",3,"=""""","=""13.8636686""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<3/2>).3d""","=""2[3/2]*""","=""2""",5,"=""""","=""13.9034546""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<3/2>).3d""","=""2[3/2]*""","=""1""",3,"=""""","=""14.1525151""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<3/2>).3d""","=""2[7/2]*""","=""4""",9,"=""""","=""13.9792373""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<3/2>).3d""","=""2[7/2]*""","=""3""",7,"=""""","=""14.0127381""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<3/2>).3d""","=""2[5/2]*""","=""2""",5,"=""""","=""14.0630272""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<3/2>).3d""","=""2[5/2]*""","=""3""",7,"=""""","=""14.0990559""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<3/2>).5s""","=""2[3/2]*""","=""2""",5,"=""""","=""14.0682977""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<3/2>).5s""","=""2[3/2]*""","=""1""",3,"=""""","=""14.0899685""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<1/2>).3d""","=""2[5/2]*""","=""2""",5,"=""""","=""14.2136715""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<1/2>).3d""","=""2[5/2]*""","=""3""",7,"=""""","=""14.2361061""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<1/2>).3d""","=""2[3/2]*""","=""2""",5,"=""""","=""14.2340226""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<1/2>).3d""","=""2[3/2]*""","=""1""",3,"=""""","=""14.3036684""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<1/2>).5s""","=""2[1/2]*""","=""0""",1,"=""""","=""14.2410277""","=""""","=""""","="""""
"=""3s2.3p5.(2P*<1/2>).5s""","=""2[1/2]*""","=""1""",3,"=""""","=""14.2550856""","=""""","=""""","="""""'''


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
    for tag in ('4s', '4p', '5s', '3d'):
        if config.endswith('.' + tag) or config.endswith(tag):
            return tag
    return 'other'


# ----------------------------------------------------------------------
# 3. Parse rows
# ----------------------------------------------------------------------
def parse(raw):
    reader = csv.reader(io.StringIO(raw))
    header = next(reader)
    rows = []
    for r in reader:
        if not r or all(c.strip() == '' for c in r):
            continue
        config = clean(r[0])
        term   = clean(r[1])
        J_raw  = clean(r[2])
        g      = int(r[3]) if r[3].strip().isdigit() else int(clean(r[3]))
        E      = float(clean(r[5]))
        # J may be fraction "3/2" (not here, all integer) — handle anyway
        if '/' in J_raw:
            num, den = J_raw.split('/')
            J = float(num) / float(den)
        else:
            J = float(J_raw)
        rows.append({
            'config': config, 'term': term, 'J': J, 'g': g,
            'energy_eV': E, 'manifold': manifold_of(config),
        })
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

    for man, prefix in (('4s','4s'), ('4p','4p'), ('3d','3d'), ('5s','5s')):
        for rank, i in enumerate(idx_sorted(man)):
            rows[i]['label'] = f'{prefix}{rank+1}'   # 1-based, energy-ascending

    for i, r in enumerate(rows):
        if r['manifold'] == 'ground':
            rows[i]['label'] = 'ground'
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
    rows = parse(RAW)
    rows = assign_labels(rows)
    rows = flag_4s(rows)
    species = build(rows)

    # pretty print
    print(f"{'label':<8} {'manifold':<8} {'term':<9} {'J':>3} {'g':>3} "
          f"{'E(eV)':>11}  kind")
    print('-'*60)
    for k, v in sorted(species.items(), key=lambda kv: kv[1]['energy_eV']):
        print(f"{v['label']:<8} {v['manifold']:<8} {v['term']:<9} "
              f"{v['J']:>3.0f} {v['g']:>3} {v['energy_eV']:>11.5f}  {v['kind']}")

    with open('ar_levels.json', 'w') as f:
        json.dump(species, f, indent=2)
    print(f"\nWrote {len(species)} levels to ar_levels.json")