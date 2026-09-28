

"""
FixNitrogenLevels.py

Re-assign manifold / label / kind in AtomicNitrogen_levels.json.

The original builder only recognised the argon manifold set
{4s, 3d, 4p, 5s, 4d, 5p, 6s, 6p} and matched it against the end of the
configuration string, so:
  - N's own low manifolds (2p3 ground config, 3s, 3p, 2s.2p4) and every
    nf / n>=5 d / n>=7 s,p series fell through to "other"
  - "2s.2p3.(5S*).14p" matched "4p"  (labelled 4p19-4p21)
  - kind was never set (ground = "normal", no metastables/resonant)

Rules used here
---------------
manifold : outer-electron nl from the last configuration token, matched as a
           whole token (so 14p -> "14p", not "4p"). jK tags ("4f D") and
           parent sub-levels ("(3P<1>)") are stripped first.
           2s2.2p3 -> "ground" (4S*) / "2p3" (2D*, 2P*);  2s.2p4 -> "2s2p4"
label    : manifold + index, energy-ascending within the manifold
           (ties broken by J, then id) -- same convention as before, so
           existing 4s/3d/4p/5s/4d/5p/6s/6p labels are unchanged.
           Manifolds ending in a digit use an underscore: 2p3_1, 2s2p4_1.
kind     : "ground"       lowest level
           "metastable"   2D*, 2P* of the ground configuration
           "autoionizing" E > IP(N I) = 14.53413 eV
           "resonant"     even-parity 4P terms (E1-allowed to 4S*3/2 in LS coupling)
           "normal"       everything else
parent   : ion-core term, e.g. "3P", "1D", "1S", "5S*" (new field)
"""

import json
import re
import sys
from collections import defaultdict

IP_N_eV = 14.53413          # N I first ionization energy (117225.7 cm^-1)


def ParseNitrogenConfig(config):
    """'2s2.2p2.(3P<1>).5f' -> ('3P', '5f');  '2s2.2p3' -> (None, '2p3')"""
    cfg = config.split(" ")[0]                           # drop jK tag, e.g. "4f D"
    m = re.search(r"\(([^)<]+)", cfg)
    parent = m.group(1) if m else None
    last = cfg.split(".")[-1]
    mo = re.fullmatch(r"(\d+)([spdfghik])(\d*)", last)
    outer = f"{mo.group(1)}{mo.group(2)}" if mo else last
    return parent, outer


def ClassifyNitrogenLevel(lev, E_ground, ip=IP_N_eV):
    cfg = lev["configuration"].split(" ")[0]
    parent, outer = ParseNitrogenConfig(cfg)
    E = lev["energy_eV"] - E_ground

    if cfg == "2s2.2p3":
        manifold = "ground" if E == 0.0 else "2p3"
    elif cfg == "2s.2p4":
        manifold = "2s2p4"
    else:
        manifold = outer

    if E > ip:
        kind = "autoionizing"
    elif manifold == "ground":
        kind = "ground"
    elif manifold == "2p3":
        kind = "metastable"
    elif lev["term"] == "4P":            # even parity (no '*'), quartet P
        kind = "resonant"
    else:
        kind = "normal"

    return manifold, kind, parent


def RelabelNitrogenLevels(levels, ip=IP_N_eV):
    E_ground = min(l["energy_eV"] for l in levels.values())

    info = {}
    groups = defaultdict(list)
    for old, lev in levels.items():
        manifold, kind, parent = ClassifyNitrogenLevel(lev, E_ground, ip)
        info[old] = (manifold, kind, parent)
        groups[manifold].append(old)

    label_map = {}
    for manifold, olds in groups.items():
        olds.sort(key=lambda o: (levels[o]["energy_eV"], levels[o]["J"], levels[o]["id"]))
        for i, old in enumerate(olds, 1):
            if manifold == "ground":
                new = "ground"
            elif manifold[-1].isdigit():
                new = f"{manifold}_{i}"
            else:
                new = f"{manifold}{i}"
            label_map[old] = new

    out = {}
    for old, lev in levels.items():                      # keep original id order
        manifold, kind, parent = info[old]
        new = label_map[old]
        out[new] = {
            "id": lev["id"],
            "label": new,
            "configuration": lev["configuration"],
            "parent": parent,
            "term": lev["term"],
            "J": lev["J"],
            "g": lev["g"],
            "energy_eV": lev["energy_eV"],
            "manifold": manifold,
            "kind": kind,
        }
    return out, label_map


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "AtomicNitrogen_levels.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "AtomicNitrogen_levels_v2.json"

    with open(src, "r", encoding="utf-8") as f:
        levels = json.load(f)

    new_levels, label_map = RelabelNitrogenLevels(levels)

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(new_levels, f, indent=2)
    with open(dst.replace(".json", "_label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    changed = {o: n for o, n in label_map.items() if o != n and not o.startswith("other")}
    counts = defaultdict(int)
    for l in new_levels.values():
        counts[(l["manifold"], l["kind"])] += 1

    print(f"{src} -> {dst}: {len(new_levels)} levels, "
          f"{sum(o.startswith('other') for o in label_map)} 'other' labels replaced")
    print(f"previously-named labels that changed: {changed if changed else 'none'}")
    print(f"kinds: { {k: sum(1 for l in new_levels.values() if l['kind'] == k) for k in ('ground','metastable','resonant','normal','autoionizing')} }")