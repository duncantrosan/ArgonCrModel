
"""
LXCatToJSON.py

Parse an LXCat download (BOLSIG+ format) into a JSON file.

Output layout (keyed by product state for excitation/ionization):

{
  "2p2_3P_3s_4Pe": {
    "key": "2p2_3P_3s_4Pe",
    "type": "EXCITATION",
    "database": "BSR",
    "target": "N",
    "product": "N(2p2_3P_3s_4Pe)",
    "product_state": "2p2_3P_3s_4Pe",
    "reversible": false,                 # True if written with "<->"
    "threshold_eV": 10.422,
    "stat_weight_ratio": null,           # only present for "<->" processes
    "configuration": "2s2.2p2.(3P).3s",  # decoded to NIST notation
    "term": "4P",                        # "*" = odd parity, same as NIST
    "g_term": 12,                        # (2S+1)(2L+1)
    "process": "E + N -> E + N(2p2_3P_3s_4Pe), Excitation",
    "updated": "...",
    "reference": "Y. Wang, O. Zatsarinny, ... (2014).",
    "energy_eV": [...],
    "cross_section_m2": [...]
  },
  ...
}

Units: energy in eV, cross section in m^2 (as in the LXCat file).
"""

import json
import re
import sys
from pathlib import Path

PROCESS_TYPES = {"EXCITATION", "ELASTIC", "EFFECTIVE", "IONIZATION", "ATTACHMENT"}
L_LETTERS = "SPDFGHIK"


# ---------------------------------------------------------------------------
# State-name decoding (BSR naming -> NIST configuration / term)
# ---------------------------------------------------------------------------
def ParseBSRStateName(state):
    """
    Decode BSR state names used on LXCat into NIST-style configuration/term.

        2p3_2Do        -> ("2s2.2p3",          "2D*")
        2s_2p4_4Pe     -> ("2s.2p4",           "4P")
        2p2_3P_3s_4Pe  -> ("2s2.2p2.(3P).3s",  "4P")
        2p2_1D_3s_2De  -> ("2s2.2p2.(1D).3s",  "2D")

    Returns (configuration, term, g_term) or (None, None, None) if the
    name does not follow the pattern.
    """
    tokens = state.split("_")
    m = re.fullmatch(r"(\d+)([SPDFGHIK])([eo])", tokens[-1])
    if m is None:
        return None, None, None

    mult, L, parity = int(m.group(1)), m.group(2), m.group(3)
    term = f"{mult}{L}" + ("*" if parity == "o" else "")
    g_term = mult * (2 * L_LETTERS.index(L) + 1)

    parts = []
    for tok in tokens[:-1]:
        if re.fullmatch(r"\d+[SPDFGHIK]", tok):      # parent term, e.g. 3P
            parts.append(f"({tok})")
        else:                                        # orbital, e.g. 2p2, 3s
            parts.append(tok)

    # BSR drops the closed 2s2 shell; NIST writes it explicitly
    if parts and parts[0].startswith("2p"):
        parts.insert(0, "2s2")

    return ".".join(parts), term, g_term


# ---------------------------------------------------------------------------
# LXCat parser
# ---------------------------------------------------------------------------
def _IsDashLine(line):
    s = line.strip()
    return len(s) >= 5 and set(s) == {"-"}


def _IsNumericLine(line):
    try:
        [float(x) for x in line.split()]
        return len(line.split()) > 0
    except ValueError:
        return False


def ParseLXCatFile(path):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()

    processes = {}
    databases = {}
    current_db = None
    group_comment = None
    warnings = []

    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        line = raw.strip()

        # ---- database header block --------------------------------------
        if line.startswith("DATABASE:"):
            name = line.split(":", 1)[1].strip()
            current_db = name.split("(")[0].strip()
            databases[current_db] = {"name": name}
            group_comment = None
            i += 1
            continue

        if current_db and line.startswith(("PERMLINK:", "HOW TO REFERENCE:")):
            field, value = line.split(":", 1)
            value = [value.strip()]
            i += 1
            # continuation lines are indented and have no "KEY:" prefix
            while i < n and lines[i].startswith(" ") and lines[i].strip() \
                    and not lines[i].strip().startswith("x"):
                value.append(lines[i].strip())
                i += 1
            databases[current_db][field.lower().replace(" ", "_")] = " ".join(value)
            continue

        # ---- group comment (applies to the processes that follow) --------
        if line.startswith("COMMENT:"):
            value = [line.split(":", 1)[1].strip()]
            i += 1
            while i < n and lines[i].startswith(" ") and lines[i].strip():
                value.append(lines[i].strip())
                i += 1
            group_comment = " ".join(value)
            continue

        # ---- process block -----------------------------------------------
        if line in PROCESS_TYPES:
            ptype = line
            reaction = lines[i + 1].strip()
            i += 2

            threshold = None
            stat_ratio = None
            if ptype != "ATTACHMENT":
                params = [float(x) for x in lines[i].split()]
                threshold = params[0]
                stat_ratio = params[1] if len(params) > 1 else None
                i += 1

            # reaction line: "N -> N(2p3_2Do)" or "N <-> N*" or just "N"
            reversible = "<->" in reaction
            if "->" in reaction:
                target, product = [s.strip() for s in re.split(r"<?->", reaction, 1)]
            else:
                target, product = reaction, None

            product_state = None
            if product:
                m = re.search(r"\((.+)\)\s*$", product)
                product_state = m.group(1) if m else product

            header = {}
            while i < n and not _IsDashLine(lines[i]):
                h = lines[i].strip()
                if ":" in h:
                    k, v = h.split(":", 1)
                    header[k.strip().rstrip(".").lower()] = v.strip()
                i += 1
            i += 1  # skip opening dashes

            energy, sigma = [], []
            while i < n and not _IsDashLine(lines[i]):
                if _IsNumericLine(lines[i]):
                    e, s = [float(x) for x in lines[i].split()[:2]]
                    energy.append(e)
                    sigma.append(s)
                i += 1
            i += 1  # skip closing dashes

            # key: product state for excitation/ionization, else type_target
            key = product_state if product_state else f"{ptype.lower()}_{target}"
            base, k = key, 2
            while key in processes:
                key = f"{base}__{k}"
                k += 1

            # sanity checks on the energy grid
            dE = [b - a for a, b in zip(energy, energy[1:])]
            if any(d < 0 for d in dE):
                warnings.append(f"{key}: energy grid not monotonic")
            if any(d == 0 for d in dE):
                warnings.append(f"{key}: duplicate energy points")
            if threshold is not None and energy and energy[0] > threshold + 1e-3:
                warnings.append(f"{key}: first energy point {energy[0]} > threshold {threshold}")

            config, term, g_term = (ParseBSRStateName(product_state)
                                    if product_state else (None, None, None))
            if product_state and config is None:
                warnings.append(f"{key}: could not decode state name '{product_state}'")

            processes[key] = {
                "key": key,
                "type": ptype,
                "database": current_db,
                "target": target,
                "product": product,
                "product_state": product_state,
                "reversible": reversible,
                "threshold_eV": threshold,
                "stat_weight_ratio": stat_ratio,
                "configuration": config,
                "term": term,
                "g_term": g_term,
                "process": header.get("process"),
                "updated": header.get("updated"),
                "reference": group_comment,
                "energy_eV": energy,
                "cross_section_m2": sigma,
            }
            continue

        i += 1

    return processes, databases, warnings


def LXCatToJSON(txt_path, json_path=None):
    txt_path = Path(txt_path)
    if json_path is None:
        json_path = txt_path.with_suffix(".json")

    processes, databases, warnings = ParseLXCatFile(txt_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(processes, f, indent=2)

    print(f"Parsed {len(processes)} processes from {txt_path.name} -> {Path(json_path).name}")
    for db, info in databases.items():
        print(f"  database: {db}")
    for key, p in processes.items():
        print(f"  {p['type']:<10} {key:<18} E_th = {p['threshold_eV']!s:>7} eV  "
              f"{p['configuration']!s:<20} {p['term']!s:<4} g_term = {p['g_term']!s:>2}  "
              f"npts = {len(p['energy_eV'])}")
    for w in warnings:
        print(f"  WARNING: {w}")

    return processes


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "AtomixNitrogenLXCat.txt"
    dst = sys.argv[2] if len(sys.argv) > 2 else "AtomicNitrogen_LXCat.json"
    LXCatToJSON(src, dst)