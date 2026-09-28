"""
BuildNitrogenLevelData.py

Combine
    AtomicNitrogen_levels.json         (NIST J-resolved level list)
    AtomicNitrogenReactionList.json    (NIST radiative transitions)
    AtomicNitrogen_LXCat.json          (from LXCatToJSON.py, BSR term-resolved)
into one dict keyed by level label, written to AtomicNitrogen_LevelData.json.

Each level keeps all of its original fields and gains:

    "electron_impact_excitation": [      # e + N(ground) -> e + N(this level)
        {
          "from": "other0",
          "lxcat_key": "2p2_3P_3s_4Pe",
          "database": "BSR",
          "reference": "...",
          "branching_factor": 0.1667,     # g_J / g_term (statistical J-split)
          "g_term": 12,
          "threshold_eV_lxcat": 10.422,
          "threshold_offset_eV": -0.096,  # (E_level - E_from) - threshold_eV_lxcat
          "energy_shifted": false,
          "energy_eV": [...],
          "cross_section_m2": [...]       # already multiplied by branching_factor
        }
    ],
    "radiative_out": [ {"to": label, "wl_nm", "Aki", "acc", "type"} ],   # this level is upper
    "radiative_in":  [ {"from": label, "wl_nm", "Aki", "acc", "type"} ], # this level is lower
    "A_listed_sum_s-1": float    # sum of Aki in radiative_out -- ONLY the lines in
                                 # the reaction list, NOT the total radiative loss

J-splitting of the LS-term cross sections
-----------------------------------------
BSR on LXCat gives e + N(4S*_3/2) -> N(2S+1 L) summed over J. Each fine-structure
level gets
        sigma_J = sigma_term * (2J+1) / ((2S+1)(2L+1))
which assumes the fine-structure splitting is negligible compared to the
collision energy (meV vs eV here). Summing over J returns the LXCat term value.

Threshold handling
------------------
BSR thresholds are calculated and differ from the NIST level energies by up to
~0.1 eV for some terms. By default the LXCat energy grid is kept as-is and the
offset is stored. Set SHIFT_TO_LEVEL_ENERGY = True to shift each J-level's
grid so the cross section starts at the NIST excitation energy (keeps
superelastic/detailed-balance rates consistent with the level energies).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

SHIFT_TO_LEVEL_ENERGY = False
E_MATCH_TOL_eV = 2e-3      # tolerance when matching reaction-list levels by energy


def _Load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _GroundLabel(levels):
    return min(levels, key=lambda k: levels[k]["energy_eV"])


# ---------------------------------------------------------------------------
# Electron-impact excitation (LXCat)
# ---------------------------------------------------------------------------
def AttachLXCatExcitation(levels, lxcat, shift_to_level_energy=SHIFT_TO_LEVEL_ENERGY):
    ground = _GroundLabel(levels)
    E_ground = levels[ground]["energy_eV"]

    by_term = defaultdict(list)
    for label, lev in levels.items():
        by_term[(lev["configuration"], lev["term"])].append(label)

    report = []
    unmatched = []

    for key, proc in lxcat.items():
        if proc["type"] != "EXCITATION" or proc["configuration"] is None:
            continue

        labels = by_term.get((proc["configuration"], proc["term"]), [])
        if not labels:
            unmatched.append(key)
            continue

        g_term = proc["g_term"]
        g_sum = sum(levels[l]["g"] for l in labels)
        th = proc["threshold_eV"]

        for label in sorted(labels, key=lambda l: levels[l]["J"]):
            lev = levels[label]
            bf = lev["g"] / g_term
            offset = (lev["energy_eV"] - E_ground) - th
            energy = proc["energy_eV"]
            if shift_to_level_energy:
                energy = [E + offset for E in energy]

            lev["electron_impact_excitation"].append({
                "from": ground,
                "lxcat_key": key,
                "database": proc["database"],
                "reference": proc["reference"],
                "branching_factor": bf,
                "g_term": g_term,
                "threshold_eV_lxcat": th,
                "threshold_offset_eV": offset,
                "energy_shifted": shift_to_level_energy,
                "energy_eV": list(energy),
                "cross_section_m2": [s * bf for s in proc["cross_section_m2"]],
            })

        g_w_mean_E = sum(levels[l]["g"] * levels[l]["energy_eV"] for l in labels) / g_sum
        report.append((key, proc["configuration"], proc["term"], labels,
                       g_sum, g_term, th, g_w_mean_E - E_ground))

    return report, unmatched


# ---------------------------------------------------------------------------
# Radiative transitions (NIST reaction list)
# ---------------------------------------------------------------------------
def AttachRadiativeTransitions(levels, reactions, tol=E_MATCH_TOL_eV):
    by_state = {}
    duplicates = []
    for label, lev in levels.items():
        k = (lev["configuration"], lev["term"], float(lev["J"]))
        if k in by_state:
            duplicates.append((k, by_state[k], label))
        by_state[k] = label

    def Find(state):
        label = by_state.get((state["config"], state["term"], float(state["J"])))
        if label is None:
            return None
        if abs(levels[label]["energy_eV"] - state["E_eV"]) > tol:
            return None
        return label

    unmatched = []
    n_ok = 0
    for tr in reactions["transitions"]:
        up, lo = Find(tr["upper"]), Find(tr["lower"])
        if up is None or lo is None:
            unmatched.append((tr["wl_nm"], tr["lower"], tr["upper"], up, lo))
            continue

        common = {"wl_nm": tr["wl_nm"], "Aki": tr["Aki"],
                  "acc": tr.get("acc", ""), "type": tr.get("type", "")}
        levels[up]["radiative_out"].append({"to": lo, **common})
        levels[lo]["radiative_in"].append({"from": up, **common})
        n_ok += 1

    for lev in levels.values():
        lev["radiative_out"].sort(key=lambda d: d["wl_nm"])
        lev["radiative_in"].sort(key=lambda d: d["wl_nm"])
        lev["A_listed_sum_s-1"] = sum(d["Aki"] for d in lev["radiative_out"])

    return n_ok, unmatched, duplicates


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def BuildNitrogenLevelData(levels_path="AtomicNitrogen_levels.json",
                           reactions_path="AtomicNitrogenReactionList.json",
                           lxcat_path="AtomicNitrogen_LXCat.json",
                           out_path="AtomicNitrogen_LevelData.json",
                           shift_to_level_energy=SHIFT_TO_LEVEL_ENERGY):
    levels = _Load(levels_path)
    reactions = _Load(reactions_path)
    lxcat = _Load(lxcat_path)

    for lev in levels.values():
        lev["electron_impact_excitation"] = []
        lev["radiative_out"] = []
        lev["radiative_in"] = []

    report, lx_unmatched = AttachLXCatExcitation(levels, lxcat, shift_to_level_energy)
    n_rad, rad_unmatched, dup_states = AttachRadiativeTransitions(levels, reactions)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(levels, f, indent=2)

    # ---- summary ---------------------------------------------------------
    print(f"Wrote {out_path}: {len(levels)} levels")
    print(f"\nLXCat terms -> levels (statistical J-split), shift_to_level_energy={shift_to_level_energy}")
    print(f"  {'LXCat key':<16} {'config':<17} {'term':<4} {'levels':<28} "
          f"{'g_sum/g_LS':>10} {'E_th LX':>8} {'<E> NIST':>9} {'dE':>7}")
    for key, cfg, term, labels, g_sum, g_term, th, Ebar in report:
        flag = "" if g_sum == g_term else "  <-- g mismatch"
        print(f"  {key:<16} {cfg:<17} {term:<4} {','.join(labels):<28} "
              f"{g_sum:>4}/{g_term:<5} {th:>8.3f} {Ebar:>9.3f} {Ebar - th:>+7.3f}{flag}")
    for key in lx_unmatched:
        print(f"  WARNING: LXCat process {key} matched no level")

    n_with = sum(1 for l in levels.values() if l["electron_impact_excitation"])
    print(f"\n  {n_with} levels have an LXCat excitation cross section")

    print(f"\nRadiative: {n_rad}/{len(reactions['transitions'])} transitions attached")
    for wl, lo, up, lab_up, lab_lo in rad_unmatched:
        miss = []
        if lab_up is None:
            miss.append(f"upper {up['config']} {up['term']} J={up['J']} E={up['E_eV']}")
        if lab_lo is None:
            miss.append(f"lower {lo['config']} {lo['term']} J={lo['J']} E={lo['E_eV']}")
        print(f"  WARNING: {wl} nm not matched ({'; '.join(miss)})")
    for k, a, b in dup_states:
        print(f"  WARNING: levels {a} and {b} share config/term/J {k}")

    return levels


if __name__ == "__main__":
    args = sys.argv[1:]
    BuildNitrogenLevelData(*args) if args else BuildNitrogenLevelData()
    
DF = BuildNitrogenLevelData(*args)
    