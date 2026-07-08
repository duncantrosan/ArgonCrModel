# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:01:49 2026

@author: dptro
"""

"""
Module 1 of 3: parse_nist_lines.py

Turn a NIST ASD *lines* CSV export into a transitions JSON.

This module knows NOTHING about your level dictionary or your labels.
Its only job: read the messy NIST file, strip the ="..." wrappers, and emit a
clean list of radiative transitions, each described by the spectroscopic
identity of its upper and lower level (configuration, term, J), plus A_ki and
wavelength.

Matching to your modeled levels happens later (Module 2).

Output schema (ar_transitions_raw.json):
{
  "transitions": [
    {
      "wl_nm":   104.822,            # vacuum wavelength if available, else ritz
      "Aki":     5.32e8,             # s^-1, or null if NIST gave none
      "acc":     "AA",              # NIST accuracy rating
      "lower":   {"config": "...", "term": "...", "J": 1.0},
      "upper":   {"config": "...", "term": "...", "J": 1.0}
    },
    ...
  ]
}
"""

import csv, re, io, json


def _clean(cell):
    """Strip NIST's  ="..."  wrapper and surrounding quotes/space."""
    if cell is None:
        return ''
    s = cell.strip()
    m = re.match(r'^="(.*)"$', s)
    if m:
        s = m.group(1)
    return s.strip().strip('"')


def _norm_J(s):
    s = _clean(s)
    if s == '':
        return None
    if '/' in s:
        n, d = s.split('/')
        return float(n) / float(d)
    return float(s)


def _to_float(s):
    s = _clean(s)
    try:
        return float(s) if s != '' else None
    except ValueError:
        return None


def parse_nist_lines(path):
    """Read a NIST lines CSV; return list of transition dicts."""
    with open(path) as f:
        text = f.read()
    reader = csv.reader(io.StringIO(text))

    transitions = []
    for row in reader:
        if not row:
            continue
        c0 = _clean(row[0])
        # skip the repeated header rows (vac or air variants)
        if c0.startswith('obs_wl'):
            continue
        if len(row) < 14:
            continue

        conf_i = _clean(row[8]);  term_i = _clean(row[9]);  J_i = row[10]
        conf_k = _clean(row[11]); term_k = _clean(row[12]); J_k = row[13]
        # rows with no level identity are wavelength-catalogue lines -> skip
        if conf_i == '' or conf_k == '':
            continue

        # wavelength: prefer observed vacuum (col 0), fall back to ritz (col 1)
        wl = _to_float(row[0])
        if wl is None:
            wl = _to_float(row[1])

        transitions.append({
            'wl_nm': wl,
            'Aki':   _to_float(row[4]),
            'acc':   _clean(row[5]),
            'lower': {'config': conf_i, 'term': term_i, 'J': _norm_J(J_i)},
            'upper': {'config': conf_k, 'term': term_k, 'J': _norm_J(J_k)},
        })
    return transitions


def build_transitions_json(in_csv, out_json):
    transitions = parse_nist_lines(in_csv)
    with open(out_json, 'w') as f:
        json.dump({'transitions': transitions}, f, indent=2)
    return transitions


if __name__ == '__main__':
    from pathlib import Path
    MainDir = Path(__file__).resolve().parent.parent
    DataFolder = MainDir / 'InputData'
    InCSV = DataFolder / 'ArgonReactionList.csv' # Change path for Alternative Reaction LIST (default is NIST)

    OutJSON = DataFolder / 'ArgonReactionList.json' # Change For alternative ReactionList
    t = build_transitions_json(InCSV,OutJSON)
    print(f"Parsed {len(t)} transitions from {InCSV} -> {OutJSON}")