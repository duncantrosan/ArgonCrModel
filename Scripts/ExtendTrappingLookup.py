# -*- coding: utf-8 -*-
"""
ExtendTrappingLookup.py

Rebuilds the radiation-trapping escape-factor table with the Voigt parameter a down to 1e-6.

The original table (InputData/RadiationTrappingLookupTable.json, CreateTrappingLookups.py) covers
a = 0.01-0.2. The resonance lines of the 4s levels have a ~ 2e-3 at 1 Torr and ~2e-5 at 1 Pa
(resonance broadening, FindTauInModel), so the model extrapolated log10(eta) linearly in a below
the table. In the Lorentz-wing regime eta ~ sqrt(a), so that extrapolation overestimates eta
(by ~2-3x at 1 Torr), i.e. lets the resonance radiation escape too easily.

Same physics as MonteCarloEscapeFactorSolve.escape_factor_fast: g = < T(tau0_R * ell / R, a) >
over uniform emission points in a hemisphere of radius R and isotropic directions, with the
single-chord transmission T from the Voigt profile (T_of_tau). The chord lengths depend only on
the geometry, so they are computed once (vectorised) and reused for every (tau, a); per a, T is
tabulated on a log grid of chord optical depth and interpolated. The grid: tau0_R = 1e-3 - 1e6
(log, 28 points), a = 1e-6 - 0.2 (log, 26 points incl. the old 0.01-0.2 values). Writes
InputData/RadiationTrappingLookupTable_v2.json in the old row format (Tau_R, Shape, EscapeFactor =
[g, standard error]); HelperFunctions.EscapeFactorInterpolator interpolates it in
(log10 tau, log10 a).

Run: python Scripts/ExtendTrappingLookup.py   (~1 min)
"""
import json
from pathlib import Path

import numpy as np

from MonteCarloEscapeFactorSolve import T_of_tau

OUT = Path(__file__).resolve().parent.parent / 'InputData' / 'RadiationTrappingLookupTable_v2.json'
OLD = Path(__file__).resolve().parent.parent / 'InputData' / 'RadiationTrappingLookupTable.json'
N_PHOTONS = 2_000_000
TAU = np.logspace(-3, 6, 28)
A_OLD = np.round(np.linspace(0.01, 0.2, 20), 4)
A = np.unique(np.r_[np.logspace(-6, -2.2, 6), [7e-3], A_OLD])


def hemisphere_chords(N, seed=0):
    """Chord length / R from uniform points in a unit hemisphere (z >= 0) along isotropic directions."""
    rng = np.random.default_rng(seed)
    pts, filled = np.empty((N, 3)), 0
    while filled < N:
        c = rng.uniform(-1, 1, size=(int((N - filled) * 2.2), 3))
        c[:, 2] = np.abs(c[:, 2])
        good = c[np.einsum('ij,ij->i', c, c) <= 1.0]
        take = min(len(good), N - filled)
        pts[filled:filled + take] = good[:take]
        filled += take
    phi = rng.uniform(0, 2 * np.pi, N)
    cost = rng.uniform(-1, 1, N)
    sint = np.sqrt(1 - cost ** 2)
    d = np.column_stack([sint * np.cos(phi), sint * np.sin(phi), cost])
    b = np.einsum('ij,ij->i', pts, d)
    c = np.einsum('ij,ij->i', pts, pts) - 1.0
    t_dome = -b + np.sqrt(b * b - c)                   # the one positive root (inside the sphere)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_base = np.where(d[:, 2] < 0, -pts[:, 2] / d[:, 2], np.inf)
    return np.minimum(t_dome, t_base)


def escape_factor(ell, tau0_R, a, n_grid=400):
    tc = tau0_R * ell
    grid = np.concatenate([[0.0], np.logspace(-4, np.log10(max(tc.max(), 1e-3) * 1.01), n_grid)])
    T = np.array([T_of_tau(t, a) for t in grid])
    v = np.interp(tc, grid, T)
    return v.mean(), v.std() / np.sqrt(len(v))


def main():
    ell = hemisphere_chords(N_PHOTONS)
    rows = []
    for t in TAU:
        for a in A:
            g, se = escape_factor(ell, t, a)
            rows.append({'Tau_R': float(t), 'Shape': float(a), 'EscapeFactor': [float(g), float(se)]})
        print(f'tau0_R = {t:.3g}: done', flush=True)
    OUT.write_text(json.dumps(rows))
    # check against the old table where both exist (same method, other seed / N)
    old = {(round(r['Tau_R'], 9), round(r['Shape'], 4)): r['EscapeFactor'][0] for r in json.loads(OLD.read_text())}
    new = {(round(r['Tau_R'], 9), round(r['Shape'], 4)): r['EscapeFactor'][0] for r in rows}
    from scipy.interpolate import RegularGridInterpolator
    tau_old = np.unique([k[0] for k in old])
    a_old = np.unique([k[1] for k in old])
    f = RegularGridInterpolator((np.log10(TAU), np.log10(A)), np.log10([[new[(round(t, 9), round(a, 4))] for a in A] for t in TAU]))
    ratio = [10 ** f([np.log10(t), np.log10(a)])[0] / old[(t, a)] for t in tau_old for a in a_old]
    print(f'new / old table at the old grid points (a = 0.01-0.2): median {np.median(ratio):.4f}, '
          f'range {np.min(ratio):.4f}-{np.max(ratio):.4f}')
    print(f'wrote {OUT} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
