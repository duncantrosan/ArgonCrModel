# -*- coding: utf-8 -*-
"""
Einstein A coefficients for the Ar I transitions that NIST does not tabulate.

Follows Bogaerts, Gijbels & Vlcek, J. Appl. Phys. 84, 121 (1998): measured /
recommended data where they exist (NIST ASD; for 4p-4s and 5p-4s these are the
Wiese et al. 1989 values), otherwise transition probabilities calculated in
(j,K) coupling, as they took from Vlcek (1989) / Katsonis & Drawin (1980).
Every line between modelled levels that is allowed in (j,K) coupling (same ion
core, outer electron l -> l +- 1) and has no NIST A is estimated, by one of:

  'jK multiplet (NIST)'         other components of the same [K] -> [K'] multiplet
                                are in NIST: the multiplet line strength S(K,K') is
                                fitted to them, the missing ones follow from the jK
                                angular factors
  'jK + Coulomb approximation'  no component in NIST: S(K,K') from the Bates-Damgaard
                                Coulomb-approximation radial integral

Line strength in (j,K) coupling (ion core j_c and outer-electron spin s = 1/2 are
spectators; K = j_c + l, J = K + s):
    S = l_> sigma^2 (2K+1)(2K'+1){l K j_c; K' l' 1}^2 (2J+1)(2J'+1){K J s; J' K' 1}^2
    A = 2.0261e18 S / (g_k lambda^3)      [s^-1; S in atomic units, lambda in Angstrom]

Writes InputData/ArgonReactionListSupplement.json (merged by he.ImportReactionList)
and prints a validation of both methods against the NIST values:
    python EstimateTransitionProbabilities.py
"""
import json
import math
from collections import defaultdict
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
from scipy.special import gamma, gammaincc
import HelperFunctions as he

RYDBERG_EV = 13.605693                  # R_inf hc (reduced-mass correction for Ar is 1e-5)
E_LIMIT = {'3/2': 15.7596117,           # Ar I ionization limit to Ar II 3p5 2P_3/2 (NIST)
           '1/2': 15.7596117 + 0.1774930}   # + Ar II 2P fine structure, 1431.58 cm^-1 (NIST)
HC_EV_NM = 1239.841984
A_CONST = 2.0261e18
S_HALF = Q(1, 2)
OUT = Path(__file__).resolve().parent.parent / 'InputData' / 'ArgonReactionListSupplement.json'

REFERENCES = {
    'Bogaerts98': 'A. Bogaerts, R. Gijbels and J. Vlcek, J. Appl. Phys. 84, 121 (1998)',
    'KatsonisDrawin80': 'K. Katsonis and H. W. Drawin, J. Quant. Spectrosc. Radiat. Transfer 23, 1 (1980)',
    'BatesDamgaard49': 'D. R. Bates and A. Damgaard, Phil. Trans. R. Soc. Lond. A 242, 101 (1949)',
    'NIST': 'NIST Atomic Spectra Database (A values of the fitted components)',
}


# ---------------------------------------------------------------------------
# Angular momentum
# ---------------------------------------------------------------------------
def _delta(a, b, c):
    s = [a + b - c, a - b + c, -a + b + c]
    if any(x < 0 or x.denominator != 1 for x in s) or (a + b + c).denominator != 1:
        return None
    return (math.factorial(int(s[0])) * math.factorial(int(s[1])) * math.factorial(int(s[2]))
            / math.factorial(int(a + b + c + 1)))


def wigner_6j(a, b, c, d, e, f):
    """{a b c; d e f} from the Racah formula (arguments integer or half-integer)."""
    a, b, c, d, e, f = (Q(x) for x in (a, b, c, d, e, f))
    tri = [_delta(a, b, c), _delta(a, e, f), _delta(d, b, f), _delta(d, e, c)]
    if any(x is None for x in tri):
        return 0.0
    lo = int(max(a + b + c, a + e + f, d + b + f, d + e + c))
    hi = int(min(a + b + d + e, a + c + d + f, b + c + e + f))
    tot = sum((-1)**t * math.factorial(t + 1) / (
        math.factorial(int(t - a - b - c)) * math.factorial(int(t - a - e - f)) *
        math.factorial(int(t - d - b - f)) * math.factorial(int(t - d - e - c)) *
        math.factorial(int(a + b + d + e - t)) * math.factorial(int(a + c + d + f - t)) *
        math.factorial(int(b + c + e + f - t))) for t in range(lo, hi + 1))
    return math.sqrt(tri[0] * tri[1] * tri[2] * tri[3]) * tot


def jk_state(level):
    """Quantum numbers of a (j,K)-coupled Ar I level, None for the ground state."""
    if level['kind'] == 'ground':
        return None
    parts = level['configuration'].split('.')
    core = '1/2' if '<1/2>' in parts[-2] else '3/2'
    outer = parts[-1]
    n, l = int(outer[:-1]), 'spdfg'.index(outer[-1])
    K = Q(level['term'][level['term'].index('[') + 1:level['term'].index(']')])
    return dict(label=level['label'], core=core, jc=Q(core), n=n, l=l, K=K, J=Q(int(level['J'])),
                E=level['energy_eV'], g=level['g'], config=level['configuration'],
                nstar=math.sqrt(RYDBERG_EV / (E_LIMIT[core] - level['energy_eV'])))


def jk_weight(u, d):
    """(2K+1)(2K'+1){l K jc; K' l' 1}^2 (2J+1)(2J'+1){K J s; J' K' 1}^2, 0 if jK-forbidden."""
    if u['core'] != d['core'] or abs(u['l'] - d['l']) != 1:
        return 0.0
    return ((2*u['K'] + 1) * (2*d['K'] + 1) * wigner_6j(u['l'], u['K'], u['jc'], d['K'], d['l'], 1)**2 *
            (2*u['J'] + 1) * (2*d['J'] + 1) * wigner_6j(u['K'], u['J'], S_HALF, d['J'], d['K'], 1)**2)


# ---------------------------------------------------------------------------
# Bates-Damgaard Coulomb approximation
# ---------------------------------------------------------------------------
def _bd_series(n, l):
    """Coefficients of the asymptotic Whittaker series, truncated where it starts to grow."""
    zbar = 2 * (1.5 * n * n - 0.5 * l * (l + 1)) / n          # z = 2r/n* at the mean radius
    c = [1.0]
    for t in range(1, 16):
        ct = c[-1] * (l + n - t + 1) * (l - n + t) / t
        if ct == 0 or abs(ct) / zbar**t > abs(c[-1]) / zbar**(t - 1):
            break
        c.append(ct)
    return c


R_CUT = 0.5   # a0: inner cut-off of the radial integral. The Coulomb functions are not valid
              # inside the core, and terms with small Gamma argument blow up without it; 0.5-2 a0
              # all give the same agreement with NIST (validation printed by main)


def bd_radial_integral(n1, l1, n2, l2, r_cut=R_CUT):
    """sigma = int_rcut^inf P(n1*,l1) r P(n2*,l2) dr in Bohr radii, term by term
    (upper incomplete Gamma functions; r_cut = 0 is the original Bates-Damgaard form)."""
    norm = lambda n, l: 1 / math.sqrt(n * n * math.gamma(n + l + 1) * math.gamma(n - l))
    beta = 1 / n1 + 1 / n2
    s = 0.0
    for t, a in enumerate(_bd_series(n1, l1)):
        for u, b in enumerate(_bd_series(n2, l2)):
            p = n1 + n2 + 2 - t - u
            if p > 0:
                G = gamma(p) * gammaincc(p, beta * r_cut)     # int_rcut^inf r^(p-1) e^(-beta r) dr * beta^p
                s += a * b * (2 / n1)**(n1 - t) * (2 / n2)**(n2 - u) * G / beta**p
    return norm(n1, l1) * norm(n2, l2) * s


def self_checks():
    assert abs(wigner_6j(1, 1, 1, 1, 1, 1) - 1 / 6) < 1e-12
    # hydrogen radial integrals |<n l|r|n' l'>| (exact values in a0; the method is exact for H)
    for (n1, l1, n2, l2), exact in {(2, 1, 1, 0): 1.2903, (3, 1, 2, 0): 3.0648, (3, 2, 2, 1): 4.7480}.items():
        assert abs(abs(bd_radial_integral(n1, l1, n2, l2, r_cut=0)) - exact) < 2e-4, (n1, l1, n2, l2)


def vac_to_air(lam):
    s2 = (1e3 / lam) ** 2
    return lam / (1 + 8.34254e-5 + 2.406147e-2 / (130 - s2) + 1.5998e-4 / (38.9 - s2))


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------
def build():
    self_checks()
    levels = he.ImportLevelList()
    nist = he.ImportReactionList(IncludeSupplement=False)['transitions']
    states = {lab: jk_state(v) for lab, v in levels.items() if v['kind'] != 'ground'}
    key = {(v['configuration'], v['term'], float(v['J'])): lab for lab, v in levels.items()}

    # sum rule: sum over all final J', K' of w = (2J+1)/(2l+1) for each l' = l +- 1
    for s in states.values():
        for lp in (s['l'] - 1, s['l'] + 1):
            if lp < 0:
                continue
            tot = 0.0
            for Kp2 in range(int(2 * abs(s['jc'] - lp)), int(2 * (s['jc'] + lp)) + 1, 2):
                Kp = Q(Kp2, 2)
                for Jp in (Kp - S_HALF, Kp + S_HALF):
                    if Jp >= 0:
                        tot += jk_weight(s, dict(s, l=lp, K=Kp, J=Jp))
            assert abs(tot - (2*s['J'] + 1) / (2*s['l'] + 1)) < 1e-9, (s['label'], lp)

    A_nist = {}
    for t in nist:
        u = key.get((t['upper']['config'], t['upper']['term'], float(t['upper']['J'])))
        d = key.get((t['lower']['config'], t['lower']['term'], float(t['lower']['J'])))
        if u and d and t.get('Aki'):
            A_nist[(u, d)] = (t['Aki'], t.get('acc'), t['wl_nm'])

    def lam_vac_A(u, d):                                        # Angstrom
        return 10 * HC_EV_NM / (states[u]['E'] - states[d]['E'])

    def S_from_A(A, u, d):
        return A * states[u]['g'] * lam_vac_A(u, d)**3 / A_CONST

    def A_from_S(S, u, d):
        return A_CONST * S / (states[u]['g'] * lam_vac_A(u, d)**3)

    def multiplet(u, d):
        U, D = states[u], states[d]
        return (U['config'], U['core'], U['K'], D['config'], D['core'], D['K'])

    # all jK-allowed pairs between modelled levels, grouped by multiplet
    groups = defaultdict(list)
    for u, U in states.items():
        for d, D in states.items():
            if D['E'] < U['E'] and jk_weight(U, D) > 0:
                groups[multiplet(u, d)].append((u, d))

    def S_ca(u, d):
        U, D = states[u], states[d]
        sigma = bd_radial_integral(U['nstar'], U['l'], D['nstar'], D['l'])
        return max(U['l'], D['l']) * sigma**2 * jk_weight(U, D), sigma

    # ---- validation against NIST --------------------------------------------
    val_ca, val_mult = defaultdict(list), []
    for comps in groups.values():
        have = [c for c in comps if c in A_nist]
        for u, d in have:
            arr = f"{states[u]['n']}{'spdf'[states[u]['l']]}-{states[d]['n']}{'spdf'[states[d]['l']]}"
            val_ca[arr].append(A_from_S(S_ca(u, d)[0], u, d) / A_nist[(u, d)][0])
            others = [c for c in have if c != (u, d)]
            if others:                                          # leave-one-out multiplet test
                S_red = sum(S_from_A(A_nist[c][0], *c) for c in others) / sum(jk_weight(states[c[0]], states[c[1]]) for c in others)
                val_mult.append(A_from_S(S_red * jk_weight(states[u], states[d]), u, d) / A_nist[(u, d)][0])

    # ---- estimates for the missing lines ---------------------------------------
    out = []
    for comps in groups.values():
        have = [c for c in comps if c in A_nist]
        S_red = None
        if have:
            S_red = sum(S_from_A(A_nist[c][0], *c) for c in have) / sum(jk_weight(states[c[0]], states[c[1]]) for c in have)
        for u, d in comps:
            if (u, d) in A_nist:
                continue
            U, D = states[u], states[d]
            w = jk_weight(U, D)
            lam = lam_vac_A(u, d) / 10
            entry = {
                'wl_nm': round(vac_to_air(lam) if lam > 200 else lam, 3),
                'lower': {'config': levels[d]['configuration'], 'term': levels[d]['term'], 'J': float(levels[d]['J'])},
                'upper': {'config': levels[u]['configuration'], 'term': levels[u]['term'], 'J': float(levels[u]['J'])},
                'acc': 'E',
            }
            if S_red is not None:
                entry.update(Aki=float(f'{A_from_S(S_red * w, u, d):.3g}'), source='jK multiplet (NIST)',
                             method=f'S(K,K\') fitted to {len(have)} NIST component(s) of the same jK multiplet; jK weight {w:.4f}',
                             fitted_to=[{'upper_J': float(states[a]['J']), 'lower_J': float(states[b]['J']),
                                         'wl_nm': A_nist[(a, b)][2], 'Aki': A_nist[(a, b)][0], 'acc': A_nist[(a, b)][1]}
                                        for a, b in have],
                             references=['Bogaerts98', 'KatsonisDrawin80', 'NIST'])
            else:
                S, sigma = S_ca(u, d)
                entry.update(Aki=float(f'{A_from_S(S, u, d):.3g}'), source='jK + Coulomb approximation',
                             method=(f"n*_upper={U['nstar']:.4f}, n*_lower={D['nstar']:.4f}, "
                                     f"radial integral {abs(sigma):.4f} a0, jK weight {w:.4f}"),
                             references=['Bogaerts98', 'KatsonisDrawin80', 'BatesDamgaard49'])
            if entry['Aki'] > 0:
                out.append(entry)
    out.sort(key=lambda e: (e['source'], -e['Aki']))
    return out, val_ca, val_mult


def main():
    out, val_ca, val_mult = build()
    print('Validation against NIST, estimate / NIST for lines NIST does have:')
    print(f"  jK multiplet (leave-one-out, {len(val_mult)} lines): median {np.median(val_mult):.2f}, "
          f"central 68 % {np.percentile(val_mult, 16):.2f} - {np.percentile(val_mult, 84):.2f}")
    allr = [r for v in val_ca.values() for r in v]
    print(f"  jK + Coulomb approximation ({len(allr)} lines): median {np.median(allr):.2f}, "
          f"central 68 % {np.percentile(allr, 16):.2f} - {np.percentile(allr, 84):.2f}")
    for arr, r in sorted(val_ca.items()):
        print(f"     {arr:6s} {len(r):3d} lines   median {np.median(r):5.2f}   "
              f"range {min(r):5.2f} - {max(r):5.2f}")
    doc = {
        'description': ('Ar I transitions between modelled levels that the NIST ASD export '
                        '(ArgonReactionList.json) does not have, generated by '
                        'Scripts/EstimateTransitionProbabilities.py and merged into the model by '
                        'HelperFunctions.ImportReactionList(). NIST lines always take precedence.'),
        'method': ('Following Bogaerts, Gijbels & Vlcek (1998): where no measured / recommended A '
                   'exists, transition probabilities calculated in (j,K) coupling. Only jK-allowed '
                   'lines are estimated (same ion core, outer electron l -> l+-1). '
                   "'jK multiplet (NIST)': multiplet line strength fitted to the NIST components of "
                   "the same [K]->[K'] multiplet; 'jK + Coulomb approximation': Bates-Damgaard radial "
                   'integral with n* from the level energies and the 2P3/2 / 2P1/2 ionization limits, '
                   f'cut off at r = {R_CUT} a0. '
                   "Accuracy grade 'E' (>50 %); see the validation printed by the script."),
        'references': REFERENCES,
        'transitions': out,
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2)
    n = defaultdict(int)
    for e in out:
        n[e['source']] += 1
    print(f"\nWrote {len(out)} transitions to {OUT}: " + ', '.join(f'{k}: {v}' for k, v in n.items()))


if __name__ == "__main__":
    main()
