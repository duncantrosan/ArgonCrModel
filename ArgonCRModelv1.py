# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:30:34 2026

@author: dptro
"""

"""
find_ratio.py - Collisional-Radiative model for pure Argon (Python port)

------------------------------------------------------------------
HOW TO ADD THINGS
------------------------------------------------------------------
* Add a 1s production channel  -> append a dict to PROD_S3 / PROD_S5
* Add a 1s loss channel        -> append a dict to LOSS_S3 / LOSS_S5
* Add a 2p excitation channel  -> append a dict to CHANNELS
* Add an emission line         -> append a dict to EMISSION_LINES
* Change a branching ratio     -> edit the 'br' field of a channel
------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt

# Boltzmann constant [J/K]
K_B = 1.380649e-23


# =====================================================================
#  REACTION / CHANNEL DEFINITIONS  (edit these to add physics)
# =====================================================================

# --- 1s3 production channels -----------------------------------------
#   rxn   : reaction string (looked up in reaction_list)
#   br    : branching ratio into 1s3
#   n_ref : 'n_Ar' | 's_3' | 's_5'  (density of the reactant)
PROD_S3 = [
    {'rxn': 'Ar -> Ar(1S3)', 'br': 1.00,    'n_ref': 'n_Ar'},   # direct GS excitation
    {'rxn': 'Ar -> Ar(2p1)', 'br': 0.02388, 'n_ref': 'n_Ar'},   # 2p1 -> 1s3 cascade
    {'rxn': 'Ar -> Ar(2p2)', 'br': 0.31,    'n_ref': 'n_Ar'},   # 2p2 -> 1s3 cascade
    {'rxn': 'Ar -> Ar(2p4)', 'br': 0.57,    'n_ref': 'n_Ar'},   # 2p4 -> 1s3 cascade
    {'rxn': 'Ar -> Ar(2p7)', 'br': 0.072,   'n_ref': 'n_Ar'},   # 2p7 -> 1s3 cascade
    # ADD new 1s3 production rows here
]

# --- 1s3 loss channels -----------------------------------------------
#   type 'diffusion'      : fixed rate 1/tau_D (no indices needed)
#   type 'excitation_sum' : sum of te_rates over [idx_lo, idx_hi] (0-based, inclusive)
#   type 'single_rxn'     : single rate coefficient * ne (give 'rxn')
LOSS_S3 = [
    {'type': 'diffusion'},
    {'type': 'excitation_sum', 'idx_lo': 45, 'idx_hi': 63},     # 1s3 -> upper levels
    # ADD new 1s3 loss rows here, e.g.:
    # {'type': 'single_rxn', 'rxn': 'Ar(1S3) + e -> Ar + e'},
]

# --- 1s5 production channels -----------------------------------------
PROD_S5 = [
    {'rxn': 'Ar -> Ar(1S5)',  'br': 1.00,    'n_ref': 'n_Ar'},  # direct GS excitation
    {'rxn': 'Ar -> Ar(2p2)',  'br': 0.172,   'n_ref': 'n_Ar'},  # 2p2  -> 1s5
    {'rxn': 'Ar -> Ar(2p3)',  'br': 0.10792, 'n_ref': 'n_Ar'},  # 2p3  -> 1s5
    {'rxn': 'Ar -> Ar(2p4)',  'br': 0.0192,  'n_ref': 'n_Ar'},  # 2p4  -> 1s5
    {'rxn': 'Ar -> Ar(2p6)',  'br': 0.687,   'n_ref': 'n_Ar'},  # 2p6  -> 1s5
    {'rxn': 'Ar -> Ar(2p7)',  'br': 0.153,   'n_ref': 'n_Ar'},  # 2p7  -> 1s5
    {'rxn': 'Ar -> Ar(2p8)',  'br': 0.293,   'n_ref': 'n_Ar'},  # 2p8  -> 1s5
    {'rxn': 'Ar -> Ar(2p9)',  'br': 0.988,   'n_ref': 'n_Ar'},  # 2p9  -> 1s5
    {'rxn': 'Ar -> Ar(2p10)', 'br': 0.762,   'n_ref': 'n_Ar'},  # 2p10 -> 1s5
    # ADD new 1s5 production rows here
]

LOSS_S5 = [
    {'type': 'diffusion'},
    {'type': 'excitation_sum', 'idx_lo': 64, 'idx_hi': 87},     # 1s5 -> upper levels
    # ADD new 1s5 loss rows here
]


# =====================================================================
#  2p UPPER LEVELS
# =====================================================================

# Total spontaneous emission rates [s^-1] for 2p1 ... 2p10
A_2P = np.array([
    41025641.03,   # 2p1
    37230081.91,   # 2p2
    35211267.61,   # 2p3
    32594524.12,   # 2p4
    45045045.05,   # 2p5
    35650623.89,   # 2p6
    33898305.08,   # 2p7
    31289111.39,   # 2p8
    33500837.52,   # 2p9
    24813895.78,   # 2p10
])

# --- 2p excitation channel table -------------------------------------
#   level : 2p level number (1..10)
#   source: 'ground' -> te_rates(idx)*n_Ar*ne
#           's3'     -> te_rates(idx)*s_3*ne
#           's5'     -> te_rates(idx)*s_5*ne
#   idx   : column index into reaction_list / te_rates (0-based)
#
# NOTE: original MATLAB indices were 1-based; subtract 1 here.
CHANNELS = [
    {'level': 1,  'source': 'ground', 'idx': 13}, {'level': 1,  'source': 's3', 'idx': 54}, {'level': 1,  'source': 's5', 'idx': 76},
    {'level': 2,  'source': 'ground', 'idx': 12}, {'level': 2,  'source': 's3', 'idx': 53}, {'level': 2,  'source': 's5', 'idx': 75},
    {'level': 3,  'source': 'ground', 'idx': 11}, {'level': 3,  'source': 's3', 'idx': 52}, {'level': 3,  'source': 's5', 'idx': 74},
    {'level': 4,  'source': 'ground', 'idx': 10}, {'level': 4,  'source': 's3', 'idx': 51}, {'level': 4,  'source': 's5', 'idx': 73},
    {'level': 5,  'source': 'ground', 'idx': 9},  {'level': 5,  'source': 's3', 'idx': 50}, {'level': 5,  'source': 's5', 'idx': 72},
    {'level': 6,  'source': 'ground', 'idx': 8},  {'level': 6,  'source': 's3', 'idx': 49}, {'level': 6,  'source': 's5', 'idx': 71},
    {'level': 7,  'source': 'ground', 'idx': 7},  {'level': 7,  'source': 's3', 'idx': 48}, {'level': 7,  'source': 's5', 'idx': 70},
    {'level': 8,  'source': 'ground', 'idx': 6},  {'level': 8,  'source': 's3', 'idx': 47}, {'level': 8,  'source': 's5', 'idx': 69},
    {'level': 9,  'source': 'ground', 'idx': 5},  {'level': 9,  'source': 's3', 'idx': 46}, {'level': 9,  'source': 's5', 'idx': 68},
    {'level': 10, 'source': 'ground', 'idx': 4},  {'level': 10, 'source': 's3', 'idx': 45}, {'level': 10, 'source': 's5', 'idx': 67},
    # ADD new 2p excitation rows here
]


# =====================================================================
#  EMISSION LINES
# =====================================================================
#   wl   : wavelength [nm]
#   level: 2p level index (1..10)
#   A    : A_ki weighting (lambda-units, kept from original formula)
EMISSION_LINES = [
    {'wl': 750.384,  'level': 1,  'A': 750384000},
    {'wl': 826.452,  'level': 2,  'A': 826452000},
    {'wl': 840.821,  'level': 3,  'A': 840821000},
    {'wl': 852.144,  'level': 4,  'A': 852144000},
    {'wl': 857.806,  'level': 2,  'A': 857806000},   # 2p2 -> 1s3 at 857 nm
    {'wl': 922.450,  'level': 6,  'A': 922450000},
    {'wl': 935.422,  'level': 7,  'A': 935422000},
    {'wl': 978.450,  'level': 8,  'A': 978450000},
    {'wl': 1148.811, 'level': 1,  'A': 1148811000},
    {'wl': 772.421,  'level': 2,  'A': 772421000},
    {'wl': 794.818,  'level': 4,  'A': 794818000},
    {'wl': 866.794,  'level': 7,  'A': 866794000},
    {'wl': 1047.005, 'level': 1,  'A': 1047005000},
    {'wl': 667.728,  'level': 5,  'A': 667728000},
    {'wl': 727.293,  'level': 2,  'A': 727293000},
    {'wl': 738.398,  'level': 3,  'A': 738398000},
    {'wl': 747.117,  'level': 4,  'A': 747117000},
    {'wl': 751.465,  'level': 5,  'A': 751465000},
    {'wl': 800.616,  'level': 6,  'A': 800616000},
    {'wl': 810.369,  'level': 7,  'A': 810369000},
    {'wl': 842.465,  'level': 8,  'A': 842465000},
    {'wl': 965.778,  'level': 2,  'A': 965778000},
    {'wl': 696.543,  'level': 2,  'A': 696543000},
    {'wl': 706.722,  'level': 3,  'A': 706722000},
    {'wl': 714.704,  'level': 4,  'A': 714704000},
    {'wl': 763.511,  'level': 6,  'A': 763511000},
    {'wl': 772.376,  'level': 7,  'A': 772376000},
    {'wl': 801.479,  'level': 8,  'A': 801479000},
    {'wl': 811.531,  'level': 9,  'A': 811531000},
    {'wl': 912.297,  'level': 10, 'A': 912297000},
    # ADD new emission lines here
]


# =====================================================================
#  HELPERS
# =====================================================================

def _rxn_idx(rxn_str, reaction_list):
    """Return 0-based column index of a reaction string, or None."""
    for i, r in enumerate(reaction_list):
        if r == rxn_str:
            return i
    return None


def compute_metastable(prod_channels, loss_channels, ne, n_Ar,
                       te_rates, reaction_list, tau_D,
                       densities=None):
    """
    Steady-state balance for one 1s metastable level: n = Production / Loss.

    densities : dict mapping 'n_Ar','s_3','s_5' -> value, used to resolve
                the 'n_ref' field of a production channel.
    """
    if densities is None:
        densities = {'n_Ar': n_Ar}

    production = 0.0
    for ch in prod_channels:
        idx = _rxn_idx(ch['rxn'], reaction_list)
        if idx is None:
            print(f"WARNING: reaction not found in reaction_list: {ch['rxn']}")
            continue
        n_ref = densities.get(ch['n_ref'], n_Ar)
        production += ch['br'] * te_rates[idx] * n_ref * ne

    loss = 0.0
    for ch in loss_channels:
        t = ch['type']
        if t == 'diffusion':
            loss += 1.0 / tau_D
        elif t == 'excitation_sum':
            lo, hi = ch['idx_lo'], ch['idx_hi']
            loss += np.sum(te_rates[lo:hi + 1]) * ne   # inclusive range
        elif t == 'single_rxn':
            idx = _rxn_idx(ch['rxn'], reaction_list)
            if idx is None:
                print(f"WARNING: loss reaction not found: {ch['rxn']}")
            else:
                loss += te_rates[idx] * ne
        else:
            print(f"WARNING: unknown loss channel type: {t}")

    return production / loss


# =====================================================================
#  MAIN
# =====================================================================

def find_ratio(ne, Te, p_mtorr, reaction_table, reaction_list,
               show_plots=False):
    """
    Collisional-Radiative model for pure Argon.

    Parameters
    ----------
    ne             : electron density [m^-3]
    Te             : electron temperature [eV]
    p_mtorr        : pressure [mTorr]
    reaction_table : (nTe x nRxn) array of rate coefficients
    reaction_list  : list of reaction strings, length nRxn
    show_plots     : if True, draw the EEDF plot

    Returns
    -------
    data : (N x 2) array of [wavelength_nm, relative_intensity]
    s_3  : 1s3 metastable density [m^-3]
    s_5  : 1s5 metastable density [m^-3]
    """
    reaction_table = np.asarray(reaction_table)

    # ---- SECTION 1: constants & plasma conditions -------------------
    R = 0.1                      # [m] plasma radius
    T_g = 300.0                  # [K] gas temperature
    P = p_mtorr / 1000.0         # mTorr -> Torr
    n_Ar = (P * 133.322) / (K_B * T_g)   # ideal gas [m^-3]

    # Interpolate rate coefficients at requested Te
    te_find = np.linspace(0.1, 10, 100)
    te_idx = int(np.argmin(np.abs(te_find - Te)))
    te_rates = reaction_table[te_idx, :]

    # Diffusion
    D_1torr = 0.2                # [cm^2/s] at 1 Torr
    D = D_1torr / P              # [cm^2/s]
    tau_D = R**2 / (5.78 * D)    # first-zero diffusion time [s]

    # ---- SECTION 2: EEDF (Maxwell-Boltzmann) ------------------------
    if show_plots:
        E = np.linspace(0, 20, 500)
        fE = 2.0 / np.sqrt(np.pi) * E**0.5 / Te**1.5 * np.exp(-E / Te)
        plt.figure()
        plt.plot(E, fE)
        plt.xlabel('E (eV)')
        plt.ylabel('f(E)')
        plt.title('Maxwell-Boltzmann EEDF')

    # ---- SECTIONS 4 & 5: metastable balances ------------------------
    # 1s3 first (depends only on ground), then 1s5
    s_3 = compute_metastable(PROD_S3, LOSS_S3, ne, n_Ar,
                             te_rates, reaction_list, tau_D,
                             densities={'n_Ar': n_Ar})
    s_5 = compute_metastable(PROD_S5, LOSS_S5, ne, n_Ar,
                             te_rates, reaction_list, tau_D,
                             densities={'n_Ar': n_Ar})

    # ---- SECTION 6: 2p upper-level corona populations ---------------
    exc_rate = np.zeros(10)
    src_density = {'ground': n_Ar, 's3': s_3, 's5': s_5}
    for ch in CHANNELS:
        lvl = ch['level'] - 1            # 0-based
        n_src = src_density[ch['source']]
        exc_rate[lvl] += te_rates[ch['idx']] * n_src * ne

    p = exc_rate / A_2P                  # corona populations, index 0 = 2p1

    # ---- SECTION 7: emission line intensities -----------------------
    wavelengths = np.array([ln['wl'] for ln in EMISSION_LINES])
    rel_den = np.array([p[ln['level'] - 1] * ln['A'] for ln in EMISSION_LINES])
    rel_den = rel_den / np.max(rel_den)

    data = np.column_stack([wavelengths, rel_den])

    print(f"1s3 metastable density = {s_3:.6g} m^-3")
    print(f"1s5 metastable density = {s_5:.6g} m^-3")

    return data, s_3, s_5


# =====================================================================
#  UTILITY FUNCTIONS (ported from original)
# =====================================================================

def split_reactions(reactions):
    """Split each 'A -> B' string into (reactants, products) lists."""
    reactants, products = [], []
    for r in reactions:
        parts = r.split('->')
        if len(parts) == 2:
            reactants.append(parts[0].strip())
            products.append(parts[1].strip())
        else:
            raise ValueError(f'Reaction string does not contain a single "->": {r}')
    return reactants, products


def track_species(species_list, reactants, products):
    """For each species, list the reaction indices where it appears."""
    results = []
    for sp in species_list:
        react_idx, prod_idx = [], []
        for ri in range(len(reactants)):
            react_split = [s.strip() for s in reactants[ri].split('+')]
            prod_split = [s.strip() for s in products[ri].split('+')]
            if sp in react_split:
                react_idx.append(ri)
            if sp in prod_split:
                prod_idx.append(ri)
        results.append({'species': sp,
                        'reactant_idx': react_idx,
                        'product_idx': prod_idx})
    return results


# =====================================================================
#  EXAMPLE / SELF-TEST
# =====================================================================
if __name__ == '__main__':
    # Dummy reaction list + table just to exercise the code.
    reaction_list = [f'rxn_{i}' for i in range(90)]
    # Overwrite the names that the channels actually look up:
    named = ['Ar -> Ar(1S3)', 'Ar -> Ar(1S5)',
             'Ar -> Ar(2p1)', 'Ar -> Ar(2p2)', 'Ar -> Ar(2p3)',
             'Ar -> Ar(2p4)', 'Ar -> Ar(2p6)', 'Ar -> Ar(2p7)',
             'Ar -> Ar(2p8)', 'Ar -> Ar(2p9)', 'Ar -> Ar(2p10)']
    for i, nm in enumerate(named):
        reaction_list[i] = nm

    nTe, nRxn = 100, len(reaction_list)
    rng = np.random.default_rng(0)
    reaction_table = rng.uniform(1e-16, 1e-14, size=(nTe, nRxn))

    data, s_3, s_5 = find_ratio(ne=1e17, Te=3.0, p_mtorr=10.0,
                                reaction_table=reaction_table,
                                reaction_list=reaction_list)
    print("\nFirst few emission lines [nm, rel. intensity]:")
    print(data[:5])