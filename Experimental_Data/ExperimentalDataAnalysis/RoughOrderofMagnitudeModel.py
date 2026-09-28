# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:46:11 2026

@author: dptro
"""

"""
Rough 0D (global) argon model, Lieberman & Lichtenberg style, for a
hemispherical plasma of radius R.

Geometry / loss channels
------------------------
  * Flat face (area pi R^2): absorbing wall. Ions leave at the Bohm speed
    with an edge-to-centre density ratio h_l (Lieberman/Lee form that
    covers low -> high pressure, l = R).
  * Curved face (area 2 pi R^2): no wall, plasma leaks out by ambipolar
    diffusion. Modelled as a volume loss rate nu_D = D_a / Lambda^2 with
    Lambda = R/pi (lowest spherical diffusion mode).

Balances
--------
  Particle:  n_g K_iz(Te) = nu_flat(Te) + nu_D(Te)      -> Te
  Power:     P_abs = e n_e V [ n_g K_iz E_c + nu_flat (2Te + Te/2 + V_s)
                                            + nu_D    (2Te + Te/2) ]  -> n_e

Rate coefficients: Maxwellian fits from Lieberman & Lichtenberg Table 3.3.
Everything SI except Te / energies in eV.
"""

import numpy as np
from scipy.optimize import brentq

# ---- constants ------------------------------------------------------------
e = 1.602e-19
k_B = 1.381e-23
m_e = 9.109e-31
M_Ar = 39.948 * 1.6605e-27
SIG_IN = 1.0e-18      # Ar+ / Ar momentum-transfer + CX cross section [m^2]
E_IZ = 15.76          # eV
E_EX = 12.14          # eV (lumped excitation level)


# ---- rate coefficients (Te in eV, result m^3/s) ---------------------------
def K_iz(Te):
    return 2.34e-14 * Te**0.59 * np.exp(-17.44 / Te)

def K_ex(Te):
    return 2.48e-14 * Te**0.33 * np.exp(-12.78 / Te)

def K_el(Te):
    lnT = np.log(Te)
    return 2.336e-14 * Te**1.609 * np.exp(0.0618 * lnT**2 - 0.1171 * lnT**3)

def E_c(Te):
    """Collisional energy lost per electron-ion pair created [eV]."""
    return E_IZ + E_EX * K_ex(Te) / K_iz(Te) + 3 * (m_e / M_Ar) * Te * K_el(Te) / K_iz(Te)


# ---- transport ------------------------------------------------------------
def gas_density(p_Torr, Tg):
    return p_Torr * 133.32 / (k_B * Tg)

def ion_props(n_g, Tg):
    lam_i = 1.0 / (n_g * SIG_IN)
    v_i = np.sqrt(8 * k_B * Tg / (np.pi * M_Ar))
    mu_i = e * lam_i / (M_Ar * v_i)          # e/(M nu_mi), nu_mi = v_i/lam_i
    return lam_i, mu_i

def u_B(Te):
    return np.sqrt(e * Te / M_Ar)

def D_a(Te, mu_i):
    return mu_i * Te                          # Te >> Ti

def h_l(Te, R, lam_i, mu_i):
    """Edge-to-centre ratio, valid low -> high pressure (L&L eq. 10.2.x)."""
    return 0.86 / np.sqrt(3 + R / (2 * lam_i) + (0.86 * R * u_B(Te) / (np.pi * D_a(Te, mu_i)))**2)


# ---- model ----------------------------------------------------------------
def loss_rates(Te, R, n_g, Tg):
    lam_i, mu_i = ion_props(n_g, Tg)
    V = (2 / 3) * np.pi * R**3
    A_flat = np.pi * R**2
    nu_flat = h_l(Te, R, lam_i, mu_i) * u_B(Te) * A_flat / V
    Lam = R / np.pi
    nu_D = D_a(Te, mu_i) / Lam**2
    return nu_flat, nu_D

def solve_Te(R, n_g, Tg, f_iz=1.0, Te_lo=0.2, Te_hi=30.0):
    f = lambda Te: n_g * f_iz * K_iz(Te) - sum(loss_rates(Te, R, n_g, Tg))
    return brentq(f, Te_lo, Te_hi)

def solve(p_Torr, R, P_abs, Tg=300.0, f_iz=1.0):
    n_g = gas_density(p_Torr, Tg)
    Te = solve_Te(R, n_g, Tg, f_iz)          # <-- f_iz enters here only
    nu_flat, nu_D = loss_rates(Te, R, n_g, Tg)
    V = (2 / 3) * np.pi * R**3
    V_s = Te * np.log(np.sqrt(M_Ar / (2 * np.pi * m_e)))
    E_loss_wall = 2 * Te + 0.5 * Te + V_s
    E_loss_diff = 2 * Te + 0.5 * Te
    # NOTE: collisional term keeps the DIRECT K_iz(Te)*E_c(Te), not f_iz*K_iz.
    denom = e * V * (n_g * K_iz(Te) * E_c(Te) + nu_flat * E_loss_wall + nu_D * E_loss_diff)
    n_e = P_abs / denom
    return dict(p_Torr=p_Torr, R=R, P_abs=P_abs, n_g=n_g, Te=Te, n_e=n_e,
                nu_flat=nu_flat, nu_D=nu_D, E_c=E_c(Te), V_s=V_s, f_iz=f_iz,
                h_l=h_l(Te, R, *ion_props(n_g, Tg)))

# ---- demo -----------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Tg = 700.0
    p  = np.logspace(-1, 3, 80)

    # ---- pick what to sweep -------------------------------------------------
    sweep = "f_iz"          # "P_abs" or "R"
    if sweep == "P_abs":
        R      = 4e-2                          # fixed geometry
        values = np.linspace(10, 120, 10)      # W
        label  = lambda v: f"$P_{{abs}}$ = {v:.0f} W"
        get    = lambda v, pi: solve(pi, R, v, Tg)
    elif sweep == "R":  # sweep R, fixed power
        P_abs  = 80.0
        values = np.linspace(1e-2, 6e-2, 6)    # m
        label  = lambda v: f"R = {v*1e2:.1f} cm"
        get    = lambda v, pi: solve(pi, v, P_abs, Tg)
    elif sweep == "Tg":
        P_abs  = 80.0
        R = 2/100
        values = np.linspace(1e-2, 6e-2, 6)    # m
        label  = lambda v: f"R = {v*1e2:.1f} cm"
        get    = lambda v, pi: solve(pi, v, P_abs, Tg)
    else:  # sweep f_iz (metastable proxy), fixed R and power
        R, P_abs = 2e-2, 100
        values = [1.0, 1.5, 2.0, 3.0, 5.0]
        label  = lambda v: f"$f_{{iz}}$ = {v:.1f}"
        get    = lambda v, pi: solve(pi, R, P_abs, Tg, f_iz=v)

        
        
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    colors  = plt.cm.viridis(np.linspace(0, 1, len(values)))

    for v, c in zip(values, colors):
        res  = [get(v, pi) for pi in p]
        Te   = np.array([x["Te"] for x in res])
        ne   = np.array([x["n_e"] for x in res])
        frac = np.array([x["nu_flat"] / (x["nu_flat"] + x["nu_D"]) for x in res])
        ax[0].semilogx(p, Te,   color=c, label=label(v))
        ax[1].loglog  (p, ne,   color=c, label=label(v))
        ax[2].semilogx(p, frac, color=c, label=label(v))

    ax[0].set_ylabel("Te [eV]")
    ax[1].set_ylabel("n_e [m$^{-3}$]")
    ax[2].set_ylabel("fraction of loss to flat face")
    for a in ax:
        a.set_xlabel("p [Torr]"); a.grid(True, which="both", alpha=0.3)
    ax[1].legend(fontsize=8, title=sweep)
    fig.suptitle(f"Sweep over {sweep}   (Tg = {Tg:.0f} K)")
    fig.tight_layout()
    plt.show()