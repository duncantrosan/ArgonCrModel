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

def solve_Te(R, n_g, Tg, Te_lo=0.2, Te_hi=30.0):
    f = lambda Te: n_g * K_iz(Te) - sum(loss_rates(Te, R, n_g, Tg))
    return brentq(f, Te_lo, Te_hi)

def solve(p_Torr, R, P_abs, Tg=300.0):
    n_g = gas_density(p_Torr, Tg)
    Te = solve_Te(R, n_g, Tg)
    nu_flat, nu_D = loss_rates(Te, R, n_g, Tg)
    V = (2 / 3) * np.pi * R**3
    V_s = Te * np.log(np.sqrt(M_Ar / (2 * np.pi * m_e)))      # floating sheath, eV
    E_loss_wall = 2 * Te + 0.5 * Te + V_s                     # per pair to flat face
    E_loss_diff = 2 * Te + 0.5 * Te                           # per pair out curved face
    denom = e * V * (n_g * K_iz(Te) * E_c(Te) + nu_flat * E_loss_wall + nu_D * E_loss_diff)
    n_e = P_abs / denom
    return dict(p_Torr=p_Torr, R=R, P_abs=P_abs, n_g=n_g, Te=Te, n_e=n_e,
                nu_flat=nu_flat, nu_D=nu_D, E_c=E_c(Te), V_s=V_s,
                h_l=h_l(Te, R, *ion_props(n_g, Tg)))


# ---- "how Maxwellian is my plasma?" self-consistency check ----------------
# Rate race, energy by energy: e-e energy exchange (Maxwellianizes) vs energy
# loss to the gas (elastic recoil + inelastic tail-scalping). The crossover
# energy eps_M is Maxwellian below, drooping above. The question is not "is it
# Maxwellian?" but "is eps_M above the thresholds I care about?".
#
# CIRCULARITY NOTE: n_e here came from a model that ASSUMED a Maxwellian EEDF,
# so feeding n_e back in is a self-consistency flag, not an independent proof.
# If it flags "tail depleted", the model is questioning its own assumption at
# that operating point -- treat it as a ballpark warning.
#
# Cross sections are representative Ar order-of-magnitude values; swap in your
# CR-model sigmas for numbers you'd quote. The eps^-3/2 vs inelastic-wall
# conclusion is robust regardless.
SIG_EN_M = 2.0e-19      # e-neutral elastic momentum-transfer x-sec [m^2]
SIG_EXC0 = 1.0e-21      # lumped e-impact excitation x-sec scale  [m^2]
E_EXC_THR = 11.55       # lowest Ar excitation threshold          [eV]

def _v_e(eps):                               # electron speed at energy eps [eV]
    return np.sqrt(2 * eps * e / m_e)

def _sig_exc(eps):                           # ramp above threshold
    return np.where(eps > E_EXC_THR,
                    SIG_EXC0 * (1 - E_EXC_THR / np.maximum(eps, E_EXC_THR)), 0.0)

def nu_ee(eps, n_e, Te):
    """e-e energy-exchange rate for a test electron of energy eps [1/s].
    Standard fast-test-electron scaling; note the eps^-3/2 -- the tail is the
    hardest part to keep Maxwellian."""
    n_cm = n_e * 1e-6
    lnL = max(5.0, 23.5 - np.log(np.sqrt(n_cm)) + 1.25 * np.log(Te))
    return 2.91e-6 * n_cm * lnL * eps**-1.5

def nu_eloss(eps, n_g):
    """Energy-loss rate to the gas [1/s]: elastic recoil + inelastic."""
    nu_m = n_g * SIG_EN_M * _v_e(eps)
    return (2 * m_e / M_Ar) * nu_m + n_g * _sig_exc(eps) * _v_e(eps)

def eps_M(n_e, n_g, Te, grid=None):
    """Maxwellization energy: highest eps where nu_ee still >= nu_eloss [eV]."""
    if grid is None:
        grid = np.linspace(1.0, 40.0, 600)
    below = np.where(nu_ee(grid, n_e, Te) - nu_eloss(grid, n_g) < 0)[0]
    return grid[below[0]] if len(below) else grid[-1]

def maxwellian_check(res, thresholds=None):
    """Take a solve() result dict, return eps_M and a per-threshold verdict.
    thresholds: {label: energy_eV} to test (defaults cover the Ar levels)."""
    if thresholds is None:
        thresholds = {"bulk (~2Te)": 2 * res["Te"],
                      "excite 4p": E_EXC_THR,
                      "4p/5p/4d": 13.5,
                      "ioniz.": E_IZ}
    em = eps_M(res["n_e"], res["n_g"], res["Te"])
    verdict = {lbl: ("Maxwellian" if E <= em else "tail-depleted")
               for lbl, E in thresholds.items()}
    return dict(eps_M=em, ionization_frac=res["n_e"] / res["n_g"], verdict=verdict)


# ---- demo -----------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    R = 2/100        # m
    P_abs = 120     # W absorbed by the plasma
    Tg = 700.0

    r = solve(1.0, R, P_abs, Tg)
    for k, v in r.items():
        print(f"{k:8s} = {v:.4g}")

    mx = maxwellian_check(r)
    print(f"\neps_M    = {mx['eps_M']:.1f} eV   (ionization frac = {mx['ionization_frac']:.1e})")
    for lbl, v in mx["verdict"].items():
        print(f"  {lbl:12s} -> {v}")

    p = np.logspace(-1, 3, 80)   # 0.1 Torr floor: below that Te exceeds the
    res = [solve(pi, R, P_abs, Tg) for pi in p]   # [0.2,30] bracket in solve_Te
    Te = np.array([x["Te"] for x in res])
    ne = np.array([x["n_e"] for x in res])
    frac_flat = np.array([x["nu_flat"] / (x["nu_flat"] + x["nu_D"]) for x in res])
    em = np.array([eps_M(x["n_e"], x["n_g"], x["Te"]) for x in res])

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    ax[0].loglog(p, Te); ax[0].set_ylabel("Te [eV]")
    ax[1].loglog(p, ne); ax[1].set_ylabel("n_e [m$^{-3}$]")
    ax[2].semilogx(p, frac_flat); ax[2].set_ylabel("fraction of loss to flat face")
    # eps_M panel: Maxwellian below the curve, depleted above. Threshold lines
    # show where the levels you measure sit relative to eps_M.
    ax[3].semilogx(p, em, "k-", label=r"$\varepsilon_M$")
    ax[3].axhline(E_EXC_THR, color="tab:orange", ls="--", lw=1, label="4p excite")
    ax[3].axhline(E_IZ, color="tab:red", ls="--", lw=1, label="ionization")
    ax[3].fill_between(p, em, 40, color="tab:red", alpha=0.08)
    ax[3].set_ylabel(r"energy [eV] -- Maxwellian below $\varepsilon_M$")
    ax[3].set_ylim(0, 25); ax[3].legend(fontsize=8)
    for a in ax:
        a.set_xlabel("p [Torr]"); a.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"Hemisphere R = {R*1e3:.1f} mm, P_abs = {P_abs} W")
    fig.tight_layout()
    plt.show()