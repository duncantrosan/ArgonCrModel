# -*- coding: utf-8 -*-
"""
LiebermanGlobalModel.py

Lieberman & Lichtenberg (Principles of Plasma Discharges, 2nd ed., Sec. 10.2) global model of a
cylindrical argon discharge, used two ways:

  1. Reproduce the low-pressure RF numbers: the universal Te vs n_g d_eff curve (Fig. 10.1), the
     worked example (R = 15 cm, l = 30 cm, 1 mTorr, 800 W absorbed) and Te vs p / n0 vs P_abs.
  2. Run the SAME model, unchanged, out to 1.5 Torr and ask how much absorbed power it takes to
     hold the centre density at the 2.45 GHz critical density n_c.

Model (Maxwellian electrons, L&L Table 3.3 rate fits)
  h_l = 0.86 [3 + l/(2 lam_i) + (0.86 l u_B / (pi D_a))^2]^-1/2       axial edge/centre
  h_R = 0.80 [4 + R/lam_i     + (0.80 R u_B / (chi01 J1(chi01) D_a))^2]^-1/2   radial
        (Lee & Lieberman form: free fall at low p, ambipolar diffusion at high p)
  A_eff = 2 pi R (R h_l + l h_R),   d_eff = R l / (2 (R h_l + l h_R))
  particle:  K_iz(Te) / u_B(Te) = 1 / (n_g d_eff)                   -> Te
  power:     n0 = P_abs / (e u_B A_eff E_T),  E_T = E_c + 2 Te + E_i,  E_i = V_s + Te/2

Output in Experimental_Data/Output/LiebermanGlobalModel/.  Run from Spyder (F5) or
python SmallAnalysisScripts/LiebermanGlobalModel.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import j1

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT_DIR, "Experimental_Data", "Output", "LiebermanGlobalModel")

e = 1.602e-19
k_B = 1.381e-23
m_e = 9.109e-31
eps0 = 8.854e-12
M_Ar = 39.948 * 1.6605e-27
SIG_I = 1.0e-18           # Ar+/Ar ion-neutral cross section, L&L lam_i = 1/(n_g sigma_i)
E_IZ, E_EX = 15.76, 12.14
CHI01 = 2.405
N_C = (2 * np.pi * 2.45e9)**2 * eps0 * m_e / e**2   # 7.4e16 m^-3

# L&L worked example (Sec. 10.2) and its gas temperature (1 mTorr -> n_g = 3.3e19 m^-3)
R_LL, L_LL, TG_LL = 0.15, 0.30, 293.0


# ---- L&L Table 3.3 Maxwellian rate coefficients (Te in eV, m^3/s) ----------------------------
def K_iz(Te):
    return 2.34e-14 * Te**0.59 * np.exp(-17.44 / Te)

def K_ex(Te):
    return 2.48e-14 * Te**0.33 * np.exp(-12.78 / Te)

def K_el(Te):
    lnT = np.log(Te)
    return 2.336e-14 * Te**1.609 * np.exp(0.0618 * lnT**2 - 0.1171 * lnT**3)

def E_c(Te):
    """Collisional energy lost per electron-ion pair created [eV] (L&L Fig. 3.17)."""
    return E_IZ + E_EX * K_ex(Te) / K_iz(Te) + 3 * (m_e / M_Ar) * Te * K_el(Te) / K_iz(Te)

def u_B(Te):
    return np.sqrt(e * Te / M_Ar)


# ---- geometry / transport --------------------------------------------------------------------
def n_gas(p_Torr, Tg):
    return p_Torr * 133.32 / (k_B * Tg)

def h_factors(Te, R, l, n_g, Tg):
    lam_i = 1.0 / (n_g * SIG_I)
    v_i = np.sqrt(8 * k_B * Tg / (np.pi * M_Ar))
    D_a = e * lam_i / (M_Ar * v_i) * Te                     # mu_i Te  (Te >> Ti)
    hl = 0.86 / np.sqrt(3 + l / (2 * lam_i) + (0.86 * l * u_B(Te) / (np.pi * D_a))**2)
    hR = 0.80 / np.sqrt(4 + R / lam_i + (0.80 * R * u_B(Te) / (CHI01 * j1(CHI01) * D_a))**2)
    return hl, hR, lam_i

def d_eff(Te, R, l, n_g, Tg):
    hl, hR, _ = h_factors(Te, R, l, n_g, Tg)
    return R * l / (2 * (R * hl + l * hR))

def Te_universal(ngd):
    """Fig. 10.1: Te from K_iz/u_B = 1/(n_g d_eff)."""
    return brentq(lambda T: K_iz(T) / u_B(T) - 1.0 / ngd, 0.1, 500.0)

def solve(p_Torr, P_abs, R=R_LL, l=L_LL, Tg=TG_LL):
    n_g = n_gas(p_Torr, Tg)
    Te = brentq(lambda T: n_g * d_eff(T, R, l, n_g, Tg) * K_iz(T) / u_B(T) - 1.0, 0.3, 200.0)
    hl, hR, lam_i = h_factors(Te, R, l, n_g, Tg)
    A_eff = 2 * np.pi * R * (R * hl + l * hR)
    V_s = Te * np.log(np.sqrt(M_Ar / (2 * np.pi * m_e)))
    E_T = E_c(Te) + 2 * Te + V_s + 0.5 * Te
    n0 = P_abs / (e * u_B(Te) * A_eff * E_T)
    return dict(p_Torr=p_Torr, P_abs=P_abs, R=R, l=l, n_g=n_g, lam_i=lam_i, h_l=hl, h_R=hR,
                A_eff=A_eff, d_eff=R * l / (2 * (R * hl + l * hR)), Te=Te, u_B=u_B(Te),
                E_c=E_c(Te), E_T=E_T, n0=n0, P_per_n0=P_abs / n0)


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # ---- 1. L&L worked example -----------------------------------------------------------------
    ex = solve(1e-3, 800.0)
    print("L&L Sec. 10.2 example  (R = 15 cm, l = 30 cm, 1 mTorr, 800 W)")
    for k in ("n_g", "lam_i", "h_l", "h_R", "A_eff", "d_eff", "Te", "u_B", "E_c", "E_T", "n0"):
        print(f"  {k:6s} = {ex[k]:.4g}")

    # ---- 2. Te and n0 vs p from 1 mTorr to 3 Torr, same chamber --------------------------------
    p = np.logspace(np.log10(5e-4), np.log10(3.0), 120)
    sweep = [solve(pi, 800.0) for pi in p]
    Te_p = np.array([s["Te"] for s in sweep])
    n0_p = np.array([s["n0"] for s in sweep])

    # ---- 3. 1.5 Torr: P_abs for n0 = n_c vs plasma size (cylinder, l = 2R) ---------------------
    Tg_hp = 350.0                       # LAS Tg at 1.5 Torr low power is 324-372 K
    print(f"\n1.5 Torr, Tg = {Tg_hp:.0f} K, n_c(2.45 GHz) = {N_C:.3g} m^-3")
    hp = solve(1.5, 1.0, Tg=Tg_hp)
    print(f"  L&L chamber: Te = {hp['Te']:.2f} eV, E_T = {hp['E_T']:.0f} eV, "
          f"h_l = {hp['h_l']:.2e}, h_R = {hp['h_R']:.2e}, P(n_c) = {hp['P_per_n0'] * N_C:.2f} W")
    Rs = np.logspace(np.log10(0.005), np.log10(0.15), 40)
    P_nc = []
    for R in Rs:
        s = solve(1.5, 1.0, R=R, l=2 * R, Tg=Tg_hp)
        P_nc.append(s["P_per_n0"] * N_C)
    P_nc = np.array(P_nc)
    for R in (0.01, 0.02, 0.04, 0.08):
        s = solve(1.5, 1.0, R=R, l=2 * R, Tg=Tg_hp)
        print(f"  R = {R*100:4.0f} cm, l = 2R: Te = {s['Te']:.2f} eV, "
              f"P_abs(n_c) = {s['P_per_n0'] * N_C:.3g} W, n0(6 W) = {6 / s['P_per_n0']:.3g} m^-3")

    # ---- figure --------------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

    a = ax[0, 0]
    ngd = np.logspace(17, 25, 200)
    a.loglog(ngd, [Te_universal(x) for x in ngd], "k-")
    a.plot(ex["n_g"] * ex["d_eff"], ex["Te"], "o", color="tab:blue", label="L&L example, 1 mTorr")
    a.plot(hp["n_g"] * hp["d_eff"], hp["Te"], "s", color="tab:red", label="same chamber, 1.5 Torr")
    a.set_xlabel(r"$n_g d_{eff}$ [m$^{-2}$]"); a.set_ylabel(r"$T_e$ [eV]")
    a.set_title("Universal Te curve (L&L Fig. 10.1)"); a.legend(fontsize=8)

    a = ax[0, 1]
    a.semilogx(p * 1e3, Te_p, "k-")
    a.axvline(1.5e3, color="tab:red", ls="--", lw=1)
    a.text(1.5e3, Te_p.max() * 0.95, " 1.5 Torr", color="tab:red", fontsize=8, va="top")
    a.set_xlabel("p [mTorr]"); a.set_ylabel(r"$T_e$ [eV]")
    a.set_title("Te vs p, R = 15 cm, l = 30 cm")

    a = ax[1, 0]
    P = np.logspace(0, np.log10(3000), 100)
    for pi, c in ((1e-3, "C0"), (1e-2, "C1"), (0.1, "C2"), (1.5, "C3")):
        Tg = TG_LL if pi < 0.5 else Tg_hp
        k = solve(pi, 1.0, Tg=Tg)["P_per_n0"]
        a.loglog(P, P / k, color=c, label=f"{pi*1e3:g} mTorr")
    a.plot(800, ex["n0"], "o", color="C0")
    a.axhline(N_C, color="k", ls=":", lw=1)
    a.text(1.2, N_C * 1.15, r"$n_c$ (2.45 GHz)", fontsize=8)
    a.set_xlabel(r"$P_{abs}$ [W]"); a.set_ylabel(r"$n_0$ [m$^{-3}$]")
    a.set_title("n0 vs absorbed power, R = 15 cm, l = 30 cm"); a.legend(fontsize=8)

    a = ax[1, 1]
    a.loglog(Rs * 100, P_nc, "k-")
    a.axhspan(1, 10, color="tab:orange", alpha=0.15, label="1-10 W sweep")
    a.set_xlabel("plasma radius R [cm] (l = 2R)"); a.set_ylabel(r"$P_{abs}$ for $n_0 = n_c$ [W]")
    a.set_title(f"1.5 Torr, Tg = {Tg_hp:.0f} K"); a.legend(fontsize=8)

    for a in ax.flat:
        a.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "LiebermanGlobalModel.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")
    plt.show()


if __name__ == "__main__":
    main()
