# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:27:23 2026

@author: dptro
"""

"""
Verification of the Monte Carlo hemisphere escape factor against
Bhatia & Kastner (1997), JQSRT 58, 347 -- Table 2, column HEF_D
(Doppler-profile hemisphere escape factor).

CRITICAL compatibility conditions:
  1. Their profile is PURE DOPPLER -> run MC at a = 0.
  2. Their x-scaling uses the 1/e Doppler half-width -> our tau0 must be
     line-center optical depth across one radius with the same convention.
  3. tau0 axis is log10(tau0) from -3.0 to +5.0.

Their notation "A + B" means A x 10^B  (e.g. "0.91584 - 1" = 0.091584).
"""

import numpy as np
# from MonteCarloEscapeFactorSolve 
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Bhatia & Kastner Table 2, HEF_D column.
# Key = log10(tau0); value = HEF_D (already decoded from A + B notation).
# Transcribed from the paper. A representative subset across the full range is
# used here for the check; extend with the full 0.1-spaced table for production.
# ---------------------------------------------------------------------------
HEF_D = {
    -3.0: 0.99960,
    -2.0: 0.99603,
    -1.0: 0.96156,
    -0.5: 0.88683,
     0.0: 0.70777,
     0.5: 0.42665,
     1.0: 0.19301,
     1.5: 0.075385,     # "0.75385 - 1"
     2.0: 0.027542,     # "0.27542 - 1"
     2.5: 0.0097627,    # "0.97627 - 2"
     3.0: 0.0033607,    # "0.33607 - 2"
     3.5: 0.0011500,    # "0.11500 - 2"
     4.0: 0.00038601,   # "0.38601 - 3"
     4.5: 0.00012695,   # "0.12695 - 3"
     5.0: 0.000042335,  # "0.42335 - 4"
}

# ---------------------------------------------------------------------------
# Bhatia & Kastner logistic fit for HEF_D (their Table 3):
#   y(tau0) = a / (1 + exp[b (log10 tau0 - c)])
# ---------------------------------------------------------------------------
def hef_d_logistic(tau0, a=1.0010280, b=2.3024323, c=0.3787757):
    return a / (1.0 + np.exp(b*(np.log10(tau0) - c)))


def escape_factor_fast(tau0_R, a, N=2_000_000, R=1.0, seed=0, n_grid=400):
    """Same physics as escape_factor_hemisphere but precomputes T(tau_chord)
    on a log grid and interpolates -- much faster than per-photon quad."""
    from MonteCarloEscapeFactorSolve  import chord_length, T_of_tau
    rng = np.random.default_rng(seed)

    pts = np.empty((N,3)); filled = 0
    while filled < N:
        c = rng.uniform(-1,1,size=(int((N-filled)*2.2),3))
        c[:,2] = np.abs(c[:,2])
        good = c[np.einsum('ij,ij->i',c,c) <= 1.0]
        take = min(len(good), N-filled)
        pts[filled:filled+take] = good[:take]; filled += take
    pts *= R

    phi = rng.uniform(0,2*np.pi,N)
    cost = rng.uniform(-1,1,N); sint = np.sqrt(1-cost**2)
    dirs = np.column_stack([sint*np.cos(phi), sint*np.sin(phi), cost])

    ells = np.array([chord_length(pts[i], dirs[i], R) for i in range(N)])
    tau_chords = tau0_R * (ells / R)

    tmax = max(tau_chords.max(), 1e-3)
    grid = np.concatenate([[0.0], np.logspace(-3, np.log10(tmax*1.01), n_grid)])
    Tgrid = np.array([T_of_tau(t, a) for t in grid])
    Tvals = np.interp(tau_chords, grid, Tgrid)
    return Tvals.mean(), Tvals.std()/np.sqrt(N)


def run_verification(N=2_000_000, seed=5):
    print(f"{'log tau0':>9} {'tau0':>11} {'HEF_D(tab)':>12} "
          f"{'MC(a=0)':>11} {'rel.err %':>10} {'logistic':>11}")
    print('-'*70)
    logs = sorted(HEF_D.keys())
    rel_errors = []
    Taus = []
    g_all = []
    for lg in logs:
        tau0 = 10.0**lg
        g_mc, se = escape_factor_fast(tau0, a=0.0, N=N, seed=seed)
        ref = HEF_D[lg]
        rel = (g_mc - ref)/ref * 100.0
        rel_errors.append(rel)
        fit = hef_d_logistic(tau0)
        Taus.append(tau0)
        g_all.append(g_mc)
        print(f"{lg:>9.1f} {tau0:>11.3e} {ref:>12.5e} "
              f"{g_mc:>11.4e} {rel:>+10.2f} {fit:>11.4e}")
    rel_errors = np.array(rel_errors)
    print('-'*70)
    print(f"Max |rel err|: {np.abs(rel_errors).max():.2f}%   "
          f"RMS: {np.sqrt(np.mean(rel_errors**2)):.2f}%")
    return rel_errors, Taus ,g_all , HEF_D

def ConvergenceTest(logTau = 1,NSteps = 10 , NMax = 2_000_000):
    Tau0 = 10**logTau
    g_Converged, STD = escape_factor_fast(Tau0, 0, NMax)
    
    print(f'The converged result is {g_Converged}')
    logMax = np.log10(NMax)
    LogSpace = np.logspace(2,logMax,NSteps)
    
    print(f"{'Num. Part':>9.1} {'Error':>10.2} ")
    
    Error = []
    g = []
    for n in LogSpace :
        nn = int(n)
        g_Low, STD = escape_factor_fast(Tau0, 0, nn)
        rel = (g_Low - g_Converged)/g_Converged * 100.0
        Error.append(rel)
        g.append(g_Low)
        print(f"{n:>9.1f} {rel:>+10.2f}")
    return Error,g,nn
        
    

if __name__ == '__main__':
    RelativeErrors , Tau , g_MonteCarlo , ReferenceValues = run_verification()
    Ref = list(ReferenceValues.values())
    plt.figure()
    plt.loglog(Tau , g_MonteCarlo , label = 'Monte Carlo Results')
    plt.loglog(Tau, Ref, label = 'Reference Data')
    plt.xlabel(r'$\tau / \mathrm{ m}^2 \cdot \mathrm{ s}^{-1}$')
    plt.ylabel('Escape Factor / unitless')
    plt.legend()
    Error , gs,Npart = ConvergenceTest(1,80)
    plt.figure()
    LogSpace = np.logspace(2,np.log10(2_000_000),80)
    Error = np.abs(np.array(Error))
    x = 1/np.sqrt(LogSpace)
    plt.plot(x , Error , label = 'Error')
    plt.xlabel(r'1/\sqrt{N} / unitless')
    plt.ylabel('Convergence Error / unitless')
    