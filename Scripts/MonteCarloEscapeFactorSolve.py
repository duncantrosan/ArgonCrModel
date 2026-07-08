# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:37:55 2026

@author: dptro
"""

# Radation Trapping Factor 

# def RedTrapHemisphere(R = 3,Tgas = 300 ,Pressure = 1,n_ls = 10^15):
    
"""
Monte Carlo radiation-trapping escape factor g for a uniformly emitting,
homogeneous hemisphere (curved dome of radius R sitting on a flat base in
the z=0 plane, occupying z >= 0).

g = < T(ell) >  averaged over
      - uniform emission point in the hemisphere volume
      - isotropic emission direction
where
      T(ell) = integral phi(nu) exp(-k0 phi(nu) ell) dnu
is the single-chord, frequency-averaged escape probability (method B:
the line profile is handled inside T by quadrature, geometry by Monte Carlo).

phi(nu) is a normalized Voigt profile. We work in dimensionless frequency
x = (nu - nu0)/dnu_D (Doppler units). The Voigt 'a' parameter and the
center optical depth tau0 = k0_centerless * R fully specify the problem.
"""

import numpy as np
from scipy.special import wofz   # Faddeeva -> Voigt
from scipy.integrate import quad


# ----------------------------------------------------------------------
# 1. Line profile  phi(x)  (normalized so integral over x = 1)
# ----------------------------------------------------------------------
def voigt_phi(x, a):
    """Normalized Voigt profile in Doppler-frequency units x, parameter a.
    a = dnu_Lorentz / (2 dnu_Doppler) * (1/sqrt? ) -- standard convention:
    H(a,x) = Re[w(x + i a)] / sqrt(pi); profile is H/(sqrt(pi)) so it
    integrates to 1 in x. Peak (a->0) = 1/sqrt(pi)."""
    z = x + 1j * a
    return wofz(z).real / np.sqrt(np.pi)


# ----------------------------------------------------------------------
# 2. Single-chord transmission  T(ell)  via 1-D quadrature over frequency
#    tau0 here is the LINE-CENTER optical depth for this particular chord:
#       tau0_chord = k0_peak * ell,  where k0_peak = k0 * phi(0).
#    We pass the chord's center optical depth directly.
# ----------------------------------------------------------------------
def T_of_tau(tau0_center, a):
    """Frequency-averaged escape prob for a chord of center optical depth tau0.
       tau0_center = (peak absorption coeff) * (chord length)."""
    phi0 = voigt_phi(0.0, a)            # profile peak value
    def integrand(x):
        phx = voigt_phi(x, a)
        # optical depth at freq x = tau0_center * phi(x)/phi(0)
        return phx * np.exp(-tau0_center * phx / phi0)
    # symmetric profile -> integrate 0..inf, double. Wings matter, go wide.
    val, _ = quad(integrand, 0, 60, limit=400)
    return 2.0 * val


# ----------------------------------------------------------------------
# 3. Hemisphere geometry: distance from point p, direction d, to boundary
#    Hemisphere: x^2+y^2+z^2 <= R^2  AND  z >= 0.
#    Two surfaces: the sphere |r|=R (dome) and the disk z=0 (base).
# ----------------------------------------------------------------------
def chord_length(p, d, R):
    """Distance along unit direction d from interior point p to the
       hemisphere boundary (min positive hit of dome or base)."""
    hits = []

    # --- sphere intersection: |p + t d|^2 = R^2 ---
    # t^2 + 2 (p.d) t + (|p|^2 - R^2) = 0  ; |d|=1
    b = np.dot(p, d)
    c = np.dot(p, p) - R * R
    disc = b * b - c
    if disc > 0:
        sq = np.sqrt(disc)
        for t in (-b - sq, -b + sq):
            if t > 1e-12:
                z = p[2] + t * d[2]
                if z >= -1e-9:            # valid only on upper hemisphere
                    hits.append(t)

    # --- base plane z = 0 intersection ---
    if abs(d[2]) > 1e-15:
        t = -p[2] / d[2]
        if t > 1e-12:
            hit = p + t * d
            if hit[0] ** 2 + hit[1] ** 2 <= R * R + 1e-9:  # within disk
                hits.append(t)

    return min(hits) if hits else 0.0


# ----------------------------------------------------------------------
# 4. Monte Carlo escape factor
# ----------------------------------------------------------------------
def escape_factor_hemisphere(tau0_R, a, N=200_000, R=1.0, seed=0):
    """
    tau0_R : center optical depth across one radius = k0_peak * R.
    a      : Voigt parameter.
    Returns g = <T(ell)> and its standard error.
    """
    rng = np.random.default_rng(seed)
    k0_peak_times_R = tau0_R          # so center optical depth of chord ell
                                      # is tau0_R * (ell/R)

    # --- uniform points in hemisphere volume by rejection in box ---
    pts = np.empty((N, 3))
    filled = 0
    while filled < N:
        m = N - filled
        cand = rng.uniform(-1, 1, size=(int(m * 2.2), 3))
        cand[:, 2] = np.abs(cand[:, 2])           # force z>=0 (upper half)
        r2 = np.einsum('ij,ij->i', cand, cand)
        good = cand[r2 <= 1.0]
        take = min(len(good), m)
        pts[filled:filled + take] = good[:take]
        filled += take
    pts *= R

    # --- isotropic directions: uniform cos(theta), uniform phi ---
    phi = rng.uniform(0, 2 * np.pi, N)
    cost = rng.uniform(-1, 1, N)
    sint = np.sqrt(1 - cost ** 2)
    dirs = np.column_stack([sint * np.cos(phi), sint * np.sin(phi), cost])

    # --- per-photon chord, then T(ell) ---
    Tvals = np.empty(N)
    for i in range(N):
        ell = chord_length(pts[i], dirs[i], R)
        tau_chord = k0_peak_times_R * (ell / R)
        Tvals[i] = T_of_tau(tau_chord, a)

    g = Tvals.mean()
    se = Tvals.std() / np.sqrt(N)
    return g, se




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


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    a = 0.05          # Voigt parameter (example; compute from your conditions)
    for tau0_R in [0.0, 1.0, 10.0, 100.0, 1000.0]:
        # N kept modest in demo because T_of_tau quad is the cost
        g, se = escape_factor_hemisphere(tau0_R, a, N=8000, seed=1)
        print(f"tau0(R) = {tau0_R:8.1f}   g = {g:.4e} +/- {se:.1e}")