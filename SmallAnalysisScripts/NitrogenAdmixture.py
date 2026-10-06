# -*- coding: utf-8 -*-
"""
NitrogenAdmixture.py

Metastable densities and Ar I line ratios as a function of the N2 admixture,
from the CR model with the N2 physics of Scripts/MainFileWithNitrogen.py
(quenching of the Ar(4s) levels by N2, dilution of the Ar ground state, and -
with EEDF_MODE 'bolsig' - the EEDF of each Ar/N2 mixture).

  EEDF_MODE 'maxwell' : one curve per Te in TE_LIST (the EEDF does not see the N2)
  EEDF_MODE 'bolsig'  : one curve per E/N in EN_LIST; BOLSIG+ is run for each
                        mixture at those E/N only (work files in OUTDIR/bolsig_work).
                        Te_eff then changes along each curve as N2 cools the EEDF.

Figure and csv in Experimental_Data/Output/NitrogenAdmixture/.
Run from Spyder: edit the settings, press F5.
"""
import ast
import contextlib
import csv
import io
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "Scripts")
sys.path.insert(0, SCRIPTS_DIR)
import HelperFunctions as he                        # noqa: E402

# ---- settings -------------------------------------------------------------------
PRESSURE, TG, R = 1.0, 300.0, 0.02                  # Torr (total), K, m - as MainFileWithNitrogen
N2_PERCENT = np.round(np.arange(0, 10.01, 0.5), 2)  # % N2, the x-axis (studied up to 10 %)
NE = 1e17                                           # m^-3
QUENCH_CHOICE = "median"                            # or a reference, e.g. 'Velazco1978'
TRAP_LINES = "all"
EEDF_MODE = "bolsig"                               # 'maxwell' or 'bolsig'
TE_LIST = [1.0, 1.5, 2.0, 3.0]                      # eV, 'maxwell'
EN_LIST = [5, 10, 20, 50]                           # Td, 'bolsig'
RATIOS = [(750.387, 751.465), (811.531, 750.387), (420.067, 750.387)]   # nm (air)
OUTDIR = os.path.join(ROOT_DIR, "Experimental_Data", "Output", "NitrogenAdmixture")


def load_defs(name, namespace):
    """Function definitions (and imports) of a Scripts/ file, without running it."""
    path = os.path.join(SCRIPTS_DIR, name)
    tree = ast.parse(open(path, encoding="utf-8").read())
    defs = ast.Module([n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))], [])
    exec(compile(defs, path, "exec"), namespace)


cr = {"__file__": os.path.join(SCRIPTS_DIR, "MainFileV2.py"), "OUTPUT_DIR": OUTDIR, "plt": plt,
      "MB_XSEC_DIR": os.path.join(os.path.dirname(os.path.dirname(str(he.MULTIBOLT_EXE))), "cross-sections")}
load_defs("MainFileV2.py", cr)                      # CRModel
load_defs("MainFileWithNitrogen.py", cr)            # SetNitrogenQuenching, MixtureXsecFile, line


def escape_interpolator():
    with contextlib.redirect_stdout(io.StringIO()):
        MD, RTM = he.GetData()
    return MD, he.EscapeFactorInterpolator(RTM)


def eedfs_for(pct):
    """[(curve label, EEDF)] at this N2 fraction, one per TE_LIST / EN_LIST entry."""
    if EEDF_MODE == "maxwell":
        return [(f"$T_e$ = {t:g} eV", he.MaxwellianEEDF(t)) for t in TE_LIST]
    if EEDF_MODE != "bolsig":
        raise ValueError(f"EEDF_MODE must be 'maxwell' or 'bolsig', not {EEDF_MODE!r}")
    x = pct / 100
    runs = he.RunBolsigScript(cr["MixtureXsecFile"](), ["Ar", "N2"], [1 - x, x],
                              [{"EN_Td": en} for en in EN_LIST],
                              os.path.join(OUTDIR, "bolsig_work", f"{pct:g}pct"),
                              n_grid=500, precision=1e-25, max_iter=20000)
    return [(f"E/N = {en:g} Td", he.BolsigEEDF(r)) for en, r in zip(EN_LIST, runs)]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    MD, interp = escape_interpolator()
    MD = he.AddDiffusionLoss(MD, PRESSURE, TG, R)              # Ar* diffusing in the total gas
    Q = he.ImportArQuenchingData("N2", QUENCH_CHOICE)
    N = he.Torr2Volume(PRESSURE, TG)
    rows = []
    for pct in N2_PERCENT:
        x = pct / 100
        cr["SetNitrogenQuenching"](MD, Q, x * N, (1 - x) * N)
        for label, eedf in eedfs_for(pct):
            with contextlib.redirect_stdout(io.StringIO()):
                D, _, EI, solver = cr["CRModel"](MD, eedf, NE, (1 - x) * PRESSURE, TG, R, interp,
                                                 trap_lines=TRAP_LINES)
            row = {"N2_percent": pct, "curve": label, "Te_eff_eV": eedf["Te_eff"], "converged": solver["converged"],
                   "n_1s5": D["4s1"]["density_m^-3"], "n_1s3": D["4s3"]["density_m^-3"]}
            for a, b in RATIOS:
                row[f"{a:.1f}/{b:.1f}"] = cr["line"](EI, a) / cr["line"](EI, b)
            rows.append(row)
        print(f"{pct:5.1f} % N2 done" + ("" if all(r["converged"] for r in rows) else "  (some not converged)"))

    tag = EEDF_MODE
    with open(os.path.join(OUTDIR, f"nitrogen_admixture_{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- figure: metastables (top), line ratios (bottom) ---------------------------
    np.seterr(divide="ignore", invalid="ignore")
    curves = list(dict.fromkeys(r["curve"] for r in rows))
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(curves)))
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5), sharex=True)
    for curve, c in zip(curves, colors):
        sel = [r for r in rows if r["curve"] == curve]
        p = np.array([r["N2_percent"] for r in sel])
        n5, n3 = (np.array([r[k] for r in sel]) for k in ("n_1s5", "n_1s3"))
        ax[0, 0].semilogy(p, n5, "o-", ms=3, color=c, label=curve)
        ax[0, 1].semilogy(p, n3, "o-", ms=3, color=c, label=curve)
        ax[0, 2].plot(p, n5 / n5[0], "o-", ms=3, color=c, label=f"1s$_5$, {curve}")
        ax[0, 2].plot(p, n3 / n3[0], "s--", ms=3, color=c, label=f"1s$_3$, {curve}")
        for k, (a, b) in enumerate(RATIOS):
            ax[1, k].plot(p, [r[f"{a:.1f}/{b:.1f}"] for r in sel], "o-", ms=3, color=c, label=curve)
    ax[0, 0].set_title("1s$_5$ (4s1) metastable")
    ax[0, 1].set_title("1s$_3$ (4s3) metastable")
    ax[0, 2].set_title("Metastable density relative to pure Ar")
    ax[0, 0].set_ylabel("Density [m$^{-3}$]")
    ax[0, 2].set_ylabel("$n$ / $n$(0 % N$_2$)")
    ax[1, 0].set_ylabel("Line ratio")
    for k, (a, b) in enumerate(RATIOS):
        ax[1, k].set_title(f"{a:.1f} / {b:.1f} nm")
    for a in ax.flat:
        a.grid(True, alpha=0.3, which="both")
    for a in ax[1]:
        a.set_xlabel("N$_2$ admixture [%]")
    ax[0, 0].legend(fontsize=8)
    ax[0, 2].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Ar/N$_2$ at {PRESSURE:g} Torr, $N_e$ = {NE:.0e} m$^{{-3}}$, {EEDF_MODE} EEDF, "
                 f"N$_2$ quenching: {QUENCH_CHOICE}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"nitrogen_admixture_{tag}.png"), dpi=200)
    print(f"figure and csv in {OUTDIR}")
    plt.show()


if __name__ == "__main__":
    main()
