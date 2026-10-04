# -*- coding: utf-8 -*-
"""
EEDFComparison.py

EEDFs from MultiBolt and BOLSIG+ - with or without electron-electron collisions,
superelastic collisions from the four Ar(4s) levels and a microwave field - and the
ground-state excitation rate coefficients of a few actinometry upper levels they
give, compared with a Maxwellian.

Each entry of CASES is one E/N sweep:
  dict(solver="multibolt", terms=6)     MultiBolt, 6 Legendre terms (DC)
  dict(solver="bolsig")                 BOLSIG+ (two-term), DC field
  dict(solver="bolsig", ee=True)        + e-e collisions at ionization degree NE/N
  dict(solver="bolsig", se=True)        + superelastic collisions with Ar(4s); the 4s
                                          populations come from the CR model at NE and
                                          are iterated to self-consistency (slow)
  dict(solver="bolsig", mw=True)        microwave field at MW_FREQ_MHZ instead of DC
The BOLSIG+ flags combine, e.g. dict(solver="bolsig", ee=True, se=True, mw=True).
Optional keys of a case: mix, EN, Ne (override MIXTURE, EN_DC / EN_MW, NE), label,
color, ls.

Runs land in InputData/MultiBolt/EEDFComparison/<name> and
InputData/Bolsig/EEDFComparison/<name>; the name spells out the mixture, solver,
physics, Ne, P, Tg and E/N range, and existing runs are reused (delete a folder to
rerun it). With N2 in the mixture the superelastic 4s populations come from the CR
model with N2 quenching at that N2 fraction (CRFitNeTe N2_percent).

Figures in Experimental_Data/Output/EEDFComparison/:
  <RUN_TAG>_rates.png       rate coefficient vs Te_eff = 2/3 <E> (or vs E/N)
  <RUN_TAG>_eepf.png        EEPF of every case at the EEDF_AT values, with the
                            Maxwellian of the same mean energy; 2nd row EEPF / Maxwellian
  <RUN_TAG>_same_Te<SAME_TE>eV.png               every case on one EEPF plot at the same
                                                 mean energy (log-EEPF interpolation between
                                                 sweep points), with the Maxwellian
  <RUN_TAG>_same_Te<SAME_TE>eV_over_maxwell.png  the same EEPFs / Maxwellian
  <RUN_TAG>_eepf_sweep.png  all EEPFs of each case, coloured by E/N
Run from Spyder: edit the settings, press F5.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "Experimental_Data", "ExperimentalDataAnalysis"))
import ActinometryRates as ar                       # noqa: E402
import CRFitNeTe as crf                             # noqa: E402

he = crf._helpers()

# ---- settings -------------------------------------------------------------------
XSEC_DIR = r"C:\Users\dptro\Documents\Work\Python\Multibolt\MultiBolt-master\MultiBolt-master\cross-sections"
MIXTURE = {"Ar": 0.95, "N2": 0.05}                 # mole fractions (Biagi_<species>.txt); 0 = left out
P_TORR, TG = 1.0, 300.0                      # Torr, K
EN_DC = [1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 75, 100, 150, 200]   # Td, Te_eff ~1.9-5.1 eV in Ar
EN_MW = [6, 8, 10, 15, 20, 30, 40, 60, 80, 100, 130, 170, 250, 330, 400, 500, 600, 700, 850, 1000]   # Td,
                                             # Te_eff ~0.7-4.7 eV in Ar: a microwave field heats
                                             # less than DC, so it needs a much higher E/N
MW_FREQ_MHZ = 2420                           # microwave frequency (BOLSIG+ omega/N = 2 pi f / N)
NE = 1e17                                    # m^-3, electron density of the e-e / superelastic cases
CASES = [
    dict(solver="multibolt", terms=2),
    dict(solver="multibolt", terms=4),
    dict(solver="multibolt", terms=6),
    dict(solver="bolsig"),
    #dict(solver="bolsig", ee=True),
    #dict(solver="bolsig", se=True),
    #dict(solver="bolsig", ee=True, se=True),
    dict(solver="bolsig", mw=True),
    #dict(solver="bolsig", mw=True, ee=True, se=True),
]
LEVELS = [("Ar I", "5p5", "Ar I 415.9 nm (5p5)"),
          ("Ar I", "4p10", "Ar I 750.4 nm (4p10)"),
          ("N I", "other21", "N I 744.2 nm (3p 4S)")]
RATE_X = "Te_eff"                            # x axis of the rate plot: "Te_eff" or "EN"
PLOT_MAXWELL = True                          # Maxwellian rate curve (Te_eff axis only)
TE_MAXWELL = np.linspace(0.5, 8, 60)         # eV
EEDF_AXIS = "Te_eff"                         # EEPF panels at these Te_eff [eV] ("Te_eff") or E/N [Td] ("EN")
EEDF_AT = [2, 3, 4]                        #   each case shows its sweep point nearest to the value
PLOT_EEPF_RATIO = True                       # 2nd row: EEPF / Maxwellian of the same mean energy
PLOT_EEPF_SWEEP = True                       # one panel per case with every E/N of its sweep
SAME_TE = 3.0                                # eV, one figure with every case at exactly this mean energy
                                             #   (Te_eff = 2/3 <E>); None = skip
E_MAX_PLOT = 25                              # eV, x range of the EEPF plots
RUN_TAG = "comparison"                       # prefix of the figure files
BOLSIG_SETTINGS = dict(n_grid=500, precision=1e-25, max_iter=20000)   # as CreateEEDFLibraries
N_WORKERS = 12                               # parallel BOLSIG+ calls (e-e / superelastic cases)
MB_DIR = he.MULTIBOLT_FOLDER / "EEDFComparison"
BS_DIR = he.BOLSIG_FOLDER / "EEDFComparison"
OUTDIR = os.path.join(crf.ROOT_DIR, "Experimental_Data", "Output", "EEDFComparison")

THRESHOLDS = [(11.55, "4s"), (13.08, "4p"), (14.46, "5p"), (15.76, "ion.")]
SUPERELASTIC = {k: lbl for k, (lbl, _) in he.AR_PASCHEN_4S.items()}    # Ar(1S5) -> 4s1, ...


# ---- cases ------------------------------------------------------------------------
def species_of(mix):
    sp = [s for s, v in mix.items() if v > 0]
    return sp, [float(mix[s]) for s in sp]


def mix_tag(mix):
    """'Ar', or e.g. 'ArN2_5pct' (fractions of all species after the first)."""
    sp, fr = species_of(mix)
    return sp[0] if len(sp) == 1 else "".join(sp) + "_" + "_".join(f"{100 * f:.3g}pct" for f in fr[1:])


def mix_title(mix):
    return " + ".join(f"{100 * f:.3g} % {s}" for s, f in zip(*species_of(mix)))


def expand(case, k):
    """Case dict with every setting filled in, its run folder name and its label."""
    c = {"solver": "bolsig", "terms": 6, "ee": False, "se": False, "mw": False,
         "mix": MIXTURE, "Ne": NE, **case}
    if c["solver"] not in ("multibolt", "bolsig"):
        raise ValueError(f"case {k}: solver must be 'multibolt' or 'bolsig', not {c['solver']!r}")
    if c["solver"] == "multibolt" and (c["ee"] or c["se"] or c["mw"]):
        raise ValueError(f"case {k}: e-e, superelastic and microwave need solver='bolsig'")
    c["EN"] = [float(x) for x in case.get("EN", EN_MW if c["mw"] else EN_DC)]
    if c["solver"] == "multibolt":
        what = f"multibolt_{c['terms']}terms"
        label = f"MultiBolt, {c['terms']} terms"
    else:
        what = "bolsig" + "_ee" * c["ee"] + "_se" * c["se"] + (f"_mw{MW_FREQ_MHZ:g}MHz" if c["mw"] else "")
        label = ", ".join(["BOLSIG+", f"{MW_FREQ_MHZ:g} MHz" if c["mw"] else "DC"]
                          + ["e-e"] * c["ee"] + ["superelastic"] * c["se"])
    ne = f"_Ne{c['Ne']:.0e}".replace("+", "") if c["ee"] or c["se"] else ""
    c["name"] = (f"{mix_tag(c['mix'])}_{what}{ne}_{P_TORR:g}Torr_{TG:g}K"
                 f"_EN{c['EN'][0]:g}-{c['EN'][-1]:g}x{len(c['EN'])}")
    if c["mix"] != MIXTURE:
        label += f" ({mix_title(c['mix'])})"
    c.setdefault("label", label)
    c.setdefault("color", f"C{k % 10}")
    c.setdefault("ls", "--" if c["solver"] == "multibolt" else "-")
    return c


def bolsig_xsec(species, superelastic):
    """BOLSIG+ cross-section file of a species set (MultiBolt's Biagi files joined), or
    its version with the Ar(4s) excitations written '<->' for superelastics."""
    tag = "".join(species)
    plain = he.BOLSIG_XSEC_FOLDER / f"Biagi_{tag}.txt"
    if not plain.is_file():
        parts = [open(os.path.join(XSEC_DIR, f"Biagi_{s}.txt"), errors="replace").read().rstrip()
                 for s in species]
        plain.write_text("\n\n".join(parts) + "\n")
    if not superelastic:
        return plain
    se = he.BOLSIG_XSEC_FOLDER / f"Biagi_{tag}_superelastic.txt"
    if not se.is_file():
        he.MakeSuperelasticXsecFile(plain, se, {k: g for k, (_, g) in he.AR_PASCHEN_4S.items()})
    return se


_cr_solvers = {}


def cr_solver(mix):
    """CR-model 4s populations for the superelastic cases (one per N2 fraction)."""
    n2 = 100 * mix.get("N2", 0) / sum(mix.values())
    if n2 not in _cr_solvers:
        _cr_solvers[n2] = crf.cr_density_solver(dict(crf.CONFIG, P_Torr=P_TORR, Tg=TG, N2_percent=n2))
    return _cr_solvers[n2]


def run(c):
    """EEDFs of a case along its E/N sweep; runs the solver if the folder is missing."""
    sp, fr = species_of(c["mix"])
    if c["solver"] == "multibolt":
        folder = MB_DIR / c["name"]
        if not (folder / "EEDFs_f0").is_dir():
            print(f"\n== MultiBolt: {c['name']}")
            he.RunMultiBolt([os.path.join(XSEC_DIR, f"Biagi_{s}.txt") for s in sp], c["name"], c["EN"],
                            species=dict(zip(sp, fr)), P_Torr=P_TORR, T_K=TG, N_terms=c["terms"],
                            remap_span=12, overwrite=True, ExportFolder=MB_DIR, verbose=False)
        return he.ImportMultiBoltEEDFs(folder, verbose=False)

    folder = BS_DIR / c["name"]
    if not he.IsBolsigLibrary(folder):
        print(f"\n== BOLSIG+: {c['name']}")
        xsec = bolsig_xsec(sp, c["se"])
        extra = dict(omega_N=2 * np.pi * MW_FREQ_MHZ * 1e6 / he.Torr2Volume(P_TORR, TG)) if c["mw"] else {}
        if c["ee"] or c["se"]:
            he.BuildBolsigLibrary(xsec, c["name"], c["EN"], [c["Ne"]], P_Torr=P_TORR, Tg=TG,
                                  species=sp, fractions=fr, electron_electron=c["ee"],
                                  superelastic=SUPERELASTIC if c["se"] else None,
                                  cr_solve=cr_solver(c["mix"]) if c["se"] else None, n_iter=8, tol=0.03,
                                  n_workers=N_WORKERS, overwrite=True, ExportFolder=BS_DIR,
                                  **BOLSIG_SETTINGS, **extra)
        else:
            he.RunBolsig(xsec, c["name"], c["EN"], species=sp, fractions=fr, Tg=TG, overwrite=True,
                         ExportFolder=BS_DIR, verbose=False, **BOLSIG_SETTINGS, **extra)
    lib = he.ImportBolsigLibrary(folder, verbose=False)
    return lib["EEDFs"] if lib["kind"] == "1D" else [row[0] for row in lib["EEDFs"]]


# ---- figures ------------------------------------------------------------------------
def thresholds(ax):
    for e, name in THRESHOLDS:
        ax.axvline(e, color="0.75", lw=0.8, zorder=0)
        ax.text(e, 1.0, f" {name}", transform=ax.get_xaxis_transform(), va="top", ha="left",
                fontsize=8, color="0.45")


def axis_values(eedfs, axis):
    return np.array([e["Te_eff"] if axis == "Te_eff" else e["sweep"]["value"] for e in eedfs])


def plot_rates(data, xs):
    fig, axs = plt.subplots(1, len(LEVELS), figsize=(5.5 * len(LEVELS), 4.8), squeeze=False)
    for ax, (sp, lbl, title) in zip(axs[0], LEVELS):
        if PLOT_MAXWELL and RATE_X == "Te_eff":
            ax.semilogy(TE_MAXWELL, ar.rate_coefficient(xs[sp][lbl], Te=TE_MAXWELL), "k:", lw=2,
                        label="Maxwellian")
        for c, eedfs in data:
            k = np.array([ar.rate_coefficient(xs[sp][lbl], eedf=(e["E"], e["EEDF"])) for e in eedfs])
            ax.semilogy(axis_values(eedfs, RATE_X), np.where(k > 0, k, np.nan), marker="o", ms=3,
                        color=c["color"], ls=c["ls"], label=c["label"])
        ax.set_title(title)
        if RATE_X == "EN":
            ax.set_xscale("log")
        ax.set_xlabel(r"$T_{e,\mathrm{eff}} = \frac{2}{3}\langle\varepsilon\rangle$ (eV)"
                      if RATE_X == "Te_eff" else "E/N (Td)")
        ax.set_ylim(1e-24, None)
        ax.grid(alpha=0.3, which="both")
    axs[0, 0].set_ylabel(r"ground-state excitation rate $k$ (m$^3$ s$^{-1}$)")
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(f"Excitation rate coefficients, {mix_title(MIXTURE)}, {P_TORR:g} Torr, {TG:g} K")
    fig.tight_layout()
    return fig


def plot_eepf(data):
    rows = 2 if PLOT_EEPF_RATIO else 1
    fig, axs = plt.subplots(rows, len(EEDF_AT), figsize=(5 * len(EEDF_AT), 4.4 * rows), squeeze=False)
    unit = "eV" if EEDF_AXIS == "Te_eff" else "Td"
    print(f"\nEEPF panels at {EEDF_AXIS} = {EEDF_AT} {unit}")
    for col, target in enumerate(EEDF_AT):
        for c, eedfs in data:
            x = axis_values(eedfs, EEDF_AXIS)
            e = eedfs[int(np.argmin(np.abs(x - target)))]
            got = e["Te_eff"] if EEDF_AXIS == "Te_eff" else e["sweep"]["value"]
            if abs(got - target) > 0.1 * target:
                print(f"  {c['label']}: nearest {EEDF_AXIS} to {target:g} is {got:.3g} {unit}")
            ok = e["EEPF"] > 0
            axs[0, col].semilogy(e["E"][ok], e["EEPF"][ok], color=c["color"], ls=c["ls"], lw=1.4,
                                 label=f"{c['label']} ({e['sweep']['value']:g} Td, {e['Te_eff']:.2f} eV)")
            if PLOT_EEPF_RATIO:
                mx = he.MaxwellianEEDF(e["Te_eff"], E=e["E"])["EEPF"]
                axs[1, col].semilogy(e["E"][ok], e["EEPF"][ok] / mx[ok], color=c["color"], ls=c["ls"], lw=1.4)
        if EEDF_AXIS == "Te_eff":
            mx = he.MaxwellianEEDF(target, E=np.linspace(0, E_MAX_PLOT, 501))
            axs[0, col].semilogy(mx["E"], mx["EEPF"], ":", color="m", lw=1.6, label=f"Maxwellian, {target:g} eV")
        axs[0, col].set_title((r"$T_{e,\mathrm{eff}}$" if EEDF_AXIS == "Te_eff" else "E/N")
                              + f" = {target:g} {unit}", fontsize=11)
        axs[0, col].set_ylim(1e-22, 1)
        axs[0, col].legend(fontsize=6.5, loc="lower left")
        if PLOT_EEPF_RATIO:
            axs[1, col].axhline(1, color="m", ls=":", lw=1.6)
            axs[1, col].set_ylim(1e-8, 1e4)
        for ax in axs[:, col]:
            ax.set_xlim(0, E_MAX_PLOT)
            ax.set_xlabel("Electron energy (eV)")
            ax.grid(alpha=0.25)
            thresholds(ax)
    axs[0, 0].set_ylabel(r"EEPF  $f_0$ (eV$^{-3/2}$)")
    if PLOT_EEPF_RATIO:
        axs[1, 0].set_ylabel("EEPF / Maxwellian of the same mean energy")
    fig.suptitle(f"EEPF, {mix_title(MIXTURE)}, {P_TORR:g} Torr, {TG:g} K"
                 + (f"  (e-e / superelastic at Ne = {NE:.0e} m$^{{-3}}$)" if any(c["ee"] or c["se"] for c, _ in data)
                    else ""), fontsize=12)
    fig.tight_layout()
    return fig


def at_mean_energy(eedfs, te):
    """(E, EEPF, E/N) of a sweep at Te_eff = te exactly: log(EEPF) interpolated between the
    two neighbouring sweep points around te, with the weight that gives the mean energy
    3/2 te. The highest-E/N crossing is used (a microwave Te_eff is not monotonic at low
    E/N). None if te is outside the sweep."""
    t = axis_values(eedfs, "Te_eff")
    cross = [i for i in range(len(t) - 1) if (t[i] - te) * (t[i + 1] - te) <= 0]
    if not cross:
        return None
    a, b = eedfs[cross[-1]], eedfs[cross[-1] + 1]
    E = np.arange(0, max(a["E"][-1], b["E"][-1]) + 1e-9, 0.02)
    fa, fb = (np.interp(E, e["E"], e["EEPF"], right=0.0) for e in (a, b))
    ok = (fa > 0) & (fb > 0)
    la, lb = np.log(np.where(ok, fa, 1)), np.log(np.where(ok, fb, 1))

    def eepf(w):
        f = np.where(ok, np.exp((1 - w) * la + w * lb), 0.0)
        return f / np.trapezoid(np.sqrt(E) * f, E)

    def te_of(w):
        return 2 / 3 * np.trapezoid(E ** 1.5 * eepf(w), E)
    lo, hi = 0.0, 1.0
    for _ in range(50):                    # bisection on the weight
        mid = 0.5 * (lo + hi)
        if (te_of(mid) - te) * (te_of(lo) - te) <= 0:
            hi = mid
        else:
            lo = mid
    w = 0.5 * (lo + hi)
    EN = a["sweep"]["value"] ** (1 - w) * b["sweep"]["value"] ** w
    return E, eepf(w), EN


def plot_same_te(data, xs):
    """All cases at Te_eff = SAME_TE with the Maxwellian: a figure of the EEPFs and a figure
    of EEPF / Maxwellian. Prints the rate coefficients."""
    figs = [plt.subplots(figsize=(9, 6)) for _ in range(2)]
    axs = [ax for _, ax in figs]
    mx = he.MaxwellianEEDF(SAME_TE, E=np.linspace(0, E_MAX_PLOT, 1001))
    axs[0].semilogy(mx["E"], mx["EEPF"], ":", color="m", lw=2, label=f"Maxwellian, {SAME_TE:g} eV")
    axs[1].axhline(1, color="m", ls=":", lw=2, label="Maxwellian")
    print(f"\nAt Te_eff = {SAME_TE:g} eV: rate coefficient (m^3/s) and / Maxwellian")
    print(f"{'case':45s} {'E/N (Td)':>9s}" + "".join(f"{t.split(' (')[0]:>24s}" for *_, t in LEVELS))
    k_mx = [float(ar.rate_coefficient(xs[sp][lbl], Te=np.array([SAME_TE]))[0]) for sp, lbl, _ in LEVELS]
    print(f"{'Maxwellian':45s} {'':>9s}" + "".join(f"{k:24.3e}" for k in k_mx))
    for c, eedfs in data:
        got = at_mean_energy(eedfs, SAME_TE)
        if got is None:
            t = axis_values(eedfs, "Te_eff")
            print(f"{c['label']:45s} Te_eff {t.min():.2f}-{t.max():.2f} eV does not reach {SAME_TE:g} eV - skipped")
            continue
        E, f, EN = got
        ok = f > 0
        axs[0].semilogy(E[ok], f[ok], color=c["color"], ls=c["ls"], lw=1.6, label=f"{c['label']} ({EN:.3g} Td)")
        m = he.MaxwellianEEDF(SAME_TE, E=E[ok])["EEPF"]
        axs[1].semilogy(E[ok], f[ok] / m, color=c["color"], ls=c["ls"], lw=1.6, label=c["label"])
        k = [ar.rate_coefficient(xs[sp][lbl], eedf=(E, np.sqrt(E) * f)) for sp, lbl, _ in LEVELS]
        print(f"{c['label']:45s} {EN:9.3g}" + "".join(f"{kk:13.3e} ({kk / km:7.2g})" for kk, km in zip(k, k_mx)))
    axs[0].set_ylim(1e-22, 1)
    axs[0].set_ylabel(r"EEPF  $f_0$ (eV$^{-3/2}$)")
    axs[1].set_ylim(1e-8, 1e4)
    axs[1].set_ylabel("EEPF / Maxwellian of the same mean energy")
    conditions = (f"\n{mix_title(MIXTURE)}, {P_TORR:g} Torr, {TG:g} K"
                  + (f", e-e / superelastic at Ne = {NE:.0e} m$^{{-3}}$"
                     if any(c["ee"] or c["se"] for c, _ in data) else ""))
    for ax, what in zip(axs, ("EEPF", "EEPF relative to a Maxwellian")):
        ax.set_xlim(0, E_MAX_PLOT)
        ax.set_xlabel("Electron energy (eV)")
        ax.grid(alpha=0.25)
        thresholds(ax)
        ax.legend(fontsize=8, loc="lower left")
        ax.set_title(rf"{what}, all methods at $T_{{e,\mathrm{{eff}}}}$ = {SAME_TE:g} eV" + conditions, fontsize=11)
    for fig, _ in figs:
        fig.tight_layout()
    return [fig for fig, _ in figs]


def plot_eepf_sweep(data):
    ncol = min(3, len(data))
    nrow = int(np.ceil(len(data) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.2 * nrow), squeeze=False, sharey=True,
                            layout="constrained")
    EN_all = np.concatenate([axis_values(eedfs, "EN") for _, eedfs in data])
    norm = LogNorm(EN_all.min(), EN_all.max())
    for ax, (c, eedfs) in zip(axs.flat, data):
        for e in eedfs:
            ok = e["EEPF"] > 0
            ax.semilogy(e["E"][ok], e["EEPF"][ok], color=plt.cm.viridis(norm(e["sweep"]["value"])), lw=1.1)
        ax.set_title(c["label"], fontsize=10)
        ax.set_xlim(0, E_MAX_PLOT)
        ax.set_ylim(1e-22, 1)
        ax.set_xlabel("Electron energy (eV)")
        ax.grid(alpha=0.25)
        thresholds(ax)
    for ax in axs.flat[len(data):]:
        ax.axis("off")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"EEPF  $f_0$ (eV$^{-3/2}$)")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axs, label="E/N (Td)", shrink=0.8)
    fig.suptitle(f"EEPF along each E/N sweep, {mix_title(MIXTURE)}, {P_TORR:g} Torr, {TG:g} K", fontsize=12)
    return fig


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    np.seterr(divide="ignore", invalid="ignore")      # EEPF / Maxwellian far in the tail
    cases = [expand(case, k) for k, case in enumerate(CASES)]
    data = [(c, run(c)) for c in cases]
    print(f"\n{'case':45s} {'E/N (Td)':>14s} {'Te_eff (eV)':>14s}  folder")
    for c, eedfs in data:
        EN, te = axis_values(eedfs, "EN"), axis_values(eedfs, "Te_eff")
        print(f"{c['label']:45s} {f'{EN.min():g}-{EN.max():g}':>14s} {f'{te.min():.2f}-{te.max():.2f}':>14s}  {c['name']}")

    xs = ar.load_all_cross_sections()
    figs = {"rates": plot_rates(data, xs), "eepf": plot_eepf(data)}
    if SAME_TE:
        figs[f"same_Te{SAME_TE:g}eV"], figs[f"same_Te{SAME_TE:g}eV_over_maxwell"] = plot_same_te(data, xs)
    if PLOT_EEPF_SWEEP:
        figs["eepf_sweep"] = plot_eepf_sweep(data)
    for key, fig in figs.items():
        fig.savefig(os.path.join(OUTDIR, f"{RUN_TAG}_{key}.png"), dpi=150)
    print(f"\nfigures in {OUTDIR}")
    plt.show()


if __name__ == "__main__":
    main()
