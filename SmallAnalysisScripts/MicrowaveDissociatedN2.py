# -*- coding: utf-8 -*-
"""
MicrowaveDissociatedN2.py

Does a mostly dissociated N2 admixture let the microwave (2.45 GHz) EEDF fit the Ar/N2 spectra?

With the N2 feed fraction f and a fraction d of it dissociated (N2 -> 2 N at constant pressure)
the gas is Ar : N2 : N = (1 - f) : f (1 - d) : 2 f d (CRFitNeTe.gas_fractions).  Removing N2
removes the 2-4 eV vibrational sink, but atomic N has its own low-lying losses: N(2D) at
2.39 eV (BSR peak ~9e-21 m^2 at 4-5 eV) and N(2P) at 3.57 eV, at twice the particle density.
BOLSIG+ decides which wins; N(2D, 2P) -> N(4S) superelastics hand energy back when the
metastables are populated.

Cross sections: Bolsig/Cross-Sections/Biagi_ArN2_BSR_N.txt = Biagi Ar + N2 and the BSR atomic-N
set of InputData/AtomicNitrogeBolsigCrossSections.txt (momentum-transfer elastic only - the file
also holds the integral elastic, which BOLSIG+ would add on top -, 26 excitations, ionization),
with N -> N(2D), N(2P) written reversible (g ratios 10/4, 6/4).  The BSR N(2D), N(2P) tables
(~1200-1300 points) are thinned to <= MAX_POINTS (every k-th point, ends kept).

VARIANTS  name: (d, N(2D)/N, N(2P)/N)
  diss95   95 % dissociated, metastables empty
  diss100  fully dissociated (Ar + N only)
  diss95m  95 % dissociated with N(2D)/N = 5 %, N(2P)/N = 1 % (assumed, for the superelastics)

1. BOLSIG+ microwave E/N sweep per N2 fraction and variant (E/N grid and precision as
   MicrowaveLowENFit) -> InputData/Bolsig/ArN2_<f>pct_bolsig_mw_<variant>
   eepf_compare.png  EEPF at matched Te_eff, Te_eff and Ar 4p / 5p excitation rate coefficients
                     along E/N: pure-Ar microwave, undissociated Ar/N2 microwave, the variants
2. CR table around n_c (MicrowaveLowENFit.NE_GRID_NC) with N2 quenching by the undissociated
   rest only and the Ar ground state diluted by N2 + N (cfg N2_dissociation; quenching of Ar
   levels by N atoms left out); per condition at Ne = n_c along E/N: chi^2/dof, Te_eff, CR 1s5,
   and the best chi^2/dof with the CR 1s5 inside the absorption range.
   diss_summary.png  against DC, the undissociated microwave fit and pure-Ar microwave

Output in Experimental_Data/Output/MicrowaveDissociatedN2/.  Run from Spyder (F5) or
  python SmallAnalysisScripts/MicrowaveDissociatedN2.py            libraries, EEPFs, tables, fits
  python SmallAnalysisScripts/MicrowaveDissociatedN2.py --eepf     libraries and EEPF figure only
"""
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import MicrowaveLowENFit as mlf                     # noqa: E402

crf, anc, he = mlf.crf, mlf.anc, mlf.he

# ---- settings -------------------------------------------------------------------
FRACTIONS = [0.5, 1, 1.5, 2, 2.4, 2.5, 3]           # % N2 feed of the conditions (power sweep at 2.4 %)
VARIANTS = {"diss95": (0.95, 0.0, 0.0),             # name: (dissociated fraction, N(2D)/N, N(2P)/N)
            "diss100": (1.0, 0.0, 0.0),
            "diss95m": (0.95, 0.05, 0.01)}
N_XSEC_SRC = os.path.join(mlf.ROOT_DIR, "InputData", "AtomicNitrogeBolsigCrossSections.txt")
XSEC = he.BOLSIG_XSEC_FOLDER / "Biagi_ArN2_BSR_N.txt"
N_META = {"N(2p3_2Do)": 10 / 4, "N(2p3_2Po)": 6 / 4}   # g(excited) / g(4S)
MAX_POINTS = 900                                    # BOLSIG+ rejects longer tables ("too many data points")
BOLSIG_TIMEOUT = 1200                               # s per library; then retried at precision 1e-25
COMPARE_PCT = [1, 2.4]                              # N2 fractions of the EEPF figure
TE_MATCH = 0.95                                     # eV, Te_eff of the pure-Ar microwave fit at n_c
REF_PURE_AR_MW = mlf.lib_folder(0, "lowEN")
STYLE = {"pure Ar": ("0.4", ":"), "undissociated": ("k", "-"),
         "diss95": ("C0", "-"), "diss95m": ("C2", "--"), "diss100": ("C3", "-")}
LOWEN_SUMMARY = os.path.join(mlf.OUTDIR, "summary.csv")          # DC, pure-Ar microwave at n_c
N2A_SUMMARY = os.path.join(mlf.ROOT_DIR, "Experimental_Data", "Output", "MicrowaveN2A", "n2a_summary.csv")
OUTDIR = os.path.join(mlf.ROOT_DIR, "Experimental_Data", "Output", "MicrowaveDissociatedN2")


# ---- 1. cross sections and BOLSIG+ libraries --------------------------------------
def _blocks(lines):
    """LXCat blocks (list of lines from the keyword to the closing dashes) of a file."""
    out, i = [], 0
    while i < len(lines):
        if lines[i].strip() in ("ELASTIC", "EFFECTIVE", "EXCITATION", "IONIZATION", "ATTACHMENT"):
            j, dashes = i, 0
            while dashes < 2:
                j += 1
                dashes += lines[j].startswith("-----")
            out.append(lines[i:j + 1])
            i = j
        i += 1
    return out


def _thin(block, n_max=MAX_POINTS):
    """The block with its data table thinned to <= n_max points (every k-th, first and last kept)."""
    d = [i for i, l in enumerate(block) if l.startswith("-----")]
    rows = block[d[0] + 1:d[1]]
    if len(rows) <= n_max:
        return block
    k = int(np.ceil(len(rows) / n_max))
    keep = rows[::k] + ([rows[-1]] if (len(rows) - 1) % k else [])
    return block[:d[0] + 1] + keep + block[d[1]:]


def write_xsec():
    """Biagi Ar/N2 + BSR atomic N (see the docstring) -> XSEC (written if missing)."""
    if XSEC.is_file():
        return XSEC
    blocks = _blocks(open(N_XSEC_SRC, errors="replace").read().splitlines())
    keep = []
    for b in blocks:
        if b[0].strip() == "ELASTIC" and not any("momentum transfer" in l for l in b):
            continue                                  # integral elastic
        m = re.fullmatch(r"\s*N\s*->\s*(\S+)\s*", b[1])
        if b[0].strip() == "EXCITATION" and m and m.group(1) in N_META:
            b = [b[0], f"N <-> {m.group(1)}", f"{float(b[2].split()[0]):.6e}  {N_META[m.group(1)]:g}"] + b[3:]
        keep.append(_thin(b))
    kinds = [b[0].strip() for b in keep]
    if kinds.count("ELASTIC") != 1 or "IONIZATION" not in kinds:
        raise ValueError(f"{N_XSEC_SRC}: expected one momentum-transfer elastic and an ionization, got {kinds}")
    base = (he.BOLSIG_XSEC_FOLDER / "Biagi_ArN2.txt").read_text(errors="replace").rstrip()
    XSEC.write_text(base + "\n\n" + "\n\n".join("\n".join(b) for b in keep) + "\n")
    print(f"wrote {XSEC} ({len(keep)} atomic-N processes)")
    return XSEC


def composition(pct, variant):
    d, m2d, m2p = VARIANTS[variant]
    x_ar, x_n2, x_n = crf.gas_fractions(dict(N2_percent=pct, N2_dissociation=d))
    return (["Ar", "N2", "N", "N(2p3_2Do)", "N(2p3_2Po)"],
            [x_ar, x_n2, x_n * (1 - m2d - m2p), x_n * m2d, x_n * m2p])


def library(pct, variant):
    folder = mlf.lib_folder(pct, variant)
    if he.IsBolsigLibrary(folder):
        return folder
    species, fractions = composition(pct, variant)
    N = he.Torr2Volume(anc.CONFIG["fit"]["P_Torr"], anc.CONFIG["fit"]["Tg"])
    kw = dict(mlf.BOLSIG_SETTINGS, precision=mlf.PRECISION, omega_N=2 * np.pi * mlf.FREQ_HZ / N,
              overwrite=True, verbose=False, timeout=BOLSIG_TIMEOUT)
    try:
        he.RunBolsig(write_xsec(), folder.name, mlf.EN_MW, species=species, fractions=fractions, **kw)
    except (subprocess.TimeoutExpired, RuntimeError) as err:
        print(f"{folder.name}: {type(err).__name__} at precision 1e-30 - retrying at 1e-25", flush=True)
        he.RunBolsig(write_xsec(), folder.name, mlf.EN_MW, species=species, fractions=fractions,
                     **dict(kw, precision=1e-25))
        return folder
    bad = runaway_rows(folder)
    if bad:     # 1e-30 can give runaway tails to 9990 eV (here with the N(2D, 2P) superelastics)
        print(f"{folder.name}: runaway tails at {[mlf.EN_MW[i] for i in bad]} Td - those rows at 1e-25", flush=True)
        prec = [1e-25 if i in bad else p for i, p in enumerate(mlf.PRECISION)]
        he.RunBolsig(write_xsec(), folder.name, mlf.EN_MW, species=species, fractions=fractions,
                     **dict(kw, precision=prec))
    return folder


def runaway_rows(folder, e_max=1000.0):
    """Rows of a library whose EEDF tail runs to >= e_max eV (BOLSIG+ artifact)."""
    lib = he.ImportBolsigLibrary(folder, verbose=False)
    return [i for i, e in enumerate(lib["EEDFs"]) if e["E"][e["EEPF"] > 0].max() >= e_max]


def job_cfg(pct, variant):
    return mlf.fit_cfg(pct, mlf.lib_folder(pct, variant), os.path.join(OUTDIR, f"fit_{variant}_{pct:g}pct_N2"),
                       Ne_grid=mlf.NE_GRID_NC, N2_dissociation=VARIANTS[variant][0])


def _run_workers(flag, keys, what):
    """Run this script with flag for each key, mlf.N_WORKERS processes at a time."""
    if not keys:
        return
    logs = os.path.join(OUTDIR, "logs")
    os.makedirs(logs, exist_ok=True)
    print(f"{what}: {len(keys)} jobs ({mlf.N_WORKERS} at a time, logs in {logs}) ...", flush=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0, procs = time.time(), {}
    for k in keys:
        while sum(p.poll() is None for p in procs.values()) >= mlf.N_WORKERS:
            time.sleep(5)
        log = open(os.path.join(logs, f"{flag.strip('-')}_{k.replace(':', '_')}.log"), "w")
        procs[k] = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), flag, k],
                                    stdout=log, stderr=subprocess.STDOUT, cwd=mlf.ROOT_DIR, env=env)
    failed = [k for k, p in procs.items() if p.wait() != 0]
    print(f"{what} done ({time.time() - t0:.0f} s)", flush=True)
    if failed:
        raise RuntimeError(f"{what} failed for {failed} - see {logs}")


def _split(key):
    pct, variant = key.split(":")
    return float(pct), variant


def build_libraries():
    write_xsec()
    _run_workers("--lib", [f"{p:g}:{v}" for p in FRACTIONS for v in VARIANTS
                           if not he.IsBolsigLibrary(mlf.lib_folder(p, v))], "BOLSIG+ libraries")


def build_tables():
    _run_workers("--build", [f"{p:g}:{v}" for p in FRACTIONS for v in VARIANTS
                             if not crf.table_is_current(job_cfg(p, v))], "CR tables")


# ---- EEPF comparison -------------------------------------------------------------
def _k(e, lo, hi):
    """Sum of the Ar excitation rate coefficients with threshold in [lo, hi) eV."""
    return sum(q["rate_m3s"] for q in e["rates"] if q["species"] == "Ar" and q["process"] == "Excitation"
               and lo <= (q["threshold_eV"] or 0) < hi)


def family_libraries(pct):
    out = {"pure Ar": REF_PURE_AR_MW, "undissociated": mlf.lib_folder(pct, "lowEN")}
    out.update({v: mlf.lib_folder(pct, v) for v in VARIANTS})
    return {k: he.ImportBolsigLibrary(f, verbose=False) for k, f in out.items() if he.IsBolsigLibrary(f)}


def tail_table(libs, pct):
    rows = []
    for fam, lib in libs.items():
        for en, e in zip(lib["EN_Td"], lib["EEDFs"]):
            E, f = e["E"], e["EEDF"]
            rows.append(dict(N2_percent=pct, family=fam, EN=en, Te_eff=e["Te_eff"], k_4s=_k(e, 11.5, 11.9),
                             k_4p=_k(e, 12.8, 13.6), k_5p=_k(e, 14.4, 14.95),
                             frac_above_11p5=np.trapezoid(f[E >= 11.55], E[E >= 11.55]) / np.trapezoid(f, E)))
    return pd.DataFrame(rows)


def plot_eepfs(path):
    fig, axs = plt.subplots(len(COMPARE_PCT), 4, figsize=(22, 5 * len(COMPARE_PCT)), squeeze=False)
    tabs = []
    for r, pct in enumerate(COMPARE_PCT):
        libs = family_libraries(pct)
        t = tail_table(libs, pct)
        tabs.append(t)
        for fam, lib in libs.items():
            c, ls = STYLE[fam]
            lab = fam if fam in ("pure Ar", "undissociated") else f"{fam} ({VARIANTS[fam][0]:.0%} diss."
            lab += "" if fam in ("pure Ar", "undissociated") else (", N(2D) 5 %, N(2P) 1 %)" if VARIANTS[fam][1] else ")")
            g = t[t.family == fam]
            i = int(np.argmin(np.abs(g.Te_eff.to_numpy() - TE_MATCH)))
            e = lib["EEDFs"][i]
            axs[r, 0].semilogy(e["E"], np.where(e["EEPF"] > 0, e["EEPF"], np.nan), color=c, ls=ls, lw=1.4,
                               label=f"{lab}: {g.EN.iloc[i]:g} Td, Te_eff {g.Te_eff.iloc[i]:.2f} eV")
            axs[r, 1].loglog(g.EN, g.Te_eff, color=c, ls=ls, lw=1.4, label=lab)
            axs[r, 2].loglog(g.EN, g.k_4p.where(g.k_4p > 0), color=c, ls=ls, lw=1.4, label=lab)
            axs[r, 3].loglog(g.k_4p.where(g.k_4p > 0), (g.k_5p / g.k_4p).where(g.k_5p > 0), color=c, ls=ls, lw=1.4,
                             label=lab)
        axs[r, 0].axvline(11.55, color="0.6", lw=0.8)
        axs[r, 0].set_xlim(0, 25)
        axs[r, 0].set_ylim(1e-14, 3)
        axs[r, 0].set_xlabel("electron energy [eV]")
        axs[r, 0].set_ylabel("EEPF [eV$^{-3/2}$]")
        axs[r, 0].set_title(f"{pct:g} % N$_2$ feed: EEPF at Te_eff ~ {TE_MATCH} eV (pure-Ar fit at n$_c$)", fontsize=10)
        axs[r, 1].set_ylabel(r"$T_{e,\mathrm{eff}}$ [eV]")
        axs[r, 2].set_ylabel("k(Ar -> 4p) [m$^3$/s]")
        axs[r, 3].set_ylabel("k(Ar -> 5p) / k(Ar -> 4p)  (tail hardness)")
        axs[r, 3].set_xlabel("k(Ar -> 4p) [m$^3$/s]")
        for ax in axs[r, 1:3]:
            ax.set_xlabel("E/N [Td]")
        for ax in axs[r]:
            ax.grid(alpha=0.3, which="both")
        axs[r, 0].legend(fontsize=7)
        axs[r, 2].legend(fontsize=7)
    fig.suptitle("Microwave (2.45 GHz) BOLSIG+ EEDFs with the N$_2$ feed dissociated to N atoms "
                 "(Biagi Ar/N$_2$ + BSR N)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return pd.concat(tabs, ignore_index=True)


# ---- 2. fits at n_c ----------------------------------------------------------------
def fit_all():
    rows, ncs = [], []
    lo, hi = mlf.MEASURED_1S5
    for pct in FRACTIONS:
        for variant in VARIANTS:
            cfg = job_cfg(pct, variant)
            fit = mlf.prepare(cfg, with_nu=False)
            for sw, x in mlf.conditions_of(fit["ft"], pct):
                d = fit["ft"][(fit["ft"].sweep == sw) & (fit["ft"].x == x)]
                s, post, _ = crf.fit_block(d, fit["M"], fit["feats"], fit["grid"], fit["Te_f"], cfg)
                nc = mlf.nc_row(fit, post, s)
                ib = int(nc.chi2_red.idxmin())
                row = dict(variant=variant, sweep=sw, x=x, N2_percent=pct, chi2_nc=nc.chi2_red[ib], EN_nc=nc.EN[ib],
                           Te_nc=nc.Te_eff[ib], n_1s5_nc=nc.n_1s5[ib])
                ok = (nc.n_1s5 >= lo) & (nc.n_1s5 <= hi)
                if ok.any():
                    k = int(nc.chi2_red.where(ok).idxmin())
                    row.update(chi2_1s5=nc.chi2_red[k], EN_1s5=nc.EN[k], Te_1s5=nc.Te_eff[k], n_1s5_1s5=nc.n_1s5[k])
                rows.append(row)
                ncs.append(nc.assign(variant=variant, sweep=sw, x=x))
            print(f"  {pct:g} % N2, {variant}: done", flush=True)
    return pd.DataFrame(rows), pd.concat(ncs, ignore_index=True)


def plot_summary(res, path):
    ref = pd.read_csv(LOWEN_SUMMARY)
    base = pd.read_csv(N2A_SUMMARY) if os.path.exists(N2A_SUMMARY) else None
    sweeps = crf.CONFIG["sweeps"]
    keys = [("chi2_nc", r"best $\chi^2$/dof at $n_c$", "linear"),
            ("chi2_1s5", r"best $\chi^2$/dof at $n_c$ with CR 1s$_5$ in 1-8e17", "linear"),
            ("EN_1s5", "E/N there [Td]", "log"),
            ("Te_1s5", r"$T_{e,\mathrm{eff}}$ there [eV]", "linear")]
    ref_keys = {"chi2_nc": "chi2_red_nc", "chi2_1s5": "chi2_red_nc", "EN_1s5": "EN_nc", "Te_1s5": "Te_nc"}
    fig, axs = plt.subplots(len(keys), len(sweeps), figsize=(7 * len(sweeps), 3.4 * len(keys)), squeeze=False)
    ar = ref[(ref.family == "mw_lowEN") & (ref.N2_percent == 0)]
    for j, sw in enumerate(sweeps):
        r = res[res.sweep == sw["name"]]
        dc = ref[(ref.family == "dc") & (ref.sweep == sw["name"]) & (ref.N2_percent > 0)].sort_values("x")
        b = base[(base.fA == 0) & (base.sweep == sw["name"])].sort_values("x") if base is not None else None
        for i, (k, ylab, scale) in enumerate(keys):
            ax = axs[i, j]
            for v in VARIANTS:
                g = r[r.variant == v].sort_values("x")
                c, ls = STYLE[v]
                ax.plot(g.x, g[k], "o", ls=ls, color=c, lw=1.4, ms=5, label=v)
            if b is not None and k in b:
                ax.plot(b.x, b[k], "^-", color="k", lw=1.2, ms=5, label="microwave, undissociated N$_2$")
            ax.plot(dc.x, dc[ref_keys[k]], "s-.", color="C1", mfc="white", ms=5, lw=1.2, label="DC EEDF (undissociated)")
            if len(ar):
                ax.axhline(ar[ref_keys[k]].iloc[0], color="0.5", ls=":", lw=1.2, label="pure Ar, microwave at n$_c$")
            ax.set_yscale(scale)
            ax.set_ylabel(ylab, fontsize=9)
            ax.set_xlabel(sw["xlabel"])
            ax.grid(alpha=0.3, which="both")
        for i in (0, 1):
            axs[i, j].axhline(1, color="0.7", lw=0.8)
            axs[i, j].set_ylim(0, None)
        axs[0, j].set_title(f"{sw['name']} sweep ({sw['note']}), $N_e = n_c$", fontsize=10)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Microwave EEDF with the N$_2$ feed dissociated to N atoms: CR fit at the critical density",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(eepf_only=False):
    os.makedirs(OUTDIR, exist_ok=True)
    build_libraries()
    tails = plot_eepfs(os.path.join(OUTDIR, "eepf_compare.png"))
    tails.to_csv(os.path.join(OUTDIR, "eepf_tails.csv"), index=False)
    at = (tails.assign(dTe=(tails.Te_eff - TE_MATCH).abs()).sort_values("dTe")
          .groupby(["N2_percent", "family"], sort=False).head(1).sort_values(["N2_percent", "family"]))
    print(f"\nat Te_eff ~ {TE_MATCH} eV:")
    print(at[["N2_percent", "family", "EN", "Te_eff", "frac_above_11p5", "k_4s", "k_4p", "k_5p"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    if eepf_only:
        return tails
    build_tables()
    print("fitting ...")
    res, ncs = fit_all()
    res.to_csv(os.path.join(OUTDIR, "diss_summary.csv"), index=False)
    ncs.to_csv(os.path.join(OUTDIR, "diss_nc_rows.csv"), index=False)
    plot_summary(res, os.path.join(OUTDIR, "diss_summary.png"))
    cols = ["sweep", "x", "variant", "chi2_nc", "EN_nc", "Te_nc", "n_1s5_nc", "chi2_1s5", "EN_1s5", "Te_1s5"]
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(res.sort_values(["sweep", "x", "variant"])[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nfigures in {OUTDIR}")
    return res, ncs


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--lib":
        library(*_split(sys.argv[2]))
    elif len(sys.argv) > 2 and sys.argv[1] == "--build":
        tab = crf.build_model_table(job_cfg(*_split(sys.argv[2])))
        print(f"{sys.argv[2]}: {len(tab['x_grid'])} E/N rows, {(~tab['converged']).sum()} not converged")
    else:
        OUT = main(eepf_only="--eepf" in sys.argv)
