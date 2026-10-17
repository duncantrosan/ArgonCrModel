# -*- coding: utf-8 -*-
"""
EEDFSchemes.py  -  EEDF schemes: reading, checking and using the scheme files of CreateEEDFScheme.py
=======================================================================================================================
TAKES IN   InputData/EEDFSchemes/<name>.toml (written by CreateEEDFScheme.py, or by hand) and the BOLSIG+
           cross-section files in Bolsig/Cross-Sections
GIVES OUT  load_scheme(name) -> a checked Scheme with what the pipeline needs:
             xsec_file()                 the cross-section file BOLSIG+ is given (written once, InputData/EEDFSchemes/xsec)
             composition(x)              species and mole fractions at x % admixture
             eedf_conditions(x, p, Tg)   (x, p, Tg) of the library a condition needs
             lib_folder(x, p, Tg)        that library's folder in InputData/Bolsig
             label(), summary()          text for plots and the list
           write_list() rebuilds InputData/EEDFSchemes/EEDF_schemes.md; save_scheme writes a scheme file
USED BY    LASAndEEDFLibraries (eedf_scheme, library), CRTableBuilder (table_cfg), RunDatasets (labels), Main.py
           (checks [eedf] model), and the question tool CreateEEDFScheme.py in the repo folder

The choices a scheme holds, and what each one means, are explained at the top of CreateEEDFScheme.py.
=======================================================================================================================
"""
import hashlib
import json
import math
import os
import re
import sys
import textwrap
import tomllib
from datetime import date
from fnmatch import fnmatchcase
from functools import cached_property
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # pipeline/: its modules import each other by name
import Paths                                        # noqa: E402 - the folder of Main.py

# ---------------------------------------------------------------------------------------------------------------------
#  Folders (all from the folder of Main.py)
# ---------------------------------------------------------------------------------------------------------------------
ROOT = Path(Paths.ROOT)                                    # the repo folder (ArgonCrModel)
XSEC_DIR = Path(Paths.BOLSIG_CROSS_SECTIONS)               # BOLSIG+ cross-section files to choose from
LIB_DIR = Path(Paths.BOLSIG_LIBRARIES)                     # BOLSIG+ libraries (PhysicsHelpers.BOLSIG_FOLDER)
SCHEME_DIR = Path(Paths.EEDF_SCHEMES)                      # the scheme files <name>.toml
SCHEME_XSEC_DIR = SCHEME_DIR / "xsec"                      # cross-section files written for the schemes
LIST_FILE = SCHEME_DIR / "EEDF_schemes.md"                 # the readable list of all schemes

# ---------------------------------------------------------------------------------------------------------------------
#  Defaults offered by the questions
# ---------------------------------------------------------------------------------------------------------------------
# E/N [Td] of the built-in libraries (PressureSweepAnalysis.EN_MW / EN_DC); copied here so that this script does not
# have to import the analysis modules.
EN_MICROWAVE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 35, 40, 45, 50, 55, 60,
                70, 80, 90, 100, 115, 130, 150, 170, 210, 250, 300, 350, 400, 500, 600, 700, 850, 1000]
EN_DC = [1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50, 75, 100, 150, 200]
DEFAULT_FREQUENCY_GHZ = 2.42                               # our microwave source
DEFAULT_XSEC = "Biagi_ArN2.txt"                            # offered first when it holds the gases of the scheme

# The EEDF options built into Main.py: they have no scheme file and their names cannot be used for a scheme.
BUILT_IN = {
    "pure_Ar": "Pure-Ar microwave BOLSIG+ EEDF (Biagi Ar, 2.42 GHz, 42 E/N 3-1000 Td, Tg of the condition); "
               "the N2 only acts in the CR model (quenching, dilution).  Libraries Ar_Biagi_bolsig_mw2.42GHz_<Tg>K_<p>mTorr.",
    "ArN2_vib_saturated": "Ar/N2 microwave BOLSIG+ EEDF (Biagi Ar/N2) with the N2 vibrational excitations left out "
                          "(saturated N2(v), no net e-V loss).  Libraries ArN2_<x>pct_bolsig_mw..._vibsat.",
    "ArN2_vib_Tv": "Ar/N2 microwave BOLSIG+ EEDF with a reversible N2(v) ladder in a Boltzmann distribution at "
                   "[eedf] Tv_K (or Tv per condition from the LAS 1s5 with Tv_K = \"LAS\", scanned over Tv_grid_K).  "
                   "Libraries ArN2_<x>pct_bolsig_mw..._Tv<Tv>K.",
}

KEYWORDS = ("ELASTIC", "EFFECTIVE", "EXCITATION", "IONIZATION", "ATTACHMENT")   # first line of a BOLSIG+ process
VIB_TREATMENTS = ("cold", "saturated", "boltzmann")
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,39}")  # scheme names: they end up in folder and file names
K_B_EV = 8.617333262e-5                                    # Boltzmann constant [eV/K]


# =====================================================================================================================
#  1. Reading BOLSIG+ cross-section files
# =====================================================================================================================
def read_processes(path):
    """Every process (collision block) of a BOLSIG+ / LXCat cross-section file, in file order, as dicts:
        kind      ELASTIC, EFFECTIVE, EXCITATION, IONIZATION or ATTACHMENT
        process   the line under the keyword, spaces collapsed, e.g. 'N2 -> N2( VIB 2V1)'  (what 'exclude' matches)
        target    the species before the arrow ('N2'); the whole line for elastic / effective
        product   the part after the arrow ('N2( VIB 2V1)'; '' for elastic / effective)
        energy    the number on the third line: threshold [eV] (excitation, ionization) or mass ratio (elastic)
        lines     the block as written in the file, keyword line to the closing line of dashes
        file      the file name
    A block is: keyword line, process line, more lines (energy, comments), a line of dashes, the table, a line of
    dashes.  Everything outside the blocks (the database header) is comment for BOLSIG+ and is not kept."""
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()
    out, i = [], 0
    while i < len(lines):
        if lines[i].strip() not in KEYWORDS:
            i += 1
            continue
        j = i + 1                                          # first line of dashes: start of the table
        while j < len(lines) and not lines[j].startswith("-----"):
            j += 1
        k = j + 1                                          # second line of dashes: end of the table
        while k < len(lines) and not lines[k].startswith("-----"):
            k += 1
        if k >= len(lines):
            raise ValueError(f"{path.name}: the {lines[i].strip()} block at line {i + 1} has no closing line of dashes")
        proc = " ".join(lines[i + 1].split())
        m = re.match(r"(.*?)\s*<?->\s*(.*)", proc)
        try:
            energy = float(lines[i + 2].split()[0])
        except (IndexError, ValueError):
            energy = None                                  # attachment: no third line
        out.append(dict(kind=lines[i].strip(), process=proc, target=m.group(1) if m else proc,
                        product=m.group(2) if m else "", energy=energy, lines=lines[i:k + 1], file=path.name))
        i = k + 1
    return out


def species_of(processes):
    """The target species of the processes, in order of first appearance."""
    return list(dict.fromkeys(p["target"] for p in processes))


def matches(process, pattern):
    """True if the process line matches the exclude entry: the exact process line, or a pattern with the wildcards
    * (anything) and ? (one character).  Spaces are collapsed in both, upper / lower case must match."""
    return fnmatchcase(process, " ".join(pattern.split()))


def vib_number(product, pattern):
    """Vibrational quantum number w of an excitation to the product name: the first whole number after the word
    pattern ('N2( VIB 3V1)' -> 3, 'N2( VIB V1)' -> 1, 'O2(VIB v=2)' -> 2); 1 if there is none."""
    rest = re.split(re.escape(pattern), product, maxsplit=1, flags=re.IGNORECASE)[-1]
    m = re.search(r"\d+", rest)
    return int(m.group()) if m else 1


# =====================================================================================================================
#  2. A scheme
# =====================================================================================================================
class Scheme:
    """One EEDF scheme (the contents of InputData/EEDFSchemes/<name>.toml), with what the pipeline needs from it:
        xsec_file()                    the cross-section file for BOLSIG+ (written once)
        composition(x)                 species and mole fractions at x % admixture
        eedf_conditions(x, p, Tg)      (x, p, Tg) of the library a condition needs
        lib_folder(x, p, Tg)           folder of that library (in InputData/Bolsig)
    Built from a dict with the sections of the scheme file; checked by check(), which raises ValueError with a
    message saying what is wrong."""

    def __init__(self, d, path=None):
        self.path = Path(path) if path else None
        s, g, x, v, f, t = (d.get(k, {}) for k in ("scheme", "gases", "cross_sections", "vibrations", "field",
                                                    "gas_temperature"))
        self.name = s.get("name", "")
        self.author = s.get("author", "")
        self.created = str(s.get("created", ""))
        self.comment = s.get("comment", "").strip()
        self.background = g.get("background", "Ar")
        self.admixture = g.get("admixture", "")
        self.admixture_in_eedf = g.get("admixture_in_eedf", True)
        self.files = list(x.get("files", []))
        self.exclude = list(x.get("exclude", []))
        self.vib_treatment = v.get("treatment", "cold")
        self.Tv_K = v.get("Tv_K", 0.0)
        self.vib_pattern = v.get("pattern", "VIB")
        self.frequency_GHz = f.get("frequency_GHz", DEFAULT_FREQUENCY_GHZ)
        self.EN_Td = list(f.get("EN_Td", EN_MICROWAVE))
        self.Tg_source = t.get("source", "LAS")
        self.Tg_K = t.get("Tg_K", 0.0)

    # ---- checks ------------------------------------------------------------------------------------------------
    def check(self):
        """ValueError if anything in the scheme is wrong or missing; returns self."""
        def num(v):
            return isinstance(v, (int, float)) and not isinstance(v, bool)
        if not NAME_RE.fullmatch(self.name or ""):
            raise ValueError(f"name {self.name!r}: letters, digits, '-', '_' and '.', starting with a letter or digit, "
                             f"at most 40 characters")
        if self.name in BUILT_IN:
            raise ValueError(f"name {self.name!r} is a built-in EEDF option of Main.py - pick another name")
        if not (isinstance(self.background, str) and self.background):
            raise ValueError("[gases] background: the species name of the main gas, e.g. \"Ar\"")
        if not isinstance(self.admixture, str) or not isinstance(self.admixture_in_eedf, bool):
            raise ValueError("[gases] admixture must be a species name (or \"\") and admixture_in_eedf true / false")
        if not self.files or not all(isinstance(n, str) for n in self.files):
            raise ValueError("[cross_sections] files: at least one file name of Bolsig/Cross-Sections, in quotes")
        for n in self.files:
            if not (XSEC_DIR / n).is_file():
                raise ValueError(f"[cross_sections] files: {n} is not in {XSEC_DIR}")
        if not all(isinstance(e, str) and e.strip() for e in self.exclude):
            raise ValueError("[cross_sections] exclude: process lines or patterns, in quotes")
        if self.vib_treatment not in VIB_TREATMENTS:
            raise ValueError(f"[vibrations] treatment = {self.vib_treatment!r}: use "
                             f"{', '.join(map(repr, VIB_TREATMENTS))}")
        if self.vib_treatment == "boltzmann" and not (num(self.Tv_K) and self.Tv_K > 0):
            raise ValueError("[vibrations] Tv_K: the vibrational temperature in K (> 0) for treatment = \"boltzmann\"")
        if not (isinstance(self.vib_pattern, str) and self.vib_pattern):
            raise ValueError("[vibrations] pattern: the word that marks a vibrational excitation, e.g. \"VIB\"")
        if not (num(self.frequency_GHz) and self.frequency_GHz >= 0):
            raise ValueError("[field] frequency_GHz: microwave frequency in GHz, or 0 for DC")
        if len(self.EN_Td) < 2 or not all(num(e) and e > 0 for e in self.EN_Td):
            raise ValueError("[field] EN_Td: at least two E/N values in Td, all > 0")
        if self.Tg_source not in ("LAS", "fixed"):
            raise ValueError(f"[gas_temperature] source = {self.Tg_source!r}: \"LAS\" or \"fixed\"")
        if self.Tg_source == "fixed" and not (num(self.Tg_K) and self.Tg_K > 0):
            raise ValueError("[gas_temperature] Tg_K: the gas temperature in K (> 0) for source = \"fixed\"")
        self.Tv_K, self.frequency_GHz, self.Tg_K = float(self.Tv_K), float(self.frequency_GHz), float(self.Tg_K)
        self.EN_Td = sorted({float(e) for e in self.EN_Td})
        # the species must be in the cross sections, every exclude entry must match a process
        have = species_of(self.processes)
        for sp in self.gases:
            if sp not in have:
                raise ValueError(f"species {sp!r} has no processes in {', '.join(self.files)} "
                                 f"(species there: {', '.join(have)})")
        for e in self.exclude:
            if not any(matches(p["process"], e) for p in self.processes):
                raise ValueError(f"[cross_sections] exclude: {e!r} matches no process of {', '.join(self.files)} "
                                 f"(python CreateEEDFScheme.py --show {self.name} lists them)")
        if self.vib_treatment == "boltzmann":
            self.vib_levels                                # raises if the ladder cannot be made
        return self

    # ---- the choices in short ----------------------------------------------------------------------------------
    @property
    def gases(self):
        """Species in the Boltzmann equation (when the condition has admixture)."""
        return [self.background] + ([self.admixture] if self.admixture and self.admixture_in_eedf else [])

    @property
    def freq_Hz(self):
        return self.frequency_GHz * 1e9

    @property
    def Tg_fixed(self):
        return self.Tg_source == "fixed"

    @property
    def field_tag(self):
        return "dc" if self.frequency_GHz == 0 else f"mw{self.frequency_GHz:g}GHz"

    def physics(self):
        """Everything that changes the libraries (not name, author, comment), with the contents of the files."""
        return dict(background=self.background, admixture=self.admixture, admixture_in_eedf=self.admixture_in_eedf,
                    files={n: hashlib.sha1((XSEC_DIR / n).read_bytes()).hexdigest() for n in self.files},
                    exclude=self.exclude, vib_treatment=self.vib_treatment,
                    Tv_K=self.Tv_K if self.vib_treatment == "boltzmann" else None, vib_pattern=self.vib_pattern,
                    frequency_GHz=self.frequency_GHz, EN_Td=self.EN_Td,
                    Tg_K=self.Tg_K if self.Tg_fixed else "LAS")

    @cached_property
    def code(self):
        """6-character fingerprint of physics(): part of the library folder and cross-section file names."""
        return hashlib.sha1(json.dumps(self.physics(), sort_keys=True).encode()).hexdigest()[:6]

    def summary(self):
        """One paragraph saying what the scheme does (for the list and the printout)."""
        if not self.admixture:
            gas = f"{self.background} only"
        elif not self.admixture_in_eedf:
            gas = (f"{self.background} only in the Boltzmann equation; the {self.admixture} of the condition only "
                   f"acts in the CR model")
        else:
            gas = (f"{self.background} + {self.admixture} (mole fraction = the condition's {self.admixture} %, "
                   f"conditions without {self.admixture}: {self.background} only)")
        vib = ""
        if self.admixture and self.admixture_in_eedf and self.vib_processes:
            vib = {"cold": f"{self.admixture} vibrations cold (excitation from v = 0 only, as in the file)",
                   "saturated": f"{self.admixture} vibrations saturated ({len(self.vib_processes)} vibrational "
                                f"excitations left out: no net e-V energy loss)",
                   "boltzmann": f"{self.admixture}(v) in a Boltzmann distribution at Tv = {self.Tv_K:g} K "
                                f"(reversible ladder of {len(self.vib_levels)} levels)"}[self.vib_treatment] + "; "
        excl = (f"{len(self.excluded)} processes left out by hand ({'; '.join(self.exclude)}); " if self.exclude
                else "")
        field = "DC field" if self.frequency_GHz == 0 else f"{self.frequency_GHz:g} GHz microwave field"
        tg = (f"Tg = {self.Tg_K:g} K for every library" if self.Tg_fixed
              else "Tg = the condition's gas temperature (LAS trend or the data set's Tg_K)")
        return (f"{gas}; cross sections {' + '.join(self.files)}; {vib}{excl}{field}; {len(self.EN_Td)} E/N "
                f"{self.EN_Td[0]:g}-{self.EN_Td[-1]:g} Td; {tg}.")

    def label(self):
        """Short text for plot titles."""
        mixed = self.admixture and self.admixture_in_eedf
        gas = f"{self.background}/{self.admixture}" if mixed else self.background
        out = f"EEDF scheme '{self.name}': {gas} {'DC' if self.frequency_GHz == 0 else 'microwave'} EEDF"
        if mixed and self.vib_processes:
            out += {"cold": f", cold {self.admixture}(v)", "saturated": f", saturated {self.admixture}(v)",
                    "boltzmann": f", {self.admixture}(v) at T$_v$ = {self.Tv_K:g} K"}[self.vib_treatment]
        return out + (f", T$_g$ = {self.Tg_K:g} K" if self.Tg_fixed else "")

    # ---- the processes ---------------------------------------------------------------------------------------
    @cached_property
    def processes(self):
        """All processes of the scheme's files, joined in file order."""
        return [p for n in self.files for p in read_processes(XSEC_DIR / n)]

    @cached_property
    def excluded(self):
        """Processes left out by the exclude list."""
        return [p for p in self.processes if any(matches(p["process"], e) for e in self.exclude)]

    @cached_property
    def vib_processes(self):
        """Vibrational excitations of the admixture (excitation of the admixture, vib_pattern in the product) that
        are not excluded by hand."""
        if not self.admixture:
            return []
        return [p for p in self.processes if p["kind"] == "EXCITATION" and p["target"] == self.admixture
                and self.vib_pattern.lower() in p["product"].lower() and "<->" not in p["process"]
                and p not in self.excluded]

    @cached_property
    def vib_levels(self):
        """{w: level energy [eV]} of the admixture's vibrational ladder (v = 0 included), from the thresholds of the
        v = 0 -> w excitations."""
        E = {0: 0.0}
        for p in self.vib_processes:
            w = vib_number(p["product"], self.vib_pattern)
            if w in E:
                raise ValueError(f"two vibrational excitations of {self.admixture} to level {w} "
                                 f"('{p['process']}' and another one): cannot build the ladder - exclude one")
            E[w] = p["energy"]
        if len(E) < 2:
            raise ValueError(f"{self.admixture} has no vibrational excitations (processes with '{self.vib_pattern}' "
                             f"in the product) in {', '.join(self.files)}: use treatment = \"cold\"")
        if sorted(E) != list(range(len(E))):
            raise ValueError(f"vibrational levels of {self.admixture}: {sorted(E)} - the numbers must run 0, 1, 2, ... "
                             f"without gaps (check [vibrations] pattern)")
        return E

    def duplicates(self):
        """Processes (kind + process line) that appear more than once in the joined files."""
        seen, dup = set(), []
        for p in self.processes:
            k = (p["kind"], p["process"])
            if k in seen:
                dup.append(f"{p['kind']} {p['process']} ({p['file']})")
            seen.add(k)
        return dup

    # ---- what the pipeline uses -------------------------------------------------------------------------------
    def xsec_file(self):
        """The cross-section file BOLSIG+ is given, written if missing: the processes of the files without the
        excluded ones; 'saturated': without the vibrational excitations; 'boltzmann': these replaced by the
        reversible ladder 'X(vv) <-> X(vw)' for every v < w (cross section of 0 -> w-v, shifted to the threshold
        E_w - E_v; statistical weight ratio 1), as LASTgComparison.ladder_xsec does for the built-in Tv option."""
        path = SCHEME_XSEC_DIR / f"{self.name}_{self.code}.txt"
        if path.is_file():
            return path
        vib = self.vib_processes if self.admixture_in_eedf else []
        drop = self.excluded + (vib if self.vib_treatment in ("saturated", "boltzmann") else [])
        out = [f"Cross sections of the EEDF scheme {self.name} (code {self.code}), written by CreateEEDFScheme.py "
               f"on {date.today().isoformat()}.",
               f"Source files (Bolsig/Cross-Sections): {', '.join(self.files)}.",
               f"Left out by hand: {len(self.excluded)} processes; vibrations '{self.vib_treatment}'.", ""]
        for p in self.processes:
            if p not in drop:
                out += p["lines"] + [""]
        if self.vib_treatment == "boltzmann" and vib:
            E, tab = self.vib_levels, {vib_number(p["product"], self.vib_pattern): p for p in vib}
            g, vmax = self.admixture, max(E)
            for v in range(vmax):
                for w in range(v + 1, vmax + 1):
                    src, thr = tab[w - v], E[w] - E[v]
                    rows = src["lines"][next(i for i, s in enumerate(src["lines"]) if s.startswith("-----")) + 1:-1]
                    out += ["EXCITATION", f"{g}(v{v}) <-> {g}(v{w})", f"{thr:.6e}  1", f"SPECIES: e / {g}",
                            f"PROCESS: {g}(v{v}) <-> {g}(v{w}), cross section of v = 0 -> {w - v} shifted to the "
                            f"threshold", "COLUMNS: Energy (eV) | Cross section (m2)", "-----------------------------"]
                    for r in rows:
                        a, b = (float(c) for c in r.split()[:2])
                        out.append(f"{max(a - src['energy'] + thr, 0.0):.6e}\t{b:.6e}")
                    out += ["-----------------------------", ""]
        SCHEME_XSEC_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")   # written in one go: safe with parallel workers
        tmp.write_text("\n".join(out) + "\n")
        os.replace(tmp, path)
        return path

    def composition(self, x_pct):
        """(species, mole fractions) of the Boltzmann equation at x_pct % admixture.  With 'boltzmann' vibrations
        the admixture keeps its elastic / electronic / ionization processes at fraction x and its levels X(vk) get
        x p_k (Boltzmann at Tv): the fractions add up to 1 + x, as in the built-in Tv option (BOLSIG+ takes them as
        given: a 1 + x error on the background density, 2 % at 2 % admixture)."""
        x = x_pct / 100
        if x == 0 or len(self.gases) == 1:
            return [self.background], [1.0]
        species, fractions = [self.background, self.admixture], [1 - x, x]
        if self.vib_treatment == "boltzmann":
            lev = sorted(self.vib_levels)
            p = [math.exp(-self.vib_levels[k] / (K_B_EV * self.Tv_K)) for k in lev]
            species += [f"{self.admixture}(v{k})" for k in lev]
            fractions += [x * q / sum(p) for q in p]
        return species, fractions

    def eedf_conditions(self, x_pct, p_mTorr, Tg):
        """(x, p, Tg) of the library for a condition at x_pct % admixture, p_mTorr and gas temperature Tg: x = 0 when
        the admixture is not in the Boltzmann equation (one library for all admixture fractions), Tg fixed or not."""
        x = x_pct if len(self.gases) > 1 else 0.0
        return float(x), float(p_mTorr), float(self.Tg_K if self.Tg_fixed else Tg)

    def lib_folder(self, x_pct, p_mTorr, Tg):
        """Library folder of the EEDF conditions (eedf_conditions) in InputData/Bolsig."""
        mix = self.background if x_pct == 0 else f"{self.background}{self.admixture}_{x_pct:g}pct"
        return LIB_DIR / f"{self.name}_{mix}_{self.field_tag}_{Tg:g}K_{p_mTorr:g}mTorr_{self.code}"

    # ---- writing ---------------------------------------------------------------------------------------------
    def to_toml(self):
        """The scheme file: every choice with a comment saying what it does."""
        def val(v):
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, (int, float)):
                return f"{v:g}"
            if isinstance(v, list):
                return "[" + ", ".join(val(e) for e in v) + "]"
            return json.dumps(v, ensure_ascii=False)       # a TOML basic string
        en = textwrap.fill(", ".join(f"{e:g}" for e in self.EN_Td), 100, initial_indent="    ",
                           subsequent_indent="    ")
        excl = "".join(f"    {val(e)},\n" for e in self.exclude)
        comment = (f"'''\n{self.comment}\n'''" if "'''" not in self.comment and self.comment else val(self.comment))
        bar = "# " + "=" * 117
        return f"""{bar}
#  EEDF scheme  {self.name}
#  made with CreateEEDFScheme.py by {self.author or '?'} on {self.created}
{bar}
#  Use it in a Main.py input file:      [eedf]
#                                       model = "{self.name}"
#  This file can be edited by hand; then run  python CreateEEDFScheme.py --check <this file>  (checks it and updates
#  EEDF_schemes.md).  A change of the physics below gives new library folders (their names end in a code made from
#  these settings), so libraries made with the old settings are never reused by mistake.
#  What each setting means: the comments here, and the text at the top of CreateEEDFScheme.py.
{bar}

[scheme]
name = {val(self.name)}                 # = the file name; what the input files use
author = {val(self.author)}
created = {val(self.created)}
# What the scheme is for (shown in EEDF_schemes.md):
comment = {comment}

[gases]
# Species names as in the cross-section files.  The admixture's mole fraction is the condition's admixture % (the
# N2_percent / gas2_percent of the data set); the background gets the rest.  "" = no admixture.
background = {val(self.background)}
admixture = {val(self.admixture)}
# false = the admixture is left out of the Boltzmann equation (EEDF of the background gas alone for every
# condition); it then only acts in the CR model.
admixture_in_eedf = {val(self.admixture_in_eedf)}

[cross_sections]
# BOLSIG+ / LXCat files in Bolsig/Cross-Sections, joined in this order.
files = {val(self.files)}
# Processes left out: the process line as listed by  python CreateEEDFScheme.py --show {self.name}  (the line under
# ELASTIC / EXCITATION / ... in the file), or a pattern with * and ? wildcards, e.g. "Ar -> Ar(3D*".
exclude = [
{excl}]

[vibrations]
# What is done with the vibrational excitations of the admixture (processes whose product contains `pattern`):
#   "cold"       as in the file (excitation from v = 0 only, no superelastic collisions)
#   "saturated"  left out: no net energy loss of the electrons to the vibrations
#   "boltzmann"  reversible ladder X(vv) <-> X(vw) with the levels in a Boltzmann distribution at Tv_K
treatment = {val(self.vib_treatment)}
Tv_K = {val(self.Tv_K)}                       # vibrational temperature [K], only for "boltzmann"
pattern = {val(self.vib_pattern)}                # marks a vibrational excitation in the product name

[field]
# Microwave frequency [GHz] (omega/N with the gas density of each condition), 0 = DC.
frequency_GHz = {val(self.frequency_GHz)}
# E/N values [Td] of every library: the fit can only pick E/N within them.
EN_Td = [
{en}
]

[gas_temperature]
# "LAS"   = the gas temperature of each condition (the one the CR model uses: LAS trend or the data set's Tg_K)
# "fixed" = Tg_K below for every library (the CR model keeps each condition's own Tg)
source = {val(self.Tg_source)}
Tg_K = {val(self.Tg_K)}
"""


# =====================================================================================================================
#  3. Scheme files and the list
# =====================================================================================================================
def scheme_path(name):
    return SCHEME_DIR / f"{name}.toml"


def read_scheme_file(path):
    """The scheme in a scheme file, checked (ValueError with the file name if it is wrong)."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            return Scheme(tomllib.load(f), path).check()
    except (tomllib.TOMLDecodeError, ValueError) as err:
        raise ValueError(f"{path.name}: {err}") from None


def load_scheme(name):
    """The scheme <name> (InputData/EEDFSchemes/<name>.toml), checked.  What Main.py and the pipeline call."""
    path = scheme_path(name)
    if not path.is_file():
        raise ValueError(f"no EEDF scheme {name!r}: the built-in options are {', '.join(map(repr, BUILT_IN))}; "
                         f"schemes: {', '.join(map(repr, scheme_names())) or 'none yet'} "
                         f"(make one with  python CreateEEDFScheme.py)")
    s = read_scheme_file(path)
    if s.name != name:
        raise ValueError(f"{path.name}: name = {s.name!r} - it must be the file name ({name!r})")
    return s


def scheme_names():
    return sorted(p.stem for p in SCHEME_DIR.glob("*.toml"))


def write_list():
    """Rebuild EEDF_schemes.md from the built-in options and every scheme file -> its text."""
    out = ["# EEDF options", "",
           "Written by `CreateEEDFScheme.py` - do not edit by hand: edit the scheme files (`<name>.toml` in this "
           "folder) and run `python CreateEEDFScheme.py --list`.", "",
           "Use one in a Main.py input file:", "", "```toml", "[eedf]", 'model = "<name>"', "```", "",
           "## Built into Main.py", ""]
    out += [f"- **{n}** - {t}" for n, t in BUILT_IN.items()]
    out += ["", "## Schemes made with CreateEEDFScheme.py", ""]
    names = scheme_names()
    if not names:
        out.append("(none yet)")
    for n in names:
        try:
            s = load_scheme(n)
        except ValueError as err:
            out += [f"### {n}", "", f"**This scheme file has an error and cannot be used:** {err}", ""]
            continue
        out += [f"### {s.name}", "",
                f"- made by {s.author or '?'} on {s.created}; file `InputData/EEDFSchemes/{s.name}.toml`; "
                f"code `{s.code}`",
                f"- {s.summary()}", ""]
        if s.comment:
            out += ["> " + ln if ln.strip() else ">" for ln in s.comment.splitlines()] + [""]
    text = "\n".join(out).rstrip() + "\n"
    SCHEME_DIR.mkdir(parents=True, exist_ok=True)
    LIST_FILE.write_text(text, encoding="utf-8")
    return text


def save_scheme(s, overwrite=False):
    """Write the scheme file and rebuild the list -> path of the scheme file."""
    path = scheme_path(s.name)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists")
    SCHEME_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(s.to_toml(), encoding="utf-8")
    read_scheme_file(path)                                 # the written file reads back
    write_list()
    return path
