# -*- coding: utf-8 -*-
"""
CreateEEDFScheme.py

Defines new EEDF options ("EEDF schemes") for the main analysis (Main.py).

=======================================================================================================================
 WHAT AN EEDF SCHEME IS
=======================================================================================================================
The CR fit of Main.py needs, for every plasma condition (gas mixture, pressure, gas temperature), a BOLSIG+ "library":
the EEDF solved at a list of E/N values.  The fit then picks the E/N (and so the EEDF) that matches the measured lines.
HOW that library is computed is a choice: which gases enter the Boltzmann equation, which cross sections, what is done
with the vibrational levels of a molecular admixture, the field frequency, the E/N values and the gas temperature.

An EEDF scheme is one such set of choices, saved under a name.  This script asks the questions, suggests a name
(e.g. "Ar-only", "ArN2-Tvib-saturated", "ArN2-Tvib-8000K") and writes

    InputData/EEDFSchemes/<name>.toml      the scheme itself (a commented text file; can be edited by hand)
    InputData/EEDFSchemes/EEDF_schemes.md  the readable list of every scheme with its summary and your comments
                                           (rebuilt every time this script saves a scheme or runs with --list)

(The schemes themselves - reading, checking, the library folders - are in
pipeline/EEDFSchemes.py; this file only asks the questions.)

To use a scheme, put its name in the [eedf] section of a Main.py input file:

    [eedf]
    model = "ArN2-Tvib-saturated"

Main.py then builds the libraries the scheme needs (one per gas fraction, pressure and gas temperature of the data
sets, only the ones that do not exist yet) with the same BOLSIG+ routine as always (LASTgComparison.library), and the
CR tables on top of them.  Nothing is built by this script: it only defines the scheme.  Use  python Main.py
<input file> --plan  to see which libraries a run would build.

The three original EEDF options of Main.py ("pure_Ar", "ArN2_vib_saturated", "ArN2_vib_Tv") are built in and keep
working exactly as before; they are listed in EEDF_schemes.md too, but have no scheme file.

=======================================================================================================================
 HOW TO RUN IT
=======================================================================================================================
    python CreateEEDFScheme.py                    answer the questions (Enter = keep the default in [brackets])
    python CreateEEDFScheme.py --list             print the list of schemes and rebuild EEDF_schemes.md
    python CreateEEDFScheme.py --show <name>      everything about one scheme, including its processes
    python CreateEEDFScheme.py --check <file>     check a scheme file written or edited by hand and add it to the
                                                  list (the file is copied to InputData/EEDFSchemes/<name>.toml)
In Spyder: run the file (F5) and answer the questions in the console; for the other modes use
Run > Configuration per file > Command line options, e.g.  --list

=======================================================================================================================
 THE CHOICES  (the questions, in the order they are asked)
=======================================================================================================================
 1. Gases.  "background" is the main gas (Ar).  "admixture" is the second gas of the data sets (in the input files
    and file lists it is the N2_percent / gas2_percent column): its mole fraction in the Boltzmann equation is that
    percentage of each condition, the background gets the rest.  Leave the admixture empty for a one-gas scheme.
    You can also keep the admixture OUT of the Boltzmann equation ("Ar-only" EEDF for every condition, the admixture
    then only acts in the CR model: quenching and dilution) - this is what the built-in "pure_Ar" does.
    NOTE: the CR model itself only knows N2 as an admixture (quenching rates).  An EEDF scheme with another gas (O2,
    H2, ...) changes the EEDF only; the CR model still treats the second gas as N2.
 2. Cross sections.  One or more BOLSIG+ / LXCat cross-section files from Bolsig/Cross-Sections/.  Several files are
    joined into one (e.g. Biagi_Ar.txt + an N2 file from another database).  Every species of the scheme must have
    its processes in the file(s); a process that appears in two files is reported (it would count twice - exclude
    one of them).
 3. Processes to leave out ("exclude").  The script lists every process with a number; give the numbers (e.g.
    3, 7-9) and/or name patterns with wildcards (e.g. "Ar -> Ar(3D*", "* -> *HIGH*").  A pattern is matched against
    the process line exactly as listed (the line under the keyword ELASTIC / EXCITATION / ... in the file).
 4. Vibrations of the admixture (only asked if the admixture has vibrational excitations: processes of the admixture
    whose product contains the word in vib_pattern, "VIB" for the Biagi files):
       "cold"       as in the file: electrons excite N2(v=0) only, no way back (cold vibrations, Tv = Tg).  For our
                    Ar/N2 plasmas this EEDF fails the line fit (too much energy lost to the vibrations).
       "saturated"  the vibrational excitations are left out: the vibrations are in balance with the electrons,
                    no net energy loss to them (elastic, electronic excitation and ionization of N2 are kept).
                    Same physics as the built-in "ArN2_vib_saturated".
       "boltzmann"  the vibrational levels are given a Boltzmann population at a vibrational temperature Tv
                    (asked next, in K): the excitations v -> w are written as reversible  N2(vv) <-> N2(vw)  for every
                    pair of levels, so BOLSIG+ also adds the superelastic collisions that give energy back to the
                    electrons.  Tv ~ Tg is the cold case, Tv ~ 10000 K close to saturation.  Same physics as the
                    built-in "ArN2_vib_Tv" at a fixed Tv_K.  (Choosing Tv per condition from the LAS 1s5 is only
                    available in the built-in "ArN2_vib_Tv" with Tv_K = "LAS".)
       The level energies are the thresholds of the v = 0 -> w excitations; the number w is the first whole number
       after the vib_pattern word in the product name ("N2( VIB 3V1)" -> 3, "N2( VIB V1)" -> 1).
 5. Field.  Microwave frequency in GHz (2.42 = our source; the omega/N of each library uses the condition's gas
    density), or 0 for a DC field.
 6. E/N values [Td] of the library.  The fit can only pick E/N inside this list (and interpolates between the
    rows), so it must span the fitted range.  The default is the list of the built-in libraries (42 values,
    3-1000 Td for microwave, 1-200 Td for DC).
 7. Gas temperature of the EEDF (BOLSIG+'s gas temperature and the gas density in omega/N):
       "LAS"   the gas temperature of each condition, the same one the CR model uses (the LAS trend rounded to
               [eedf] Tg_step_K of the input file, or the Tg_K of the data set if it gives one)  <- default
       a number in K, e.g. 700: every library at that Tg; the CR model still uses each condition's own Tg.
 8. Name.  A name is suggested from the choices; press Enter to take it or type your own (letters, digits, - _ .;
    at most 40 characters).  Main.py input files refer to the scheme by this name.
 9. Comment.  Free text: why you made the scheme, what you want to test.  It goes into the scheme file and into the
    list EEDF_schemes.md so the others know what the scheme is for.

=======================================================================================================================
 WHERE THINGS GO
=======================================================================================================================
    InputData/EEDFSchemes/<name>.toml          the scheme
    InputData/EEDFSchemes/EEDF_schemes.md      the list of schemes
    InputData/EEDFSchemes/xsec/<name>_<code>.txt   the cross-section file BOLSIG+ is given (joined, processes left out,
                                                    vibrational ladder), written by Main.py before the first library
    InputData/Bolsig/<name>_<mixture>_<field>_<Tg>K_<p>mTorr_<code>/   the libraries (built by Main.py)
    <code> is a 6-character fingerprint of the scheme's physics (everything except name, author and comment, plus
    the contents of the cross-section files).  If you edit a scheme file by hand, or a cross-section file changes,
    the code changes and Main.py builds new libraries instead of reusing ones made with the old settings.
"""
import getpass
import os
import re
import subprocess
import sys
import textwrap
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline"))
from EEDFSchemes import *                           # noqa: E402,F401,F403 - Scheme, load_scheme, the folders, ...


# =====================================================================================================================
#  4. The questions
# =====================================================================================================================
def say(text):
    """Explanation printed before a question, indented."""
    print(textwrap.indent(textwrap.dedent(text).strip("\n"), "    "))


def ask(question, default="", check=None):
    """Ask until the answer passes check (a function that returns the value or raises ValueError).  Enter = the
    default (shown in [brackets])."""
    while True:
        shown = f" [{default}]" if default != "" else ""
        a = input(f"\n>>> {question}{shown}: ").strip()
        if not a:
            a = str(default)
        try:
            return check(a) if check else a
        except ValueError as err:
            print(f"    !! {err}")


def yes(a):
    if a.lower() in ("y", "yes", "true", "1"):
        return True
    if a.lower() in ("n", "no", "false", "0"):
        return False
    raise ValueError("answer y or n")


def positive(a):
    v = float(a)
    if not v > 0:
        raise ValueError("must be > 0")
    return v


def treatment(a):
    if a.lower() not in VIB_TREATMENTS:
        raise ValueError("cold, saturated or boltzmann")
    return a.lower()


def frequency(a):
    v = float(a)
    if v < 0:
        raise ValueError("a frequency in GHz, or 0 for DC")
    return v


def parse_numbers(a, n):
    """'3, 7-9' -> [3, 7, 8, 9] (1-based, checked against n)."""
    out = []
    for part in a.split(","):
        lo, _, hi = part.strip().partition("-")
        lo, hi = int(lo), int(hi or lo)
        if not 1 <= lo <= hi <= n:
            raise ValueError(f"{part.strip()}: numbers 1-{n}")
        out += range(lo, hi + 1)
    return out


def parse_en(a, default):
    """E/N answer: '' (default), 'min max n' (n values evenly spaced in log), or a list '3, 5, 10, 30'."""
    if a == "default":
        return list(default)
    parts = a.replace(",", " ").split()
    v = [float(p) for p in parts]
    if len(parts) == 3 and "," not in a and v[2].is_integer() and v[2] >= 2 and v[0] < v[1]:
        lo, hi, n = v[0], v[1], int(v[2])
        return [float(f"{lo * (hi / lo) ** (i / (n - 1)):.3g}") for i in range(n)]
    if len(v) < 2 or min(v) <= 0:
        raise ValueError("at least two E/N values > 0")
    return sorted(set(v))


def git_user():
    try:
        r = subprocess.run(["git", "config", "user.name"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or getpass.getuser()
    except (OSError, subprocess.SubprocessError):
        return getpass.getuser()


def print_processes(procs, mark=()):
    """Numbered list of processes (1-based), * in front of the ones in mark."""
    for i, p in enumerate(procs, 1):
        e = f"{p['energy']:.4g} eV" if p["energy"] is not None and p["kind"] in ("EXCITATION", "IONIZATION") else ""
        print(f"  {'*' if p in mark else ' '}{i:4d}  {p['kind']:<10s} {p['process']:<42s} {e:>11s}   {p['file']}")


def suggest_name(s):
    """Name from the choices, e.g. 'Ar-only', 'ArN2-Tvib-saturated', 'ArN2-Tvib-8000K-DC-Tg700K'; '-2', '-3', ...
    added if the name is taken."""
    if not s.admixture or not s.admixture_in_eedf:
        name = f"{s.background}-only"
    else:
        name = f"{s.background}{s.admixture}"
        if s.vib_processes:
            name += {"cold": "-cold-vib", "saturated": "-Tvib-saturated",
                     "boltzmann": f"-Tvib-{s.Tv_K:g}K"}[s.vib_treatment]
    if s.frequency_GHz == 0:
        name += "-DC"
    elif s.frequency_GHz != DEFAULT_FREQUENCY_GHZ:
        name += f"-{s.frequency_GHz:g}GHz"
    if s.Tg_fixed:
        name += f"-Tg{s.Tg_K:g}K"
    if s.exclude:
        name += "-custom"
    name = re.sub(r"[^A-Za-z0-9_.-]", "", name)[:36]
    base, i = name, 2
    while scheme_path(name).exists() or name in BUILT_IN:
        name, i = f"{base}-{i}", i + 1
    return name


def interactive():
    """Ask the questions, show the result, save the scheme on 'y'."""
    print("=" * 100)
    print(" CreateEEDFScheme: a new EEDF option for Main.py.  Press Enter to keep the value in [brackets].")
    print(" (The full explanation of every question is at the top of CreateEEDFScheme.py.)")
    print("=" * 100)
    d = {"scheme": {}, "gases": {}, "cross_sections": {}, "vibrations": {}, "field": {}, "gas_temperature": {}}

    # ---- 1. gases -------------------------------------------------------------------------------------------------
    say("""
        1. GASES.  The background is the main gas; the admixture is the second gas of the data sets (its % is the
           N2_percent / gas2_percent of each condition).  Names as in the cross-section files (e.g. Ar, N2, O2).""")
    g = d["gases"]
    g["background"] = ask("Background gas", "Ar")
    g["admixture"] = ask("Admixture gas ('none' for a one-gas scheme)", "N2",
                         lambda a: "" if a.lower() in ("none", "-", "") else a)
    g["admixture_in_eedf"] = False
    if g["admixture"]:
        say(f"""
            Should the {g['admixture']} be in the Boltzmann equation?  n = the EEDF of {g['background']} alone for every
            condition ("{g['background']}-only"; the {g['admixture']} then only acts in the CR model).""")
        g["admixture_in_eedf"] = ask(f"{g['admixture']} in the EEDF (y/n)", "y", yes)
    gases = [g["background"]] + ([g["admixture"]] if g["admixture_in_eedf"] else [])

    # ---- 2. cross sections ----------------------------------------------------------------------------------------
    files = sorted(p.name for p in XSEC_DIR.glob("*.txt"))
    have = {n: species_of(read_processes(XSEC_DIR / n)) for n in files}
    say(f"""
        2. CROSS SECTIONS.  Files in {XSEC_DIR.relative_to(ROOT)} (give one number, or several separated by commas
           to join them):""")
    for i, n in enumerate(files, 1):
        print(f"      {i:2d}  {n:<34s} species: {', '.join(have[n])}")
    full = [i for i, n in enumerate(files, 1) if all(sp in have[n] for sp in gases)]
    dflt = files.index(DEFAULT_XSEC) + 1 if DEFAULT_XSEC in files and files.index(DEFAULT_XSEC) + 1 in full else \
        (full[0] if full else 1)

    def pick_files(a):
        chosen = [files[i - 1] for i in dict.fromkeys(parse_numbers(a, len(files)))]
        sp = {s for n in chosen for s in have[n]}
        if miss := [s for s in gases if s not in sp]:
            raise ValueError(f"no processes of {', '.join(miss)} in {', '.join(chosen)}")
        return chosen
    d["cross_sections"]["files"] = ask("Cross-section file number(s)", dflt, pick_files)
    s = Scheme(d)
    if dup := s.duplicates():
        print(f"    !! these processes are in more than one of the files - leave one copy out in the next step:")
        for t in dup:
            print(f"       {t}")

    # ---- 3. processes to leave out -------------------------------------------------------------------------------
    procs = [p for p in s.processes if p["target"] in gases or p["target"].startswith(tuple(gases))]
    say(f"""
        3. PROCESSES TO LEAVE OUT.  The processes of {', '.join(gases)} in the file(s) (number, kind, process line,
           threshold).  Vibrational excitations are handled in the next question - do not list them here.""")
    print_processes(procs)

    def pick_exclude(a):
        if a.lower() in ("none", ""):
            return []
        out = []
        for part in [t.strip() for t in a.split(",") if t.strip()]:
            if re.fullmatch(r"\d+(\s*-\s*\d+)?", part):
                out += [procs[i - 1]["process"] for i in parse_numbers(part.replace(" ", ""), len(procs))]
            elif any(matches(p["process"], part) for p in s.processes):
                out.append(part)
            else:
                raise ValueError(f"{part!r} matches no process (patterns: * = anything, ? = one character)")
        return list(dict.fromkeys(out))
    d["cross_sections"]["exclude"] = ask("Leave out (numbers like 3, 7-9 and/or patterns like Ar -> Ar(3D*; "
                                         "'none')", "none", pick_exclude)
    s = Scheme(d)
    if s.excluded:
        print(f"    {len(s.excluded)} processes left out:")
        print_processes(s.excluded)

    # ---- 4. vibrations -------------------------------------------------------------------------------------------
    v = d["vibrations"]
    v["pattern"], v["treatment"], v["Tv_K"] = "VIB", "cold", 0.0
    if g["admixture_in_eedf"]:
        say(f"""
            4. VIBRATIONS OF {g['admixture']}.  Vibrational excitations are the {g['admixture']} excitations whose
               product contains a marker word ("VIB" in the Biagi files).""")
        v["pattern"] = ask("Marker word of a vibrational excitation", "VIB")
        s = Scheme(d)
        if s.vib_processes:
            print(f"    {len(s.vib_processes)} vibrational excitations of {g['admixture']}:")
            print_processes(s.vib_processes)
            say(f"""
                cold       as in the file: excitation from v = 0 only, no way back (vibrations at Tg)
                saturated  vibrations in balance with the electrons: these excitations left out, no net energy loss
                           to them  (like the built-in ArN2_vib_saturated)
                boltzmann  levels in a Boltzmann distribution at a vibrational temperature Tv, reversible ladder:
                           energy lost to cold and given back by hot vibrations  (like ArN2_vib_Tv)""")
            v["treatment"] = ask("Vibrations: cold / saturated / boltzmann", "saturated", treatment)
            if v["treatment"] == "boltzmann":
                say("""
                    Tv ~ Tg (300-700 K) is the cold case; ~10000 K is close to saturation.  Keep Tv <= ~10000 K:
                    hotter vibrations heat the electrons into a runaway and BOLSIG+ then takes > 20 min per library.""")
                v["Tv_K"] = ask("Vibrational temperature Tv [K]", 8000, positive)
                try:
                    Scheme(d).vib_levels
                except ValueError as err:
                    print(f"    !! {err}\n    -> vibrations set to 'cold'")
                    v["treatment"], v["Tv_K"] = "cold", 0.0
        else:
            print(f"    {g['admixture']} has no processes with '{v['pattern']}' in the product: no vibrational "
                  f"treatment needed (cold = as in the file).")

    # ---- 5. field ------------------------------------------------------------------------------------------------
    say("""
        5. FIELD.  Microwave frequency in GHz (our source: 2.42), or 0 for a DC field.""")
    f = d["field"]
    f["frequency_GHz"] = ask("Frequency [GHz]", f"{DEFAULT_FREQUENCY_GHZ:g}", frequency)

    # ---- 6. E/N ---------------------------------------------------------------------------------------------------
    en0 = EN_MICROWAVE if f["frequency_GHz"] > 0 else EN_DC
    say(f"""
        6. E/N VALUES [Td] of every library.  The fit can only pick E/N within them.
           default = the {len(en0)} values of the built-in libraries ({en0[0]:g}-{en0[-1]:g} Td):
           {', '.join(f'{e:g}' for e in en0)}
           or 'min max n' = n values evenly spaced in log (e.g. 3 1000 40), or a list like 3, 5, 10, 30, 100""")
    f["EN_Td"] = ask("E/N", "default", lambda a: parse_en(a, en0))

    # ---- 7. gas temperature --------------------------------------------------------------------------------------
    say("""
        7. GAS TEMPERATURE of the EEDF.  LAS = the gas temperature of each condition (the one the CR model uses:
           LAS trend, or the data set's Tg_K), or a number in K for every library.""")
    t = d["gas_temperature"]
    tg = ask("Gas temperature: LAS or a number in K", "LAS",
             lambda a: "LAS" if a.upper() == "LAS" else positive(a))
    t["source"], t["Tg_K"] = ("LAS", 0.0) if tg == "LAS" else ("fixed", tg)

    # ---- 8. name, 9. comment, author -----------------------------------------------------------------------------
    s = Scheme(d)
    say("""
        8. NAME of the scheme (what the input files use: [eedf] model = "<name>").  Letters, digits, - _ .""")

    def pick_name(a):
        if not NAME_RE.fullmatch(a):
            raise ValueError("letters, digits, '-', '_' and '.', at most 40 characters")
        if a in BUILT_IN:
            raise ValueError(f"{a} is a built-in option")
        return a
    d["scheme"]["name"] = ask("Name", suggest_name(s), pick_name)
    say("""
        9. COMMENT: what the scheme is for / what you want to test.  Several lines; an empty line ends it.""")
    lines = []
    while (ln := input("    | ")) != "":
        lines.append(ln)
    d["scheme"]["comment"] = "\n".join(lines)
    d["scheme"]["author"] = ask("Your name", git_user())
    d["scheme"]["created"] = date.today().isoformat()

    # ---- check, show, save ---------------------------------------------------------------------------------------
    try:
        s = Scheme(d).check()
    except ValueError as err:
        print(f"\n!! the scheme is not valid: {err}\nNothing saved.")
        return None
    print("\n" + "=" * 100)
    print(f" {s.name}  (code {s.code})")
    print(textwrap.fill(s.summary(), 98, initial_indent="   ", subsequent_indent="   "))
    if s.comment:
        print(textwrap.indent(s.comment, "   > "))
    print("=" * 100)
    over = scheme_path(s.name).exists()
    if over:
        print(f"!! {scheme_path(s.name).name} exists already - saving replaces it (input files using the name then "
              f"get the new scheme; libraries of the old one stay in InputData/Bolsig).")
    if not ask("Save this scheme (y/n)", "y", yes):
        print("Nothing saved.")
        return None
    path = save_scheme(s, overwrite=True)
    print(f"\nsaved {path}\nlist  {LIST_FILE}")
    print(f'\nUse it in an input file:   [eedf]\n                           model = "{s.name}"')
    print("then  python Main.py <input file> --plan   lists the libraries it will build.")
    return s


# =====================================================================================================================
#  5. Command line
# =====================================================================================================================
def show(name):
    s = load_scheme(name)
    print(s.path.read_text(encoding="utf-8"))
    print(f"code {s.code}\n{s.summary()}\n")
    print(f"processes of {' + '.join(s.files)} (* = left out by hand):")
    print_processes(s.processes, mark=s.excluded)
    if s.vib_processes:
        print(f"\nvibrational excitations of {s.admixture} (treatment '{s.vib_treatment}'):")
        print_processes(s.vib_processes)


def check_file(path):
    """Check a hand-written / edited scheme file and put it (as <name>.toml) into InputData/EEDFSchemes."""
    s = read_scheme_file(path)
    dst = scheme_path(s.name)
    if Path(path).resolve() != dst.resolve():
        if dst.exists() and not ask(f"{dst.name} exists - replace it (y/n)", "n", yes):
            return None
        SCHEME_DIR.mkdir(parents=True, exist_ok=True)
        dst.write_text(Path(path).read_text(encoding="utf-8"), encoding="utf-8")
    write_list()
    print(f"{s.name}: OK (code {s.code})\n{s.summary()}\n-> {dst}\nlist updated: {LIST_FILE}")
    return s


def main(argv):
    if "--list" in argv:
        print(write_list())
        print(f"(written to {LIST_FILE})")
    elif "--show" in argv:
        show(argv[argv.index("--show") + 1])
    elif "--check" in argv:
        check_file(argv[argv.index("--check") + 1])
    else:
        return interactive()


if __name__ == "__main__":
    main(sys.argv[1:])
