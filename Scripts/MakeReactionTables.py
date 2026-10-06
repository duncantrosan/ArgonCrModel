# -*- coding: utf-8 -*-
"""
LaTeX tables of every argon level and process in the CR model, with data sources.

Everything is read from the model input data (he.GetData()), so rerun after
changing the input data or adding species:
    python MakeReactionTables.py   ->   Scripts/Output/ArgonReactionTables.tex

The .tex is a fragment for \\input{}; it needs \\usepackage{booktabs,longtable}.
"""
import os
import re
import csv
import contextlib
import io
import numpy as np
import HelperFunctions as he
from ParserforLXCatData import PASCHEN_TO_LABEL

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

E_ION = 15.7596     # eV, Ar first ionization energy (as in he.CreateIonizationCrossSections)
K_MET = 6.4e-16     # m^3/s, metastable-metastable collisions (SolveLabelEquation in MainFileV2)
K_AR2 = 2.3e-21     # m^3/s, two-body quenching by ground-state Ar (GroundQuenchingLoss in MainFileV2)
K_AR3 = 1.4e-44     # m^6/s, three-body quenching by ground-state Ar (GroundQuenchingLoss in MainFileV2)
SIGMA_CAP = 1.0e-18 # m^2, cap on analytic cross sections (he.AddAnalyticExcitationCrossSections)
QUENCH_2P_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "InputData", "Ar_2p_quenching_Sadeghi2001.csv")

# Numbered in this order in the tables
REFERENCES = {
    'NIST': r'A.~Kramida, Yu.~Ralchenko, J.~Reader and NIST ASD Team, \emph{NIST Atomic Spectra '
            r'Database} (level energies, statistical weights, $A_{ki}$ and accuracy grades).',
    'EstMultiplet': r'This work: $A_{ki}$ not tabulated by NIST, from the $jK$-coupling line strength '
                    r'$S(KJ\rightarrow K^\prime J^\prime)=S(K,K^\prime)(2J+1)(2J^\prime+1)'
                    r'\{K\;J\;s;\,J^\prime\;K^\prime\;1\}^2$ with $S(K,K^\prime)$ fitted to the NIST '
                    r'components of the same $[K]\rightarrow[K^\prime]$ multiplet '
                    r'(\texttt{Scripts/EstimateTransitionProbabilities.py}).',
    'EstCoulomb': r'This work: $A_{ki}$ not tabulated by NIST, calculated in $jK$ coupling with '
                  r'Coulomb-approximation radial integrals (Bates--Damgaard, effective quantum numbers '
                  r'from the level energies and the $^2P_{3/2}$/$^2P_{1/2}$ limits), as the calculated '
                  r'transition probabilities used by Bogaerts \emph{et al.} '
                  r'(\texttt{Scripts/EstimateTransitionProbabilities.py}).',
    'KatsonisDrawin80': r'K.~Katsonis and H.~W.~Drawin, J.~Quant.~Spectrosc.~Radiat.~Transf. '
                        r'\textbf{23}, 1 (1980) ($jK$-coupling transition probabilities used by '
                        r'Bogaerts \emph{et al.} where no measured data exist).',
    'BatesDamgaard49': r'D.~R.~Bates and A.~Damgaard, Phil.~Trans.~R.~Soc.~Lond.~A \textbf{242}, 101 (1949).',
    'BSR': r'BSR database (B-spline R-matrix calculations by O.~Zatsarinny and K.~Bartschat, BSR-500 '
           r'model), \texttt{www.lxcat.net/BSR}, retrieved 4 October 2026; O.~Zatsarinny, '
           r'Comput.~Phys.~Commun. \textbf{174}, 273 (2006); O.~Zatsarinny and K.~Bartschat, '
           r'J.~Phys.~B \textbf{46}, 112001 (2013).',
    'NGFSRDW': r'NGFSRDW database (relativistic distorted-wave cross sections), '
               r'\texttt{www.lxcat.net/NGFSRDW}, retrieved 1 July 2026.',
    'Khakoo04': r'M.~A.~Khakoo \emph{et al.}, J.~Phys.~B \textbf{37}, 247 (2004).',
    'Kaur98': r'S.~Kaur, R.~Srivastava, R.~P.~McEachran and A.~D.~Stauffer, '
              r'J.~Phys.~B \textbf{31}, 4833 (1998).',
    'Gangwar10': r'R.~K.~Gangwar, L.~Sharma, R.~Srivastava and A.~D.~Stauffer, '
                 r'Phys.~Rev.~A \textbf{81}, 052707 (2010).',
    'Srivastava06': r'R.~Srivastava, A.~D.~Stauffer and L.~Sharma, Phys.~Rev.~A \textbf{74}, 012715 (2006).',
    'Sharma07': r'L.~Sharma, R.~Srivastava and A.~D.~Stauffer, Phys.~Rev.~A \textbf{76}, 024701 (2007).',
    'Gangwar12': r'R.~K.~Gangwar, L.~Sharma, R.~Srivastava and A.~D.~Stauffer, '
                 r'J.~Appl.~Phys. \textbf{111}, 053307 (2012).',
    'Bogaerts98': r'A.~Bogaerts, R.~Gijbels and J.~Vl\v{c}ek, J.~Appl.~Phys. \textbf{84}, 121 (1998), '
                  r'Sec.~III~A (Drawin formulae; H.~W.~Drawin (1967); J.~Vl\v{c}ek, '
                  r'J.~Phys.~D \textbf{22}, 623 (1989)).',
    'Vriens80': r'L.~Vriens and A.~H.~M.~Smeets, Phys.~Rev.~A \textbf{22}, 940 (1980).',
    'BK97': r'A.~K.~Bhatia and S.~O.~Kastner, J.~Quant.~Spectrosc.~Radiat.~Transf. '
            r'\textbf{58}, 347 (1997) (validation of the Monte Carlo escape factors).',
    'SAB07': r'Spectrochim.~Acta Part~B \textbf{62}, 344--356 (2007) (metastable diffusion coefficients).',
    'Ferreira85': r'C.~M.~Ferreira, J.~Loureiro and A.~Ricard, J.~Appl.~Phys. \textbf{57}, 82 (1985), '
                  r'as used by Bogaerts \emph{et al.} (1998).',
    'Tachibana86': r'K.~Tachibana, Phys.~Rev.~A \textbf{34}, 1007 (1986) (two- and three-body '
                   r'quenching of the metastables by ground-state argon).',
    'ZhuPu10': r'X.-M.~Zhu and Y.-K.~Pu, J.~Phys.~D: Appl.~Phys. \textbf{43}, 015204 (2010), Table~3 '
               r'(Ar-atom collisional transfer of the $2\mathrm{p}$ levels, measured at 300~K; original '
               r'measurements in Refs.~36--39 therein).',
    'Holstein47': r'T.~Holstein, Phys.~Rev. \textbf{72}, 1212 (1947).',
    'Walsh59': r'P.~J.~Walsh, Phys.~Rev. \textbf{116}, 511 (1959) (escape factors of a Voigt line in an '
               r'infinite cylinder, as used in [Bogaerts98]).',
    'BOLSIG': r'G.~J.~M.~Hagelaar and L.~C.~Pitchford, Plasma Sources Sci.~Technol. \textbf{14}, 722 '
              r'(2005) (BOLSIG+, two-term Boltzmann solver; command-line version 11/2019).',
    'MultiBolt': r'M.~Flynn \emph{et al.}, J.~Phys.~D: Appl.~Phys. \textbf{55}, 015201 (2022) '
                 r'(MultiBolt, multi-term Boltzmann solver, used as a cross-check of the EEDF).',
    'Biagi': r'S.~F.~Biagi, Magboltz 8.97 cross sections for Ar and N$_2$ (Biagi database on LXCat), '
             r'input to BOLSIG+ and MultiBolt for the EEDF.',
    'Ar1sN2': r'Quenching of Ar($1\mathrm{s}$) by N$_2$: median of the measurements compiled in '
              r'\texttt{InputData/Ar\_1s\_quenching\_data.csv} (Velazco \emph{et al.} 1978; Loeb 1981; '
              r'De~Jong 1974; Bochkova 1974; Bour\`ene and Le~Calv\'e 1975, 1977; Firestone 1978; '
              r'McNeely 1975; Chapman 1972; Clark 1972; see the file for the individual values).',
    'Sadeghi01': r'N.~Sadeghi, D.~W.~Setser, A.~Francis, U.~Czarnetzki and H.~F.~D\"obele, '
                 r'J.~Chem.~Phys. \textbf{115}, 3144 (2001) (total quenching of Ar($2\mathrm{p}_1$, '
                 r'$2\mathrm{p}_5$, $2\mathrm{p}_6$, $2\mathrm{p}_8$) by 22 gases at 300~K; all values in '
                 r'\texttt{InputData/Ar\_2p\_quenching\_Sadeghi2001.csv}).',
}
REF_NUM = {key: i + 1 for i, key in enumerate(REFERENCES)}


# reference keys for each value of the 'source' field of a transition
A_SOURCE = {'NIST ASD': ('NIST',),
            'jK multiplet (NIST)': ('EstMultiplet',),
            'jK + Coulomb approximation': ('EstCoulomb', 'BatesDamgaard49')}


def cite(*keys):
    return '[' + ','.join(str(REF_NUM[k]) for k in keys) + ']'


# references between entries are written [Key] in REFERENCES and numbered here
REFERENCES = {k: re.sub(r'\[(\w+)\]', lambda m: cite(m.group(1)) if m.group(1) in REF_NUM else m.group(0), v)
              for k, v in REFERENCES.items()}


def sci(x, digits=2, math=True):
    """3.13e8 -> $3.13\\times10^{8}$ (math=False: without the $ delimiters)"""
    if x == 0:
        s = '0'
    else:
        e = int(np.floor(np.log10(abs(x))))
        m = x / 10**e
        if round(m, digits) >= 10:
            m, e = m / 10, e + 1
        s = rf'{m:.{digits}f}\times10^{{{e}}}'
    return f'${s}$' if math else s


def paschen(label):
    """Paschen notation for the 1s, 2p and 2s levels (the 3d names in the
    parser are the NGFSRDW energy index, not Paschen notation)."""
    inverse = {v: k for k, v in PASCHEN_TO_LABEL.items()}
    m = re.match(r'([12])([sp])(\d+)$', inverse.get(label, ''))
    return rf'${m.group(1)}\mathrm{{{m.group(2)}}}_{{{m.group(3)}}}$' if m else '--'


def config_tex(cfg):
    """NIST configuration '3s2.3p5.(2P*<3/2>).4s' -> 3s^2 3p^5 (^2P^o_3/2) 4s (math, no $)."""
    parts = []
    for p in cfg.split('.'):
        core = re.fullmatch(r'\((\d)([SPDF])(\*?)<(.+?)>\)', p)
        if core:
            mult, L, odd, j = core.groups()
            parts.append(rf'({{}}^{{{mult}}}\mathrm{{{L}}}' + (r'^{\circ}' if odd else '') + rf'_{{{j}}})')
            continue
        shell = re.fullmatch(r'(\d+)([spdfg])(\d*)', p)
        if shell is None:
            raise ValueError(f'unrecognised configuration {cfg!r}')
        n, l, occ = shell.groups()
        parts.append(rf'{n}\mathrm{{{l}}}' + (rf'^{{{occ}}}' if occ else ''))
    return ''.join(parts)


def term_tex(term, J=None):
    """NIST term '2[3/2]*' -> ^2[3/2]^o (jK) or '1S' -> ^1S (LS), with J as subscript."""
    m = re.fullmatch(r'(\d)(\[.+?\]|[SPDFGHIK])(\*?)', term)
    if m is None:
        raise ValueError(f'unrecognised term {term!r}')
    mult, K, odd = m.groups()
    K = K if K.startswith('[') else rf'\mathrm{{{K}}}'
    J = '' if J is None else rf'_{{{int(J) if float(J).is_integer() else J}}}'
    return rf'{{}}^{{{mult}}}{K}' + (r'^{\circ}' if odd else '') + J


def designation(level):
    """Full NIST level designation, configuration + term + J, as inline math."""
    return '$' + config_tex(level['configuration']) + r'\;' + term_tex(level['term'], level['J']) + '$'


def longtable(colspec, header, rows, caption, label):
    """rows: lists of cell strings, or the string 'midrule' for a separator."""
    ncols = len(header)
    head = ' & '.join(header) + r' \\'
    out = [r'\begingroup\small\setlength{\tabcolsep}{4pt}',
           r'\begin{longtable}{' + colspec + '}',
           r'\caption{' + caption + r'}\label{' + label + r'}\\',
           r'\toprule', head, r'\midrule', r'\endfirsthead',
           r'\multicolumn{%d}{l}{\small\itshape Table~\thetable{} (continued)}\\' % ncols,
           r'\toprule', head, r'\midrule', r'\endhead',
           r'\midrule',
           r'\multicolumn{%d}{r}{\small\itshape continued on next page}\\' % ncols,
           r'\endfoot',
           r'\bottomrule', r'\endlastfoot']
    for row in rows:
        if row == 'midrule':
            out.append(r'\midrule')
            continue
        assert len(row) == ncols, (caption[:40], row)
        out.append(' & '.join(row) + r' \\')
    out += [r'\end{longtable}', r'\endgroup', '']
    return '\n'.join(out)


def rdw_reference(MD, lo, up):
    """Paper behind each NGFSRDW channel, per the database description."""
    mlo, mup = MD[lo]['manifold'], MD[up]['manifold']
    if mlo == 'ground':
        return {'4s': 'Khakoo04', '4p': 'Kaur98'}.get(mup, 'Gangwar10')   # 3d, 5s, 5p
    if mlo == '4s' and mup == '4p':
        return 'Srivastava06' if MD[lo]['kind'] == 'metastable' else 'Gangwar12'
    if mlo == '4s' and mup == '5p':
        return 'Sharma07'                                                 # metastable 4s -> 5p
    return 'Gangwar12'                                                    # 4s-4s, 4p-4p


def build(MD):
    levels = sorted(MD.values(), key=lambda v: v['energy_eV'])
    by_energy = {v['label']: i for i, v in enumerate(levels)}
    kind_name = {'ground': 'ground', 'metastable': 'metastable',
                 'resonant': 'resonant', 'normal': '--'}

    # ---- collect channels -------------------------------------------------
    rdw, allowed, forbidden = [], [], {}
    for lvl in levels:
        for r in lvl['Electron Impact CrossSections']['Reactants']:
            lo, up = lvl['label'], r['partner_label']
            if not r.get('analytic'):
                ref = 'BSR' if r.get('database') == 'BSR-500' else rdw_reference(MD, lo, up)
                rdw.append((lo, up, r['threshold_eV'], ref))
            elif r['method'].startswith('Drawin allowed'):
                f = float(re.search(r'f_lu=([-+0-9.eE]+)', r['method']).group(1))
                capped = np.max(r['cross_section']) >= SIGMA_CAP
                allowed.append((lo, up, r['threshold_eV'], f, capped))
            else:
                forbidden.setdefault(lo, []).append(up)
    rdw.sort(key=lambda c: (by_energy[c[0]], by_energy[c[1]]))
    allowed.sort(key=lambda c: (by_energy[c[0]], by_energy[c[1]]))

    LevelList = he.ImportLevelList()
    update = he.combine(LevelList, he.ImportReactionList()['transitions'])
    lines = [t for t in update['transitions_in_set'] if t['Aki'] is not None]
    lines.sort(key=lambda t: (-by_energy[t['upper_label']], t['wl_nm']))
    no_decay = [designation(v) for v in levels if v['kind'] not in ('ground', 'metastable')
                and not any(t['upper_label'] == v['label'] for t in lines)]
    D = {label: designation(v) for label, v in MD.items()}    # model label -> NIST designation

    n_ion = sum(1 for v in levels if v['kind'] != 'ground')
    ND_s3, ND_s5 = (D * he.Torr2Volume(1, 300) for D in he.FindDiffusionCoeff(1, 300))
    n_forb = sum(len(v) for v in forbidden.values())
    n_exc = len(rdw) + len(allowed) + n_forb
    n_bsr = sum(1 for c in rdw if c[3] == 'BSR')
    n_ngf = len(rdw) - n_bsr
    with open(he.ATOM_TRANSFER_FILE, newline='', encoding='utf-8') as file:
        at = list(csv.DictReader(line for line in file if not line.startswith('#')))
    q1s = he.ImportArQuenchingData('N2', verbose=False)

    tex = [r'% Generated by Scripts/MakeReactionTables.py from the CR-model input data - do not edit by hand.',
           r'% Requires \usepackage{booktabs,longtable}.', '']

    # ---- 1. levels --------------------------------------------------------
    rows = [['$' + config_tex(v['configuration']) + '$', '$' + term_tex(v['term']) + '$',
             f"{int(v['J'])}", f"{int(v['g'])}", f"{v['energy_eV']:.4f}", paschen(v['label']),
             kind_name[v['kind']]] for v in levels]
    tex.append(longtable(
        'llrrrll',
        ['Configuration', 'Term', '$J$', '$g$', '$E$ (eV)', 'Paschen', 'Type'],
        rows,
        rf'Argon levels in the CR model ({len(levels)} levels): configuration, term, $J$, statistical '
        rf'weight $g$ and energy $E$ above the ground state, from NIST {cite("NIST")}. '
        r'Paschen notation is given for the $1\mathrm{s}$, $2\mathrm{p}$ and $2\mathrm{s}$ levels.',
        'tab:ar_levels'))

    # ---- 2. summary of processes -----------------------------------------
    R_exc = r'$e+\mathrm{Ar}_i\rightarrow e+\mathrm{Ar}_j$'
    rows = [
        ['Electron-impact excitation, B-spline R-matrix', R_exc, f'{n_bsr}',
         r'$\sigma(\varepsilon)$ integrated over the EEDF; ground, $4\mathrm{s}$, $4\mathrm{p}$, '
         r'$3\mathrm{d}$, $5\mathrm{s}$ levels', cite('BSR')],
        ['Electron-impact excitation, RDW', R_exc, f'{n_ngf}',
         r'$\sigma(\varepsilon)$ integrated over the EEDF; only to the $5\mathrm{p}$ levels (not in BSR)',
         cite('NGFSRDW', 'Gangwar10', 'Sharma07')],
        ['Electron-impact excitation, analytic', R_exc, f'{len(allowed) + n_forb}',
         rf'Drawin $\sigma(\varepsilon)$ integrated over the EEDF; {len(allowed)} optically allowed, '
         rf'{n_forb} forbidden', cite('Bogaerts98', 'NIST')],
        ['Superelastic de-excitation', r'$e+\mathrm{Ar}_j\rightarrow e+\mathrm{Ar}_i$', f'{n_exc}',
         r'Maxwellian: $k_{ji}=k_{ij}(g_i/g_j)\,e^{\Delta E/T_e}$; numerical EEDF: Klein--Rosseland '
         r'$\sigma_{ji}$ integrated over the EEDF', 'from the two rows above'],
        ['Electron-impact ionization', r'$e+\mathrm{Ar}_i\rightarrow 2e+\mathrm{Ar}^+$', f'{n_ion}',
         r'Vriens--Smeets $\sigma(\varepsilon)$ integrated over the EEDF; all excited levels '
         rf'(not the ground state), threshold $E_\mathrm{{ion}}-E_i$ with $E_\mathrm{{ion}}={E_ION}$~eV',
         cite('Vriens80')],
        ['Spontaneous emission', r'$\mathrm{Ar}_k\rightarrow\mathrm{Ar}_i+h\nu$', f'{len(lines)}',
         r'$A_{ki}\Lambda_{ki}$; NIST where available, otherwise $jK$-coupling estimates',
         cite('NIST', 'EstMultiplet', 'EstCoulomb', 'KatsonisDrawin80')],
        ['Radiation trapping', 'all emission lines', f'{len(lines)}',
         r'Escape factor $\Lambda(\tau_0,a)$ from a Monte Carlo calculation for a uniformly emitting '
         r'hemisphere (\texttt{' + he.TRAPPING_TABLE_FILE.replace('_', r'\_') + r'}, '
         r'$a=10^{-6}$--$0.2$, $\tau_0=10^{-3}$--$10^{6}$), Voigt profile (Doppler and resonance '
         r'broadening), lower level as absorber; optionally Holstein--Walsh for the resonance lines',
         'this work, ' + cite('BK97', 'Holstein47', 'Walsh59')],
        ['Metastable diffusion', r'$\mathrm{Ar}(1\mathrm{s}_5,1\mathrm{s}_3)\rightarrow$ wall', '2',
         rf'$\tau_D^{{-1}}=D/\Lambda_D^2$, $\Lambda_D=R/4.493$ (hemisphere) or a given diffusion length, '
         rf'$ND={sci(ND_s5, 1, False)}$ and ${sci(ND_s3, 1, False)}$~m$^{{-1}}$\,s$^{{-1}}$', cite('SAB07')],
        ['Collisional transfer by Ar', r'$\mathrm{Ar}(2\mathrm{p}_x)+\mathrm{Ar}\rightarrow'
         r'\mathrm{Ar}(2\mathrm{p}_y,1\mathrm{s})+\mathrm{Ar}$', f'{len(at)}',
         r'$k(T_g)=k_{300}(T_g/300)^{1/2}$ downhill, uphill by detailed balance at $T_g$, loss '
         r'frequency $kN_g$ (Table~\ref{tab:ar_atom_transfer})', cite('ZhuPu10')],
        ['Quenching by N$_2$', r'$\mathrm{Ar}(1\mathrm{s},2\mathrm{p})+\mathrm{N}_2\rightarrow$ products',
         f'{len(q1s)}+4', r'loss frequency $k_Q x_{\mathrm{N}_2}N$ (+$k_{QM}x_{\mathrm{N}_2}x_{\mathrm{Ar}}N^2$); '
         r'only with N$_2$ admixture, which also dilutes the Ar ground state '
         r'(Tables~\ref{tab:ar_1s_n2_quenching}, \ref{tab:ar_n2_quenching})', cite('Ar1sN2', 'Sadeghi01')],
        ['Metastable--metastable collisions', r'$\mathrm{Ar}^m+\mathrm{Ar}^m\rightarrow$ products', '1',
         rf'$k={sci(K_MET, 1, False)}$~m$^3$\,s$^{{-1}}$', cite('Ferreira85')],
        ['Two-body quenching by Ar', r'$\mathrm{Ar}^m+\mathrm{Ar}\rightarrow 2\mathrm{Ar}$', '2',
         rf'$k_2={sci(K_AR2, 1, False)}$~m$^3$\,s$^{{-1}}$, loss frequency $k_2N_g$',
         cite('Tachibana86')],
        ['Three-body quenching by Ar', r'$\mathrm{Ar}^m+2\mathrm{Ar}\rightarrow\mathrm{Ar}_2^*+\mathrm{Ar}$',
         '2', rf'$k_3={sci(K_AR3, 1, False)}$~m$^6$\,s$^{{-1}}$, loss frequency $k_3N_g^2$; '
         r'the excimer $\mathrm{Ar}_2^*$ is not followed', cite('Tachibana86')],
    ]
    tex.append(longtable(
        r'p{0.19\linewidth}p{0.2\linewidth}rp{0.33\linewidth}p{0.12\linewidth}',
        ['Process', 'Reaction', '$N$', 'Rate coefficient', 'Source'], rows,
        r'Processes for argon in the CR model. Electron-impact rates are '
        r'$k=\sqrt{2e/m_e}\int\varepsilon\,\sigma(\varepsilon)f_0(\varepsilon)\,d\varepsilon$ '
        r'($\int\sqrt{\varepsilon}f_0\,d\varepsilon=1$), with $f_0$ either a Maxwellian at $T_e$ or a '
        r'numerical EEDF from BOLSIG+ ' + cite('BOLSIG') + ' (DC or microwave field, Ar or Ar/N$_2$) '
        'computed with the Biagi cross sections ' + cite('Biagi') + '.',
        'tab:ar_processes'))

    # ---- 3. excitation with numerical cross sections (BSR, NGFSRDW) ---------
    rows, prev = [], None
    for lo, up, thr, ref in rdw:
        if prev is not None and MD[lo]['manifold'] != MD[prev]['manifold']:
            rows.append('midrule')
        rows.append([D[lo], D[up], f'{thr:.4f}', cite(ref)])
        prev = lo
    tex.append(longtable(
        'llrl', ['Lower', 'Upper', r'$\varepsilon_\mathrm{th}$ (eV)', 'Source'], rows,
        rf'Electron-impact excitation channels with numerical cross sections ({len(rdw)} channels): '
        rf'{n_bsr} from the B-spline R-matrix BSR-500 calculations {cite("BSR")}, from threshold, with '
        r'the BSR thresholds shifted by $-0.22$ to $+0.15$~eV onto the NIST level energies (nine '
        r'near-degenerate $5\mathrm{s}\rightarrow3\mathrm{d}$ pairs whose order is reversed in BSR are '
        rf'left out); {n_ngf} to the $5\mathrm{{p}}$ levels, which BSR does not include, from the '
        rf'relativistic distorted-wave NGFSRDW database {cite("NGFSRDW")} (tabulated from 5--6~eV above '
        r'threshold; $\sigma=0$ below). Each channel is also included as its superelastic reverse process.',
        'tab:ar_numerical'))

    # ---- 4. analytic, optically allowed ------------------------------------
    rows = [[D[lo], D[up] + (r'\textsuperscript{*}' if capped else ''), f'{thr:.4f}', sci(f, 2)]
            for lo, up, thr, f, capped in allowed]
    tex.append(longtable(
        'llrr', ['Lower', 'Upper', r'$\Delta E$ (eV)', r'$f_{ij}$'], rows,
        rf'Optically allowed excitation channels without numerical cross sections ({len(allowed)} '
        rf'channels): Drawin '
        rf'cross section $\sigma=4\pi a_0^2(E_H/\Delta E)^2 f_{{ij}}\,\alpha\,(u-1)u^{{-2}}\ln(1.25\beta u)$, '
        rf'$u=\varepsilon/\Delta E$, $\alpha=\beta=1$ {cite("Bogaerts98")}, with the oscillator strength '
        rf'$f_{{ij}}$ from $A_{{ki}}$ (Table~\ref{{tab:ar_radiative}}; NIST {cite("NIST")} or the '
        rf'estimates of {cite("EstMultiplet", "EstCoulomb")}). $^*$Peak cross section capped at '
        rf'{sci(SIGMA_CAP, 0)}~m$^2$. Each channel is also included as its superelastic reverse process.',
        'tab:ar_allowed'))

    # ---- 5. analytic, forbidden ------------------------------------------
    rows = [[D[lo], str(len(forbidden[lo])), ', '.join(D[u] for u in sorted(forbidden[lo], key=by_energy.get))]
            for lo in sorted(forbidden, key=lambda l: by_energy[l])]
    tex.append(longtable(
        r'lrp{0.58\linewidth}', ['Lower', '$N$', 'Upper levels'], rows,
        rf'Optically forbidden excitation channels ({n_forb} channels): the remaining pairs of levels '
        r'without numerical cross sections or a radiative transition, with the Drawin forbidden cross section '
        rf'$\sigma=4\pi a_0^2\,\alpha\,(u-1)u^{{-2}}$, $\alpha=0.01$ {cite("Bogaerts98")}. As in '
        rf'{cite("Bogaerts98")}, parity-forbidden channels ($\Delta l\neq\pm1$) between the primed '
        r'($^2P_{1/2}$ core) and unprimed ($^2P_{3/2}$ core) systems are neglected, except among the '
        r'$4\mathrm{s}$ levels. Each channel is also included as its superelastic reverse process.',
        'tab:ar_forbidden'))

    # ---- 6. radiative transitions ------------------------------------------
    rows, prev = [], None
    for t in lines:
        up, lo = t['upper_label'], t['lower_label']
        if prev is not None and up != prev:
            rows.append('midrule')
        rows.append([D[up] if up != prev else '', D[lo] + (r'\textsuperscript{\dag}' if lo == 'ground' else ''),
                     f"{t['wl_nm']:.3f}", sci(t['Aki'], 2), t['acc'] or '--',
                     cite(*A_SOURCE[t.get('source', 'NIST ASD')])])
        prev = up
    n_src = {s: sum(1 for t in lines if t.get('source', 'NIST ASD') == s) for s in A_SOURCE}
    tex.append(longtable(
        'llrrll', ['Upper', 'Lower', r'$\lambda$ (nm)', r'$A_{ki}$ (s$^{-1}$)', 'Acc.', 'Source'], rows,
        rf'Spontaneous emission lines in the CR model ({len(lines)} lines): {n_src["NIST ASD"]} from '
        rf'NIST {cite("NIST")}; where NIST gives no $A_{{ki}}$, {n_src["jK multiplet (NIST)"]} completed '
        rf'from the NIST components of the same $jK$ multiplet {cite("EstMultiplet")} and '
        rf'{n_src["jK + Coulomb approximation"]} calculated in $jK$ coupling with Coulomb-approximation '
        rf'radial integrals {cite("EstCoulomb", "BatesDamgaard49")}, following Bogaerts \emph{{et al.}} '
        rf'{cite("Bogaerts98")}; '
        r'$\lambda$ in vacuum below 200~nm and in air above. Accuracy of $A_{ki}$: AA $\le1\%$, '
        r'A$+$ $\le2\%$, A $\le3\%$, B$+$ $\le7\%$, B $\le10\%$, C$+$ $\le18\%$, C $\le25\%$, '
        r'D$+$ $\le40\%$, D $\le50\%$, E $>50\%$. Lines are reduced by an escape factor $\Lambda$ '
        r'computed with the lower-level density as absorber, on every line or, as in '
        rf'{cite("Bogaerts98")}, only on lines to the ground state. $^\dagger$Resonance line to '
        r'the ground state (VUV). The metastable $1\mathrm{s}_5$ and $1\mathrm{s}_3$ levels have no '
        r'radiative decay'
        + (', and ' + ' and '.join(no_decay) + ' have no tabulated decay to a modelled level' if no_decay else '')
        + '.',
        'tab:ar_radiative'))

    # ---- 7. collisional transfer by ground-state Ar ---------------------------
    def pas(p):                                   # '2p3' -> $2\mathrm{p}_{3}$ ; '1s' -> 1s (any)
        m = re.match(r'([12])([sp])(\d*)$', p)
        return rf'${m.group(1)}\mathrm{{{m.group(2)}}}' + (rf'_{{{m.group(3)}}}$' if m.group(3) else '$')
    rows = [[r['process'], pas(r['from']), pas(r['to']) + ('$^\\ddagger$' if r['to'] == '1s' else ''),
             sci(float(r['k_cm3_s_300K']) * 1e-6, 1), r['Tg_exponent'], r['source']] for r in at]
    tex.append(longtable(
        'rllrrl', ['No.', 'From', 'To', r'$k_{300}$ (m$^3$\,s$^{-1}$)', 'Exp.', r'Refs.\ in ' + cite('ZhuPu10')],
        rows,
        rf'Population transfer between excited levels by collisions with ground-state argon, '
        r'$\mathrm{Ar}(x)+\mathrm{Ar}\rightarrow\mathrm{Ar}(y)+\mathrm{Ar}$, processes 9--26 of Zhu and Pu '
        rf'{cite("ZhuPu10")} (\texttt{{InputData/Ar\_2p\_atom\_transfer.csv}}). Listed downhill, '
        r'$k=k_{300}(T_g/300)^{\mathrm{Exp.}}$; the uphill rates follow from detailed balance, '
        r'$k_{y\rightarrow x}=k_{x\rightarrow y}(g_x/g_y)\exp[-(E_x-E_y)/k_BT_g]$. '
        r'$^\ddagger$Shared among the four $1\mathrm{s}$ levels in proportion to their statistical weights.',
        'tab:ar_atom_transfer'))

    # ---- 8. quenching of the 1s levels by N2 ---------------------------------
    rows = [[D[lab], paschen(lab), sci(q['kQ'], 2), sci(q['kQM'], 2) if q['kQM'] else '--',
             q['source_kQ'].split(' (')[0]] for lab, q in sorted(q1s.items(), key=lambda t: by_energy[t[0]])]
    tex.append(longtable(
        'llrrl', ['Level', 'Paschen', r'$k_Q$ (m$^3$\,s$^{-1}$)', r'$k_{QM}$ (m$^6$\,s$^{-1}$)', 'Value'], rows,
        r'Quenching of the $\mathrm{Ar}(1\mathrm{s})$ levels by N$_2$, $\mathrm{Ar}(1\mathrm{s})+\mathrm{N}_2'
        r'\rightarrow$ products (two-body $k_Q$; three-body $k_{QM}$ with Ar as third body where measured) '
        rf'{cite("Ar1sN2")}; loss frequency $k_Q x_{{\mathrm{{N}}_2}}N+k_{{QM}}x_{{\mathrm{{N}}_2}}x_{{\mathrm{{Ar}}}}N^2$. '
        r'With a dissociated fraction of the N$_2$ feed, only the remaining N$_2$ quenches.',
        'tab:ar_1s_n2_quenching'))

    # ---- 9. quenching of the 2p levels by N2 ---------------------------------
    with open(QUENCH_2P_FILE, newline='', encoding='utf-8') as file:
        q2p = [r for r in csv.DictReader(file) if r['quencher'] == 'N2' and r['Ar_state'].startswith('2p')]
    q2p.sort(key=lambda r: by_energy[r['CR_label']])
    rows = [[D[r['CR_label']], paschen(r['CR_label']), f"{MD[r['CR_label']]['energy_eV']:.4f}",
             sci(float(r['10^10_kQ_cm3_s']) * 1e-16, 1), f"{100 * float(r['kQ_rel_uncertainty']):.0f}",
             sci(float(r['10^-16_sigmaQ_cm2']) * 1e-20, 1)] for r in q2p]
    tex.append(longtable(
        'llrrrr', ['Level', 'Paschen', '$E$ (eV)', r'$k_Q$ (m$^3$\,s$^{-1}$)', r'$\Delta k_Q/k_Q$ (\%)',
                   r'$\sigma_Q$ (m$^2$)'], rows,
        r'Total quenching of $\mathrm{Ar}(2\mathrm{p})$ levels by N$_2$ at 300~K, '
        r'$\mathrm{Ar}(2\mathrm{p})+\mathrm{N}_2\rightarrow$ products, measured by two-photon laser '
        r'excitation from the ground state and time-resolved fluorescence (so only $J=0$ and 2 levels) '
        rf'{cite("Sadeghi01")}; $\sigma_Q=k_Q/\langle v\rangle$, loss frequency $k_Q\,x_{{\mathrm{{N}}_2}}N_g$. '
        r'In the CR model as a loss of these four levels (CRFitNeTe \texttt{quench\_4p}); the other '
        r'$4\mathrm{p}$ levels have no data and are not quenched.',
        'tab:ar_n2_quenching'))

    # ---- 10. references ----------------------------------------------------
    rows = [[f'[{n}]', text] for text, n in ((REFERENCES[k], REF_NUM[k]) for k in REFERENCES)]
    tex.append(longtable(r'lp{0.9\linewidth}', ['', 'Source'], rows,
                         'Data sources for Tables~\\ref{tab:ar_levels}--\\ref{tab:ar_n2_quenching}.',
                         'tab:ar_sources'))
    return '\n'.join(tex), dict(levels=len(levels), bsr=n_bsr, ngfsrdw=n_ngf, allowed=len(allowed),
                                forbidden=n_forb, ionization=n_ion, radiative=len(lines),
                                atom_transfer=len(at), metastable_levels=2)


# ============================================================================
# Model description (ArgonCRModel.tex): text, BOLSIG+ results and current fit
# results; \input{}s ArgonReactionTables.tex
# ============================================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = dict(   # outputs of the analysis scripts used in the status section (skipped if missing)
    sweep=os.path.join(ROOT_DIR, 'Experimental_Data', 'Output', 'PressureSweep', 'pressure_sweep_fits.csv'),
    sweep_fig=os.path.join(ROOT_DIR, 'Experimental_Data', 'Output', 'PressureSweep', 'figures',
                           '1s_densities_vs_pressure_microwave_nc.png'),
    act=os.path.join(ROOT_DIR, 'Experimental_Data', 'Output', 'ActinometryN2Content', 'nN_nAr_per_condition_Ne_nc.csv'),
    act_fig=os.path.join(ROOT_DIR, 'Experimental_Data', 'Output', 'ActinometryN2Content',
                         'n2_sweep_microwave_dissociation_Ne_nc.png'),
)
BOLSIG_LIBS = [   # (LaTeX label, plot label, library folder) of the Te_eff table and the EEPF figure
    (r'Ar, DC, 800~K', 'Ar, DC, 800 K', 'Ar_Biagi_bolsig_dc_800K'),
    (r'Ar, 2.42~GHz, 1~Torr, 800~K', 'Ar, 2.42 GHz, 1 Torr, 800 K', 'Ar_Biagi_bolsig_mw2.42GHz_800K_1000mTorr'),
    (r'Ar + 2\,\% N$_2$, 2.45~GHz, 1~Torr, 300~K', 'Ar + 2 % N2, 2.45 GHz, 1 Torr, 300 K', 'ArN2_2pct_bolsig_mw'),
]
EN_TABLE = [5, 10, 20, 30, 50, 100, 200, 500, 1000]   # Td
TE_EEPF = 1.0                                          # eV, Te_eff of the EEPFs compared in the figure


def bolsig_results():
    """Te_eff(E/N) table rows and the EEPF comparison figure from the BOLSIG+ libraries
    (a library whose Te_eff does not come within 10 % of TE_EEPF is left out of the figure)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, fig = [], plt.figure(figsize=(7, 4.6))
    ax = fig.gca()
    for (label, plot_label, name), col in zip(BOLSIG_LIBS, ('C0', 'C3', 'C2')):
        folder = he.BOLSIG_FOLDER / name
        if not he.IsBolsigLibrary(folder):
            continue
        lib = he.ImportBolsigLibrary(folder, verbose=False)
        en, te = lib['EN_Td'], np.array([e['Te_eff'] for e in lib['EEDFs']])
        rows.append([label] + [f'{np.interp(np.log(x), np.log(en), te):.2f}' if en[0] <= x <= en[-1] else '--'
                               for x in EN_TABLE])
        i = int(np.argmin(np.abs(te - TE_EEPF)))
        if abs(te[i] - TE_EEPF) > 0.1 * TE_EEPF:
            continue
        e, x = lib['EEDFs'][i], en[i]
        ok = e['EEPF'] > 0
        ax.semilogy(e['E'][ok], e['EEPF'][ok], color=col,
                    label=f'{plot_label}, {x:g} Td, Te,eff = {e["Te_eff"]:.2f} eV')
    E = np.linspace(0, 30, 600)
    ax.semilogy(E, 2 / np.sqrt(np.pi) * TE_EEPF ** -1.5 * np.exp(-E / TE_EEPF), 'k--',
                label=f'Maxwellian, Te = {TE_EEPF:g} eV')
    for thr, txt in ((11.55, 'Ar 1s$_5$'), (13.08, 'Ar 2p'), (12.0, 'N 3p $^4$S$^o$')):
        ax.axvline(thr, color='0.6', lw=0.8, ls=':')
        ax.text(thr, 2e-9, txt, rotation=90, fontsize=7, color='0.4', ha='right', va='bottom')
    ax.set_xlim(0, 25)
    ax.set_ylim(1e-9, 2)
    ax.set_xlabel('electron energy (eV)')
    ax.set_ylabel(r'EEPF $f_0$ (eV$^{-3/2}$)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'eepf_comparison.png'), dpi=200)
    plt.close(fig)
    return rows


def _tabular(colspec, header, rows):
    return '\n'.join([r'\begin{tabular}{' + colspec + '}', r'\toprule', ' & '.join(header) + r' \\', r'\midrule']
                     + [' & '.join(r) + r' \\' for r in rows] + [r'\bottomrule', r'\end{tabular}'])


def _figure(path, caption, width=0.8):
    rel = os.path.relpath(path, OUTPUT_DIR).replace('\\', '/')
    return (r'\IfFileExists{' + rel + r'}{\begin{figure}[htbp]\centering\includegraphics[width=' + f'{width}'
            + r'\linewidth]{' + rel + r'}\caption{' + caption + r'}\end{figure}}{}')


def results_section():
    """Tables of the current pressure-sweep and actinometry results (from the analysis outputs)."""
    import pandas as pd
    out = []
    if os.path.exists(RESULTS['sweep']):
        g = pd.read_csv(RESULTS['sweep'])
        g = g[g.family == 'microwave'].sort_values('p_mTorr')
        rows = [[f'{r.p_mTorr:.0f}', f'{r.Te_nc:.2f}', f'{r.EN_nc:.0f}', f'{r.chi2_red_nc:.2f}', f'{r.dchi2_nc:.0f}']
                + [sci(getattr(r, f'n_{m}_nc'), 1) for m in ('1s5', '1s4', '1s3', '1s2')] for r in g.itertuples()]
        out.append(r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3.5pt}' + '\n' + _tabular(
            'rrrrrrrrr', ['$p$ (mTorr)', r'$T_{e,\mathrm{eff}}$ (eV)', '$E/N$ (Td)', r'$\chi^2$/dof',
                          r'$\Delta\chi^2/s^2$', r'$n(1\mathrm{s}_5)$', r'$n(1\mathrm{s}_4)$', r'$n(1\mathrm{s}_3)$',
                          r'$n(1\mathrm{s}_2)$'], rows)
            + '\n' + r'\caption{Pure-Ar pressure sweep (80~W, $T_g=800$~K), microwave EEDF with $N_e=n_c$: '
            r'fitted $T_{e,\mathrm{eff}}$ and $E/N$, fit quality, the cost $\Delta\chi^2/s^2$ of pinning $N_e$ '
            r'relative to the free fit, and the CR densities (m$^{-3}$) of the four $1\mathrm{s}$ levels. '
            r'From \texttt{SmallAnalysisScripts/PressureSweepAnalysis.py}.}\label{tab:res_sweep}\end{table}')
        out.append(_figure(RESULTS['sweep_fig'], r'CR densities of the $1\mathrm{s}$ levels along the pressure '
                           r'sweep (microwave EEDF, $N_e=n_c$; band: $\Delta\chi^2\le1\,s^2$ along $E/N$).'))
    if os.path.exists(RESULTS['act']):
        c = pd.read_csv(RESULTS['act'])
        c = c[(c.sweep == 'N2 fraction') & (c.eedf == 'microwave')]
        rows = []
        for x, h in c.groupby('x'):
            d, s = (h[h.variant == v].iloc[0] for v in ('direct', 'stepwise'))
            rows.append([f'{x:g}', f'{s.Te_eff:.2f}', f'{d.dissociation:.3f}', f'{s.dissociation:.3f}',
                         rf'$\times{np.exp(s.ln_err_stat):.2f}^{{\pm1}}$', f'{s.n_pairs:.0f}'])
        out.append(r'\begin{table}[htbp]\centering\small' + '\n' + _tabular(
            'rrrrrr', [r'N$_2$ (\%)', r'$T_{e,\mathrm{eff}}$ (eV)', r'$D$ direct', r'$D$ stepwise',
                       'error (stepwise)', 'pairs'], rows)
            + '\n' + r'\caption{N$_2$ dissociation degree $D=n_\mathrm{N}/(2n_{\mathrm{N}_2})$ from N\,I/Ar\,I '
            r'actinometry (N\,I 744.2~nm), microwave EEDF of each Ar/N$_2$ mixture with $N_e=n_c$, '
            r'85~W, 1~Torr; direct: Ar($2\mathrm{p}$) from the ground state only; stepwise: with the CR '
            r'model\textquoteright s production of the Ar level. From '
            r'\texttt{Experimental\_Data/ExperimentalDataAnalysis/ActinometryNitrogenContent.py}.}'
            r'\label{tab:res_act}\end{table}')
        out.append(_figure(RESULTS['act_fig'], r'N$_2$ dissociation degree along the N$_2$ sweep '
                           r'(microwave EEDF, $N_e=n_c$).'))
    return '\n\n'.join(out)


DESCRIPTION = r"""\documentclass[11pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,graphicx}
\usepackage[hidelinks]{hyperref}
\title{Collisional--radiative model of argon for optical emission spectroscopy of
microwave Ar and Ar/N$_2$ discharges}
\author{}
\date{Generated <<DATE>> by \texttt{Scripts/MakeReactionTables.py}}
\begin{document}
\maketitle

\section{Overview}
The collisional--radiative (CR) model (\texttt{Scripts/MainFileV2.py}, function \texttt{CRModel})
computes the steady-state densities of <<NEXC>> excited argon levels (Table~\ref{tab:ar_levels}:
<<MANIFOLDS>>) for a given electron energy distribution function (EEDF), electron density $N_e$,
pressure $p$, gas temperature $T_g$ and plasma size $R$. The ground-state density is fixed,
$N_g=p/k_BT_g$ (reduced by the N$_2$ fraction for Ar/N$_2$). From the densities follow the emission
intensities of <<NRAD>> lines, which are compared with calibrated echelle spectra
(\texttt{Experimental\_Data/ExperimentalDataAnalysis/CRFitNeTe.py}) to infer the EEDF and $N_e$, and
which give the excitation rates of the argon reference lines in N\,I/Ar\,I actinometry
(\texttt{ActinometryNitrogenContent.py}). Table~\ref{tab:ar_processes} lists all processes;
the following sections describe how they enter the model.

\section{Balance equations}
For each excited level $i$,
\begin{equation}
\frac{dn_i}{dt}=0=\sum_{j} N_e k^{e}_{ji} n_j+\sum_{k>i} A_{ki}\Lambda_{ki} n_k
 +N_g\sum_j k^{\mathrm{Ar}}_{ji} n_j
 -n_i\Big[N_e\big(\textstyle\sum_j k^{e}_{ij}+k^{\mathrm{ion}}_i\big)+\sum_{l<i}A_{il}\Lambda_{il}
 +\tau_{D,i}^{-1}+k_m n_m+\nu_{Q,i}+N_g\textstyle\sum_j k^{\mathrm{Ar}}_{ij}\Big],
\end{equation}
where $k^e_{ji}$ is electron-impact excitation or superelastic de-excitation from level $j$
(including the ground state), $A_{ki}\Lambda_{ki}$ spontaneous emission reduced by the escape factor,
$k^{\mathrm{Ar}}$ population transfer by ground-state atoms (Table~\ref{tab:ar_atom_transfer}),
$k^{\mathrm{ion}}_i$ ionization, $\tau_{D,i}$ diffusion to the wall (metastables only),
$k_m n_m$ metastable--metastable collisions ($n_m=n_{1s_5}+n_{1s_3}$) and $\nu_{Q,i}$ quenching by
ground-state Ar (metastables) and by N$_2$. Radiative decay to levels outside the model is kept as a
loss. The equations are linear in the $n_i$ except for $k_m n_m n_i$ and the escape factors,
which depend on the lower-level densities. They are solved as one linear system (with a Newton
step for the quadratic term), and the escape factors are recomputed from the new densities until
the largest relative change is below $10^{-6}$.

\section{Electron-impact processes}
\paragraph{Excitation.} <<NBSR>> channels between the ground, $4\mathrm{s}$, $4\mathrm{p}$, $3\mathrm{d}$ and
$5\mathrm{s}$ levels use the B-spline R-matrix (BSR-500) cross sections of Zatsarinny and Bartschat,
from threshold, with the thresholds moved onto the NIST level energies
(Table~\ref{tab:ar_numerical}). BSR does not include the $5\mathrm{p}$ levels; <<NNGF>> channels to them
use relativistic distorted-wave (NGFSRDW) cross sections, which are tabulated only from 5--6~eV above
threshold. All other pairs use Drawin's formulae (Bogaerts \emph{et al.}): <<NALLOW>> optically
allowed channels with the oscillator strength from $A_{ki}$ (Table~\ref{tab:ar_allowed}) and <<NFORB>>
forbidden ones (Table~\ref{tab:ar_forbidden}). Until 6 October 2026 the model used NGFSRDW for all
numerical channels; its ground$\rightarrow4\mathrm{s}/4\mathrm{p}$ cross sections are 10--30 times
larger than BSR and the Biagi set used in the Boltzmann solver, which made the model and the EEDF
inconsistent.
\paragraph{Superelastic de-excitation} is included for every excitation channel: by detailed balance
for a Maxwellian, and from the Klein--Rosseland cross section for a numerical EEDF.
\paragraph{Ionization} from each excited level uses the Vriens--Smeets cross section; the ion
density is not followed.
\paragraph{Rate coefficients} are
$k=\sqrt{2e/m_e}\int_0^\infty\varepsilon\,\sigma(\varepsilon)f_0(\varepsilon)\,d\varepsilon$ with the
EEPF $f_0$ normalized to $\int\sqrt{\varepsilon}f_0\,d\varepsilon=1$.

\section{Electron energy distribution: BOLSIG+}
The EEDF is either a Maxwellian at $T_e$ or a numerical solution of the two-term Boltzmann equation
from BOLSIG+ (Hagelaar and Pitchford) with the Biagi cross sections for Ar and N$_2$
(\texttt{HelperFunctions.RunBolsig}), as a library over the reduced field $E/N$
(\texttt{InputData/Bolsig/}). A library is computed for each gas mixture, gas temperature and, for
the microwave field, each pressure: in an HF field of angular frequency $\omega$ the EEDF depends on
$E/N$ and $\omega/N$ (2.42 or 2.45~GHz). Electron--electron collisions and superelastic collisions
with the CR excited-state populations can be switched on, which makes the EEDF depend on $N_e$ as well
(\texttt{BuildBolsigLibrary}); the results here are without them. The temperature quoted for a
numerical EEDF is $T_{e,\mathrm{eff}}=\tfrac23\langle\varepsilon\rangle$ (Table~\ref{tab:bolsig}).
Multi-term solutions (MultiBolt, 2--8 terms) give the same EEDF in Ar + 10\,\% N$_2$.

\begin{table}[htbp]\centering\small
<<BOLSIGTABLE>>
\caption{$T_{e,\mathrm{eff}}$ (eV) of BOLSIG+ EEDFs against the reduced field $E/N$ (Td); -- outside
the computed range. In the microwave field of pure Ar, $T_{e,\mathrm{eff}}$ is not monotonic in $E/N$
(Ramsauer minimum) and stays between about 0.8 and 1.4~eV over 25--600~Td, while the DC EEDF reaches
3--5~eV. The Ar/N$_2$ library is for 300~K (larger $N$, so smaller $\omega/N$) and is not directly
comparable with the 800~K pure-Ar one.}\label{tab:bolsig}
\end{table}

\IfFileExists{eepf_comparison.png}{\begin{figure}[htbp]\centering
\includegraphics[width=0.8\linewidth]{eepf_comparison.png}
\caption{Microwave BOLSIG+ EEPFs with $T_{e,\mathrm{eff}}\approx<<TEEEPF>>$~eV compared with a
Maxwellian of the same temperature (the DC EEDFs of pure Ar do not reach $T_{e,\mathrm{eff}}$ below
1.6~eV). At the same mean energy the microwave EEPFs have a plateau up to the inelastic thresholds
(dotted), where they lie about two orders of magnitude above the Maxwellian, and fall off steeply
above them. Excitation rates, and ratios of rates with different thresholds, therefore depend
strongly on the EEDF shape, not only on $T_{e,\mathrm{eff}}$.}\label{fig:eepf}\end{figure}}{}

\section{Radiation transport}
Each line $k\rightarrow i$ is multiplied by an escape factor $\Lambda(\tau_0,a)$ with the lower level
as absorber: line-centre optical depth $\tau_0=k_0R$ with $k_0=\frac{\lambda^2}{8\pi}\frac{g_k}{g_i}
A_{ki}\,\phi(0)\,n_i$ and the Voigt parameter $a$ from Doppler and resonance broadening
(\texttt{HelperFunctions.FindTauInModel}). $\Lambda$ is interpolated from a Monte Carlo calculation
for a uniformly emitting hemisphere of radius $R$ over $a=10^{-6}$--$0.2$ and $\tau_0=10^{-3}$--$10^6$.
The first version of that table started at $a=0.01$, while the Ar resonance lines have
$a\approx10^{-3}$--$5\times10^{-3}$ at 0.5--1.5~Torr; the extrapolation overestimated their escape by
1.2--1.6 times. As a systematic, the resonance lines can instead use the Holstein--Walsh
fundamental-mode escape factor of a cylinder (\texttt{escape\_mode='walsh'}), 1.4--3.4 times lower.
For tests, the escape factors of the lines to excited levels can be frozen at the solution for a
reference $N_e$ (\texttt{trap\_ref\_Ne}) or switched off (\texttt{trap\_lines='ground'}).

\section{Heavy-particle processes}
The metastables $1\mathrm{s}_5$ and $1\mathrm{s}_3$ diffuse to the wall with
$\tau_D^{-1}=D/\Lambda_D^2$, $\Lambda_D=R/4.493$ for a hemisphere, and are lost by
metastable--metastable collisions and by two- and three-body quenching with ground-state Ar.
Collisions with ground-state atoms transfer population among the $2\mathrm{p}$ levels and from
$2\mathrm{p}$ to $1\mathrm{s}$ (Zhu and Pu; Table~\ref{tab:ar_atom_transfer}), with the reverse
rates from detailed balance at $T_g$. With N$_2$ in the feed, the four $1\mathrm{s}$ levels
(Table~\ref{tab:ar_1s_n2_quenching}) and the four measured $2\mathrm{p}$ levels
(Table~\ref{tab:ar_n2_quenching}) are quenched by N$_2$, and the Ar ground state is diluted; a
dissociated fraction of the N$_2$ feed can be set, which then neither quenches nor counts as N$_2$.

\section{Fitting the emission spectra}
Argon lines in 400--475~nm ($5\mathrm{p}\rightarrow4\mathrm{s}$) and 690--860~nm
($4\mathrm{p}\rightarrow4\mathrm{s}$) that are cleanly fitted, free of N$_2$ bands, rated reliable
and away from the echelle order edges are integrated in every spectrum; lines closer than 0.1~nm form
one feature. The model intensity of a feature is $\sum n_kA_{ki}\Lambda_{ki}/\lambda$ (calibrated
radiance). A CR table is computed on a grid of EEDF ($E/N$ or $T_e$) and $N_e=10^{15}$--$3\times10^{19}$
m$^{-3}$; at each grid point the measured features are compared with the model through
\begin{equation}
\chi^2=\sum_{s,f}\frac{\big[\ln I^{\mathrm{meas}}_{sf}-\ln I^{\mathrm{mod}}_f-c_s-\beta(\lambda_f-\lambda_0)\big]^2}
{\sigma_{sf}^2+\sigma_\mathrm{model}^2},
\end{equation}
with a free scale $c_s$ per spectrum, a linear spectral-response tilt $\beta$, the measurement error
$\sigma_{sf}$ (fit error, a 2\,\% floor and the repeat-to-repeat scatter) and a model error
$\sigma_\mathrm{model}=20\,\%$. The posterior $\propto e^{-\chi^2/2}$ on a fine grid gives
$T_{e,\mathrm{eff}}$ and $N_e$ with their uncertainties (widened by the Birge ratio when
$\chi^2/\mathrm{dof}>1$). Because the line ratios alone push $N_e$ to the grid edge (Section~\ref{sec:status}),
$N_e$ can be pinned at the critical density of the microwave field,
$n_c=\varepsilon_0m_e\omega^2/e^2$ ($7.3\times10^{16}$~m$^{-3}$ at 2.42~GHz in the pressure sweep,
$7.45\times10^{16}$~m$^{-3}$ at 2.45~GHz in the actinometry, as their EEDF libraries), and only $E/N$
is fitted.

\section{Actinometry}
For an N\,I line and an Ar\,I reference line excited from the ground state,
\begin{equation}
\frac{n_\mathrm{N}}{n_\mathrm{Ar}}=\frac{I_\mathrm{N}\lambda_\mathrm{N}}{I_\mathrm{Ar}\lambda_\mathrm{Ar}}\,
\frac{k_\mathrm{Ar}b_\mathrm{Ar}}{k_\mathrm{N}b_\mathrm{N}},
\end{equation}
with branching ratios $b=A_\mathrm{line}/\sum A$ and the wavelengths converting calibrated radiance to
photon rates. In the stepwise variant, $k_\mathrm{Ar}$ is replaced by $k_\mathrm{Ar}/f_\mathrm{direct}$,
where $f_\mathrm{direct}$ is the share of direct excitation in the CR model's production of the Ar
level. Each N/Ar line pair is evaluated over the fit posterior; per condition, pairs whose
$\ln(n_\mathrm{N}/n_\mathrm{Ar})$ lies more than 1.5 interquartile ranges beyond the quartiles of
all pairs are rejected (Tukey), and the rest are combined with weights from their line-ratio errors.
The dissociation degree is $D=n_\mathrm{N}/(2n_{\mathrm{N}_2})=(n_\mathrm{N}/n_\mathrm{Ar})(1-x)/(2x)$
for an N$_2$ feed fraction $x$.

\section{Status (<<DATE>>)}\label{sec:status}
\begin{itemize}
\item Against the literature, the model reproduces the measured $2\mathrm{p}$ distribution of
Zhu and Pu's 100~Pa CCP (mean deviation 1.3 on a sum-100 scale) and the level patterns of
Bogaerts \emph{et al.} within about 2. At 100~Pa its $1\mathrm{s}$ densities are 2--5 times lower than
Zhu and Pu's, traced to the resonance escape factor.
\item With free $N_e$, the fits of the pure-Ar pressure sweep run to the upper $N_e$ grid edge for
both the DC and the microwave EEDF, with no interior minimum of $\chi^2(N_e)$; pinning $N_e=n_c$ costs
$\Delta\chi^2/s^2\approx20$--45. The preference comes from the primed-core $2\mathrm{p}$ lines
(706.7, 727.3, 750.4, 794.8~nm): without them the microwave fit accepts $n_c$. 794.8~nm lies at the
start of an echelle order and reads 1.45 times high against 852.1~nm from the same level.
\item Freezing the escape factors at their $n_c$ values leaves this unchanged, so trapping is not
the cause. The unfitted 772.38/772.42~nm ($2\mathrm{p}_7/2\mathrm{p}_2$) ratio agrees with the model at
$n_c$ within 4\,\% and differs by a factor 1.46 at the free-fit $N_e$.
\item In actinometry, the two $5\mathrm{p}$ reference lines (415.9, 430.0~nm) disagree with the seven
$2\mathrm{p}$ lines by about two orders of magnitude and are rejected as outliers; their cross
sections (NGFSRDW, starting 5--6~eV above threshold) are the weakest part of the data.
\end{itemize}

<<RESULTS>>

\clearpage
\section{Tables}
\input{ArgonReactionTables.tex}

\end{document}
"""


def description(MD, counts):
    import collections
    import datetime
    man = collections.Counter(v['manifold'] for v in MD.values() if v['kind'] != 'ground')
    manifolds = ', '.join(rf'{n} ${m[0]}\mathrm{{{m[1:]}}}$' if re.match(r'\d[spdf]$', m) else f'{n} {m}'
                          for m, n in man.items())
    rep = {'DATE': datetime.date.today().isoformat(), 'NEXC': str(sum(man.values())), 'MANIFOLDS': manifolds,
           'NRAD': str(counts['radiative']), 'NBSR': str(counts['bsr']), 'NNGF': str(counts['ngfsrdw']),
           'NALLOW': str(counts['allowed']), 'NFORB': str(counts['forbidden']), 'TEEEPF': f'{TE_EEPF:g}',
           'BOLSIGTABLE': _tabular('l' + 'r' * len(EN_TABLE), ['EEDF'] + [f'{x:g}' for x in EN_TABLE],
                                   bolsig_results()),
           'RESULTS': results_section()}
    tex = DESCRIPTION
    for k, v in rep.items():
        tex = tex.replace(f'<<{k}>>', v)
    return tex


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        ModelData, _ = he.GetData()
    tex, counts = build(ModelData)
    path = os.path.join(OUTPUT_DIR, 'ArgonReactionTables.tex')
    with open(path, 'w', encoding='utf-8') as file:
        file.write(tex)
    print(f'Wrote {path}')
    print('  ' + ', '.join(f'{k}: {v}' for k, v in counts.items()))
    path = os.path.join(OUTPUT_DIR, 'ArgonCRModel.tex')
    with open(path, 'w', encoding='utf-8') as file:
        file.write(description(ModelData, counts))
    print(f'Wrote {path} (compile with pdflatex in {OUTPUT_DIR})')
