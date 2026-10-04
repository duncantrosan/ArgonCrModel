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
    'MultiBolt': r'M.~Flynn \emph{et al.}, J.~Phys.~D: Appl.~Phys. \textbf{55}, 015201 (2022) '
                 r'(MultiBolt, used for numerical EEDFs).',
    'Biagi': r'S.~F.~Biagi, Magboltz 8.97 cross sections (Biagi database on LXCat), '
             r'input to MultiBolt for the EEDF.',
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
                rdw.append((lo, up, r['threshold_eV'], rdw_reference(MD, lo, up)))
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
        ['Electron-impact excitation, RDW', R_exc, f'{len(rdw)}',
         r'$\sigma(\varepsilon)$ integrated over the EEDF',
         cite('NGFSRDW', 'Khakoo04', 'Kaur98', 'Gangwar10', 'Srivastava06', 'Gangwar12')],
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
         r'Escape factor $\Lambda(\tau_0,a)$ from a Monte Carlo hemisphere calculation, Voigt '
         r'profile (Doppler and resonance broadening), lower level as absorber',
         'this work, ' + cite('BK97')],
        ['Metastable diffusion', r'$\mathrm{Ar}(1\mathrm{s}_5,1\mathrm{s}_3)\rightarrow$ wall', '2',
         rf'$\tau_D^{{-1}}=D\,(4.493/R)^2$, $ND={sci(ND_s5, 1, False)}$ and '
         rf'${sci(ND_s3, 1, False)}$~m$^{{-1}}$\,s$^{{-1}}$', cite('SAB07')],
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
        r'$k=\sqrt{2e/m_e}\int\sqrt{\varepsilon}\,\sigma(\varepsilon)F(\varepsilon)\,d\varepsilon$, '
        r'with $F$ either a Maxwellian at $T_e$ or a numerical EEDF from MultiBolt '
        + cite('MultiBolt') + ' computed with the Biagi cross sections ' + cite('Biagi') + '.',
        'tab:ar_processes'))

    # ---- 3. RDW excitation ------------------------------------------------
    rows, prev = [], None
    for lo, up, thr, ref in rdw:
        if prev is not None and MD[lo]['manifold'] != MD[prev]['manifold']:
            rows.append('midrule')
        rows.append([D[lo], D[up], f'{thr:.4f}', cite(ref)])
        prev = lo
    tex.append(longtable(
        'llrl', ['Lower', 'Upper', r'$\varepsilon_\mathrm{th}$ (eV)', 'Source'], rows,
        rf'Electron-impact excitation channels with relativistic distorted-wave cross sections from '
        rf'the NGFSRDW database {cite("NGFSRDW")} ({len(rdw)} channels; thresholds as tabulated). '
        r'Each channel is also included as its superelastic reverse process.',
        'tab:ar_rdw'))

    # ---- 4. analytic, optically allowed ------------------------------------
    rows = [[D[lo], D[up] + (r'\textsuperscript{*}' if capped else ''), f'{thr:.4f}', sci(f, 2)]
            for lo, up, thr, f, capped in allowed]
    tex.append(longtable(
        'llrr', ['Lower', 'Upper', r'$\Delta E$ (eV)', r'$f_{ij}$'], rows,
        rf'Optically allowed excitation channels without RDW data ({len(allowed)} channels): Drawin '
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
        r'without RDW data or a radiative transition, with the Drawin forbidden cross section '
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

    # ---- 7. quenching of the 2p levels by N2 ---------------------------------
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
        r'Not yet in the CR model, which quenches only the $4\mathrm{s}$ levels by N$_2$.',
        'tab:ar_n2_quenching'))

    # ---- 8. references -----------------------------------------------------
    rows = [[f'[{n}]', text] for text, n in ((REFERENCES[k], REF_NUM[k]) for k in REFERENCES)]
    tex.append(longtable(r'lp{0.9\linewidth}', ['', 'Source'], rows,
                         'Data sources for Tables~\\ref{tab:ar_levels}--\\ref{tab:ar_n2_quenching}.',
                         'tab:ar_sources'))
    return '\n'.join(tex), dict(levels=len(levels), rdw=len(rdw), allowed=len(allowed),
                                forbidden=n_forb, ionization=n_ion, radiative=len(lines))


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        ModelData, _ = he.GetData()
    tex, counts = build(ModelData)
    path = os.path.join(OUTPUT_DIR, 'ArgonReactionTables.tex')
    with open(path, 'w', encoding='utf-8') as file:
        file.write(tex)
    print(f'Wrote {path}')
    print('  ' + ', '.join(f'{k}: {v}' for k, v in counts.items()))
