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
import contextlib
import io
import numpy as np
import HelperFunctions as he
from ParserforLXCatData import PASCHEN_TO_LABEL

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

E_ION = 15.7596     # eV, Ar first ionization energy (as in he.CreateIonizationCrossSections)
K_MET = 6.4e-16     # m^3/s, metastable-metastable collisions (SolveLabelEquation in MainFileV2)
SIGMA_CAP = 1.0e-18 # m^2, cap on analytic cross sections (he.AddAnalyticExcitationCrossSections)

# Numbered in order of first use in the tables
REFERENCES = {
    'NIST': r'A.~Kramida, Yu.~Ralchenko, J.~Reader and NIST ASD Team, \emph{NIST Atomic Spectra '
            r'Database} (level energies, statistical weights, $A_{ki}$ and accuracy grades).',
    'NGFSRDW': r'NGFSRDW database (relativistic distorted-wave cross sections), '
               r'\texttt{www.lxcat.net/NGFSRDW}, retrieved 1 July 2026.',
    'Khakoo04': r'M.~A.~Khakoo \emph{et al.}, J.~Phys.~B \textbf{37}, 247 (2004).',
    'Kaur98': r'S.~Kaur, R.~Srivastava, R.~P.~McEachran and A.~D.~Stauffer, '
              r'J.~Phys.~B \textbf{31}, 4833 (1998).',
    'Gangwar10': r'R.~K.~Gangwar, L.~Sharma, R.~Srivastava and A.~D.~Stauffer, '
                 r'Phys.~Rev.~A \textbf{81}, 052707 (2010).',
    'Srivastava06': r'R.~Srivastava, A.~D.~Stauffer and L.~Sharma, Phys.~Rev.~A \textbf{74}, 012715 (2006).',
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
    'MultiBolt': r'M.~Flynn \emph{et al.}, J.~Phys.~D: Appl.~Phys. \textbf{55}, 015201 (2022) '
                 r'(MultiBolt, used for numerical EEDFs).',
    'Biagi': r'S.~F.~Biagi, Magboltz 8.97 cross sections (Biagi database on LXCat), '
             r'input to MultiBolt for the EEDF.',
}
REF_NUM = {key: i + 1 for i, key in enumerate(REFERENCES)}


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
    inverse = {v: k for k, v in PASCHEN_TO_LABEL.items()}
    p = inverse.get(label)
    if p is None:
        return '--'
    m = re.match(r'(\d)([spd])(\d+)$', p)
    return rf'${m.group(1)}{m.group(2)}_{{{m.group(3)}}}$'


def designation(level):
    """Racah notation, e.g. 4s[3/2]o_2; a prime marks the 2P_1/2 core."""
    if level['kind'] == 'ground':
        return r'$3p^6\;{}^1S_0$'
    cfg, term = level['configuration'], level['term']
    nl = cfg.split('.')[-1]
    prime = "'" if '<1/2>' in cfg else ''
    K = re.search(r'\[(.+?)\]', term).group(1)
    parity = r'^{\circ}' if term.endswith('*') else ''
    return rf'$\mathrm{{{nl}}}{prime}[{K}]{parity}_{{{int(level["J"])}}}$'


def longtable(colspec, header, rows, caption, label):
    """rows: lists of cell strings, or the string 'midrule' for a separator."""
    ncols = len(header)
    head = ' & '.join(header) + r' \\'
    out = [r'\begingroup\small',
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
        return {'4s': 'Khakoo04', '4p': 'Kaur98'}.get(mup, 'Gangwar10')   # 3d, 5s
    if mlo == '4s' and mup == '4p':
        return 'Srivastava06' if MD[lo]['kind'] == 'metastable' else 'Gangwar12'
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
    no_decay = [v['label'] for v in levels if v['kind'] not in ('ground', 'metastable')
                and not any(t['upper_label'] == v['label'] for t in lines)]

    n_ion = sum(1 for v in levels if v['kind'] != 'ground')
    ND_s3, ND_s5 = (D * he.Torr2Volume(1, 300) for D in he.FindDiffusionCoeff(1, 300))
    n_forb = sum(len(v) for v in forbidden.values())
    n_exc = len(rdw) + len(allowed) + n_forb

    tex = [r'% Generated by Scripts/MakeReactionTables.py from the CR-model input data - do not edit by hand.',
           r'% Requires \usepackage{booktabs,longtable}.', '']

    # ---- 1. levels --------------------------------------------------------
    rows = [[v['label'], paschen(v['label']), designation(v), f"{int(v['J'])}", f"{int(v['g'])}",
             f"{v['energy_eV']:.4f}", kind_name[v['kind']],
             '--' if v['kind'] == 'ground' else f"{E_ION - v['energy_eV']:.3f}"] for v in levels]
    tex.append(longtable(
        'llllrrlr',
        ['Label', 'Paschen', 'Designation', '$J$', '$g$', '$E$ (eV)', 'Type', r'$E_\mathrm{ion}-E$ (eV)'],
        rows,
        rf'Argon levels in the CR model ({len(levels)} levels). Energies and statistical weights from '
        rf'NIST {cite("NIST")}; designations in Racah notation, where a prime marks the '
        r'$^2P^\circ_{1/2}$ ion core. For the 3d levels the Paschen column gives the NGFSRDW '
        r'energy index ($3d_{12}$ lowest), not Paschen notation. $E_\mathrm{ion}-E$ is the threshold for electron-impact '
        rf'ionization out of the level ($E_\mathrm{{ion}} = {E_ION}$~eV).',
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
         r'(not the ground state)', cite('Vriens80')],
        ['Spontaneous emission', r'$\mathrm{Ar}_k\rightarrow\mathrm{Ar}_i+h\nu$', f'{len(lines)}',
         r'$A_{ki}\Lambda_{ki}$', cite('NIST')],
        ['Radiation trapping', 'all emission lines', f'{len(lines)}',
         r'Escape factor $\Lambda(\tau_0,a)$ from a Monte Carlo hemisphere calculation, Voigt '
         r'profile (Doppler and resonance broadening), lower level as absorber',
         'this work, ' + cite('BK97')],
        ['Metastable diffusion', r'$\mathrm{Ar}(1s_5,1s_3)\rightarrow$ wall', '2',
         rf'$\tau_D^{{-1}}=D\,(4.493/R)^2$, $ND={sci(ND_s3, 1, False)}$ and '
         rf'${sci(ND_s5, 1, False)}$~m$^{{-1}}$\,s$^{{-1}}$', cite('SAB07')],
        ['Metastable--metastable collisions', r'$\mathrm{Ar}^m+\mathrm{Ar}^m\rightarrow$ products', '1',
         rf'$k={sci(K_MET, 1, False)}$~m$^3$\,s$^{{-1}}$', cite('Ferreira85')],
    ]
    tex.append(longtable(
        'p{3.1cm}p{3.3cm}rp{5.2cm}p{1.9cm}',
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
        rows.append([lo, up, f'{thr:.4f}', cite(ref)])
        prev = lo
    tex.append(longtable(
        'llrl', ['Lower', 'Upper', r'$\varepsilon_\mathrm{th}$ (eV)', 'Source'], rows,
        rf'Electron-impact excitation channels with relativistic distorted-wave cross sections from '
        rf'the NGFSRDW database {cite("NGFSRDW")} ({len(rdw)} channels; thresholds as tabulated). '
        r'Each channel is also included as its superelastic reverse process.',
        'tab:ar_rdw'))

    # ---- 4. analytic, optically allowed ------------------------------------
    rows = [[lo, up + (r'$^*$' if capped else ''), f'{thr:.4f}', sci(f, 2)]
            for lo, up, thr, f, capped in allowed]
    tex.append(longtable(
        'llrr', ['Lower', 'Upper', r'$\Delta E$ (eV)', r'$f_{ij}$'], rows,
        rf'Optically allowed excitation channels without RDW data ({len(allowed)} channels): Drawin '
        rf'cross section $\sigma=4\pi a_0^2(E_H/\Delta E)^2 f_{{ij}}\,\alpha\,(u-1)u^{{-2}}\ln(1.25\beta u)$, '
        rf'$u=\varepsilon/\Delta E$, $\alpha=\beta=1$ {cite("Bogaerts98")}, with the oscillator strength '
        rf'$f_{{ij}}$ from the NIST $A_{{ki}}$ {cite("NIST")}. $^*$Peak cross section capped at '
        rf'{sci(SIGMA_CAP, 0)}~m$^2$. Each channel is also included as its superelastic reverse process.',
        'tab:ar_allowed'))

    # ---- 5. analytic, forbidden ------------------------------------------
    rows = [[lo, str(len(forbidden[lo])), ', '.join(sorted(forbidden[lo], key=by_energy.get))]
            for lo in sorted(forbidden, key=lambda l: by_energy[l])]
    tex.append(longtable(
        'lrp{11.2cm}', ['Lower', '$N$', 'Upper levels'], rows,
        rf'Optically forbidden excitation channels ({n_forb} channels): all remaining pairs of levels '
        r'without RDW data or a radiative transition, with the Drawin forbidden cross section '
        rf'$\sigma=4\pi a_0^2\,\alpha\,(u-1)u^{{-2}}$, $\alpha=0.01$ {cite("Bogaerts98")}. Each channel '
        r'is also included as its superelastic reverse process.',
        'tab:ar_forbidden'))

    # ---- 6. radiative transitions ------------------------------------------
    rows, prev = [], None
    for t in lines:
        up, lo = t['upper_label'], t['lower_label']
        if prev is not None and up != prev:
            rows.append('midrule')
        rows.append([up if up != prev else '', lo + (r'$^\dagger$' if lo == 'ground' else ''),
                     f"{t['wl_nm']:.3f}", sci(t['Aki'], 2), t['acc'] or '--'])
        prev = up
    tex.append(longtable(
        'llrrl', ['Upper', 'Lower', r'$\lambda$ (nm)', r'$A_{ki}$ (s$^{-1}$)', 'Acc.'], rows,
        rf'Spontaneous emission lines in the CR model ({len(lines)} lines), from NIST {cite("NIST")}; '
        r'$\lambda$ in vacuum below 200~nm and in air above. Accuracy of $A_{ki}$: AA $\le1\%$, '
        r'A$+$ $\le2\%$, A $\le3\%$, B$+$ $\le7\%$, B $\le10\%$, C$+$ $\le18\%$, C $\le25\%$, '
        r'D$+$ $\le40\%$, D $\le50\%$, E $>50\%$. Every line is reduced by an escape factor '
        r'$\Lambda$ computed with the lower-level density as absorber. $^\dagger$Resonance line to '
        r'the ground state (VUV). The metastables 4s1 and 4s3 have no radiative decay'
        + (', and ' + ', '.join(no_decay) + ' have no tabulated decay to a modelled level' if no_decay else '')
        + '.',
        'tab:ar_radiative'))

    # ---- 7. references -----------------------------------------------------
    rows = [[f'[{n}]', text] for text, n in ((REFERENCES[k], REF_NUM[k]) for k in REFERENCES)]
    tex.append(longtable('lp{14cm}', ['', 'Source'], rows,
                         'Data sources for Tables~\\ref{tab:ar_levels}--\\ref{tab:ar_radiative}.',
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
