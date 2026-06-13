"""
Generate the St. Yau Research Outline DOCX.
Matches the 2026 S.T. Yau High School Science Award (Asia) template format.
Run: python abstract/make_docx.py
Output: abstract/Research_Outline_Aarush_Gupta.docx
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

doc = Document()

# ── Page margins ──────────────────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin   = Cm(2.54)
    section.right_margin  = Cm(2.54)

# ── Styles ────────────────────────────────────────────────────────────────────
normal = doc.styles['Normal']
normal.font.name = 'Times New Roman'
normal.font.size = Pt(11)

def para(text='', bold=False, italic=False, size=11, align=WD_ALIGN_PARAGRAPH.LEFT,
         space_before=0, space_after=6, keep_with_next=False, color=None):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after  = Pt(space_after)
    p.paragraph_format.keep_with_next = keep_with_next
    if text:
        run = p.add_run(text)
        run.bold   = bold
        run.italic = italic
        run.font.name = 'Times New Roman'
        run.font.size = Pt(size)
        if color:
            run.font.color.rgb = RGBColor(*color)
    return p

def add_run(p, text, bold=False, italic=False, size=11):
    run = p.add_run(text)
    run.bold   = bold
    run.italic = italic
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size)
    return run

def heading(text, level=1):
    """Bold heading that matches the outline style."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after  = Pt(4)
    run = p.add_run(text)
    run.bold = True
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)
    return p

def bullet(text, level=1, italic_part=None):
    """Bullet point. italic_part = substring to italicise (first occurrence)."""
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(2)
    p.paragraph_format.left_indent  = Inches(0.25 * level)
    if italic_part and italic_part in text:
        idx = text.index(italic_part)
        if idx > 0:
            r = p.add_run(text[:idx])
            r.font.name = 'Times New Roman'; r.font.size = Pt(11)
        r2 = p.add_run(italic_part)
        r2.italic = True; r2.font.name = 'Times New Roman'; r2.font.size = Pt(11)
        rest = text[idx + len(italic_part):]
        if rest:
            r3 = p.add_run(rest)
            r3.font.name = 'Times New Roman'; r3.font.size = Pt(11)
    else:
        r = p.add_run(text)
        r.font.name = 'Times New Roman'; r.font.size = Pt(11)
    return p

def set_cell_border(cell, **kwargs):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for edge in ('top','left','bottom','right','insideH','insideV'):
        val = kwargs.get(edge, 'single')
        sz  = kwargs.get(f'{edge}_sz', 4)
        tag = OxmlElement(f'w:{edge}')
        tag.set(qn('w:val'),   val)
        tag.set(qn('w:sz'),    str(sz))
        tag.set(qn('w:space'), '0')
        tag.set(qn('w:color'), '000000')
        tcBorders.append(tag)
    tcPr.append(tcBorders)

def cell_text(cell, text, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT, size=10):
    cell.text = ''
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after  = Pt(1)
    run = p.add_run(text)
    run.bold = bold
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER (manual — Word's header/footer would require section XML; simpler
# to put it as a styled first paragraph)
# ══════════════════════════════════════════════════════════════════════════════

# Add actual header using the header section
header = doc.sections[0].header
hdr_para = header.paragraphs[0]
hdr_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
hdr_run = hdr_para.add_run('Research Outline                    2026 S.T. Yau High School Science Award (Asia)')
hdr_run.font.name = 'Times New Roman'
hdr_run.font.size = Pt(10)

# Footer with page number
footer = doc.sections[0].footer
ftr_para = footer.paragraphs[0]
ftr_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
ftr_run = ftr_para.add_run('Page ')
ftr_run.font.name = 'Times New Roman'
ftr_run.font.size = Pt(10)
fldChar1 = OxmlElement('w:fldChar')
fldChar1.set(qn('w:fldCharType'), 'begin')
instrText = OxmlElement('w:instrText')
instrText.text = 'PAGE'
fldChar2 = OxmlElement('w:fldChar')
fldChar2.set(qn('w:fldCharType'), 'end')
ftr_para.runs[-1]._r.append(fldChar1)
ftr_para.runs[-1]._r.append(instrText)
ftr_para.runs[-1]._r.append(fldChar2)
ftr_para.add_run(' of 4').font.size = Pt(10)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE BLOCK
# ══════════════════════════════════════════════════════════════════════════════

para('2026 S.T. Yau High School Science Award (Asia)',
     bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER,
     space_before=0, space_after=0)
para('Research Outline',
     bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER,
     space_before=0, space_after=6)

# ── Registration table ────────────────────────────────────────────────────────
tbl = doc.add_table(rows=5, cols=4)
tbl.style = 'Table Grid'
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

# Header row — Team Member ×3
for i, col in enumerate(tbl.columns):
    if i == 0:
        cell_text(tbl.cell(0, 0), '', bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    else:
        cell_text(tbl.cell(0, i), 'Team Member', bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

labels = ['Name', 'School', 'City, Country', 'Registration No.']
values = ['Aarush Gupta', 'UWCSEA, East', 'Singapore',
          '[Physics-0XX — assigned at registration]']

for r, (lbl, val) in enumerate(zip(labels, values)):
    cell_text(tbl.cell(r+1, 0), lbl)
    cell_text(tbl.cell(r+1, 1), val)
    cell_text(tbl.cell(r+1, 2), '')
    cell_text(tbl.cell(r+1, 3), '')

# Merge cols 1–3 for the value cells
for r in range(1, 5):
    tbl.cell(r, 1).merge(tbl.cell(r, 3))

doc.add_paragraph()  # spacer

# ── Supervising Teacher table ─────────────────────────────────────────────────
tbl2 = doc.add_table(rows=5, cols=2)
tbl2.style = 'Table Grid'
tbl2.alignment = WD_TABLE_ALIGNMENT.CENTER

# Merge top row for the "Supervising Teacher" heading
tbl2.cell(0, 0).merge(tbl2.cell(0, 1))
cell_text(tbl2.cell(0, 0), 'Supervising Teacher', bold=True,
          align=WD_ALIGN_PARAGRAPH.CENTER)

sup_labels = ['Name', 'Position', 'School/Institution', 'City, Country']
sup_values = ['[Name]', '[Position]', '[School / College]', '[City, Country]']
for r, (lbl, val) in enumerate(zip(sup_labels, sup_values)):
    cell_text(tbl2.cell(r+1, 0), lbl)
    cell_text(tbl2.cell(r+1, 1), val)

doc.add_paragraph()  # spacer

# ── Paper title ───────────────────────────────────────────────────────────────
para('Structure over Resolution: an Operating Envelope for Stable Integration of the '
     'Chaotic Three-Body Problem',
     bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER,
     space_before=6, space_after=10)

# ══════════════════════════════════════════════════════════════════════════════
# ABSTRACT
# ══════════════════════════════════════════════════════════════════════════════

heading('Abstract')

abstract_body = [
    ("The Newtonian three-body problem has no general closed-form solution and is a prototype of "
     "deterministic chaos, so every quantitative statement about such a system is obtained numerically — "
     "which makes both the integration scheme and the yardstick by which it is judged decisive. "
     "Two difficulties are usually left implicit. Because neighbouring trajectories diverge exponentially, "
     "the customary measure — global position error against a reference — saturates within a few Lyapunov "
     "times and cannot rank methods over long time scales. And neural network models proposed to accelerate "
     "integration are seldom tested at equal computational cost against the obvious alternative of taking "
     "a finer step."),
    ("I compare five integrator methods and two neural network correctors across three test systems — "
     "analytic three-body configurations of varying chaoticity, binary–single scattering encounters, and "
     "the real Sun–Earth–Moon system initialised from JPL Horizons ephemerides — using chaos-appropriate "
     "diagnostics: maximum relative energy drift, short-horizon position RMS within one Lyapunov time, "
     "bounded/ejected status, and — critically for hierarchical systems — the preserved separation of "
     "bound pairs."),
]
for body in abstract_body:
    p = para(space_before=0, space_after=4)
    add_run(p, body)

# Three findings with italic lead-ins
findings = [
    ("First, ", "time-reversal symmetry, not brute-force resolution, governs long-term energy drift",
     ": the reversible adaptive leapfrog (η = 0.05) stays energy-bounded across all dynamical "
     "regimes at a single control setting, whereas a fixed step using eight times as many force "
     "evaluations ejects the same close-encounter system at 128% energy error. The efficiency "
     "advantage is regime-dependent — the scheme is 1.5–3.6× cheaper in force evaluations on "
     "close-encounter and under-resolved systems but slower on near-regular orbits — so the "
     "contribution is an operating map, not a universal speedup."),
    ("Second, ", "energy conservation can actively mislead",
     ": a fixed-step Sun–Earth–Moon integration conserves total energy to max|ΔE/E₀| = 0.008% "
     "and reports the system as bound, yet the Earth–Moon separation grows to 774× its true "
     "value — a destroyed hierarchy that only a pair-separation metric detects."),
    ("Third, ", "neural network force and state corrections do not beat physics at equal cost",
     ": under a pre-registered equal-compute protocol on 27 held-out configurations, a per-step "
     "scalar corrector (A1) beats an equal-cost finer leapfrog step on short-horizon position "
     "RMS in 0 of 27 cases — even when charged no computational cost — and an encounter-triggered "
     "state corrector (A2) in at most 2 of 27. An oracle (best-possible) correction yields 0% "
     "improvement on under-resolved binaries at every separation tested, localising the failure "
     "to temporal resolution, not force accuracy."),
]
for prefix, italic, suffix in findings:
    p = para(space_before=0, space_after=4)
    add_run(p, prefix)
    add_run(p, italic, italic=True)
    add_run(p, suffix)

p = para(space_before=0, space_after=6)
add_run(p, "Together these results map an ")
add_run(p, "operating envelope", italic=True)
add_run(p, " — which integrator is necessary and sufficient in each dynamical regime — and "
           "establish a selection of stability diagnostics for chaotic three-body work: stability "
           "must be judged on preserved dynamical structure, not energy alone.")

# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════

heading('Research questions')

rqs = [
    ("Q1 (structure). ",
     "Which structural property — symplecticity, time-reversal symmetry, or adaptive step-sizing — "
     "governs stable integration, and does the answer depend on the dynamical ",
     "regime",
     " (the chaoticity and hierarchical structure of the initial conditions: regular, moderate "
     "scattering, or tight binary)?"),
    ("Q2 (measurement). ",
     "Which diagnostics are sufficient to classify integrator ",
     "stability",
     " — that is, to determine whether a numerical solution preserves the physical structure of "
     "the three-body system — when long-time position error is uninformative due to chaos?"),
    ("Q3 (learning). ",
     "Can a neural network force or state correction outperform spending the same compute on a "
     "finer classical step?",
     None, None),
]
for item in rqs:
    p = doc.add_paragraph(style='List Number')
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(3)
    label, body, italic_word, suffix = item if len(item) == 4 else (*item, None, None)
    add_run(p, label, bold=True)
    if italic_word:
        add_run(p, body)
        add_run(p, italic_word, italic=True)
        add_run(p, suffix)
    else:
        add_run(p, body)

# ══════════════════════════════════════════════════════════════════════════════
# METHODS
# ══════════════════════════════════════════════════════════════════════════════

heading('Methods')

heading_sub = lambda t: para(t, bold=True, space_before=4, space_after=2)

heading_sub('Integrator methods — benchmarked against IAS15 (fifteenth-order Gauss–Radau, REBOUND [3,4])')

methods = [
    'Fixed-step leapfrog (dt ∈ {0.04, 0.005} yr; 25 and 200 force evaluations/yr).',
    ('Heuristic inverse-distance adaptive leapfrog: step size scales as the cube of the minimum '
     'inter-body separation. This is the SIMON integrator without its neural network component.'),
    ('Time-symmetric (reversible) adaptive leapfrog [Hut, Makino & McMillan, 1995 [6]]: step size '
     'is found by fixed-point iteration to satisfy a time-reversibility condition, bounding secular '
     'energy drift. Tuning parameter η = 0.05 is held fixed across all regimes.'),
    'Fourth-order symplectic integrator [Yoshida, 1990 [2]]: a composition of leapfrog steps preserving symplectic structure.',
    ('IAS15 [3,4]: fifteenth-order Gauss–Radau adaptive reference scheme; used as the '
     'high-accuracy standard throughout.'),
]
for m in methods:
    bullet(m)

heading_sub('Neural network correctors — applied to the leapfrog baseline')
bullet('A1 — per-step scalar force-magnitude correction: outputs a scalar multiplier on the '
       'gravitational force at every leapfrog step, trained to match IAS15 forces.')
bullet('A2 — encounter-triggered state-to-state residual correction: activated only at close '
       'approach, predicts the residual correction to the integrated state; fires ~once per '
       '100-yr rollout on the test configurations (not a per-step corrector).')
p = para(space_before=2, space_after=4)
add_run(p, 'Each corrector is evaluated under a ')
add_run(p, 'pre-registered equal-compute protocol', bold=True)
add_run(p, ': the network is charged a cost κ (in units of one force evaluation) and is '
           'deemed to "click" only if it falls below the plain-leapfrog cost–accuracy frontier '
           'on held-out configurations not seen during training.')

heading_sub('Test systems (all integrated over T = 100 yr)')
bullet('Analytic three-body initial conditions spanning Lyapunov exponents: near-regular '
       '(IC6, λ ≈ 0.019 yr⁻¹), moderate scattering (IC1, λ ≈ 0.17 yr⁻¹; IC4, λ ≈ 0.10 yr⁻¹).')
bullet('Binary–single scattering: inner binary separation r_bin ∈ {0.05, 0.10, 0.20, 0.40} AU, '
       'perturber at 3 AU.')
bullet('Real Sun–Earth–Moon system initialised from JPL Horizons ephemerides [7].')

heading_sub('Stability diagnostics — chaos-appropriate, reported together')
diags = [
    'Maximum relative energy drift, max|ΔE/E₀|, over T = 100 yr.',
    'Short-horizon position RMS within ~1 Lyapunov time.',
    'Time-to-divergence (time at which position error exceeds a fixed threshold).',
    'Bounded vs ejected status.',
    ('Range of bound-pair separation [AU] — required for hierarchical systems; detects '
     'sub-system unbinding invisible to global energy metrics.'),
]
for d in diags:
    bullet(d)

p = para(space_before=2, space_after=6)
add_run(p, 'Method efficiency (force evaluations to reach equal accuracy) is compared only at matched '
           'accuracy, by interpolating each method\'s Pareto cost–accuracy frontier within the measured '
           'range. Wall-clock speed is noted only as a secondary observation.')

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

heading('Results')

# Finding 1
p = para(space_before=4, space_after=2)
add_run(p, 'Finding 1 — time-reversal symmetry governs energy drift', bold=True)
add_run(p, ' (answers Q1).')
body1 = (
    'The reversible adaptive leapfrog (η = 0.05) stays energy-bounded across all test '
    'systems at a single setting: max|ΔE/E₀| = 0.087%, 0.224%, and 0.003% on IC1, IC4, '
    'and IC6 respectively. A fixed step with dt = 0.005 (eight times more force evaluations '
    'per year) ejects the IC1 close-encounter system at 128% energy error. The heuristic '
    'adaptive leapfrog also fails on close encounters (IC1: 1.24%; BS_0.05: 4905%, ejected); '
    'only time-symmetric adaptation and IAS15 remain bounded across all regimes.'
)
para(body1, space_before=0, space_after=3)
body1b = (
    'The efficiency advantage is regime-dependent. On the energy metric, the time-symmetric '
    'scheme is 1.5–3.6× cheaper in force evaluations on close-encounter (IC1) and '
    'under-resolved (IC4) configurations. On near-regular IC3/IC6 the fixed step is already '
    'cheap and the time-symmetric scheme uses 2–3× more evaluations for no accuracy gain. '
    'The contribution is an operating map, not a universal speedup (Figure 1: cost vs '
    'energy drift across all 9 configurations).'
)
para(body1b, space_before=0, space_after=6)

# Finding 2
p = para(space_before=4, space_after=2)
add_run(p, 'Finding 2 — the energy-metric trap', bold=True)
add_run(p, ' (answers Q2, first part).')
body2 = (
    'The real Sun–Earth–Moon system integrated with leapfrog at dt = 0.01 yr reports '
    'max|ΔE/E₀| = 0.008% and bounded = True. Yet over 100 yr the Earth–Moon separation '
    'grows from its true value of 0.00257 AU to a peak of 1.99 AU — 774× the true distance '
    '— while IAS15, the time-symmetric scheme, and the heuristic-adaptive integrator all '
    'hold it within [0.0024, 0.0027] AU. A global energy or ejection criterion thus certifies '
    'a run whose internal hierarchical structure is destroyed. Only the pair-separation '
    'diagnostic detects the failure (Figure 2).'
)
para(body2, space_before=0, space_after=6)

# Finding 3
p = para(space_before=4, space_after=2)
add_run(p, 'Finding 3 — neural network corrections are null at equal compute', bold=True)
add_run(p, ' (answers Q3).')
body3a = 'Under the pre-registered equal-compute protocol on 27 held-out configurations:'
para(body3a, space_before=0, space_after=2)
bullet('A1 (per-step scalar): 0 of 27 configurations beat an equal-cost finer leapfrog step '
       'on short-horizon position RMS — even when the network is charged zero cost (κ = 0).')
bullet('A2 (encounter-triggered state residual): 2 of 27 at κ = 0, dropping to 1 of 27 once '
       'compute is charged. Both correctors are null.')
body3b = (
    'A perfect oracle correction applied at every close encounter yields 0% improvement on '
    'the binary–single system at every separation r_bin ∈ {0.05–0.40 AU}, confirming that '
    'the failure is temporal under-resolution, not force-model error. No force or state '
    'correction can repair a step-size problem; adaptive resolution (time-symmetric leapfrog '
    'or IAS15) can and does (Figure 3). The A2 prototype demonstrates that encounter dynamics '
    'contain learnable structure, but the learned model does not translate into a net gain at '
    'equal computational cost.'
)
para(body3b, space_before=3, space_after=6)

# Operating envelope table
p = para(space_before=4, space_after=2)
add_run(p, 'Operating envelope', bold=True)
add_run(p, ' — three-regime summary:')

oe_tbl = doc.add_table(rows=4, cols=4)
oe_tbl.style = 'Table Grid'
hdrs = ['Regime', 'Sufficient integrator', 'Example config', 'Cost (fe/yr)']
rows_data = [
    ['Regular / well-separated',     'Fixed leapfrog dt = 0.04',         'IC6: 0.0006% energy',      '25'],
    ['Moderate scattering',           'Time-symmetric adaptive η = 0.05', 'IC1: 0.087%, IC4: 0.224%', '75–88'],
    ['Tight binary (r_bin ≤ 0.05 AU)','IAS15',                           'BS_0.05: bounded',         '~31,600'],
]
for i, h in enumerate(hdrs):
    cell_text(oe_tbl.cell(0, i), h, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
for r, row in enumerate(rows_data):
    for c, val in enumerate(row):
        cell_text(oe_tbl.cell(r+1, c), val)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# CONCLUSIONS AND CONTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════

heading('Conclusions and Contributions')

conclusions = [
    ('C1 — Operating envelope (answers Q1). ',
     'Fixed-step leapfrog is necessary and sufficient for regular, well-separated systems. '
     'The time-symmetric adaptive leapfrog is necessary and sufficient for moderate scattering: '
     'it is the only method that stays bounded at a single control setting across this regime '
     'and reaches accuracies that fixed leapfrog cannot attain. IAS15 is necessary for tight '
     'binaries, where all fixed-step and heuristic methods eject the binary and the '
     'time-symmetric scheme, while bounded, is 65× slower in wall-clock time.'),
    ('C2 — Stability diagnostics (answers Q2). ',
     'For chaotic three-body work, stability cannot be judged from energy drift or ejection '
     'status alone. The full required set is: (a) max|ΔE/E₀| over the full integration time; '
     '(b) short-horizon position RMS within one Lyapunov time; (c) bounded/ejected status; '
     '(d) bound-pair separation range for hierarchical systems. Finding 2 demonstrates '
     'concretely on a real system that (a)–(c) all pass while the Moon\'s orbit is destroyed; '
     'only (d) detects the failure.'),
    ('C3 — Neural network corrections do not beat equal-cost refinement (answers Q3). ',
     'Under a pre-registered protocol, neither corrector outperforms an equal-cost finer '
     'classical step on the short-horizon metric. The oracle analysis localises the failure '
     'to temporal resolution. The A2 prototype shows that encounter dynamics have learnable '
     'structure, but learning alone does not overcome the timestep problem; adaptive '
     'resolution does.'),
]
for label, body in conclusions:
    p = para(space_before=4, space_after=3)
    add_run(p, label, bold=True)
    add_run(p, body)

p = para(space_before=6, space_after=3)
add_run(p, 'Novelty. ', bold=True)
add_run(p, 'The time-symmetric adaptive leapfrog is the method of Hut, Makino & McMillan '
           '(1995) [6]. The original contributions of this work are: the controlled '
           'characterisation of this method across dynamical regimes on a common test suite; '
           'the demonstration that an energy criterion misclassifies a physically destroyed '
           'real system and the identification of pair-separation as a necessary diagnostic; '
           'and the pre-registered equal-compute null result on learned corrections with the '
           'failure mechanism identified.')

# ── Keywords ──────────────────────────────────────────────────────────────────
p = para(space_before=8, space_after=3)
add_run(p, 'Keywords: ', bold=True)
add_run(p, 'three-body problem; symplectic and time-symmetric integration; deterministic chaos; '
           'computational celestial mechanics; neural network force corrections; operating envelope.')

# ── References ────────────────────────────────────────────────────────────────
heading('References')
refs = [
    '[1] H. Poincaré (1890). Sur le problème des trois corps et les équations de la dynamique. Acta Mathematica, 13, 1–270.',
    '[2] H. Yoshida (1990). Construction of higher order symplectic integrators. Physics Letters A, 150, 262–268.',
    '[3] H. Rein & D. S. Spiegel (2015). IAS15: a fast, adaptive, high-order integrator for gravitational dynamics. MNRAS, 446, 1424–1437.',
    '[4] H. Rein & S.-F. Liu (2012). REBOUND: an open-source multi-purpose N-body code for collisional dynamics. A&A, 537, A128.',
    '[5] P. G. Breen, C. N. Foley, T. Boekholt & S. Portegies Zwart (2020). Newton versus the machine: solving the chaotic three-body problem using deep neural networks. MNRAS, 494, 2465–2470.',
    '[6] P. Hut, J. Makino & S. McMillan (1995). Building a better leapfrog. ApJ Letters, 443, L93–L96.',
    '[7] J. D. Giorgini et al. (1996). JPL\'s on-line Solar System data service (HORIZONS). BAAS, 28, 1158.',
]
for ref in refs:
    p = para(space_before=0, space_after=2)
    add_run(p, ref)

p = para(space_before=6, space_after=0)
add_run(p, '(Bibliographic details should be confirmed against the originals before final submission.)',
        italic=True)

# ── Save ──────────────────────────────────────────────────────────────────────
out = 'abstract/Research_Outline_Aarush_Gupta.docx'
doc.save(out)
print(f'Saved: {out}')
