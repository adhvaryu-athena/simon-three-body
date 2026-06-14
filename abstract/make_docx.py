"""
Generate the St. Yau 2026 Research Outline DOCX.
Matches the official 2-page template exactly:
  Page 1: form tables + title + Abstract heading + abstract text start
  Page 2: abstract text cont. + Keywords + References
Run from repo root: python abstract/make_docx.py
Output: abstract/Research_Outline_Aarush_Gupta.docx
"""

from docx import Document
from docx.shared import Pt, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ── Document and page setup ───────────────────────────────────────────────────
doc = Document()

for sec in doc.sections:
    sec.page_width       = Cm(21.0)
    sec.page_height      = Cm(29.7)
    sec.top_margin       = Cm(2.5)
    sec.bottom_margin    = Cm(2.5)
    sec.left_margin      = Cm(2.5)
    sec.right_margin     = Cm(2.5)
    sec.header_distance  = Cm(1.25)
    sec.footer_distance  = Cm(1.25)

# Default style: Calibri 11 pt
ns = doc.styles['Normal']
ns.font.name = 'Calibri'
ns.font.size = Pt(11)
ns.paragraph_format.space_before      = Pt(0)
ns.paragraph_format.space_after       = Pt(0)
ns.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE

CONTENT_WIDTH_CM = 16.0   # 21 - 2×2.5

# ── Header: "Research Outline    2026 S.T. Yau…" ────────────────────────────
hdr = doc.sections[0].header
hp  = hdr.paragraphs[0]
hp.paragraph_format.space_before = Pt(0)
hp.paragraph_format.space_after  = Pt(0)
# Left text
r1 = hp.add_run('Research Outline')
r1.font.name = 'Calibri'; r1.font.size = Pt(10)
# Right-aligned tab + text
tab_twips = int(Cm(CONTENT_WIDTH_CM).pt * 20 / 0.75)   # cm → twips
pPr = hp._p.get_or_add_pPr()
tabs_el = OxmlElement('w:tabs')
tab_el  = OxmlElement('w:tab')
tab_el.set(qn('w:val'),   'right')
tab_el.set(qn('w:pos'),   str(int(Cm(CONTENT_WIDTH_CM) * 1440 / 914400)))
# simpler: just measure in EMUs → twips.  1 EMU = 1/914400 inch; 1 twip = 1/1440 inch
tab_el.set(qn('w:pos'),   str(int(CONTENT_WIDTH_CM / 2.54 * 1440)))  # cm→twips
tabs_el.append(tab_el)
pPr.append(tabs_el)
r2 = hp.add_run('\t2026 S.T. Yau High School Science Award (Asia)')
r2.font.name = 'Calibri'; r2.font.size = Pt(10)

# ── Footer: "Page X of 2" ────────────────────────────────────────────────────
ftr = doc.sections[0].footer
fp  = ftr.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
fp.paragraph_format.space_before = Pt(0)
fp.paragraph_format.space_after  = Pt(0)

def add_field(para, field_instr):
    """Insert a Word field (e.g. PAGE) into a paragraph run."""
    r = OxmlElement('w:r')
    fld_begin = OxmlElement('w:fldChar'); fld_begin.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText'); instr.text = field_instr; instr.set(qn('xml:space'), 'preserve')
    fld_end = OxmlElement('w:fldChar'); fld_end.set(qn('w:fldCharType'), 'end')
    r.append(fld_begin); r.append(instr); r.append(fld_end)
    para._p.append(r)

def frun(para, text, size=10, bold=False):
    r = para.add_run(text); r.font.name = 'Calibri'; r.font.size = Pt(size); r.bold = bold; return r

frun(fp, 'Page ')
add_field(fp, 'PAGE')
frun(fp, ' of 2')

# ── Helpers ───────────────────────────────────────────────────────────────────
def body_para(text='', bold=False, italic=False, size=11,
              align=WD_ALIGN_PARAGRAPH.LEFT,
              space_before=0, space_after=0):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before      = Pt(space_before)
    p.paragraph_format.space_after       = Pt(space_after)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if text:
        r = p.add_run(text)
        r.bold = bold; r.italic = italic
        r.font.name = 'Calibri'; r.font.size = Pt(size)
    return p

def mixed_para(align=WD_ALIGN_PARAGRAPH.LEFT, space_before=0, space_after=0):
    """Return an empty paragraph; caller adds runs manually."""
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before      = Pt(space_before)
    p.paragraph_format.space_after       = Pt(space_after)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    return p

def prun(p, text, bold=False, italic=False, size=11):
    r = p.add_run(text)
    r.bold = bold; r.italic = italic
    r.font.name = 'Calibri'; r.font.size = Pt(size)
    return r

def cell_fmt(cell, text='', bold=False, align=WD_ALIGN_PARAGRAPH.LEFT, size=11):
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before      = Pt(1)
    p.paragraph_format.space_after       = Pt(1)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    if text:
        r = p.add_run(text)
        r.bold = bold; r.font.name = 'Calibri'; r.font.size = Pt(size)

# ── Title block ───────────────────────────────────────────────────────────────
body_para('2026 S.T. Yau High School Science Award (Asia)',
          bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
          space_before=0, space_after=0)
body_para('Research Outline',
          bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
          space_before=0, space_after=4)

# ── Team member table ─────────────────────────────────────────────────────────
# 5 rows × 4 cols  (label | TM1 | TM2 | TM3)
t1 = doc.add_table(rows=5, cols=4)
t1.style = 'Table Grid'
t1.alignment = WD_TABLE_ALIGNMENT.CENTER

# Column widths: label 3.5 cm, each TM col 4.17 cm
col_widths = [Cm(3.5), Cm(4.17), Cm(4.17), Cm(4.16)]
for i, w in enumerate(col_widths):
    for cell in t1.columns[i].cells:
        cell.width = w

# Row 0: header
cell_fmt(t1.cell(0, 0), '')
for col in (1, 2, 3):
    cell_fmt(t1.cell(0, col), 'Team Member', align=WD_ALIGN_PARAGRAPH.CENTER)

# Rows 1-3: Name, School, City Country
labels1 = ['Name', 'School', 'City, Country']
values1 = ['Aarush Gupta', 'UWCSEA, East', 'Singapore']
for row, (lbl, val) in enumerate(zip(labels1, values1), start=1):
    cell_fmt(t1.cell(row, 0), lbl)
    cell_fmt(t1.cell(row, 1), val)
    cell_fmt(t1.cell(row, 2), '')
    cell_fmt(t1.cell(row, 3), '')

# Row 4: Registration No. (merge cols 1-3)
cell_fmt(t1.cell(4, 0), 'Registration No.')
t1.cell(4, 1).merge(t1.cell(4, 3))
cell_fmt(t1.cell(4, 1), '[Physics-0XX, assigned at registration]')

doc.add_paragraph().paragraph_format.space_after = Pt(4)

# ── Supervising teacher table ─────────────────────────────────────────────────
# 5 rows × 2 cols  (label | value)
t2 = doc.add_table(rows=5, cols=2)
t2.style = 'Table Grid'
t2.alignment = WD_TABLE_ALIGNMENT.CENTER

t2.columns[0].width = Cm(3.5)
t2.columns[1].width = Cm(12.5)
for cell in t2.columns[0].cells:
    cell.width = Cm(3.5)
for cell in t2.columns[1].cells:
    cell.width = Cm(12.5)

# Row 0: merged header
t2.cell(0, 0).merge(t2.cell(0, 1))
cell_fmt(t2.cell(0, 0), 'Supervising Teacher',
         align=WD_ALIGN_PARAGRAPH.CENTER)

sup_labels = ['Name', 'Position', 'School/Institution', 'City, Country']
sup_vals   = ['[Name]', '[Position]', '[School / College]', '[City, Country]']
for row, (lbl, val) in enumerate(zip(sup_labels, sup_vals), start=1):
    cell_fmt(t2.cell(row, 0), lbl)
    cell_fmt(t2.cell(row, 1), val)

doc.add_paragraph().paragraph_format.space_after = Pt(4)

# ── Paper title and Abstract heading ─────────────────────────────────────────
body_para('Structure over Resolution: an Operating Envelope for Stable Integration '
          'of the Chaotic Three-Body Problem',
          bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
          space_before=0, space_after=4)

body_para('Abstract', bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
          space_before=0, space_after=0)

# ── Abstract text ─────────────────────────────────────────────────────────────
# ~290 words, condensed to fit 2 pages at 1.5 line spacing.

paras = [
    # Background
    ("The Newtonian three-body problem has no general closed-form solution and is the canonical "
     "prototype of deterministic chaos: every quantitative statement about such a system rests on a "
     "numerical integrator and on the yardstick by which it is judged. Two difficulties are usually "
     "left implicit. Because neighbouring trajectories diverge exponentially, global position error "
     "against a reference saturates within a few Lyapunov times and cannot rank methods over long "
     "time scales. And neural network models proposed to accelerate integration are rarely "
     "benchmarked at equal computational cost against the obvious alternative of a finer classical step."),
    # Methods + findings lead-in
    ("I compare five integrator methods and two neural network correctors across analytic three-body "
     "configurations, binary–single scattering encounters, and the real Sun–Earth–Moon system "
     "initialised from JPL Horizons ephemerides, using chaos-appropriate diagnostics: maximum "
     "relative energy drift, short-horizon position RMS within one Lyapunov time, bounded/ejected "
     "status, and the preserved separation of bound pairs."),
]

for text in paras:
    body_para(text, space_before=0, space_after=0)

# Three findings with italic lead-ins
findings = [
    ("First, ", "time-symmetry, not brute-force resolution, governs long-term energy drift",
     ": the reversible adaptive leapfrog (η = 0.05) stays energy-bounded across every "
     "dynamical regime at a single control setting, and reaches the accuracy of a much finer fixed "
     "step at lower cost on close-encounter and under-resolved systems, with no advantage on "
     "near-regular orbits. The contribution is an operating map of where adaptivity pays, not a universal speedup."),
    ("Second, ", "energy conservation can actively mislead",
     ": a fixed-step integration conserves total energy to 0.008% and reports the Sun–Earth–Moon "
     "system as bound, yet the Earth–Moon separation grows to 774× its true value, a "
     "destroyed hierarchy detectable only by a pair-separation metric."),
    ("Third, ", "neural network force and state corrections do not beat equal-cost classical refinement",
     ": under a pre-registered equal-compute protocol on 27 held-out configurations, a per-step "
     "scalar corrector wins just 1 of 25 bounded cases (and a state corrector at most 2 of 24) on "
     "short-horizon position RMS even when charged no computational cost; a perfect oracle correction "
     "yields 0% improvement on under-resolved binaries, localising the failure to temporal resolution "
     "rather than force accuracy."),
]

for prefix, italic_text, suffix in findings:
    p = mixed_para(space_before=0, space_after=0)
    prun(p, prefix)
    prun(p, italic_text, italic=True)
    prun(p, suffix)

# Synthesis
p = mixed_para(space_before=0, space_after=0)
prun(p, "Together these results define an ")
prun(p, "operating envelope", italic=True)
prun(p, ", identifying which integrator is necessary and sufficient in each dynamical regime, and establish "
        "which stability diagnostics are required for chaotic three-body work.")

# ── Keywords ──────────────────────────────────────────────────────────────────
p = mixed_para(space_before=6, space_after=0)
prun(p, 'Keywords', bold=True)
prun(p, ': three-body problem; symplectic and time-symmetric integration; deterministic chaos; '
        'computational celestial mechanics; neural network force corrections; operating envelope.')

# ── References ────────────────────────────────────────────────────────────────
body_para('References', bold=True, space_before=6, space_after=0)

refs = [
    ('[1] H. Poincaré (1890). Sur le problème des trois corps et les équations de la dynamique. '
     'Acta Mathematica, 13, 1–270.'),
    ('[2] H. Hut, J. Makino & S. McMillan (1995). Building a better leapfrog. '
     'ApJ Letters, 443, L93–L96.'),
    ('[3] H. Yoshida (1990). Construction of higher order symplectic integrators. '
     'Physics Letters A, 150, 262–268.'),
    ('[4] H. Rein & S.-F. Liu (2012). REBOUND: an open-source multi-purpose N-body code for '
     'collisional dynamics. A&A, 537, A128.'),
    ('[5] H. Rein & D. S. Spiegel (2015). IAS15: a fast, adaptive, high-order integrator for '
     'gravitational dynamics. MNRAS, 446, 1424–1437.'),
    ('[6] P. G. Breen et al. (2020). Newton versus the machine: solving the chaotic three-body '
     'problem using deep neural networks. MNRAS, 494, 2465–2470.'),
    ('[7] J. D. Giorgini et al. (1996). JPL’s on-line Solar System data service (HORIZONS). '
     'BAAS, 28, 1158.'),
]
for ref in refs:
    body_para(ref, space_before=0, space_after=0)

# ── Save ──────────────────────────────────────────────────────────────────────
out = 'abstract/Research_Outline_Aarush_Gupta.docx'
doc.save(out)
print(f'Saved: {out}')
