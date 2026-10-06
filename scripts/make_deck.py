#!/usr/bin/env python3
"""Build the LiteRev-Evidence slide deck (.pptx).

The deck describes the product and runs one scenario through it as the worked
example. The example's figures are NOT hard-coded in the slides: they live in
`USE_CASE` below, and `--api` refreshes them from a running instance so the deck
can be rebuilt the day of a talk with the numbers of that day.

    python3 scripts/make_deck.py -o literev-evidence.pptx
    python3 scripts/make_deck.py -o literev-evidence.pptx \
        --api https://literev-scenario.com/api --scenario usr-69cc64786731

Without `--api`, the stored figures are used and the deck says as of when they
were read, so a slide never passes off a stale count as a current one.

Needs python-pptx (pip install python-pptx). No other dependency, and no network
access unless `--api` is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# ── Charte ───────────────────────────────────────────────────────────────────
# Reprise de l'interface : fond forêt très sombre, vert de marque, or pour les
# accents, blanc cassé pour le texte.
INK = RGBColor(0x0B, 0x14, 0x11)          # fond
INK_SOFT = RGBColor(0x12, 0x1F, 0x1A)     # cartes
BRAND = RGBColor(0x3F, 0xB9, 0x7E)        # vert de marque
BRAND_DIM = RGBColor(0x1E, 0x5C, 0x40)
GOLD = RGBColor(0xD9, 0xB3, 0x5C)
PAPER = RGBColor(0xF2, 0xF6, 0xF4)
MUTED = RGBColor(0x9A, 0xAE, 0xA5)
RED = RGBColor(0xE0, 0x6C, 0x75)

W, H = Inches(13.333), Inches(7.5)         # 16:9
MARGIN = Inches(0.72)
FONT = "Inter"
MONO = "Consolas"


@dataclass
class UseCase:
    """Les chiffres de l'exemple, lus sur une exécution réelle."""
    scenario_id: str = "usr-69cc64786731"
    name: str = "Early warning indicators for respiratory infections in Western Switzerland"
    question: str = ("What are the early warning indicators for respiratory infections "
                     "in Western Switzerland?")
    as_of: str = "6 October 2026, while the pipeline was still ingesting"
    total: int = 449
    above_threshold: int = 52
    below_threshold: int = 390
    unscored: int = 7
    from_local: int = 228
    newly_fetched: int = 221
    threshold: float = 0.45
    years: list[tuple[int, int]] = field(default_factory=lambda: [
        (2016, 14), (2017, 6), (2018, 11), (2019, 16), (2020, 47), (2021, 53),
        (2022, 23), (2023, 36), (2024, 37), (2025, 56), (2026, 49)])
    sources: list[tuple[str, int]] = field(default_factory=list)

    def refresh(self, api: str, scenario_id: str | None = None) -> str:
        """Relit les compteurs sur une instance en marche. Renvoie une note d'état."""
        import datetime as _dt
        import urllib.request

        sid = scenario_id or self.scenario_id

        def _get(path: str):
            with urllib.request.urlopen(f"{api.rstrip('/')}{path}", timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))

        detail = _get(f"/user-scenarios/{sid}/detail")
        corpus = _get(f"/user-scenarios/{sid}/corpus?limit=1&abstract_chars=0")
        # Le jeu de compteurs commun : une instruction, un instantané (cf.
        # api/scenario_store.py). Repli sur les champs historiques si l'instance
        # est antérieure.
        c = corpus.get("counts") or detail.get("counts") or {}
        self.scenario_id = sid
        self.name = detail.get("name") or self.name
        self.question = detail.get("query") or self.question
        self.total = int(c.get("total", corpus.get("total", self.total)))
        self.above_threshold = int(c.get("above_threshold", corpus.get("above_threshold", 0)))
        self.below_threshold = int(c.get("below_threshold", corpus.get("below_threshold", 0)))
        self.unscored = int(c.get("unscored", corpus.get("unscored", 0)))
        self.from_local = int(c.get("from_local", corpus.get("from_local") or 0))
        self.newly_fetched = int(c.get("newly_fetched", corpus.get("newly_fetched") or 0))
        self.threshold = float(c.get("threshold", corpus.get("threshold", self.threshold)))
        self.years = [(int(y["year"]), int(y["count"]))
                      for y in corpus.get("year_distribution", []) if y.get("year")]
        self.years.sort()
        self.sources = [(str(s["source"]).upper(), int(s["count"]))
                        for s in corpus.get("source_distribution", [])][:10]
        self.as_of = _dt.datetime.now().strftime("%-d %B %Y, %H:%M")
        return f"figures read from {api} for {sid}"


# ── Primitives de mise en page ───────────────────────────────────────────────
def _text(frame, runs, size=16, color=PAPER, bold=False, align=PP_ALIGN.LEFT,
          space_after=6, font=FONT, line=None):
    """Écrit un paragraphe. `runs` est une chaîne, ou une liste (texte, surcharges)."""
    p = frame.paragraphs[0] if not frame.paragraphs[0].runs and not frame.paragraphs[0].text \
        else frame.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    if line:
        p.line_spacing = line
    for item in (runs if isinstance(runs, list) else [runs]):
        txt, over = item if isinstance(item, tuple) else (item, {})
        r = p.add_run()
        r.text = txt
        r.font.size = Pt(over.get("size", size))
        r.font.bold = over.get("bold", bold)
        r.font.color.rgb = over.get("color", color)
        r.font.name = over.get("font", font)
    return p


def _box(slide, x, y, w, h, fill=None, line=None, radius=True, shadow=False):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, x, y, w, h)
    if radius:
        try:
            shape.adjustments[0] = 0.06
        except (IndexError, KeyError):
            pass
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    shape.shadow.inherit = shadow
    return shape


# Mesurer un bloc de texte AVANT de le poser. Sans cela, un titre qui passe à deux
# lignes recouvre la diapositive et une carte dont le texte dépasse déborde de son
# cadre : deux défauts visibles sur un vidéoprojecteur et sur rien d'autre.
# Largeur moyenne d'un caractère, en fraction de la taille de police, pour une
# linéale. Majorée à dessein : mieux vaut une carte un peu haute qu'un texte coupé.
_CHAR_W = 0.63
_CHAR_W_BOLD = 0.66


def _lines(text: str, width: Emu, size: float, bold: bool = False) -> int:
    """Nombre de lignes qu'occupera `text` dans `width` à la taille `size` (points)."""
    if not text:
        return 0
    per_line = max(1, int((width / Inches(1)) * 72 / (size * (_CHAR_W_BOLD if bold else _CHAR_W))))
    total = 0
    for para in text.split("\n"):
        total += max(1, -(-len(para) // per_line))
    return total


def _height(text: str, width: Emu, size: float, bold: bool = False, line: float = 1.18) -> Emu:
    return Emu(int(_lines(text, width, size, bold) * size * line * 12700))


def _tf(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def _slide(prs, title=None, kicker=None, subtitle=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])     # vierge
    bg = _box(s, 0, 0, W, H, fill=INK, radius=False)
    bg.line.fill.background()
    y = MARGIN
    if kicker:
        tf = _tf(s, MARGIN, y, W - 2 * MARGIN, Inches(0.3))
        _text(tf, kicker.upper(), size=11, color=BRAND, bold=True, space_after=0)
        y += Inches(0.34)
    if title:
        th = _height(title, W - 2 * MARGIN, 30, bold=True, line=1.05)
        tf = _tf(s, MARGIN, y, W - 2 * MARGIN, th)
        _text(tf, title, size=30, color=PAPER, bold=True, space_after=0, line=1.05)
        y += th + Inches(0.2)
    if subtitle:
        sh = _height(subtitle, W - 2 * MARGIN, 14, line=1.2)
        tf = _tf(s, MARGIN, y, W - 2 * MARGIN, sh)
        _text(tf, subtitle, size=14, color=MUTED, space_after=0, line=1.2)
        y += sh + Inches(0.1)
    return s, y + Inches(0.18)


def _cards(slide, y, items, cols=3, height=None, gap=Inches(0.26),
           accent=BRAND, body_size=12):
    """Une grille de cartes titre + corps.

    La hauteur est MESURÉE sur la carte la plus longue, pas fixée d'avance : une
    hauteur en dur coupait le dernier mot de la carte la plus chargée."""
    total_w = W - 2 * MARGIN
    cw = int((total_w - gap * (cols - 1)) / cols)
    inner = cw - Inches(0.44)
    rows = (len(items) + cols - 1) // cols

    def _needed(bs):
        return max(_height(h, inner, 13, bold=True, line=1.1)
                   + _height(b, inner, bs, line=1.18)
                   for h, b in items) + Inches(0.58)

    # La grille doit tenir SOUS le titre, pied de page compris. Si elle déborde, on
    # réduit le corps d'un point à la fois plutôt que de laisser les cartes du bas
    # sortir de la diapositive - ce qui n'est visible qu'une fois projeté.
    avail = H - y - MARGIN - Inches(0.2)
    cap = int((avail - gap * (rows - 1)) / rows)
    needed = _needed(body_size)
    while needed > cap and body_size > 9:
        body_size -= 0.5
        needed = _needed(body_size)
    height = min(max(height or 0, needed), cap) if cap > 0 else needed
    for i, (head, body) in enumerate(items):
        col, row = i % cols, i // cols
        x = MARGIN + col * (cw + gap)
        yy = y + row * (height + gap)
        _box(slide, x, yy, cw, height, fill=INK_SOFT, line=BRAND_DIM)
        tf = _tf(slide, x + Inches(0.22), yy + Inches(0.18), cw - Inches(0.44),
                 height - Inches(0.36))
        _text(tf, head, size=13, color=accent, bold=True, space_after=5, line=1.1)
        _text(tf, body, size=body_size, color=MUTED, space_after=0, line=1.18)
    return y + rows * (height + gap)


def _bullets(slide, x, y, w, lines, size=15, gap=Pt(11), marker=BRAND):
    tf = _tf(slide, x, y, w, H - y - MARGIN)
    for i, line in enumerate(lines):
        head, _, rest = line.partition(" · ")
        runs = [("•  ", {"color": marker, "bold": True})]
        if rest:
            runs += [(head + " ", {"color": PAPER, "bold": True}), (rest, {"color": MUTED})]
        else:
            runs += [(head, {"color": MUTED})]
        _text(tf, runs, size=size, space_after=int(gap.pt) if i < len(lines) - 1 else 0,
              line=1.18)
    return tf


def _stat_row(slide, y, stats, accent=BRAND):
    """Une rangée de grands nombres avec leur légende."""
    total_w = W - 2 * MARGIN
    cw = int(total_w / len(stats))
    for i, (value, label, sub) in enumerate(stats):
        x = MARGIN + i * cw
        tf = _tf(slide, x, y, cw - Inches(0.2), Inches(1.5))
        _text(tf, str(value), size=38, color=accent, bold=True, space_after=2)
        _text(tf, label, size=12, color=PAPER, bold=True, space_after=2, line=1.1)
        if sub:
            _text(tf, sub, size=10, color=MUTED, space_after=0, line=1.15)
    return y + Inches(1.5)


def _histogram(slide, x, y, w, h, pairs, bar=BRAND, label_every=2):
    """Un histogramme : l'axe des abscisses est le TEMPS, années creuses comprises."""
    if not pairs:
        return y
    years = {int(a): int(b) for a, b in pairs}
    lo, hi = min(years), max(years)
    cols = [(yv, years.get(yv, 0)) for yv in range(lo, hi + 1)]
    peak = max((c for _, c in cols), default=1) or 1
    slot = int(w / len(cols))
    bw = max(Emu(1), int(slot * 0.74))
    axis_h = Inches(0.3)
    plot_h = h - axis_h
    for i, (yv, count) in enumerate(cols):
        bh = int(plot_h * (count / peak)) if count else Emu(1)
        bh = max(bh, Emu(9000)) if count else Emu(4000)
        rect = _box(slide, x + i * slot, y + plot_h - bh, bw, bh,
                    fill=bar if count else BRAND_DIM, radius=False)
        rect.line.fill.background()
        if count:
            tf = _tf(slide, x + i * slot - Inches(0.1), y + plot_h - bh - Inches(0.22),
                     bw + Inches(0.2), Inches(0.2))
            _text(tf, str(count), size=8, color=MUTED, align=PP_ALIGN.CENTER, space_after=0)
        if i % label_every == 0 or i == len(cols) - 1:
            tf = _tf(slide, x + i * slot - Inches(0.12), y + plot_h + Inches(0.06),
                     bw + Inches(0.24), Inches(0.2))
            _text(tf, str(yv), size=8, color=MUTED, align=PP_ALIGN.CENTER, space_after=0,
                  font=MONO)
    rule = _box(slide, x, y + plot_h, w, Pt(1), fill=BRAND_DIM, radius=False)
    rule.line.fill.background()
    return y + h


def _bars(slide, x, y, w, rows, accent=BRAND, label_w=Inches(2.2), row_h=Inches(0.3)):
    """Des barres horizontales, étiquette à gauche, valeur à droite."""
    peak = max((v for _, v in rows), default=1) or 1
    track_w = w - label_w - Inches(0.75)
    for i, (label, value) in enumerate(rows):
        yy = y + i * row_h
        tf = _tf(slide, x, yy, label_w, row_h)
        _text(tf, label, size=10, color=MUTED, space_after=0, font=MONO)
        bg = _box(slide, x + label_w, yy + Inches(0.06), track_w, Inches(0.11),
                  fill=INK_SOFT, radius=False)
        bg.line.fill.background()
        bw = max(Emu(6000), int(track_w * (value / peak)))
        fg = _box(slide, x + label_w, yy + Inches(0.06), bw, Inches(0.11),
                  fill=accent, radius=False)
        fg.line.fill.background()
        tf = _tf(slide, x + label_w + track_w + Inches(0.1), yy, Inches(0.65), row_h)
        _text(tf, f"{value:,}".replace(",", " "), size=10, color=PAPER, space_after=0,
              font=MONO, align=PP_ALIGN.RIGHT)
    return y + len(rows) * row_h


def _flow(slide, y, steps, accent=BRAND):
    """Une bande d'étapes reliées par des chevrons."""
    total_w = W - 2 * MARGIN
    gap = Inches(0.12)
    cw = int((total_w - gap * (len(steps) - 1)) / len(steps))
    for i, (num, head, body) in enumerate(steps):
        x = MARGIN + i * (cw + gap)
        _box(slide, x, y, cw, Inches(1.55), fill=INK_SOFT, line=BRAND_DIM)
        tf = _tf(slide, x + Inches(0.14), y + Inches(0.14), cw - Inches(0.28), Inches(1.3))
        _text(tf, num, size=10, color=accent, bold=True, space_after=3, font=MONO)
        _text(tf, head, size=11.5, color=PAPER, bold=True, space_after=3, line=1.05)
        _text(tf, body, size=9, color=MUTED, space_after=0, line=1.12)
    return y + Inches(1.55)


def _footer(slide, left, right=""):
    tf = _tf(slide, MARGIN, H - Inches(0.52), W - 2 * MARGIN, Inches(0.3))
    _text(tf, left, size=9, color=MUTED, space_after=0)
    if right:
        tf2 = _tf(slide, MARGIN, H - Inches(0.52), W - 2 * MARGIN, Inches(0.3))
        _text(tf2, right, size=9, color=BRAND_DIM, align=PP_ALIGN.RIGHT, space_after=0)


# ── Les diapositives ─────────────────────────────────────────────────────────
def build(uc: UseCase, out: str, live: bool) -> str:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # 1 ── Titre
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = _box(s, 0, 0, W, H, fill=INK, radius=False)
    bg.line.fill.background()
    band = _box(s, 0, 0, Inches(0.16), H, fill=BRAND, radius=False)
    band.line.fill.background()
    tf = _tf(s, MARGIN, Inches(2.1), W - 2 * MARGIN, Inches(3))
    _text(tf, "LITEREV", size=13, color=BRAND, bold=True, space_after=8)
    _text(tf, "Evidence to Scenario", size=54, color=PAPER, bold=True, space_after=14,
          line=1.0)
    _text(tf, "A living evidence review that reads the whole corpus, says what it is "
              "certain of, and hands the result to a model.",
          size=18, color=MUTED, space_after=26, line=1.25)
    _text(tf, [("Institut de Santé Globale", {"color": PAPER, "bold": True}),
               ("   ·   Université de Genève", {"color": MUTED})], size=13, space_after=0)

    # 2 ── Le problème
    s, y = _slide(prs, "A systematic review takes a year. An outbreak does not wait.",
                  kicker="The problem")
    y = _cards(s, y, [
        ("Searching is the easy part",
         "Thirteen databases answer in seconds. Deciding which of the four thousand "
         "results belong in the review is the work, and it is still done by hand."),
        ("Reading does not scale",
         "A reviewer reads a few hundred abstracts. The corpus has thousands. "
         "Whatever is not read is not in the conclusion, and nobody says so."),
        ("The tools that help, sample",
         "Commercial synthesis tools summarise a few dozen papers out of the eligible "
         "set. An empty cell then means 'none of those fifty', not 'none at all'."),
        ("Certainty is asserted, not graded",
         "A claim drawn from two case reports reads like a claim drawn from three "
         "randomised trials, because nothing on the page distinguishes them."),
        ("The result is a document",
         "A PDF cannot be re-run next month, cannot be audited, and cannot be handed "
         "to a transmission model as parameters."),
        ("And it is not reproducible",
         "Re-run the same question six months later and nobody can say what changed: "
         "the literature, the search, or the reviewer."),
    ], cols=3)
    _footer(s, "LiteRev-Evidence")

    # 3 ── Ce que c'est
    s, y = _slide(prs, "One question in. A graded, citable, re-runnable review out.",
                  kicker="What LiteRev-Evidence is")
    y = _flow(s, y, [
        ("01", "Ask", "A question in plain language, or a boolean query if you have one."),
        ("02", "Assemble", "Thirteen sources plus the local base, deduplicated, PRISMA-counted."),
        ("03", "Rank", "Semantic score over the whole corpus, then a cross-encoder rerank."),
        ("04", "Screen", "Threshold, reviewer inclusion or exclusion, double-blind if needed."),
        ("05", "Extract", "PICO, study design, concepts and parameters, cached per article."),
        ("06", "Synthesise", "Brief, claims with GRADE certainty, gap matrix, citable report."),
        ("07", "Model", "Variables, model spec, epidemiological parameters, SEIR."),
    ], )
    y += Inches(0.3)
    _bullets(s, MARGIN, y, W - 2 * MARGIN, [
        "It is a living review · the same scenario re-runs on a schedule, and what changed is visible.",
        "It is auditable · every figure on screen can be traced to the articles it was counted over.",
        "It is a database, not a document · exports to CSV, Excel, RIS, BibTeX, JSON and Markdown.",
    ], size=14)
    _footer(s, "LiteRev-Evidence")

    # 4 ── La règle de la maison
    s, y = _slide(prs, "Every extraction reads every relevant article",
                  kicker="The rule that makes the difference",
                  subtitle="Not a sample. Not the top twenty. Not the fifty the budget allowed.")
    y = _cards(s, y, [
        ("The constraint",
         "Several thousand abstracts do not fit in one prompt. Every tool hits this "
         "wall. Most answer it by sampling and not saying so."),
        ("Map: once per article",
         "Each article's facts are extracted once and cached on its row (PICO, study "
         "design, concepts). The cost is paid once and is incremental."),
        ("Reduce: in SQL, over all of them",
         "The aggregation runs in SQL over the entire relevant subset. No LLM, no "
         "sampling. The generator writes over that digest."),
    ], cols=3)
    y += Inches(0.2)
    _box(s, MARGIN, y, W - 2 * MARGIN, Inches(1.5), fill=INK_SOFT, line=GOLD)
    tf = _tf(s, MARGIN + Inches(0.3), y + Inches(0.22), W - 2 * MARGIN - Inches(0.6),
             Inches(1.1))
    _text(tf, "Why it matters, concretely", size=12, color=GOLD, bold=True, space_after=6)
    _text(tf, "A commercial report prints a theme-by-dimension matrix with \"Potential "
              "gap\" where a cell is empty, and synthesises 50 of its 216 eligible "
              "papers. An empty cell means none of those fifty. Counted over the whole "
              "relevant subset, the same cell says something a sample cannot: no "
              "article in this corpus studies these two things together.",
          size=12.5, color=MUTED, space_after=0, line=1.22)
    _footer(s, "Pinned by tests/test_full_corpus_digest.py and tests/test_gap_matrix.py")

    # 5 ── Les sources
    s, y = _slide(prs, "Thirteen sources, one corpus", kicker="Where the literature comes from",
                  subtitle="Queried in parallel, under a time budget, with the local base "
                           "treated as one source among the others.")
    y = _cards(s, y, [
        ("Boolean-native", "PubMed (MeSH), Europe PMC, Europe PMC preprints. They apply "
                           "the real boolean query, so their results join the corpus "
                           "directly, without a second local filter."),
        ("Keyword sources", "OpenAlex, Crossref, Semantic Scholar, DOAJ, CORE, OpenAIRE. "
                            "Their ranking is loose, so their results are re-matched "
                            "against the boolean query locally."),
        ("Preprints and trials", "bioRxiv, medRxiv, arXiv, ClinicalTrials.gov. Typed as "
                                 "preprints and registrations, not as journal articles."),
        ("The local base", "Everything previously ingested. A new scenario reuses it "
                           "instantly, which is why a search returns in seconds rather "
                           "than minutes."),
        ("Full text where open", "PubMed Central and Europe PMC full text is chunked and "
                                 "indexed, so the assistant can quote a result section, "
                                 "not only an abstract."),
        ("Field data", "ReliefWeb situation reports alongside the literature, for "
                       "questions where the grey literature is the evidence."),
    ], cols=3)
    _footer(s, "Per-source caps and date windows: docs/SOURCE_LIMITS.md")

    # 6 ── De la question à la requête
    s, y = _slide(prs, "From a question to the query the databases actually received",
                  kicker="Search strategy")
    _box(s, MARGIN, y, W - 2 * MARGIN, Inches(1.0), fill=INK_SOFT, line=BRAND_DIM)
    tf = _tf(s, MARGIN + Inches(0.26), y + Inches(0.16), W - 2 * MARGIN - Inches(0.52),
             Inches(0.7))
    _text(tf, "THE QUESTION, AS ASKED", size=9.5, color=BRAND, bold=True, space_after=5)
    _text(tf, uc.question, size=15, color=PAPER, space_after=0, line=1.15)
    y += Inches(1.22)
    y = _cards(s, y, [
        ("Translated, deterministically",
         "A natural-language question is translated into a boolean expression at "
         "temperature zero with a fixed seed, and cached: the same phrasing always "
         "gives the same strategy, so the corpus is reproducible."),
        ("Shaped per source",
         "A MeSH-tagged expression for PubMed, a portable boolean for the APIs that "
         "accept operators, flattened keywords for those that do not, and arXiv's "
         "own syntax. One query sent four ways."),
        ("Shown, not hidden",
         "The generated boolean is displayed as what it is: the query sent to the "
         "databases, and the definition of the corpus. A reviewer can copy it into "
         "PubMed and get the same set."),
    ], cols=3)
    y += Inches(0.2)
    _bullets(s, MARGIN, y, W - 2 * MARGIN, [
        "PubMed is queried on the union of the MeSH expression and the portable boolean: "
        "a MeSH query alone returned 35 results where the plain boolean returned 306.",
    ], size=13)
    _footer(s, "Stored on the scenario, so the search can be audited and repeated")

    # 7 ── Ce qui définit le corpus
    s, y = _slide(prs, "What is in the corpus, and why",
                  kicker="Corpus assembly",
                  subtitle="Membership is a lexical property of the boolean query, "
                           "independent of any semantic score.")
    y = _flow(s, y, [
        ("1", "Local match", "Every document in the base that satisfies the boolean query."),
        ("2", "Live union", "Plus what the boolean-native sources returned, unfiltered."),
        ("3", "Re-match", "Keyword-source results re-checked against the boolean query."),
        ("4", "Quality rule", "No abstract, no entry: an article nobody can read is not evidence."),
        ("5", "Deduplicate", "One link per distinct article, across sources and runs."),
        ("6", "Freeze", "What arrives after this point belongs to the next run, not this one."),
    ])
    y += Inches(0.3)
    _bullets(s, MARGIN, y, W - 2 * MARGIN, [
        "The corpus is reset to the boolean match on every run · otherwise stale links accumulate and the corpus only ever grows.",
        "A legitimately empty result is allowed to empty the corpus, but a transient source failure is not.",
        "Lowering the relevance threshold brings articles back: the threshold filters, it never deletes.",
    ], size=13.5)
    _footer(s, "api/pipeline.py")

    # 8 ── PRISMA
    s, y = _slide(prs, "PRISMA, counted rather than asserted", kicker="Accounting")
    y = _cards(s, y, [
        ("Records identified",
         "One record per source that returned the article, the local base included. "
         "It is the overlap between sources that makes a duplicate, so both halves "
         "have to be counted."),
        ("Duplicates removed",
         "Counted from the links actually merged, not from a flag nobody sets. That "
         "flag is why the figure used to read zero forever."),
        ("Removed for other reasons",
         "Split explicitly: no abstract (the quality rule), or retrieved by a keyword "
         "source but not matching the boolean query."),
        ("Records screened",
         "The corpus as it stands. Reconciled with the figures frozen at search time, "
         "so later additions and removals appear on their own lines."),
        ("Included and excluded",
         "Per scenario, not per document: the same article can be included in one "
         "review and excluded from another."),
        ("Double-blind available",
         "Two reviewers, blind to each other, with disagreements surfaced. Required "
         "for a formal systematic review."),
    ], cols=3)
    _footer(s, "api/review.py")

    # 9 ── Pertinence
    s, y = _slide(prs, "Relevance: a threshold, a reviewer, and one definition of 'relevant'",
                  kicker="Ranking and screening")
    y = _cards(s, y, [
        ("Semantic score",
         "Every article in the corpus is scored by cosine similarity to the question, "
         "reusing stored embeddings and embedding the rest on the fly. The whole "
         "corpus, not a page of it."),
        ("Cross-encoder rerank",
         "A Cohere rerank refines the ordering of the relevant subset, where the "
         "difference between rank 5 and rank 50 actually matters."),
        ("The shared gate",
         "'Relevant' means: never a duplicate, never excluded by a reviewer, and "
         "otherwise included by hand OR above the threshold. Written once, in SQL, "
         "used by every panel and every extraction."),
    ], cols=3)
    y += Inches(0.22)
    _box(s, MARGIN, y, W - 2 * MARGIN, Inches(1.35), fill=INK_SOFT, line=RED)
    tf = _tf(s, MARGIN + Inches(0.3), y + Inches(0.2), W - 2 * MARGIN - Inches(0.6),
             Inches(1.0))
    _text(tf, "Why one definition, written once", size=12, color=RED, bold=True,
          space_after=6)
    _text(tf, "The condition had been copied by hand into each module, and the copies "
              "drifted. The assistant's copy had lost both the duplicate exclusion and "
              "the reviewer exclusion, so it could cite an article a reviewer had just "
              "thrown out, while the counter underneath the answer counted the correct "
              "subset. One function, one place to fix.",
          size=12.5, color=MUTED, space_after=0, line=1.22)
    _footer(s, "relevant_gate_sql() in api/scenario_store.py")

    # 10 ── Niveaux de preuve
    s, y = _slide(prs, "Sixteen study designs, four certainties, from official lists",
                  kicker="Study design and GRADE",
                  subtitle="MeSH Publication Types (tree V03) and Epidemiologic Study "
                           "Characteristics (E05.318), graded by GRADE.")
    y = _cards(s, y, [
        ("High", "Systematic review and meta-analysis OF randomised trials. Randomised "
                 "controlled trials."),
        ("Moderate", "Controlled clinical trials where allocation is not stated. "
                     "Clinical practice guidelines."),
        ("Low", "Cohort, case-control and cross-sectional studies. Non-randomised and "
                "quasi-experimental trials. Surveillance. Syntheses of observational "
                "studies."),
        ("Very low", "Case reports and case series. Narrative reviews. Qualitative "
                     "studies. Modelling and preclinical work."),
    ], cols=4, accent=GOLD)
    y += Inches(0.24)
    _bullets(s, MARGIN, y, W - 2 * MARGIN, [
        "Randomised trials start high, observational studies start low · and a synthesis never upgrades its inputs.",
        "The vocabulary is written once, in Python, and the SQL that classifies articles is generated from it, with a test running both against a real database.",
        "The page explains, in plain words, which designs sit at which level: a grade nobody can check is a decoration.",
    ], size=13)
    _footer(s, "api/study_design.py")

    # 11 ── Le brief et les affirmations
    s, y = _slide(prs, "Claims, each with the designs behind it",
                  kicker="Evidence synthesis")
    y = _cards(s, y, [
        ("A claim, not a summary",
         "Each statement carries the articles that support it, by number, so a reader "
         "can go and check."),
        ("Graded by its weakest honest reading",
         "The certainty of a claim is driven by the designs that actually support it, "
         "capped by what the corpus as a whole can sustain."),
        ("A ceiling the corpus imposes",
         "A corpus with no randomised evidence cannot produce a high-certainty claim, "
         "whatever the wording of the abstracts."),
        ("Counted over everything relevant",
         "The brief is written over an SQL digest of the entire relevant subset; only "
         "a handful of articles are reproduced, for quotation."),
        ("Invalidated when the corpus moves",
         "The cached brief carries a fingerprint of the threshold and the article "
         "identifiers, so it expires rather than describing a corpus that has changed."),
        ("Exportable as a document",
         "A citable report with renumbered references, ready to paste into a paper or "
         "hand to a committee."),
    ], cols=3)
    _footer(s, "api/evidence.py, api/report.py")

    # 12 ── La matrice des manques
    s, y = _slide(prs, "The gap matrix: what nobody has studied together",
                  kicker="Finding the hole",
                  subtitle="Interventions against outcomes, populations against settings, "
                           "counted in SQL over every relevant article.")
    y = _cards(s, y, [
        ("Every cell is counted",
         "Not sampled. A cell reading zero means no article in this corpus pairs these "
         "two concepts, which is a finding."),
        ("It declares its denominator",
         "The matrix says how many relevant articles it could not read, because "
         "nothing was extracted from them. A gap figure that hides its denominator is "
         "the thing being replaced."),
        ("An axis is truncated, and says so",
         "A forty-by-forty grid is not read. It is cut to its largest labels, and the "
         "untruncated count stays on screen so the grid never looks more complete "
         "than it is."),
        ("A gap is only claimed inside the grid",
         "Outside it, a zero may be a label that was cut rather than a pairing nobody "
         "studied. Reporting those would invent findings."),
        ("Stable between runs",
         "Ties are broken by label, so a matrix on identical data does not reorder "
         "itself. A figure that moves cannot be cited."),
        ("Screening reaches it",
         "An article a reviewer excluded does not fill a cell, so screening decisions "
         "change the gaps, as they should."),
    ], cols=3)
    _footer(s, "api/digest.py, pinned by tests/test_gap_matrix.py")

    # 13 ── Au-delà de la revue
    s, y = _slide(prs, "From evidence to a model", kicker="What the review feeds")
    y = _cards(s, y, [
        ("Variables",
         "Predictor and outcome variables proposed from the PICO extracted across the "
         "whole relevant subset, with the articles that justify each one."),
        ("Model specification",
         "A candidate specification: outcome, predictors, lags, functional form, with "
         "the evidence behind each choice."),
        ("Epidemiological parameters",
         "R0, incubation, serial interval, case fatality, read from every relevant "
         "article and reported with their spread, not a single borrowed number."),
        ("SEIR",
         "A compartmental model parameterised from those extracted ranges, so the "
         "simulation's assumptions are traceable to papers."),
        ("Recommended actions",
         "What the evidence supports doing, with the certainty of each, in the "
         "language of the person who has to act."),
        ("Situation reports and alerts",
         "ReliefWeb reports beside the literature, and thresholds that raise an alert "
         "when the field data moves."),
    ], cols=3)
    _footer(s, "api/variables.py, api/model_spec.py, api/seir.py, api/situation_reports.py")

    # 14 ── Le cas d'usage : la question
    s, y = _slide(prs, uc.name, kicker="Worked example",
                  subtitle=f"Scenario {uc.scenario_id} · figures as of {uc.as_of}.")
    y = _stat_row(s, y, [
        (f"{uc.total:,}".replace(",", " "), "articles in the corpus",
         f"{uc.from_local} already in the local base, {uc.newly_fetched} fetched for this scenario"),
        (f"{uc.above_threshold:,}".replace(",", " "), "above the threshold",
         f"cosine similarity ≥ {uc.threshold:.2f}"),
        (f"{uc.below_threshold:,}".replace(",", " "), "below, and kept",
         "lowering the threshold brings them back; nothing is deleted"),
        (f"{uc.unscored:,}".replace(",", " "), "not yet scored",
         "the pipeline was still running when this was read"),
    ])
    y += Inches(0.2)
    tf = _tf(s, MARGIN, y, W - 2 * MARGIN, Inches(0.8))
    _text(tf, "A question in plain English, asked once. Thirteen sources queried, a "
              "boolean strategy generated and stored, a corpus assembled and "
              "deduplicated, and every article scored against the question. No "
              "reviewer has validated this selection yet, and the interface says so "
              "on the page rather than in a footnote.",
          size=13.5, color=MUTED, space_after=0, line=1.25)
    _footer(s, "Counts read from one SQL statement, so every panel agrees")

    # 15 ── Le cas d'usage : les chiffres
    s, y = _slide(prs, "The corpus, described", kicker="Worked example",
                  subtitle="Publication years as a histogram, so the axis is time and "
                           "empty years show as empty.")
    half = int((W - 2 * MARGIN - Inches(0.5)) / 2)
    tf = _tf(s, MARGIN, y, half, Inches(0.3))
    _text(tf, "PUBLICATION YEAR", size=9.5, color=BRAND, bold=True, space_after=0)
    _histogram(s, MARGIN, y + Inches(0.34), half, Inches(2.5), uc.years)
    rx = MARGIN + half + Inches(0.5)
    if uc.sources:
        tf = _tf(s, rx, y, half, Inches(0.3))
        _text(tf, "LITERATURE SOURCES", size=9.5, color=BRAND, bold=True, space_after=0)
        _bars(s, rx, y + Inches(0.4), half, uc.sources)
    else:
        # Sans instance en marche, on montre ce que l'on sait VRAIMENT de ce corpus :
        # sa composition. Recopier la ventilation par source d'un autre scénario
        # ferait un graphique faux sur une diapositive qui a l'air juste.
        tf = _tf(s, rx, y, half, Inches(0.3))
        _text(tf, "WHERE THE CORPUS CAME FROM", size=9.5, color=BRAND, bold=True,
              space_after=0)
        _bars(s, rx, y + Inches(0.4), half, [
            ("LOCAL BASE", uc.from_local),
            ("FETCHED LIVE", uc.newly_fetched),
        ], label_w=Inches(1.5), row_h=Inches(0.34))
        tf = _tf(s, rx, y + Inches(1.2), half, Inches(0.3))
        _text(tf, "RELEVANCE, AT A THRESHOLD OF %.2f" % uc.threshold, size=9.5,
              color=GOLD, bold=True, space_after=0)
        _bars(s, rx, y + Inches(1.6), half, [
            ("ABOVE", uc.above_threshold),
            ("BELOW, KEPT", uc.below_threshold),
            ("NOT YET SCORED", uc.unscored),
        ], accent=GOLD, label_w=Inches(1.5), row_h=Inches(0.34))
    y += Inches(3.1)
    _bullets(s, MARGIN, y, W - 2 * MARGIN, [
        "The surge from 2020 is the pandemic literature, and the 2026 count is a year still in progress.",
        "Half the corpus was already in the local base: a new question on a covered area returns in seconds, not minutes.",
        "Run this deck with --api against a live instance to replace these figures, and the per-source breakdown, with the ones of the day.",
    ], size=13)
    _footer(s, f"Scenario {uc.scenario_id}")

    # 16 ── Ce que l'exemple produit
    s, y = _slide(prs, "What comes out of it", kicker="Worked example")
    y = _cards(s, y, [
        ("An evidence brief",
         "What the literature establishes about early warning indicators, written over "
         "a digest of every relevant article, with claims carrying their certainty."),
        ("A design and certainty profile",
         "How much of this corpus is surveillance, modelling, cohort or trial work, "
         "and therefore what the corpus can and cannot support."),
        ("A gap matrix",
         "Which indicators have been studied against which outcomes, and which "
         "pairings nobody has published on."),
        ("Candidate variables",
         "The indicators the literature actually uses, with the articles behind each, "
         "ready to become a model specification."),
        ("A citable report",
         "Renumbered references, the figures, the method, downloadable."),
        ("A bibliography",
         "The relevant articles as RIS, BibTeX, CSV or Excel. The RIS imports into "
         "Zotero with its journals, which it did not before."),
    ], cols=3)
    _footer(s, f"Scenario {uc.scenario_id}")

    # 17 ── Compter une fois
    s, y = _slide(prs, "One corpus, one number, wherever it is shown",
                  kicker="What correctness looks like here",
                  subtitle="A real screenshot: 433 in the banner, 449 in the corpus "
                           "title, and 441 of 433 articles scored.")
    y = _cards(s, y, [
        ("The symptom",
         "More articles scored than exist. Three panels of one page, disagreeing at "
         "one moment, while a pipeline ran."),
        ("The cause",
         "Four endpoints each ran their own COUNT, in their own connection, at their "
         "own instant. Under READ COMMITTED, two statements of the same connection "
         "already see two snapshots."),
        ("The fix",
         "Every count comes from one SQL statement, so one snapshot, by construction. "
         "The threshold is read inside it. Figures still move while a search runs, "
         "but they move together."),
    ], cols=3)
    y += Inches(0.24)
    _bullets(s, MARGIN, y, W - 2 * MARGIN, [
        "The endpoint whose job was to reconcile the counters was itself a fifth independent count.",
        "A test now asserts that the parts add up to the whole, that scored can never exceed the total, and that every endpoint reports the same object.",
        "This is the shape of most of the work: a number on screen that nobody could trace, traced and then made impossible to lose.",
    ], size=13)
    _footer(s, "scenario_counts() in api/scenario_store.py")

    # 18 ── Architecture
    s, y = _slide(prs, "How it is built", kicker="Architecture")
    y = _cards(s, y, [
        ("FastAPI, one module per domain",
         "Thirty-six modules in api/, imported in a fixed order; a module only imports "
         "from the ones before it. main.py is the composition root."),
        ("PostgreSQL with pgvector",
         "Documents, chunks, embeddings, scenario links, screening decisions and "
         "cached extractions, all in one database. The aggregations are SQL."),
        ("Lexical and semantic, together",
         "A GIN full-text index for the boolean query, cosine similarity for ranking. "
         "Membership is lexical, ordering is semantic."),
        ("Model roles, not model names",
         "Four roles (bulk, write, chat, embedding) bound to models in one registry, "
         "with request shaping and learned repairs when an API rejects a parameter."),
        ("A React front end",
         "One page per scenario, French and English in parallel locale files, with a "
         "test that the two carry the same keys and the same placeholders."),
        ("Deployed on merge",
         "Every merge to main deploys and restarts the API. Searches in flight are "
         "relaunched at startup."),
    ], cols=3)
    _footer(s, "docs/ARCHITECTURE.md")

    # 19 ── Les garanties
    s, y = _slide(prs, "What is guaranteed, and by what",
                  kicker="Quality",
                  subtitle="Each of these is a test that fails the build, not an "
                           "intention in a document.")
    y = _cards(s, y, [
        ("Extractions read everything",
         "test_full_corpus_digest.py pins both halves of map-then-reduce: the per-"
         "article cache, and the SQL aggregation over the whole relevant subset."),
        ("Counts cannot diverge",
         "test_one_article_count.py pins one statement, one snapshot, and that every "
         "endpoint reports the same object."),
        ("Gaps declare their denominator",
         "test_gap_matrix.py pins that a gap is only claimed inside the shown grid, "
         "and that the unread articles are reported."),
        ("One relevance gate",
         "The SQL predicate is generated from one function; a second copy cannot "
         "drift because there is no second copy."),
        ("One study-design vocabulary",
         "The classification SQL is generated from the Python table, with a test "
         "running both against a real database and comparing them."),
        ("Both languages stay in step",
         "A locale test fails when a key or a placeholder exists in one language and "
         "not the other."),
    ], cols=3)
    _footer(s, "827 backend tests, 44 front-end tests")

    # 20 ── Comparaison
    s, y = _slide(prs, "Against the tools it is usually compared to",
                  kicker="Where it differs")
    rows = [
        ("Coverage of the eligible set",
         "A sample: a few dozen of the eligible papers, which the page does not say.",
         "All of it. Map per article, reduce in SQL, no sampling anywhere."),
        ("Certainty",
         "Asserted, or absent. A case report reads like a trial.",
         "GRADE, from a study-design vocabulary taken from MeSH, with a corpus ceiling."),
        ("Gaps",
         "'Potential gap' over the sample, so an empty cell may be a sampling artefact.",
         "Counted over every relevant article, with the unread ones declared."),
        ("Reviewer control",
         "Little or none: the tool decides what is eligible.",
         "Threshold, manual inclusion and exclusion, double-blind screening, PRISMA."),
        ("Reproducibility",
         "A chat answer. Ask twice, get two answers.",
         "A stored boolean strategy, a frozen corpus, a cached brief with a fingerprint."),
        ("What you leave with",
         "A PDF.",
         "A database, a bibliography, a citable report, and model parameters."),
    ]
    row_h = Inches(0.72)
    col1, col2 = Inches(2.9), int((W - 2 * MARGIN - Inches(2.9)) / 2)
    hdr = [("", col1), ("Typical synthesis tool", col2), ("LiteRev-Evidence", col2)]
    x = MARGIN
    for label, cw in hdr:
        tf = _tf(s, x, y, cw - Inches(0.16), Inches(0.3))
        _text(tf, label.upper(), size=9.5,
              color=BRAND if "LiteRev" in label else MUTED, bold=True, space_after=0)
        x += cw
    y += Inches(0.36)
    for i, (label, them, us) in enumerate(rows):
        if i % 2 == 0:
            band = _box(s, MARGIN - Inches(0.14), y - Inches(0.06),
                        W - 2 * MARGIN + Inches(0.28), row_h, fill=INK_SOFT)
            band.line.fill.background()
        tf = _tf(s, MARGIN, y, col1 - Inches(0.16), row_h)
        _text(tf, label, size=11, color=PAPER, bold=True, space_after=0, line=1.12)
        tf = _tf(s, MARGIN + col1, y, col2 - Inches(0.16), row_h)
        _text(tf, them, size=10, color=MUTED, space_after=0, line=1.14)
        tf = _tf(s, MARGIN + col1 + col2, y, col2 - Inches(0.16), row_h)
        _text(tf, us, size=10, color=PAPER, space_after=0, line=1.14)
        y += row_h
    _footer(s, "Comparison drawn from a published Consensus report on a real question")

    # 21 ── Limites
    s, y = _slide(prs, "What it does not do, and what is next",
                  kicker="Honest limits")
    y = _cards(s, y, [
        ("It does not replace a reviewer",
         "An automatic selection is labelled as such on every page. A formal "
         "systematic review needs the double-blind screening tab and two humans."),
        ("Abstract-first",
         "Full text is used where it is open. Behind a paywall, the extraction reads "
         "the abstract, and the article says so."),
        ("Extraction is as good as the abstract",
         "A study design absent from the abstract cannot be classified from it. "
         "Unclassified is reported as unclassified, never guessed."),
        ("Non-English literature",
         "Indexed and searched, but the extraction prompts are strongest in English. "
         "A known asymmetry, not a solved problem."),
        ("Next: screening at scale",
         "Active learning on reviewer decisions, so the threshold stops being the "
         "only lever."),
        ("Next: between two runs",
         "A diff of the review itself: which claims changed, which are new, which "
         "lost their support, when the literature moves."),
    ], cols=3)
    _footer(s, "ROADMAP.md")

    # 22 ── Fin
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = _box(s, 0, 0, W, H, fill=INK, radius=False)
    bg.line.fill.background()
    band = _box(s, 0, 0, Inches(0.16), H, fill=GOLD, radius=False)
    band.line.fill.background()
    tf = _tf(s, MARGIN, Inches(2.4), W - 2 * MARGIN, Inches(3))
    _text(tf, "LiteRev-Evidence", size=46, color=PAPER, bold=True, space_after=16)
    _text(tf, "Ask once. Read everything. Say how sure you are.",
          size=22, color=BRAND, space_after=24, line=1.1)
    _text(tf, "Institut de Santé Globale, Université de Genève",
          size=14, color=MUTED, space_after=6)
    _text(tf, "literev-scenario.com", size=14, color=MUTED, space_after=0, font=MONO)

    prs.save(out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="literev-evidence.pptx")
    ap.add_argument("--api", help="base URL of a running instance, to refresh the "
                                  "worked example's figures")
    ap.add_argument("--scenario", help="scenario id for the worked example")
    args = ap.parse_args(argv)

    uc = UseCase()
    live = False
    if args.api:
        try:
            print(uc.refresh(args.api, args.scenario), file=sys.stderr)
            live = True
        except Exception as e:                                       # noqa: BLE001
            print(f"could not refresh from {args.api}: {e}\n"
                  f"falling back to the stored figures", file=sys.stderr)
    elif args.scenario:
        uc.scenario_id = args.scenario

    out = build(uc, args.out, live)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
