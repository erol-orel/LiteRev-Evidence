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

YEAR_AXIS_FROM = 1995      # début de l'axe des années dans l'histogramme
DESIGN_BARS = 10           # devis montrés en barres, le reste étant regroupé

# Les seize devis, tels que le serveur les nomme, et leur étiquette courte dans
# la figure. Une barre ne dispose que de deux pouces : « Surveillance / registre
# / écologique » y serait coupé.
_DESIGN_SHORT = {
    "Cohort study": "Cohort",
    "Cross-sectional study": "Cross-sectional",
    "Case-control study": "Case-control",
    "Randomized controlled trial": "Randomised trial",
    "Clinical trial (allocation unstated)": "Clinical trial",
    "Non-randomised / quasi-experimental": "Quasi-experimental",
    "Systematic review / meta-analysis": "Systematic review",
    "Narrative review / editorial / opinion": "Narrative review",
    "Surveillance / registry / ecological": "Surveillance",
    "Observational (subtype unstated)": "Observational",
    "Case report / case series": "Case report or series",
    "Guideline / practice guideline": "Guideline",
    "Modelling / simulation": "Modelling",
    "Qualitative research": "Qualitative",
    "Experimental / preclinical": "Preclinical",
    "Design not stated": "Design not stated",
}


def _short_design(label: str) -> str:
    """L'étiquette courte d'un devis. Un devis ajouté côté serveur et absent de
    la table est coupé au premier séparateur plutôt que perdu."""
    if label in _DESIGN_SHORT:
        return _DESIGN_SHORT[label]
    head = label.split(" / ")[0].split(" (")[0].strip()
    return head[:21] if len(head) > 21 else head


# Le barème, si la table de l'application n'est pas importable (diapositives
# construites hors du dépôt). Sert de repli, jamais de source.
_GRADE_FALLBACK = {
    "Inherited": ["Systematic review"],
    "High": ["Randomised trial"],
    "Moderate": ["Clinical trial"],
    "Low": ["Quasi-experimental", "Cohort", "Case-control", "Cross-sectional",
            "Surveillance", "Observational"],
    "Very low": ["Case report or series", "Narrative review"],
    "Not applicable": ["Guideline", "Modelling", "Qualitative", "Preclinical"],
    "Not assessed": ["Design not stated"],
}


def grade_groups() -> dict[str, list[str]]:
    """Les devis rangés par niveau de certitude, LUS dans la table de
    l'application plutôt que recopiés ici.

    Deux cartes de cette diapositive décrivaient encore un barème antérieur :
    les recommandations y étaient données comme « modérée » et le travail
    qualitatif comme « très faible », quand l'application les marque toutes deux
    hors barème. Une table recopiée vieillit ; celle-ci est la même que celle de
    l'écran."""
    try:
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from api.study_design import vocabulary
    except Exception:
        return dict(_GRADE_FALLBACK)
    out: dict[str, list[str]] = {}
    for g in vocabulary("en"):
        key = "Inherited" if g.get("level") is None else str(g.get("label") or "")
        out[key] = [_short_design(str(d.get("label") or "")) for d in g.get("designs", [])]
    return out or dict(_GRADE_FALLBACK)


@dataclass
class UseCase:
    """Les chiffres de l'exemple, lus sur une exécution réelle."""
    scenario_id: str = "usr-69cc64786731"
    name: str = "Early warning indicators for respiratory infections in Western Switzerland"
    question: str = ("What are the early warning indicators for respiratory infections "
                     "in Western Switzerland?")
    as_of: str = "7 October 2026, the pipeline having finished"
    total: int = 6564
    above_threshold: int = 467
    below_threshold: int = 6097
    unscored: int = 0
    from_local: int = 1246
    newly_fetched: int = 5318
    threshold: float = 0.45
    with_fulltext: int = 4275
    journals: int = 133
    relevant_with_pico: int = 467
    relevant_with_fulltext: int = 420
    year_min: int = 1917
    # L'axe des années est long (1917-2026) ; la diapositive le coupe à 1995 et le
    # dit, parce que cent dix barres dont cent sont vides ne se lisent pas.
    years: list[tuple[int, int]] = field(default_factory=lambda: [
        (1995, 5), (1996, 11), (1997, 27), (1998, 11), (1999, 18), (2000, 67),
        (2001, 12), (2002, 42), (2003, 33), (2004, 29), (2005, 41), (2006, 56),
        (2007, 42), (2008, 61), (2009, 85), (2010, 99), (2011, 134),
        (2012, 155), (2013, 165), (2014, 184), (2015, 196), (2016, 232),
        (2017, 243), (2018, 290), (2019, 317), (2020, 623), (2021, 613),
        (2022, 552), (2023, 573), (2024, 505), (2025, 570), (2026, 520)])
    sources: list[tuple[str, int]] = field(default_factory=lambda: [
        ("OPENALEX", 5284), ("EUROPEPMC", 823), ("PUBMED", 184), ("CORE", 71),
        ("CROSSREF", 70), ("PREPRINT", 59), ("SEMANTIC SCHOLAR", 35),
        ("DOAJ", 20), ("ARXIV", 7), ("PROSPERO", 5), ("OPENAIRE", 5),
        ("COCHRANE", 1)])
    # Le profil de preuve, tel que la page l'affiche. Les dix devis les plus
    # fréquents ; `designs_other` porte le reste, pour que la figure ne se lise
    # pas comme le corpus entier.
    designs: list[tuple[str, int]] = field(default_factory=lambda: [
        ("Cohort", 190), ("Cross-sectional", 61), ("Surveillance", 50),
        ("Systematic review", 43), ("Observational", 37), ("Narrative review", 18),
        ("Case-control", 16), ("Design not stated", 13),
        ("Randomised trial", 12), ("Case report or series", 7)])
    designs_other: tuple[int, int] = (6, 20)      # (combien de devis, combien d'articles)
    clusters: list[tuple[str, int]] = field(default_factory=lambda: [
        ("CLUSTER 5", 183), ("CLUSTER 1", 73), ("CLUSTER 3", 71),
        ("CLUSTER 2", 35), ("CLUSTER 4", 32)])
    levels: list[tuple[str, int]] = field(default_factory=lambda: [
        ("Low", 400), ("Very low", 25), ("Not applicable", 16),
        ("Not assessed", 13), ("High", 12), ("Moderate", 1)])

    def refresh(self, api: str, scenario_id: str | None = None) -> str:
        """Relit les compteurs sur une instance en marche. Renvoie une note d'état."""
        import datetime as _dt
        import urllib.request

        sid = scenario_id or self.scenario_id

        def _get(path: str, timeout: int = 30):
            with urllib.request.urlopen(f"{api.rstrip('/')}{path}", timeout=timeout) as r:
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
        self.with_fulltext = int(c.get("with_fulltext", self.with_fulltext))
        self.journals = int(c.get("journals_count", self.journals))
        self.year_min = int(c.get("year_min") or self.year_min)
        # L'axe commence en 1995 : le corpus remonte à 1917, et cent dix barres
        # dont quatre-vingts valent un ou deux ne se lisent pas. La diapositive
        # porte l'année de départ dans son intitulé.
        self.years = sorted((int(y["year"]), int(y["count"]))
                            for y in corpus.get("year_distribution", [])
                            if y.get("year") and int(y["year"]) >= YEAR_AXIS_FROM)
        self.sources = [(str(s["source"]).upper().replace("_", " "), int(s["count"]))
                        for s in corpus.get("source_distribution", [])][:10]

        # Le profil de preuve. Ces deux distributions sont celles que l'écran
        # affiche et que le sélecteur de corpus reprend, lues au même endroit :
        # les recopier à la main les laissait vieillir d'une version sur l'autre.
        ev = _get(f"/user-scenarios/{sid}/evidence-brief", timeout=180)
        designs = [(_short_design(str(d.get("design_en") or d.get("design") or "")),
                    int(d["count"])) for d in ev.get("study_design_distribution", [])]
        designs = [(lbl, n) for lbl, n in designs if lbl and n]
        if designs:
            self.designs = designs[:DESIGN_BARS]
            rest = designs[DESIGN_BARS:]
            self.designs_other = (len(rest), sum(n for _, n in rest))
        levels = [(str(e.get("level_en") or e.get("level") or ""), int(e["count"]))
                  for e in ev.get("evidence_level_distribution", [])]
        levels = [(lbl, n) for lbl, n in levels if lbl and n]
        if levels:
            self.levels = levels
        stats = ev.get("corpus_stats") or {}
        self.relevant_with_pico = int(stats.get("relevant_with_pico", self.relevant_with_pico))
        self.relevant_with_fulltext = int(
            stats.get("relevant_with_fulltext", self.relevant_with_fulltext))

        # Le clustering est recalculé après un redémarrage : une instance encore
        # froide répond sans groupes, et les chiffres précédents valent mieux
        # qu'une figure vide.
        try:
            cl = _get(f"/user-scenarios/{sid}/clustering", timeout=300)
            found = [(str(g.get("cluster_name") or "").upper(), int(g.get("n_docs") or 0))
                     for g in cl.get("clusters", []) if not g.get("is_noise")]
            found = [(lbl, n) for lbl, n in found if lbl and n]
            if found:
                self.clusters = sorted(found, key=lambda r: -r[1])
        except Exception:
            pass

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
    """Nombre de lignes qu'occupera `text` dans `width` à la taille `size` (points).

    Le retour se fait AU MOT, comme dans le rendu. Diviser la longueur par le
    nombre de caractères par ligne sous-estimait d'une ou deux lignes dès que les
    mots étaient longs, parce que la fin de chaque ligne reste vide : la dernière
    ligne d'une carte passait alors sous sa bordure."""
    if not text:
        return 0
    per_line = max(1, int((width / Inches(1)) * 72 / (size * (_CHAR_W_BOLD if bold else _CHAR_W))))
    total = 0
    for para in text.split("\n"):
        words = para.split()
        if not words:
            total += 1
            continue
        count, cur = 1, 0
        for word in words:
            need = len(word) if cur == 0 else cur + 1 + len(word)
            if need <= per_line:
                cur = need
                continue
            count += 1
            cur = len(word)
            while cur > per_line:          # un mot plus long qu'une ligne
                count += 1
                cur -= per_line
        total += count
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
        _text(tf, kicker.upper(), size=12, color=BRAND, bold=True, space_after=0)
        y += Inches(0.34)
    if title:
        th = _height(title, W - 2 * MARGIN, 30, bold=True, line=1.05)
        tf = _tf(s, MARGIN, y, W - 2 * MARGIN, th)
        _text(tf, title, size=30, color=PAPER, bold=True, space_after=0, line=1.05)
        y += th + Inches(0.2)
    if subtitle:
        sh = _height(subtitle, W - 2 * MARGIN, 15, line=1.2)
        tf = _tf(s, MARGIN, y, W - 2 * MARGIN, sh)
        _text(tf, subtitle, size=15, color=MUTED, space_after=0, line=1.2)
        y += sh + Inches(0.1)
    return s, y + Inches(0.18)


def _cards(slide, y, items, cols=3, height=None, gap=Inches(0.26),
           accent=BRAND, body_size=13):
    """Une grille de cartes titre + corps.

    La hauteur est MESURÉE sur la carte la plus longue, pas fixée d'avance : une
    hauteur en dur coupait le dernier mot de la carte la plus chargée."""
    total_w = W - 2 * MARGIN
    cw = int((total_w - gap * (cols - 1)) / cols)
    inner = cw - Inches(0.44)
    rows = (len(items) + cols - 1) // cols

    def _needed(bs):
        return max(_height(h, inner, 14, bold=True, line=1.1)
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
        _text(tf, head, size=14, color=accent, bold=True, space_after=6, line=1.1)
        _text(tf, body, size=body_size, color=MUTED, space_after=0, line=1.18)
    return y + rows * (height + gap)


def _bullets(slide, x, y, w, lines, size=15, gap=Pt(12), marker=BRAND):
    """Une liste à puces, dont le cadre est MESURÉ sur son texte plutôt qu'étendu
    jusqu'au bas de la diapositive : un cadre trop grand passe les contrôles de
    géométrie alors que son texte, lui, déborde sur le pied de page."""
    h = Emu(0)
    for line in lines:
        h += _height("\u2022  " + line, w, size, line=1.18) + Emu(int(gap.pt * 12700))
    tf = _tf(slide, x, y, w, h)
    for i, line in enumerate(lines):
        head, _, rest = line.partition(" · ")
        runs = [("•  ", {"color": marker, "bold": True})]
        if rest:
            runs += [(head + " ", {"color": PAPER, "bold": True}), (rest, {"color": MUTED})]
        else:
            runs += [(head, {"color": MUTED})]
        _text(tf, runs, size=size, space_after=int(gap.pt) if i < len(lines) - 1 else 0,
              line=1.18)
    return y + h


def _stat_row(slide, y, stats, accent=BRAND):
    """Une rangée de grands nombres avec leur légende."""
    total_w = W - 2 * MARGIN
    cw = int(total_w / len(stats))
    for i, (value, label, sub) in enumerate(stats):
        x = MARGIN + i * cw
        tf = _tf(slide, x, y, cw - Inches(0.2), Inches(1.5))
        _text(tf, str(value), size=38, color=accent, bold=True, space_after=2)
        _text(tf, label, size=13, color=PAPER, bold=True, space_after=2, line=1.1)
        if sub:
            _text(tf, sub, size=11, color=MUTED, space_after=0, line=1.18)
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
    # Une colonne sur deux porte sa valeur, et la parité est choisie pour que le
    # SOMMET en fasse partie : l'ajouter en plus de sa voisine superposait les
    # deux nombres.
    parity = next((i for i, (_, c) in enumerate(cols) if c == peak), 0) % 2
    for i, (yv, count) in enumerate(cols):
        bh = int(plot_h * (count / peak)) if count else Emu(1)
        bh = max(bh, Emu(9000)) if count else Emu(4000)
        rect = _box(slide, x + i * slot, y + plot_h - bh, bw, bh,
                    fill=bar if count else BRAND_DIM, radius=False)
        rect.line.fill.background()
        if count and (len(cols) <= 16 or i % 2 == parity):
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


def _flow(slide, y, steps, accent=BRAND, per_row=6):
    """Une bande d'étapes numérotées.

    La hauteur est MESURÉE sur l'étape la plus longue, et au-delà de `per_row`
    étapes la bande passe sur deux rangs : huit colonnes sur une seule rangée
    laissent 1,4 pouce par carte, où le texte était tout simplement coupé."""
    total_w = W - 2 * MARGIN
    gap = Inches(0.12)
    rows = 1 if len(steps) <= per_row else 2
    cols = len(steps) if rows == 1 else -(-len(steps) // 2)
    cw = int((total_w - gap * (cols - 1)) / cols)
    inner = cw - Inches(0.28)
    h = max(_height(num, inner, 11, bold=True)
            + _height(head, inner, 12.5, bold=True, line=1.05)
            + _height(body, inner, 10.5, line=1.14)
            for num, head, body in steps) + Inches(0.45)
    for i, (num, head, body) in enumerate(steps):
        col, row = i % cols, i // cols
        x = MARGIN + col * (cw + gap)
        yy = y + row * (h + gap)
        _box(slide, x, yy, cw, h, fill=INK_SOFT, line=BRAND_DIM)
        tf = _tf(slide, x + Inches(0.14), yy + Inches(0.16), inner, h - Inches(0.3))
        _text(tf, num, size=11, color=accent, bold=True, space_after=3, font=MONO)
        _text(tf, head, size=12.5, color=PAPER, bold=True, space_after=4, line=1.05)
        _text(tf, body, size=10.5, color=MUTED, space_after=0, line=1.14)
    return y + rows * h + (rows - 1) * gap


def _shot(slide, path, x, y, w, h, caption=None, accent=BRAND):
    """Une copie d'écran, cadrée dans `w` x `h` en conservant ses proportions.

    L'image est posée dans un cadre de la couleur des cartes : une capture
    d'interface sombre posée nue sur un fond sombre n'a plus de bord. La légende,
    quand il y en a une, est centrée sous l'image : elle fait partie de la figure."""
    import os
    if not (path and os.path.exists(path)):
        # Pas de capture disponible : un cadre vide vaut mieux qu'une image absente
        # qui décalerait tout le reste de la diapositive.
        _box(slide, x, y, w, h, fill=INK_SOFT, line=BRAND_DIM)
        tf = _tf(slide, x + Inches(0.2), Emu(int(y + h / 2 - Inches(0.15))),
                 w - Inches(0.4), Inches(0.3))
        _text(tf, "screenshot not available at build time", size=10, color=MUTED,
              align=PP_ALIGN.CENTER, space_after=0)
        return y + h
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    dw, dh = int(iw * scale), int(ih * scale)
    dx = x + int((w - dw) / 2)
    # Pas de cadre : l'image porte déjà ses propres bords, et un rectangle arrondi
    # par-dessus ne faisait qu'ajouter une ligne de plus à regarder.
    slide.shapes.add_picture(path, dx, y, dw, dh)
    out = y + dh
    if caption:
        ch = _height(caption, w, 10.5, line=1.2)
        tf = _tf(slide, x, out + Inches(0.16), w, ch)
        _text(tf, caption, size=10.5, color=MUTED, align=PP_ALIGN.CENTER, space_after=0,
              line=1.2)
        out += Inches(0.16) + ch
    return out


def _chips(slide, x, y, w, items, accent=BRAND, size=11.5, gap=Inches(0.11)):
    """Une rangée d'étiquettes qui se replie : un vocabulaire se lit mieux ainsi
    qu'en liste à puces."""
    cx, cy = x, y
    line_h = Inches(0.38)
    for label in items:
        cw = _height(label, Inches(10), size, bold=True)  # largeur approchée via la mesure
        cw = Emu(int(len(label) * size * _CHAR_W_BOLD * 12700) + Inches(0.34))
        if cx + cw > x + w:
            cx, cy = x, cy + line_h + Inches(0.08)
        chip = _box(slide, cx, cy, cw, line_h, fill=INK_SOFT, line=accent)
        chip.line.width = Pt(0.75)
        tf = _tf(slide, cx, cy, cw, line_h, anchor=MSO_ANCHOR.MIDDLE)
        _text(tf, label, size=size, color=accent, bold=True, align=PP_ALIGN.CENTER,
              space_after=0, font=MONO)
        cx += cw + gap
    return cy + line_h


def _pipe(slide, x, y, w, steps, accent=BRAND, size=12.5, gap=Inches(0.18)):
    """Une colonne d'étapes reliées verticalement, pour un enchaînement qui ne tient
    pas en bande horizontale.

    La hauteur de chaque rang est MESURÉE : une hauteur fixe faisait se chevaucher
    les rangs dont le texte passe à trois lignes."""
    tw = w - Inches(0.3)
    yy = y
    for i, (head, body) in enumerate(steps):
        h = _height(f"{head}  {body}", tw, size, line=1.15) + gap
        dot = _box(slide, x, yy + Inches(0.07), Inches(0.14), Inches(0.14),
                   fill=accent, radius=False)
        dot.line.fill.background()
        if i < len(steps) - 1:
            rule = _box(slide, x + Inches(0.06), yy + Inches(0.21), Pt(1),
                        h - Inches(0.14), fill=BRAND_DIM, radius=False)
            rule.line.fill.background()
        tf = _tf(slide, x + Inches(0.3), yy, tw, h)
        _text(tf, [(head + "  ", {"color": PAPER, "bold": True}), (body, {"color": MUTED})],
              size=size, space_after=0, line=1.15)
        yy += h
    return yy


def _panel(slide, x, y, w, title, paragraphs, accent=BRAND, size=13):
    """Un encadré titré dont la hauteur est MESURÉE sur son contenu.

    Une hauteur fixe laissait le dernier paragraphe déborder du cadre, ce qui ne se
    voit qu'une fois projeté."""
    inner = w - Inches(0.52)
    body = [paragraphs] if isinstance(paragraphs, str) else list(paragraphs)
    h = _height(title, inner, size, bold=True) + Inches(0.66)
    for para in body:
        h += _height(para, inner, size, line=1.26) + Inches(0.14)
    # Borné au-dessus du pied de page : un encadré qui le dépasse déséquilibre la
    # diapositive, même quand les deux ne se recouvrent pas.
    h = min(h, H - Inches(0.72) - y)
    _box(slide, x, y, w, h, fill=INK_SOFT, line=accent)
    tf = _tf(slide, x + Inches(0.26), y + Inches(0.2), inner, h - Inches(0.4))
    _text(tf, title, size=size, color=accent, bold=True, space_after=8)
    for i, para in enumerate(body):
        _text(tf, para, size=size, color=MUTED if i < len(body) - 1 or len(body) == 1 else PAPER,
              space_after=0 if i == len(body) - 1 else 10, line=1.26)
    return y + h


def _scatter(slide, x, y, w, h, groups, seed=7):
    """Un nuage de points groupé, pour montrer ce qu'est une projection de corpus.

    Les positions sont tirées d'un générateur à graine fixe : la figure est la même
    d'une génération à l'autre, ce qu'une illustration doit être."""
    import math
    import random as _random
    rng = _random.Random(seed)
    for (cxf, cyf, spread, n, colour, label) in groups:
        gx, gy = x + int(w * cxf), y + int(h * cyf)
        # Une seule échelle pour les deux axes, sinon les groupes sont aplatis
        # par le rapport largeur/hauteur du cadre et ne ressemblent plus à rien.
        scale = min(w, h)
        d = Inches(0.075)
        for _ in range(n):
            a = rng.uniform(0, 2 * math.pi)
            r = abs(rng.gauss(0, 1)) * spread
            # Borné au cadre : un point tiré loin de son centre ne doit pas finir
            # sous le pied de page.
            px = min(max(gx + int(math.cos(a) * r * scale), x), x + w - d)
            py = min(max(gy + int(math.sin(a) * r * scale), y), y + h - d)
            dot = _box(slide, px, py, d, d, fill=colour)
            dot.line.fill.background()
        if label:
            ly = min(max(gy - int(0.34 * scale), y), y + h - Inches(0.26))
            tf = _tf(slide, gx - Inches(0.9), ly, Inches(1.8), Inches(0.26))
            _text(tf, label, size=9.5, color=colour, bold=True, align=PP_ALIGN.CENTER,
                  space_after=0)
    return y + h


def _stacked(slide, x, y, w, segments, h=Inches(0.42), colours=None):
    """Une barre empilée : des parts d'un même tout, en une seule ligne.

    Six barres séparées disaient la même chose en six fois plus de hauteur."""
    total = sum(n for _, n in segments) or 1
    palette = colours or [BRAND, RGBColor(0x5F, 0xA8, 0x86), GOLD,
                          RGBColor(0x6E, 0x9E, 0xE8), RGBColor(0xB4, 0x8A, 0xD4),
                          BRAND_DIM]
    cx = x
    for i, (label, n) in enumerate(segments):
        seg_w = int(w * n / total) if i < len(segments) - 1 else (x + w - cx)
        if seg_w <= 0:
            continue
        box = _box(slide, cx, y, seg_w, h, fill=palette[i % len(palette)], radius=False)
        box.line.color.rgb = INK
        box.line.width = Pt(1.25)
        if seg_w > Inches(0.5):
            tf = _tf(slide, cx, y + Inches(0.07), seg_w, h, anchor=MSO_ANCHOR.TOP)
            _text(tf, str(n), size=11, color=INK, bold=True, align=PP_ALIGN.CENTER,
                  space_after=0)
        cx += seg_w
    # La légende sous la barre, sur une ligne.
    lx = x
    for i, (label, n) in enumerate(segments):
        chip = _box(slide, lx, y + h + Inches(0.14), Inches(0.12), Inches(0.12),
                    fill=palette[i % len(palette)], radius=False)
        chip.line.fill.background()
        tw = Emu(int(len(label) * 9 * _CHAR_W * 12700) + Inches(0.1))
        tf = _tf(slide, lx + Inches(0.18), y + h + Inches(0.08), tw, Inches(0.24))
        _text(tf, label, size=9, color=MUTED, space_after=0)
        lx += Inches(0.18) + tw + Inches(0.16)
    return y + h + Inches(0.42)


# ── Les diapositives ─────────────────────────────────────────────────────────
def build(uc: UseCase, out: str, live: bool, shots: str = "") -> str:
    import os

    def shot(name):
        return os.path.join(shots, f"{name}.png") if shots else ""

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # 1 ── Titre
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = _box(s, 0, 0, W, H, fill=INK, radius=False)
    bg.line.fill.background()
    band = _box(s, 0, 0, Inches(0.16), H, fill=BRAND, radius=False)
    band.line.fill.background()
    tf = _tf(s, MARGIN, Inches(1.9), W - 2 * MARGIN, Inches(3.4))
    _text(tf, "LITEREV", size=13, color=BRAND, bold=True, space_after=8)
    _text(tf, "Evidence to Scenario", size=54, color=PAPER, bold=True, space_after=14,
          line=1.0)
    _text(tf, "From a question, to the whole literature on it, to a graded answer, to a "
              "model you can run and a dashboard a practitioner can watch.",
          size=18, color=MUTED, space_after=26, line=1.25)
    _text(tf, [("Institut de Santé Globale", {"color": PAPER, "bold": True}),
               ("   ·   Université de Genève", {"color": MUTED})], size=13, space_after=0)

    # 2 ── Le problème
    s, y = _slide(prs, "A systematic review takes a year. A decision does not wait.",
                  kicker="The problem")
    y = _cards(s, y, [
        ("Reading does not scale",
         "A reviewer reads a few hundred abstracts; the corpus holds thousands. "
         "Whatever goes unread is absent from the conclusion, and nothing on the page "
         "says which part that was."),
        ("The tools that help, sample",
         "Synthesis tools summarise a few dozen of the eligible papers. An empty cell "
         "in their gap table then means \"none of those few dozen\", which is not the "
         "same claim at all."),
        ("Certainty is asserted",
         "A finding resting on two case reports is printed like one resting on three "
         "randomised trials, and nothing on the page separates them."),
        ("The output is inert",
         "A PDF cannot be re-run next month, cannot be audited, and cannot hand its "
         "parameters to a model or a dashboard."),
        ("Evidence and operations stay apart",
         "What the literature says about an indicator, and what that indicator is "
         "doing in your region this week, live in different tools."),
        ("Nothing is reproducible",
         "Re-run the question six months later and nobody can say what moved: the "
         "literature, the search, or the reviewer."),
    ], cols=3)

    # 3 ── L'arc complet
    s, y = _slide(prs, "One question in. A review, a model and a dashboard out.",
                  kicker="The whole arc")
    y = _flow(s, y, [
        ("01", "Ask", "Plain language or boolean. Translated deterministically and stored."),
        ("02", "Assemble", "Bibliographic APIs plus the local base, deduplicated, PRISMA-counted."),
        ("03", "Rank", "Semantic score across the whole corpus, then a cross-encoder rerank."),
        ("04", "Screen", "Threshold, reviewer decisions, double-blind where it is required."),
        ("05", "Extract", "PICO, design, concepts and parameters, cached per article."),
        ("06", "Synthesise", "Graded claims, gap matrix, clusters, knowledge graph, report."),
        ("07", "Model", "Outcome, variables, algorithm, tuning, SEIR, validation."),
        ("08", "Monitor", "Field data beside the evidence, alerts, and a living review."),
    ])
    y += Inches(0.3)
    _bullets(s, MARGIN, y, W - 2 * MARGIN, [
        "Steps 1 to 6 are the review. Steps 7 and 8 are what makes it operational, and are the part no synthesis tool offers.",
        "Every step writes to the same database, so each one can be re-run on its own without redoing the ones before it.",
    ], size=14)

    # 4 ── Capture : l'espace de travail
    s, y = _slide(prs, "The workspace", kicker="One scenario, one page")
    _shot(s, shot("workspace"), MARGIN, y - Inches(0.1), W - 2 * MARGIN,
          H - y - Inches(0.9),
          caption="Threshold, corpus, screening, search inside the corpus, and the "
                  "profile of what was retrieved, on one screen")

    # 5 ── De la question à la requête
    s, y = _slide(prs, "The query the databases actually received",
                  kicker="Search strategy")
    sw = int((W - 2 * MARGIN) * 0.54)
    bx = MARGIN + sw + Inches(0.42)
    bw = W - MARGIN - bx
    # La colonne de texte d'abord : sa hauteur dit où centrer la capture, qui est
    # large et courte. Alignée en haut, elle laissait un vide sous elle.
    pipe_bottom = _pipe(s, bx, y + Inches(0.1), bw, [
        ("Translated.", "A question in plain language becomes a boolean expression at "
                        "temperature zero with a fixed seed, and is cached."),
        ("Reproducible.", "The same phrasing always yields the same strategy, so the "
                          "corpus can be rebuilt identically."),
        ("Shaped per source.", "MeSH for PubMed, a portable boolean for the APIs that "
                               "accept operators, keywords for those that do not."),
        ("Widened for recall.", "PubMed is queried on the union of the MeSH expression "
                                "and the plain boolean: a MeSH query alone returned 35 "
                                "results where the boolean returned 306."),
        ("Shown, not hidden.", "A reviewer can paste it into PubMed and obtain the same "
                               "set. That is what makes the corpus auditable."),
    ])
    from PIL import Image as _Img
    _Img.MAX_IMAGE_PIXELS = None
    _band = pipe_bottom - y
    try:
        _iw, _ih = _Img.open(shot("strategy")).size
        _dh = int(sw * _ih / _iw)
    except Exception:                                             # noqa: BLE001
        _dh = int(_band)
    _shot(s, shot("strategy"), MARGIN, y + max(0, int((_band - _dh) / 2)), sw, _band)

    # 6 ── Le corpus
    s, y = _slide(prs, "What is in the corpus, and why",
                  kicker="Corpus assembly",
                  subtitle="Membership is a lexical property of the boolean query, "
                           "independent of any semantic score.")
    y = _flow(s, y, [
        ("1", "Local match", "Everything already in the base that satisfies the query."),
        ("2", "Live union", "Plus what the boolean-native sources returned, unfiltered."),
        ("3", "Re-match", "Keyword-source results re-checked against the boolean query."),
        ("4", "Quality rule", "No abstract, no entry: an unreadable record is not evidence."),
        ("5", "Deduplicate", "One link per distinct article, across sources and runs."),
        ("6", "Freeze", "What arrives later belongs to the next run, not this one."),
    ])
    y += Inches(0.26)
    bx, bw, cx, cw = MARGIN, int((W - 2 * MARGIN) * 0.48), MARGIN + int((W - 2 * MARGIN) * 0.48) + Inches(0.42), 0
    cw = W - MARGIN - cx
    tf = _tf(s, bx, y, bw, Inches(0.3))
    _text(tf, "SOURCES QUERIED IN PARALLEL", size=9.5, color=BRAND, bold=True, space_after=0)
    _chips(s, bx, y + Inches(0.34), bw, [
        "PubMed", "Europe PMC", "OpenAlex", "Crossref", "Semantic Scholar",
        "DOAJ", "CORE", "OpenAIRE", "ClinicalTrials.gov", "bioRxiv", "medRxiv",
        "arXiv", "local base",
    ], size=9.5)
    _bullets(s, cx, y + Inches(0.06), cw, [
        "The corpus is reset to the boolean match on each run, so stale links cannot accumulate.",
        "A legitimately empty result may empty the corpus; a transient source failure may not.",
        "Lowering the threshold brings articles back: it filters, it never deletes.",
        "A free-text search runs inside the corpus, over titles, abstracts, authors, journals, keywords and identifiers, accents ignored, and can be held to the relevant subset alone.",
    ], size=13.5)

    # 7 ── Pertinence et sélection
    s, y = _slide(prs, "Relevance, screening, and one definition of \"relevant\"",
                  kicker="Ranking and screening")
    y = _cards(s, y, [
        ("Scored in full",
         "Every article in the corpus gets a cosine score against the question, reusing "
         "stored embeddings and embedding the rest on the fly."),
        ("Reranked where it matters",
         "A cross-encoder refines the ordering of the relevant subset, where the gap "
         "between rank 5 and rank 50 changes what gets read."),
        ("The shared gate",
         "Relevant means: never a duplicate, never excluded by a reviewer, and otherwise "
         "included by hand OR above the threshold. One SQL predicate, used by every "
         "panel and every extraction."),
        ("Reviewer decisions win",
         "An article rescued below the threshold feeds the analyses; one excluded above "
         "it does not. Screening is per scenario, not per document."),
        ("Double-blind when needed",
         "Two reviewers, blind to each other, disagreements surfaced. The path a formal "
         "systematic review requires."),
        ("PRISMA, counted",
         "Identified, duplicates removed, removed for other reasons, screened: taken "
         "from the links actually merged, not from a flag nobody sets."),
    ], cols=3)

    # 8 ── Lire tout le corpus
    s, y = _slide(prs, "Every extraction reads every relevant article",
                  kicker="The rule the rest depends on",
                  subtitle="Not a sample, not a top twenty, not the fifty the budget allowed.")
    bx, bw, cx, cw = MARGIN, int((W - 2 * MARGIN) * 0.52), 0, 0
    cx = MARGIN + bw + Inches(0.44)
    cw = W - MARGIN - cx
    _pipe(s, bx, y, bw, [
        ("The constraint.", "Several thousand abstracts do not fit in one prompt. Every "
                            "tool meets this wall; most answer it by sampling quietly."),
        ("Map.", "Each article's facts are extracted once and cached on its row: PICO, "
                 "study design, concepts, epidemiological parameters. Paid once, "
                 "incremental afterwards."),
        ("Reduce.", "The aggregation runs in SQL over the entire relevant subset. No "
                    "model call, no sampling, no ceiling."),
        ("Write.", "The generator writes over that digest and reproduces a handful of "
                   "articles for quotation, under an instruction that its conclusions "
                   "must hold for the whole corpus."),
    ])
    _panel(s, cx, y, cw, "What it buys", [
        "Counted over the whole relevant subset, an empty cell in the gap matrix says "
        "something a sample cannot: no article in this corpus studies these two things "
        "together. A design profile describes the corpus rather than the sample. A "
        "claim's certainty is bounded by what the evidence actually contains.",
        "Every figure downstream inherits this property, which is why it is a rule and "
        "not a setting.",
    ], accent=GOLD)

    # 9 ── Capture : profils du corpus
    s, y = _slide(prs, "What the corpus is made of", kicker="Evidence profile",
                  subtitle="Study designs, sources and certainty levels, counted over "
                           "every relevant article, and usable as a filter.")
    _shot(s, shot("evidence"), MARGIN, y, W - 2 * MARGIN, H - y - Inches(0.95),
          caption="Clicking a design or a level narrows the corpus to it, the way a "
                  "cluster selection does, and the choice is reversible")

    # 10 ── Niveaux de preuve
    s, y = _slide(prs, "Sixteen study designs, four certainties, from official lists",
                  kicker="Study design and GRADE",
                  subtitle="Taken from the MeSH publication-type and epidemiologic study "
                           "trees. A level is the CEILING the design allows, not an "
                           "appraisal: bias, inconsistency and imprecision only lower it.")
    gg = grade_groups()
    _join = lambda k: ", ".join(gg.get(k) or []) + "."
    # Les devis viennent de la table ; la raison est écrite ici, courte, parce
    # que celle de l'écran fait un paragraphe et qu'une carte n'en veut pas.
    _why = {
        "High": "GRADE starts randomised evidence at the highest certainty.",
        "Moderate": "An intervention study that does not state whether it randomised.",
        "Low": "Observational evidence starts low.",
        "Very low": "No comparison group and no reproducible selection method: no "
                    "effect estimate.",
    }
    y = _cards(s, y, [(k, f"{_join(k)} {_why[k]}")
                      for k in ("High", "Moderate", "Low", "Very low")],
               cols=4, accent=GOLD)
    y += Inches(0.24)
    _bullets(s, MARGIN, y, W - 2 * MARGIN, [
        "A synthesis inherits the certainty of what it includes rather than upgrading it: "
        "high over randomised trials, low over observational ones.",
        f"{_join('Not applicable')[:-1]}: not applicable rather than graded, not being "
        f"primary evidence. {_join('Not assessed')[:-1]}: not assessed, which is not the "
        f"same as graded low.",
    ], size=14)

    # 11 ── Affirmations et rapport
    s, y = _slide(prs, "Claims, each carrying the evidence behind it",
                  kicker="Synthesis")
    y = _cards(s, y, [
        ("A claim, not a paragraph",
         "Each statement carries the articles supporting it as numbered references, "
         "title, journal and year, each opening the paper itself."),
        ("Graded on its own support",
         "Certainty follows the designs that actually support the claim, capped by what "
         "the corpus as a whole can sustain."),
        ("A ceiling the corpus imposes",
         "A corpus holding no randomised evidence cannot yield a high-certainty claim, "
         "whatever the abstracts assert."),
        ("Expires when the corpus moves",
         "The cached synthesis carries a fingerprint of the threshold and the article "
         "identifiers, so it cannot describe a corpus that has since changed."),
        ("A citable document",
         "Renumbered references, methods, figures, downloadable as PDF or Markdown and "
         "ready to paste into a paper."),
        ("A bibliography that imports",
         "RIS, BibTeX, CSV, Excel, JSON. The RIS lands in Zotero with journals, authors "
         "and DOIs in the right fields."),
    ], cols=3)

    # 12 ── Matrice des manques
    s, y = _slide(prs, "The gap matrix: what nobody has studied together",
                  kicker="Finding the hole")
    bx, bw = MARGIN, int((W - 2 * MARGIN) * 0.5)
    cx = MARGIN + bw + Inches(0.44)
    cw = W - MARGIN - cx
    # Une petite matrice dessinée : la figure vaut mieux que sa description.
    rows = ["vector control", "vaccination", "surveillance", "case management"]
    cols = ["incidence", "severity", "mortality", "cost"]
    cells = {(0, 0): 34, (0, 1): 11, (0, 2): 4, (0, 3): 0,
             (1, 0): 22, (1, 1): 19, (1, 2): 7, (1, 3): 3,
             (2, 0): 41, (2, 1): 6, (2, 2): 2, (2, 3): 0,
             (3, 0): 8, (3, 1): 26, (3, 2): 13, (3, 3): 1}
    lab_w = Inches(1.35)
    cell = Inches(0.78)
    gx, gy = bx + lab_w, y + Inches(0.42)
    for j, c in enumerate(cols):
        tf = _tf(s, gx + j * cell, y + Inches(0.06), cell, Inches(0.34))
        _text(tf, c, size=8, color=MUTED, align=PP_ALIGN.CENTER, space_after=0)
    peak = max(cells.values()) or 1
    for i, r in enumerate(rows):
        tf = _tf(s, bx, gy + i * cell + Inches(0.22), lab_w - Inches(0.1), Inches(0.34))
        _text(tf, r, size=9, color=PAPER, align=PP_ALIGN.RIGHT, space_after=0)
        for j in range(len(cols)):
            n = cells[(i, j)]
            box = _box(s, gx + j * cell + Emu(9000), gy + i * cell + Emu(9000),
                       cell - Emu(18000), cell - Emu(18000),
                       fill=INK_SOFT if not n else BRAND_DIM, line=GOLD if not n else None)
            if n:
                box.fill.solid()
                t = 0.25 + 0.75 * (n / peak)
                box.fill.fore_color.rgb = RGBColor(int(0x1E + (0x3F - 0x1E) * t),
                                                   int(0x5C + (0xB9 - 0x5C) * t),
                                                   int(0x40 + (0x7E - 0x40) * t))
            tf = _tf(s, gx + j * cell, gy + i * cell + Inches(0.22), cell, Inches(0.34))
            _text(tf, str(n) if n else "gap", size=10 if n else 8,
                  color=INK if n and n > peak * 0.5 else (GOLD if not n else PAPER),
                  bold=True, align=PP_ALIGN.CENTER, space_after=0)
    _bullets(s, cx, y, cw, [
        "Every cell is counted. A zero means no article in this corpus pairs these two concepts, which is itself a finding.",
        "The matrix declares how many relevant articles it could not read, because nothing was extracted from them.",
        "A truncated axis keeps its untruncated count on screen, so the grid never looks more complete than it is.",
        "A gap is only claimed inside the shown grid; outside it a zero may be a label that was cut.",
        "Ties break by label, so a matrix on identical data does not reorder itself between runs.",
        "A reviewer's exclusion empties a cell, as it should.",
    ], size=13.5)

    # 13 ── Clustering
    s, y = _slide(prs, "Clustering: the shape of the literature",
                  kicker="Topic structure",
                  subtitle="What themes exist in this corpus, how big each is, and which "
                           "articles sit between them.")
    _shot(s, shot("clusters"), MARGIN, y, W - 2 * MARGIN, H - y - Inches(0.95),
          caption="Each cluster is a selectable subset: pick one and every downstream "
                  "panel describes that theme alone")

    # Le détail de la chaîne, sur sa propre diapositive.
    s, y = _slide(prs, "How the clusters are found", kicker="Topic structure",
                  subtitle="Found rather than chosen: the number of topics follows the "
                           "density of the corpus, not a parameter someone picked.")
    bx, bw = MARGIN, int((W - 2 * MARGIN) * 0.52)
    cx = MARGIN + bw + Inches(0.44)
    cw = W - MARGIN - cx
    pipe_bottom = _pipe(s, bx, y, bw, [
        ("Embeddings.", "The vectors already stored for the corpus, with a TF-IDF "
                        "fallback so it works without a model call."),
        ("UMAP.", "Reduction to two dimensions, under a timeout."),
        ("HDBSCAN.", "Density clustering, minimum cluster size scaled to the corpus. "
                     "K-means over a truncated SVD takes over if either step fails."),
        ("Summaries.", "A label and a short description per cluster, in the reader's "
                       "language."),
    ])
    panel_bottom = _panel(s, cx, y, cw, "What it is for", [
        "A cluster is a selectable subset. Picking one narrows the corpus to that theme, "
        "and every downstream panel, the brief, the variables, the gap matrix, then "
        "describes that theme alone.",
        "It is also how a reviewer discovers that a question asked as one thing is in "
        "fact three separate literatures.",
    ])
    fig_y = max(pipe_bottom, panel_bottom) + Inches(0.3)
    tf = _tf(s, MARGIN, fig_y, W - 2 * MARGIN, Inches(0.28))
    _text(tf, "WHAT IT FOUND IN THE WORKED EXAMPLE", size=9.5, color=BRAND, bold=True,
          space_after=0)
    _clustered = sum(n for _, n in uc.clusters)
    _fringe = max(0, uc.above_threshold - _clustered)
    _stacked(s, MARGIN, fig_y + Inches(0.34), W - 2 * MARGIN,
             [(lbl.replace("CLUSTER ", "Cluster "), n) for lbl, n in uc.clusters]
             + [(f"Unclustered ({_fringe})", _fringe)])

    # 14 ── Graphe de connaissances
    s, y = _slide(prs, "The knowledge graph: what sits next to what",
                  kicker="Similarity network")
    bx, bw = MARGIN, int((W - 2 * MARGIN) * 0.44)
    cx = MARGIN + bw + Inches(0.44)
    cw = W - MARGIN - cx
    # Un petit graphe dessiné.
    import math as _math
    cxn, cyn, rad = int(bx + bw / 2), int(y + Inches(1.95)), Inches(1.6)
    nodes = []
    for i in range(9):
        a = 2 * _math.pi * i / 9 - _math.pi / 2
        nodes.append((cxn + int(rad * _math.cos(a)), cyn + int(rad * _math.sin(a))))
    groups = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    cols_g = [BRAND, GOLD, RGBColor(0x6E, 0x9E, 0xE8)]
    for i, (nx, ny) in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            if groups[i] == groups[j] or (i + j) % 7 == 0:
                mx, my = nodes[j]
                conn = s.shapes.add_connector(1, nx, ny, mx, my)
                conn.line.color.rgb = BRAND_DIM
                conn.line.width = Pt(0.75)
    for i, (nx, ny) in enumerate(nodes):
        d = Inches(0.24) if i % 3 else Inches(0.34)
        dot = _box(s, nx - d // 2, ny - d // 2, d, d, fill=cols_g[groups[i]])
        dot.line.fill.background()
    _bullets(s, cx, y, cw, [
        "Articles are nodes; an edge means their abstracts are close enough in meaning to be about the same thing.",
        "Communities fall out of the edges and are labelled from the titles they contain, so a theme gets a name without a model call.",
        "It answers a different question from clustering: not \"what themes exist\" but \"which papers are the bridges, and which sit alone\".",
        "An isolated node with a high relevance score is usually either a mis-filed article or the only paper on something.",
        "Like clustering, it is a projection, and the interface says how many articles it covers.",
    ], size=14)

    # 15 ── Du texte aux variables
    s, y = _slide(prs, "From abstracts to candidate variables",
                  kicker="The bridge",
                  subtitle="The step that turns a literature review into something a "
                           "model can be built from.")
    y = _flow(s, y, [
        ("1", "PICO per article", "Population, intervention, comparator, outcome, cached on the row."),
        ("2", "Concepts", "Normalised concept terms per article, typed as intervention, outcome, population or setting."),
        ("3", "Aggregate", "Counted in SQL across every relevant article: which concepts this literature actually uses."),
        ("4", "Propose", "Candidate predictors and outcomes, each with the articles that justify it."),
        ("5", "Decide", "The reviewer keeps, renames, retypes or drops each one. Nothing is adopted silently."),
    ])
    y += Inches(0.3)
    _bullets(s, MARGIN, y, W - 2 * MARGIN, [
        "A proposed variable that no article supports does not appear: the provenance is the point, not the suggestion.",
        "Each variable carries a machine name, a data type and a source, so the spec that follows is directly usable as a schema.",
        "Epidemiological parameters are extracted the same way: R0, incubation, serial interval, case fatality, reported with their spread across the corpus rather than as a single borrowed number.",
    ], size=14)

    # 16 ── Les outcomes
    s, y = _slide(prs, "Defining the outcome properly, before anything is fitted",
                  kicker="Outcomes",
                  subtitle="The step most often rushed, and the one that decides whether "
                           "the model answers a real question.")
    bx, bw = MARGIN, int((W - 2 * MARGIN) * 0.5)
    cx = MARGIN + bw + Inches(0.44)
    cw = W - MARGIN - cx
    _pipe(s, bx, y, bw, [
        ("Named and typed.", "A target carries a machine name, a task type, a unit and, "
                             "for a classification, its positive class."),
        ("Ready-made templates.", "Emergency-department overload, bed occupancy, incoming "
                                  "call volume, call surge: well-specified targets applied "
                                  "in one click."),
        ("A template brings its own.", "Suggested predictors, a fitting algorithm family, "
                                       "the metric that suits the target, and alert bands "
                                       "where they make sense."),
        ("A data schema falls out.", "The template produces the column list the operator "
                                     "must supply, so an upload can be validated instead "
                                     "of guessed."),
    ])
    tf = _tf(s, cx, y, cw, Inches(0.3))
    _text(tf, "TASK TYPES", size=9.5, color=BRAND, bold=True, space_after=0)
    yy = _chips(s, cx, y + Inches(0.34), cw, ["classification", "regression", "count", "survival"])
    tf = _tf(s, cx, yy + Inches(0.26), cw, Inches(0.3))
    _text(tf, "METRICS", size=9.5, color=BRAND, bold=True, space_after=0)
    yy = _chips(s, cx, yy + Inches(0.6), cw,
                ["roc_auc", "average_precision", "rmse", "mae", "r2", "c_index"])
    tf = _tf(s, cx, yy + Inches(0.26), cw, Inches(0.3))
    _text(tf, "VARIABLE TYPES", size=9.5, color=BRAND, bold=True, space_after=0)
    _chips(s, cx, yy + Inches(0.6), cw, ["float", "int", "bool", "category", "datetime"])

    # 17 ── Choix de l'algorithme
    s, y = _slide(prs, "Choosing the algorithm, and saying why",
                  kicker="Model specification",
                  subtitle="The family follows the task and the data, not fashion. The "
                           "spec is editable, and every field is explicit.")
    tf = _tf(s, MARGIN, y, W - 2 * MARGIN, Inches(0.3))
    _text(tf, "ALGORITHM FAMILIES", size=9.5, color=BRAND, bold=True, space_after=0)
    yy = _chips(s, MARGIN, y + Inches(0.34), W - 2 * MARGIN, [
        "gradient_boosting", "lightgbm", "xgboost", "random_forest", "extremal_rf",
        "logistic_regression", "linear_regression", "elasticnet", "svm", "mlp",
        "knn", "cox_ph", "prophet", "sarimax",
    ], size=10)
    yy += Inches(0.3)
    yy = _cards(s, yy, [
        ("Tabular and tree-based",
         "Boosting and forests for mixed operational tables, where interactions matter "
         "and the sample is modest."),
        ("Linear and regularised",
         "When the coefficient itself is the deliverable and has to be defensible to a "
         "committee."),
        ("Time series",
         "Prophet and SARIMAX are routed out of the tabular path: a dated target is a "
         "forecasting problem, not a regression on rows."),
        ("Survival",
         "Cox proportional hazards with the concordance index, for time-to-event "
         "outcomes rather than a yes or no at one horizon."),
        ("Extremal",
         "A quantile random forest for surge and overload: it predicts the upper tail, "
         "scored with pinball loss, because the mean is not what breaks a service."),
        ("Comparison, not assertion",
         "Several families can be fitted on the same data and ranked on the same metric, "
         "with each run's parameters kept."),
    ], cols=3)

    # 18 ── Entraînement et hyperparamètres
    s, y = _slide(prs, "Training, tuning and validation",
                  kicker="Fitting the model",
                  subtitle="Hyperparameters are searched, not guessed, and the search is "
                           "recorded with the model.")
    bx, bw = MARGIN, int((W - 2 * MARGIN) * 0.5)
    cx = MARGIN + bw + Inches(0.44)
    cw = W - MARGIN - cx
    _pipe(s, bx, y, bw, [
        ("Bayesian search.", "Optuna proposes parameter sets over a defined trial budget, "
                             "concentrating on the promising region rather than sweeping "
                             "a grid."),
        ("Cross-validated.", "Stratified k-fold for classification, k-fold for regression, "
                             "and a time-series split when the target is dated, so no "
                             "future row leaks into training."),
        ("Scored on the stated metric.", "The metric chosen with the outcome is the one "
                                         "optimised: average precision for a rare event, "
                                         "pinball for a quantile, c-index for survival."),
        ("Kept, with its parameters.", "Every run stores its family, its best "
                                       "hyperparameters, its metrics and its feature "
                                       "importances, and can be exported."),
    ])
    _panel(s, cx, y, cw, "Searched per family", [
        "Boosting: number of estimators, learning rate, depth, subsample, column sample.",
        "Forests: estimators, depth, minimum samples per split and per leaf.",
        "Regularised linear: penalty strength and L1 ratio. SVM: C and kernel coefficient.",
        "Neural: hidden layout and weight decay. Nearest neighbours: count and weighting.",
        "Feature importances come back with the model, so the variables the literature "
        "proposed can be checked against the ones the data actually used.",
    ])

    # Capture : la spécification telle qu'elle est produite.
    s, y = _slide(prs, "The specification the literature produces",
                  kicker="Model specification",
                  subtitle="An outcome defined to the unit and the time horizon, with "
                           "interpretation bands, each carrying the articles behind it.")
    _shot(s, shot("model"), MARGIN, y, W - 2 * MARGIN, H - y - Inches(0.95),
          caption="Generated from the relevant articles, and withheld until a "
                  "reviewer validates it")

    # 19 ── SEIR
    s, y = _slide(prs, "SEIR, parameterised from the literature",
                  kicker="Mechanistic models",
                  subtitle="A compartmental projection whose assumptions are traceable to "
                           "the papers they came from.")
    bx, bw = MARGIN, int((W - 2 * MARGIN) * 0.5)
    cx = MARGIN + bw + Inches(0.44)
    cw = W - MARGIN - cx
    # Les compartiments.
    comps = [("S", "susceptible"), ("E", "exposed"), ("I", "infectious"), ("R", "removed")]
    bwid = int((bw - Inches(0.3) * 3) / 4)
    for i, (letter, label) in enumerate(comps):
        x = bx + i * (bwid + Inches(0.3))
        _box(s, x, y, bwid, Inches(0.92), fill=INK_SOFT, line=BRAND)
        tf = _tf(s, x, y + Inches(0.12), bwid, Inches(0.7))
        _text(tf, letter, size=22, color=BRAND, bold=True, align=PP_ALIGN.CENTER,
              space_after=0)
        _text(tf, label, size=8.5, color=MUTED, align=PP_ALIGN.CENTER, space_after=0)
        if i < 3:
            arr = s.shapes.add_connector(2, x + bwid, y + Inches(0.46),
                                         x + bwid + Inches(0.3), y + Inches(0.46))
            arr.line.color.rgb = BRAND_DIM
            arr.line.width = Pt(1.5)
    yy = y + Inches(1.2)
    _bullets(s, bx, yy, bw, [
        "Parameters are drawn from the distributions extracted across the relevant corpus, not from one convenient paper.",
        "Where the corpus gives no R0, the projection says so rather than falling back on a hard-coded value.",
        "The exposed population is derived from the geography the corpus is actually about.",
        "Observed case data can be overlaid, and the model calibrated against it.",
    ], size=13.5)
    _panel(s, cx, y, cw, "A mechanistic model as an input to a statistical one", [
        "The projection is not only an output. Its compartments can be fed back as "
        "predictors: projected incidence, the share of the population still susceptible, "
        "days since the inflection.",
        "That is what the \"seir\" feature source means in a model specification. A "
        "demand forecast for a service then rests on operational variables, public data, "
        "and the epidemic state the literature-parameterised model implies, together.",
        "The variants carrying vaccination and quarantine compartments extend the same "
        "idea to interventions.",
    ], accent=GOLD)

    # 20 ── Revue vivante (capture)
    s, y = _slide(prs, "A review that keeps itself current",
                  kicker="Living review",
                  subtitle="The same question, re-asked on a schedule, with only what "
                           "changed brought to your attention.")
    _shot(s, shot("living"), MARGIN, y, int((W - 2 * MARGIN) * 0.58), H - y - Inches(0.8))
    bx = MARGIN + int((W - 2 * MARGIN) * 0.58) + Inches(0.42)
    bwr = W - MARGIN - bx
    _pipe(s, bx, y + Inches(0.1), bwr, [
        ("Re-queried.", "The scenario's own query is re-run against every source, and "
                        "new references are inserted and deduplicated."),
        ("Re-indexed.", "New articles are embedded, their PICO extracted, their full text "
                        "fetched where it is open."),
        ("Re-derived.", "Clustering and the derived figures are recomputed; the cached "
                        "synthesis expires rather than describing the old corpus."),
        ("Reported.", "Email digests, immediate, daily or weekly, and a dry run that "
                      "shows what an update would change before it changes it."),
    ])

    # 21 ── Tableau de bord praticien
    s, y = _slide(prs, "A monitoring dashboard a practitioner can actually watch",
                  kicker="Where this is heading",
                  subtitle="Evidence, field data and a fitted model on one screen, "
                           "refreshing on their own.")
    y = _cards(s, y, [
        ("What the evidence says",
         "The graded claims for this question, kept current by the living review, with "
         "the certainty of each visible at a glance."),
        ("What the field is doing",
         "Situation reports and operational series beside the literature, on the same "
         "page and the same timeline."),
        ("What the model expects",
         "The trained model's prediction for the coming period, with its intervals and "
         "the variables driving it."),
        ("What the mechanism implies",
         "The SEIR projection for the same geography, calibrated against observed cases."),
        ("When to look up",
         "Alert bands on the predicted outcome, and a notification when new evidence "
         "changes a claim rather than when any paper appears."),
        ("Why it can be trusted",
         "Every number traces back: to the articles counted, the screening decisions "
         "taken, the parameters searched. Nothing on the dashboard is unattributable."),
    ], cols=3)

    # 22 ── L'assistant
    s, y = _slide(prs, "Asking the corpus a question", kicker="Retrieval",
                  subtitle="The fastest way to interrogate a corpus you have just built, "
                           "and the one place where a conversational answer is the right "
                           "shape.")
    y = _cards(s, y, [
        ("Scoped to what counts",
         "The question is answered through the same relevance gate as every other "
         "panel: never a duplicate, never an article a reviewer excluded."),
        ("Figures from the whole corpus",
         "Any statement of how many, which years or which designs comes from the "
         "aggregation over every relevant article. The retrieved passages supply the "
         "quotations, never the counts."),
        ("Passage-level where it can be",
         "Where full text is open it is indexed passage by passage, so an answer can "
         "quote a results section rather than a summary, and names the article."),
    ], cols=3)
    y += Inches(0.26)
    _bullets(s, MARGIN, y, W - 2 * MARGIN, [
        "Narrow the corpus first, by cluster, design or certainty level, and the same question is answered of that subset alone.",
        "An answer resting on nothing above the threshold says so, instead of producing an unsupported paragraph.",
    ], size=14)

    # Les questions deviennent un actif du scénario.
    s, y = _slide(prs, "Questions that are kept, exported and acted on",
                  kicker="From an answer to a change",
                  subtitle="An answer is a dated result on a named corpus, not a chat "
                           "message that scrolls away.")
    bx, bw = MARGIN, int((W - 2 * MARGIN) * 0.5)
    cx = MARGIN + bw + Inches(0.44)
    cw = W - MARGIN - cx
    _pipe(s, bx, y, bw, [
        ("Kept.", "Every question is stored with its answer, the scenario it was asked "
                  "on, the threshold and any narrowing in force, and the articles it "
                  "cited."),
        ("Comparable.", "The same question re-asked after the corpus has grown produces "
                        "a second dated answer beside the first, so a change in the "
                        "literature is visible rather than inferred."),
        ("Exportable.", "An answer leaves as a formatted document, references included, "
                        "ready for a report or a committee paper."),
        ("Actionable.", "Where an answer names a parameter that differs from the one the "
                        "scenario holds, the difference is proposed as a change the "
                        "reviewer accepts or rejects. Accepted, it is written where the "
                        "projection reads it, with its provenance."),
    ])
    _panel(s, cx, y, cw, "Why this matters", [
        "A question asked of a corpus is a small piece of research: it has a scope, a "
        "date and a set of sources. Treating it as a disposable chat turn throws all "
        "three away.",
        "Kept instead, the history becomes the record of what was asked of this "
        "evidence base and what it answered, which is what an audit, a co-author or a "
        "reviewer six months later actually needs.",
        "And an answer that can propose a parameter update closes the loop: the "
        "literature stops being something you read and becomes something that updates "
        "the model.",
    ], accent=GOLD)

    # 23 ── Capture : la liste des scénarios
    s, y = _slide(prs, "Many questions, side by side", kicker="Scenarios",
                  subtitle="Folders, pinned reviews and recent searches, each with its "
                           "own corpus, screening state and model, and searchable by "
                           "name or by the question asked.")
    _shot(s, shot("scenarios"), MARGIN, y, W - 2 * MARGIN, H - y - Inches(0.9),
          caption="Each card opens a full workspace, with its own corpus, screening "
                  "state and model")

    # 24 ── Cas d'usage : la question
    s, y = _slide(prs, uc.name, kicker="Worked example",
                  subtitle=f"Scenario {uc.scenario_id} · figures as of {uc.as_of}.")
    _n = lambda v: f"{v:,}".replace(",", " ")
    y = _stat_row(s, y, [
        (_n(uc.total), "articles in the corpus",
         f"{_n(uc.from_local)} already in the local base, {_n(uc.newly_fetched)} "
         f"fetched for this question"),
        (_n(uc.above_threshold), "feed the analyses",
         f"at or above a cosine similarity of {uc.threshold:.2f}"),
        (_n(uc.below_threshold), "below, and kept",
         "the threshold filters; lowering it brings them back"),
        (_n(uc.with_fulltext), "with full text",
         f"and {uc.journals} distinct journals across {len(uc.years)} years shown"),
    ])
    y += Inches(0.26)
    qbx, qbw = MARGIN, int((W - 2 * MARGIN) * 0.52)
    qcx = MARGIN + qbw + Inches(0.44)
    qcw = W - MARGIN - qcx
    _box(s, qbx, y, qbw, Inches(1.15), fill=INK_SOFT, line=BRAND_DIM)
    tf = _tf(s, qbx + Inches(0.24), y + Inches(0.18), qbw - Inches(0.48), Inches(0.8))
    _text(tf, "THE QUESTION, AS ASKED", size=9.5, color=BRAND, bold=True, space_after=5)
    _text(tf, uc.question, size=14, color=PAPER, space_after=0, line=1.18)
    _pipe(s, qcx, y + Inches(0.02), qcw, [
        ("Automatic so far.", "No reviewer has screened this selection yet, and every "
                              "page that uses it says so rather than implying otherwise."),
        ("Reversible.", "The threshold is a filter, not a deletion: the articles below "
                        "it are in the corpus and come back when it is lowered."),
    ])
    y += Inches(1.45)
    _bullets(s, MARGIN, y, W - 2 * MARGIN, [
        "Asked once in plain English. Sources queried in parallel, a boolean strategy generated and stored, a corpus assembled, deduplicated and scored against the question.",
    ], size=13.5)

    # 25 ── Cas d'usage : le corpus décrit
    s, y = _slide(prs, "The corpus, described", kicker="Worked example")
    half = int((W - 2 * MARGIN - Inches(0.5)) / 2)
    tf = _tf(s, MARGIN, y, half, Inches(0.3))
    _text(tf, f"PUBLICATION YEAR, {uc.years[0][0]} ONWARDS", size=9.5, color=BRAND,
          bold=True, space_after=0)
    _histogram(s, MARGIN, y + Inches(0.34), half, Inches(2.6), uc.years, label_every=4)
    rx = MARGIN + half + Inches(0.5)
    if uc.sources:
        tf = _tf(s, rx, y, half, Inches(0.3))
        _text(tf, "LITERATURE SOURCES", size=9.5, color=BRAND, bold=True, space_after=0)
        _bars(s, rx, y + Inches(0.4), half, uc.sources[:8], label_w=Inches(1.9),
              row_h=Inches(0.31))
    else:
        tf = _tf(s, rx, y, half, Inches(0.3))
        _text(tf, "WHERE THE CORPUS CAME FROM", size=9.5, color=BRAND, bold=True,
              space_after=0)
        _bars(s, rx, y + Inches(0.4), half, [
            ("LOCAL BASE", uc.from_local), ("FETCHED LIVE", uc.newly_fetched),
        ], label_w=Inches(1.5), row_h=Inches(0.34))
        tf = _tf(s, rx, y + Inches(1.24), half, Inches(0.3))
        _text(tf, "RELEVANCE AT %.2f" % uc.threshold, size=9.5, color=GOLD, bold=True,
              space_after=0)
        _bars(s, rx, y + Inches(1.64), half, [
            ("ABOVE", uc.above_threshold),
            ("BELOW, KEPT", uc.below_threshold),
            ("UNSCORED", uc.unscored),
        ], accent=GOLD, label_w=Inches(1.5), row_h=Inches(0.34))
    y += Inches(3.2)
    # Lu dans la distribution, et non affirmé : la phrase précédente classait les
    # quatre dernières années en tête, ce que les chiffres ne disent pas.
    _peak_year, _peak_n = max(uc.years, key=lambda r: r[1]) if uc.years else (0, 0)
    _prior = dict(uc.years).get(_peak_year - 1)
    _since = [n for yr, n in uc.years if yr > _peak_year]
    _bullets(s, MARGIN, y, W - 2 * MARGIN, [
        f"The corpus reaches back to {uc.year_min}; the axis starts at "
        f"{uc.years[0][0] if uc.years else YEAR_AXIS_FROM} because what precedes it is a "
        f"handful of articles a year.",
        f"The jump at {_peak_year} is the pandemic surveillance literature: {_peak_n} "
        f"articles against {_prior} the year before"
        + (f", and no year since has fallen below {min(_since)}." if _since else "."),
    ], size=14)

    # Cas d'usage : le profil de preuve, chiffré.
    s, y = _slide(prs, "What this corpus can and cannot support",
                  kicker="Worked example",
                  subtitle="The design and certainty profile of the "
                           f"{uc.above_threshold} articles that feed the analyses.")
    half = int((W - 2 * MARGIN - Inches(0.5)) / 2)
    tf = _tf(s, MARGIN, y, half, Inches(0.3))
    _text(tf, "STUDY DESIGN", size=9.5, color=BRAND, bold=True, space_after=0)
    # Le reste des devis en une barre : les dix premiers seuls se liraient comme
    # le corpus entier, et la somme ne tomberait pas juste.
    _other_kinds, _other_n = uc.designs_other
    _design_rows = list(uc.designs)
    if _other_n:
        _design_rows.append((f"{_other_kinds} others", _other_n))
    _bottom = _bars(s, MARGIN, y + Inches(0.4), half, _design_rows,
                    label_w=Inches(2.0), row_h=Inches(0.285))
    rx = MARGIN + half + Inches(0.5)
    tf = _tf(s, rx, y, half, Inches(0.3))
    _text(tf, "CERTAINTY, BY GRADE", size=9.5, color=GOLD, bold=True, space_after=0)
    _lv_bottom = _bars(s, rx, y + Inches(0.4), half, uc.levels, accent=GOLD,
                       label_w=Inches(2.0), row_h=Inches(0.285))
    _lv = dict(uc.levels)
    tf = _tf(s, rx, _lv_bottom + Inches(0.18), half, Inches(1.0))
    _text(tf, "Not applicable covers guidelines and other non-primary records; "
              "not assessed means no design was identifiable. Both are stated rather "
              "than folded into a low grade.", size=11, color=MUTED, line=1.2,
          space_after=0)
    # Mesuré, et non décalé d'une valeur devinée : le texte recouvrait la dernière barre.
    y = max(_bottom, _lv_bottom) + Inches(0.26)
    # Les chiffres de la phrase sont LUS dans les deux distributions ci-dessus.
    # Écrits à la main, ils décrivaient la version précédente du comptage.
    _dz = dict(uc.designs)
    _obs = sum(n for lbl, n in uc.designs if lbl in (
        "Cohort", "Cross-sectional", "Case-control", "Surveillance", "Observational"))
    _bullets(s, MARGIN, y, W - 2 * MARGIN, [
        f"{_dz.get('Randomised trial', 0)} randomised trials against {_obs} cohort, "
        f"cross-sectional, case-control and surveillance studies: an observational "
        f"literature, and the ceiling on any claim follows from that.",
        f"{_lv.get('High', 0)} articles could carry a high-certainty claim; "
        f"{_lv.get('Low', 0)} sit at low certainty on their design alone.",
    ], size=13.5)

    # 26 ── Cas d'usage : ce qu'il produit
    s, y = _slide(prs, "What this scenario produces", kicker="Worked example",
                  subtitle="Six artefacts from one question, each counted over the same "
                           "relevant subset and each traceable to the articles behind it.")
    y = _cards(s, y, [
        ("A graded brief",
         "What the literature establishes about early warning indicators, with each "
         "claim carrying its certainty and its supporting articles."),
        ("An evidence profile",
         "How much of this corpus is surveillance, modelling, cohort or trial work, and "
         "therefore what it can and cannot support."),
        ("A gap matrix",
         "Which indicators have been studied against which outcomes, and which pairings "
         "nobody has published on."),
        ("Candidate variables",
         "The indicators this literature actually uses, each with its provenance, ready "
         "to become a model specification."),
        ("A model and a projection",
         "An outcome, a fitted and tuned model with its importances, and an SEIR "
         "projection for the same geography."),
        ("A bibliography and a report",
         "A citable document with renumbered references, and the relevant articles as "
         "RIS, BibTeX, CSV or Excel."),
    ], cols=3)

    # 27 ── Architecture
    s, y = _slide(prs, "How it is built", kicker="Architecture")
    y = _cards(s, y, [
        ("FastAPI, one module per domain",
         "Imported in a fixed order, each module importing only from the ones before it, "
         "with a single composition root."),
        ("PostgreSQL with pgvector",
         "Documents, chunks, embeddings, scenario links, screening decisions and cached "
         "extractions in one database. The aggregations are SQL."),
        ("Lexical and semantic together",
         "A full-text index decides membership; cosine similarity decides ordering. The "
         "two are never confused."),
        ("Model roles, not model names",
         "Bulk, write, chat and embedding are roles bound to models in one registry, with "
         "request shaping and learned repairs when an API rejects a parameter."),
        ("A React front end",
         "French and English in parallel locale files, with a test that both carry the "
         "same keys and the same placeholders."),
        ("Deployed on merge",
         "Every merge to the main branch deploys and restarts the API; work in flight is "
         "relaunched at startup."),
    ], cols=3)

    # 28 ── Garanties
    s, y = _slide(prs, "What is guaranteed, and by what",
                  kicker="Quality",
                  subtitle="Each of these is a test that fails the build, not an "
                           "intention in a document.")
    y = _cards(s, y, [
        ("Extractions read everything",
         "Both halves of map-then-reduce are pinned: the per-article cache, and the SQL "
         "aggregation over the whole relevant subset."),
        ("One definition of relevant",
         "The SQL predicate is generated from one function, so a second copy cannot "
         "drift from the first."),
        ("One study-design vocabulary",
         "The classification SQL is generated from the Python table, with a test running "
         "both against a real database and comparing them."),
        ("Figures that cannot contradict",
         "Every article count comes from a single statement, so the panels of one page "
         "always agree with each other."),
        ("Gaps declare their denominator",
         "A gap is only claimed inside the shown grid, and the articles the matrix could "
         "not read are reported."),
        ("Both languages stay in step",
         "A locale test fails when a key or a placeholder exists in one language and not "
         "the other."),
    ], cols=3)

    # 29 ── Comparaison
    s, y = _slide(prs, "Against the tools it is usually compared to",
                  kicker="Where it differs",
                  subtitle="Drawn from a published synthesis report answering a real "
                           "question, not from a feature list.")
    rows = [
        ("Coverage of the eligible set",
         "A sample: a few dozen of the eligible papers, not stated on the page.",
         "All of it. Map per article, reduce in SQL, no sampling anywhere."),
        ("Certainty",
         "Asserted, or absent. A case report reads like a trial.",
         "GRADE, from a study-design vocabulary taken from MeSH, with a corpus ceiling."),
        ("Reviewer control",
         "Little or none: the tool decides what is eligible.",
         "Threshold, manual inclusion and exclusion, double-blind screening, PRISMA."),
        ("Reproducibility",
         "A chat answer. Ask twice, get two answers.",
         "A stored boolean strategy, a frozen corpus, a synthesis with a fingerprint."),
        ("Beyond the review",
         "Nothing. The review is the end of the road.",
         "Variables, outcome, tuned model, SEIR projection, alerts, living review."),
        ("What you leave with",
         "A PDF.",
         "A database, a bibliography, a citable report, a fitted model and a dashboard."),
    ]
    row_h = Inches(0.74)
    col1 = Inches(2.9)
    col2 = int((W - 2 * MARGIN - col1) / 2)
    x = MARGIN
    for label, cw in [("", col1), ("Typical synthesis tool", col2), ("LiteRev-Evidence", col2)]:
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
        _text(tf, label, size=12, color=PAPER, bold=True, space_after=0, line=1.12)
        tf = _tf(s, MARGIN + col1, y, col2 - Inches(0.16), row_h)
        _text(tf, them, size=11, color=MUTED, space_after=0, line=1.16)
        tf = _tf(s, MARGIN + col1 + col2, y, col2 - Inches(0.16), row_h)
        _text(tf, us, size=11, color=PAPER, space_after=0, line=1.16)
        y += row_h

    # 30 ── Limites
    s, y = _slide(prs, "What it does not do, and what is next",
                  kicker="Honest limits")
    y = _cards(s, y, [
        ("It does not replace a reviewer",
         "An automatic selection is labelled as such on every page. A formal systematic "
         "review needs the double-blind tab and two humans."),
        ("Abstract-first",
         "Full text is used where it is open. Behind a paywall the extraction reads the "
         "abstract, and the article says so."),
        ("Extraction is as good as the abstract",
         "A design absent from the abstract cannot be classified from it. Unclassified "
         "is reported as unclassified, never guessed."),
        ("A model needs real data",
         "Variables proposed from the literature are candidates. Fitting one still "
         "requires an operational extract that nobody else can supply."),
        ("Next: screening at scale",
         "Active learning on reviewer decisions, so the threshold stops being the only "
         "lever on a corpus of thousands."),
        ("Next: diffing two runs",
         "Which claims changed, which are new, which lost their support, when the "
         "literature moves under a living review."),
    ], cols=3)

    # 31 ── Fin
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = _box(s, 0, 0, W, H, fill=INK, radius=False)
    bg.line.fill.background()
    band = _box(s, 0, 0, Inches(0.16), H, fill=GOLD, radius=False)
    band.line.fill.background()
    tf = _tf(s, MARGIN, Inches(2.4), W - 2 * MARGIN, Inches(3))
    _text(tf, "LiteRev-Evidence", size=46, color=PAPER, bold=True, space_after=16)
    _text(tf, "Ask once. Read everything. Say how sure you are. Then run it.",
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
    ap.add_argument("--shots", default="assets/shots",
                    help="directory of interface screenshots to embed")
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

    out = build(uc, args.out, live, args.shots)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
