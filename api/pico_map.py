"""The per-article map: the PICO and the main finding of each article, one contract.

House rule (CLAUDE.md): every extraction reads ALL the relevant articles, by map then
reduce. This module is the MAP half for the PICO: what each article is asked, what is
kept of the answer, and which text the model reads. Five writers fill `pico_json` (the
pipeline's PICO step, the background worker, the per-article endpoint, the batch endpoint
and scripts/extract_pico_batch.py). They carried four prompts, three study design
vocabularies and three ways of choosing the text, and none of them asked for a result:
the brief, the recommended actions and the model spec all read a `key_finding` that no
writer stored, so whatever they said about results rested on the 20 to 30 articles they
reproduce.

Every article now carries `key_finding` (one sentence: the main result, with its number
and its direction when the paper gives them) and `finding_direction` (a closed
vocabulary), cached on the row with the rest of its PICO, and api/digest.py counts the
directions over the whole relevant subset in SQL.

Pure: no FastAPI app and no engine, so the batch script imports it without starting the
API. The one database helper takes the caller's connection.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from sqlalchemy import text

from llm_usage import LLMTruncated, json_content
from llm_usage import model_for as _model

logger = logging.getLogger("literev-api")

# ─────────────────────────────────────────────────────────────────────────────
# What each article is asked
# ─────────────────────────────────────────────────────────────────────────────
#: La direction du résultat principal, en vocabulaire FERMÉ : le digest la compte en SQL,
#: et une valeur libre ferait autant de catégories que de tournures.
FINDING_DIRECTIONS = ("increase", "decrease", "no_difference", "mixed", "descriptive", "none")

#: Le vocabulaire des devis, le même pour tous les rédacteurs (cf. api/study_design). Le
#: travailleur de fond et l'extraction à la demande demandaient « RCT|Cohort|Systematic
#: review|etc », et ce « etc » laissait le modèle écrire ce qu'il voulait.
PICO_DESIGNS = ("Randomized controlled trial | Clinical trial | Non-randomized trial | "
                "Systematic review | Meta-analysis | Cohort study | Case-control study | "
                "Cross-sectional study | Surveillance | Case report | Modelling study | "
                "Qualitative study | Guideline | Narrative review | Preclinical | Not stated")

_KEY_FINDING_SPEC = ('"key_finding":"ONE sentence: the main result, with its number and its '
                     'direction when the article gives them; an empty string when the article '
                     'reports no result (a protocol, a commentary, a methods paper)",'
                     '"finding_direction":"ONE of: ' + " | ".join(FINDING_DIRECTIONS) + '"')

_DIRECTION_DEFINITIONS = (
    "finding_direction: increase = the outcome is higher, or the exposure raises it, in the "
    "group or condition studied; decrease = lower, reduced or protective; no_difference = no "
    "significant difference or association; mixed = results in both directions, or that "
    "differ across outcomes or groups; descriptive = a result without a comparison (a "
    "prevalence, an incidence, a description); none = no result reported, and key_finding "
    "is then empty.")

_PICO_SYSTEM = (
    "You are a systematic review expert. Extract the PICO elements and the main finding of "
    "the article, and return ONLY valid JSON:\n"
    '{"P":"Population","I":"Intervention or exposure","C":"Comparator or Not specified",'
    '"O":"Outcome(s)",'
    '"study_design":"ONE of: ' + PICO_DESIGNS + '",'
    + _KEY_FINDING_SPEC + ","
    '"pico_confidence":0.0-1.0,"pico_notes":""}\n'
    + _DIRECTION_DEFINITIONS + "\n"
    "For a review, say in study_design what it includes when the text says so (e.g. "
    '"Systematic review of cohort studies"): the level of evidence depends on it.\n'
    "When a full-text excerpt follows the abstract, base the fields on it as well: its "
    "methods and results say more than the abstract.\n"
    "Be concise (max 2 sentences per field, one for key_finding). Do not use the em dash "
    "character. Return ONLY the JSON."
)

FINDING_SYSTEM_PROMPT = (
    "You read a scientific article and report its main finding. Return ONLY valid JSON:\n"
    "{" + _KEY_FINDING_SPEC + "}\n"
    + _DIRECTION_DEFINITIONS + "\n"
    "Do not use the em dash character. Return ONLY the JSON."
)


def _with_language(prompt: str, lang_directive: str, identifiers: str) -> str:
    """Le prompt dans la langue de l'interface, sauf les identifiants : le code regroupe
    `study_design` et compte `finding_direction`, et « cohort » et « cohorte » feraient
    deux catégories pour le même devis."""
    if not lang_directive:
        return prompt
    return (prompt + lang_directive
            + f" Exception: keep {identifiers} in English, from the lists above, as stable "
              "identifiers.")


def pico_system_prompt(lang_directive: str = "") -> str:
    """Le prompt PICO de tous les rédacteurs. Sans directive, il répond en anglais, comme
    le travailleur de fond l'a toujours fait ; le pipeline passe la langue de l'interface,
    parce que la prose PICO est affichée telle quelle."""
    return _with_language(_PICO_SYSTEM, lang_directive,
                          '"study_design" and "finding_direction"')


# ─────────────────────────────────────────────────────────────────────────────
# What is kept of the answer
# ─────────────────────────────────────────────────────────────────────────────
_DIRECTION_SYNONYMS = {
    "increased": "increase", "increases": "increase", "higher": "increase", "rise": "increase",
    "positive": "increase", "positive_association": "increase", "risk_factor": "increase",
    "decreased": "decrease", "decreases": "decrease", "lower": "decrease",
    "reduction": "decrease", "reduced": "decrease", "negative": "decrease",
    "protective": "decrease", "no_effect": "no_difference", "no_association": "no_difference",
    "not_significant": "no_difference", "non_significant": "no_difference",
    "nonsignificant": "no_difference", "no_significant_difference": "no_difference",
    "similar": "no_difference", "unchanged": "no_difference", "inconsistent": "mixed",
    "conflicting": "mixed", "both": "mixed", "description": "descriptive",
    "prevalence": "descriptive", "incidence": "descriptive", "no_finding": "none",
    "not_reported": "none", "n_a": "none", "na": "none",
}
#: Ce qu'un modèle écrit à la place d'un résultat absent.
_NO_FINDING = {"", "none", "n/a", "na", "null", "nil", "-", "not reported", "not stated",
               "not applicable", "no result", "no results", "no finding", "no findings",
               "aucun", "aucune", "aucun résultat", "non rapporté", "non rapporte"}
#: Une phrase, mais une phrase de résultat porte parfois trois intervalles de confiance.
#: Le plafond borne un emballement, il ne dimensionne pas la réponse.
_MAX_FINDING_CHARS = 600
_EM_DASH = re.compile(r"\s*" + chr(0x2014) + r"\s*")


def normalise_finding(finding: Any, direction: Any) -> tuple[str, str]:
    """(key_finding, finding_direction) tels qu'on les garde. PUR.

    Un résultat vide impose « none » ; un résultat présent ne peut pas être « none », ni
    une direction inconnue : il devient « descriptive », la seule étiquette qui n'affirme
    aucun sens."""
    kf = re.sub(r"\s+", " ", _EM_DASH.sub(" - ", finding)).strip() if isinstance(finding, str) else ""
    if kf.lower().rstrip(".") in _NO_FINDING:
        kf = ""
    if len(kf) > _MAX_FINDING_CHARS:
        kf = kf[:_MAX_FINDING_CHARS].rsplit(" ", 1)[0] + "..."
    key = re.sub(r"[\s\-/]+", "_", str(direction or "").strip().lower())
    key = _DIRECTION_SYNONYMS.get(key, key)
    if not kf:
        return "", "none"
    return kf, (key if key in FINDING_DIRECTIONS and key != "none" else "descriptive")


def normalise_pico(raw: Any, source: str | None = None) -> dict | None:
    """Le PICO tel qu'on l'écrit sur la ligne, ou None si la réponse n'est pas un objet.

    Tolérant, comme les rédacteurs l'étaient chacun de leur côté : une clé manquante est
    complétée plutôt que rejetée (et ré-extraite en boucle), et l'aval filtre déjà sur
    `pico_confidence`."""
    if not isinstance(raw, dict):
        return None
    pico = dict(raw)
    for key in ("P", "I", "C", "O"):
        pico.setdefault(key, "")
    if not isinstance(pico.get("study_design"), str) or not pico["study_design"].strip():
        # Identifiant stable, lu comme une absence par api/study_design.raw_design_sql.
        pico["study_design"] = "not specified"
    pico["key_finding"], pico["finding_direction"] = normalise_finding(
        pico.get("key_finding"), pico.get("finding_direction"))
    try:
        pico["pico_confidence"] = float(pico.get("pico_confidence", 0.3))
    except (TypeError, ValueError):
        pico["pico_confidence"] = 0.3
    pico["pico_notes"] = pico.get("pico_notes", "")
    if source:
        pico["pico_source"] = source
    return pico


class PicoUnreadable(Exception):
    """The answer came back and cannot be used: cut twice, not JSON, or not an object.

    Deterministic at temperature 0 and seed 42, so the caller counts an attempt. Any
    OTHER exception out of `extract_pico` comes from the API call (quota, timeout,
    outage): transient, and charged to nobody."""


# ─────────────────────────────────────────────────────────────────────────────
# A JSON answer, asked once more in a shorter form when it is cut
# ─────────────────────────────────────────────────────────────────────────────
#: Ajouté au message système quand une réponse a été coupée au plafond : la même
#: demande, en plus court. Une réponse JSON coupée n'est pas une réponse plus courte,
#: c'est une absence de réponse, et la redemander à l'identique la recouperait.
BRIEF_RETRY_INSTRUCTION = (
    "\n\nYOUR PREVIOUS ANSWER WAS CUT OFF at the length limit and was lost. Answer again "
    "with the same JSON structure and every required field, but much shorter: shorten "
    "the free-text values and leave out anything optional.")


class CutAnswer(LLMTruncated):
    """Cut even after the shorter retry. Carries the cut text for the one caller that can
    repair it (the model spec, api/variables._loads_lenient); every other caller reads it
    as no answer."""

    def __init__(self, message: str, content: str = ""):
        super().__init__(message)
        self.content = content


def _with_system_suffix(messages: list[dict], suffix: str) -> list[dict]:
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        out[0]["content"] = f"{out[0].get('content') or ''}{suffix}"
    else:
        out.insert(0, {"role": "system", "content": suffix.strip()})
    return out


def _content_of(response) -> str:
    try:
        return getattr(response.choices[0].message, "content", None) or ""
    except (AttributeError, IndexError, TypeError):
        return ""


def json_completion(call: Callable[[list[dict]], Any], messages: list[dict], label: str = "",
                    brief: str = BRIEF_RETRY_INSTRUCTION) -> str:
    """The text of a JSON completion, never a cut one.

    `call(messages)` makes the request (the call site keeps its own model, ceiling and
    temperature). An answer cut at its ceiling (`finish_reason == "length"`, which
    `llm_usage.json_content` names) used to surface as a JSONDecodeError that only the
    log saw; it is now asked once more with an instruction to be brief. Cut again, it
    raises `CutAnswer`, which carries the cut text."""
    try:
        return json_content(call(messages), label)
    except LLMTruncated as cut:
        logger.warning(f"{label or 'JSON'}: {cut} Nouvelle demande, plus courte.")
    response = call(_with_system_suffix(messages, brief))
    try:
        return json_content(response, f"{label} (shorter retry)" if label else "shorter retry")
    except LLMTruncated as cut:
        raise CutAnswer(str(cut), _content_of(response)) from None


# ─────────────────────────────────────────────────────────────────────────────
# What the model reads
# ─────────────────────────────────────────────────────────────────────────────
# Le texte intégral REMPLAÇAIT le résumé, et seuls ses 14 000 premiers caractères étaient
# lus : l'en-tête, l'introduction, le début des méthodes (et, pour une page HTML, le
# bandeau du site). Le pipeline, lui, ne lisait que le résumé, alors que sa docstring
# annonçait le texte intégral. Le résumé est désormais TOUJOURS lu, et le texte intégral
# y ajoute ses méthodes et ses résultats : la population et le devis sont dans les
# premières, le résultat principal dans les secondes.
PICO_ABSTRACT_CHARS = 3000
PICO_FULLTEXT_CHARS = 11000
#: Un texte intégral plus court ne dit rien que le résumé ne dise (une page d'erreur, un
#: résumé seul) : rien n'est ajouté.
_MIN_EXCERPT_CHARS = 300
#: Part du budget réservée aux méthodes quand les deux sections le dépassent ; ce que l'une
#: n'emploie pas revient à l'autre.
_METHODS_SHARE = 0.4
#: En deçà, les sections trouvées sont celles d'un résumé structuré recopié en tête du
#: texte intégral (« Methods… Results… » en trois phrases), pas celles de l'article.
_MIN_SECTIONS_CHARS = 1500

# Titres de section, anglais et français. Les morceaux de texte intégral sont stockés sans
# retour à la ligne (le découpage replie les blancs), donc un titre se reconnaît à son
# contexte : en tête de texte, après une fin de phrase ou un numéro de section, et suivi
# d'une capitale ou d'un chiffre. La casse compte : « Methods We recruited… » est un titre,
# « results were compared… » n'en est pas un.
_HEADINGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("methods", ("Materials and methods", "Materials and Methods", "MATERIALS AND METHODS",
                 "Material and methods", "Material and Methods", "MATERIAL AND METHODS",
                 "Patients and methods", "Patients and Methods", "PATIENTS AND METHODS",
                 "Methods", "METHODS", "Methodology", "METHODOLOGY",
                 "Matériels et méthodes", "Matériel et méthodes", "MATÉRIEL ET MÉTHODES",
                 "Méthodes", "MÉTHODES", "Méthodologie", "MÉTHODOLOGIE")),
    ("results", ("Results and discussion", "Results and Discussion", "RESULTS AND DISCUSSION",
                 "Results", "RESULTS", "Findings", "FINDINGS",
                 "Résultats", "RÉSULTATS", "Resultats", "RESULTATS")),
    ("discussion", ("Discussion", "DISCUSSION", "Conclusions", "CONCLUSIONS", "Conclusion",
                    "CONCLUSION")),
    ("introduction", ("Introduction", "INTRODUCTION", "Background", "BACKGROUND", "Contexte",
                      "CONTEXTE")),
    ("end", ("References", "REFERENCES", "Bibliography", "BIBLIOGRAPHY", "Références",
             "RÉFÉRENCES", "Bibliographie", "Acknowledgements", "Acknowledgments",
             "ACKNOWLEDGEMENTS", "ACKNOWLEDGMENTS", "Remerciements")),
)
_HEADING_KIND = {h: kind for kind, names in _HEADINGS for h in names}
_HEADING_RE = re.compile(
    r"(?:^|(?<=[.!?:;)\]]\s)|(?<=\d\s)|(?<=\d\.\s))"
    r"(?P<h>" + "|".join(re.escape(h) for h in sorted(_HEADING_KIND, key=len, reverse=True)) + r")"
    r"(?=\s*[:.]?\s+[A-Z0-9ÀÂÄÇÉÈÊËÎÏÔÖÙÛÜ(]|\s*$)")
_TRAILING_SECTION_NUMBER = re.compile(r"\s\d{1,2}(?:\.\d{1,2})*\.?\s*$")


def _join_chunks(chunks: list[str]) -> str:
    """Les morceaux bout à bout, sans leur recouvrement : chaque morceau répète la fin du
    précédent (jusqu'à 400 caractères, en phrases entières)."""
    out = ""
    for chunk in chunks:
        c = re.sub(r"\s+", " ", chunk or "").strip()
        if not c:
            continue
        if not out:
            out = c
            continue
        # Le recouvrement le plus LONG : la première position de la fin du texte d'où ce
        # qui suit est exactement le début du morceau.
        probe = c[:60]
        pos = out.find(probe, max(0, len(out) - 800)) if len(probe) >= 20 else -1
        while pos >= 0 and not c.startswith(out[pos:]):
            pos = out.find(probe, pos + 1)
        out += c[len(out) - pos:] if pos >= 0 else " " + c
    return out


def _cut(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n].rsplit(" ", 1)[0] + " [...]"


def _section_spans(body: str) -> tuple[dict[str, tuple[int, int]], list[tuple[int, str]]]:
    """La plus longue section de chaque sorte, d'un titre au titre suivant. La plus longue,
    parce qu'un texte intégral commence souvent par un résumé structuré dont « Methods » et
    « Results » tiennent en trois phrases."""
    marks = [(m.start("h"), _HEADING_KIND[m.group("h")]) for m in _HEADING_RE.finditer(body)]
    spans: dict[str, tuple[int, int]] = {}
    for i, (pos, kind) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(body)
        # Le numéro du titre suivant (« 4. Discussion ») n'appartient pas à cette section.
        number = _TRAILING_SECTION_NUMBER.search(body, pos, end)
        end = number.start() if number else end
        best = spans.get(kind)
        if best is None or end - pos > best[1] - best[0]:
            spans[kind] = (pos, end)
    return spans, marks


def fulltext_excerpt(chunks: list[str], budget: int = PICO_FULLTEXT_CHARS) -> str:
    """Les méthodes et les résultats d'un texte intégral, en `budget` caractères au plus.

    Sans titre reconnu, les morceaux qui suivent le premier : celui-ci porte l'en-tête,
    l'introduction et, pour une page HTML, le bandeau du site. La lecture s'arrête aux
    références. PUR."""
    body = _join_chunks(chunks)
    if len(body) < _MIN_EXCERPT_CHARS:
        return ""
    spans, _marks = _section_spans(body)
    methods = body[slice(*spans["methods"])] if "methods" in spans else ""
    results = body[slice(*spans["results"])] if "results" in spans else ""
    if len(methods) + len(results) >= _MIN_SECTIONS_CHARS:
        m_take = min(len(methods), int(budget * _METHODS_SHARE))
        r_take = min(len(results), budget - m_take)
        m_take = min(len(methods), budget - r_take)
        return " [...] ".join(_cut(s, n) for s, n in ((methods, m_take), (results, r_take)) if n > 0)
    rest = _join_chunks(chunks[1:]) if len(chunks) > 1 else body
    end = next((pos for pos, kind in _section_spans(rest)[1] if kind == "end"), len(rest))
    rest = rest[:end].strip()
    return _cut(rest, budget) if len(rest) >= _MIN_EXCERPT_CHARS else ""


def pico_source_text(title: str | None, abstract: str | None,
                     chunks: list[str] | None = None) -> tuple[str, str]:
    """(le texte que lit le modèle, sa source : « fulltext » ou « abstract »).

    Toujours le titre et le résumé (3 000 caractères au plus) ; quand le texte intégral
    existe, ses méthodes et ses résultats en plus (environ 11 000 caractères). PUR."""
    parts = [f"Title: {(title or '').strip()}"]
    abstract = (abstract or "").strip()[:PICO_ABSTRACT_CHARS]
    if abstract:
        parts.append(f"Abstract: {abstract}")
    excerpt = fulltext_excerpt(chunks or [])
    if excerpt:
        parts.append(f"Full text, methods and results (excerpt): {excerpt}")
    return "\n\n".join(parts), ("fulltext" if excerpt else "abstract")


def pico_fulltext_chunks(conn, doc_id: int) -> list[str]:
    """Les morceaux de texte intégral d'un article, dans l'ordre du document. Le type
    actuel (`fulltext_section`) d'abord ; l'ancien (`full_text`) seulement à défaut, pour
    ne pas lire deux fois le même texte."""
    rows = conn.execute(text("""
        SELECT chunk_type, content
        FROM document_chunk
        WHERE document_id = :id
          AND chunk_type IN ('fulltext_section', 'full_text')
          AND content IS NOT NULL AND length(trim(content)) > 0
        ORDER BY (chunk_type = 'fulltext_section') DESC, chunk_index, id
    """), {"id": doc_id}).all()
    if not rows:
        return []
    kind = rows[0][0]
    return [r[1] for r in rows if r[0] == kind]


# ─────────────────────────────────────────────────────────────────────────────
# The two calls
# ─────────────────────────────────────────────────────────────────────────────
#: 800 suffisait au PICO seul (400 tronquait le JSON des articles verbeux) ; la phrase du
#: résultat principal s'y ajoute.
_PICO_MAX_TOKENS = 1000
_FINDING_MAX_TOKENS = 300


def _parse_object(content: str, what: str) -> dict:
    try:
        raw = json.loads(content)
    except ValueError as e:
        raise PicoUnreadable(f"{what}: JSON invalide ({e})") from None
    if not isinstance(raw, dict):
        raise PicoUnreadable(f"{what}: la réponse n'est pas un objet JSON")
    return raw


def extract_pico(client, title: str | None, abstract: str | None, chunks: list[str] | None = None,
                 lang_directive: str = "", label: str = "") -> tuple[dict, str]:
    """(pico, source) pour UN article : le seul appel PICO de l'application.

    Lève `PicoUnreadable` quand la réponse est inutilisable (une tentative à compter) ;
    toute autre exception vient de l'appel lui-même (transitoire)."""
    user, source = pico_source_text(title, abstract, chunks)
    messages = [{"role": "system", "content": pico_system_prompt(lang_directive)},
                {"role": "user", "content": user}]

    def _ask(msgs: list[dict]):
        return client.chat.completions.create(
            model=_model("bulk"),
            messages=msgs,
            temperature=0,
            seed=42,
            max_tokens=_PICO_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

    try:
        content = json_completion(_ask, messages, label or "PICO")
    except LLMTruncated as e:
        raise PicoUnreadable(str(e)) from None
    pico = normalise_pico(_parse_object(content, "PICO"), source)
    return pico, source


def extract_finding(client, title: str | None, body: str | None, body_label: str = "Abstract",
                    lang_directive: str = "", label: str = "") -> tuple[str, str]:
    """(key_finding, finding_direction) d'un article dont le PICO précède ces deux champs :
    le titre et le résumé en entrée, les deux champs en sortie, un appel court."""
    system = _with_language(FINDING_SYSTEM_PROMPT, lang_directive, '"finding_direction"')
    user = f"Title: {(title or '').strip()}\n\n{body_label}: {(body or '').strip()[:PICO_FULLTEXT_CHARS]}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _ask(msgs: list[dict]):
        return client.chat.completions.create(
            model=_model("bulk"),
            messages=msgs,
            temperature=0,
            seed=42,
            max_tokens=_FINDING_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

    try:
        content = json_completion(_ask, messages, label or "finding")
    except LLMTruncated as e:
        raise PicoUnreadable(str(e)) from None
    raw = _parse_object(content, "finding")
    return normalise_finding(raw.get("key_finding"), raw.get("finding_direction"))
