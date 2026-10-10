"""The citable report speaks the language of its brief, and so does the brief's level.

Found preparing HPAI_last for a team that reads English. The brief was English, but:

- its global `evidence_level` read "Faible": the prompt's template lists the four values in
  French, and the model sometimes copies them into an English brief, so the interface
  showed "Level: Faible";
- the downloadable report was French whatever the brief's language: "## 1. Méthodes",
  "Niveau de preuve global", a claim table headed "Affirmation | Force" with strengths
  reading "Faible".

The level is now normalised to the brief's language, when the brief is written and when
it is served; the report takes a language (the interface's, else the brief's) and writes
its headings, notes and level labels in it. The French report is unchanged.
"""
from api.evidence import brief_level
from api.report import build_report, format_reference

_ARTICLES = {
    11: {"id": 11, "title": "Serosurvey of poultry workers", "year": 2020, "journal": "EID",
         "doi": "10.1/abc", "authors": "Smith J; Doe A"},
}


def _brief(level="Faible", strength="Faible"):
    return {
        "executive_summary": "Exposed workers seroconvert [11].",
        "key_findings": ["Seroprevalence is low [11]."],
        "limitations": ["Observational designs."],
        "methodological_quality": "Mostly observational.",
        "evidence_level": level,
        "grade_recommendation": "C",
        "claims": [{"claim": "Workers seroconvert", "strength": strength, "article_ids": [11],
                    "basis": {"n_articles": 1, "designs": {"cross-sectional": 1},
                              "from_designs": "Faible", "downgraded_single_study": True,
                              "capped_by_corpus": True}}],
        "_meta": {"model": "m", "grade_ceiling": "Faible", "generated_at": "2026-10-10T21:26:05",
                  "lang": "en"},
    }


_DIGEST = {"n_articles": 1043, "n_with_pico": 1042, "n_with_fulltext": 869,
           "year_min": 1997, "year_max": 2026, "complete": True}


# ── the brief's level ────────────────────────────────────────────────────────

def test_a_french_level_in_an_english_brief_is_translated():
    assert brief_level("Faible", "en") == "Low"
    assert brief_level("Très faible", "en") == "Very low"
    assert brief_level("Modéré", "en") == "Moderate"
    assert brief_level("Fort", "en") == "Strong"
    assert brief_level("Insuffisant", "en") == "Insufficient"


def test_an_english_level_in_a_french_brief_is_translated_back():
    assert brief_level("Low", "fr") == "Faible"
    assert brief_level("Very low", "fr") == "Très faible"
    assert brief_level("moderate", "fr") == "Modéré"


def test_the_head_is_translated_and_the_rest_kept():
    assert brief_level("Faible (corpus observationnel)", "en") == "Low (corpus observationnel)"
    assert brief_level("Low - observational corpus", "en") == "Low - observational corpus"


def test_an_unknown_or_empty_level_is_returned_as_is():
    assert brief_level("Mixed", "en") == "Mixed"
    assert brief_level("", "en") == ""
    assert brief_level(None, "en") is None
    # "low" inside a word is not the level "low"
    assert brief_level("Lower than expected", "fr") == "Lower than expected"


# ── the report ───────────────────────────────────────────────────────────────

def _report(lang, **kw):
    return build_report(_brief(**kw), _DIGEST, None, None, _ARTICLES, "HPAI_last",
                        '"avian influenza" AND workers', 0.5482, lang=lang)["markdown"]


def test_an_english_report_has_english_headings_and_levels():
    md = _report("en")
    for heading in ("## 1. Methods", "## 2. Results", "### Executive summary", "### Key findings",
                    "### Figure 1. Claims and strength of evidence", "## 3. Overall level of evidence",
                    "## 4. References"):
        assert heading in md, heading
    assert "- Level of evidence: Low" in md
    assert "| Claim | Strength | Basis | References |" in md
    assert "| Workers seroconvert | Low |" in md
    assert "a single study, downgraded one level" in md
    assert "corpus certainty ceiling `Low`" in md
    for french in ("Méthodes", "Résultats", "Niveau de preuve", "Références", "Affirmation",
                   "Faible", "non renseigné", "Généré le", "plafonné"):
        assert french not in md, french


def test_the_french_report_is_unchanged():
    md = _report("fr")
    for heading in ("## 1. Méthodes", "## 2. Résultats", "## 3. Niveau de preuve global",
                    "## 4. Références", "- Niveau de preuve : Faible",
                    "| Affirmation | Force | Base du calcul | Références |"):
        assert heading in md, heading


def test_a_brief_level_written_in_the_wrong_language_is_shown_in_the_reports():
    assert "- Level of evidence: Low" in _report("en", level="Faible")
    assert "- Niveau de preuve : Faible" in _report("fr", level="Low")


def test_an_older_label_spelling_still_reads_in_english():
    md = _report("en", strength="Modéré")
    assert "| Workers seroconvert | Moderate |" in md


def test_a_reference_with_missing_fields_says_so_in_the_reports_language():
    assert "[authors not recorded]" in format_reference(1, {"id": 9}, "en")
    assert "[title not recorded]" in format_reference(1, {"id": 9}, "en")
    assert "[auteurs non renseignés]" in format_reference(1, {"id": 9})
