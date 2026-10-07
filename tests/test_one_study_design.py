"""One study design per article, wherever it is read (api/study_design.py).

The screen that prompted this: the evidence-level chart said 372 articles were Faible,
and the panel underneath, offering to narrow the corpus to exactly that selection, said
400 would be kept. Both were counting the same articles through the same GRADE table, so
the disagreement could only come from reading the design itself differently.

It did. Two independent extractions write a design from the same abstract, the PICO pass
and the metadata pass, and six places in the code combined them: the charts read the
column first, the corpus selector read PICO first, and an article whose two passes differ
landed in one level on the chart and another in the selection. Worse, either pass writes a
marker like "non précisé" when it finds nothing, and that marker won against a real design
found by the other.

So these tests pin one expression, used everywhere, treating every such marker as an
absence on both sides.
"""
import main
from conftest import ensure_document_columns  # noqa: E402

from api.study_design import classify, grade_level, raw_design_sql

SID = "usr-design-test"


def test_the_expression_reads_both_extractions():
    sql = raw_design_sql("d")
    assert "d.pico_json->>'study_design'" in sql and "d.study_design" in sql


def test_a_marker_of_absence_is_not_a_design():
    sql = raw_design_sql("d")
    for marker in ("non précisé", "not specified", "unknown", "n/a"):
        assert f"'{marker}'" in sql, marker


# ── contre une vraie base ────────────────────────────────────────────────────
def _seed(db_conn, rows):
    """rows: (id, study_design_column, pico_study_design)"""
    import json
    ensure_document_columns(db_conn.cursor())
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 8900 AND 8999")
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, created_at, updated_at) "
                    "VALUES (%s, 'Designs', 'dengue', NOW(), NOW())", (SID,))
        for doc_id, column, pico in rows:
            cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, "
                        "project_context, quality_score, study_design, pico_json) "
                        "VALUES (%s,%s,'abstract long enough',2024,'pubmed','literev',0.8,%s,%s)",
                        (doc_id, f"Article {doc_id}", column,
                         json.dumps({"study_design": pico}) if pico is not None else None))
            cur.execute("INSERT INTO article_scenarios (document_id, scenario_id, "
                        "similarity_score) VALUES (%s,%s,0.9)", (doc_id, SID))


def _cleanup(db_conn):
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 8900 AND 8999")
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))


#: Les cas qui faisaient diverger les deux lectures.
_ROWS = [
    # (id, colonne, pico)
    (8900, "Randomized controlled trial", "Cohort study"),   # les deux parlent, et diffèrent
    (8901, "Cohort study", "non précisé"),                   # le PICO se tait : la colonne gagne
    (8902, "non précisé", "Randomized controlled trial"),    # la colonne se tait : le PICO gagne
    (8903, None, "Case-control study"),                      # colonne absente
    (8904, "Cross-sectional study", None),                   # PICO absent
    (8905, "not specified", "unknown"),                      # les deux se taisent
]


def test_a_marker_never_wins_against_a_real_design(db_conn):
    """Le cas le plus coûteux : « non précisé » écrasait un devis réel trouvé par
    l'autre passe, et l'article tombait en « Non évaluée » sur une carte."""
    _seed(db_conn, _ROWS)
    try:
        _, level_of, meta = main._design_membership(SID, 0.45)
        assert level_of[8901] == grade_level("Cohort study")
        assert level_of[8902] == grade_level("Randomized controlled trial")
        assert level_of[8903] == grade_level("Case-control study")
        assert level_of[8904] == grade_level("Cross-sectional study")
        # Les deux se taisent : non évalué, ce qui est une réponse et pas une erreur.
        assert level_of[8905] == grade_level("")
        assert meta["judged"] == len(_ROWS)
    finally:
        _cleanup(db_conn)


def test_the_chart_and_the_corpus_selector_count_the_same_articles(db_conn):
    """THE property. The chart offered a level and the panel underneath offered to keep
    exactly that level; the two numbers have to be the same number."""
    _seed(db_conn, _ROWS)
    try:
        brief = main._build_evidence_brief(SID)
        chart = {d["level"]: d["count"] for d in brief["evidence_level_distribution"]}
        _, level_of, _ = main._design_membership(SID, 0.45)
        selector: dict[str, int] = {}
        for level in level_of.values():
            selector[level] = selector.get(level, 0) + 1
        assert chart == selector, f"chart {chart} != selector {selector}"
    finally:
        _cleanup(db_conn)


def test_the_design_chart_agrees_with_the_selector_too(db_conn):
    _seed(db_conn, _ROWS)
    try:
        brief = main._build_evidence_brief(SID)
        chart = {d["design"]: d["count"] for d in brief["study_design_distribution"]}
        design_of, _, _ = main._design_membership(SID, 0.45)
        selector: dict[str, int] = {}
        for design in design_of.values():
            selector[design] = selector.get(design, 0) + 1
        assert chart == selector, f"chart {chart} != selector {selector}"
    finally:
        _cleanup(db_conn)


def test_every_relevant_article_is_in_the_distribution(db_conn):
    """A distribution that silently drops articles cannot be offered as a filter."""
    _seed(db_conn, _ROWS)
    try:
        brief = main._build_evidence_brief(SID)
        assert sum(d["count"] for d in brief["evidence_level_distribution"]) == len(_ROWS)
        assert sum(d["count"] for d in brief["study_design_distribution"]) == len(_ROWS)
    finally:
        _cleanup(db_conn)


def test_the_distributions_carry_an_english_label_beside_the_value(db_conn):
    """The value is what the selector sends back, so it stays as the server writes it;
    the label is what the reader sees. Translating the value would break selection."""
    _seed(db_conn, _ROWS)
    try:
        brief = main._build_evidence_brief(SID)
        for row in brief["evidence_level_distribution"]:
            assert row["level_en"] and isinstance(row["level_en"], str)
        levels = {r["level"]: r["level_en"] for r in brief["evidence_level_distribution"]}
        assert levels.get("Faible", "Low") == "Low"
        designs = {r["design"]: r["design_en"] for r in brief["study_design_distribution"]}
        assert designs.get("Cohorte", "Cohort study") == "Cohort study"
    finally:
        _cleanup(db_conn)


def test_the_average_citation_count_reports_its_denominator(db_conn):
    """"Avg. citations 24.0 · Max 24" described one article out of 467, and said so
    nowhere. The count the average is taken over now travels with it."""
    _seed(db_conn, _ROWS)
    with db_conn.cursor() as cur:
        cur.execute("UPDATE literature_document SET citation_count = 24 WHERE id = 8900")
    try:
        stats = main._build_evidence_brief(SID)["corpus_stats"]
        assert stats["citations_known"] == 1
        assert stats["avg_citations"] == 24.0 and stats["max_citations"] == 24
    finally:
        _cleanup(db_conn)


# ── le tableau des niveaux, groupé et traduit ───────────────────────────────
def test_the_legend_groups_by_level_rather_than_listing_every_design():
    """Sixteen rows restated one rule nine times. Six groups state it once each."""
    groups = main.get_study_design_vocabulary(lang="en")["groups"]
    assert 5 <= len(groups) <= 8
    assert sum(len(g["designs"]) for g in groups) >= 15
    low = next(g for g in groups if g["level"] == "Faible")
    assert len(low["designs"]) >= 4


def test_the_legend_is_translated_including_the_levels_and_the_note():
    en = main.get_study_design_vocabulary(lang="en")
    assert all(not _has_french(g["label"]) for g in en["groups"]), [g["label"] for g in en["groups"]]
    assert "ceiling" in en["note"].lower()
    assert {lv["label"] for lv in en["levels"]} >= {"High", "Low", "Very low"}
    fr = main.get_study_design_vocabulary(lang="fr")
    assert any(g["label"] == "Faible" for g in fr["groups"])
    assert "PLAFOND" in fr["note"]


def _has_french(text: str) -> bool:
    return any(w in (text or "") for w in ("Élevée", "Modérée", "Faible", "Très faible",
                                           "Non applicable", "Non évaluée", "hérité"))


def test_the_level_value_stays_the_servers_even_in_english():
    """The interface sends this value back to narrow the corpus; translating it at the
    source would have broken the selection it is offered beside."""
    en = main.get_study_design_vocabulary(lang="en")
    values = {g["level"] for g in en["groups"] if g["level"]}
    assert "Faible" in values and "Élevée" in values
