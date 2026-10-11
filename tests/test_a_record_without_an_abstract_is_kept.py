"""Une notice SANS résumé reste au corpus, signalée, hors du sous-ensemble pertinent tant
qu'un relecteur ne l'inclut pas.

Elles en étaient retirées après la fédération (2 434 notices sur HPAI_last : lettres,
éditoriaux, notices que les éditeurs livrent sans résumé) ; la correspondance dans la
bibliothèque locale exigeait un résumé de 30 caractères, si bien qu'un document sans
résumé ne correspondait jamais, même par son titre ; le PRISMA les disait « retirées » ;
et le backfill des résumés, qui passait après sans jamais les relier, marquait « tentées »
toutes les notices d'un lot dont la requête à Europe PMC avait échoué. Ces tests tiennent
chaque morceau : le lien gardé, le compte à part, la porte, l'inclusion à la main, la
correspondance par le titre, l'arithmétique du PRISMA et le backfill.
"""
import json
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

import main  # noqa: E402
from api.scenario_store import has_abstract_sql, relevant_gate_sql, scenario_counts  # noqa: E402
from conftest import patch_app  # noqa: E402
from search_tables import ensure_search_tables  # noqa: E402

SID = "usr-noabstract001"
LONG = "Seroprevalence of H5N1 antibodies among exposed poultry workers in two provinces."


# ── le prédicat et la porte (purs) ────────────────────────────────────────────

def test_the_helper_is_the_thirty_character_rule():
    assert has_abstract_sql("x") == "(x.abstract IS NOT NULL AND length(TRIM(x.abstract)) >= 30)"


def test_the_gate_reads_the_abstract_only_on_the_score_branch():
    """Inclus à la main : pertinent, résumé ou non. Par le score : seulement avec un résumé."""
    gate = relevant_gate_sql("d", "ars", ":thr")
    included, by_score = gate.split("= 'included' OR (", 1)
    assert has_abstract_sql("d") not in included
    assert has_abstract_sql("d") in by_score and "similarity_score" in by_score


# ── contre une vraie base ────────────────────────────────────────────────────

ROWS = [
    # (id, abstract, score, screening_status)
    (9810, LONG, 0.90, None),             # au-dessus du seuil, pertinent
    (9811, None, 0.95, None),             # sans résumé, score de titre élevé : PAS pertinent
    (9812, "Short.", 0.20, "included"),   # sans résumé, inclus à la main : pertinent
    (9813, None, None, None),             # sans résumé, jamais scoré : pas « non scoré »
    (9814, LONG, None, None),             # non scoré
    (9815, LONG, 0.10, None),             # en dessous du seuil
    (9816, None, 0.99, "excluded"),       # sans résumé, exclu
]
NO_ABSTRACT = {9811, 9812, 9813, 9816}


@pytest.fixture()
def corpus(db_conn, monkeypatch):
    created_chunk_table = ensure_search_tables(db_conn.cursor())
    main._ensure_scenario_settings_table()
    # Le corpus a un article non scoré : la lecture du corpus lancerait le scoring.
    patch_app(monkeypatch, "_maybe_autorerank", lambda *a, **k: False)
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 9810 AND 9839")
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, article_count, created_at, updated_at) "
                    "VALUES (%s, 'No abstract', 'h5n1', 'boolean', '{}', %s, NOW(), NOW())", (SID, len(ROWS)))
        for doc_id, abstract, score, status in ROWS:
            cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, project_context, "
                        "is_duplicate) VALUES (%s, %s, %s, 2023, 'crossref', 'literev', false)",
                        (doc_id, f"H5N1 in poultry workers, record {doc_id}", abstract))
            cur.execute("INSERT INTO article_scenarios (document_id, scenario_id, similarity_score, screening_status) "
                        "VALUES (%s, %s, %s, %s)", (doc_id, SID, score, status))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 9810 AND 9839")
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        if created_chunk_table:
            cur.execute("DROP TABLE document_chunk")


def _relevant(conn, thr=0.45) -> set[int]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT d.id FROM literature_document d JOIN article_scenarios ars "
                    f"ON ars.document_id = d.id AND ars.scenario_id = %(sid)s "
                    f"WHERE {relevant_gate_sql('d', 'ars', '%(thr)s')}", {"sid": SID, "thr": thr})
        return {r[0] for r in cur.fetchall()}


def test_they_are_counted_apart_and_the_parts_still_add_up(corpus):
    c = scenario_counts(SID, threshold=0.45)
    assert c["total"] == 7 and c["no_abstract"] == 4
    assert (c["above_threshold"], c["below_threshold"], c["unscored"]) == (1, 1, 1)
    assert c["above_threshold"] + c["below_threshold"] + c["unscored"] + c["no_abstract"] == c["total"]
    assert c["scored"] + c["unscored"] + c["no_abstract"] == c["total"]
    assert c["relevant"] == 2


def test_never_relevant_on_the_score_but_relevant_when_included_by_hand(corpus):
    assert _relevant(corpus) == {9810, 9812}
    assert 9811 not in _relevant(corpus, thr=0.0), "un score de titre n'ouvre jamais la porte"
    # Le résumé arrive (backfill, autre source) : la notice devient éligible d'elle-même.
    with corpus.cursor() as cur:
        cur.execute("UPDATE literature_document SET abstract = %s WHERE id = 9811", (LONG,))
    assert 9811 in _relevant(corpus)


def test_the_corpus_lists_them_apart_for_screening_by_title(corpus):
    got = main.get_user_scenario_corpus(SID, limit=50, no_abstract=True, abstract_chars=None)
    assert {a["id"] for a in got["articles"]} == NO_ABSTRACT
    assert got["filtered_total"] == 4 and got["total"] == 7 and got["no_abstract"] == 4
    assert all(a["has_abstract"] is False for a in got["articles"])
    every = {a["id"]: a for a in main.get_user_scenario_corpus(SID, limit=50, abstract_chars=None)["articles"]}
    assert every[9810]["has_abstract"] is True and every[9810]["above_threshold"] is True
    assert every[9811]["above_threshold"] is False, "0.95 sur le seul titre ne passe pas le seuil"


def test_the_counts_endpoint_stays_consistent(corpus):
    from fastapi.testclient import TestClient
    body = TestClient(main.app).get(f"/user-scenarios/{SID}/counts").json()
    assert body["consistent"] is True, body["mismatches"]
    assert body["no_abstract"] == 4 and body["counts"]["no_abstract"] == 4
    assert body["above_threshold"] + body["below_threshold"] == body["corpus_links"] == 7


def test_the_prisma_card_screens_them_by_title_instead_of_removing_them(corpus):
    p = main.get_user_scenario_prisma(SID, threshold=0.45)
    sem, ident = p["semantic_screening"], p["identification"]
    assert sem["no_abstract"] == 4 and sem["above_threshold"] == 1
    assert sem["above_threshold"] + sem["below_threshold"] + sem["no_abstract"] == ident["records_screened"] == 7
    assert ident["kept_no_abstract"] == 4 and ident["removed_no_abstract"] == 0
    assert p["evidence"]["total"] == 2
    # Inclus à la main sans résumé : un repêchage. Exclu sans résumé : pas un veto (le
    # score ne l'avait pas présélectionné).
    assert p["manual_curation"]["manually_rescued"] == 1
    assert p["manual_curation"]["manually_vetoed"] == 0


def test_the_figures_of_a_search_say_kept_and_the_arithmetic_holds():
    f = main._prisma_identification_figures({"crossref": 10, "pubmed": 6}, 14, 1, 11,
                                            kept_no_abstract=3, removed_not_matching=2)
    assert f["removed_no_abstract"] == 0 and f["kept_no_abstract"] == 3
    assert (f["records_identified"] - f["duplicates_removed"] - f["removed_before_screening"]
            == f["records_screened"] == 11)
    # Jamais plus que les passées au screening, dont elles sont un sous-compte.
    assert main._prisma_identification_figures({"crossref": 2}, 2, 0, 2, kept_no_abstract=9)["kept_no_abstract"] == 2


def test_the_library_matches_a_record_without_an_abstract_on_its_title(corpus, monkeypatch):
    monkeypatch.setenv("LEXICAL_SEARCH_ENGINE", "like")
    with corpus.cursor() as cur:
        cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, project_context, "
                    "is_duplicate) VALUES (9820, 'Quokkapox among zookeepers: a letter', NULL, 2024, "
                    "'crossref', 'literev', false)")
    ids = set(main._boolean_corpus_ids("quokkapox", {}))
    assert 9820 in ids, "sans résumé, le titre suffit à correspondre"
    assert 9820 in set(main._boolean_corpus_ids("quokkapox AND zookeepers", {}))


# ── la recherche, assemblée de bout en bout sur la bibliothèque locale ──────────

def test_the_assembly_keeps_the_link_and_the_figures_say_so(corpus, monkeypatch):
    from api import pipeline as P
    from api import relevance as R
    monkeypatch.setenv("LEXICAL_SEARCH_ENGINE", "like")
    monkeypatch.setattr(R, "_run_semantic_rerank_inline", lambda *a, **k: 0)
    monkeypatch.setattr(R, "_run_cross_encoder_rerank", lambda *a, **k: {})
    monkeypatch.setattr(P, "_run_clustering_background", lambda *a, **k: None)
    monkeypatch.setattr(P, "_precompute_user_kg", lambda *a, **k: None)
    sid = "usr-noabstract002"
    with corpus.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (sid,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (sid,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, article_count, search_strategy) "
                    "VALUES (%s, 'Quokkapox', 'quokkapox', 'boolean', '{}', 0, %s::jsonb)",
                    (sid, json.dumps({"general": "quokkapox OR wallabypox", "pubmed": "quokkapox OR wallabypox",
                                      "explanation": "", "synonyms": []})))
        cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, project_context, "
                    "is_duplicate) VALUES "
                    "(9830, 'Quokkapox in a zoo: a letter to the editor', NULL, 2024, 'crossref', 'literev', false),"
                    "(9831, 'Wallabypox transmission in captive marsupials', %s, 2023, 'pubmed', 'literev', false)",
                    (LONG,))
    try:
        P._run_user_scenario_populate(sid, "quokkapox", {}, include_live=False, auto_pipeline=False)
        job = P._user_scenario_populate_jobs.get(sid) or {}
        with corpus.cursor() as cur:
            cur.execute("SELECT document_id FROM article_scenarios WHERE scenario_id = %s", (sid,))
            linked = {r[0] for r in cur.fetchall()}
        assert {9830, 9831} <= linked, "la notice sans résumé a été retirée du corpus"
        fig = job.get("prisma_identification") or {}
        assert fig.get("removed_no_abstract") == 0 and fig.get("kept_no_abstract") == 1, fig
        assert (fig["records_identified"] - fig["duplicates_removed"] - fig["removed_before_screening"]
                == fig["records_screened"]), fig
        assert scenario_counts(sid)["no_abstract"] == 1
    finally:
        with corpus.cursor() as cur:
            cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (sid,))
            cur.execute("DELETE FROM user_scenarios WHERE id = %s", (sid,))


# ── le backfill des résumés ──────────────────────────────────────────────────

class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_the_lookup_says_when_a_request_failed():
    from api.schema_boot import _epmc_abstract_lookup

    def _down(url, params=None, timeout=None):
        import requests
        raise requests.ConnectionError("unreachable")

    assert _epmc_abstract_lookup("doi", ["10.1/a", "10.1/b"], _down) == ({}, False)
    assert _epmc_abstract_lookup("doi", ["10.1/a"], lambda *a, **k: _Resp(503)) == ({}, False)
    assert _epmc_abstract_lookup("doi", ["10.1/a"], lambda *a, **k: _Resp(429)) == ({}, False)


def test_the_lookup_answers_only_what_the_page_holds():
    from api.schema_boot import _EPMC_ABSTRACT_PAGE, _epmc_abstract_lookup
    sent = []

    def _page(url, params=None, timeout=None):
        sent.append(dict(params or {}))
        # Deux notices trouvées en tout, une seule dans la page : l'autre DOI n'est PAS
        # « sans résumé », il n'est pas répondu.
        return _Resp(200, {"hitCount": 2, "resultList": {"result": [
            {"doi": "10.1/A", "abstractText": "<p>" + LONG + "</p>"}]}})

    answered, ok = _epmc_abstract_lookup("doi", ["10.1/a", "10.1/b"], _page)
    assert ok and answered == {"10.1/a": LONG}
    assert sent[0]["pageSize"] == _EPMC_ABSTRACT_PAGE >= 1000, "la page à la taille du lot poussait des notices dehors"
    assert 'DOI:"10.1/a"' in sent[0]["query"] and 'DOI:"10.1/b"' in sent[0]["query"]
    # La page complète : un DOI absent est un vrai « pas de notice ».
    full = lambda *a, **k: _Resp(200, {"hitCount": 1, "resultList": {"result": [  # noqa: E731
        {"doi": "10.1/a", "abstractText": LONG}]}})
    assert _epmc_abstract_lookup("doi", ["10.1/a", "10.1/b"], full) == ({"10.1/a": LONG, "10.1/b": None}, True)


def test_a_pmid_is_looked_up_in_medline():
    from api.schema_boot import _epmc_abstract_lookup
    sent = []

    def _medline(url, params=None, timeout=None):
        sent.append(dict(params or {}))
        return _Resp(200, {"hitCount": 1, "resultList": {"result": [{"pmid": "31234567", "abstractText": LONG}]}})

    assert _epmc_abstract_lookup("pmid", ["31234567"], _medline) == ({"31234567": LONG}, True)
    assert sent[0]["query"] == "(EXT_ID:31234567 AND SRC:MED)"


def test_a_refused_batch_is_asked_again_one_value_at_a_time():
    from api.schema_boot import _epmc_abstract_lookup

    def _picky(url, params=None, timeout=None):
        q = (params or {}).get("query", "")
        if " OR " in q or "bad" in q:
            return _Resp(400)
        return _Resp(200, {"hitCount": 1, "resultList": {"result": [{"doi": "10.1/good", "abstractText": LONG}]}})

    answered, ok = _epmc_abstract_lookup("doi", ["10.1/good", "10.1/bad"], _picky)
    assert ok and answered == {"10.1/good": LONG, "10.1/bad": None}


@pytest.fixture()
def backfill_rows(db_conn, monkeypatch):
    created_chunk_table = ensure_search_tables(db_conn.cursor())
    from api import relevance as R
    # Le rattrapage GLOBAL des chunks manquants n'est pas l'objet ici : il toucherait toute
    # la bibliothèque de la base de test.
    monkeypatch.setattr(R, "_backfill_title_abstract_chunks", lambda *a, **k: 0)
    tag = str(int(time.time() * 1000))[-8:]
    with db_conn.cursor() as cur:
        cur.execute("ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS doi TEXT")
        cur.execute("ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS "
                    "abstract_backfill_attempted BOOLEAN DEFAULT FALSE")
        cur.execute("ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS embedding TEXT")
        cur.execute("DELETE FROM document_chunk WHERE document_id BETWEEN 9840 AND 9849")
        cur.execute("DELETE FROM article_scenarios WHERE document_id BETWEEN 9840 AND 9849")
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 9840 AND 9849")
        cur.execute("INSERT INTO literature_document (id, title, abstract, doi, pmid, source, project_context) VALUES "
                    "(9840, 'Backfill one, a title long enough', NULL, %s, NULL, 'crossref', 'literev'),"
                    "(9841, 'Backfill two, a title long enough', NULL, %s, NULL, 'crossref', 'literev'),"
                    "(9842, 'Backfill three, a title long enough', NULL, NULL, %s, 'pubmed', 'literev')",
                    (f"10.9999/nab.{tag}.1", f"10.9999/nab.{tag}.2", f"9{tag[-7:]}"))
        # Un chunk fait du seul titre, déjà vectorisé, et un lien scoré sur ce titre. La
        # colonne est un vector(1536) sur une base au schéma complet, du texte ailleurs.
        cur.execute("SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                    "WHERE attrelid = 'document_chunk'::regclass AND attname = 'embedding'")
        kind = cur.fetchone()[0]
        vector = "[" + ",".join(["0.1"] * 1536) + "]" if kind.startswith("vector") else "[0.1]"
        cur.execute("INSERT INTO document_chunk (document_id, chunk_index, content, chunk_type, embedding) "
                    "VALUES (9840, 0, 'Backfill one, a title long enough', 'title_abstract', %s)", (vector,))
        cur.execute("INSERT INTO article_scenarios (document_id, scenario_id, similarity_score) "
                    "VALUES (9840, %s, 0.91)", (SID,))
    yield db_conn, tag
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM document_chunk WHERE document_id BETWEEN 9840 AND 9849")
        cur.execute("DELETE FROM article_scenarios WHERE document_id BETWEEN 9840 AND 9849")
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 9840 AND 9849")
        if created_chunk_table:
            cur.execute("DROP TABLE document_chunk")


def _state(conn, doc_id):
    with conn.cursor() as cur:
        cur.execute("SELECT abstract, abstract_backfill_attempted FROM literature_document WHERE id = %s", (doc_id,))
        return cur.fetchone()


def test_a_failed_request_marks_nothing_as_attempted(backfill_rows):
    from api.schema_boot import _backfill_abstracts_once
    conn, _tag = backfill_rows

    def _down(url, params=None, timeout=None):
        import requests
        raise requests.Timeout("Europe PMC timed out")

    stats = _backfill_abstracts_once(limit=100000, http_get=_down)
    assert stats["failed_requests"] >= 1
    for doc_id in (9840, 9841, 9842):
        abstract, attempted = _state(conn, doc_id)
        assert abstract is None and attempted is not True, f"{doc_id} marquée tentée sur une requête échouée"


def test_answered_records_are_filled_or_marked_and_the_chunk_is_rebuilt(backfill_rows):
    from api.schema_boot import _backfill_abstracts_once
    conn, tag = backfill_rows
    doi1, pmid = f"10.9999/nab.{tag}.1", f"9{tag[-7:]}"

    def _epmc(url, params=None, timeout=None):
        q = (params or {}).get("query", "")
        if f"EXT_ID:{pmid}" in q:
            return _Resp(200, {"hitCount": 1, "resultList": {"result": [
                {"pmid": pmid, "abstractText": "From MEDLINE: " + LONG}]}})
        if f"nab.{tag}" in q:
            # Complète : le second DOI n'a pas de notice, il est répondu sans résumé.
            return _Resp(200, {"hitCount": 1, "resultList": {"result": [{"doi": doi1, "abstractText": LONG}]}})
        return _Resp(200, {"hitCount": 0, "resultList": {"result": []}})

    _backfill_abstracts_once(limit=100000, http_get=_epmc)
    assert _state(conn, 9840) == (LONG, True)
    assert _state(conn, 9841) == (None, True), "répondu sans notice : tenté, pour ne pas bloquer la file"
    assert _state(conn, 9842) == ("From MEDLINE: " + LONG, True), "le PMID seul suffit"
    with conn.cursor() as cur:
        cur.execute("SELECT content, embedding FROM document_chunk WHERE document_id = 9840 "
                    "AND chunk_type = 'title_abstract'")
        content, embedding = cur.fetchone()
        cur.execute("SELECT similarity_score FROM article_scenarios WHERE document_id = 9840")
        score = cur.fetchone()[0]
    assert LONG[:40] in content and embedding is None, "le chunk du titre seul doit être refait, et revectorisé"
    assert score is None, "le score de titre ne vaut pas un score de résumé"
