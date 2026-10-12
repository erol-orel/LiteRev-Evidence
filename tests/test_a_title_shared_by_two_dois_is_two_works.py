"""Deux notices de même titre et de DOI différents sont deux travaux.

`_ingest_doc_direct` repliait une notice sur une ligne existante de même titre normalisé
quel que soit son DOI, et l'index unique (project_context, title_norm) l'y forçait : dix
éditions d'un « Avian influenza situation update », ou deux « Correction » de même
libellé, ne faisaient qu'une ligne, la seconde perdue sans trace. Le titre ne réunit plus
deux notices qui portent chacune un DOI, différent ; un DOI manquant d'un côté, ou le même
DOI, réunit encore. L'index devient (project_context, title_norm, DOI normalisé), bâti
avant que l'ancien ne soit retiré ; la dédup des liens d'un scénario suit la même règle.
"""
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

import main  # noqa: E402
from search_tables import ensure_search_tables  # noqa: E402

TITLE = "Avian influenza situation update for the European region"


def _ingestion_works() -> bool:
    from api.sources import _ingest_doc_direct
    probe = "crossref:probe-title-doi-0001"
    try:
        did, _new = _ingest_doc_direct(
            source="crossref", title="Ingestion probe for the title and DOI test",
            abstract="A probe document, inserted then deleted by the test.", year=2021,
            url=None, external_id=probe, doi=None)
    except Exception:                                    # noqa: BLE001
        return False
    try:
        with main.engine.begin() as c:
            if did is not None:
                c.execute(text("DELETE FROM document_chunk WHERE document_id = :d"), {"d": did})
            c.execute(text("DELETE FROM literature_document WHERE external_id = :e"), {"e": probe})
    except Exception:                                    # noqa: BLE001
        pass
    return did is not None


@pytest.fixture
def index(db_conn):
    from api.schema_boot import _ensure_title_doi_unique_index
    created_chunk_table = ensure_search_tables(db_conn.cursor())
    assert _ensure_title_doi_unique_index() == "ok"
    tag = str(int(time.time() * 1000))[-8:]
    yield tag
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM document_chunk WHERE document_id IN (SELECT id FROM literature_document "
                    "WHERE title LIKE %s)", (f"%{tag}%",))
        cur.execute("DELETE FROM article_scenarios WHERE document_id IN (SELECT id FROM literature_document "
                    "WHERE title LIKE %s)", (f"%{tag}%",))
        cur.execute("DELETE FROM literature_document WHERE title LIKE %s", (f"%{tag}%",))
        if created_chunk_table:
            cur.execute("DROP TABLE document_chunk")


def test_the_migration_replaces_the_index_and_can_run_again(index, db_conn):
    from api.schema_boot import TITLE_DOI_INDEX, _ensure_title_doi_unique_index
    assert _ensure_title_doi_unique_index() == "ok", "rejouée au démarrage suivant, elle ne doit rien casser"
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('uq_litdoc_title_norm') IS NULL, "
                    "(SELECT indisvalid FROM pg_index WHERE indexrelid = to_regclass(%s))", (TITLE_DOI_INDEX,))
        old_gone, valid = cur.fetchone()
    assert old_gone and valid is True


def test_the_index_lets_two_dois_share_a_title_and_still_refuses_a_true_duplicate(index, db_conn):
    import psycopg
    tag = index
    title = f"{TITLE} {tag}"
    tn = main._normalize_title(title)
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 9880 AND 9889")
        for doc_id, doi in ((9880, f"10.5555/upd.{tag}.1"), (9881, f"10.5555/upd.{tag}.2"), (9882, None)):
            cur.execute("INSERT INTO literature_document (id, title, abstract, source, project_context, doi, "
                        "title_norm) VALUES (%s, %s, 'x', 'crossref', 'literev', %s, %s)", (doc_id, title, doi, tn))
        with pytest.raises(psycopg.errors.UniqueViolation):
            cur.execute("INSERT INTO literature_document (id, title, abstract, source, project_context, doi, "
                        "title_norm) VALUES (9883, %s, 'x', 'crossref', 'literev', NULL, %s)", (title, tn))


@pytest.fixture
def ingest(index):
    if not _ingestion_works():
        pytest.skip("l'ingestion ne marche pas sur ce schéma : rien à mesurer ici")
    from api.sources import _ingest_doc_direct

    def _ingest(title, doi, ext):
        return _ingest_doc_direct(source="crossref", title=title, abstract=None, year=2024, url=None,
                                  external_id=ext, doi=doi)
    return _ingest, index


def test_the_same_generic_title_with_two_dois_is_two_rows(ingest):
    _ingest, tag = ingest
    title = f"{TITLE} {tag}"
    first, new1 = _ingest(title, f"10.5555/upd.{tag}.1", f"crossref:{tag}.1")
    second, new2 = _ingest(title, f"10.5555/upd.{tag}.2", f"crossref:{tag}.2")
    assert new1 and new2 and first != second


def test_the_same_doi_still_merges(ingest):
    _ingest, tag = ingest
    first, _ = _ingest(f"{TITLE} {tag}", f"10.5555/same.{tag}", f"crossref:{tag}")
    again, new = _ingest(f"{TITLE.upper()} {tag}.", f"https://doi.org/10.5555/SAME.{tag}", f"openalex:W{tag}")
    assert (again, new) == (first, False)


def test_a_missing_doi_on_either_side_still_merges_on_the_title(ingest):
    _ingest, tag = ingest
    with_doi, _ = _ingest(f"{TITLE} {tag} a", f"10.5555/one.{tag}", f"crossref:{tag}a")
    assert _ingest(f"{TITLE} {tag} a", None, f"arxiv:{tag}a") == (with_doi, False)
    without, _ = _ingest(f"{TITLE} {tag} b", None, f"arxiv:{tag}b")
    assert _ingest(f"{TITLE} {tag} b", f"10.5555/two.{tag}", f"crossref:{tag}b") == (without, False)
    with main.engine.connect() as c:
        doi = c.execute(text("SELECT doi FROM literature_document WHERE id = :i"), {"i": without}).scalar()
    assert doi == f"10.5555/two.{tag}", "le DOI arrivé ensuite comble la ligne existante"


def test_the_link_dedup_follows_the_same_rule(index, db_conn):
    tag = index
    sid = "usr-titledoi0001"
    upd, gen = f"{TITLE} {tag}", f"A long generic title about poultry workers health {tag}"
    long_abstract = "Seroprevalence of H5N1 antibodies among exposed poultry workers in two provinces."
    rows = [  # (id, title, doi, external_id, abstract)
        (9870, upd, f"10.5555/upd.{tag}.1", f"crossref:{tag}.1", long_abstract),
        (9871, upd, f"10.5555/upd.{tag}.2", f"crossref:{tag}.2", long_abstract),
        (9872, gen, f"10.5555/gen.{tag}", f"crossref:{tag}.g", None),
        (9873, gen, None, f"openalex:W{tag}", long_abstract),       # sans DOI : rejoint 9872
        (9874, upd, None, f"arxiv:{tag}", long_abstract),           # deux DOI pour ce titre : seule
    ]
    with db_conn.cursor() as cur:
        main._ensure_user_scenarios_table()
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (sid,))
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 9870 AND 9879")
        for doc_id, title, doi, ext, abstract in rows:
            cur.execute("INSERT INTO literature_document (id, title, abstract, source, project_context, doi, "
                        "external_id, title_norm, is_duplicate) VALUES (%s, %s, %s, 'crossref', 'literev', "
                        "%s, %s, %s, false)", (doc_id, title, abstract, doi, ext, main._normalize_title(title)))
            cur.execute("INSERT INTO article_scenarios (document_id, scenario_id, similarity_score) "
                        "VALUES (%s, %s, 0.5)", (doc_id, sid))
    try:
        assert main._dedup_scenario_links(sid) == 1
        with db_conn.cursor() as cur:
            cur.execute("SELECT document_id FROM article_scenarios WHERE scenario_id = %s", (sid,))
            assert {r[0] for r in cur.fetchall()} == {9870, 9871, 9872, 9874}
            # Le survivant sans résumé reçoit celui du doublon qu'il absorbe.
            cur.execute("SELECT abstract FROM literature_document WHERE id = 9872")
            assert cur.fetchone()[0] == long_abstract
    finally:
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (sid,))
            cur.execute("DELETE FROM literature_document WHERE id BETWEEN 9870 AND 9879")
