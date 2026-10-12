"""Un travail compte une fois, même quand ses notices portent des DOI différents.

Depuis qu'une notice de même titre et d'un autre DOI est un autre travail (dix éditions
d'un « situation update » restent dix), la recherche HPAI du 12 octobre gardait à part ce
qui n'était qu'un article : son préprint (Research Square, Authorea), ses versions (/v1,
/v2, Wellcome Open Research .1 et .2), les figures que figshare dépose sous son titre, et
une copie au point final ('10.1371/journal.pone.0073200.'). 40 paires de ce genre étaient
entières dans le sous-ensemble pertinent : l'extraction comptait l'étude deux fois.

La règle : le DOI d'un travail ignore ponctuation, casse et version ; sous un même titre
long, un préprint ou un dépôt de données rejoint LA version publiée quand il n'y en a
qu'une. Deux DOI de revue de racines différentes restent deux travaux, et la version
publiée survit plutôt que le préprint ou la figure.
"""
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from api.documents import doi_kind, doi_work_key, same_work_doi_match, same_work_survivors  # noqa: E402

TITLE = "live poultry exposure and public response to influenza a h7n9 in urban and rural china"


# ── Le DOI d'un travail ───────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,key", [
    ("10.1371/journal.pone.0073200.", "10.1371/journal.pone.0073200"),          # CORE
    ("https://doi.org/10.3201/EID2104.141281.", "10.3201/eid2104.141281"),
    ("10.21203/rs.2.14161/v2", "10.21203/rs.2.14161"),                            # Research Square
    ("10.22541/au.174938958.80823095/v1", "10.22541/au.174938958.80823095"),     # Authorea
    ("10.6084/m9.figshare.1528105.v1", "10.6084/m9.figshare.1528105"),           # figshare
    ("10.12688/wellcomeopenres.23064.2", "10.12688/wellcomeopenres.23064"),      # F1000 : .N
    ("10.5555/upd.2024.2", "10.5555/upd.2024.2"),          # un numéro d'édition n'est pas une version
    (None, None), ("", None),
])
def test_the_work_doi_drops_punctuation_case_and_versions(raw, key):
    assert doi_work_key(raw) == key


@pytest.mark.parametrize("doi,kind", [
    ("10.1186/1471-2334-9-170", "journal"),
    ("10.21203/rs.2.14161/v1", "preprint"),
    ("10.1101/2020.01.01.123456", "preprint"),
    ("10.64898/2025.12.08.693062", "preprint"),
    ("10.6084/m9.figshare.1528105.v1", "data"),
    ("10.5281/zenodo.1234567", "data"),
    (None, None),
])
def test_each_doi_says_what_it_is(doi, kind):
    assert doi_kind(doi) == kind


# ── Les liens d'un scénario ───────────────────────────────────────────────────

def _rec(i, doi, title=TITLE, ext=None):
    return {"id": i, "doi": doi, "title_norm": title, "external_id": ext}


def test_figshare_items_join_the_published_article_which_survives():
    # Le cas réel : l'article de PLOS ONE et trois de ses figures sur figshare. L'article
    # a ici le plus GRAND id : il survit quand même.
    recs = [_rec(10, "10.6084/m9.figshare.1528109.v3"), _rec(20, "10.6084/m9.figshare.1528103"),
            _rec(30, "10.6084/m9.figshare.1528109"), _rec(40, "10.1371/journal.pone.0137831")]
    assert same_work_survivors(recs) == {10: 40, 20: 40, 30: 40}


def test_a_preprint_joins_its_one_published_version():
    recs = [_rec(1, "10.22541/au.174938958.80823095/v1"), _rec(2, "10.1111/irv.70189")]
    assert same_work_survivors(recs) == {1: 2}


def test_preprint_versions_join_each_other_when_nothing_is_published():
    recs = [_rec(1, "10.21203/rs.2.14161/v1"), _rec(2, "10.21203/rs.2.14161/v2")]
    assert same_work_survivors(recs) == {2: 1}


def test_a_trailing_dot_is_the_same_doi():
    recs = [_rec(5, "10.1371/journal.pone.0073200"), _rec(9, "10.1371/journal.pone.0073200.")]
    assert same_work_survivors(recs) == {9: 5}


def test_two_editions_with_one_title_stay_two_works():
    title = "avian influenza situation update for the european region"
    recs = [_rec(1, "10.5555/upd.2024.1", title), _rec(2, "10.5555/upd.2024.2", title)]
    assert same_work_survivors(recs) == {}


def test_a_preprint_under_a_title_two_journals_share_stays_apart():
    recs = [_rec(1, "10.5555/upd.2024.1"), _rec(2, "10.5555/upd.2024.2"), _rec(3, "10.21203/rs.3.rs-9/v1")]
    assert same_work_survivors(recs) == {}


def test_a_short_title_never_joins_two_dois():
    recs = [_rec(1, "10.5555/c.1", "correction"), _rec(2, "10.21203/rs.3.rs-1/v1", "correction")]
    assert same_work_survivors(recs) == {}


def test_a_record_without_doi_joins_the_one_work_of_its_title():
    assert same_work_survivors([_rec(1, "10.1186/x"), _rec(2, None, ext="openalex:W1")]) == {2: 1}
    two = [_rec(1, "10.5555/a.1"), _rec(2, "10.5555/a.2"), _rec(3, None, ext="arxiv:9")]
    assert same_work_survivors(two) == {}


def test_the_pmid_still_joins_records_without_doi():
    recs = [_rec(7, None, "short", "pmid:123"), _rec(8, None, "short", "123")]
    assert same_work_survivors(recs) == {8: 7}


# ── Le choix à l'ingestion ────────────────────────────────────────────────────

def test_ingestion_folds_a_version_or_a_preprint_into_the_published_record():
    published = [(1, "10.1371/journal.pone.0073200")]
    assert same_work_doi_match("10.1371/journal.pone.0073200.", published) == 1
    assert same_work_doi_match("10.21203/rs.3.rs-1/v1", published) == 1
    assert same_work_doi_match("10.6084/m9.figshare.7.v1", published) == 1


def test_ingestion_never_folds_a_published_article_into_a_preprint_or_an_edition():
    assert same_work_doi_match("10.1111/irv.70189", [(1, "10.22541/au.1/v1")]) is None
    assert same_work_doi_match("10.5555/upd.2024.2", [(1, "10.5555/upd.2024.1")]) is None
    two = [(1, "10.5555/upd.2024.1"), (2, "10.5555/upd.2024.2")]
    assert same_work_doi_match("10.21203/rs.3.rs-1/v1", two) is None


# ── Contre la base ────────────────────────────────────────────────────────────

@pytest.fixture
def library(db_conn):
    import main
    from search_tables import ensure_search_tables

    from api.schema_boot import _ensure_title_doi_unique_index
    created_chunk_table = ensure_search_tables(db_conn.cursor())
    # Un test antérieur de la session peut avoir remplacé la table par une version minimale
    # (sans les colonnes de l'ingestion, sans défaut sur l'id) : l'ingestion y échouait et
    # son test se sautait. On complète ce qui manque, sans rien retirer.
    with db_conn.cursor() as cur:
        for col, typ in (("year", "integer"), ("url", "text"), ("source_type", "text"),
                         ("authors", "text"), ("journal", "text"), ("pmid", "text")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        for col, typ in (("token_count", "integer"), ("chunk_weight", "double precision"),
                         ("metadata_json", "jsonb")):
            cur.execute(f"ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("SELECT column_default FROM information_schema.columns "
                    "WHERE table_name = 'literature_document' AND column_name = 'id'")
        row = cur.fetchone()
        if row is not None and row[0] is None:
            cur.execute("CREATE SEQUENCE IF NOT EXISTS literature_document_onework_seq START 900000000")
            cur.execute("ALTER TABLE literature_document ALTER COLUMN id "
                        "SET DEFAULT nextval('literature_document_onework_seq')")
    db_conn.commit()
    try:
        assert _ensure_title_doi_unique_index() == "ok"
        main._ensure_user_scenarios_table()
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"main.engine cannot reach the database ({type(e).__name__})")
    tag = str(int(time.time() * 1000))[-8:]
    yield tag
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE document_id IN (SELECT id FROM literature_document "
                    "WHERE title LIKE %s)", (f"%{tag}%",))
        cur.execute("DELETE FROM document_chunk WHERE document_id IN (SELECT id FROM literature_document "
                    "WHERE title LIKE %s)", (f"%{tag}%",))
        cur.execute("DELETE FROM literature_document WHERE title LIKE %s", (f"%{tag}%",))
        if created_chunk_table:
            cur.execute("DROP TABLE document_chunk")
    db_conn.commit()


def test_the_scenario_keeps_the_published_article_and_the_reviewers_decision(library, db_conn):
    import main
    tag, sid = library, "usr-onework00001"
    title = f"Live poultry exposure and public response to influenza A(H7N9) {tag}"
    update = f"Avian influenza situation update for the European region {tag}"
    abstract = "Seroprevalence of H5N1 antibodies among exposed poultry workers in two provinces."
    rows = [  # (id, title, doi)
        (9860, title, f"10.22541/au.{tag}/v1"),               # le préprint, jugé par un relecteur
        (9861, title, f"10.6084/m9.figshare.{tag}.v1"),       # une figure déposée
        (9862, title, f"10.1371/journal.pone.{tag}"),         # l'article publié, le plus grand id
        (9863, update, f"10.5555/upd.{tag}.1"),
        (9864, update, f"10.5555/upd.{tag}.2"),
    ]
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (sid,))
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 9860 AND 9869")
        for doc_id, t, doi in rows:
            cur.execute("INSERT INTO literature_document (id, title, abstract, source, project_context, doi, "
                        "title_norm, is_duplicate) VALUES (%s, %s, %s, 'crossref', 'literev', %s, %s, false)",
                        (doc_id, t, abstract, doi, main._normalize_title(t)))
            cur.execute("INSERT INTO article_scenarios (document_id, scenario_id, similarity_score) "
                        "VALUES (%s, %s, 0.5)", (doc_id, sid))
        cur.execute("UPDATE article_scenarios SET screening_status = 'excluded', screening_reason = 'wrong population' "
                    "WHERE scenario_id = %s AND document_id = 9860", (sid,))
    db_conn.commit()
    try:
        assert main._dedup_scenario_links(sid) == 2
        with db_conn.cursor() as cur:
            cur.execute("SELECT document_id, screening_status, screening_reason FROM article_scenarios "
                        "WHERE scenario_id = %s ORDER BY document_id", (sid,))
            got = cur.fetchall()
        assert [r[0] for r in got] == [9862, 9863, 9864], "the published article survives; the editions stay two"
        assert got[0][1:] == ("excluded", "wrong population"), "the reviewer's decision follows the work"
    finally:
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (sid,))
            cur.execute("DELETE FROM literature_document WHERE id BETWEEN 9860 AND 9869")
        db_conn.commit()


def test_ingestion_adds_no_record_for_another_version_of_a_published_article(library):
    from api.sources import _ingest_doc_direct
    tag = library

    def ingest(title, doi, ext):
        return _ingest_doc_direct(source="crossref", title=title, abstract=None, year=2024, url=None,
                                  external_id=ext, doi=doi)

    title = f"Occupational exposure of poultry workers to avian influenza {tag}"
    try:
        first, new = ingest(title, f"10.5555/jrn.{tag}", f"crossref:{tag}")
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"l'ingestion ne marche pas sur ce schéma ({type(e).__name__})")
    if first is None:
        pytest.skip("l'ingestion ne marche pas sur ce schéma")
    assert new
    assert ingest(title, f"https://doi.org/10.5555/JRN.{tag}.", f"core:{tag}") == (first, False)
    assert ingest(title, f"10.21203/rs.3.rs-{tag}/v1", f"preprint:{tag}") == (first, False)
    assert ingest(title, f"10.6084/m9.figshare.{tag}.v1", f"core:fig{tag}") == (first, False)
    # Le préprint d'abord, l'article ensuite : l'article garde sa notice (la dédup du
    # scénario le préférera), il n'est pas replié sur le préprint.
    other = f"Risk perception among live bird market vendors {tag}"
    pre, _ = ingest(other, f"10.21203/rs.3.rs-{tag}b/v1", f"preprint:{tag}b")
    pub, new_pub = ingest(other, f"10.5555/jrn.{tag}b", f"crossref:{tag}b")
    assert new_pub and pub != pre
