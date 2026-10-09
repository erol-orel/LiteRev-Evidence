"""La dédup intra-scénario FUSIONNE avant de supprimer.

`_dedup_scenario_links` garde, pour chaque clé (DOI, external_id normalisé, titre long),
le lien de plus PETIT `document_id`. Ce choix est voulu : il doit rester celui de
`scripts/_softdedup.py`, sinon la dédup globale marquera d'autres lignes que celles déjà
retirées ici.

Mais le plus petit id n'a aucun rapport avec ce que le lien PORTE. Les deux lignes
viennent de deux chemins d'ingestion (live « pmid:123 », populate « 123 ») et c'est
souvent la SECONDE, celle de plus grand id, qui a été scorée et jugée : son lien portait
la décision du relecteur, ses motifs, ses votes de double aveugle et ses deux scores, et
la suppression les emportait sans un mot. Un relecteur voyait son exclusion revenir « en
attente » après une relance de la recherche.

On ne remplace jamais une valeur du survivant : on ne comble que ses trous.
"""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("psycopg")

import main  # noqa: E402

SID = "dedup-merge-test"


def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


@pytest.fixture()
def two_links(db_conn):
    """Deux lignes pour le MÊME article (même DOI), liées au même scénario. Celle de
    plus grand id porte tout ; celle de plus petit id ne porte rien."""
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    with db_conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS document_search")
        cur.execute("DROP TABLE IF EXISTS document_chunk, article_scenarios, "
                    "literature_document CASCADE")
        cur.execute(
            "CREATE TABLE literature_document ("
            "id bigint PRIMARY KEY, title text NOT NULL, source text NOT NULL,"
            "abstract text, doi text, external_id text, title_norm text,"
            "is_duplicate boolean DEFAULT false, screening_status text,"
            "project_context text DEFAULT 'literev')")
        cur.execute(
            "CREATE TABLE article_scenarios (scenario_id text, document_id bigint,"
            "similarity_score double precision, rerank_score double precision,"
            "screening_status text, screening_reason text, screening_notes text,"
            "screened_at timestamp, reviewer_1_status varchar(20), reviewer_1_reason text,"
            "reviewer_2_status varchar(20), reviewer_2_reason text,"
            "kappa_resolved boolean DEFAULT false, kappa_final_status varchar(20),"
            "cluster_id integer, cluster_label text,"
            "PRIMARY KEY (scenario_id, document_id))")
        cur.execute(
            "INSERT INTO literature_document (id, title, source, abstract, doi, external_id)"
            " VALUES (10, 'Same paper', 'pubmed', 'abstract long enough to count',"
            "         '10.1000/same', 'pmid:123'),"
            "        (20, 'Same paper', 'openalex', 'abstract long enough to count',"
            "         '10.1000/same', '123')")
        # 10 = canonique (plus petit id), vide. 20 = tout le travail du relecteur.
        cur.execute(
            "INSERT INTO article_scenarios (scenario_id, document_id) VALUES (%s, 10)", (SID,))
        cur.execute(
            "INSERT INTO article_scenarios (scenario_id, document_id, similarity_score,"
            " rerank_score, screening_status, screening_reason, reviewer_1_status,"
            " reviewer_1_reason, cluster_id, cluster_label)"
            " VALUES (%s, 20, 0.71, 0.93, 'excluded', 'population hors sujet',"
            "         'excluded', 'pas la bonne espèce', 4, 'volailles')", (SID,))
        db_conn.commit()
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        db_conn.commit()


def _survivor(db_conn):
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT document_id, similarity_score, rerank_score, screening_status,"
            " screening_reason, reviewer_1_status, reviewer_1_reason, cluster_id,"
            " cluster_label FROM article_scenarios WHERE scenario_id = %s", (SID,))
        return cur.fetchall()


def test_the_reviewer_decision_survives_the_deduplication(two_links):
    removed = main._dedup_scenario_links(SID)
    assert removed == 1, "le lien doublon n'a pas été retiré"
    rows = _survivor(two_links)
    assert len(rows) == 1
    (doc_id, sim, rer, status, reason, r1, r1r, cl, cll) = rows[0]
    assert doc_id == 10, "le canonique doit rester le plus petit id (cf. _softdedup.py)"
    assert status == "excluded", (
        "l'exclusion du relecteur a été supprimée avec le lien qui la portait ; "
        "elle revenait « en attente » après chaque relance de la recherche")
    assert reason == "population hors sujet"
    assert (r1, r1r) == ("excluded", "pas la bonne espèce")
    assert sim == pytest.approx(0.71) and rer == pytest.approx(0.93)
    assert (cl, cll) == (4, "volailles")


def test_a_value_already_on_the_survivor_is_never_overwritten(two_links):
    with two_links.cursor() as cur:
        cur.execute("UPDATE article_scenarios SET screening_status = 'included',"
                    " similarity_score = 0.42 WHERE scenario_id = %s AND document_id = 10",
                    (SID,))
        two_links.commit()
    main._dedup_scenario_links(SID)
    rows = _survivor(two_links)
    assert len(rows) == 1
    (_doc, sim, rer, status, *_rest) = rows[0]
    assert status == "included", "la fusion a écrasé la décision du survivant"
    assert sim == pytest.approx(0.42), "la fusion a écrasé un score du survivant"
    assert rer == pytest.approx(0.93), "le trou du survivant n'a pas été comblé"


def test_nothing_happens_when_there_is_no_duplicate(two_links):
    with two_links.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s AND document_id = 10",
                    (SID,))
        two_links.commit()
    assert main._dedup_scenario_links(SID) == 0
    rows = _survivor(two_links)
    assert len(rows) == 1 and rows[0][0] == 20
    assert rows[0][3] == "excluded"
