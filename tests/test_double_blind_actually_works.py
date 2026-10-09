"""Le double aveugle n'avait jamais fonctionné, et rien dans tests/ ne le touchait.

Quatre défauts qui tenaient ensemble :

1. `ars_row["reviewer_1_status"]` indexait par NOM un `Row` de SQLAlchemy 2, qui ne
   s'indexe que par position. Le TypeError partait à l'intérieur de `engine.begin()`, la
   transaction était annulée, le vote n'était jamais écrit et l'API répondait 500.
   Aucun kappa ne pouvait exister, puisqu'aucun vote ne pouvait être écrit.

2. Le rôle (relecteur 1 ou 2) était attribué PAR NAVIGATEUR, en lisant un
   `sessionStorage` propre à l'onglet. Deux relecteurs sur deux machines, ce qui est la
   définition du double aveugle, recevaient tous les deux le rôle 1 : le second écrasait
   le premier, `reviewer_2_status` restait NULL, et la requête du kappa (qui exige les
   deux colonnes) ne renvoyait rien. Le code envoyé par le client était ignoré : rien ne
   gardait trace de qui avait voté.

3. Les boutons d'arbitrage appelaient l'endpoint de DÉCISION, parce que la route
   d'arbitrage n'existait que sous `/gesica/scenarios/...`, inatteignable pour un
   scénario réel. L'arbitre réécrivait donc `reviewer_1_status`, `r1 == r2` devenait
   vrai, `kappa_resolved` passait à vrai, et le kappa comptait une concordance que
   personne n'avait exprimée.

4. Il n'existait aucun moyen de voter : les deux seuls boutons du panneau étaient ceux
   d'arbitrage, et la liste des conflits ne se remplit que si les deux relecteurs ont
   voté. Le panneau affichait un kappa qui ne pouvait jamais exister.
"""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("psycopg")

import main  # noqa: E402

SID = "double-blind-works"
R1, R2 = "R-1111", "R-2222"


def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


@pytest.fixture()
def seeded(db_conn):
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    with db_conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS document_search")
        cur.execute("DROP TABLE IF EXISTS document_chunk, article_scenarios, "
                    "literature_document CASCADE")
        cur.execute(
            "CREATE TABLE literature_document ("
            "id bigint PRIMARY KEY, title text NOT NULL, source text NOT NULL,"
            "abstract text, doi text, year int, journal text, external_id text,"
            "title_norm text, is_duplicate boolean DEFAULT false, screening_status text,"
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
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("SELECT to_regclass('scenario_settings') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_scenario_settings_table()
        cur.execute("DELETE FROM scenario_reviewer WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters)"
                    " VALUES (%s, 'Double blind', 'avian influenza', 'boolean', '{}')", (SID,))
        cur.executemany(
            "INSERT INTO literature_document (id, title, source, abstract, year)"
            " VALUES (%s, %s, 'pubmed', %s, 2024)",
            [(700 + i, f"Paper {i}", "an abstract long enough to be screened") for i in range(4)])
        cur.executemany(
            "INSERT INTO article_scenarios (scenario_id, document_id, similarity_score)"
            " VALUES (%s, %s, %s)",
            [(SID, 700 + i, 0.9 - i * 0.05) for i in range(4)])
        db_conn.commit()
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM scenario_reviewer WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        db_conn.commit()


def _client():
    from fastapi.testclient import TestClient
    return TestClient(main.app)


KEY = {"X-API-Key": "test-write-key"}


def _vote(c, doc_id: int, status: str, code: str):
    return c.post(f"/user-scenarios/{SID}/double-blind/decision", headers=KEY,
                  json={"article_id": doc_id, "status": status, "reviewer_code": code})


# ── 1. Le 500 ────────────────────────────────────────────────────────────────

def test_a_decision_returns_200_and_is_actually_stored(seeded):
    r = _vote(_client(), 700, "included", R1)
    assert r.status_code == 200, r.text
    assert r.json()["reviewer"] == 1
    with seeded.cursor() as cur:
        cur.execute("SELECT reviewer_1_status FROM article_scenarios"
                    " WHERE scenario_id = %s AND document_id = 700", (SID,))
        assert cur.fetchone()[0] == "included", (
            "le vote n'a pas été écrit : la transaction était annulée par un TypeError")


# ── 2. Le rôle vient du serveur ──────────────────────────────────────────────

def test_two_codes_get_two_different_roles(seeded):
    c = _client()
    assert _vote(c, 700, "included", R1).json()["reviewer"] == 1
    assert _vote(c, 700, "excluded", R2).json()["reviewer"] == 2, (
        "le second relecteur a reçu le rôle 1 : il écrase le premier et le kappa reste "
        "vide après une journée de screening à deux")
    with seeded.cursor() as cur:
        cur.execute("SELECT reviewer_1_status, reviewer_2_status FROM article_scenarios"
                    " WHERE scenario_id = %s AND document_id = 700", (SID,))
        assert cur.fetchone() == ("included", "excluded")


def test_the_role_of_a_code_never_changes(seeded):
    c = _client()
    _vote(c, 700, "included", R1)
    _vote(c, 700, "excluded", R2)
    assert _vote(c, 701, "included", R1).json()["reviewer"] == 1
    assert _vote(c, 701, "included", R2).json()["reviewer"] == 2


def test_a_third_reviewer_is_refused_rather_than_overwriting_one_of_the_two(seeded):
    c = _client()
    _vote(c, 700, "included", R1)
    _vote(c, 700, "excluded", R2)
    r = _vote(c, 701, "included", "R-3333")
    assert r.status_code == 409, r.text
    assert R1 in r.json()["detail"] and R2 in r.json()["detail"]


def test_a_decision_without_a_reviewer_code_is_refused(seeded):
    r = _client().post(f"/user-scenarios/{SID}/double-blind/decision", headers=KEY,
                       json={"article_id": 700, "status": "included"})
    assert r.status_code == 422
    assert "code" in r.json()["detail"].lower()


def test_the_client_cannot_choose_its_own_role(seeded):
    """`reviewer` est encore accepté dans le corps, et doit être IGNORÉ."""
    c = _client()
    _vote(c, 700, "included", R1)                      # R-1111 devient relecteur 1
    r = c.post(f"/user-scenarios/{SID}/double-blind/decision", headers=KEY,
               json={"article_id": 701, "status": "included", "reviewer": 2,
                     "reviewer_code": R1})
    assert r.status_code == 200, r.text
    assert r.json()["reviewer"] == 1, "le client s'est donné le rôle 2 en le demandant"


def test_the_registration_endpoint_returns_the_role_and_both_codes(seeded):
    c = _client()
    a = c.post(f"/user-scenarios/{SID}/double-blind/register?reviewer_code=1111", headers=KEY)
    assert a.status_code == 200, a.text
    assert a.json()["reviewer"] == 1 and a.json()["reviewer_code"] == R1
    b = c.post(f"/user-scenarios/{SID}/double-blind/register?reviewer_code=R-2222", headers=KEY)
    assert b.json()["reviewer"] == 2
    assert b.json()["registered"] == {"1": R1, "2": R2}


# ── 3. L'arbitrage ne fabrique plus la concordance ───────────────────────────

def test_a_vote_is_written_once_and_an_arbitration_cannot_rewrite_it(seeded):
    c = _client()
    _vote(c, 700, "included", R1)
    _vote(c, 700, "excluded", R2)
    again = _vote(c, 700, "excluded", R1)
    assert again.status_code == 409, (
        "un second vote du même relecteur a été accepté : c'est ainsi que l'arbitrage "
        "réécrivait un vote et fabriquait la concordance que comptait le kappa")
    with seeded.cursor() as cur:
        cur.execute("SELECT reviewer_1_status, reviewer_2_status, kappa_resolved"
                    " FROM article_scenarios WHERE scenario_id = %s AND document_id = 700",
                    (SID,))
        r1, r2, resolved = cur.fetchone()
    assert (r1, r2) == ("included", "excluded")
    assert resolved is False, "un désaccord a été marqué résolu"


def test_the_arbitration_route_exists_for_a_real_scenario(seeded):
    c = _client()
    _vote(c, 700, "included", R1)
    _vote(c, 700, "excluded", R2)
    r = c.post(f"/user-scenarios/{SID}/double-blind/resolve"
               f"?article_id=700&final_status=included", headers=KEY)
    assert r.status_code == 200, r.text
    with seeded.cursor() as cur:
        cur.execute("SELECT reviewer_1_status, reviewer_2_status, kappa_final_status,"
                    " screening_status FROM article_scenarios"
                    " WHERE scenario_id = %s AND document_id = 700", (SID,))
        r1, r2, final, screening = cur.fetchone()
    assert (r1, r2) == ("included", "excluded"), "l'arbitrage a modifié un vote"
    assert final == "included" and screening == "included"


def test_the_kappa_counts_the_two_real_votes(seeded):
    c = _client()
    for doc in (700, 701, 702):
        _vote(c, doc, "included", R1)
    _vote(c, 700, "included", R2)
    _vote(c, 701, "excluded", R2)
    _vote(c, 702, "included", R2)
    k = c.get(f"/user-scenarios/{SID}/double-blind/kappa").json()
    assert k["n_evaluated"] == 3
    assert k["conflicts"] == 1
    assert k["kappa"] is not None


def test_a_conflict_names_the_two_reviewers(seeded):
    c = _client()
    _vote(c, 700, "included", R1)
    _vote(c, 700, "excluded", R2)
    rows = c.get(f"/user-scenarios/{SID}/double-blind/conflicts").json()
    assert len(rows) == 1
    assert rows[0]["reviewer_1_code"] == R1 and rows[0]["reviewer_2_code"] == R2, (
        "le panneau affichait ces deux colonnes alors qu'elles n'existaient nulle part : "
        "un conflit ne nommait personne")


# ── 4. Il est possible de voter ──────────────────────────────────────────────

def test_the_queue_offers_the_relevant_articles_this_reviewer_has_not_judged(seeded):
    c = _client()
    q = c.get(f"/user-scenarios/{SID}/double-blind/queue?reviewer_code={R1}").json()
    assert q["reviewer"] == 1
    assert q["remaining"] == 4 and len(q["articles"]) == 4
    # Trié par pertinence : on ne fait pas juger n'importe quoi en premier.
    assert [a["id"] for a in q["articles"]] == [700, 701, 702, 703]
    _vote(c, 700, "included", R1)
    q2 = c.get(f"/user-scenarios/{SID}/double-blind/queue?reviewer_code={R1}").json()
    assert q2["remaining"] == 3
    assert 700 not in [a["id"] for a in q2["articles"]]
    # Et le lot de l'AUTRE relecteur est intact : chacun juge en aveugle.
    q3 = c.get(f"/user-scenarios/{SID}/double-blind/queue?reviewer_code={R2}").json()
    assert q3["reviewer"] == 2 and q3["remaining"] == 4


def test_the_queue_leaves_out_what_the_review_did_not_retain(seeded):
    """Le lot est le sous-ensemble pertinent : on ne demande pas de juger un article que
    la revue a écarté."""
    with seeded.cursor() as cur:
        cur.execute("UPDATE article_scenarios SET screening_status = 'excluded'"
                    " WHERE scenario_id = %s AND document_id = 703", (SID,))
        cur.execute("UPDATE literature_document SET is_duplicate = TRUE WHERE id = 702")
        seeded.commit()
    q = _client().get(f"/user-scenarios/{SID}/double-blind/queue?reviewer_code={R1}").json()
    assert [a["id"] for a in q["articles"]] == [700, 701]
    assert q["remaining"] == 2
