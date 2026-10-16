"""Un score calculé n'est pas recalculé, et un échec n'est pas réessayé sans fin.

La carte des scores d'un scénario affichait « 48 578 sur 48 583 · 5 en cours… » des jours
après la fin de la recherche. Les cinq n'étaient pas en cours : l'API d'embedding les
refusait ou les omettait de sa réponse, rien ne le notait, donc chaque passage automatique
et chaque ouverture de la page du corpus les re-sélectionnait, pour échouer de nouveau.

Deux garanties sont pinées ici :

1. un article qui PORTE un score n'est plus sélectionné par un passage automatique, donc
   il n'est jamais recalculé pour rien (et le seuil n'y change rien : le score est le
   cosinus question-article, le seuil ne fait que trancher ensuite) ;
2. un article que l'API ne parvient pas à scorer sort de la sélection automatique au bout
   de SCORE_MAX_ATTEMPTS essais, et l'interface le dit au lieu d'afficher « en cours ».
"""
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from api import relevance as R  # noqa: E402

pytestmark = pytest.mark.skipif(not (os.getenv("DB_URL") or os.getenv("DATABASE_URL")),
                                reason="pas de base")


def test_an_article_with_a_score_is_not_picked_up_again():
    """La sélection d'un passage automatique porte sur `similarity_score IS NULL` : un
    article scoré en est exclu par construction, quel que soit le seuil."""
    from api.scenario_store import unscored_sql
    sql = unscored_sql("d", "ars")
    assert "similarity_score IS NULL" in sql
    # Et rien dans cette condition ne regarde le seuil : le score ne dépend pas de lui.
    assert "threshold" not in sql.lower()


def test_an_article_that_cannot_be_scored_leaves_the_automatic_selection():
    gate = R._not_given_up_sql("asn")
    assert "score_attempts" in gate
    assert str(R.SCORE_MAX_ATTEMPTS) in gate


def test_the_cap_is_at_least_one_try():
    assert R.SCORE_MAX_ATTEMPTS >= 1


def test_the_counts_leave_out_what_will_never_be_scored(monkeypatch):
    """`retryable` pilote le réarmement du scoring automatique : s'il comptait les
    articles déjà abandonnés, le scoring se relancerait sans fin pour eux."""
    import inspect
    src = inspect.getsource(R._retryable_unscored)
    assert "_not_given_up_sql" in src, "le compte des articles à réessayer doit exclure les abandons"


def test_a_requested_rescore_gives_the_failures_another_chance(monkeypatch):
    """Un rescore COMPLET demandé par l'utilisateur repart de zéro : qui clique veut qu'on
    réessaie, y compris ce qui avait échoué."""
    import inspect
    src = inspect.getsource(R._run_semantic_rerank_inline)
    assert "_clear_score_attempts(scenario_id)" in src
    assert "if not missing_only:" in src


def test_an_omitted_article_is_counted_as_a_failed_try():
    """Le cas qui a produit les cinq : l'API renvoie moins d'embeddings qu'on ne lui a
    donné de textes, et les manquants étaient simplement sautés."""
    import inspect
    src = inspect.getsource(R._run_semantic_rerank_inline)
    assert "omitted by the embedding API" in src
    assert "_note_score_attempts" in src


def test_a_quota_pause_is_not_blamed_on_the_articles():
    """Une pause de quota n'est pas la faute de l'article : la compter contre lui le
    ferait abandonner après trois pauses, alors qu'il n'a jamais été essayé."""
    import inspect
    src = inspect.getsource(R._run_semantic_rerank_inline)
    assert "if not _is_openai_quota_error(_e):" in src


def test_the_status_says_what_could_not_be_scored():
    import inspect
    src = inspect.getsource(R.get_rerank_status)
    for field in ("similarity_failed", "similarity_failed_reason", "similarity_retryable"):
        assert field in src, f"{field} doit être exposé pour que l'écran cesse de dire « en cours »"
