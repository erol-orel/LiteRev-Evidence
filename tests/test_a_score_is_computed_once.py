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


# ── Un corpus déjà noté n'est pas renvoyé en entier aux APIs ─────────────────
# La plainte d'origine, mesurée : le pipeline complet relançait les deux balayages
# ENTIERS à chaque passage, et il repart à chaque redémarrage de l'API. Sur le scénario
# COVID exposition, cela veut dire 81 766 articles renvoyés à Cohere, lot par lot, pour
# réécrire exactement les mêmes nombres.

def test_the_question_decides_whether_everything_is_rescanned(db_conn):
    """Un balayage complet n'est refait que si la question a changé."""
    sid = "score-pass-scenario"
    R.forget_score_passes(sid)
    assert R.needs_full_score_pass(sid, "influenza AND birds", "rerank") is True
    R.note_full_score_pass(sid, "influenza AND birds", "rerank")
    assert R.needs_full_score_pass(sid, "influenza AND birds", "rerank") is False
    # Une autre question, un autre balayage. Un autre score, aussi : les deux sont séparés.
    assert R.needs_full_score_pass(sid, "influenza AND cats", "rerank") is True
    assert R.needs_full_score_pass(sid, "influenza AND birds", "semantic") is True
    # Et un rescore demandé à la main efface la trace : tout est refait.
    R.forget_score_passes(sid)
    assert R.needs_full_score_pass(sid, "influenza AND birds", "rerank") is True


def test_the_fingerprint_is_the_question_and_nothing_else():
    assert R.score_query_sha("  influenza AND birds ") == R.score_query_sha("influenza AND birds")
    assert R.score_query_sha("influenza AND birds") != R.score_query_sha("influenza AND cats")


def test_the_pipeline_only_fills_the_gaps_when_the_question_has_not_changed():
    import inspect
    from api import pipeline as P
    src = inspect.getsource(P._run_user_scenario_full_pipeline)
    assert 'needs_full_score_pass(scenario_id, query, "semantic")' in src
    assert "missing_only=not _full_sem" in src
    assert 'needs_full_score_pass(scenario_id, query, "rerank")' in src
    assert "only_missing=not _full_ce" in src


def test_a_search_does_not_rerank_the_whole_corpus_again():
    """Le chemin /populate note en arrière-plan ce que la recherche vient d'ajouter."""
    import inspect
    from api import pipeline as P
    src = inspect.getsource(P._run_user_scenario_populate)
    assert 'needs_full_score_pass(_sid, _query, "rerank")' in src
    assert "only_missing=not _full" in src


def test_a_rescore_asked_for_by_hand_still_redoes_everything():
    import inspect
    src = inspect.getsource(R.trigger_rerank)
    assert "forget_score_passes(scenario_id)" in src
    assert 'note_full_score_pass(scenario_id, query, "semantic")' in src
    assert 'note_full_score_pass(scenario_id, query, "rerank")' in src


def test_an_interrupted_pass_is_not_recorded_as_complete():
    """Un balayage que le fournisseur a interrompu laisse des articles sans note : le
    noter comme complet les condamnerait à ne jamais être notés."""
    import inspect
    from api import pipeline as P
    src = inspect.getsource(P._run_user_scenario_full_pipeline)
    assert 'if _full_ce and not _ce_res.get("stopped"):' in src
