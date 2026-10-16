"""Un décorateur de route porte sur la fonction qu'il annonce, pas sur la suivante.

Un helper privé inséré entre `@app.get(...)` et son handler a suffi : la route
`/user-scenarios/{id}/embedding-status` s'est retrouvée servie par le helper, qui renvoie
deux clés au lieu du statut complet. L'API répondait 200, aucun test ne regardait la forme
de la réponse, et l'onglet Corpus tombait en erreur sur `score_availability` indéfini.

Deux garde-fous, purs et sans base :

1. aucune route de l'application n'est servie par un nom privé (la convention du dépôt :
   les handlers sont publics, les helpers commencent par un souligné) ;
2. le statut d'embedding rend bien les champs que l'interface lit.
"""
import inspect

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402

API_ROUTES = [r for r in main.app.routes if getattr(r, "endpoint", None) and getattr(r, "path", "")]


def test_no_route_is_served_by_a_private_helper():
    private = sorted(
        f"{sorted(r.methods or [])} {r.path} -> {r.endpoint.__name__}"
        for r in API_ROUTES
        if r.endpoint.__name__.startswith("_")
    )
    assert private == [], (
        "une route est servie par un helper privé, signe d'un décorateur posé sur la "
        f"mauvaise fonction : {private}"
    )


def _route(path: str, method: str = "GET"):
    for r in API_ROUTES:
        if r.path == path and method in (r.methods or set()):
            return r
    raise AssertionError(f"route absente : {method} {path}")


def test_the_embedding_status_route_is_the_status_handler():
    route = _route("/user-scenarios/{scenario_id}/embedding-status")
    assert route.endpoint.__name__ == "get_user_scenario_embedding_status"


def test_the_embedding_status_answers_what_the_scores_card_reads():
    """La carte des scores lit `score_availability.semantic` et `ranking.failed` : les
    deux doivent sortir du handler, sinon l'onglet Corpus ne s'affiche plus du tout."""
    src = inspect.getsource(_route("/user-scenarios/{scenario_id}/embedding-status").endpoint)
    for field in ('"score_availability"', '"semantic"', '"ranking"', "_score_failures_for"):
        assert field in src, f"le statut d'embedding doit rendre {field}"
