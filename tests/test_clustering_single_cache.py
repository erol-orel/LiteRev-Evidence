"""Le clustering avait DEUX caches, et l'invalidation n'en connaissait qu'un.

`_persist_clustering_result` écrivait le payload en base ET sous
`/tmp/literev_clustering_cache/{sid}.json`, avec son propre TTL de 24 h.
`_run_clustering_background` lisait la copie /tmp AVANT tout calcul et, sur un succès,
la réinstallait en base avec un horodatage frais.

Déroulé observé en production : bouger le seuil vide `clustering_json` par
CORPUS_DERIVED_CACHE_RESET, la copie /tmp du pool PRÉCÉDENT est encore dans son TTL, elle
est servie, puis réécrite en base. Le scénario HPAI annonçait « 602 articles, 5 clusters »
pendant que son corpus pertinent en comptait 201, avec clustering_generated_at égal à
updated_at à la microseconde près, APRÈS le PATCH du seuil. Trois scénarios sur les dix-neuf
portant un clustering annonçaient plus d'articles que leur sous-ensemble pertinent n'en
contient, ce qui ne peut pas arriver : l'entrée du clustering est un sous-ensemble strict
du pool pertinent.

Un cache que l'invalidation ne peut pas atteindre n'est pas un cache, c'est une seconde
vérité. Ces tests épinglent qu'il n'y en a plus qu'une.
"""
import ast
import inspect

import pytest

pytest.importorskip("fastapi")

from api import clustering as C  # noqa: E402


def _code_only(fn) -> str:
    """La source SANS les docstrings.

    Première version de ces tests : ils cherchaient "/tmp" dans la source et tombaient sur
    ma propre docstring, qui raconte justement le bug. Un test qui lit sa propre prose ne
    teste rien."""
    src = inspect.getsource(fn)
    tree = ast.parse(src.lstrip() if src.startswith((" ", "\t")) else src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return ast.unparse(tree)


def test_nothing_in_the_module_writes_a_second_cache():
    src = _code_only(C)
    assert "literev_clustering_cache" not in src, (
        "la copie /tmp est revenue : elle survit à l'invalidation et ressuscite "
        "le clustering du pool précédent")


def test_persist_writes_only_the_database():
    src = _code_only(C._persist_clustering_result)
    assert "_save_viz_cache" in src
    assert "open(" not in src and "/tmp" not in src


def test_the_background_run_does_not_read_a_file_before_computing():
    """C'était le chemin de la résurrection : lire /tmp d'abord, servir, réécrire."""
    src = _code_only(C._run_clustering_background)
    for forbidden in ("os.path.exists(", "getmtime(", "/tmp"):
        assert forbidden not in src, f"lecture de cache fichier réintroduite : {forbidden}"


def test_the_database_cache_is_still_the_one_that_remains():
    """On a retiré un cache, pas les deux : le durable doit rester."""
    assert callable(C._load_viz_cache) and callable(C._save_viz_cache)
    assert C.VIZ_CACHE_TTL_S > 0


def test_the_reset_list_reaches_the_cache_that_is_left():
    """`clustering_json` est bien la colonne que l'invalidation vide."""
    from api.scenario_store import CORPUS_DERIVED_CACHE_RESET
    assert "clustering_json" in CORPUS_DERIVED_CACHE_RESET
    assert "clustering_generated_at" in CORPUS_DERIVED_CACHE_RESET
