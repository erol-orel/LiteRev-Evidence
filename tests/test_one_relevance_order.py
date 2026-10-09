"""Il n'y a qu'UN ordre de pertinence, et l'export le partage avec l'écran.

Il y en avait trois, pour la même question :

  - l'onglet Corpus triait par « au-dessus du seuil, puis reranké, puis rerank, puis
    similarité, puis année, puis citations, puis titre » ;
  - l'export relisait « inclus à la main, puis similarité, puis citations, puis id »,
    sous une docstring promettant « le même ensemble et le même ordre que l'onglet
    Corpus » ;
  - l'export par identifiants (un cluster, un concept, une réponse du RAG) en avait un
    troisième, « inclus, puis COALESCE(rerank, similarité) ».

Un relecteur qui compare son écran au fichier qu'il vient de télécharger ne retrouvait
pas ses dix premiers articles, et rien ne lui disait pourquoi.
"""
import pathlib
import re

import pytest

pytest.importorskip("fastapi")

from api.scenario_store import relevance_order_sql  # noqa: E402

API = pathlib.Path(__file__).resolve().parent.parent / "api"


def test_the_order_is_a_pure_function_of_its_aliases():
    sql = relevance_order_sql("d", "ars")
    assert "ars.rerank_score DESC NULLS LAST" in sql
    assert "ars.similarity_score DESC NULLS LAST" in sql
    assert "d.citation_count DESC NULLS LAST" in sql
    assert sql.endswith("d.id")
    other = relevance_order_sql("ld", "asn")
    assert "ld." in other and "asn." in other
    assert " d." not in other and "ars." not in other


def test_a_threshold_puts_the_relevant_first_and_nothing_else_changes():
    without = relevance_order_sql("d", "ars")
    with_thr = relevance_order_sql("d", "ars", ":thr")
    assert with_thr.endswith(without)
    assert with_thr.startswith(
        "CASE WHEN COALESCE(ars.similarity_score, 0) >= :thr THEN 0 ELSE 1 END ASC, ")


def test_the_screening_status_read_is_the_per_scenario_one():
    """Un article exclu dans une AUTRE revue ne doit pas remonter en tête de celle-ci."""
    sql = relevance_order_sql("d", "ars")
    assert "(ars.screening_status = 'included') DESC" in sql
    assert "d.screening_status" not in sql


def test_no_module_writes_its_own_relevance_order():
    """Un ORDER BY qui réécrit cette suite est une quatrième vérité.

    On cherche la signature de l'ancien tri : un ORDER BY qui classe par
    `similarity_score DESC` sans passer par la fonction. Les tris qui répondent à une
    AUTRE question (par année, par titre, par date de création) ne sont pas concernés."""
    #: Le seul tri par score qui ne classe PAS un corpus : « dans quels scénarios cet
    #: article se trouve-t-il ? », ordonné par son score dans chacun. Ce ne sont pas des
    #: articles qui sont classés, ce sont des scénarios.
    allowed = {("gesica.py", "ars.scenario_id, ars.similarity_score, ars.assigned_at")}
    offenders = []
    for path in sorted(API.glob("*.py")):
        if path.name == "scenario_store.py":
            continue
        src = path.read_text(encoding="utf-8")
        for m in re.finditer(r"ORDER BY((?:[^\"']|\n){0,400}?)(?:LIMIT|\"\"\"|\n\s*\n)", src):
            block = m.group(1)
            if "relevance_order_sql" in block:
                continue
            if "similarity_score DESC" not in block and "rerank_score DESC" not in block:
                continue
            _ctx = src[max(0, m.start() - 400):m.start()]
            if any(path.name == n and sig in _ctx for n, sig in allowed):
                continue
            line = src[: m.start()].count("\n") + 1
            offenders.append(f"{path.name}:{line}: {' '.join(block.split())[:110]}")
    assert not offenders, (
        "un ordre de pertinence écrit à la main ; utilisez relevance_order_sql() pour "
        "que l'écran et le fichier téléchargé donnent la même liste :\n  "
        + "\n  ".join(offenders))
