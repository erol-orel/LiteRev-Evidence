"""Il n'y a qu'UNE définition du sous-ensemble pertinent, et personne ne la réécrit.

Le module qui la porte le dit déjà dans son commentaire : la condition avait été recopiée
à la main dans chaque module, les copies ont divergé, et celle du RAG avait perdu
l'exclusion des doublons ET celle des articles écartés, si bien que l'assistant citait un
article qu'un relecteur venait d'exclure pendant que le compteur affiché sous la réponse
comptait le bon sous-ensemble.

La leçon n'avait pas été tenue : `relevant_gate_sql` existait, et 40 copies écrites à la
main vivaient à côté, dans 11 modules. Le jour où la porte a gagné une clause (le seuil de
rerank), elle l'a gagnée pour UNE requête et pour aucune des quarante autres. Régler un
seuil de rerank vidait donc les artefacts, qui se recalculaient aussitôt sur un ensemble
qui l'ignorait : le compteur disait 201, le brief lisait 602.

Ce test échoue si une copie revient. Il n'interdit pas de compter autre chose : « au-dessus
du seuil » et « écarté à la main alors qu'il était pertinent » sont d'autres questions, et
elles sont listées nommément ci-dessous.
"""
import ast
import pathlib
import re

import pytest

API = pathlib.Path(__file__).resolve().parent.parent / "api"

#: Les seuls endroits où comparer la similarité à un seuil SANS passer par la porte est
#: légitime, parce que la question posée n'est pas « qui est pertinent ».
ALLOWED = {
    # Le compteur « au-dessus du seuil », qui est affiché À CÔTÉ de « pertinent » et doit
    # rester différent de lui : c'est ce qui permet de voir ce que le seuil écarte.
    ("review.py", "above_threshold"),
    # « Écarté à la main alors qu'il passait le seuil » : par construction, des articles
    # que la porte REFUSE. Les faire passer par elle donnerait toujours zéro.
    ("review.py", "manually_vetoed"),
}


def _docstring_lines(src: str) -> set[int]:
    """Les lignes occupées par une docstring.

    Une première version de ce test signalait relevance.py:408, qui est une PHRASE
    expliquant la règle, pas du SQL. Un test qui lit la prose du module ne teste rien."""
    out: set[int] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            d = body[0]
            out.update(range(d.lineno, (d.end_lineno or d.lineno) + 1))
    return out


def _sql_lines():
    for path in sorted(API.glob("*.py")):
        if path.name == "scenario_store.py":        # c'est là que vit la porte
            continue
        src = path.read_text(encoding="utf-8")
        skip = _docstring_lines(src)
        for i, line in enumerate(src.splitlines(), 1):
            if i not in skip:
                yield path.name, i, line


def test_no_module_writes_its_own_relevance_gate():
    offenders = []
    for name, i, line in _sql_lines():
        if "similarity_score" not in line:
            continue
        if not re.search(r"similarity_score,?\s*0?\)?\s*>=", line):
            continue
        if line.lstrip().startswith(("#", "--")) or "--" in line.split("similarity_score")[0]:
            continue
        if any(tag in line for _n, tag in ALLOWED if _n == name):
            continue
        offenders.append(f"{name}:{i}: {line.strip()[:110]}")
    assert not offenders, (
        "une porte de pertinence écrite à la main est revenue ; utilisez "
        "relevant_gate_sql(), sinon la prochaine clause ajoutée à la porte ne vaudra que "
        "pour elle :\n  " + "\n  ".join(offenders))


def test_the_gate_still_carries_both_scores():
    from api.scenario_store import relevant_gate_sql
    sql = relevant_gate_sql()
    assert "similarity_score" in sql and "rerank_score" in sql
    assert "is_duplicate IS NOT TRUE" in sql
    assert "IS DISTINCT FROM 'excluded'" in sql


def _pasted_by_file() -> dict:
    """Les sites qui portent le TEXTE de la porte au lieu de l'appeler.

    Il y en avait 37, dans 11 modules. La liste des copies tolérées a disparu avec
    elles : la clause de rerank, elle, était restée dans la seule requête qui appelle la
    fonction, et c'est ce qui faisait dire 201 au compteur et 602 au brief."""
    out: dict[str, int] = {}
    for name, _i, line in _sql_lines():
        if "rerank_score IS NULL OR" in line:
            out[name] = out.get(name, 0) + 1
    return out


def test_no_module_carries_the_text_of_the_gate():
    now = _pasted_by_file()
    assert not now, (
        "le texte de la porte a été recopié au lieu d'appeler relevant_gate_sql() ou "
        f"relevant_gate_tail_sql() ; fichier -> nombre de copies : {now}")


def test_the_tail_helper_is_the_gate_minus_its_first_clause():
    """La queue est obtenue en coupant la porte sur son premier « AND ». Si la clause des
    doublons cessait d'arriver en tête, treize requêtes perdraient autre chose qu'elle."""
    from api.scenario_store import relevant_gate_sql, relevant_gate_tail_sql
    full, tail = relevant_gate_sql(), relevant_gate_tail_sql()
    assert full.endswith(tail)
    assert full[: -len(tail)] == "d.is_duplicate IS NOT TRUE AND "
    assert "rerank_score" in tail and "IS DISTINCT FROM 'excluded'" in tail


def test_the_shared_sql_constants_really_interpolate_the_gate():
    """Les constantes SQL partagées portent la SORTIE de la fonction, donc elles suivront
    la prochaine clause. Épinglé en comparant leur texte à ce que la fonction rend
    aujourd'hui, sans recharger les modules (un reload rebâtirait des routeurs)."""
    from api.scenario_store import relevant_gate_sql, relevant_gate_tail_sql
    from api import alerts, digest, knowledge_graph
    assert relevant_gate_sql("d", "ars", ":thr") in digest._RELEVANT
    assert relevant_gate_tail_sql("d", "a", ":thr") in alerts._RELEVANT_GATE
    assert relevant_gate_tail_sql("d", "ars", ":thr") in knowledge_graph._CONCEPT_ROWS_SQL


def test_the_allowed_exceptions_still_exist():
    """Si l'un de ces compteurs disparaît, l'exception doit disparaître avec lui, sinon
    elle devient une porte dérobée silencieuse."""
    src = (API / "review.py").read_text(encoding="utf-8")
    for name, tag in ALLOWED:
        assert tag in src, f"l'exception {tag} ne correspond plus à rien dans {name}"
