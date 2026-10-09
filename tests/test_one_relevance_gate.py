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
        if "rerank_score IS NULL OR" in line:       # copie connue, voir KNOWN_PASTED
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


#: Les sites qui portent le TEXTE de la porte au lieu de l'appeler. La requête qu'ils
#: exécutent est aujourd'hui exactement celle que la fonction produit, parce qu'elle y a
#: été copiée depuis sa sortie ; mais une clause ajoutée demain à la fonction ne les
#: suivra pas, ce qui est précisément le défaut que l'unification devait supprimer.
#: Ce nombre ne doit que DIMINUER. Le convertir demande de transformer 34 chaînes SQL en
#: f-strings, ce qui se fait fichier par fichier et se vérifie requête par requête.
KNOWN_PASTED = {
    "evidence.py": 20, "digest.py": 4, "knowledge_graph.py": 4, "assistant.py": 2,
    "alerts.py": 1, "clustering.py": 1, "relevance.py": 1, "review.py": 1,
    "seir.py": 1, "sources.py": 1, "variables.py": 1,
}


def _pasted_by_file() -> dict:
    out: dict[str, int] = {}
    for name, _i, line in _sql_lines():
        if "rerank_score IS NULL OR" in line:
            out[name] = out.get(name, 0) + 1
    return out


def test_the_pasted_copies_never_grow():
    """Une copie de plus est une régression, même si son SQL est juste le jour où elle
    est écrite. C'est le texte qui est le problème, pas sa valeur actuelle."""
    now = _pasted_by_file()
    worse = {f: (n, KNOWN_PASTED.get(f, 0)) for f, n in now.items() if n > KNOWN_PASTED.get(f, 0)}
    assert not worse, (
        "le texte de la porte a été recopié dans de nouveaux endroits au lieu d'appeler "
        f"relevant_gate_sql() ; fichier -> (maintenant, connu) : {worse}")


def test_the_known_list_is_not_stale():
    """Si un fichier a été converti, il doit sortir de la liste, sinon elle protège un
    problème qui n'existe plus et masque le suivant."""
    now = _pasted_by_file()
    stale = {f: n for f, n in KNOWN_PASTED.items() if now.get(f, 0) < n}
    assert not stale, (
        "des copies ont été converties sans mettre KNOWN_PASTED à jour ; "
        f"fichier -> ancien compte : {stale}")


def test_the_allowed_exceptions_still_exist():
    """Si l'un de ces compteurs disparaît, l'exception doit disparaître avec lui, sinon
    elle devient une porte dérobée silencieuse."""
    src = (API / "review.py").read_text(encoding="utf-8")
    for name, tag in ALLOWED:
        assert tag in src, f"l'exception {tag} ne correspond plus à rien dans {name}"
