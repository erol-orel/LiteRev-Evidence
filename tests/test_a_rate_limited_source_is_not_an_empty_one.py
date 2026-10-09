"""Une source limitée en débit, ou coupée en cours, ne se lit pas « aucun résultat ».

Vu en déroulant un scénario NEUF de bout en bout, sans réseau : Semantic Scholar
répondait 429 et le fetcher réessayait **sans limite** (`if 429: sleep(2); continue`).
Trois conséquences, toutes sur la première recherche d'un nouveau scénario :

1. La boucle tenait son fil jusqu'au budget de fédération (180 s par défaut). Les onze
   autres sources avaient fini en quelques secondes ; la recherche affichait pourtant
   « en cours » pendant trois minutes, et rien dans l'interface n'en disait la raison.
2. Elle frappait l'API une quarantaine de fois de plus pendant ce temps, ce qui durcit
   précisément la limite qu'elle attendait.
3. Elle sortait par le garde-temps avec zéro article et SANS marquer d'erreur. L'issue
   calculée était donc `empty`, c'est-à-dire une affirmation sur la LITTÉRATURE
   (« cette base ne contient rien sur le sujet ») pour une source qui n'avait jamais
   répondu.

Le même silence valait pour les treize boucles de pagination : elles testaient le budget
et sortaient sans rien marquer, si bien qu'une source arrêtée au milieu de ses pages
annonçait `ok` comme si elle avait fini, et que `_fed_incomplete` restait faux, ce qui
autorisait le corpus à se vider sur une fédération pourtant partielle.

DEUXIÈME VERSION de ce fichier. La première lisait le TEXTE du module : elle comptait des
occurrences, mesurait des décalages, et la relecture adversariale a montré qu'elle laissait
passer la suppression de la borne (le compteur retiré chez les appelants, la boucle de
nouveau infinie, le fichier vert) tout en cassant sur une modification correcte (une
ligne ajoutée dans le bloc des issues, le fichier rouge). Les comportements sont désormais
mesurés dans `test_the_rate_limit_bound_actually_holds.py`, qui pilote la fédération
réelle et COMPTE les requêtes. Ce fichier ne garde que ce qui se vérifie structurellement
sans dépendre d'un décalage : l'arbre syntaxique, pas la position des caractères.
"""
import ast
import inspect
import textwrap

import pytest

pytest.importorskip("fastapi")

from api import pipeline as P  # noqa: E402
from api.search import SOURCE_OUTCOMES, SOURCE_OUTCOMES_COUNTED  # noqa: E402

SRC = inspect.getsource(P._run_user_scenario_populate)
TREE = ast.parse(textwrap.dedent(SRC))


def _func(name: str) -> ast.FunctionDef:
    for n in ast.walk(TREE):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(f"{name} a disparu de _run_user_scenario_populate")


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [c for c in ast.walk(node)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == name]


# ── La borne existe, et c'est une borne ──────────────────────────────────────

def test_the_rate_limit_wait_raises_when_the_attempts_run_out():
    """Épuiser les tentatives doit LEVER : c'est ce qui déclenche `_mark_source_error`.

    Lu dans l'arbre : la fonction contient un `raise` dont la condition compare `attempt`
    au plafond. Le comportement mesuré, lui, est dans l'autre fichier."""
    fn = _func("_wait_out_rate_limit")
    raises = [n for n in ast.walk(fn) if isinstance(n, ast.Raise)]
    assert len(raises) >= 2, "il faut lever à l'épuisement ET quand le budget ne laisse plus le temps"
    guards = [n for n in ast.walk(fn) if isinstance(n, ast.If)
              and "attempt" in ast.unparse(n.test) and "_RATE_LIMIT_ATTEMPTS" in ast.unparse(n.test)]
    assert guards, "aucune condition ne compare le nombre de tentatives au plafond"


def test_the_wait_never_outlives_the_federation_deadline():
    fn = _func("_wait_out_rate_limit")
    src = ast.unparse(fn)
    assert "_fed_deadline[0]" in src and "min(" in src, (
        "l'attente ne borne pas le délai par le temps qui reste au budget")


def test_every_429_retry_site_passes_a_counter_that_it_increments():
    """La borne vit chez les APPELANTS, qui passent le compteur : c'est là que la relecture
    a montré qu'on pouvait la supprimer sans qu'aucun test ne bronche.

    Pour chaque appel `_wait_out_rate_limit(resp, <compteur>, source)`, le même compteur
    doit être incrémenté dans la même boucle. Sans cela il vaut toujours 0 et le plafond
    n'est jamais atteint."""
    sites = _calls(TREE, "_wait_out_rate_limit")
    assert len(sites) >= 3, f"{len(sites)} site(s) de retry trouvés, 3 attendus"
    for call in sites:
        assert len(call.args) >= 2, "l'appel ne passe pas de compteur"
        counter = call.args[1]
        assert isinstance(counter, ast.Name), "le compteur doit être une variable, pas une constante"
        # Trouver la boucle `while` englobante et vérifier qu'elle incrémente ce nom.
        loop = None
        for n in ast.walk(TREE):
            if isinstance(n, ast.While) and any(c is call for c in ast.walk(n)):
                loop = n
        assert loop is not None, "le retry n'est pas dans une boucle"
        increments = [n for n in ast.walk(loop) if isinstance(n, ast.AugAssign)
                      and isinstance(n.target, ast.Name) and n.target.id == counter.id]
        assert increments, (
            f"`{counter.id}` est passé à _wait_out_rate_limit mais jamais incrémenté dans "
            "sa boucle : la borne n'existe pas, la boucle est de nouveau infinie")


# ── Les boucles de pagination marquent la source qu'elles coupent ────────────

def test_every_pagination_loop_checks_the_budget_through_the_marking_helper():
    """Aucune boucle ne teste `_fed_deadline` à la main : toutes passent par
    `_budget_exhausted()`, qui marque la source. Compté sur l'ARBRE, et sans nombre en dur :
    ajouter une source correctement ne doit pas casser ce test."""
    manual = [n for n in ast.walk(TREE) if isinstance(n, ast.Compare)
              and "_fed_deadline[0]" in ast.unparse(n)
              and any(isinstance(op, (ast.GtE, ast.Gt)) for op in n.ops)]
    # La seule comparaison « >= deadline » tolérée est celle de _budget_exhausted lui-même.
    inside_helper = {id(n) for n in ast.walk(_func("_budget_exhausted")) if isinstance(n, ast.Compare)}
    offenders = [ast.unparse(n) for n in manual if id(n) not in inside_helper]
    assert not offenders, (
        "une boucle teste encore le budget à la main, sans marquer la source : "
        + "; ".join(offenders))
    loops_with_check = [n for n in ast.walk(TREE) if isinstance(n, ast.While)
                        and _calls(n, "_budget_exhausted")]
    assert len(loops_with_check) >= 12, (
        f"seulement {len(loops_with_check)} boucles de pagination passent par "
        "_budget_exhausted() ; une source ajoutée sans ce garde se lira « vide » quand le "
        "budget la coupe")


def test_a_cut_source_also_makes_the_federation_incomplete():
    """Sinon le corpus est autorisé à se vider sur un fetch partiel."""
    fn = _func("_budget_exhausted")
    src = ast.unparse(fn)
    assert "_fed_incomplete[0] = True" in src
    assert "_fetcher_cut.add" in src


# ── La table des issues ──────────────────────────────────────────────────────

def _outcome_loop() -> ast.For:
    """La boucle `for _fn in source_funcs:` qui décide l'issue de chaque source."""
    for n in ast.walk(TREE):
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "_fn" \
                and "source_funcs" in ast.unparse(n.iter):
            return n
    raise AssertionError("la boucle des issues a disparu")


def test_cut_by_budget_is_decided_before_ok_or_empty():
    """Une source coupée ne doit pas retomber sur ok/empty.

    Lu sur la chaîne de `if/elif` de la boucle : la branche qui assigne `cut_by_budget`
    précède celle qui assigne `ok`/`empty`, et sa condition lit `_fetcher_cut`."""
    loop = _outcome_loop()
    order = []
    for n in ast.walk(loop):
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id == "_outcome":
            order.append(ast.unparse(n.value))
    cut_i = next((i for i, v in enumerate(order) if "cut_by_budget" in v), None)
    ok_i = next((i for i, v in enumerate(order) if "'ok'" in v or '"ok"' in v), None)
    assert cut_i is not None and ok_i is not None, order
    assert cut_i < ok_i, f"`ok`/`empty` est décidé avant `cut_by_budget` : {order}"
    conds = [ast.unparse(n.test) for n in ast.walk(loop) if isinstance(n, ast.If)]
    assert any("_fetcher_cut" in c for c in conds), (
        "aucune condition de la table ne lit `_fetcher_cut` : une source arrêtée en cours "
        "de pagination retombe sur ok/empty")


def test_the_counted_outcomes_come_from_the_shared_constant():
    """Deux vérités sinon : la liste réécrite à la main ici, et SOURCE_OUTCOMES_COUNTED."""
    loop = _outcome_loop()
    assert "SOURCE_OUTCOMES_COUNTED" in ast.unparse(loop)
    assert "cut_by_budget" in SOURCE_OUTCOMES
    assert "cut_by_budget" not in SOURCE_OUTCOMES_COUNTED


def test_the_total_found_is_what_the_sources_returned_not_what_was_inserted():
    """`total_found = ingested` annonçait moins d'enregistrements que d'identifiés."""
    # Le `total_found` du JOB, au niveau de la fonction : `_fetch_pubmed` a le sien,
    # local, qui porte le compte de l'esearch et n'est pas en cause.
    outer = next(n for n in TREE.body if isinstance(n, ast.FunctionDef))

    def _top_level(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            yield child
            yield from _top_level(child)

    assigns = [n for n in _top_level(outer) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "total_found" for t in n.targets)]
    assert assigns, "total_found n'est plus assigné au niveau du job"
    assert all("_ident_records" in ast.unparse(a.value) for a in assigns), (
        [ast.unparse(a) for a in assigns])
