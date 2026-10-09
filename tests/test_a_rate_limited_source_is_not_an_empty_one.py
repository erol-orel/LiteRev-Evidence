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
"""
import inspect
import re

import pytest

pytest.importorskip("fastapi")

from api import pipeline as P  # noqa: E402
from api.search import SOURCE_OUTCOMES, SOURCE_OUTCOMES_COUNTED  # noqa: E402

SRC = inspect.getsource(P._run_user_scenario_populate)


def test_no_fetcher_waits_out_a_429_without_a_bound():
    """La signature de l'ancienne boucle : un sleep nu suivi d'un continue."""
    offenders = []
    for m in re.finditer(r"status_code == 429:\n(?P<body>(?:[ \t]*[^\n]*\n){1,4})", SRC):
        body = m.group("body")
        if "_wait_out_rate_limit" in body:
            continue
        if "for " in body or "range(" in body:      # une boucle bornée convient aussi
            continue
        offenders.append(" ".join(body.split())[:90])
    assert not offenders, (
        "un 429 réessayé sans limite : la source tient son fil jusqu'au budget et en "
        "sort « vide »\n  " + "\n  ".join(offenders))


def test_the_rate_limit_wait_gives_up_and_raises_so_the_source_reads_error():
    """Épuiser les tentatives doit LEVER : c'est ce qui déclenche `_mark_source_error`.

    Rendre la main en silence aurait laissé l'issue à `empty`, qui est le bug."""
    assert "_RATE_LIMIT_ATTEMPTS = 3" in SRC
    _body = SRC[SRC.index("def _wait_out_rate_limit"):][:1600]
    assert "raise RuntimeError" in _body
    assert "attempt + 1 >= _RATE_LIMIT_ATTEMPTS" in _body
    # Et l'attente ne doit jamais déborder du budget de fédération.
    assert "_fed_deadline[0] - _time.time()" in _body
    assert "Retry-After" in _body, "l'en-tête qui dit combien attendre est ignoré"


def test_every_pagination_loop_marks_the_source_it_cuts():
    """Treize boucles, treize marquages : aucune sortie muette sur le budget."""
    assert "if _time.time() >= _fed_deadline[0]:\n" not in SRC.replace(
        "if _time.time() < _fed_deadline[0]:", ""), (
        "une boucle teste encore le budget à la main, sans marquer la source")
    assert SRC.count("if _budget_exhausted():") == 13, (
        f"13 boucles de pagination attendues, {SRC.count('if _budget_exhausted():')} trouvées : "
        "une source ajoutée sans marquage se lira « vide » quand le budget la coupe")


def test_a_cut_source_also_makes_the_federation_incomplete():
    """Sinon le corpus est autorisé à se vider sur un fetch partiel.

    `_fed_incomplete` ne se levait que si `as_completed` débordait lui-même. Une source
    coupée AU MILIEU de ses pages alors que toutes les futures rentrent dans le délai
    laissait donc `_fetch_ok` vrai, et `allow_empty` avec lui."""
    _body = SRC[SRC.index("def _budget_exhausted"):][:900]
    assert "_fed_incomplete[0] = True" in _body
    assert "_fetcher_cut.add(_f)" in _body


def test_the_outcome_of_a_cut_source_is_cut_by_budget_and_is_not_counted_as_searched():
    _m = re.search(r"for _fn in source_funcs:.{0,1600}?_queried\.add\(_n\)", SRC, re.S)
    assert _m, "la table des issues a bougé"
    table = _m.group(0)
    assert "_n in _fetcher_cut or _n not in _returned" in table, (
        "une source arrêtée en cours de pagination retombe sur ok/empty")
    _i_cut = table.index('"cut_by_budget"')
    _i_empty = table.index('"empty"')
    assert _i_cut < _i_empty, "l'issue `empty` capture encore les sources coupées"
    assert "cut_by_budget" in SOURCE_OUTCOMES
    assert "cut_by_budget" not in SOURCE_OUTCOMES_COUNTED, (
        "une source coupée serait comptée parmi les sources interrogées")
    assert "SOURCE_OUTCOMES_COUNTED" in table, (
        "la liste des issues comptées est réécrite à la main ici : deux vérités")


def test_the_total_found_is_what_the_sources_returned_not_what_was_inserted():
    """`total_found = ingested` annonçait moins d'enregistrements que d'identifiés.

    `ingested` ne compte que les LIGNES NOUVELLES : dès qu'une source redonne un article
    qu'une autre a déjà donné, le recoupement, qui est justement ce que le PRISMA appelle
    un doublon, disparaissait du total annoncé."""
    assert "total_found = ingested" not in SRC
    assert "total_found = sum(_ident_records.values())" in SRC
