"""OpenAlex et DOAJ répondaient 400 au booléen portable recopié tel quel.

Mesuré sur le scénario HPAI_last de production (requête recommandée, 1 012 caractères
portables, sous la limite d'URL) : OpenAlex et DOAJ ont toutes deux répondu « 400 Bad
Request » au booléen, pendant que CORE (1 990 notices) et ClinicalTrials.gov (49)
acceptaient exactement la même chaîne, et que la requête de contrôle, sans troncature,
passait chez les quatre. Le booléen portable n'était que le texte de l'utilisateur, tags
retirés : « "occupational*" », « "farm worker*" », deux espaces là où un tag avait été, la
ponctuation telle quelle.

Les quatre API qui reçoivent le booléen dans une URL reçoivent désormais sa forme PROPRE,
rendue depuis l'arbre : AND/OR/NOT en capitales, phrases entre guillemets, un seul
espace, pas d'étoile (ces moteurs racinisent), exclusions comprises. Et la sonde
/sources/health a un mode `raw` qui envoie une requête telle quelle à OpenAlex et à DOAJ,
pour bissecter depuis la production sans créer de scénario.
"""
import ast
import inspect
import textwrap

import pytest

pytest.importorskip("fastapi")

from api.search import _clean_boolean, _plain_keywords, _strip_field_tags  # noqa: E402

V4_LIKE = (
    '( "Occupational Exposure"[mh] OR "occupational*"[tiab] OR "farm worker*"[tiab] '
    'OR "Health Knowledge, Attitudes, Practice"[mh] ) AND ( "Influenza in Birds"[mh] '
    'OR "H5N1"[tiab] OR "2.3.4.4b"[tiab] )'
)


def test_the_clean_form_keeps_the_structure_and_drops_the_stars():
    out = _clean_boolean(_strip_field_tags(V4_LIKE))
    assert out, "rien n'en sort : les quatre API retomberaient sur la requête réduite"
    assert " AND " in out and " OR " in out, out
    assert "*" not in out and "[" not in out and "  " not in out, out
    assert '"occupational exposure"' in out and "occupational" in out and '"farm worker"' in out, out
    assert "h5n1" in out and "2344b" in out, out
    assert out.count("(") == out.count(")") and out.count('"') % 2 == 0, out


def test_the_clean_form_keeps_the_exclusions():
    """Ce n'est pas la requête réduite : le NOT reste, ces quatre API le parlent."""
    q = '("H5N1"[tiab] OR "avian influenza"[tiab]) NOT ("H1N1"[tiab] OR "swine"[tiab])'
    out = _clean_boolean(_strip_field_tags(q))
    assert out == '(h5n1 OR "avian influenza") AND NOT (h1n1 OR swine)', out
    assert _clean_boolean('"H5N1"[tiab] NOT "H1N1"[tiab]') == "h5n1 AND NOT h1n1"


def test_a_publication_type_exclusion_does_not_survive_into_the_clean_form():
    q = '"H5N1"[tiab] NOT ("news"[pt] OR "letter"[pt])'
    assert _clean_boolean(_strip_field_tags(q)) == "h5n1"


@pytest.mark.parametrize("query", ["", "   ", "AND OR", "()", '""'])
def test_nothing_gives_nothing(query):
    assert _clean_boolean(query) == ""


def test_the_populate_sends_the_clean_form_and_measures_the_gate_on_it():
    from api import pipeline as P
    tree = ast.parse(textwrap.dedent(inspect.getsource(P._run_user_scenario_populate)))
    assigned: dict[str, list[str]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for tgt in n.targets:
                if isinstance(tgt, ast.Name):
                    assigned.setdefault(tgt.id, []).append(ast.unparse(n.value))
    assert any("_clean_boolean(_portable_bool)" in v for v in assigned.get("_clean_bool", [])), assigned.get("_clean_bool")
    assert any("len(_clean_bool) <= 1200" in v for v in assigned.get("_send_bool", [])), assigned.get("_send_bool")
    assert any(v.startswith("_clean_bool if _send_bool") for v in assigned.get("_bool_query", [])), assigned.get("_bool_query")
    # Le booléen portable est replié sur une espace : un retour à la ligne dans la requête
    # collée partait tel quel dans l'URL.
    assert any(".split()" in v and "join" in v for v in assigned.get("_portable_bool", [])), assigned.get("_portable_bool")


class _Resp:
    status_code = 200

    def json(self):
        return {"meta": {"count": 1}, "total": 1, "hitCount": 1, "esearchresult": {"count": "1"},
                "message": {"total-results": 1}, "header": {"numFound": 1}, "messages": [{"total": 1}],
                "totalCount": 1}

    text = ""


def _capture(monkeypatch):
    import requests
    calls: list[tuple[str, dict]] = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append((url, dict(params or {})))
        return _Resp()

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


def test_the_probe_sends_keywords_to_openalex_by_default_and_the_raw_query_on_demand(monkeypatch):
    from api import sources as S
    calls = _capture(monkeypatch)
    q = '("H5N1" OR "avian influenza") AND "farmers"'
    out = S.sources_health(q, timeout=1)
    openalex = [p for u, p in calls if "openalex" in u]
    assert openalex and openalex[0]["search"] == _plain_keywords(q)
    assert not any("doaj.org" in u for u, _ in calls)
    assert "DOAJ" in out["not_probed"]

    calls.clear()
    out = S.sources_health(q, timeout=1, raw=True)
    openalex = [p for u, p in calls if "openalex" in u]
    assert openalex and openalex[0]["search"] == q, "en mode raw, OpenAlex doit recevoir la requête telle quelle"
    doaj = [u for u, _ in calls if "doaj.org/api/search/articles/" in u]
    assert doaj and "%28%22H5N1%22" in doaj[0], doaj
    assert "DOAJ" in out["probed"] and "DOAJ" not in out["not_probed"]
