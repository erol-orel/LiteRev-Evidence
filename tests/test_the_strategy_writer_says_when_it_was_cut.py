"""La stratégie booléenne coupée à son plafond de tokens se dit coupée.

Le rédacteur de stratégie avait 500 tokens : une stratégie riche en synonymes s'arrêtait
au milieu d'une chaîne JSON, `json.loads` levait, et la recherche partait en mode dégradé
avec la phrase brute comme booléen (un ET de chacun de ses mots), sans que rien ne dise
pourquoi. Le plafond passe à 1 500, une coupure qui reste est détectée par
`llm_usage.json_content` (finish_reason « length »), et la réponse de l'API le dit.
"""
import types

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402

QUERY = "occupational exposure of poultry workers to avian influenza, a question cut short 7d1e"


class _Completions:
    def __init__(self, seen: dict, finish_reason: str, content: str):
        self.seen, self.finish_reason, self.content = seen, finish_reason, content

    def create(self, **kw):
        self.seen.update(kw)
        msg = types.SimpleNamespace(content=self.content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason=self.finish_reason)])


def _fake_client(seen: dict, finish_reason: str, content: str):
    class _Client:
        def __init__(self, *a, **k):
            self.chat = types.SimpleNamespace(completions=_Completions(seen, finish_reason, content))
    return _Client


@pytest.fixture
def cut_writer(monkeypatch):
    import llm_usage
    from api import search as S
    seen: dict = {}
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(llm_usage, "MeteredOpenAI",
                        _fake_client(seen, "length", '{"general": "(\\"poultry worker\\" OR \\"farm'))
    monkeypatch.setattr(S, "_STRATEGY_CACHE", {})
    return seen


def test_a_cut_strategy_is_degraded_with_its_reason(cut_writer):
    from api import search as S
    out = S._generate_search_strategy(QUERY)
    assert out["degraded"] is True and out["truncated"] is True
    assert out["degraded_reason"] == "truncated" and "coupée" in out["reason"]
    assert out["general"] == QUERY, "la recherche part sur la requête telle que saisie"
    assert S._strategy_key(QUERY) not in S._STRATEGY_CACHE, "une coupure ne se met jamais en cache"
    # Le plafond relevé (1 500, puis 6 000 pour qu'une correction proposée puisse réécrire
    # une stratégie de 8 000 caractères), et la consigne sur le tiret long gardée.
    assert cut_writer["max_tokens"] == S.SEARCH_STRATEGY_MAX_TOKENS == 6000
    assert "em dash" in cut_writer["messages"][0]["content"]


def test_the_api_response_says_the_strategy_was_cut(cut_writer):
    from fastapi.testclient import TestClient
    body = TestClient(main.app).post("/search-strategy", json={"query": QUERY + " via the API"}).json()
    assert body["truncated"] is True and body["degraded"] is True and body["degraded_reason"] == "truncated"


def test_a_complete_answer_is_not_reported_as_cut(monkeypatch):
    import llm_usage
    from api import search as S
    seen: dict = {}
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", _fake_client(
        seen, "stop", '{"general": "(\\"poultry worker\\" OR farmer) AND (h5n1 OR \\"avian influenza\\")",'
                      ' "pubmed": "h5n1[tiab]", "explanation": "e", "synonyms": []}'))
    monkeypatch.setattr(S, "_STRATEGY_CACHE", {})
    # Pas d'écriture dans le cache persistant pendant le test.
    monkeypatch.setattr(S, "engine", _NoDb())
    out = S._generate_search_strategy(QUERY + " complete")
    assert not out.get("truncated") and not out.get("degraded")
    assert out["general"].startswith('("poultry worker" OR farmer)')


class _NoDb:
    """Un moteur dont chaque connexion échoue : le cache persistant est sauté (lecture et
    écriture sont best-effort dans _generate_search_strategy)."""

    def connect(self):
        raise RuntimeError("no database in this test")

    def begin(self):
        raise RuntimeError("no database in this test")
