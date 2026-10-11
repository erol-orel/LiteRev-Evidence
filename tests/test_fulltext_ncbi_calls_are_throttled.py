"""Les appels NCBI de l'étape « texte intégral » passent par le helper throttlé.

Ils partaient par un `requests.get` à eux : sans l'espacement partagé (3 requêtes par
seconde sans clé, 10 avec), sans la clé d'API, sans le passage en POST, pendant que cinq
fils les lançaient en parallèle. Au-delà de la limite de NCBI : des 429, et des textes
intégraux PMC perdus. Et pdftotext (30 s) comme le téléchargement du PDF coupaient les
gros documents : 120 s et 60 s.
"""
import inspect

import pytest

pytest.importorskip("fastapi")

from api import pipeline as P  # noqa: E402


class _Resp:
    def __init__(self, payload=None, text_body="<pmc-articleset/>"):
        self.status_code = 200
        self._payload = payload or {}
        self.text = text_body

    def json(self):
        return self._payload


@pytest.fixture
def ncbi(monkeypatch):
    calls: list[tuple[str, dict, dict]] = []

    def _throttled(url, params, timeout=12, use_key=True):
        calls.append((url, dict(params), {"timeout": timeout, "use_key": use_key}))
        if "esummary" in url:
            return _Resp({"result": {"31234567": {"articleids": [{"idtype": "pmcid", "value": "PMC7654321"}]}}})
        if "idconv" in url:
            return _Resp({"records": [{"pmcid": "PMC1111111"}]})
        return _Resp()

    def _forbidden(*a, **k):
        raise AssertionError("un appel NCBI est parti hors du helper throttlé")

    import requests
    monkeypatch.setattr(P, "_ncbi_get", _throttled)
    monkeypatch.setattr(requests, "get", _forbidden)
    monkeypatch.setattr(requests, "post", _forbidden)
    return calls


def test_the_pmcid_is_resolved_through_the_throttled_helper(ncbi):
    assert P._ft_resolve_pmcid("31234567", None, None, "a@b.c") == "PMC7654321"
    assert ncbi[0][0].endswith("/esummary.fcgi") and ncbi[0][1]["id"] == "31234567"
    assert ncbi[0][2]["use_key"] is True


def test_the_doi_converter_shares_the_spacing_but_not_the_eutils_key(ncbi):
    assert P._ft_resolve_pmcid("", None, "10.1/x", "a@b.c") == "PMC1111111"
    url, params, opts = ncbi[0]
    assert "idconv" in url and params["ids"] == "10.1/x" and opts["use_key"] is False


def test_the_pmc_full_text_is_fetched_through_the_throttled_helper(ncbi):
    r = P._ft_fetch_pmc_xml("PMC7654321")
    assert r is not None and r.text.startswith("<")
    url, params, opts = ncbi[0]
    assert url.endswith("/efetch.fcgi") and params == {"db": "pmc", "id": "7654321", "rettype": "full",
                                                       "retmode": "xml"}
    assert opts["timeout"] == 30


def test_a_failed_ncbi_call_is_none_not_an_exception(monkeypatch):
    def _down(*a, **k):
        raise RuntimeError("NCBI down")
    monkeypatch.setattr(P, "_ncbi_get", _down)
    assert P._ft_fetch_pmc_xml("PMC1") is None
    assert P._ft_resolve_pmcid("31234567", None, None, "a@b.c") is None


def test_the_fulltext_step_has_no_ncbi_call_of_its_own():
    src = inspect.getsource(P._run_user_scenario_full_pipeline)
    assert "ncbi.nlm.nih.gov" not in src, "un appel NCBI contourne encore le helper throttlé"
    assert "_ft_resolve_pmcid(" in src and "_ft_fetch_pmc_xml(" in src
    # Les gros PDF : 60 s pour les télécharger, 120 s pour pdftotext.
    assert "_requests.get(pdf_url, timeout=60" in src
    pdftotext = src[src.index('["pdftotext"'):]
    assert "timeout=120" in pdftotext[:200]


def test_the_throttle_spaces_the_start_of_requests_without_holding_the_lock(monkeypatch):
    """Le verrou espace les DÉPARTS : un texte intégral lent ne bloque plus l'esearch d'une
    recherche derrière lui."""
    from api import sources as S
    src = inspect.getsource(S._ncbi_get)
    body = src[src.index("for attempt in range(3):"):]
    lock_block = body[body.index("with _NCBI_LOCK:"):body.index("try:")]
    assert "_req.get" not in lock_block and "_req.post" not in lock_block
