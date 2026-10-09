"""Une requête PubMed longue part en POST, pas dans l'URL d'un GET.

Le tableau d'identification du scénario HPAI de production porte `pubmed: 0`, pendant que
327 de ses 640 articles screenés portent `pubmed` comme source : ces notices venaient de
la bibliothèque locale, pas du fetch PubMed de ce run. PubMed n'avait rien rapporté.

Sa requête fait 3 075 caractères, et `_ncbi_get` l'envoyait en GET : une URL de 4 247
caractères une fois encodée (les guillemets et les crochets deviennent %22, %5B, %5D).
Au-delà de ~2 000 caractères, une URL de GET est refusée ou tronquée par les proxys, et
NCBI documente le POST pour les requêtes longues. Le dépôt le savait déjà pour `efetch`,
qui poste sa longue liste d'identifiants ; c'était `esearch`, celui qui porte la REQUÊTE,
qui partait dans l'URL.

Le run étant antérieur au registre des issues par source, on ne peut pas prouver après
coup que c'est la cause de ce zéro : ces tests épinglent la forme de la requête, qui est
vérifiable, et le nouveau registre nommera l'issue au prochain run.
"""
import json
import urllib.parse

import pytest

pytest.importorskip("fastapi")

import api.sources as S  # noqa: E402


class _Resp:
    status_code = 200

    def json(self):
        return {"esearchresult": {"count": "306", "idlist": ["39000001"]}}

    def raise_for_status(self):
        return None


class _Spy:
    """Note la méthode employée, et où les paramètres ont voyagé."""

    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None, **kw):
        self.calls.append(("GET", url, params, None))
        return _Resp()

    def post(self, url, data=None, timeout=None, **kw):
        self.calls.append(("POST", url, None, data))
        return _Resp()


@pytest.fixture
def spy(monkeypatch):
    s = _Spy()
    monkeypatch.setitem(__import__("sys").modules, "requests", s)
    # `_ncbi_get` fait `import requests as _req` dans son corps : remplacer l'entrée du
    # cache des modules suffit, et n'affecte rien d'autre que cet appel.
    monkeypatch.setattr(S, "_NCBI_LAST", [0.0], raising=False)
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    return s


SHORT = '"H5N1"[tiab] AND "domestic cat*"[tiab]'
#: La forme de la requête HPAI : longue, pleine de guillemets et de crochets, donc dont
#: l'encodage pour une URL fait gonfler la taille de moitié.
LONG = " OR ".join(f'"Occupational Exposure Term Number {i:03d}"[tiab]' for i in range(60))


def test_a_short_query_still_goes_in_the_url(spy):
    S._ncbi_get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                {"db": "pubmed", "term": SHORT, "retmode": "json"})
    assert [c[0] for c in spy.calls] == ["GET"], (
        "une requête courte n'a aucune raison de changer de méthode")
    assert spy.calls[0][2]["term"] == SHORT


def test_a_long_query_goes_in_the_body(spy):
    S._ncbi_get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                {"db": "pubmed", "term": LONG, "retmode": "json"})
    assert [c[0] for c in spy.calls] == ["POST"], (
        f"une requête de {len(LONG)} caractères part encore dans l'URL d'un GET ; "
        f"encodée, elle y occupe {len(urllib.parse.quote(LONG))} caractères")
    method, url, params, data = spy.calls[0]
    assert params is None and data is not None, "les paramètres doivent être dans le corps"
    assert data["term"] == LONG, "la requête doit arriver ENTIÈRE, non tronquée"
    assert data["db"] == "pubmed" and data["retmode"] == "json"


def test_the_switch_sits_below_the_url_limit_that_breaks_things():
    """Le seuil doit laisser de la marge sous les 2 048 caractères d'URL."""
    assert S._NCBI_POST_ABOVE <= 2000
    # Un GET au seuil doit rester sous la limite, en-tête d'URL et clé d'API compris.
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
    assert len(base) + S._NCBI_POST_ABOVE + 60 < 2048, (
        "au seuil, l'URL d'un GET dépasse encore la limite pratique de 2 048")


def test_the_real_hpai_query_would_have_gone_in_the_body():
    """La requête du scénario de production, mesurée : 4 247 caractères d'URL."""
    pq = ('( ( ("Environmental Exposure"[mh] OR "Environmental Exposure*"[tiab]) '
          'OR ("Occupational Exposure"[mh] OR "Occupational Diseases"[mh]) ) '
          'AND ( "Influenza in Birds"[mh] OR "H5N1"[tiab] ) ) ') * 8
    params = {"db": "pubmed", "term": pq, "retmax": 2000, "retmode": "json",
              "email": "api@literev.app"}
    encoded = len(urllib.parse.urlencode(params))
    assert encoded > S._NCBI_POST_ABOVE, (
        f"la requête de ce test ({encoded} caractères encodés) ne déclenche plus le POST ; "
        "elle doit rester représentative de celle de production (4 247)")


def test_the_api_key_is_counted_in_the_length_decision(monkeypatch):
    """La clé ajoute 45 caractères à l'URL : elle doit entrer dans le calcul.

    Sinon une requête juste sous le seuil le franchit en production, où la clé existe, et
    pas en recette, où elle n'existe pas : le défaut ne se reproduit alors jamais."""
    import inspect
    src = inspect.getsource(S._ncbi_get)
    _i = src.index("_long =")
    decision = src[_i:_i + 200]
    _i_key = src.index('params = {**params, "api_key": key}')
    assert _i_key < _i, (
        "la longueur est calculée AVANT l'ajout de la clé d'API : la décision ne porte "
        "pas sur l'URL réellement émise")


def test_a_none_parameter_does_not_break_the_measurement(spy):
    """`urlencode` d'un None donne la chaîne « None » : on les écarte."""
    S._ncbi_get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                {"db": "pubmed", "term": SHORT, "sort": None})
    assert spy.calls[0][0] == "GET"


def test_the_retry_still_works_through_the_post_path(monkeypatch):
    """Le retry sur 429 ne doit pas être réservé au GET."""
    class _Flaky(_Spy):
        def __init__(self):
            super().__init__()
            self.n = 0

        def post(self, url, data=None, timeout=None, **kw):
            self.n += 1
            self.calls.append(("POST", url, None, data))
            if self.n == 1:
                class _TooMany:
                    status_code = 429
                return _TooMany()
            return _Resp()

    f = _Flaky()
    monkeypatch.setitem(__import__("sys").modules, "requests", f)
    monkeypatch.setattr(S, "_NCBI_LAST", [0.0], raising=False)
    monkeypatch.setattr(S, "_NCBI_MIN_INTERVAL", 0.0, raising=False)
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    r = S._ncbi_get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                    {"db": "pubmed", "term": LONG})
    assert f.n == 2 and r.status_code == 200
    assert all(c[0] == "POST" for c in f.calls), "le retry est reparti en GET"


def test_esearch_and_efetch_now_agree_on_how_a_long_payload_travels():
    """`efetch` postait déjà ; l'asymétrie était le défaut."""
    import inspect
    from api import pipeline as P
    src = inspect.getsource(P._run_user_scenario_populate)
    assert "_requests.post(f\"{ENTREZ_BASE}/efetch.fcgi\"" in src, (
        "efetch ne poste plus sa liste d'identifiants")
    # Et l'esearch du populate passe par le helper, donc par la même décision.
    _i = src.index("esearch.fcgi")
    assert "_ncbi_get(" in src[max(0, _i - 200):_i], (
        "l'esearch du populate n'emprunte plus le helper qui choisit la méthode")
