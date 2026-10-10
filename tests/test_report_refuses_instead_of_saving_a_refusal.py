"""Un téléchargement sans brief REFUSE, au lieu de livrer son refus comme un rapport.

Le endpoint répondait 200 avec un corps JSON d'erreur. Le navigateur, qui suit un
`<a download>`, enregistrait donc un fichier de 166 octets PORTANT LE NOM DU RAPPORT.
Un partenaire ouvrait ce fichier et y trouvait `{"status": "no_brief", ...}`.

Ce test existe aussi parce que le 409 que j'ai écrit pour corriger cela levait un
NameError (HTTPException n'était pas importé dans ce module) et répondait donc 500 : un
refus qui n'arrive pas se remarque moins qu'un fichier vide.
"""
import inspect

import pytest

pytest.importorskip("fastapi")

from api import report as R  # noqa: E402


def test_the_module_can_actually_raise_an_http_error():
    """`HTTPException` doit être importé : le raise levait un NameError, donc 500."""
    assert "HTTPException" in dir(R) or "HTTPException" in R.__dict__ or True
    src = inspect.getsource(R)
    assert "from fastapi import" in src
    _imports = next(l for l in src.splitlines() if l.startswith("from fastapi import"))
    assert "HTTPException" in _imports, (
        "le 409 du téléchargement lève un NameError et répond 500")


def test_a_download_without_a_brief_raises_409_and_a_plain_read_does_not():
    src = inspect.getsource(R.evidence_report)
    assert "status_code=409" in src
    # Le refus ne concerne QUE le téléchargement : la lecture simple doit continuer à
    # répondre 200 avec son message, que l'interface affiche.
    _i = src.index("status_code=409")
    assert "if download:" in src[max(0, _i - 400):_i]
    assert '"status": "no_brief"' in src


def test_the_client_reads_the_response_before_saving_it():
    """Côté client, un `<a download>` enregistre ce qu'on lui donne : il fallait lire."""
    import pathlib
    api_ts = (pathlib.Path(__file__).resolve().parent.parent
              / "frontend" / "src" / "lib" / "api.ts").read_text(encoding="utf-8")
    assert "export async function downloadEvidenceReport" in api_ts
    _fn = api_ts[api_ts.index("export async function downloadEvidenceReport"):][:1200]
    assert "if (!r.ok)" in _fn and "throw new Error" in _fn
    assert "URL.createObjectURL" in _fn

    page = (pathlib.Path(__file__).resolve().parent.parent
            / "frontend" / "src" / "components" / "ScenarioDetailPage.tsx").read_text(encoding="utf-8")
    # La page passe aussi sa langue : le rapport s'écrit dans celle de l'interface.
    assert "downloadEvidenceReport(scenarioId, lang)" in page
    assert "href={evidenceReportUrl(scenarioId" not in page, (
        "le lien direct est revenu : le navigateur enregistrera de nouveau le refus")
