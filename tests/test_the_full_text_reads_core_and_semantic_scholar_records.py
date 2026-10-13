"""The full-text download finds the records that carry neither a DOI nor a PMID.

On 12 October 2026, 162 of the 536 relevant articles of HPAI_last still without a text named
no DOI and no PMID: records from CORE, Semantic Scholar and OpenAlex. Every source of the
download looked a paper up by one of those, or by an OpenAlex id. A CORE record carries its
download link, the open-access PDF itself (41 of them), and a Semantic Scholar record its own
paper id, which the Graph API reads as it is.

The PDF of CORE's link did not read from the server (1 of 41 on 13 October), so CORE's API comes
first: it returns the text CORE already extracted (`fullText`), by CORE id for its own records and
by DOI for the others, with the key the search uses."""
import inspect

import pytest

pytest.importorskip("fastapi")

from api import pipeline as P  # noqa: E402
from conftest import patch_app  # noqa: E402

CORE_PDF = "https://core.ac.uk/download/144159216.pdf"
TEXT = "Highly pathogenic avian influenza in poultry workers. " * 40


@pytest.fixture
def calls(monkeypatch):
    seen: dict[str, list] = {"get": [], "pdf": []}

    def fake_get(url, params=None, timeout=20, headers=None):
        seen["get"].append((str(url), dict(params or {}), dict(headers or {})))
        return None

    def fake_pdf(url):
        seen["pdf"].append(str(url))
        return TEXT if url == CORE_PDF else None

    monkeypatch.delenv("CORE_API_KEY", raising=False)
    patch_app(monkeypatch, "_ft_get", fake_get)
    patch_app(monkeypatch, "_ft_resolve_pmcid", lambda *a, **k: None)
    patch_app(monkeypatch, "_extract_pdf_text_ft", fake_pdf)
    return seen


class _Answer:
    def __init__(self, body, status=200):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


def _core_answers(monkeypatch, calls, answers):
    """CORE's API answering `answers` (the end of a URL -> its JSON body); everything else None."""
    monkeypatch.setenv("CORE_API_KEY", "core-test-key-0123456789")

    def fake_get(url, params=None, timeout=20, headers=None):
        calls["get"].append((str(url), dict(params or {}), dict(headers or {})))
        return next((_Answer(body) for end, body in answers.items() if str(url).endswith(end)), None)
    patch_app(monkeypatch, "_ft_get", fake_get)


def test_a_core_record_is_read_through_the_core_api_with_the_key(calls, monkeypatch):
    _core_answers(monkeypatch, calls, {"/works/12345": {"id": 12345, "fullText": TEXT}})
    text, source, _ = P._fulltext_of({"external_id": "core:12345", "doi": None, "pmid": None, "url": CORE_PDF})
    assert (text, source) == (TEXT.strip(), "core") and calls["pdf"] == []          # no PDF to read
    core = [c for c in calls["get"] if "api.core.ac.uk" in c[0]]
    assert core[0][2]["Authorization"] == "Bearer core-test-key-0123456789"


def test_a_paper_with_a_doi_is_found_in_core_by_that_doi_and_no_other(calls, monkeypatch):
    results = [{"doi": "10.9999/another.paper", "fullText": "Another paper. " * 100},
               {"doi": "https://doi.org/10.1000/XYZ", "fullText": TEXT}]
    _core_answers(monkeypatch, calls, {"/search/works": {"results": results}})
    text, source, _ = P._fulltext_of({"external_id": "x:1", "doi": "10.1000/xyz", "pmid": None})
    assert (text, source) == (TEXT.strip(), "core")
    assert [c[1] for c in calls["get"] if c[0].endswith("/search/works")][0]["q"] == 'doi:"10.1000/xyz"'
    results.pop()                                       # only another paper's text is left
    text, source, fails = P._fulltext_of({"external_id": "x:1", "doi": "10.1000/xyz", "pmid": None})
    assert source is None and "core_api:no_text" in fails


def test_without_text_from_the_api_the_core_pdf_is_still_read(calls, monkeypatch):
    _core_answers(monkeypatch, calls, {"/works/12345": {"id": 12345, "fullText": None}})
    text, source, fails = P._fulltext_of({"external_id": "core:12345", "doi": None, "pmid": None, "url": CORE_PDF})
    assert (text, source) == (TEXT, "core") and calls["pdf"] == [CORE_PDF] and "core_api:no_text" in fails


def test_without_the_core_key_the_api_is_not_called(calls):
    P._fulltext_of({"external_id": "core:12345", "doi": "10.1000/xyz", "pmid": None, "url": CORE_PDF})
    assert not [c for c in calls["get"] if "api.core.ac.uk" in c[0]]


def test_a_core_record_is_read_from_its_own_download_link(calls):
    text, source, fails = P._fulltext_of({"external_id": "core:12345", "doi": None, "pmid": None,
                                          "url": CORE_PDF})
    assert source == "core" and text == TEXT
    assert calls["pdf"] == [CORE_PDF]


def test_a_core_copy_that_reads_empty_is_a_named_failure(calls):
    text, source, fails = P._fulltext_of({"external_id": "core:9", "doi": None, "pmid": None,
                                          "url": "https://core.ac.uk/download/1.pdf"})
    assert text is None and source is None and "core:pdf_empty" in fails


def test_only_a_core_download_link_is_fetched_as_the_paper(calls):
    """Any other link may be a landing page or a paywall: the other sources decide."""
    for url in ("https://example.org/paper.pdf", "https://doi.org/10.1000/xyz", None, ""):
        calls["pdf"].clear()
        text, source, _ = P._fulltext_of({"external_id": "x:1", "doi": None, "pmid": None, "url": url})
        assert source is None and calls["pdf"] == []


def test_a_semantic_scholar_record_is_looked_up_by_its_own_paper_id(calls, monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "s2-test-key-0123456789")
    P._fulltext_of({"external_id": "s2:0b8a1d0257d1784a5f8b3fe7ecb21d3e21e4fc5c", "doi": None, "pmid": None})
    s2 = [c for c in calls["get"] if "semanticscholar" in c[0]]
    assert s2 and s2[0][0].endswith("/paper/0b8a1d0257d1784a5f8b3fe7ecb21d3e21e4fc5c")
    assert s2[0][2].get("x-api-key") == "s2-test-key-0123456789"
    # A DOI still comes first when the record has one.
    calls["get"].clear()
    P._fulltext_of({"external_id": "s2:abc", "doi": "10.1000/xyz", "pmid": None})
    s2 = [c for c in calls["get"] if "semanticscholar" in c[0]]
    assert s2 and s2[0][0].endswith("/paper/DOI:10.1000/xyz")


def test_the_rows_to_download_carry_the_record_link():
    assert "ld.url" in inspect.getsource(P.rows_without_fulltext)
