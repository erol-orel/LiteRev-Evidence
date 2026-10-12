"""The open-access full text: one download, for the pipeline and the enrichment panel, after
the scores, and on the relevant subset unless asked otherwise (Erol, 12 October 2026).

It lived in a closure of the full pipeline: only the pipeline could download a text, before the
scores, over the whole scenario (17,510 articles for HPAI_last, 1,075 of them relevant), while the
panel's "full text" batch only found links. And it selected by the `has_fulltext` flag, which a
link found sets without any text stored: such an article was never downloaded at all."""
import inspect

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402,F401
from api import pipeline as P  # noqa: E402


def test_an_article_without_a_text_is_one_without_a_full_text_chunk():
    src = inspect.getsource(P.rows_without_fulltext)
    assert "NO_FULLTEXT_CHUNK_SQL" in src and "has_fulltext" not in src
    assert "document_chunk" in P.NO_FULLTEXT_CHUNK_SQL and "fulltext_section" in P.NO_FULLTEXT_CHUNK_SQL
    assert "scenario_scope_sql(scope" in src


def test_the_download_counts_texts_and_reports_progress(monkeypatch):
    monkeypatch.setattr(P, "_fetch_and_store_fulltext", lambda row, emb_client=None: (row["id"] % 2 == 1, None))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = []
    out = P.download_fulltext([{"id": i} for i in range(1, 22)], progress=lambda d, t, f: seen.append((d, t, f)))
    assert out == {"total": 21, "fetched": 11, "failed": 10}
    assert seen[-1] == (21, 21, 10) and all(t == 21 for _d, t, _f in seen)
    assert P.download_fulltext([]) == {"total": 0, "fetched": 0, "failed": 0}


def test_the_pipeline_downloads_after_the_scores_and_waits_for_a_chosen_threshold():
    src = inspect.getsource(P._run_user_scenario_full_pipeline)
    order = src[src.index("STEP_ORDER = ["):src.index("]", src.index("STEP_ORDER = ["))]
    assert order.index('"cross_encoder"') < order.index('"fulltext"') < order.index('"pico"')
    # The step itself sits after the cross-encoder, and reads the shared download.
    assert src.index('update_step("cross_encoder", "running")') < src.index("download_fulltext(")
    assert "rows_without_fulltext(scenario_id, _enrich_scope)" in src
    # Full text, PICO and metadata wait while nobody has chosen the threshold.
    assert 'scenario_threshold_info(scenario_id)["chosen"]' in src
    assert src.count("if openai_key and not _enrich_waits:") == 2
    assert '["waiting_threshold"] = True' in src
