"""The structured extraction runs by itself, as a step of the pipeline.

It used to run ONLY when somebody pressed "Extract the N remaining articles". So a
reviewer who set a threshold, which is the moment the relevant subset exists, found the
panel asking for that press, run after run, while everything that reads the extraction
(the brief, the geography, the pooled estimates, the assistant's answers) quietly rested
on a partial one.

It is now a step like the others: after the metadata, before the clustering, over ALL the
relevant articles that have no extraction yet, with no cap. The pipeline WAITS for it,
within a budget, so the steps after it read what it has just written; past the budget the
extraction carries on in the background and the step says so rather than claiming to be
done.
"""
from __future__ import annotations

import inspect

import pytest

from api import pipeline as P


def _pipeline_src() -> str:
    return inspect.getsource(P._run_user_scenario_full_pipeline)


def test_the_extraction_is_a_step_between_the_metadata_and_the_clustering():
    src = _pipeline_src()
    order = src[src.index("STEP_ORDER = ["):src.index("]", src.index("STEP_ORDER = ["))]
    assert '"extraction"' in order, "the extraction is not a step of the pipeline"
    assert order.index('"metadata"') < order.index('"extraction"') < order.index('"clustering"')
    # And before everything that READS it, so those steps see what it wrote.
    assert order.index('"extraction"') < order.index('"evidence"')
    assert order.index('"extraction"') < order.index('"variables"')


def test_the_step_extracts_every_relevant_article_with_no_cap_and_waits_for_it():
    src = _pipeline_src()
    step = src[src.index('update_step("extraction", "running")'):src.index('update_step("clustering"')]
    assert "start_extraction(scenario_id, None, 0)" in step, "the step caps the articles"
    assert "extraction_status(scenario_id)" in step, "the step does not follow the run"
    assert "_extraction_wait_budget_s()" in step, "the wait is not bounded"
    # A run still going when the budget runs out is NOT reported as done.
    assert '"running" if _ex.get("running") else "done"' in step
    assert "still_running" in step


def test_the_step_waits_for_a_threshold_like_the_other_enrichments():
    """Without a chosen threshold the relevant subset is the default one, which can be a
    third of a corpus: the full text, the PICO and the metadata all wait, and an
    extraction that costs one model call per article must wait with them."""
    src = _pipeline_src()
    step = src[src.index('update_step("extraction", "running")'):src.index('update_step("clustering"')]
    assert "if _enrich_waits:" in step and '"skipped"' in step
    assert "_WAIT_REASON" in step


def test_the_wait_budget_reads_the_environment_and_keeps_its_default(monkeypatch):
    monkeypatch.delenv("PIPELINE_EXTRACTION_WAIT_S", raising=False)
    assert P._extraction_wait_budget_s() == float(P.EXTRACTION_WAIT_S)
    monkeypatch.setenv("PIPELINE_EXTRACTION_WAIT_S", "120")
    assert P._extraction_wait_budget_s() == 120.0
    monkeypatch.setenv("PIPELINE_EXTRACTION_WAIT_S", "0")
    assert P._extraction_wait_budget_s() == 0.0          # start it, do not wait
    for bad in ("", "soon", "-5"):
        monkeypatch.setenv("PIPELINE_EXTRACTION_WAIT_S", bad)
        assert P._extraction_wait_budget_s() == float(P.EXTRACTION_WAIT_S), bad


@pytest.mark.parametrize("module, name", [("api.scenarios", "_launch_full_pipeline"),
                                          ("api.schema_boot", "startup_event")])
def test_every_seeded_step_list_carries_the_extraction(module, name):
    """Three places seed the step map (the launcher, the restart after a deploy, the
    documented list). A step missing from one of them shows as a blank row in the
    interface while it runs."""
    import importlib

    mod = importlib.import_module(module)
    src = inspect.getsource(getattr(mod, name))
    assert '"extraction"' in src, f"{module}.{name} does not seed the extraction step"


def test_the_documented_step_list_matches_the_pipeline():
    from api import scenarios as S

    documented = inspect.getsource(S)
    assert ('"extraction", "clustering", "knowledge_graph", "evidence", "variables", "actions"]'
            in documented), "the documented list no longer matches STEP_ORDER"
