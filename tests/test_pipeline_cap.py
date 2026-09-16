"""The interface's pipeline and the API's populate must cap live sources identically.

Observed on 14 Sep: the same query gave 30,511 documents when populated from the API and
25,140 when the pipeline was started from the interface - the pipeline path carried a
hard-coded 500 per source (PubMed 500, CORE 500, DOAJ 500… in the PRISMA box) while the
populate path applied LIVE_MAX_PER_SOURCE (2,000 by default). Same query, same corpus,
whichever button started it.
"""
import inspect

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402


def _default(fn, name):
    return inspect.signature(fn).parameters[name].default


def test_every_pipeline_entry_point_defaults_to_the_shared_cap():
    assert main.LIVE_MAX_PER_SOURCE >= 500
    for fn in (main._run_user_scenario_full_pipeline, main._launch_full_pipeline,
               main.start_user_scenario_pipeline):
        assert _default(fn, "max_results") == main.LIVE_MAX_PER_SOURCE, fn.__name__


def test_the_populate_clamps_a_larger_request_to_the_same_cap():
    """The interface no longer sends a cap; the API's populate default (100 000) is
    clamped inside the run - so both paths end on LIVE_MAX_PER_SOURCE."""
    assert _default(main.populate_user_scenario, "max_results") >= main.LIVE_MAX_PER_SOURCE
    src = inspect.getsource(main._run_user_scenario_populate)
    assert "max_results = min(max_results, LIVE_MAX_PER_SOURCE)" in src
