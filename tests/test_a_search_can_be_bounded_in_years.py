"""A search bounded in years keeps only those years, whichever source found the record.

The search screen had a year range that nothing read: the search sent only the project
context and the preprint option. It now sends `year_min` / `year_max` with the scenario's
filters when the user narrows the range. The local match already applied them
(`_build_where`), but the records of the boolean-native sources (PubMed, Europe PMC)
enter the corpus WITHOUT that re-match: a 2021-onwards search kept a 2020 Europe PMC
article. The same condition is now applied to the assembled corpus.

The sources are the in-memory fakes of the preprint test: Semantic Scholar (keyword source,
re-matched locally) returns a 2021 article and a 2021 preprint; Europe PMC (boolean-native)
a 2020 article and, from its preprint facet, a 2022 preprint.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from test_a_search_can_leave_preprints_out import federation  # noqa: E402,F401


def test_the_lower_bound_reaches_the_native_sources_too(federation):  # noqa: F811
    run, _fake, corpus_titles = federation
    run({"year_min": 2021})
    titles = corpus_titles()
    assert not any("epmc article" in t for t in titles), titles        # 2020, native: out
    assert any("s2 article" in t for t in titles), titles               # 2021, re-matched: in
    assert any("epmc preprint" in t for t in titles), titles            # 2022, native: in


def test_the_upper_bound_too(federation):  # noqa: F811
    run, _fake, corpus_titles = federation
    run({"year_max": 2020})
    titles = corpus_titles()
    assert titles and all("epmc article" in t for t in titles), titles


def test_a_search_without_bounds_is_unchanged(federation):  # noqa: F811
    run, _fake, corpus_titles = federation
    run({})
    titles = corpus_titles()
    for what in ("epmc article", "s2 article", "epmc preprint", "s2 preprint"):
        assert any(what in t for t in titles), (what, titles)
