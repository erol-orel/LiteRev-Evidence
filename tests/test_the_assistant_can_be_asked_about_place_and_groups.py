"""Two questions the assistant used to decline, and the context it was missing.

A reviewer asked for a European corpus broken down by sex, and then for two thematic
clusters broken down the same way. The assistant declined both, honestly: it answers from
article passages and from two digests, and neither digest held what was asked.

- The coverage counts said how many articles report by sex, the corpus digest said how
  many are European, and nothing crossed the two, so the intersection could only be
  guessed. It is now counted in SQL over the same relevant, extracted set.
- The clusters were never in its context at all: it could not name one or say how large
  it was. They are a cached projection, so naming them costs a cache read.

Pure tests: the blocks are built from a digest and a cache, so neither needs a database.
"""
import pytest

pytest.importorskip("fastapi")

from api import clustering as C  # noqa: E402
from api import extraction as E  # noqa: E402


def _digest(**over):
    d = {"complete": True, "n_relevant": 1000, "n_extracted": 900, "n_fulltext": 500,
         "n_abstract": 400, "n_unread": 100,
         "coverage": {k: 0 for k in E.COVERAGE_KEYS}, "coverage_fulltext": {k: 0 for k in E.COVERAGE_KEYS}}
    d["coverage"]["sex_gender"] = 243
    d["coverage_fulltext"]["sex_gender"] = 120
    d.update(over)
    return d


def test_the_place_crossing_answers_how_many_european_articles_report_by_sex():
    cov_eu = {k: 0 for k in E.COVERAGE_KEYS}
    cov_eu["sex_gender"] = 42
    block = E.extraction_to_prompt(_digest(by_place={
        "n_europe": 298, "n_elsewhere": 600, "n_no_country": 2, "coverage_europe": cov_eu}))
    assert "PAR LIEU" in block
    assert "298" in block and "42" in block, "the intersection must be in the block, not inferable"
    assert "243" in block, "and the whole-corpus count stays, so the two are not confused"


def test_the_place_crossing_never_calls_the_nuts_set_the_eu():
    """The set holds Turkey, Serbia, Albania, North Macedonia and Montenegro beside the EU
    and the EEA. A count presented as "in the EU" would be wrong, and a partner would
    quote it."""
    cov_eu = {k: 0 for k in E.COVERAGE_KEYS}
    cov_eu["sex_gender"] = 42
    block = E.extraction_to_prompt(_digest(by_place={
        "n_europe": 298, "n_elsewhere": 600, "n_no_country": 2, "coverage_europe": cov_eu}))
    assert "pays NUTS" in block, "the count is labelled by the set it is counted over"
    assert "ne doivent jamais etre presentes comme" in block, "and the model is told not to call it the EU"
    from api.geography import NUTS_ISO
    assert {"TR", "RS", "AL", "MK", "ME"} <= set(NUTS_ISO), "the wider set is why the wording matters"


def test_a_digest_without_the_crossing_says_nothing_about_place():
    """An older scenario, aggregated before this existed, must not grow an empty claim."""
    block = E.extraction_to_prompt(_digest())
    assert "PAR LIEU" not in block


def test_the_groups_block_names_each_cluster_with_its_size(monkeypatch):
    monkeypatch.setattr(C, "_load_viz_cache", lambda sid, col, **kw: {"clusters": [
        {"cluster_id": 8, "cluster_name": "Cluster 9", "n_docs": 31,
         "summary": "Human exposure to H5N1 in poultry-keeping communities across Asia.",
         "points": [{"x": 1, "y": 2, "id": 1}]},
        {"cluster_id": -1, "cluster_name": "Unclassified", "n_docs": 344,
         "summary": "Background noise (articles not grouped)."},
    ]})
    block = C.clusters_to_prompt("usr-x")
    assert "Cluster 9" in block and "31 articles" in block
    assert "poultry-keeping communities across Asia" in block
    assert "Unclassified" in block and "n'est pas un theme" in block
    assert '"x"' not in block and "1, 'y'" not in block, "coordinates are not context"


def test_no_clustering_computed_says_nothing(monkeypatch):
    monkeypatch.setattr(C, "_load_viz_cache", lambda sid, col, **kw: None)
    assert C.clusters_to_prompt("usr-x") == ""
    monkeypatch.setattr(C, "_load_viz_cache", lambda sid, col, **kw: {"clusters": []})
    assert C.clusters_to_prompt("usr-x") == ""
