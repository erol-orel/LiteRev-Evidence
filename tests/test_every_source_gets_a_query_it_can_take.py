"""A source's query is measured before it is sent, for every source.

Nine APIs, nine sets of limits, and the pipeline tailors the query to each: field tags
stripped, the boolean re-rendered, its length measured on the ENCODED form, then sent
whole, reduced, or split into parts whose union is the query. Semantic Scholar was the
exception: its bulk query rode in a URL like OpenAlex's and nothing weighed it, so the
longest T2.4 string (COVID exposure, 4,519 characters) came back 400 Bad Request and the
source contributed nothing to a 27,405 article corpus.

These are pure tests on the renderers and the limits, no database and no network.
"""
import pytest

pytest.importorskip("fastapi")

from api import search as S  # noqa: E402

#: The shape that failed: many quoted phrases joined by OR, then ANDed with another block.
#: Every term is distinct, so nothing is folded away, and the rendering is long.
LONG = (
    "(" + " OR ".join(f'"environmental exposure variant {i}"' for i in range(60)) + ")"
    " AND (" + " OR ".join(f'"severe acute respiratory syndrome coronavirus term {i}"'
                           for i in range(60)) + ")"
)


def _s2(portable: str):
    return S._boolean_to_s2(S._parse_boolean_ast(S._tokenize_boolean(portable)))


def test_the_long_string_really_is_too_long_for_a_url():
    """The premise of the fix, stated as a measurement rather than assumed."""
    whole = _s2(LONG)
    assert whole, "the renderer handles this shape"
    assert S._url_encoded_len(whole) > S.S2_MAX_QUERY_ENCODED


def test_the_reduced_form_fits_and_keeps_the_structure():
    """The fallback is the reduced boolean, not a bag of words: the same tree with each OR
    block cut to its first terms. A bag of words was read as OR by some engines and as AND
    by others, which is how five sources once returned thousands of unrelated records."""
    short = S._shorten_boolean(LONG, S.S2_MAX_QUERY_ENCODED, render=S._boolean_to_s2,
                               measure=S._url_encoded_len)
    assert short, "something fits under the limit"
    assert S._url_encoded_len(short) <= S.S2_MAX_QUERY_ENCODED
    assert "|" in short and "(" in short, "still a boolean, not a word list"


def test_a_short_query_is_sent_whole():
    """The limit must not touch an ordinary query: most strategies ride in the URL intact."""
    whole = _s2('("avian influenza" OR "H5N1") AND ("poultry worker" OR "culler")')
    assert whole and S._url_encoded_len(whole) <= S.S2_MAX_QUERY_ENCODED


def test_the_limit_is_measured_on_the_encoded_form():
    """Quotes and parentheses cost three bytes each once encoded, so a plain character
    count said nothing useful: that is why every other source measures the encoded form."""
    q = '("a b c" OR "d e f")'
    assert S._url_encoded_len(q) > len(q)


def test_semantic_scholar_is_now_measured_like_the_others():
    """The limits of the sources that put their query in a URL, side by side: the point of
    the fix is that this list has no hole in it."""
    assert S.S2_MAX_QUERY_ENCODED == S.URL_QUERY_MAX_ENCODED, (
        "one URL, one limit: a different number here would be a second thing to maintain")


def test_europe_pmc_keeps_a_long_query_out_of_the_url():
    """It takes its query in a body above this length. The COVID exposure run answered 503
    on a GET under the old threshold of 3,000, while the same query in a body had worked
    all week, so the threshold sits where only short queries stay in the URL."""
    from api.sources import EPMC_POST_ABOVE_ENCODED
    assert EPMC_POST_ABOVE_ENCODED <= 1200


def test_the_long_string_is_split_rather_than_reduced():
    """A split keeps everything: the parts' union IS the query, so no synonym is dropped
    and the records still answer the whole boolean. Reduction is the fallback for a query
    that will not fit in SPLIT_MAX_PARTS pieces, not the first move."""
    parts = S._split_boolean(LONG, S.S2_MAX_QUERY_ENCODED, render=S._boolean_to_s2,
                             measure=S._url_encoded_len)
    assert parts, "this query does fit in parts"
    assert len(parts) <= S.SPLIT_MAX_PARTS
    assert all(S._url_encoded_len(p) <= S.S2_MAX_QUERY_ENCODED for p in parts)


def test_the_plan_says_what_each_source_will_receive():
    """What a reviewer could only learn by running the search and reading the notes after.
    On the string that failed, the plan names the body, the parts and the reductions."""
    plan = {r["source"]: r for r in S.source_query_plan(LONG)}
    assert set(plan) >= {"pubmed", "europepmc", "openalex", "semantic_scholar", "doaj",
                         "openaire", "crossref"}
    assert plan["pubmed"]["how"] == "body", "a long query goes in the body, no URL limit"
    assert plan["europepmc"]["how"] == "body"
    assert plan["semantic_scholar"]["how"] == "parts" and plan["semantic_scholar"]["parts"] > 1
    assert plan["crossref"]["how"] == "keywords", "this API reads free text only"
    assert plan["doaj"]["limit"] == "DOAJ_MAX_QUERY_CHARS"


def test_an_ordinary_query_is_whole_everywhere_it_can_be():
    """The plan must not cry wolf: a normal strategy reaches every boolean source intact."""
    plan = {r["source"]: r for r in S.source_query_plan(
        '("avian influenza" OR "H5N1") AND ("poultry worker" OR "culler")')}
    for name in ("openalex", "semantic_scholar", "doaj", "clinicaltrials", "arxiv"):
        assert plan[name]["how"] == "whole", f"{name} can take this one as written"


def test_no_boolean_means_keywords_everywhere():
    plan = S.source_query_plan("avian influenza in European poultry farms")
    assert {r["how"] for r in plan} == {"keywords"}
