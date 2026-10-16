"""The register of variables of interest: one list of what a review asks of every paper.

Three things are worth pinning, and all three are the kind of thing that breaks quietly.

1. THE COUNTS ARE OVER THE WHOLE RELEVANT SUBSET, never a sample and never the whole
   corpus. A paper a reviewer excluded is out; a paper below the threshold is out; a paper
   a reviewer included by hand is in whatever its score. The house rule, on this page too.

2. EVERY COUNT NAMES ITS DENOMINATOR. The place of a study is read from the extraction, so
   its denominator is the papers that were READ, not the relevant ones; a coverage figure
   whose denominator is implied is the bug this field exists to prevent.

3. A SOURCE THAT CANNOT BE READ LOSES ITS COUNTS, NOT THE PAGE. The clustering may not be
   computed, the place resolver may have no list loaded: the register still serves every
   other variable, and says which one it could not read and why. A panel that fails whole
   is a panel nobody can use.

Plus the one that is not about counting: a scenario's own variables are the GROUPS of its
codebook, named the way the codebook names them, and a group the model wrote that maps to
nothing is listed as unmapped rather than dropped, because it is work waiting to be done.
"""
import json

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api.variables_of_interest import (  # noqa: E402
    COST_FREE,
    FAMILIES,
    OF_EXTRACTED,
    OF_RELEVANT,
    STANDARD,
    STANDARD_KEYS,
    _cut,
    variables_of_interest,
)
from conftest import ensure_document_columns  # noqa: E402

SID = "usr-register-test"
OTHER = "usr-register-other"
IDS = (9701, 9702, 9703, 9704, 9705)


# ── Pure: the shape of the register itself ──────────────────────────────────
def test_every_standard_variable_is_well_formed():
    """A row the frontend cannot read is a blank line in the interface. Each one needs a
    unique key, a family the page knows, a cost, a type, and a denominator."""
    assert len(set(STANDARD_KEYS)) == len(STANDARD_KEYS), "duplicate key in the register"
    for v in STANDARD:
        assert v["family"] in FAMILIES, v
        assert v["cost"] in ("free", "already_run", "one_pass"), v
        assert v["type"] in ("category", "number", "boolean", "date", "text"), v
        assert v["denominator_of"] in ("relevant", "extracted", "corpus"), v
        assert v["shown_in"], f"{v['key']} says nothing about where it is shown today"


def test_most_of_the_standard_register_costs_nothing():
    """The point of the standard register: the answer is mostly already in the database.
    If this ever drops below half, a variable has been added that needs a reading pass and
    was filed as if it were free."""
    free = [v for v in STANDARD if v["cost"] == COST_FREE]
    assert len(free) >= len(STANDARD) / 2


def test_a_value_list_is_cut_but_its_count_is_not():
    values = [{"value": str(i), "n": 100 - i} for i in range(40)]
    shown, total = _cut(values)
    assert total == 40 and len(shown) == 10
    assert shown[0]["label"] == "0", "a value with no label of its own reads as itself"


# ── Over the database ───────────────────────────────────────────────────────
def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:                                        # noqa: BLE001
        return False


def _extraction(location: str, sheet: str, group: str, covariate: str) -> str:
    return json.dumps({
        "source": "fulltext", "version": 1,
        "ref": {"location": location, "risk_pop": 100, "positive": 7},
        "coverage": {"sex_gender": True},
        "observations": [{"sheet": sheet, "group": group, "covariate": covariate,
                          "disease": "HPAI H5N1", "value": 1, "quote": "q",
                          "quote_verified": True}],
    })


@pytest.fixture()
def seeded(db_conn):
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("CREATE TABLE IF NOT EXISTS literature_document (id BIGINT PRIMARY KEY)")
        cur.execute("CREATE TABLE IF NOT EXISTS article_scenarios ("
                    "scenario_id TEXT, document_id BIGINT, PRIMARY KEY (scenario_id, document_id))")
        created_chunk_table = ensure_document_columns(cur)
        for col, typ in (("title", "TEXT"), ("abstract", "TEXT"), ("year", "INTEGER"),
                         ("journal", "TEXT"), ("language", "TEXT"), ("publication_type", "TEXT"),
                         ("open_access", "BOOLEAN"), ("citation_count", "INTEGER"),
                         ("doi", "TEXT"), ("has_fulltext", "BOOLEAN"),
                         ("pico_json", "JSONB"), ("extraction_json", "JSONB"),
                         ("is_duplicate", "BOOLEAN DEFAULT FALSE")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        for col, typ in (("similarity_score", "DOUBLE PRECISION"), ("screening_reason", "TEXT"),
                         ("screening_notes", "TEXT"), ("screened_at", "TIMESTAMP"),
                         ("kappa_final_status", "VARCHAR(20)")):
            cur.execute(f"ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS embedding TEXT")
        for sid in (SID, OTHER):
            cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (sid,))
            cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (sid,))
            cur.execute("DELETE FROM user_scenarios WHERE id = %s", (sid,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        for sid, name in ((SID, "Register"), (OTHER, "Another scenario")):
            cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, pinned) "
                        "VALUES (%s, %s, 'avian influenza', 'boolean', '{}', TRUE)", (sid, name))
        # Five papers, and only three of them are relevant:
        #   9701 above the threshold, read from its full text, placed in Italy;
        #   9702 above the threshold, read, placed in China;
        #   9703 BELOW the threshold, so out of the subset entirely;
        #   9704 below the threshold but INCLUDED by a reviewer, so in, and never read;
        #   9705 above the threshold but EXCLUDED by a reviewer, so out.
        cur.execute("""
            INSERT INTO literature_document
              (id, title, abstract, source, year, journal, language, publication_type,
               open_access, citation_count, doi, has_fulltext, is_duplicate, project_context,
               pico_json, extraction_json)
            VALUES
              (9701, 'Poultry workers in Lombardy', %(a)s, 'pubmed', 2024, 'Eurosurveillance',
               'en', 'Journal Article', TRUE, 12, '10.1/a', TRUE, FALSE, 'literev',
               %(p1)s::jsonb, %(x1)s::jsonb),
              (9702, 'Live bird markets', %(a)s, 'openalex', 2023, 'Emerg Infect Dis',
               'en', 'Journal Article', FALSE, 3, '10.1/b', FALSE, FALSE, 'literev',
               %(p2)s::jsonb, %(x2)s::jsonb),
              (9703, 'Off topic', %(a)s, 'pubmed', 2010, NULL, NULL, NULL,
               NULL, NULL, NULL, NULL, FALSE, 'literev', NULL, NULL),
              (9704, 'Found by hand', %(a)s, 'uploaded', 2022, NULL, 'fr', NULL,
               NULL, NULL, NULL, NULL, FALSE, 'literev', NULL, NULL),
              (9705, 'Wrong population', %(a)s, 'pubmed', 2021, NULL, NULL, NULL,
               NULL, NULL, NULL, NULL, FALSE, 'literev', NULL, NULL)
        """, {"a": "a study of exposure among poultry workers, with counts by sex",
              "p1": json.dumps({"study_design": "Cross-sectional study", "population": "poultry workers",
                                "key_finding": "exposure was higher among cullers",
                                "finding_direction": "increase"}),
              "p2": json.dumps({"study_design": "Cohort study", "population": "market vendors",
                                "key_finding": "seropositivity rose with market exposure",
                                "finding_direction": "increase"}),
              "x1": _extraction("Lombardy, Italy", "human_exp", "occupation", "poultry worker"),
              "x2": _extraction("Guangdong, China", "animal", "species", "chicken")})
        cur.execute("""
            INSERT INTO article_scenarios
              (scenario_id, document_id, similarity_score, screening_status, screening_reason,
               kappa_final_status)
            VALUES (%s, 9701, 0.90, NULL, NULL, 'agreed'),
                   (%s, 9702, 0.85, NULL, NULL, NULL),
                   (%s, 9703, 0.10, NULL, NULL, NULL),
                   (%s, 9704, 0.05, 'included', 'found by hand', NULL),
                   (%s, 9705, 0.95, 'excluded', 'wrong population', NULL)
        """, (SID,) * 5)
        cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score) "
                    "VALUES (%s, 9701, 0.88)", (OTHER,))
        for sid in (SID, OTHER):
            cur.execute("INSERT INTO scenario_settings (scenario_id, similarity_threshold) VALUES (%s, 0.45) "
                        "ON CONFLICT (scenario_id) DO UPDATE SET similarity_threshold = 0.45", (sid,))
    yield db_conn
    with db_conn.cursor() as cur:
        for sid in (SID, OTHER):
            cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (sid,))
            cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (sid,))
            cur.execute("DELETE FROM user_scenarios WHERE id = %s", (sid,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        if created_chunk_table:
            cur.execute("DROP TABLE document_chunk")


def _by_key(out: dict) -> dict:
    return {v["key"]: v for v in out["variables"]}


def test_the_register_counts_the_relevant_subset_and_nothing_else(seeded):
    """Three papers are relevant out of five: two above the threshold and one a reviewer
    included by hand. The one below the threshold and the one a reviewer excluded are not
    counted anywhere, and the corpus total is reported separately so the two can be told
    apart."""
    out = variables_of_interest(SID, lang="en")
    assert out["n_relevant"] == 3
    assert out["n_corpus"] == 5
    v = _by_key(out)
    assert v["year"]["n_with_value"] == 3            # every relevant paper has a year
    assert v["year"]["denominator"] == 3
    assert v["journal"]["n_with_value"] == 2         # 9704 has none
    assert v["open_access"]["n_with_value"] == 2     # 9704 says nothing either way


def test_the_threshold_variable_is_counted_over_the_corpus_not_the_subset(seeded):
    """"Above the threshold" is the only variable whose question is about the whole corpus.
    Counting it over the relevant subset would print "3 of 3", which says nothing."""
    v = _by_key(variables_of_interest(SID, lang="en"))["above_threshold"]
    assert v["denominator_of"] == "corpus" and v["denominator"] == 5
    assert v["n_with_value"] == 3
    assert {x["value"]: x["n"] for x in v["values"]} == {"above": 3, "below": 2}


def test_a_place_is_counted_out_of_the_papers_that_were_read(seeded):
    """Two of the three relevant papers were read, and both gave a place. The denominator
    is therefore 2, not 3: the third says nothing because nobody has read it yet, and
    printing "2 of 3" would read as a third of the corpus having no place."""
    out = variables_of_interest(SID, lang="en")
    assert out["n_extracted"] == 2
    country = _by_key(out)["country"]
    assert country["denominator_of"] == OF_EXTRACTED and country["denominator"] == 2
    assert country["n_with_value"] == 2
    assert {x["value"] for x in country["values"]} == {"IT", "CN"}


def test_the_design_is_folded_and_named_in_the_asked_language(seeded):
    """The design is stored as the model worded it. The register shows the folded term, and
    names it in the reader's language: the same count, two spellings."""
    en = _by_key(variables_of_interest(SID, lang="en"))["design"]
    fr = _by_key(variables_of_interest(SID, lang="fr"))["design"]
    assert en["n_with_value"] == 3 and fr["n_with_value"] == 3
    labels_en = {x["label"] for x in en["values"]}
    labels_fr = {x["label"] for x in fr["values"]}
    assert "Cross-sectional" in " ".join(labels_en)
    assert "Transversale" in labels_fr
    assert labels_en != labels_fr, "the label follows the language, the value does not"


def test_europe_elsewhere_and_no_place_add_up_to_the_papers_read(seeded):
    """The three buckets of the place question must sum to the papers that were read, and
    "elsewhere" is not "placed minus European".

    A paper can be European without naming one country ("Europe", "Germany and the
    Netherlands"), so it is not among the papers with a named country. Subtracting all the
    European papers from the placed ones removes those twice: on the HPAI exposure review
    that printed 339 where the truth is 382, while the country selection, counting the same
    thing another way, said 382. Making the three sum to the denominator makes that class of
    mistake impossible to repeat unseen."""
    out = variables_of_interest(SID, lang="en")
    europe = _by_key(out)["europe"]
    buckets = {x["value"]: x["n"] for x in europe["values"]}
    assert set(buckets) == {"europe", "elsewhere", "no_place"}
    assert sum(buckets.values()) == out["n_extracted"] == europe["denominator"]
    # Of the two papers read, one is in Italy and one in China, and both name a country.
    assert buckets == {"europe": 1, "elsewhere": 1, "no_place": 0}


def test_the_scenarios_own_variables_are_its_codebook_groups(seeded):
    """One variable per codebook group, named as the codebook names it, counted over the
    papers that answered it. A group nobody answered is listed with zero: that is the gap
    map, and a blank there is a result rather than a missing row."""
    out = variables_of_interest(SID, lang="en")
    own = {v["key"]: v for v in out["variables"] if v["family"] == "own"}
    assert out["n_own"] == len(own) >= 10
    assert own["human_exp.occupation"]["name"] == "Occupation"
    assert own["human_exp.occupation"]["n_with_value"] == 1
    assert own["animal.species"]["n_with_value"] == 1
    # A group the codebook declares and no paper answered: zero, listed, not dropped.
    assert own["human_susc.pregnancy"]["n_with_value"] == 0
    assert own["human_exp.occupation"]["values"][0]["label"] == "Poultry worker"


def test_a_misplaced_group_is_rescued_by_the_value_beside_it(seeded, db_conn):
    """"wild bird" written where a group belongs is a value of Species in the wrong column.
    When the value beside it is one the codebook knows ("mallard" is a duck), the position is
    recovered from that value and the row joins Species: no unmapped group, no lost paper."""
    with db_conn.cursor() as cur:
        cur.execute("UPDATE literature_document SET extraction_json = %s::jsonb WHERE id = 9702",
                    (_extraction("Guangdong, China", "animal", "wild bird", "mallard"),))
    out = variables_of_interest(SID, lang="en")
    own = {v["key"]: v for v in out["variables"] if v["family"] == "own"}
    assert out["n_unmapped"] == 0
    assert own["animal.species"]["n_with_value"] == 1
    assert own["animal.species"]["values"][0]["label"] == "Duck"


def test_a_group_the_codebook_maps_to_nothing_is_listed_as_unmapped(seeded, db_conn):
    """When neither the group nor the value beside it matches anything, the row cannot be
    placed. Dropping it would hide a hundred papers on a real corpus, so the register lists
    it under its own spelling, marked, with the papers behind it, and a reviewer can fold it
    in with one synonym."""
    with db_conn.cursor() as cur:
        cur.execute("UPDATE literature_document SET extraction_json = %s::jsonb WHERE id = 9702",
                    (_extraction("Guangdong, China", "animal", "wild bird",
                                 "sampled in wetlands"),))
    out = variables_of_interest(SID, lang="en")
    own = {v["key"]: v for v in out["variables"] if v["family"] == "own"}
    assert out["n_unmapped"] == 1
    assert out["n_papers_unmapped"] == 1, "the papers behind the stray labels, not the labels"
    stray = own["animal.wild bird"]
    assert stray["unmapped"] is True and stray["n_with_value"] == 1
    assert own["animal.species"]["n_with_value"] == 0, "and it did not join Species by accident"


def test_a_declared_group_is_never_cut_by_the_cap_on_stray_labels(seeded, db_conn):
    """On a real corpus the model writes a long tail of group labels the codebook cannot
    place: 102 of them on the HPAI exposure review. A single cap over both kinds let that
    tail push the review's own questions off the page, which turns the gap map into a list of
    whatever happens to be present. The declared groups are therefore never cut, the tail is,
    and the payload says how long it really is."""
    import json as _json

    from api.variables_of_interest import _MAX_UNMAPPED_SHOWN

    obs = [{"sheet": "animal", "group": f"stray label {i}", "covariate": "nothing known",
            "disease": "HPAI H5N1", "quote": "q", "quote_verified": True}
           for i in range(_MAX_UNMAPPED_SHOWN + 12)]
    with db_conn.cursor() as cur:
        cur.execute("UPDATE literature_document SET extraction_json = %s::jsonb WHERE id = 9702",
                    (_json.dumps({"source": "fulltext", "version": 1,
                                  "ref": {"location": "Guangdong, China"},
                                  "coverage": {}, "observations": obs}),))
    out = variables_of_interest(SID, lang="en")
    own = [v for v in out["variables"] if v["family"] == "own"]
    declared = [v for v in own if not v.get("unmapped")]
    strays = [v for v in own if v.get("unmapped")]
    assert out["n_unmapped"] == _MAX_UNMAPPED_SHOWN + 12, "the true length, not the shown one"
    assert len(strays) == _MAX_UNMAPPED_SHOWN == out["n_unmapped_shown"]
    assert len(declared) == out["n_own"] >= 19, "every declared group survived the tail"
    assert "human_susc.sex_gender" in {v["key"] for v in declared}


def test_a_source_that_cannot_be_read_loses_its_counts_not_the_page(seeded):
    """No clustering is cached for this scenario, which is the normal state of a new review
    and not a failure. The theme variable is served with nothing counted and the reason
    said; every other variable is still there, and the register does not call itself
    incomplete for it."""
    out = variables_of_interest(SID, lang="en")
    assert out["complete"] is True
    assert "cluster" in out["notes"]
    cluster = _by_key(out)["cluster"]
    assert cluster["n_with_value"] == 0 and cluster["values"] == []
    assert _by_key(out)["year"]["n_with_value"] == 3, "the rest of the register is served"


def test_the_endpoint_serves_the_register_read_only(seeded):
    from fastapi.testclient import TestClient

    client = TestClient(main.app)
    r = client.get(f"/user-scenarios/{SID}/variables-of-interest", params={"lang": "en"})
    assert r.status_code == 200
    body = r.json()
    assert body["n_relevant"] == 3 and body["n_standard"] == len(STANDARD_KEYS)
    assert {v["key"] for v in body["variables"]} >= set(STANDARD_KEYS)
    assert client.get("/user-scenarios/usr-nope/variables-of-interest").status_code == 404


def test_the_same_paper_in_another_scenario_is_not_counted_here(seeded):
    """Paper 9701 belongs to two scenarios. The other scenario has its own link row and no
    settings of its own beyond the threshold, so its register counts one paper, not three."""
    out = variables_of_interest(OTHER, lang="en")
    assert out["n_relevant"] == 1 and out["n_corpus"] == 1
    assert _by_key(out)["year"]["n_with_value"] == 1
