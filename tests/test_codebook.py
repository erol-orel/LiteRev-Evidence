"""The label codebook (api/codebook.py): hierarchy per sheet, normalised at read time.

Pure tests cover the folding and the matching (the cases that go wrong in a naive version:
"male" inside "female", a plural turning "species" into "specie", an unmatched label merging
with a matched one); the integration tests run the endpoints against the test database and
check that editing the codebook corrects extractions already stored."""
import json
import types

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api import codebook, extraction  # noqa: E402
from conftest import patch_app  # noqa: E402
from test_extraction import SID, HDR, PAPER, _fake_client, extracted, seeded  # noqa: E402,F401

IDX = codebook.Index(codebook.DEFAULT_NODES)


def _n(sheet, group, cov):
    return codebook.normalise(IDX, sheet, group, cov)


# ── Folding ─────────────────────────────────────────────────────────────────
def test_labels_fold_case_accents_punctuation_and_plurals():
    f = codebook.clean_label
    assert f("  Males ") == "male" and f("MEN") == "man" and f("Women") == "woman"
    assert f("Hôtes   animaux") == "hote animaux"
    assert f("Sex/Gender") == "sex gender"
    assert f("aged 65+") == "aged 65+" and f("<5 years") == "<5 year"
    assert f(None) == "" and f("") == ""


def test_a_node_key_is_never_singularised():
    assert codebook.key_slug("Species") == "species"
    assert codebook.key_slug("Sex / Gender") == "sex_gender"
    assert codebook.key_slug("Biosécurité") == "biosecurite"


# ── Matching ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("label,expected", [
    ("Males", "male"), ("men", "male"), ("female", "female"), ("Women", "female"), ("girls", "female"),
])
def test_sex_labels_reach_their_value_and_male_never_matches_inside_female(label, expected):
    r = _n("human_susc", "sex", label)
    assert (r["l1"], r["l2"], r["matched"]) == ("sex_gender", expected, True)


def test_the_group_decides_between_two_nodes_that_both_fit():
    # "male farmers" holds both a sex and an occupation: the group the paper filed it under decides.
    assert _n("human_susc", "sex", "male farmers")["l2"] == "male"
    assert _n("human_exp", "occupation", "male farmers")["l2"] == "farmer"


def test_the_longest_synonym_wins():
    assert _n("human_exp", "occupation", "poultry farmers")["l2"] == "poultry_worker"
    assert _n("human_exp", "occupation", "farmers")["l2"] == "farmer"


def test_latin_and_common_names_reach_the_same_species():
    assert _n("animal", "species", "Felis catus")["l2"] == "cat"
    assert _n("animal", "species", "Domestic cats")["l2"] == "cat"
    assert _n("animal", "species", "Gallus gallus domesticus")["l2"] == "chicken"


def test_an_unmatched_label_keeps_its_own_spelling_and_never_merges_by_accident():
    r = _n("human_susc", "sex", "zzz unknown")
    assert r["matched"] is False and r["l1"] == "sex_gender" and r["l2"] is None
    assert r["covariate_key"] == "zzz unknown"
    r2 = _n("human_exp", "weird group", "zzz")
    assert r2["l1"] is None and r2["group_key"] == "weird group" and r2["covariate_key"] == "zzz"


def test_a_group_is_recognised_without_a_value_and_a_value_without_a_group():
    assert _n("human_susc", "Sex/Gender", "")["l1"] == "sex_gender"
    assert _n("human_susc", "", "Men")["l2"] == "male"


def test_the_annotation_adds_to_an_observation_and_changes_nothing_else():
    o = {"sheet": "human_susc", "group": "sex", "covariate": "Males", "value": 3}
    a = codebook.annotate(IDX, o)
    assert a["value"] == 3 and a["covariate"] == "Males"             # the stored words are untouched
    assert (a["group_key"], a["covariate_key"], a["label_path"]) == ("sex_gender", "male", "sex_gender > male")
    assert codebook.annotate(IDX, {"sheet": "env", "covariate": "zzz"})["label_path"] is None


def test_every_sheet_has_a_default_vocabulary():
    for sheet in codebook.CODEBOOK_SHEETS:
        assert any(n["sheet"] == sheet for n in codebook.DEFAULT_NODES), sheet
    assert chr(0x2014) not in json.dumps(codebook.DEFAULT_NODES, ensure_ascii=False)


def test_the_default_codebook_is_itself_valid():
    assert len(codebook.validate_nodes([dict(n) for n in codebook.DEFAULT_NODES])) == len(codebook.DEFAULT_NODES)


# ── Validation and CSV ──────────────────────────────────────────────────────
def test_validation_says_what_is_wrong():
    v = codebook.validate_nodes
    with pytest.raises(ValueError, match="at least one"):
        v([])
    with pytest.raises(ValueError, match="node 1: unknown sheet"):
        v([{"sheet": "nope", "l1": "a"}])
    with pytest.raises(ValueError, match="level 1 is required"):
        v([{"sheet": "env", "l1": ""}])
    with pytest.raises(ValueError, match="level 3 needs level 2"):
        v([{"sheet": "env", "l1": "a", "l3": "c"}])
    with pytest.raises(ValueError, match="duplicate"):
        v([{"sheet": "env", "l1": "a"}, {"sheet": "env", "l1": "A"}])
    ok = v([{"sheet": "HUMAN_COV_SUSC", "l1": "Sex", "l2": "Male", "synonyms": "Men|boys; he"}])
    assert ok[0]["sheet"] == "human_susc" and ok[0]["synonyms"] == ["man", "boy", "he"]


def test_a_csv_in_the_reviewers_shape_is_read():
    csv_text = ("sheet,level1,level2,level3,synonyms,label_en,label_fr\n"
                "HUMAN_COV_SUSC,sex,,,sexe|gender,Sex,Sexe\n"
                "HUMAN_COV_SUSC,sex,male,,men|males,Male,Homme\n"
                "HUMAN_COV_SUSC,sex,female,,women,Female,Femme\n"
                ",,,,,,\n")
    nodes = codebook.parse_codebook_csv("﻿" + csv_text)
    assert [(n["l1"], n["l2"]) for n in nodes] == [("sex", None), ("sex", "male"), ("sex", "female")]
    idx = codebook.Index(nodes)
    assert codebook.normalise(idx, "human_susc", "gender", "Women")["l2"] == "female"
    with pytest.raises(ValueError, match="needs at least the columns"):
        codebook.parse_codebook_csv("a,b\n1,2\n")
    back = codebook.parse_codebook_csv(codebook.codebook_to_csv(nodes))
    assert [(n["l1"], n["l2"], n["synonyms"]) for n in back] == [(n["l1"], n["l2"], n["synonyms"]) for n in nodes]


def test_the_extraction_prompt_carries_the_vocabulary():
    block = codebook.vocabulary_prompt(codebook.DEFAULT_NODES)
    assert "PREFERRED GROUP NAMES" in block and "sex_gender (male, female" in block
    assert "human_exp:" in block and "occupation" in block and chr(0x2014) not in block


# ── Endpoints and the effect on stored extractions ──────────────────────────
def _client():
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def test_the_codebook_endpoints_round_trip(extracted):
    c = _client()
    cb = c.get(f"/user-scenarios/{SID}/codebook").json()
    assert cb["source"] == "default" and cb["n_nodes"] == len(codebook.DEFAULT_NODES)

    # Writing needs the key and a valid body.
    assert c.put(f"/user-scenarios/{SID}/codebook", json={"nodes": []}).status_code in (401, 403)
    bad = c.put(f"/user-scenarios/{SID}/codebook", json={"nodes": [{"sheet": "x", "l1": "a"}]}, headers=HDR)
    assert bad.status_code == 422 and "unknown sheet" in bad.json()["detail"]

    csv_text = ("sheet,level1,level2,synonyms\n"
                "human_susc,sex,,\nhuman_susc,sex,male,men|males\nhuman_susc,sex,female,women\n")
    r = c.post(f"/user-scenarios/{SID}/codebook/import", content=csv_text, headers=HDR)
    assert r.status_code == 200 and r.json() == {"scenario_id": SID, "source": "custom", "n_nodes": 3}
    assert c.get(f"/user-scenarios/{SID}/codebook").json()["source"] == "custom"
    assert c.post(f"/user-scenarios/{SID}/codebook/import", content="x,y\n1,2", headers=HDR).status_code == 422
    exported = c.get(f"/user-scenarios/{SID}/codebook/export")
    assert exported.status_code == 200 and "human_susc,sex,male" in exported.text

    reset = c.delete(f"/user-scenarios/{SID}/codebook", headers=HDR).json()
    assert reset["source"] == "default"
    assert c.get(f"/user-scenarios/{SID}/codebook").json()["source"] == "default"
    assert c.get("/user-scenarios/nope/codebook").status_code == 404


def test_unmapped_labels_are_listed_and_a_synonym_maps_them(extracted):
    c = _client()
    un = c.get(f"/user-scenarios/{SID}/codebook/unmapped").json()
    # The seeded rows: Male x2 (sex), female (sex), farmer (occupation). All map by default.
    assert un["rows_unmapped"] == 0 and un["rows_mapped"] == 4
    with extracted.cursor() as cur:
        cur.execute("UPDATE literature_document SET extraction_json = jsonb_set(extraction_json, '{observations,0,covariate}', "
                    "'\"Hombres\"') WHERE id = 9701")
    un = c.get(f"/user-scenarios/{SID}/codebook/unmapped").json()
    assert un["rows_unmapped"] == 1 and un["unmapped"][0]["covariate"] == "Hombres" and un["unmapped"][0]["l1"] == "sex_gender"

    r = c.post(f"/user-scenarios/{SID}/codebook/synonym", headers=HDR,
               json={"sheet": "human_susc", "l1": "sex_gender", "l2": "male", "label": "Hombres"})
    assert r.status_code == 200 and "hombre" in r.json()["synonyms"]
    # The stored extraction is untouched; the label now reads as male.
    un = c.get(f"/user-scenarios/{SID}/codebook/unmapped").json()
    assert un["rows_unmapped"] == 0
    art = c.get(f"/user-scenarios/{SID}/articles/9701/extraction").json()["extraction"]["observations"][0]
    assert art["covariate"] == "Hombres" and art["covariate_key"] == "male" and art["label_path"] == "sex_gender > male"

    assert c.post(f"/user-scenarios/{SID}/codebook/synonym", headers=HDR,
                  json={"sheet": "human_susc", "l1": "sex_gender", "l2": "nope", "label": "x"}).status_code == 404
    assert c.post(f"/user-scenarios/{SID}/codebook/synonym", json={}).status_code in (401, 403)
    c.delete(f"/user-scenarios/{SID}/codebook", headers=HDR)


def test_a_scenario_run_sends_the_codebook_vocabulary_and_records_its_own_prompt(seeded, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    systems = []
    base = _fake_client()
    real = base.chat.completions.create

    def create(**kw):
        systems.append(kw["messages"][0]["content"])
        return real(**kw)
    base.chat.completions.create = create
    patch_app(monkeypatch, "_llm_client", lambda: base)
    extraction._jobs[SID] = {"running": True, "total": 0, "done": 0, "failed": 0}
    extraction._extract_scenario(SID, "Avian influenza")
    assert systems and all("PREFERRED GROUP NAMES" in s_ for s_ in systems)
    with seeded.cursor() as cur:
        cur.execute("SELECT extraction_json->>'prompt_sha' FROM literature_document WHERE id = 9701")
        sha = cur.fetchone()[0]
    import hashlib
    assert sha == hashlib.sha256(systems[0].encode()).hexdigest()[:10] and sha != extraction.PROMPT_SHA


def test_the_workbook_and_csv_carry_the_codebook_label(extracted):
    openpyxl = pytest.importorskip("openpyxl")
    import io
    c = _client()
    wb = openpyxl.load_workbook(io.BytesIO(c.get(f"/user-scenarios/{SID}/extraction/export").content))
    head = [x.value for x in wb["HUMAN_COV_SUSC"][1]]
    labels = {r[head.index("Codebook label")].value for r in wb["HUMAN_COV_SUSC"].iter_rows(min_row=2)}
    assert {"sex_gender > male", "sex_gender > female"} <= labels
    csv_text = c.get(f"/user-scenarios/{SID}/extraction/export?format=csv").text
    assert "l1,l2" in csv_text.splitlines()[0] and "sex_gender,male" in csv_text


# ── Synonyms beyond the limit are refused, never dropped ────────────────────
def test_a_node_takes_200_synonyms_and_the_201st_is_refused_with_its_count():
    """The import kept the first 40 synonyms of a node and dropped the rest without a word."""
    syn = "|".join(f"label number {i}" for i in range(200))
    ok = codebook.validate_nodes([{"sheet": "human_exp", "l1": "occupation", "l2": "farmer", "synonyms": syn}])
    assert codebook._MAX_SYNONYMS == 200 and len(ok[0]["synonyms"]) == 200
    with pytest.raises(ValueError, match="201 distinct synonyms; the maximum is 200"):
        codebook.validate_nodes([{"sheet": "human_exp", "l1": "occupation", "l2": "farmer", "synonyms": syn + "|one more"}])
    # A sheet written another way is the sheet it names.
    assert codebook.sheet_key("Human_Susc") == "human_susc" and codebook.sheet_key("Environment") == "env"
    assert codebook.sheet_key("ANIMALorRESERVOIR_COV") == "animal" and codebook.sheet_key("lab") is None


def test_the_import_and_the_mapping_refuse_instead_of_cutting(extracted):
    c = _client()
    many = "|".join(f"job {i}" for i in range(201))
    r = c.post(f"/user-scenarios/{SID}/codebook/import", headers=HDR,
               content=f"sheet,level1,level2,synonyms\nhuman_exp,occupation,,\nhuman_exp,occupation,farmer,{many}\n")
    assert r.status_code == 422 and "the maximum is 200" in r.json()["detail"]
    full = "|".join(f"job {i}" for i in range(200))
    assert c.post(f"/user-scenarios/{SID}/codebook/import", headers=HDR,
                  content=f"sheet,level1,level2,synonyms\nhuman_exp,occupation,,\nhuman_exp,occupation,farmer,{full}\n").status_code == 200
    # The node is full: a new label is refused with a reason, not appended then cut with a 200.
    add = c.post(f"/user-scenarios/{SID}/codebook/synonym", headers=HDR,
                 json={"sheet": "HUMAN_EXP", "l1": "occupation", "l2": "farmer", "label": "Granjeros"})
    assert add.status_code == 422 and "already has 200 synonyms" in add.json()["detail"]
    nodes = c.get(f"/user-scenarios/{SID}/codebook").json()["nodes"]
    farmer = next(n for n in nodes if n.get("l2") == "farmer")
    assert len(farmer["synonyms"]) == 200 and "granjero" not in farmer["synonyms"]
    # A label it already holds is not refused.
    assert c.post(f"/user-scenarios/{SID}/codebook/synonym", headers=HDR,
                  json={"sheet": "human_exp", "l1": "occupation", "l2": "farmer", "label": "job 7"}).status_code == 200
    c.delete(f"/user-scenarios/{SID}/codebook", headers=HDR)
