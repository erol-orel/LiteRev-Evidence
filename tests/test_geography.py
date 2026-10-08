"""Where the evidence comes from (api/geography.py): a place, as a paper words it, to a country and a
NUTS code.

The resolver is built to be conservative, and these tests pin the ways a naive one goes wrong: a bird
read as a country, two countries read as one, a region guessed from a name that exists in several
places, and a code made up where the list does not know the region. The official list is loaded
from a small CSV in the shape Eurostat publishes."""
import io

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api import geography as G  # noqa: E402
from test_extraction import HDR, SID, extracted, seeded  # noqa: E402,F401

BUILTIN = G.NutsIndex(G._builtin_entries(), "builtin")


def _r(place, hint=None, index=BUILTIN):
    return G.resolve_location(place, hint, index)


# ── Folding and countries ───────────────────────────────────────────────────
def test_places_fold_to_plain_ascii_words():
    assert G.fold("Baden-Württemberg, Germany") == "baden wurttemberg germany"
    assert G.fold("Valle d'Aosta") == "valle d aosta" and G.fold(None) == ""


def test_every_country_with_a_nuts_code_is_known_and_two_codes_differ_from_iso():
    assert all(iso in G.COUNTRY_NAME for iso in G._NUTS_ISO)
    assert G.nuts0_of("GR") == "EL" and G.nuts0_of("GB") == "UK" and G.nuts0_of("DE") == "DE"
    assert G.nuts0_of("VN") is None and G.nuts0_of(None) is None
    assert G.iso_of_nuts0("EL") == "GR" and G.iso_of_nuts0("UK") == "GB" and G.iso_of_nuts0("FR") == "FR"


def test_a_country_is_recognised_in_several_languages():
    for place, iso in (("Germany", "DE"), ("Allemagne", "DE"), ("Deutschland", "DE"), ("Italie", "IT"), ("Spagna", "ES"),
                       ("The Netherlands", "NL"), ("Pays-Bas", "NL"), ("England", "GB"), ("UK", "GB"), ("Suisse", "CH")):
        assert _r(place)["country"] == iso, place


def test_turkey_the_bird_is_never_a_country():
    assert _r("a commercial farm keeping turkeys and ducks")["country"] is None
    assert _r("Turkey")["country"] is None


def test_two_countries_are_several_countries_not_a_guess():
    r = _r("Italy and France")
    assert r["several_countries"] is True and r["country"] is None and r["method"] == "unresolved"
    assert _r("Germany, Italy")["several_countries"] is True


def test_a_word_only_matches_whole_words():
    assert _r("Chadwick")["country"] is None and _r("Marseille")["country"] is None
    assert _r("Poland")["country"] == "PL" and _r("Polandia")["country"] is None


# ── Regions ─────────────────────────────────────────────────────────────────
def test_a_region_resolves_with_its_whole_hierarchy():
    r = _r("Sigmaringen, Baden-Württemberg, Germany")
    assert (r["country"], r["nuts0"], r["nuts1"], r["nuts2"], r["level"], r["method"]) == ("DE", "DE", "DE1", None, 1, "region")
    r = _r("Lombardy, northern Italy")
    assert (r["nuts0"], r["nuts1"], r["nuts2"], r["level"]) == ("IT", "ITC", "ITC4", 2)
    assert _r("Saxony")["nuts1"] == "DED" and _r("Saxony-Anhalt")["nuts1"] == "DEE"      # the longest name wins


def test_the_country_the_paper_names_decides_and_the_articles_country_fills_in():
    assert _r("Bolzano", "IT")["nuts2"] == "ITH1"
    assert _r("", "DE")["method"] == "article_country" and _r("", "DE")["nuts0"] == "DE"
    assert _r("somewhere rural")["method"] == "unresolved"
    # The paper's own country beats the article's.
    assert _r("Berlin, Germany", "IT")["nuts1"] == "DE3"


def test_a_country_without_nuts_is_a_country_without_a_code():
    r = _r("Hanoi, Vietnam")
    assert r["country"] == "VN" and r["nuts0"] is None and r["level"] is None and r["method"] == "country"


def test_a_region_is_never_invented_where_the_list_does_not_know_it():
    r = _r("Provence, France")
    assert r["country"] == "FR" and r["nuts1"] is None and r["level"] == 0


# ── The official list ───────────────────────────────────────────────────────
GISCO = ("NUTS_ID,LEVL_CODE,CNTR_CODE,NAME_LATN,NUTS_NAME,MOUNT_TYPE\n"
         "DE,0,DE,Deutschland,Deutschland,\nDE1,1,DE,Baden-Württemberg,Baden-Württemberg,\nDE13,2,DE,Freiburg,Freiburg,\n"
         "DE139,3,DE,Sigmaringen,Sigmaringen,\nFR1,1,FR,Île-de-France,Île-de-France,\nEL3,1,EL,Attiki,Αττική,\n")


def test_the_gisco_file_is_parsed_and_the_country_rows_are_skipped():
    rows = G.parse_nuts_csv(GISCO)
    assert [r[0] for r in rows] == ["DE1", "DE13", "DE139", "FR1", "EL3"]
    assert [(r[0], r[1], r[2]) for r in rows if r[0] == "EL3"] == [("EL3", 1, "GR")]        # NUTS EL is ISO GR
    assert G.parse_nuts_csv("code,level,name\nITC4,2,Lombardia\n")[0][:4] == ("ITC4", 2, "IT", "Lombardia")


def test_a_bad_file_says_what_is_wrong():
    with pytest.raises(ValueError, match="needs the columns"):
        G.parse_nuts_csv("a,b\n1,2\n")
    with pytest.raises(ValueError, match="no NUTS 1, 2 or 3 row"):
        G.parse_nuts_csv("NUTS_ID,NUTS_NAME\nDE,Deutschland\n")
    with pytest.raises(ValueError, match="does not match the code"):
        G.parse_nuts_csv("NUTS_ID,LEVL_CODE,NUTS_NAME\nDE139,1,Sigmaringen\n")


def test_a_loaded_list_resolves_the_district_the_builtin_one_cannot():
    idx = G.NutsIndex(G._builtin_entries() + [(c, l, k, n) for c, l, k, n, _a in G.parse_nuts_csv(GISCO)], "loaded")
    r = G.resolve_location("Sigmaringen, Germany", None, idx)
    assert (r["nuts1"], r["nuts2"], r["nuts3"], r["level"]) == ("DE1", "DE13", "DE139", 3)
    assert G.resolve_location("Sigmaringen", None, BUILTIN)["country"] is None            # the built-in list does not know it


# ── Endpoints ───────────────────────────────────────────────────────────────
def _client():
    from fastapi.testclient import TestClient
    return TestClient(main.app)


@pytest.fixture()
def nuts_clean(extracted):
    with extracted.cursor() as cur:
        cur.execute("DELETE FROM geo_nuts")
    G._cache["key"] = None
    yield extracted
    with extracted.cursor() as cur:
        cur.execute("DELETE FROM geo_nuts")
    G._cache["key"] = None


def test_the_import_needs_the_key_and_a_valid_file_and_changes_what_resolves(nuts_clean):
    c = _client()
    assert c.get("/geo/nuts/status").json()["source"] == "builtin"
    assert c.post("/geo/nuts/import", content=GISCO).status_code in (401, 403)
    assert c.post("/geo/nuts/import", content="a,b\n1,2", headers=HDR).status_code == 422
    r = c.post("/geo/nuts/import", content=GISCO, headers=HDR)
    assert r.status_code == 200 and r.json() == {"loaded": 5, "by_level": {"1": 3, "2": 1, "3": 1}, "countries": 3}
    assert c.get("/geo/nuts/status").json()["source"] == "loaded"
    got = c.get("/geo/resolve", params={"place": "Sigmaringen, Germany"}).json()
    assert got["nuts3"] == "DE139" and got["level"] == 3
    # Loading again replaces, it does not add.
    assert c.post("/geo/nuts/import", content="code,level,name\nITC4,2,Lombardia\n", headers=HDR).json()["loaded"] == 1
    assert c.get("/geo/resolve", params={"place": "Sigmaringen, Germany"}).json()["level"] == 0


def _set_places(conn, places):
    import json
    with conn.cursor() as cur:
        for i, (place, country) in places.items():
            cur.execute("UPDATE literature_document SET country = %s, extraction_json = jsonb_set(extraction_json, '{ref}', %s::jsonb) WHERE id = %s",
                        (country, json.dumps({"location": place}), i))


def test_the_geography_counts_every_relevant_extracted_article(nuts_clean):
    c = _client()
    # 9701, 9702, 9703 are the relevant extracted articles; 9704 (below the threshold) and 9705 (excluded) carry places too.
    _set_places(nuts_clean, {9701: ("Sigmaringen, Baden-Württemberg, Germany", None), 9702: ("Lombardy", "IT"),
                             9703: ("nowhere in particular", None), 9704: ("Paris, France", None), 9705: ("Madrid, Spain", None)})
    g = c.get(f"/user-scenarios/{SID}/extraction/geography").json()
    assert (g["n_papers"], g["n_resolved"], g["n_unresolved"], g["nuts_source"]) == (3, 2, 1, "builtin")
    by = {x["iso2"]: x for x in g["countries"]}
    assert set(by) == {"DE", "IT"}                                    # not France, not Spain: not relevant
    assert by["DE"]["n_papers"] == 1 and by["DE"]["regions"] == [{"code": "DE1", "n_papers": 1, "name": "Baden-Wuerttemberg"}]
    assert by["IT"]["regions"][0]["code"] == "ITC4" and by["IT"]["nuts0"] == "IT"
    assert g["unresolved"] == [{"location": "nowhere in particular", "n": 1}]
    assert c.get("/user-scenarios/nope/extraction/geography").status_code == 404


def test_the_ref_sheet_carries_the_nuts_codes(nuts_clean):
    openpyxl = pytest.importorskip("openpyxl")
    _set_places(nuts_clean, {9701: ("Lombardy, Italy", None)})
    wb = openpyxl.load_workbook(io.BytesIO(_client().get(f"/user-scenarios/{SID}/extraction/export").content))
    head = [x.value for x in wb["REF"][1]]
    rows = {r[0].value: {h: x.value for h, x in zip(head, r)} for r in wb["REF"].iter_rows(min_row=2)}
    assert rows["LR9701"]["study area NUTS level 1 "] == "ITC" and rows["LR9701"]["study area NUTS level 2 "] == "ITC4"
    assert rows["LR9701"]["study area NUTS level 3"] is None and rows["LR9701"]["geographic location and area"] == "Lombardy, Italy"
    readme = " ".join(str(r[0].value) for r in wb["README"].iter_rows())
    assert "NUTS columns are filled from the place name" in readme


def test_the_report_has_a_geography_section(nuts_clean):
    _set_places(nuts_clean, {9701: ("Berlin, Germany", None), 9702: ("Lombardy", "IT"), 9703: ("Berlin", "DE")})
    md = _client().get(f"/user-scenarios/{SID}/extraction/report", params={"format": "md"}).text
    assert "## Where the evidence comes from" in md and "| Germany | DE | 2 |" in md and "| Italy | IT | 1 |" in md
    assert "Berlin DE3 (2)" in md and "Lombardia ITC4 (1)" in md
    assert "3 of the 3 extracted articles" in md
