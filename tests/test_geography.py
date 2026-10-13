"""Where the evidence comes from (api/geography.py): a place, as a paper words it, to a country and a
NUTS code.

The resolver is built to be conservative, and these tests pin the ways a naive one goes wrong: a bird
read as a country, two countries read as one, a region guessed from a name that exists in several
places, a code made up where the list does not know the region, a short name read inside a longer
one (Mexico in New Mexico, US in Belarus), and a continent given the article's country. The official
list is loaded from a small CSV in the shape Eurostat publishes."""
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
    assert G.fold("Trinidad & Tobago") == "trinidad and tobago"                              # "&" joins a list, as "and"
    assert G.fold("Hong Kong; China") == "hong kong and china"                                  # so does ";"
    assert G.fold("Valle d\u2019Aosta") == G.fold("Valle d'Aosta")                               # a typographic apostrophe


def test_every_country_with_a_nuts_code_is_known_and_two_codes_differ_from_iso():
    assert all(iso in G.COUNTRY_NAME for iso in G._NUTS_ISO)
    assert G.nuts0_of("GR") == "EL" and G.nuts0_of("GB") == "UK" and G.nuts0_of("DE") == "DE" and G.nuts0_of("TR") == "TR"
    assert G.nuts0_of("VN") is None and G.nuts0_of("HK") is None and G.nuts0_of(None) is None
    assert G.iso_of_nuts0("EL") == "GR" and G.iso_of_nuts0("UK") == "GB" and G.iso_of_nuts0("FR") == "FR"


def test_a_country_is_recognised_in_several_languages():
    for place, iso in (("Germany", "DE"), ("Allemagne", "DE"), ("Deutschland", "DE"), ("Italie", "IT"), ("Spagna", "ES"),
                       ("The Netherlands", "NL"), ("Pays-Bas", "NL"), ("England", "GB"), ("UK", "GB"), ("Suisse", "CH")):
        assert _r(place)["country"] == iso, place


def test_turkey_the_bird_is_not_turkiye_the_country():
    assert _r("a commercial farm keeping turkeys and ducks")["country"] is None
    assert _r("Turkey")["country"] is None
    assert _r("Ankara, Türkiye")["country"] == "TR" and _r("Republic of Turkey")["country"] == "TR"
    # Written as a country, it is one: the 2006 human cases of Dogubeyazit and Van were unresolved.
    assert _r("Dogubeyazit and Van, Turkey")["country"] == "TR" and _r("Dogubayazit and Van, Eastern Turkey")["country"] == "TR"
    assert _r("Azerbaijan, Indonesia, Pakistan and Turkey")["several_countries"] is True
    # Still the bird: first word, a bird word before it, any word after it.
    for place, iso in (("Turkey farms, Minnesota", "US"), ("commercial Turkey farms, Minnesota", "US"), ("Iowa, Turkey Creek", "US"),
                       ("wild Turkey, Ohio", "US")):
        assert _r(place)["country"] == iso, place


def test_a_semicolon_lists_places_and_a_typographic_apostrophe_is_an_apostrophe():
    assert _r("Hong Kong; China")["several_countries"] is True and _r("Hong Kong, China")["country"] == "HK"
    assert _r("Lao People\u2019s Democratic Republic")["country"] == "LA"


def test_the_cities_and_places_named_in_the_hpai_scenarios_resolve():
    for place, iso in (("Baiyin City", "CN"), ("Yangquan", "CN"), ("Nanchong City", "CN"), ("Zhaoqing", "CN"), ("Yangjiang City", "CN"),
                       ("Taipei City", "TW"), ("Kowloon", "HK"), ("Inner Niger Delta", "ML"), ("West Azerbaijan Province", "IR"),
                       ("Jamaica Bay, New York", "US")):
        assert _r(place)["country"] == iso, place
    assert _r("Jordan Valley")["reason"] == "multi_country_area" and _r("Bermuda grass pastures")["country"] is None


def test_two_countries_are_several_countries_not_a_guess():
    r = _r("Italy and France")
    assert r["several_countries"] is True and r["country"] is None and r["method"] == "unresolved"
    assert _r("Germany, Italy")["several_countries"] is True


def test_a_word_only_matches_whole_words():
    assert _r("Chadwick")["country"] is None and _r("Marseille")["country"] is None
    assert _r("Poland")["country"] == "PL" and _r("Polandia")["country"] is None


# ── Every country, its abbreviations, the places inside one ─────────────────
#: ISO 3166-1, written out here apart from the table it checks: every code and its common English
#: name. Three have a safer name here; their usual one is pinned as unresolved further down.
ISO_3166 = {
    "AD": "Andorra", "AE": "United Arab Emirates", "AF": "Afghanistan", "AG": "Antigua and Barbuda", "AI": "Anguilla",
    "AL": "Albania", "AM": "Armenia", "AO": "Angola", "AQ": "Antarctica", "AR": "Argentina", "AS": "American Samoa",
    "AT": "Austria", "AU": "Australia", "AW": "Aruba", "AX": "Aland Islands", "AZ": "Azerbaijan",
    "BA": "Bosnia and Herzegovina", "BB": "Barbados", "BD": "Bangladesh", "BE": "Belgium", "BF": "Burkina Faso",
    "BG": "Bulgaria", "BH": "Bahrain", "BI": "Burundi", "BJ": "Benin", "BL": "St Barts", "BM": "Bermuda", "BN": "Brunei",
    "BO": "Bolivia", "BQ": "Caribbean Netherlands", "BR": "Brazil", "BS": "Bahamas", "BT": "Bhutan", "BV": "Bouvet Island",
    "BW": "Botswana", "BY": "Belarus", "BZ": "Belize", "CA": "Canada", "CC": "Cocos (Keeling) Islands",
    "CD": "Democratic Republic of the Congo", "CF": "Central African Republic", "CG": "Republic of the Congo",
    "CH": "Switzerland", "CI": "Côte d'Ivoire", "CK": "Cook Islands", "CL": "Chile", "CM": "Cameroon", "CN": "China",
    "CO": "Colombia", "CR": "Costa Rica", "CU": "Cuba", "CV": "Cabo Verde", "CW": "Curaçao", "CX": "Christmas Island",
    "CY": "Cyprus", "CZ": "Czechia", "DE": "Germany", "DJ": "Djibouti", "DK": "Denmark", "DM": "Dominica",
    "DO": "Dominican Republic", "DZ": "Algeria", "EC": "Ecuador", "EE": "Estonia", "EG": "Egypt", "EH": "Western Sahara",
    "ER": "Eritrea", "ES": "Spain", "ET": "Ethiopia", "FI": "Finland", "FJ": "Fiji", "FK": "Falkland Islands",
    "FM": "Federated States of Micronesia", "FO": "Faroe Islands", "FR": "France", "GA": "Gabon", "GB": "United Kingdom",
    "GD": "Grenada", "GE": "Georgia", "GF": "French Guiana", "GG": "Guernsey", "GH": "Ghana", "GI": "Gibraltar",
    "GL": "Greenland", "GM": "Gambia", "GN": "Guinea", "GP": "Guadeloupe", "GQ": "Equatorial Guinea", "GR": "Greece",
    "GS": "South Georgia and the South Sandwich Islands", "GT": "Guatemala", "GU": "Guam", "GW": "Guinea-Bissau",
    "GY": "Guyana", "HK": "Hong Kong", "HM": "Heard Island and McDonald Islands", "HN": "Honduras", "HR": "Croatia",
    "HT": "Haiti", "HU": "Hungary", "ID": "Indonesia", "IE": "Ireland", "IL": "Israel", "IM": "Isle of Man", "IN": "India",
    "IO": "British Indian Ocean Territory", "IQ": "Iraq", "IR": "Iran", "IS": "Iceland", "IT": "Italy", "JE": "Jersey",
    "JM": "Jamaica", "JO": "Jordan", "JP": "Japan", "KE": "Kenya", "KG": "Kyrgyzstan", "KH": "Cambodia", "KI": "Kiribati",
    "KM": "Comoros", "KN": "Saint Kitts and Nevis", "KP": "North Korea", "KR": "South Korea", "KW": "Kuwait",
    "KY": "Cayman Islands", "KZ": "Kazakhstan", "LA": "Laos", "LB": "Lebanon", "LC": "Saint Lucia", "LI": "Liechtenstein",
    "LK": "Sri Lanka", "LR": "Liberia", "LS": "Lesotho", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia",
    "LY": "Libya", "MA": "Morocco", "MC": "Monaco", "MD": "Moldova", "ME": "Montenegro",
    "MF": "Saint Martin (French part)", "MG": "Madagascar", "MH": "Marshall Islands", "MK": "North Macedonia",
    "ML": "Mali", "MM": "Myanmar", "MN": "Mongolia", "MO": "Macao", "MP": "Northern Mariana Islands", "MQ": "Martinique",
    "MR": "Mauritania", "MS": "Montserrat", "MT": "Malta", "MU": "Mauritius", "MV": "Maldives", "MW": "Malawi",
    "MX": "Mexico", "MY": "Malaysia", "MZ": "Mozambique", "NA": "Namibia", "NC": "New Caledonia", "NE": "Niger",
    "NF": "Norfolk Island", "NG": "Nigeria", "NI": "Nicaragua", "NL": "Netherlands", "NO": "Norway", "NP": "Nepal",
    "NR": "Nauru", "NU": "Niue", "NZ": "New Zealand", "OM": "Oman", "PA": "Panama", "PE": "Peru", "PF": "French Polynesia",
    "PG": "Papua New Guinea", "PH": "Philippines", "PK": "Pakistan", "PL": "Poland", "PM": "Saint Pierre and Miquelon",
    "PN": "Pitcairn", "PR": "Puerto Rico", "PS": "Palestine", "PT": "Portugal", "PW": "Palau", "PY": "Paraguay",
    "QA": "Qatar", "RE": "Réunion", "RO": "Romania", "RS": "Serbia", "RU": "Russia", "RW": "Rwanda", "SA": "Saudi Arabia",
    "SB": "Solomon Islands", "SC": "Seychelles", "SD": "Sudan", "SE": "Sweden", "SG": "Singapore", "SH": "Saint Helena",
    "SI": "Slovenia", "SJ": "Svalbard and Jan Mayen", "SK": "Slovakia", "SL": "Sierra Leone", "SM": "San Marino",
    "SN": "Senegal", "SO": "Somalia", "SR": "Suriname", "SS": "South Sudan", "ST": "Sao Tome and Principe",
    "SV": "El Salvador", "SX": "Sint Maarten", "SY": "Syria", "SZ": "Eswatini", "TC": "Turks and Caicos Islands",
    "TD": "Chad", "TF": "French Southern Territories", "TG": "Togo", "TH": "Thailand", "TJ": "Tajikistan", "TK": "Tokelau",
    "TL": "Timor-Leste", "TM": "Turkmenistan", "TN": "Tunisia", "TO": "Tonga", "TR": "Türkiye",
    "TT": "Trinidad and Tobago", "TV": "Tuvalu", "TW": "Taiwan", "TZ": "Tanzania", "UA": "Ukraine", "UG": "Uganda",
    "UM": "United States Minor Outlying Islands", "US": "United States", "UY": "Uruguay", "UZ": "Uzbekistan",
    "VA": "Vatican City", "VC": "Saint Vincent and the Grenadines", "VE": "Venezuela", "VG": "British Virgin Islands",
    "VI": "US Virgin Islands", "VN": "Vietnam", "VU": "Vanuatu", "WF": "Wallis and Futuna", "WS": "Samoa", "YE": "Yemen",
    "YT": "Mayotte", "ZA": "South Africa", "ZM": "Zambia", "ZW": "Zimbabwe",
}
#: The six NUTS counts inside a member state: France's overseas departments and the Aland Islands.
COUNTED_IN = {"GP": "FR", "MQ": "FR", "GF": "FR", "RE": "FR", "YT": "FR", "AX": "FI"}


def test_every_iso_3166_country_resolves_by_its_common_english_name():
    assert len(ISO_3166) == 249
    assert set(G.COUNTRY_NAME) | set(COUNTED_IN) == set(ISO_3166) | {"XK"}          # and Kosovo, the EU's XK
    assert {iso: parent for iso, (parent, _n) in G._COUNTED_IN.items()} == COUNTED_IN
    for iso, name in ISO_3166.items():
        assert _r(name)["country"] == COUNTED_IN.get(iso, iso), name
    assert _r("Reunion")["nuts0"] == "FR" and _r("Kosovo")["country"] == "XK"


def test_names_kept_out_on_purpose_resolve_to_no_country():
    # Turkey is a bird, Congo alone is two countries, Saint-Barthelemy and Saint-Martin are also villages in
    # France, and Island, Iceland in German, is any island.
    for place in ("Turkey", "Congo", "Saint-Barthélemy", "Saint-Martin-de-Seignanx", "Island"):
        assert _r(place)["country"] is None, place
    assert _r("Vancouver Island, Canada")["country"] == "CA" and _r("Long Island, New York")["country"] == "US"


def test_the_usual_abbreviations_and_other_names():
    for place, iso in (("U.S.", "US"), ("US", "US"), ("USA", "US"), ("U.S.A.", "US"), ("United States of America", "US"),
                       ("UK", "GB"), ("U.K.", "GB"), ("UAE", "AE"), ("U.A.E.", "AE"), ("PRC", "CN"), ("P.R. China", "CN"),
                       ("People's Republic of China", "CN"), ("DRC", "CD"), ("DR Congo", "CD"), ("DPRK", "KP"), ("ROK", "KR"),
                       ("Korea", "KR"), ("KSA", "SA"), ("PNG", "PG"), ("NZ", "NZ"), ("HKSAR", "HK"), ("Hong Kong SAR", "HK"),
                       ("Macau", "MO"), ("Lao PDR", "LA"), ("BiH", "BA"), ("Ivory Coast", "CI"), ("Burma", "MM"),
                       ("East Timor", "TL"), ("Swaziland", "SZ"), ("Cape Verde", "CV"), ("Viet Nam", "VN"),
                       ("Russian Federation", "RU"), ("Taiwan, Province of China", "TW"), ("Korea, Republic of", "KR"),
                       ("Korea, Democratic People's Republic of", "KP"), ("Republic of Guinea-Bissau", "GW"),
                       ("Turks & Caicos", "TC"), ("Wallis & Futuna", "WF")):
        assert _r(place)["country"] == iso, place


def test_the_places_production_left_unresolved_now_resolve():
    # Scenario HPAI_last: 578 of 1,013 extracted articles resolved, and these places were among the rest.
    for place, iso in (("Hong Kong", "HK"), ("Taiwan", "TW"), ("United Arab Emirates", "AE"), ("U.S.", "US"), ("US", "US"),
                       ("Cuba", "CU"), ("Peru", "PE"), ("Mongolia", "MN"), ("Guangzhou", "CN"), ("Ningxia", "CN"),
                       ("California", "US"), ("Iowa", "US"), ("Colorado", "US"), ("Alaska", "US"), ("Macao", "MO")):
        r = _r(place)
        assert (r["country"], r["method"], r["reason"], r["nuts0"], r["level"]) == (iso, "country", None, None, None), place


def test_an_abbreviation_only_matches_a_whole_word():
    # US is not inside Belarus, Cyprus, Russia or Australia, UK not inside Ukraine.
    for place, iso in (("Belarus", "BY"), ("Cyprus", "CY"), ("Mauritius", "MU"), ("Russia", "RU"), ("Australia", "AU"),
                       ("Ukraine", "UA"), ("Minsk, Belarus", "BY"), ("Iowa, US", "US"), ("Busan, Korea", "KR")):
        assert _r(place)["country"] == iso, place
    for place in ("Columbus", "USDA field station", "avian influenza virus surveillance", "Uskudar", "Brus", "Ukhta"):
        assert _r(place)["country"] is None, place
    assert _r("U.S.-Mexico border")["several_countries"] is True


US_STATES = ("Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida",
             "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
             "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
             "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
             "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
             "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming")
CA_PROVINCES = ("Alberta", "British Columbia", "Manitoba", "New Brunswick", "Newfoundland and Labrador", "Nova Scotia", "Ontario",
                "Prince Edward Island", "Québec", "Saskatchewan", "Northwest Territories", "Nunavut", "Yukon")
CN_PROVINCES = ("Anhui", "Beijing", "Chongqing", "Fujian", "Gansu", "Guangdong", "Guangxi", "Guizhou", "Hainan", "Hebei",
                "Heilongjiang", "Henan", "Hubei", "Hunan", "Inner Mongolia", "Jiangsu", "Jiangxi", "Jilin", "Liaoning", "Ningxia",
                "Qinghai", "Shaanxi", "Shandong", "Shanghai", "Shanxi", "Sichuan", "Tianjin", "Tibet", "Xinjiang", "Yunnan",
                "Zhejiang", "Guangzhou", "Shenzhen", "Wuhan", "Nanjing", "Hangzhou", "Xi'an", "Urumqi", "Kunming", "Changsha")


def test_a_state_or_a_province_is_its_country():
    assert len(US_STATES) == 50
    for state in US_STATES:
        assert _r(state)["country"] == ("GE" if state == "Georgia" else "US"), state      # Georgia alone: see below
        assert _r(f"{state}, USA")["country"] == "US", state
    for province in CA_PROVINCES:
        assert _r(province)["country"] == "CA" and _r(f"{province}, Canada")["country"] == "CA", province
    for province in CN_PROVINCES:
        assert _r(province)["country"] == "CN" and _r(f"{province}, China")["country"] == "CN", province
    r = _r("Guangzhou, Guangdong, China")
    assert (r["country"], r["method"], r["nuts1"]) == ("CN", "country", None)                # a country, never a NUTS code


def test_a_longer_name_takes_its_words():
    # The shorter name inside a longer one is not read: no Mexico in New Mexico, no Ireland in Northern Ireland.
    for place, iso in (("New Mexico", "US"), ("New Jersey", "US"), ("New England", "US"), ("Inner Mongolia, China", "CN"),
                       ("Belfast, Northern Ireland", "GB"), ("New South Wales, Australia", "AU"), ("Papua New Guinea", "PG"),
                       ("Guinea-Bissau", "GW"), ("Equatorial Guinea", "GQ"), ("South Sudan", "SS"), ("American Samoa", "AS"),
                       ("Democratic Republic of the Congo", "CD"), ("Democratic People's Republic of Korea", "KP"),
                       ("Baja California, Mexico", "MX"), ("Rhode Island", "US"), ("British Columbia", "CA"),
                       ("Yukon-Kuskokwim Delta, Alaska", "US"), ("Central Macedonia, Greece", "GR"), ("Niger Delta", "NG")):
        r = _r(place)
        assert (r["country"], r["several_countries"]) == (iso, False), place


def test_georgia_is_the_country_unless_the_place_or_the_article_names_the_us():
    for place, hint, iso in (("Georgia", None, "GE"), ("Tbilisi, Georgia", None, "GE"), ("Georgia", "DE", "GE"),
                             ("Atlanta, Georgia, USA", None, "US"), ("Georgia, US", None, "US"), ("Georgia and Alabama", None, "US"),
                             ("Georgia", "US", "US"), ("Georgia, Caucasus", "US", "GE"), ("South Georgia", None, "GS"),
                             ("Strait of Georgia, British Columbia", None, "CA"), ("Montana", None, "US"),
                             ("Montana, Bulgaria", None, "BG"), ("Florida, Uruguay", None, "UY")):
        assert _r(place, hint)["country"] == iso, (place, hint)
    # Named with another country, the country stands: two countries, not a state.
    assert _r("Georgia and Armenia")["several_countries"] is True


def test_a_territory_written_with_its_country_is_the_territory():
    for place, iso in (("Hong Kong, China", "HK"), ("Hong Kong SAR, People's Republic of China", "HK"), ("Kowloon, Hong Kong, PRC", "HK"),
                       ("Macau, China", "MO"), ("Taiwan, China", "TW"), ("San Juan, Puerto Rico, USA", "PR"),
                       ("Nuuk, Greenland, Denmark", "GL"), ("Bermuda (UK)", "BM")):
        assert _r(place)["country"] == iso, place
    # A list, or another place in between, keeps them two.
    for place in ("Hong Kong and mainland China", "mainland China, Hong Kong", "China and Hong Kong", "Hong Kong, Guangzhou, China",
                  "Guangdong and Hong Kong", "Taiwan and China", "Puerto Rico and the United States", "Hong Kong & China"):
        assert _r(place)["several_countries"] is True, place


def test_a_continent_or_a_region_of_several_countries_stays_unresolved_with_its_reason():
    for place in ("Europe", "Southeast Asia", "global", "worldwide", "Sub-Saharan Africa", "Latin America", "European Union",
                  "multiple countries", "Middle East", "Gulf of Mexico", "the Wadden Sea"):
        r = _r(place)
        assert (r["country"], r["method"], r["reason"], r["several_countries"]) == (None, "unresolved", "multi_country_area", False), place
        assert _r(place, "DE")["country"] is None, place                     # never the article's own country
    # Named with a country, the country stands, and a NUTS region still resolves.
    assert _r("Netherlands, Europe")["country"] == "NL" and _r("Gulf of Mexico, Texas")["country"] == "US"
    assert _r("Lombardy, southern Europe")["nuts2"] == "ITC4"


def test_every_unresolved_place_says_why():
    for place, hint, reason in (("", None, "no_place"), (None, None, "no_place"), ("Not reported", None, "no_place"),
                                ("N/A", None, "no_place"), ("Italy and France", None, "several_countries"),
                                ("Europe", None, "multi_country_area"), ("somewhere rural", None, "not_recognised"),
                                ("Germany", None, None), ("", "DE", None), ("somewhere rural", "FR", None)):
        assert _r(place, hint)["reason"] == reason, (place, hint)
    assert _r("Not reported", "DE")["method"] == "article_country"                         # as an empty place would
    assert G.UNRESOLVED_REASONS == ("no_place", "several_countries", "multi_country_area", "not_recognised")


def test_a_bird_a_breed_or_a_lab_animal_is_not_a_place():
    for place, iso in (("a guinea fowl farm, Nigeria", "NG"), ("Canada geese, Ohio", "US"), ("Jersey cows, Texas", "US"),
                       ("Rhode Island Red hens, Kenya", "KE"), ("Maine-et-Loire, France", "FR"), ("Sierra Nevada, Spain", "ES"),
                       ("Sierra Nevada, California", "US"), ("Reunion, France", "FR")):
        assert _r(place)["country"] == iso, place
    for place in ("guinea pigs, laboratory", "a guinea fowl farm", "Canada geese", "Saint-Barthélemy-d'Anjou"):
        assert _r(place)["country"] is None, place


def test_no_name_is_claimed_by_two_tables():
    assert G._CLAIMED_TWICE == []


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
    idx = G.build_index(G.parse_nuts_csv(GISCO))
    r = G.resolve_location("Sigmaringen, Germany", None, idx)
    assert (r["nuts1"], r["nuts2"], r["nuts3"], r["level"]) == ("DE1", "DE13", "DE139", 3)
    assert G.resolve_location("Sigmaringen", None, BUILTIN)["country"] is None            # the built-in list does not know it
    assert idx.source == "loaded" and G.build_index([]).source == "builtin"


# ── The official list, loaded: read as papers write places, never wider than they do ──
#: Rows in GISCO's shape, names as Eurostat writes them: in the country's language, bilingual with
#: a slash, tagged with a country in brackets, prefixed in Belgium, suffixed for German districts.
OFFICIAL = G.build_index(G.parse_nuts_csv(
    "NUTS_ID,LEVL_CODE,CNTR_CODE,NAME_LATN,NUTS_NAME\n"
    "DE1,1,DE,Baden-Württemberg,Baden-Württemberg\nDE14,2,DE,Tübingen,Tübingen\nDE149,3,DE,Sigmaringen,Sigmaringen\n"
    "DE2,1,DE,Bayern,Bayern\nDE21,2,DE,Oberbayern,Oberbayern\n"
    'DE212,3,DE,"München, Kreisfreie Stadt","München, Kreisfreie Stadt"\nDE21H,3,DE,"München, Landkreis","München, Landkreis"\n'
    "DE9,1,DE,Niedersachsen,Niedersachsen\nDE94,2,DE,Weser-Ems,Weser-Ems\nDE948,3,DE,Cloppenburg,Cloppenburg\n"
    "DE94A,3,DE,Friesland (DE),Friesland (DE)\n"
    "ITH3,2,IT,Veneto,Veneto\nITH31,3,IT,Verona,Verona\nITC4,2,IT,Lombardia,Lombardia\n"
    "IE0,1,IE,Ireland,Ireland\nIE04,2,IE,Northern and Western,Northern and Western\nIE041,3,IE,Border,Border\nIE042,3,IE,West,West\n"
    "ES3,1,ES,Comunidad de Madrid,Comunidad de Madrid\nES300,3,ES,Madrid,Madrid\nES4,1,ES,Centro (ES),Centro (ES)\n"
    "ES523,3,ES,Valencia/València,Valencia/València\n"
    "NL12,2,NL,Friesland (NL),Friesland (NL)\nNL42,2,NL,Limburg (NL),Limburg (NL)\n"
    "BE21,2,BE,Prov. Antwerpen,Prov. Antwerpen\nBE211,3,BE,Arr. Antwerpen,Arr. Antwerpen\nBE22,2,BE,Prov. Limburg (BE),Prov. Limburg (BE)\n"
    "FRJ25,3,FR,Lot,Lot\nFRI14,3,FR,Lot-et-Garonne,Lot-et-Garonne\n"
    "MT0,1,MT,Malta,Malta\nMT00,2,MT,Malta,Malta\nMT001,3,MT,Malta,Malta\nRO42,2,RO,Vest,Vest\n"))


def _o(place, hint=None):
    return G.resolve_location(place, hint, OFFICIAL)


def test_the_english_names_stay_beside_the_official_ones():
    # Eurostat writes Bayern and Lombardia, papers Bavaria and Lombardy: loading the list used to drop the second.
    for place, level, code in (("Bavaria, Germany", 1, "DE2"), ("Bavaria", 1, "DE2"), ("Lower Saxony", 1, "DE9"),
                               ("Lombardy, northern Italy", 2, "ITC4"), ("Tuscany", 2, "ITI1"), ("Bayern", 1, "DE2")):
        r = _o(place)
        assert (r[f"nuts{level}"], r["method"]) == (code, "region"), place
    assert OFFICIAL.name_of["DE1"] == "Baden-Württemberg" and OFFICIAL.name_of["ITI1"] == "Toscana"   # the official name is shown


def test_a_district_resolves_inside_its_state_and_a_shared_name_to_what_it_shares():
    r = _o("Sigmaringen, Baden-Württemberg, Germany")
    assert (r["nuts1"], r["nuts2"], r["nuts3"], r["level"]) == ("DE1", "DE14", "DE149", 3)
    assert _o("Cloppenburg, Lower Saxony, Germany")["nuts3"] == "DE948"
    # The town and the district of Munich share a name: Upper Bavaria, not a guess between them.
    r = _o("München, Germany")
    assert (r["nuts2"], r["nuts3"], r["level"]) == ("DE21", None, 2)
    # Two regions of one country, no country named: the country; of two countries: several.
    r = _r("Bavaria and Saxony")
    assert (r["country"], r["nuts1"], r["level"], r["method"]) == ("DE", None, 0, "region")
    assert _r("Bavaria and Lombardy")["several_countries"] is True


def test_an_official_name_is_read_only_inside_a_country_the_place_or_the_article_names():
    # A district's name is often a word or a town elsewhere: Lot, Verona (Wisconsin).
    for place in ("Sigmaringen", "Lot", "a lot of backyard flocks", "Verona"):
        assert _o(place)["country"] is None, place
    assert _o("Sigmaringen", "DE")["nuts3"] == "DE149" and _o("Verona", "IT")["nuts3"] == "ITH31"
    assert _o("Lot, France")["nuts3"] == "FRJ25" and _o("Lot-et-Garonne, France")["nuts3"] == "FRI14"
    assert _o("Verona, Wisconsin, USA")["country"] == "US"


def test_a_compass_word_or_a_countrys_name_is_not_a_region():
    # Ireland's West and Border, Spain's Centro, Romania's Vest: a direction, not a place.
    for place, hint in (("West Africa", None), ("West Africa", "IE"), ("a border area", None)):
        assert _o(place, hint)["nuts0"] is None, (place, hint)
    for place, iso in (("West of Ireland", "IE"), ("border area, Ireland", "IE"), ("Vest, Romania", "RO")):
        r = _o(place)
        assert (r["country"], r["level"], r["nuts1"]) == (iso, 0, None), place
    assert _o("Centro de Investigacion en Sanidad Animal, Madrid, Spain")["nuts3"] == "ES300"
    # Malta is also the name of one of Malta's two NUTS 3: the country, not the island.
    r = _o("Malta")
    assert (r["country"], r["level"]) == ("MT", 0)


def test_bilingual_tagged_and_prefixed_official_names_match_as_papers_write_them():
    for place, level, code in (("Valencia, Spain", 3, "ES523"), ("València, Spain", 3, "ES523"),
                               ("Limburg, the Netherlands", 2, "NL42"), ("Limburg, Belgium", 2, "BE22"),
                               ("Friesland, the Netherlands", 2, "NL12"), ("Friesland, Germany", 3, "DE94A"),
                               ("Antwerpen, Belgium", 3, "BE211")):
        assert _o(place)[f"nuts{level}"] == code, place


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
    assert g["unresolved"] == [{"location": "nowhere in particular", "n": 1, "reason": "not_recognised"}]
    assert c.get("/user-scenarios/nope/extraction/geography").status_code == 404


def test_the_geography_says_why_a_place_does_not_resolve(nuts_clean):
    c = _client()
    # "Europe" stays unresolved even though the article's own country is Germany.
    _set_places(nuts_clean, {9701: ("Hong Kong, China", None), 9702: ("Europe", "DE"), 9703: ("", None)})
    g = c.get(f"/user-scenarios/{SID}/extraction/geography").json()
    assert (g["n_papers"], g["n_resolved"], g["n_unresolved"], g["n_several_countries"]) == (3, 1, 2, 0)
    assert [(x["iso2"], x["name"], x["nuts0"], x["n_papers"]) for x in g["countries"]] == [("HK", "Hong Kong", None, 1)]
    assert g["unresolved_by_reason"] == {"no_place": 1, "several_countries": 0, "multi_country_area": 1, "not_recognised": 0}
    assert g["unresolved"] == [{"location": "(no place stated)", "n": 1, "reason": "no_place"},
                               {"location": "Europe", "n": 1, "reason": "multi_country_area"}]
    md = c.get(f"/user-scenarios/{SID}/extraction/report", params={"format": "md"}).text
    assert ("2 articles do not resolve to a single country: no place stated 1, several countries named 0, a continent or a "
            "region spanning several countries 1, a place the list does not know 0.") in md


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
