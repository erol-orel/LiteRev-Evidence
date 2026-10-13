"""Where the evidence comes from: a study's place, resolved to a country and a NUTS region.

The extraction records a place as the paper words it ("Sigmaringen, Baden-Wuerttemberg,
Germany"). To combine the literature with official statistics by area (Eurostat, national
databases), that has to become a country and, where it can, a NUTS code (the EU's hierarchy of
territorial units: NUTS 0 a country, 1 a major region, 2 a basic region, 3 a small one; the codes
nest, so DE139 sits in DE13, DE1 and DE).

What this does and does not know, said plainly:

  - Countries: every entry of ISO 3166-1 (and Kosovo, XK), by its common English name and the
    usual abbreviations (US, U.S., USA, UK, UAE, PRC, DRC...), the NUTS countries and the larger
    others also in French, Italian, German or Spanish, with the NUTS 0 code where there is one
    (Greece is EL and the United Kingdom UK, not GR and GB). Hong Kong, Macao and Taiwan are their
    own codes (HK, MO, TW). France's overseas departments and the Aland Islands count as France
    and Finland, where NUTS counts them. "Turkey" is deliberately NOT recognised: in this
    literature it is a bird ("Türkiye" is).
  - Places inside a country: the fifty US states, Canada's provinces and territories, China's
    provinces and major cities, Australia's states, each to its country only. A name that is a
    place in two countries is taken in the second only when the place says so: "Georgia" is the
    country unless the place, or the article, names the United States.
  - Continents and regions that span several countries ("Europe", "Southeast Asia", "global")
    resolve to no country, with that as the reason, and never take the article's country.
  - Regions: only those this code is sure of, Germany's sixteen Laender (NUTS 1) and Italy's five
    macro-areas (NUTS 1) and twenty-one regions (NUTS 2). A wrong code is worse than a blank, so
    nothing else is guessed.
  - The OFFICIAL list. An administrator loads Eurostat's NUTS attribute table (`/geo/nuts/import`;
    GISCO's NUTS_AT_2024.csv has NUTS_ID, LEVL_CODE and NUTS_NAME) and the whole of Europe resolves
    to NUTS 3, with no code change. Until then the built-in regions are what is used, and the API
    says which. Loaded, its names are read only inside the country the place or the article names
    (a district's name is often a town or a word elsewhere: "Guadalajara", "Lot"), each side of a
    bilingual name and without its tags ("Valencia/València", "Limburg (NL)", "Prov. Antwerpen",
    "München, Landkreis"), and never when they are only compass words ("West" and "Border" are
    Irish regions, "Centro" an Italian, Spanish and Portuguese one) or a country's name ("Malta"
    is also one of Malta's two NUTS 3). The built-in names stay beside it, English ones included:
    Eurostat writes "Bayern" and "Lombardia", papers "Bavaria" and "Lombardy".

Resolution is deterministic and conservative, and uses a static table, no network. A place name
must match whole words, and the longest name found takes its words ("New Mexico" is not Mexico,
"Northern Ireland" not Ireland, a guinea fowl not Guinea); of the regions a place names, the deepest
wins when it lies inside the others ("Sigmaringen, Baden-Württemberg"), and only what they share is
kept when it does not ("Bavaria and Saxony" is Germany); a name that exists in several countries is
taken only in the country the paper names (or the article's own country); a place that names two
countries is "several countries", not a guess, unless it is a territory written with its country
("Hong Kong, China"); and what cannot be resolved is counted, listed and given its reason, so that
it can be fixed.
"""
from __future__ import annotations

import csv
import io
import re
import unicodedata
from collections.abc import Iterable
from typing import Any, NamedTuple

from fastapi import Depends, HTTPException, Query, Request
from sqlalchemy import text

from .core import app, engine, logger, require_api_key
from .scenario_store import (_get_scenario_threshold, _get_user_scenario_or_404,
                             relevant_gate_sql)
from .schema_boot import _exec_ddl_isolated

_GEO_DDL = ["""CREATE TABLE IF NOT EXISTS geo_nuts (
            code TEXT PRIMARY KEY, level INTEGER NOT NULL, country TEXT NOT NULL, name TEXT NOT NULL,
            alt_name TEXT, source TEXT NOT NULL DEFAULT 'eurostat')"""]
try:
    _exec_ddl_isolated(_GEO_DDL, "_ensure_geo_nuts")
except Exception as _e:                                       # noqa: BLE001 - never blocks startup
    logger.warning(f"_ensure_geo_nuts: {_e}")


def fold(s: Any) -> str:
    """ASCII, lower case, "&" and ";" as "and" (they join a list: "Hong Kong; China" is two places,
    "Hong Kong, China" one), anything else that is not a letter or a digit as one space. A
    typographic apostrophe separates like a plain one ("Lao People’s Democratic Republic"
    folds as "Lao People's ..." does, where it was read as "peoples")."""
    t = re.sub("[\u2018\u2019\u02bc]", "'", str(s or ""))
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    t = t.replace("&", " and ").replace(";", " and ")
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Countries: ISO 3166 alpha-2 -> names; the NUTS 0 code differs for two of them
# ─────────────────────────────────────────────────────────────────────────────
#: (iso2, english name, other names). NUTS countries first; the rest are counted but have no NUTS.
#: The English name is the one shown. Every ISO 3166-1 entry is here (tests/test_geography.py checks
#: the list against the standard) but the six that NUTS counts inside France or Finland, see
#: _COUNTED_IN. A name that is also a word, a breed or another place is left out or guarded below:
#: "Turkey" (a bird), "Island" (German for Iceland, and any island), "Congo" alone (two countries),
#: "Saint-Martin" or "Saint-Barthelemy" alone (villages in France).
_COUNTRIES = [
    ("AT", "Austria", ("Osterreich", "Autriche", "Austria")), ("BE", "Belgium", ("Belgique", "Belgio", "Belgien", "Belgie")),
    ("BG", "Bulgaria", ("Bulgarie", "Bulgarien")), ("HR", "Croatia", ("Croatie", "Croazia", "Kroatien", "Hrvatska")),
    ("CY", "Cyprus", ("Chypre", "Cipro", "Zypern")), ("CZ", "Czechia", ("Czech Republic", "Czech", "Tchequie", "Cechia", "Tschechien")),
    ("DK", "Denmark", ("Danemark", "Danimarca", "Danemarca", "Danmark", "Dinamarca")), ("EE", "Estonia", ("Estonie", "Estland")),
    ("FI", "Finland", ("Finlande", "Finlandia", "Finnland", "Suomi")), ("FR", "France", ("Francia", "Frankreich")),
    ("DE", "Germany", ("Allemagne", "Germania", "Deutschland", "Alemania")), ("GR", "Greece", ("Grece", "Grecia", "Griechenland", "Hellas", "Hellenic Republic")),
    ("HU", "Hungary", ("Hongrie", "Ungheria", "Ungarn", "Hungria")), ("IE", "Ireland", ("Republic of Ireland", "Eire", "Irlande", "Irlanda", "Irland")),
    ("IT", "Italy", ("Italie", "Italia", "Italien")), ("LV", "Latvia", ("Lettonie", "Lettonia", "Lettland")),
    ("LT", "Lithuania", ("Lituanie", "Lituania", "Litauen")), ("LU", "Luxembourg", ("Lussemburgo", "Luxemburg")),
    ("MT", "Malta", ("Malte",)), ("NL", "Netherlands", ("The Netherlands", "Holland", "Pays-Bas", "Paesi Bassi", "Niederlande", "Nederland", "Paises Bajos")),
    ("PL", "Poland", ("Pologne", "Polonia", "Polen", "Polska")), ("PT", "Portugal", ()), ("RO", "Romania", ("Roumanie", "Rumanien")),
    ("SK", "Slovakia", ("Slovaquie", "Slovacchia", "Slowakei", "Slovak Republic")), ("SI", "Slovenia", ("Slovenie", "Slovenia", "Slowenien")),
    ("ES", "Spain", ("Espagne", "Spagna", "Spanien", "Espana")), ("SE", "Sweden", ("Suede", "Svezia", "Schweden", "Sverige", "Suecia")),
    ("IS", "Iceland", ("Islande", "Islanda", "Islandia")), ("LI", "Liechtenstein", ()), ("NO", "Norway", ("Norvege", "Norvegia", "Norwegen", "Norge")),
    ("CH", "Switzerland", ("Swiss Confederation", "Suisse", "Svizzera", "Schweiz", "Suiza")),
    ("GB", "United Kingdom", ("UK", "U.K.", "United Kingdom of Great Britain and Northern Ireland", "Great Britain", "Britain", "England", "Scotland", "Wales",
                              "Northern Ireland", "Royaume-Uni", "Regno Unito", "Vereinigtes Konigreich")),
    ("ME", "Montenegro", ()), ("MK", "North Macedonia", ("Macedonia", "FYROM", "Former Yugoslav Republic of Macedonia")),
    ("AL", "Albania", ("Albanie", "Albanien")), ("RS", "Serbia", ("Serbie", "Serbien")),
    ("TR", "Türkiye", ("Turkiye", "Republic of Turkiye", "Republic of Turkey", "Turquie", "Turchia", "Turkei", "Turquia")),
    # Outside NUTS: counted, never given a NUTS code. The Americas.
    ("US", "United States", ("USA", "U.S.A.", "US", "U.S.", "United States of America", "Etats-Unis", "Stati Uniti", "Vereinigte Staaten", "Estados Unidos")),
    ("CA", "Canada", ()), ("MX", "Mexico", ("Mexique", "Messico", "United Mexican States")),
    ("GT", "Guatemala", ()), ("BZ", "Belize", ()), ("SV", "El Salvador", ()), ("HN", "Honduras", ()), ("NI", "Nicaragua", ()),
    ("CR", "Costa Rica", ()), ("PA", "Panama", ()), ("CU", "Cuba", ()), ("HT", "Haiti", ()),
    ("DO", "Dominican Republic", ("Republique dominicaine",)), ("JM", "Jamaica", ("Jamaique",)), ("PR", "Puerto Rico", ("Porto Rico",)),
    ("BS", "Bahamas", ()), ("BB", "Barbados", ("Barbade",)), ("TT", "Trinidad and Tobago", ("Trinidad", "Tobago")), ("GD", "Grenada", ()),
    ("LC", "Saint Lucia", ("St Lucia", "Sainte-Lucie")), ("VC", "Saint Vincent and the Grenadines", ("St Vincent and the Grenadines",)),
    ("KN", "Saint Kitts and Nevis", ("St Kitts and Nevis", "Saint Kitts", "St Kitts")), ("AG", "Antigua and Barbuda", ("Antigua",)),
    ("DM", "Dominica", ("Commonwealth of Dominica",)), ("MS", "Montserrat", ()), ("AI", "Anguilla", ()), ("AW", "Aruba", ()),
    ("CW", "Curaçao", ()), ("SX", "Sint Maarten", ()), ("BQ", "Caribbean Netherlands", ("Bonaire, Sint Eustatius and Saba", "Bonaire", "Sint Eustatius")),
    ("KY", "Cayman Islands", ()), ("TC", "Turks and Caicos Islands", ("Turks and Caicos",)),
    ("VG", "British Virgin Islands", ("Virgin Islands, British",)),
    ("VI", "US Virgin Islands", ("U.S. Virgin Islands", "United States Virgin Islands", "Virgin Islands of the United States", "Virgin Islands, U.S.")),
    ("BM", "Bermuda", ("Bermudes",)), ("GL", "Greenland", ("Groenland", "Kalaallit Nunaat")),
    ("PM", "Saint Pierre and Miquelon", ("St Pierre and Miquelon", "Saint-Pierre-et-Miquelon")),
    ("BL", "Saint Barthélemy", ("Collectivity of Saint Barthelemy", "Saint Barts", "St Barts", "Saint Barths", "St Barths")),
    ("MF", "Saint Martin (French part)", ("Collectivity of Saint Martin",)),
    ("CO", "Colombia", ("Colombie",)), ("VE", "Venezuela", ("Bolivarian Republic of Venezuela",)), ("GY", "Guyana", ()),
    ("SR", "Suriname", ("Surinam",)), ("EC", "Ecuador", ()), ("PE", "Peru", ("Perou",)), ("BO", "Bolivia", ("Plurinational State of Bolivia", "Bolivie")),
    ("BR", "Brazil", ("Bresil", "Brasile")), ("PY", "Paraguay", ()), ("UY", "Uruguay", ()), ("AR", "Argentina", ("Argentine",)), ("CL", "Chile", ()),
    ("FK", "Falkland Islands", ("Falklands", "Malvinas")),
    ("GS", "South Georgia and the South Sandwich Islands", ("South Georgia", "South Sandwich Islands")),
    # Asia
    ("CN", "China", ("PRC", "P.R.C.", "PR China", "P.R. China", "People's Republic of China", "Mainland China", "Chinese mainland",
                     "Chine", "Cina", "Republique populaire de Chine")),
    ("HK", "Hong Kong", ("Hongkong", "HK", "HKSAR", "Hong Kong SAR", "Hong Kong Special Administrative Region")),
    ("MO", "Macao", ("Macau", "Macao SAR", "Macau SAR", "Macao Special Administrative Region")),
    ("TW", "Taiwan", ("Chinese Taipei", "Taiwan, Province of China")),
    ("MN", "Mongolia", ("Mongolie",)), ("JP", "Japan", ("Japon", "Giappone")),
    ("KR", "South Korea", ("Korea", "Republic of Korea", "Korea, Republic of", "ROK", "Coree du Sud", "Coree")),
    ("KP", "North Korea", ("DPRK", "Democratic People's Republic of Korea", "Korea, Democratic People's Republic of", "Coree du Nord")),
    ("VN", "Vietnam", ("Viet Nam",)), ("LA", "Laos", ("Lao PDR", "Lao People's Democratic Republic")), ("KH", "Cambodia", ("Cambodge",)),
    ("TH", "Thailand", ("Thailande",)), ("MM", "Myanmar", ("Burma", "Birmanie")), ("MY", "Malaysia", ("Malaisie",)),
    ("SG", "Singapore", ("Singapour",)), ("ID", "Indonesia", ("Indonesie",)), ("PH", "Philippines", ()),
    ("BN", "Brunei", ("Brunei Darussalam",)), ("TL", "Timor-Leste", ("East Timor",)),
    ("IN", "India", ("Inde",)), ("PK", "Pakistan", ()), ("BD", "Bangladesh", ()), ("NP", "Nepal", ()), ("BT", "Bhutan", ("Bhoutan",)),
    ("LK", "Sri Lanka", ()), ("MV", "Maldives", ()), ("AF", "Afghanistan", ()),
    ("KZ", "Kazakhstan", ()), ("KG", "Kyrgyzstan", ("Kyrgyz Republic", "Kirghizistan")), ("TJ", "Tajikistan", ("Tadjikistan",)),
    ("TM", "Turkmenistan", ()), ("UZ", "Uzbekistan", ("Ouzbekistan",)),
    ("IR", "Iran", ("Islamic Republic of Iran",)), ("IQ", "Iraq", ("Irak",)), ("SY", "Syria", ("Syrian Arab Republic", "Syrie")),
    ("LB", "Lebanon", ("Liban",)), ("JO", "Jordan", ("Jordanie",)), ("IL", "Israel", ()),
    ("PS", "Palestine", ("State of Palestine", "Palestinian Territories", "Occupied Palestinian Territory", "West Bank", "Gaza", "Gaza Strip",
                         "Cisjordanie")),
    ("SA", "Saudi Arabia", ("KSA", "Kingdom of Saudi Arabia", "Arabie saoudite")),
    ("AE", "United Arab Emirates", ("UAE", "U.A.E.", "Emirats arabes unis")),
    ("KW", "Kuwait", ("Koweit",)), ("QA", "Qatar", ()), ("BH", "Bahrain", ("Bahrein",)), ("OM", "Oman", ()), ("YE", "Yemen", ()),
    ("AM", "Armenia", ("Armenie",)), ("AZ", "Azerbaijan", ("Azerbaidjan",)),
    ("GE", "Georgia", ()),                                        # the name is in _HOMONYMS: also a US state
    ("IO", "British Indian Ocean Territory", ("Chagos Archipelago", "Chagos Islands")),
    ("CX", "Christmas Island", ()), ("CC", "Cocos (Keeling) Islands", ("Cocos Islands", "Keeling Islands")),
    # Europe outside NUTS
    ("RU", "Russia", ("Russian Federation", "Russie", "Russland")), ("UA", "Ukraine", ()), ("BY", "Belarus", ("Bielorussie",)),
    ("MD", "Moldova", ("Republic of Moldova", "Moldavie")),
    ("BA", "Bosnia and Herzegovina", ("Bosnia", "Bosnia-Herzegovina", "BiH", "Bosnie-Herzegovine")),
    ("XK", "Kosovo", ()),                                         # not in ISO 3166: XK is the code the EU and Eurostat use
    ("AD", "Andorra", ("Andorre",)), ("MC", "Monaco", ()), ("SM", "San Marino", ()),
    ("VA", "Vatican City", ("Vatican", "Holy See", "Vatican City State")),
    ("GI", "Gibraltar", ()), ("IM", "Isle of Man", ()), ("JE", "Jersey", ()), ("GG", "Guernsey", ()),
    ("FO", "Faroe Islands", ("Faroes", "Faeroe Islands", "Iles Feroe")), ("SJ", "Svalbard and Jan Mayen", ("Svalbard", "Jan Mayen", "Spitsbergen")),
    # Africa
    ("EG", "Egypt", ("Egypte", "Egitto")), ("LY", "Libya", ("Libye",)), ("TN", "Tunisia", ("Tunisie",)), ("DZ", "Algeria", ("Algerie",)),
    ("MA", "Morocco", ("Maroc",)), ("EH", "Western Sahara", ("Sahara occidental",)), ("MR", "Mauritania", ("Mauritanie",)),
    ("SN", "Senegal", ()), ("GM", "Gambia", ("The Gambia", "Gambie")), ("GW", "Guinea-Bissau", ("Republic of Guinea-Bissau", "Guinee-Bissau")),
    ("GN", "Guinea", ("Republic of Guinea", "Guinea-Conakry", "Guinee")), ("SL", "Sierra Leone", ()), ("LR", "Liberia", ()),
    ("CI", "Côte d'Ivoire", ("Ivory Coast",)), ("ML", "Mali", ()), ("BF", "Burkina Faso", ()), ("GH", "Ghana", ()),
    ("TG", "Togo", ()), ("BJ", "Benin", ()), ("NE", "Niger", ()), ("NG", "Nigeria", ()), ("CV", "Cabo Verde", ("Cape Verde",)),
    ("TD", "Chad", ("Tchad",)), ("CM", "Cameroon", ("Cameroun",)), ("CF", "Central African Republic", ("Centrafrique", "Republique centrafricaine")),
    ("GQ", "Equatorial Guinea", ("Guinee equatoriale",)), ("GA", "Gabon", ()), ("ST", "São Tomé and Príncipe", ("Sao Tome",)),
    ("CG", "Republic of the Congo", ("Republic of Congo", "Congo Republic", "Congo-Brazzaville", "Congo, Republic of the", "Republique du Congo")),
    ("CD", "Democratic Republic of the Congo", ("Democratic Republic of Congo", "DR Congo", "D.R. Congo", "DRC", "Congo-Kinshasa", "Zaire",
                                                "Congo, The Democratic Republic of the", "Congo, Democratic Republic of the", "RD Congo", "RDC",
                                                "Republique democratique du Congo")),
    ("AO", "Angola", ()), ("SD", "Sudan", ("Soudan",)), ("SS", "South Sudan", ("Soudan du Sud",)), ("ER", "Eritrea", ("Erythree",)),
    ("ET", "Ethiopia", ("Ethiopie",)), ("DJ", "Djibouti", ()), ("SO", "Somalia", ("Somalie",)), ("KE", "Kenya", ()), ("UG", "Uganda", ("Ouganda",)),
    ("RW", "Rwanda", ()), ("BI", "Burundi", ()), ("TZ", "Tanzania", ("United Republic of Tanzania", "Tanzanie")), ("MW", "Malawi", ()),
    ("ZM", "Zambia", ("Zambie",)), ("ZW", "Zimbabwe", ()), ("MZ", "Mozambique", ()), ("BW", "Botswana", ()), ("NA", "Namibia", ("Namibie",)),
    ("ZA", "South Africa", ("Republic of South Africa", "RSA", "Afrique du Sud", "Sudafrica")), ("LS", "Lesotho", ()), ("SZ", "Eswatini", ("Swaziland",)),
    ("MG", "Madagascar", ()), ("KM", "Comoros", ("Comores",)), ("MU", "Mauritius", ()), ("SC", "Seychelles", ()),
    ("SH", "Saint Helena", ("St Helena", "Saint Helena, Ascension and Tristan da Cunha", "Ascension Island", "Tristan da Cunha")),
    ("TF", "French Southern Territories", ("French Southern and Antarctic Lands", "Terres australes et antarctiques francaises", "Kerguelen",
                                           "Crozet Islands")),
    ("BV", "Bouvet Island", ()), ("HM", "Heard Island and McDonald Islands", ("Heard Island", "McDonald Islands")),
    ("AQ", "Antarctica", ("Antarctic Peninsula", "Antarctique")),
    # Oceania
    ("AU", "Australia", ("Australie",)), ("NZ", "New Zealand", ("NZ", "Aotearoa", "Nouvelle-Zelande")),
    ("PG", "Papua New Guinea", ("PNG", "Papouasie-Nouvelle-Guinee")), ("FJ", "Fiji", ("Fidji",)), ("SB", "Solomon Islands", ()),
    ("VU", "Vanuatu", ("New Hebrides",)), ("NC", "New Caledonia", ("Nouvelle-Caledonie",)),
    ("PF", "French Polynesia", ("Polynesie francaise", "Tahiti")), ("WS", "Samoa", ("Western Samoa",)), ("AS", "American Samoa", ()),
    ("TO", "Tonga", ()), ("TV", "Tuvalu", ()), ("KI", "Kiribati", ()), ("NR", "Nauru", ()), ("MH", "Marshall Islands", ()),
    ("FM", "Federated States of Micronesia", ("Micronesia, Federated States of",)),      # "Micronesia" alone is the region
    ("PW", "Palau", ()), ("GU", "Guam", ()), ("MP", "Northern Mariana Islands", ("Northern Marianas",)),
    ("UM", "United States Minor Outlying Islands", ("Midway Atoll", "Midway Islands", "Wake Island")),
    ("CK", "Cook Islands", ()), ("NU", "Niue", ()), ("TK", "Tokelau", ()), ("PN", "Pitcairn", ("Pitcairn Islands",)),
    ("WF", "Wallis and Futuna", ("Wallis-et-Futuna",)), ("NF", "Norfolk Island", ()),
]
#: ISO 3166-1 entries that NUTS counts inside a member state, France's overseas departments (NUTS
#: FRY) and the Aland Islands (FI2): they resolve to that state, to keep its NUTS code and line up
#: with Eurostat's figures.
_COUNTED_IN: dict[str, tuple[str, tuple[str, ...]]] = {
    "GP": ("FR", ("Guadeloupe",)), "MQ": ("FR", ("Martinique",)), "GF": ("FR", ("French Guiana", "Guyane", "Guyane francaise")),
    "RE": ("FR", ("Reunion", "La Reunion", "Reunion Island")), "YT": ("FR", ("Mayotte",)),
    "AX": ("FI", ("Aland", "Aland Islands", "Ahvenanmaa")),
}
_NUTS_ISO = frozenset({
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
    "PL", "PT", "RO", "SK", "SI", "ES", "SE", "IS", "LI", "NO", "CH", "GB", "ME", "MK", "AL", "RS", "TR"})
#: Where the NUTS 0 code is not the ISO code.
_NUTS0 = {"GR": "EL", "GB": "UK"}
COUNTRY_NAME = {iso: name for iso, name, _a in _COUNTRIES}


def nuts0_of(iso2: str | None) -> str | None:
    if not iso2 or iso2 not in _NUTS_ISO:
        return None
    return _NUTS0.get(iso2, iso2)


def iso_of_nuts0(code: str) -> str:
    return {"EL": "GR", "UK": "GB"}.get(code, code)


# ─────────────────────────────────────────────────────────────────────────────
# Places inside a country, places across several, and names that are not places
# ─────────────────────────────────────────────────────────────────────────────
#: First-level subdivisions whose name is a place in one country only, and China's major cities:
#: each resolves to its country, never to a NUTS code. Left out on purpose: Georgia, Montana,
#: Florida and Maryland, places in other countries too (see _HOMONYMS), and Victoria (Australia),
#: also a city in Canada and a lake in Africa.
_SUBDIVISIONS: dict[str, tuple[str, ...]] = {
    "US": ("Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Hawaii", "Hawai'i",
           "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Massachusetts", "Michigan",
           "Minnesota", "Mississippi", "Missouri", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
           "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
           "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin",
           "Wyoming", "District of Columbia",
           # not states, but holding another country's name (England, Jersey, Yukon)
           "New England", "Jersey City", "Jersey Shore", "Yukon-Kuskokwim Delta", "Yukon Delta", "Yukon Flats", "Jamaica Bay"),
    "CA": ("Alberta", "British Columbia", "Manitoba", "New Brunswick", "Newfoundland and Labrador", "Newfoundland", "Labrador",
           "Nova Scotia", "Ontario", "Prince Edward Island", "Quebec", "Saskatchewan", "Northwest Territories", "Nunavut", "Yukon"),
    # The provinces, autonomous regions and municipalities (Hong Kong, Macao and Taiwan have their own
    # code), then the provincial capitals and the cities the avian influenza literature names most.
    "CN": ("Anhui", "Beijing", "Chongqing", "Fujian", "Gansu", "Guangdong", "Guangxi", "Guizhou", "Hainan", "Hebei", "Heilongjiang",
           "Henan", "Hubei", "Hunan", "Inner Mongolia", "Nei Mongol", "Jiangsu", "Jiangxi", "Jilin", "Liaoning", "Ningxia", "Qinghai",
           "Shaanxi", "Shandong", "Shanghai", "Shanxi", "Sichuan", "Tianjin", "Tibet", "Xizang", "Xinjiang", "Yunnan", "Zhejiang",
           "Guangzhou", "Shenzhen", "Dongguan", "Foshan", "Zhuhai", "Shantou", "Hangzhou", "Ningbo", "Wenzhou", "Huzhou", "Jiaxing",
           "Shaoxing", "Jinhua", "Nanjing", "Suzhou", "Wuxi", "Changzhou", "Yangzhou", "Yancheng", "Wuhan", "Changsha", "Chengdu",
           "Kunming", "Fuzhou", "Xiamen", "Quanzhou", "Jinan", "Qingdao", "Yantai", "Shenyang", "Dalian", "Harbin", "Changchun",
           "Hefei", "Nanchang", "Zhengzhou", "Shijiazhuang", "Taiyuan", "Hohhot", "Xi'an", "Xian", "Lanzhou", "Xining", "Yinchuan",
           "Urumqi", "Lhasa", "Nanning", "Guilin", "Haikou", "Sanya", "Guiyang",
           # named as study sites in the three HPAI scenarios (12 October 2026) and left unresolved
           "Baiyin", "Yangquan", "Nanchong", "Zhaoqing", "Yangjiang"),
    "TW": ("Taipei", "Kaohsiung", "Taichung", "Tainan"), "HK": ("Kowloon",),
    "AU": ("New South Wales", "Queensland", "Tasmania", "Northern Territory", "Australian Capital Territory", "Western Australia",
           "South Australia"),
    # Not subdivisions, but places whose name holds another country's, or a region's, which they
    # would otherwise be read as.
    "FR": ("Maine-et-Loire", "French Antilles", "French West Indies"),
    "MX": ("Baja California", "Baja California Sur", "Gulf of California"),
    "GR": ("Central Macedonia", "Western Macedonia", "West Macedonia", "Eastern Macedonia and Thrace", "East Macedonia and Thrace"),
    "NG": ("Niger Delta", "Niger State", "Benin City"), "ML": ("Inner Niger Delta",),
    "IR": ("West Azerbaijan", "East Azerbaijan"), "BE": ("Luxembourg Province", "Province of Luxembourg"),
    "ID": ("West Timor",), "ZA": ("Lake St Lucia", "Lake Saint Lucia", "St Lucia Estuary"), "GT": ("Antigua Guatemala",),
    "PG": ("New Ireland",),
}
#: A name that is a place in more than one country: the country it is taken in by default, then
#: the others. It is taken in another of them only when the place also names that country or a
#: place in it ("Athens, Georgia, USA", "Montana, Bulgaria"), or, naming nothing else, when that is
#: the article's own country.
_HOMONYMS: dict[str, tuple[str, ...]] = {
    "Georgia": ("GE", "US"), "Georgie": ("GE", "US"),        # the country; the US state
    "Montana": ("US", "BG"),                                 # the US state; a province of Bulgaria
    "Florida": ("US", "UY"),                                 # the US state; a department of Uruguay
    "Maryland": ("US", "LR"),                                # the US state; a county of Liberia
}
#: Continents, regions and seas that span several countries, and the words that say "several
#: countries". A place that names one of them and no country resolves to none, for that reason,
#: and is never given the article's country. Named with a country ("Netherlands, Europe"), the
#: country stands.
_AREAS = (
    "global", "globally", "worldwide", "world wide", "world", "international", "internationally", "multinational",
    "multi-national", "multicountry", "multi-country", "countries", "continents", "intercontinental", "transboundary",
    "cross-border",
    "Europe", "European Union", "EU", "EEA", "European Economic Area", "Eurasia", "Scandinavia", "Fennoscandia", "Baltic",
    "Baltic states", "Baltic Sea", "Balkans", "Iberian Peninsula", "Benelux", "British Isles", "Mediterranean", "Black Sea",
    "North Sea", "Wadden Sea", "Caucasus", "Strait of Gibraltar",
    "Africa", "Sub-Saharan Africa", "Sahel", "Sahara", "Maghreb", "Horn of Africa", "Great Lakes", "Lake Victoria", "Lake Malawi",
    "Lake Chad", "Gulf of Guinea", "Niger River", "River Niger", "Senegal River", "Congo Basin", "Jordan Valley", "Jordan River",
    "Asia", "Southeast Asia", "South-East Asia", "South Asia", "East Asia", "Central Asia", "Western Asia", "Middle East",
    "Near East", "Far East", "Levant", "Arabian Peninsula", "Persian Gulf", "Arabian Gulf", "Gulf states", "Gulf Cooperation Council",
    "GCC", "Indian subcontinent", "Himalaya", "Himalayas", "Indochina", "Indo-China", "Greater Mekong Subregion", "Mekong region",
    "Mekong basin", "ASEAN", "Korean Peninsula", "Greater China", "South China Sea", "East China Sea", "Sea of Japan", "Japan Sea",
    "Gulf of Thailand", "New Guinea", "Borneo",
    "Americas", "North America", "South America", "Latin America", "Central America", "Caribbean", "West Indies", "Antilles",
    "Netherlands Antilles", "Hispaniola", "Gulf of Mexico", "Patagonia", "Amazon", "Amazonia", "Lake Ontario", "Yukon River",
    "Oceania", "Australasia", "Melanesia", "Micronesia", "Polynesia", "Pacific", "Pacific Islands", "Atlantic", "Indian Ocean",
    "Arctic", "Antarctic", "Sub-Antarctic", "Subantarctic", "Southern Ocean",
)
#: Phrases that hold a country's or a state's name but are not a place (a bird, a breed, a lab animal, a disease),
#: or are a place in two countries that only the rest of the place can tell apart: they are read as
#: nothing, and so is the name inside them.
_NOT_PLACES = (
    "Turkey",                                                # a bird, in this literature ("Türkiye" is the country)
    "Guinea fowl", "Guinea fowls", "Guinea pig", "Guinea pigs", "Guinea worm", "Guinea grass", "Bermuda grass", "Canada goose",
    "Canada geese",
    "Jersey cattle", "Jersey cow", "Jersey cows", "Jersey Giant", "Jersey Giants", "Guernsey cattle", "Guernsey cow", "Guernsey cows",
    "Rhode Island Red", "Rhode Island Reds", "Rhode Island White", "Rhode Island Whites", "New Hampshire Red", "New Hampshire Reds",
    "Hong Kong flu", "Hong Kong influenza", "cochon d'Inde", "cochons d'Inde",
    "Sierra Nevada", "Strait of Georgia", "Georgia Strait", "New Britain", "New Holland", "Saint Barthelemy",
)
#: A territory with its own code, followed by the country it belongs to or is listed under ("Hong
#: Kong, China", "Taiwan, Province of China", "San Juan, Puerto Rico, USA"): written so, it is the
#: territory, not two countries. Joined by "and" ("Hong Kong and mainland China"), or with another
#: place between them, they stay two.
_ADDRESS_PARENT = {
    "HK": "CN", "MO": "CN", "TW": "CN",
    "PR": "US", "GU": "US", "VI": "US", "AS": "US", "MP": "US", "UM": "US",
    "GL": "DK", "FO": "DK", "SJ": "NO", "BV": "NO", "AW": "NL", "CW": "NL", "SX": "NL", "BQ": "NL",
    "NC": "FR", "PF": "FR", "WF": "FR", "PM": "FR", "BL": "FR", "MF": "FR", "TF": "FR",
    "BM": "GB", "KY": "GB", "VG": "GB", "AI": "GB", "MS": "GB", "TC": "GB", "FK": "GB", "GI": "GB", "GS": "GB", "SH": "GB",
    "IO": "GB", "PN": "GB", "IM": "GB", "JE": "GB", "GG": "GB",
    "CX": "AU", "CC": "AU", "NF": "AU", "HM": "AU", "CK": "NZ", "NU": "NZ", "TK": "NZ",
}
#: The words that make a list of places rather than an address.
_LIST_WORDS = frozenset({"and", "or", "versus", "vs", "plus", "with", "between", "to", "from", "including", "excluding", "except",
                         "et", "und", "ou", "oder"})
#: A country's name that leaves its territories out: never the address of one.
_MAINLAND = frozenset({"mainland china", "chinese mainland"})

#: folded name -> (kind, iso): kind is "country", "subdivision", "homonym" (see _HOMONYMS), "area"
#: (no country) or "none" (not a place). Built once from the tables above.
_LEXICON: dict[str, tuple[str, str | None]] = {}
#: Names two tables both claim (tests/test_geography.py keeps it empty).
_CLAIMED_TWICE: list[str] = []


def _claim(names: Iterable[str], kind: str, iso: str | None, reserved: frozenset[str] = frozenset()) -> None:
    for n in names:
        key = fold(n)
        if key in reserved:
            continue
        if _LEXICON.setdefault(key, (kind, iso)) != (kind, iso):
            _CLAIMED_TWICE.append(f"{n}: {kind} {iso}, already {_LEXICON[key]}")


_claim(_NOT_PLACES, "none", None)
_claim(_HOMONYMS, "homonym", None)
_RESERVED = frozenset(fold(n) for n in (*_NOT_PLACES, *_HOMONYMS))
for _iso, _name, _alts in _COUNTRIES:
    _claim((_name, *_alts), "country", _iso, _RESERVED)
for _iso, (_parent, _names) in _COUNTED_IN.items():
    _claim(_names, "subdivision", _parent)
for _iso, _names in _SUBDIVISIONS.items():
    _claim(_names, "subdivision", _iso)
_claim(_AREAS, "area", None)
_HOMONYM_ISO = {fold(n): isos for n, isos in _HOMONYMS.items()}
_COUNTED_IN_ISO = {iso: parent for iso, (parent, _n) in _COUNTED_IN.items()}
#: Every name of a country, folded -> ISO code.
_NAME_TO_ISO = {name: iso for name, (kind, iso) in _LEXICON.items() if kind == "country"}
_MAX_WORDS = max(len(name.split()) for name in _LEXICON)


class _Hit(NamedTuple):
    start: int                                    # the first word of the name in the place
    end: int                                      # the word after its last
    kind: str
    iso: str | None
    name: str


def _scan(words: list[str]) -> list[_Hit]:
    """The names in a place (folded, split into words), leftmost then longest. A name found takes
    its words, so the shorter names inside it are not read: New Mexico is not Mexico, Northern
    Ireland not Ireland, a guinea fowl not Guinea."""
    hits: list[_Hit] = []
    i = 0
    while i < len(words):
        for j in range(min(len(words), i + _MAX_WORDS), i, -1):
            name = " ".join(words[i:j])
            entry = _LEXICON.get(name)
            if entry:
                hits.append(_Hit(i, j, entry[0], entry[1], name))
                i = j
                break
        else:
            i += 1
    return hits


#: "Turkey" is the country when the place is written that way: a capital, after a comma, a
#: semicolon, a bracket, "and" or a compass word, and nothing after it but punctuation ("Van,
#: Turkey", "eastern Turkey", "Pakistan and Turkey"). "Turkey" alone, "Turkey farms" or "wild
#: Turkey" stay the bird: the place resolves as if the word were not there.
_TURKEY_THE_COUNTRY = re.compile(r"(?:[,;(]|\band|(?i:\b(?:eastern|western|northern|southern|central|north|south|east|west)))"
                                 r"\s*Turkey\s*(?:[,;.)]|$)")


def _countries_named(folded: str, hint: str | None) -> tuple[set[str], bool]:
    """The countries a place names, and whether it names an area spanning several countries."""
    words = folded.split()
    hits = _scan(words)
    context = {h.iso for h in hits if h.kind in ("country", "subdivision")}
    homonyms = [h for h in hits if h.kind == "homonym"]
    area = any(h.kind == "area" for h in hits)
    taken: list[_Hit] = []
    for h in hits:
        if h.kind == "homonym":
            default, *others = _HOMONYM_ISO[h.name]
            around = context | {_HOMONYM_ISO[o.name][0] for o in homonyms if o is not h}
            iso = next((c for c in others if c in around), None)
            if iso is None and not around and not area and hint in others:
                iso = hint                        # "Georgia" alone, in an American article
            h = h._replace(kind="country", iso=iso or default)
        taken.append(h)
    # "Hong Kong, China": the country after a territory is its address, unless a list says otherwise.
    address = {k + 1 for k, (a, b) in enumerate(zip(taken, taken[1:]))
               if a.kind == b.kind == "country" and _ADDRESS_PARENT.get(a.iso) == b.iso
               and b.name not in _MAINLAND and _LIST_WORDS.isdisjoint(words[a.end:b.start])}
    return {h.iso for k, h in enumerate(taken) if h.kind in ("country", "subdivision") and k not in address}, area


# ─────────────────────────────────────────────────────────────────────────────
# The regions this code is sure of
# ─────────────────────────────────────────────────────────────────────────────
#: code -> (level, country iso2, names). Germany's Laender are NUTS 1; Italy's macro-areas NUTS 1
#: and regions NUTS 2. Nothing else is written from memory.
_BUILTIN: dict[str, tuple[int, str, tuple[str, ...]]] = {
    "DE1": (1, "DE", ("Baden-Wuerttemberg", "Baden-Württemberg")), "DE2": (1, "DE", ("Bayern", "Bavaria")),
    "DE3": (1, "DE", ("Berlin",)), "DE4": (1, "DE", ("Brandenburg",)), "DE5": (1, "DE", ("Bremen",)),
    "DE6": (1, "DE", ("Hamburg",)), "DE7": (1, "DE", ("Hessen", "Hesse")), "DE8": (1, "DE", ("Mecklenburg-Vorpommern", "Mecklenburg-Western Pomerania")),
    "DE9": (1, "DE", ("Niedersachsen", "Lower Saxony")), "DEA": (1, "DE", ("Nordrhein-Westfalen", "North Rhine-Westphalia")),
    "DEB": (1, "DE", ("Rheinland-Pfalz", "Rhineland-Palatinate")), "DEC": (1, "DE", ("Saarland",)),
    "DED": (1, "DE", ("Sachsen", "Saxony")), "DEE": (1, "DE", ("Sachsen-Anhalt", "Saxony-Anhalt")),
    "DEF": (1, "DE", ("Schleswig-Holstein",)), "DEG": (1, "DE", ("Thueringen", "Thüringen", "Thuringia")),
    "ITC": (1, "IT", ("Nord-Ovest", "North-West Italy")), "ITH": (1, "IT", ("Nord-Est", "North-East Italy")),
    "ITI": (1, "IT", ("Centro Italia", "Central Italy")), "ITF": (1, "IT", ("Sud Italia", "Southern Italy")), "ITG": (1, "IT", ("Isole",)),
    "ITC1": (2, "IT", ("Piemonte", "Piedmont")), "ITC2": (2, "IT", ("Valle d'Aosta", "Aosta Valley", "Vallee d'Aoste")),
    "ITC3": (2, "IT", ("Liguria",)), "ITC4": (2, "IT", ("Lombardia", "Lombardy", "Lombardie")),
    "ITH1": (2, "IT", ("Bolzano", "Bozen", "South Tyrol", "Alto Adige", "Sudtirol")), "ITH2": (2, "IT", ("Trento", "Trentino")),
    "ITH3": (2, "IT", ("Veneto",)), "ITH4": (2, "IT", ("Friuli-Venezia Giulia", "Friuli Venezia Giulia", "Friuli")),
    "ITH5": (2, "IT", ("Emilia-Romagna", "Emilia Romagna")), "ITI1": (2, "IT", ("Toscana", "Tuscany", "Toscane")),
    "ITI2": (2, "IT", ("Umbria", "Ombrie")), "ITI3": (2, "IT", ("Marche",)), "ITI4": (2, "IT", ("Lazio", "Latium")),
    "ITF1": (2, "IT", ("Abruzzo", "Abruzzes")), "ITF2": (2, "IT", ("Molise",)), "ITF3": (2, "IT", ("Campania", "Campanie")),
    "ITF4": (2, "IT", ("Puglia", "Apulia", "Pouilles")), "ITF5": (2, "IT", ("Basilicata",)), "ITF6": (2, "IT", ("Calabria", "Calabre")),
    "ITG1": (2, "IT", ("Sicilia", "Sicily", "Sicile")), "ITG2": (2, "IT", ("Sardegna", "Sardinia", "Sardaigne")),
}

#: Words that say where in a country, not which region. An official name made of nothing else is
#: never read: Ireland's "West", "Border" and "Mid-East", "Centro (IT)", Spain's "Este" and "Sur",
#: Romania's "Vest" would otherwise take "West Africa", "a border area" or "Centro de Investigacion".
_COMPASS_WORDS = frozenset({
    "north", "south", "east", "west", "northern", "southern", "eastern", "western", "northeast", "northwest", "southeast",
    "southwest", "central", "centre", "center", "middle", "mid", "midland", "midlands", "border", "upper", "lower", "inner",
    "outer", "coast", "coastal", "and", "of", "the", "region", "island", "islands",
    "nord", "sud", "est", "ouest", "ovest", "centro", "centru", "vest", "norte", "sur", "este", "oeste", "noroeste", "noreste",
    "nordeste", "sudeste", "sudoeste", "isole"})
#: Belgium's prefixes for a province and an arrondissement: "Prov. Antwerpen", "Arr. de Bruxelles-Capitale".
_NUTS_PREFIX = re.compile(r"^(?:prov|arr)\.\s*(?:(?:de|du|van)\s+)?", re.I)


def _name_forms(name: str) -> set[str]:
    """The ways a paper can write an official NUTS name, folded: each side of a bilingual name
    ("Valencia/València"), without a tag in brackets ("Limburg (NL)"), without a Belgian "Prov." or
    "Arr.", and without what follows a comma ("München, Landkreis"). A form that is only compass
    words, or a country's name, is not one: see _COMPASS_WORDS and the module's docstring."""
    forms: set[str] = set()
    for part in {name, *name.split("/")}:
        bare = re.sub(r"\([^)]*\)", " ", part)
        for form in (fold(bare), fold(_NUTS_PREFIX.sub("", bare.split(",")[0].strip()))):
            if form and not set(form.split()) <= _COMPASS_WORDS and form not in _NAME_TO_ISO:
                forms.add(form)
    return forms


class NutsIndex:
    """The NUTS names in use, ready for matching: folded name -> [(code, level, country, curated)].

    `entries` are the curated names, the built-in regions with their English and French ones: read
    wherever the place is, as before any list was loaded. `loaded` are the official list's, read only
    inside the country the place or the article names. The first name of a code is the one shown,
    the official one when there is one."""

    def __init__(self, entries: Iterable[tuple[str, int, str, str]], source: str,
                 loaded: Iterable[tuple[str, int, str, str]] = ()):
        self.source = source
        self.by_name: dict[str, list[tuple[str, int, str, bool]]] = {}
        self.name_of: dict[str, str] = {}
        for curated, rows in ((False, list(loaded)), (True, list(entries))):
            for code, level, country, name in rows:
                self.name_of.setdefault(code, name)
                for form in ({fold(name)} if curated else _name_forms(name)):
                    if (code, level, country, curated) not in self.by_name.setdefault(form, []):
                        self.by_name[form].append((code, level, country, curated))
        self.n = len(self.name_of)
        self.max_words = max((len(n.split()) for n in self.by_name), default=1)


def _builtin_entries() -> list[tuple[str, int, str, str]]:
    return [(code, lvl, c, n) for code, (lvl, c, names) in _BUILTIN.items() for n in names]


def build_index(rows: Iterable[tuple[str, int, str, str, str | None]]) -> NutsIndex:
    """The index for the official rows (code, level, country, name, Latin-script name) as stored,
    with the built-in names beside them; the built-in names alone when there are none. The
    Latin-script name is the one shown: "Attiki" and "Yugozapaden" read where "Αττική" and
    "Югозападен" do not, and for every other country the two are the same."""
    loaded: list[tuple[str, int, str, str]] = []
    for code, level, country, name, alt in rows:
        if alt and alt != name:
            loaded.append((code, level, country, alt))
        loaded.append((code, level, country, name))
    return NutsIndex(_builtin_entries(), "loaded" if loaded else "builtin", loaded)


_cache: dict[str, Any] = {"key": None, "index": None}


def get_nuts_index() -> NutsIndex:
    """The loaded official list when there is one, with the built-in regions as a floor, else
    the built-in regions alone. Rebuilt when the table changes, whichever worker changed it."""
    try:
        with engine.connect() as conn:
            key = tuple(conn.execute(text(
                "SELECT COUNT(*), COALESCE(md5(string_agg(code || name, ',' ORDER BY code)), '') FROM geo_nuts")).one())
            if _cache["key"] == key and _cache["index"] is not None:
                return _cache["index"]
            rows = conn.execute(text("SELECT code, level, country, name, alt_name FROM geo_nuts")).all() if key[0] else []
    except Exception as e:                                         # noqa: BLE001
        logger.warning(f"get_nuts_index: {e}")
        key, rows = ("error", ""), []
    idx = build_index(rows)
    _cache.update({"key": key, "index": idx})
    return idx


def _region_hits(words: list[str], index: NutsIndex, country: str | None) -> list[list[str]]:
    """The NUTS codes each region name in a place can be, leftmost then longest, a name found
    taking its words (Lower Saxony is not Saxony), at the name's deepest level. Inside a known
    country every name counts; with none, only the curated ones. A name several codes share
    gives them all ("Oldenburg": the town DE943 and the district DE94D)."""
    hits: list[list[str]] = []
    i = 0
    while i < len(words):
        for j in range(min(len(words), i + index.max_words), i, -1):
            cands = [(c, lvl) for c, lvl, cc, curated in index.by_name.get(" ".join(words[i:j]), ())
                     if (cc == country if country else curated)]
            if cands:
                deepest = max(lvl for _c, lvl in cands)
                hits.append(sorted({c for c, lvl in cands if lvl == deepest}))
                i = j
                break
        else:
            i += 1
    return hits


def _region_of(hits: list[list[str]]) -> str:
    """The region a place names, from its hits: a code that only places another ("Baden-
    Württemberg" around Sigmaringen, "Lower Saxony" around Cloppenburg) is set aside, and the
    place is in what the others share. "Sigmaringen, Baden-Württemberg" is the district;
    "Oldenburg and Cloppenburg, Lower Saxony" is Weser-Ems (DE94), where both districts lie;
    "Bavaria and Saxony" is Germany (DE); "Bavaria and Lombardy" is no country ("")."""
    codes = [c for h in hits for c in h]
    leaves = sorted({c for c in codes if not any(o != c and o.startswith(c) for o in codes)})
    return _common_prefix(leaves)


def _common_prefix(codes: list[str]) -> str:
    p = codes[0]
    for c in codes[1:]:
        while not c.startswith(p):
            p = p[:-1]
    return p


#: Why a place resolves to no country: nothing stated, several countries named, a continent or a
#: region spanning several, or a place this table does not know (the ones worth fixing).
UNRESOLVED_REASONS = ("no_place", "several_countries", "multi_country_area", "not_recognised")
#: What a location says when it states none: counted with the empty ones.
_NO_PLACE = frozenset({"n a", "na", "none", "null", "unknown", "unspecified", "not applicable", "not available", "not given",
                       "not mentioned", "not reported", "not specified", "not stated"})


def resolve_location(place: Any, article_country: Any = None, index: NutsIndex | None = None) -> dict[str, Any]:
    """A place, as a paper words it, to a country and a NUTS code.

    Returns `country` (ISO alpha-2, or None), `nuts0` .. `nuts3` (None where unknown), `level`
    (the deepest known, 0 to 3), `method` (`region`, `country`, `article_country` or `unresolved`),
    `several_countries` when the place names more than one, and `reason`, one of
    UNRESOLVED_REASONS when the place resolves to no country, else None."""
    index = index or get_nuts_index()
    folded = fold(place)
    folded = "" if folded in _NO_PLACE else folded                 # "Not reported" states no place
    hint = str(article_country or "").strip().upper()[:2]
    hint = _COUNTED_IN_ISO.get(hint, hint)                         # Reunion (RE) is counted in France
    hint = hint if hint in COUNTRY_NAME else None
    named, area = _countries_named(folded, hint) if folded else (set(), False)
    if isinstance(place, str) and _TURKEY_THE_COUNTRY.search(place):
        named = named | {"TR"}
    out: dict[str, Any] = {"country": None, "nuts0": None, "nuts1": None, "nuts2": None, "nuts3": None, "level": None,
                           "method": "unresolved", "several_countries": False, "reason": None}
    if len(named) > 1:
        out.update({"several_countries": True, "reason": "several_countries"})
        return out
    # "Europe" or "global" is not the article's country: a study across several stays unresolved.
    country = next(iter(named), None) or (None if area else hint)
    # The regions the place names: the deepest when it lies inside the others ("Sigmaringen,
    # Baden-Württemberg"), else only what they share ("Bavaria and Saxony" is Germany).
    hits = _region_hits(folded.split(), index, country)
    if hits:
        code = _region_of(hits)
        if len(code) >= 3:
            iso = iso_of_nuts0(code[:2])
            out.update({"country": iso, "nuts0": code[:2], "method": "region", "level": len(code) - 2})
            for lvl in (1, 2, 3):
                out[f"nuts{lvl}"] = code[: 2 + lvl] if len(code) >= 2 + lvl else None
            return out
        if not country and len(code) < 2:                          # regions of two countries, none named
            out.update({"several_countries": True, "reason": "several_countries"})
            return out
        if not country:                                            # two regions of one country
            iso = iso_of_nuts0(code)
            out.update({"country": iso, "nuts0": code, "level": 0, "method": "region"})
            return out
    if country:
        out.update({"country": country, "nuts0": nuts0_of(country), "level": 0 if nuts0_of(country) else None,
                    "method": "country" if named else "article_country"})
        return out
    out["reason"] = "multi_country_area" if area else "not_recognised" if folded else "no_place"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Loading the official list
# ─────────────────────────────────────────────────────────────────────────────
def parse_nuts_csv(content: str) -> list[tuple[str, int, str, str, str | None]]:
    """Rows (code, level, country, name, alt_name) from Eurostat's GISCO attribute CSV
    (NUTS_ID, LEVL_CODE, CNTR_CODE, NAME_LATN, NUTS_NAME), or from a plain `code,level,name`."""
    reader = csv.DictReader(io.StringIO(content.lstrip("﻿")))
    if not reader.fieldnames:
        raise ValueError("the file has no header row")
    cols = {re.sub(r"[^a-z0-9]", "", f.lower()): f for f in reader.fieldnames}
    code_c = cols.get("nutsid") or cols.get("code")
    if not code_c or not (cols.get("nutsname") or cols.get("namelatn") or cols.get("name")):
        raise ValueError("the file needs the columns NUTS_ID (or code) and NUTS_NAME (or NAME_LATN, or name)")
    out = []
    for row in reader:
        code = (row.get(code_c) or "").strip().upper()
        if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{0,3}", code) or len(code) < 3:
            continue                                               # a country row, or not a code
        name = (row.get(cols.get("nutsname") or "") or row.get(cols.get("namelatn") or "") or row.get(cols.get("name") or "") or "").strip()
        alt = (row.get(cols.get("namelatn") or "") or "").strip() or None
        if not name:
            continue
        level = len(code) - 2
        lc = cols.get("levlcode") or cols.get("level")
        if lc and (row.get(lc) or "").strip().isdigit():
            level = int(row[lc])
        if level not in (1, 2, 3) or level != len(code) - 2:
            raise ValueError(f"{code}: level {level} does not match the code")
        out.append((code, level, iso_of_nuts0(code[:2]), name, alt))
    if not out:
        raise ValueError("no NUTS 1, 2 or 3 row found in the file")
    return out


@app.post("/geo/nuts/import")
async def import_nuts(request: Request, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Load the official NUTS list (CSV in the request body), replacing the one loaded before."""
    try:
        rows = parse_nuts_csv((await request.body()).decode("utf-8", "replace"))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM geo_nuts"))
        conn.execute(text("INSERT INTO geo_nuts (code, level, country, name, alt_name) VALUES (:c, :l, :k, :n, :a) "
                          "ON CONFLICT (code) DO UPDATE SET level = :l, country = :k, name = :n, alt_name = :a"),
                     [{"c": c, "l": l, "k": k, "n": n, "a": a} for c, l, k, n, a in rows])
    _cache["key"] = None                                           # the next lookup rebuilds the index
    by_level = {lvl: sum(1 for r in rows if r[1] == lvl) for lvl in (1, 2, 3)}
    return {"loaded": len(rows), "by_level": by_level, "countries": len({r[2] for r in rows})}


@app.get("/geo/nuts/status")
def nuts_status() -> dict[str, Any]:
    idx = get_nuts_index()
    return {"source": idx.source, "n_regions": idx.n,
            "note": ("The official NUTS list is loaded." if idx.source == "loaded" else
                     "Only the built-in regions are known (Germany NUTS 1, Italy NUTS 1 and 2). Load Eurostat's NUTS file for the rest of Europe.")}


# ─────────────────────────────────────────────────────────────────────────────
# The geography of a scenario
# ─────────────────────────────────────────────────────────────────────────────
def geography(scenario_id: str) -> dict[str, Any]:
    """Where the relevant extracted articles' studies took place, counted over ALL of them."""
    thr = _get_scenario_threshold(scenario_id)
    gate = relevant_gate_sql("d", "ars", ":thr")
    index = get_nuts_index()
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT d.id, d.country, d.extraction_json->'ref'->>'location' AS location,
                   CASE WHEN jsonb_typeof(d.extraction_json->'observations') = 'array'
                        THEN jsonb_array_length(d.extraction_json->'observations') ELSE 0 END AS n_rows
            FROM literature_document d JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE ars.scenario_id = :sid AND {gate} AND jsonb_typeof(d.extraction_json) = 'object'
        """), {"sid": scenario_id, "thr": thr}).mappings().all()
    countries: dict[str, dict[str, Any]] = {}
    unresolved: dict[str, dict[str, Any]] = {}
    by_reason = dict.fromkeys(UNRESOLVED_REASONS, 0)
    several = n_unres = 0
    for r in rows:
        res = resolve_location(r["location"], r["country"], index)
        if res["several_countries"]:
            several += 1
        iso = res["country"]
        if not iso:
            n_unres += 1
            by_reason[res["reason"]] += 1
            key = (r["location"] or "").strip() or "(no place stated)"
            unresolved.setdefault(key, {"location": key, "n": 0, "reason": res["reason"]})["n"] += 1
            continue
        c = countries.setdefault(iso, {"iso2": iso, "name": COUNTRY_NAME.get(iso, iso), "nuts0": nuts0_of(iso),
                                       "n_papers": 0, "n_rows": 0, "regions": {}})
        c["n_papers"] += 1
        c["n_rows"] += int(r["n_rows"] or 0)
        if res["nuts1"]:
            reg = c["regions"].setdefault(res["nuts2"] or res["nuts1"], {"code": res["nuts2"] or res["nuts1"], "n_papers": 0})
            reg["n_papers"] += 1
    out_c = []
    for c in sorted(countries.values(), key=lambda c: (-c["n_papers"], c["name"])):
        regs = sorted(c.pop("regions").values(), key=lambda g: (-g["n_papers"], g["code"]))
        for g in regs:
            g["name"] = _region_name(index, g["code"])
        out_c.append({**c, "regions": regs})
    return {"scenario_id": scenario_id, "n_papers": len(rows), "n_resolved": len(rows) - n_unres, "n_unresolved": n_unres,
            "n_several_countries": several, "unresolved_by_reason": by_reason, "nuts_source": index.source, "countries": out_c,
            "unresolved": sorted(unresolved.values(), key=lambda u: (-u["n"], u["location"]))[:25]}


def _region_name(index: NutsIndex, code: str) -> str:
    return index.name_of.get(code, code)


@app.get("/user-scenarios/{scenario_id}/extraction/geography")
def get_scenario_geography(scenario_id: str) -> dict[str, Any]:
    _get_user_scenario_or_404(scenario_id)
    return geography(scenario_id)


@app.get("/geo/resolve")
def resolve_endpoint(place: str = Query("", max_length=300), country: str | None = Query(None, max_length=2)) -> dict[str, Any]:
    """Try a place: what it resolves to, with the list in use. For checking a label by hand."""
    return {"place": place, **resolve_location(place, country)}
