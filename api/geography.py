"""Where the evidence comes from: a study's place, resolved to a country and a NUTS region.

The extraction records a place as the paper words it ("Sigmaringen, Baden-Wuerttemberg,
Germany"). To combine the literature with official statistics by area (Eurostat, national
databases), that has to become a country and, where it can, a NUTS code (the EU's hierarchy of
territorial units: NUTS 0 a country, 1 a major region, 2 a basic region, 3 a small one; the codes
nest, so DE139 sits in DE13, DE1 and DE).

What this does and does not know, said plainly:

  - Countries: about fifty, recognised by their name in English, French, Italian, German and
    Spanish, with the NUTS 0 code where there is one (Greece is EL and the United Kingdom UK, not
    GR and GB). "Turkey" is deliberately NOT recognised: in this literature it is a bird.
  - Regions: only those this code is sure of, Germany's sixteen Laender (NUTS 1) and Italy's five
    macro-areas (NUTS 1) and twenty-one regions (NUTS 2). A wrong code is worse than a blank, so
    nothing else is guessed.
  - The OFFICIAL list. An administrator loads Eurostat's NUTS file (`/geo/nuts/import`; the GISCO
    attribute CSV has NUTS_ID, LEVL_CODE and NUTS_NAME) and the whole of Europe resolves to NUTS 3,
    with no code change. Until then the built-in regions are what is used, and the API says which.

Resolution is deterministic and conservative. A place name must match whole words; the longest and
deepest match wins; a name that exists in several countries is taken only in the country the paper
names (or the article's own country); a place that names two countries is "several countries", not
a guess; and what cannot be resolved is counted and listed so that it can be fixed.
"""
from __future__ import annotations

import csv
import io
import re
import unicodedata
from typing import Any

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
    """ASCII, lower case, anything that is not a letter or a digit as one space."""
    t = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Countries: ISO 3166 alpha-2 -> names; the NUTS 0 code differs for two of them
# ─────────────────────────────────────────────────────────────────────────────
#: (iso2, english name, other names). NUTS countries first; the rest are counted but have no NUTS.
_COUNTRIES = [
    ("AT", "Austria", ("Osterreich", "Autriche", "Austria")), ("BE", "Belgium", ("Belgique", "Belgio", "Belgien", "Belgie")),
    ("BG", "Bulgaria", ("Bulgarie", "Bulgarien")), ("HR", "Croatia", ("Croatie", "Croazia", "Kroatien", "Hrvatska")),
    ("CY", "Cyprus", ("Chypre", "Cipro", "Zypern")), ("CZ", "Czechia", ("Czech Republic", "Czech", "Tchequie", "Cechia", "Tschechien")),
    ("DK", "Denmark", ("Danemark", "Danimarca", "Danemarca", "Danmark", "Dinamarca")), ("EE", "Estonia", ("Estonie", "Estland")),
    ("FI", "Finland", ("Finlande", "Finlandia", "Finnland", "Suomi")), ("FR", "France", ("Francia", "Frankreich")),
    ("DE", "Germany", ("Allemagne", "Germania", "Deutschland", "Alemania")), ("GR", "Greece", ("Grece", "Grecia", "Griechenland", "Hellas", "Grecia")),
    ("HU", "Hungary", ("Hongrie", "Ungheria", "Ungarn", "Hungria")), ("IE", "Ireland", ("Irlande", "Irlanda", "Irland")),
    ("IT", "Italy", ("Italie", "Italia", "Italien")), ("LV", "Latvia", ("Lettonie", "Lettonia", "Lettland")),
    ("LT", "Lithuania", ("Lituanie", "Lituania", "Litauen")), ("LU", "Luxembourg", ("Lussemburgo", "Luxemburg")),
    ("MT", "Malta", ("Malte",)), ("NL", "Netherlands", ("The Netherlands", "Holland", "Pays-Bas", "Paesi Bassi", "Niederlande", "Nederland", "Paises Bajos")),
    ("PL", "Poland", ("Pologne", "Polonia", "Polen", "Polska")), ("PT", "Portugal", ()), ("RO", "Romania", ("Roumanie", "Rumanien")),
    ("SK", "Slovakia", ("Slovaquie", "Slovacchia", "Slowakei", "Slovak Republic")), ("SI", "Slovenia", ("Slovenie", "Slovenia", "Slowenien")),
    ("ES", "Spain", ("Espagne", "Spagna", "Spanien", "Espana")), ("SE", "Sweden", ("Suede", "Svezia", "Schweden", "Sverige", "Suecia")),
    ("IS", "Iceland", ("Islande", "Islanda", "Island")), ("LI", "Liechtenstein", ()), ("NO", "Norway", ("Norvege", "Norvegia", "Norwegen", "Norge")),
    ("CH", "Switzerland", ("Suisse", "Svizzera", "Schweiz", "Suiza")), ("GB", "United Kingdom", ("UK", "Great Britain", "Britain", "England", "Scotland", "Wales", "Northern Ireland", "Royaume-Uni", "Regno Unito", "Vereinigtes Konigreich")),
    ("ME", "Montenegro", ()), ("MK", "North Macedonia", ("Macedonia",)), ("AL", "Albania", ("Albanie", "Albanien")), ("RS", "Serbia", ("Serbie", "Serbien")),
    # Outside NUTS: counted, never given a NUTS code.
    ("US", "United States", ("USA", "United States of America", "Etats-Unis", "Stati Uniti", "Vereinigte Staaten")), ("CN", "China", ("Chine", "Cina")),
    ("IN", "India", ("Inde",)), ("BD", "Bangladesh", ()), ("VN", "Vietnam", ("Viet Nam",)), ("KH", "Cambodia", ("Cambodge",)), ("TH", "Thailand", ("Thailande",)),
    ("ID", "Indonesia", ("Indonesie",)), ("EG", "Egypt", ("Egypte", "Egitto")), ("NG", "Nigeria", ()), ("BR", "Brazil", ("Bresil", "Brasile")),
    ("CA", "Canada", ()), ("JP", "Japan", ("Japon", "Giappone")), ("KR", "South Korea", ("Republic of Korea", "Coree du Sud")), ("AU", "Australia", ("Australie",)),
    ("MX", "Mexico", ("Mexique", "Messico")), ("PK", "Pakistan", ()), ("IR", "Iran", ()), ("IL", "Israel", ()), ("RU", "Russia", ("Russie", "Russland")),
    ("UA", "Ukraine", ()), ("ZA", "South Africa", ("Afrique du Sud", "Sudafrica")), ("KE", "Kenya", ()), ("GH", "Ghana", ()),
]
_NUTS_ISO = frozenset({
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
    "PL", "PT", "RO", "SK", "SI", "ES", "SE", "IS", "LI", "NO", "CH", "GB", "ME", "MK", "AL", "RS"})
#: Where the NUTS 0 code is not the ISO code.
_NUTS0 = {"GR": "EL", "GB": "UK"}
COUNTRY_NAME = {iso: name for iso, name, _a in _COUNTRIES}
_NAME_TO_ISO: dict[str, str] = {}
for _iso, _name, _alts in _COUNTRIES:
    for _n in (_name, *_alts):
        _NAME_TO_ISO.setdefault(fold(_n), _iso)
_NAME_TO_ISO.setdefault("uk", "GB")
_NAME_TO_ISO.pop("turkey", None)                  # a bird, in this literature
_COUNTRY_NAMES_BY_LENGTH = sorted(_NAME_TO_ISO, key=len, reverse=True)


def nuts0_of(iso2: str | None) -> str | None:
    if not iso2 or iso2 not in _NUTS_ISO:
        return None
    return _NUTS0.get(iso2, iso2)


def iso_of_nuts0(code: str) -> str:
    return {"EL": "GR", "UK": "GB"}.get(code, code)


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
#: The Eurostat names of the same codes, so a loaded file overrides rather than duplicates them.


class NutsIndex:
    """The NUTS names in use, ready for matching: folded name -> [(code, level, country)]."""

    def __init__(self, entries: list[tuple[str, int, str, str]], source: str):
        self.source = source
        self.n = len({e[0] for e in entries})
        self.by_name: dict[str, list[tuple[str, int, str]]] = {}
        for code, level, country, name in entries:
            self.by_name.setdefault(fold(name), []).append((code, level, country))
        self.names_by_length = sorted((n for n in self.by_name if n), key=len, reverse=True)
        self.name_of: dict[str, str] = {}
        for code, _l, _c, name in entries:                            # the first name listed is the one shown
            self.name_of.setdefault(code, name)


def _builtin_entries() -> list[tuple[str, int, str, str]]:
    return [(code, lvl, c, n) for code, (lvl, c, names) in _BUILTIN.items() for n in names]


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
    entries = _builtin_entries()
    have = {r[0] for r in rows}
    entries = [e for e in entries if e[0] not in have]            # the official name wins for a code it has
    for code, level, country, name, alt in rows:
        entries.append((code, level, country, name))
        if alt and alt != name:
            entries.append((code, level, country, alt))
    idx = NutsIndex(entries, "loaded" if rows else "builtin")
    _cache.update({"key": key, "index": idx})
    return idx


def _words(folded: str, name: str) -> bool:
    """Whole words only. Both sides are folded (letters, digits and single spaces), so padding
    with a space is the word boundary, and it is much cheaper than a regex per name."""
    return bool(name) and f" {name} " in f" {folded} "


def _common_prefix(codes: list[str]) -> str:
    p = codes[0]
    for c in codes[1:]:
        while not c.startswith(p):
            p = p[:-1]
    return p


def resolve_location(place: Any, article_country: Any = None, index: NutsIndex | None = None) -> dict[str, Any]:
    """A place, as a paper words it, to a country and a NUTS code.

    Returns `country` (ISO alpha-2, or None), `nuts0` .. `nuts3` (None where unknown), `level`
    (the deepest known, 0 to 3), `method` (`region`, `country`, `article_country` or `unresolved`)
    and `several_countries` when the place names more than one."""
    index = index or get_nuts_index()
    folded = fold(place)
    hint = str(article_country or "").strip().upper()[:2] or None
    hint = hint if hint in COUNTRY_NAME else None
    named = {_NAME_TO_ISO[n] for n in _COUNTRY_NAMES_BY_LENGTH if _words(folded, n)} if folded else set()
    out: dict[str, Any] = {"country": None, "nuts0": None, "nuts1": None, "nuts2": None, "nuts3": None, "level": None,
                           "method": "unresolved", "several_countries": False}
    if len(named) > 1:
        out["several_countries"] = True
        return out
    country = next(iter(named), None) or hint
    # A place name in a NUTS list: longest first, restricted to the country when it is known.
    best: list[tuple[str, int]] = []
    for name in index.names_by_length:
        if not _words(folded, name):
            continue
        cands = [(c, lvl) for c, lvl, cc in index.by_name[name] if country is None or cc == country]
        if cands:
            best = cands
            break
    if best:
        deepest = max(lvl for _c, lvl in best)
        codes = sorted({c for c, lvl in best if lvl == deepest})
        code = codes[0] if len(codes) == 1 else _common_prefix(codes)
        if len(code) >= 3:
            iso = iso_of_nuts0(code[:2])
            out.update({"country": iso, "nuts0": code[:2], "method": "region", "level": len(code) - 2})
            for lvl in (1, 2, 3):
                out[f"nuts{lvl}"] = code[: 2 + lvl] if len(code) >= 2 + lvl else None
            return out
    if country:
        out.update({"country": country, "nuts0": nuts0_of(country), "level": 0 if nuts0_of(country) else None,
                    "method": "country" if named else "article_country"})
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
    unresolved: dict[str, int] = {}
    several = n_unres = 0
    for r in rows:
        res = resolve_location(r["location"], r["country"], index)
        if res["several_countries"]:
            several += 1
        iso = res["country"]
        if not iso:
            n_unres += 1
            key = (r["location"] or "").strip() or "(no place stated)"
            unresolved[key] = unresolved.get(key, 0) + 1
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
            "n_several_countries": several, "nuts_source": index.source, "countries": out_c,
            "unresolved": [{"location": k, "n": v} for k, v in sorted(unresolved.items(), key=lambda kv: (-kv[1], kv[0]))[:25]]}


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
