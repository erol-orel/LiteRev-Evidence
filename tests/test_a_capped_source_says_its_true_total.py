"""Au plafond par source, « pubmed 2 000 » se lisait comme un total. C'est un plancher.

Le scénario de contrôle de production `usr-54fc5e52fea5` (le bloc des virus seul) porte au
tableau d'identification : core 2 000, pubmed 1 994, openalex 1 993, crossref 1 968,
semantic_scholar 1 957. Cinq sources au plafond de 2 000, et rien dans la carte, dans les
chiffres stockés ni dans l'export ne le disait : un relecteur lisait « PubMed : 1 994 »
comme le nombre d'articles que PubMed contient sur le sujet.

Et le lot gardé n'est pas le même selon la source : l'esearch de PubMed est demandé avec
`sort=pub_date`, donc une PubMed plafonnée garde les 2 000 PLUS RÉCENTS et perd en silence
la littérature H5N1 de 2004 à 2012 ; OpenAlex et Europe PMC trient par pertinence. PRISMA-S
demande le nombre d'enregistrements que chaque base a retournés.

Chaque API annonce son total vrai (`esearchresult.count`, `meta.count`, `hitCount`,
`total-results`, `total`). Il est désormais relevé, servi avec les chiffres
d'identification sous `source_totals`, et toute source dont on a gardé MOINS que ce total
est nommée dans `sources_capped`, affichée « n / total · plafonnée », avec une phrase qui
dit quel lot a été gardé.
"""
import ast
import inspect
import pathlib
import textwrap

import pytest

pytest.importorskip("fastapi")

from api.pipeline import _run_user_scenario_populate  # noqa: E402
from api.search import _prisma_identification_figures  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = inspect.getsource(_run_user_scenario_populate)
TREE = ast.parse(textwrap.dedent(SRC))


# ── Les chiffres ─────────────────────────────────────────────────────────────

def test_a_source_that_announced_more_than_it_returned_is_named_capped():
    """Le cas réel du scénario de contrôle : 1 994 gardés sur un total annoncé plus grand."""
    f = _prisma_identification_figures(
        {"pubmed": 1994, "openalex": 1993, "doaj": 999}, 4986, 0, 4986,
        source_outcomes={"_fetch_pubmed": "ok", "_fetch_openalex": "ok", "_fetch_doaj": "ok"},
        per_source_cap=2000,
        source_totals={"pubmed": 7412, "openalex": 25018, "doaj": 999})
    assert f["source_totals"] == {"pubmed": 7412, "openalex": 25018, "doaj": 999}
    assert f["sources_capped"] == ["openalex", "pubmed"], (
        "les sources dont on a gardé moins que le total annoncé doivent être nommées")
    assert "doaj" not in f["sources_capped"], "999 gardés sur 999 annoncés : pas plafonnée"


def test_a_source_without_an_announced_total_is_never_called_capped():
    """Les préprints d'Europe PMC et bioRxiv/medRxiv n'annoncent pas de total : ne pas
    inventer."""
    f = _prisma_identification_figures(
        {"preprint": 2000, "pubmed": 30}, 2030, 0, 2030,
        source_outcomes={"_fetch_preprints": "ok", "_fetch_pubmed": "ok"},
        per_source_cap=2000, source_totals={"pubmed": 30})
    assert f["sources_capped"] == []
    assert "preprints" not in f["source_totals"]


def test_a_gap_under_the_cap_is_not_a_cap():
    """Le cas réel du premier run de production après #326 : PubMed 1 176 gardés sur 1 182
    annoncés, OpenAlex 1 051 sur 1 058, sous un plafond de 2 000. Des notices perdues au
    nettoyage, pas un plafond : la carte disait pourtant « plafonnée », avec la phrase sur
    les 2 000 plus pertinents. Europe PMC, 1 984 gardés sur 5 469 annoncés, l'est."""
    f = _prisma_identification_figures(
        {"pubmed": 1176, "openalex": 1051, "europepmc": 1984}, 4211, 0, 4211,
        source_outcomes={"_fetch_pubmed": "ok", "_fetch_openalex": "ok", "_fetch_europepmc": "ok"},
        per_source_cap=2000,
        source_totals={"pubmed": 1182, "openalex": 1058, "europepmc": 5469})
    assert f["sources_capped"] == ["europepmc"], f["sources_capped"]
    # Le « n / total » garde l'écart visible, lui : le total est servi tel quel.
    assert f["source_totals"] == {"pubmed": 1182, "openalex": 1058, "europepmc": 5469}


def test_without_a_cap_nothing_is_capped():
    for cap in (None, 0):
        f = _prisma_identification_figures(
            {"pubmed": 10}, 10, 0, 10, source_outcomes={"_fetch_pubmed": "ok"},
            per_source_cap=cap, source_totals={"pubmed": 500})
        assert f["sources_capped"] == [], cap
        assert f["source_totals"] == {"pubmed": 500}


def test_a_cut_or_failed_source_is_not_called_capped_on_top():
    """Son issue dit déjà pourquoi le compte n'est pas un total."""
    f = _prisma_identification_figures(
        {"pubmed": 300, "openalex": 50}, 350, 0, 350,
        source_outcomes={"_fetch_pubmed": "cut_by_budget", "_fetch_openalex": "error"},
        per_source_cap=2000, source_totals={"pubmed": 9000, "openalex": 9000})
    assert f["sources_capped"] == []


def test_a_total_that_is_not_a_number_is_ignored_not_crashed():
    f = _prisma_identification_figures(
        {"pubmed": 30}, 30, 0, 30, source_outcomes={"_fetch_pubmed": "ok"},
        source_totals={"pubmed": "beaucoup", "openalex": None})
    assert f["source_totals"] == {}
    assert f["sources_capped"] == []


def test_totals_are_keyed_by_source_label_like_everything_else():
    f = _prisma_identification_figures(
        {"pubmed": 10}, 10, 0, 10, source_outcomes={"_fetch_pubmed": "ok"},
        per_source_cap=10, source_totals={"_fetch_pubmed": 500})
    assert f["source_totals"] == {"pubmed": 500}
    assert f["sources_capped"] == ["pubmed"]


# ── Les fetchers relèvent le total de leur API ──────────────────────────────

def test_the_helper_keeps_the_largest_total_and_rejects_junk():
    """`_note_total` est une fermeture ; on la lit sur l'arbre et on en vérifie le contrat."""
    fn = next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == "_note_total")
    src = ast.unparse(fn)
    assert "int(total)" in src and "except (TypeError, ValueError)" in src
    assert "max(" in src, "deux pages du même fetcher ne doivent pas écraser le total par un plus petit"
    assert "_counter_lock" in src, "écrit depuis les fils des fetchers : sous le verrou"


@pytest.mark.parametrize("source,field", [
    # PubMed passe la variable `total_found`, elle-même lue dans `esearchresult.count` :
    # c'est cette lecture qu'on vérifie juste en dessous. arXiv passe par le parseur de
    # son flux Atom (`opensearch:totalResults`), CORE lit `totalHits`.
    ("pubmed", "total_found"), ("openalex", "count"), ("crossref", "total-results"),
    ("europepmc", "hitCount"), ("semantic_scholar", "total"), ("doaj", "total"),
    ("core", "totalHits"), ("arxiv", "_parse_arxiv_total"), ("clinicaltrials", "totalCount"),
])
def test_each_api_total_is_noted(source, field):
    calls = [n for n in ast.walk(TREE) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "_note_total"
             and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == source]
    assert calls, f"le fetcher {source} ne relève pas le total annoncé par son API"
    assert any(field in ast.unparse(c.args[1]) for c in calls), (
        f"{source} : le total doit être lu dans le champ {field!r}")


def test_pubmed_total_comes_from_the_esearch_count():
    """`total_found` est le compteur de l'esearch, le même que celui du site PubMed."""
    fn = next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == "_fetch_pubmed")
    assigns = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "total_found"]
    # `ast.unparse` normalise les guillemets : on cherche le mot, pas sa ponctuation.
    assert assigns and all("count" in ast.unparse(a.value) for a in assigns), (
        [ast.unparse(a) for a in assigns])


def test_the_totals_snapshot_is_taken_beside_the_records_snapshot():
    """Même instant que `_recs_snapshot` : des totaux pris plus tard décriraient un autre run."""
    outer = next(n for n in TREE.body if isinstance(n, ast.FunctionDef))
    lines = {}
    for n in ast.walk(outer):
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            if n.targets[0].id in ("_recs_snapshot", "_totals_snapshot"):
                lines[n.targets[0].id] = n.lineno
    assert set(lines) == {"_recs_snapshot", "_totals_snapshot"}, lines
    assert abs(lines["_totals_snapshot"] - lines["_recs_snapshot"]) <= 2


# ── La carte le dit, dans les deux langues ───────────────────────────────────

def test_the_card_shows_n_of_total_and_the_cap_note():
    page = (ROOT / "frontend" / "src" / "components" / "ScenarioDetailPage.tsx").read_text(encoding="utf-8")
    assert "source_totals" in page and "sources_capped" in page
    assert "prisma.cappedNote" in page and "prisma.capped" in page
    for loc in ("fr", "en"):
        txt = (ROOT / "frontend" / "src" / "i18n" / "locales" / f"{loc}.ts").read_text(encoding="utf-8")
        line = next(l for l in txt.splitlines() if "cappedNote:" in l)
        assert "{sources}" in line and "{cap}" in line, f"{loc}: placeholders"
        assert "PubMed" in line and "OpenAlex" in line, (
            f"{loc}: la note doit dire quel lot chaque source garde, c'est tout son intérêt")


def test_the_three_capped_sources_are_asked_by_relevance_and_the_note_says_so():
    """Le fait que la note affirme, épinglé contre le code qui le rend vrai.

    Décision du propriétaire du projet : au plafond, PubMed garde les plus pertinents
    (`sort=relevance`, le « Best Match »), comme OpenAlex et Europe PMC. Trié par date, il
    gardait les 2 000 plus récents et perdait la littérature H5N1 de 2004 à 2012. Si l'un
    des trois change d'ordre, la note de la carte devient fausse et doit changer ici."""
    assert '"sort": "relevance"' in SRC, "l'esearch du populate ne trie plus par pertinence"
    assert '"sort": "pub_date"' not in SRC, "PubMed est de nouveau trié par date"
    assert "relevance_score:desc" in SRC, "OpenAlex ne trie plus par pertinence"
    import inspect
    from api import sources as S
    assert '"sort": "relevance"' in inspect.getsource(S._live_fetch_pubmed), (
        "le panneau de recherche en direct montre une autre liste que le corpus")
    # Europe PMC : pas de paramètre `sort`, son défaut est la pertinence (commentaire du
    # fetcher) ; on vérifie qu'aucun tri par date n'y a été ajouté.
    _i = SRC.index("def _fetch_europepmc")
    assert '"sort"' not in SRC[_i:SRC.index("def _fetch_preprints")], (
        "un tri a été ajouté à Europe PMC : la note doit le dire")
    for loc in ("fr", "en"):
        line = next(l for l in (ROOT / "frontend" / "src" / "i18n" / "locales" / f"{loc}.ts")
                    .read_text(encoding="utf-8").splitlines() if "cappedNote:" in l)
        assert ("pertinen" in line or "relevan" in line) and "récent" not in line.split("pas des")[0] \
            if loc == "fr" else ("relevan" in line), f"{loc}: la note ne dit plus l'ordre réel"
