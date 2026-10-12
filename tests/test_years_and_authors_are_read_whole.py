"""Les années et les auteurs, lus en entier.

- PubMed : l'année ne se lisait que dans PubDate/Year ou ArticleDate/Year. Une notice
  datée par MedlineDate (« 1998 Spring », « Winter 2017-2018 ») entrait sans année.
- Crossref : l'année venait de `created`, la date du DÉPÔT du DOI : un article de 1995
  déposé en 2009 entrait daté de 2009. `issued` d'abord, `created` en dernier.
- Un filtre d'années (`year_min`, `year_max`) écartait toute notice sans année connue ;
  elle reste, et le corpus la compte à part (`year_unknown`).
- PubMed : les auteurs étaient coupés aux six premiers, sans « et al. », si bien que les
  exports et les citations perdaient les autres. Tous sont gardés ; l'affichage abrégé
  reste celui du rapport.
"""
import xml.etree.ElementTree as ET

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api.pipeline import _crossref_year, _pubmed_authors, _pubmed_year  # noqa: E402
from search_tables import ensure_search_tables  # noqa: E402


def _article(pubdate: str = "", article_date: str = "", authors: int = 0):
    names = "".join(f"<Author><LastName>Author{i}</LastName><ForeName>F{i}</ForeName></Author>"
                    for i in range(authors))
    return ET.fromstring(
        "<PubmedArticle><MedlineCitation><Article><Journal><JournalIssue>"
        f"<PubDate>{pubdate}</PubDate></JournalIssue></Journal>"
        f"{article_date}<AuthorList>{names}</AuthorList></Article></MedlineCitation></PubmedArticle>")


@pytest.mark.parametrize("pubdate,article_date,year", [
    ("<Year>2004</Year><Month>Jan</Month>", "", 2004),
    ("<MedlineDate>1998 Spring</MedlineDate>", "", 1998),
    ("<MedlineDate>1998-1999</MedlineDate>", "", 1998),
    ("<MedlineDate>Winter 2017-2018</MedlineDate>", "", 2017),
    ("", "<ArticleDate><Year>2021</Year></ArticleDate>", 2021),
    ("<MedlineDate>Spring</MedlineDate>", "", None),
])
def test_the_pubmed_year_reads_the_medline_date(pubdate, article_date, year):
    assert _pubmed_year(_article(pubdate, article_date)) == year


def test_the_crossref_year_is_the_publication_not_the_deposit():
    created_2009 = {"created": {"date-parts": [[2009, 5, 1]]}}
    assert _crossref_year({**created_2009, "issued": {"date-parts": [[1995]]}}) == 1995
    assert _crossref_year({**created_2009, "published-print": {"date-parts": [[1996, 2]]}}) == 1996
    assert _crossref_year({**created_2009, "published-online": {"date-parts": [[1997]]}}) == 1997
    assert _crossref_year({**created_2009, "published": {"date-parts": [[1998]]}}) == 1998
    assert _crossref_year(created_2009) == 2009, "created reste le dernier recours"
    assert _crossref_year({"issued": {"date-parts": [[None]]}, **created_2009}) == 2009
    assert _crossref_year({}) is None


def test_every_pubmed_author_is_kept():
    authors = _pubmed_authors(_article(authors=9))
    assert authors.split("; ") == [f"Author{i} F{i}" for i in range(9)]
    assert _pubmed_authors(_article()) is None


def test_the_reference_still_shortens_and_the_exports_list_everyone():
    from api.exports import to_bibtex, to_ris
    from api.report import format_reference
    authors = _pubmed_authors(_article(authors=9))
    ref = format_reference(1, {"id": 7, "title": "T", "authors": authors})
    assert "Author5 F5, et al." in ref and "Author6" not in ref
    ris = to_ris([{"id": 7, "title": "T", "authors": authors}])
    assert ris.count("AU  - ") == 9
    assert to_bibtex([{"id": 7, "title": "T", "authors": authors}]).count(" and ") == 8


def test_a_year_filter_keeps_an_unknown_year():
    where, params = main._build_where({"year_min": 2000, "year_max": 2020})
    assert "(d.year IS NULL OR d.year >= :year_min)" in where
    assert "(d.year IS NULL OR d.year <= :year_max)" in where
    assert params == {"year_min": 2000, "year_max": 2020}


def test_the_library_match_keeps_an_unknown_year_and_the_corpus_counts_it(db_conn, monkeypatch):
    from api.scenario_store import scenario_counts
    created_chunk_table = ensure_search_tables(db_conn.cursor())
    monkeypatch.setenv("LEXICAL_SEARCH_ENGINE", "like")
    sid = "usr-yearunknown01"
    abstract = "Wallaroopox outbreaks among zoo staff, a long enough abstract."
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (sid,))
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 9860 AND 9869")
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (sid,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, created_at, updated_at) "
                    "VALUES (%s, 'Years', 'wallaroopox', NOW(), NOW())", (sid,))
        for doc_id, year in ((9860, 2015), (9861, None), (9862, 1990)):
            cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, project_context, "
                        "is_duplicate) VALUES (%s, %s, %s, %s, 'crossref', 'literev', false)",
                        (doc_id, f"Wallaroopox report {doc_id}", abstract, year))
            cur.execute("INSERT INTO article_scenarios (document_id, scenario_id, similarity_score) "
                        "VALUES (%s, %s, 0.9)", (doc_id, sid))
    try:
        ids = set(main._boolean_corpus_ids("wallaroopox", {"year_min": 2000}))
        assert ids == {9860, 9861}, "l'année inconnue reste, 1990 sort"
        c = scenario_counts(sid)
        assert c["year_unknown"] == 1 and c["total"] == 3
    finally:
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (sid,))
            cur.execute("DELETE FROM literature_document WHERE id BETWEEN 9860 AND 9869")
            cur.execute("DELETE FROM user_scenarios WHERE id = %s", (sid,))
            if created_chunk_table:
                cur.execute("DROP TABLE document_chunk")
