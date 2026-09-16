"""The per-source answer cache of the federation (api/pipeline.py).

A search re-run a few hours after the first one asked every source for its whole result
again, to insert nothing. Each fetcher's answer to (query, filters, cap) is now kept for
SOURCE_CACHE_TTL_S and replayed. Integration: needs the test database (skips without)."""
import main


def test_the_cache_key_depends_on_query_filters_and_cap():
    h = main._source_query_hash("chikungunya AND europe", {}, 2000)
    assert h == main._source_query_hash("chikungunya AND europe ", {}, 2000)      # trimmed
    assert h != main._source_query_hash("chikungunya AND europe", {"year_min": 2020}, 2000)
    assert h != main._source_query_hash("chikungunya AND europe", {}, 500)
    assert len(h) == 40


def test_save_then_load_then_expire(db_conn):
    main._ensure_source_query_cache()
    h = main._source_query_hash("test cache query", {}, 2000)
    links = [(101, "pubmed", True), (102, "pubmed", True), (103, "openalex", False)]
    main._save_source_cache("_fetch_pubmed", h, "test cache query", links)
    loaded = main._load_source_cache(h)
    assert set(loaded) == {"_fetch_pubmed"}
    assert loaded["_fetch_pubmed"]["links"] == links
    assert loaded["_fetch_pubmed"]["n_records"] == 3
    assert loaded["_fetch_pubmed"]["age_s"] < 60

    # A second save replaces the entry; a TTL of zero disables the cache entirely.
    main._save_source_cache("_fetch_pubmed", h, "test cache query", links[:1])
    assert main._load_source_cache(h)["_fetch_pubmed"]["n_records"] == 1
    assert main._load_source_cache(h, ttl_s=0) == {}

    # Older than the TTL: not replayed.
    with db_conn.cursor() as cur:
        cur.execute("UPDATE source_query_cache SET fetched_at = NOW() - INTERVAL '2 days' "
                    "WHERE query_hash = %s", (h,))
    assert main._load_source_cache(h) == {}
    assert main._load_source_cache(h, ttl_s=10 * 86400)["_fetch_pubmed"]["n_records"] == 1
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM source_query_cache WHERE query_hash = %s", (h,))
