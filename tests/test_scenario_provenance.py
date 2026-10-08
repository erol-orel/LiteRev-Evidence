"""A saved search records when it was created and from where.

The date and the time were already there (`user_scenarios.created_at`). The address
was not, and the one place that already resolved it, the rate limiter, did so inside
its own middleware. Two readings of `X-Forwarded-For` drift, and the one that counts
the wrong hop grants a forged header the credit of a real address, so there is one
function and both callers use it.

The header is written by whoever is in front: the client writes the leftmost entries
and our own proxy appends what it actually saw. Only the last `TRUSTED_PROXY_HOPS`
entries are ours, so the real client is read from the END, never from the beginning.
"""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import main  # noqa: E402
from api.core import client_ip  # noqa: E402


class _Req:
    """Le minimum qu'une requête doit offrir : des en-têtes et une adresse."""

    class _Client:
        def __init__(self, host): self.host = host

    def __init__(self, xff=None, host="127.0.0.1"):
        self.headers = {"X-Forwarded-For": xff} if xff else {}
        self.client = self._Client(host) if host else None


# ── reading the address (pure) ───────────────────────────────────────────────

def test_without_a_proxy_header_the_connection_address_is_used():
    assert client_ip(_Req(host="198.51.100.4")) == "198.51.100.4"


def test_the_trusted_hop_is_read_from_the_end_not_the_beginning():
    """`client, ours` : our proxy appended what it saw, and that is the only entry
    the client could not write. Reading the first would trust a forged header."""
    assert client_ip(_Req(xff="203.0.113.7, 10.0.0.1")) == "10.0.0.1"


def test_a_single_entry_is_taken_as_it_stands():
    assert client_ip(_Req(xff="203.0.113.7")) == "203.0.113.7"


def test_empty_segments_do_not_produce_an_empty_address():
    """An empty address would put every caller in one bucket, here and in the limiter."""
    assert client_ip(_Req(xff=" , , 10.0.0.9")) == "10.0.0.9"
    assert client_ip(_Req(xff="   ", host="198.51.100.4")) == "198.51.100.4"


def test_a_request_with_no_client_at_all_still_answers():
    assert client_ip(_Req(host=None)) == "unknown"


def test_the_address_is_cut_to_what_the_column_holds():
    """45 signs is the longest an IPv6 address is written; the column is VARCHAR(45)."""
    assert len(client_ip(_Req(xff="a" * 200))) <= 45


def test_an_ipv6_address_passes_through_whole():
    v6 = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
    assert len(v6) <= 45 and client_ip(_Req(xff=v6)) == v6


# ── recording it (database) ──────────────────────────────────────────────────

def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


@pytest.fixture()
def client(db_conn):
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    from fastapi.testclient import TestClient
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
    created: list[str] = []
    yield TestClient(main.app), created
    with db_conn.cursor() as cur:
        for sid in created:
            cur.execute("DELETE FROM user_scenarios WHERE id = %s", (sid,))


def _create(tc, created, **headers):
    r = tc.post("/user-scenarios",
                json={"name": "Provenance", "query": "rsv AND geneva", "mode": "boolean",
                      "filters": {}, "result_count": 0, "pinned": False},
                headers={"X-API-Key": "test-write-key", **headers})
    assert r.status_code == 201, r.text
    body = r.json()
    created.append(body["id"])
    return body


def test_a_created_search_carries_its_date_and_its_address(client):
    tc, created = client
    body = _create(tc, created, **{"X-Forwarded-For": "203.0.113.7, 10.0.0.1"})
    assert body["created_at"]
    assert body["created_ip"] == "10.0.0.1"


def test_the_date_and_the_address_are_both_on_the_detail_page(client):
    tc, created = client
    sid = _create(tc, created, **{"X-Forwarded-For": "10.0.0.2"})["id"]
    d = tc.get(f"/user-scenarios/{sid}/detail").json()
    assert d["created_at"] and d["created_ip"] == "10.0.0.2"


def test_the_date_and_the_address_survive_to_the_list(client):
    tc, created = client
    sid = _create(tc, created, **{"X-Forwarded-For": "10.0.0.3"})["id"]
    rows = tc.get("/user-scenarios").json()
    mine = [r for r in rows if r["id"] == sid]
    assert mine and mine[0]["created_ip"] == "10.0.0.3" and mine[0]["created_at"]


def test_a_search_created_without_a_forwarded_header_still_records_an_address(client):
    """Direct access, a local run: the connection address is recorded, not nothing."""
    tc, created = client
    body = _create(tc, created)
    assert body["created_ip"]


def test_a_row_written_before_the_column_existed_reads_as_unknown(client, db_conn):
    """Absence is said, never invented: an older search has no address to show."""
    tc, created = client
    sid = _create(tc, created)["id"]
    with db_conn.cursor() as cur:
        cur.execute("UPDATE user_scenarios SET created_ip = NULL WHERE id = %s", (sid,))
    assert tc.get(f"/user-scenarios/{sid}/detail").json()["created_ip"] is None
