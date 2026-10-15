"""The limit on an uploaded file is the application's, and it says so.

Two walls stood in front of the two upload features (api/additions.py for papers,
api/review_import.py for decisions), and a real file hit both:

- nginx refuses a request body above 1 MB by default, answering 413 from the proxy. The
  API never sees the request, so it cannot explain anything: the ISS team's screening
  export (1,524 records, 4.2 MB) was refused before it arrived. `deploy.sh` now sets the
  proxy's limit above the application's own, so the limit that speaks is the one with a
  message. This file pins the two together: raise MAX_FILE_BYTES and the test says the
  proxy must follow.
- the `csv` module refuses a cell above 128 KB and raises, which the endpoint answered
  with a 500. A long cell is now read, and a file that is genuinely unreadable says so.
"""
import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from api import additions as A  # noqa: E402

DEPLOY = Path(__file__).resolve().parent.parent / "deploy.sh"
#: `client_max_body_size 32m;`, in bytes, whatever unit the script writes it in.
_UNITS = {"": 1, "k": 1024, "m": 1024 * 1024, "g": 1024 * 1024 * 1024}


def _nginx_body_limit() -> int:
    m = re.search(r"client_max_body_size\s+(\d+)\s*([kmgKMG]?)\s*;", DEPLOY.read_text(encoding="utf-8"))
    assert m, "deploy.sh no longer sets client_max_body_size: an upload over 1 MB is refused by nginx"
    return int(m.group(1)) * _UNITS[m.group(2).lower()]


def test_the_proxy_lets_through_what_the_app_accepts():
    """A file the application would accept must reach it. The proxy's limit is above
    MAX_FILE_BYTES, so a file too large is refused by the API, with its own message, and
    never by a 413 the API cannot explain."""
    assert _nginx_body_limit() > A.MAX_FILE_BYTES, (
        "nginx would refuse a file the application accepts, with a 413 the user cannot act on")


def test_the_proxy_limit_stays_within_reason():
    """And not so far above that a mistaken multi-gigabyte upload is held in memory by the
    proxy before the application ever counts its bytes."""
    assert _nginx_body_limit() <= 4 * A.MAX_FILE_BYTES


def test_a_long_cell_is_read_rather_than_refused():
    """csv's own default (128 KB per cell) is not the file's limit. A notes column holding
    a whole abstract, or a screening note carrying every label of a record, goes past it."""
    assert A.CSV_MAX_FIELD_BYTES > 128 * 1024
    long_note = "RAYYAN-INCLUSION: {\"R\"=>\"Included\"} " + ("persistence " * 20000)
    data = ('title,doi,notes\n'
            f'"A paper about poultry workers",10.5555/longcell.1,"{long_note}"\n').encode()
    recs = A.parse_csv(data)
    assert len(recs) == 1
    assert recs[0]["title"].startswith("A paper about poultry workers")
    assert len(long_note) > 128 * 1024, "the cell must really be past csv's default"


def test_a_cell_past_even_that_says_so_instead_of_breaking():
    """Past the raised limit, csv still raises, and the answer must be a sentence about the
    file rather than a 500: `_prepare` turns this ValueError into a 422, and the upload
    report into one unreadable file."""
    data = ('title,doi\n"%s",10.5555/huge.1\n' % ("x" * (A.CSV_MAX_FIELD_BYTES + 16))).encode()
    with pytest.raises(ValueError) as e:
        A.parse_csv(data)
    assert "CSV" in str(e.value)
