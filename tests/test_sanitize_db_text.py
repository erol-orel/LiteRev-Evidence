"""NUL bytes in extracted text must never reach an INSERT.

Observed in production three times in 45 minutes:

    WARNING: Fulltext doc 138471: (psycopg.DataError) PostgreSQL text fields
             cannot contain NUL (0x00) bytes

`pdftotext` emits NUL on some malformed PDFs. `_insert_fulltext_chunks_ft` does
DELETE-then-INSERT inside ONE transaction, so a single bad chunk rolls the whole thing
back: the article keeps no full text at all, and the only trace is a WARNING. The same
class reaches the main ingestion path too, because a JSON API may return "\\u0000",
which `json.loads` decodes into a real NUL.

The trap these tests exist to pin: the extractors run `re.sub(r"\\s+", " ", ...)` right
before returning, which looks like it would clean this up and does not — Python's `\\s`
does not match `\\x00`.
"""
import pytest

pytest.importorskip("fastapi")

import main


NUL = "\x00"


def test_the_production_failure_is_neutralised():
    """A NUL anywhere in the text is removed, leaving the surrounding text intact."""
    got = main.sanitize_db_text(f"Methods{NUL} and results")
    assert NUL not in got
    assert got == "Methods and results"


def test_whitespace_normalisation_alone_would_not_have_caught_it():
    """Pins WHY the bug survived: `\\s` does not cover NUL. If this ever fails, the
    sanitiser has become redundant and can go."""
    import re
    assert NUL in re.sub(r"\s+", " ", f"a{NUL}b"), "re.\\s now matches NUL — revisit"
    assert NUL not in main.sanitize_db_text(f"a{NUL}b")


def test_real_text_is_left_alone():
    """Accents, punctuation, newlines and tabs are legitimate and must survive."""
    for text in ("Étude prospective sur la grippe A (H1N1) — 2009",
                 "line one\nline two\tcolumn",
                 "R₀ = 1.28 (IQR 1.19–1.37)",
                 "emoji ok 🦠", ""):
        assert main.sanitize_db_text(text) == text


def test_other_control_characters_go_too():
    """Postgres accepts these, but in extracted text they are decoding garbage and they
    corrupt LLM prompts. Tab, newline and carriage return are explicitly kept."""
    assert main.sanitize_db_text("a\x01b\x1fc\x7fd") == "abcd"
    assert main.sanitize_db_text("keep\tthese\r\nplease") == "keep\tthese\r\nplease"


def test_non_strings_pass_through_untouched():
    """Called on optional columns (abstract, authors, journal) that are often None."""
    for value in (None, 42, b"bytes", []):
        assert main.sanitize_db_text(value) is value


def test_a_pdf_that_is_entirely_nul_yields_empty_not_a_crash():
    assert main.sanitize_db_text(NUL * 100) == ""


def test_the_json_unicode_escape_route_is_covered():
    """How this reaches the INGESTION path: an API returns "\\u0000" in an abstract."""
    import json
    abstract = json.loads('"Background:\\u0000 we studied..."')
    assert NUL in abstract, "sanity: json.loads really does produce a raw NUL"
    assert NUL not in main.sanitize_db_text(abstract)


def test_postgres_actually_rejects_what_we_strip(db_conn):
    """The premise, verified against a real server rather than trusted.

    Confirms both halves: raw NUL is refused, and the sanitised string is accepted.
    Skips cleanly when no database is reachable.
    """
    poisoned = f"Results{NUL} of the trial"
    with db_conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE nul_probe (body text)")
        with pytest.raises(Exception) as exc:
            cur.execute("INSERT INTO nul_probe (body) VALUES (%s)", (poisoned,))
        assert "NUL" in str(exc.value) or "0x00" in str(exc.value)

    # The connection needs a clean statement after the failed one.
    with db_conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE nul_probe2 (body text)")
        cur.execute("INSERT INTO nul_probe2 (body) VALUES (%s)",
                    (main.sanitize_db_text(poisoned),))
        cur.execute("SELECT body FROM nul_probe2")
        assert cur.fetchone()[0] == "Results of the trial"
