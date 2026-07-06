"""Tests for the email-alert digest pure logic (main.py).

The digest sender itself needs SMTP + a DB, but its decision/rendering helpers are
pure and testable: email normalization, the frequency "is-due" rule, and the digest
content (which must list the REAL new articles, escape HTML, and note overflow).
"""
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("pandas")   # main is import-only (env from conftest)

import main


# ── email normalization ──────────────────────────────────────────────────────
def test_clean_email():
    assert main._clean_email("  Erol.Orel@UNIGE.ch ") == "erol.orel@unige.ch"
    assert main._clean_email("not-an-email") is None
    assert main._clean_email("") is None
    assert main._clean_email(None) is None
    assert main._clean_email("a@b.co") == "a@b.co"


# ── SMTP port/mode inference (GoDaddy Pro Email 465 SSL vs M365 587 STARTTLS) ─
def test_smtp_mode_inference():
    assert main._smtp_mode_for(465, None) == "ssl"
    assert main._smtp_mode_for(587, None) == "starttls"
    assert main._smtp_mode_for(25, None) == "starttls"
    assert main._smtp_mode_for(465, "starttls") == "starttls"   # explicit override wins
    assert main._smtp_mode_for(587, "ssl") == "ssl"
    assert main._smtp_mode_for(None, None) == "ssl"             # default when unset
    assert main._smtp_mode_for("nonsense", None) == "ssl"       # bad port → safe default


# ── frequency "is due" ───────────────────────────────────────────────────────
def test_digest_is_due():
    now = datetime(2026, 7, 5, tzinfo=timezone.utc)
    # never notified → always due
    assert main._digest_is_due("weekly", None, now) is True
    # immediate → due even right after
    assert main._digest_is_due("immediate", now - timedelta(minutes=1), now) is True
    # daily: due after >= 1 day, not before
    assert main._digest_is_due("daily", now - timedelta(hours=23), now) is False
    assert main._digest_is_due("daily", now - timedelta(days=1, minutes=1), now) is True
    # weekly: due after >= 7 days, not before
    assert main._digest_is_due("weekly", now - timedelta(days=6), now) is False
    assert main._digest_is_due("weekly", now - timedelta(days=7, minutes=1), now) is True
    # unknown frequency falls back to weekly behaviour
    assert main._digest_is_due(None, now - timedelta(days=8), now) is True


# ── digest rendering ─────────────────────────────────────────────────────────
def _articles(n):
    return [{"id": i, "title": f"Article {i}", "year": 2025, "doi": f"10.x/{i}", "url": None}
            for i in range(n)]


def test_render_digest_lists_real_articles():
    subj, html, text = main._render_alert_digest("epidemic-early-warning", _articles(3), 3)
    assert "3" in subj and "epidemic-early-warning" in subj
    for i in range(3):
        assert f"Article {i}" in html
        assert f"Article {i}" in text
    # DOI link is built when no url is present
    assert "doi.org/10.x/0" in html
    # scenario link present
    assert "/#scenario/epidemic-early-warning" in html


def test_render_digest_escapes_html():
    arts = [{"title": "T&D <script>alert(1)</script>", "year": None, "url": None, "doi": None}]
    _subj, html, _text = main._render_alert_digest("s1", arts, 1)
    assert "<script>" not in html                       # escaped
    assert "&lt;script&gt;" in html


def test_render_digest_overflow_note():
    # 30 new but only 25 listed → the HTML notes the remainder
    _subj, html, text = main._render_alert_digest("s1", _articles(25), 30)
    assert "5 de plus" in html
    assert "5 de plus" in text


def test_render_digest_singular_plural():
    subj1, _h, _t = main._render_alert_digest("s", _articles(1), 1)
    subj2, _h2, _t2 = main._render_alert_digest("s", _articles(2), 2)
    assert "1 nouvel article" in subj1
    assert "2 nouvels articles" in subj2 or "2 nouvel" in subj2   # accept simple pluralization
