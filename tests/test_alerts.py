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


def test_render_digest_uses_name_and_deeplink():
    name = "Prevision des hausses ARI"
    subj, html, text = main._render_alert_digest(
        "usr-095d26e192ae", _articles(3), 3, scenario_name=name)
    # subject + body show the human NAME, not the raw id
    assert name in subj
    assert "usr-095d26e192ae" not in subj
    assert name in html
    for i in range(3):
        assert f"Article {i}" in html
        assert f"Article {i}" in text
    # DOI link built when no url is present
    assert "doi.org/10.x/0" in html
    # deep link to the scenario PAGE (?scenario=<id>), not the old /#scenario search route
    assert "/?scenario=usr-095d26e192ae" in html
    assert "/#scenario/" not in html


def test_render_digest_falls_back_to_id_without_name():
    subj, _h, _t = main._render_alert_digest("s1", _articles(1), 1)
    assert "s1" in subj                                  # no name → id is used


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


def test_render_digest_singular_plural_french_agreement():
    subj1, _h, _t = main._render_alert_digest("s", _articles(1), 1)
    subj2, _h2, _t2 = main._render_alert_digest("s", _articles(2), 2)
    assert "1 nouvel article" in subj1              # singular
    assert "2 nouveaux articles" in subj2           # correct FR plural (not "nouvels")
    assert "nouvels" not in subj2


# ── what the RUNNER feeds the renderer, and what the preview reveals ─────────
class _FakeResult:
    def __init__(self, rows): self._rows = rows
    def mappings(self): return self
    def all(self): return self._rows


class _FakeConn:
    """Answers the only raw SQL the runner still issues: the subscription list."""
    def __init__(self, subs): self._subs = subs
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, *_a, **_k): return _FakeResult(self._subs)


class _FakeEngine:
    def __init__(self, subs): self._subs = subs
    def connect(self): return _FakeConn(self._subs)
    def begin(self): return _FakeConn(self._subs)


def _drive(monkeypatch, *, listed, total, smtp, dry_run):
    """Run the real digest loop with the two article helpers stubbed."""
    from conftest import patch_app

    captured = {}

    def _fake_render(sid, articles, total_new, scenario_name=None):
        captured["listed"], captured["total"] = len(articles), total_new
        return ("subject", "<html></html>", "text")

    subs = [{"id": 1, "email": "a@b.c", "scenario_id": "usr-1",
             "frequency": "daily", "last_notified_at": None}]
    patch_app(monkeypatch, "engine", _FakeEngine(subs))
    patch_app(monkeypatch, "_render_alert_digest", _fake_render)
    patch_app(monkeypatch, "_new_articles_for_scenario",
              lambda c, s, since, limit=25: _articles(listed))
    patch_app(monkeypatch, "_count_new_articles_for_scenario", lambda c, s, since: total)
    patch_app(monkeypatch, "_get_scenario_name", lambda s: "Chikungunya")
    patch_app(monkeypatch, "_send_email_smtp", lambda *a, **k: None)
    monkeypatch.setenv("SMTP_HOST", smtp)

    out = main._process_alert_digests(None, dry_run=dry_run, respect_frequency=False)
    return out, captured


def test_the_digest_announces_the_real_total_not_the_number_of_lines(monkeypatch):
    """The renderer takes (articles, total_new) and is tested above with 25 of 30. The
    runner passed `len(arts)` as the total, and the query already caps `arts` at 25, so
    the total ALWAYS equalled the number of lines. A subscriber whose scenario had gained
    300 articles was emailed "25 nouveaux articles", and the tested "et N de plus" line
    could never fire in production. The runner counts the total separately now."""
    out, captured = _drive(monkeypatch, listed=25, total=300,
                           smtp="smtp.example.org", dry_run=False)
    assert captured["listed"] == 25
    assert captured["total"] == 300, "the email must announce the real total"
    row = out["results"][0]
    assert row["sent"] is True
    assert row["new"] == 300 and row["listed"] == 25


def test_the_preview_says_whether_anything_would_actually_be_sent(monkeypatch):
    """dry_run short-circuited BEFORE the SMTP check, so the preview answered "dry_run"
    identically on a server that would send and on one that can never send. The single
    prerequisite you run a preview to verify was the one it hid."""
    out, _ = _drive(monkeypatch, listed=3, total=3, smtp="", dry_run=True)
    assert out["smtp_configured"] is False
    assert out["results"][0]["would_send"] is False

    out2, _ = _drive(monkeypatch, listed=3, total=3,
                     smtp="smtp.example.org", dry_run=True)
    assert out2["smtp_configured"] is True
    assert out2["results"][0]["would_send"] is True
    # A preview still sends nothing and reports nothing as sent.
    assert out2["sent"] == 0 and out2["results"][0]["sent"] is False
