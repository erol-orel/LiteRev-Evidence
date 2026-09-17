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


def _drive(monkeypatch, *, listed, total, smtp, dry_run, last_notified="set"):
    """Run the real digest loop with the two article helpers stubbed.

    `last_notified="set"` gives the subscription a real timestamp (the normal case);
    None makes it a never-notified row, which is the first-digest path."""
    from datetime import datetime
    from conftest import patch_app

    captured = {}

    def _fake_render(sid, articles, total_new, scenario_name=None, first_digest=False,
                     n_relevant=None, signals=None):
        captured["listed"], captured["total"] = len(articles), total_new
        captured["first_digest"] = first_digest
        captured["n_relevant"], captured["signals"] = n_relevant, signals
        return ("subject", "<html></html>", "text")

    _ln = datetime(2026, 1, 1) if last_notified == "set" else last_notified
    subs = [{"id": 1, "email": "a@b.c", "scenario_id": "usr-1",
             "frequency": "immediate", "last_notified_at": _ln}]
    patch_app(monkeypatch, "engine", _FakeEngine(subs))
    patch_app(monkeypatch, "_render_alert_digest", _fake_render)
    patch_app(monkeypatch, "_new_articles_for_scenario",
              lambda c, s, since, limit=25: _articles(listed))
    patch_app(monkeypatch, "_count_new_articles_for_scenario", lambda c, s, since: total)
    patch_app(monkeypatch, "_get_scenario_name", lambda s: "Chikungunya")
    patch_app(monkeypatch, "_send_email_smtp", lambda *a, **k: None)
    # The signals have their own tests; here they must simply not interfere.
    patch_app(monkeypatch, "_get_scenario_threshold", lambda sid: 0.45)
    patch_app(monkeypatch, "change_signals",
              lambda sid, since, thr: {"new_relevant": 2, "signals": []})
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


# ── the FIRST digest of a never-notified subscription ────────────────────────
def test_a_never_notified_subscription_does_not_call_the_whole_corpus_new(monkeypatch):
    """With `last_notified_at` NULL there is no lower bound, so nothing is "new since
    last time": the count matched the ENTIRE corpus and the email announced "2170
    nouveaux articles" for a scenario that had gained none since the subscription. A
    first digest is an introduction, so its total is what it shows."""
    out, captured = _drive(monkeypatch, listed=25, total=2170,
                           smtp="smtp.example.org", dry_run=False, last_notified=None)
    assert captured["first_digest"] is True
    assert captured["total"] == 25, "a first digest counts what it shows"
    assert captured["listed"] == 25
    assert out["results"][0]["new"] == 25
    assert out["results"][0]["first_digest"] is True


def test_a_normal_digest_still_reports_the_real_total(monkeypatch):
    """The first-digest rule must not swallow the fix above it: once there IS a lower
    bound, the real total is what the email announces."""
    out, captured = _drive(monkeypatch, listed=25, total=300,
                           smtp="smtp.example.org", dry_run=False)
    assert captured["first_digest"] is False
    assert captured["total"] == 300
    assert out["results"][0]["new"] == 300 and out["results"][0]["listed"] == 25


def test_the_first_digest_says_recent_not_new():
    """The wording has to match the arithmetic: "les plus récents", not "nouveaux"."""
    subj, html, text = main._render_alert_digest(
        "usr-1", _articles(25), 25, scenario_name="Chikungunya", first_digest=True)
    assert "nouveaux articles" not in subj
    assert "25 articles récents" in subj and "Chikungunya" in subj
    for body in (html, text):
        assert "Première notification" in body
        assert "25 25" not in body          # the count is not printed twice
    # Singular stays grammatical.
    subj1, _h, _t = main._render_alert_digest("usr-1", _articles(1), 1, first_digest=True)
    assert "1 article récent" in subj1


def test_a_normal_digest_keeps_its_wording_and_overflow_note():
    """The ordinary path is unchanged, intro included: it must stay absent."""
    subj, html, text = main._render_alert_digest(
        "usr-1", _articles(25), 300, scenario_name="Chikungunya")
    assert "300 nouveaux articles" in subj
    assert "300 300" not in html and "300 300" not in text
    assert "275 de plus" in html and "275 de plus" in text
    for body in (html, text):
        assert "Première notification" not in body


# ── How many of the new papers are RELEVANT, and what is worth rereading ─────
def test_the_digest_says_how_many_of_the_new_papers_clear_the_threshold():
    """"300 new articles" does not say whether three of them matter or none. The relevant
    count is the number a reviewer decides on."""
    _s, html, text = main._render_alert_digest(
        "s1", _articles(3), 300, scenario_name="Chik", n_relevant=12)
    for body in (html, text):
        assert "12 sur 300 passent le seuil" in body
    # When every new article is relevant, say that rather than "300 of 300".
    _s, _h, text_all = main._render_alert_digest("s1", _articles(3), 3, n_relevant=3)
    assert "Les 3 passent le seuil" in text_all
    # Without the count the sentence is simply absent, never a guess.
    _s, _h, text_none = main._render_alert_digest("s1", _articles(3), 3, n_relevant=None)
    assert "passent le seuil" not in text_none


def test_the_signals_invite_a_review_and_never_claim_the_model_changed():
    """These are computed from cached per-article facts with no LLM, so they cannot
    establish that a conclusion moved. The caveat travels with them, always."""
    _s, html, text = main._render_alert_digest(
        "s1", _articles(2), 10, n_relevant=10,
        signals=[{"code": "epidemic_parameters", "n": 3, "detail": "r0"},
                 {"code": "new_concepts", "n": 2, "detail": "Wolbachia"},
                 {"code": "strong_designs", "n": 1, "detail": "Systematic review"}])
    for body in (html, text):
        assert "3 articles rapportant un paramètre" in body
        assert "2 concepts absents de la carte" in body
        assert "1 devis qui relève" in body          # singular agreement, no "(s)"
        assert "pistes de relecture, pas un constat" in body
        assert "seule une régénération" in body
    # No signals, no block and no caveat dangling on its own.
    _s, _h, quiet = main._render_alert_digest("s1", _articles(2), 10, n_relevant=10, signals=[])
    assert "pistes de relecture" not in quiet and "À relire" not in quiet


def test_signals_agree_in_the_plural_as_well_as_the_singular():
    _s, _h, one = main._render_alert_digest(
        "s1", _articles(1), 1, signals=[{"code": "new_concepts", "n": 1, "detail": "x"}])
    _s2, _h2, many = main._render_alert_digest(
        "s1", _articles(1), 1, signals=[{"code": "new_concepts", "n": 4, "detail": "x"}])
    assert "1 concept absent de la carte" in one
    assert "4 concepts absents de la carte" in many
    assert "(s)" not in one and "(s)" not in many


def test_change_signals_never_break_the_digest(monkeypatch):
    """A failed signal computation must not stop an email going out: the articles are the
    point, the signals are an extra."""
    from conftest import patch_app

    class _Boom:
        def connect(self): raise RuntimeError("no database")

    patch_app(monkeypatch, "engine", _Boom())
    out = main.change_signals("usr-x", None, 0.45)
    assert out == {"new_relevant": 0, "signals": []}
