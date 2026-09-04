"""The deploy smoke test's health assertion — the check that has to FAIL correctly.

A smoke test that cannot fail is decoration. The property worth proving is not that a
healthy response passes; it is that each broken one is REJECTED, and rejected with a
message someone reading a failed deploy log at 22:41 can act on.

The case this exists for: the startup DDL fails open by design, so a database missing a
table answers /health with `status: ok` while /user-scenarios, /gesica/scenarios and
/corpus/fulltext-stats return 500. That is the fresh-database bug fixed in e060f3a, and a
deploy onto it would have gone green on the old `grep '"status":"ok"'`.
"""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import check_health  # noqa: E402

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "scripts", "check_health.py")

#: The real production response, copied from the deploy log of 39759fe.
HEALTHY = {"status": "ok", "database": "ok",
           "schema": {"ok": True, "missing_tables": [], "ddl_failures": 0}}

#: What a fresh database looked like before e060f3a: 200 on /health, 500 on three
#: endpoints, and the old smoke test passing anyway.
FRESH_DB_BUG = {"status": "ok", "database": "ok",
                "schema": {"ok": False, "missing_tables": ["article_scenarios"],
                           "ddl_failures": 1,
                           "details": ["_ensure_user_scenarios_table: ALTER TABLE ..."]}}


def test_the_real_production_response_passes():
    assert check_health.problems(HEALTHY) == []


def test_the_bug_this_check_exists_for_is_caught():
    found = check_health.problems(FRESH_DB_BUG)
    assert found, "a fresh database missing article_scenarios must fail the deploy"
    assert "article_scenarios" in found[0], "the log must name the missing table"


def test_the_old_grep_would_have_passed_the_broken_response():
    """Pins WHY this replaced a grep, so nobody reverts it as over-engineering."""
    body = json.dumps(FRESH_DB_BUG, separators=(",", ":"))
    assert '"status":"ok"' in body          # the old check passed …
    assert check_health.problems(FRESH_DB_BUG)   # … while the app was unusable


@pytest.mark.parametrize("health, expect", [
    ({"status": "degraded", "database": "ok", "schema": {"ok": True}}, "status="),
    ({"status": "ok", "database": "down", "schema": {"ok": True}}, "database="),
    ({"status": "ok", "database": "ok", "schema": {"ok": False}}, "SCHEMA DEGRADED"),
])
def test_each_kind_of_unhealthy_is_named_in_the_message(health, expect):
    found = check_health.problems(health)
    assert found and any(expect in p for p in found), found


def test_every_problem_is_reported_not_just_the_first():
    """The deploy log is the only thing anyone reads; one reason at a time wastes a cycle."""
    found = check_health.problems({"status": "bad", "database": "bad", "schema": {"ok": False}})
    assert len(found) == 3


def test_a_response_with_no_schema_block_is_rejected():
    """An older build predating the schema check would otherwise pass having checked nothing."""
    found = check_health.problems({"status": "ok", "database": "ok"})
    assert found and "schema" in found[0]


def test_a_truthy_non_true_schema_ok_is_not_accepted():
    """`ok: "false"` and `ok: 1` are not True; `is not True` is deliberate, not a typo."""
    for value in ("false", "true", 1, "yes"):
        assert check_health.problems(
            {"status": "ok", "database": "ok", "schema": {"ok": value}})


def test_a_non_object_response_is_rejected():
    for junk in ([], "ok", None, 3):
        assert check_health.problems(junk)


# ── the script as the deploy actually invokes it ─────────────────────────────
def _run(payload, tmp_path, as_stdin=False):
    p = tmp_path / "health.json"
    p.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    if as_stdin:
        return subprocess.run([sys.executable, SCRIPT, "-"], input=p.read_text(),
                              capture_output=True, text=True)
    return subprocess.run([sys.executable, SCRIPT, str(p)], capture_output=True, text=True)


def test_exit_code_zero_on_a_healthy_response(tmp_path):
    r = _run(HEALTHY, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "health OK" in r.stdout


def test_exit_code_one_fails_the_deploy_step(tmp_path):
    """`set -e` in the smoke test turns a non-zero exit into a failed deploy."""
    r = _run(FRESH_DB_BUG, tmp_path)
    assert r.returncode == 1
    assert "SMOKE TEST FAILED" in r.stderr and "article_scenarios" in r.stderr


def test_unparseable_output_fails_rather_than_passing(tmp_path):
    """A truncated body or an HTML error page must not read as healthy."""
    r = _run("<html>502 Bad Gateway</html>", tmp_path)
    assert r.returncode == 1 and "could not read" in r.stderr


def test_it_also_accepts_the_response_on_stdin(tmp_path):
    assert _run(HEALTHY, tmp_path, as_stdin=True).returncode == 0


def test_wrong_arguments_exit_distinctly(tmp_path):
    """2, not 1: a usage error is not the same as an unhealthy app."""
    assert subprocess.run([sys.executable, SCRIPT], capture_output=True).returncode == 2
