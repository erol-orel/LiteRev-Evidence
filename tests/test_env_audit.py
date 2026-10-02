"""The audit has to find the real thing: a file that disagrees with the running process.

What it is for: on the production server a 64-character WRITE_API_KEY read out of
`/opt/literev-api/secrets.env` returned 401, because the key in force came from
`/etc/literev-api.env`. Nothing on the machine said so. This script has to say so, in one
command, without printing either key.
"""
import importlib.util
import pathlib
import sys

import pytest

import env_files

_SPEC = importlib.util.spec_from_file_location(
    "env_audit", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "env_audit.py")
ea = importlib.util.module_from_spec(_SPEC)
sys.modules["env_audit"] = ea
_SPEC.loader.exec_module(ea)


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A fake deployment: a canonical file, a legacy file, and a little source tree."""
    canonical = tmp_path / "etc.env"
    legacy = tmp_path / "legacy.env"
    local = tmp_path / "repo.env"
    search = (str(canonical), str(local), str(legacy))
    for module in (env_files, ea):
        monkeypatch.setattr(module, "CANONICAL", str(canonical), raising=False)
        monkeypatch.setattr(module, "LOCAL", str(local), raising=False)
        monkeypatch.setattr(module, "SECONDARY", (str(legacy),), raising=False)
        monkeypatch.setattr(module, "SEARCH_PATH", search, raising=False)

    root = tmp_path / "src"
    (root / "api").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "scripts" / "archive").mkdir(parents=True)
    (root / "api" / "core.py").write_text(
        'import os\n'
        'KEY = os.getenv("WRITE_API_KEY")\n'
        'URL = os.environ.get("DB_URL", "")\n'
        'HOST = os.environ["SMTP_HOST"]\n',
        encoding="utf-8")
    (root / "tests" / "conftest.py").write_text(
        'import os\nos.getenv("A_TEST_ONLY_VARIABLE")\n', encoding="utf-8")
    (root / "scripts" / "archive" / "old.py").write_text(
        'import os\nos.getenv("A_RETIRED_VARIABLE")\n', encoding="utf-8")
    return {"root": root, "canonical": canonical, "legacy": legacy, "local": local}


# ── finding what the code reads ──────────────────────────────────────────────
def test_all_three_ways_of_reading_the_environment_are_found(machine):
    found = ea.used_variables(str(machine["root"]))
    assert set(found) == {"WRITE_API_KEY", "DB_URL", "SMTP_HOST"}
    assert found["WRITE_API_KEY"]["at"].endswith("core.py:2"), "a finding must be locatable"


def test_the_test_suite_and_the_archive_are_not_the_services_configuration(machine):
    found = ea.used_variables(str(machine["root"]))
    assert "A_TEST_ONLY_VARIABLE" not in found
    assert "A_RETIRED_VARIABLE" not in found


# ── the findings ─────────────────────────────────────────────────────────────
def test_a_file_disagreeing_with_the_live_process_is_stale(machine, monkeypatch):
    machine["canonical"].write_text("WRITE_API_KEY=in-force\n", encoding="utf-8")
    machine["legacy"].write_text("WRITE_API_KEY=stale\n", encoding="utf-8")
    monkeypatch.setattr(ea, "live_environment",
                        lambda pid: ({"WRITE_API_KEY": "in-force"}, "pid 1"))
    report = ea.audit(str(machine["root"]), None, True)
    assert report["findings"]["stale"] == ["WRITE_API_KEY"]
    row = next(r for r in report["rows"] if r["name"] == "WRITE_API_KEY")
    assert row["stale_in"] == [str(machine["legacy"])]
    assert row["effective_from"] == str(machine["canonical"])


def test_a_setting_only_a_shadow_file_defines_is_reported(machine, monkeypatch):
    """Editing SMTP_HOST there changes nothing, and nothing says so."""
    machine["canonical"].write_text("WRITE_API_KEY=x\n", encoding="utf-8")
    machine["legacy"].write_text("SMTP_HOST=smtp.example.org\n", encoding="utf-8")
    monkeypatch.setattr(ea, "live_environment",
                        lambda pid: ({"WRITE_API_KEY": "x"}, "pid 1"))
    report = ea.audit(str(machine["root"]), None, True)
    assert report["findings"]["scattered"] == ["SMTP_HOST"]
    assert report["secondary_files"] == [str(machine["legacy"])]


def test_agreement_is_a_clean_report(machine, monkeypatch):
    machine["canonical"].write_text(
        "WRITE_API_KEY=a\nDB_URL=b\nSMTP_HOST=c\n", encoding="utf-8")
    monkeypatch.setattr(ea, "live_environment", lambda pid: (
        {"WRITE_API_KEY": "a", "DB_URL": "b", "SMTP_HOST": "c"}, "pid 1"))
    report = ea.audit(str(machine["root"]), None, True)
    assert report["findings"] == {"stale": [], "scattered": [], "missing": []}


def test_nothing_is_missing_on_a_machine_with_no_canonical_file(machine):
    """A report that fires on every developer's laptop is a report nobody reads."""
    report = ea.audit(str(machine["root"]), None, False)
    assert report["findings"]["missing"] == []
    assert report["canonical_present"] is False


def test_a_variable_the_code_reads_and_no_file_defines_is_missing(machine, monkeypatch):
    machine["canonical"].write_text("WRITE_API_KEY=a\nDB_URL=b\n", encoding="utf-8")
    monkeypatch.setattr(ea, "live_environment", lambda pid: (None, "not requested"))
    report = ea.audit(str(machine["root"]), None, False)
    assert report["findings"]["missing"] == ["SMTP_HOST"]


def test_a_variable_the_unit_supplies_inline_is_not_missing(machine, monkeypatch):
    """systemd can set a variable without any file, and that is legitimate."""
    machine["canonical"].write_text("WRITE_API_KEY=a\nDB_URL=b\n", encoding="utf-8")
    monkeypatch.setattr(ea, "live_environment", lambda pid: (
        {"WRITE_API_KEY": "a", "DB_URL": "b", "SMTP_HOST": "from-the-unit"}, "pid 1"))
    report = ea.audit(str(machine["root"]), None, True)
    assert report["findings"]["missing"] == []


# ── the command ──────────────────────────────────────────────────────────────
def test_the_exit_code_is_the_finding(machine, monkeypatch, capsys):
    machine["canonical"].write_text(
        "WRITE_API_KEY=a\nDB_URL=b\nSMTP_HOST=c\n", encoding="utf-8")
    monkeypatch.setattr(ea, "live_environment", lambda pid: (
        {"WRITE_API_KEY": "a", "DB_URL": "b", "SMTP_HOST": "c"}, "pid 1"))
    assert ea.main(["--root", str(machine["root"]), "--live"]) == 0
    assert "Nothing scattered, nothing stale" in capsys.readouterr().out

    machine["legacy"].write_text("WRITE_API_KEY=stale\n", encoding="utf-8")
    assert ea.main(["--root", str(machine["root"]), "--live"]) == 1
    out = capsys.readouterr().out
    assert "STALE" in out and str(machine["legacy"]) in out


def test_the_output_never_contains_a_value(machine, monkeypatch, capsys):
    secret = "sk-a-very-real-looking-secret"
    machine["canonical"].write_text("WRITE_API_KEY=in-force\n", encoding="utf-8")
    machine["legacy"].write_text(f"WRITE_API_KEY={secret}\n", encoding="utf-8")
    monkeypatch.setattr(ea, "live_environment",
                        lambda pid: ({"WRITE_API_KEY": "in-force"}, "pid 1"))
    for argv in (["--live"], ["--live", "--json"]):
        ea.main(["--root", str(machine["root"]), *argv])
        out = capsys.readouterr().out
        assert secret not in out and "in-force" not in out, argv


# ── required against optional ────────────────────────────────────────────────
def test_only_a_setting_that_raises_when_absent_counts_as_missing(machine):
    """Forty of this application's variables are tuning knobs read with `os.getenv` and a
    built-in default. Reporting every unset one would make a healthy server look broken,
    and a report that fires everywhere stops being read. A subscript raises; that is the
    line."""
    (machine["root"] / "api" / "extra.py").write_text(
        'import os\n'
        'REQUIRED = os.environ["REQUIRED_ONE"]\n'
        'OPTIONAL = os.getenv("OPTIONAL_ONE", "a default")\n'
        'MAYBE = os.getenv("MAYBE_ONE")\n'
        'os.environ["A_STUB_FOR_A_SUBPROCESS"] = "x"\n',
        encoding="utf-8")
    used = ea.used_variables(str(machine["root"]))
    assert used["REQUIRED_ONE"]["required"] is True
    assert used["OPTIONAL_ONE"]["required"] is False
    assert used["MAYBE_ONE"]["required"] is False
    assert "A_STUB_FOR_A_SUBPROCESS" not in used, "setting a variable is not reading one"

    machine["canonical"].write_text("WRITE_API_KEY=a\n", encoding="utf-8")
    report = ea.audit(str(machine["root"]), None, False)
    assert "REQUIRED_ONE" in report["findings"]["missing"]
    assert "OPTIONAL_ONE" not in report["findings"]["missing"]
    assert "OPTIONAL_ONE" in report["using_built_in_defaults"]
