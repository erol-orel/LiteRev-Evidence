"""One file configures the service, and the tooling can prove which one.

These tests exist because of a concrete afternoon: `/opt/literev-api/secrets.env` held a
WRITE_API_KEY, the key was 64 clean characters, and the running API rejected it, because
the real one came from `/etc/literev-api.env`. Four files claimed to hold configuration,
`alembic/env.py` read them in the opposite precedence to the service while a comment
claimed it mirrored `main.py`, and `main.py` read none of them.

So the properties under test are not "it parses a .env file" but: a variable already in
the environment is never overridden (CI and `VAR=x python3 ...` must keep working), the
canonical file wins over a repository-local one, a file the service cannot see is NAMED,
and nothing reports a value.
"""
import os

import pytest

import env_files


@pytest.fixture
def files(tmp_path, monkeypatch):
    """A machine with the three kinds of file: canonical, repository-local, secondary."""
    canonical = tmp_path / "etc-literev-api.env"
    local = tmp_path / "repo.env"
    legacy = tmp_path / "opt-secrets.env"
    monkeypatch.setattr(env_files, "CANONICAL", str(canonical))
    monkeypatch.setattr(env_files, "LOCAL", str(local))
    monkeypatch.setattr(env_files, "SECONDARY", (str(legacy),))
    monkeypatch.setattr(env_files, "SEARCH_PATH",
                        (str(canonical), str(local), str(legacy)))
    return {"canonical": canonical, "local": local, "legacy": legacy}


# ── reading a file ───────────────────────────────────────────────────────────
def test_the_shapes_a_real_env_file_turns_out_to_have(tmp_path):
    path = tmp_path / "x.env"
    path.write_text(
        "# a comment\n"
        "\n"
        "PLAIN=value\n"
        "export EXPORTED=value2\n"
        'QUOTED="in double quotes"\n'
        "SINGLE='in single quotes'\n"
        "WITH_EQUALS=postgresql://u:p@h/db?x=1\n"
        "TRAILING=value3  \n"
        "CRLF=value4\r\n"
        "not a variable line\n"
        "lowercase=ignored-name-is-fine-though\n",
        encoding="utf-8")
    got = env_files._parse(str(path))
    assert got["PLAIN"] == "value"
    assert got["EXPORTED"] == "value2"
    assert got["QUOTED"] == "in double quotes"
    assert got["SINGLE"] == "in single quotes"
    assert got["WITH_EQUALS"] == "postgresql://u:p@h/db?x=1", "cut -d= -f2 was the bug"
    assert got["TRAILING"] == "value3"
    assert got["CRLF"] == "value4", "a CRLF file must not append a carriage return"
    assert "not a variable line" not in got


def test_a_file_that_cannot_be_read_is_not_an_error(tmp_path):
    """A development machine has no /etc file, and the service may not be able to read
    one it is nonetheless given by systemd."""
    assert env_files._parse(str(tmp_path / "absent")) is None


# ── loading ──────────────────────────────────────────────────────────────────
def test_what_is_already_in_the_environment_always_wins(files, monkeypatch):
    """`DB_URL=... python3 scripts/x.py` and CI's inline environment must survive. The
    integration tests truncate the tables of DB_URL, so a file silently overriding it
    would mean a test run against the wrong database."""
    files["canonical"].write_text("DB_URL=from-the-file\n", encoding="utf-8")
    monkeypatch.setenv("DB_URL", "from-the-caller")
    env_files.load_env()
    assert os.environ["DB_URL"] == "from-the-caller"


def test_the_canonical_file_wins_over_the_one_in_the_repository(files, monkeypatch):
    """The production bug, inverted: the repo file used to be read FIRST, so a stale copy
    shadowed the file the service actually runs on."""
    files["canonical"].write_text("WRITE_API_KEY=the-real-one\n", encoding="utf-8")
    files["local"].write_text("WRITE_API_KEY=the-stale-one\n", encoding="utf-8")
    monkeypatch.delenv("WRITE_API_KEY", raising=False)
    loaded = env_files.load_env()
    assert os.environ["WRITE_API_KEY"] == "the-real-one"
    assert loaded == [str(files["canonical"]), str(files["local"])]


def test_loading_reports_which_files_it_actually_read(files):
    files["canonical"].write_text("A=1\n", encoding="utf-8")
    assert env_files.load_env() == [str(files["canonical"])], "the absent one is skipped"


# ── saying where a setting comes from ────────────────────────────────────────
def test_sources_lists_every_file_that_defines_a_name_in_precedence_order(files):
    files["canonical"].write_text("SHARED=a\nONLY_CANONICAL=b\n", encoding="utf-8")
    files["local"].write_text("SHARED=c\n", encoding="utf-8")
    files["legacy"].write_text("SHARED=d\nONLY_LEGACY=e\n", encoding="utf-8")
    got = env_files.sources()
    assert got["SHARED"] == [str(files["canonical"]), str(files["local"]),
                             str(files["legacy"])]
    assert got["ONLY_CANONICAL"] == [str(files["canonical"])]
    assert got["ONLY_LEGACY"] == [str(files["legacy"])]


def test_a_file_other_than_the_canonical_one_is_named(files):
    files["canonical"].write_text("A=1\n", encoding="utf-8")
    files["legacy"].write_text("A=2\n", encoding="utf-8")
    assert env_files.secondary_files() == [str(files["legacy"])]


def test_the_repository_file_counts_as_secondary_only_beside_a_canonical_one(files):
    """On a development machine the repo file IS the configuration, and calling it a
    shadow would be wrong. On a server, where both exist, it is silent."""
    files["local"].write_text("A=1\n", encoding="utf-8")
    assert env_files.secondary_files() == []
    files["canonical"].write_text("A=2\n", encoding="utf-8")
    assert env_files.secondary_files() == [str(files["local"])]


def test_an_empty_file_is_not_counted(files):
    files["canonical"].write_text("A=1\n", encoding="utf-8")
    files["legacy"].write_text("# nothing but a comment\n", encoding="utf-8")
    assert env_files.secondary_files() == []


# ── the afternoon this is meant to prevent ───────────────────────────────────
def test_drift_catches_a_file_that_disagrees_with_the_running_process(files):
    """THE production case: secrets.env's WRITE_API_KEY was not the one the API held, so
    a curl built from that file returned 401 and the file looked right."""
    files["legacy"].write_text("WRITE_API_KEY=stale-64-chars\n", encoding="utf-8")
    got = env_files.drift({"WRITE_API_KEY": "the-one-in-force"})
    assert str(files["legacy"]) in got["WRITE_API_KEY"]
    assert got["WRITE_API_KEY"]["environment"] == env_files._fingerprint("the-one-in-force")


def test_drift_catches_a_setting_that_exists_only_in_a_file(files):
    """Editing it there does nothing: the process does not have the variable at all."""
    files["legacy"].write_text("SMTP_HOST=smtp.example.org\n", encoding="utf-8")
    got = env_files.drift({})
    assert got["SMTP_HOST"]["environment"] == "unset"


def test_a_variable_the_unit_sets_and_no_file_mentions_is_not_drift(files):
    """Legitimate: systemd can set a variable inline, and the shell can export one."""
    files["canonical"].write_text("A=1\n", encoding="utf-8")
    assert env_files.drift({"A": "1", "SET_BY_THE_UNIT": "x"}) == {}


def test_agreement_is_silence(files):
    files["canonical"].write_text("A=1\nB=2\n", encoding="utf-8")
    assert env_files.drift({"A": "1", "B": "2"}) == {}


def test_nothing_reported_ever_contains_a_value(files):
    """This output is printed, logged and pasted into chats. A fingerprint is enough to
    tell two values apart and useless for recovering either."""
    secret = "sk-a-very-real-looking-secret-value"
    files["legacy"].write_text(f"OPENAI_API_KEY={secret}\n", encoding="utf-8")
    rendered = repr(env_files.drift({"OPENAI_API_KEY": "something-else"}))
    assert secret not in rendered and "something-else" not in rendered
    assert len(env_files._fingerprint(secret)) == 12


def test_the_fingerprint_separates_absent_from_empty_from_set():
    assert env_files._fingerprint("") != env_files._fingerprint("x")
    assert env_files._fingerprint("x") == env_files._fingerprint("x")


# ── the real list, not a fixture ─────────────────────────────────────────────
def test_the_canonical_file_is_first_in_the_real_search_path():
    """The whole bug in one assertion: /etc/literev-api.env is the file systemd passes to
    the service, and it was FIRST in none of the seven lists that used to exist."""
    assert env_files.SEARCH_PATH[0] == env_files.CANONICAL == "/etc/literev-api.env"
    assert env_files.SEARCH_PATH[1] == env_files.LOCAL


def test_no_path_any_previous_loader_read_was_dropped():
    """Replacing seven loaders with one must not stop a setting from being found. These
    are the paths those loaders read, from api/core.py, embed_corpus.py, alembic/env.py,
    deduplicate_corpus.py, extract_pico_batch.py, living_review_scheduler.py and tools/.
    A setting living only in one of them keeps working; env_audit names it so it can be
    moved deliberately."""
    previously_read = {
        "/etc/literev-api.env", "/opt/literev-api/.env", "/etc/literev/env",
        "/etc/literev/secrets", "/opt/literev-api/secrets.env",
    }
    search = set(env_files.SEARCH_PATH)
    assert previously_read <= search, previously_read - search
    assert env_files.LOCAL in search, "the repository .env was read by five of the seven"


def test_applying_a_file_touches_only_what_is_missing(tmp_path, monkeypatch):
    path = tmp_path / "x.env"
    path.write_text("ALREADY_SET=from-the-file\nNOT_SET=from-the-file\n", encoding="utf-8")
    monkeypatch.setenv("ALREADY_SET", "from-the-environment")
    monkeypatch.delenv("NOT_SET", raising=False)
    assert env_files.apply(str(path)) == ["NOT_SET"]
    assert os.environ["ALREADY_SET"] == "from-the-environment"
    assert os.environ["NOT_SET"] == "from-the-file"


def test_applying_an_absent_file_is_a_no_op(tmp_path):
    assert env_files.apply(str(tmp_path / "absent")) == []
