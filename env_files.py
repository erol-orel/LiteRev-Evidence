"""Where configuration comes from: one list, one parser, one precedence, for every entry
point.

There were SEVEN hand-rolled env loaders, each with its own list of files in its own
order, and no two of them agreed:

    api/core.py                      .env, /opt/literev-api/.env, /etc/literev/env,
                                     /etc/literev/secrets, /opt/literev-api/secrets.env
    alembic/env.py                   ./.env, /etc/literev-api.env, /etc/literev/secrets,
                                     /opt/literev-api/secrets.env
    embed_corpus.py                  .env, <script>/.env, /etc/literev/env,
                                     /opt/literev-api/.env, then the two secrets files
    scripts/deduplicate_corpus.py    .env, /opt/literev-api/.env, secrets.env,
                                     /etc/literev/secrets
    scripts/extract_pico_batch.py    /etc/literev-api.env, then others
    living_review_scheduler.py       /opt/literev-api/secrets.env, <repo>/secrets.env
    tools/*.py                       five paths in a fifth order

`/etc/literev-api.env`, the file the systemd unit actually passes to the service, appeared
in three of those seven lists and was FIRST in none. The service survived that because
systemd puts the file's contents in the process environment before Python starts and every
loader refuses to override an existing variable, so the canonical values always won. The
cost was paid elsewhere: a WRITE_API_KEY sitting in `/opt/literev-api/secrets.env` was
rejected by the very API running on that machine, because systemd's copy had won and
nothing said the file was stale. `alembic/env.py` read the repository `.env` FIRST, so a
DB_URL there would have migrated one database while the API used another.

Hence this module. Not a new loader: the only one.

  * `SEARCH_PATH` is the union of all seven lists, deduplicated, with `CANONICAL` first.
    Every path any entry point used is still read, so no setting stops being found, and
    the canonical file now wins when two of them disagree.
  * `apply` and `load_env` never override a variable already in the environment, so
    systemd keeps the last word, and `DB_URL=... python3 scripts/x.py` and CI's inline
    environment keep working untouched.
  * One parser, `_parse`. Two parsers would mean the audit could report agreement where
    the loader disagrees. It is stdlib-only, so the API boot does not depend on
    python-dotenv.
  * `sources`, `secondary_files` and `drift` answer "where does this setting come from"
    and "which file is lying", without ever returning a value: `scripts/env_audit.py`
    prints them and /health counts them.

Consolidating onto one file is now a decision someone can make with data, by reading
`scripts/env_audit.py --live` and moving what it names. Until then, reading all six is
what keeps the deployment working.
"""
from __future__ import annotations

import hashlib
import os
from typing import Iterable

#: The file the systemd unit passes to the service. Not a preference: a statement of where
#: production already reads from, and therefore the one place a setting should live.
CANONICAL = "/etc/literev-api.env"

#: The repository-local file, for a development machine. Second, so that on a server where
#: both exist the canonical one wins. It used to be first in three of the seven lists.
LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

#: Everything else any loader read, in descending precedence. Still read so that a setting
#: living only here keeps working; `secondary_files()` names them so they can be retired
#: deliberately rather than discovered during an incident.
SECONDARY = (
    "/opt/literev-api/.env",
    "/etc/literev/env",
    "/etc/literev/secrets",
    "/opt/literev-api/secrets.env",
)

#: The load order. First to define a variable wins, and nothing overrides the environment.
SEARCH_PATH = (CANONICAL, LOCAL, *SECONDARY)


def _parse(path: str) -> dict[str, str] | None:
    """A `.env` file as a mapping, or None if it cannot be read.

    `KEY=value`, `#` comments, an optional `export`, and ONE matched pair of surrounding
    quotes removed. Not a shell.

    The quoting rule is deliberate. The loader this replaces did
    `value.strip().strip('"').strip("'")`, which strips quote characters greedily from both
    ends: a password ending in a quote came out shortened, silently, and only at that one
    call site. A matched pair, or nothing."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    except OSError:
        return None
    out: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, sep, value = line.partition("=")
        if not sep:
            continue
        name = name.strip()
        if not name or not name.replace("_", "").isalnum():
            continue
        value = value.strip().rstrip("\r")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[name] = value
    return out


def apply(path: str) -> list[str]:
    """Set from `path` only the variables the environment does not already have.

    Returns the names it set. An absent or unreadable file sets nothing and is not an
    error: a development machine has no /etc file."""
    parsed = _parse(path)
    if not parsed:
        return []
    applied = []
    for name, value in parsed.items():
        if name not in os.environ:
            os.environ[name] = value
            applied.append(name)
    return applied


def load_env(extra: Iterable[str] = ()) -> list[str]:
    """Load `SEARCH_PATH` into `os.environ`. Returns the files that existed and were read.

    Called by `api/core.py` at import, by `alembic/env.py`, and by the scripts and tools
    that run outside the service and get no environment of their own. Because nothing is
    overridden, calling it twice, or after systemd has already supplied everything, is a
    no-op."""
    loaded: list[str] = []
    for path in (*SEARCH_PATH, *extra):
        if path and os.path.isfile(path):
            apply(path)
            loaded.append(path)
    return loaded


def sources(paths: Iterable[str] | None = None) -> dict[str, list[str]]:
    """{variable name: the files defining it, in precedence order}. Names only, no values.

    The first file listed for a name is the one in force, unless the environment already
    had that variable, which only `--live` can tell (systemd wins over every file)."""
    out: dict[str, list[str]] = {}
    for path in (paths if paths is not None else SEARCH_PATH):
        parsed = _parse(path)
        if not parsed:
            continue
        for name in parsed:
            out.setdefault(name, []).append(path)
    return out


def _fingerprint(value: str) -> str:
    """Twelve hex characters of SHA-256: enough to tell two values apart, useless for
    recovering either, which is what makes it safe to print, log and paste."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def secondary_files() -> list[str]:
    """Files other than the canonical one that still define settings.

    Not broken, and not ignored: each is a place a setting can hide, and the reason the
    same variable could be edited in the wrong file for an afternoon. Empty is the goal."""
    if not _parse(CANONICAL):
        return []                                     # a dev machine: these ARE the config
    return [path for path in (LOCAL, *SECONDARY)
            if path != CANONICAL and _parse(path)]


def drift(environ: dict[str, str] | None = None) -> dict[str, dict[str, str]]:
    """Variables where a file on disk does not match the environment in force.

    {name: {"environment": fingerprint or "unset", path: fingerprint, ...}}, fingerprints
    only. Two shapes, both of which were live on the server:

      - a file defines the variable differently from the running process, so the file
        describes a system that is not running (the stale WRITE_API_KEY, which cost an
        afternoon of 401s);
      - a file defines a variable the process does not have at all, so editing it there
        does nothing.

    A variable the environment has and no file mentions is NOT drift: systemd and the
    shell may legitimately supply one."""
    env = os.environ if environ is None else environ
    out: dict[str, dict[str, str]] = {}
    for name, files in sources().items():
        current = env.get(name)
        row = {path: _fingerprint((_parse(path) or {}).get(name, ""))
               for path in files
               if (_parse(path) or {}).get(name, None) != current}
        if row:
            row["environment"] = "unset" if current is None else _fingerprint(current)
            out[name] = row
    return out
