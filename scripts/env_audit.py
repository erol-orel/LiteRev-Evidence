#!/usr/bin/env python3
"""Where every setting comes from, and which files are lying about it.

Four files on the production server held configuration and nothing said which one the
service read. The symptom was a WRITE_API_KEY taken from `/opt/literev-api/secrets.env`
that the running API rejected, because the real one came from `/etc/literev-api.env`.
That is a five-minute confusion when you know to look, and an afternoon when you do not.

This answers it in one command, for all of the variables the code actually reads, and
never prints a value: files and the live process are compared by fingerprint, twelve hex
characters of SHA-256, which is enough to tell two values apart and useless for recovering
either. So the output is safe to paste into a chat or an issue.

    python3 scripts/env_audit.py                 # files on this machine
    sudo python3 scripts/env_audit.py --live      # also compare the running API
    python3 scripts/env_audit.py --json

Exits 1 when it finds something that will waste someone's time:

  stale       a file and the live process disagree, so the file describes a system that is
              not running. This is the 401: the value is there, it reads correctly, and it
              is not the one in force;
  scattered   a setting that lives outside the canonical file. It IS applied, since every
              file in the search path is read, but it is applied from a place nobody will
              look. Moving it into the canonical file is the fix;
  missing     a variable the code reads and no file defines (it may still be set by the
              unit or the shell; --live says which).

`--live` reads /proc/<pid>/environ, which needs root, and is the only authority on what
the service actually has: the API gets its environment from systemd before it starts, so
no file can be trusted to describe it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from env_files import (CANONICAL, LOCAL, SEARCH_PATH, SECONDARY,  # noqa: E402
                       _fingerprint, _parse, secondary_files, sources)

#: `os.getenv("X")` and `os.environ.get("X")` (group 1) against `os.environ["X"]` (group
#: 2). The distinction decides whether a variable being undefined is a FINDING: a
#: subscript raises KeyError, so the setting is required; `getenv` returns None, so the
#: code has already decided what to do without it, and most of these are tuning knobs with
#: a built-in default. Reporting all forty of them would make a healthy server look broken,
#: which is how a report stops being read.
#:
#: Deliberately a regex and not an import of the application: this has to run on a server
#: where the heavy ML dependencies are not installed, and it has to see the variables read
#: at import time, before any of them could be inspected.
_USE = re.compile(r"""os\.(?:getenv|environ\.get)\(\s*["']([A-Z][A-Z_0-9]*)["']"""
                  r"""|os\.environ\[\s*["']([A-Z][A-Z_0-9]*)["']\s*\]""")

#: Directories whose variables are not the service's: CI fixtures, retired one-off
#: scripts, and the test suite's own stubs. This file is skipped as well, because the
#: regex above matches its own documentation: a scanner that reports itself teaches
#: whoever reads the output to discount it.
_SKIP_DIRS = ("/tests/", "/scripts/archive/", "/.venv/", "/node_modules/", "/alembic/",
              "/scripts/env_audit.py")

#: Variables that belong to the environment rather than to this application. Reading them
#: is normal and their absence is not a finding.
_NOT_OURS = {"PATH", "HOME", "PWD", "USER", "TERM", "TMPDIR", "LANG", "CI", "PORT",
             "VIRTUAL_ENV", "PYTHONPATH", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"}


def used_variables(root: str) -> dict[str, dict]:
    """{name: {"at": "path:line", "required": bool}} over the application's own Python.

    The location matters: a finding nobody can locate gets ignored. `required` is true only
    where the value is read by subscript, which raises when it is absent."""
    found: dict[str, dict] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "__"))
                       and d not in ("node_modules", "frontend", "venv")]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            rel = "/" + os.path.relpath(path, root).replace(os.sep, "/")
            if any(skip in rel for skip in _SKIP_DIRS):
                continue
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    lines = fh.readlines()
            except OSError:
                continue
            for number, line in enumerate(lines, start=1):
                for match in _USE.finditer(line):
                    name = match.group(1) or match.group(2)
                    if not name or name in _NOT_OURS:
                        continue
                    # `os.environ["X"] = y` SETS a variable for a subprocess; it is not a
                    # setting the deployment has to supply. Counting it as a required read
                    # made a benchmark script's own stub look like a missing production key.
                    after = line[match.end():].lstrip()
                    if match.group(2) is not None and after.startswith("=") \
                            and not after.startswith("=="):
                        continue
                    entry = found.setdefault(
                        name, {"at": f"{rel.lstrip('/')}:{number}", "required": False})
                    # One required read anywhere makes it required everywhere.
                    entry["required"] = entry["required"] or match.group(2) is not None
    return found


def live_environment(pid: int | None) -> tuple[dict[str, str] | None, str]:
    """The running API's own environment, and a note saying how it was obtained.

    This is the only authority: the service is handed its environment by systemd before
    the process starts, so a file on disk can describe it but cannot prove it."""
    if pid is None:
        pid = _find_api_pid()
        if pid is None:
            return None, "no running API process found (pass --pid)"
    try:
        with open(f"/proc/{pid}/environ", "rb") as fh:
            raw = fh.read()
    except PermissionError:
        return None, f"/proc/{pid}/environ needs root"
    except OSError as exc:
        return None, f"/proc/{pid}/environ: {exc}"
    env = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        name, _, value = entry.decode("utf-8", "replace").partition("=")
        env[name] = value
    return env, f"pid {pid}"


def _find_api_pid() -> int | None:
    """The uvicorn process serving `main:app`, without asking systemd: this has to work
    the same whether the service is managed by a unit, a container or a shell."""
    for entry in sorted(os.listdir("/proc")):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                cmdline = fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if "uvicorn" in cmdline and "main:app" in cmdline:
            return int(entry)
    return None


def audit(root: str, pid: int | None, check_live: bool) -> dict:
    used = used_variables(root)
    defined = sources()
    files = {path: _parse(path) for path in SEARCH_PATH}
    live, live_note = (live_environment(pid) if check_live else (None, "not requested"))
    canonical_present = bool(files.get(CANONICAL))

    rows = []
    findings: dict[str, list[str]] = {"stale": [], "scattered": [], "missing": []}
    defaulted: list[str] = []
    for name in sorted(set(used) | set(defined)):
        in_files = defined.get(name, [])
        use = used.get(name) or {}
        row = {
            "name": name,
            "read_at": use.get("at"),
            "required": bool(use.get("required")),
            # In precedence order, so the first is the file that wins among the files. The
            # environment still beats all of them, which only --live can show.
            "defined_in": in_files,
            "effective_from": in_files[0] if in_files else None,
        }
        if live is not None:
            current = live.get(name)
            row["live"] = "unset" if current is None else _fingerprint(current)
            disagreeing = [p for p in in_files
                           if (files.get(p) or {}).get(name, None) != current]
            if disagreeing:
                row["stale_in"] = disagreeing
                findings["stale"].append(name)
        # A setting outside the canonical file is applied, but from a place nobody checks.
        if canonical_present and in_files and CANONICAL not in in_files:
            findings["scattered"].append(name)
        # "Defined nowhere" only means something when there IS somewhere to define it. On a
        # development machine with no canonical file, every variable would be reported, and
        # a report that fires on every machine is a report nobody reads.
        nowhere = (canonical_present and name in used and not in_files
                   and (live is None or name not in (live or {})))
        if nowhere:
            (findings["missing"] if row["required"] else defaulted).append(name)
        rows.append(row)

    return {
        "canonical": CANONICAL,
        "canonical_present": canonical_present,
        "search_path": list(SEARCH_PATH),
        "secondary_files": secondary_files(),
        "live": live_note,
        "variables_read_by_the_code": len(used),
        # Read with os.getenv, defined nowhere: each falls back to its built-in default.
        # Normal, and listed only so the count is not mistaken for a problem.
        "using_built_in_defaults": defaulted,
        "findings": findings,
        "rows": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true",
                        help="also read the running API's environment (needs root)")
    parser.add_argument("--pid", type=int, default=None,
                        help="the API process, when autodetection cannot find it")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--root", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".."))
    args = parser.parse_args(argv)

    report = audit(os.path.abspath(args.root), args.pid, args.live or args.pid is not None)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print(report)
    return 1 if any(report["findings"].values()) else 0


def _print(report: dict) -> None:
    print(f"canonical file : {report['canonical']}"
          f"{'' if report['canonical_present'] else '   NOT PRESENT on this machine'}")
    print(f"live process   : {report['live']}")
    print(f"variables read : {report['variables_read_by_the_code']}")
    if report["secondary_files"]:
        print("\nfiles other than the canonical one that still define settings:")
        for path in report["secondary_files"]:
            print(f"  {path}")
        print("  All are read, in the order above. Empty is the goal: one file.")

    findings = report["findings"]
    labels = {
        "stale": "a file DISAGREES with the running process (the file is not what runs)",
        "scattered": "in force, but from a file other than the canonical one",
        "missing": "REQUIRED (read by subscript, raises when absent) and defined nowhere",
    }
    by_name = {row["name"]: row for row in report["rows"]}
    for kind, label in labels.items():
        names = findings.get(kind) or []
        if not names:
            continue
        print(f"\n{kind.upper()}: {label}")
        for name in names:
            row = by_name[name]
            where = ", ".join(row.get("stale_in") or row["defined_in"]) or "-"
            read_at = f"  read at {row['read_at']}" if row.get("read_at") else ""
            live = f"  live={row['live']}" if "live" in row else ""
            print(f"  {name:28} {where}{live}{read_at}")

    defaulted = report.get("using_built_in_defaults") or []
    if defaulted:
        print(f"\n{len(defaulted)} optional settings are defined in no file and use their "
              "built-in default.\n  This is normal. --json lists them.")

    if not any(findings.values()):
        print("\nNothing scattered, nothing stale. One file, and it is the one in force.")


if __name__ == "__main__":
    raise SystemExit(main())
