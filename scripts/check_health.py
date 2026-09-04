#!/usr/bin/env python3
"""Assert that a /health response describes a HEALTHY app. Exit 1 with a reason if not.

Run by the deploy smoke test (.github/workflows/deploy.yml) against the freshly restarted
API. It exists as a file rather than a shell one-liner for two reasons: the check is real
logic and deserves tests (tests/test_check_health.py), and embedding it in the workflow
meant a python heredoc inside a bash heredoc inside a YAML block scalar — three levels of
indentation, any of which a later edit could silently break.

What it adds over the `grep '"status":"ok"'` it replaced
--------------------------------------------------------
The startup DDL fails OPEN by design: a server that refuses to boot is worse than a
degraded one. The cost is that a database missing a table answered /health with
`status: ok` while /user-scenarios, /gesica/scenarios and /corpus/fulltext-stats all
returned 500. That is not hypothetical — it is the fresh-database bug fixed in e060f3a,
and the last of those endpoints is what this very smoke test calls next.

`status` deliberately stays "ok" in that state so that a pre-existing degradation cannot
fail every deployment (including the one carrying the fix). `schema.ok` is the field that
tells the truth, and this script is what makes it BLOCKING — enabled once production was
confirmed clean (deploy 39759fe: ok=true, no missing tables, no dropped DDL).

Parsed as JSON rather than grepped: `grep '"ok":true'` would match any future boolean
named "ok" anywhere in the response, and `grep '"schema":{"ok":true'` would depend on key
order. Both would pass a degraded app.

Usage:  python3 check_health.py /tmp/literev_health.json
        curl -fsS localhost:8000/health | python3 check_health.py -
"""
from __future__ import annotations

import json
import sys


def problems(health) -> list[str]:
    """Every reason this response is not healthy. Empty list == healthy.

    Collects ALL of them rather than returning on the first: when a deploy fails at
    22:41 the log is the only thing anyone reads, and "status is degraded" alone sends
    the reader back to the server for the detail that was right there.
    """
    if not isinstance(health, dict):
        return [f"/health returned {type(health).__name__}, not an object"]
    out: list[str] = []
    if health.get("status") != "ok":
        out.append(f"status={health.get('status')!r} (expected 'ok')")
    if health.get("database") != "ok":
        out.append(f"database={health.get('database')!r} (expected 'ok')")

    schema = health.get("schema")
    if not isinstance(schema, dict):
        # An older build predating the schema block would otherwise pass silently, and
        # the deploy would go green having checked nothing it was added to check.
        out.append("no `schema` block in /health — is the deployed build older than the "
                   "schema integrity check?")
    elif schema.get("ok") is not True:
        missing = schema.get("missing_tables") or []
        out.append(
            f"SCHEMA DEGRADED: missing_tables={missing} "
            f"ddl_failures={schema.get('ddl_failures')} details={schema.get('details')}. "
            "The startup DDL fails open, so the app answers /health while endpoints that "
            "need those tables return 500.")
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0] if argv else 'check_health.py'} <health.json|->",
              file=sys.stderr)
        return 2
    try:
        raw = sys.stdin.read() if argv[1] == "-" else open(argv[1], encoding="utf-8").read()
        health = json.loads(raw)
    except Exception as e:
        print(f"SMOKE TEST FAILED — could not read /health: {e}", file=sys.stderr)
        return 1

    found = problems(health)
    if found:
        print("SMOKE TEST FAILED — " + "; ".join(found), file=sys.stderr)
        return 1
    print("health OK (status, database, schema all clean)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
