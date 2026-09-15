"""LiteRev API — entry point (`uvicorn main:app`) and composition root.

The implementation lives in the `api` package, one module per domain (see
api/__init__.py for the list and the import order). This module imports them in
order — which runs each module's startup DDL as before — and re-exports every
name, so `import main` keeps working for the scripts, tools and tests
(`main.engine`, `main._clustering_jobs`, …). To patch a function, patch it on the
module that defines it (`api.search._search_local_doc_ids`, …): a name patched on
`main` is not seen by the module that calls it.
"""
from __future__ import annotations

import importlib as _importlib

import api as _api

for _name in _api.MODULES:
    _mod = _importlib.import_module(f"api.{_name}")
    globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
    globals()[_name] = _mod

from api.core import app, engine, logger  # noqa: E402  (explicit for readers and linters)
