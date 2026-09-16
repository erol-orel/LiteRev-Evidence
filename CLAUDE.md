# House rules for this repository

## No em dashes, anywhere, ever

Never write the em dash character (U+2014) in this repository or about it: not in code,
comments, interface strings, LLM prompts, documentation, scripts, workflows, commit
messages, pull request titles or descriptions, nor in chat replies about this project.
Use a comma, a colon, a plain hyphen or a middle dot instead. LLM prompts that generate
text must keep asking the model not to use it.

`tests/test_no_em_dashes.py` scans every tracked text file and fails CI on the first one.

## Working conventions

- Backend: the `api/` package, one module per domain; `main.py` is the entry point and
  re-exports every name. Tests patch names with `patch_app` (`tests/conftest.py`).
- The integration tests truncate the tables of `DB_URL`: never point them at a database
  whose data matters.
- Every merge to `main` deploys to production and restarts the API, which cuts the
  searches and pipelines in flight (they are relaunched at startup). Do not merge during
  a presentation.
- Frontend strings live in `frontend/src/i18n/locales/{fr,en}.ts`; both files must carry
  the same keys and the same `{placeholders}`.
