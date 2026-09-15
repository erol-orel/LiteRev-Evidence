# LiteRev frontend

React 19 + TypeScript + Vite + Tailwind. Single-page application served by nginx in
production, talking to the FastAPI backend under `/api` (nginx strips the prefix).

## Develop

```bash
npm ci
npm run dev          # http://localhost:5173, /api proxied to http://127.0.0.1:8000
npm run build        # type-check (tsc -b) + production bundle in dist/
npm run preview      # serve dist/ with the same /api proxy
```

`VITE_API_PROXY_TARGET` changes where the dev/preview proxy sends `/api`
(default `http://127.0.0.1:8000`). `VITE_API_BASE_URL` changes the base the bundle
itself calls (default `/api`); it is inlined at build time, so never put a secret in
a `VITE_*` variable. The admin key is entered in the interface and stays in the
browser's storage.

## Test

```bash
npm test             # unit tests (vitest + Testing Library, jsdom): src/**/*.test.ts(x)
npm run test:watch
npm run e2e          # browser smoke test (Playwright) — needs a running API and a
                     # seeded scenario: use `python3 scripts/smoke_e2e.py` from the
                     # repository root, which sets everything up (see docs/ops-runbook.md §8)
```

Unit tests live next to the code they cover. The browser test lives in `e2e/` and
runs against the built bundle (`vite preview`), so `npm run build` first when running
Playwright by hand.

## Layout

- `src/App.tsx` — shell, navigation, search page, scenario list
- `src/components/ScenarioDetailPage.tsx` — the scenario page and its sections
- `src/lib/api.ts` — typed API client (retries, error messages, admin key)
- `src/lib/searchText.ts` — pure search helpers (combined query text, facet kinds)
- `src/i18n/` — language provider and the French/English locale files
