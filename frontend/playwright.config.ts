import { defineConfig, devices } from "@playwright/test";

// Browser smoke test of the built frontend (e2e/). `vite preview` serves dist/ and
// proxies /api to the API named by VITE_API_PROXY_TARGET (see vite.config.ts);
// scripts/smoke_e2e.py at the repository root boots that API on a throwaway
// database, seeds a scenario and runs this configuration.
const PORT = Number(process.env.E2E_WEB_PORT ?? 4173);
const isCI = !!process.env.CI;

export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: isCI ? 1 : 0,
  reporter: isCI ? [["list"], ["html", { open: "never" }]] : [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    // French by default, like a Geneva browser; the tests switch to English themselves.
    locale: "fr-FR",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `npm run preview -- --host 127.0.0.1 --port ${PORT} --strictPort`,
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: !isCI,
    timeout: 60_000,
  },
});
