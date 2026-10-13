import { defineConfig } from "@playwright/test";

// The demo video's tour (demo/tour.spec.ts), kept out of the smoke test (e2e/). Run by
// scripts/demo_video.py at the repository root, which boots the API on a throwaway
// database with the synthetic demo corpus and renders the video from what this writes.
// With DEMO_BASE_URL the tour films a running deployment instead, and no server starts.
const PORT = Number(process.env.E2E_WEB_PORT ?? 4173);
const REMOTE = process.env.DEMO_BASE_URL;

export default defineConfig({
  testDir: "./demo",
  timeout: 20 * 60_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: REMOTE || `http://127.0.0.1:${PORT}`,
    browserName: "chromium",
    // 1440 px is the width the interface is laid out for; scale 2 keeps text sharp when
    // the renderer zooms in and scales down to 1920 x 1080.
    viewport: { width: 1440, height: 810 },
    deviceScaleFactor: 2,
    locale: process.env.DEMO_LANG === "fr" ? "fr-FR" : "en-US",
    colorScheme: "dark",
    // A locator that never resolves fails the scene instead of waiting for ever.
    actionTimeout: 30_000,
  },
  webServer: REMOTE ? undefined : {
    command: `npm run preview -- --host 127.0.0.1 --port ${PORT} --strictPort`,
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
