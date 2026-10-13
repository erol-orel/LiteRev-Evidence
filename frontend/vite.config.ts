import { copyFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs'
import { extname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { Plugin } from 'vite'
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// The demo video is recorded into docs/demo/ (scripts/demo_video.py, docs/demo-video.md),
// where the README shows it. The site serves the same files under /demo/ for the header's
// "Demo video" button: copied into the build, not committed a second time.
const DEMO_DIR = fileURLToPath(new URL('../docs/demo/', import.meta.url))
const DEMO_FILES = new Set(['.mp4', '.vtt', '.jpg'])

function demoVideo(): Plugin {
  let outDir = 'dist'
  return {
    name: 'literev-demo-video',
    apply: 'build',
    configResolved(config) {
      outDir = join(config.root, config.build.outDir)
    },
    closeBundle() {
      if (!existsSync(DEMO_DIR)) return
      const dest = join(outDir, 'demo')
      mkdirSync(dest, { recursive: true })
      for (const name of readdirSync(DEMO_DIR)) {
        if (DEMO_FILES.has(extname(name))) copyFileSync(join(DEMO_DIR, name), join(dest, name))
      }
    },
  }
}

// In production the API is served under /api: nginx strips the prefix and forwards
// to the FastAPI process. The dev server and `vite preview` do the same, so
// `npm run dev` and the browser smoke test talk to a local API on the same origin
// (no CORS setup). VITE_API_PROXY_TARGET overrides the API address.
const apiProxy = {
  '/api': {
    target: process.env.VITE_API_PROXY_TARGET ?? 'http://127.0.0.1:8000',
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/api/, ''),
  },
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), demoVideo()],
  server: { proxy: apiProxy },
  preview: { proxy: apiProxy },
  build: {
    rollupOptions: {
      output: {
        // Split large third-party libraries into their own cached vendor
        // chunks, separate from application code, to keep the main entry
        // chunk small and improve long-term caching.
        manualChunks(id) {
          if (id.includes('node_modules')) {
            // React core + its runtime scheduler.
            if (
              id.includes('node_modules/react-dom/') ||
              id.includes('node_modules/react/') ||
              id.includes('node_modules/scheduler/')
            ) {
              return 'react'
            }
            // Icon library (large, tree-shakeable but heavy in aggregate).
            if (id.includes('node_modules/lucide-react/')) {
              return 'icons'
            }
            // Everything else from node_modules (e.g. clsx) in a shared vendor chunk.
            return 'vendor'
          }
        },
      },
    },
  },
  // Unit tests (vitest): src/**/*.test.ts(x), run in jsdom. The browser smoke test
  // lives in e2e/ and is run by Playwright, not vitest.
  test: {
    environment: 'jsdom',
    setupFiles: ['src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    css: false,
  },
})
