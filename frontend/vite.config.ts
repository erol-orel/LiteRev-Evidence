import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

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
  plugins: [react()],
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
