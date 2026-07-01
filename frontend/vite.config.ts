import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
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
})
