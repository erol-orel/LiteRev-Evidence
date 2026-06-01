/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // ── Palette LiteRev — couleurs exactes du logo ───────────────
        // Vert foncé (tronc + livre) : #0A3621
        // Or/Ambre  (feuillage)      : #E3AC3B
        // ─────────────────────────────────────────────────────────────
        brand: {
          50:  '#f0f7f3',
          100: '#d6eade',
          200: '#aed4bc',
          300: '#7db897',
          400: '#4d9c72',
          500: '#2d7a52',
          600: '#1f5c3c',
          700: '#0A3621',
          800: '#082a1a',
          900: '#051d12',
          950: '#030f09',
        },
        gold: {
          50:  '#fdf8ec',
          100: '#faefd0',
          200: '#f5dea0',
          300: '#efc96a',
          400: '#E3AC3B',
          500: '#d4941a',
          600: '#b07612',
          700: '#8a5a0e',
          800: '#6b4410',
          900: '#573711',
          950: '#311d05',
        },
        forest: {
          50:  '#f2f5f3',
          100: '#e0e8e3',
          200: '#c1d1c7',
          300: '#9ab4a4',
          400: '#6e9280',
          500: '#4d7461',
          600: '#3a5c4c',
          700: '#2c4739',
          800: '#1e3028',
          900: '#121e19',
          950: '#0a1410',
        },
      },
      boxShadow: {
        panel: '0 10px 40px rgba(10, 54, 33, 0.35)',
        gold:  '0 4px 20px rgba(227, 172, 59, 0.25)',
        green: '0 4px 20px rgba(10, 54, 33, 0.40)',
      },
    }
  },
  plugins: []
}
