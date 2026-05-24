/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#ecfeff',
          100: '#cffafe',
          500: '#06b6d4',
          600: '#0891b2',
          700: '#0e7490'
        }
      },
      boxShadow: {
        panel: '0 10px 40px rgba(15, 23, 42, 0.28)'
      }
    }
  },
  plugins: []
}
