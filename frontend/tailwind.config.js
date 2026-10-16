/** @type {import('tailwindcss').Config} */

// ── Les couleurs passent par des variables CSS, pour que le thème puisse les changer ──
// L'interface a été écrite en mode sombre : ~3 000 classes de couleur (`text-white/80`,
// `border-white/10`, `text-brand-300`, `bg-rose-500/10`...). Les réécrire une à une avec
// une variante `dark:` serait des milliers de modifications à la main et autant d'oublis.
// Donc chaque nuance pointe vers une variable, et c'est `src/theme.css` (généré par
// `scripts/gen-theme.mjs`) qui donne les deux jeux de valeurs. En clair, la nuance n prend
// la nuance opposée de la même teinte, ce qui convient au style du code : nuances claires
// pour le texte et les filets, fonds en transparence.
const SHADES = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950]

/** Les nuances d'une teinte, chacune lue dans sa variable, avec l'alpha de Tailwind. */
const themed = (hue) =>
  Object.fromEntries(SHADES.map(s => [s, `rgb(var(--c-${hue}-${s}) / <alpha-value>)`]))

// Toute teinte employée dans src/ doit figurer ici ET dans scripts/gen-theme.mjs, sinon
// elle garderait la valeur fixe de Tailwind et ne suivrait pas le thème.
const HUES = ['brand', 'gold', 'forest', 'rose', 'amber', 'emerald', 'violet', 'red',
              'blue', 'teal', 'slate', 'sky', 'cyan', 'pink', 'green', 'orange',
              'fuchsia', 'purple', 'indigo', 'lime', 'yellow', 'stone']

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ...Object.fromEntries(HUES.map(h => [h, themed(h)])),
        // Dans ce code, « white » veut dire premier plan et non la couleur blanche :
        // `text-white/70` est du texte, `bg-white/[0.03]` un panneau, `border-white/10`
        // un filet. C'est donc l'encre du thème, blanche en sombre, presque noire en clair.
        white: 'rgb(var(--ink) / <alpha-value>)',
        // Le blanc qui doit RESTER blanc : le texte posé sur un bouton plein de couleur,
        // dont le fond ne change pas avec le thème.
        onsolid: '#ffffff',
        // Le fond de page, pour les rares endroits qui le peignent eux-mêmes.
        page: 'rgb(var(--page) / <alpha-value>)',
        // La pastille sélectionnée et son texte, identiques dans les deux thèmes.
        pill: 'rgb(var(--pill) / <alpha-value>)',
        onpill: 'rgb(var(--on-pill) / <alpha-value>)',
      },
      boxShadow: {
        // Définies en entier dans le thème : l'opacité d'une ombre n'est pas la même sur
        // fond sombre et sur fond clair.
        panel: 'var(--shadow-panel)',
        gold: 'var(--shadow-gold)',
        green: 'var(--shadow-green)',
      },
    }
  },
  plugins: []
}
