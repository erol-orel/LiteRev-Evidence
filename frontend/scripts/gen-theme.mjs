/**
 * Génère src/theme.css : une variable CSS par teinte et par nuance, en deux jeux.
 *
 * Pourquoi : l'interface a été écrite en mode sombre, avec ~3 000 classes de couleur
 * (`text-white/80`, `border-white/10`, `text-brand-300`, `bg-rose-500/10`...). Réécrire
 * chacune avec une variante `dark:` serait des milliers de modifications à la main, donc
 * à la place les couleurs de Tailwind deviennent des variables, et c'est le thème qui
 * change leur valeur. Aucune classe de composant ne bouge.
 *
 * La règle du mode clair : la nuance n devient la nuance 1000 - n de la MÊME teinte
 * (300 ↔ 700, 400 ↔ 600, 500 inchangé). Le code utilise les nuances claires pour le
 * texte et les bordures sur fond sombre, et des fonds en transparence ; inverser l'échelle
 * donne donc du texte foncé et des teintes pâles sur fond clair, ce qui est exactement ce
 * qu'il faut, sans toucher aux composants.
 *
 * `white` est traité comme une couleur sémantique : dans ce code il veut dire « premier
 * plan » (`text-white/70` = texte, `bg-white/[0.03]` = panneau, `border-white/10` = filet).
 * Il devient donc `--ink`, blanc en sombre et presque noir en clair. Le blanc qui doit
 * RESTER blanc, sur un bouton plein, s'écrit `text-onsolid`.
 *
 * Lancer : node scripts/gen-theme.mjs  (depuis frontend/)
 */
import { writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import colors from 'tailwindcss/colors.js'

const SHADES = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950]

/** Les échelles de la marque, telles qu'elles sont dans tailwind.config.js. */
const BRAND = {
  brand: { 50: '#f0f7f3', 100: '#d6eade', 200: '#aed4bc', 300: '#7db897', 400: '#4d9c72',
           500: '#2d7a52', 600: '#1f5c3c', 700: '#0A3621', 800: '#082a1a', 900: '#051d12',
           950: '#030f09' },
  gold:  { 50: '#fdf8ec', 100: '#faefd0', 200: '#f5dea0', 300: '#efc96a', 400: '#E3AC3B',
           500: '#d4941a', 600: '#b07612', 700: '#8a5a0e', 800: '#6b4410', 900: '#573711',
           950: '#311d05' },
  forest: { 50: '#f2f5f3', 100: '#e0e8e3', 200: '#c1d1c7', 300: '#9ab4a4', 400: '#6e9280',
            500: '#4d7461', 600: '#3a5c4c', 700: '#2c4739', 800: '#1e3028', 900: '#121e19',
            950: '#0a1410' },
}

/** Les teintes de Tailwind réellement employées dans src/ (voir le commentaire du fichier). */
const BUILT_IN = ['rose', 'amber', 'emerald', 'violet', 'red', 'blue', 'teal', 'slate',
                  'sky', 'cyan', 'pink', 'green', 'orange', 'fuchsia', 'purple', 'indigo',
                  'lime', 'yellow', 'stone']

const hexToRgb = (hex) => {
  const h = hex.replace('#', '').trim()
  const full = h.length === 3 ? h.split('').map(c => c + c).join('') : h
  const n = parseInt(full, 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

/** `oklch(...)` et `color(...)`, que Tailwind 4 emploie, ne se découpent pas en RGB ici. */
const toTriplet = (value) => {
  if (typeof value !== 'string' || !value.startsWith('#')) return null
  return hexToRgb(value).join(' ')
}

const scales = { ...BRAND }
for (const hue of BUILT_IN) {
  const scale = colors[hue]
  if (!scale || typeof scale !== 'object') continue
  const kept = {}
  for (const s of SHADES) {
    const t = toTriplet(scale[s])
    if (t) kept[s] = scale[s]
  }
  if (Object.keys(kept).length) scales[hue] = kept
}

const mirror = (shade) => {
  const m = { 50: 950, 100: 900, 200: 800, 300: 700, 400: 600, 500: 500,
              600: 400, 700: 300, 800: 200, 900: 100, 950: 50 }
  return m[shade]
}

const block = (mode) => {
  const out = []
  for (const [hue, scale] of Object.entries(scales)) {
    const line = []
    for (const s of SHADES) {
      const from = mode === 'light' ? mirror(s) : s
      const hex = scale[from] ?? scale[s]
      if (!hex) continue
      line.push(`  --c-${hue}-${s}: ${toTriplet(hex)};`)
    }
    if (line.length) out.push(`  /* ${hue} */\n${line.join('\n')}`)
  }
  return out.join('\n')
}

const header = `/* Fichier GÉNÉRÉ par scripts/gen-theme.mjs. Ne pas modifier à la main :
   relancer "node scripts/gen-theme.mjs" depuis frontend/.

   Une variable par teinte et par nuance. Le jeu sombre reprend exactement les couleurs
   d'avant le mode clair, donc le mode sombre est l'interface telle qu'elle a toujours été.
   Le jeu clair prend, pour chaque nuance, la nuance opposée de la même teinte. */
`

const text = `${header}
:root {
  /* « white » dans ce code veut dire premier plan, pas la couleur blanche. */
  --ink: 255 255 255;
  /* Le fond de page et les surfaces, pour ce que le CSS pose hors de Tailwind. */
  --page: 10 20 16;
  --chart-surface: 7 15 10;
  --chart-ink: 255 255 255;
  --chart-edge: 155 231 196;
  --chart-noise: 148 163 184;
  /* Les ombres portées en entier, parce que leur opacité n'est pas la même dans les deux
     thèmes : une ombre verte marquée sur fond sombre, un filet gris discret sur fond clair. */
  --shadow-panel: 0 10px 40px rgba(10, 54, 33, 0.35);
  --shadow-gold: 0 4px 20px rgba(227, 172, 59, 0.25);
  --shadow-green: 0 4px 20px rgba(10, 54, 33, 0.40);
  /* Le fond de l'application, posé une fois sur la coque. */
  --app-shell: radial-gradient(circle at top, rgba(10, 54, 33, 0.18), transparent 35%),
               linear-gradient(180deg, #0a1410 0%, #121e19 100%);
  /* Les surfaces opaques : barre du haut, boîtes de dialogue, fond des graphiques. */
  --surface: 10 20 16;
  --surface-deep: 7 15 10;
  /* La pastille sélectionnée (onglet actif, langue, thème) : vert foncé et or dans les
     DEUX thèmes. Inversée, elle donnerait de l'or moyen sur vert pâle, illisible. */
  --pill: 10 54 33;
  --on-pill: 227 172 59;
${block('dark')}
}

:root[data-theme='light'] {
  color-scheme: light;
  /* Pas un noir pur : l'encre des textes reprend le vert très foncé du logo. */
  --ink: 9 24 17;
  --page: 247 250 248;
  --chart-surface: 255 255 255;
  --chart-ink: 9 24 17;
  --chart-edge: 45 122 82;
  --chart-noise: 100 116 139;
  --shadow-panel: 0 8px 28px rgba(9, 24, 17, 0.10);
  --shadow-gold: 0 4px 18px rgba(176, 118, 18, 0.18);
  --shadow-green: 0 4px 18px rgba(9, 24, 17, 0.12);
  --app-shell: radial-gradient(circle at top, rgba(45, 122, 82, 0.10), transparent 40%),
               linear-gradient(180deg, #ffffff 0%, #eef3f0 100%);
  --surface: 255 255 255;
  --surface-deep: 250 252 251;
  --pill: 10 54 33;
  --on-pill: 239 201 106;
${block('light')}
}
`

const here = dirname(fileURLToPath(import.meta.url))
const target = join(here, '..', 'src', 'theme.css')
writeFileSync(target, text, 'utf8')
console.log(`wrote ${target} (${Object.keys(scales).length} hues, ${SHADES.length} shades)`)
