// French translations (default language). The shape of this object is the
// source of truth; en.ts must match it (enforced via the `Translations` type).
export const fr = {
  nav: {
    search: "Recherche",
    scenarios: "Scénarios",
    terrain: "Données Terrain",
    stats: "Statistiques",
  },
  header: {
    adminKeyActive: "Clé admin",
    readOnly: "Lecture seule",
    adminKeyActiveTooltip:
      "Clé admin active sur cet appareil — cliquer pour la retirer",
    adminKeySetTooltip:
      "Définir la clé admin (X-API-Key) pour activer les écritures",
    languageLabel: "Langue",
  },
};

// Values infer as `string` (no `as const`), so en.ts must match the key
// structure but supply its own strings; missing/extra keys are compile errors.
export type Translations = typeof fr;
