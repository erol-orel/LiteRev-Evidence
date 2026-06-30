import type { Translations } from "./fr";

// English translations. Typed as `Translations` so the compiler flags any key
// that is missing or mistyped relative to the French source of truth.
export const en: Translations = {
  nav: {
    search: "Search",
    scenarios: "Scenarios",
    terrain: "Field Data",
    stats: "Statistics",
  },
  header: {
    adminKeyActive: "Admin key",
    readOnly: "Read-only",
    adminKeyActiveTooltip:
      "Admin key active on this device — click to remove it",
    adminKeySetTooltip:
      "Set the admin key (X-API-Key) to enable writes",
    languageLabel: "Language",
  },
};
