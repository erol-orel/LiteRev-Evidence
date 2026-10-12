/** The GRADE levels the server sends are its canonical French values (`api/study_design.py`):
 *  they travel as keys and are compared, so they are never translated at the source, and the
 *  interface shows a label beside them. The claim badges of the evidence brief printed the raw
 *  value, so an English reader saw "FAIBLE" or "TRÈS FAIBLE". Briefs written before the claims
 *  were graded on the GRADE scale carry the former four strengths, read the same way. */
const LEVEL_KEYS: Record<string, string> = {
  "Élevée": "high",
  "Modérée": "moderate",
  "Faible": "low",
  "Très faible": "veryLow",
  "Non applicable": "notApplicable",
  "Non évaluée": "notAssessed",
  "Fort": "strong",
  "Modéré": "moderate",
  "Insuffisant": "insufficient",
};

/** Strongest first, then the two values that are not on the scale. */
const LEVEL_BADGE: Record<string, string> = {
  high: "bg-brand-500/20 text-brand-300",
  strong: "bg-brand-500/20 text-brand-300",
  moderate: "bg-gold-500/20 text-gold-300",
  low: "bg-orange-500/20 text-orange-300",
  veryLow: "bg-rose-500/15 text-rose-300",
  insufficient: "bg-white/10 text-white/45",
  notApplicable: "bg-white/10 text-white/45",
  notAssessed: "bg-white/10 text-white/40",
};

/** The locale key of a server level, or null for a value this table does not know. */
export function levelKey(level: string | null | undefined): string | null {
  return (level && LEVEL_KEYS[level.trim()]) || null;
}

/** The label of a server level in the interface's language; an unknown value is shown as it came. */
export function levelText(level: string | null | undefined, t: (key: string) => string): string {
  const key = levelKey(level);
  return key ? t(`scenarioDetail.evidences.levels.${key}`) : (level ?? "");
}

/** The badge colours of a server level; neutral for an unknown one. */
export function levelBadge(level: string | null | undefined): string {
  const key = levelKey(level);
  return (key && LEVEL_BADGE[key]) || "bg-white/10 text-white/50";
}
