import type { ExtractionArticle } from "./api";

/** What a paper can report on, in the order the screen shows it. Same keys as the API. */
export const COVERAGE_KEYS = [
  "sex_gender", "age", "occupation", "kap_risk_perception", "ppe",
  "vaccination", "human_testing", "animal_host", "environment", "vector",
] as const;
export type CoverageKey = (typeof COVERAGE_KEYS)[number];

export const SHEET_KEYS = ["human_susc", "human_exp", "env", "animal", "vector"] as const;

/** A whole percentage, 0 when there is nothing to divide by (never NaN on screen). */
export function share(n: number, d: number): number {
  if (!d || d <= 0 || !Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(100, Math.round((n / d) * 100)));
}

export type ExtractionFilter = "all" | "extracted" | "pending" | "quote_missing" | "to_review" | "conflict";
export type ItemMode = "reports" | "missing";
export type ExtractionSort = "year" | "rows" | "quotes";

/** Rows whose quote was not found in the text the model read: the ones to check first. */
export function quotesToCheck(a: Pick<ExtractionArticle, "n_observations" | "n_quote_found">): number {
  return Math.max(0, (a.n_observations || 0) - (a.n_quote_found || 0));
}

export interface ExtractionView {
  filter: ExtractionFilter;
  search: string;
  item: CoverageKey | null;
  itemMode: ItemMode;
}

/** The articles the table shows. An item filter only looks at extracted articles: an article
 *  that was never read can neither report an item nor be said to lack it. */
export function filterArticles(articles: ExtractionArticle[], v: ExtractionView): ExtractionArticle[] {
  const q = v.search.trim().toLowerCase();
  return articles.filter((a) => {
    if (v.filter === "extracted" && !a.has_extraction) return false;
    if (v.filter === "pending" && a.has_extraction) return false;
    if (v.filter === "quote_missing" && quotesToCheck(a) === 0) return false;
    if (v.filter === "to_review" && !(a.has_extraction && a.n_observations > (a.n_reviewed ?? 0))) return false;
    if (v.filter === "conflict" && !((a.n_conflict ?? 0) > 0)) return false;
    if (q && !(a.title || "").toLowerCase().includes(q)) return false;
    if (v.item) {
      if (!a.has_extraction || !a.coverage) return false;
      const reports = a.coverage[v.item] === true;
      if (v.itemMode === "reports" ? !reports : reports) return false;
    }
    return true;
  });
}

export function sortArticles(articles: ExtractionArticle[], by: ExtractionSort): ExtractionArticle[] {
  const out = [...articles];
  out.sort((a, b) => {
    // Papers that were read come first, whatever the sort: the unread ones have nothing to rank.
    const read = Number(b.has_extraction) - Number(a.has_extraction);
    if (read !== 0) return read;
    if (by === "rows") return b.n_observations - a.n_observations || (b.year ?? 0) - (a.year ?? 0);
    if (by === "quotes") return quotesToCheck(b) - quotesToCheck(a) || (b.year ?? 0) - (a.year ?? 0);
    return (b.year ?? 0) - (a.year ?? 0) || b.id - a.id;
  });
  return out;
}

/** Kappa in words (Landis and Koch), for a reader who does not know the scale. */
export function kappaBand(k: number | null | undefined): "undefined" | "poor" | "fair" | "moderate" | "substantial" | "almost" {
  if (k == null || !Number.isFinite(k)) return "undefined";
  if (k < 0.2) return "poor";
  if (k < 0.4) return "fair";
  if (k < 0.6) return "moderate";
  if (k < 0.8) return "substantial";
  return "almost";
}

/** Fields a reviewer may type, as the payload of an edit. Empty text clears a number. */
export function editPayload(f: { value: string; n_cases: string; pop_risk: string; covariate: string }): Record<string, unknown> {
  const num = (s: string) => (s.trim() === "" ? null : Number(s.trim().replace(",", ".")));
  return { value: num(f.value), n_cases: num(f.n_cases), pop_risk: num(f.pop_risk), covariate: f.covariate.trim() };
}

/** Whether every typed number is a number, so a bad entry is caught before it is sent. */
export function editIsValid(f: { value: string; n_cases: string; pop_risk: string; covariate: string }): boolean {
  const p = editPayload(f);
  return f.covariate.trim().length > 0
    && (["value", "n_cases", "pop_risk"] as const).every((k) => p[k] === null || Number.isFinite(p[k] as number));
}
