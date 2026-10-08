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

export type ExtractionFilter = "all" | "extracted" | "pending" | "quote_missing";
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
