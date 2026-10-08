import { describe, expect, it } from "vitest";
import type { ExtractionArticle } from "./api";
import { COVERAGE_KEYS, filterArticles, quotesToCheck, share, sortArticles } from "./extraction";

const art = (id: number, over: Partial<ExtractionArticle> = {}): ExtractionArticle => ({
  id, title: `Paper ${id}`, year: 2020 + id, doi: null, journal: null, has_extraction: true,
  source: "fulltext", text_truncated: false,
  coverage: Object.fromEntries(COVERAGE_KEYS.map((k) => [k, false])),
  n_observations: 4, n_quote_found: 4, attempts: 0, ...over,
});

const view = { filter: "all" as const, search: "", item: null, itemMode: "reports" as const };

describe("share", () => {
  it("is a whole percentage and never NaN", () => {
    expect(share(34, 120)).toBe(28);
    expect(share(0, 0)).toBe(0);
    expect(share(5, 0)).toBe(0);
    expect(share(150, 100)).toBe(100);
    expect(share(Number.NaN, 10)).toBe(0);
  });
});

describe("filterArticles", () => {
  const items = [
    art(1, { coverage: { ...art(1).coverage!, sex_gender: true } }),
    art(2),
    art(3, { has_extraction: false, source: null, coverage: null, n_observations: 0, n_quote_found: 0 }),
    art(4, { n_quote_found: 1 }),
  ];

  it("splits extracted from pending", () => {
    expect(filterArticles(items, { ...view, filter: "extracted" }).map((a) => a.id)).toEqual([1, 2, 4]);
    expect(filterArticles(items, { ...view, filter: "pending" }).map((a) => a.id)).toEqual([3]);
  });

  it("lists the articles whose quotes need a check", () => {
    expect(quotesToCheck(items[3])).toBe(3);
    expect(filterArticles(items, { ...view, filter: "quote_missing" }).map((a) => a.id)).toEqual([4]);
  });

  it("filters by an item reported, or not reported, and never counts an unread article either way", () => {
    expect(filterArticles(items, { ...view, item: "sex_gender" }).map((a) => a.id)).toEqual([1]);
    const missing = filterArticles(items, { ...view, item: "sex_gender", itemMode: "missing" }).map((a) => a.id);
    expect(missing).toEqual([2, 4]);
    expect(missing).not.toContain(3);
  });

  it("searches the title without caring about case", () => {
    expect(filterArticles(items, { ...view, search: "PAPER 2" }).map((a) => a.id)).toEqual([2]);
  });
});

describe("sortArticles", () => {
  it("sorts by year, rows or quotes to check, without touching the input", () => {
    const items = [art(1, { n_observations: 9, n_quote_found: 9 }), art(2, { n_observations: 2, n_quote_found: 2 }), art(3, { n_quote_found: 0 })];
    const before = items.map((a) => a.id);
    expect(sortArticles(items, "year").map((a) => a.id)).toEqual([3, 2, 1]);
    expect(sortArticles(items, "rows").map((a) => a.id)).toEqual([1, 3, 2]);
    expect(sortArticles(items, "quotes")[0].id).toBe(3);
    expect(items.map((a) => a.id)).toEqual(before);
  });

  it("puts the papers that were read before the ones that were not, in every sort", () => {
    const items = [art(9, { has_extraction: false, source: null, coverage: null, n_observations: 0, n_quote_found: 0 }), art(1)];
    for (const by of ["year", "rows", "quotes"] as const) {
      expect(sortArticles(items, by).map((a) => a.id)).toEqual([1, 9]);
    }
  });
});
