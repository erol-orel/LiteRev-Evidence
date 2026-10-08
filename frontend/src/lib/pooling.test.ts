import { describe, expect, it } from "vitest";
import type { PooledComparison, PooledGroup } from "./api";
import { bandTone, csvOfPooled, fmtOr, forestScale, intervalText, orDomain, pct, proportionDomain } from "./pooling";

const study = (p: number, lo: number, hi: number) => ({ article_id: 1, title: "t", year: 2020, first_author: "A", x: 1, n: 10, review_status: "accepted", p, ci_low: lo, ci_high: hi });
const group = (over: Partial<PooledGroup> = {}): PooledGroup => ({
  sheet: "human_susc", group: "sex_gender", label: "male", disease: "hpai", label_path: "sex_gender > male", mapped: true,
  k: 2, n_total: 20, events_total: 4, pooled: { p: 0.2, ci_low: 0.1, ci_high: 0.35, pi_low: 0.05, pi_high: 0.6 },
  heterogeneity: { Q: 3, df: 1, p: 0.08, I2: 60, tau2: 0.2, band: "substantial" }, reason: null,
  studies: [study(0.1, 0.02, 0.4), study(0.3, 0.1, 0.55)], ...over,
});

describe("formatting", () => {
  it("shows a proportion with one decimal and never NaN", () => {
    expect(pct(0.2236)).toBe("22.4%");
    expect(pct(0.5)).toBe("50%");
    expect(pct(null)).toBe("-");
    expect(pct(Number.NaN)).toBe("-");
  });
  it("shows an odds ratio with three significant digits", () => {
    expect(fmtOr(1.23456)).toBe("1.23");
    expect(fmtOr(0.047312)).toBe("0.0473");
    expect(fmtOr(250.4)).toBe("250");
    expect(fmtOr(null)).toBe("-");
  });
  it("writes an interval or a dash", () => {
    expect(intervalText(0.1, 0.35, pct)).toBe("10% to 35%");
    expect(intervalText(0.1, null, pct)).toBe("-");
  });
});

describe("forestScale", () => {
  it("is linear for proportions and clamps to the plot", () => {
    const x = forestScale(0, 1, 200, false);
    expect(x(0)).toBe(0);
    expect(x(0.5)).toBe(100);
    expect(x(2)).toBe(200);
    expect(x(-1)).toBe(0);
  });
  it("is logarithmic for odds ratios, with 1 at the middle of a symmetric domain", () => {
    const x = forestScale(0.25, 4, 200, true);
    expect(x(1)).toBeCloseTo(100);
    expect(x(0.25)).toBeCloseTo(0);
    expect(x(4)).toBeCloseTo(200);
    expect(x(0)).toBe(0);                                            // a zero never gives NaN
  });
});

describe("domains", () => {
  it("rounds a proportion domain up and caps it at 1", () => {
    expect(proportionDomain(group())).toEqual([0, 0.6]);
    expect(proportionDomain(group({ studies: [study(0.9, 0.5, 0.99)], pooled: null }))).toEqual([0, 1]);
  });
  it("keeps the null value 1 inside an odds-ratio domain", () => {
    const c: PooledComparison = {
      sheet: "human_susc", group: "sex_gender", disease: null, a: "male", b: "female", k: 2,
      pooled: { or: 3, ci_low: 2, ci_high: 4.5, pi_low: null, pi_high: null },
      heterogeneity: group().heterogeneity!, studies: [{ ...study(0, 0, 0), ci_low: 1.8, ci_high: 5 }],
    };
    const [lo, hi] = orDomain(c);
    expect(lo).toBeLessThan(1);
    expect(hi).toBeGreaterThan(5);
  });
});

describe("tone and csv", () => {
  it("gets more worried as heterogeneity grows", () => {
    expect(bandTone("low")).not.toBe(bandTone("considerable"));
    expect(bandTone("substantial")).toContain("gold");
  });
  it("writes a csv that quotes text and leaves unpooled values empty", () => {
    const csv = csvOfPooled([group({ label: 'say "hi"' }), group({ pooled: null, heterogeneity: null })]).split("\n");
    expect(csv[0].startsWith("sheet,group,label,disease,k")).toBe(true);
    expect(csv[1]).toContain('"say ""hi"""');
    expect(csv[2]).toContain(",,,,,,");
  });
});
