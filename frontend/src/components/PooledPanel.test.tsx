import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "../i18n/LanguageProvider";
import { en } from "../i18n/locales/en";
import type { PooledResponse, PooledStudy } from "../lib/api";
import { PooledPanel } from "./PooledPanel";

const { fetchPooled } = vi.hoisted(() => ({ fetchPooled: vi.fn() }));
vi.mock("../lib/api", async (importOriginal) => ({ ...(await importOriginal<typeof import("../lib/api")>()), fetchPooled }));

const P = en.scenarioDetail.extraction.pooled;
const het = { Q: 3, df: 2, p: 0.2, I2: 30, tau2: 0.1, band: "moderate" as const };
const study = (id: number, author: string, x: number, n: number, composition: boolean): PooledStudy => ({
  article_id: id, title: "t", year: 2025, first_author: author, x, n, review_status: "unreviewed",
  p: x / n, ci_low: x / n - 0.05, ci_high: x / n + 0.05, weight_pct: 33, composition,
});

// Two of the three "male" studies give who was studied (142 men among 225 people), and the
// comparison left out two papers whose male and female rows split one sample.
const response: PooledResponse = {
  scenario_id: "s", filters: { reviewed_only: false, verified_only: true, split_disease: true, min_studies: 3 },
  n_rows_used: 12, n_duplicate_rows_dropped: 0,
  excluded: { rejected: 0, quote_not_found: 1, composition: 5 },
  pooled: [{
    sheet: "human_susc", group: "sex_gender", label: "male", disease: "hpai a h5n1", label_path: "sex_gender > male",
    mapped: true, k: 3, n_total: 371, events_total: 234, k_composition: 2,
    pooled: { p: 0.6, ci_low: 0.4, ci_high: 0.75, pi_low: 0.1, pi_high: 0.95 }, heterogeneity: het, reason: null,
    studies: [study(1, "Whitlow", 142, 225, true), study(2, "Charania", 80, 106, true), study(3, "Smith", 12, 40, false)],
  }],
  comparisons: [{
    sheet: "human_susc", group: "sex_gender", disease: "hpai a h5n1", a: "male", b: "female", k: 3, excluded_composition: 2,
    pooled: { or: 1.3, ci_low: 0.8, ci_high: 2.1, pi_low: 0.4, pi_high: 4 }, heterogeneity: het,
    studies: [3, 4, 5].map((i) => ({ ...study(i, "Smith", 12, 40, false), x1: 12, n1: 40, x2: 10, n2: 45, or: 1.3, ci_low: 0.5, ci_high: 3 })),
  }],
};

describe("PooledPanel", () => {
  beforeEach(() => {
    localStorage.setItem("literev-lang", "en");
    fetchPooled.mockResolvedValue(response);
  });

  it("says which rows give the composition of a sample and what the comparisons left out", async () => {
    render(<LanguageProvider><PooledPanel scenarioId="s" tick={0} /></LanguageProvider>);
    expect(await screen.findByText(new RegExp(`${P.excl.composition} 5\\.`))).toBeInTheDocument();
    expect(screen.getByText(P.caution)).toBeInTheDocument();
    expect(screen.getByText(/composition of the sample in 2 of 3/)).toBeInTheDocument();
    expect(screen.getByText(/3 papers, 2 left out as the composition of one sample/)).toBeInTheDocument();
    // The studies are named under the forest plot of the label.
    fireEvent.click(screen.getByText(/sex gender > male/));
    expect(screen.getByText("Give the composition of their sample, not a group with its cases: Whitlow 2025, Charania 2025.")).toBeInTheDocument();
  });

  it("says nothing of the kind where no row gives the composition of a sample", async () => {
    fetchPooled.mockResolvedValue({
      ...response, excluded: { quote_not_found: 1, composition: 0 },
      pooled: response.pooled.map((g) => ({ ...g, k_composition: 0, studies: g.studies.map((s) => ({ ...s, composition: false })) })),
      comparisons: response.comparisons.map((c) => ({ ...c, excluded_composition: 0 })),
    });
    render(<LanguageProvider><PooledPanel scenarioId="s" tick={0} /></LanguageProvider>);
    expect(await screen.findByText(/quote not found 1\./)).toBeInTheDocument();
    expect(screen.queryByText(new RegExp(P.excl.composition))).toBeNull();
    expect(screen.queryByText(/composition of the sample in/)).toBeNull();
    expect(screen.queryByText(/left out as the composition/)).toBeNull();
    fireEvent.click(screen.getByText(/sex gender > male/));
    expect(screen.queryByText(/Give the composition of their sample/)).toBeNull();
  });
});
