import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "../i18n/LanguageProvider";
import { en } from "../i18n/locales/en";
import { fr } from "../i18n/locales/fr";
import type { PooledGroup, PooledResponse } from "../lib/api";
import { PooledPanel } from "./PooledPanel";

/** HPAI_last's six spellings of A(H5N1) under immune_status > seropositive, as the API returns them. */
const SPELT = [
  { spelling: "HPAI H5N1", n: 5 }, { spelling: "Avian influenza A(H5N1)", n: 4 }, { spelling: "HPAI A(H5N1)", n: 4 },
  { spelling: "Influenza A(H5N1)", n: 4 }, { spelling: "H5N1", n: 3 }, { spelling: "H5N1 avian influenza", n: 3 },
];
const LIST = "HPAI H5N1 (5) · Avian influenza A(H5N1) (4) · HPAI A(H5N1) (4) · Influenza A(H5N1) (4) · H5N1 (3) · H5N1 avian influenza (3)";

const study = (i: number) => ({ article_id: i, title: `Paper ${i}`, year: 2024, first_author: "Lee", x: 3, n: 40, review_status: "unreviewed",
  p: 0.075, ci_low: 0.03, ci_high: 0.2, weight_pct: 50 });
const group = (over: Partial<PooledGroup>): PooledGroup => ({
  sheet: "human_susc", group: "immune_status", label: "seropositive", disease: "A(H5N1)", disease_spellings: SPELT,
  label_path: "immune_status > seropositive", mapped: true, k: 23, n_total: 920, events_total: 69,
  pooled: { p: 0.075, ci_low: 0.05, ci_high: 0.11, pi_low: 0.02, pi_high: 0.25 },
  heterogeneity: { Q: 30, df: 22, p: 0.12, I2: 26.7, tau2: 0.1, band: "moderate" }, reason: null,
  studies: [study(1), study(2)], ...over,
});
const response = (pooled: PooledGroup[]): PooledResponse => ({
  scenario_id: "usr-test", filters: { reviewed_only: false, verified_only: true, split_disease: true, min_studies: 3 },
  n_rows_used: 25, n_duplicate_rows_dropped: 0, excluded: {}, pooled, comparisons: [],
  groups_total: pooled.length, groups_returned: pooled.length, groups_truncated: false,
  groups_too_few_studies: pooled.filter((g) => !g.pooled).length,
});

function serve(body: PooledResponse) {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
    ok: true, status: 200, headers: { get: () => null }, json: async () => body, text: async () => JSON.stringify(body),
  }));
}

function renderPanel(lang: "en" | "fr", pooled: PooledGroup[]) {
  localStorage.setItem("literev-lang", lang);
  serve(response(pooled));
  render(<LanguageProvider><PooledPanel scenarioId="usr-test" tick={0} /></LanguageProvider>);
}

describe("PooledPanel diseases", () => {
  it("names the disease once and shows the spellings it merged, in a line and on hover", async () => {
    renderPanel("en", [group({})]);
    const line = await screen.findByText(`As written: ${LIST}`);
    const cell = line.parentElement!;
    expect(cell.firstChild?.textContent).toBe("A(H5N1)");
    expect(cell).toHaveAttribute("title", `As written: ${LIST}`);
    // Opened, the row gives the whole list again under its forest plot, uncut.
    fireEvent.click(screen.getByText("immune status > seropositive"));
    expect(screen.getAllByText(`As written: ${LIST}`)).toHaveLength(2);
  });

  it("says nothing more when the one spelling already reads as the disease", async () => {
    renderPanel("en", [group({ disease: "covid 19", disease_spellings: [{ spelling: "COVID-19", n: 4 }], k: 4 })]);
    expect(await screen.findByText("covid 19")).not.toHaveAttribute("title");
    expect(screen.queryByText(/^As written:/)).toBeNull();
  });

  it("explains on the toggle how diseases are matched", async () => {
    renderPanel("en", [group({})]);
    await screen.findByText(`As written: ${LIST}`);
    expect(screen.getByText(en.scenarioDetail.extraction.pooled.optDisease).closest("label"))
      .toHaveAttribute("title", en.scenarioDetail.extraction.pooled.optDiseaseHint);
  });

  it("speaks French, and says a disease is not stated in words", async () => {
    const few = group({ disease: "(not stated)", disease_spellings: [{ spelling: "N/A", n: 1 }], k: 2, pooled: null, heterogeneity: null,
      reason: "too_few_studies" });
    renderPanel("fr", [group({}), few]);
    await screen.findByText(`Tel qu'écrit : ${LIST}`);
    fireEvent.click(screen.getByText(/^Libellés avec moins de 3 études/));
    const notStated = screen.getByText(fr.scenarioDetail.extraction.pooled.diseaseNotStated);
    expect(notStated).toHaveAttribute("title", "Tel qu'écrit : N/A (1)");
    expect(screen.queryByText("(not stated)")).toBeNull();
  });
});
