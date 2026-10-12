import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "../i18n/LanguageProvider";
import { en } from "../i18n/locales/en";
import { fr } from "../i18n/locales/fr";
import type { PooledGroup, PooledResponse, PooledStudy } from "../lib/api";
import { PooledPanel } from "./PooledPanel";

// Every test serves the panel its own answer through fetchPooled.
const { fetchPooled } = vi.hoisted(() => ({ fetchPooled: vi.fn() }));
vi.mock("../lib/api", async (importOriginal) => ({ ...(await importOriginal<typeof import("../lib/api")>()), fetchPooled }));

const P = en.scenarioDetail.extraction.pooled;

// ── Rows that give the composition of a sample ──────────────────────────────
const het = { Q: 3, df: 2, p: 0.2, I2: 30, tau2: 0.1, band: "moderate" as const };
const sampleStudy = (id: number, author: string, x: number, n: number, composition: boolean): PooledStudy => ({
  article_id: id, title: "t", year: 2025, first_author: author, x, n, review_status: "unreviewed",
  p: x / n, ci_low: x / n - 0.05, ci_high: x / n + 0.05, weight_pct: 33, composition,
});

// Two of the three "male" studies give who was studied (142 men among 225 people), and the
// comparison left out two papers whose male and female rows split one sample.
const composed: PooledResponse = {
  scenario_id: "s", filters: { reviewed_only: false, verified_only: true, split_disease: true, min_studies: 3 },
  n_rows_used: 12, n_duplicate_rows_dropped: 0,
  excluded: { rejected: 0, quote_not_found: 1, composition: 5 },
  pooled: [{
    sheet: "human_susc", group: "sex_gender", label: "male", disease: "A(H5N1)", label_path: "sex_gender > male",
    mapped: true, k: 3, n_total: 371, events_total: 234, k_composition: 2,
    pooled: { p: 0.6, ci_low: 0.4, ci_high: 0.75, pi_low: 0.1, pi_high: 0.95 }, heterogeneity: het, reason: null,
    studies: [sampleStudy(1, "Whitlow", 142, 225, true), sampleStudy(2, "Charania", 80, 106, true), sampleStudy(3, "Smith", 12, 40, false)],
  }],
  comparisons: [{
    sheet: "human_susc", group: "sex_gender", disease: "A(H5N1)", a: "male", b: "female", k: 3, excluded_composition: 2,
    pooled: { or: 1.3, ci_low: 0.8, ci_high: 2.1, pi_low: 0.4, pi_high: 4 }, heterogeneity: het,
    studies: [3, 4, 5].map((i) => ({ ...sampleStudy(i, "Smith", 12, 40, false), x1: 12, n1: 40, x2: 10, n2: 45, or: 1.3, ci_low: 0.5, ci_high: 3 })),
  }],
};

describe("PooledPanel composition", () => {
  beforeEach(() => {
    localStorage.setItem("literev-lang", "en");
    fetchPooled.mockResolvedValue(composed);
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
      ...composed, excluded: { quote_not_found: 1, composition: 0 },
      pooled: composed.pooled.map((g) => ({ ...g, k_composition: 0, studies: g.studies.map((s) => ({ ...s, composition: false })) })),
      comparisons: composed.comparisons.map((c) => ({ ...c, excluded_composition: 0 })),
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

// ── One disease, however the papers spell it ────────────────────────────────
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

function renderPanel(lang: "en" | "fr", body: PooledResponse) {
  localStorage.setItem("literev-lang", lang);
  fetchPooled.mockResolvedValue(body);
  render(<LanguageProvider><PooledPanel scenarioId="usr-test" tick={0} /></LanguageProvider>);
}

describe("PooledPanel diseases", () => {
  it("names the disease once and shows the spellings it merged, in a line and on hover", async () => {
    renderPanel("en", response([group({})]));
    const line = await screen.findByText(`As written: ${LIST}`);
    const cell = line.parentElement!;
    expect(cell.firstChild?.textContent).toBe("A(H5N1)");
    expect(cell).toHaveAttribute("title", `As written: ${LIST}`);
    // Opened, the row gives the whole list again under its forest plot, uncut.
    fireEvent.click(screen.getByText("immune status > seropositive"));
    expect(screen.getAllByText(`As written: ${LIST}`)).toHaveLength(2);
  });

  it("says nothing more when the one spelling already reads as the disease", async () => {
    renderPanel("en", response([group({ disease: "covid 19", disease_spellings: [{ spelling: "COVID-19", n: 4 }], k: 4 })]));
    expect(await screen.findByText("covid 19")).not.toHaveAttribute("title");
    expect(screen.queryByText(/^As written:/)).toBeNull();
  });

  it("explains on the toggle how diseases are matched", async () => {
    renderPanel("en", response([group({})]));
    await screen.findByText(`As written: ${LIST}`);
    expect(screen.getByText(en.scenarioDetail.extraction.pooled.optDisease).closest("label"))
      .toHaveAttribute("title", en.scenarioDetail.extraction.pooled.optDiseaseHint);
  });

  it("speaks French, and says a disease is not stated in words", async () => {
    const few = group({ disease: "(not stated)", disease_spellings: [{ spelling: "N/A", n: 1 }], k: 2, pooled: null, heterogeneity: null,
      reason: "too_few_studies" });
    renderPanel("fr", response([group({}), few]));
    await screen.findByText(`Tel qu'écrit : ${LIST}`);
    fireEvent.click(screen.getByText(/^Libellés avec moins de 3 études/));
    const notStated = screen.getByText(fr.scenarioDetail.extraction.pooled.diseaseNotStated);
    expect(notStated).toHaveAttribute("title", "Tel qu'écrit : N/A (1)");
    expect(screen.queryByText("(not stated)")).toBeNull();
  });
});

// ── Both at once: a pool that merged spellings and holds rows that split a sample ──
const GROUP_SPELT = [{ spelling: "HPAI H5N1", n: 2 }, { spelling: "H5N1", n: 1 }];
const CMP_SPELT = [{ spelling: "HPAI H5N1", n: 2 }, { spelling: "Influenza A(H5N1)", n: 1 }];
const both: PooledResponse = {
  ...composed,
  pooled: composed.pooled.map((g) => ({ ...g, disease_spellings: GROUP_SPELT })),
  comparisons: composed.comparisons.map((c) => ({ ...c, disease_spellings: CMP_SPELT })),
};

describe("PooledPanel composition and diseases together", () => {
  it("marks the composition beside the label and gives the spellings beside the disease", async () => {
    renderPanel("en", both);
    const line = await screen.findByText("As written: HPAI H5N1 (2) · H5N1 (1)");
    expect(line.parentElement!.firstChild?.textContent).toBe("A(H5N1)");
    expect(screen.getByText(/composition of the sample in 2 of 3/)).toBeInTheDocument();
    // The comparison names its disease, with its own spellings on hover, and the papers it left out.
    expect(screen.getByTitle("As written: HPAI H5N1 (2) · Influenza A(H5N1) (1)")).toHaveTextContent("A(H5N1)");
    expect(screen.getByText(/3 papers, 2 left out as the composition of one sample/)).toBeInTheDocument();
    expect(screen.getByText(new RegExp(`${P.excl.composition} 5\\.`))).toBeInTheDocument();
    // Opened, the label names the studies that split a sample and gives its spellings again.
    fireEvent.click(screen.getByText(/sex gender > male/));
    expect(screen.getByText("Give the composition of their sample, not a group with its cases: Whitlow 2025, Charania 2025.")).toBeInTheDocument();
    expect(screen.getAllByText("As written: HPAI H5N1 (2) · H5N1 (1)")).toHaveLength(2);
    // Opened, the comparison gives its own spellings under its forest plot.
    fireEvent.click(screen.getByText(/3 papers, 2 left out/));
    expect(screen.getByText("As written: HPAI H5N1 (2) · Influenza A(H5N1) (1)")).toBeInTheDocument();
  });

  it("says both in French", async () => {
    renderPanel("fr", both);
    await screen.findByText("Tel qu'écrit : HPAI H5N1 (2) · H5N1 (1)");
    expect(screen.getByText(/composition de l'échantillon dans 2 sur 3/)).toBeInTheDocument();
    expect(screen.getByText(/3 articles, 2 écartés comme composition d'un échantillon/)).toBeInTheDocument();
    expect(screen.getByTitle("Tel qu'écrit : HPAI H5N1 (2) · Influenza A(H5N1) (1)")).toHaveTextContent("A(H5N1)");
  });
});
