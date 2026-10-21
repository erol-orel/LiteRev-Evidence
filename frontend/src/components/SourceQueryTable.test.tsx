import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SourceQueryTable } from "./SourceQueryTable";
import { LanguageProvider } from "../i18n/LanguageProvider";
import type { SourceQueryPlan } from "../lib/api";

/** The plan of the COVID exposure strategy of production, abridged: PubMed with its field
 *  tags, OpenAlex in parts, OpenAIRE down to three terms, Semantic Scholar on keywords. */
const PUBMED = '("Environmental Exposure"[Majr] OR "Workplace"[mh]) AND "SARS-CoV-2"[mh]';
const PARTS = ["(workplace OR worksite) AND (sars-cov-2)", "(fomites OR fomite) AND (sars-cov-2)"];
const PLAN: SourceQueryPlan = {
  query_chars: 4519,
  planned_from: "strategy",
  plan: [
    { source: "pubmed", how: "body", sent: [PUBMED] },
    { source: "openalex", how: "parts", parts: 2, sent: PARTS,
      encoded: 3901, limit_value: 1800 },
    { source: "openaire", how: "reduced", sent: ["workplace AND sars-cov-2 AND epidemiolog"] },
    { source: "semantic_scholar", how: "keywords",
      note: "this API cannot express the query (a NOT it has no syntax for)",
      sent: ["environmental exposure sars-cov-2 workplace"] },
  ],
  whole: [],
  body: ["pubmed"],
  parts: { openalex: 2 },
  reduced: ["openaire"],
  keywords: ["semantic_scholar"],
};

function show(plan: SourceQueryPlan, props: Partial<Parameters<typeof SourceQueryTable>[0]> = {}) {
  localStorage.setItem("literev-lang", "en");
  render(
    <LanguageProvider>
      <SourceQueryTable plan={plan} {...props} />
    </LanguageProvider>,
  );
}

describe("the query each database receives", () => {
  it("names every database, with what happens to its query", () => {
    show(PLAN);
    for (const name of ["pubmed", "openalex", "openaire", "semantic_scholar"]) {
      expect(screen.getByText(name)).toBeTruthy();
    }
    expect(screen.getByText(/sent whole, in the request body/i)).toBeTruthy();
    expect(screen.getByText(/2 parts/i)).toBeTruthy();
    expect(screen.getByText(/reduced, drops terms/i)).toBeTruthy();
    expect(screen.getByText(/keywords only/i)).toBeTruthy();
  });

  it("shows the exact query once a database is opened, and each part of a split one", () => {
    show(PLAN);
    fireEvent.click(screen.getByRole("button", { name: /openalex/i }));
    for (const part of PARTS) expect(screen.getByText(part)).toBeTruthy();
    expect(screen.getByText(/part 1 of 2/i)).toBeTruthy();
    expect(screen.getByText(/part 2 of 2/i)).toBeTruthy();
  });

  it("shows PubMed the query with its field tags, which no other database receives", () => {
    show(PLAN);
    fireEvent.click(screen.getByRole("button", { name: /pubmed/i }));
    expect(screen.getByText(PUBMED)).toBeTruthy();
  });

  it("says plainly what OpenAIRE receives of a 4,519 character strategy", () => {
    // The single most surprising disclosure of this panel, and the reason it exists.
    show(PLAN);
    fireEvent.click(screen.getByRole("button", { name: /openaire/i }));
    expect(screen.getByText("workplace AND sars-cov-2 AND epidemiolog")).toBeTruthy();
  });

  it("carries the note that says why a database gets keywords", () => {
    show(PLAN);
    expect(screen.getByText(/cannot express the query/i)).toBeTruthy();
  });

  it("measures the query against the limit that forced the split", () => {
    show(PLAN);
    expect(screen.getByText(/3,901 of 1,800/)).toBeTruthy();
  });

  it("asks for one database alone when its button is pressed", () => {
    const onRerun = vi.fn();
    show(PLAN, { onRerun });
    const buttons = screen.getAllByText(/search again/i);
    expect(buttons).toHaveLength(4);
    fireEvent.click(buttons[1]);
    expect(onRerun).toHaveBeenCalledWith("openalex");
  });

  it("offers no re-run where there is no scenario to re-run", () => {
    show(PLAN);
    expect(screen.queryByText(/search again/i)).toBeNull();
  });

  it("says when the plan describes the typed query rather than a stored strategy", () => {
    show({ ...PLAN, planned_from: "query" });
    expect(screen.getByText(/No usable boolean strategy is stored/i)).toBeTruthy();
  });

  it("stays quiet about it when the stored strategy is what was planned", () => {
    show(PLAN);
    expect(screen.queryByText(/No usable boolean strategy is stored/i)).toBeNull();
  });

  it("renders nothing rather than an empty frame when there is no plan", () => {
    show({ ...PLAN, plan: [] });
    expect(screen.queryByTestId("source-query-table")).toBeNull();
  });
});
