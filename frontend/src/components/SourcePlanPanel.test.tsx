import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SourcePlanPanel } from "./SourcePlanPanel";
import { LanguageProvider } from "../i18n/LanguageProvider";
import type { SourceQueryPlan } from "../lib/api";

/** The plan of the T2.4 COVID exposure string, as the server answers it: PubMed in the
 *  body, OpenAlex and Semantic Scholar in parts, DOAJ reduced, Crossref keywords. */
const PLAN: SourceQueryPlan = {
  query_chars: 2138,
  plan: [
    { source: "pubmed", how: "body" },
    { source: "europepmc", how: "body" },
    { source: "openalex", how: "parts", parts: 9 },
    { source: "semantic_scholar", how: "parts", parts: 9 },
    { source: "doaj", how: "reduced", limit: "DOAJ_MAX_QUERY_CHARS" },
    { source: "crossref", how: "keywords" },
    { source: "clinicaltrials", how: "whole" },
  ],
  whole: ["clinicaltrials"],
  body: ["pubmed", "europepmc"],
  parts: { openalex: 9, semantic_scholar: 9 },
  reduced: ["doaj"],
  keywords: ["crossref"],
};

function show(plan: SourceQueryPlan, lang: "en" | "fr" = "en") {
  localStorage.setItem("literev-lang", lang);
  render(
    <LanguageProvider>
      <SourcePlanPanel plan={plan} />
    </LanguageProvider>,
  );
}

describe("the plan of a query, before the search", () => {
  it("names the sources of every carriage, with the number of parts", () => {
    show(PLAN);
    const box = screen.getByTestId("source-plan");
    expect(box.textContent).toContain("2138");
    expect(box.textContent).toContain("pubmed, europepmc");
    expect(box.textContent).toContain("openalex (9), semantic_scholar (9)");
    expect(box.textContent).toContain("doaj");
    expect(box.textContent).toContain("crossref");
    expect(box.textContent).toContain("clinicaltrials");
  });

  it("says that only the reduced form drops terms", () => {
    show(PLAN);
    const box = screen.getByTestId("source-plan");
    // The word is the user's warning: a split loses nothing, a reduction does.
    expect(box.textContent).toMatch(/DROPS terms/);
    expect(box.textContent).toMatch(/nothing lost/);
  });

  it("speaks French when the interface does", () => {
    show(PLAN, "fr");
    expect(screen.getByTestId("source-plan").textContent).toMatch(/RETIRE des termes/);
  });

  it("renders nothing at all when there is no plan to show", () => {
    show({ ...PLAN, plan: [] });
    expect(screen.queryByTestId("source-plan")).toBeNull();
  });
});
