import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { QueryProposal } from "./QueryProposal";
import { LanguageProvider } from "../i18n/LanguageProvider";
import { en } from "../i18n/locales/en";
import { fr } from "../i18n/locales/fr";
import { fetchSearchStrategy, type SearchStrategy } from "../lib/api";
import { acceptProposal, changedSpan, keepEnteredQuery } from "../lib/queryProposal";

const P = en.search.proposal;
const ENTERED = '(poultry worker* OR veterinarian*) AND (h5n1 OR "avian influenza" AND seroprevalence';
const PROPOSED = `${ENTERED})`;
const REASON = "Opening parentheses never closed: 1. The proposal closes them at the end of the query.";

function show(lang: "en" | "fr" = "en") {
  localStorage.setItem("literev-lang", lang);
  const onAccept = vi.fn();
  const onKeep = vi.fn();
  render(
    <LanguageProvider>
      <QueryProposal entered={ENTERED} proposed={PROPOSED} reason={REASON} onAccept={onAccept} onKeep={onKeep} />
    </LanguageProvider>,
  );
  return { onAccept, onKeep };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("QueryProposal", () => {
  it("shows the query as entered, the proposed one, why, and two choices", () => {
    show();
    expect(screen.getByRole("region", { name: P.title })).toBeInTheDocument();
    expect(screen.getByTestId("proposal-entered").textContent).toBe(ENTERED);
    expect(screen.getByTestId("proposal-proposed").textContent).toBe(PROPOSED);
    expect(screen.getByTestId("proposal-reason").textContent).toBe(REASON);
    expect(screen.getByText(P.nothingChanges)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: P.accept })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: P.keep })).toBeInTheDocument();
  });

  it("highlights the change instead of leaving two walls of text to compare", () => {
    show();
    const marks = Array.from(document.querySelectorAll("mark"));
    expect(marks.map((m) => m.textContent)).toEqual([")"]);
    expect(screen.getByTestId("proposal-proposed").contains(marks[0])).toBe(true);
  });

  it("only reports the choice: accepting or keeping is the caller's to apply", () => {
    const { onAccept, onKeep } = show();
    fireEvent.click(screen.getByRole("button", { name: P.keep }));
    expect(onKeep).toHaveBeenCalledTimes(1);
    expect(onAccept).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: P.accept }));
    expect(onAccept).toHaveBeenCalledTimes(1);
  });

  it("speaks the interface language", () => {
    show("fr");
    expect(screen.getByRole("button", { name: fr.search.proposal.accept })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: fr.search.proposal.keep })).toBeInTheDocument();
    expect(screen.getByText(fr.search.proposal.entered)).toBeInTheDocument();
  });
});

describe("the choice", () => {
  const pending: SearchStrategy = {
    general: ENTERED, pubmed: ENTERED, explanation: "kept as entered", synonyms: [],
    entered_query: ENTERED, input_kind: "boolean", proposed_query: PROPOSED,
    proposal_reason: REASON, proposal_reasons: { en: REASON, fr: "x" }, proposal_issues: ["unclosed_parenthesis"],
  };

  it("accepting makes the proposal the query, for PubMed too, with nothing left pending", () => {
    const used = acceptProposal(pending);
    expect(used.general).toBe(PROPOSED);
    expect(used.pubmed).toBe(PROPOSED);
    expect(used.entered_query).toBe(ENTERED);
    for (const k of ["proposed_query", "proposal_reason", "proposal_reasons", "proposal_issues"]) {
      expect(k in used).toBe(false);
    }
  });

  it("keeping changes nothing but the pending proposal", () => {
    const used = keepEnteredQuery(pending);
    expect(used.general).toBe(ENTERED);
    expect(used.pubmed).toBe(ENTERED);
    expect("proposed_query" in used).toBe(false);
    expect(pending.proposed_query).toBe(PROPOSED);              // the server's answer is not mutated
  });

  it("finds where two versions differ", () => {
    expect(changedSpan("abc", "abXc")).toEqual({ prefix: "ab", removed: "", added: "X", suffix: "c" });
    expect(changedSpan("(a OR b)) AND c", "(a OR b) AND c"))
      .toEqual({ prefix: "(a OR b)", removed: ")", added: "", suffix: " AND c" });
    expect(changedSpan("same", "same")).toEqual({ prefix: "same", removed: "", added: "", suffix: "" });
  });
});

describe("fetchSearchStrategy", () => {
  it("sends the interface language with the query, a long strategy whole", async () => {
    localStorage.setItem("literev-lang", "en");
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true, status: 200, headers: { get: () => null },
      json: async () => ({ general: ENTERED, proposed_query: PROPOSED }),
    } as unknown as Response);
    vi.stubGlobal("fetch", fetchMock);
    const long = "a".repeat(5000);
    const out = await fetchSearchStrategy(long);
    expect(out.proposed_query).toBe(PROPOSED);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/search-strategy");
    expect(JSON.parse((init as RequestInit).body as string)).toEqual({ query: long, lang: "en" });
  });
});
