import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ConceptAnnotationStatus } from "./ConceptAnnotationStatus";
import { LanguageProvider } from "../i18n/LanguageProvider";
import { en } from "../i18n/locales/en";
import { clearApiKey, setApiKey } from "../lib/api";
import type { ConceptGraphData } from "../lib/api";

const A = en.scenarioDetail.knowledgeGraph.annotation;
const BASE: ConceptGraphData = {
  kind: "concepts", version: 2, scenario_id: "usr-abc", n_articles: 10, n_total: 12, n_with_concepts: 5,
  n_missing_concepts: 0, source: "llm", latest_year: 2024, types: [], nodes: [], edges: [], triples: [], gaps: [],
  articles: {},
};

function show(over: Partial<ConceptGraphData>, onStarted = vi.fn()) {
  localStorage.setItem("literev-lang", "en");
  const view = render(
    <LanguageProvider>
      <ConceptAnnotationStatus scenarioId="usr-abc" data={{ ...BASE, ...over }} onStarted={onStarted} />
    </LanguageProvider>,
  );
  return { onStarted, view };
}

function answer(body: unknown) {
  return { ok: true, status: 200, json: async () => body } as unknown as Response;
}

afterEach(() => {
  clearApiKey();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("ConceptAnnotationStatus", () => {
  it("says nothing when every relevant article is annotated", () => {
    const { view } = show({});
    expect(view.container.textContent).toBe("");
  });

  it("counts what is not annotated yet and what cannot be, and starts the pass only on request", async () => {
    setApiKey("secret");
    const fetchMock = vi.fn().mockResolvedValue(answer({ status: "started" }));
    vi.stubGlobal("fetch", fetchMock);
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const { onStarted } = show({ n_missing_concepts: 7, n_unannotatable: 3, n_concepts_given_up: 2, n_without_text: 1 });
    expect(screen.getByText(A.missing.replace("{n}", "7"))).toBeInTheDocument();
    expect(screen.getByText(A.unannotatable.replace("{n}", "3").replace("{given_up}", "2").replace("{no_text}", "1")))
      .toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();                     // showing the counts costs nothing

    fireEvent.click(screen.getByRole("button", { name: A.run.replace("{n}", "7") }));
    await waitFor(() => expect(onStarted).toHaveBeenCalledTimes(1));
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/user-scenarios/usr-abc/concept-graph/annotate");
    expect((init as RequestInit).method).toBe("POST");
    expect((init as RequestInit).headers).toEqual({ "X-API-Key": "secret" });

    fireEvent.click(screen.getByRole("button", { name: A.retry }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(fetchMock.mock.calls[1][0]).toBe("/api/user-scenarios/usr-abc/concept-graph/annotate?retry_given_up=true");
  });

  it("says why nothing started when no call can be made", async () => {
    setApiKey("secret");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(answer({ status: "paused", reason: "calls refused: budget" })));
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const { onStarted } = show({ n_missing_concepts: 4 });
    fireEvent.click(screen.getByRole("button", { name: A.run.replace("{n}", "4") }));
    expect(await screen.findByText(A.paused.replace("{reason}", "calls refused: budget"))).toBeInTheDocument();
    expect(onStarted).not.toHaveBeenCalled();
  });

  it("says when the last pass stopped early, and offers no button without the key", () => {
    show({ n_missing_concepts: 4, annotation: { status: "paused", reason: "4 failures in a row", annotated: 0, charged: 0, pending: 4 } });
    expect(screen.getByText(A.lastPaused.replace("{reason}", "4 failures in a row"))).toBeInTheDocument();
    expect(screen.getByText(A.needsKey)).toBeInTheDocument();
    expect(screen.queryByRole("button")).toBeNull();
  });
});
