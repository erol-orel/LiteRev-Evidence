import { afterEach, describe, expect, it, vi } from "vitest";
import {
  addCodebookSynonym, clearApiKey, pooledExportUrl, reverifyExtractionQuotes, setApiKey, startConceptAnnotation,
  startExtraction,
} from "./api";

function reply(status: number, body: unknown = {}): Response {
  return { ok: status >= 200 && status < 300, status, json: async () => body } as unknown as Response;
}

afterEach(() => {
  clearApiKey();
  vi.unstubAllGlobals();
});

describe("the extraction's own calls", () => {
  it("downloads every pooled label and comparison from the server, with the panel's filters", () => {
    expect(pooledExportUrl("usr-abc", { reviewedOnly: false, verifiedOnly: true, splitDisease: false }))
      .toBe("/api/user-scenarios/usr-abc/extraction/pooled/export?reviewed_only=false&verified_only=true&split_disease=false");
  });

  it("re-checks the quotes with the key, and says why when it is refused", async () => {
    setApiKey("secret");
    const counts = { n_articles: 3, n_articles_changed: 1, n_observations: 9, verified_before: 5, verified_after: 7,
                     newly_found: 2, no_longer_found: 0 };
    const fetchMock = vi.fn().mockResolvedValueOnce(reply(200, counts))
      .mockResolvedValueOnce(reply(409, { detail: "An extraction run is in progress" }));
    vi.stubGlobal("fetch", fetchMock);
    expect(await reverifyExtractionQuotes("usr-abc")).toEqual(counts);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/user-scenarios/usr-abc/extraction/reverify");
    expect(init).toEqual({ method: "POST", headers: { "X-API-Key": "secret" } });
    await expect(reverifyExtractionQuotes("usr-abc")).rejects.toThrow("An extraction run is in progress");
  });

  it("passes on why a run did not start", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(reply(200, { status: "paused", reason: "calls refused: budget" })));
    expect(await startExtraction("usr-abc")).toEqual({ status: "paused", reason: "calls refused: budget" });
  });

  it("starts the concept annotation of a built-in scenario through the same route", async () => {
    setApiKey("secret");
    const fetchMock = vi.fn().mockResolvedValue(reply(200, { status: "started" }));
    vi.stubGlobal("fetch", fetchMock);
    expect(await startConceptAnnotation("influenza-surveillance", true)).toEqual({ status: "started" });
    expect(fetchMock.mock.calls[0][0])
      .toBe("/api/user-scenarios/influenza-surveillance/concept-graph/annotate?retry_given_up=true");
    expect((fetchMock.mock.calls[0][1] as RequestInit).headers).toEqual({ "X-API-Key": "secret" });
  });

  it("shows why a synonym was refused instead of a bare status", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      reply(422, { detail: "human_exp > occupation > farmer already has 200 synonyms, the maximum" })));
    await expect(addCodebookSynonym("usr-abc", { sheet: "human_exp", l1: "occupation", l2: "farmer", label: "x" }))
      .rejects.toThrow("already has 200 synonyms");
  });
});
