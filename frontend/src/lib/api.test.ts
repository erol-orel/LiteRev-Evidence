import { afterEach, describe, expect, it, vi } from "vitest";
import {
  clearApiKey,
  fetchConceptGraph,
  fetchGesicaScenarios,
  fetchScenarioCorpus,
  fetchScenarioDetail,
  getApiKey,
  hasApiKey,
  httpMessage,
  patchScenarioSettings,
  patchUserScenario,
  populateUserScenario,
  safeFetch,
  setApiKey,
  startUserScenarioPipeline,
} from "./api";
import { en } from "../i18n/locales/en";

/** A minimal Response stand-in: only what the client reads. */
function reply(status: number, body: unknown = {}, headers: Record<string, string> = {}): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (name: string) => headers[name] ?? null },
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as unknown as Response;
}

function stubFetch(...responses: Response[]) {
  const fetchMock = vi.fn();
  for (const r of responses) fetchMock.mockResolvedValueOnce(r);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.useRealTimers();
});

describe("safeFetch", () => {
  it("returns a successful response without retrying", async () => {
    const fetchMock = stubFetch(reply(200, { ok: true }));
    const r = await safeFetch("/api/health");
    expect(r.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("retries a 429 with a back-off, for any method", async () => {
    vi.useFakeTimers();
    const fetchMock = stubFetch(reply(429), reply(200));
    const pending = safeFetch("/api/x", { method: "POST" });
    await vi.runAllTimersAsync();
    expect((await pending).status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("retries a 503 only for idempotent requests", async () => {
    vi.useFakeTimers();
    const getMock = stubFetch(reply(503), reply(200));
    const pendingGet = safeFetch("/api/x");
    await vi.runAllTimersAsync();
    expect((await pendingGet).status).toBe(200);
    expect(getMock).toHaveBeenCalledTimes(2);

    const postMock = stubFetch(reply(503), reply(200));
    const r = await safeFetch("/api/x", { method: "POST" });
    expect(r.status).toBe(503);
    expect(postMock).toHaveBeenCalledTimes(1);
  });

  it("gives up after the configured number of retries", async () => {
    vi.useFakeTimers();
    const fetchMock = stubFetch(reply(502), reply(502), reply(502), reply(200));
    const pending = safeFetch("/api/x", undefined, { retries: 2 });
    await vi.runAllTimersAsync();
    expect((await pending).status).toBe(502);
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("does not retry a client error", async () => {
    const fetchMock = stubFetch(reply(404), reply(200));
    expect((await safeFetch("/api/x")).status).toBe(404);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe("httpMessage", () => {
  it("maps the status to a message in the interface language", () => {
    localStorage.setItem("literev-lang", "en");
    expect(httpMessage(429)).toBe(en.errors.tooManyRequests);
    expect(httpMessage(401)).toBe(en.errors.unauthorized);
    expect(httpMessage(403)).toBe(en.errors.unauthorized);
    expect(httpMessage(404)).toBe(en.errors.notFound);
    expect(httpMessage(503)).toBe(en.errors.serviceUnavailable);
    expect(httpMessage(500)).toBe(en.errors.serverError);
    expect(httpMessage(418)).toBe(`${en.errors.genericPrefix} 418.`);
  });
});

describe("API key storage", () => {
  it("prefers the session key, persists on request and clears both", () => {
    expect(hasApiKey()).toBe(false);
    setApiKey("  persisted  ");
    expect(localStorage.getItem("api_key")).toBe("persisted");
    setApiKey("session-only", false);
    expect(sessionStorage.getItem("api_key")).toBe("session-only");
    expect(getApiKey()).toBe("session-only");
    clearApiKey();
    expect(getApiKey()).toBe("");
    setApiKey("   ");                                              // blank = remove
    expect(hasApiKey()).toBe(false);
  });
});

describe("calls that start server-side work carry the interface language", () => {
  it("passes lang when pinning, starting the pipeline and building the corpus", async () => {
    localStorage.setItem("literev-lang", "en");
    setApiKey("secret");
    const fetchMock = stubFetch(reply(200, {}), reply(200, {}), reply(200, {}));
    await patchUserScenario("usr-abc", { pinned: true });
    await startUserScenarioPipeline("usr-abc", 500);
    await populateUserScenario("usr-abc", { includeLive: true, maxResults: 2000 });
    const urls = fetchMock.mock.calls.map((c) => String(c[0]));
    expect(urls[0]).toBe("/api/user-scenarios/usr-abc?lang=en");
    expect(urls[1]).toBe("/api/user-scenarios/usr-abc/pipeline?max_results=500&lang=en");
    expect(urls[2]).toBe("/api/user-scenarios/usr-abc/populate?max_results=2000&include_live=true&lang=en");
  });
});

describe("scenario endpoints", () => {
  it("builds the corpus query string from the options", async () => {
    const fetchMock = stubFetch(reply(200, { articles: [], total: 0 }));
    await fetchScenarioCorpus("usr-abc", { limit: 200, abstractChars: 600, threshold: 0.42, fulltextOnly: true });
    expect(fetchMock.mock.calls[0][0])
      .toBe("/api/user-scenarios/usr-abc/corpus?limit=200&fulltext_only=true&threshold=0.42&abstract_chars=600");
  });

  it("reads the concept map from the scenario's own base path", async () => {
    const fetchMock = stubFetch(reply(200, { kind: "concepts", nodes: [] }), reply(200, { kind: "concepts", nodes: [] }));
    await fetchConceptGraph("usr-abc");
    await fetchConceptGraph("influenza-surveillance", true);
    expect(fetchMock.mock.calls[0][0]).toBe("/api/user-scenarios/usr-abc/concept-graph");
    expect(fetchMock.mock.calls[1][0]).toBe("/api/gesica/scenarios/influenza-surveillance/concept-graph?refresh=true");
  });

  it("routes built-in and user scenarios to their own base path, with the language", async () => {
    localStorage.setItem("literev-lang", "en");
    const fetchMock = stubFetch(reply(200, {}), reply(200, {}));
    await fetchScenarioDetail("usr-abc");
    await fetchScenarioDetail("influenza-surveillance");
    expect(fetchMock.mock.calls[0][0]).toBe("/api/user-scenarios/usr-abc/detail?lang=en");
    expect(fetchMock.mock.calls[1][0]).toBe("/api/gesica/scenarios/influenza-surveillance/detail?lang=en");
  });

  it("asks for the catalogue in the interface language and maps the rows", async () => {
    localStorage.setItem("literev-lang", "fr");
    const row = {
      id: "s1", title: "Titre", label_short: "T", description: "d", cluster: "c", article_count: 3,
      living_evidence_note: "note", recommended_actions: ["a"], relevant_articles: [],
    };
    const fetchMock = stubFetch(reply(200, [row]));
    const out = await fetchGesicaScenarios();
    expect(fetchMock.mock.calls[0][0]).toBe("/api/gesica/scenarios?lang=fr");
    expect(out).toEqual([{
      id: "s1", title: "Titre", labelShort: "T", description: "d", cluster: "c", articleCount: 3,
      livingEvidenceNote: "note", recommendedActions: ["a"], relevantArticles: [],
    }]);
  });

  it("sends the admin key with a mutation and turns a 401 into a readable error", async () => {
    localStorage.setItem("literev-lang", "en");
    setApiKey("secret");
    const fetchMock = stubFetch(reply(200, { status: "ok", scenario_id: "usr-abc", updated: ["similarity_threshold"] }));
    await patchScenarioSettings("usr-abc", { similarity_threshold: 0.5 });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/scenarios/usr-abc/settings");
    expect(init).toEqual({
      method: "PATCH",
      headers: { "X-API-Key": "secret", "Content-Type": "application/json" },
      body: JSON.stringify({ similarity_threshold: 0.5 }),
    });

    clearApiKey();
    const denied = stubFetch(reply(401));
    await expect(patchScenarioSettings("usr-abc", { similarity_threshold: 0.5 })).rejects.toThrow(en.errors.unauthorized);
    expect((denied.mock.calls[0][1] as RequestInit).headers).toEqual({ "Content-Type": "application/json" });
  });
});
