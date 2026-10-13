import { afterEach, describe, expect, it, vi } from "vitest";
import {
  askScenarioRagStreamFiltered,
  clearApiKey,
  exportArticleIds,
  exportConceptArticles,
  exportRelevantArticles,
  exportScenarioCorpus,
  fetchScenarioQuestionsUpTo,
  fetchUnmappedLabelsUpTo,
  startEnrichmentBatch,
  fetchConceptGraph,
  fetchGesicaScenarios,
  fetchScenarioCorpus,
  fetchScenarioDetail,
  fetchSearchLimits,
  fetchThresholdCurve,
  applyScenarioSubset,
  previewScenarioSubset,
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

/** A streamed SSE response, delivered in the pieces given (so a line can straddle two). */
function sseReply(pieces: string[]): Response {
  const encoder = new TextEncoder();
  let i = 0;
  return {
    ok: true,
    status: 200,
    headers: { get: () => null },
    body: {
      getReader: () => ({
        read: async () => (i < pieces.length
          ? { done: false, value: encoder.encode(pieces[i++]) }
          : { done: true, value: undefined }),
      }),
    },
  } as unknown as Response;
}

describe("the assistant stream", () => {
  it("reports a cut answer before it is done, and lets the server choose top_k", async () => {
    const fetchMock = stubFetch(sseReply([
      'event: meta\ndata: {"papers_used": 40, "papers_with_fulltext": 3, "threshold": 0.45}\n\n',
      'data: {"token": "An answer that "}\n\ndata: {"tok',
      'en": "stops mid"}\n\n',
      'event: cut\ndata: {"cut": true, "reason": "length", "max_tokens": 3000}\n\n',
      "event: done\ndata: {}\n\n",
    ]));
    const order: string[] = [];
    let text = "";
    await new Promise<void>((resolve, reject) => {
      askScenarioRagStreamFiltered("usr-abc", "What did the studies find?", {
        onSources: () => order.push("sources"),
        onMeta: () => order.push("meta"),
        onToken: (t) => { text += t; },
        onCut: () => order.push("cut"),
        onDone: () => { order.push("done"); resolve(); },
        onError: (e) => reject(new Error(e)),
      });
    });
    expect(text).toBe("An answer that stops mid");
    expect(order).toEqual(["meta", "cut", "done"]);
    const body = JSON.parse(String((fetchMock.mock.calls[0][1] as RequestInit).body));
    expect(body).not.toHaveProperty("top_k");
    expect(body.scenario_id).toBe("usr-abc");
  });

  it("does not report a cut for a complete answer", async () => {
    stubFetch(sseReply(['data: {"token": "Whole."}\n\n', "event: done\ndata: {}\n\n"]));
    let cut = false;
    await new Promise<void>((resolve, reject) => {
      askScenarioRagStreamFiltered("usr-abc", "Q?", {
        onSources: () => {}, onToken: () => {}, onCut: () => { cut = true; },
        onDone: () => resolve(), onError: (e) => reject(new Error(e)),
      });
    });
    expect(cut).toBe(false);
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

  it("sends no search mode and leaves the per-source cap to the server", async () => {
    localStorage.setItem("literev-lang", "en");
    setApiKey("secret");
    const fetchMock = stubFetch(reply(200, {}), reply(200, {}), reply(200, {}));
    // What the search screen sends: never a cap of its own, and no mode: every search is
    // exhaustive.
    await populateUserScenario("usr-abc", { includeLive: true, excludePreprints: false });
    await populateUserScenario("usr-abc", { includeLive: false });
    // A pipeline started without a cap: the server applies LIVE_MAX_PER_SOURCE, as the
    // search does. The old default sent max_results=100000.
    await startUserScenarioPipeline("usr-abc");
    const urls = fetchMock.mock.calls.map((c) => String(c[0]));
    expect(urls[0]).toBe("/api/user-scenarios/usr-abc/populate?include_live=true&lang=en&exclude_preprints=false");
    expect(urls[1]).toBe("/api/user-scenarios/usr-abc/populate?include_live=false&lang=en");
    expect(urls[2]).toBe("/api/user-scenarios/usr-abc/pipeline?lang=en");
    for (const u of urls) {
      expect(u).not.toContain("max_results");
      expect(u).not.toContain("exhaustive");
    }
  });

  it("reads the server's caps and time budget for the search screen", async () => {
    const fetchMock = stubFetch(reply(200, { per_source: 100000, ranked_per_source: 2000, budget_s: 2700 }),
                                reply(200, { standard_per_source: 2000, exhaustive_per_source: 20000, ranked_per_source: 2000 }));
    expect(await fetchSearchLimits()).toEqual({ per_source: 100000, ranked_per_source: 2000, budget_s: 2700 });
    // An older server's answer: the screen keeps its defaults.
    expect(await fetchSearchLimits()).toBeNull();
    expect(String(fetchMock.mock.calls[0][0])).toBe("/api/search/limits");
  });
});

describe("scenario endpoints", () => {
  it("builds the corpus query string from the options", async () => {
    const fetchMock = stubFetch(reply(200, { articles: [], total: 0 }));
    await fetchScenarioCorpus("usr-abc", { limit: 200, abstractChars: 600, threshold: 0.42, fulltextOnly: true });
    expect(fetchMock.mock.calls[0][0])
      .toBe("/api/user-scenarios/usr-abc/corpus?limit=200&fulltext_only=true&threshold=0.42&abstract_chars=600");
  });

  it("lists the records kept without an abstract on request", async () => {
    const fetchMock = stubFetch(reply(200, { articles: [], total: 0 }), reply(200, { articles: [], total: 0 }));
    await fetchScenarioCorpus("usr-abc", { limit: 100, noAbstract: true });
    expect(fetchMock.mock.calls[0][0]).toBe("/api/user-scenarios/usr-abc/corpus?limit=100&no_abstract=true");
    await fetchScenarioCorpus("usr-abc", { limit: 100, noAbstract: false });
    expect(fetchMock.mock.calls[1][0]).toBe("/api/user-scenarios/usr-abc/corpus?limit=100");
  });

  it("exports the relevant articles as a named file", async () => {
    const body = { ok: true, status: 200, headers: { get: (n: string) => n === "Content-Disposition" ? 'attachment; filename="chik_relevant-articles_12.ris"' : null },
      blob: async () => new Blob(["TY  - JOUR"]), json: async () => ({}), text: async () => "" } as unknown as Response;
    const fetchMock = stubFetch(body);
    const out = await exportRelevantArticles("usr-abc", "ris");
    expect(fetchMock.mock.calls[0][0]).toBe("/api/user-scenarios/usr-abc/relevant/export?format=ris");
    expect(out.filename).toBe("chik_relevant-articles_12.ris");
    expect(await out.blob.text()).toBe("TY  - JOUR");
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

  it("asks for a threshold curve, with the target only when there is one", async () => {
    const body = {
      scenario_id: "usr-abc", current_threshold: 0.45, candidates: 2, included: 0,
      unscored: 0, corpus: 2, with_parameter_total: 1, scoring_in_progress: false,
      curve: [], suggestion: null,
    };
    const fetchMock = stubFetch(reply(200, body), reply(200, body), reply(200, body));
    expect(await fetchThresholdCurve("usr-abc", 100)).toEqual(body);
    expect(fetchMock.mock.calls[0][0]).toBe("/api/scenarios/usr-abc/threshold-curve?target=100");
    await fetchThresholdCurve("usr-abc");
    expect(fetchMock.mock.calls[1][0]).toBe("/api/scenarios/usr-abc/threshold-curve");
    // A blank or nonsense box must not travel as `target=NaN`: the API would 422 it.
    await fetchThresholdCurve("usr-abc", Number.NaN);
    expect(fetchMock.mock.calls[2][0]).toBe("/api/scenarios/usr-abc/threshold-curve");
  });

  it("previews a narrowing without the write key and applies it with one", async () => {
    const plan = {
      relevant: 5, keep: 3, exclude: 2, undecided: 1, by_dimension: {}, combine: "all",
      unassigned: "keep", reason: "scope: ...", meta: {}, exclude_sample: [1, 2],
      undecided_sample: [3],
    };
    setApiKey("secret");
    const fetchMock = stubFetch(reply(200, plan), reply(200, { ...plan, applied: 2 }));
    await previewScenarioSubset("usr-abc", { clusters: [0, 1] });
    const [previewUrl, previewInit] = fetchMock.mock.calls[0];
    expect(previewUrl).toBe("/api/user-scenarios/usr-abc/subset/preview");
    // A preview writes nothing, so it must not carry the admin key around.
    expect((previewInit as RequestInit).headers).toEqual({ "Content-Type": "application/json" });
    expect((previewInit as RequestInit).body).toBe(JSON.stringify({ clusters: [0, 1] }));

    await applyScenarioSubset("usr-abc", { clusters: [0], unassigned: "exclude" });
    const [applyUrl, applyInit] = fetchMock.mock.calls[1];
    expect(applyUrl).toBe("/api/user-scenarios/usr-abc/subset/apply");
    expect((applyInit as RequestInit).headers).toEqual({
      "X-API-Key": "secret", "Content-Type": "application/json",
    });
    clearApiKey();
  });

  it("surfaces the API's own reason when a narrowing is refused", async () => {
    stubFetch(reply(404, { detail: "Cluster 42 inconnu (disponibles : [0, 1])." }));
    await expect(previewScenarioSubset("usr-abc", { clusters: [42] }))
      .rejects.toThrow("Cluster 42 inconnu");
  });
});

/** A file the server sends back, with its name in Content-Disposition. */
function fileReply(filename: string, content = "x"): Response {
  return {
    ok: true, status: 200,
    headers: { get: (n: string) => (n === "Content-Disposition" ? `attachment; filename="${filename}"` : null) },
    blob: async () => new Blob([content]), json: async () => ({}), text: async () => content,
  } as unknown as Response;
}

describe("the full set, never the screen's cut", () => {
  it("asks the server to sort the whole corpus by year, and sends nothing for relevance", async () => {
    const fetchMock = stubFetch(reply(200, { articles: [], total: 0 }), reply(200, { articles: [], total: 0 }));
    await fetchScenarioCorpus("usr-abc", { limit: 200, abstractChars: 600, sort: "year_desc" });
    await fetchScenarioCorpus("usr-abc", { limit: 200, sort: "relevance" });
    expect(fetchMock.mock.calls[0][0]).toBe("/api/user-scenarios/usr-abc/corpus?limit=200&sort=year_desc&abstract_chars=600");
    expect(fetchMock.mock.calls[1][0]).toBe("/api/user-scenarios/usr-abc/corpus?limit=200");
  });

  it("exports the whole corpus of a search from its own route", async () => {
    const fetchMock = stubFetch(fileReply("q_whole-corpus_812.xlsx"));
    const out = await exportScenarioCorpus("usr-abc", "xlsx");
    expect(fetchMock.mock.calls[0][0]).toBe("/api/user-scenarios/usr-abc/corpus/export?format=xlsx");
    expect(out.filename).toBe("q_whole-corpus_812.xlsx");
  });

  it("sends a list of ids in the body of a POST, not in the URL", async () => {
    const ids = Array.from({ length: 3000 }, (_, i) => 100000 + i);
    const fetchMock = stubFetch(fileReply("q_filtered-map_3000.csv"));
    await exportArticleIds("usr-abc", ids, "csv", { label: "Filtered map" });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/user-scenarios/usr-abc/articles/export");
    expect((init as RequestInit).method).toBe("POST");
    expect(JSON.parse(String((init as RequestInit).body))).toEqual({ ids, format: "csv", label: "Filtered map" });
  });

  it("exports the concept map as composed, as filters rather than the concepts drawn", async () => {
    const fetchMock = stubFetch(fileReply("q_concept-map_40.ris"));
    await exportConceptArticles("usr-abc", [], "ris", {
      composed: { exclude_types: ["place", "vector"], min_count: 3, exclude: ["pathogen:dengue virus"] },
    });
    const url = new URL(String(fetchMock.mock.calls[0][0]), "http://localhost");
    expect(url.pathname).toBe("/api/user-scenarios/usr-abc/concepts/export");
    expect(url.searchParams.get("composed")).toBe("true");
    expect(url.searchParams.get("exclude_types")).toBe("place,vector");
    expect(url.searchParams.get("min_count")).toBe("3");
    expect(url.searchParams.get("exclude")).toBe("pathogen:dengue virus");
    expect(url.searchParams.has("concepts")).toBe(false);
  });

  it("starts an enrichment batch for every pending article unless a size is given", async () => {
    setApiKey("secret");
    const job = { status: "started", job: { kind: "pico", status: "running", processed: 0, total: 12 } };
    const fetchMock = stubFetch(reply(200, job), reply(200, job));
    await startEnrichmentBatch("pico", "usr-abc");
    await startEnrichmentBatch("fulltext", "usr-abc", 50, "relevant");
    expect(fetchMock.mock.calls[0][0]).toBe("/api/pico/extract?scenario_id=usr-abc");
    expect((fetchMock.mock.calls[0][1] as RequestInit).method).toBe("POST");
    expect(fetchMock.mock.calls[1][0]).toBe("/api/fulltext/fetch?limit=50&scenario_id=usr-abc&scope=relevant");
    clearApiKey();
  });

  it("reaches every unmapped label and every question, page after page", async () => {
    const labels = (from: number, n: number) => Array.from({ length: n }, (_, i) => ({ covariate: `l${from + i}` }));
    const fetchMock = stubFetch(
      reply(200, { n_distinct_unmapped: 650, unmapped: labels(0, 500) }),
      reply(200, { n_distinct_unmapped: 650, unmapped: labels(500, 150) }),
    );
    const out = await fetchUnmappedLabelsUpTo("usr-abc", 700);
    expect(out.unmapped).toHaveLength(650);
    expect(fetchMock.mock.calls[1][0]).toBe("/api/user-scenarios/usr-abc/codebook/unmapped?top=200&offset=500");

    const q = (from: number, n: number) => Array.from({ length: n }, (_, i) => ({ id: from + i }));
    const qMock = stubFetch(reply(200, { total: 120, items: q(0, 100) }), reply(200, { total: 120, items: q(100, 20) }));
    const page = await fetchScenarioQuestionsUpTo("usr-abc", 150);
    expect(page.items).toHaveLength(120);
    expect(qMock.mock.calls[1][0]).toBe("/api/user-scenarios/usr-abc/questions?limit=50&offset=100");
  });
});
