import React from "react";
import {
  AlertTriangle, CheckCircle2, ChevronDown, ChevronUp, Download, Info, Loader2,
  RotateCcw, Sparkles, Table2,
} from "lucide-react";
import { currentLang, useI18n } from "../i18n/LanguageProvider";
import { CodebookPanel } from "./CodebookPanel";
import { GeographyPanel } from "./GeographyPanel";
import { ObservationPanel } from "./ObservationPanel";
import { PooledPanel } from "./PooledPanel";
import type { Detail } from "./ObservationPanel";
import { ReviewSummaryCard } from "./ReviewSummaryCard";
import {
  extractionExportUrl, extractionReportUrl, fetchAllExtractionArticles, fetchArticleExtraction, fetchExtractionDigest,
  fetchExtractionStatus, hasApiKey, startExtraction,
} from "../lib/api";
import type {
  ExtractionArticle, ExtractionArticlesResponse, ExtractionDigest, ExtractionObservation, ExtractionStatus,
} from "../lib/api";
import { COVERAGE_KEYS, SHEET_KEYS, filterArticles, quotesToCheck, share, sortArticles } from "../lib/extraction";
import type { CoverageKey, ExtractionFilter, ExtractionSort, ItemMode } from "../lib/extraction";

const PAGE = 100;
const POLL_MS = 4000;

/** Structured extraction: what each relevant paper reports, in the review template's shape,
 *  with the exact quote behind every value. Counts always say out of how many papers. */
export function ExtractionSection({ scenarioId }: { scenarioId: string }) {
  const { t } = useI18n();
  const T = (k: string) => t(`scenarioDetail.extraction.${k}`);

  const [status, setStatus] = React.useState<ExtractionStatus | null>(null);
  const [digest, setDigest] = React.useState<ExtractionDigest | null>(null);
  const [list, setList] = React.useState<ExtractionArticlesResponse | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [notice, setNotice] = React.useState<string | null>(null);

  const [filter, setFilter] = React.useState<ExtractionFilter>("all");
  const [search, setSearch] = React.useState("");
  const [sortBy, setSortBy] = React.useState<ExtractionSort>("year");
  const [item, setItem] = React.useState<CoverageKey | null>(null);
  const [itemMode, setItemMode] = React.useState<ItemMode>("reports");
  // How many rows are listed. Tied to the view it was set for, so a new filter or sort
  // starts again at the first page without an effect to reset it.
  const viewKey = [filter, search, item, itemMode, sortBy].join("|");
  const [shownRows, setShownRows] = React.useState({ key: viewKey, n: PAGE });
  const visible = shownRows.key === viewKey ? shownRows.n : PAGE;
  const [open, setOpen] = React.useState<number | null>(null);
  const [details, setDetails] = React.useState<Record<number, Detail>>({});
  // Who is reviewing: a name kept in this browser. It labels each decision and the agreement
  // between reviewers; the application has one write key, not user accounts.
  const [reviewer, setReviewerState] = React.useState<string>(() => {
    try { return localStorage.getItem("literev-reviewer") ?? ""; } catch { return ""; }
  });
  const setReviewer = (v: string) => {
    setReviewerState(v);
    try { localStorage.setItem("literev-reviewer", v); } catch { /* storage unavailable */ }
  };
  const [reviewTick, setReviewTick] = React.useState(0);

  const load = React.useCallback(
    () => Promise.all([
      fetchExtractionStatus(scenarioId),
      fetchExtractionDigest(scenarioId),
      fetchAllExtractionArticles(scenarioId),
    ]),
    [scenarioId],
  );
  const apply = React.useCallback(
    ([st, dg, ls]: [ExtractionStatus, ExtractionDigest, ExtractionArticlesResponse]) => {
      setStatus(st);
      setDigest(dg);
      setList(ls);
    },
    [],
  );
  const reload = React.useCallback(() => load().then(apply), [load, apply]);

  // The parent keys this component by scenario, so a new scenario starts from a clean state.
  React.useEffect(() => {
    let alive = true;
    load()
      .then((r) => { if (alive) apply(r); })
      .catch((e: Error) => { if (alive) setError(e.message); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [load, apply]);

  // While a run is going: follow its progress, and refresh the figures now and then.
  const running = !!status?.running;
  React.useEffect(() => {
    if (!running) return;
    let tick = 0;
    const id = window.setInterval(() => {
      tick += 1;
      fetchExtractionStatus(scenarioId)
        .then((st) => {
          setStatus(st);
          if (!st.running || tick % 3 === 0) reload().catch(() => undefined);
        })
        .catch(() => undefined);
    }, POLL_MS);
    return () => window.clearInterval(id);
  }, [running, scenarioId, reload]);

  const run = async () => {
    if (!status) return;
    if (!window.confirm(T("confirmRun").replace("{n}", String(status.n_pending)))) return;
    setNotice(null);
    try {
      const r = await startExtraction(scenarioId);
      if (r.status === "no_llm") setNotice(T("runNoLlm"));
      else setStatus(await fetchExtractionStatus(scenarioId));
    } catch (e) {
      setNotice(`${T("runFailed")} ${(e as Error).message}`);
    }
  };

  const toggle = (a: ExtractionArticle) => {
    if (!a.has_extraction) return;
    const next = open === a.id ? null : a.id;
    setOpen(next);
    if (next !== null && !details[a.id]) {
      setDetails((d) => ({ ...d, [a.id]: { loading: true } }));
      fetchArticleExtraction(scenarioId, a.id)
        .then((data) => setDetails((d) => ({ ...d, [a.id]: { loading: false, data } })))
        .catch((e: Error) => setDetails((d) => ({ ...d, [a.id]: { loading: false, error: e.message } })));
    }
  };

  const afterReview = (articleId: number, updated: ExtractionObservation | null) => {
    if (updated?.obs_key) {
      setDetails((d) => {
        const cur = d[articleId];
        const ex = cur?.data?.extraction;
        if (!cur || !ex) return d;
        const observations = ex.observations.map((o) => (o.obs_key === updated.obs_key ? { ...o, ...updated } : o));
        return { ...d, [articleId]: { ...cur, data: { ...cur.data!, extraction: { ...ex, observations } } } };
      });
    } else {
      setDetails((d) => { const n = { ...d }; delete n[articleId]; return n; });    // re-read on next open
      setOpen(null);
    }
    setReviewTick((n) => n + 1);                                    // the counts and the summary
    reload().catch(() => undefined);
  };

  const shown = React.useMemo(
    () => sortArticles(filterArticles(list?.articles ?? [], { filter, search, item, itemMode }), sortBy),
    [list, filter, search, item, itemMode, sortBy],
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8 text-white/50 gap-2">
        <RotateCcw size={16} className="animate-spin" /><span className="text-sm">{T("loading")}</span>
      </div>
    );
  }
  if (error || !status || !digest || !list) {
    return (
      <div className="rounded-2xl border border-rose-500/20 bg-rose-500/5 px-4 py-3 text-sm text-rose-300">
        <AlertTriangle size={14} className="inline mr-2" />{error ?? T("error")}
      </div>
    );
  }

  const extracted = status.n_extracted;
  const progress = share(extracted, status.n_relevant);
  const toCheck = list.articles.reduce((n, a) => n + quotesToCheck(a), 0);
  const sheetName = (k: string) => T(`sheet.${k}`);

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-3 mb-4">
        <div className="rounded-xl border border-white/10 bg-white/5 p-2 shrink-0">
          <Table2 size={14} className="text-brand-400" />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-white uppercase tracking-wider">{T("title")}</h3>
          <p className="text-xs text-white/50 mt-0.5">{T("subtitle")}</p>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        {[
          { label: T("statRelevant"), value: status.n_relevant, color: "text-white" },
          { label: T("statExtracted"), value: extracted, color: "text-brand-300" },
          { label: T("statFulltext"), value: status.n_from_fulltext, color: "text-brand-300" },
          { label: T("statAbstract"), value: status.n_from_abstract, color: status.n_from_abstract ? "text-gold-400" : "text-white/50" },
          { label: T("statRows"), value: status.n_observations, color: "text-white" },
        ].map((s) => (
          <div key={s.label} className="rounded-2xl border border-white/5 bg-white/3 p-3 text-center">
            <div className={`text-xl font-bold ${s.color}`}>{s.value.toLocaleString()}</div>
            <div className="text-[10px] text-white/40 mt-0.5">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Progress, run, downloads */}
      <div className="rounded-xl border border-white/5 bg-white/2 p-3 space-y-3">
        <div>
          <div className="flex justify-between text-[10px] text-white/40 mb-1.5">
            <span>{T("progressLabel")}</span>
            <span>{extracted}/{status.n_relevant} ({progress}%)</span>
          </div>
          <div className="h-2 bg-white/5 rounded-full overflow-hidden">
            <div className="h-full bg-brand-500 rounded-full transition-all" style={{ width: `${progress}%` }} />
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {status.running ? (
            <span className="flex items-center gap-1.5 rounded-xl border border-gold-400/30 bg-gold-400/10 px-3 py-1.5 text-[11px] text-gold-400">
              <Loader2 size={12} className="animate-spin" />
              {T("runningLabel")
                .replace("{done}", String((status.job?.done ?? 0) + (status.job?.failed ?? 0)))
                .replace("{total}", String(status.job?.total ?? status.n_pending))}
            </span>
          ) : status.n_pending > 0 && hasApiKey() ? (
            <button onClick={run}
              className="flex items-center gap-1.5 rounded-xl border border-brand-500/40 bg-brand-500/15 px-3 py-1.5 text-[11px] font-medium text-brand-300 hover:bg-brand-500/25 transition">
              <Sparkles size={12} />{T("runPending").replace("{n}", String(status.n_pending))}
            </button>
          ) : status.n_pending > 0 ? (
            <span className="text-[10px] text-white/40">{T("runNeedsKey")}</span>
          ) : (
            <span className="flex items-center gap-1.5 text-[11px] text-brand-300">
              <CheckCircle2 size={12} />{T("runNothing")}
            </span>
          )}
          {extracted > 0 && (["xlsx", "csv"] as const).map((f) => (
            <a key={f} href={extractionExportUrl(scenarioId, f)} download
              className="flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-[11px] text-white/70 hover:bg-white/10 transition">
              <Download size={11} />{f === "xlsx" ? T("downloadExcel") : T("downloadCsv")}
            </a>
          ))}
          {extracted > 0 && (["pdf", "docx"] as const).map((f) => (
            <a key={f} href={extractionReportUrl(scenarioId, f, currentLang())} download
              className="flex items-center gap-1.5 rounded-xl border border-brand-500/30 bg-brand-500/10 px-3 py-1.5 text-[11px] text-brand-300 hover:bg-brand-500/20 transition">
              <Download size={11} />{f === "pdf" ? T("downloadReportPdf") : T("downloadReportWord")}
            </a>
          ))}
        </div>
        {notice && <p className="text-[11px] text-rose-300">{notice}</p>}
        {status.n_given_up > 0 && (
          <p className="text-[10px] text-white/40">{T("givenUp").replace("{n}", String(status.n_given_up))}</p>
        )}
      </div>

      {status.n_from_abstract > 0 && (
        <div className="flex items-start gap-2 rounded-xl border border-gold-400/20 bg-gold-400/5 px-3 py-2 text-[11px] text-gold-400/90">
          <Info size={13} className="mt-0.5 shrink-0" />
          <span>{T("abstractWarning").replace("{n}", String(status.n_from_abstract))}</span>
        </div>
      )}

      {extracted === 0 ? (
        <div className="rounded-2xl border border-white/5 bg-white/2 px-4 py-8 text-center text-xs text-white/45">
          {T("nothingYet")}
        </div>
      ) : (
        <>
          {/* What the papers report: click to filter the table */}
          <div className="space-y-2">
            <div>
              <h4 className="text-xs font-semibold text-white/80 uppercase tracking-wider">{T("coverageTitle")}</h4>
              <p className="text-[10px] text-white/40 mt-0.5">{T("coverageHint").replace("{n}", String(extracted))}</p>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              {COVERAGE_KEYS.map((k) => {
                const n = digest.coverage[k] ?? 0;
                const ft = digest.coverage_fulltext[k] ?? 0;
                const sel = item === k;
                return (
                  <button key={k} onClick={() => setItem(sel ? null : k)} aria-pressed={sel}
                    className={`text-left rounded-2xl border p-3 transition ${
                      sel ? "border-brand-500/60 bg-brand-500/10" : "border-white/5 bg-white/3 hover:bg-white/6"}`}>
                    <div className="text-[10px] text-white/55 leading-4 min-h-[2rem]">{T(`cov.${k}`)}</div>
                    <div className="mt-1 flex items-baseline gap-1.5">
                      <span className="text-xl font-bold text-white">{n}</span>
                      <span className="text-[10px] text-white/40">{share(n, extracted)}%</span>
                    </div>
                    <div className="mt-1.5 h-1.5 rounded-full bg-white/5 overflow-hidden flex">
                      <div className="h-full bg-brand-500" style={{ width: `${share(ft, extracted)}%` }} />
                      <div className="h-full bg-brand-500/35" style={{ width: `${share(n - ft, extracted)}%` }} />
                    </div>
                    <div className="mt-1 text-[9px] text-white/35">
                      {T("coverageOf").replace("{n}", String(extracted))} · {T("coverageFulltext").replace("{n}", String(ft))}
                    </div>
                  </button>
                );
              })}
            </div>
            {item && (
              <div className="flex flex-wrap items-center gap-2 text-[10px]">
                <span className="text-white/50">{T(`cov.${item}`)}:</span>
                {(["reports", "missing"] as const).map((m) => (
                  <button key={m} onClick={() => setItemMode(m)}
                    className={`px-2.5 py-1 rounded-lg font-medium transition ${
                      itemMode === m ? "bg-brand-700 text-gold-400 font-semibold" : "text-white/60 hover:text-white hover:bg-white/8"}`}>
                    {m === "reports" ? T("modeReports") : T("modeMissing")}
                  </button>
                ))}
                <button onClick={() => setItem(null)} className="text-white/40 hover:text-white underline">{T("clearItem")}</button>
              </div>
            )}
          </div>

          <CodebookPanel scenarioId={scenarioId} onChanged={() => { void reload(); }} />

          <GeographyPanel scenarioId={scenarioId} />

          {/* Rows by sheet */}
          {digest.by_sheet.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-xs font-semibold text-white/80 uppercase tracking-wider">{T("sheetsTitle")}</h4>
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                {SHEET_KEYS.map((k) => {
                  const r = digest.by_sheet.find((x) => x.sheet === k);
                  return (
                    <div key={k} className="rounded-2xl border border-white/5 bg-white/3 p-3">
                      <div className="text-[10px] text-white/55">{sheetName(k)}</div>
                      <div className="text-lg font-bold text-white mt-0.5">{(r?.n_rows ?? 0).toLocaleString()}</div>
                      <div className="text-[9px] text-white/35">
                        {T("sheetDetail")
                          .replace("{articles}", String(r?.n_articles ?? 0))
                          .replace("{pct}", String(share(r?.n_quote_found ?? 0, r?.n_rows ?? 0)))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </>
      )}

      {extracted > 0 && (
        <ReviewSummaryCard scenarioId={scenarioId} reviewer={reviewer} onReviewer={setReviewer} tick={reviewTick} />
      )}

      {extracted > 0 && <PooledPanel scenarioId={scenarioId} tick={reviewTick} />}

      {/* Controls */}
      <div className="flex flex-wrap gap-2 items-center">
        <input type="text" placeholder={T("searchPlaceholder")} value={search} onChange={(e) => setSearch(e.target.value)}
          className="flex-1 min-w-[200px] rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-white focus:outline-none focus:border-brand-500/50" />
        <div className="flex gap-1">
          {(["all", "extracted", "pending", "quote_missing", "to_review", "conflict"] as const).map((f) => (
            <button key={f} onClick={() => setFilter(f)}
              className={`px-2.5 py-1.5 rounded-lg text-[10px] font-medium transition ${
                filter === f ? "bg-brand-700 text-gold-400 font-semibold" : "text-white/60 hover:text-white hover:bg-white/8"}`}>
              {T(`filter.${f}`)}{f === "quote_missing" && toCheck > 0 ? ` (${toCheck})` : ""}
            </button>
          ))}
        </div>
        <select value={sortBy} onChange={(e) => setSortBy(e.target.value as ExtractionSort)}
          className="rounded-lg border border-white/10 bg-white/5 px-2 py-1.5 text-[10px] text-white/70 focus:outline-none">
          <option value="year">{T("sortYear")}</option>
          <option value="rows">{T("sortRows")}</option>
          <option value="quotes">{T("sortQuotes")}</option>
        </select>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-2xl border border-white/5">
        <table className="w-full text-[10px] border-collapse">
          <thead>
            <tr className="border-b border-white/5 bg-white/3">
              {["colTitle", "colYear", "colSource", "colReports", "colRows", "colQuotes", "colReview"].map((h) => (
                <th key={h} className="text-left px-3 py-2 text-white/40 font-semibold uppercase tracking-wider whitespace-nowrap">{T(h)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.slice(0, visible).map((a, i) => {
              const check = quotesToCheck(a);
              const isOpen = open === a.id;
              return (
                <React.Fragment key={a.id}>
                  <tr onClick={() => toggle(a)}
                    className={`border-b border-white/5 transition ${i % 2 === 0 ? "bg-white/1" : "bg-transparent"} ${
                      a.has_extraction ? "cursor-pointer hover:bg-white/4" : ""}`}>
                    <td className="px-3 py-2 max-w-[260px]">
                      <div className="flex items-start gap-1.5">
                        {a.has_extraction
                          ? (isOpen ? <ChevronUp size={11} className="mt-0.5 text-brand-300 shrink-0" /> : <ChevronDown size={11} className="mt-0.5 text-white/40 shrink-0" />)
                          : <span className="mt-1 h-1.5 w-1.5 rounded-full bg-white/20 shrink-0" />}
                        <span className="text-white/70 leading-4 line-clamp-2">{a.title}</span>
                      </div>
                    </td>
                    <td className="px-3 py-2 text-white/50 font-mono whitespace-nowrap">{a.year || "-"}</td>
                    <td className="px-3 py-2 whitespace-nowrap">
                      {a.source === "fulltext" && <span className="rounded-md bg-brand-500/10 border border-brand-500/20 px-1.5 py-0.5 text-brand-300">{T("sourceFulltext")}</span>}
                      {a.source === "abstract" && <span className="rounded-md bg-gold-400/10 border border-gold-400/20 px-1.5 py-0.5 text-gold-400">{T("sourceAbstract")}</span>}
                      {!a.source && <span className="text-white/25">{T("sourceNone")}</span>}
                      {a.text_truncated && <span className="ml-1 text-white/35" title={T("textTruncated")}>…</span>}
                    </td>
                    <td className="px-3 py-2">
                      <div className="flex flex-wrap gap-1">
                        {a.coverage && COVERAGE_KEYS.filter((k) => a.coverage![k]).map((k) => (
                          <span key={k} title={T(`cov.${k}`)}
                            className="rounded bg-brand-500/10 border border-brand-500/20 px-1 py-px text-brand-300">{T(`chip.${k}`)}</span>
                        ))}
                        {a.has_extraction && !COVERAGE_KEYS.some((k) => a.coverage?.[k]) && <span className="text-white/25">-</span>}
                      </div>
                    </td>
                    <td className="px-3 py-2 text-white/60 font-mono">{a.has_extraction ? a.n_observations : "-"}</td>
                    <td className="px-3 py-2 whitespace-nowrap">
                      {a.has_extraction && a.n_observations > 0
                        ? (check === 0
                          ? <span className="text-brand-300">{T("quotesOk")}</span>
                          : <span className="font-semibold text-gold-400">{T("quotesCheck").replace("{n}", String(check))}</span>)
                        : <span className="text-white/25">-</span>}
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap">
                      {a.has_extraction && a.n_observations > 0 ? (
                        <span className={a.n_reviewed >= a.n_observations ? "text-brand-300" : "text-white/55"}>
                          {a.n_reviewed}/{a.n_observations}
                          {a.n_conflict > 0 && <span className="ml-1.5 font-semibold text-gold-400">{T("review.conflicts").replace("{n}", String(a.n_conflict))}</span>}
                        </span>
                      ) : <span className="text-white/25">-</span>}
                    </td>
                  </tr>
                  {isOpen && (
                    <tr className="border-b border-white/5 bg-white/2">
                      <td colSpan={7} className="px-4 py-3">
                        <ObservationPanel scenarioId={scenarioId} articleId={a.id} detail={details[a.id]} reviewer={reviewer}
                          onUpdated={(u) => afterReview(a.id, u)} />
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
        {shown.length > visible && (
          <div className="text-center py-3 text-[10px] text-white/45">
            {T("showingFirst").replace("{n}", String(visible)).replace("{total}", String(shown.length))}{" "}
            <button onClick={() => setShownRows({ key: viewKey, n: visible + PAGE })} className="text-brand-300 underline">{T("showMore")}</button>
          </div>
        )}
        {shown.length === 0 && <div className="text-center py-8 text-xs text-white/35">{T("noMatch")}</div>}
      </div>
    </div>
  );
}
