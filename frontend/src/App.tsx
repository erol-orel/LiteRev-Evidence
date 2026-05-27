import { useEffect, useMemo, useState } from "react";
import { Download, ExternalLink, RotateCcw } from "lucide-react";
import { getFilterOptions, searchDocuments } from "./lib/api";
import type { FilterOptions } from "./lib/api";
import type {
  ProjectContext,
  RelevanceLabel,
  SearchFilters,
  SearchMode,
  SearchResult,
} from "./types/search";

type DocumentDetail = {
  id: number;
  source?: string | null;
  title?: string | null;
  abstract?: string | null;
  year?: number | null;
  url?: string | null;
  externalid?: string | null;
  projectcontext?: string | null;
  sourcetype?: string | null;
  diseaseorcondition?: string | null;
  scenariotype?: string | null;
  geographicscope?: string | null;
  evidencecategory?: string | null;
};

const FILTER_FIELDS: Array<[keyof FilterOptions, string]> = [
  ["sourcetype", "Type de source"],
  ["diseaseorcondition", "Maladie / pathologie"],
  ["scenariotype", "Type de scénario"],
  ["geographicscope", "Zone géographique"],
  ["evidencecategory", "Catégorie de preuve"],
];

const PAGE_SIZE = 10;
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function fetchDocumentDetail(documentId: number): Promise<DocumentDetail> {
  const response = await fetch(`${API_BASE_URL}/documents/${documentId}`);
  if (!response.ok) {
    throw new Error(`Impossible de charger le document ${documentId}`);
  }
  return response.json();
}

export default function App() {
  const [projectContext, setProjectContext] = useState<ProjectContext>("eva");
  const [mode, setMode] = useState<SearchMode>("semantic");
  const [query, setQuery] = useState("");
  const [filters, setFilters] = useState<SearchFilters>({ projectcontext: "eva" });
  const [yearRange, setYearRange] = useState<[number, number]>([
    2000,
    new Date().getFullYear(),
  ]);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [relevanceMap, setRelevanceMap] = useState<Record<number, RelevanceLabel>>({});
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null);
  const [selectedDocument, setSelectedDocument] = useState<DocumentDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [filterOptions, setFilterOptions] = useState<FilterOptions | null>(null);
  const [page, setPage] = useState(1);

  useEffect(() => {
    getFilterOptions()
      .then((opts) => {
        setFilterOptions(opts);
        if (opts.year?.length) {
          const years = opts.year
            .map((y) => Number(y.value))
            .filter((y) => Number.isFinite(y));
          if (years.length) {
            setYearRange([Math.min(...years), Math.max(...years)]);
          }
        }
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    setFilters((prev) => ({ ...prev, projectcontext: projectContext }));
  }, [projectContext]);

  const effectiveFilters = useMemo(
    () => ({
      ...filters,
      projectcontext: projectContext,
    }),
    [filters, projectContext],
  );

  const dedupedResults = useMemo(() => {
    const seen = new Set<string>();
    return results.filter((r) => {
      const key = `${r.documentid}-${r.chunkindex}-${r.content}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }, [results]);

  const totalPages = Math.max(1, Math.ceil(dedupedResults.length / PAGE_SIZE));

  const pagedResults = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE;
    return dedupedResults.slice(start, start + PAGE_SIZE);
  }, [dedupedResults, page]);

  const detailView = useMemo(() => {
    if (!selectedResult) return null;

    return {
      id: selectedDocument?.id ?? selectedResult.documentid,
      title: selectedDocument?.title ?? selectedResult.title,
      abstract: selectedDocument?.abstract ?? selectedResult.abstract,
      source: selectedDocument?.source ?? selectedResult.source,
      year: selectedDocument?.year ?? selectedResult.year,
      url: selectedDocument?.url ?? selectedResult.url,
      externalid: selectedDocument?.externalid ?? null,
      projectcontext: selectedDocument?.projectcontext ?? selectedResult.projectcontext,
      sourcetype: selectedDocument?.sourcetype ?? selectedResult.sourcetype,
      diseaseorcondition:
        selectedDocument?.diseaseorcondition ?? selectedResult.diseaseorcondition,
      scenariotype: selectedDocument?.scenariotype ?? selectedResult.scenariotype,
      geographicscope:
        selectedDocument?.geographicscope ?? selectedResult.geographicscope,
      evidencecategory:
        selectedDocument?.evidencecategory ?? selectedResult.evidencecategory,
      excerpt: selectedResult.highlight ?? selectedResult.content,
    };
  }, [selectedDocument, selectedResult]);

  async function loadDocumentDetail(result: SearchResult) {
    setSelectedResult(result);
    setSelectedDocument(null);
    setDetailLoading(true);

    try {
      const detail = await fetchDocumentDetail(result.documentid);
      setSelectedDocument(detail);
    } catch (err) {
      console.error(err);
      setSelectedDocument(null);
    } finally {
      setDetailLoading(false);
    }
  }

  async function handleSearch() {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setPage(1);

    try {
      const data = await searchDocuments({
        querytext: query,
        mode,
        limit: 100,
        filters: {
          ...effectiveFilters,
          yearmin: yearRange[0],
          yearmax: yearRange[1],
        },
      });

      setResults(data.results);

      const first = data.results[0] ?? null;
      setSelectedResult(first);
      setSelectedDocument(null);

      if (first) {
        try {
          const detail = await fetchDocumentDetail(first.documentid);
          setSelectedDocument(detail);
        } catch (err) {
          console.error(err);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur inconnue");
      setSelectedResult(null);
      setSelectedDocument(null);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setFilters({ projectcontext: projectContext });
    setSelectedDocument(null);
    setSelectedResult(null);
    setResults([]);
    setPage(1);

    if (filterOptions?.year?.length) {
      const years = filterOptions.year
        .map((y) => Number(y.value))
        .filter((y) => Number.isFinite(y));
      if (years.length) {
        setYearRange([Math.min(...years), Math.max(...years)]);
      }
    } else {
      setYearRange([2000, new Date().getFullYear()]);
    }
  }

  function handleExport(fmt: "csv" | "json") {
    if (!dedupedResults.length) return;

    if (fmt === "json") {
      const blob = new Blob([JSON.stringify(dedupedResults, null, 2)], {
        type: "application/json",
      });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "literev-results.json";
      a.click();
      URL.revokeObjectURL(a.href);
      return;
    }

    const headers = [
      "title",
      "score",
      "source",
      "year",
      "projectcontext",
      "sourcetype",
      "diseaseorcondition",
      "scenariotype",
      "geographicscope",
      "evidencecategory",
      "url",
    ];

    const rows = dedupedResults.map((r) =>
      headers
        .map((h) => JSON.stringify((r as Record<string, unknown>)[h] ?? ""))
        .join(","),
    );

    const blob = new Blob([[headers.join(","), ...rows].join("\n")], {
      type: "text/csv;charset=utf-8",
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "literev-results.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  const hasResults = dedupedResults.length > 0;

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,rgba(34,211,238,0.10),transparent_30%),linear-gradient(180deg,#020617_0%,#081226_100%)] text-white">
      <header className="border-b border-white/10 bg-slate-950/70 backdrop-blur-xl">
        <div className="mx-auto max-w-[1380px] px-6 py-8">
          <div className="flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
            <div className="max-w-3xl">
              <p className="text-xs font-semibold uppercase tracking-[0.28em] text-cyan-300">
                LiteRev
              </p>
              <h1 className="mt-3 text-4xl font-semibold tracking-tight text-white">
                Evidence-to-scenario search
              </h1>
              <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-300">
                Interface unifiée pour GeoAI4EI, GESICA et EVA, connectée au moteur
                FastAPI PostgreSQL/pgvector.
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              {[
                ["geoai4ei", "GeoAI4EI"],
                ["gesica", "GESICA"],
                ["eva", "EVA"],
              ].map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setProjectContext(value as ProjectContext)}
                  className={
                    projectContext === value
                      ? "rounded-2xl border border-cyan-400 bg-cyan-500/10 px-5 py-3 text-left text-white shadow-2xl transition"
                      : "rounded-2xl border border-white/10 bg-white/5 px-5 py-3 text-left text-slate-300 transition hover:border-white/20 hover:bg-white/10"
                  }
                >
                  <div className="text-sm font-semibold">{label}</div>
                </button>
              ))}
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1380px] px-6 py-8">
        <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
          <aside className="space-y-5 rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl xl:sticky xl:top-6 xl:self-start">
            <div className="flex items-center justify-between">
              <div>
                <div className="mb-2 text-xs font-semibold uppercase tracking-[0.22em] text-cyan-300">
                  Filtres
                </div>
                <p className="text-sm text-slate-300">
                  Affinez la recherche par métadonnées canoniques.
                </p>
              </div>

              <button
                type="button"
                onClick={handleReset}
                title="Réinitialiser les filtres"
                className="flex items-center gap-1 rounded-xl border border-white/10 px-2 py-1 text-xs text-slate-400 transition hover:border-white/20 hover:text-slate-200"
              >
                <RotateCcw size={12} />
                Reset
              </button>
            </div>

            <div className="space-y-4">
              {FILTER_FIELDS.map(([key, label]) => (
                <label key={key} className="block space-y-2">
                  <span className="text-sm text-slate-200">{label}</span>
                  <select
                    value={String((filters as Record<string, unknown>)[key] ?? "")}
                    onChange={(e) =>
                      setFilters((prev) => ({
                        ...prev,
                        [key]: e.target.value || undefined,
                      }))
                    }
                    className="w-full rounded-2xl border border-white/10 bg-slate-950/80 px-4 py-3 text-sm text-white outline-none focus:border-cyan-400"
                  >
                    <option value="">Tous</option>
                    {(filterOptions?.[key] ?? []).map((opt) => (
                      <option key={String(opt.value)} value={String(opt.value)}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </label>
              ))}

              <label className="block space-y-2">
                <span className="flex items-center justify-between text-sm text-slate-200">
                  <span>Année</span>
                  <span className="font-mono text-cyan-300">
                    {yearRange[0]} - {yearRange[1]}
                  </span>
                </span>

                <div className="space-y-2">
                  <input
                    type="range"
                    min={
                      filterOptions?.year?.length
                        ? Math.min(...filterOptions.year.map((y) => Number(y.value)))
                        : 2000
                    }
                    max={yearRange[1]}
                    value={yearRange[0]}
                    onChange={(e) =>
                      setYearRange([Number(e.target.value), yearRange[1]])
                    }
                    className="w-full accent-cyan-400"
                  />
                  <input
                    type="range"
                    min={yearRange[0]}
                    max={
                      filterOptions?.year?.length
                        ? Math.max(...filterOptions.year.map((y) => Number(y.value)))
                        : new Date().getFullYear()
                    }
                    value={yearRange[1]}
                    onChange={(e) =>
                      setYearRange([yearRange[0], Number(e.target.value)])
                    }
                    className="w-full accent-cyan-400"
                  />
                </div>
              </label>
            </div>
          </aside>

          <section className="space-y-6">
            <section className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
              <div className="mb-4 flex items-center gap-2 rounded-2xl border border-white/10 bg-slate-900/80 p-1 text-sm">
                {(["semantic", "boolean"] as SearchMode[]).map((item) => (
                  <button
                    key={item}
                    type="button"
                    onClick={() => setMode(item)}
                    className={
                      mode === item
                        ? "rounded-xl bg-cyan-500 px-4 py-2 text-slate-950 transition"
                        : "rounded-xl px-4 py-2 text-slate-300 transition hover:bg-white/10"
                    }
                  >
                    {item === "semantic" ? "Sémantique" : "Booléen"}
                  </button>
                ))}
              </div>

              <div className="flex flex-col gap-3 lg:flex-row">
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleSearch();
                  }}
                  placeholder={
                    mode === "semantic"
                      ? "Ex. respiratory outbreak"
                      : "Ex. respiratory AND outbreak"
                  }
                  className="min-h-14 flex-1 rounded-2xl border border-white/10 bg-slate-950/80 px-4 text-white outline-none placeholder:text-slate-500 focus:border-cyan-400"
                />
                <button
                  type="button"
                  onClick={handleSearch}
                  disabled={loading}
                  className="min-h-14 rounded-2xl bg-cyan-400 px-6 font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {loading ? "Recherche..." : "Rechercher"}
                </button>
              </div>
            </section>

            {error && (
              <div className="rounded-2xl border border-rose-400/30 bg-rose-500/10 p-4 text-sm text-rose-100">
                {error}
              </div>
            )}

            {!loading && !error && !hasResults && (
              <div className="rounded-3xl border border-dashed border-white/10 bg-white/5 p-10 text-center text-slate-300">
                Lancez une recherche pour afficher les résultats.
              </div>
            )}

            {hasResults && (
              <>
                <div className="flex items-center justify-between">
                  <p className="text-sm text-slate-400">
                    <span className="font-semibold text-white">{dedupedResults.length}</span>{" "}
                    résultat{dedupedResults.length > 1 ? "s" : ""} · page{" "}
                    <span className="text-white">{page}</span> / {totalPages}
                  </p>

                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => handleExport("csv")}
                      className="flex items-center gap-1.5 rounded-xl border border-white/10 px-3 py-1.5 text-xs text-slate-300 transition hover:border-white/20 hover:text-white"
                    >
                      <Download size={12} />
                      CSV
                    </button>
                    <button
                      type="button"
                      onClick={() => handleExport("json")}
                      className="flex items-center gap-1.5 rounded-xl border border-white/10 px-3 py-1.5 text-xs text-slate-300 transition hover:border-white/20 hover:text-white"
                    >
                      <Download size={12} />
                      JSON
                    </button>
                  </div>
                </div>

                <div className="grid gap-6 2xl:grid-cols-[minmax(0,1fr)_340px]">
                  <div className="space-y-4">
                    {pagedResults.map((result) => (
                      <article
                        key={`${result.documentid}-${result.chunkindex}-${result.content}`}
                        className={
                          selectedResult?.id === result.id
                            ? "rounded-3xl border border-cyan-400/60 bg-white/5 p-5 shadow-2xl transition"
                            : "rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl transition hover:border-cyan-400/40"
                        }
                      >
                        <div className="flex items-start justify-between gap-4">
                          <button
                            type="button"
                            onClick={() => loadDocumentDetail(result)}
                            className="text-left"
                          >
                            <h3 className="text-2xl font-semibold text-white hover:text-cyan-300">
                              {result.title}
                            </h3>
                          </button>

                          {result.url && (
                            <a
                              href={result.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex shrink-0 items-center gap-2 rounded-xl border border-white/10 px-3 py-2 text-sm text-slate-200 hover:bg-white/10"
                            >
                              Source
                              <ExternalLink size={16} />
                            </a>
                          )}
                        </div>

                        <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-400">
                          <span className="rounded-full bg-white/5 px-2 py-1">
                            Score {(result.score ?? 0).toFixed(3)}
                          </span>
                          {result.source && (
                            <span className="rounded-full bg-white/5 px-2 py-1">
                              {result.source}
                            </span>
                          )}
                          {result.year && (
                            <span className="rounded-full bg-white/5 px-2 py-1">
                              {result.year}
                            </span>
                          )}
                          {result.projectcontext && (
                            <span className="rounded-full bg-cyan-500/10 px-2 py-1 text-cyan-200">
                              {result.projectcontext}
                            </span>
                          )}
                        </div>

                        <p className="mt-4 max-w-none text-sm leading-6 text-slate-200">
                          {result.highlight ?? result.content}
                        </p>

                        <div className="mt-5 flex flex-wrap gap-2">
                          {(["pertinent", "non-pertinent", "incertain"] as RelevanceLabel[]).map(
                            (tag) => (
                              <button
                                key={tag}
                                type="button"
                                onClick={() =>
                                  setRelevanceMap((prev) => ({
                                    ...prev,
                                    [result.id]: tag,
                                  }))
                                }
                                className={
                                  relevanceMap[result.id] === tag
                                    ? "rounded-full border border-cyan-400 bg-cyan-500/15 px-3 py-1 text-xs text-cyan-200 transition"
                                    : "rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-400 transition hover:border-white/20 hover:text-slate-200"
                                }
                              >
                                {tag}
                              </button>
                            ),
                          )}
                        </div>
                      </article>
                    ))}

                    {totalPages > 1 && (
                      <div className="flex items-center justify-center gap-3 pt-2">
                        <button
                          type="button"
                          disabled={page === 1}
                          onClick={() => setPage((p) => Math.max(1, p - 1))}
                          className="rounded-xl border border-white/10 px-4 py-2 text-sm text-slate-300 transition hover:bg-white/10 disabled:opacity-40"
                        >
                          Précédent
                        </button>

                        <span className="text-sm text-slate-300">
                          Page <span className="text-white">{page}</span> / {totalPages}
                        </span>

                        <button
                          type="button"
                          disabled={page === totalPages}
                          onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                          className="rounded-xl border border-white/10 px-4 py-2 text-sm text-slate-300 transition hover:bg-white/10 disabled:opacity-40"
                        >
                          Suivant
                        </button>
                      </div>
                    )}
                  </div>

                  <aside className="2xl:sticky 2xl:top-8 2xl:self-start">
                    <div className="min-h-[180px] rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
                      {!selectedResult ? (
                        <div className="text-sm leading-6 text-slate-300">
                          Cliquez sur un résultat pour afficher le détail.
                        </div>
                      ) : detailLoading ? (
                        <div className="text-sm leading-6 text-slate-300">
                          Chargement du document...
                        </div>
                      ) : (
                        <div className="space-y-5 text-sm text-slate-200">
                          <div>
                            <p className="text-xs uppercase tracking-[0.2em] text-cyan-300">
                              Document detail
                            </p>
                            <h2 className="mt-2 text-xl font-semibold text-white">
                              {detailView?.title}
                            </h2>
                          </div>

                          {detailView?.url && (
                            <a
                              href={detailView.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center gap-2 rounded-xl border border-white/10 px-3 py-2 text-sm text-slate-200 hover:bg-white/10"
                            >
                              Open source
                              <ExternalLink size={16} />
                            </a>
                          )}

                          <section>
                            <h3 className="mb-2 font-medium text-white">Extrait</h3>
                            <p className="rounded-2xl border border-white/10 bg-white/5 p-4 leading-6">
                              {detailView?.excerpt ?? ""}
                            </p>
                          </section>

                          <section>
                            <h3 className="mb-2 font-medium text-white">Résumé</h3>
                            <p className="rounded-2xl border border-white/10 bg-white/5 p-4 leading-6">
                              {detailView?.abstract ?? "—"}
                            </p>
                          </section>

                          <section>
                            <h3 className="mb-2 font-medium text-white">Métadonnées</h3>
                            <dl className="grid grid-cols-1 gap-3 rounded-2xl border border-white/10 bg-white/5 p-4">
                              <div>
                                <dt className="text-slate-400">ID</dt>
                                <dd>{detailView?.id ?? "—"}</dd>
                              </div>
                              <div>
                                <dt className="text-slate-400">Source</dt>
                                <dd>{detailView?.source ?? "—"}</dd>
                              </div>
                              <div>
                                <dt className="text-slate-400">Année</dt>
                                <dd>{detailView?.year ?? "—"}</dd>
                              </div>
                              <div>
                                <dt className="text-slate-400">External ID</dt>
                                <dd>{detailView?.externalid ?? "—"}</dd>
                              </div>
                              <div>
                                <dt className="text-slate-400">Projet</dt>
                                <dd>{detailView?.projectcontext ?? "—"}</dd>
                              </div>
                              <div>
                                <dt className="text-slate-400">Type</dt>
                                <dd>{detailView?.sourcetype ?? "—"}</dd>
                              </div>
                              <div>
                                <dt className="text-slate-400">Pathologie</dt>
                                <dd>{detailView?.diseaseorcondition ?? "—"}</dd>
                              </div>
                              <div>
                                <dt className="text-slate-400">Scénario</dt>
                                <dd>{detailView?.scenariotype ?? "—"}</dd>
                              </div>
                              <div>
                                <dt className="text-slate-400">Zone</dt>
                                <dd>{detailView?.geographicscope ?? "—"}</dd>
                              </div>
                              <div>
                                <dt className="text-slate-400">Preuve</dt>
                                <dd>{detailView?.evidencecategory ?? "—"}</dd>
                              </div>
                            </dl>
                          </section>
                        </div>
                      )}
                    </div>
                  </aside>
                </div>
              </>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}