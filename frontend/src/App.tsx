import { useEffect, useMemo, useState } from 'react'
import { ExternalLink } from 'lucide-react'
import type { ProjectContext, RelevanceLabel, SearchFilters, SearchMode, SearchResult } from './types/search'
import { getFilterOptions, searchDocuments } from './lib/api'
import type { FilterOptions } from './lib/api'

export default function App() {
  const [projectContext, setProjectContext] = useState<ProjectContext>('eva')
  const [mode, setMode] = useState<SearchMode>('semantic')
  const [query, setQuery] = useState('respiratory outbreak')
  const [filters, setFilters] = useState<SearchFilters>({ project_context: 'eva' })
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [relevanceMap, setRelevanceMap] = useState<Record<number, RelevanceLabel>>({})
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null)
  const [filterOptions, setFilterOptions] = useState<FilterOptions | null>(null)

  useEffect(() => {
    getFilterOptions().then(setFilterOptions).catch(console.error)
  }, [])

  const effectiveFilters = useMemo(
    () => ({ ...filters, project_context: projectContext }),
    [filters, projectContext]
  )

  const dedupedResults = useMemo(() => {
    const seen = new Set<string>()
    return results.filter((result) => {
      const key = `${result.document_id}::${result.chunk_index}::${result.content}`
      if (seen.has(key)) return false
      seen.add(key)
      return true
    })
  }, [results])

  async function handleSearch() {
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    try {
      const data = await searchDocuments({
        query_text: query,
        mode,
        limit: 10,
        filters: effectiveFilters,
      })
      setResults(data.results)
      setSelectedResult(data.results[0] ?? null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erreur inconnue')
      setSelectedResult(null)
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  const hasResults = dedupedResults.length > 0

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(34,211,238,0.10),_transparent_30%),linear-gradient(180deg,_#020617_0%,_#081226_100%)] text-white">
      <header className="border-b border-white/10 bg-slate-950/70 backdrop-blur-xl">
        <div className="mx-auto max-w-[1380px] px-6 py-8">
          <div className="flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
            <div className="max-w-3xl">
              <p className="text-xs font-semibold uppercase tracking-[0.28em] text-cyan-300">LiteRev++</p>
              <h1 className="mt-3 text-4xl font-semibold tracking-tight text-white">Evidence-to-scenario search</h1>
              <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-300">
                Interface unifiée pour GeoAI4EI, GESICA et EVA, connectée au moteur FastAPI + PostgreSQL/pgvector.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-3">
              {[
                ['geoai4ei', 'GeoAI4EI'],
                ['gesica', 'GESICA'],
                ['eva', 'EVA'],
              ].map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setProjectContext(value as ProjectContext)}
                  className={`rounded-2xl border px-5 py-3 text-left transition ${
                    projectContext === value
                      ? 'border-cyan-400 bg-cyan-500/10 text-white shadow-2xl'
                      : 'border-white/10 bg-white/5 text-slate-300 hover:border-white/20 hover:bg-white/10'
                  }`}
                >
                  <div className="text-sm font-semibold">{label}</div>
                </button>
              ))}
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1380px] px-6 py-8">
        <div className="grid gap-6 xl:grid-cols-[280px_minmax(0,1fr)]">
          <aside className="xl:sticky xl:top-8 xl:self-start">
            <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
              <h2 className="text-lg font-semibold text-white">Filtres</h2>
              <div className="mt-5 space-y-4">
                {[
                  ['source_type', 'Type de source'],
                  ['disease_or_condition', 'Maladie / pathologie'],
                  ['scenario_type', 'Type de scénario'],
                  ['geographic_scope', 'Zone géographique'],
                  ['evidence_category', 'Catégorie de preuve'],
                ].map(([key, label]) => {
                  const options = filterOptions?.[key as keyof FilterOptions] ?? []
                  return (
                    <label key={key} className="block">
                      <span className="mb-2 block text-sm font-medium text-slate-200">{label}</span>
                      <select
                        value={(filters as Record<string, string | undefined>)[key] ?? ''}
                        onChange={(e) =>
                          setFilters((prev) => ({ ...prev, [key]: e.target.value || undefined }))
                        }
                        className="w-full rounded-2xl border border-white/10 bg-slate-950/80 px-3 py-3 text-sm text-white focus:border-cyan-400 focus:outline-none appearance-none"
                      >
                        <option value="">Tous</option>
                        {options.map((opt) => (
                          <option key={String(opt.value)} value={String(opt.value)}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  )
                })}
              </div>
            </div>
          </aside>

          <section className="space-y-6">
            <section className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
              <div className="mb-4 flex items-center gap-2 rounded-2xl border border-white/10 bg-slate-900/80 p-1 text-sm">
                {(['semantic', 'boolean'] as SearchMode[]).map((item) => (
                  <button
                    key={item}
                    type="button"
                    onClick={() => setMode(item)}
                    className={`rounded-xl px-4 py-2 capitalize transition ${
                      mode === item ? 'bg-cyan-500 text-slate-950' : 'text-slate-300 hover:bg-white/10'
                    }`}
                  >
                    {item === 'semantic' ? 'Sémantique' : 'Booléen'}
                  </button>
                ))}
              </div>
              <div className="flex flex-col gap-3 lg:flex-row">
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder={mode === 'semantic' ? 'Ex. respiratory outbreak' : 'Ex. respiratory AND outbreak'}
                  className="min-h-14 flex-1 rounded-2xl border border-white/10 bg-slate-950/80 px-4 text-white outline-none placeholder:text-slate-500 focus:border-cyan-400"
                />
                <button
                  type="button"
                  onClick={handleSearch}
                  disabled={loading}
                  className="min-h-14 rounded-2xl bg-cyan-400 px-6 font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {loading ? 'Recherche…' : 'Rechercher'}
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
              <div className="grid gap-6 2xl:grid-cols-[minmax(0,1fr)_340px]">
                <div className="space-y-4">
                  {dedupedResults.map((result) => (
                    <article
                      key={`${result.document_id}-${result.chunk_index}-${result.content}`}
                      className={`rounded-3xl border bg-white/5 p-5 shadow-2xl transition ${
                        selectedResult?.id === result.id ? 'border-cyan-400/60' : 'border-white/10 hover:border-cyan-400/40'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-4">
                        <button type="button" onClick={() => setSelectedResult(result)} className="text-left">
                          <h3 className="text-2xl font-semibold text-white hover:text-cyan-300">{result.title}</h3>
                        </button>
                        {result.url && (
                          <a href={result.url} target="_blank" rel="noopener noreferrer"
                            className="inline-flex items-center gap-2 rounded-xl border border-white/10 px-3 py-2 text-sm text-slate-200 hover:bg-white/10">
                            Source <ExternalLink size={16} />
                          </a>
                        )}
                      </div>
                      <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-400">
                        <span className="rounded-full bg-white/5 px-2 py-1">Score {result.score.toFixed(3)}</span>
                        {result.source && <span className="rounded-full bg-white/5 px-2 py-1">{result.source}</span>}
                        {result.year && <span className="rounded-full bg-white/5 px-2 py-1">{result.year}</span>}
                        {result.project_context && (
                          <span className="rounded-full bg-cyan-500/10 px-2 py-1 text-cyan-200">{result.project_context}</span>
                        )}
                      </div>
                      <p className="mt-4 max-w-none text-sm leading-6 text-slate-200">
                        {result.highlight || result.content}
                      </p>
                      <div className="mt-5 flex flex-wrap gap-2">
                        {(['pertinent', 'non_pertinent', 'incertain'] as RelevanceLabel[]).map((tag) => (
                          <button key={tag} type="button"
                            onClick={() => setRelevanceMap((prev) => ({ ...prev, [result.id]: tag }))}
                            className={`rounded-full border px-3 py-1 text-xs transition ${
                              relevanceMap[result.id] === tag
                                ? 'border-cyan-400 bg-cyan-500/15 text-cyan-200'
                                : 'border-white/10 bg-white/5 text-slate-400 hover:border-white/20 hover:text-slate-200'
                            }`}>
                            {tag}
                          </button>
                        ))}
                      </div>
                    </article>
                  ))}
                </div>

                <aside className="2xl:sticky 2xl:top-8 2xl:self-start">
                  <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl min-h-[180px]">
                    {!selectedResult ? (
                      <div className="text-sm leading-6 text-slate-300">Cliquez sur un résultat pour afficher le détail.</div>
                    ) : (
                      <div className="space-y-5 text-sm text-slate-200">
                        <div>
                          <p className="text-xs uppercase tracking-[0.2em] text-cyan-300">Document detail</p>
                          <h2 className="mt-2 text-2xl font-semibold text-white">{selectedResult.title}</h2>
                        </div>
                        {selectedResult.url && (
                          <a href={selectedResult.url} target="_blank" rel="noopener noreferrer"
                            className="inline-flex items-center gap-2 rounded-xl border border-white/10 px-3 py-2 text-sm text-slate-200 hover:bg-white/10">
                            Open source <ExternalLink size={16} />
                          </a>
                        )}
                        <section>
                          <h3 className="mb-2 font-medium text-white">Extrait</h3>
                          <p className="rounded-2xl border border-white/10 bg-white/5 p-4 leading-6">
                            {selectedResult.highlight || selectedResult.content}
                          </p>
                        </section>
                        <section>
                          <h3 className="mb-2 font-medium text-white">Métadonnées</h3>
                          <dl className="grid grid-cols-1 gap-3 rounded-2xl border border-white/10 bg-white/5 p-4">
                            <div><dt className="text-slate-400">Source</dt><dd>{selectedResult.source ?? '—'}</dd></div>
                            <div><dt className="text-slate-400">Année</dt><dd>{selectedResult.year ?? '—'}</dd></div>
                            <div><dt className="text-slate-400">Projet</dt><dd>{selectedResult.project_context ?? '—'}</dd></div>
                            <div><dt className="text-slate-400">Type</dt><dd>{selectedResult.source_type ?? '—'}</dd></div>
                            <div><dt className="text-slate-400">Pathologie</dt><dd>{selectedResult.disease_or_condition ?? '—'}</dd></div>
                            <div><dt className="text-slate-400">Scénario</dt><dd>{selectedResult.scenario_type ?? '—'}</dd></div>
                            <div><dt className="text-slate-400">Zone</dt><dd>{selectedResult.geographic_scope ?? '—'}</dd></div>
                            <div><dt className="text-slate-400">Preuve</dt><dd>{selectedResult.evidence_category ?? '—'}</dd></div>
                          </dl>
                        </section>
                      </div>
                    )}
                  </div>
                </aside>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  )
}
