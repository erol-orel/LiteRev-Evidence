import { useEffect, useState, useCallback } from "react";
import {
  ArrowLeft, Brain,
  ChevronDown, ChevronUp, Code, Database, ExternalLink, FileText,
  Layers, MessageSquare, RefreshCw, RotateCcw, Search,
  Shield, Terminal, Zap, AlertTriangle,
  Globe,
} from "lucide-react";
import {
  fetchScenarioDetail,
  fetchScenarioCorpus,
  fetchScenarioModelStatus,
  runScenarioModel,
  fetchScenarioClustering,
  askScenarioRag,
  fetchScenarioPrisma,
  type ScenarioDetail,
  type ScenarioCorpus,
  type ModelStatus,
  type ScenarioClustering,
  type ScenarioRagResponse,
  type ScenarioPrisma,
  type CorpusArticle,
} from "../lib/api";

// ─── Helpers ──────────────────────────────────────────────────────────────────

const STATUS_COLORS = {
  green: {
    bg: "bg-emerald-500/10",
    border: "border-emerald-500/30",
    text: "text-emerald-300",
    dot: "bg-emerald-400",
    badge: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
  },
  orange: {
    bg: "bg-amber-500/10",
    border: "border-amber-500/30",
    text: "text-amber-300",
    dot: "bg-amber-400",
    badge: "bg-amber-500/20 text-amber-300 border-amber-500/30",
  },
  red: {
    bg: "bg-rose-500/10",
    border: "border-rose-500/30",
    text: "text-rose-300",
    dot: "bg-rose-400",
    badge: "bg-rose-500/20 text-rose-300 border-rose-500/30",
  },
};

function SectionHeader({ icon, title, subtitle }: { icon: React.ReactNode; title: string; subtitle?: string }) {
  return (
    <div className="flex items-center gap-3 mb-4">
      <div className="rounded-xl border border-white/10 bg-white/5 p-2 shrink-0">{icon}</div>
      <div>
        <h3 className="text-sm font-semibold text-white uppercase tracking-wider">{title}</h3>
        {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
      </div>
    </div>
  );
}

function LoadingSpinner({ text }: { text?: string }) {
  return (
    <div className="flex items-center justify-center py-8 text-slate-400 gap-2">
      <RotateCcw size={16} className="animate-spin" />
      <span className="text-sm">{text ?? "Chargement..."}</span>
    </div>
  );
}

function ErrorBox({ message }: { message: string }) {
  return (
    <div className="rounded-2xl border border-rose-500/20 bg-rose-500/5 px-4 py-3 text-sm text-rose-300">
      <AlertTriangle size={14} className="inline mr-2" />
      {message}
    </div>
  );
}

// ─── Section: Queries ─────────────────────────────────────────────────────────

function QueriesSection({ detail }: { detail: ScenarioDetail }) {
  const [showPrompt, setShowPrompt] = useState(false);
  return (
    <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-5">
      <SectionHeader
        icon={<Search size={14} className="text-cyan-400" />}
        title="Stratégie de Recherche"
        subtitle="Requêtes utilisées pour récupérer les articles du corpus"
      />
      {/* Boolean Queries PubMed */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <Terminal size={12} className="text-violet-400" />
          <span className="text-xs font-semibold text-violet-300 uppercase tracking-wider">
            Requêtes Booléennes PubMed ({detail.boolean_queries.length})
          </span>
        </div>
        <div className="space-y-2">
          {detail.boolean_queries.length > 0 ? detail.boolean_queries.map((q, i) => (
            <div key={i} className="group relative rounded-xl border border-violet-500/10 bg-violet-500/5 px-3 py-2">
              <div className="flex items-start gap-2">
                <span className="text-[10px] font-mono text-violet-500 shrink-0 mt-0.5">Q{i + 1}</span>
                <code className="text-xs text-violet-200 font-mono break-all leading-5">{q}</code>
              </div>
              <a
                href={`https://pubmed.ncbi.nlm.nih.gov/?term=${encodeURIComponent(q)}`}
                target="_blank"
                rel="noopener noreferrer"
                className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition text-violet-400 hover:text-violet-300"
                title="Ouvrir dans PubMed"
              >
                <ExternalLink size={11} />
              </a>
            </div>
          )) : (
            <p className="text-xs text-slate-500 italic">Aucune requête booléenne définie pour ce scénario.</p>
          )}
        </div>
      </div>
      {/* Natural Language Queries */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <MessageSquare size={12} className="text-sky-400" />
          <span className="text-xs font-semibold text-sky-300 uppercase tracking-wider">
            Requêtes Langage Naturel — Recherche Sémantique ({detail.nl_queries.length})
          </span>
        </div>
        <div className="space-y-2">
          {detail.nl_queries.length > 0 ? detail.nl_queries.map((q, i) => (
            <div key={i} className="rounded-xl border border-sky-500/10 bg-sky-500/5 px-3 py-2">
              <div className="flex items-start gap-2">
                <span className="text-[10px] font-mono text-sky-500 shrink-0 mt-0.5">NL{i + 1}</span>
                <span className="text-xs text-sky-200 leading-5">{q}</span>
              </div>
            </div>
          )) : (
            <p className="text-xs text-slate-500 italic">Aucune requête NL définie pour ce scénario.</p>
          )}
        </div>
      </div>
      {/* Evidence Extraction Prompt */}
      {detail.evidence_extraction_prompt && (
        <div>
          <button
            onClick={() => setShowPrompt(!showPrompt)}
            className="flex items-center gap-2 text-xs font-semibold text-amber-300 uppercase tracking-wider hover:text-amber-200 transition"
          >
            <Brain size={12} className="text-amber-400" />
            Prompt d'Extraction d'Évidence (RAG)
            {showPrompt ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </button>
          {showPrompt && (
            <div className="mt-2 rounded-xl border border-amber-500/10 bg-amber-500/5 px-4 py-3">
              <p className="text-xs text-amber-200 leading-6 whitespace-pre-wrap font-mono">
                {detail.evidence_extraction_prompt}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Section: Modèle ──────────────────────────────────────────────────────────

function ModelSection({ scenarioId }: { scenarioId: string }) {
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchScenarioModelStatus(scenarioId);
      setStatus(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [scenarioId]);

  useEffect(() => { load(); }, [load]);

  const handleRerun = async () => {
    setRunning(true);
    try {
      const data = await runScenarioModel(scenarioId);
      setStatus(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  };

  const colors = status ? STATUS_COLORS[status.status_color] : STATUS_COLORS.green;

  return (
    <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-5">
      <div className="flex items-center justify-between">
        <SectionHeader
          icon={<Brain size={14} className="text-violet-400" />}
          title="Modèle Prédictif"
          subtitle="Algorithme, variables et statut en temps réel"
        />
        <button
          onClick={handleRerun}
          disabled={running || loading}
          className="flex items-center gap-1.5 rounded-xl border border-violet-500/20 bg-violet-500/10 px-3 py-1.5 text-xs text-violet-300 hover:bg-violet-500/20 transition disabled:opacity-50"
          title="Re-lancer le modèle"
        >
          <RefreshCw size={11} className={running ? "animate-spin" : ""} />
          {running ? "Calcul..." : "Re-lancer"}
        </button>
      </div>

      {loading && <LoadingSpinner text="Chargement du statut du modèle..." />}
      {error && <ErrorBox message={error} />}

      {status && !loading && (
        <>
          {/* Statut coloré */}
          <div className={`rounded-2xl border ${colors.border} ${colors.bg} px-4 py-3`}>
            <div className="flex items-center gap-3">
              <span className={`h-3 w-3 rounded-full ${colors.dot} animate-pulse shrink-0`} />
              <div className="flex-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className={`text-sm font-semibold ${colors.text}`}>{status.status_label}</span>
                  <span className={`rounded-full border px-2 py-0.5 text-[10px] font-mono ${colors.badge}`}>
                    {status.status_color.toUpperCase()}
                  </span>
                  {status.recent_articles_30d > 0 && (
                    <span className="rounded-full border border-cyan-500/20 bg-cyan-500/10 px-2 py-0.5 text-[10px] text-cyan-300">
                      +{status.recent_articles_30d} articles (30j)
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-400 mt-0.5">
                  Dernière mise à jour : {new Date(status.timestamp).toLocaleString("fr-FR")}
                </p>
              </div>
            </div>
          </div>

          {/* Seuils d'alerte */}
          <div>
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Seuils d'alerte</p>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
              {(["green", "orange", "red"] as const).map((color) => {
                const threshold = status.alert_thresholds[color];
                const c = STATUS_COLORS[color];
                return threshold ? (
                  <div key={color} className={`rounded-xl border ${c.border} ${c.bg} px-3 py-2`}>
                    <div className="flex items-center gap-1.5 mb-1">
                      <span className={`h-2 w-2 rounded-full ${c.dot}`} />
                      <span className={`text-xs font-semibold ${c.text}`}>{threshold.label}</span>
                    </div>
                    <code className={`text-[10px] font-mono ${c.text} opacity-70`}>{threshold.condition}</code>
                  </div>
                ) : null;
              })}
            </div>
          </div>

          {/* Informations modèle */}
          {status.model_info && Object.keys(status.model_info).length > 0 && (
            <div className="space-y-3">
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Détails du modèle</p>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div className="rounded-xl border border-white/5 bg-white/3 px-3 py-2">
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Algorithme</p>
                  <p className="text-xs text-white font-mono">{status.model_info.algorithm}</p>
                </div>
                <div className="rounded-xl border border-white/5 bg-white/3 px-3 py-2">
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Fréquence de mise à jour</p>
                  <p className="text-xs text-white">{status.model_info.update_frequency}</p>
                </div>
                <div className="rounded-xl border border-white/5 bg-white/3 px-3 py-2 sm:col-span-2">
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Sortie du modèle</p>
                  <p className="text-xs text-white">{status.model_info.output}</p>
                </div>
              </div>
              {/* Variables */}
              {status.model_info.variables && status.model_info.variables.length > 0 && (
                <div className="rounded-xl border border-white/5 bg-white/3 px-3 py-2">
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Variables du modèle</p>
                  <div className="flex flex-wrap gap-1.5">
                    {status.model_info.variables.map((v, i) => (
                      <span key={i} className="rounded-lg border border-violet-500/20 bg-violet-500/10 px-2 py-0.5 text-[10px] text-violet-300 font-mono">
                        {v}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Résultat brut du modèle */}
          {status.model_result && (
            <details className="group">
              <summary className="cursor-pointer text-xs text-slate-500 hover:text-slate-300 transition flex items-center gap-1">
                <Code size={11} />
                Résultat brut du modèle (JSON)
              </summary>
              <pre className="mt-2 rounded-xl border border-white/5 bg-black/30 p-3 text-[10px] text-slate-300 font-mono overflow-auto max-h-48">
                {JSON.stringify(status.model_result, null, 2)}
              </pre>
            </details>
          )}
          {status.model_error && (
            <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-xs text-amber-300">
              <AlertTriangle size={11} className="inline mr-1" />
              Erreur modèle : {status.model_error}
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ─── Section: Corpus ──────────────────────────────────────────────────────────

function CorpusSection({ scenarioId, detail }: { scenarioId: string; detail: ScenarioDetail }) {
  const [corpus, setCorpus] = useState<ScenarioCorpus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [fulltextOnly, setFulltextOnly] = useState(false);
  const [expandedArticle, setExpandedArticle] = useState<number | null>(null);
  const PAGE_SIZE = 20;

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchScenarioCorpus(scenarioId, {
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
        fulltextOnly,
      });
      setCorpus(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [scenarioId, page, fulltextOnly]);

  useEffect(() => { load(); }, [load]);

  const stats = detail.corpus_stats;

  return (
    <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-5">
      <SectionHeader
        icon={<Database size={14} className="text-sky-400" />}
        title="Corpus d'Articles"
        subtitle={`${stats.total} articles indexés · ${stats.with_fulltext} avec full-text · ${stats.years_covered} années couvertes`}
      />

      {/* Stats rapides */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          { label: "Articles totaux", value: stats.total, color: "text-cyan-300" },
          { label: "Avec full-text", value: stats.with_fulltext, color: "text-emerald-300" },
          { label: "Journaux", value: stats.journals_count, color: "text-violet-300" },
          {
            label: "Période",
            value: stats.year_min && stats.year_max ? `${stats.year_min}–${stats.year_max}` : "N/A",
            color: "text-amber-300",
          },
        ].map((s) => (
          <div key={s.label} className="rounded-xl border border-white/5 bg-white/3 px-3 py-2 text-center">
            <p className={`text-lg font-bold ${s.color}`}>{s.value}</p>
            <p className="text-[10px] text-slate-500 mt-0.5">{s.label}</p>
          </div>
        ))}
      </div>

      {/* Filtres */}
      <div className="flex items-center gap-3 flex-wrap">
        <button
          onClick={() => { setFulltextOnly(!fulltextOnly); setPage(0); }}
          className={`flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs transition ${
            fulltextOnly
              ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
              : "border-white/10 bg-white/5 text-slate-400 hover:bg-white/10"
          }`}
        >
          <FileText size={11} />
          Full-text uniquement
        </button>
        {corpus && (
          <span className="text-xs text-slate-500">
            {corpus.total} article{corpus.total > 1 ? "s" : ""} · Page {page + 1}/{Math.ceil(corpus.total / PAGE_SIZE)}
          </span>
        )}
      </div>

      {loading && <LoadingSpinner text="Chargement du corpus..." />}
      {error && <ErrorBox message={error} />}

      {corpus && !loading && (
        <>
          {/* Distribution par source */}
          {corpus.source_distribution.length > 0 && (
            <div>
              <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Distribution par source</p>
              <div className="flex flex-wrap gap-2">
                {corpus.source_distribution.slice(0, 8).map((s) => (
                  <span key={s.source} className="rounded-lg border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] text-slate-300">
                    {s.source}: <span className="text-cyan-300 font-mono">{s.count}</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Liste des articles */}
          <div className="space-y-2">
            {corpus.articles.map((article) => (
              <ArticleRow
                key={article.id}
                article={article}
                expanded={expandedArticle === article.id}
                onToggle={() => setExpandedArticle(expandedArticle === article.id ? null : article.id)}
              />
            ))}
          </div>

          {/* Pagination */}
          {corpus.total > PAGE_SIZE && (
            <div className="flex items-center justify-center gap-3 pt-2">
              <button
                onClick={() => setPage(Math.max(0, page - 1))}
                disabled={page === 0}
                className="rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-300 hover:bg-white/10 disabled:opacity-30 transition"
              >
                ← Précédent
              </button>
              <span className="text-xs text-slate-500">
                {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, corpus.total)} / {corpus.total}
              </span>
              <button
                onClick={() => setPage(page + 1)}
                disabled={(page + 1) * PAGE_SIZE >= corpus.total}
                className="rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-300 hover:bg-white/10 disabled:opacity-30 transition"
              >
                Suivant →
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function ArticleRow({
  article,
  expanded,
  onToggle,
}: {
  article: CorpusArticle;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="rounded-2xl border border-white/5 bg-white/3 overflow-hidden">
      <div
        className="flex items-start gap-3 px-3 py-2.5 cursor-pointer hover:bg-white/5 transition"
        onClick={onToggle}
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <p className="text-sm text-white font-medium leading-5 line-clamp-1">{article.title}</p>
            {article.has_fulltext && (
              <span className="shrink-0 rounded-md border border-emerald-500/20 bg-emerald-500/10 px-1.5 py-0.5 text-[10px] text-emerald-300">
                Full-text
              </span>
            )}
            {article.open_access && (
              <span className="shrink-0 rounded-md border border-amber-500/20 bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-300">
                OA
              </span>
            )}
          </div>
          <div className="flex items-center gap-2 mt-0.5 flex-wrap">
            {article.journal && <span className="text-[11px] text-slate-400 italic">{article.journal}</span>}
            {article.year && <span className="text-[11px] text-slate-500">({article.year})</span>}
            {article.source && (
              <span className="rounded border border-white/5 bg-white/5 px-1.5 py-0.5 text-[10px] text-slate-500 font-mono">
                {article.source}
              </span>
            )}
            {article.citation_count !== null && article.citation_count > 0 && (
              <span className="text-[10px] text-slate-500">{article.citation_count} citations</span>
            )}
          </div>
        </div>
        <div className="shrink-0 flex items-center gap-2">
          {article.url && (
            <a
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="text-slate-500 hover:text-cyan-400 transition"
            >
              <ExternalLink size={12} />
            </a>
          )}
          {expanded ? <ChevronUp size={12} className="text-slate-500" /> : <ChevronDown size={12} className="text-slate-500" />}
        </div>
      </div>
      {expanded && (
        <div className="border-t border-white/5 px-3 py-3 space-y-2">
          {article.authors && (
            <p className="text-xs text-slate-400">
              <span className="text-slate-500">Auteurs : </span>{article.authors}
            </p>
          )}
          {article.abstract && (
            <p className="text-xs text-slate-300 leading-5">{article.abstract}</p>
          )}
          <div className="flex flex-wrap gap-2 pt-1">
            {article.doi && (
              <a
                href={`https://doi.org/${article.doi}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[10px] text-violet-400 hover:underline font-mono"
              >
                DOI: {article.doi}
              </a>
            )}
            {article.study_design && (
              <span className="text-[10px] text-slate-500 border border-white/5 rounded px-1.5 py-0.5">
                {article.study_design}
              </span>
            )}
            {article.country && (
              <span className="text-[10px] text-slate-500 border border-white/5 rounded px-1.5 py-0.5">
                <Globe size={9} className="inline mr-0.5" />{article.country}
              </span>
            )}
            {article.keywords && (
              <div className="flex flex-wrap gap-1">
                {article.keywords.split(",").slice(0, 5).map((kw, i) => (
                  <span key={i} className="text-[10px] text-slate-500 bg-slate-800/50 px-1 rounded">
                    #{kw.trim()}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Section: Clustering ──────────────────────────────────────────────────────

function ClusteringSection({ scenarioId }: { scenarioId: string }) {
  const [data, setData] = useState<ScenarioClustering | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nClusters, setNClusters] = useState(5);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchScenarioClustering(scenarioId, nClusters);
      setData(result);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [scenarioId, nClusters]);

  return (
    <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-5">
      <div className="flex items-center justify-between">
        <SectionHeader
          icon={<Layers size={14} className="text-emerald-400" />}
          title="Clustering & Topic Modelling"
          subtitle="K-means + LDA sur le corpus du scénario"
        />
        <div className="flex items-center gap-2">
          <select
            value={nClusters}
            onChange={(e) => setNClusters(Number(e.target.value))}
            className="rounded-xl border border-white/10 bg-white/5 px-2 py-1.5 text-xs text-slate-300 focus:outline-none"
          >
            {[3, 4, 5, 6, 7, 8].map((n) => (
              <option key={n} value={n}>{n} clusters</option>
            ))}
          </select>
          <button
            onClick={load}
            disabled={loading}
            className="flex items-center gap-1.5 rounded-xl border border-emerald-500/20 bg-emerald-500/10 px-3 py-1.5 text-xs text-emerald-300 hover:bg-emerald-500/20 transition disabled:opacity-50"
          >
            <RotateCcw size={11} className={loading ? "animate-spin" : ""} />
            {loading ? "Calcul..." : "Analyser"}
          </button>
        </div>
      </div>

      {!data && !loading && (
        <div className="text-center py-6">
          <Layers size={24} className="text-slate-600 mx-auto mb-2" />
          <p className="text-sm text-slate-500">Cliquez sur "Analyser" pour lancer le clustering sur le corpus.</p>
        </div>
      )}

      {loading && <LoadingSpinner text="Clustering en cours (K-means + LDA)..." />}
      {error && <ErrorBox message={error} />}

      {data && !loading && (
        <>
          {data.message && (
            <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-xs text-amber-300">
              {data.message}
            </div>
          )}

          {/* Topics LDA */}
          {data.topics && data.topics.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                Topics LDA ({data.topics.length} topics sur {data.n_docs} articles)
              </p>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {data.topics.map((topic) => (
                  <div key={topic.topic_id} className="rounded-xl border border-white/5 bg-white/3 px-3 py-2">
                    <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5">
                      Topic {topic.topic_id + 1}
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {topic.top_words.slice(0, 8).map((w, i) => (
                        <span
                          key={i}
                          className="rounded-md border border-emerald-500/15 bg-emerald-500/8 px-1.5 py-0.5 text-[10px] text-emerald-300 font-mono"
                          style={{ opacity: 1 - i * 0.08 }}
                        >
                          {w}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Clusters K-means */}
          {data.clusters && data.clusters.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                Clusters K-means ({data.n_clusters} clusters)
              </p>
              <div className="space-y-3">
                {data.clusters.map((cluster) => (
                  <div key={cluster.cluster_id} className="rounded-xl border border-white/5 bg-white/3 px-3 py-3">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="rounded-full bg-cyan-500/20 border border-cyan-500/20 px-2 py-0.5 text-[10px] text-cyan-300 font-mono">
                          Cluster {cluster.cluster_id + 1}
                        </span>
                        <span className="text-xs text-slate-400">{cluster.n_docs} articles</span>
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-1 mb-2">
                      {cluster.top_words.slice(0, 6).map((w, i) => (
                        <span key={i} className="text-[10px] text-slate-400 bg-slate-800/50 px-1.5 py-0.5 rounded font-mono">
                          {w}
                        </span>
                      ))}
                    </div>
                    <div className="text-xs text-slate-500">
                      <span className="text-slate-400">Article représentatif : </span>
                      <span className="text-slate-300">{cluster.representative_doc.title}</span>
                      {cluster.representative_doc.year && (
                        <span className="text-slate-500"> ({cluster.representative_doc.year})</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ─── Section: RAG ─────────────────────────────────────────────────────────────

function RagSection({ scenarioId, detail }: { scenarioId: string; detail: ScenarioDetail }) {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<ScenarioRagResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAsk = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await askScenarioRag(scenarioId, question);
      setResponse(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const suggestedQuestions = [
    `Quels sont les principaux facteurs de risque identifiés dans la littérature pour ${detail.title.toLowerCase()} ?`,
    `Quels algorithmes de prédiction sont les plus performants pour ce scénario ?`,
    `Quelles sont les recommandations opérationnelles pour le Grand Genève ?`,
  ];

  return (
    <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-5">
      <SectionHeader
        icon={<MessageSquare size={14} className="text-amber-400" />}
        title="Assistant RAG — Evidence Scientifique"
        subtitle={`Interrogez directement le corpus de ${detail.corpus_stats.total} articles de ce scénario`}
      />

      {/* Questions suggérées */}
      <div>
        <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Questions suggérées</p>
        <div className="flex flex-wrap gap-2">
          {suggestedQuestions.map((q, i) => (
            <button
              key={i}
              onClick={() => setQuestion(q)}
              className="rounded-xl border border-amber-500/10 bg-amber-500/5 px-2.5 py-1.5 text-xs text-amber-200 hover:bg-amber-500/10 transition text-left"
            >
              {q.length > 80 ? q.slice(0, 80) + "..." : q}
            </button>
          ))}
        </div>
      </div>

      {/* Zone de saisie */}
      <div className="space-y-2">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleAsk(); }}
          placeholder={`Posez une question sur le corpus "${detail.title}"...`}
          rows={3}
          className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white placeholder-slate-500 focus:border-amber-500/30 focus:outline-none resize-none"
        />
        <div className="flex items-center justify-between">
          <p className="text-[10px] text-slate-500">Ctrl+Entrée pour envoyer · Prompt spécifique au scénario</p>
          <button
            onClick={handleAsk}
            disabled={loading || !question.trim()}
            className="flex items-center gap-2 rounded-xl border border-amber-500/20 bg-amber-500/10 px-4 py-2 text-sm text-amber-300 hover:bg-amber-500/20 transition disabled:opacity-40"
          >
            {loading ? <RotateCcw size={13} className="animate-spin" /> : <Zap size={13} />}
            {loading ? "Analyse en cours..." : "Interroger le corpus"}
          </button>
        </div>
      </div>

      {error && <ErrorBox message={error} />}

      {/* Réponse */}
      {response && (
        <div className="space-y-4">
          <div className="rounded-2xl border border-amber-500/15 bg-amber-500/5 px-4 py-4">
            <div className="flex items-center gap-2 mb-3">
              <Brain size={13} className="text-amber-400" />
              <span className="text-xs font-semibold text-amber-300 uppercase tracking-wider">
                Réponse basée sur {response.sources.length} source{response.sources.length > 1 ? "s" : ""}
                {response.model && <span className="ml-2 text-amber-500 font-mono">({response.model})</span>}
              </span>
            </div>
            <p className="text-sm text-slate-200 leading-6 whitespace-pre-wrap">{response.answer}</p>
          </div>

          {/* Sources */}
          {response.sources.length > 0 && (
            <div>
              <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Sources utilisées</p>
              <div className="space-y-2">
                {response.sources.map((s, i) => (
                  <div key={s.document_id} className="flex items-start gap-2 rounded-xl border border-white/5 bg-white/3 px-3 py-2">
                    <span className="text-[10px] font-mono text-slate-500 shrink-0 mt-0.5">[{i + 1}]</span>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs text-white font-medium line-clamp-1">{s.title}</p>
                      <div className="flex items-center gap-2 mt-0.5 flex-wrap">
                        {s.journal && <span className="text-[10px] text-slate-400 italic">{s.journal}</span>}
                        {s.year && <span className="text-[10px] text-slate-500">({s.year})</span>}
                        {s.authors && <span className="text-[10px] text-slate-500 truncate max-w-[200px]">{s.authors}</span>}
                        <span className="text-[10px] text-cyan-400 font-mono">
                          score: {(s.score * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                    {s.url && (
                      <a href={s.url} target="_blank" rel="noopener noreferrer" className="text-slate-500 hover:text-cyan-400 transition shrink-0">
                        <ExternalLink size={11} />
                      </a>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Section: PRISMA ──────────────────────────────────────────────────────────

function PrismaSection({ scenarioId }: { scenarioId: string }) {
  const [data, setPrisma] = useState<ScenarioPrisma | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchScenarioPrisma(scenarioId)
      .then(setPrisma)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [scenarioId]);

  return (
    <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-5">
      <SectionHeader
        icon={<Shield size={14} className="text-rose-400" />}
        title="Flow PRISMA"
        subtitle="Identification · Screening · Éligibilité · Inclusion"
      />

      {loading && <LoadingSpinner text="Calcul du flow PRISMA..." />}
      {error && <ErrorBox message={error} />}

      {data && !loading && (
        <div className="space-y-4">
          {/* Flow visuel */}
          <div className="flex flex-col items-center gap-2">
            {/* Identification */}
            <PrismaBox
              color="sky"
              title="Identification"
              value={data.identification.total_records_identified}
              subtitle={`dont ${data.identification.duplicates_removed} doublons supprimés`}
            />
            <PrismaArrow value={`−${data.identification.duplicates_removed} doublons`} />
            {/* Screening */}
            <PrismaBox
              color="violet"
              title="Screening"
              value={data.screening.records_screened}
              subtitle={`${data.screening.records_excluded_title_abstract} exclus · ${data.screening.records_pending} en attente`}
            />
            <PrismaArrow value={`−${data.screening.records_excluded_title_abstract} exclus titre/résumé`} />
            {/* Éligibilité */}
            <PrismaBox
              color="amber"
              title="Éligibilité"
              value={data.eligibility.fulltext_retrieved}
              subtitle={`${data.eligibility.fulltext_not_retrieved} sans full-text`}
            />
            <PrismaArrow value={`−${data.eligibility.fulltext_excluded} exclus full-text`} />
            {/* Inclusion */}
            <PrismaBox
              color="emerald"
              title="Inclus"
              value={data.included.total_included}
              subtitle={`${data.included.pending_assessment} en attente d'évaluation`}
            />
          </div>

          {/* Détail par source */}
          <div>
            <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Articles identifiés par source</p>
            <div className="flex flex-wrap gap-2">
              {Object.entries(data.identification.by_source)
                .filter(([, v]) => v > 0)
                .sort(([, a], [, b]) => b - a)
                .map(([source, count]) => (
                  <span key={source} className="rounded-lg border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] text-slate-300">
                    {source}: <span className="text-cyan-300 font-mono">{count}</span>
                  </span>
                ))}
            </div>
          </div>

          {data.included.note && (
            <p className="text-[10px] text-slate-500 italic">{data.included.note}</p>
          )}
        </div>
      )}
    </div>
  );
}

function PrismaBox({
  color,
  title,
  value,
  subtitle,
}: {
  color: "sky" | "violet" | "amber" | "emerald";
  title: string;
  value: number;
  subtitle?: string;
}) {
  const colors = {
    sky: "border-sky-500/30 bg-sky-500/10 text-sky-300",
    violet: "border-violet-500/30 bg-violet-500/10 text-violet-300",
    amber: "border-amber-500/30 bg-amber-500/10 text-amber-300",
    emerald: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  };
  return (
    <div className={`rounded-2xl border ${colors[color]} px-6 py-3 w-full max-w-sm text-center`}>
      <p className="text-[10px] uppercase tracking-wider opacity-70 mb-1">{title}</p>
      <p className="text-2xl font-bold font-mono">{value.toLocaleString()}</p>
      {subtitle && <p className="text-[10px] opacity-60 mt-0.5">{subtitle}</p>}
    </div>
  );
}

function PrismaArrow({ value }: { value: string }) {
  return (
    <div className="flex flex-col items-center gap-0.5">
      <div className="h-4 w-px bg-slate-600" />
      <span className="text-[10px] text-slate-500 italic">{value}</span>
      <div className="h-4 w-px bg-slate-600" />
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

type SectionKey = "queries" | "model" | "corpus" | "clustering" | "rag" | "prisma";

const SECTIONS: Array<{ key: SectionKey; label: string; icon: React.ReactNode }> = [
  { key: "queries", label: "Stratégie de recherche", icon: <Search size={13} /> },
  { key: "model", label: "Modèle prédictif", icon: <Brain size={13} /> },
  { key: "corpus", label: "Corpus", icon: <Database size={13} /> },
  { key: "clustering", label: "Clustering & Topics", icon: <Layers size={13} /> },
  { key: "rag", label: "Assistant RAG", icon: <MessageSquare size={13} /> },
  { key: "prisma", label: "PRISMA", icon: <Shield size={13} /> },
];

interface ScenarioDetailPageProps {
  scenarioId: string;
  onBack: () => void;
}

export function ScenarioDetailPage({ scenarioId, onBack }: ScenarioDetailPageProps) {
  const [detail, setDetail] = useState<ScenarioDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<SectionKey>("queries");

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchScenarioDetail(scenarioId)
      .then(setDetail)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [scenarioId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24 text-slate-400 gap-2">
        <RotateCcw size={18} className="animate-spin" />
        <span>Chargement du scénario...</span>
      </div>
    );
  }

  if (error || !detail) {
    return (
      <div className="space-y-4">
        <button onClick={onBack} className="flex items-center gap-2 text-sm text-slate-400 hover:text-white transition">
          <ArrowLeft size={14} /> Retour aux scénarios
        </button>
        <ErrorBox message={error ?? "Scénario introuvable"} />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* En-tête */}
      <div className="flex items-start gap-4">
        <button
          onClick={onBack}
          className="mt-1 flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-400 hover:text-white hover:bg-white/10 transition shrink-0"
        >
          <ArrowLeft size={12} /> Retour
        </button>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h2 className="text-xl font-semibold text-white">{detail.title}</h2>
            <span className="rounded-full border border-cyan-500/20 bg-cyan-500/10 px-2 py-0.5 text-xs text-cyan-300">
              {detail.cluster}
            </span>
            <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-xs text-slate-400 font-mono">
              {detail.corpus_stats.total} articles
            </span>
          </div>
          <p className="mt-1 text-sm text-slate-400 leading-5">{detail.description}</p>
          {/* Actions recommandées */}
          {detail.recommended_actions.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {detail.recommended_actions.slice(0, 2).map((action, i) => (
                <span key={i} className="flex items-center gap-1.5 rounded-xl border border-white/5 bg-white/3 px-2.5 py-1 text-xs text-slate-300">
                  <span className="h-1.5 w-1.5 rounded-full bg-cyan-400 shrink-0" />
                  {action.length > 80 ? action.slice(0, 80) + "..." : action}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Navigation par onglets */}
      <div className="flex flex-wrap gap-1.5 border-b border-white/5 pb-4">
        {SECTIONS.map((section) => (
          <button
            key={section.key}
            onClick={() => setActiveSection(section.key)}
            className={`flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-medium transition ${
              activeSection === section.key
                ? "border border-cyan-500/30 bg-cyan-500/10 text-cyan-300"
                : "border border-transparent text-slate-400 hover:text-white hover:bg-white/5"
            }`}
          >
            {section.icon}
            {section.label}
          </button>
        ))}
      </div>

      {/* Contenu de la section active */}
      {activeSection === "queries" && <QueriesSection detail={detail} />}
      {activeSection === "model" && <ModelSection scenarioId={scenarioId} />}
      {activeSection === "corpus" && <CorpusSection scenarioId={scenarioId} detail={detail} />}
      {activeSection === "clustering" && <ClusteringSection scenarioId={scenarioId} />}
      {activeSection === "rag" && <RagSection scenarioId={scenarioId} detail={detail} />}
      {activeSection === "prisma" && <PrismaSection scenarioId={scenarioId} />}
    </div>
  );
}
