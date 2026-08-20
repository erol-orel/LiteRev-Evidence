import React, { useState, useEffect, useLayoutEffect, useCallback, useRef, useMemo } from "react";
import { ErrorBoundary } from "./ErrorBoundary";
import { useI18n } from "../i18n/LanguageProvider";
import {
  ArrowLeft, Brain,
  ChevronDown, ChevronUp, Database, ExternalLink, FileText,
  Layers, MessageSquare, RefreshCw, RotateCcw, Search,
  Shield, Terminal, Zap, AlertTriangle,
  Globe, Upload, CheckCircle2, AlertCircle, Info,
  Microscope, Loader2, Download, Table2, BookOpen,
  Network, Bell, Users, Rss, Sparkles, ClipboardList,
  TrendingUp, X, Plus, Sliders, Activity, Target
} from "lucide-react";
import {
  fetchScenarioDetail,
  fetchScenarioCorpus,
  fetchScenarioClustering,
  askScenarioRagStreamFiltered,
  type RagMeta,
  fetchScenarioPrisma,
  uploadModelData,
  screenArticle,
  fetchArticlePico,
  fetchScenarioPicoBulk,
  fetchEvidenceBrief,
  getLlmEvidenceBrief,
  generateEvidenceBrief,
  getBriefGenerationStatus,
  getScenarioVariables,
  generateScenarioVariables,
  getVariablesGenerationStatus,
  validateScenarioVariables,
  getModelRun,
  getModelDataset,
  listDataConnectors,
  autoFetchModelData,
  type DataConnector,
  type AutoFetchResponse,
  getScenarioModelSpec,
  trainModel,
  compareModels,
  generateSyntheticData,
  getModelTrainStatus,
  getModelMonitor,
  exportModelBundle,
  exportModelXlsx,
  fetchSeirProjection,
  postSeirProjection,
  postSeirObserved,
  deleteSeirObserved,
  calibrateSeir,
  type SeirProjection,
  type SeirBand,
  type SeirParamValue,
  type SeirOverride,
  type SeirCalibration,
  proposeSpec,
  getSpecProposal,
  validateSpecProposal,
  getScenarioSettings,
  patchScenarioSettings,
  triggerRerank,
  getRerankStatus,
  fetchKnowledgeGraph,
  fetchKappaStats,
  fetchDoubleBlindConflicts,
  submitDoubleBlindDecision,
  subscribeAlerts,
  triggerLivingReview,
  extractPicoBatchGlobal,
  extractMetadataBatch,
  fetchFulltextBatch,
  fetchEnrichmentStatus,
  fetchUserScenarioEmbeddingStatus,
  searchLive,
  getSearchStrategy,
  type LiveSearchResult,
  type LiveSearchResponse,
  type SearchStrategy,
  type EnrichmentStatus,
  type EmbeddingStatus,
  type ScenarioDetail,
  type ScenarioCorpus,
  type ScenarioClustering,
  type ScenarioRagResponse,
  type ScenarioPrisma,
  type CorpusArticle,
  type ClusterResult,
  type ClusterPoint,
  type PicoData,
  type PicoBulkResponse,
  type EvidenceBriefData,
  type LlmEvidenceBrief,
  type ScenarioVariables,
  type KnowledgeGraphData,
  type KGNode,
  type KappaStats,
  editModelSpec,
  type EditSpecPayload,
  type ModelRun,
  type ModelDataset,
  type ModelSpecResponse,
  type ProvArticle,
  type ModelMonitor,
  type SpecProposal,
  scenarioBase,
  isUserScenario,
  safeFetch,
} from "../lib/api";

// ─── Helpers ──────────────────────────────────────────────────────────────────

const STATUS_COLORS = {
  green: {
    bg: "bg-brand-500/10",
    border: "border-brand-500/30",
    text: "text-brand-300",
    dot: "bg-brand-400",
    badge: "bg-brand-500/20 text-brand-300 border-brand-500/30",
  },
  orange: {
    bg: "bg-gold-500/10",
    border: "border-gold-500/30",
    text: "text-gold-300",
    dot: "bg-gold-400",
    badge: "bg-gold-500/20 text-gold-300 border-gold-500/30",
  },
  red: {
    bg: "bg-rose-500/10",
    border: "border-rose-500/30",
    text: "text-rose-300",
    dot: "bg-rose-400",
    badge: "bg-rose-500/20 text-rose-300 border-rose-500/30",
  },
  unavailable: {
    bg: "bg-white/3",
    border: "border-white/15",
    text: "text-white/50",
    dot: "bg-white/40",
    badge: "bg-white/10 text-white/50 border-white/15",
  },
};

function SectionHeader({ icon, title, subtitle }: { icon: React.ReactNode; title: string; subtitle?: string }) {
  return (
    <div className="flex items-center gap-3 mb-4">
      <div className="rounded-xl border border-white/10 bg-white/5 p-2 shrink-0">{icon}</div>
      <div>
        <h3 className="text-sm font-semibold text-white uppercase tracking-wider">{title}</h3>
        {subtitle && <p className="text-xs text-white/50 mt-0.5">{subtitle}</p>}
      </div>
    </div>
  );
}

function LoadingSpinner({ text }: { text?: string }) {
  const { t } = useI18n();
  return (
    <div className="flex items-center justify-center py-8 text-white/50 gap-2">
      <RotateCcw size={16} className="animate-spin" />
      <span className="text-sm">{text ?? t("scenarioDetail.common.loading")}</span>
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

function QueriesSection({ detail, scenarioId }: { detail: ScenarioDetail; scenarioId: string }) {
  const { t } = useI18n();
  const [showPrompt, setShowPrompt] = useState(false);
  const [strategy, setStrategy] = useState<SearchStrategy | null>(null);
  const [strategyLoading, setStrategyLoading] = useState(false);
  const [strategyError, setStrategyError] = useState<string | null>(null);
  const [liveData, setLiveData] = useState<LiveSearchResponse | null>(null);
  const [liveLoading, setLiveLoading] = useState(false);
  const [liveError, setLiveError] = useState<string | null>(null);
  const [liveSort, setLiveSort] = useState<"hybrid" | "semantic" | "lexical" | "year_desc">("hybrid");

  const sortedLiveResults = useMemo(() => {
    if (!liveData) return [];
    const arr = [...liveData.results];
    arr.sort((a, b) => {
      if (liveSort === "semantic") return (b.semantic_score ?? 0) - (a.semantic_score ?? 0);
      if (liveSort === "lexical") return (b.lexical_score ?? 0) - (a.lexical_score ?? 0);
      if (liveSort === "year_desc") return (b.year ?? 0) - (a.year ?? 0);
      return (b.hybrid_score ?? 0) - (a.hybrid_score ?? 0);
    });
    return arr;
  }, [liveData, liveSort]);

  function loadStrategy() {
    if (!isUserScenario(scenarioId)) return;
    setStrategyLoading(true);
    setStrategyError(null);
    getSearchStrategy(scenarioId)
      .then(setStrategy)
      .catch(e => setStrategyError(e.message))
      .finally(() => setStrategyLoading(false));
  }

  function runLiveSearch() {
    if (!isUserScenario(scenarioId)) return;
    setLiveLoading(true);
    setLiveError(null);
    searchLive(scenarioId)
      .then(setLiveData)
      .catch(e => setLiveError(e.message))
      .finally(() => setLiveLoading(false));
  }

  return (
    <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-5">
      <SectionHeader
        icon={<Search size={14} className="text-brand-400" />}
        title={t("scenarioDetail.queries.title")}
        subtitle={t("scenarioDetail.queries.subtitle")}
      />
      {/* Boolean Queries multi-sources */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <Terminal size={12} className="text-brand-400" />
          <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">
            {t("scenarioDetail.queries.booleanQueries")} ({detail.boolean_queries.length})
          </span>
        </div>
        <div className="space-y-2">
          {detail.boolean_queries.length > 0 ? detail.boolean_queries.map((q, i) => (
            <div key={i} className="group relative rounded-xl border border-brand-500/10 bg-brand-500/5 px-3 py-2">
              <div className="flex items-start gap-2">
                <span className="text-[10px] font-mono text-brand-500 shrink-0 mt-0.5">Q{i + 1}</span>
                <code className="text-xs text-brand-200 font-mono break-all leading-5">{q}</code>
              </div>
              <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition flex gap-1">
                <a
                  href={`https://pubmed.ncbi.nlm.nih.gov/?term=${encodeURIComponent(q)}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-brand-400 hover:text-brand-300 text-[9px] font-mono"
                  title={t("scenarioDetail.queries.openInPubmed")}
                >PubMed</a>
                <span className="text-white/20">|</span>
                <a
                  href={`https://scholar.google.com/scholar?q=${encodeURIComponent(q)}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-brand-400 hover:text-brand-300 text-[9px] font-mono"
                  title={t("scenarioDetail.queries.openInScholar")}
                >Scholar</a>
                <span className="text-white/20">|</span>
                <a
                  href={`https://europepmc.org/search?query=${encodeURIComponent(q)}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-brand-400 hover:text-brand-300 text-[9px] font-mono"
                  title={t("scenarioDetail.queries.openInEuropepmc")}
                >EuropePMC</a>
              </div>
            </div>
          )) : (
            <p className="text-xs text-white/35 italic">{t("scenarioDetail.queries.noBooleanQuery")}</p>
          )}
        </div>
      </div>
      {/* Natural Language Queries */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <MessageSquare size={12} className="text-brand-400" />
          <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">
            {t("scenarioDetail.queries.nlQueries")} ({detail.nl_queries.length})
          </span>
        </div>
        <div className="space-y-2">
          {detail.nl_queries.length > 0 ? detail.nl_queries.map((q, i) => (
            <div key={i} className="rounded-xl border border-brand-500/10 bg-brand-500/5 px-3 py-2">
              <div className="flex items-start gap-2">
                <span className="text-[10px] font-mono text-brand-500 shrink-0 mt-0.5">NL{i + 1}</span>
                <span className="text-xs text-brand-200 leading-5">{q}</span>
              </div>
            </div>
          )) : (
            <p className="text-xs text-white/35 italic">{t("scenarioDetail.queries.noNlQuery")}</p>
          )}
        </div>
      </div>
      {/* Evidence Extraction Prompt */}
      {detail.evidence_extraction_prompt && (
        <div>
          <button
            onClick={() => setShowPrompt(!showPrompt)}
            className="flex items-center gap-2 text-xs font-semibold text-gold-300 uppercase tracking-wider hover:text-gold-200 transition"
          >
            {showPrompt ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            {showPrompt ? t("scenarioDetail.queries.hidePrompt") : t("scenarioDetail.queries.showPrompt")} {t("scenarioDetail.queries.extractionPromptSuffix")}
          </button>
          {showPrompt && (
            <div className="mt-3 rounded-xl border border-gold-500/15 bg-gold-500/5 p-4">
              <pre className="text-xs text-gold-100 font-mono whitespace-pre-wrap leading-5">
                {detail.evidence_extraction_prompt}
              </pre>
            </div>
          )}
        </div>
      )}

      {/* AI Search Strategy (user scenarios only) */}
      {isUserScenario(scenarioId) && (
        <div>
          <div className="flex items-center gap-2 mb-3">
            <Brain size={12} className="text-violet-400" />
            <span className="text-xs font-semibold text-violet-300 uppercase tracking-wider">
              {t("scenarioDetail.queries.booleanStrategy")}
            </span>
            <button
              onClick={loadStrategy}
              disabled={strategyLoading}
              className="ml-auto rounded-xl border border-violet-500/30 bg-violet-500/10 px-2.5 py-1 text-[10px] text-violet-300 hover:bg-violet-500/20 transition disabled:opacity-50"
            >
              {strategyLoading ? <Loader2 size={10} className="animate-spin inline" /> : t("scenarioDetail.queries.generateRefresh")}
            </button>
          </div>
          {strategyError && <p className="text-xs text-rose-400">{strategyError}</p>}
          {strategy && (
            <div className="space-y-3">
              <div className="rounded-xl border border-violet-500/15 bg-violet-500/5 p-3">
                <p className="text-[10px] font-semibold text-violet-400 uppercase tracking-wider mb-1">{t("scenarioDetail.queries.generalQuery")}</p>
                <code className="text-xs text-violet-200 font-mono break-all leading-5">{strategy.general}</code>
              </div>
              <div className="rounded-xl border border-blue-500/15 bg-blue-500/5 p-3">
                <div className="flex items-center justify-between mb-1">
                  <p className="text-[10px] font-semibold text-blue-400 uppercase tracking-wider">{t("scenarioDetail.queries.pubmedQueryMesh")}</p>
                  <a
                    href={`https://pubmed.ncbi.nlm.nih.gov/?term=${encodeURIComponent(strategy.pubmed)}`}
                    target="_blank" rel="noopener noreferrer"
                    className="text-[9px] text-blue-400 hover:text-blue-300"
                  >{t("scenarioDetail.queries.openInPubmedArrow")}</a>
                </div>
                <code className="text-xs text-blue-200 font-mono break-all leading-5">{strategy.pubmed}</code>
              </div>
              {strategy.explanation && (
                <div className="rounded-xl border border-white/10 bg-white/3 p-3">
                  <p className="text-[10px] font-semibold text-white/50 uppercase tracking-wider mb-1">{t("scenarioDetail.queries.explanation")}</p>
                  <p className="text-xs text-white/70 leading-5">{strategy.explanation}</p>
                </div>
              )}
              {strategy.synonyms && strategy.synonyms.length > 0 && (
                <div>
                  <p className="text-[10px] font-semibold text-white/50 uppercase tracking-wider mb-2">{t("scenarioDetail.queries.synonymGroups")}</p>
                  <div className="flex flex-wrap gap-2">
                    {strategy.synonyms.map((group, i) => (
                      <div key={i} className="rounded-xl border border-white/10 bg-white/3 px-3 py-1.5 flex gap-1.5 flex-wrap">
                        {group.map((term, j) => (
                          <span key={j} className={`text-[10px] rounded-full px-1.5 py-0.5 ${j === 0 ? 'bg-brand-500/20 text-brand-300' : 'bg-white/5 text-white/60'}`}>
                            {term}
                          </span>
                        ))}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Live Federated Search (user scenarios only) */}
      {isUserScenario(scenarioId) && (
        <div>
          <div className="flex items-center gap-2 mb-3">
            <Globe size={12} className="text-brand-400" />
            <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">
              {t("scenarioDetail.queries.liveSearchTitle")}
            </span>
            <button
              onClick={runLiveSearch}
              disabled={liveLoading}
              className="ml-auto rounded-xl border border-brand-500/30 bg-brand-500/10 px-2.5 py-1 text-[10px] text-brand-300 hover:bg-brand-500/20 transition disabled:opacity-50 flex items-center gap-1"
            >
              {liveLoading ? <><Loader2 size={10} className="animate-spin" /> {t("scenarioDetail.queries.searching")}</> : t("scenarioDetail.queries.searchAllSources")}
            </button>
          </div>
          {liveError && <p className="text-xs text-rose-400">{liveError}</p>}
          {liveData && (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-3 text-xs text-forest-400">
                <span className="text-white font-semibold">{liveData.total} {t("scenarioDetail.queries.liveResultsSuffix")}</span>
                <span>·</span>
                {liveData.new_count > 0 && (
                  <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-amber-300">
                    {liveData.new_count} {t("scenarioDetail.queries.newUnindexed")}
                    {liveData.ingesting_background && t("scenarioDetail.queries.ingestingInProgress")}
                  </span>
                )}
                <span>{liveData.sources_queried.join(", ")}</span>
              </div>
              {typeof liveData.corpus_total === "number" && (
                <div className="rounded-xl border border-brand-500/20 bg-brand-500/5 px-3 py-2 text-[11px] text-brand-200"
                     title={t("scenarioDetail.queries.corpusTooltip")}>
                  {t("scenarioDetail.queries.corpusPrefix")} <span className="font-semibold text-white">{liveData.corpus_total.toLocaleString()}</span> {t("scenarioDetail.queries.corpusDocuments")}
                  {" · "}<span className="text-brand-300">{(liveData.corpus_above_threshold ?? 0).toLocaleString()} {t("scenarioDetail.queries.corpusAboveThresholdPrefix")} {liveData.threshold ?? 0.45}</span>
                  {" "}<span className="text-white/40">{t("scenarioDetail.queries.corpusLexicalNote")}</span>
                </div>
              )}
              <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
                <span className="text-forest-500">{t("scenarioDetail.queries.sortLabel")}</span>
                {([
                  ["hybrid", t("scenarioDetail.queries.sortRelevance")],
                  ["semantic", t("scenarioDetail.queries.sortSemantic")],
                  ["lexical", t("scenarioDetail.queries.sortLexical")],
                  ["year_desc", t("scenarioDetail.queries.sortYear")],
                ] as [typeof liveSort, string][]).map(([val, label]) => (
                  <button
                    key={val}
                    type="button"
                    onClick={() => setLiveSort(val)}
                    className={`rounded-full border px-2 py-0.5 transition ${
                      liveSort === val
                        ? "border-brand-400/60 bg-brand-500/20 text-brand-300"
                        : "border-white/10 bg-white/5 text-forest-400 hover:border-white/20 hover:text-white"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {sortedLiveResults.map((r: LiveSearchResult, i: number) => (
                  <div key={i} className={`rounded-xl border p-3 ${r.in_local_db ? 'border-white/10 bg-white/3' : 'border-amber-500/30 bg-amber-500/5'}`}>
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-medium text-white leading-5 truncate">{r.title || t("scenarioDetail.queries.untitled")}</p>
                        <div className="flex flex-wrap gap-1.5 mt-1 items-center">
                          {r.hybrid_score != null && (
                            <span className="text-[10px] rounded-full bg-violet-500/20 px-1.5 py-0.5 text-violet-300" title={t("scenarioDetail.queries.relevanceScoreTooltip")}>
                              {r.hybrid_score.toFixed(2)}
                            </span>
                          )}
                          {r.semantic_score != null && (
                            <span className="text-[10px] rounded-full bg-blue-500/10 px-1.5 py-0.5 text-blue-300 border border-blue-500/20" title={t("scenarioDetail.queries.semanticComponentTooltip")}>
                              {t("scenarioDetail.queries.semanticShort")} {r.semantic_score.toFixed(2)}
                            </span>
                          )}
                          {r.lexical_score != null && (
                            <span className="text-[10px] rounded-full bg-amber-500/10 px-1.5 py-0.5 text-amber-300 border border-amber-500/20" title={t("scenarioDetail.queries.lexicalComponentTooltip")}>
                              {t("scenarioDetail.queries.lexicalShort")} {r.lexical_score.toFixed(2)}
                            </span>
                          )}
                          <span className="text-[10px] text-forest-400">{r.source_name}</span>
                          {r.also_in_sources && r.also_in_sources.length > 0 && (
                            <span className="text-[10px] text-forest-500">+{r.also_in_sources.join(", ")}</span>
                          )}
                          {r.year && <span className="text-[10px] text-forest-400">{r.year}</span>}
                          {r.journal && <span className="text-[10px] text-forest-500 truncate max-w-[120px]">{r.journal}</span>}
                          {r.in_local_db
                            ? <span className="text-[10px] rounded-full border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 text-emerald-300">{t("scenarioDetail.queries.alreadyIndexed")}</span>
                            : <span className="text-[10px] rounded-full border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 text-amber-300">{t("scenarioDetail.queries.notIndexed")}</span>}
                        </div>
                      </div>
                      {r.url && (
                        <a href={r.url} target="_blank" rel="noopener noreferrer" className="shrink-0 text-[10px] text-brand-400 hover:text-brand-300">
                          <ExternalLink size={10} />
                        </a>
                      )}
                    </div>
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

// ─── Section: Variables & Databases (NOUVEAU) ──────────────────────────────────

// Lien vers l'article source (le plus récent / le plus cité) d'un élément du spec.
function ArticleSourceLink({ a, label }: { a?: ProvArticle | null; label?: string }) {
  const { t } = useI18n();
  if (!a) return null;
  const resolvedLabel = label ?? t("scenarioDetail.common.source");
  const text = `${(a.title ?? 'article').slice(0, 70)}${a.year ? ` (${a.year})` : ''}`;
  const inner = (
    <span className="inline-flex items-center gap-1 text-[10px] text-brand-300/80 hover:text-brand-300">
      <FileText size={9} /> {resolvedLabel} : {text}
      {typeof a.citation_count === 'number' && a.citation_count > 0 && (
        <span className="text-white/30">· {a.citation_count} {t("scenarioDetail.articleSourceLink.citations")}</span>
      )}
    </span>
  );
  return a.url
    ? <a href={a.url} target="_blank" rel="noreferrer" title={a.title} className="block mt-0.5">{inner}</a>
    : <span title={a.title} className="block mt-0.5">{inner}</span>;
}

// Familles sélectionnables (alignées sur model_trainer + _ALGO_FAMILIES côté API).
// prophet/sarimax = prévision de série temporelle (nécessite une colonne date).
const ALGO_FAMILY_OPTIONS = [
  "lightgbm", "xgboost", "gradient_boosting", "random_forest", "extremal_rf",
  "logistic_regression", "linear_regression", "elasticnet", "svm", "mlp", "knn",
  "prophet", "sarimax",
];
const TASK_TYPE_OPTIONS = ["classification", "regression", "count", "survival"];
const DTYPE_OPTIONS = ["float", "int", "bool", "category", "datetime"];

// Éditeur de spec : ajuster l'algorithme (parmi les candidats suggérés par
// l'évidence ou toute famille valide), le type de tâche, et ajouter/supprimer des
// variables — puis ré-entraîner. Le backend reconstruit le data_template et
// incrémente la version.
function SpecEditor({ scenarioId, spec, onApplied }: { scenarioId: string; spec: ModelSpecResponse; onApplied: () => void }) {
  const { t } = useI18n();
  const curFamily = spec.algorithm?.family ?? "gradient_boosting";
  const curTask = spec.outcome?.task_type ?? "classification";
  const [open, setOpen] = useState(false);
  const [family, setFamily] = useState(curFamily);
  const [taskType, setTaskType] = useState(curTask);
  const [removed, setRemoved] = useState<Set<string>>(new Set());
  const [added, setAdded] = useState<{ name: string; dtype: string; importance: string }[]>([]);
  const [addName, setAddName] = useState("");
  const [addDtype, setAddDtype] = useState("float");
  const [busy, setBusy] = useState(false);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);

  // Candidats issus de l'évidence en tête de liste, puis les autres familles.
  const candidates = (spec.algorithm?.candidates ?? []).filter(c => ALGO_FAMILY_OPTIONS.includes(c));
  const familyOptions = Array.from(new Set([...candidates, ...ALGO_FAMILY_OPTIONS]));
  const feats = spec.features ?? [];
  const remainingCount = feats.filter(f => f.machine_name && !removed.has(f.machine_name)).length + added.length;
  const dirty = family !== curFamily || taskType !== curTask || removed.size > 0 || added.length > 0;

  const toggleRemove = (mn?: string) => {
    if (!mn) return;
    setRemoved(prev => { const n = new Set(prev); if (n.has(mn)) n.delete(mn); else n.add(mn); return n; });
  };
  const addFeature = () => {
    const name = addName.trim();
    if (!name) return;
    setAdded(prev => [...prev, { name, dtype: addDtype, importance: "medium" }]);
    setAddName("");
  };

  const apply = async () => {
    setBusy(true); setErr(null); setWarnings([]);
    try {
      const payload: EditSpecPayload = { retrain: true };
      if (family !== curFamily) payload.algorithm_family = family;
      if (taskType !== curTask) payload.task_type = taskType;
      if (removed.size > 0) payload.remove_features = Array.from(removed);
      if (added.length > 0) payload.add_features = added;
      const res = await editModelSpec(scenarioId, payload);
      setWarnings(res.warnings ?? []);
      setRemoved(new Set()); setAdded([]);
      onApplied();
    } catch (e: any) { setErr(e.message); }
    finally { setBusy(false); }
  };

  return (
    <div className="rounded-2xl border border-white/8 bg-white/3 p-4 space-y-3">
      <button onClick={() => setOpen(o => !o)} className="w-full flex items-center justify-between">
        <span className="text-[10px] font-bold uppercase tracking-wider text-white/35 flex items-center gap-1.5">
          <Sliders size={12} /> {t("scenarioDetail.specEditor.title")}
        </span>
        {open ? <ChevronUp size={14} className="text-white/40" /> : <ChevronDown size={14} className="text-white/40" />}
      </button>
      {open && (
        <div className="space-y-3">
          <p className="text-[11px] text-white/45">{t("scenarioDetail.specEditor.subtitle")}</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <label className="block">
              <span className="text-[10px] text-white/40 uppercase tracking-wider">{t("scenarioDetail.specEditor.algorithm")}</span>
              <select value={family} onChange={e => setFamily(e.target.value)}
                className="mt-1 w-full rounded-lg bg-forest-950/60 border border-white/10 px-2 py-1.5 text-xs text-white">
                {familyOptions.map(f => (
                  <option key={f} value={f}>{f}{candidates.includes(f) ? ` · ${t("scenarioDetail.specEditor.suggested")}` : ""}</option>
                ))}
              </select>
            </label>
            <label className="block">
              <span className="text-[10px] text-white/40 uppercase tracking-wider">{t("scenarioDetail.specEditor.taskType")}</span>
              <select value={taskType} onChange={e => setTaskType(e.target.value)}
                className="mt-1 w-full rounded-lg bg-forest-950/60 border border-white/10 px-2 py-1.5 text-xs text-white">
                {TASK_TYPE_OPTIONS.map(tt => <option key={tt} value={tt}>{t(`scenarioDetail.specEditor.task_${tt}`)}</option>)}
              </select>
            </label>
          </div>
          <div>
            <span className="text-[10px] text-white/40 uppercase tracking-wider">{t("scenarioDetail.specEditor.variables")} ({remainingCount})</span>
            <div className="mt-1.5 space-y-1">
              {feats.map(f => {
                const gone = f.machine_name ? removed.has(f.machine_name) : false;
                return (
                  <div key={f.machine_name} className={`flex items-center gap-2 text-[11px] rounded-lg px-2 py-1 border ${gone ? "border-rose-500/20 bg-rose-500/5" : "border-white/8 bg-white/2"}`}>
                    <span className={`flex-1 truncate ${gone ? "line-through text-white/40" : "text-white/70"}`}>{f.name || f.machine_name}</span>
                    <span className="text-[9px] text-white/30 font-mono">{f.dtype}</span>
                    <button onClick={() => toggleRemove(f.machine_name)} className="text-white/40 hover:text-rose-300"
                      title={gone ? t("scenarioDetail.specEditor.restore") : t("scenarioDetail.specEditor.remove")}>
                      {gone ? <RotateCcw size={12} /> : <X size={12} />}
                    </button>
                  </div>
                );
              })}
              {added.map((a, i) => (
                <div key={`add-${i}`} className="flex items-center gap-2 text-[11px] rounded-lg px-2 py-1 border border-brand-500/25 bg-brand-500/5">
                  <span className="flex-1 truncate text-brand-200">{a.name}</span>
                  <span className="text-[9px] text-white/30 font-mono">{a.dtype}</span>
                  <span className="text-[9px] text-brand-300/60">{t("scenarioDetail.specEditor.new")}</span>
                  <button onClick={() => setAdded(prev => prev.filter((_, j) => j !== i))} className="text-white/40 hover:text-rose-300"><X size={12} /></button>
                </div>
              ))}
            </div>
          </div>
          <div className="flex flex-wrap items-end gap-2">
            <input value={addName} onChange={e => setAddName(e.target.value)} placeholder={t("scenarioDetail.specEditor.newVarName")}
              onKeyDown={e => { if (e.key === "Enter") { e.preventDefault(); addFeature(); } }}
              className="flex-1 min-w-[120px] rounded-lg bg-forest-950/60 border border-white/10 px-2 py-1.5 text-xs text-white" />
            <select value={addDtype} onChange={e => setAddDtype(e.target.value)}
              className="rounded-lg bg-forest-950/60 border border-white/10 px-2 py-1.5 text-xs text-white">
              {DTYPE_OPTIONS.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
            <button onClick={addFeature} disabled={!addName.trim()}
              className="flex items-center gap-1 rounded-lg border border-white/15 text-white/70 px-2.5 py-1.5 text-xs hover:bg-white/5 disabled:opacity-40">
              <Plus size={12} /> {t("scenarioDetail.specEditor.add")}
            </button>
          </div>
          {warnings.map((w, i) => (
            <p key={i} className="text-[11px] text-amber-300 flex items-start gap-1.5"><AlertTriangle size={12} className="mt-px shrink-0" />{w}</p>
          ))}
          {err && <p className="text-[11px] text-rose-300">{err}</p>}
          <button onClick={apply} disabled={busy || !dirty || remainingCount === 0}
            className="w-full flex items-center justify-center gap-1.5 rounded-xl bg-brand-600 text-white font-semibold py-2 text-xs hover:bg-brand-500 transition disabled:opacity-40">
            <RotateCcw size={12} className={busy ? "animate-spin" : ""} />
            {busy ? t("scenarioDetail.specEditor.applying") : t("scenarioDetail.specEditor.applyRetrain")}
          </button>
          {remainingCount === 0 && <p className="text-[10px] text-rose-300/70 text-center">{t("scenarioDetail.specEditor.needOne")}</p>}
        </div>
      )}
    </div>
  );
}

// Graphe de prévision (SVG autonome) : historique récent + ajustement sur le
// holdout (réel vs prévu) + prévision future avec intervalle. Rendu quand le
// modèle actif est un forecaster (task_type === "forecast").
function ForecastPanel({ run }: { run: ModelRun }) {
  const { t, lang } = useI18n();
  const s = (run.summary ?? {}) as any;
  const hist = s.history_tail ?? { dates: [], actual: [] };
  const hold = s.holdout ?? { dates: [], actual: [], predicted: [] };
  const fore = s.forecast ?? { dates: [], predicted: [], lower: [], upper: [] };
  const actual: number[] = (hist.actual ?? []).map(Number);
  const holdPred: number[] = (hold.predicted ?? []).map(Number);
  const foreVals: number[] = (fore.predicted ?? []).map(Number);
  const lower: number[] = (fore.lower ?? []).map(Number);
  const upper: number[] = (fore.upper ?? []).map(Number);
  const Lh = actual.length;
  const Hh = holdPred.length;      // holdout predicted aligns to the last Hh of history
  const Hf = foreVals.length;
  if (Lh < 2 || Hf < 1) return null;

  const totalN = Lh + Hf;
  const W = 640, H = 200, padL = 8, padR = 8, padT = 10, padB = 22;
  const allY = [...actual, ...holdPred, ...foreVals, ...lower, ...upper].filter(v => Number.isFinite(v));
  const yMin = Math.min(...allY), yMax = Math.max(...allY);
  const yPad = (yMax - yMin) * 0.08 || 1;
  const y0 = yMin - yPad, y1 = yMax + yPad;
  const xAt = (i: number) => padL + (i / Math.max(1, totalN - 1)) * (W - padL - padR);
  const yAt = (v: number) => padT + (1 - (v - y0) / Math.max(1e-9, y1 - y0)) * (H - padT - padB);
  const line = (vals: number[], startIdx: number) =>
    vals.map((v, k) => `${k === 0 ? "M" : "L"}${xAt(startIdx + k).toFixed(1)},${yAt(v).toFixed(1)}`).join(" ");

  // CI band polygon over the forecast region (top edge L→R, bottom edge R→L).
  const bandPts = (upper.length === Hf && lower.length === Hf)
    ? [
        ...upper.map((v, k) => `${xAt(Lh + k).toFixed(1)},${yAt(v).toFixed(1)}`),
        ...Array.from({ length: Hf }, (_unused, k) => {
          const j = Hf - 1 - k;
          return `${xAt(Lh + j).toFixed(1)},${yAt(lower[j]).toFixed(1)}`;
        }),
      ].join(" ")
    : "";
  // Bridge the actual line into the forecast (continuity at the seam).
  const foreLineStart = Lh - 1 >= 0 ? [actual[Lh - 1], ...foreVals] : foreVals;
  const foreLineStartIdx = Lh - 1 >= 0 ? Lh - 1 : Lh;

  const fmtDate = (iso?: string) => {
    if (!iso) return "";
    const d = new Date(iso);
    return isNaN(d.getTime()) ? "" : d.toLocaleDateString(lang === "fr" ? "fr-FR" : "en-US", { month: "short", day: "numeric" });
  };
  const firstDate = fmtDate(hist.dates?.[0]);
  const seamDate = fmtDate(hold.dates?.[0] ?? fore.dates?.[0]);
  const lastDate = fmtDate(fore.dates?.[fore.dates.length - 1]);
  const mape = run.metrics?.mape;

  return (
    <div className="rounded-xl border border-white/5 bg-white/2 px-3 py-2.5">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <span className="text-[10px] text-white/35 uppercase tracking-wider flex items-center gap-1">
          <TrendingUp size={11} /> {t("scenarioDetail.model.forecastTitle")}
        </span>
        <span className="text-[10px] text-white/40">
          {t("scenarioDetail.model.forecastHorizon")}: {s.horizon ?? Hf} · {run.metrics?.rmse != null ? `RMSE ${Number(run.metrics.rmse).toFixed(2)}` : ""}
          {typeof mape === "number" ? ` · MAPE ${mape.toFixed(1)}%` : ""}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="mt-2 w-full" style={{ height: "auto" }} preserveAspectRatio="none">
        {bandPts && <polygon points={bandPts} fill="rgba(56,189,248,0.12)" stroke="none" />}
        {/* seam between observed and forecast */}
        <line x1={xAt(Lh - 1)} y1={padT} x2={xAt(Lh - 1)} y2={H - padB} stroke="rgba(255,255,255,0.12)" strokeDasharray="3 3" />
        {/* actual (observed history incl. holdout) */}
        <path d={line(actual, 0)} fill="none" stroke="rgba(255,255,255,0.75)" strokeWidth={1.6} />
        {/* holdout predicted (model fit on held-out window) */}
        {Hh > 0 && <path d={line(holdPred, Lh - Hh)} fill="none" stroke="#fbbf24" strokeWidth={1.4} strokeDasharray="4 2" />}
        {/* forward forecast */}
        <path d={line(foreLineStart, foreLineStartIdx)} fill="none" stroke="#38bdf8" strokeWidth={1.6} strokeDasharray="4 2" />
      </svg>
      <div className="flex items-center justify-between text-[9px] text-white/30 font-mono">
        <span>{firstDate}</span><span>{seamDate}</span><span>{lastDate}</span>
      </div>
      <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1 text-[9px]">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-[2px] bg-white/70" />{t("scenarioDetail.model.forecastObserved")}</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-[2px]" style={{ background: "#fbbf24" }} />{t("scenarioDetail.model.forecastHoldout")}</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-[2px]" style={{ background: "#38bdf8" }} />{t("scenarioDetail.model.forecastFuture")}</span>
      </div>
    </div>
  );
}

// ── SEIR projection (modèle compartimental paramétré par la littérature) ─────
const _SEIR_SERIES = ["incidence", "prevalence", "cumulative", "deaths"] as const;
// Séries optionnelles n'apparaissant que si le modèle a le compartiment (V / Q).
const _SEIR_SERIES_OPT = ["vaccinated", "quarantined"] as const;
type SeirSeriesKey = typeof _SEIR_SERIES[number] | typeof _SEIR_SERIES_OPT[number];
const _SEIR_COLOR: Record<SeirSeriesKey, string> = {
  incidence: "#38bdf8", prevalence: "#a78bfa", cumulative: "#34d399", deaths: "#f87171",
  vaccinated: "#2dd4bf", quarantined: "#fbbf24",
};
const _SEIR_PARAM_LABEL: Record<string, string> = {
  r0: "R₀", beta: "β", infectious_period_days: "Infectious (d)", incubation_period_days: "Incubation (d)",
  cfr: "CFR", immunity_duration_days: "Immunity (d)", serial_interval_days: "Serial (d)",
  vaccination_rate: "Vaccination rate", vaccine_efficacy: "Vaccine efficacy", quarantine_rate: "Quarantine rate",
};

// ─── Onglet SEIR : modèle · formule · paramètres éditables (liés aux articles) ─
const _SEIR_PARAM_ORDER = ["r0", "incubation_period_days", "infectious_period_days", "cfr", "immunity_duration_days", "serial_interval_days"];
// Paramètres d'INTERVENTION (politique, pas extraits de la littérature) : affichés dans
// leur propre bloc « avancé » pour activer les compartiments V (vaccination) / Q (quarantaine).
const _SEIR_INTERVENTION_KEYS = ["vaccination_rate", "vaccine_efficacy", "quarantine_rate"];
const _SEIR_SYMBOL: Record<string, string> = {
  r0: "R₀ = β/γ", incubation_period_days: "σ = 1/incub.", infectious_period_days: "γ = 1/infect.",
  cfr: "f (CFR)", immunity_duration_days: "ω = 1/immun.", serial_interval_days: "—",
  vaccination_rate: "ν (/day)", vaccine_efficacy: "ε (0–1)", quarantine_rate: "κ (/day)",
};
// Équations du membre de droite selon les compartiments actifs du modèle sélectionné.
function _seirEquations(model: string): { compartments: string; eqs: string[] } {
  const hasV = model.includes("V");   // vaccination : S→V
  const hasE = model.includes("E");
  const hasQ = model.includes("Q");   // quarantaine/isolement : I→Q (Q ne transmet plus)
  const hasD = model.includes("D");
  const waning = model.endsWith("S");   // R→S (immunité décroissante) : SEIRS / SEIRDS / SIRS
  const compartments = ["S", hasV ? "V" : "", hasE ? "E" : "", "I", hasQ ? "Q" : "", "R", hasD ? "D" : ""].filter(Boolean).join(" · ");
  const inflow = hasQ ? "(I+Q)" : "I";  // le flux sortant infectieux (I et Q) alimente R/D
  const eqs: string[] = [`dS/dt = −β·S·I/N${hasV ? " − ν·S" : ""}${waning ? " + ω·R" : ""}`];
  if (hasV) eqs.push("dV/dt = ν·S");
  if (hasE) { eqs.push("dE/dt = β·S·I/N − σ·E", `dI/dt = σ·E − γ·I${hasQ ? " − κ·I" : ""}`); }
  else { eqs.push(`dI/dt = β·S·I/N − γ·I${hasQ ? " − κ·I" : ""}`); }
  if (hasQ) eqs.push("dQ/dt = κ·I − γ·Q");
  eqs.push(`dR/dt = γ·${hasD ? "(1−f)·" : ""}${inflow}${waning ? " − ω·R" : ""}`);
  if (hasD) eqs.push(`dD/dt = γ·f·${inflow}`);
  return { compartments, eqs };
}

// Format a magnitude compactly for a chart axis (1.2M, 340k, 4.5, 0).
function _fmtCompact(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toFixed(a >= 1e10 ? 0 : 1)}B`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(a >= 1e7 ? 0 : 1)}M`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(a >= 1e4 ? 0 : 1)}k`;
  if (a >= 10) return v.toFixed(0);
  if (a >= 1) return v.toFixed(1);
  return v === 0 ? "0" : v.toFixed(2);
}

function SeirBandChart({ band, dates, color, lang, yUnit, observed, observedColor }: {
  band: SeirBand; dates: (string | number)[]; color: string; lang: string; yUnit?: string;
  observed?: { day: number; value: number }[]; observedColor?: string;
}) {
  const median = (band?.median ?? []).map(Number);
  const lower = (band?.lower ?? []).map(Number);
  const upper = (band?.upper ?? []).map(Number);
  const N = median.length;
  if (N < 2) return null;
  const W = 680, H = 240, padL = 50, padR = 14, padT = 14, padB = 30;
  const obs = (observed ?? []).filter(p => Number.isFinite(p.value) && p.day >= 0 && p.day <= N - 1);
  const allY = [...median, ...lower, ...upper, ...obs.map(p => p.value)].filter(v => Number.isFinite(v));
  const dataMax = Math.max(...allY), dataMin = Math.min(...allY);
  const y0 = Math.min(0, dataMin);                                     // baseline at 0 (épidémio)
  const y1 = dataMax > y0 ? dataMax + (dataMax - y0) * 0.08 : y0 + 1;
  const xAt = (i: number) => padL + (i / Math.max(1, N - 1)) * (W - padL - padR);
  const yAt = (v: number) => padT + (1 - (v - y0) / Math.max(1e-9, y1 - y0)) * (H - padT - padB);
  const linePath = (vals: number[]) => vals.map((v, k) => `${k === 0 ? "M" : "L"}${xAt(k).toFixed(1)},${yAt(v).toFixed(1)}`).join(" ");
  const bandPts = (upper.length === N && lower.length === N)
    ? [...upper.map((v, k) => `${xAt(k).toFixed(1)},${yAt(v).toFixed(1)}`),
       ...Array.from({ length: N }, (_u, k) => { const j = N - 1 - k; return `${xAt(j).toFixed(1)},${yAt(lower[j]).toFixed(1)}`; })].join(" ")
    : "";
  const fmtX = (d: string | number | undefined) => {
    if (d == null) return "";
    if (typeof d === "number") return `${lang === "fr" ? "J" : "D"}${d}`;   // jour/day depuis le début
    const dt = new Date(d); return isNaN(dt.getTime()) ? String(d) : dt.toLocaleDateString(lang === "fr" ? "fr-FR" : "en-US", { month: "short", day: "numeric" });
  };
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map(f => y0 + f * (y1 - y0));
  const nX = Math.min(6, N);
  const xTicks = Array.from({ length: nX }, (_v, i) => Math.round((i * (N - 1)) / (nX - 1)));
  const peakI = median.reduce((bi, v, i) => (v > median[bi] ? i : bi), 0);   // pic de la médiane
  const xAxisLabel = typeof dates[0] === "number"
    ? (lang === "fr" ? "Jours depuis le début" : "Days from start")
    : (lang === "fr" ? "Date" : "Date");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="mt-2 w-full text-white/70" style={{ height: "auto" }} role="img" aria-label="SEIR projection">
      {/* y gridlines + value labels */}
      {yTicks.map((v, i) => (
        <g key={`y${i}`}>
          <line x1={padL} x2={W - padR} y1={yAt(v)} y2={yAt(v)} stroke="currentColor" strokeOpacity={0.1} strokeWidth={0.5} />
          <text x={padL - 6} y={yAt(v) + 3} textAnchor="end" fontSize={9} fill="currentColor" fillOpacity={0.6} fontFamily="monospace">{_fmtCompact(v)}</text>
        </g>
      ))}
      {yUnit && <text x={12} y={padT + (H - padT - padB) / 2} textAnchor="middle" fontSize={8.5} fill="currentColor" fillOpacity={0.45}
        transform={`rotate(-90 12 ${padT + (H - padT - padB) / 2})`}>{yUnit}</text>}
      {/* x axis line + ticks + labels */}
      <line x1={padL} x2={W - padR} y1={H - padB} y2={H - padB} stroke="currentColor" strokeOpacity={0.3} strokeWidth={0.6} />
      {xTicks.map((i, k) => (
        <g key={`x${k}`}>
          <line x1={xAt(i)} x2={xAt(i)} y1={H - padB} y2={H - padB + 4} stroke="currentColor" strokeOpacity={0.4} strokeWidth={0.6} />
          <text x={xAt(i)} y={H - padB + 16} textAnchor="middle" fontSize={9} fill="currentColor" fillOpacity={0.65} fontFamily="monospace">{fmtX(dates[i])}</text>
        </g>
      ))}
      <text x={padL + (W - padL - padR) / 2} y={H - 2} textAnchor="middle" fontSize={8.5} fill="currentColor" fillOpacity={0.4}>{xAxisLabel}</text>
      {/* uncertainty band + median line */}
      {bandPts && <polygon points={bandPts} fill={color} fillOpacity={0.16} stroke="none" />}
      <path d={linePath(median)} fill="none" stroke={color} strokeWidth={1.8} />
      {/* observed data overlay (real series, re-expressed in model units at the aligned day) */}
      {obs.map((p, k) => (
        <circle key={`obs${k}`} cx={xAt(p.day)} cy={yAt(p.value)} r={2.3}
          fill={observedColor ?? "#f472b6"} fillOpacity={0.92} stroke="currentColor"
          strokeWidth={0.3} strokeOpacity={0.25} />
      ))}
      {/* peak marker (median), labelled with its day/date */}
      <circle cx={xAt(peakI)} cy={yAt(median[peakI])} r={2.6} fill={color} />
      <text x={Math.min(Math.max(xAt(peakI), padL + 16), W - padR - 16)} y={yAt(median[peakI]) - 6}
        textAnchor="middle" fontSize={8.5} fill={color} fontFamily="monospace">{fmtX(dates[peakI])}</text>
    </svg>
  );
}

function SeirModelView({ scenarioId }: { scenarioId: string }) {
  const { t, lang } = useI18n();
  const [proj, setProj] = useState<SeirProjection | null>(null);
  const [provIndex, setProvIndex] = useState<Record<string, ProvArticle>>({});
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [series, setSeries] = useState<SeirSeriesKey>("incidence");
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [popEdit, setPopEdit] = useState("");
  const [i0Edit, setI0Edit] = useState("");
  // Série observée (réelle) : upload/connecteur → superposition + calibration.
  const [connectors, setConnectors] = useState<DataConnector[]>([]);
  const [obsText, setObsText] = useState("");
  const [obsConnector, setObsConnector] = useState("");
  const [obsVariable, setObsVariable] = useState("");
  const [obsRegion, setObsRegion] = useState("");
  const [obsColumn, setObsColumn] = useState<SeirSeriesKey>("incidence");
  const [obsBusy, setObsBusy] = useState(false);
  const [obsError, setObsError] = useState("");
  const [calib, setCalib] = useState<SeirCalibration | null>(null);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    Promise.all([
      fetchSeirProjection(scenarioId).catch(() => null),
      getScenarioModelSpec(scenarioId).catch(() => null),
    ]).then(([p, s]) => {
      if (!alive) return;
      setProj(p);
      setProvIndex((s?.provenance_index as Record<string, ProvArticle>) ?? {});
    }).finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [scenarioId]);

  useEffect(() => { listDataConnectors().then(setConnectors).catch(() => setConnectors([])); }, []);

  if (loading && !proj) return <LoadingSpinner text={t("scenarioDetail.seirTab.loading")} />;
  if (!proj || !proj.applicable) {
    return (
      <div className="rounded-2xl border border-white/8 bg-white/2 px-5 py-10 text-center space-y-2">
        <TrendingUp size={22} className="mx-auto text-white/25" />
        <p className="text-sm font-medium text-white/70">{t("scenarioDetail.seirTab.notApplicableTitle")}</p>
        <p className="text-[12px] text-white/45 max-w-md mx-auto leading-relaxed">
          {proj?.reason ?? t("scenarioDetail.seirTab.notApplicable")}</p>
      </div>
    );
  }

  const model = proj.model ?? "SEIR";
  const { compartments, eqs } = _seirEquations(model);
  const params = proj.effective_parameters ?? proj.parameters ?? {};
  const fmtPop = (n?: number) => (typeof n === "number"
    ? (n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : `${Math.round(n)}`) : "—");
  const hasEdits = Object.values(edits).some(v => v.trim() !== "") || popEdit.trim() !== "" || i0Edit.trim() !== "";
  const hasOverrides = !!proj.overrides_applied && Object.keys(proj.overrides_applied).length > 0;

  const recompute = () => {
    const overrides: Record<string, SeirOverride> = {};
    Object.entries(edits).forEach(([k, v]) => { const n = parseFloat(v); if (v.trim() !== "" && isFinite(n)) overrides[k] = { value: n }; });
    const body: { overrides: Record<string, SeirOverride>; population?: number; initial_infected?: number } = { overrides };
    const pop = parseFloat(popEdit); if (popEdit.trim() !== "" && isFinite(pop)) body.population = pop;
    const i0 = parseFloat(i0Edit); if (i0Edit.trim() !== "" && isFinite(i0)) body.initial_infected = i0;
    setBusy(true);
    postSeirProjection(scenarioId, body).then(p => { if (p?.applicable) setProj(p); }).finally(() => setBusy(false));
  };
  const reset = () => {
    setEdits({}); setPopEdit(""); setI0Edit(""); setBusy(true);
    fetchSeirProjection(scenarioId).then(p => setProj(p)).finally(() => setBusy(false));
  };

  // ── Série observée (réelle) : upload/connecteur → superposition + calibration ──
  const parseObsText = (txt: string): { date?: string; day?: number; value: number }[] => {
    const pts: { date?: string; day?: number; value: number }[] = [];
    txt.split(/\r?\n/).forEach(line => {
      const cells = line.split(/[,;\t]/).map(s => s.trim());
      if (cells.length < 2) return;
      const v = parseFloat(cells[1]); if (!isFinite(v)) return;
      if (/^\d{4}-\d{2}-\d{2}$/.test(cells[0])) pts.push({ date: cells[0], value: v });
      else { const d = parseFloat(cells[0]); if (isFinite(d)) pts.push({ day: d, value: v }); }
    });
    return pts;
  };
  const refetchProj = () => fetchSeirProjection(scenarioId).then(p => { if (p?.applicable) setProj(p); });
  const attachObserved = () => {
    setObsBusy(true); setObsError("");
    const body: Parameters<typeof postSeirObserved>[1] = { column: obsColumn };
    if (obsConnector && obsVariable) {
      body.connector_id = obsConnector; body.connector_variable = obsVariable;
      if (obsRegion.trim()) body.region = obsRegion.trim();
    } else {
      body.points = parseObsText(obsText);
    }
    postSeirObserved(scenarioId, body)
      .then(refetchProj)
      .then(() => { setObsText(""); setCalib(null); })
      .catch(e => setObsError(String(e?.message ?? e)))
      .finally(() => setObsBusy(false));
  };
  const clearObserved = () => {
    setObsBusy(true); setObsError("");
    deleteSeirObserved(scenarioId).then(refetchProj).then(() => setCalib(null))
      .catch(e => setObsError(String(e?.message ?? e))).finally(() => setObsBusy(false));
  };
  const runCalibrate = () => {
    setObsBusy(true); setObsError("");
    const overrides: Record<string, SeirOverride> = {};
    Object.entries(edits).forEach(([k, v]) => { const n = parseFloat(v); if (v.trim() !== "" && isFinite(n)) overrides[k] = { value: n }; });
    calibrateSeir(scenarioId, { column: proj.observed?.column, overrides })
      .then(setCalib).catch(e => setObsError(String(e?.message ?? e))).finally(() => setObsBusy(false));
  };
  const applyFitted = () => {
    if (calib?.fitted_r0 == null) return;
    const next = { ...edits, r0: String(calib.fitted_r0) };
    setEdits(next);
    const overrides: Record<string, SeirOverride> = {};
    Object.entries(next).forEach(([k, v]) => { const n = parseFloat(v); if (v.trim() !== "" && isFinite(n)) overrides[k] = { value: n }; });
    const body: { overrides: Record<string, SeirOverride>; population?: number; initial_infected?: number } = { overrides };
    const pop = parseFloat(popEdit); if (popEdit.trim() !== "" && isFinite(pop)) body.population = pop;
    const i0 = parseFloat(i0Edit); if (i0Edit.trim() !== "" && isFinite(i0)) body.initial_infected = i0;
    setBusy(true);
    postSeirProjection(scenarioId, body).then(p => { if (p?.applicable) setProj(p); }).finally(() => setBusy(false));
  };

  const sm = proj.summary;
  const seriesLabel: Record<SeirSeriesKey, string> = {
    incidence: t("scenarioDetail.seir.seriesIncidence"), prevalence: t("scenarioDetail.seir.seriesPrevalence"),
    cumulative: t("scenarioDetail.seir.seriesCumulative"), deaths: t("scenarioDetail.seir.seriesDeaths"),
    vaccinated: t("scenarioDetail.seir.seriesVaccinated"), quarantined: t("scenarioDetail.seir.seriesQuarantined"),
  };
  const perDay = lang === "fr" ? "personnes/j" : "people/day";
  const people = lang === "fr" ? "personnes" : "people";
  const seriesUnit: Record<SeirSeriesKey, string> = {
    incidence: perDay, prevalence: people, cumulative: people, deaths: people,
    vaccinated: people, quarantined: people,
  };
  // Séries de base + celles des compartiments V/Q seulement si le modèle les possède.
  const availableSeries: SeirSeriesKey[] = [..._SEIR_SERIES, ..._SEIR_SERIES_OPT.filter(k => proj.series?.[k])];
  const activeSeries: SeirSeriesKey = availableSeries.includes(series) ? series : "incidence";
  const band = proj.series?.[activeSeries];
  const orderedKeys = [..._SEIR_PARAM_ORDER.filter(k => k in params),
    ...Object.keys(params).filter(k => !_SEIR_PARAM_ORDER.includes(k) && !_SEIR_INTERVENTION_KEYS.includes(k))];

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-white/8 bg-white/3 p-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="flex items-center gap-2">
            <span className="rounded-lg bg-brand-500/15 border border-brand-500/30 text-brand-300 px-2.5 py-1 text-sm font-bold font-mono">{model}</span>
            {proj.disease && <span className="text-sm text-white/70">{proj.disease}</span>}
          </div>
          <p className="text-[11px] text-white/40 mt-1">{t("scenarioDetail.seirTab.compartments")}: <span className="font-mono text-white/60">{compartments}</span></p>
        </div>
        <div className="text-right text-[11px] text-white/50">
          <p>{t("scenarioDetail.seir.population")}: <span className="font-mono text-white/80">{fmtPop(proj.population)}</span>{proj.geography ? ` · ${proj.geography}` : ""}</p>
          <p className="text-white/35">{proj.n_samples} {t("scenarioDetail.seir.draws")}</p>
        </div>
      </div>

      <div className="rounded-2xl border border-white/8 bg-white/2 p-4">
        <p className="text-[10px] uppercase tracking-wider text-white/40 mb-2">{t("scenarioDetail.seirTab.equations")}</p>
        <div className="font-mono text-[13px] text-white/80 space-y-1 overflow-x-auto">
          {eqs.map((e, i) => <div key={i}>{e}</div>)}
        </div>
        <p className="text-[10px] text-white/35 mt-2 leading-relaxed">
          β = {t("scenarioDetail.seirTab.beta")}, σ = 1/{t("scenarioDetail.seirTab.incubation")}, γ = 1/{t("scenarioDetail.seirTab.infectious")}
          {model.includes("D") ? `, f = ${t("scenarioDetail.seirTab.cfr")}` : ""}{model.endsWith("S") ? `, ω = 1/${t("scenarioDetail.seirTab.immunity")}` : ""}
          {model.includes("V") ? `, ν = ${t("scenarioDetail.seirTab.vaccination")}` : ""}{model.includes("Q") ? `, κ = ${t("scenarioDetail.seirTab.quarantine")}` : ""}
        </p>
      </div>

      <div className="rounded-2xl border border-white/5 bg-white/2 p-3">
        <div className="flex flex-wrap gap-1.5">
          {availableSeries.map(k => (
            <button key={k} type="button" onClick={() => setSeries(k)}
              className={`rounded-full px-2 py-0.5 text-[10px] border transition ${activeSeries === k ? "border-white/30 bg-white/10 text-white" : "border-white/10 text-white/50 hover:bg-white/5"}`}>{seriesLabel[k]}</button>
          ))}
        </div>
        {band && <SeirBandChart band={band} dates={proj.dates ?? []} color={_SEIR_COLOR[activeSeries]} lang={lang} yUnit={seriesUnit[activeSeries]}
          observed={proj.observed && proj.observed.column === activeSeries ? proj.observed.points : undefined} observedColor="#f472b6" />}
        {proj.observed && (
          <p className="text-[9px] text-white/35 mt-1">
            <span className="inline-block w-2 h-2 rounded-full bg-pink-400 mr-1 align-middle" />
            {t("scenarioDetail.seirTab.observedLegend")}{proj.observed.column !== activeSeries ? ` (${seriesLabel[proj.observed.column as SeirSeriesKey] ?? proj.observed.column})` : ""}
            {proj.observed.fit_r2 != null ? ` · R²=${proj.observed.fit_r2}` : ""}
          </p>
        )}
        {sm && (
          <div className="mt-2 grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px]">
            <div><div className="text-white/40">{t("scenarioDetail.seir.peakPrevalence")}</div><div className="font-mono text-white/80">{fmtPop(sm.peak_prevalence.median)}</div></div>
            <div><div className="text-white/40">{t("scenarioDetail.seir.peakDay")}</div><div className="font-mono text-white/80">{lang === "fr" ? "J" : "D"}{sm.peak_prevalence_day.median}</div></div>
            <div><div className="text-white/40">{t("scenarioDetail.seir.attackRate")}</div><div className="font-mono text-white/80">{(sm.attack_rate.median * 100).toFixed(1)}%</div></div>
            <div><div className="text-white/40">{t("scenarioDetail.seir.deaths")}</div><div className="font-mono text-white/80">{sm.total_deaths.median}</div></div>
            {sm.total_vaccinated && sm.total_vaccinated.median > 0 && (
              <div><div className="text-white/40">{t("scenarioDetail.seir.totalVaccinated")}</div><div className="font-mono text-teal-300">{fmtPop(sm.total_vaccinated.median)}</div></div>)}
            {sm.peak_quarantine && sm.peak_quarantine.median > 0 && (
              <div><div className="text-white/40">{t("scenarioDetail.seir.peakQuarantine")}</div><div className="font-mono text-amber-300">{fmtPop(sm.peak_quarantine.median)}</div></div>)}
          </div>
        )}
      </div>

      <div className="rounded-2xl border border-white/8 bg-white/2 p-4 space-y-3">
        <div className="flex items-center justify-between">
          <p className="text-[10px] uppercase tracking-wider text-white/40">{t("scenarioDetail.seirTab.parameters")}</p>
          <p className="text-[9px] text-white/30">{t("scenarioDetail.seirTab.editHint")}</p>
        </div>
        <div className="space-y-2">
          {orderedKeys.map(k => {
            const p = params[k] as SeirParamValue;
            const arts = (p.provenance ?? []).map(id => provIndex[String(id)]).filter(Boolean) as ProvArticle[];
            return (
              <div key={k} className="grid grid-cols-12 items-center gap-2 text-[11px] border-b border-white/5 pb-2">
                <div className="col-span-4 sm:col-span-3">
                  <div className="text-white/80 font-medium">{_SEIR_PARAM_LABEL[k] ?? k}</div>
                  <div className="text-[9px] text-white/35 font-mono">{_SEIR_SYMBOL[k] ?? ""}</div>
                </div>
                <div className="col-span-4 sm:col-span-3">
                  <input type="number" step="any" value={edits[k] ?? ""} placeholder={String(p.value)}
                    onChange={e => setEdits(prev => ({ ...prev, [k]: e.target.value }))}
                    className="w-full rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-white/90 font-mono focus:border-brand-400/50 outline-none" />
                  {p.overridden && <span className="text-[8px] text-gold-300">{t("scenarioDetail.seirTab.overridden")}</span>}
                </div>
                <div className="col-span-4 sm:col-span-3 font-mono text-white/45">
                  {p.ci_low != null && p.ci_high != null ? `[${p.ci_low}–${p.ci_high}]` : (p.n_studies ? `${p.n_studies} ${t("scenarioDetail.seir.studies")}` : "—")}
                  {p.unit ? ` ${p.unit}` : ""}
                </div>
                <div className="hidden sm:flex col-span-3 flex-wrap gap-1 justify-end">
                  {arts.slice(0, 3).map((a, i) => <ArticleSourceLink key={i} a={a} label={t("scenarioDetail.common.ref")} />)}
                </div>
              </div>
            );
          })}
        </div>
        <div className="grid grid-cols-2 gap-2 text-[11px]">
          <label className="space-y-1 block"><span className="text-white/40">{t("scenarioDetail.seir.population")}</span>
            <input type="number" step="any" value={popEdit} placeholder={String(Math.round(proj.population ?? 0))}
              onChange={e => setPopEdit(e.target.value)} className="w-full rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-white/90 font-mono outline-none focus:border-brand-400/50" /></label>
          <label className="space-y-1 block"><span className="text-white/40">{t("scenarioDetail.seirTab.initialInfected")}</span>
            <input type="number" step="any" value={i0Edit} placeholder={String(Math.round(proj.initial_infected ?? 0))}
              onChange={e => setI0Edit(e.target.value)} className="w-full rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-white/90 font-mono outline-none focus:border-brand-400/50" /></label>
        </div>

        {/* Paramètres d'intervention (avancés) : activent les compartiments V / Q d'un modèle plus riche */}
        <div className="rounded-xl border border-dashed border-white/12 bg-white/2 p-3 space-y-2">
          <div className="flex items-center gap-1.5">
            <Sliders size={12} className="text-teal-300/80" />
            <p className="text-[10px] uppercase tracking-wider text-white/45">{t("scenarioDetail.seirTab.interventions")}</p>
          </div>
          <p className="text-[9px] text-white/30 leading-relaxed">{t("scenarioDetail.seirTab.interventionsHint")}</p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-[11px]">
            {_SEIR_INTERVENTION_KEYS.map(k => {
              const cur = params[k] as SeirParamValue | undefined;
              return (
                <label key={k} className="space-y-1 block">
                  <span className="text-white/55">{_SEIR_PARAM_LABEL[k]} <span className="text-white/30 font-mono">{_SEIR_SYMBOL[k]}</span></span>
                  <input type="number" step="any" min="0" value={edits[k] ?? ""}
                    placeholder={cur ? String(cur.value) : (k === "vaccine_efficacy" ? "1.0" : "0")}
                    onChange={e => setEdits(prev => ({ ...prev, [k]: e.target.value }))}
                    className="w-full rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-white/90 font-mono outline-none focus:border-teal-400/50" />
                  {cur?.overridden && <span className="text-[8px] text-teal-300">{t("scenarioDetail.seirTab.active")}</span>}
                </label>
              );
            })}
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <button type="button" onClick={recompute} disabled={busy || !hasEdits}
            className="flex items-center gap-1.5 rounded-xl bg-brand-600 text-white font-semibold py-1.5 px-4 text-xs hover:bg-brand-500 transition disabled:opacity-50">
            {busy ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}{t("scenarioDetail.seirTab.recompute")}</button>
          {(hasEdits || hasOverrides) && (
            <button type="button" onClick={reset} disabled={busy}
              className="rounded-xl border border-white/15 text-white/60 py-1.5 px-3 text-xs hover:bg-white/5 transition disabled:opacity-50">{t("scenarioDetail.seirTab.reset")}</button>
          )}
          {hasOverrides && <span className="text-[10px] text-gold-300">{t("scenarioDetail.seirTab.usingOverrides")}</span>}
        </div>
      </div>

      {/* Données observées (réelles) : superposition sur le graphe + calibration du modèle */}
      <div className="rounded-2xl border border-white/8 bg-white/2 p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <Target size={12} className="text-pink-300/80" />
            <p className="text-[10px] uppercase tracking-wider text-white/45">{t("scenarioDetail.seirTab.observed")}</p>
          </div>
          {proj.observed && <span className="text-[9px] text-white/40">{proj.observed.n} {t("scenarioDetail.seirTab.obsPoints")}</span>}
        </div>

        {proj.observed ? (
          <div className="space-y-3">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px]">
              <div className="truncate"><div className="text-white/40">{t("scenarioDetail.seirTab.obsSource")}</div><div className="font-mono text-white/80 truncate" title={proj.observed.label ?? ""}>{proj.observed.label ?? proj.observed.source ?? "—"}</div></div>
              <div><div className="text-white/40">{t("scenarioDetail.seirTab.obsSeries")}</div><div className="font-mono text-white/80">{seriesLabel[proj.observed.column as SeirSeriesKey] ?? proj.observed.column}</div></div>
              <div><div className="text-white/40">{t("scenarioDetail.seirTab.obsFit")}</div><div className="font-mono text-white/80">{proj.observed.fit_r2 != null ? `R²=${proj.observed.fit_r2}` : "—"}</div></div>
              <div><div className="text-white/40">{t("scenarioDetail.seirTab.obsShift")}</div><div className="font-mono text-white/80">{proj.observed.shift_days != null ? `${proj.observed.shift_days} ${lang === "fr" ? "j" : "d"}` : "—"}</div></div>
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              <button type="button" onClick={runCalibrate} disabled={obsBusy}
                className="flex items-center gap-1.5 rounded-xl bg-teal-600 text-white font-semibold py-1.5 px-4 text-xs hover:bg-teal-500 transition disabled:opacity-50">
                {obsBusy ? <Loader2 size={12} className="animate-spin" /> : <Target size={12} />}{t("scenarioDetail.seirTab.calibrate")}</button>
              <button type="button" onClick={clearObserved} disabled={obsBusy}
                className="rounded-xl border border-white/15 text-white/60 py-1.5 px-3 text-xs hover:bg-white/5 transition disabled:opacity-50">{t("scenarioDetail.seirTab.obsClear")}</button>
            </div>
            {calib?.ok && (
              <div className="rounded-xl border border-teal-500/20 bg-teal-500/5 p-3 text-[11px] space-y-2">
                <div className="flex flex-wrap gap-x-4 gap-y-1">
                  <span className="text-white/60">{t("scenarioDetail.seirTab.fittedR0")}: <span className="font-mono text-teal-300 font-semibold">{calib.fitted_r0}</span></span>
                  {calib.literature_r0 != null && <span className="text-white/60">{t("scenarioDetail.seirTab.literatureR0")}: <span className="font-mono text-white/80">{calib.literature_r0}</span></span>}
                  <span className="text-white/60">R²: <span className="font-mono text-white/80">{calib.r2}</span></span>
                  {calib.shift_days != null && <span className="text-white/60">{t("scenarioDetail.seirTab.obsShift")}: <span className="font-mono text-white/80">{calib.shift_days} {lang === "fr" ? "j" : "d"}</span></span>}
                </div>
                <button type="button" onClick={applyFitted} disabled={busy}
                  className="rounded-lg bg-teal-600/80 text-white py-1 px-3 text-[11px] hover:bg-teal-500 transition disabled:opacity-50">{t("scenarioDetail.seirTab.applyFitted")}</button>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            <p className="text-[9px] text-white/30 leading-relaxed">{t("scenarioDetail.seirTab.observedHint")}</p>
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-2 text-[11px]">
              <select value={obsConnector} onChange={e => { setObsConnector(e.target.value); setObsVariable(""); }}
                className="rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-white/90 outline-none focus:border-teal-400/50">
                <option value="" className="bg-slate-800">{t("scenarioDetail.seirTab.obsUpload")}</option>
                {connectors.map(c => <option key={c.id} value={c.id} className="bg-slate-800">{c.name}</option>)}
              </select>
              {obsConnector && (
                <select value={obsVariable} onChange={e => setObsVariable(e.target.value)}
                  className="rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-white/90 outline-none focus:border-teal-400/50">
                  <option value="" className="bg-slate-800">{t("scenarioDetail.seirTab.obsVariable")}</option>
                  {(connectors.find(c => c.id === obsConnector)?.variables ?? []).map(v => <option key={v.machine_name} value={v.machine_name} className="bg-slate-800">{v.label}</option>)}
                </select>
              )}
              {obsConnector && <input value={obsRegion} onChange={e => setObsRegion(e.target.value)} placeholder={t("scenarioDetail.seirTab.obsRegion")}
                className="rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-white/90 outline-none focus:border-teal-400/50" />}
              <select value={obsColumn} onChange={e => setObsColumn(e.target.value as SeirSeriesKey)}
                className="rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-white/90 outline-none focus:border-teal-400/50">
                {(["incidence", "prevalence", "cumulative", "deaths"] as SeirSeriesKey[]).map(k => <option key={k} value={k} className="bg-slate-800">{seriesLabel[k]}</option>)}
              </select>
            </div>
            {!obsConnector && (
              <textarea value={obsText} onChange={e => setObsText(e.target.value)} rows={3}
                placeholder={"2023-01-08, 12.4\n2023-01-15, 15.1\n2023-01-22, 18.2"}
                className="w-full rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-white/90 font-mono text-[11px] outline-none focus:border-teal-400/50" />
            )}
            <button type="button" onClick={attachObserved}
              disabled={obsBusy || (obsConnector ? !obsVariable : parseObsText(obsText).length < 3)}
              className="flex items-center gap-1.5 rounded-xl bg-teal-600 text-white font-semibold py-1.5 px-4 text-xs hover:bg-teal-500 transition disabled:opacity-50">
              {obsBusy ? <Loader2 size={12} className="animate-spin" /> : <Upload size={12} />}{t("scenarioDetail.seirTab.obsAttach")}</button>
          </div>
        )}
        {obsError && <p className="text-[10px] text-rose-300">{obsError}</p>}
      </div>

      <p className="text-[9px] text-white/25 leading-tight">{t("scenarioDetail.seirTab.note")}</p>
    </div>
  );
}

function VariablesSection({ detail, scenarioId, onGoToModel }: { detail: ScenarioDetail; scenarioId: string; onGoToModel?: () => void }) {
  const { t, lang } = useI18n();
  // État du modèle entraîné + des données branchées, pour relier ce panneau au
  // "Modèle Prédictif" : on montre par variable si elle est branchée, et quel
  // algorithme a réellement été entraîné.
  const [modelRun, setModelRun] = useState<ModelRun | null>(null);
  const [modelDataset, setModelDataset] = useState<ModelDataset | null>(null);
  const [modelSpec, setModelSpec] = useState<ModelSpecResponse | null>(null);
  const reloadModelState = React.useCallback(() => {
    getModelRun(scenarioId).then(setModelRun).catch(() => setModelRun(null));
    getModelDataset(scenarioId).then(setModelDataset).catch(() => setModelDataset(null));
    getScenarioModelSpec(scenarioId).then(setModelSpec).catch(() => setModelSpec(null));
  }, [scenarioId]);
  React.useEffect(() => { reloadModelState(); }, [reloadModelState]);

  // machine_name -> article source (le plus récent / cité) pour chaque variable.
  const sourceByVar: Record<string, ProvArticle | null> = {};
  (modelSpec?.features ?? []).forEach(f => { if (f.machine_name) sourceByVar[f.machine_name] = f.best_article ?? null; });

  const trained = modelRun?.status === "ready";
  const trainedMetricValue = (() => {
    if (!trained || !modelRun?.metrics || !modelRun.metric) return null;
    const v = modelRun.metrics[modelRun.metric];
    return typeof v === "number" ? v : null;
  })();
  const val = modelDataset?.validation;
  const dataStatusFor = (machineName?: string): "plugged" | "missing_user" | "public" | "unknown" => {
    if (!machineName || !val) return "unknown";
    if ((val.matched_features ?? []).includes(machineName)) return "plugged";
    if ((val.missing_public ?? []).includes(machineName)) return "public";
    if ((val.missing_user ?? []).includes(machineName)) return "missing_user";
    return "unknown";
  };

  const [file, setFile] = useState<File | null>(null);
  const [uploading, setLoading] = useState(false);
  const [uploadResult, setUploadResult] = useState<any | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // LLM Variables auto-fill
  const [llmVars, setLlmVars] = useState<ScenarioVariables | null>(null);
  const [llmLoading, setLlmLoading] = useState(false);
  const [llmGenerating, setLlmGenerating] = useState(false);
  const [llmGenStatus, setLlmGenStatus] = useState<string | null>(null);
  const [llmValidating, setLlmValidating] = useState(false);
  const [llmError, setLlmError] = useState<string | null>(null);
  const llmPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Anti-setState-après-démontage : garde-fou pour le suivi d'entraînement lancé
  // depuis l'upload (le poll récursif ci-dessous peut survivre au démontage).
  const mountedRef = useRef(true);
  const trainPollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const loadLlmVars = React.useCallback(() => {
    setLlmLoading(true);
    getScenarioVariables(scenarioId)
      .then(d => { setLlmVars(d); setLlmLoading(false); })
      .catch(() => { setLlmLoading(false); });
  }, [scenarioId]);

  React.useEffect(() => {
    loadLlmVars();
    return () => { if (llmPollRef.current) clearInterval(llmPollRef.current); };
  }, [loadLlmVars]);

  // Démontage : stoppe le suivi d'entraînement et bloque tout setState tardif.
  React.useEffect(() => () => {
    mountedRef.current = false;
    if (trainPollRef.current) clearTimeout(trainPollRef.current);
  }, []);

  // Polling si generation en cours
  React.useEffect(() => {
    if (!llmVars || llmVars.status !== 'generating') return;
    setLlmGenStatus(t("scenarioDetail.variables.genInProgress"));
    if (llmPollRef.current) clearInterval(llmPollRef.current);   // pas d'orphelin
    llmPollRef.current = setInterval(() => {
      getVariablesGenerationStatus(scenarioId).then(s => {
        if (s.status === 'done') {
          if (llmPollRef.current) clearInterval(llmPollRef.current);
          setLlmGenStatus(null);
          loadLlmVars();
          reloadModelState();   // le model_spec devient "ready" à la fin de la génération
                                // → rafraîchir pour afficher « Ajuster le modèle » sans reload
        } else if (s.status === 'error') {
          if (llmPollRef.current) clearInterval(llmPollRef.current);
          setLlmGenStatus(null);
          setLlmError(s.error ?? t("scenarioDetail.variables.genError"));
        }
      }).catch(() => {
        // Erreur transitoire (réseau/500) : SANS ce catch, l'intervalle n'était
        // jamais nettoyé (rejet non géré) → spinner "génération" bloqué à vie.
        if (llmPollRef.current) clearInterval(llmPollRef.current);
        setLlmGenStatus(null);
        setLlmError(t("scenarioDetail.variables.genError"));
      });
    }, 5000);
    return () => { if (llmPollRef.current) clearInterval(llmPollRef.current); };
  }, [llmVars, scenarioId, loadLlmVars, reloadModelState, t]);

  const handleGenerateLlm = async () => {
    setLlmGenerating(true);
    setLlmGenStatus(t("scenarioDetail.variables.genStarting"));
    setLlmError(null);
    try {
      await generateScenarioVariables(scenarioId);
      if (llmPollRef.current) clearInterval(llmPollRef.current);   // avoid orphaning the effect's poll
      llmPollRef.current = setInterval(() => {
        getVariablesGenerationStatus(scenarioId).then(s => {
          if (s.status === 'done') {
            if (llmPollRef.current) clearInterval(llmPollRef.current);
            setLlmGenStatus(null);
            setLlmGenerating(false);
            loadLlmVars();
            reloadModelState();   // idem : rendre « Ajuster le modèle » visible immédiatement
          } else if (s.status === 'error') {
            if (llmPollRef.current) clearInterval(llmPollRef.current);
            setLlmGenStatus(null);
            setLlmGenerating(false);
            setLlmError(s.error ?? t("scenarioDetail.variables.genError"));
          }
        }).catch(() => {
          // Erreur transitoire : nettoyer l'intervalle + sortir de l'état "génération"
          // (sinon rejet non géré + spinner bloqué à vie).
          if (llmPollRef.current) clearInterval(llmPollRef.current);
          setLlmGenStatus(null);
          setLlmGenerating(false);
          setLlmError(t("scenarioDetail.variables.genError"));
        });
      }, 5000);
    } catch (e: any) {
      setLlmGenerating(false);
      setLlmGenStatus(null);
      setLlmError(e.message);
    }
  };

  const handleValidateLlm = async () => {
    setLlmValidating(true);
    try {
      await validateScenarioVariables(scenarioId, {});
      loadLlmVars();
      reloadModelState();   // garder le spec (et « Ajuster le modèle ») synchrone après validation
    } catch (e: any) {
      setLlmError(e.message);
    } finally {
      setLlmValidating(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
      setUploadResult(null);
      setUploadError(null);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setUploadResult(null);
      setUploadError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setUploadError(null);
    setUploadResult(null);
    try {
      // Branche les données sur le pipeline modèle : validation + (si suffisant)
      // entraînement automatique. Plus besoin de cliquer "Entraîner".
      const res = await uploadModelData(scenarioId, file);
      setUploadResult(res);
      setFile(null);
      // Rafraîchit l'état modèle/données ; si l'entraînement a démarré, on le suit.
      getModelDataset(scenarioId).then(setModelDataset).catch(() => {});
      if (res.training_started) {
        const poll = () => getModelTrainStatus(scenarioId).then(s => {
          if (!mountedRef.current) return;   // composant démonté → on arrête
          if (s.status === 'running') { trainPollRef.current = setTimeout(poll, 3000); }
          else { getModelRun(scenarioId).then(r => { if (mountedRef.current) setModelRun(r); }).catch(() => {}); }
        }).catch(() => {});
        trainPollRef.current = setTimeout(poll, 2000);
      }
    } catch (err: any) {
      setUploadError(err.message || t("scenarioDetail.variables.uploadError"));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">

      {/* Banniere LLM Variables - pretes a valider */}
      {llmVars && !llmVars.status && !llmVars._validated && llmVars.predictor_variables && llmVars.predictor_variables.length > 0 && (
        <div className="rounded-2xl border border-gold-500/30 bg-gold-500/8 px-4 py-3 flex items-start gap-3">
          <Bell size={14} className="text-gold-400 shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-gold-300 mb-1">
              {t("scenarioDetail.variables.llmBannerTitle")}
            </p>
            <p className="text-[10px] text-gold-200/70 leading-relaxed">
              {llmVars.predictor_variables?.length ?? 0} {t("scenarioDetail.variables.llmBannerBodyPart1")} {llmVars._meta?.pico_articles_used ?? '?'} {t("scenarioDetail.variables.llmBannerBodyPart2")}{llmVars._meta?.relevant_total != null ? `${t("scenarioDetail.variables.llmBannerBodyRelevantPrefix")}${llmVars._meta.relevant_total}${t("scenarioDetail.variables.llmBannerBodyRelevantSuffix")}` : ''}{llmVars._meta?.corpus_total != null ? `, ${llmVars._meta.corpus_total}${t("scenarioDetail.variables.llmBannerBodyCorpusSuffix")}` : (llmVars._meta?.relevant_total != null ? ')' : '')}{t("scenarioDetail.variables.llmBannerBodyClose")}
            </p>
          </div>
          <button onClick={handleValidateLlm} disabled={llmValidating}
            className="shrink-0 flex items-center gap-1.5 rounded-xl bg-gold-500/20 hover:bg-gold-500/30 border border-gold-500/30 text-gold-300 font-semibold px-3 py-1.5 text-xs transition disabled:opacity-50">
            {llmValidating ? (<Loader2 size={11} className="animate-spin" />) : (<CheckCircle2 size={11} />)}
            {t("scenarioDetail.variables.validate")}
          </button>
        </div>
      )}

      {/* Banniere : variables validees */}
      {llmVars && llmVars._validated && (
        <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 px-4 py-2.5 flex items-center gap-2">
          <CheckCircle2 size={12} className="text-brand-400 shrink-0" />
          <p className="text-[10px] text-brand-300">
            {t("scenarioDetail.variables.validatedBannerPrefix")} {llmVars._generated_at ? new Date(llmVars._generated_at).toLocaleDateString(lang === "fr" ? "fr-FR" : "en-US") : ''}.
          </p>
        </div>
      )}

      {/* Etat vide : pas d'articles disponibles */}
      {llmVars && llmVars.status === 'empty' && (
        <div className="rounded-2xl border border-slate-500/20 bg-slate-500/5 px-4 py-3 flex items-start gap-3">
          <div className="text-xs text-slate-300/80">
            <strong className="text-slate-200">{t("scenarioDetail.variables.emptyTitle")}</strong> : {(llmVars as any).message ?? t("scenarioDetail.variables.emptyDefaultMsg")}
          </div>
        </div>
      )}

      {/* Bouton generer + erreur */}
      <div className="flex items-center gap-3 flex-wrap">
        <button onClick={handleGenerateLlm} disabled={llmGenerating || llmLoading}
          className="flex items-center gap-1.5 rounded-xl border border-brand-500/30 bg-brand-500/10 hover:bg-brand-500/20 text-brand-300 font-medium px-3 py-1.5 text-xs transition disabled:opacity-50">
          {llmGenerating ? (<Loader2 size={11} className="animate-spin" />) : (<Brain size={11} />)}
          {t("scenarioDetail.variables.generateFromPico")}
        </button>
        {llmGenStatus && (
          <span className="text-[10px] text-gold-400 flex items-center gap-1">
            <Loader2 size={10} className="animate-spin" />{llmGenStatus}
          </span>
        )}
        {llmError && <span className="text-[10px] text-rose-400">{llmError}</span>}
      </div>

      {/* Variables LLM generees */}
      {llmVars && !llmVars.status && llmVars.predictor_variables && llmVars.predictor_variables.length > 0 && (
        <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-4">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <SectionHeader
              icon={<Brain size={14} className="text-brand-400" />}
              title={t("scenarioDetail.variables.modelVariablesTitle")}
              subtitle={`${t("scenarioDetail.variables.modelVariablesSubtitlePart1")} ${llmVars._meta?.pico_articles_used ?? '?'} ${t("scenarioDetail.variables.modelVariablesSubtitlePart2")}${llmVars._meta?.relevant_total != null ? ` · ${llmVars._meta.relevant_total}${t("scenarioDetail.variables.modelVariablesSubtitleRelevant")}` : ''}${llmVars._meta?.corpus_total != null ? `${t("scenarioDetail.variables.modelVariablesSubtitleCorpusPrefix")}${llmVars._meta.corpus_total}${t("scenarioDetail.variables.modelVariablesSubtitleCorpusSuffix")}` : ''}`}
            />
            <div className="flex gap-2">
              {llmVars._validated ? (
                <span className="rounded-full bg-brand-500/10 border border-brand-500/20 px-2.5 py-1 text-[10px] font-semibold text-brand-300">{t("scenarioDetail.variables.validated")}</span>
              ) : (
                <span className="rounded-full bg-gold-500/10 border border-gold-500/20 px-2.5 py-1 text-[10px] font-semibold text-gold-300">{t("scenarioDetail.variables.pendingValidation")}</span>
              )}
            </div>
          </div>

          {/* Lien opérationnel avec le Modèle Prédictif */}
          <div className="rounded-2xl border border-white/10 bg-white/2 px-4 py-2.5 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px]">
            <span className="flex items-center gap-1.5 text-white/60">
              <span className={`h-1.5 w-1.5 rounded-full ${llmVars._validated ? 'bg-brand-400' : 'bg-gold-400'}`} />
              {llmVars._validated ? t("scenarioDetail.variables.specValidated") : t("scenarioDetail.variables.specToValidate")}
            </span>
            <span className="text-white/20">·</span>
            <span className="flex items-center gap-1.5 text-white/60">
              <Database size={11} />
              {modelDataset?.status === 'ready' ? `${modelDataset.n_rows?.toLocaleString() ?? '?'} ${t("scenarioDetail.variables.rowsPlugged")}` : t("scenarioDetail.variables.noDataPlugged")}
            </span>
            <span className="text-white/20">·</span>
            <span className="flex items-center gap-1.5 text-white/60">
              <Brain size={11} />
              {trained ? `${t("scenarioDetail.variables.modelPrefix")} ${modelRun?.family}${trainedMetricValue != null ? ` · ${modelRun?.metric} ${trainedMetricValue.toFixed(3)}` : ''}` : t("scenarioDetail.variables.modelNotTrained")}
            </span>
            {onGoToModel && (
              <button onClick={onGoToModel}
                className="ml-auto flex items-center gap-1 rounded-lg border border-brand-500/30 bg-brand-500/10 text-brand-300 px-2.5 py-1 font-semibold hover:bg-brand-500/20 transition">
                {t("scenarioDetail.common.openPredictiveModel")}
              </button>
            )}
          </div>

          {/* Outcome principal LLM */}
          {llmVars.primary_outcome && (
            <div className="rounded-2xl border border-gold-500/10 bg-gold-500/5 p-4 space-y-1">
              <p className="text-[10px] font-bold uppercase tracking-wider text-gold-400">{t("scenarioDetail.variables.primaryOutcome")}</p>
              <p className="text-sm font-medium text-gold-200">{llmVars.primary_outcome.name}</p>
              <p className="text-xs text-white/55">{llmVars.primary_outcome.definition}</p>
              <p className="text-[10px] text-white/35">{t("scenarioDetail.variables.measurement")} {llmVars.primary_outcome.measurement} - {t("scenarioDetail.variables.timeframe")} {llmVars.primary_outcome.timeframe}{llmVars.primary_outcome.unit ? ` · ${t("scenarioDetail.variables.unit")} ${llmVars.primary_outcome.unit}` : ''}</p>
              <ArticleSourceLink a={modelSpec?.outcome?.best_article} />
            </div>
          )}

          {/* Seuils d'interprétation de l'outcome (issus des évidences) */}
          {llmVars.alert_thresholds && (
            <div className="rounded-2xl border border-white/10 bg-white/3 p-4 space-y-2.5">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/50">
                {t("scenarioDetail.variables.outcomeThresholdsTitle")} <span className="text-white/30 normal-case font-normal">{t("scenarioDetail.variables.outcomeThresholdsSubtitle")}</span>
              </p>
              {(["green", "orange", "red"] as const).map((lvl) => {
                const band0 = llmVars.alert_thresholds?.[lvl];
                if (!band0) return null;
                const cfg = {
                  green:  { dot: "bg-emerald-400", cls: "text-emerald-300", def: t("scenarioDetail.variables.thresholdNormal") },
                  orange: { dot: "bg-amber-400",   cls: "text-amber-300",   def: t("scenarioDetail.variables.thresholdTension") },
                  red:    { dot: "bg-rose-400",     cls: "text-rose-300",    def: t("scenarioDetail.variables.thresholdAlert") },
                }[lvl];
                // Articles sources de la modalité (pool pertinent), résolus côté serveur.
                const band = modelSpec?.alert_thresholds?.[lvl];
                const srcArts = band?.provenance_articles?.length
                  ? band.provenance_articles
                  : (band?.best_article ? [band.best_article] : []);
                return (
                  <div key={lvl} className="flex items-start gap-2.5">
                    <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${cfg.dot}`} />
                    <div className="min-w-0">
                      <p className="text-xs">
                        <span className={`font-semibold ${cfg.cls}`}>{band0.label || cfg.def}</span>
                        {band0.range && <span className="ml-2 font-mono text-white/70">{band0.range}</span>}
                      </p>
                      {(band0.rationale || band0.description) && (
                        <p className="text-[10px] text-white/40 leading-snug">{band0.rationale || band0.description}</p>
                      )}
                      {srcArts.slice(0, 3).map((a, i) => <ArticleSourceLink key={i} a={a} label={t("scenarioDetail.common.ref")} />)}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left">
              <thead>
                <tr className="border-b border-white/5 text-[10px] text-white/50 uppercase tracking-wider">
                  <th className="py-2.5 px-3">{t("scenarioDetail.variables.tableVariable")}</th>
                  <th className="py-2.5 px-3">{t("scenarioDetail.variables.tableType")}</th>
                  <th className="py-2.5 px-3">{t("scenarioDetail.variables.tableDefinition")}</th>
                  <th className="py-2.5 px-3">{t("scenarioDetail.variables.tableSource")}</th>
                  <th className="py-2.5 px-3 text-center">{t("scenarioDetail.variables.tableAvailability")}</th>
                  <th className="py-2.5 px-3 text-center">{t("scenarioDetail.variables.tableImportance")}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 text-xs">
                {llmVars.predictor_variables?.map((v, i) => (
                  <tr key={i} className="hover:bg-white/1">
                    <td className="py-3 px-3">
                      <p className="font-mono text-brand-300 font-medium">{v.name}</p>
                      {v.machine_name && <p className="text-[10px] text-white/35 font-mono mt-0.5">{t("scenarioDetail.variables.columnPrefix")} {v.machine_name}</p>}
                      {v.machine_name && <ArticleSourceLink a={sourceByVar[v.machine_name]} />}
                    </td>
                    <td className="py-3 px-3">
                      <span className="rounded bg-white/5 border border-white/10 px-1.5 py-0.5 text-[10px] text-white/50">{v.type}</span>
                    </td>
                    <td className="py-3 px-3 text-white/70 leading-5 max-w-[200px]">{v.definition}</td>
                    <td className="py-3 px-3 text-white/50 font-mono text-[11px]">
                      {(v.source === 'seir' || v._seir_role === 'derived')
                        ? t("scenarioDetail.variables.seirSourceLabel")
                        : v._seir_role === 'parameter'
                        ? t("scenarioDetail.variables.seirParamSourceLabel")
                        : v.data_source}
                    </td>
                    <td className="py-3 px-3 text-center">
                      {(() => {
                        // Le sous-modèle SEIR prend le pas : une variable DÉRIVÉE est
                        // remplie automatiquement ; un PARAMÈTRE (R0/CFR…) n'est pas une
                        // feature du prédicteur (il alimente le SEIR).
                        if (v._seir_role === 'parameter') {
                          return (
                            <span title={t("scenarioDetail.variables.tipSeirParam")} className="inline-flex items-center gap-1.5 text-[10px] font-semibold text-white/40">
                              <span className="h-1.5 w-1.5 rounded-full bg-white/30" />{t("scenarioDetail.variables.statusSeirParam")}
                            </span>
                          );
                        }
                        if (v._seir_role === 'derived' || v.source === 'seir') {
                          return (
                            <span title={t("scenarioDetail.variables.tipSeirDerived")} className="inline-flex items-center gap-1.5 text-[10px] font-semibold text-violet-300">
                              <span className="h-1.5 w-1.5 rounded-full bg-violet-400" />{t("scenarioDetail.variables.statusSeirDerived")}
                            </span>
                          );
                        }
                        // Statut clair (pastille + libellé + couleur, jamais la couleur
                        // seule — accessibilité). Sans jeu de données chargé, on déduit
                        // la disponibilité de la source déclarée de la variable.
                        let st = dataStatusFor(v.machine_name);
                        if (st === 'unknown') {
                          const auto = !!(v as any).public_provider
                            || (v as any).source === 'public_api'
                            || /public|api|m[ée]t[ée]o|open[- ]?data|open-?meteo/i.test(v.data_source ?? '');
                          st = auto ? 'public' : 'missing_user';
                        }
                        const cfg = ({
                          plugged:      { dot: 'bg-emerald-400', cls: 'text-emerald-300', label: t("scenarioDetail.variables.statusPlugged") },
                          public:       { dot: 'bg-sky-400',     cls: 'text-sky-300',     label: t("scenarioDetail.variables.statusPublic") },
                          missing_user: { dot: 'bg-amber-400',   cls: 'text-amber-300',   label: t("scenarioDetail.variables.statusMissingUser") },
                        } as const)[st as 'plugged' | 'public' | 'missing_user'];
                        const tip = st === 'plugged' ? t("scenarioDetail.variables.tipPlugged")
                          : st === 'public' ? t("scenarioDetail.variables.tipPublic")
                          : t("scenarioDetail.variables.tipMissingUser");
                        return (
                          <span title={`${v.machine_name ?? ''} — ${tip}`} className={`inline-flex items-center gap-1.5 text-[10px] font-semibold ${cfg.cls}`}>
                            <span className={`h-1.5 w-1.5 rounded-full ${cfg.dot}`} />
                            {cfg.label}
                          </span>
                        );
                      })()}
                    </td>
                    <td className="py-3 px-3 text-center">
                      <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                        v.importance === 'high' ? 'bg-brand-500/15 text-brand-300' :
                        v.importance === 'medium' ? 'bg-gold-500/15 text-gold-300' :
                        'bg-white/5 text-white/40'
                      }`}>{v.importance}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Algorithme recommande */}
          {llmVars.recommended_algorithm && (
            <div className="rounded-2xl border border-white/8 bg-white/3 p-4 space-y-2">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">{t("scenarioDetail.variables.recommendedAlgorithm")}</p>
                {trained && (
                  <span className="rounded-full bg-brand-500/15 text-brand-300 px-2 py-0.5 text-[10px] font-semibold">
                    {t("scenarioDetail.variables.trainedPrefix")} {modelRun?.family}{trainedMetricValue != null ? ` · ${modelRun?.metric} ${trainedMetricValue.toFixed(3)}` : ''}
                  </span>
                )}
              </div>
              <p className="text-sm font-semibold text-white">{llmVars.recommended_algorithm.primary}</p>
              <p className="text-xs text-white/55">{llmVars.recommended_algorithm.rationale}</p>
              {llmVars.recommended_algorithm.alternatives?.length > 0 && (
                <p className="text-[10px] text-white/35">{t("scenarioDetail.variables.alternatives")} {llmVars.recommended_algorithm.alternatives.join(', ')}</p>
              )}
              <ArticleSourceLink a={modelSpec?.algorithm?.best_article} />
            </div>
          )}

          {/* Éditeur de spec : choisir l'algorithme (candidats de l'évidence),
              le type de tâche, ajouter/supprimer des variables, puis ré-entraîner. */}
          {modelSpec?.status === "ready" && (
            <SpecEditor scenarioId={scenarioId} spec={modelSpec} onApplied={reloadModelState} />
          )}
        </div>
      )}

    {/* L'outcome et les variables (générés par LLM) sont présentés plus haut dans
        cet onglet ; les anciens encadrés (vides pour les scénarios utilisateur) ont
        été retirés. Restent les bases de données requises et l'import de données. */}
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Sources de données requises par le MODÈLE (et non la source de la
            littérature). On préfère required_databases du spec généré ; à défaut
            on liste detail.databases. */}
        <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-4">
          <SectionHeader
            icon={<Globe size={14} className="text-brand-400" />}
            title={t("scenarioDetail.variables.requiredSourcesTitle")}
            subtitle={t("scenarioDetail.variables.requiredSourcesSubtitle")}
          />
          {(() => {
            const feeds = ((llmVars as any)?.required_databases as string[] | undefined)
              ?? detail.databases ?? [];
            return feeds.length > 0 ? (
              <div className="space-y-2">
                {feeds.map((db, i) => (
                  <div key={i} className="flex items-center gap-2.5 rounded-xl border border-white/5 bg-white/3 px-3 py-2.5 text-xs text-white/70">
                    <Database size={12} className="text-brand-400 shrink-0" />
                    <span>{db}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-white/35 italic">{t("scenarioDetail.variables.requiredSourcesEmpty")}</p>
            );
          })()}
        </div>

        {/* Auto-remplissage des variables publiques (Phase 2) */}
        <AutoFetchPanel
          scenarioId={scenarioId}
          spec={modelSpec}
          onFetched={() => { getModelDataset(scenarioId).then(setModelDataset).catch(() => {}); }}
        />

        {/* Zone d'upload interactif */}
        <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-4">
          <SectionHeader
            icon={<Upload size={14} className="text-brand-400" />}
            title={t("scenarioDetail.variables.importTitle")}
            subtitle={t("scenarioDetail.variables.importSubtitle")}
          />

          <div
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-white/10 hover:border-brand-500/30 bg-white/2 hover:bg-white/5 rounded-2xl p-6 text-center cursor-pointer transition flex flex-col items-center gap-2"
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".csv,.xlsx,.xls"
              className="hidden"
            />
            <Upload size={24} className="text-white/35" />
            <p className="text-xs text-white/70 font-medium">{t("scenarioDetail.variables.dropHere")}</p>
            <p className="text-[10px] text-white/35">{t("scenarioDetail.variables.acceptedFormats")}</p>
            {file && (
              <div className="mt-2 rounded-lg bg-brand-500/10 border border-brand-500/20 px-2.5 py-1 text-[11px] text-brand-300 font-mono">
                {file.name} ({(file.size / 1024).toFixed(1)} KB)
              </div>
            )}
          </div>

          {file && (
            <button
              onClick={handleUpload}
              disabled={uploading}
              className="w-full flex items-center justify-center gap-1.5 rounded-xl bg-brand-500 hover:bg-brand-400 text-forest-950 font-semibold py-2 text-xs transition disabled:opacity-50"
            >
              <RotateCcw size={12} className={uploading ? "animate-spin" : ""} />
              {uploading ? t("scenarioDetail.variables.importing") : t("scenarioDetail.variables.launchImport")}
            </button>
          )}

          {uploadError && <ErrorBox message={uploadError} />}

          {uploadResult && (
            <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 p-4 space-y-3">
              <div className="flex items-center gap-1.5 text-brand-300 text-xs font-semibold">
                <CheckCircle2 size={14} /> {t("scenarioDetail.variables.dataPlugged")}
              </div>
              <div className="rounded-lg bg-forest-900/50 p-2 text-[10px] font-mono text-white/50 space-y-1">
                <div>{t("scenarioDetail.variables.rows")} <span className="text-brand-300">{uploadResult.n_rows}</span> · {t("scenarioDetail.variables.columns")} <span className="text-brand-300">{uploadResult.n_cols}</span></div>
                {uploadResult.validation?.matched_features && (
                  <div>{t("scenarioDetail.variables.recognizedVariables")} <span className="text-brand-300">{uploadResult.validation.matched_features.length}</span></div>
                )}
                {(uploadResult.validation?.missing_user?.length ?? 0) > 0 && (
                  <div className="text-gold-300">{t("scenarioDetail.variables.toProvide")} {uploadResult.validation!.missing_user!.join(", ")}</div>
                )}
              </div>
              {uploadResult.training_started ? (
                <div className="flex items-start gap-1.5 text-[11px] text-brand-300">
                  <Loader2 size={12} className="animate-spin shrink-0 mt-0.5" />
                  <span>{t("scenarioDetail.variables.trainingStartedPrefix")} <strong>{t("scenarioDetail.variables.trainingStartedTab")}</strong>.</span>
                </div>
              ) : (
                <div className="flex items-start gap-1 text-[10px] text-white/45">
                  <Info size={10} className="shrink-0 mt-0.5" />
                  <span>{uploadResult.validation?.readiness?.reasons?.[0] ?? t("scenarioDetail.variables.insufficientData")}</span>
                </div>
              )}
              {onGoToModel && (
                <button onClick={onGoToModel} className="text-[10px] font-semibold text-brand-300 hover:text-brand-200">
                  {t("scenarioDetail.common.openPredictiveModel")}
                </button>
              )}
            </div>
          )}
        </div>
    </div>
    </div>
  );
}

// ─── Section: Corpus ──────────────────────────────────────────────────────────

function CorpusSection({ scenarioId, threshold }: { scenarioId: string; detail: ScenarioDetail; threshold?: number }) {
  const { t } = useI18n();
  const [data, setData] = useState<ScenarioCorpus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [embeddingStatus, setEmbeddingStatus] = useState<EmbeddingStatus | null>(null);

  // Recharge le corpus quand le seuil (curseur) change, avec un léger debounce.
  useEffect(() => {
    if (threshold == null) return;
    let cancelled = false;   // évite qu'une réponse lente d'un ancien seuil écrase un seuil plus récent
    const tid = setTimeout(() => {
      fetchScenarioCorpus(scenarioId, { threshold })
        .then(d => { if (!cancelled) setData(d); })
        .catch(() => {});
    }, 250);
    return () => { cancelled = true; clearTimeout(tid); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threshold]);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchScenarioCorpus(scenarioId)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
    // Charger le statut d'embedding si c'est un user-scenario
    fetchUserScenarioEmbeddingStatus(scenarioId)
      .then(setEmbeddingStatus)
      .catch(() => {}); // Silencieux si pas un user-scenario
  }, [scenarioId]);

  // Tant qu'un scoring (rerank Cohere) tourne en arrière-plan, on rafraîchit le
  // corpus toutes les 4 s pour que les badges de pertinence (⊕ Cohere) et le
  // compteur "auto-sélectionnés" apparaissent en direct, sans refresh manuel.
  useEffect(() => {
    if (!data?.rerank_running) return;
    const id = setTimeout(() => {
      fetchScenarioCorpus(scenarioId, threshold != null ? { threshold } : undefined)
        .then(setData)
        .catch(() => {});
    }, 4000);
    return () => clearTimeout(id);
  }, [data, scenarioId, threshold]);

  if (loading) return <LoadingSpinner text={t("scenarioDetail.corpus.loadingCorpus")} />;
  if (error || !data) return <ErrorBox message={error ?? t("scenarioDetail.common.errorCorpus")} />;

  return (
    <div className="space-y-4">
      {/* Bannière avertissement sélection automatique */}
      <div className="rounded-2xl border border-gold-500/20 bg-gold-500/5 px-4 py-3 flex items-start gap-3">
        <AlertTriangle size={14} className="text-gold-400 shrink-0 mt-0.5" />
        <div className="text-xs text-gold-200/80 leading-relaxed">
          <strong className="text-gold-300">{t("scenarioDetail.corpus.autoSelectionTitle")}</strong>{t("scenarioDetail.corpus.autoSelectionBody1")} <strong>{t("scenarioDetail.corpus.autoSelectionBody2")}</strong> {t("scenarioDetail.corpus.autoSelectionBody3")}
        </div>
      </div>

      {/* Scores de pertinence = le classement AFFICHÉ (calculé en ligne pendant le
          scoring). L'indexation RAG (Assistant) vit désormais dans l'onglet
          « Assistant IA », pas ici : elle n'affecte pas ce classement. */}
      {embeddingStatus && (() => {
        const total = embeddingStatus.corpus_total ?? embeddingStatus.ranking?.total ?? 0;
        const scored = embeddingStatus.ranking?.scored ?? 0;
        const semanticReady = embeddingStatus.score_availability.semantic;
        const cohereReady = embeddingStatus.score_availability.cohere;
        const cohereConfigured = embeddingStatus.score_availability.cohere_configured ?? false;
        // Voyant tri-état : vert (prêt) / ambre (en cours) / gris (indisponible).
        type S = 'on' | 'pending' | 'off';
        const dot = (s: S) => s === 'on' ? 'bg-brand-400' : s === 'pending' ? 'bg-gold-400' : 'bg-white/20';
        const txt = (s: S) => s === 'on' ? 'text-forest-400' : s === 'pending' ? 'text-gold-300' : 'text-white/25';
        const semanticState: S = semanticReady ? 'on' : (scored > 0 ? 'pending' : 'off');
        const cohereState: S = cohereReady ? 'on' : (cohereConfigured ? 'pending' : 'off');
        return (
          <div className="rounded-2xl border border-white/10 bg-white/3 px-4 py-3">
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                {semanticReady ? <CheckCircle2 size={14} className="text-forest-400" />
                  : <Loader2 size={14} className="text-gold-400 animate-spin" />}
                <span className={`text-xs font-semibold ${semanticReady ? 'text-forest-300' : 'text-gold-300'}`}>
                  {t("scenarioDetail.corpus.relevanceScores")}
                </span>
              </div>
              <p className="text-[11px] text-white/70">
                <span className="text-white font-semibold">{scored}</span> / {total} {t("scenarioDetail.corpus.articlesScoredSuffix")}
                {!semanticReady && scored < total && ` · ${total - scored} ${t("scenarioDetail.corpus.inProgressSuffix")}`}
              </p>
              <div className="flex items-center gap-3 flex-wrap">
                <span title={t("scenarioDetail.corpus.semanticTooltip")}
                  className={`text-[10px] flex items-center gap-1 ${txt(semanticState)}`}>
                  <span className={`h-1.5 w-1.5 rounded-full ${dot(semanticState)}`} />
                  {t("scenarioDetail.corpus.semantic")}{semanticState === 'pending' ? t("scenarioDetail.corpus.semanticInProgress") : ''}
                </span>
                <span title={t("scenarioDetail.corpus.cohereTooltip")}
                  className={`text-[10px] flex items-center gap-1 ${txt(cohereState)}`}>
                  <span className={`h-1.5 w-1.5 rounded-full ${dot(cohereState)}`} />
                  {t("scenarioDetail.corpus.cohere")}{cohereState === 'pending' ? t("scenarioDetail.corpus.cohereWaiting") : cohereState === 'off' ? t("scenarioDetail.corpus.cohereNotConfigured") : ''}
                </span>
              </div>
            </div>
          </div>
        );
      })()}
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      {/* Liste des articles */}
      <div className="lg:col-span-2 space-y-4">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <SectionHeader
            icon={<FileText size={14} className="text-brand-400" />}
            title={`${t("scenarioDetail.corpus.corpusTitlePrefix")} (${data.total} ${t("scenarioDetail.corpus.corpusTitleArticles")})`}
            subtitle={t("scenarioDetail.corpus.corpusSubtitle")}
          />
          {data.above_threshold !== undefined && (
            <div className="flex items-center gap-2 flex-wrap">
              <span className="rounded-full bg-brand-500/15 border border-brand-500/30 px-3 py-1 text-[10px] font-semibold text-brand-300">
                {data.above_threshold} {t("scenarioDetail.corpus.aboveThreshold")}
              </span>
              {(() => {
                const below = data.below_threshold ?? Math.max(0, data.total - data.above_threshold! - (data.unscored ?? 0));
                return below > 0 ? (
                  <span className="rounded-full bg-white/5 border border-white/10 px-3 py-1 text-[10px] text-white/40">
                    {below} {t("scenarioDetail.corpus.belowThresholdKept")}
                  </span>
                ) : null;
              })()}
              {(data.unscored ?? 0) > 0 && (
                <span className="rounded-full bg-gold-500/10 border border-gold-500/30 px-3 py-1 text-[10px] text-gold-300 flex items-center gap-1">
                  {data.rerank_running && <Loader2 size={9} className="animate-spin" />}
                  {data.unscored} {t("scenarioDetail.corpus.unscored")}{data.rerank_running ? t("scenarioDetail.corpus.scoringInProgress") : ''}
                </span>
              )}
            </div>
          )}
        </div>
        {typeof data.from_local === "number" && (
          <p className="text-[11px] text-white/40"
             title={t("scenarioDetail.corpus.fromLocalTooltip")}>
            {data.from_local.toLocaleString()} {t("scenarioDetail.corpus.alreadyInLocal")} · {(data.newly_fetched ?? 0).toLocaleString()} {t("scenarioDetail.corpus.fetchedForScenario")}
          </p>
        )}
        <div className="space-y-3">
          {data.articles.length > 0 ? data.articles.map((article) => (
            <ArticleRow
              key={article.id}
              article={article}
              scenarioId={scenarioId}
              isExpanded={expandedId === article.id}
              onToggle={() => setExpandedId(expandedId === article.id ? null : article.id)}
              onScreeningChange={(_id, _status) => {
                // Rafraîchir les stats PRISMA si nécessaire
              }}
            />
          )) : (
            <p className="text-xs text-white/35 italic">{t("scenarioDetail.corpus.noArticles")}</p>
          )}
        </div>
      </div>

      {/* Distribution et stats */}
      <div className="space-y-6">
        {/* Années */}
        <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-4">
          <SectionHeader
            icon={<Globe size={14} className="text-brand-400" />}
            title={t("scenarioDetail.corpus.distributionByYear")}
          />
          <div className="space-y-2 text-xs">
            {data.year_distribution.map((item) => (
              <div key={item.year} className="flex items-center gap-3">
                <span className="w-10 text-white/50 font-mono">{item.year}</span>
                <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-brand-500 rounded-full"
                    style={{ width: `${(item.count / data.total) * 100}%` }}
                  />
                </div>
                <span className="w-6 text-right text-white/70 font-mono">{item.count}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Sources */}
        <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-4">
          <SectionHeader
            icon={<Database size={14} className="text-brand-400" />}
            title={t("scenarioDetail.corpus.literatureSources")}
          />
          <div className="space-y-2 text-xs">
            {data.source_distribution.map((item) => (
              <div key={item.source} className="flex items-center gap-3">
                <span className="w-20 text-white/50 uppercase font-mono tracking-wider">{item.source}</span>
                <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-brand-500 rounded-full"
                    style={{ width: `${(item.count / data.total) * 100}%` }}
                  />
                </div>
                <span className="w-6 text-right text-white/70 font-mono">{item.count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
    </div>
  );
}

function ArticleRow({
  article,
  scenarioId,
  isExpanded,
  onToggle,
  onScreeningChange,
}: {
  article: CorpusArticle;
  scenarioId: string;
  isExpanded: boolean;
  onToggle: () => void;
  onScreeningChange?: (id: number, status: string) => void;
}) {
  const { t } = useI18n();
  const [screeningStatus, setScreeningStatus] = React.useState<string>(article.screening_status ?? 'pending');
  const [screeningLoading, setScreeningLoading] = React.useState(false);
  const [exclusionReason, setExclusionReason] = React.useState('');
  const [screeningNotes, setScreeningNotes] = React.useState('');
  const [pico, setPico] = React.useState<PicoData | null>(null);
  const [picoLoading, setPicoLoading] = React.useState(false);
  const [picoLoaded, setPicoLoaded] = React.useState(false);

  const handleScreen = async (status: 'included' | 'excluded' | 'pending') => {
    if (status === 'excluded' && !exclusionReason) return;
    setScreeningLoading(true);
    try {
      await screenArticle(scenarioId, article.id, status, exclusionReason || undefined, screeningNotes || undefined);
      setScreeningStatus(status);
      onScreeningChange?.(article.id, status);
    } catch (e) {
      console.error('Screening error:', e);
    } finally {
      setScreeningLoading(false);
    }
  };

  const loadPico = async () => {
    if (picoLoaded) return;
    setPicoLoading(true);
    try {
      const res = await fetchArticlePico(scenarioId, article.id);
      setPico(res.pico);
      setPicoLoaded(true);
    } catch (e) {
      console.error('PICO error:', e);
    } finally {
      setPicoLoading(false);
    }
  };

  React.useEffect(() => {
    if (isExpanded && !picoLoaded) loadPico();
  }, [isExpanded]);

  const statusBadge = {
    included: 'bg-brand-500/20 text-brand-300 border border-brand-500/30',
    excluded: 'bg-rose-500/20 text-rose-300 border border-rose-500/30',
    pending: 'bg-gold-500/10 text-gold-400 border border-gold-500/20',
  }[screeningStatus] ?? 'bg-white/5 text-white/50 border border-white/10';

  const statusLabel = { included: t("scenarioDetail.articleRow.statusIncluded"), excluded: t("scenarioDetail.articleRow.statusExcluded"), pending: t("scenarioDetail.articleRow.statusPending") }[screeningStatus] ?? screeningStatus;

  return (
    <div className={`rounded-2xl border transition overflow-hidden ${
      screeningStatus === 'included' ? 'border-brand-500/20 bg-brand-500/3' :
      screeningStatus === 'excluded' ? 'border-rose-500/15 bg-rose-500/3 opacity-60' :
      'border-white/5 bg-white/2 hover:bg-white/3'
    }`}>
      <div onClick={onToggle} className="p-4 flex items-start gap-3 cursor-pointer">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="rounded bg-white/5 px-1.5 py-0.5 text-[10px] font-mono text-white/50 uppercase tracking-wider">
              {article.source}
            </span>
            {article.year && (
              <span className="text-[10px] font-mono text-white/35">{article.year}</span>
            )}
            {article.has_fulltext && (
              <span className="rounded-full bg-brand-500/10 border border-brand-500/20 px-2 py-0.5 text-[9px] text-brand-300 font-medium">
                {t("scenarioDetail.articleRow.fulltext")}
              </span>
            )}
            {/* Score de PERTINENCE (Cohere) = celui qui détermine le classement,
                affiché en premier pour que l'ordre soit lisible. */}
            {article.rerank_score !== undefined && article.rerank_score !== null && (
              <span className="rounded-full px-2 py-0.5 text-[9px] font-medium bg-violet-500/15 border border-violet-500/30 text-violet-300"
                title={t("scenarioDetail.articleRow.rerankTooltip")}>
                ⊕ {article.rerank_score.toFixed(3)}
              </span>
            )}
            {article.similarity_score !== undefined && article.similarity_score !== null && (
              <span className={`rounded-full px-2 py-0.5 text-[9px] font-medium ${
                article.similarity_score >= 0.45
                  ? 'bg-brand-500/15 border border-brand-500/30 text-brand-300'
                  : 'bg-white/5 border border-white/10 text-white/30'
              }`}
              title={t("scenarioDetail.articleRow.similarityTooltip")}>
                ◎ {article.similarity_score.toFixed(3)}
              </span>
            )}
            <span className={`rounded-full px-2 py-0.5 text-[9px] font-medium ${statusBadge}`}>
              {statusLabel}
            </span>
          </div>
          <h4 className="text-sm font-semibold text-white mt-1.5 leading-5">{article.title}</h4>
          {article.authors && (
            <p className="text-xs text-white/35 mt-1 truncate">{article.authors}</p>
          )}
        </div>
        <button className="text-white/35 hover:text-white shrink-0 mt-1">
          {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>

      {isExpanded && (
        <div className="border-t border-white/5 bg-white/1 p-4 text-xs space-y-4">
          {/* Screening PRISMA : Interface avancée */}
          <div className="rounded-xl border border-white/8 bg-white/3 p-3 space-y-3">
            <h5 className="text-[10px] font-semibold text-white/50 uppercase tracking-wider flex items-center gap-1">
              <CheckCircle2 size={10} />{t("scenarioDetail.articleRow.screeningDecision")}
            </h5>
            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <label className="block text-[10px] text-white/40 mb-1">{t("scenarioDetail.articleRow.exclusionReasonLabel")}</label>
                <select
                  value={exclusionReason}
                  onChange={(e) => { e.stopPropagation(); setExclusionReason(e.target.value); }}
                  onClick={(e) => e.stopPropagation()}
                  className="w-full rounded-lg border border-white/10 bg-forest-950/80 px-2.5 py-1.5 text-xs text-white outline-none focus:border-brand-400"
                >
                  <option value="">{t("scenarioDetail.articleRow.selectReason")}</option>
                  <option value="wrong-population">{t("scenarioDetail.articleRow.reasonWrongPopulation")}</option>
                  <option value="wrong-intervention">{t("scenarioDetail.articleRow.reasonWrongIntervention")}</option>
                  <option value="wrong-outcome">{t("scenarioDetail.articleRow.reasonWrongOutcome")}</option>
                  <option value="no-fulltext">{t("scenarioDetail.articleRow.reasonNoFulltext")}</option>
                  <option value="duplicate">{t("scenarioDetail.articleRow.reasonDuplicate")}</option>
                  <option value="other">{t("scenarioDetail.articleRow.reasonOther")}</option>
                </select>
              </div>
              <div>
                <label className="block text-[10px] text-white/40 mb-1">{t("scenarioDetail.articleRow.screeningNotesLabel")}</label>
                <input
                  value={screeningNotes}
                  onChange={(e) => { e.stopPropagation(); setScreeningNotes(e.target.value); }}
                  onClick={(e) => e.stopPropagation()}
                  placeholder={t("scenarioDetail.articleRow.screeningNotesPlaceholder")}
                  className="w-full rounded-lg border border-white/10 bg-forest-950/80 px-2.5 py-1.5 text-xs text-white outline-none focus:border-brand-400"
                />
              </div>
            </div>
            {screeningLoading ? (
              <div className="flex items-center gap-2 text-white/40 text-xs">
                <Loader2 size={12} className="animate-spin" />{t("scenarioDetail.articleRow.saving")}
              </div>
            ) : (
              <div className="flex items-center gap-2 flex-wrap">
                <button
                  onClick={(e) => { e.stopPropagation(); handleScreen('excluded'); }}
                  disabled={!exclusionReason}
                  className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-semibold transition ${
                    screeningStatus === 'excluded'
                      ? 'border-rose-500/50 bg-rose-500/20 text-rose-200'
                      : 'border-rose-500/30 bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 disabled:opacity-40 disabled:cursor-not-allowed'
                  }`}
                >
                  <AlertCircle size={11} />{t("scenarioDetail.articleRow.excludeArticle")}
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); handleScreen('included'); }}
                  className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-semibold transition ${
                    screeningStatus === 'included'
                      ? 'border-brand-500/50 bg-brand-500/20 text-brand-200'
                      : 'border-brand-500/30 bg-brand-500/10 hover:bg-brand-500/20 text-brand-300'
                  }`}
                >
                  <CheckCircle2 size={11} />{t("scenarioDetail.articleRow.includeInFinalCorpus")}
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); handleScreen('pending'); }}
                  className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-semibold transition ${
                    screeningStatus === 'pending'
                      ? 'border-gold-500/30 bg-gold-500/15 text-gold-300'
                      : 'border-white/10 bg-white/5 hover:bg-white/10 text-white/50'
                  }`}
                >
                  {t("scenarioDetail.articleRow.pending")}
                </button>
              </div>
            )}
          </div>

          {/* PICO */}
          {picoLoading ? (
            <div className="flex items-center gap-2 text-white/35">
              <Loader2 size={12} className="animate-spin" />
              <span>{t("scenarioDetail.articleRow.loadingPico")}</span>
            </div>
          ) : pico ? (
            <div className="rounded-xl border border-white/5 bg-white/2 p-3 space-y-2">
              <p className="text-[10px] font-semibold text-white/50 uppercase tracking-wider flex items-center gap-1">
                <Microscope size={10} />PICO
                {pico.pico_confidence != null && (
                  <span className="ml-auto font-mono text-white/35">{t("scenarioDetail.articleRow.confidence")} {Math.round(pico.pico_confidence * 100)}%</span>
                )}
              </p>
              <div className="grid grid-cols-2 gap-2">
                {[['P', t("scenarioDetail.articleRow.populationLabel"), pico.P], ['I', t("scenarioDetail.articleRow.interventionLabel"), pico.I], ['C', t("scenarioDetail.articleRow.comparatorLabel"), pico.C], ['O', t("scenarioDetail.articleRow.outcomeLabel"), pico.O]].map(([key, label, val]) => val && (
                  <div key={key} className="rounded-lg bg-white/3 border border-white/5 p-2">
                    <span className="text-[9px] font-bold text-brand-400 uppercase">{key} : {label}</span>
                    <p className="text-white/70 mt-0.5 leading-4">{val as string}</p>
                  </div>
                ))}
              </div>
              {pico.study_design && (
                <p className="text-[10px] text-white/35">{t("scenarioDetail.articleRow.studyType")} <span className="text-white/70">{pico.study_design}</span></p>
              )}
            </div>
          ) : picoLoaded ? (
            <p className="text-[10px] text-white/25 italic">{t("scenarioDetail.articleRow.picoNotExtracted")}</p>
          ) : null}

          {article.abstract && (
            <div>
              <p className="font-semibold text-white/50 mb-1">{t("scenarioDetail.articleRow.abstract")}</p>
              <p className="text-white/70 leading-5">{article.abstract}</p>
            </div>
          )}
          <div className="flex items-center gap-4 flex-wrap text-white/50 font-mono text-[10px] pt-1">
            {article.journal && <span>{t("scenarioDetail.articleRow.journal")} <span className="text-white/70">{article.journal}</span></span>}
            {article.doi && (
              <span>
                DOI:{" "}
                <a
                  href={`https://doi.org/${article.doi}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-brand-400 hover:underline"
                >
                  {article.doi}
                </a>
              </span>
            )}
            {article.url && (
              <a
                href={article.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 text-brand-400 hover:underline"
              >
                {t("scenarioDetail.articleRow.directLink")} <ExternalLink size={10} />
              </a>
            )}
            {article.country && (
              <span className="text-[10px] text-white/35 border border-white/5 rounded px-1.5 py-0.5">
                <Globe size={9} className="inline mr-0.5" />{article.country}
              </span>
            )}
            {article.keywords && (
              <div className="flex flex-wrap gap-1">
                {article.keywords.split(",").slice(0, 5).map((kw, i) => (
                  <span key={i} className="text-[10px] text-white/35 bg-forest-800/50 px-1 rounded">
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

// ─── Section: Clustering UMAP & HDBSCAN (ENRICHI) ──────────────────────────────

function ClusteringSection({ scenarioId }: { scenarioId: string }) {
  const { t } = useI18n();
  const [data, setData] = useState<ScenarioClustering | null>(null);
  const [loading, setLoading] = useState(false);
  const [polling, setPolling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    setPolling(false);
  }, []);

  const handleResult = useCallback((result: ScenarioClustering) => {
    setData(result);
    if (result.clusters && result.clusters.length > 0) {
      const firstDense = result.clusters.find((c: any) => !c.is_noise);
      setSelectedCluster(firstDense ? firstDense.cluster_id : result.clusters[0].cluster_id);
    }
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    stopPolling();
    try {
      const result = await fetchScenarioClustering(scenarioId);
      // Si le backend répond "running", démarrer le polling
      if ((result as any).status === "running") {
        setLoading(false);
        setPolling(true);
        setData(result);
        pollRef.current = setInterval(async () => {
          try {
            const r = await safeFetch(`${scenarioBase(scenarioId)}/${scenarioId}/clustering/status`);
            if (!r.ok) return;
            const status = await r.json();
            if (status.status === "done" || (status.clusters && status.clusters.length > 0)) {
              stopPolling();
              handleResult(status);
            } else if (status.status === "error") {
              stopPolling();
              setError(status.error || t("scenarioDetail.clustering.errorClustering"));
            }
          } catch (_) {}
        }, 5000);
      } else {
        handleResult(result);
        setLoading(false);
      }
    } catch (e: any) {
      setError(e.message);
      setLoading(false);
    }
  }, [scenarioId, stopPolling, handleResult, t]);

  useEffect(() => {
    load();
    return () => stopPolling();
  }, [load, stopPolling]);

  const activeClusterData = data?.clusters.find((c) => c.cluster_id === selectedCluster);

  return (
    <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-5">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <SectionHeader
          icon={<Layers size={14} className="text-brand-400" />}
          title={t("scenarioDetail.clustering.title")}
          subtitle={t("scenarioDetail.clustering.subtitle")}
        />
        <button
          onClick={load}
          disabled={loading}
          className="flex items-center gap-1.5 rounded-xl border border-brand-500/20 bg-brand-500/10 px-3 py-1.5 text-xs text-brand-300 hover:bg-brand-500/20 transition disabled:opacity-50"
        >
          <RotateCcw size={11} className={loading ? "animate-spin" : ""} />
          {loading ? t("scenarioDetail.clustering.calculating") : t("scenarioDetail.clustering.recalculate")}
        </button>
      </div>

      {(loading || polling) && (
        <LoadingSpinner text={polling
          ? t("scenarioDetail.clustering.pollingText")
          : t("scenarioDetail.clustering.startingText")
        } />
      )}
      {error && <ErrorBox message={error} />}

      {data && !loading && (
        <>
          {data.message && (
            <div className="rounded-xl border border-gold-500/20 bg-gold-500/5 px-3 py-2.5 text-xs text-gold-300">
              <Info size={12} className="inline mr-1.5 shrink-0" />
              {data.message}
            </div>
          )}

          {data.clusters && data.clusters.length > 0 && (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
              {/* Carte UMAP 2D (SVG) */}
              <div className="lg:col-span-1 space-y-4">
                <div className="rounded-2xl border border-white/5 bg-white/2 p-4 flex flex-col gap-3">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-gold-400">{t("scenarioDetail.clustering.umapProjection")}</span>
                  <p className="text-[10px] text-white/40 leading-4">{t("scenarioDetail.clustering.umapCaption")}</p>
                  <div className="w-full bg-[#0a1410] rounded-xl border border-white/5 overflow-hidden">
                    <UmapScatterPlot clusters={data.clusters} selectedCluster={selectedCluster} onSelectCluster={setSelectedCluster}/>
                  </div>
                  <div className="flex items-center justify-between text-[10px] text-white/25 font-mono">
                    <span>← UMAP dim 1 →</span>
                    <span>{data.n_docs} {t("scenarioDetail.clustering.articles")} · {data.n_clusters} {t("scenarioDetail.clustering.articlesClustersSeparator")}</span>
                  </div>
                </div>

                {/* Sélecteur de cluster de gauche */}
                <div className="space-y-2">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-white/50 block px-1">{t("scenarioDetail.clustering.selectGroup")}</span>
                  <div className="space-y-1 max-h-64 overflow-y-auto pr-1">
                    {data.clusters.map((c) => (
                      <button
                        key={c.cluster_id}
                        onClick={() => setSelectedCluster(c.cluster_id)}
                        className={`w-full text-left rounded-xl px-3 py-2 text-xs transition flex items-center justify-between border ${
                          selectedCluster === c.cluster_id
                            ? "border-brand-500/30 bg-brand-500/10 text-brand-300"
                            : "border-transparent text-white/50 hover:text-white hover:bg-white/3"
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <span
                            className="h-2 w-2 rounded-full shrink-0"
                            style={{ backgroundColor: getClusterColor(c.cluster_id, c.is_noise) }}
                          />
                          <span className="font-medium truncate max-w-[150px]">{c.cluster_name}</span>
                        </div>
                        <span className="text-[10px] font-mono opacity-70 bg-white/5 rounded px-1.5 py-0.5">{c.n_docs} {t("scenarioDetail.clustering.articles")}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Détails du cluster sélectionné */}
              <div className="lg:col-span-2 space-y-6">
                {activeClusterData && (
                  <div className="rounded-3xl border border-white/5 bg-white/2 p-5 space-y-5">
                    {/* En-tête */}
                    <div className="flex items-center justify-between flex-wrap gap-2 border-b border-white/5 pb-4">
                      <div className="flex items-center gap-3">
                        <span
                          className="h-3.5 w-3.5 rounded-full border border-white/10"
                          style={{ backgroundColor: getClusterColor(activeClusterData.cluster_id, activeClusterData.is_noise) }}
                        />
                        <div>
                          <h4 className="text-sm font-bold text-white uppercase tracking-wider">{activeClusterData.cluster_name}</h4>
                          <p className="text-xs text-white/50 mt-0.5">{activeClusterData.n_docs} {t("scenarioDetail.clustering.denseArticlesInGroup")}</p>
                        </div>
                      </div>
                    </div>

                    {/* Résumé clinique LLM */}
                    <div className="space-y-2">
                      <div className="flex items-center gap-1.5 text-xs font-semibold text-brand-300 uppercase tracking-wider">
                        <Brain size={13} />
                        {t("scenarioDetail.clustering.groupSynthesis")}
                      </div>
                      <div className="rounded-2xl border border-brand-500/15 bg-brand-500/5 p-4 text-xs text-white/80 leading-6 italic">
                        "{activeClusterData.summary}"
                      </div>
                    </div>

                    {/* Mots-clés TF-IDF */}
                    <div className="space-y-2">
                      <p className="text-[10px] font-bold uppercase tracking-wider text-white/50">{t("scenarioDetail.clustering.topKeywords")}</p>
                      <div className="flex flex-wrap gap-1.5">
                        {activeClusterData.top_words.map((w, i) => (
                          <span
                            key={i}
                            className="rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-[10px] text-white/70 font-mono"
                          >
                            {w}
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Article représentatif (peut être absent pour un petit cluster) */}
                    {activeClusterData.representative_doc && (
                      <div className="space-y-2 border-t border-white/5 pt-4">
                        <p className="text-[10px] font-bold uppercase tracking-wider text-white/50">{t("scenarioDetail.clustering.representativeArticle")}</p>
                        <div className="rounded-xl border border-white/5 bg-white/3 p-3">
                          <div className="flex items-center gap-1.5 text-[10px] text-white/35 font-mono">
                            <span>ID: #{activeClusterData.representative_doc.id}</span>
                            {activeClusterData.representative_doc.year && <span>• {activeClusterData.representative_doc.year}</span>}
                            {activeClusterData.representative_doc.journal && <span>• {activeClusterData.representative_doc.journal}</span>}
                          </div>
                          <h5 className="text-xs font-semibold text-white mt-1.5 leading-5">{activeClusterData.representative_doc.title}</h5>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// Palette de couleurs pour les clusters
const CLUSTER_VIVID = [
  "#34d399","#38bdf8","#a78bfa","#fbbf24","#f472b6","#60a5fa","#2dd4bf","#c084fc","#fb7185",
];
function getClusterColor(clusterId: number, isNoise: boolean): string {
  if (isNoise) return "#94a3b8";
  return CLUSTER_VIVID[clusterId % CLUSTER_VIVID.length];
}
// Convex hull (gift wrapping)
function convexHull(pts: Array<{x:number;y:number}>): Array<{x:number;y:number}> {
  if (pts.length < 3) return pts;
  const sorted = [...pts].sort((a,b) => a.x-b.x||a.y-b.y);
  const cross = (o:{x:number;y:number},a:{x:number;y:number},b:{x:number;y:number}) =>
    (a.x-o.x)*(b.y-o.y)-(a.y-o.y)*(b.x-o.x);
  const lower: Array<{x:number;y:number}> = [];
  for (const p of sorted) {
    while (lower.length>=2 && cross(lower[lower.length-2],lower[lower.length-1],p)<=0) lower.pop();
    lower.push(p);
  }
  const upper: Array<{x:number;y:number}> = [];
  for (let i=sorted.length-1;i>=0;i--) {
    const p=sorted[i];
    while (upper.length>=2 && cross(upper[upper.length-2],upper[upper.length-1],p)<=0) upper.pop();
    upper.push(p);
  }
  upper.pop(); lower.pop();
  return lower.concat(upper);
}
function expandHull(hull: Array<{x:number;y:number}>, margin: number): Array<{x:number;y:number}> {
  if (!hull.length) return hull;
  const cx=hull.reduce((s,p)=>s+p.x,0)/hull.length;
  const cy=hull.reduce((s,p)=>s+p.y,0)/hull.length;
  return hull.map(p=>{const dx=p.x-cx,dy=p.y-cy,d=Math.sqrt(dx*dx+dy*dy)||1;return{x:p.x+(dx/d)*margin,y:p.y+(dy/d)*margin};});
}
// Scatter plot UMAP moderne avec nuages pastel
function UmapScatterPlot({clusters,selectedCluster,onSelectCluster}:{clusters:ClusterResult[];selectedCluster:number|null;onSelectCluster:(id:number)=>void}) {
  const { t } = useI18n();
  const allPoints: Array<ClusterPoint&{cluster_id:number;is_noise:boolean}>=[];
  clusters.forEach(c=>{if(c.points) c.points.forEach(p=>allPoints.push({...p,cluster_id:c.cluster_id,is_noise:c.is_noise}));});
  if(!allPoints.length) return <span className="text-xs text-white/40">{t("scenarioDetail.clustering.noPoint")}</span>;
  const xs=allPoints.map(p=>p.x),ys=allPoints.map(p=>p.y);
  const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
  const rX=maxX-minX||1,rY=maxY-minY||1;
  const W=420,H=370,PAD=28;
  const sv=(x:number,y:number)=>({cx:PAD+((x-minX)/rX)*(W-2*PAD),cy:H-PAD-((y-minY)/rY)*(H-2*PAD)});
  const hulls: Record<number,string>={};
  clusters.filter(c=>!c.is_noise&&c.points&&c.points.length>=3).forEach(c=>{
    const pts=c.points!.map(p=>{const s=sv(p.x,p.y);return{x:s.cx,y:s.cy};});
    const hull=expandHull(convexHull(pts),16);
    if(hull.length>=3) hulls[c.cluster_id]=hull.map(p=>`${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  });
  const pastelColors=["rgba(52,211,153,0.13)","rgba(56,189,248,0.13)","rgba(167,139,250,0.13)","rgba(251,191,36,0.13)","rgba(244,114,182,0.13)","rgba(96,165,250,0.13)","rgba(45,212,191,0.13)","rgba(192,132,252,0.13)","rgba(251,113,133,0.13)"];
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="overflow-visible">
      <defs>
        <filter id="ptglow"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>
      {[0.25,0.5,0.75].map(f=>(
        <React.Fragment key={f}>
          <line x1={PAD} y1={PAD+f*(H-2*PAD)} x2={W-PAD} y2={PAD+f*(H-2*PAD)} stroke="rgba(255,255,255,0.04)" strokeDasharray="4,4"/>
          <line x1={PAD+f*(W-2*PAD)} y1={PAD} x2={PAD+f*(W-2*PAD)} y2={H-PAD} stroke="rgba(255,255,255,0.04)" strokeDasharray="4,4"/>
        </React.Fragment>
      ))}
      {Object.entries(hulls).map(([cid,pts])=>{
        const id=parseInt(cid),sel=selectedCluster===id;
        return(<polygon key={`h-${cid}`} points={pts}
          fill={pastelColors[id%pastelColors.length]}
          stroke={getClusterColor(id,false)} strokeWidth={sel?1.5:0.7}
          strokeOpacity={sel?0.55:0.2} strokeDasharray={sel?"none":"5,3"}
          className="transition-all duration-300 cursor-pointer" onClick={()=>onSelectCluster(id)}/>);
      })}
      {allPoints.map(p=>{
        const{cx,cy}=sv(p.x,p.y);
        const sel=selectedCluster===p.cluster_id;
        const col=getClusterColor(p.cluster_id,p.is_noise);
        const r=p.is_noise?2.5:sel?7:4.5;
        return(<g key={p.id} className="cursor-pointer" onClick={()=>onSelectCluster(p.cluster_id)}>
          {sel&&!p.is_noise&&<circle cx={cx} cy={cy} r={r+6} fill={col} opacity={0.14}/>}
          <circle cx={cx} cy={cy} r={r} fill={col}
            opacity={p.is_noise?0.22:sel?1:0.78}
            stroke={sel?"#fff":p.is_noise?"transparent":col}
            strokeWidth={sel?2:0.5} strokeOpacity={0.5}
            filter={sel?"url(#ptglow)":undefined}
            className="transition-all duration-200">
            <title>{p.title} ({p.year||"?"})</title>
          </circle>
        </g>);
      })}
      {clusters.filter(c=>!c.is_noise&&c.points&&c.points.length>0).map(c=>{
        const pts=c.points!.map(p=>sv(p.x,p.y));
        const mx=pts.reduce((s,p)=>s+p.cx,0)/pts.length;
        const my=pts.reduce((s,p)=>s+p.cy,0)/pts.length;
        const sel=selectedCluster===c.cluster_id;
        return(<text key={`lbl-${c.cluster_id}`} x={mx} y={my} textAnchor="middle" dominantBaseline="middle"
          fontSize="9" fontWeight="700" fill={getClusterColor(c.cluster_id,false)}
          opacity={sel?1:0.5} letterSpacing="0.05em" className="pointer-events-none select-none uppercase">
          {c.cluster_name.replace("Cluster ","C")}
        </text>);
      })}
    </svg>
  );
}
// ─── Section: RAG ─────────────────────────────────────────────────────────────

function RagSection({ scenarioId, detail }: { scenarioId: string; detail: ScenarioDetail }) {
  const { t } = useI18n();
  const [question, setQuestion] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamedText, setStreamedText] = useState("");
  const [sources, setSources] = useState<ScenarioRagResponse['sources']>([]);
  const [meta, setMeta] = useState<RagMeta | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const cancelRef = useRef<(() => void) | null>(null);
  const answerRef = useRef<HTMLDivElement>(null);

  // Indexation RAG du corpus : combien de documents l'Assistant peut déjà
  // interroger. Rafraîchi tant que des chunks restent à vectoriser, pour que la
  // couverture se mette à jour en direct sous les yeux de l'utilisateur.
  const [embeddingStatus, setEmbeddingStatus] = useState<EmbeddingStatus | null>(null);
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setInterval> | null = null;
    const stop = () => { if (timer) { clearInterval(timer); timer = null; } };
    const load = () => fetchUserScenarioEmbeddingStatus(scenarioId)
      .then((s) => {
        if (stopped) return;
        setEmbeddingStatus(s);
        if ((s.total_pending_chunks ?? 0) === 0 && (s.chunkless ?? 0) === 0) stop();
      })
      .catch(() => { /* non bloquant */ });
    load();
    timer = setInterval(load, 4000);
    return () => { stopped = true; stop(); };
  }, [scenarioId]);

  // Abandonner un stream RAG en cours au démontage (navigation hors du scénario) :
  // sinon le reader SSE continue de tourner et appelle setState sur un composant
  // démonté (fuite de connexion + avertissement React).
  useEffect(() => () => { cancelRef.current?.(); }, []);

  const ask = (qText: string) => {
    if (!qText.trim() || streaming) return;
    // Cancel previous
    if (cancelRef.current) cancelRef.current();
    setStreaming(true);
    setStreamedText("");
    setSources([]);
    setMeta(null);
    setError(null);
    setDone(false);

    const cancel = askScenarioRagStreamFiltered(scenarioId, qText, {
      onSources: (s) => setSources(s),
      onMeta: (m) => setMeta(m),
      onToken: (t) => {
        setStreamedText(prev => prev + t);
        // Auto-scroll
        if (answerRef.current) {
          answerRef.current.scrollTop = answerRef.current.scrollHeight;
        }
      },
      onDone: () => { setStreaming(false); setDone(true); },
      onError: (e) => { setError(e); setStreaming(false); },
    });
    cancelRef.current = cancel;
  };

  const reset = () => {
    if (cancelRef.current) cancelRef.current();
    setStreamedText("");
    setSources([]);
    setMeta(null);
    setError(null);
    setDone(false);
    setStreaming(false);
    setQuestion("");
  };

  const suggestedQuestions = detail.nl_queries.slice(0, 3);

  return (
    <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-5">
      <SectionHeader
        icon={<MessageSquare size={14} className="text-brand-400" />}
        title={t("scenarioDetail.rag.title")}
        subtitle={t("scenarioDetail.rag.subtitle")}
      />

      {/* Indexation du corpus pour l'Assistant (RAG) — couverture documentaire.
          Déplacée ici depuis l'onglet Corpus : c'est la fonctionnalité concernée. */}
      {embeddingStatus && (() => {
        const total = embeddingStatus.corpus_total ?? embeddingStatus.ranking?.total ?? 0;
        const chunkless = embeddingStatus.chunkless ?? 0;
        const pendingChunks = embeddingStatus.total_pending_chunks ?? 0;
        // Progrès réconcilié AU NIVEAU DOCUMENT (somme == total).
        const pendingDocs = (embeddingStatus.title_abstract_chunks?.pending_docs ?? 0)
          + (embeddingStatus.fulltext?.docs_pending ?? 0) + chunkless;
        const indexedDocs = Math.max(0, total - pendingDocs);
        const complete = pendingChunks === 0 && chunkless === 0;
        return (
          <div className={`rounded-2xl border px-4 py-3 flex items-start gap-3 ${complete ? 'border-forest-500/20 bg-forest-500/5' : 'border-gold-500/20 bg-gold-500/5'}`}>
            <div className="shrink-0 mt-0.5">
              {complete ? <CheckCircle2 size={14} className="text-forest-400" />
                : <Loader2 size={14} className="text-gold-400 animate-spin" />}
            </div>
            <div className="flex-1 min-w-0 space-y-1">
              <p className={`text-xs font-semibold ${complete ? 'text-forest-300' : 'text-gold-300'}`}>
                {complete ? t("scenarioDetail.rag.indexingReady") : t("scenarioDetail.rag.indexingInProgress")}
              </p>
              <p className="text-[11px] text-white/70">
                <span className="text-white font-semibold">{indexedDocs}</span> / {total} {t("scenarioDetail.rag.documentsIndexed")}
                {pendingDocs > 0 && ` · ${pendingDocs} ${t("scenarioDetail.rag.docsInProgressSuffix")}`}
                {(embeddingStatus.fulltext?.total_docs ?? 0) > 0 && (
                  <span className="text-white/40">{t("scenarioDetail.rag.inFulltextPrefix")}{embeddingStatus.fulltext?.docs_fully_embedded}/{embeddingStatus.fulltext?.total_docs}{t("scenarioDetail.rag.inFulltextSuffix")}</span>
                )}
              </p>
              {chunkless > 0 ? (
                <p className="text-[10px] text-gold-300">{t("scenarioDetail.rag.chunklessWarningPrefix")}{chunkless}{t("scenarioDetail.rag.chunklessWarningSuffix")}</p>
              ) : complete ? (
                <p className="text-[10px] text-forest-400">{t("scenarioDetail.rag.coverageComplete")}</p>
              ) : (
                <p className="text-[10px] text-white/40">
                  {t("scenarioDetail.rag.coverageExpandingPrefix")}{pendingChunks > 0 ? `${t("scenarioDetail.rag.chunksRemainingPrefix")}${pendingChunks}${t("scenarioDetail.rag.chunksRemainingSuffix")}` : ''}.
                </p>
              )}
            </div>
          </div>
        );
      })()}

      {/* Questions suggérées */}
      {suggestedQuestions.length > 0 && !streamedText && !streaming && (
        <div className="space-y-2">
          <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">{t("scenarioDetail.rag.suggestedQuestions")}</p>
          <div className="flex flex-wrap gap-2">
            {suggestedQuestions.map((q, i) => (
              <button
                key={i}
                onClick={() => { setQuestion(q); ask(q); }}
                disabled={streaming}
                className="text-left rounded-xl border border-white/5 bg-white/2 hover:bg-white/5 px-3 py-2 text-xs text-white/70 hover:text-white transition disabled:opacity-50"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Zone de saisie */}
      <div className="flex gap-2">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask(question)}
          placeholder={t("scenarioDetail.rag.questionPlaceholder")}
          disabled={streaming}
          className="flex-1 rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-xs text-white focus:outline-none focus:border-brand-500/50 transition disabled:opacity-50"
        />
        {streaming ? (
          <button onClick={reset}
            className="rounded-xl bg-rose-500/20 border border-rose-500/30 hover:bg-rose-500/30 text-rose-300 font-semibold px-4 text-xs transition shrink-0"
          >{t("scenarioDetail.rag.stop")}</button>
        ) : (
          <button
            onClick={() => ask(question)}
            disabled={!question.trim()}
            className="rounded-xl bg-brand-500 hover:bg-brand-400 text-forest-950 font-semibold px-4 text-xs transition disabled:opacity-50 shrink-0"
          >{t("scenarioDetail.rag.ask")}</button>
        )}
      </div>

      {error && <ErrorBox message={error} />}

      {(streaming || streamedText) && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3 border-t border-white/5 pt-5">
          {/* Réponse streaming */}
          <div className="lg:col-span-2 space-y-3">
            <div className="flex items-center justify-between">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/50">{t("scenarioDetail.rag.answerTitle")}</p>
              {streaming && (
                <div className="flex items-center gap-1.5 text-[10px] text-brand-300">
                  <span className="h-1.5 w-1.5 rounded-full bg-brand-400 animate-pulse"/>
                  {t("scenarioDetail.rag.generating")}
                </div>
              )}
              {done && (
                <button onClick={reset} className="text-[10px] text-white/30 hover:text-white transition">{t("scenarioDetail.rag.newQuestion")}</button>
              )}
            </div>
            <div
              ref={answerRef}
              className="rounded-2xl border border-white/5 bg-white/2 p-4 text-xs text-white/80 leading-6 whitespace-pre-wrap max-h-[400px] overflow-y-auto"
            >
              {streamedText}
              {streaming && <span className="inline-block w-0.5 h-3 bg-brand-400 animate-pulse ml-0.5 align-middle"/>}
            </div>
            {meta && meta.papers_used > 0 && (
              <p className="mt-2 text-[10px] text-white/40">
                {meta.papers_used} {t("scenarioDetail.rag.papersUsedSuffix")} (≥ {meta.threshold}) · {meta.papers_with_fulltext} {t("scenarioDetail.rag.withFulltextSuffix")}
              </p>
            )}
          </div>

          {/* Sources citées */}
          {sources.length > 0 && (
            <div className="space-y-3">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/50">{t("scenarioDetail.rag.citedSources")}</p>
              <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1">
                {sources.map((src, i) => (
                  <div key={i} className="rounded-xl border border-white/5 bg-white/3 p-2.5 text-xs">
                    <div className="flex items-center justify-between gap-2">
                      <span className="rounded bg-brand-500/10 border border-brand-500/20 px-1.5 py-0.5 text-[9px] text-brand-300 font-mono">
                        {t("scenarioDetail.rag.sourceLabel")} {i + 1}
                      </span>
                      <span className="text-[10px] text-white/35 font-mono">{t("scenarioDetail.rag.relevance")} {(src.score * 100).toFixed(0)}%</span>
                    </div>
                    <h5 className="font-semibold text-white mt-1.5 leading-4 line-clamp-2">{src.title}</h5>
                    <p className="text-[10px] text-white/35 mt-1 truncate">
                      {src.authors} • {src.year || "N/A"}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
          {streaming && sources.length === 0 && (
            <div className="space-y-2">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/50">{t("scenarioDetail.rag.citedSources")}</p>
              <div className="rounded-xl border border-white/5 bg-white/2 p-3 text-center text-[10px] text-white/30">
                <Loader2 size={12} className="animate-spin mx-auto mb-1"/>
                {t("scenarioDetail.rag.loadingSources")}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Section: PRISMA ──────────────────────────────────────────────────────────

// ─── PRISMA 2020 SVG Diagram ──────────────────────────────────────────────────
// Layout PRISMA 2020 officiel :
//   Phase 1 : Identification par source (noeuds en haut, un par base)
//   Phase 2 : Regroupement (total) + deduplication
//   Phase 3 : Screening titre/résumé
//   Phase 4 : Eligibilité (texte intégral)
//   Phase 5 : Inclus
// Inspiré du template officiel PRISMA 2020 (Page, McKenzie et al. 2021)

// Heuristic default: map a public variable's machine_name to a likely connector+variable.
function _guessConnector(mn: string, connectors: DataConnector[]): { id: string; var: string } | null {
  const n = (mn || "").toLowerCase();
  const first = (cid: string): { id: string; var: string } | null => {
    const c = connectors.find(x => x.id === cid);
    return c && c.variables[0] ? { id: cid, var: c.variables[0].machine_name } : null;
  };
  const match = (cid: string): { id: string; var: string } | null => {
    const c = connectors.find(x => x.id === cid);
    const v = c?.variables.find(v => {
      const vn = v.machine_name.toLowerCase();
      return n.includes(vn.split("_")[0]) || vn.includes(n.split("_")[0]);
    });
    return c && v ? { id: cid, var: v.machine_name } : first(cid);
  };
  if (/(temp|humid|precip|wind|weather|met[eé]o)/.test(n)) return match("open-meteo-weather");
  if (/(pm2|pm10|no2|o3|ozone|air|pollut)/.test(n)) return match("open-meteo-air-quality");
  if (/(rsv|sars|sewage)/.test(n)) return match("eawag-wastewater");   // per-virus loads (archive)
  // clinical influenza-like-illness / ARI consultation incidence → FOPH Sentinella (live)
  if (/(ili|ari|grippal|grippe|influenza.?like|consultation|sentinella|incidence)/.test(n)) return match("foph-sentinella-ili");
  if (/(wastewater|viral.?load|influenza|flu|ww)/.test(n)) return match("foph-wastewater");  // live influenza WW
  return null;
}

const _REGION_OPTS = ["geneva", "lausanne", "sion", "neuchatel", "fribourg", "jura"];

// Slice 3b — per-variable pickers: for each PUBLIC data-template column, choose a
// connector + variable, set a shared region/date window + frequency, and fetch the
// dataset automatically instead of uploading a CSV.
function AutoFetchPanel({ scenarioId, spec, onFetched }: {
  scenarioId: string; spec: ModelSpecResponse | null; onFetched: () => void;
}) {
  const { t } = useI18n();
  const [connectors, setConnectors] = useState<DataConnector[]>([]);
  const [region, setRegion] = useState("geneva");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [freq, setFreq] = useState("W");
  const [autoTrain, setAutoTrain] = useState(true);
  const [sel, setSel] = useState<Record<string, { enabled: boolean; connectorId: string; variable: string }>>({});
  const [fetching, setFetching] = useState(false);
  const [result, setResult] = useState<AutoFetchResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const publicFeatures = (spec?.features ?? []).filter(f => f.source === "public_api" && f.machine_name);

  useEffect(() => { listDataConnectors().then(setConnectors).catch(() => setConnectors([])); }, []);
  useEffect(() => {
    if (!publicFeatures.length || !connectors.length) return;
    setSel(prev => {
      const next = { ...prev };
      for (const f of publicFeatures) {
        const mn = f.machine_name!;
        if (next[mn]) continue;
        const g = _guessConnector(mn, connectors);
        next[mn] = { enabled: !!g, connectorId: g?.id ?? "", variable: g?.var ?? "" };
      }
      return next;
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connectors.length, publicFeatures.length]);

  if (!publicFeatures.length) return null;   // nothing public → panel hidden entirely
  const AF = "scenarioDetail.variables.autoFetch";
  const varsFor = (cid: string) => connectors.find(c => c.id === cid)?.variables ?? [];
  const setRow = (mn: string, patch: Partial<{ enabled: boolean; connectorId: string; variable: string }>) =>
    setSel(prev => ({ ...prev, [mn]: { ...prev[mn], ...patch } }));

  const doFetch = async () => {
    const mappings = publicFeatures
      .filter(f => { const s = sel[f.machine_name!]; return s?.enabled && s.connectorId && s.variable; })
      .map(f => ({ template_column: f.machine_name!, connector_id: sel[f.machine_name!].connectorId, connector_variable: sel[f.machine_name!].variable }));
    if (!mappings.length || !startDate || !endDate) { setErr(t(`${AF}.needSelection`)); return; }
    setFetching(true); setErr(null); setResult(null);
    try {
      const res = await autoFetchModelData(scenarioId, { region, start_date: startDate, end_date: endDate, frequency: freq, mappings, auto_train: autoTrain });
      setResult(res); onFetched();
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setFetching(false); }
  };

  return (
    <div className="rounded-3xl border border-brand-500/20 bg-brand-500/5 p-5 space-y-4">
      <SectionHeader icon={<Database size={14} className="text-brand-400" />} title={t(`${AF}.title`)} subtitle={t(`${AF}.subtitle`)} />
      <div className="grid grid-cols-2 gap-2 text-[11px]">
        <label className="space-y-1"><span className="text-white/50">{t(`${AF}.region`)}</span>
          <select value={region} onChange={e => setRegion(e.target.value)} className="w-full rounded-lg bg-forest-900/60 border border-white/10 px-2 py-1.5 text-white/80">
            {_REGION_OPTS.map(r => <option key={r} value={r}>{r}</option>)}
          </select>
        </label>
        <label className="space-y-1"><span className="text-white/50">{t(`${AF}.frequency`)}</span>
          <select value={freq} onChange={e => setFreq(e.target.value)} className="w-full rounded-lg bg-forest-900/60 border border-white/10 px-2 py-1.5 text-white/80">
            <option value="W">{t(`${AF}.freqWeekly`)}</option>
            <option value="D">{t(`${AF}.freqDaily`)}</option>
            <option value="MS">{t(`${AF}.freqMonthly`)}</option>
          </select>
        </label>
        <label className="space-y-1"><span className="text-white/50">{t(`${AF}.from`)}</span>
          <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} className="w-full rounded-lg bg-forest-900/60 border border-white/10 px-2 py-1.5 text-white/80" />
        </label>
        <label className="space-y-1"><span className="text-white/50">{t(`${AF}.to`)}</span>
          <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} className="w-full rounded-lg bg-forest-900/60 border border-white/10 px-2 py-1.5 text-white/80" />
        </label>
      </div>
      <div className="space-y-1.5">
        {publicFeatures.map(f => {
          const mn = f.machine_name!; const s = sel[mn] ?? { enabled: false, connectorId: "", variable: "" };
          return (
            <div key={mn} className="flex items-center gap-2 rounded-xl border border-white/5 bg-white/3 px-2.5 py-2 text-[11px]">
              <input type="checkbox" checked={s.enabled} onChange={e => setRow(mn, { enabled: e.target.checked })} className="accent-brand-500" />
              <span className="w-28 shrink-0 truncate text-white/70" title={f.name || mn}>{f.name || mn}</span>
              <select value={s.connectorId} onChange={e => setRow(mn, { connectorId: e.target.value, variable: varsFor(e.target.value)[0]?.machine_name ?? "" })}
                className="flex-1 min-w-0 rounded-lg bg-forest-900/60 border border-white/10 px-1.5 py-1 text-white/80">
                <option value="">{t(`${AF}.pickConnector`)}</option>
                {connectors.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
              </select>
              <select value={s.variable} onChange={e => setRow(mn, { variable: e.target.value })} disabled={!s.connectorId}
                className="flex-1 min-w-0 rounded-lg bg-forest-900/60 border border-white/10 px-1.5 py-1 text-white/80 disabled:opacity-40">
                <option value="">{t(`${AF}.pickVariable`)}</option>
                {varsFor(s.connectorId).map(v => <option key={v.machine_name} value={v.machine_name}>{v.label}</option>)}
              </select>
            </div>
          );
        })}
      </div>
      <label className="flex items-center gap-1.5 text-[10px] text-white/50">
        <input type="checkbox" checked={autoTrain} onChange={e => setAutoTrain(e.target.checked)} className="accent-brand-500" />
        {t(`${AF}.autoTrain`)}
      </label>
      <button onClick={doFetch} disabled={fetching}
        className="w-full flex items-center justify-center gap-1.5 rounded-xl bg-brand-500 hover:bg-brand-400 text-forest-950 font-semibold py-2 text-xs transition disabled:opacity-50">
        {fetching ? <Loader2 size={12} className="animate-spin" /> : <Database size={12} />}
        {fetching ? t(`${AF}.fetching`) : t(`${AF}.fetch`)}
      </button>
      {err && <ErrorBox message={err} />}
      {result && (
        <div className="rounded-lg bg-forest-900/50 p-2 text-[10px] font-mono text-white/50 space-y-1">
          <div>{t(`${AF}.rows`)} <span className="text-brand-300">{result.n_rows}</span> · {t(`${AF}.filled`)} <span className="text-brand-300">{(result.filled_columns || []).join(", ") || "—"}</span></div>
          {(result.still_needed_user_columns?.length ?? 0) > 0 && (
            <div className="text-gold-300">{t(`${AF}.stillNeeded`)} {result.still_needed_user_columns.join(", ")}</div>
          )}
          {Object.keys(result.fetch_errors || {}).length > 0 && (
            <div className="text-rose-300">{t(`${AF}.errors`)} {Object.entries(result.fetch_errors).map(([k, v]) => `${k}: ${v}`).join(" · ")}</div>
          )}
          {result.training_started && <div className="text-brand-300">{t(`${AF}.trainingStarted`)}</div>}
        </div>
      )}
    </div>
  );
}

const SOURCE_LABELS_MAP: Record<string, string> = {
  pubmed: "PubMed",
  pmc: "PMC",
  openalex: "OpenAlex",
  europepmc: "EuropePMC",
  crossref: "Crossref",
  medrxiv: "medRxiv",
  biorxiv: "bioRxiv",
  semantic_scholar: "Semantic Scholar",
  doaj: "DOAJ",
  clinicaltrials: "ClinicalTrials.gov",
  core: "CORE",
  arxiv: "arXiv",
  openaire: "OpenAIRE",
  db_cache: "DB Cache",
  preprint: "Preprints (Europe PMC)",
  preprints: "Preprints (Europe PMC)",
};

function PrismaStageCard({
  color,
  label,
  icon,
  children,
}: {
  color: string;
  label: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className={`rounded-2xl border p-4 space-y-3 ${color}`}>
      <div className="flex items-center gap-2">
        {icon}
        <span className="text-[10px] font-bold uppercase tracking-widest opacity-70">{label}</span>
      </div>
      {children}
    </div>
  );
}

function PrismaBigNum({ value, sub }: { value: number; sub?: string }) {
  return (
    <div>
      <span className="text-3xl font-black font-mono">{value.toLocaleString()}</span>
      {sub && <p className="text-[10px] opacity-60 mt-0.5">{sub}</p>}
    </div>
  );
}

function PrismaRow({ label, value, accent }: { label: string; value: number | string; accent?: string }) {
  return (
    <div className="flex items-center justify-between text-[11px]">
      <span className="opacity-60">{label}</span>
      <span className={`font-mono font-semibold ${accent ?? ""}`}>{typeof value === "number" ? value.toLocaleString() : value}</span>
    </div>
  );
}

function PrismaConnector() {
  return (
    <div className="flex justify-center">
      <div className="flex flex-col items-center gap-0.5">
        <div className="w-px h-3 bg-white/20" />
        <div className="w-0 h-0 border-l-[5px] border-l-transparent border-r-[5px] border-r-transparent border-t-[6px] border-t-white/20" />
      </div>
    </div>
  );
}

function PrismaSection({ scenarioId }: { scenarioId: string }) {
  const { t } = useI18n();
  const [data, setData] = useState<ScenarioPrisma | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchScenarioPrisma(scenarioId)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [scenarioId]);

  if (loading) return <LoadingSpinner text={t("scenarioDetail.prisma.calculatingFlow")} />;
  if (error || !data) return <ErrorBox message={error ?? t("scenarioDetail.common.errorPrisma")} />;

  // Garde-fous : selon l'avancement du scénario, l'API peut renvoyer un payload
  // partiel (section ou champ absent). Le panneau PRISMA ne doit jamais planter
  // pour autant — on substitue des objets vides / des 0 (PrismaBigNum appelle
  // value.toLocaleString() sans condition, donc un undefined ferait crasher l'app).
  const num = (v: unknown): number => (typeof v === "number" && isFinite(v) ? v : 0);
  const ident = data.identification ?? ({} as ScenarioPrisma["identification"]);
  const sem = data.semantic_screening ?? ({} as ScenarioPrisma["semantic_screening"]);
  const ft = data.full_text ?? ({} as ScenarioPrisma["full_text"]);
  const mc = data.manual_curation ?? ({} as ScenarioPrisma["manual_curation"]);
  const ev = data.evidence ?? ({} as ScenarioPrisma["evidence"]);

  const activeSources = Object.entries(ident.by_source ?? {})
    .filter(([, v]) => num(v) > 0)
    .sort(([, a], [, b]) => num(b) - num(a));

  const totalRecords = num(ident.total_records ?? ident.total_records_identified);
  const evidenceTotal = num(ev.ai_auto_selected) + num(ev.manually_rescued);

  return (
    <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-4">
      <SectionHeader
        icon={<Shield size={14} className="text-brand-400" />}
        title={t("scenarioDetail.prisma.title")}
        subtitle={t("scenarioDetail.prisma.subtitle")}
      />

      <div className="max-w-xl mx-auto space-y-1">

        {/* ── Stage 1: Identification ── */}
        <PrismaStageCard
          color="border-emerald-800/50 bg-emerald-950/30 text-emerald-200"
          label={t("scenarioDetail.prisma.stage1")}
          icon={<Database size={13} className="text-emerald-400" />}
        >
          <PrismaBigNum value={totalRecords} sub={`${activeSources.length} ${activeSources.length !== 1 ? t("scenarioDetail.prisma.sourceSearchedPlural") : t("scenarioDetail.prisma.sourceSearchedSingular")}`} />
          {/* « records identified » = étape PRISMA d'identification : compté AVANT
              déduplication (norme PRISMA 2020), donc légitimement ≥ au corpus dédupliqué
              affiché ailleurs. La note l'explicite pour éviter la lecture « incohérence ». */}
          <div className="text-center text-[10px] text-emerald-300/50 -mt-1">{t("scenarioDetail.prisma.recordsIdentifiedNote")}</div>
          <div className="space-y-1">
            <PrismaRow label={t("scenarioDetail.prisma.embeddedSearchable")} value={num(ident.embedded)} />
            <PrismaRow label={t("scenarioDetail.prisma.duplicatesRemoved")} value={num(ident.duplicates_removed)} />
          </div>
          {activeSources.length > 0 && (
            <div className="flex flex-wrap gap-1.5 pt-1 border-t border-white/5">
              {activeSources.map(([src, cnt]) => (
                <span key={src} className="rounded px-2 py-0.5 bg-emerald-900/40 text-[10px] font-mono text-emerald-300">
                  {SOURCE_LABELS_MAP[src] ?? src.toUpperCase()} {(cnt as number).toLocaleString()}
                </span>
              ))}
            </div>
          )}
        </PrismaStageCard>

        <PrismaConnector />

        {/* ── Stage 2: AI Semantic Pre-screening ── */}
        <PrismaStageCard
          color="border-violet-800/50 bg-violet-950/30 text-violet-200"
          label={t("scenarioDetail.prisma.stage2")}
          icon={<Sparkles size={13} className="text-violet-400" />}
        >
          <div className="grid grid-cols-2 gap-3">
            <div>
              <PrismaBigNum value={num(sem.above_threshold)} sub={t("scenarioDetail.prisma.aboveThresholdPreselected")} />
            </div>
            <div>
              <PrismaBigNum value={num(sem.below_threshold)} sub={t("scenarioDetail.prisma.belowThresholdDeprioritised")} />
            </div>
          </div>
          <div className="space-y-1">
            <PrismaRow label={t("scenarioDetail.prisma.similarityThreshold")} value={`≥ ${(num(sem.threshold) * 100).toFixed(0)}%`} />
            <PrismaRow label={t("scenarioDetail.prisma.method")} value={sem.method ?? "—"} />
            <PrismaRow
              label={t("scenarioDetail.prisma.fullTextsAvailable")}
              value={`${num(ft.with_fulltext).toLocaleString()} (${num(ft.pct).toFixed(0)}%)`}
              accent="text-violet-300"
            />
          </div>
          <p className="text-[9px] opacity-40 italic">{ft.note}</p>
        </PrismaStageCard>

        <PrismaConnector />

        {/* ── Stage 3: Manual Curation ── */}
        <PrismaStageCard
          color="border-amber-800/50 bg-amber-950/30 text-amber-200"
          label={t("scenarioDetail.prisma.stage3")}
          icon={<ClipboardList size={13} className="text-amber-400" />}
        >
          <div className="grid grid-cols-3 gap-3">
            <div>
              <PrismaBigNum value={num(mc.included)} sub={t("scenarioDetail.prisma.included")} />
            </div>
            <div>
              <PrismaBigNum value={num(mc.excluded)} sub={t("scenarioDetail.prisma.excluded")} />
            </div>
            <div>
              <PrismaBigNum value={num(mc.pending)} sub={t("scenarioDetail.prisma.pending")} />
            </div>
          </div>
          <div className="space-y-1 border-t border-white/5 pt-2">
            <PrismaRow label={t("scenarioDetail.prisma.rescued")} value={num(mc.manually_rescued)} accent="text-green-400" />
            <PrismaRow label={t("scenarioDetail.prisma.vetoed")} value={num(mc.manually_vetoed)} accent="text-red-400" />
            <PrismaRow label={t("scenarioDetail.prisma.screeningComplete")} value={mc.screening_complete ? t("scenarioDetail.prisma.yes") : t("scenarioDetail.prisma.inProgress")} />
          </div>
        </PrismaStageCard>

        <PrismaConnector />

        {/* ── Stage 4: Evidence Synthesis ── */}
        <PrismaStageCard
          color="border-cyan-800/50 bg-cyan-950/30 text-cyan-200"
          label={t("scenarioDetail.prisma.stage4")}
          icon={<BookOpen size={13} className="text-cyan-400" />}
        >
          <PrismaBigNum
            value={evidenceTotal}
            sub={mc.screening_complete ? t("scenarioDetail.prisma.finalEvidenceSet") : t("scenarioDetail.prisma.evidenceSetInProgress")}
          />
          <div className="space-y-1">
            <PrismaRow label={t("scenarioDetail.prisma.aiPreselected")} value={num(ev.ai_auto_selected)} />
            <PrismaRow label={t("scenarioDetail.prisma.manuallyRescued")} value={num(ev.manually_rescued)} accent="text-green-400" />
            <PrismaRow label={t("scenarioDetail.prisma.withFullTextLabel")} value={`${num(ev.with_fulltext).toLocaleString()} ${t("scenarioDetail.prisma.withFullTextOf")} ${evidenceTotal.toLocaleString()}`} />
          </div>
        </PrismaStageCard>
      </div>

      {!mc.screening_complete && (
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 px-4 py-3 mt-2">
          <p className="text-[10px] text-amber-400">
            <span className="font-semibold">{t("scenarioDetail.prisma.manualScreeningInProgress")}</span> · {num(mc.pending).toLocaleString()} {t("scenarioDetail.prisma.articlesStillPending")}
          </p>
        </div>
      )}
    </div>
  );
}


// ─── Section: PICO Tableau Comparatif ─────────────────────────────────────────
function PicoSection({ scenarioId }: { scenarioId: string }) {
  const { t } = useI18n();
  const [data, setData] = React.useState<PicoBulkResponse | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [filter, setFilter] = React.useState<'all'|'with_pico'|'without_pico'>('all');
  const [search, setSearch] = React.useState('');
  const [sortBy, setSortBy] = React.useState<'year'|'confidence'|'design'>('year');

  React.useEffect(() => {
    setLoading(true);
    fetchScenarioPicoBulk(scenarioId, 200, 0)
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [scenarioId]);

  const exportCsv = () => {
    if (!data) return;
    const headers = ['ID',t("scenarioDetail.pico.csvTitle"),t("scenarioDetail.pico.csvYear"),t("scenarioDetail.pico.csvJournal"),t("scenarioDetail.pico.csvStudyType"),t("scenarioDetail.pico.csvConfidence"),t("scenarioDetail.pico.csvPopulation"),t("scenarioDetail.pico.csvIntervention"),t("scenarioDetail.pico.csvComparator"),t("scenarioDetail.pico.csvOutcome"),t("scenarioDetail.pico.csvNotes")];
    const rows = data.articles.filter(a => a.has_pico).map(a => [
      a.id, `"${(a.title||'').replace(/"/g,'""')}"`, a.year||'',
      `"${(a.journal||'').replace(/"/g,'""')}"`,
      a.study_design||'', a.pico_confidence!=null?Math.round(a.pico_confidence*100)+'%':'',
      `"${(a.P||'').replace(/"/g,'""')}"`,
      `"${(a.I||'').replace(/"/g,'""')}"`,
      `"${(a.C||'').replace(/"/g,'""')}"`,
      `"${(a.O||'').replace(/"/g,'""')}"`,
      `"${(a.pico_notes||'').replace(/"/g,'""')}"`,
    ]);
    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csv], {type:'text/csv;charset=utf-8'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `pico_${scenarioId}.csv`; a.click();
    URL.revokeObjectURL(url);
  };

  if (loading) return <LoadingSpinner text={t("scenarioDetail.pico.loadingData")} />;
  if (error || !data) return <ErrorBox message={error ?? t("scenarioDetail.common.errorPico")} />;

  const filtered = data.articles
    .filter(a => filter === 'all' || (filter === 'with_pico' ? a.has_pico : !a.has_pico))
    .filter(a => !search || a.title.toLowerCase().includes(search.toLowerCase()))
    .sort((a,b) => {
      if (sortBy === 'year') return (b.year||0)-(a.year||0);
      if (sortBy === 'confidence') return (b.pico_confidence||0)-(a.pico_confidence||0);
      return (a.study_design||'').localeCompare(b.study_design||'');
    });

  const coverage = data.total > 0 ? Math.round(data.with_pico/data.total*100) : 0;

  return (
    <div className="space-y-5">
      <SectionHeader
        icon={<Table2 size={14} className="text-brand-400" />}
        title={t("scenarioDetail.pico.title")}
        subtitle={t("scenarioDetail.pico.subtitle")}
      />
      {/* Stats bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          {label:t("scenarioDetail.pico.statTotalArticles"),value:data.total,color:'text-white'},
          {label:t("scenarioDetail.pico.statWithPico"),value:data.with_pico,color:'text-brand-300'},
          {label:t("scenarioDetail.pico.statCoverage"),value:coverage+'%',color:coverage>70?'text-brand-300':coverage>40?'text-gold-400':'text-rose-300'},
          {label:t("scenarioDetail.pico.statWithoutPico"),value:data.total-data.with_pico,color:'text-white/50'},
        ].map(s=>(
          <div key={s.label} className="rounded-2xl border border-white/5 bg-white/3 p-3 text-center">
            <div className={`text-xl font-bold ${s.color}`}>{s.value}</div>
            <div className="text-[10px] text-white/40 mt-0.5">{s.label}</div>
          </div>
        ))}
      </div>
      {/* Coverage bar */}
      <div className="rounded-xl border border-white/5 bg-white/2 p-3">
        <div className="flex justify-between text-[10px] text-white/40 mb-1.5">
          <span>{t("scenarioDetail.pico.coverageLabel")}</span><span>{data.with_pico}/{data.total}</span>
        </div>
        <div className="h-2 bg-white/5 rounded-full overflow-hidden">
          <div className="h-full bg-brand-500 rounded-full transition-all" style={{width:`${coverage}%`}}/>
        </div>
      </div>
      {/* Controls */}
      <div className="flex flex-wrap gap-2 items-center">
        <input
          type="text" placeholder={t("scenarioDetail.pico.searchArticlePlaceholder")} value={search}
          onChange={e=>setSearch(e.target.value)}
          className="flex-1 min-w-[200px] rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-white focus:outline-none focus:border-brand-500/50"
        />
        <div className="flex gap-1">
          {(['all','with_pico','without_pico'] as const).map(f=>(
            <button key={f} onClick={()=>setFilter(f)}
              className={`px-2.5 py-1.5 rounded-lg text-[10px] font-medium transition ${
                filter===f?'bg-brand-700 text-gold-400 font-semibold':'text-white/60 hover:text-white hover:bg-white/8'
              }`}
            >
              {f==='all'?t("scenarioDetail.pico.filterAll"):f==='with_pico'?t("scenarioDetail.pico.filterWithPico"):t("scenarioDetail.pico.filterWithoutPico")}
            </button>
          ))}
        </div>
        <select value={sortBy} onChange={e=>setSortBy(e.target.value as any)}
          className="rounded-lg border border-white/10 bg-white/5 px-2 py-1.5 text-[10px] text-white/70 focus:outline-none"
        >
          <option value="year">{t("scenarioDetail.pico.sortByYear")}</option>
          <option value="confidence">{t("scenarioDetail.pico.sortByConfidence")}</option>
          <option value="design">{t("scenarioDetail.pico.sortByDesign")}</option>
        </select>
        {data.with_pico > 0 && (
          <button onClick={exportCsv}
            className="flex items-center gap-1.5 rounded-xl border border-brand-500/30 bg-brand-500/10 px-3 py-1.5 text-[10px] text-brand-300 hover:bg-brand-500/20 transition"
          >
            <Download size={10}/>{t("scenarioDetail.pico.exportCsv")}
          </button>
        )}
      </div>
      {/* Table */}
      <div className="overflow-x-auto rounded-2xl border border-white/5">
        <table className="w-full text-[10px] border-collapse">
          <thead>
            <tr className="border-b border-white/5 bg-white/3">
              {[t("scenarioDetail.pico.colTitle"),t("scenarioDetail.pico.colYear"),t("scenarioDetail.pico.colStudyType"),t("scenarioDetail.pico.colConfidence"),t("scenarioDetail.pico.colPopulation"),t("scenarioDetail.pico.colIntervention"),t("scenarioDetail.pico.colComparator"),t("scenarioDetail.pico.colOutcome")].map(h=>(
                <th key={h} className="text-left px-3 py-2 text-white/40 font-semibold uppercase tracking-wider whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0,100).map((a,i)=>(
              <tr key={a.id} className={`border-b border-white/3 transition ${i%2===0?'bg-white/1':'bg-transparent'} hover:bg-white/4`}>
                <td className="px-3 py-2 max-w-[200px]">
                  <div className="flex items-start gap-1.5">
                    {a.has_pico
                      ? <span className="mt-0.5 h-1.5 w-1.5 rounded-full bg-brand-400 shrink-0"/>
                      : <span className="mt-0.5 h-1.5 w-1.5 rounded-full bg-white/20 shrink-0"/>
                    }
                    <span className="text-white/70 leading-4 line-clamp-2">{a.title}</span>
                  </div>
                </td>
                <td className="px-3 py-2 text-white/50 font-mono whitespace-nowrap">{a.year||'—'}</td>
                <td className="px-3 py-2 whitespace-nowrap">
                  {a.study_design
                    ? <span className="rounded-md bg-brand-500/10 border border-brand-500/20 px-1.5 py-0.5 text-brand-300">{a.study_design}</span>
                    : <span className="text-white/25">—</span>
                  }
                </td>
                <td className="px-3 py-2 whitespace-nowrap">
                  {a.pico_confidence!=null
                    ? <span className={`font-mono font-semibold ${a.pico_confidence>0.7?'text-brand-300':a.pico_confidence>0.4?'text-gold-400':'text-rose-300'}`}>
                        {Math.round(a.pico_confidence*100)}%
                      </span>
                    : <span className="text-white/25">—</span>
                  }
                </td>
                {(['P','I','C','O'] as const).map(key=>(
                  <td key={key} className="px-3 py-2 max-w-[160px]">
                    <span className="text-white/60 leading-4 line-clamp-3">{(a as any)[key]||<span className="text-white/20">—</span>}</span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {filtered.length > 100 && (
          <div className="text-center py-3 text-[10px] text-white/35">
            {t("scenarioDetail.pico.showingFirstPart1")} {filtered.length}. {t("scenarioDetail.pico.showingFirstPart2")}
          </div>
        )}
        {filtered.length === 0 && (
          <div className="text-center py-8 text-xs text-white/35">{t("scenarioDetail.pico.noArticleMatch")}</div>
        )}
      </div>
    </div>
  );
}

// ─── Section: Knowledge Graph (co-citations) ─────────────────────────────────

function KnowledgeGraphSection({ scenarioId }: { scenarioId: string }) {
  const { t } = useI18n();
  const [data, setData] = React.useState<KnowledgeGraphData | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = React.useState<number | null>(null);
  const [selectedNode, setSelectedNode] = React.useState<number | null>(null);
  const [tooltip, setTooltip] = React.useState<{x:number;y:number;node:KGNode}|null>(null);
  const [minSim, setMinSim] = React.useState(0.35);

  React.useEffect(() => {
    setLoading(true);
    fetchKnowledgeGraph(scenarioId, 400, minSim)
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [scenarioId, minSim]);

  if (loading) return <LoadingSpinner text={t("scenarioDetail.knowledgeGraph.calculating")} />;
  if (error || !data) return <ErrorBox message={error ?? t("scenarioDetail.common.errorKnowledgeGraph")} />;
  if (!data.nodes.length) return <div className="text-xs text-white/40 p-4">{t("scenarioDetail.knowledgeGraph.noArticleEmbeddings")}</div>;

  const W = 700, H = 500;
  const CLUSTER_COLORS = [
    "#22c55e","#38bdf8","#a78bfa","#fb923c","#f472b6",
    "#34d399","#fbbf24","#60a5fa","#e879f9","#4ade80",
  ];
  const PASTEL = [
    "rgba(34,197,94,0.08)","rgba(56,189,248,0.08)","rgba(167,139,250,0.08)",
    "rgba(251,146,60,0.08)","rgba(244,114,182,0.08)","rgba(52,211,153,0.08)",
    "rgba(251,191,36,0.08)","rgba(96,165,250,0.08)","rgba(232,121,249,0.08)",
  ];

  // Force-directed layout simulation (static, pre-computed)
  // Use cluster assignment to position nodes in cluster groups
  const clusterCenters: Record<number, {x:number;y:number}> = {};
  const uniqueClusters = [...new Set(data.nodes.map(n => n.cluster))];
  uniqueClusters.forEach((cid, i) => {
    const angle = (2 * Math.PI * i) / uniqueClusters.length - Math.PI / 2;
    const r = Math.min(W, H) * 0.3;
    clusterCenters[cid] = { x: W/2 + r * Math.cos(angle), y: H/2 + r * Math.sin(angle) };
  });

  // Position nodes around their cluster center with jitter based on degree
  const nodePositions: Record<number, {x:number;y:number}> = {};
  const clusterNodeCounts: Record<number, number> = {};
  data.nodes.forEach(n => { clusterNodeCounts[n.cluster] = (clusterNodeCounts[n.cluster] || 0) + 1; });
  const clusterNodeIdx: Record<number, number> = {};
  data.nodes.forEach(n => {
    const idx = clusterNodeIdx[n.cluster] || 0;
    clusterNodeIdx[n.cluster] = idx + 1;
    const count = clusterNodeCounts[n.cluster];
    const center = clusterCenters[n.cluster] || {x: W/2, y: H/2};
    const spread = Math.min(80, 20 + count * 4);
    const angle2 = (2 * Math.PI * idx) / count;
    const r2 = spread * (0.3 + 0.7 * (idx / count));
    nodePositions[n.id] = {
      x: Math.max(20, Math.min(W-20, center.x + r2 * Math.cos(angle2))),
      y: Math.max(20, Math.min(H-20, center.y + r2 * Math.sin(angle2))),
    };
  });

  const nodeMap = new Map(data.nodes.map(n => [n.id, n]));
  const maxDegree = Math.max(1, ...data.nodes.map(n => n.degree));

  const isActive = (nodeId: number) => {
    if (!selectedNode) return true;
    if (nodeId === selectedNode) return true;
    return data.edges.some(e =>
      (e.source === nodeId || e.target === nodeId) &&
      (e.source === selectedNode || e.target === selectedNode)
    );
  };

  const selectedNodeData = selectedNode ? nodeMap.get(selectedNode) : null;

  return (
    <div className="space-y-4">
      <SectionHeader
        icon={<Network size={14} className="text-brand-400" />}
        title={t("scenarioDetail.knowledgeGraph.title")}
        subtitle={
          (data.n_total && data.n_total > data.n_nodes
            ? `${data.n_nodes} ${t("scenarioDetail.knowledgeGraph.subtitleMostRelevantPrefix")} ${data.n_total} · `
            : `${data.n_nodes} ${t("scenarioDetail.knowledgeGraph.subtitleArticles")} · `) +
          `${data.n_edges} ${t("scenarioDetail.knowledgeGraph.subtitleLinks")} · ${data.n_clusters} ${t("scenarioDetail.knowledgeGraph.subtitleCommunities")} · ${t("scenarioDetail.knowledgeGraph.subtitleProximity")}`
        }
      />

      {/* Contrôles */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2 text-xs text-white/50">
          <span>{t("scenarioDetail.knowledgeGraph.similarityThreshold")}</span>
          <input type="range" min={0.2} max={0.7} step={0.05} value={minSim}
            onChange={e => setMinSim(parseFloat(e.target.value))}
            className="w-28 accent-brand-500"
          />
          <span className="font-mono text-brand-300">{minSim.toFixed(2)}</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {[...data.clusters].sort((a,b)=>b.size-a.size).slice(0,5).map((c) => {
            const i = uniqueClusters.indexOf(c.id);
            return (
              <span key={c.id}
                className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/3 px-2 py-1 text-[10px] text-white/50"
              >
                <span className="h-2 w-2 rounded-full shrink-0" style={{background: CLUSTER_COLORS[i % CLUSTER_COLORS.length]}}/>
                <span className="text-white/70">{c.label || `${t("scenarioDetail.knowledgeGraph.groupFallback")} ${c.id + 1}`}</span>
                <span className="text-white/30 font-mono">· {c.size}</span>
              </span>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* SVG Graph */}
        <div className="lg:col-span-2">
          <svg width="100%" viewBox={`0 0 ${W} ${H}`}
            className="bg-[#070f0a] rounded-2xl border border-white/5 overflow-visible"
            style={{maxHeight: 500}}
          >
            <defs>
              <filter id="kg-glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
              <filter id="kg-glow-sm"><feGaussianBlur stdDeviation="1.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
              {uniqueClusters.map((cid, i) => (
                <radialGradient key={cid} id={`kg-rg-${cid}`} cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor={CLUSTER_COLORS[i % CLUSTER_COLORS.length]} stopOpacity="0.9"/>
                  <stop offset="100%" stopColor={CLUSTER_COLORS[i % CLUSTER_COLORS.length]} stopOpacity="0.3"/>
                </radialGradient>
              ))}
            </defs>

            {/* Cluster halos */}
            {uniqueClusters.map((cid, i) => {
              const center = clusterCenters[cid];
              const count = clusterNodeCounts[cid] || 1;
              const r = Math.min(90, 30 + count * 5);
              return (
                <ellipse key={cid}
                  cx={center.x} cy={center.y}
                  rx={r * 1.4} ry={r}
                  fill={PASTEL[i % PASTEL.length]}
                  stroke={CLUSTER_COLORS[i % CLUSTER_COLORS.length]}
                  strokeWidth="0.5" strokeOpacity="0.3"
                />
              );
            })}

            {/* Edges */}
            {data.edges.map((e, i) => {
              const pa = nodePositions[e.source];
              const pb = nodePositions[e.target];
              if (!pa || !pb) return null;
              const active = isActive(e.source) && isActive(e.target);
              const highlighted = selectedNode && (e.source === selectedNode || e.target === selectedNode);
              return (
                <line key={i}
                  x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y}
                  stroke={highlighted ? "rgba(255,255,255,0.4)" : "rgba(255,255,255,0.06)"}
                  strokeWidth={highlighted ? e.weight * 2 : e.weight * 0.5}
                  opacity={active ? 1 : 0.1}
                />
              );
            })}

            {/* Nodes */}
            {data.nodes.map(n => {
              const pos = nodePositions[n.id];
              if (!pos) return null;
              const ci = uniqueClusters.indexOf(n.cluster);
              const color = CLUSTER_COLORS[ci % CLUSTER_COLORS.length];
              const r = Math.max(4, Math.min(12, 3 + (n.degree / maxDegree) * 9));
              const active = isActive(n.id);
              const sel = selectedNode === n.id;
              const hov = hoveredNode === n.id;
              return (
                <g key={n.id}
                  className="cursor-pointer"
                  onClick={() => setSelectedNode(sel ? null : n.id)}
                  onMouseEnter={_e => { setHoveredNode(n.id); setTooltip({x: pos.x, y: pos.y, node: n}); }}
                  onMouseLeave={() => { setHoveredNode(null); setTooltip(null); }}
                >
                  {(sel || hov) && <circle cx={pos.x} cy={pos.y} r={r + 6} fill={color} opacity={0.15}/>}
                  {sel && <circle cx={pos.x} cy={pos.y} r={r + 4} fill="none" stroke={color} strokeWidth="1.5" strokeDasharray="3,2" opacity={0.7}/>}
                  <circle
                    cx={pos.x} cy={pos.y} r={sel || hov ? r + 2 : r}
                    fill={`url(#kg-rg-${n.cluster})`}
                    stroke={sel ? "#fff" : color}
                    strokeWidth={sel ? 1.5 : 0.8}
                    opacity={active ? 1 : 0.15}
                    filter={(sel || hov) ? "url(#kg-glow)" : undefined}
                  />
                </g>
              );
            })}

            {/* Tooltip */}
            {tooltip && (() => {
              const n = tooltip.node;
              const x = tooltip.x;
              const y = tooltip.y;
              const title = (n.title || '').slice(0, 50);
              const meta = `${n.year || 'N/A'} · ${n.design} · ${n.degree} ${t("scenarioDetail.knowledgeGraph.tooltipConnections")}`;
              const tw = Math.max(title.length, meta.length) * 4.5 + 16;
              const tx = Math.max(tw/2 + 4, Math.min(W - tw/2 - 4, x));
              const ty = y - 30;
              return (
                <g>
                  <rect x={tx - tw/2} y={ty - 22} width={tw} height={28}
                    rx="5" fill="rgba(7,15,10,0.95)" stroke="rgba(255,255,255,0.15)" strokeWidth="0.8"/>
                  <text x={tx} y={ty - 10} textAnchor="middle" fontSize="7.5" fill="rgba(255,255,255,0.9)"
                    className="pointer-events-none select-none" fontWeight="600">{title}</text>
                  <text x={tx} y={ty + 1} textAnchor="middle" fontSize="6.5" fill="rgba(255,255,255,0.45)"
                    className="pointer-events-none select-none">{meta}</text>
                </g>
              );
            })()}
          </svg>
        </div>

        {/* Panel latéral */}
        <div className="space-y-3">
          {selectedNodeData ? (
            <div className="rounded-2xl border border-white/10 bg-white/3 p-4 space-y-3">
              <div className="flex items-start justify-between gap-2">
                <p className="text-xs font-bold text-white leading-4">{selectedNodeData.title}</p>
                <button onClick={() => setSelectedNode(null)} className="text-white/30 hover:text-white text-xs shrink-0">✕</button>
              </div>
              <div className="space-y-1.5 text-[10px] text-white/50">
                <div className="flex justify-between"><span>{t("scenarioDetail.knowledgeGraph.year")}</span><span className="text-white/70">{selectedNodeData.year || 'N/A'}</span></div>
                <div className="flex justify-between"><span>{t("scenarioDetail.knowledgeGraph.journal")}</span><span className="text-white/70 text-right max-w-[120px] truncate">{selectedNodeData.journal || '—'}</span></div>
                <div className="flex justify-between"><span>{t("scenarioDetail.knowledgeGraph.studyType")}</span><span className="rounded bg-brand-500/10 border border-brand-500/20 px-1 text-brand-300">{selectedNodeData.design}</span></div>
                <div className="flex justify-between"><span>{t("scenarioDetail.knowledgeGraph.connections")}</span><span className="text-brand-300 font-semibold">{selectedNodeData.degree}</span></div>
                <div className="flex justify-between"><span>{t("scenarioDetail.knowledgeGraph.quality")}</span><span className="text-gold-400 font-semibold">{selectedNodeData.quality > 0 ? Math.round(selectedNodeData.quality * 100) + '%' : '—'}</span></div>
              </div>
            </div>
          ) : (
            <div className="rounded-2xl border border-white/5 bg-white/2 p-4 text-center">
              <Network size={20} className="text-white/20 mx-auto mb-2"/>
              <p className="text-[10px] text-white/35">{t("scenarioDetail.knowledgeGraph.clickNode")}</p>
            </div>
          )}

          {/* Articles pivots (les plus connectés) */}
          {(() => {
            const hubs = [...data.nodes].filter(n => n.degree > 0).sort((a,b)=>b.degree-a.degree).slice(0,5);
            if (!hubs.length) return null;
            return (
              <div className="rounded-2xl border border-white/5 bg-white/2 p-3 space-y-2">
                <p className="text-[10px] font-bold uppercase tracking-wider text-white/40">{t("scenarioDetail.knowledgeGraph.hubArticles")}</p>
                {hubs.map(n => (
                  <button key={n.id} onClick={() => setSelectedNode(n.id)}
                    className="w-full text-left flex items-start gap-2 text-[10px] hover:bg-white/3 rounded px-1 py-0.5 transition">
                    <span className="text-brand-300 font-mono shrink-0 w-5">{n.degree}</span>
                    <span className="text-white/60 leading-3 line-clamp-2">{n.title}</span>
                  </button>
                ))}
              </div>
            );
          })()}

          {/* Légende des communautés thématiques */}
          <div className="rounded-2xl border border-white/5 bg-white/2 p-3 space-y-2">
            <p className="text-[10px] font-bold uppercase tracking-wider text-white/40">{t("scenarioDetail.knowledgeGraph.thematicCommunities")}</p>
            {[...data.clusters].filter(c => c.size > 1).sort((a,b)=>b.size-a.size).slice(0, 8).map((c) => {
              const i = uniqueClusters.indexOf(c.id);
              return (
                <div key={c.id} className="flex items-center justify-between gap-2 text-[10px]">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <span className="h-2 w-2 rounded-full shrink-0" style={{background: CLUSTER_COLORS[i % CLUSTER_COLORS.length]}}/>
                    <span className="text-white/60 truncate">{c.label || `${t("scenarioDetail.knowledgeGraph.groupFallback")} ${c.id + 1}`}</span>
                  </div>
                  <span className="text-white/40 font-mono shrink-0">{c.size}</span>
                </div>
              );
            })}
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 gap-2">
            {[
              {label: t("scenarioDetail.knowledgeGraph.statArticles"), value: data.n_nodes},
              {label: t("scenarioDetail.knowledgeGraph.statConnections"), value: data.n_edges},
              {label: t("scenarioDetail.knowledgeGraph.statGroups"), value: data.n_clusters},
              {label: t("scenarioDetail.knowledgeGraph.statThreshold"), value: data.min_similarity.toFixed(2)},
            ].map(s => (
              <div key={s.label} className="rounded-xl border border-white/5 bg-white/2 p-2 text-center">
                <div className="text-sm font-bold text-brand-300">{s.value}</div>
                <div className="text-[9px] text-white/35">{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Section: Double-aveugle Screening + Kappa ────────────────────────────────

// Reviewer code storage key
const REVIEWER_CODE_KEY = 'literev_reviewer_code';
const REVIEWER_ROLE_KEY = 'literev_reviewer_role';

function DoubleBlindSection({ scenarioId }: { scenarioId: string }) {
  const { t } = useI18n();
  const [kappa, setKappa] = React.useState<KappaStats | null>(null);
  const [conflicts, setConflicts] = React.useState<any[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [submitting, setSubmitting] = React.useState<number|null>(null);

  // Reviewer identity system
  const [reviewerCode, setReviewerCode] = React.useState<string>(
    () => localStorage.getItem(REVIEWER_CODE_KEY) ?? ''
  );
  const [reviewerRole, setReviewerRole] = React.useState<1|2|null>(
    () => { const r = localStorage.getItem(REVIEWER_ROLE_KEY); return r ? (parseInt(r) as 1|2) : null; }
  );
  const [codeInput, setCodeInput] = React.useState('');
  const [codeError, setCodeError] = React.useState('');
  const reviewer: 1|2 = reviewerRole ?? 1;

  const handleCodeSubmit = () => {
    const code = codeInput.trim().toUpperCase();
    if (!/^R-?\d{4}$/.test(code) && !/^\d{4}$/.test(code)) {
      setCodeError(t("scenarioDetail.doubleBlind.invalidCodeFormat"));
      return;
    }
    const normalized = code.startsWith('R-') ? code : `R-${code}`;
    // Assign role based on existing registrations
    const existingCode = localStorage.getItem(REVIEWER_CODE_KEY);
    const existingRole = localStorage.getItem(REVIEWER_ROLE_KEY);
    let role: 1|2;
    if (existingCode === normalized && existingRole) {
      role = parseInt(existingRole) as 1|2;
    } else {
      // Check if R1 slot is taken (stored in sessionStorage for cross-tab)
      const r1Code = sessionStorage.getItem(`literev_r1_${scenarioId}`);
      if (!r1Code || r1Code === normalized) {
        role = 1;
        sessionStorage.setItem(`literev_r1_${scenarioId}`, normalized);
      } else {
        role = 2;
      }
    }
    localStorage.setItem(REVIEWER_CODE_KEY, normalized);
    localStorage.setItem(REVIEWER_ROLE_KEY, String(role));
    setReviewerCode(normalized);
    setReviewerRole(role);
    setCodeError('');
  };

  const handleResetCode = () => {
    localStorage.removeItem(REVIEWER_CODE_KEY);
    localStorage.removeItem(REVIEWER_ROLE_KEY);
    setReviewerCode('');
    setReviewerRole(null);
    setCodeInput('');
  };

  const reload = React.useCallback(() => {
    setLoading(true);
    Promise.all([
      fetchKappaStats(scenarioId),
      fetchDoubleBlindConflicts(scenarioId),
    ]).then(([k, c]) => { setKappa(k); setConflicts(c); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [scenarioId]);

  React.useEffect(() => { reload(); }, [reload]);

  const decide = async (articleId: number, status: "included"|"excluded") => {
    if (!reviewerCode) { alert(t("scenarioDetail.doubleBlind.enterCodeFirst")); return; }
    setSubmitting(articleId);
    try {
      await submitDoubleBlindDecision(scenarioId, { article_id: articleId, reviewer, status, reviewer_code: reviewerCode });
      reload();
    } catch (e: any) {
      alert(e.message);
    } finally {
      setSubmitting(null);
    }
  };

  if (loading) return <LoadingSpinner text={t("scenarioDetail.doubleBlind.loading")} />;

  const kappaColor = !kappa?.kappa ? 'text-white/40'
    : kappa.kappa >= 0.61 ? 'text-brand-300'
    : kappa.kappa >= 0.41 ? 'text-gold-400'
    : 'text-rose-300';

  return (
    <div className="space-y-5">
      <SectionHeader
        icon={<Users size={14} className="text-brand-400" />}
        title={t("scenarioDetail.doubleBlind.title")}
        subtitle={t("scenarioDetail.doubleBlind.subtitle")}
      />

      {/* Identification reviewer */}
      {!reviewerCode ? (
        <div className="rounded-2xl border border-white/10 bg-white/3 p-5 space-y-3">
          <p className="text-xs font-semibold text-white/70">{t("scenarioDetail.doubleBlind.reviewerIdentification")}</p>
          <p className="text-[10px] text-white/40 leading-relaxed">
            {t("scenarioDetail.doubleBlind.reviewerIdentificationHint")}
          </p>
          <div className="flex gap-2">
            <input
              type="text"
              value={codeInput}
              onChange={e => { setCodeInput(e.target.value); setCodeError(''); }}
              onKeyDown={e => e.key === 'Enter' && handleCodeSubmit()}
              placeholder={t("scenarioDetail.doubleBlind.codePlaceholder")}
              maxLength={7}
              className="flex-1 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-white placeholder-white/25 focus:outline-none focus:border-brand-500/50"
            />
            <button onClick={handleCodeSubmit}
              className="rounded-xl bg-brand-500 hover:bg-brand-400 text-white font-semibold px-4 py-2 text-xs transition">
              {t("scenarioDetail.doubleBlind.confirm")}
            </button>
          </div>
          {codeError && <p className="text-[10px] text-red-400">{codeError}</p>}
          <p className="text-[9px] text-white/25 italic">{t("scenarioDetail.doubleBlind.codeSavedLocally")}</p>
        </div>
      ) : (
        <div className="flex items-center justify-between gap-3 rounded-2xl border border-brand-500/20 bg-brand-500/5 px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-xl bg-brand-500/20 flex items-center justify-center">
              <span className="text-xs font-bold text-brand-300">R{reviewer}</span>
            </div>
            <div>
              <p className="text-xs font-semibold text-white/80">{t("scenarioDetail.doubleBlind.reviewerPrefix")} {reviewer} : <span className="font-mono text-brand-300">{reviewerCode}</span></p>
              <p className="text-[9px] text-white/35">{t("scenarioDetail.doubleBlind.identitySaved")}</p>
            </div>
          </div>
          <button onClick={handleResetCode} className="text-[9px] text-white/25 hover:text-white/50 transition">{t("scenarioDetail.doubleBlind.change")}</button>
        </div>
      )}

      {/* Score Kappa */}
      {kappa && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            {label: t("scenarioDetail.doubleBlind.statEvaluated"), value: kappa.n_evaluated, color: 'text-white'},
            {label: t("scenarioDetail.doubleBlind.statKappa"), value: kappa.kappa != null ? kappa.kappa.toFixed(3) : 'N/A', color: kappaColor},
            {label: t("scenarioDetail.doubleBlind.statAgreement"), value: kappa.po_observed != null ? Math.round(kappa.po_observed * 100) + '%' : 'N/A', color: 'text-brand-300'},
            {label: t("scenarioDetail.doubleBlind.statConflicts"), value: kappa.conflicts, color: kappa.conflicts > 0 ? 'text-gold-400' : 'text-white/40'},
          ].map(s => (
            <div key={s.label} className="rounded-2xl border border-white/5 bg-white/3 p-3 text-center">
              <div className={`text-xl font-bold ${s.color}`}>{s.value}</div>
              <div className="text-[10px] text-white/35 mt-0.5">{s.label}</div>
            </div>
          ))}
        </div>
      )}
      {kappa?.kappa != null && (
        <div className="rounded-xl border border-white/5 bg-white/2 p-3 flex items-center gap-3">
          <div className={`text-sm font-bold ${kappaColor}`}>{kappa.kappa.toFixed(3)}</div>
          <div>
            <div className="text-xs font-semibold text-white/70">{kappa.interpretation}</div>
            <div className="text-[10px] text-white/35">{t("scenarioDetail.doubleBlind.expectedByChance")} {Math.round((kappa.pe_expected || 0) * 100)}%</div>
          </div>
        </div>
      )}
      {kappa?.n_evaluated === 0 && (
        <div className="rounded-xl border border-gold-500/20 bg-gold-500/5 p-3 text-xs text-gold-300">
          {t("scenarioDetail.doubleBlind.noEvaluationYet")}
        </div>
      )}

      {/* Conflits */}
      {conflicts.length > 0 && (
        <div className="space-y-3">
          <p className="text-[10px] font-bold uppercase tracking-wider text-gold-400">
            {conflicts.length} {t("scenarioDetail.doubleBlind.conflictsToResolve")}
          </p>
          <div className="space-y-2 max-h-[400px] overflow-y-auto">
            {conflicts.map(art => (
              <div key={art.id} className="rounded-xl border border-gold-500/20 bg-gold-500/5 p-3 space-y-2">
                <p className="text-xs font-semibold text-white/80 leading-4">{art.title}</p>
                <div className="flex items-center gap-4 text-[10px] text-white/50">
                  <span>R1 {art.reviewer_1_code ? <span className="font-mono text-white/40">[{art.reviewer_1_code}]</span> : ''} : <span className={art.reviewer_1_status === 'included' ? 'text-brand-300' : 'text-rose-300'}>{art.reviewer_1_status}</span></span>
                  <span>R2 {art.reviewer_2_code ? <span className="font-mono text-white/40">[{art.reviewer_2_code}]</span> : ''} : <span className={art.reviewer_2_status === 'included' ? 'text-brand-300' : 'text-rose-300'}>{art.reviewer_2_status}</span></span>
                </div>
                <div className="flex gap-2">
                  <button onClick={() => decide(art.id, 'included')} disabled={submitting === art.id}
                    className="flex-1 rounded-lg bg-brand-500/20 border border-brand-500/30 text-brand-300 text-[10px] py-1.5 hover:bg-brand-500/30 transition disabled:opacity-50"
                  >{t("scenarioDetail.doubleBlind.includeArbitration")}</button>
                  <button onClick={() => decide(art.id, 'excluded')} disabled={submitting === art.id}
                    className="flex-1 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-[10px] py-1.5 hover:bg-rose-500/20 transition disabled:opacity-50"
                  >{t("scenarioDetail.doubleBlind.excludeArbitration")}</button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Section: Alertes & Living Review ────────────────────────────────────────

function AlertsSection({ scenarioId }: { scenarioId: string }) {
  const { t } = useI18n();
  const [email, setEmail] = React.useState('');
  const [frequency, setFrequency] = React.useState<'daily'|'weekly'|'immediate'>('weekly');
  const [subscribed, setSubscribed] = React.useState(false);
  const [ownerCovered, setOwnerCovered] = React.useState(false);
  const [subscribing, setSubscribing] = React.useState(false);
  const [lrStatus, setLrStatus] = React.useState<any>(null);
  const [lrLoading, setLrLoading] = React.useState(false);

  const handleSubscribe = async () => {
    if (!email.trim()) return;
    setSubscribing(true);
    try {
      const res = await subscribeAlerts(email, scenarioId, frequency);
      setSubscribed(true);
      setOwnerCovered(!!res.owner_set);
    } catch (e: any) {
      alert(e.message);
    } finally {
      setSubscribing(false);
    }
  };

  const handleDryRun = async () => {
    setLrLoading(true);
    try {
      const res = await triggerLivingReview(scenarioId, true);
      setLrStatus(res);
    } catch (e: any) {
      alert(e.message);
    } finally {
      setLrLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <SectionHeader
        icon={<Bell size={14} className="text-brand-400" />}
        title={t("scenarioDetail.alerts.title")}
        subtitle={t("scenarioDetail.alerts.subtitle")}
      />

      {/* Alertes email */}
      <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-4">
        <div className="flex items-center gap-2">
          <Bell size={13} className="text-gold-400"/>
          <h4 className="text-sm font-semibold text-white">{t("scenarioDetail.alerts.emailAlerts")}</h4>
        </div>
        <p className="text-xs text-white/50">{t("scenarioDetail.alerts.emailAlertsHint")}</p>
        {subscribed ? (
          <div className="space-y-2">
            <div className="rounded-xl border border-brand-500/20 bg-brand-500/5 p-3 flex items-center gap-2 text-xs text-brand-300">
              <CheckCircle2 size={13}/>
              {t("scenarioDetail.alerts.subscriptionConfirmedPrefix")} <strong>{email}</strong> {t("scenarioDetail.alerts.subscriptionFrequency")} {frequency}
            </div>
            {ownerCovered && (
              <div className="rounded-xl border border-gold-500/20 bg-gold-500/5 p-2.5 flex items-start gap-2 text-[11px] text-gold-300">
                <Rss size={12} className="mt-0.5 shrink-0"/>
                {t("scenarioDetail.alerts.ownerCovered")}
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-3">
            <input
              type="email" value={email} onChange={e => setEmail(e.target.value)}
              placeholder={t("scenarioDetail.alerts.emailPlaceholder")}
              className="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-white focus:outline-none focus:border-brand-500/50"
            />
            <div className="flex gap-2">
              {(['immediate','daily','weekly'] as const).map(f => (
                <button key={f} onClick={() => setFrequency(f)}
                  className={`px-3 py-1.5 rounded-lg text-[10px] font-medium transition ${
                    frequency === f
                      ? 'bg-brand-700 text-gold-400 font-semibold'
                      : 'text-white/60 hover:text-white hover:bg-white/8'
                  }`}
                >{f === 'immediate' ? t("scenarioDetail.alerts.frequencyImmediate") : f === 'daily' ? t("scenarioDetail.alerts.frequencyDaily") : t("scenarioDetail.alerts.frequencyWeekly")}</button>
              ))}
            </div>
            <button onClick={handleSubscribe} disabled={subscribing || !email.trim()}
              className="flex items-center gap-2 rounded-xl bg-brand-500 hover:bg-brand-400 text-white font-semibold px-4 py-2 text-xs transition disabled:opacity-50"
            >
              {subscribing ? <Loader2 size={12} className="animate-spin"/> : <Bell size={12}/>}
              {t("scenarioDetail.alerts.subscribeButton")}
            </button>
          </div>
        )}
      </div>

      {/* Living Review */}
      <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-4">
        <div className="flex items-center gap-2">
          <Rss size={13} className="text-brand-400"/>
          <h4 className="text-sm font-semibold text-white">{t("scenarioDetail.alerts.livingReviewPipeline")}</h4>
        </div>
        <p className="text-xs text-white/50">
          {t("scenarioDetail.alerts.livingReviewDesc")}
        </p>
        <div className="rounded-xl border border-white/5 bg-white/2 p-3 space-y-1.5 text-[10px] text-white/40">
          {[
            t("scenarioDetail.alerts.stepMultiSource"),
            t("scenarioDetail.alerts.stepInsert"),
            t("scenarioDetail.alerts.stepEmbeddings"),
            t("scenarioDetail.alerts.stepPico"),
            t("scenarioDetail.alerts.stepFulltext"),
            t("scenarioDetail.alerts.stepClustering"),
            t("scenarioDetail.alerts.stepRerank"),
          ].map((step, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="h-1 w-1 rounded-full bg-brand-400"/>
              {step}
            </div>
          ))}
        </div>
        <button onClick={handleDryRun} disabled={lrLoading}
          className="flex items-center gap-2 rounded-xl border border-brand-500/30 bg-brand-500/10 hover:bg-brand-500/20 text-brand-300 font-semibold px-4 py-2 text-xs transition disabled:opacity-50"
        >
          {lrLoading ? <Loader2 size={12} className="animate-spin"/> : <RefreshCw size={12}/>}
          {t("scenarioDetail.alerts.simulateDryRun")}
        </button>
        {lrStatus && (
          <div className="rounded-xl border border-brand-500/20 bg-brand-500/5 p-3 text-xs text-brand-300 space-y-1">
            <div className="font-semibold">{lrStatus.message}</div>
            {lrStatus.scenarios?.map((s: any, i: number) => (
              <div key={i} className="text-[10px] text-white/50">• {s.title} · {s.action}</div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Section: Enrichissement LLM Batch ────────────────────────────────────────

export function EnrichmentSection({ scenarioId }: { scenarioId?: string }) {
  const { t } = useI18n();
  const [status, setStatus] = React.useState<EnrichmentStatus | null>(null);
  const [loadingStatus, setLoadingStatus] = React.useState(false);
  const [running, setRunning] = React.useState<string | null>(null);
  const [lastResult, setLastResult] = React.useState<{ type: string; msg: string; error: boolean } | null>(null);
  const [limit, setLimit] = React.useState(100000);

  const loadStatus = React.useCallback(async () => {
    setLoadingStatus(true);
    try {
      const s = await fetchEnrichmentStatus(scenarioId);
      setStatus(s);
    } catch {/* ignore */} finally {
      setLoadingStatus(false);
    }
  }, [scenarioId]);

  React.useEffect(() => { loadStatus(); }, [loadStatus]);

  const run = async (type: "pico" | "metadata" | "fulltext") => {
    setRunning(type);
    setLastResult(null);
    try {
      let res;
      if (type === "pico") res = await extractPicoBatchGlobal(scenarioId, limit);
      else if (type === "metadata") res = await extractMetadataBatch(scenarioId, limit);
      else res = await fetchFulltextBatch(scenarioId, Math.min(limit, 1000));
      setLastResult({ type, msg: res.message, error: false });
      await loadStatus();
    } catch (e: any) {
      setLastResult({ type, msg: `${t("scenarioDetail.enrichment.errorPrefix")} ${e.message}`, error: true });
    } finally {
      setRunning(null);
    }
  };

  const JOBS = [
    {
      key: "pico" as const,
      label: t("scenarioDetail.enrichment.picoLabel"),
      desc: t("scenarioDetail.enrichment.picoDesc"),
      icon: <Microscope size={15} className="text-brand-400" />,
      stat: status ? `${status.pico.count} / ${status.total} (${status.pico.pct}%)` : "—",
      pct: status ? status.pico.pct : 0,
      color: "bg-brand-500",
    },
    {
      key: "metadata" as const,
      label: t("scenarioDetail.enrichment.metadataLabel"),
      desc: t("scenarioDetail.enrichment.metadataDesc"),
      icon: <Database size={15} className="text-gold-400" />,
      stat: status ? `${status.metadata.count} / ${status.total} (${status.metadata.pct}%)` : "—",
      pct: status ? status.metadata.pct : 0,
      color: "bg-gold-500",
    },
    {
      key: "fulltext" as const,
      label: t("scenarioDetail.enrichment.fulltextLabel"),
      desc: t("scenarioDetail.enrichment.fulltextDesc"),
      icon: <Globe size={15} className="text-forest-300" />,
      stat: status ? `${status.fulltext.count} / ${status.total} (${status.fulltext.pct}%)` : "—",
      pct: status ? status.fulltext.pct : 0,
      color: "bg-forest-400",
    },
  ];

  // Global (Corpus) usage passes no scenarioId; the auto-enrichment note applies
  // corpus-wide, so show it there too (and for user scenarios, as before).
  const showAutoNote = !scenarioId || isUserScenario(scenarioId);

  return (
    <div className="space-y-5">
      <SectionHeader
        icon={<Zap size={16} className="text-gold-400" />}
        title={t("scenarioDetail.enrichment.title")}
        subtitle={t("scenarioDetail.enrichment.subtitle")}
      />

      {showAutoNote && (
        <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 px-4 py-3 flex items-start gap-3">
          <Info size={14} className="text-brand-400 shrink-0 mt-0.5" />
          <div className="text-xs text-brand-200/80 leading-relaxed">
            <strong className="text-brand-300">{t("scenarioDetail.enrichment.autoEnrichmentTitle")}</strong>{t("scenarioDetail.enrichment.autoEnrichmentBody")}
          </div>
        </div>
      )}

      {/* Paramètre lot */}
      <div className="flex items-center gap-3 rounded-2xl border border-white/8 bg-white/3 px-4 py-3">
        <label className="text-xs text-white/50 shrink-0">{t("scenarioDetail.enrichment.batchSizeLabel")}</label>
        <input
          type="number"
          min={5} max={100000} step={100}
          value={limit}
          onChange={e => setLimit(Number(e.target.value))}
          className="w-20 rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-xs text-white text-center focus:outline-none focus:border-brand-500/50"
        />
        <span className="text-xs text-white/30">{t("scenarioDetail.enrichment.articlesPerRun")}</span>
        <button
          onClick={loadStatus}
          disabled={loadingStatus}
          className="ml-auto flex items-center gap-1.5 rounded-xl border border-white/10 px-3 py-1.5 text-xs text-white/50 hover:text-white hover:bg-white/8 transition"
        >
          <RefreshCw size={11} className={loadingStatus ? "animate-spin" : ""} />
          {t("common.refresh")}
        </button>
      </div>

      {/* Cartes de jobs */}
      <div className="grid gap-4 sm:grid-cols-3">
        {JOBS.map(job => (
          <div key={job.key} className="rounded-2xl border border-white/8 bg-white/3 p-4 space-y-3">
            <div className="flex items-start gap-2.5">
              {job.icon}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-white">{job.label}</p>
                <p className="text-xs text-white/40 mt-0.5 leading-relaxed">{job.desc}</p>
              </div>
            </div>

            {/* Barre de progression */}
            <div className="space-y-1">
              <div className="flex justify-between text-[10px] text-white/40">
                <span>{t("scenarioDetail.enrichment.coverage")}</span>
                <span>{job.stat}</span>
              </div>
              <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${job.color}`}
                  style={{ width: `${job.pct}%` }}
                />
              </div>
            </div>

            <button
              onClick={() => run(job.key)}
              disabled={running !== null}
              className="w-full flex items-center justify-center gap-2 rounded-xl bg-brand-500/15 border border-brand-500/25 py-2 text-xs font-medium text-brand-300 hover:bg-brand-500/25 transition disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {running === job.key ? (
                <><Loader2 size={12} className="animate-spin" /> {t("scenarioDetail.enrichment.running")}</>
              ) : (
                <><Zap size={12} /> {t("scenarioDetail.enrichment.launchPrefix")}{limit} {t("scenarioDetail.enrichment.launchSuffix")}</>
              )}
            </button>
          </div>
        ))}
      </div>

      {/* Résultat dernier job */}
      {lastResult && (
        <div className={`flex items-start gap-2.5 rounded-2xl border px-4 py-3 text-xs ${
          lastResult.error
            ? "border-red-500/20 bg-red-500/5 text-red-300"
            : "border-brand-500/20 bg-brand-500/5 text-brand-200"
        }`}>
          {lastResult.error ? <AlertCircle size={13} className="mt-0.5 shrink-0" /> : <CheckCircle2 size={13} className="mt-0.5 shrink-0" />}
          <span><strong className="font-semibold capitalize">{lastResult.type}</strong> : {lastResult.msg}</span>
        </div>
      )}
    </div>
  );
}

// ─── Composite Tabs ──────────────────────────────────────────────────────────


// ─── Seuil de similarite ajustable ─────────────────────────────────────────

function SeuilSection({ scenarioId, onSaved, onThresholdChange }: { scenarioId: string; onSaved?: () => void; onThresholdChange?: (v: number) => void }) {
  const { t } = useI18n();
  const [threshold, setThreshold] = React.useState<number>(0.45);
  const [saving, setSaving] = React.useState(false);
  const [saved, setSaved] = React.useState(false);
  const [rerankStatus, setRerankStatus] = React.useState<string | null>(null);
  const pollRef = React.useRef<ReturnType<typeof setInterval> | null>(null);

  React.useEffect(() => {
    getScenarioSettings(scenarioId)
      .then(s => setThreshold(s.similarity_threshold ?? 0.45))
      .catch(() => {});
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [scenarioId]);

  const handleSave = async () => {
    setSaving(true);
    setSaved(false);
    try {
      await patchScenarioSettings(scenarioId, { similarity_threshold: threshold });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
      onSaved?.();
    } catch {}
    setSaving(false);
  };

  const handleRerank = async () => {
    setRerankStatus(t("scenarioDetail.seuil.launchScoring"));
    try {
      await triggerRerank(scenarioId);
      // Refresh the corpus so it picks up rerank_running and auto-refreshes the
      // relevance (Cohere) badges live, instead of only on a manual page reload.
      onSaved?.();
      if (pollRef.current) clearInterval(pollRef.current);   // pas d'orphelin
      pollRef.current = setInterval(() => {
        getRerankStatus(scenarioId).then(s => {
          if (s.status === "done") {
            if (pollRef.current) clearInterval(pollRef.current);
            setRerankStatus(t("scenarioDetail.seuil.scoringDonePrefix") + (s.updated ?? "?") + t("scenarioDetail.seuil.scoringDoneSuffix"));
            setTimeout(() => setRerankStatus(null), 4000);
            onSaved?.();   // final refresh so the computed scores/badges show
          } else if (s.status === "error") {
            if (pollRef.current) clearInterval(pollRef.current);
            setRerankStatus(t("scenarioDetail.seuil.scoringError"));
          }
        }).catch(() => {
          // Erreur transitoire : nettoyer l'intervalle (sinon poll infini + rejet non géré).
          if (pollRef.current) clearInterval(pollRef.current);
          setRerankStatus(t("scenarioDetail.seuil.scoringError"));
        });
      }, 3000);
    } catch (e: any) {
      setRerankStatus(t("scenarioDetail.seuil.errorPrefix") + e.message);
    }
  };

  return (
    <div className="rounded-2xl border border-white/8 bg-white/2 px-4 py-3 flex flex-wrap items-center gap-4">
      <div className="flex items-center gap-2 text-xs text-white/60">
        <Zap size={12} className="text-brand-400 shrink-0" />
        <span className="font-medium text-white/70">{t("scenarioDetail.seuil.thresholdLabel")}</span>
        <input
          type="range" min={0.1} max={0.9} step={0.05}
          value={threshold}
          onChange={e => { const v = parseFloat(e.target.value); setThreshold(v); onThresholdChange?.(v); }}
          className="w-28 accent-brand-500"
        />
        <span className="font-mono text-brand-300 w-10 text-center">{threshold.toFixed(2)}</span>
      </div>
      <div className="flex items-center gap-2">
        <button onClick={handleSave} disabled={saving}
          className="flex items-center gap-1 rounded-lg bg-brand-500/15 hover:bg-brand-500/25 border border-brand-500/20 text-brand-300 px-2.5 py-1 text-[11px] font-medium transition disabled:opacity-50">
          {saving ? (<Loader2 size={10} className="animate-spin" />) : (<CheckCircle2 size={10} />)}
          {saved ? t("scenarioDetail.seuil.saved") : t("scenarioDetail.seuil.save")}
        </button>
        <button onClick={handleRerank}
          className="flex items-center gap-1 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-white/50 hover:text-white/70 px-2.5 py-1 text-[11px] font-medium transition">
          <RefreshCw size={10} />
          {t("scenarioDetail.seuil.recalculateScores")}
        </button>
        {rerankStatus && (
          <span className="text-[10px] text-gold-400">{rerankStatus}</span>
        )}
      </div>
      <p className="text-[10px] text-white/30 w-full">
        {t("scenarioDetail.seuil.footerMain")}
        <span className="ml-1 text-white/20">{t("scenarioDetail.seuil.footerLegend")}</span>
      </p>
    </div>
  );
}

/** ReviewTab : Corpus + PRISMA + Double-Aveugle (sous-tabs) */
function ReviewTab({ scenarioId, detail }: { scenarioId: string; detail: ScenarioDetail }) {
  const { t } = useI18n();
  const [sub, setSub] = React.useState<"corpus" | "prisma" | "screening">("corpus");
  const [corpusRefreshKey, setCorpusRefreshKey] = React.useState(0);
  const [liveThreshold, setLiveThreshold] = React.useState<number | undefined>(undefined);
  const SUB = [
    { key: "corpus" as const,    label: t("scenarioDetail.review.subCorpus"),    icon: <FileText size={12} /> },
    { key: "prisma" as const,    label: t("scenarioDetail.review.subPrisma"),    icon: <Shield size={12} /> },
    { key: "screening" as const, label: t("scenarioDetail.review.subScreening"), icon: <Users size={12} /> },
  ];
  return (
    <div className="space-y-4">
      <div className="flex gap-1.5 border-b border-white/5 pb-3">
        {SUB.map(s => (
          <button key={s.key} onClick={() => setSub(s.key)}
            className={`flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-medium transition ${
              sub === s.key ? "bg-brand-700 text-gold-400 font-semibold" : "text-white/60 hover:text-white hover:bg-white/8"
            }`}>
            {s.icon}{s.label}
          </button>
        ))}
      </div>
      {sub === "corpus" && (
        <div className="space-y-4">
          <SeuilSection scenarioId={scenarioId} onSaved={() => setCorpusRefreshKey(k => k + 1)} onThresholdChange={setLiveThreshold} />
          <CorpusSection key={corpusRefreshKey} scenarioId={scenarioId} detail={detail} threshold={liveThreshold} />
        </div>
      )}
      {sub === "prisma" && <PrismaSection scenarioId={scenarioId} />}
      {sub === "screening" && <DoubleBlindSection scenarioId={scenarioId} />}
    </div>
  );
}

/** EvidencesSection : Stats corpus + 5 articles PICO + Brief narratif LLM fusionnés */
function EvidencesSection({ scenarioId, detail }: { scenarioId: string; detail: ScenarioDetail }) {
  const { t, lang } = useI18n();
  // ── Evidence Brief data ──────────────────────────────────────────────────────
  const [briefData, setBriefData] = React.useState<EvidenceBriefData | null>(null);
  const [briefLoading, setBriefLoading] = React.useState(true);
  const [briefError, setBriefError] = React.useState<string | null>(null);

  // ── LLM Brief data ───────────────────────────────────────────────────────────
  const [llmData, setLlmData] = React.useState<LlmEvidenceBrief | null>(null);
  const [llmLoading, setLlmLoading] = React.useState(true);
  const [llmError, setLlmError] = React.useState<string | null>(null);
  const [regenerating, setRegenerating] = React.useState(false);
  const [genStatus, setGenStatus] = React.useState<string | null>(null);
  const pollRef = React.useRef<ReturnType<typeof setInterval> | null>(null);

  // ── PDF export ───────────────────────────────────────────────────────────────
  const [exporting, setExporting] = React.useState(false);

  // ── Load both ────────────────────────────────────────────────────────────────
  React.useEffect(() => {
    setBriefLoading(true);
    fetchEvidenceBrief(scenarioId)
      .then(setBriefData)
      .catch(e => setBriefError(e.message))
      .finally(() => setBriefLoading(false));
  }, [scenarioId]);

  const loadLlm = React.useCallback(() => {
    setLlmLoading(true);
    getLlmEvidenceBrief(scenarioId)
      .then(d => { setLlmData(d); setLlmLoading(false); })
      .catch(e => { setLlmError(e.message); setLlmLoading(false); });
    // `lang` in deps: switching the UI language re-requests the brief in that
    // language (the backend regenerates it when the cached language differs).
  }, [scenarioId, lang]);

  React.useEffect(() => {
    loadLlm();
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [loadLlm]);

  // Polling si génération LLM en cours
  React.useEffect(() => {
    if (!llmData || llmData.status !== 'generating') return;
    setGenStatus(t("scenarioDetail.evidences.generatingInProgress"));
    if (pollRef.current) clearInterval(pollRef.current);   // pas d'orphelin
    pollRef.current = setInterval(() => {
      getBriefGenerationStatus(scenarioId).then(s => {
        if (s.status === 'done') {
          if (pollRef.current) clearInterval(pollRef.current);
          setGenStatus(null);
          loadLlm();
        } else if (s.status === 'error') {
          if (pollRef.current) clearInterval(pollRef.current);
          setGenStatus(null);
          setLlmError(s.error ?? t("scenarioDetail.evidences.genError"));
        }
      }).catch(() => {
        // Erreur transitoire : nettoyer l'intervalle + sortir de "génération"
        // (sinon rejet non géré + spinner bloqué à vie).
        if (pollRef.current) clearInterval(pollRef.current);
        setGenStatus(null);
        setLlmError(t("scenarioDetail.evidences.genError"));
      });
    }, 5000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [llmData, scenarioId, loadLlm, t]);

  const handleRegenerate = async () => {
    setRegenerating(true);
    setGenStatus(t("scenarioDetail.evidences.regenLaunched"));
    try {
      await generateEvidenceBrief(scenarioId, true);
      if (pollRef.current) clearInterval(pollRef.current);   // avoid orphaning the effect's poll
      pollRef.current = setInterval(() => {
        getBriefGenerationStatus(scenarioId).then(s => {
          if (s.status === 'done') {
            if (pollRef.current) clearInterval(pollRef.current);
            setGenStatus(null);
            setRegenerating(false);
            loadLlm();
          } else if (s.status === 'error') {
            if (pollRef.current) clearInterval(pollRef.current);
            setGenStatus(null);
            setRegenerating(false);
            setLlmError(s.error ?? t("scenarioDetail.evidences.regenError"));
          }
        }).catch(() => {
          if (pollRef.current) clearInterval(pollRef.current);
          setGenStatus(null);
          setRegenerating(false);
          setLlmError(t("scenarioDetail.evidences.regenError"));
        });
      }, 5000);
    } catch (e: any) {
      setRegenerating(false);
      setGenStatus(null);
      setLlmError(e.message);
    }
  };

  // ── PDF export ───────────────────────────────────────────────────────────────
  const handleExport = async () => {
    if (!briefData) return;
    setExporting(true);
    try {
      const llm = llmData;
      const b = briefData;
      const dups_pdf = b.corpus_stats.duplicates ?? 0;
      // `total` renvoyé par le backend EXCLUT déjà les doublons (COUNT FILTER
      // is_duplicate IS NOT TRUE) — on l'affiche tel quel. Soustraire `dups` ici
      // dédupliquait une SECONDE fois → chiffre plus bas que la carte scénario.
      const uniqueTotal_pdf = b.corpus_stats.total;
      // Le PDF reflète exactement les mêmes chiffres que le panneau Evidences à
      // l'écran : stats clés ET distributions portent sur le SOUS-ENSEMBLE
      // PERTINENT (≥ seuil sémantique), pas le corpus complet.
      const relevant_pdf = b.corpus_stats.relevant ?? uniqueTotal_pdf;
      const rTotal_pdf = relevant_pdf || 1;
      const thr_pdf = b.corpus_stats.threshold ?? 0.45;
      const picoRel_pdf = b.corpus_stats.relevant_with_pico ?? b.corpus_stats.with_pico ?? 0;
      const ftRel_pdf = b.corpus_stats.relevant_with_fulltext ?? b.corpus_stats.with_fulltext ?? 0;
      const html = `<!DOCTYPE html><html><head><meta charset="utf-8">
<title>${t("scenarioDetail.evidences.pdf.docTitlePrefix")} ${detail.title}</title>
<style>
  body{font-family:Georgia,serif;max-width:820px;margin:40px auto;color:#1a1a1a;line-height:1.65;font-size:13px}
  h1{color:#0A3621;border-bottom:3px solid #E3AC3B;padding-bottom:8px;font-size:1.6em}
  h2{color:#0A3621;margin-top:32px;font-size:1.05em;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid #d0e8d8;padding-bottom:4px}
  h3{color:#2d7a52;font-size:.95em;margin-top:16px}
  .meta{color:#666;font-size:.82em;margin-bottom:24px}
  .stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:14px 0}
  .stat{background:#f0f7f3;border:1px solid #aed4bc;border-radius:8px;padding:10px;text-align:center}
  .stat-val{font-size:1.6em;font-weight:700;color:#0A3621}
  .stat-sub{font-size:.7em;color:#888;font-family:monospace}
  .stat-label{font-size:.7em;color:#4d7461;margin-top:3px}
  .dist-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:12px 0}
  .dist-box{background:#f9fafb;border:1px solid #e0e8e3;border-radius:8px;padding:12px}
  .dist-title{font-size:.72em;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#666;margin-bottom:8px}
  .bar-row{display:flex;align-items:center;gap:6px;margin-bottom:4px;font-size:.75em}
  .bar-label{width:110px;color:#555;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .bar-track{flex:1;background:#e0e8e3;border-radius:4px;height:6px}
  .bar-fill-green{background:#2d7a52;border-radius:4px;height:6px}
  .bar-fill-blue{background:#3b82f6;border-radius:4px;height:6px}
  .bar-fill-gold{background:#d97706;border-radius:4px;height:6px}
  .bar-count{width:28px;text-align:right;color:#888;font-family:monospace}
  .article-card{border:1px solid #e0e8e3;border-radius:10px;padding:14px;margin-bottom:14px;page-break-inside:avoid}
  .article-num{font-size:.75em;font-family:monospace;color:#2d7a52;font-weight:700}
  .article-title{font-weight:700;color:#0A3621;font-size:.95em;margin:4px 0}
  .article-meta{font-size:.75em;color:#666;margin-bottom:8px}
  .badge{display:inline-block;background:#d6eade;color:#0A3621;border-radius:4px;padding:1px 5px;font-size:.72em;font-weight:600;margin-right:4px}
  .badge-status-included{background:#d6eade;color:#0A3621}
  .badge-status-pending{background:#f3f4f6;color:#555}
  .badge-status-excluded{background:#fee2e2;color:#991b1b}
  .badge-sim{background:#fef3c7;color:#92400e}
  .pico-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:8px}
  .pico-cell{background:#f9fafb;border:1px solid #e0e8e3;border-radius:6px;padding:8px}
  .pico-label{font-size:.65em;font-weight:700;text-transform:uppercase;color:#888;margin-bottom:3px}
  .pico-val{font-size:.78em;color:#333;line-height:1.4}
  .doi-link{font-size:.75em;color:#2d7a52;text-decoration:none}
  .section-divider{border:none;border-top:2px solid #e3ac3b;margin:28px 0}
  .llm-box{background:#f0f7f3;border:1px solid #aed4bc;border-radius:10px;padding:16px;margin-bottom:16px}
  .llm-label{font-size:.72em;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#2d7a52;margin-bottom:6px}
  .llm-text{font-size:.85em;color:#1a1a1a;line-height:1.6}
  .llm-gold{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:14px;margin-bottom:14px}
  .llm-gold-label{font-size:.72em;font-weight:700;text-transform:uppercase;color:#92400e;margin-bottom:6px}
  ul.llm-list{margin:0;padding-left:18px}
  ul.llm-list li{font-size:.85em;color:#1a1a1a;line-height:1.6;margin-bottom:4px}
  .pico-summary-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:10px}
  .pico-summary-cell{background:#f9fafb;border:1px solid #e0e8e3;border-radius:6px;padding:10px}
  .ref-card{border:1px solid #e0e8e3;border-radius:6px;padding:8px 10px;margin-bottom:6px}
  .ref-title{font-size:.82em;font-weight:600;color:#0A3621}
  .ref-meta{font-size:.73em;color:#666}
  .ref-contrib{font-size:.73em;color:#555;font-style:italic}
  footer{margin-top:40px;padding-top:14px;border-top:1px solid #e0e8e3;font-size:.72em;color:#999;text-align:center}
  .level-badge-forte{background:#d6eade;color:#0A3621;border-radius:6px;padding:3px 10px;font-weight:700;font-size:.85em;display:inline-block}
  .level-badge-mod{background:#fef3c7;color:#92400e;border-radius:6px;padding:3px 10px;font-weight:700;font-size:.85em;display:inline-block}
  .level-badge-faible{background:#f3f4f6;color:#555;border-radius:6px;padding:3px 10px;font-weight:700;font-size:.85em;display:inline-block}
  .grade-badge{background:#0A3621;color:#E3AC3B;border-radius:6px;padding:3px 10px;font-weight:700;font-size:.85em;display:inline-block;margin-left:6px}
  @media print{.no-print{display:none}}
</style></head><body>
<h1>${t("scenarioDetail.evidences.pdf.heading")}</h1>
<div class="meta">
  <strong>${detail.title}</strong> · LiteRev Evidence<br>
  ${t("scenarioDetail.evidences.pdf.subheadingGeneratedOn")} ${new Date().toLocaleDateString(lang === "fr" ? "fr-FR" : "en-US",{year:'numeric',month:'long',day:'numeric'})}
  ${llm?._meta ? ` · ${llm._meta.articles_used} ${t("scenarioDetail.evidences.pdf.subheadingArticles")} ${llm._meta.threshold?.toFixed(2)}` : ''}
</div>

<h2>${t("scenarioDetail.evidences.pdf.corpusStats")}</h2>
<div class="stat-grid">
  <div class="stat"><div class="stat-val">${uniqueTotal_pdf}</div><div class="stat-sub">${dups_pdf>0?`${t("scenarioDetail.evidences.pdf.statDuplicatesPrefix")} ${dups_pdf} ${t("scenarioDetail.evidences.pdf.statDuplicatesSuffix")}`:t("scenarioDetail.evidences.pdf.statUnique")}</div><div class="stat-label">${t("scenarioDetail.evidences.pdf.articlesCorpus")}</div></div>
  <div class="stat"><div class="stat-val" style="color:#2d7a52">${relevant_pdf}</div><div class="stat-sub">${t("scenarioDetail.evidences.pdf.thresholdPrefix")} ${thr_pdf.toFixed(2)} · ${Math.round(relevant_pdf/(uniqueTotal_pdf||1)*100)}% ${t("scenarioDetail.evidences.pdf.ofCorpus")}</div><div class="stat-label">${t("scenarioDetail.evidences.pdf.relevantUsed")}</div></div>
  <div class="stat"><div class="stat-val" style="color:#d97706">${picoRel_pdf}</div><div class="stat-sub">${Math.round(picoRel_pdf/rTotal_pdf*100)}% ${t("scenarioDetail.evidences.pdf.ofRelevant")}</div><div class="stat-label">${t("scenarioDetail.evidences.pdf.picoExtracted")}</div></div>
  <div class="stat"><div class="stat-val" style="color:#3b82f6">${ftRel_pdf}</div><div class="stat-sub">${Math.round(ftRel_pdf/rTotal_pdf*100)}% ${t("scenarioDetail.evidences.pdf.ofRelevant")}</div><div class="stat-label">${t("scenarioDetail.evidences.pdf.fulltext")}</div></div>
</div>
${(b.corpus_stats.included||b.corpus_stats.excluded||(b.corpus_stats.pending??0)) ? `<p class="meta">${t("scenarioDetail.evidences.pdf.screeningPrefix")} <strong>${b.corpus_stats.included}</strong> ${t("scenarioDetail.evidences.pdf.screeningIncluded")} · <strong>${b.corpus_stats.excluded}</strong> ${t("scenarioDetail.evidences.pdf.screeningExcluded")} · <strong>${b.corpus_stats.pending ?? Math.max(0, uniqueTotal_pdf - b.corpus_stats.included - b.corpus_stats.excluded)}</strong> ${t("scenarioDetail.evidences.pdf.screeningPending")}</p>` : ''}
${b.corpus_stats.year_min && b.corpus_stats.year_max ? `<p class="meta">${t("scenarioDetail.evidences.pdf.coveragePrefix")} <strong>${b.corpus_stats.year_min} – ${b.corpus_stats.year_max}</strong>${b.corpus_stats.avg_citations != null ? ` · ${t("scenarioDetail.evidences.pdf.avgCitations")} <strong>${b.corpus_stats.avg_citations.toFixed(1)}</strong>` : ''}</p>` : ''}

<div class="dist-grid">
  <div class="dist-box">
    <div class="dist-title">${t("scenarioDetail.evidences.pdf.studyTypes")}</div>
    ${(()=>{const top=b.study_design_distribution;const rem=relevant_pdf-top.reduce((s,d)=>s+d.count,0);const rows=rem>0?[...top,{design:t("scenarioDetail.evidences.pdf.other"),count:rem}]:top;return rows.map(d=>`<div class="bar-row"><span class="bar-label">${d.design}</span><div class="bar-track"><div class="bar-fill-green" style="width:${Math.round(d.count/rTotal_pdf*100)}%"></div></div><span class="bar-count">${d.count}</span></div>`).join('');})()}
  </div>
  <div class="dist-box">
    <div class="dist-title">${t("scenarioDetail.evidences.pdf.sources")}</div>
    ${(()=>{const top=(b.source_distribution??[]);const rem=relevant_pdf-top.reduce((s,d)=>s+d.count,0);const rows=rem>0?[...top,{source:t("scenarioDetail.evidences.pdf.other"),count:rem}]:top;return rows.map(d=>`<div class="bar-row"><span class="bar-label">${d.source}</span><div class="bar-track"><div class="bar-fill-blue" style="width:${Math.round(d.count/rTotal_pdf*100)}%"></div></div><span class="bar-count">${d.count}</span></div>`).join('');})()}
  </div>
  <div class="dist-box">
    <div class="dist-title">${t("scenarioDetail.evidences.pdf.evidenceLevels")}</div>
    ${(b.evidence_level_distribution??[]).slice(0,6).map(d=>`<div class="bar-row"><span class="bar-label">${d.level}</span><div class="bar-track"><div class="bar-fill-gold" style="width:${Math.round(d.count/rTotal_pdf*100)}%"></div></div><span class="bar-count">${d.count}</span></div>`).join('')}
  </div>
</div>

${llm && (llm.executive_summary || (llm.key_findings?.length ?? 0) > 0) ? `
<hr class="section-divider">
<h2>${t("scenarioDetail.evidences.pdf.narrativeBrief")}</h2>
${llm._meta ? `<p class="meta">${llm._meta.articles_used} ${t("scenarioDetail.evidences.pdf.articlesAnalyzed")} ${llm._meta.threshold?.toFixed(2)} · ${llm._meta.human_validated} ${t("scenarioDetail.evidences.pdf.humanValidated")} · ${llm._meta.year_range}</p>` : ''}
${(llm.evidence_level||llm.grade_recommendation) ? `<p>${llm.evidence_level?`<span class="level-badge-${/fort|strong|élev|elev|high/.test(llm.evidence_level.toLowerCase())?'forte':/mod/.test(llm.evidence_level.toLowerCase())?'mod':'faible'}">${t("scenarioDetail.evidences.pdf.level")} ${llm.evidence_level}</span>`:''} ${llm.grade_recommendation?`<span class="grade-badge">${t("scenarioDetail.evidences.pdf.grade")} ${llm.grade_recommendation}</span>`:''}</p>` : ''}
${llm.executive_summary ? `<div class="llm-box"><div class="llm-label">${t("scenarioDetail.evidences.pdf.executiveSummary")}</div><div class="llm-text">${llm.executive_summary}</div></div>` : ''}
${llm.clinical_context ? `<h3>${t("scenarioDetail.evidences.pdf.clinicalContext")}</h3><p class="llm-text">${llm.clinical_context}</p>` : ''}
${(llm.key_findings?.length??0)>0 ? `<h3>${t("scenarioDetail.evidences.pdf.keyFindings")}</h3><ul class="llm-list">${llm.key_findings!.map(f=>`<li>${f}</li>`).join('')}</ul>` : ''}
${llm.evidence_synthesis ? `<h3>${t("scenarioDetail.evidences.pdf.evidenceSynthesis")}</h3><p class="llm-text">${llm.evidence_synthesis}</p>` : ''}
${(llm.population_summary||llm.intervention_summary||llm.outcome_summary) ? `<div class="pico-summary-grid">${llm.population_summary?`<div class="pico-summary-cell"><div class="pico-label">${t("scenarioDetail.evidences.pdf.population")}</div><div class="llm-text">${llm.population_summary}</div></div>`:''} ${llm.intervention_summary?`<div class="pico-summary-cell"><div class="pico-label">${t("scenarioDetail.evidences.pdf.intervention")}</div><div class="llm-text">${llm.intervention_summary}</div></div>`:''} ${llm.outcome_summary?`<div class="pico-summary-cell"><div class="pico-label">${t("scenarioDetail.evidences.pdf.outcome")}</div><div class="llm-text">${llm.outcome_summary}</div></div>`:''}</div>` : ''}
${(llm.recommended_actions?.length??0)>0 ? `<div class="llm-gold"><div class="llm-gold-label">${t("scenarioDetail.evidences.pdf.recommendedActions")}</div><ul class="llm-list">${llm.recommended_actions!.map(a=>`<li>${a}</li>`).join('')}</ul></div>` : ''}
${llm.clinical_implications ? `<h3>${t("scenarioDetail.evidences.pdf.clinicalImplications")}</h3><p class="llm-text">${llm.clinical_implications}</p>` : ''}
${(llm.implementation_recommendations?.length??0)>0 ? `<h3>${t("scenarioDetail.evidences.pdf.implementationRecommendations")}</h3><ul class="llm-list">${llm.implementation_recommendations!.map(r=>`<li>${r}</li>`).join('')}</ul>` : ''}
${((llm.limitations?.length??0)>0||(llm.research_gaps?.length??0)>0) ? `<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px">${(llm.limitations?.length??0)>0?`<div><h3>${t("scenarioDetail.evidences.pdf.limitations")}</h3><ul class="llm-list">${llm.limitations!.map(l=>`<li>${l}</li>`).join('')}</ul></div>`:''} ${(llm.research_gaps?.length??0)>0?`<div><h3>${t("scenarioDetail.evidences.pdf.researchGaps")}</h3><ul class="llm-list">${llm.research_gaps!.map(g=>`<li>${g}</li>`).join('')}</ul></div>`:''}</div>` : ''}
${llm.future_research ? `<h3>${t("scenarioDetail.evidences.pdf.futureResearch")}</h3><p class="llm-text">${llm.future_research}</p>` : ''}

` : ''}

<footer>
  LiteRev Evidence · ${new Date().getFullYear()} · ${t("scenarioDetail.evidences.pdf.footer")}
</footer>
</body></html>`;
      const blob = new Blob([html], {type:'text/html;charset=utf-8'});
      const url = URL.createObjectURL(blob);
      const win = window.open(url, '_blank');
      if (win) setTimeout(() => { win.print(); }, 800);
      URL.revokeObjectURL(url);
    } finally {
      setExporting(false);
    }
  };

  // Les distributions (types d'étude / sources / niveaux de preuve) sont calculées
  // sur le SOUS-ENSEMBLE PERTINENT (≥ seuil), pas le corpus entier : on divise donc
  // par `relevant`, sinon les % étaient minuscules et le reste « Autres » gonflait
  // artificiellement (corpus entier − comptes pertinents).
  const relevant = briefData ? ((briefData.corpus_stats.relevant ?? briefData.corpus_stats.total) || 1) : 1;
  const meta = llmData?._meta;
  const hasLlmContent = !!(llmData?.executive_summary || (llmData?.key_findings?.length ?? 0) > 0);

  return (
    <div className="space-y-6">
      {/* ─── HEADER ─────────────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <SectionHeader
            icon={<BookOpen size={14} className="text-brand-400" />}
            title={t("scenarioDetail.evidences.title")}
            subtitle={t("scenarioDetail.evidences.subtitle")}
          />
          {meta && (
            <p className="text-[10px] text-white/35 mt-1">
              {meta.articles_used} {t("scenarioDetail.evidences.metaArticlesAnalyzed")} {meta.threshold?.toFixed(2)} · {meta.human_validated} {t("scenarioDetail.evidences.metaHumanValidated")}
              {meta.year_range ? ` · ${meta.year_range}` : ''}
            </p>
          )}
          {!meta && briefData && (
            <p className="text-[10px] text-white/35 mt-1">
              {t("scenarioDetail.evidences.generatedOn")} {new Date(briefData.generated_at).toLocaleDateString(lang === "fr" ? "fr-FR" : "en-US",{year:'numeric',month:'long',day:'numeric'})}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {genStatus && (
            <span className="text-[10px] text-gold-400 flex items-center gap-1">
              <Loader2 size={10} className="animate-spin" />{genStatus}
            </span>
          )}
          <button onClick={handleRegenerate} disabled={regenerating || llmLoading}
            className="flex items-center gap-1.5 rounded-xl border border-brand-500/30 bg-brand-500/10 hover:bg-brand-500/20 text-brand-300 font-medium px-3 py-1.5 text-xs transition disabled:opacity-50">
            {regenerating ? (<Loader2 size={11} className="animate-spin" />) : (<RefreshCw size={11} />)}
            {t("scenarioDetail.evidences.regenerateBrief")}
          </button>
          {briefData && (
            <button onClick={handleExport} disabled={exporting}
              className="flex items-center gap-2 rounded-2xl bg-brand-500 hover:bg-brand-400 text-white font-semibold px-4 py-2 text-xs transition disabled:opacity-50">
              {exporting ? <Loader2 size={12} className="animate-spin"/> : <Download size={12}/>}
              {exporting ? t("scenarioDetail.evidences.generating") : t("scenarioDetail.evidences.exportPdf")}
            </button>
          )}
        </div>
      </div>

      {/* ─── BANNIÈRE AVERTISSEMENT ──────────────────────────────────────────── */}
      {briefData && briefData.corpus_stats.included === 0 && (
        <div className="rounded-2xl border border-gold-500/20 bg-gold-500/5 px-4 py-3 flex items-start gap-3">
          <AlertTriangle size={14} className="text-gold-400 shrink-0 mt-0.5" />
          <div className="text-xs text-gold-200/80 leading-relaxed">
            <strong className="text-gold-300">{t("scenarioDetail.evidences.noHumanValidatedTitle")}</strong>{t("scenarioDetail.evidences.noHumanValidatedBody")}
          </div>
        </div>
      )}

      {/* ─── STATS CORPUS ────────────────────────────────────────────────────── */}
      {briefLoading && <LoadingSpinner text={t("scenarioDetail.evidences.loadingCorpusStats")} />}
      {briefError && <ErrorBox message={briefError} />}
      {briefData && (
        <>
          {/* KPIs — l'analyse (Evidence Brief, PICO, modèle) repose sur les
              articles PERTINENTS (au-dessus du seuil sémantique). On montre le
              corpus total pour le contexte, puis tout le reste sur les pertinents.
              Les compteurs de screening (inclus/exclus/en attente) vivent dans
              l'onglet Revue, pas ici. */}
          {(()=>{
            const cs = briefData.corpus_stats;
            const dups = cs.duplicates ?? 0;
            // `total` EXCLUT déjà les doublons côté backend (COUNT FILTER
            // is_duplicate IS NOT TRUE) → même valeur que la carte scénario.
            // (Soustraire `dups` de nouveau ici = double déduplication.)
            const uniqueTotal = cs.total;
            const relevant = cs.relevant ?? uniqueTotal;
            const rTotal = relevant || 1;
            const thr = cs.threshold ?? 0.45;
            const boxes = [
              {label:t("scenarioDetail.evidences.statCorpusArticles"), value:uniqueTotal,                            color:'text-white',       sub:dups>0?`${t("scenarioDetail.evidences.subDuplicatesPrefix")} ${dups} ${t("scenarioDetail.evidences.subDuplicatesSuffix")}`:t("scenarioDetail.evidences.subUnique")},
              {label:t("scenarioDetail.evidences.statRelevantUsed"), value:relevant,                           color:'text-brand-300',   sub:`${t("scenarioDetail.evidences.subThresholdPrefix")} ${thr} · ${Math.round(relevant/(uniqueTotal||1)*100)}% ${t("scenarioDetail.evidences.subOfCorpus")}`},
              {label:t("scenarioDetail.evidences.statPicoExtracted"),     value:cs.relevant_with_pico ?? cs.with_pico,  color:'text-gold-400',    sub:`${Math.round((cs.relevant_with_pico ?? cs.with_pico)/rTotal*100)}% ${t("scenarioDetail.evidences.subOfRelevant")}`},
              {label:t("scenarioDetail.evidences.statFulltext"),    value:cs.relevant_with_fulltext ?? cs.with_fulltext ?? 0, color:'text-blue-300', sub:`${Math.round((cs.relevant_with_fulltext ?? cs.with_fulltext ?? 0)/rTotal*100)}% ${t("scenarioDetail.evidences.subOfRelevant")}`},
            ];
            return (
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5">
                {boxes.map(s=>(
                  <div key={s.label} className="rounded-2xl border border-white/5 bg-white/2 p-3 text-center">
                    <div className={`text-xl font-bold ${s.color}`}>{s.value}</div>
                    {s.sub && <div className="text-[9px] text-white/30 font-mono">{s.sub}</div>}
                    <div className="text-[9px] text-white/35 mt-0.5 leading-tight">{s.label}</div>
                  </div>
                ))}
              </div>
            );
          })()}

          {/* Couverture temporelle */}
          {briefData.corpus_stats.year_min && briefData.corpus_stats.year_max && (
            <div className="flex flex-wrap gap-4 text-xs text-white/50">
              <span>{t("scenarioDetail.evidences.coverage")} <span className="text-white/70 font-semibold">{briefData.corpus_stats.year_min} – {briefData.corpus_stats.year_max}</span></span>
              {briefData.corpus_stats.avg_citations != null && <span>{t("scenarioDetail.evidences.avgCitations")} <span className="text-white/70 font-semibold">{briefData.corpus_stats.avg_citations.toFixed(1)}</span></span>}
              {briefData.corpus_stats.max_citations != null && <span>{t("scenarioDetail.evidences.max")} <span className="text-white/70 font-semibold">{briefData.corpus_stats.max_citations}</span></span>}
            </div>
          )}

          {/* Distributions */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="rounded-2xl border border-white/5 bg-white/2 p-4 space-y-2">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/40">{t("scenarioDetail.evidences.studyTypes")}</p>
              <div className="space-y-1.5">
                {(() => {
                  // Toutes les catégories + « Autres » (reste) pour que la somme corresponde au corpus
                  const top = briefData.study_design_distribution;
                  const remainder = relevant - top.reduce((s,d)=>s+d.count,0);
                  const rows = remainder > 0 ? [...top, {design:t("scenarioDetail.evidences.other"), count:remainder}] : top;
                  return rows.map(d=>(
                  <div key={d.design} className="flex items-center gap-2 text-[10px]">
                    <span className="w-28 text-white/60 truncate">{d.design}</span>
                    <div className="flex-1 h-1 bg-white/5 rounded-full overflow-hidden">
                      <div className="h-full bg-brand-500 rounded-full" style={{width:`${Math.round(d.count/relevant*100)}%`}}/>
                    </div>
                    <span className="w-7 text-right text-white/40 font-mono">{d.count}</span>
                  </div>
                  ));
                })()}
              </div>
            </div>
            <div className="rounded-2xl border border-white/5 bg-white/2 p-4 space-y-2">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/40">{t("scenarioDetail.evidences.sources")}</p>
              <div className="space-y-1.5">
                {(() => {
                  const top = (briefData.source_distribution ?? []);
                  const remainder = relevant - top.reduce((s,d)=>s+d.count,0);
                  const rows = remainder > 0 ? [...top, {source:t("scenarioDetail.evidences.other"), count:remainder}] : top;
                  return rows.map(d=>(
                  <div key={d.source} className="flex items-center gap-2 text-[10px]">
                    <span className="w-28 text-white/60 truncate capitalize">{d.source}</span>
                    <div className="flex-1 h-1 bg-white/5 rounded-full overflow-hidden">
                      <div className="h-full bg-blue-500/60 rounded-full" style={{width:`${Math.round(d.count/relevant*100)}%`}}/>
                    </div>
                    <span className="w-7 text-right text-white/40 font-mono">{d.count}</span>
                  </div>
                  ));
                })()}
              </div>
            </div>
            <div className="rounded-2xl border border-white/5 bg-white/2 p-4 space-y-2">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/40">{t("scenarioDetail.evidences.evidenceLevels")}</p>
              <div className="space-y-1.5">
                {(briefData.evidence_level_distribution ?? []).slice(0,6).map(d=>(
                  <div key={d.level} className="flex items-center gap-2 text-[10px]">
                    <span className="w-28 text-white/60 truncate capitalize">{d.level}</span>
                    <div className="flex-1 h-1 bg-white/5 rounded-full overflow-hidden">
                      <div className="h-full bg-gold-400/60 rounded-full" style={{width:`${Math.round(d.count/relevant*100)}%`}}/>
                    </div>
                    <span className="w-7 text-right text-white/40 font-mono">{d.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}

      {/* ─── SÉPARATEUR ─────────────────────────────────────────────────────── */}
      {briefData && (hasLlmContent || llmLoading) && (
        <div className="border-t border-gold-500/20 pt-2">
          <p className="text-[10px] font-bold uppercase tracking-wider text-gold-400/60">{t("scenarioDetail.evidences.narrativeBriefLlm")}</p>
        </div>
      )}

      {/* ─── BRIEF LLM ──────────────────────────────────────────────────────── */}
      {llmLoading && <LoadingSpinner text={t("scenarioDetail.evidences.loadingBrief")} />}
      {llmError && <ErrorBox message={llmError} />}
      {llmData && llmData.status === 'empty' && (
        <div className="rounded-2xl border border-slate-500/20 bg-slate-500/5 px-4 py-3">
          <p className="text-xs text-slate-300/80">
            <strong className="text-slate-200">{t("scenarioDetail.evidences.noArticleForBriefTitle")}</strong> : {llmData.message ?? t("scenarioDetail.evidences.noArticleForBriefDefault")}
          </p>
        </div>
      )}
      {llmData && llmData.status === 'generating' && (
        <div className="rounded-2xl border border-gold-500/20 bg-gold-500/5 px-5 py-4 flex items-start gap-3">
          <Loader2 size={14} className="text-gold-400 animate-spin shrink-0 mt-0.5" />
          <div className="text-xs text-gold-200/80">
            <strong className="text-gold-300">{t("scenarioDetail.evidences.briefGeneratingTitle")}</strong> : {llmData.message ?? t("scenarioDetail.evidences.briefGeneratingDefault")}
          </div>
        </div>
      )}
      {llmData && !llmData.status && !hasLlmContent && (
        <div className="rounded-2xl border border-white/10 bg-white/3 px-4 py-6 text-center text-xs text-white/40">
          {t("scenarioDetail.evidences.noBriefAvailable")}
        </div>
      )}
      {llmData && hasLlmContent && (
        <div className="space-y-5">
          {/* Niveau de preuve + Grade */}
          {(llmData.evidence_level || llmData.grade_recommendation) && (
            <div className="flex flex-wrap gap-2">
              {llmData.evidence_level && (
                <span className={`rounded-xl px-3 py-1 text-xs font-semibold border ${
                  llmData.evidence_level.toLowerCase().includes('fort') ? 'bg-brand-500/15 border-brand-500/30 text-brand-300' :
                  llmData.evidence_level.toLowerCase().includes('mod') ? 'bg-gold-500/15 border-gold-500/30 text-gold-300' :
                  'bg-white/5 border-white/10 text-white/50'
                }`}>{t("scenarioDetail.evidences.levelPrefix")} {llmData.evidence_level}</span>
              )}
              {llmData.grade_recommendation && (
                <span className="rounded-xl px-3 py-1 text-xs font-semibold border bg-brand-500/10 border-brand-500/20 text-brand-200">
                  {t("scenarioDetail.evidences.gradePrefix")} {llmData.grade_recommendation}
                </span>
              )}
            </div>
          )}
          {/* Résumé exécutif */}
          {llmData.executive_summary && (
            <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 p-4">
              <p className="text-[10px] font-bold uppercase tracking-wider text-brand-400 mb-2">{t("scenarioDetail.evidences.executiveSummary")}</p>
              <p className="text-sm text-white/80 leading-relaxed">{llmData.executive_summary}</p>
            </div>
          )}
          {/* Contexte clinique */}
          {llmData.clinical_context && (
            <div className="space-y-1">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">{t("scenarioDetail.evidences.clinicalContext")}</p>
              <p className="text-xs text-white/65 leading-relaxed">{llmData.clinical_context}</p>
            </div>
          )}
          {/* Résultats clés */}
          {llmData.key_findings && llmData.key_findings.length > 0 && (
            <div className="space-y-2">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">{t("scenarioDetail.evidences.keyFindings")}</p>
              <ul className="space-y-1.5">
                {llmData.key_findings.map((f, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-white/70">
                    <span className="shrink-0 mt-0.5 h-4 w-4 rounded-full bg-brand-500/20 border border-brand-500/30 flex items-center justify-center text-[9px] font-bold text-brand-300">{i+1}</span>
                    {f}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {/* Synthèse des évidences */}
          {llmData.evidence_synthesis && (
            <div className="space-y-1">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">{t("scenarioDetail.evidences.evidenceSynthesis")}</p>
              <p className="text-xs text-white/65 leading-relaxed whitespace-pre-line">{llmData.evidence_synthesis}</p>
            </div>
          )}
          {/* PICO Summary */}
          {(llmData.population_summary || llmData.intervention_summary || llmData.outcome_summary) && (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {llmData.population_summary && (
                <div className="rounded-xl border border-white/8 bg-white/3 p-3">
                  <p className="text-[9px] font-bold uppercase tracking-wider text-white/30 mb-1">{t("scenarioDetail.evidences.population")}</p>
                  <p className="text-xs text-white/60 leading-relaxed">{llmData.population_summary}</p>
                </div>
              )}
              {llmData.intervention_summary && (
                <div className="rounded-xl border border-white/8 bg-white/3 p-3">
                  <p className="text-[9px] font-bold uppercase tracking-wider text-white/30 mb-1">{t("scenarioDetail.evidences.intervention")}</p>
                  <p className="text-xs text-white/60 leading-relaxed">{llmData.intervention_summary}</p>
                </div>
              )}
              {llmData.outcome_summary && (
                <div className="rounded-xl border border-white/8 bg-white/3 p-3">
                  <p className="text-[9px] font-bold uppercase tracking-wider text-white/30 mb-1">{t("scenarioDetail.evidences.outcome")}</p>
                  <p className="text-xs text-white/60 leading-relaxed">{llmData.outcome_summary}</p>
                </div>
              )}
            </div>
          )}
          {/* Actions recommandées */}
          {llmData.recommended_actions && llmData.recommended_actions.length > 0 && (
            <div className="rounded-2xl border border-gold-500/20 bg-gold-500/5 p-4 space-y-2">
              <p className="text-[10px] font-bold uppercase tracking-wider text-gold-400">{t("scenarioDetail.evidences.recommendedActions")}</p>
              <ul className="space-y-1.5">
                {llmData.recommended_actions.map((a, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-gold-200/80">
                    <Zap size={10} className="shrink-0 mt-0.5 text-gold-400" />
                    {a}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {/* Implications cliniques */}
          {llmData.clinical_implications && (
            <div className="space-y-1">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">{t("scenarioDetail.evidences.clinicalImplications")}</p>
              <p className="text-xs text-white/65 leading-relaxed">{llmData.clinical_implications}</p>
            </div>
          )}
          {/* Recommandations d'implémentation */}
          {llmData.implementation_recommendations && llmData.implementation_recommendations.length > 0 && (
            <div className="space-y-2">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">{t("scenarioDetail.evidences.implementationRecommendations")}</p>
              <ul className="space-y-1">
                {llmData.implementation_recommendations.map((r, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-white/60">
                    <CheckCircle2 size={10} className="shrink-0 mt-0.5 text-brand-400" />
                    {r}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {/* Limites + Lacunes */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {llmData.limitations && llmData.limitations.length > 0 && (
              <div className="space-y-1.5">
                <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">{t("scenarioDetail.evidences.limitations")}</p>
                <ul className="space-y-1">
                  {llmData.limitations.map((l, i) => (
                    <li key={i} className="flex items-start gap-1.5 text-xs text-white/50">
                      <AlertCircle size={9} className="shrink-0 mt-0.5 text-rose-400" />
                      {l}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {llmData.research_gaps && llmData.research_gaps.length > 0 && (
              <div className="space-y-1.5">
                <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">{t("scenarioDetail.evidences.researchGaps")}</p>
                <ul className="space-y-1">
                  {llmData.research_gaps.map((g, i) => (
                    <li key={i} className="flex items-start gap-1.5 text-xs text-white/50">
                      <Search size={9} className="shrink-0 mt-0.5 text-gold-400" />
                      {g}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
          {/* Recherches futures */}
          {llmData.future_research && (
            <div className="space-y-1">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">{t("scenarioDetail.evidences.futureResearch")}</p>
              <p className="text-xs text-white/55 leading-relaxed">{llmData.future_research}</p>
            </div>
          )}

        </div>
      )}
    </div>
  );
}

/** EvidenceTab : Evidences fusionnées + Tableau PICO (sous-tabs) */
function EvidenceTab({ scenarioId, detail }: { scenarioId: string; detail: ScenarioDetail }) {
  const { t } = useI18n();
  const [sub, setSub] = React.useState<"evidences" | "pico">("evidences");
  const SUB = [
    { key: "evidences" as const, label: t("scenarioDetail.evidenceTab.subEvidences"), icon: <BookOpen size={12} /> },
    { key: "pico" as const,      label: t("scenarioDetail.evidenceTab.subPicoTable"), icon: <Table2 size={12} /> },
  ];
  return (
    <div className="space-y-4">
      <div className="flex gap-1.5 border-b border-white/5 pb-3">
        {SUB.map(s => (
          <button key={s.key} onClick={() => setSub(s.key)}
            className={`flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-medium transition ${
              sub === s.key ? "bg-brand-700 text-gold-400 font-semibold" : "text-white/60 hover:text-white hover:bg-white/8"
            }`}>
            {s.icon}{s.label}
          </button>
        ))}
      </div>
      {sub === "evidences" && <EvidencesSection scenarioId={scenarioId} detail={detail} />}
      {sub === "pico" && <PicoSection scenarioId={scenarioId} />}
    </div>
  );
}

/** VizTab : Clustering + Knowledge Graph (sous-tabs) */
function VizTab({ scenarioId }: { scenarioId: string }) {
  const { t } = useI18n();
  const [sub, setSub] = React.useState<"clustering" | "kg">("clustering");
  const SUB = [
    { key: "clustering" as const, label: t("scenarioDetail.vizTab.subClustering"), icon: <Layers size={12} /> },
    { key: "kg" as const,         label: t("scenarioDetail.vizTab.subKnowledgeGraph"),     icon: <Network size={12} /> },
  ];
  return (
    <div className="space-y-4">
      <div className="flex gap-1.5 border-b border-white/5 pb-3">
        {SUB.map(s => (
          <button key={s.key} onClick={() => setSub(s.key)}
            className={`flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-medium transition ${
              sub === s.key ? "bg-brand-700 text-gold-400 font-semibold" : "text-white/60 hover:text-white hover:bg-white/8"
            }`}>
            {s.icon}{s.label}
          </button>
        ))}
      </div>
      {sub === "clustering" && <ClusteringSection scenarioId={scenarioId} />}
      {sub === "kg" && <KnowledgeGraphSection scenarioId={scenarioId} />}
    </div>
  );
}

/** VariablesModelTab : Variables & Données + Modèle prédictif (sous-tabs) */
function VariablesModelTab({ scenarioId, detail, initialSub }: { scenarioId: string; detail: ScenarioDetail; initialSub?: "variables" | "monitor" }) {
  const { t } = useI18n();
  const [sub, setSub] = React.useState<"variables" | "monitor" | "seir">(initialSub ?? "seir");
  // Onglet SEIR TOUJOURS affiché et en PREMIER. Quand la projection n'est pas applicable
  // (maladie non transmissible / aucune variable dérivable par SEIR), SeirModelView
  // l'énonce clairement plutôt que de masquer l'onglet.
  const SUB = [
    { key: "seir" as const, label: t("scenarioDetail.variablesModelTab.subSeir"), icon: <TrendingUp size={12} /> },
    { key: "variables" as const, label: t("scenarioDetail.variablesModelTab.subData"), icon: <Database size={12} /> },
    { key: "monitor" as const, label: t("scenarioDetail.variablesModelTab.subModel"), icon: <Brain size={12} /> },
  ];
  return (
    <div className="space-y-4">
      <div className="flex gap-1.5 border-b border-white/5 pb-3">
        {SUB.map(s => (
          <button key={s.key} onClick={() => setSub(s.key)}
            className={`flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-medium transition ${
              sub === s.key ? "bg-brand-700 text-gold-400 font-semibold" : "text-white/60 hover:text-white hover:bg-white/8"
            }`}>
            {s.icon}{s.label}
          </button>
        ))}
      </div>
      {sub === "variables" && <VariablesSection detail={detail} scenarioId={scenarioId} onGoToModel={() => setSub("monitor")} />}
      {sub === "monitor" && <ModelMonitorSection scenarioId={scenarioId} />}
      {sub === "seir" && <SeirModelView scenarioId={scenarioId} />}
    </div>
  );
}

// ─── Tableau de bord du modèle (vue clinicien) ────────────────────────────────
// Statut clair (pastille + libellé + couleur — jamais la couleur seule, accessibilité).
const _DASH_STATUS = {
  green:  { dot: "bg-emerald-400", text: "text-emerald-300", bg: "bg-emerald-500/10", border: "border-emerald-500/30", ring: "ring-emerald-400/40" },
  orange: { dot: "bg-amber-400",   text: "text-amber-300",   bg: "bg-amber-500/10",   border: "border-amber-500/30",   ring: "ring-amber-400/40" },
  red:    { dot: "bg-rose-400",    text: "text-rose-300",     bg: "bg-rose-500/10",    border: "border-rose-500/30",    ring: "ring-rose-400/40" },
  unavailable: { dot: "bg-white/30", text: "text-white/60", bg: "bg-white/5", border: "border-white/10", ring: "ring-white/20" },
} as const;
type DashStatus = keyof typeof _DASH_STATUS;

function ModelDashboard({ scenarioId, run, monitor, spec }: { scenarioId: string; run: ModelRun; monitor: ModelMonitor | null; spec: ModelSpecResponse | null }) {
  const { t, lang } = useI18n();
  const locale = lang === "fr" ? "fr-FR" : "en-US";
  const fmtDate = (d?: string | null) => (d ? new Date(d).toLocaleString(locale, { dateStyle: "medium", timeStyle: "short" }) : "—");
  const [exporting, setExporting] = useState(false);
  const [exportingXlsx, setExportingXlsx] = useState(false);
  const _download = (blob: Blob, name: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = name; a.click();
    URL.revokeObjectURL(url);
  };
  const doExport = async () => {
    setExporting(true);
    try {
      const bundle = await exportModelBundle(scenarioId);
      _download(new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" }), `model_${scenarioId}.json`);
    } catch { /* silencieux : l'export ne doit jamais casser la page */ }
    finally { setExporting(false); }
  };
  const doExportXlsx = async () => {
    setExportingXlsx(true);
    try {
      _download(await exportModelXlsx(scenarioId), `model_${scenarioId}.xlsx`);
    } catch { /* silencieux */ }
    finally { setExportingXlsx(false); }
  };

  const sc: DashStatus = (monitor?.status_color as DashStatus) in _DASH_STATUS ? (monitor!.status_color as DashStatus) : "unavailable";
  const cfg = _DASH_STATUS[sc];
  const outcomeName = spec?.outcome?.name || monitor?.outcome || t("scenarioDetail.model.dashboard.outcome");
  const unit = monitor?.unit || spec?.outcome?.unit || "";
  const hasPrediction = monitor?.status === "ready" && typeof monitor?.value === "number";
  const valueLabel = hasPrediction
    ? (monitor!.kind === "probability" ? `${(monitor!.value! * 100).toFixed(0)}%` : monitor!.value!.toLocaleString(locale))
    : "—";
  const metricKey = run.metric ?? monitor?.model?.metric;
  const metricVal = (metricKey && run.metrics && typeof run.metrics[metricKey] === "number") ? run.metrics[metricKey] : null;

  // Facteurs clés (importances par VARIABLE, noms lisibles) — repli sur les brutes.
  const nameByMachine: Record<string, string> = {};
  (spec?.features ?? []).forEach(f => { if (f.machine_name) nameByMachine[f.machine_name] = f.name || f.machine_name; });
  const byVar = (run.summary?.importances_by_variable ?? null) as Record<string, number> | null;
  const drivers = (byVar && Object.keys(byVar).length > 0
    ? Object.entries(byVar).map(([mn, v]) => ({ label: nameByMachine[mn] ?? mn, value: Number(v) }))
    : (run.feature_importances ?? []).map(fi => {
        const base = fi.feature.includes("__") ? fi.feature.split("__").slice(1).join("__") : fi.feature;
        return { label: nameByMachine[base] ?? base, value: fi.importance };
      })
  ).sort((a, b) => b.value - a.value).slice(0, 5);
  const maxDriver = drivers.length ? Math.max(...drivers.map(d => Math.abs(d.value))) || 1 : 1;

  const bands = (["green", "orange", "red"] as const)
    .map(lvl => ({ lvl, ...(spec?.alert_thresholds?.[lvl] ?? {}) }))
    .filter(b => b.range || b.label);

  return (
    <div className={`rounded-3xl border ${cfg.border} ${cfg.bg} p-5 space-y-4`}>
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <SectionHeader icon={<Activity size={14} className="text-brand-400" />}
          title={t("scenarioDetail.model.dashboard.title")} subtitle={outcomeName} />
        <div className="flex items-center gap-2 shrink-0">
          <button type="button" onClick={doExportXlsx} disabled={exportingXlsx}
            title={t("scenarioDetail.model.dashboard.exportExcelTip")}
            className="flex items-center gap-1.5 rounded-full border border-emerald-500/25 text-emerald-300/80 px-2.5 py-1 text-[10px] font-semibold hover:bg-emerald-500/10 hover:text-emerald-200 transition disabled:opacity-50">
            {exportingXlsx ? <Loader2 size={11} className="animate-spin" /> : <Table2 size={11} />}{t("scenarioDetail.model.dashboard.exportExcel")}
          </button>
          <button type="button" onClick={doExport} disabled={exporting}
            title={t("scenarioDetail.model.dashboard.exportTip")}
            className="flex items-center gap-1.5 rounded-full border border-white/15 text-white/60 px-2.5 py-1 text-[10px] font-semibold hover:bg-white/5 hover:text-white transition disabled:opacity-50">
            {exporting ? <Loader2 size={11} className="animate-spin" /> : <Download size={11} />}{t("scenarioDetail.model.dashboard.export")}
          </button>
          <span className={`rounded-full px-3 py-1 text-xs font-bold border ${cfg.bg} ${cfg.text} ${cfg.border} flex items-center gap-1.5`}>
            <span className={`h-2 w-2 rounded-full ${cfg.dot}`} />
            {monitor?.status_label ?? t("scenarioDetail.model.dashboard.noSignal")}
          </span>
        </div>
      </div>

      {/* Prédiction actuelle (héros) */}
      <div className="rounded-2xl border border-white/5 bg-white/2 px-4 py-3">
        <p className="text-[10px] uppercase tracking-wider text-white/40">
          {monitor?.kind === "probability"
            ? `${t("scenarioDetail.model.dashboard.currentRisk")}${monitor?.positive_class ? ` (${monitor.positive_class})` : ""}`
            : t("scenarioDetail.model.dashboard.currentPrediction")}
        </p>
        <div className="flex items-baseline gap-2 mt-1">
          <span className={`text-5xl font-black font-mono ${cfg.text}`}>{valueLabel}</span>
          {hasPrediction && unit && <span className="text-sm font-normal text-white/50">{unit}</span>}
        </div>
        {!hasPrediction && (
          <p className="text-[11px] text-white/45 mt-1.5">{t("scenarioDetail.model.dashboard.awaitingData")}</p>
        )}
        {typeof monitor?.n_scored === "number" && monitor.n_scored > 0 && (
          <p className="text-[10px] text-white/35 mt-1 font-mono">
            {monitor.n_scored.toLocaleString(locale)} {t("scenarioDetail.model.dashboard.scored")}
            {typeof monitor.window === "number" ? ` · ${t("scenarioDetail.model.dashboard.window")} ${monitor.window}` : ""}
          </p>
        )}
      </div>

      {/* Bandes d'alerte (Normal / Tension / Alerte) — la bande courante est mise en avant */}
      {bands.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          {bands.map(b => {
            const bc = _DASH_STATUS[b.lvl];
            const active = b.lvl === sc && hasPrediction;
            return (
              <div key={b.lvl} title={b.rationale || undefined}
                className={`rounded-xl border ${bc.border} ${bc.bg} px-3 py-2 ${active ? `ring-2 ${bc.ring}` : ""}`}>
                <div className={`flex items-center gap-1.5 text-[11px] font-semibold ${bc.text}`}>
                  <span className={`h-1.5 w-1.5 rounded-full ${bc.dot}`} />{b.label || b.lvl}
                  {active && <span className="ml-auto text-[9px] uppercase tracking-wide">{t("scenarioDetail.model.dashboard.here")}</span>}
                </div>
                {b.range && <p className="text-[11px] text-white/55 font-mono mt-0.5">{b.range}</p>}
              </div>
            );
          })}
        </div>
      )}

      {/* Méta : algorithme · métrique · dernier entraînement · dernier calcul */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
        <div className="rounded-xl border border-white/5 bg-white/2 px-3 py-2">
          <div className="text-white/40">{t("scenarioDetail.model.algorithm")}</div>
          <div className="font-semibold text-white mt-0.5 font-mono truncate">{run.family ?? "—"}</div>
        </div>
        <div className="rounded-xl border border-white/5 bg-white/2 px-3 py-2">
          <div className="text-white/40 uppercase">{metricKey ?? t("scenarioDetail.model.dashboard.metric")}</div>
          <div className="font-semibold text-brand-300 mt-0.5 font-mono">{metricVal != null ? metricVal.toFixed(3) : "—"}</div>
        </div>
        <div className="rounded-xl border border-white/5 bg-white/2 px-3 py-2">
          <div className="text-white/40">{t("scenarioDetail.model.dashboard.trainedOn")}</div>
          <div className="font-medium text-white/80 mt-0.5">{fmtDate(run.created_at)}</div>
        </div>
        <div className="rounded-xl border border-white/5 bg-white/2 px-3 py-2">
          <div className="text-white/40">{t("scenarioDetail.model.dashboard.computedOn")}</div>
          <div className="font-medium text-white/80 mt-0.5">{fmtDate(monitor?.generated_at)}</div>
        </div>
      </div>

      {/* Facteurs clés (importances) */}
      {drivers.length > 0 && (
        <div className="rounded-2xl border border-white/5 bg-white/2 px-4 py-3">
          <p className="text-[10px] uppercase tracking-wider text-white/40 flex items-center gap-1 mb-2">
            <TrendingUp size={11} /> {t("scenarioDetail.model.dashboard.keyDrivers")}
          </p>
          <div className="space-y-1.5">
            {drivers.map((d, i) => (
              <div key={i} className="flex items-center gap-2 text-[11px]">
                <span className="w-40 truncate text-white/70" title={d.label}>{d.label}</span>
                <span className="flex-1 h-1.5 rounded-full bg-white/5 overflow-hidden">
                  <span className="block h-full rounded-full bg-brand-400/70" style={{ width: `${Math.max(4, (Math.abs(d.value) / maxDriver) * 100)}%` }} />
                </span>
                <span className="w-12 text-right font-mono text-white/45">{d.value.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


// ─── Section: Suivi & Évolution du modèle entraîné (Phases 3-5) ────────────────

function ModelMonitorSection({ scenarioId }: { scenarioId: string }) {
  const { t, lang } = useI18n();
  const [run, setRun] = useState<ModelRun | null>(null);
  const [monitor, setMonitor] = useState<ModelMonitor | null>(null);
  const [proposal, setProposal] = useState<SpecProposal | null>(null);
  const [spec, setSpec] = useState<ModelSpecResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    Promise.all([
      getModelRun(scenarioId).catch(() => null),
      getModelMonitor(scenarioId).catch(() => null),
      getSpecProposal(scenarioId).catch(() => null),
      getScenarioModelSpec(scenarioId).catch(() => null),
    ]).then(([r, m, p, s]) => {
      setRun(r); setMonitor(m); setProposal(p); setSpec(s);
    }).catch((e) => setError(e.message)).finally(() => setLoading(false));
  }, [scenarioId]);

  useEffect(() => { load(); }, [load]);

  // Rafraîchissement live : le statut se met à jour seul (sans spinner).
  useEffect(() => {
    const id = setInterval(() => {
      getModelMonitor(scenarioId).then(setMonitor).catch(() => {});
      getModelRun(scenarioId).then(setRun).catch(() => {});
    }, 30000);
    return () => clearInterval(id);
  }, [scenarioId]);

  // Anti-setState-après-démontage : le poll récursif ci-dessous appelle done()
  // (→ setBusy/load) et peut survivre au démontage de l'onglet Modèle.
  const mountedRef = useRef(true);
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => {
    mountedRef.current = false;
    if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
  }, []);

  const poll = (fn: () => Promise<{ status: string }>, done: () => void) => {
    let tries = 0;
    const tick = async () => {
      if (!mountedRef.current) return;
      tries += 1;
      try {
        const s = await fn();
        if (!mountedRef.current) return;
        if (s.status === "running") {
          if (tries < 40) { pollTimerRef.current = setTimeout(tick, 3000); return; }
        }
      } catch { /* ignore transient */ }
      if (mountedRef.current) done();
    };
    pollTimerRef.current = setTimeout(tick, 2000);
  };

  const doTrain = async () => {
    setBusy("train"); setError(null);
    try {
      await trainModel(scenarioId);
      poll(() => getModelTrainStatus(scenarioId), () => { setBusy(null); load(); });
    } catch (e: any) { setError(e.message); setBusy(null); }
  };

  const doCompare = async () => {
    setBusy("compare"); setError(null);
    try {
      await compareModels(scenarioId);
      poll(() => getModelTrainStatus(scenarioId), () => { setBusy(null); load(); });
    } catch (e: any) { setError(e.message); setBusy(null); }
  };

  const doSynthetic = async () => {
    setBusy("synthetic"); setError(null);
    try {
      // L'endpoint synthétique déclenche déjà l'entraînement (auto_train) ;
      // on se contente de suivre le job.
      await generateSyntheticData(scenarioId);
      poll(() => getModelTrainStatus(scenarioId), () => { setBusy(null); load(); });
    } catch (e: any) { setError(e.message); setBusy(null); }
  };

  const doPropose = async () => {
    setBusy("propose"); setError(null);
    try {
      await proposeSpec(scenarioId);
      poll(async () => { const p = await getSpecProposal(scenarioId); return { status: p.status === "generating" ? "running" : "done" }; },
        () => { setBusy(null); load(); });
    } catch (e: any) { setError(e.message); setBusy(null); }
  };

  const doValidate = async (action: 'accept' | 'reject') => {
    setBusy(action); setError(null);
    try {
      await validateSpecProposal(scenarioId, action);
      if (action === 'accept') {
        poll(() => getModelTrainStatus(scenarioId), () => { setBusy(null); load(); });
      } else { setBusy(null); load(); }
    } catch (e: any) { setError(e.message); setBusy(null); }
  };

  if (loading) return <LoadingSpinner text={t("scenarioDetail.model.loadingTrained")} />;

  const mcolors = STATUS_COLORS[monitor?.status_color ?? "unavailable"] || STATUS_COLORS.green;
  const diff = proposal?.diff;

  // Importances par VARIABLE (mêmes noms que l'onglet Variables), repliées et
  // mappées machine_name -> nom lisible. Fallback: feature_importances nettoyées.
  const nameByMachine: Record<string, string> = {};
  (spec?.features ?? []).forEach(f => { if (f.machine_name) nameByMachine[f.machine_name] = f.name || f.machine_name; });
  const byVar = (run?.summary?.importances_by_variable ?? null) as Record<string, number> | null;
  const topImportances: { label: string; value: number }[] = byVar && Object.keys(byVar).length > 0
    ? Object.entries(byVar)
        .map(([mn, v]) => ({ label: nameByMachine[mn] ?? mn, value: Number(v) }))
        .sort((a, b) => b.value - a.value).slice(0, 6)
    : (run?.feature_importances ?? []).slice(0, 6).map(fi => {
        const base = fi.feature.includes("__") ? fi.feature.split("__").slice(1).join("__") : fi.feature;
        return { label: nameByMachine[base] ?? base, value: fi.importance };
      });

  return (
    <div className="space-y-6">
      {error && <ErrorBox message={error} />}

      {/* Tableau de bord clinicien — dès qu'un modèle est entraîné : prédiction + bande
          d'alerte (couleur), performance, dates d'entraînement/calcul, facteurs clés. */}
      {run?.status === "ready" && <ModelDashboard scenarioId={scenarioId} run={run} monitor={monitor} spec={spec} />}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Statut live piloté par le modèle entraîné */}
        <div className={`rounded-3xl border ${mcolors.border} ${mcolors.bg} p-6 flex flex-col justify-between space-y-6`}>
          <div>
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold uppercase tracking-wider text-white/50">{t("scenarioDetail.model.liveStatus")}</span>
              <span className="flex h-2 w-2 rounded-full relative">
                <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${mcolors.dot}`} />
                <span className={`relative inline-flex rounded-full h-2 w-2 ${mcolors.dot}`} />
              </span>
            </div>
            {monitor?.status === "ready" ? (
              <>
                <p className="mt-4 text-3xl font-extrabold text-white">{monitor.status_label}</p>
                <div className="mt-4 rounded-2xl bg-white/5 p-4 border border-white/5">
                  <p className="text-[10px] text-white/50 uppercase tracking-wider">
                    {monitor.kind === "probability" ? `${t("scenarioDetail.model.riskPrefix")}${monitor.positive_class ?? t("scenarioDetail.model.riskPositiveClass")})` : t("scenarioDetail.model.predictedValue")}
                  </p>
                  <p className="text-3xl font-black text-brand-300 mt-1 font-mono">
                    {monitor.kind === "probability" && typeof monitor.value === "number"
                      ? `${(monitor.value * 100).toFixed(0)}%`
                      : (typeof monitor.value === "number" ? monitor.value.toLocaleString() : monitor.value)}
                    {monitor.unit && <span className="text-sm font-normal ml-1 text-white/50">{monitor.unit}</span>}
                  </p>
                  {monitor.generated_at && (
                    <p className="text-[10px] text-white/35 mt-1.5 font-mono">{t("scenarioDetail.model.computedOn")} {new Date(monitor.generated_at).toLocaleString(lang === "fr" ? "fr-FR" : "en-US")}</p>
                  )}
                </div>
              </>
            ) : (
              <p className="mt-4 text-sm text-white/60">{monitor?.message ?? t("scenarioDetail.model.modelNotTrainedOrNoData")}</p>
            )}
          </div>
        </div>

        {/* Modèle entraîné : métriques */}
        <div className="lg:col-span-2 rounded-3xl border border-white/10 bg-white/3 p-5 space-y-4">
          <SectionHeader icon={<Brain size={14} className="text-brand-400" />} title={t("scenarioDetail.model.trainedModel")}
            subtitle={t("scenarioDetail.model.trainedModelSubtitle")} />
          {run?.status === "ready" && (run.summary as any)?.is_synthetic && (
            <div className="rounded-xl border border-amber-500/25 bg-amber-500/10 px-3 py-2 flex items-start gap-2 text-[11px] text-amber-300">
              <Sparkles size={12} className="mt-0.5 shrink-0" />
              {t("scenarioDetail.model.syntheticBadge")}
            </div>
          )}
          {run?.status === "ready" ? (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 text-xs">
                <div className="rounded-xl border border-white/5 bg-white/2 px-3 py-2">
                  <span className="text-white/35">{t("scenarioDetail.model.algorithm")}</span>
                  <p className="font-semibold text-white mt-1">{run.family}</p>
                </div>
                <div className="rounded-xl border border-white/5 bg-white/2 px-3 py-2">
                  <span className="text-white/35">{t("scenarioDetail.model.task")}</span>
                  <p className="font-semibold text-white mt-1">{run.task_type}</p>
                </div>
                <div className="rounded-xl border border-white/5 bg-white/2 px-3 py-2">
                  <span className="text-white/35">{t("scenarioDetail.model.metricPrefix")}{run.metric})</span>
                  <p className="font-semibold text-brand-300 mt-1 font-mono">
                    {run.metrics && run.metric && typeof run.metrics[run.metric] === "number"
                      ? run.metrics[run.metric].toFixed(3)
                      : "—"}
                  </p>
                </div>
              </div>
              {run.task_type === "forecast" && <ForecastPanel run={run} />}
              {topImportances.length > 0 && (
                <div className="rounded-xl border border-white/5 bg-white/2 px-3 py-2.5">
                  <span className="text-[10px] text-white/35 uppercase tracking-wider flex items-center gap-1"><TrendingUp size={11} /> {t("scenarioDetail.model.influentialVariables")}</span>
                  <div className="mt-2 space-y-1.5">
                    {topImportances.map((fi) => (
                      <div key={fi.label} className="flex items-center gap-2">
                        <div className="h-1.5 rounded-full bg-brand-500/60" style={{ width: `${Math.max(4, Math.min(100, fi.value * 100))}%` }} />
                        <span className="text-[10px] text-white/50 truncate">{fi.label}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {run.best_params && Object.keys(run.best_params).length > 0 && (
                <div className="rounded-xl border border-white/5 bg-white/2 px-3 py-2.5">
                  <span className="text-[10px] text-white/35 uppercase tracking-wider">{t("scenarioDetail.model.hyperparameters")}</span>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {Object.entries(run.best_params).map(([k, v]) => (
                      <span key={k} className="rounded-lg border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] font-mono text-white/70">
                        {k} = <span className="text-brand-300">{typeof v === 'number' ? (Number.isInteger(v) ? v : (v as number).toFixed(3)) : String(v)}</span>
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {/* Hypothèses de régression (Gauss-Markov) : multicolinéarité, homoscédasticité,
                  autocorrélation, normalité des résidus. Uniquement pour les tâches de régression. */}
              {(((run.summary as any)?.assumptions?.checks?.length ?? 0) > 0) && (
                <div className="rounded-xl border border-white/5 bg-white/2 px-3 py-2.5">
                  <span className="text-[10px] text-white/35 uppercase tracking-wider">{t("scenarioDetail.model.assumptions")}</span>
                  <div className="mt-2 space-y-1.5">
                    {(run.summary as any).assumptions.checks.map((c: any) => (
                      <div key={c.key} className="flex items-start gap-2 text-[10px]" title={c.detail || ""}>
                        <span className={`mt-px shrink-0 rounded px-1 py-0.5 font-semibold ${
                          c.status === "ok" ? "bg-forest-500/20 text-forest-300"
                          : c.status === "warn" ? "bg-amber-500/20 text-amber-300"
                          : "bg-rose-500/20 text-rose-300"}`}>
                          {c.status === "ok" ? "OK" : c.status === "warn" ? "!" : "×"}
                        </span>
                        <span className="text-white/60">
                          <span className="text-white/80">{c.name}</span>
                          {typeof c.p_value === "number" && <span className="text-white/40"> · p={c.p_value}</span>}
                          {typeof c.statistic === "number" && <span className="text-white/40"> · {c.statistic}</span>}
                        </span>
                      </div>
                    ))}
                  </div>
                  {(run.summary as any).assumptions.applies === false && (
                    <p className="mt-1.5 text-[9px] text-white/30 leading-snug">{t("scenarioDetail.model.assumptionsTreeNote")}</p>
                  )}
                </div>
              )}
              {/* Garde-fous : fuite de cible, équilibre des classes, stabilité CV +
                  intervalle de confiance bootstrap de la métrique. */}
              {(((run.summary as any)?.guardrails?.checks?.length ?? 0) > 0) && (() => {
                const gr = (run.summary as any).guardrails;
                const ci = gr.metric_ci;
                return (
                  <div className="rounded-xl border border-white/5 bg-white/2 px-3 py-2.5">
                    <span className="text-[10px] text-white/35 uppercase tracking-wider flex items-center gap-1"><Shield size={11} /> {t("scenarioDetail.model.guardrails")}</span>
                    <div className="mt-2 space-y-1.5">
                      {gr.checks.map((c: any) => (
                        <div key={c.key} className="flex items-start gap-2 text-[10px]" title={c.detail || ""}>
                          <span className={`mt-px shrink-0 rounded px-1 py-0.5 font-semibold ${
                            c.status === "ok" ? "bg-forest-500/20 text-forest-300"
                            : c.status === "warn" ? "bg-amber-500/20 text-amber-300"
                            : "bg-rose-500/20 text-rose-300"}`}>
                            {c.status === "ok" ? "OK" : c.status === "warn" ? "!" : "×"}
                          </span>
                          <span className="text-white/60">
                            <span className="text-white/80">{c.name}</span>
                            {typeof c.statistic === "number" && <span className="text-white/40"> · {c.statistic}</span>}
                            {c.detail && <span className="text-white/40"> — {c.detail}</span>}
                          </span>
                        </div>
                      ))}
                    </div>
                    {ci && typeof ci.low === "number" && (
                      <p className="mt-2 text-[10px] text-white/50">
                        {t("scenarioDetail.model.metricCi")} ({ci.metric}) : <span className="font-mono text-white/70">[{ci.low.toFixed(3)}, {ci.high.toFixed(3)}]</span> <span className="text-white/30">· 95% · {ci.n_boot} bootstraps</span>
                      </p>
                    )}
                  </div>
                );
              })()}
              {/* Classement des familles comparées (bouton « Comparer les modèles »).
                  Le rang 1 = modèle actif retenu. Les familles en échec (paquet absent,
                  données insuffisantes) sont listées à part avec leur raison. */}
              {(((run.summary as any)?.leaderboard?.length ?? 0) > 0) && (() => {
                const board: any[] = (run.summary as any).leaderboard;
                const scored = board.filter((e) => e.value != null);
                const failed = board.filter((e) => e.value == null);
                const lowerBetter = (run.summary as any).leaderboard_lower_is_better === true;
                return (
                  <div className="rounded-xl border border-white/5 bg-white/2 px-3 py-2.5">
                    <span className="text-[10px] text-white/35 uppercase tracking-wider flex items-center gap-1">
                      <TrendingUp size={11} /> {t("scenarioDetail.model.leaderboardTitle")}
                      <span className="text-white/25 normal-case tracking-normal">
                        · {run.metric} {lowerBetter ? t("scenarioDetail.model.leaderboardLower") : t("scenarioDetail.model.leaderboardHigher")}
                      </span>
                    </span>
                    <div className="mt-2 space-y-1">
                      {scored.map((e) => (
                        <div key={e.family} className={`flex items-center gap-2 text-[11px] rounded-lg px-2 py-1 ${
                          e.rank === 1 ? "bg-brand-500/10 border border-brand-500/25" : ""}`}>
                          <span className="w-4 shrink-0 text-white/35 font-mono">{e.rank}</span>
                          <span className={`flex-1 truncate ${e.rank === 1 ? "text-brand-200 font-semibold" : "text-white/70"}`}>
                            {e.family}
                            {e.rank === 1 && <span className="ml-1.5 text-[9px] text-brand-300/70">{t("scenarioDetail.model.leaderboardActive")}</span>}
                          </span>
                          <span className="font-mono text-white/70">{typeof e.value === "number" ? e.value.toFixed(3) : "—"}</span>
                        </div>
                      ))}
                      {failed.map((e) => (
                        <div key={e.family} className="flex items-center gap-2 text-[11px] px-2 py-1 opacity-40" title={e.error || ""}>
                          <span className="w-4 shrink-0 text-white/25">·</span>
                          <span className="flex-1 truncate text-white/50">{e.family}</span>
                          <span className="text-[9px] text-white/40">{t("scenarioDetail.model.leaderboardUnavailable")}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })()}
            </div>
          ) : (
            <p className="text-xs text-white/50">{run?.message ?? t("scenarioDetail.model.noTrainedModel")}</p>
          )}
          <div className="flex flex-col gap-2 sm:flex-row">
            <button onClick={doTrain} disabled={busy !== null}
              className="flex-1 flex items-center justify-center gap-1.5 rounded-xl bg-white text-forest-950 font-semibold py-2 text-xs hover:bg-forest-200 transition disabled:opacity-50">
              <RotateCcw size={12} className={busy === "train" ? "animate-spin" : ""} />
              {busy === "train" ? t("scenarioDetail.model.trainingInProgress") : (run?.status === "ready" ? t("scenarioDetail.model.retrainModel") : t("scenarioDetail.model.trainModel"))}
            </button>
            <button onClick={doCompare} disabled={busy !== null} title={t("scenarioDetail.model.compareTooltip")}
              className="flex items-center justify-center gap-1.5 rounded-xl border border-white/15 text-white/70 font-semibold py-2 px-3 text-xs hover:bg-white/5 transition disabled:opacity-50">
              <TrendingUp size={12} className={busy === "compare" ? "animate-spin" : ""} />
              {busy === "compare" ? t("scenarioDetail.model.comparingInProgress") : t("scenarioDetail.model.compareModels")}
            </button>
            <button onClick={doSynthetic} disabled={busy !== null} title={t("scenarioDetail.model.syntheticTooltip")}
              className="flex items-center justify-center gap-1.5 rounded-xl border border-white/15 text-white/70 font-semibold py-2 px-3 text-xs hover:bg-white/5 transition disabled:opacity-50">
              <Sparkles size={12} className={busy === "synthetic" ? "animate-spin" : ""} />
              {busy === "synthetic" ? t("scenarioDetail.model.demoInProgress") : t("scenarioDetail.model.demoDataTrain")}
            </button>
          </div>
        </div>
      </div>

      {/* La projection SEIR a désormais son propre onglet « SEIR » (modèle, formule,
          paramètres éditables liés aux articles) — plus de doublon ici. */}

      {/* Évolution pilotée par l'évidence (Phase 5) */}
      <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-4">
        <SectionHeader icon={<Sparkles size={14} className="text-gold-400" />} title={t("scenarioDetail.model.evidenceEvolution")}
          subtitle={t("scenarioDetail.model.evidenceEvolutionSubtitle")} />
        {diff && proposal?.status === "ready" ? (
          diff.has_changes ? (
            <div className="space-y-3">
              <div className="flex flex-wrap gap-2 text-[11px]">
                {diff.outcome_changed && <span className="rounded-lg bg-gold-500/10 border border-gold-500/30 text-gold-300 px-2 py-1">{t("scenarioDetail.model.outcomeChanged")}</span>}
                {diff.summary.added > 0 && <span className="rounded-lg bg-brand-500/10 border border-brand-500/30 text-brand-300 px-2 py-1">+{diff.summary.added} {t("scenarioDetail.model.variablesSuffix")}</span>}
                {diff.summary.removed > 0 && <span className="rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 px-2 py-1">-{diff.summary.removed} {t("scenarioDetail.model.variablesSuffix")}</span>}
                {diff.summary.changed > 0 && <span className="rounded-lg bg-white/5 border border-white/15 text-white/70 px-2 py-1">{diff.summary.changed} {t("scenarioDetail.model.changedSuffix")}</span>}
                {diff.algorithm_changed && <span className="rounded-lg bg-gold-500/10 border border-gold-500/30 text-gold-300 px-2 py-1">{t("scenarioDetail.model.algorithmChangedPrefix")} {String(diff.algorithm_fields.family?.new ?? "")}</span>}
              </div>
              {(diff.features_added.length > 0 || diff.features_removed.length > 0) && (
                <div className="text-[11px] font-mono text-white/50 space-y-0.5">
                  {diff.features_added.map((f) => <p key={f} className="text-brand-300">+ {f}</p>)}
                  {diff.features_removed.map((f) => <p key={f} className="text-rose-300">− {f}</p>)}
                </div>
              )}
              <div className="flex gap-2">
                <button onClick={() => doValidate('accept')} disabled={busy !== null}
                  className="flex items-center gap-1.5 rounded-xl bg-brand-600 text-white font-semibold py-2 px-4 text-xs hover:bg-brand-500 transition disabled:opacity-50">
                  <CheckCircle2 size={13} />{busy === "accept" ? t("scenarioDetail.model.applying") : t("scenarioDetail.model.validateRetrain")}
                </button>
                <button onClick={() => doValidate('reject')} disabled={busy !== null}
                  className="flex items-center gap-1.5 rounded-xl border border-white/15 text-white/70 font-semibold py-2 px-4 text-xs hover:bg-white/5 transition disabled:opacity-50">
                  <X size={13} />{t("scenarioDetail.model.reject")}
                </button>
              </div>
            </div>
          ) : (
            <p className="text-xs text-brand-300 flex items-center gap-1.5"><CheckCircle2 size={13} /> {t("scenarioDetail.model.upToDate")}</p>
          )
        ) : (
          <p className="text-xs text-white/50">{t("scenarioDetail.model.noProposal")}</p>
        )}
        <button onClick={doPropose} disabled={busy !== null}
          className="flex items-center gap-1.5 rounded-xl border border-gold-500/30 bg-gold-500/10 text-gold-300 font-semibold py-2 px-4 text-xs hover:bg-gold-500/20 transition disabled:opacity-50">
          <RefreshCw size={12} className={busy === "propose" ? "animate-spin" : ""} />
          {busy === "propose" ? t("scenarioDetail.model.analyzingEvidence") : t("scenarioDetail.model.checkNewEvidence")}
        </button>
      </div>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

type SectionKey = "review" | "evidence" | "assistant" | "viz" | "variables" | "queries" | "alerts";

const SECTIONS: Array<{ key: SectionKey; icon: React.ReactNode }> = [
  { key: "review",      icon: <FileText size={13} /> },
  { key: "evidence",    icon: <BookOpen size={13} /> },
  { key: "assistant",   icon: <MessageSquare size={13} /> },
  { key: "viz",         icon: <Layers size={13} /> },
  { key: "variables",   icon: <Database size={13} /> },
  { key: "queries",     icon: <Search size={13} /> },
  { key: "alerts",      icon: <Bell size={13} /> },
];

interface ScenarioDetailPageProps {
  scenarioId: string;
  onBack: () => void;
  initialTab?: "model";
}

export function ScenarioDetailPage({ scenarioId, onBack, initialTab }: ScenarioDetailPageProps) {
  const { t } = useI18n();
  const [detail, setDetail] = useState<ScenarioDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // initialTab="model" ouvre directement l'onglet Variables & Modèle (sous-tab Modèle prédictif).
  const [activeSection, setActiveSection] = useState<SectionKey>(initialTab === "model" ? "variables" : "review");

  // Scroll tout en haut à l'ouverture d'un scénario, en garantissant le tout
  // début (en-tête + logos). Un scrollTo(0,0) seul est repoussé par l'inertie
  // (momentum) reportée depuis la liste/recherche. On va donc en haut PUIS on
  // verrouille brièvement le défilement (overflow:hidden) : tant que la page ne
  // peut pas défiler, l'inertie ne peut pas la faire redescendre. On relâche
  // après 500 ms (inertie épuisée). Dépendance `loading` : on rejoue le verrou
  // une fois le vrai contenu monté.
  useLayoutEffect(() => {
    const root = document.documentElement;
    const body = document.body;
    const prevRoot = root.style.overflow;
    const prevBody = body.style.overflow;
    window.scrollTo(0, 0);
    root.style.overflow = "hidden";
    body.style.overflow = "hidden";
    const release = () => {
      root.style.overflow = prevRoot;
      body.style.overflow = prevBody;
    };
    const timer = window.setTimeout(release, 500);
    return () => {
      window.clearTimeout(timer);
      release();
    };
  }, [scenarioId, loading]);

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
      <div className="flex items-center justify-center py-24 text-white/50 gap-2">
        <RotateCcw size={18} className="animate-spin" />
        <span>{t("scenarioDetail.page.loading")}</span>
      </div>
    );
  }

  if (error || !detail) {
    return (
      <div className="space-y-4">
        <button onClick={onBack} className="flex items-center gap-2 text-sm text-white/50 hover:text-white transition">
          <ArrowLeft size={14} /> {t("scenarioDetail.page.backToScenarios")}
        </button>
        <ErrorBox message={error ?? t("scenarioDetail.page.scenarioNotFound")} />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* En-tête */}
      <div className="flex items-start gap-4">
        <button
          onClick={onBack}
          className="mt-1 flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-white/50 hover:text-white hover:bg-white/10 transition shrink-0"
        >
          <ArrowLeft size={12} /> {t("scenarioDetail.page.back")}
        </button>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h2 className="text-xl font-semibold text-white">{detail.title}</h2>
            <span className="rounded-full border border-brand-500/20 bg-brand-500/10 px-2 py-0.5 text-xs text-brand-300">
              {detail.cluster}
            </span>
            <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-xs text-white/50 font-mono">
              {detail.corpus_stats.total} {t("scenarioDetail.page.articles")}
            </span>
          </div>
          <p className="mt-1 text-sm text-white/50 leading-5">
            {isUserScenario(scenarioId) && detail.query ? `${t("scenarios.savedSearchPrefix")}${detail.query}` : detail.description}
          </p>
          
          {/* Mots-clés */}
          {detail.keywords && detail.keywords.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {detail.keywords.map((kw, i) => (
                <span key={i} className="rounded-md border border-white/5 bg-white/2 px-1.5 py-0.5 text-[10px] text-white/40 font-medium tracking-wide">
                  #{kw}
                </span>
              ))}
            </div>
          )}

          {/* Contexte Clinique */}
          {detail.clinical_rationale && (
            <div className="mt-3 rounded-xl border border-brand-500/10 bg-brand-500/3 p-3.5 text-xs text-brand-200/90 leading-relaxed flex gap-2.5 items-start">
              <span className="mt-0.5 flex h-4 w-4 items-center justify-center rounded-full bg-brand-500/10 text-brand-400 shrink-0 text-[10px] font-bold">i</span>
              <div>
                <strong className="text-brand-300 font-semibold block mb-0.5">{t("scenarioDetail.page.clinicalStake")}</strong>
                {detail.clinical_rationale}
              </div>
            </div>
          )}

          {/* Actions recommandées */}
          {detail.recommended_actions.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {detail.recommended_actions.slice(0, 2).map((action, i) => (
                <span key={i} className="flex items-center gap-1.5 rounded-xl border border-white/5 bg-white/3 px-2.5 py-1 text-xs text-white/70">
                  <span className="h-1.5 w-1.5 rounded-full bg-brand-400 shrink-0" />
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
                ? "bg-brand-700 text-gold-400 font-semibold"
                : "border border-transparent text-white/60 hover:text-white hover:bg-white/8"
            }`}
          >
            {section.icon}
            {t(`scenarioDetail.page.sections.${section.key}`)}
          </button>
        ))}
      </div>

      {/* Contenu de la section active — isolé par une limite d'erreur : un crash
          de rendu (ex. visualisation clustering) n'emporte plus toute la page. */}
      <ErrorBoundary resetKey={`${activeSection}:${scenarioId}`} label="scenarioDetail.page.errorBoundaryLabel">
        {activeSection === "review" && <ReviewTab scenarioId={scenarioId} detail={detail} />}
        {activeSection === "evidence" && <EvidenceTab scenarioId={scenarioId} detail={detail} />}
        {activeSection === "assistant" && <RagSection scenarioId={scenarioId} detail={detail} />}
        {activeSection === "viz" && <VizTab scenarioId={scenarioId} />}
        {activeSection === "variables" && <VariablesModelTab detail={detail} scenarioId={scenarioId} initialSub={initialTab === "model" ? "monitor" : undefined} />}
        {activeSection === "queries" && <QueriesSection detail={detail} scenarioId={scenarioId} />}
        {activeSection === "alerts" && <AlertsSection scenarioId={scenarioId} />}
      </ErrorBoundary>
    </div>
  );
}
