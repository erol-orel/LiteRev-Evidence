import { useEffect, useMemo, useState } from "react";
import { Activity, BarChart2, BookOpen, Download, ExternalLink, RotateCcw, Zap, CheckSquare, XCircle, CheckCircle, HelpCircle, ArrowDown } from "lucide-react";

import {
  fetchDocumentDetail,
  fetchEvidenceSummary,
  fetchGesicaScenarios,
  fetchGesicaStats,
  fetchCorpusStats,
  getFilterOptions,
  getReadableExcerpt,
  askAssistant,
  fetchScreeningList,
  submitScreeningDecision,
  fetchPrismaFlow,
  type CorpusStats,
  type DocumentDetailResponse,
  type EvidenceSummaryResponse,
  type FilterOptions,
  type GesicaScenario,
  type GesicaStats,
  type AskResponse,
  type ScreeningDocument,
  type PrismaFlow,
  searchDocuments,
} from "./lib/api";
import type {
  ProjectContext,
  RelevanceLabel,
  SearchFilters,
  SearchMode,
  SearchResult,
} from "./types/search";

const FILTER_FIELDS: Array<[keyof FilterOptions, string]> = [
  ["sourceType", "Type de source"],
  ["diseaseOrCondition", "Maladie / pathologie"],
  ["scenarioType", "Type de scénario"],
  ["geographicScope", "Zone géographique"],
  ["evidenceCategory", "Catégorie de preuve"],
];

const PAGE_SIZE = 10;

type AppTab = "search" | "scenarios" | "stats" | "assistant" | "screening";

type DetailView = {
  id: number | null;
  title: string;
  abstract: string;
  excerpt: string;
  source: string;
  year: string;
  url: string;
  externalId: string;
  projectContext: string;
  sourceType: string;
  disease: string;
  scenario: string;
  geography: string;
  evidence: string;
  chunkCount: number;
};

function csvEscape(value: unknown): string {
  return JSON.stringify(value ?? "");
}

function EvidenceStrengthBadge({ strength }: { strength: "weak" | "moderate" | "strong" | null }) {
  if (!strength) return null;
  const config = {
    strong: { label: "Forte", className: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30" },
    moderate: { label: "Modérée", className: "bg-amber-500/20 text-amber-300 border-amber-500/30" },
    weak: { label: "Faible", className: "bg-rose-500/20 text-rose-300 border-rose-500/30" },
  };
  const { label, className } = config[strength];
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${className}`}>
      <Zap size={10} />
      {label}
    </span>
  );
}

function SignalBadge({ label }: { label: string }) {
  return (
    <span className="rounded-full bg-cyan-500/10 border border-cyan-500/20 px-2 py-0.5 text-xs text-cyan-300">
      {label}
    </span>
  );
}

function GesicaSignalsPanel({ summary }: { summary: EvidenceSummaryResponse }) {
  const s = summary.gesicaSignals;
  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="font-medium text-white flex items-center gap-2">
          <Zap size={14} className="text-cyan-400" />
          Signaux GESICA
        </h3>
        <EvidenceStrengthBadge strength={s.evidenceStrength} />
      </div>

      {s.forecastHorizon && (
        <div className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-300">
          <span className="text-slate-400">Horizon prévisionnel :</span>{" "}
          <span className="font-mono text-cyan-300">{s.forecastHorizon}</span>
        </div>
      )}

      {s.demandSignals.length > 0 && (
        <div>
          <p className="mb-1 text-xs text-slate-400">Signaux de demande</p>
          <div className="flex flex-wrap gap-1">
            {s.demandSignals.slice(0, 8).map((sig) => (
              <SignalBadge key={sig} label={sig} />
            ))}
          </div>
        </div>
      )}

      {s.scenarioTags.length > 0 && (
        <div>
          <p className="mb-1 text-xs text-slate-400">Scénarios détectés</p>
          <div className="flex flex-wrap gap-1">
            {s.scenarioTags.map((tag) => (
              <span key={tag} className="rounded-full bg-violet-500/10 border border-violet-500/20 px-2 py-0.5 text-xs text-violet-300">
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}

      {s.reportedMetrics.length > 0 && (
        <div>
          <p className="mb-1 text-xs text-slate-400">Métriques rapportées</p>
          <div className="flex flex-wrap gap-1">
            {s.reportedMetrics.map((m) => (
              <span key={m} className="rounded-full bg-slate-700/60 border border-white/10 px-2 py-0.5 text-xs text-slate-300 font-mono uppercase">
                {m}
              </span>
            ))}
          </div>
        </div>
      )}

      {s.crossBorder && (
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-xs text-amber-300">
          Pertinence transfrontalière détectée (France / Suisse)
        </div>
      )}
    </section>
  );
}

function StatsView({ corpusStats, gesicaStats }: { corpusStats: CorpusStats | null; gesicaStats: GesicaStats | null }) {
  if (!corpusStats && !gesicaStats) {
    return <div className="text-sm text-slate-400">Chargement des statistiques...</div>;
  }

  return (
    <div className="space-y-6">
      {corpusStats && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold text-white">
            <BarChart2 size={18} className="text-cyan-400" />
            Corpus global
          </h2>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-4 text-center">
              <p className="text-2xl font-bold text-cyan-300">{corpusStats.totalDocuments.toLocaleString()}</p>
              <p className="mt-1 text-xs text-slate-400">Documents</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-4 text-center">
              <p className="text-2xl font-bold text-cyan-300">{corpusStats.totalChunks.toLocaleString()}</p>
              <p className="mt-1 text-xs text-slate-400">Chunks</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-4 text-center">
              <p className="text-2xl font-bold text-cyan-300">{Object.keys(corpusStats.byProject).length}</p>
              <p className="mt-1 text-xs text-slate-400">Projets</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-4 text-center">
              <p className="text-2xl font-bold text-cyan-300">{Object.keys(corpusStats.bySource).length}</p>
              <p className="mt-1 text-xs text-slate-400">Sources</p>
            </div>
          </div>

          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <div>
              <h3 className="mb-2 text-sm font-medium text-slate-300">Par projet</h3>
              <div className="space-y-2">
                {Object.entries(corpusStats.byProject).map(([proj, count]) => (
                  <div key={proj} className="flex items-center justify-between rounded-xl border border-white/10 bg-slate-900/40 px-3 py-2 text-sm">
                    <span className="text-slate-200 capitalize">{proj}</span>
                    <span className="font-mono text-cyan-300">{count}</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h3 className="mb-2 text-sm font-medium text-slate-300">Par source</h3>
              <div className="space-y-2">
                {Object.entries(corpusStats.bySource).map(([src, count]) => (
                  <div key={src} className="flex items-center justify-between rounded-xl border border-white/10 bg-slate-900/40 px-3 py-2 text-sm">
                    <span className="text-slate-200">{src}</span>
                    <span className="font-mono text-cyan-300">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {gesicaStats && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold text-white">
            <Activity size={18} className="text-cyan-400" />
            Corpus GESICA
          </h2>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            {Object.entries(gesicaStats.evidenceStrengthDistribution).map(([strength, count]) => {
              const colors: Record<string, string> = {
                strong: "text-emerald-300",
                moderate: "text-amber-300",
                weak: "text-rose-300",
              };
              return (
                <div key={strength} className="rounded-2xl border border-white/10 bg-slate-900/60 p-4 text-center">
                  <p className={`text-2xl font-bold ${colors[strength] ?? "text-white"}`}>{count}</p>
                  <p className="mt-1 text-xs text-slate-400 capitalize">Preuve {strength}</p>
                </div>
              );
            })}
          </div>

          {Object.keys(gesicaStats.forecastHorizons).length > 0 && (
            <div className="mt-4">
              <h3 className="mb-2 text-sm font-medium text-slate-300">Horizons prévisionnels</h3>
              <div className="flex flex-wrap gap-2">
                {Object.entries(gesicaStats.forecastHorizons).slice(0, 10).map(([h, count]) => (
                  <span key={h} className="rounded-full border border-cyan-500/20 bg-cyan-500/10 px-3 py-1 text-xs text-cyan-300">
                    {h} <span className="opacity-60">({count})</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ScenariosView({ scenarios }: { scenarios: GesicaScenario[] }) {
  if (scenarios.length === 0) {
    return <div className="text-sm text-slate-400">Chargement des scénarios...</div>;
  }

  return (
    <div className="space-y-5">
      {scenarios.map((scenario) => (
        <div key={scenario.id} className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <div className="flex items-start gap-3">
            <div className="mt-1 rounded-xl border border-cyan-500/20 bg-cyan-500/10 p-2">
              <Activity size={16} className="text-cyan-400" />
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-white">{scenario.title}</h3>
              <p className="mt-1 text-sm leading-6 text-slate-300">{scenario.description}</p>
            </div>
          </div>

          <div className="mt-4">
            <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
              Actions recommandées
            </h4>
            <ul className="space-y-1.5">
              {scenario.recommendedActions.map((action, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-slate-200">
                  <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-cyan-400" />
                  {action}
                </li>
              ))}
            </ul>
          </div>

          {scenario.relevantArticles.length > 0 && (
            <div className="mt-4">
              <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                Articles associés
              </h4>
              <div className="space-y-2">
                {scenario.relevantArticles.map((article) => (
                  <div key={article.id} className="rounded-xl border border-white/10 bg-slate-900/40 px-3 py-2 text-sm">
                    <p className="text-slate-200">{article.title}</p>
                    <p className="mt-0.5 text-xs text-slate-400">{article.source} · {article.year ?? "—"}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

interface ScreeningViewProps {
  docs: ScreeningDocument[];
  prismaFlow: PrismaFlow | null;
  loading: boolean;
  error: string | null;
  selectedDoc: ScreeningDocument | null;
  setSelectedDoc: (doc: ScreeningDocument | null) => void;
  reason: string;
  setReason: (r: string) => void;
  notes: string;
  setNotes: (n: string) => void;
  onDecision: (status: "included" | "excluded") => void;
  projectContext: string;
}

function ScreeningView({
  docs,
  prismaFlow,
  loading,
  error,
  selectedDoc,
  setSelectedDoc,
  reason,
  setReason,
  notes,
  setNotes,
  onDecision,
  projectContext,
}: ScreeningViewProps) {
  if (loading && docs.length === 0) {
    return <div className="text-sm text-slate-400">Chargement du module de screening...</div>;
  }

  if (error) {
    return <div className="text-sm text-rose-400">Erreur : {error}</div>;
  }

  const pendingDocs = docs.filter(d => !d.screeningStatus || d.screeningStatus === "pending");

  return (
    <div className="space-y-6">
      {/* Diagramme PRISMA */}
      {prismaFlow && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl">
          <h3 className="mb-4 text-lg font-semibold text-white flex items-center gap-2">
            <CheckSquare size={18} className="text-emerald-400" />
            Diagramme de Flux PRISMA — {projectContext.toUpperCase()}
          </h3>
          <div className="grid gap-4 md:grid-cols-4 text-center">
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-4">
              <div className="text-2xl font-bold text-cyan-300">{prismaFlow.recordsIdentified}</div>
              <div className="mt-1 text-xs text-slate-400 uppercase tracking-wider">Identifiés (Stage 1)</div>
            </div>
            <div className="flex flex-col justify-center items-center">
              <ArrowDown size={16} className="text-slate-500 mb-2" />
              <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-4 w-full">
                <div className="text-2xl font-bold text-yellow-400">{prismaFlow.recordsScreened}</div>
                <div className="mt-1 text-xs text-slate-400 uppercase tracking-wider">Screenés (Titre/Abstract)</div>
              </div>
            </div>
            <div className="flex flex-col justify-center items-center">
              <div className="text-xs text-rose-400 font-semibold mb-1">-{prismaFlow.recordsExcluded} Exclus</div>
              <ArrowDown size={16} className="text-slate-500 mb-2" />
              <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/5 p-4 w-full">
                <div className="text-2xl font-bold text-emerald-400">{prismaFlow.recordsIncluded}</div>
                <div className="mt-1 text-xs text-slate-400 uppercase tracking-wider">Inclus (Full-text)</div>
              </div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-4 flex flex-col justify-center">
              <div className="text-xs text-slate-400">Taux d'Inclusion</div>
              <div className="text-xl font-bold text-white mt-1">
                {prismaFlow.recordsScreened > 0 
                  ? `${((prismaFlow.recordsIncluded / prismaFlow.recordsScreened) * 100).toFixed(1)}%`
                  : "0%"}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Interface de Screening Double-Panel */}
      <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
        {/* Liste des Documents */}
        <aside className="space-y-4">
          <div className="rounded-3xl border border-white/10 bg-white/5 p-4 shadow-2xl h-[600px] flex flex-col">
            <h4 className="text-sm font-semibold text-white mb-3 px-2 flex items-center justify-between">
              <span>Articles ({docs.length})</span>
              <span className="text-xs text-slate-400 font-normal">{pendingDocs.length} en attente</span>
            </h4>
            <div className="flex-1 overflow-y-auto space-y-2 pr-1">
              {docs.map((d) => (
                <button
                  key={d.id}
                  onClick={() => {
                    setSelectedDoc(d);
                    setReason(d.screeningReason || "");
                    setNotes(d.screeningNotes || "");
                  }}
                  className={`w-full text-left rounded-xl p-3 border transition flex flex-col gap-1.5 ${
                    selectedDoc?.id === d.id
                      ? "border-cyan-400 bg-cyan-500/10"
                      : "border-white/5 bg-white/5 hover:border-white/20"
                  }`}
                >
                  <span className="text-xs font-semibold text-slate-200 line-clamp-2">{d.title}</span>
                  <div className="flex items-center justify-between text-[10px] text-slate-400 w-full">
                    <span>{d.source} · {d.year ?? "—"}</span>
                    {d.screeningStatus === "included" && (
                      <span className="flex items-center gap-0.5 text-emerald-400 font-semibold">
                        <CheckCircle size={10} /> Inclus
                      </span>
                    )}
                    {d.screeningStatus === "excluded" && (
                      <span className="flex items-center gap-0.5 text-rose-400 font-semibold">
                        <XCircle size={10} /> Exclu
                      </span>
                    )}
                    {(!d.screeningStatus || d.screeningStatus === "pending") && (
                      <span className="flex items-center gap-0.5 text-yellow-400 font-semibold">
                        <HelpCircle size={10} /> À screené
                      </span>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </div>
        </aside>

        {/* Détail et Décision */}
        <section className="space-y-4">
          {selectedDoc ? (
            <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl space-y-5">
              <div>
                <span className="rounded bg-white/5 px-2 py-1 text-xs text-slate-400">
                  {selectedDoc.source} · {selectedDoc.year ?? "—"}
                </span>
                <h3 className="text-2xl font-semibold text-white mt-3">{selectedDoc.title}</h3>
                {selectedDoc.url && (
                  <a
                    href={selectedDoc.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300 mt-2"
                  >
                    Voir l'article d'origine <ExternalLink size={10} />
                  </a>
                )}
              </div>

              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Abstract</h4>
                <p className="rounded-2xl border border-white/10 bg-slate-950/60 p-4 leading-6 text-sm text-slate-200">
                  {selectedDoc.abstract || "Aucun abstract disponible."}
                </p>
              </div>

              {/* Formulaire de Décision */}
              <div className="border-t border-white/10 pt-5 space-y-4">
                <h4 className="text-sm font-semibold text-white">Décision de Screening (Inclusion / Exclusion)</h4>
                
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Raison de l'exclusion (obligatoire si exclu)</label>
                    <select
                      value={reason}
                      onChange={(e) => setReason(e.target.value)}
                      className="w-full rounded-xl border border-white/10 bg-slate-950/80 p-3 text-sm text-white outline-none focus:border-cyan-400"
                    >
                      <option value="">-- Sélectionner une raison --</option>
                      <option value="wrong-population">Population non cible</option>
                      <option value="wrong-intervention">Pas d'IA / Méthode non cible</option>
                      <option value="wrong-outcome">Pas de métriques d'évaluation</option>
                      <option value="no-fulltext">Pas de texte intégral disponible</option>
                      <option value="duplicate">Doublon d'un autre article</option>
                      <option value="other">Autre (spécifier en notes)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Notes de screening</label>
                    <input
                      value={notes}
                      onChange={(e) => setNotes(e.target.value)}
                      placeholder="Ex. Excellente étude de prévision par LSTM à Genève"
                      className="w-full rounded-xl border border-white/10 bg-slate-950/80 p-3 text-sm text-white outline-none focus:border-cyan-400"
                    />
                  </div>
                </div>

                <div className="flex gap-3 justify-end pt-2">
                  <button
                    onClick={() => onDecision("excluded")}
                    className="rounded-xl border border-rose-500/30 bg-rose-500/10 hover:bg-rose-500/20 px-5 py-3 text-sm font-semibold text-rose-200 transition flex items-center gap-2"
                  >
                    <XCircle size={16} />
                    Exclure l'article
                  </button>
                  <button
                    onClick={() => onDecision("included")}
                    className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 hover:bg-emerald-500/20 px-5 py-3 text-sm font-semibold text-emerald-200 transition flex items-center gap-2"
                  >
                    <CheckCircle size={16} />
                    Inclure dans le corpus final
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl text-center text-slate-400 py-20">
              Sélectionnez un document dans la liste pour commencer le screening.
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

interface AssistantViewProps {
  question: string;
  setQuestion: (q: string) => void;
  response: AskResponse | null;
  loading: boolean;
  error: string | null;
  onAsk: () => void;
  projectContext: string;
}

function AssistantView({
  question,
  setQuestion,
  response,
  loading,
  error,
  onAsk,
  projectContext,
}: AssistantViewProps) {
  return (
    <div className="space-y-6">
      <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl">
        <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold text-white">
          <Zap size={18} className="text-cyan-400" />
          Assistant Scientifique RAG — {projectContext.toUpperCase()}
        </h2>
        <p className="mb-4 text-sm text-slate-300 leading-6">
          Posez une question complexe à l'assistant. Il va interroger les chunks de la base de données les plus pertinents pour votre projet, puis synthétiser une réponse scientifiquement étayée et citer ses sources.
        </p>

        <div className="flex flex-col gap-3 sm:flex-row">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onAsk()}
            placeholder="Ex. Quelles sont les meilleures méthodes d'IA pour prédire l'afflux de patients aux urgences ?"
            className="min-h-14 flex-1 rounded-2xl border border-white/10 bg-slate-950/80 px-4 text-white outline-none placeholder:text-slate-500 focus:border-cyan-400"
          />
          <button
            type="button"
            onClick={onAsk}
            disabled={loading || !question.trim()}
            className="min-h-14 rounded-2xl bg-cyan-400 px-6 font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-60 flex items-center justify-center gap-2"
          >
            {loading ? "Synthèse en cours..." : "Interroger"}
            <Zap size={14} />
          </button>
        </div>

        {error && (
          <div className="mt-4 rounded-2xl border border-rose-400/30 bg-rose-500/10 p-4 text-sm text-rose-100">
            {error}
          </div>
        )}
      </div>

      {loading && (
        <div className="rounded-3xl border border-dashed border-white/10 bg-white/5 p-10 text-center text-slate-300 flex flex-col items-center justify-center gap-3">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-cyan-400 border-t-transparent" />
          <p className="text-sm">L'assistant analyse les articles scientifiques et rédige sa synthèse...</p>
        </div>
      )}

      {response && (
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
          <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl space-y-4">
            <h3 className="text-lg font-semibold text-white">Synthèse de l'Assistant</h3>
            <div className="prose prose-invert max-w-none text-sm leading-7 text-slate-200 whitespace-pre-wrap">
              {response.answer}
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl space-y-4 h-fit">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              <BookOpen size={16} className="text-cyan-400" />
              Sources utilisées ({response.sources.length})
            </h3>
            <div className="space-y-3">
              {response.sources.map((s, i) => (
                <div key={s.documentId} className="rounded-2xl border border-white/10 bg-slate-900/60 p-4 space-y-2">
                  <div className="flex items-start justify-between gap-2">
                    <span className="inline-flex shrink-0 h-5 w-5 items-center justify-center rounded-full bg-cyan-400/20 text-xs font-bold text-cyan-300">
                      {i + 1}
                    </span>
                    {s.url && (
                      <a
                        href={s.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-slate-400 hover:text-white"
                      >
                        <ExternalLink size={12} />
                      </a>
                    )}
                  </div>
                  <h4 className="text-sm font-semibold text-white leading-5">{s.title}</h4>
                  <div className="flex flex-wrap gap-1 text-[10px]">
                    <span className="rounded bg-white/5 px-1.5 py-0.5 text-slate-400">{s.source}</span>
                    {s.year && <span className="rounded bg-white/5 px-1.5 py-0.5 text-slate-400">{s.year}</span>}
                    {s.evidenceStrength && (
                      <span className="rounded bg-cyan-500/10 px-1.5 py-0.5 text-cyan-300 capitalize">
                        Preuve {s.evidenceStrength}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [projectContext, setProjectContext] = useState<ProjectContext>("gesica");
  const [activeTab, setActiveTab] = useState<AppTab>("search");
  const [mode, setMode] = useState<SearchMode>("semantic");
  const [query, setQuery] = useState("");
  const [filters, setFilters] = useState<SearchFilters>({
    projectContext: "gesica",
  });
  const [yearRange, setYearRange] = useState<[number, number]>([
    2000,
    new Date().getFullYear(),
  ]);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [relevanceMap, setRelevanceMap] = useState<Record<string, RelevanceLabel>>({});
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null);
  const [selectedDocument, setSelectedDocument] = useState<DocumentDetailResponse | null>(null);
  const [evidenceSummary, setEvidenceSummary] = useState<EvidenceSummaryResponse | null>(null);
  const [filterOptions, setFilterOptions] = useState<FilterOptions | null>(null);
  const [page, setPage] = useState(1);
  const [corpusStats, setCorpusStats] = useState<CorpusStats | null>(null);
  const [gesicaStats, setGesicaStats] = useState<GesicaStats | null>(null);
  const [gesicaScenarios, setGesicaScenarios] = useState<GesicaScenario[]>([]);
  
  // States Assistant RAG
  const [assistantQuestion, setAssistantQuestion] = useState("");
  const [assistantResponse, setAssistantResponse] = useState<AskResponse | null>(null);
  const [assistantLoading, setAssistantLoading] = useState(false);
  const [assistantError, setAssistantLoadingError] = useState<string | null>(null);

  const handleAskAssistant = async () => {
    if (!assistantQuestion.trim()) return;
    setAssistantLoading(true);
    setAssistantLoadingError(null);
    setAssistantResponse(null);
    try {
      const res = await askAssistant({
        question: assistantQuestion,
        projectContext: projectContext,
        filters: filters
      });
      setAssistantResponse(res);
    } catch (err: any) {
      setAssistantLoadingError(err.message || "Une erreur est survenue lors de l'appel à l'assistant.");
    } finally {
      setAssistantLoading(false);
    }
  };

  // States Screening PRISMA
  const [screeningDocs, setScreeningList] = useState<ScreeningDocument[]>([]);
  const [prismaFlow, setPrismaFlow] = useState<PrismaFlow | null>(null);
  const [screeningLoading, setScreeningLoading] = useState(false);
  const [screeningError, setScreeningError] = useState<string | null>(null);
  const [selectedScreeningDoc, setSelectedScreeningDoc] = useState<ScreeningDocument | null>(null);
  const [decisionReason, setDecisionReason] = useState("");
  const [decisionNotes, setDecisionNotes] = useState("");

  const loadScreeningData = async () => {
    setScreeningLoading(true);
    setScreeningError(null);
    try {
      const [list, flow] = await Promise.all([
        fetchScreeningList(projectContext),
        fetchPrismaFlow(projectContext)
      ]);
      setScreeningList(list);
      setPrismaFlow(flow);
      if (list.length > 0 && !selectedScreeningDoc) {
        setSelectedScreeningDoc(list[0]);
      }
    } catch (err: any) {
      setScreeningError(err.message || "Erreur de chargement des données de screening.");
    } finally {
      setScreeningLoading(false);
    }
  };

  const handleScreeningDecision = async (status: "included" | "excluded") => {
    if (!selectedScreeningDoc) return;
    try {
      await submitScreeningDecision({
        documentId: selectedScreeningDoc.id,
        status,
        reason: decisionReason || undefined,
        notes: decisionNotes || undefined
      });
      
      // Mettre à jour la liste locale
      setScreeningList(prev => prev.map(d => 
        d.id === selectedScreeningDoc.id 
          ? { ...d, screeningStatus: status, screeningReason: decisionReason, screeningNotes: decisionNotes }
          : d
      ));
      
      // Recharger le diagramme PRISMA
      const flow = await fetchPrismaFlow(projectContext);
      setPrismaFlow(flow);
      
      // Sélectionner le document suivant s'il y en a un en attente
      const currentIndex = screeningDocs.findIndex(d => d.id === selectedScreeningDoc.id);
      const nextPending = screeningDocs.slice(currentIndex + 1).find(d => !d.screeningStatus || d.screeningStatus === "pending")
                       || screeningDocs.slice(0, currentIndex).find(d => !d.screeningStatus || d.screeningStatus === "pending");
      
      if (nextPending) {
        setSelectedScreeningDoc(nextPending);
        setDecisionReason("");
        setDecisionNotes("");
      } else {
        // Si aucun en attente, prendre juste le suivant de la liste globale
        const nextDoc = screeningDocs[currentIndex + 1] || screeningDocs[0] || null;
        setSelectedScreeningDoc(nextDoc);
        if (nextDoc) {
          setDecisionReason(nextDoc.screeningReason || "");
          setDecisionNotes(nextDoc.screeningNotes || "");
        }
      }
    } catch (err: any) {
      alert(err.message || "Erreur lors de la soumission de la décision.");
    }
  };

  useEffect(() => {
    if (activeTab === "screening") {
      loadScreeningData();
    }
  }, [activeTab, projectContext]);

  useEffect(() => {
    getFilterOptions()
      .then((opts) => {
        setFilterOptions(opts);
        const years =
          opts.year
            ?.map((y) => Number(y.value))
            .filter((y) => Number.isFinite(y) && y > 0) ?? [];
        if (years.length > 0) {
          setYearRange([Math.min(...years), Math.max(...years)]);
        }
      })
      .catch((err) => console.error(err));
  }, []);

  useEffect(() => {
    setFilters((prev) => ({ ...prev, projectContext }));
    setPage(1);
    setSelectedResult(null);
    setSelectedDocument(null);
    setEvidenceSummary(null);
  }, [projectContext]);

  useEffect(() => {
    if (activeTab === "stats") {
      fetchCorpusStats().then(setCorpusStats).catch(console.error);
      fetchGesicaStats().then(setGesicaStats).catch(console.error);
    }
    if (activeTab === "scenarios") {
      fetchGesicaScenarios().then(setGesicaScenarios).catch(console.error);
    }
  }, [activeTab]);

  const effectiveFilters = useMemo<SearchFilters>(
    () => ({
      ...filters,
      projectContext,
      yearMin: yearRange[0],
      yearMax: yearRange[1],
    }),
    [filters, projectContext, yearRange],
  );

  const dedupedResults = useMemo(() => {
    const seen = new Set<string>();
    return results.filter((result) => {
      const key = `${result.documentId}-${result.chunkIndex}-${result.content}`;
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

  async function loadDocumentDetail(result: SearchResult) {
    setSelectedResult(result);
    setSelectedDocument(null);
    setEvidenceSummary(null);
    setDetailLoading(true);
    try {
      const [detail, summary] = await Promise.all([
        fetchDocumentDetail(result.documentId),
        fetchEvidenceSummary(result.documentId).catch(() => null),
      ]);
      setSelectedDocument(detail);
      setEvidenceSummary(summary);
    } catch (err) {
      console.error(err);
    } finally {
      setDetailLoading(false);
    }
  }

  async function handleSearch() {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setPage(1);
    setSelectedResult(null);
    setSelectedDocument(null);
    setEvidenceSummary(null);
    try {
      const data = await searchDocuments({
        queryText: query,
        mode,
        limit: 100,
        filters: effectiveFilters,
      });
      setResults(data.results);
      const first = data.results[0] ?? null;
      if (first) {
        await loadDocumentDetail(first);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur inconnue");
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setFilters({ projectContext });
    setResults([]);
    setError(null);
    setPage(1);
    setSelectedResult(null);
    setSelectedDocument(null);
    setEvidenceSummary(null);
    const years =
      filterOptions?.year
        ?.map((y) => Number(y.value))
        .filter((y) => Number.isFinite(y) && y > 0) ?? [];
    if (years.length > 0) {
      setYearRange([Math.min(...years), Math.max(...years)]);
    }
  }

  function handleExport(format: "csv" | "json") {
    if (!dedupedResults.length) return;
    if (format === "json") {
      const blob = new Blob([JSON.stringify(dedupedResults, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "literev-results.json";
      a.click();
      URL.revokeObjectURL(a.href);
      return;
    }
    const headers = ["title", "score", "source", "year", "projectContext", "sourceType", "diseaseOrCondition", "scenarioType", "geographicScope", "evidenceCategory", "url"];
    const rows = dedupedResults.map((r) =>
      headers.map((h) => csvEscape((r as Record<string, unknown>)[h])).join(","),
    );
    const blob = new Blob([[headers.join(","), ...rows].join("\n")], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "literev-results.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  const hasResults = dedupedResults.length > 0;

  const detailView = useMemo<DetailView | null>(() => {
    if (!selectedResult) return null;
    const doc = selectedDocument?.document;
    const excerpt = getReadableExcerpt(selectedResult, selectedDocument);
    return {
      id: doc?.id ?? selectedResult.documentId ?? null,
      title: doc?.title ?? selectedResult.title ?? "Sans titre",
      abstract: doc?.abstract ?? "",
      excerpt,
      source: doc?.source ?? selectedResult.source ?? "—",
      year: doc?.year?.toString() ?? selectedResult.year?.toString() ?? "—",
      url: doc?.url ?? selectedResult.url ?? "",
      externalId: doc?.externalId ?? "—",
      projectContext: doc?.projectContext ?? selectedResult.projectContext ?? "—",
      sourceType: doc?.sourceType ?? selectedResult.sourceType ?? "—",
      disease: doc?.diseaseOrCondition ?? selectedResult.diseaseOrCondition ?? "—",
      scenario: doc?.scenarioType ?? selectedResult.scenarioType ?? "—",
      geography: doc?.geographicScope ?? selectedResult.geographicScope ?? "—",
      evidence: doc?.evidenceCategory ?? selectedResult.evidenceCategory ?? "—",
      chunkCount: selectedDocument?.chunks?.length ?? 0,
    };
  }, [selectedDocument, selectedResult]);

  const tabs: Array<{ id: AppTab; label: string; icon: React.ReactNode }> = [
    { id: "search", label: "Recherche", icon: <BookOpen size={14} /> },
    { id: "assistant", label: "Assistant RAG", icon: <Zap size={14} className="text-cyan-400" /> },
    { id: "screening", label: "Screening PRISMA", icon: <CheckSquare size={14} className="text-emerald-400" /> },
    { id: "scenarios", label: "Scénarios GESICA", icon: <Activity size={14} /> },
    { id: "stats", label: "Statistiques", icon: <BarChart2 size={14} /> },
  ];

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
                Interface unifiée pour GeoAI4EI, GESICA et EVA — moteur FastAPI + PostgreSQL/pgvector.
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              {(
                [
                  ["geoai4ei", "GeoAI4EI"],
                  ["gesica", "GESICA"],
                  ["eva", "EVA"],
                ] as const
              ).map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setProjectContext(value)}
                  className={`rounded-2xl border px-5 py-3 text-left transition ${
                    projectContext === value
                      ? "border-cyan-400 bg-cyan-500/10 text-white shadow-2xl"
                      : "border-white/10 bg-white/5 text-slate-300 hover:border-white/20 hover:bg-white/10"
                  }`}
                >
                  <div className="text-sm font-semibold">{label}</div>
                </button>
              ))}
            </div>
          </div>

          <div className="mt-6 flex gap-1 rounded-2xl border border-white/10 bg-slate-900/60 p-1 w-fit">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 rounded-xl px-4 py-2 text-sm transition ${
                  activeTab === tab.id
                    ? "bg-cyan-500 text-slate-950 font-semibold"
                    : "text-slate-300 hover:bg-white/10"
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1380px] px-6 py-8">

        {activeTab === "stats" && (
          <StatsView corpusStats={corpusStats} gesicaStats={gesicaStats} />
        )}

        {activeTab === "scenarios" && (
          <ScenariosView scenarios={gesicaScenarios} />
        )}

        {activeTab === "assistant" && (
          <AssistantView
            question={assistantQuestion}
            setQuestion={setAssistantQuestion}
            response={assistantResponse}
            loading={assistantLoading}
            error={assistantError}
            onAsk={handleAskAssistant}
            projectContext={projectContext}
          />
        )}

        {activeTab === "screening" && (
          <ScreeningView
            docs={screeningDocs}
            prismaFlow={prismaFlow}
            loading={screeningLoading}
            error={screeningError}
            selectedDoc={selectedScreeningDoc}
            setSelectedDoc={setSelectedScreeningDoc}
            reason={decisionReason}
            setReason={setDecisionReason}
            notes={decisionNotes}
            setNotes={setDecisionNotes}
            onDecision={handleScreeningDecision}
            projectContext={projectContext}
          />
        )}

        {activeTab === "search" && (
          <div className="grid gap-6 xl:grid-cols-[280px_minmax(0,1fr)]">
            <aside className="xl:sticky xl:top-8 xl:self-start">
              <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold text-white">Filtres</h2>
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

                <div className="mt-5 space-y-4">
                  {FILTER_FIELDS.map(([key, label]) => {
                    const options = filterOptions?.[key] ?? [];
                    return (
                      <label key={key} className="block">
                        <span className="mb-2 block text-sm font-medium text-slate-200">
                          {label}
                        </span>
                        <select
                          value={(filters as Record<string, string | undefined>)[key] ?? ""}
                          onChange={(e) =>
                            setFilters((prev) => ({
                              ...prev,
                              [key]: e.target.value || undefined,
                            }))
                          }
                          className="w-full appearance-none rounded-2xl border border-white/10 bg-slate-950/80 px-3 py-3 text-sm text-white focus:border-cyan-400 focus:outline-none"
                        >
                          <option value="">Tous</option>
                          {options.map((opt) => (
                            <option key={String(opt.value)} value={String(opt.value)}>
                              {opt.label}
                            </option>
                          ))}
                        </select>
                      </label>
                    );
                  })}

                  <div>
                    <span className="mb-2 block text-sm font-medium text-slate-200">
                      Année{" "}
                      <span className="font-mono text-cyan-300">
                        {yearRange[0]} — {yearRange[1]}
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
                  </div>
                </div>
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
                      className={`rounded-xl px-4 py-2 capitalize transition ${
                        mode === item
                          ? "bg-cyan-500 text-slate-950"
                          : "text-slate-300 hover:bg-white/10"
                      }`}
                    >
                      {item}
                    </button>
                  ))}
                </div>

                <div className="flex flex-col gap-3 lg:flex-row">
                  <input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                    placeholder={
                      mode === "semantic"
                        ? "Ex. ambulance demand forecasting"
                        : "Ex. ambulance AND forecasting"
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
                      résultat{dedupedResults.length > 1 ? "s" : ""} · {totalPages > 1 ? `page ${page}/${totalPages}` : "1 page"}
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

                  <div className="grid gap-6 2xl:grid-cols-[minmax(0,1fr)_380px]">
                    <div className="space-y-4">
                      {pagedResults.map((result) => (
                        <article
                          key={`${result.documentId}-${result.chunkIndex}-${result.content}`}
                          className={`rounded-3xl border bg-white/5 p-5 shadow-2xl transition ${
                            selectedResult?.id === result.id
                              ? "border-cyan-400/60"
                              : "border-white/10 hover:border-cyan-400/40"
                          }`}
                        >
                          <div className="flex items-start justify-between gap-4">
                            <button
                              type="button"
                              onClick={() => loadDocumentDetail(result)}
                              className="text-left"
                            >
                              <h3 className="text-xl font-semibold text-white hover:text-cyan-300">
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
                                <ExternalLink size={14} />
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
                            {result.projectContext && (
                              <span className="rounded-full bg-cyan-500/10 px-2 py-1 text-cyan-200">
                                {result.projectContext}
                              </span>
                            )}
                            {result.scenarioType && (
                              <span className="rounded-full bg-violet-500/10 px-2 py-1 text-violet-200">
                                {result.scenarioType}
                              </span>
                            )}
                            {result.evidenceCategory && (
                              <span className="rounded-full bg-slate-700/60 px-2 py-1 text-slate-300">
                                {result.evidenceCategory}
                              </span>
                            )}
                          </div>

                          <p className="mt-4 text-sm leading-6 text-slate-200">
                            {result.highlight || result.content}
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
                                  className={`rounded-full border px-3 py-1 text-xs transition ${
                                    relevanceMap[result.id] === tag
                                      ? "border-cyan-400 bg-cyan-500/15 text-cyan-200"
                                      : "border-white/10 bg-white/5 text-slate-400 hover:border-white/20 hover:text-slate-200"
                                  }`}
                                >
                                  {tag}
                                </button>
                              ),
                            )}
                          </div>
                        </article>
                      ))}

                      {totalPages > 1 && (
                        <div className="flex items-center justify-center gap-2 pt-2">
                          <button
                            type="button"
                            disabled={page === 1}
                            onClick={() => setPage((p) => p - 1)}
                            className="rounded-xl border border-white/10 px-4 py-2 text-sm text-slate-300 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-30"
                          >
                            Précédent
                          </button>
                          <span className="text-sm text-slate-400">
                            {page} / {totalPages}
                          </span>
                          <button
                            type="button"
                            disabled={page === totalPages}
                            onClick={() => setPage((p) => p + 1)}
                            className="rounded-xl border border-white/10 px-4 py-2 text-sm text-slate-300 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-30"
                          >
                            Suivant
                          </button>
                        </div>
                      )}
                    </div>

                    <aside className="2xl:sticky 2xl:top-8 2xl:self-start">
                      <div className="min-h-[220px] rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
                        {!selectedResult ? (
                          <div className="text-sm leading-6 text-slate-300">
                            Cliquez sur un résultat pour afficher le détail du document.
                          </div>
                        ) : detailLoading ? (
                          <div className="text-sm leading-6 text-slate-300">
                            Chargement du document complet...
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
                              <div>
                                <a
                                  href={detailView.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center gap-2 rounded-xl border border-white/10 px-3 py-2 text-sm text-slate-200 hover:bg-white/10"
                                >
                                  Open source
                                  <ExternalLink size={16} />
                                </a>
                              </div>
                            )}

                            <section>
                              <h3 className="mb-2 font-medium text-white">Extrait</h3>
                              <p className="rounded-2xl border border-white/10 bg-white/5 p-4 leading-6">
                                {detailView?.excerpt || "—"}
                              </p>
                            </section>

                            <section>
                              <h3 className="mb-2 font-medium text-white">Abstract</h3>
                              <p className="rounded-2xl border border-white/10 bg-white/5 p-4 leading-6">
                                {detailView?.abstract || "—"}
                              </p>
                            </section>

                            {evidenceSummary && (
                              <GesicaSignalsPanel summary={evidenceSummary} />
                            )}

                            <section>
                              <h3 className="mb-2 font-medium text-white">Métadonnées</h3>
                              <dl className="grid grid-cols-1 gap-3 rounded-2xl border border-white/10 bg-white/5 p-4">
                                <div><dt className="text-slate-400">ID</dt><dd>{detailView?.id ?? "—"}</dd></div>
                                <div><dt className="text-slate-400">Source</dt><dd>{detailView?.source}</dd></div>
                                <div><dt className="text-slate-400">Année</dt><dd>{detailView?.year}</dd></div>
                                <div><dt className="text-slate-400">External ID</dt><dd>{detailView?.externalId}</dd></div>
                                <div><dt className="text-slate-400">Projet</dt><dd>{detailView?.projectContext}</dd></div>
                                <div><dt className="text-slate-400">Type</dt><dd>{detailView?.sourceType}</dd></div>
                                <div><dt className="text-slate-400">Pathologie</dt><dd>{detailView?.disease}</dd></div>
                                <div><dt className="text-slate-400">Scénario</dt><dd>{detailView?.scenario}</dd></div>
                                <div><dt className="text-slate-400">Zone</dt><dd>{detailView?.geography}</dd></div>
                                <div><dt className="text-slate-400">Preuve</dt><dd>{detailView?.evidence}</dd></div>
                                <div><dt className="text-slate-400">Chunks</dt><dd>{detailView?.chunkCount}</dd></div>
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
        )}
      </main>
    </div>
  );
}
