import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  ArrowLeft, Brain,
  ChevronDown, ChevronUp, Database, ExternalLink, FileText,
  Layers, MessageSquare, RefreshCw, RotateCcw, Search,
  Shield, Terminal, Zap, AlertTriangle,
  Globe, Upload, CheckCircle2, AlertCircle, Info,
  Microscope, Loader2, Download, Table2, BookOpen,
  Network, Bell, Users, Rss
} from "lucide-react";
import {
  fetchScenarioDetail,
  fetchScenarioCorpus,
  fetchScenarioModelStatus,
  runScenarioModel,
  fetchScenarioClustering,
  askScenarioRagStreamFiltered,
  fetchScenarioPrisma,
  uploadScenarioDataset,
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
  type EnrichmentStatus,
  type ScenarioDetail,
  type ScenarioCorpus,
  type ModelStatus,
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
  scenarioBase,
  isUserScenario,
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
  return (
    <div className="flex items-center justify-center py-8 text-white/50 gap-2">
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
        icon={<Search size={14} className="text-brand-400" />}
        title="Stratégie de Recherche"
        subtitle="Requêtes utilisées pour récupérer les articles du corpus"
      />
      {/* Boolean Queries PubMed */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <Terminal size={12} className="text-brand-400" />
          <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">
            Requêtes Booléennes PubMed ({detail.boolean_queries.length})
          </span>
        </div>
        <div className="space-y-2">
          {detail.boolean_queries.length > 0 ? detail.boolean_queries.map((q, i) => (
            <div key={i} className="group relative rounded-xl border border-brand-500/10 bg-brand-500/5 px-3 py-2">
              <div className="flex items-start gap-2">
                <span className="text-[10px] font-mono text-brand-500 shrink-0 mt-0.5">Q{i + 1}</span>
                <code className="text-xs text-brand-200 font-mono break-all leading-5">{q}</code>
              </div>
              <a
                href={`https://pubmed.ncbi.nlm.nih.gov/?term=${encodeURIComponent(q)}`}
                target="_blank"
                rel="noopener noreferrer"
                className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition text-brand-400 hover:text-brand-300"
                title="Ouvrir dans PubMed"
              >
                <ExternalLink size={11} />
              </a>
            </div>
          )) : (
            <p className="text-xs text-white/35 italic">Aucune requête booléenne définie pour ce scénario.</p>
          )}
        </div>
      </div>
      {/* Natural Language Queries */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <MessageSquare size={12} className="text-brand-400" />
          <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">
            Requêtes Langage Naturel : Recherche Sémantique ({detail.nl_queries.length})
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
            <p className="text-xs text-white/35 italic">Aucune requête NL définie pour ce scénario.</p>
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
            {showPrompt ? "Masquer" : "Afficher"} le prompt d'extraction d'évidence
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
    </div>
  );
}

// ─── Section: Variables & Databases (NOUVEAU) ──────────────────────────────────

function VariablesSection({ detail, scenarioId }: { detail: ScenarioDetail; scenarioId: string }) {
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

  // Polling si generation en cours
  React.useEffect(() => {
    if (!llmVars || llmVars.status !== 'generating') return;
    setLlmGenStatus('Generation en cours...');
    llmPollRef.current = setInterval(() => {
      getVariablesGenerationStatus(scenarioId).then(s => {
        if (s.status === 'done') {
          if (llmPollRef.current) clearInterval(llmPollRef.current);
          setLlmGenStatus(null);
          loadLlmVars();
        } else if (s.status === 'error') {
          if (llmPollRef.current) clearInterval(llmPollRef.current);
          setLlmGenStatus(null);
          setLlmError(s.error ?? 'Erreur de generation');
        }
      });
    }, 5000);
    return () => { if (llmPollRef.current) clearInterval(llmPollRef.current); };
  }, [llmVars, scenarioId, loadLlmVars]);

  const handleGenerateLlm = async () => {
    setLlmGenerating(true);
    setLlmGenStatus('Lancement de la generation...');
    setLlmError(null);
    try {
      await generateScenarioVariables(scenarioId);
      llmPollRef.current = setInterval(() => {
        getVariablesGenerationStatus(scenarioId).then(s => {
          if (s.status === 'done') {
            if (llmPollRef.current) clearInterval(llmPollRef.current);
            setLlmGenStatus(null);
            setLlmGenerating(false);
            loadLlmVars();
          } else if (s.status === 'error') {
            if (llmPollRef.current) clearInterval(llmPollRef.current);
            setLlmGenStatus(null);
            setLlmGenerating(false);
            setLlmError(s.error ?? 'Erreur de generation');
          }
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
      const res = await uploadScenarioDataset(scenarioId, file);
      setUploadResult(res);
      setFile(null);
    } catch (err: any) {
      setUploadError(err.message || "Une erreur est survenue lors de l'upload.");
    } finally {
      setLoading(false);
    }
  };

  const variables = detail.variables_detail ? Object.entries(detail.variables_detail) : [];
  const totalVars = variables.length;
  const pluggedVars = variables.filter(([, v]) => v.plugged).length;
  const missingVars = totalVars - pluggedVars;

  return (
    <div className="space-y-4">

      {/* Banniere LLM Variables - pretes a valider */}
      {llmVars && !llmVars.status && !llmVars._validated && llmVars.predictor_variables && llmVars.predictor_variables.length > 0 && (
        <div className="rounded-2xl border border-gold-500/30 bg-gold-500/8 px-4 py-3 flex items-start gap-3">
          <Bell size={14} className="text-gold-400 shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-gold-300 mb-1">
              Variables & Modele generes automatiquement - validation requise
            </p>
            <p className="text-[10px] text-gold-200/70 leading-relaxed">
              {llmVars.predictor_variables?.length ?? 0} variables predictives et le modele recommande ont ete extraits automatiquement depuis les PICO de {llmVars._meta?.pico_articles_used ?? '?'} articles. Verifiez et validez avant utilisation.
            </p>
          </div>
          <button onClick={handleValidateLlm} disabled={llmValidating}
            className="shrink-0 flex items-center gap-1.5 rounded-xl bg-gold-500/20 hover:bg-gold-500/30 border border-gold-500/30 text-gold-300 font-semibold px-3 py-1.5 text-xs transition disabled:opacity-50">
            {llmValidating ? (<Loader2 size={11} className="animate-spin" />) : (<CheckCircle2 size={11} />)}
            Valider
          </button>
        </div>
      )}

      {/* Banniere : variables validees */}
      {llmVars && llmVars._validated && (
        <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 px-4 py-2.5 flex items-center gap-2">
          <CheckCircle2 size={12} className="text-brand-400 shrink-0" />
          <p className="text-[10px] text-brand-300">
            Variables & Modele valides par un relecteur humain le {llmVars._generated_at ? new Date(llmVars._generated_at).toLocaleDateString('fr-FR') : ''}.
          </p>
        </div>
      )}

      {/* Etat vide : pas d'articles disponibles */}
      {llmVars && llmVars.status === 'empty' && (
        <div className="rounded-2xl border border-slate-500/20 bg-slate-500/5 px-4 py-3 flex items-start gap-3">
          <div className="text-xs text-slate-300/80">
            <strong className="text-slate-200">Aucun article disponible</strong> — {(llmVars as any).message ?? 'Ajoutez des articles ou abaissez le seuil de similarité pour générer les variables.'}
          </div>
        </div>
      )}

      {/* Bouton generer + erreur */}
      <div className="flex items-center gap-3 flex-wrap">
        <button onClick={handleGenerateLlm} disabled={llmGenerating || llmLoading}
          className="flex items-center gap-1.5 rounded-xl border border-brand-500/30 bg-brand-500/10 hover:bg-brand-500/20 text-brand-300 font-medium px-3 py-1.5 text-xs transition disabled:opacity-50">
          {llmGenerating ? (<Loader2 size={11} className="animate-spin" />) : (<Brain size={11} />)}
          Generer automatiquement depuis les PICO
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
              title="Variables generees par LLM"
              subtitle={`Extraites depuis les PICO de ${llmVars._meta?.pico_articles_used ?? '?'} articles`}
            />
            <div className="flex gap-2">
              {llmVars._validated ? (
                <span className="rounded-full bg-brand-500/10 border border-brand-500/20 px-2.5 py-1 text-[10px] font-semibold text-brand-300">Validees</span>
              ) : (
                <span className="rounded-full bg-gold-500/10 border border-gold-500/20 px-2.5 py-1 text-[10px] font-semibold text-gold-300">En attente de validation</span>
              )}
            </div>
          </div>

          {/* Outcome principal LLM */}
          {llmVars.primary_outcome && (
            <div className="rounded-2xl border border-gold-500/10 bg-gold-500/5 p-4 space-y-1">
              <p className="text-[10px] font-bold uppercase tracking-wider text-gold-400">Outcome principal</p>
              <p className="text-sm font-medium text-gold-200">{llmVars.primary_outcome.name}</p>
              <p className="text-xs text-white/55">{llmVars.primary_outcome.definition}</p>
              <p className="text-[10px] text-white/35">Mesure : {llmVars.primary_outcome.measurement} - Horizon : {llmVars.primary_outcome.timeframe}</p>
            </div>
          )}

          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left">
              <thead>
                <tr className="border-b border-white/5 text-[10px] text-white/50 uppercase tracking-wider">
                  <th className="py-2.5 px-3">Variable</th>
                  <th className="py-2.5 px-3">Type</th>
                  <th className="py-2.5 px-3">Definition</th>
                  <th className="py-2.5 px-3">Source</th>
                  <th className="py-2.5 px-3 text-center">Importance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 text-xs">
                {llmVars.predictor_variables?.map((v, i) => (
                  <tr key={i} className="hover:bg-white/1">
                    <td className="py-3 px-3 font-mono text-brand-300 font-medium">{v.name}</td>
                    <td className="py-3 px-3">
                      <span className="rounded bg-white/5 border border-white/10 px-1.5 py-0.5 text-[10px] text-white/50">{v.type}</span>
                    </td>
                    <td className="py-3 px-3 text-white/70 leading-5 max-w-[200px]">{v.definition}</td>
                    <td className="py-3 px-3 text-white/50 font-mono text-[11px]">{v.data_source}</td>
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
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">Algorithme recommande</p>
              <p className="text-sm font-semibold text-white">{llmVars.recommended_algorithm.primary}</p>
              <p className="text-xs text-white/55">{llmVars.recommended_algorithm.rationale}</p>
              {llmVars.recommended_algorithm.alternatives?.length > 0 && (
                <p className="text-[10px] text-white/35">Alternatives : {llmVars.recommended_algorithm.alternatives.join(', ')}</p>
              )}
            </div>
          )}
        </div>
      )}

    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      {/* Colonne de gauche: Variables & Outcomes */}
      <div className="lg:col-span-2 space-y-6">
        {/* Outcome */}
        <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-4">
          <SectionHeader
            icon={<Zap size={14} className="text-gold-400" />}
            title="Outcome étudié & surveillé"
            subtitle="Définition clinique de l'indicateur principal du modèle"
          />
          <div className="rounded-2xl border border-gold-500/10 bg-gold-500/5 p-4">
            <p className="text-sm font-medium text-gold-200 leading-6">
              {detail.outcome_definition || "Outcome clinique non spécifié."}
            </p>
          </div>
        </div>

        {/* Liste des variables */}
        <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-4">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <SectionHeader
              icon={<Database size={14} className="text-brand-400" />}
              title="Variables du modèle"
              subtitle="Paramètres d'entrée du modèle prédictif"
            />
            <div className="flex gap-2">
              <span className="rounded-full bg-brand-500/10 border border-brand-500/20 px-2.5 py-1 text-[10px] font-semibold text-brand-300">
                {pluggedVars} branchées
              </span>
              {missingVars > 0 && (
                <span className="rounded-full bg-rose-500/10 border border-rose-500/20 px-2.5 py-1 text-[10px] font-semibold text-rose-300">
                  {missingVars} manquantes
                </span>
              )}
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left">
              <thead>
                <tr className="border-b border-white/5 text-[10px] text-white/50 uppercase tracking-wider">
                  <th className="py-2.5 px-3">Variable</th>
                  <th className="py-2.5 px-3">Définition clinique / Rôle</th>
                  <th className="py-2.5 px-3">Source de données</th>
                  <th className="py-2.5 px-3 text-center">Statut</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 text-xs">
                {variables.map(([name, varInfo]) => (
                  <tr key={name} className="hover:bg-white/1">
                    <td className="py-3 px-3 font-mono text-brand-300 font-medium">{name}</td>
                    <td className="py-3 px-3 text-white/70 leading-5">{varInfo.definition}</td>
                    <td className="py-3 px-3 text-white/50 font-mono text-[11px]">{varInfo.source}</td>
                    <td className="py-3 px-3 text-center">
                      {varInfo.plugged ? (
                        <span className="inline-flex items-center gap-1 rounded-full bg-brand-500/15 border border-brand-500/20 px-2 py-0.5 text-[10px] text-brand-300">
                          <CheckCircle2 size={10} /> Connecté
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 rounded-full bg-rose-500/15 border border-rose-500/20 px-2 py-0.5 text-[10px] text-rose-300" title="Données réelles manquantes">
                          <AlertCircle size={10} /> Non connecté
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Colonne de droite: Bases de données & Upload */}
      <div className="space-y-6">
        {/* Bases de données utilisées */}
        <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-4">
          <SectionHeader
            icon={<Globe size={14} className="text-brand-400" />}
            title="Bases de données requises"
            subtitle="Flux d'informations à connecter pour ce scénario"
          />
          <div className="space-y-2">
            {detail.databases && detail.databases.length > 0 ? (
              detail.databases.map((db, i) => (
                <div key={i} className="flex items-center gap-2.5 rounded-xl border border-white/5 bg-white/3 px-3 py-2.5 text-xs text-white/70">
                  <Database size={12} className="text-brand-400 shrink-0" />
                  <span>{db}</span>
                </div>
              ))
            ) : (
              <p className="text-xs text-white/35 italic">Aucune base de données répertoriée.</p>
            )}
          </div>
        </div>

        {/* Zone d'upload interactif */}
        <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-4">
          <SectionHeader
            icon={<Upload size={14} className="text-brand-400" />}
            title="Importer des données réelles"
            subtitle="Uploadez vos fichiers CSV/Excel pour alimenter les variables manquantes"
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
            <p className="text-xs text-white/70 font-medium">Glissez-déposez votre fichier ici</p>
            <p className="text-[10px] text-white/35">Formats acceptés : CSV, Excel (.xlsx, .xls)</p>
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
              {uploading ? "Importation..." : "Lancer l'importation"}
            </button>
          )}

          {uploadError && <ErrorBox message={uploadError} />}

          {uploadResult && (
            <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 p-4 space-y-3">
              <div className="flex items-center gap-1.5 text-brand-300 text-xs font-semibold">
                <CheckCircle2 size={14} /> Importation réussie !
              </div>
              <p className="text-[11px] text-white/70 leading-4">
                {uploadResult.message}
              </p>
              <div className="rounded-lg bg-forest-900/50 p-2 text-[10px] font-mono text-white/50 space-y-1">
                <div>Lignes détectées : <span className="text-brand-300">{uploadResult.detected_rows}</span></div>
                <div>Colonnes : <span className="text-brand-300">{uploadResult.detected_columns?.slice(0, 5).join(", ")}{uploadResult.detected_columns?.length > 5 ? "..." : ""}</span></div>
              </div>
              <div className="flex items-start gap-1 text-[10px] text-white/35">
                <Info size={10} className="shrink-0 mt-0.5" />
                <span>Les variables manquantes du modèle seront automatiquement branchées lors du prochain recalcul.</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
    </div>
  );
}

// ─── Section: Model ───────────────────────────────────────────────────────────

function ModelSection({ scenarioId }: { scenarioId: string }) {
  const [data, setData] = useState<ModelStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [recalculating, setRecalculating] = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    fetchScenarioModelStatus(scenarioId)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [scenarioId]);

  useEffect(() => {
    load();
  }, [load]);

  const runModel = async () => {
    setRecalculating(true);
    setError(null);
    try {
      const res = await runScenarioModel(scenarioId);
      setData(res);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setRecalculating(false);
    }
  };

  if (loading) return <LoadingSpinner text="Chargement du statut du modèle..." />;
  if (error || !data) return <ErrorBox message={error ?? "Erreur statut modèle"} />;

  const colors = STATUS_COLORS[data.status_color] || STATUS_COLORS.green;

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      {/* Statut Live Card */}
      <div className={`rounded-3xl border ${colors.border} ${colors.bg} p-6 flex flex-col justify-between space-y-6 lg:col-span-1`}>
        <div>
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold uppercase tracking-wider text-white/50">Statut Live du Modèle</span>
            <span className="flex h-2 w-2 rounded-full relative">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${colors.dot}`} />
              <span className={`relative inline-flex rounded-full h-2 w-2 ${colors.dot}`} />
            </span>
          </div>
          <p className="mt-4 text-3xl font-extrabold text-white">{data.status_label}</p>
          {data.model_result && data.model_result.value !== undefined && (
            <div className="mt-4 rounded-2xl bg-white/5 p-4 border border-white/5">
              <p className="text-[10px] text-white/50 uppercase tracking-wider">Dernière valeur live calculée</p>
              <p className="text-3xl font-black text-brand-300 mt-1 font-mono">
                {typeof data.model_result.value === "number" ? data.model_result.value.toLocaleString() : data.model_result.value}
                {data.model_result.unit && <span className="text-sm font-normal ml-1 text-white/50">{data.model_result.unit}</span>}
              </p>
              <p className="text-[10px] text-white/35 mt-1.5 font-mono">Calculé le {new Date(data.timestamp).toLocaleString()}</p>
            </div>
          )}
        </div>

        <div className="space-y-3">
          <div className="flex items-center gap-2 text-xs text-white/50">
            <RefreshCw size={12} />
            <span>Mise à jour automatique à chaque nouvelle valeur</span>
          </div>
          <button
            onClick={runModel}
            disabled={recalculating}
            className="w-full flex items-center justify-center gap-1.5 rounded-xl bg-white text-forest-950 font-semibold py-2 text-xs hover:bg-forest-200 transition disabled:opacity-50"
          >
            <RotateCcw size={12} className={recalculating ? "animate-spin" : ""} />
            {recalculating ? "Recalcul..." : "Rerun le modèle manuellement"}
          </button>
        </div>
      </div>

      {/* Algorithme & Seuils */}
      <div className="lg:col-span-2 space-y-6">
        {/* Info algorithme */}
        <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-4">
          <SectionHeader
            icon={<Brain size={14} className="text-brand-400" />}
            title="Algorithme & Paramètres"
            subtitle="Spécifications techniques du modèle prédictif"
          />
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 text-xs">
            <div className="rounded-xl border border-white/5 bg-white/2 px-3 py-2">
              <span className="text-white/35">Modèle mathématique</span>
              <p className="font-semibold text-white mt-1">{data.model_info.algorithm}</p>
            </div>
            <div className="rounded-xl border border-white/5 bg-white/2 px-3 py-2">
              <span className="text-white/35">Fréquence de calcul</span>
              <p className="font-semibold text-white mt-1">{data.model_info.update_frequency}</p>
            </div>
            <div className="rounded-xl border border-white/5 bg-white/2 px-3 py-2 sm:col-span-2">
              <span className="text-white/35">Indicateur de sortie (Outcome)</span>
              <p className="font-semibold text-white mt-1">{data.model_info.output}</p>
            </div>
          </div>
        </div>

        {/* Seuils d'alerte */}
        <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-4">
          <SectionHeader
            icon={<Shield size={14} className="text-brand-400" />}
            title="Seuils d'alerte et de décision"
            subtitle="Séparations définissant la couleur d'alerte"
          />
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 text-xs">
            <div className="rounded-xl border border-brand-500/10 bg-brand-500/5 px-3 py-2.5">
              <span className="font-semibold text-brand-300">Vert (Situation normale)</span>
              <p className="text-white/50 mt-1 font-mono">{data.alert_thresholds.green.condition}</p>
              <p className="text-[10px] text-white/35 mt-1">{data.alert_thresholds.green.label}</p>
            </div>
            <div className="rounded-xl border border-gold-500/10 bg-gold-500/5 px-3 py-2.5">
              <span className="font-semibold text-gold-300">Orange (Vigilance)</span>
              <p className="text-white/50 mt-1 font-mono">{data.alert_thresholds.orange.condition}</p>
              <p className="text-[10px] text-white/35 mt-1">{data.alert_thresholds.orange.label}</p>
            </div>
            <div className="rounded-xl border border-rose-500/10 bg-rose-500/5 px-3 py-2.5">
              <span className="font-semibold text-rose-300">Rouge (Alerte critique)</span>
              <p className="text-white/50 mt-1 font-mono">{data.alert_thresholds.red.condition}</p>
              <p className="text-[10px] text-white/35 mt-1">{data.alert_thresholds.red.label}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Section: Corpus ──────────────────────────────────────────────────────────

function CorpusSection({ scenarioId }: { scenarioId: string; detail: ScenarioDetail }) {
  const [data, setData] = useState<ScenarioCorpus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchScenarioCorpus(scenarioId)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [scenarioId]);

  if (loading) return <LoadingSpinner text="Chargement du corpus d'articles..." />;
  if (error || !data) return <ErrorBox message={error ?? "Erreur corpus"} />;

  return (
    <div className="space-y-4">
      {/* Bannière avertissement sélection automatique */}
      <div className="rounded-2xl border border-gold-500/20 bg-gold-500/5 px-4 py-3 flex items-start gap-3">
        <AlertTriangle size={14} className="text-gold-400 shrink-0 mt-0.5" />
        <div className="text-xs text-gold-200/80 leading-relaxed">
          <strong className="text-gold-300">Sélection automatique</strong> — Ces articles ont été récupérés par recherche lexicale (requête PubMed) et/ou sémantique (similarité vectorielle). <strong>Aucun n'a été validé par un relecteur humain.</strong> Pour une revue systématique formelle, un screening humain en double-aveugle est requis (onglet Revue).
        </div>
      </div>
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      {/* Liste des articles */}
      <div className="lg:col-span-2 space-y-4">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <SectionHeader
            icon={<FileText size={14} className="text-brand-400" />}
            title={`Corpus d'articles (${data.total} articles indexés)`}
            subtitle="Articles sélectionnés automatiquement — en attente de validation humaine (screening)"
          />
          {data.above_threshold !== undefined && (
            <div className="flex items-center gap-2">
              <span className="rounded-full bg-brand-500/15 border border-brand-500/30 px-3 py-1 text-[10px] font-semibold text-brand-300">
                {data.above_threshold} auto-sélectionnés (seuil)
              </span>
              {data.total > data.above_threshold && (
                <span className="rounded-full bg-white/5 border border-white/10 px-3 py-1 text-[10px] text-white/40">
                  {data.total - data.above_threshold} sous le seuil
                </span>
              )}
            </div>
          )}
        </div>
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
            <p className="text-xs text-white/35 italic">Aucun article dans ce corpus.</p>
          )}
        </div>
      </div>

      {/* Distribution et stats */}
      <div className="space-y-6">
        {/* Années */}
        <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-4">
          <SectionHeader
            icon={<Globe size={14} className="text-brand-400" />}
            title="Distribution par Année"
          />
          <div className="space-y-2 text-xs">
            {data.year_distribution.slice(0, 6).map((item) => (
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
            title="Sources Littérature"
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

  const statusLabel = { included: 'Inclus', excluded: 'Exclu', pending: 'En attente' }[screeningStatus] ?? screeningStatus;

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
                Full-text
              </span>
            )}
            {article.similarity_score !== undefined && article.similarity_score !== null && (
              <span className={`rounded-full px-2 py-0.5 text-[9px] font-medium ${
                article.similarity_score >= 0.45
                  ? 'bg-brand-500/15 border border-brand-500/30 text-brand-300'
                  : 'bg-white/5 border border-white/10 text-white/30'
              }`}>
                sim {article.similarity_score.toFixed(2)}
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
          {/* Screening PRISMA — Interface avancée */}
          <div className="rounded-xl border border-white/8 bg-white/3 p-3 space-y-3">
            <h5 className="text-[10px] font-semibold text-white/50 uppercase tracking-wider flex items-center gap-1">
              <CheckCircle2 size={10} />Décision de Screening (Inclusion / Exclusion)
            </h5>
            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <label className="block text-[10px] text-white/40 mb-1">Raison de l'exclusion (obligatoire si exclu)</label>
                <select
                  value={exclusionReason}
                  onChange={(e) => { e.stopPropagation(); setExclusionReason(e.target.value); }}
                  onClick={(e) => e.stopPropagation()}
                  className="w-full rounded-lg border border-white/10 bg-forest-950/80 px-2.5 py-1.5 text-xs text-white outline-none focus:border-brand-400"
                >
                  <option value="">-- Sélectionner une raison --</option>
                  <option value="wrong-population">Population non cible</option>
                  <option value="wrong-intervention">Méthode non cible</option>
                  <option value="wrong-outcome">Pas de métriques d'évaluation</option>
                  <option value="no-fulltext">Pas de texte intégral disponible</option>
                  <option value="duplicate">Doublon d'un autre article</option>
                  <option value="other">Autre (spécifier en notes)</option>
                </select>
              </div>
              <div>
                <label className="block text-[10px] text-white/40 mb-1">Notes de screening</label>
                <input
                  value={screeningNotes}
                  onChange={(e) => { e.stopPropagation(); setScreeningNotes(e.target.value); }}
                  onClick={(e) => e.stopPropagation()}
                  placeholder="Observations, commentaires..."
                  className="w-full rounded-lg border border-white/10 bg-forest-950/80 px-2.5 py-1.5 text-xs text-white outline-none focus:border-brand-400"
                />
              </div>
            </div>
            {screeningLoading ? (
              <div className="flex items-center gap-2 text-white/40 text-xs">
                <Loader2 size={12} className="animate-spin" />Enregistrement...
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
                  <AlertCircle size={11} />Exclure l'article
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); handleScreen('included'); }}
                  className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-semibold transition ${
                    screeningStatus === 'included'
                      ? 'border-brand-500/50 bg-brand-500/20 text-brand-200'
                      : 'border-brand-500/30 bg-brand-500/10 hover:bg-brand-500/20 text-brand-300'
                  }`}
                >
                  <CheckCircle2 size={11} />Inclure dans le corpus final
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); handleScreen('pending'); }}
                  className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-semibold transition ${
                    screeningStatus === 'pending'
                      ? 'border-gold-500/30 bg-gold-500/15 text-gold-300'
                      : 'border-white/10 bg-white/5 hover:bg-white/10 text-white/50'
                  }`}
                >
                  En attente
                </button>
              </div>
            )}
          </div>

          {/* PICO */}
          {picoLoading ? (
            <div className="flex items-center gap-2 text-white/35">
              <Loader2 size={12} className="animate-spin" />
              <span>Chargement PICO...</span>
            </div>
          ) : pico ? (
            <div className="rounded-xl border border-white/5 bg-white/2 p-3 space-y-2">
              <p className="text-[10px] font-semibold text-white/50 uppercase tracking-wider flex items-center gap-1">
                <Microscope size={10} />PICO
                {pico.pico_confidence != null && (
                  <span className="ml-auto font-mono text-white/35">Confiance : {Math.round(pico.pico_confidence * 100)}%</span>
                )}
              </p>
              <div className="grid grid-cols-2 gap-2">
                {[['P', 'Population', pico.P], ['I', 'Intervention', pico.I], ['C', 'Comparateur', pico.C], ['O', 'Outcome', pico.O]].map(([key, label, val]) => val && (
                  <div key={key} className="rounded-lg bg-white/3 border border-white/5 p-2">
                    <span className="text-[9px] font-bold text-brand-400 uppercase">{key} : {label}</span>
                    <p className="text-white/70 mt-0.5 leading-4">{val as string}</p>
                  </div>
                ))}
              </div>
              {pico.study_design && (
                <p className="text-[10px] text-white/35">Type d'étude : <span className="text-white/70">{pico.study_design}</span></p>
              )}
            </div>
          ) : picoLoaded ? (
            <p className="text-[10px] text-white/25 italic">PICO non encore extrait pour cet article.</p>
          ) : null}

          {article.abstract && (
            <div>
              <p className="font-semibold text-white/50 mb-1">Abstract</p>
              <p className="text-white/70 leading-5">{article.abstract}</p>
            </div>
          )}
          <div className="flex items-center gap-4 flex-wrap text-white/50 font-mono text-[10px] pt-1">
            {article.journal && <span>Journal: <span className="text-white/70">{article.journal}</span></span>}
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
                Lien direct <ExternalLink size={10} />
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
            const r = await fetch(`${scenarioBase(scenarioId)}/${scenarioId}/clustering/status`);
            if (!r.ok) return;
            const status = await r.json();
            if (status.status === "done" || (status.clusters && status.clusters.length > 0)) {
              stopPolling();
              handleResult(status);
            } else if (status.status === "error") {
              stopPolling();
              setError(status.error || "Erreur lors du clustering");
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
  }, [scenarioId, stopPolling, handleResult]);

  useEffect(() => {
    load();
    return () => stopPolling();
  }, [load, stopPolling]);

  const [vizTab, setVizTab] = useState<'scatter'|'graph'>('scatter');
  const activeClusterData = data?.clusters.find((c) => c.cluster_id === selectedCluster);

  return (
    <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-5">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <SectionHeader
          icon={<Layers size={14} className="text-brand-400" />}
          title="Clustering & Topic Modelling Avancé"
          subtitle="UMAP (Réduction 2D) + HDBSCAN (Clustering à densité) + Synthèses automatiques"
        />
        <button
          onClick={load}
          disabled={loading}
          className="flex items-center gap-1.5 rounded-xl border border-brand-500/20 bg-brand-500/10 px-3 py-1.5 text-xs text-brand-300 hover:bg-brand-500/20 transition disabled:opacity-50"
        >
          <RotateCcw size={11} className={loading ? "animate-spin" : ""} />
          {loading ? "Calcul..." : "Recalculer"}
        </button>
      </div>

      {(loading || polling) && (
        <LoadingSpinner text={polling
          ? "Calcul en cours en arrière-plan (UMAP + HDBSCAN)... Mise à jour automatique toutes les 5s"
          : "Lancement de l'analyse (Embeddings → UMAP 2D → HDBSCAN)..."
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
                  <div className="flex items-center gap-1">
                    <button onClick={()=>setVizTab("scatter")} className={`px-3 py-1 rounded-lg text-[10px] font-semibold uppercase tracking-wider transition ${vizTab==="scatter"?"bg-brand-700 text-gold-400":"text-white/60 hover:text-white hover:bg-white/8"}`}>UMAP 2D</button>
                    <button onClick={()=>setVizTab("graph")} className={`px-3 py-1 rounded-lg text-[10px] font-semibold uppercase tracking-wider transition ${vizTab==="graph"?"bg-brand-700 text-gold-400":"text-white/60 hover:text-white hover:bg-white/8"}`}>Knowledge Graph</button>
                  </div>
                  {vizTab==="scatter"&&(
                    <>
                      <p className="text-[10px] text-white/40 leading-4">Points = articles. Proximité = similarité. Nuages = groupes thématiques.</p>
                      <div className="w-full bg-[#0a1410] rounded-xl border border-white/5 overflow-hidden">
                        <UmapScatterPlot clusters={data.clusters} selectedCluster={selectedCluster} onSelectCluster={setSelectedCluster}/>
                      </div>
                      <div className="flex items-center justify-between text-[10px] text-white/25 font-mono">
                        <span>← UMAP dim 1 →</span>
                        <span>{data.n_docs} articles · {data.n_clusters} clusters</span>
                      </div>
                    </>
                  )}
                  {vizTab==="graph"&&(
                    <KnowledgeGraph clusters={data.clusters} selectedCluster={selectedCluster} onSelectCluster={setSelectedCluster}/>
                  )}
                </div>

                {/* Sélecteur de cluster de gauche */}
                <div className="space-y-2">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-white/50 block px-1">Sélectionner un groupe</span>
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
                        <span className="text-[10px] font-mono opacity-70 bg-white/5 rounded px-1.5 py-0.5">{c.n_docs} articles</span>
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
                          <p className="text-xs text-white/50 mt-0.5">{activeClusterData.n_docs} articles scientifiques denses dans ce groupe</p>
                        </div>
                      </div>
                    </div>

                    {/* Résumé clinique LLM */}
                    <div className="space-y-2">
                      <div className="flex items-center gap-1.5 text-xs font-semibold text-brand-300 uppercase tracking-wider">
                        <Brain size={13} />
                        Synthèse du groupe
                      </div>
                      <div className="rounded-2xl border border-brand-500/15 bg-brand-500/5 p-4 text-xs text-white/80 leading-6 italic">
                        "{activeClusterData.summary}"
                      </div>
                    </div>

                    {/* Mots-clés TF-IDF */}
                    <div className="space-y-2">
                      <p className="text-[10px] font-bold uppercase tracking-wider text-white/50">Mots-clés prépondérants</p>
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

                    {/* Article représentatif */}
                    <div className="space-y-2 border-t border-white/5 pt-4">
                      <p className="text-[10px] font-bold uppercase tracking-wider text-white/50">Article le plus central / représentatif</p>
                      <div className="rounded-xl border border-white/5 bg-white/3 p-3">
                        <div className="flex items-center gap-1.5 text-[10px] text-white/35 font-mono">
                          <span>ID: #{activeClusterData.representative_doc.id}</span>
                          {activeClusterData.representative_doc.year && <span>• {activeClusterData.representative_doc.year}</span>}
                          {activeClusterData.representative_doc.journal && <span>• {activeClusterData.representative_doc.journal}</span>}
                        </div>
                        <h5 className="text-xs font-semibold text-white mt-1.5 leading-5">{activeClusterData.representative_doc.title}</h5>
                      </div>
                    </div>
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
// Knowledge Graph : nœuds = clusters, arêtes = mots-clés partagés
function KnowledgeGraph({clusters,selectedCluster,onSelectCluster}:{clusters:ClusterResult[];selectedCluster:number|null;onSelectCluster:(id:number)=>void}) {
  const W=560,H=420;
  const denseC=clusters.filter(c=>!c.is_noise);
  const [hoveredNode,setHoveredNode]=React.useState<string|null>(null);
  const [tooltip,setTooltip]=React.useState<{x:number;y:number;text:string}|null>(null);
  if (!denseC.length) return <span className="text-xs text-white/40">Aucun cluster.</span>;

  // Build keyword nodes + cluster nodes
  type GNode={id:string;label:string;type:'cluster'|'keyword';cluster_id?:number;count?:number;x:number;y:number;r:number;color:string};
  type GEdge={from:string;to:string;weight:number};

  // Gather top keywords across all clusters
  const kwMap=new Map<string,{clusters:number[];count:number}>();
  denseC.forEach(c=>{
    (c.top_words||[]).slice(0,8).forEach((w,i)=>{
      const key=w.toLowerCase();
      if(!kwMap.has(key)) kwMap.set(key,{clusters:[],count:0});
      const entry=kwMap.get(key)!;
      entry.clusters.push(c.cluster_id);
      entry.count+=8-i; // TF-IDF-like weight: first words have higher weight
    });
  });

  // Keep only keywords that appear in ≥1 cluster (all are interesting) but limit to top 20
  const sortedKw=[...kwMap.entries()]
    .sort((a,b)=>b[1].count-a[1].count)
    .slice(0,Math.min(20,kwMap.size));

  // Layout: clusters in outer ring, keywords in inner area
  const clusterNodes:GNode[]=denseC.map((c,i)=>{
    const angle=(2*Math.PI*i)/denseC.length-Math.PI/2;
    const r0=Math.min(W,H)*0.36;
    return{
      id:`c_${c.cluster_id}`,
      label:c.cluster_name,
      type:'cluster',
      cluster_id:c.cluster_id,
      count:c.n_docs,
      x:W/2+r0*Math.cos(angle),
      y:H/2+r0*Math.sin(angle),
      r:Math.max(22,Math.min(42,14+c.n_docs/6)),
      color:getClusterColor(c.cluster_id,false),
    };
  });

  const kwNodes:GNode[]=sortedKw.map(([w,info],i)=>{
    // Position keywords in a grid-like inner area
    const cols=Math.ceil(Math.sqrt(sortedKw.length));
    const row=Math.floor(i/cols);
    const col=i%cols;
    const cellW=(W*0.5)/cols;
    const cellH=(H*0.5)/Math.ceil(sortedKw.length/cols);
    const jitter=(Math.random()-0.5)*cellW*0.3;
    return{
      id:`k_${w}`,
      label:w,
      type:'keyword',
      count:info.count,
      x:W*0.25+col*cellW+cellW/2+jitter,
      y:H*0.25+row*cellH+cellH/2,
      r:Math.max(10,Math.min(20,6+info.count*1.5)),
      color:'rgba(227,172,59,0.85)',
    };
  });

  const allNodes=[...clusterNodes,...kwNodes];

  // Edges: cluster → keyword
  const edges:GEdge[]=[];
  sortedKw.forEach(([w,info])=>{
    info.clusters.forEach(cid=>{
      edges.push({from:`c_${cid}`,to:`k_${w}`,weight:1});
    });
  });
  // Cross-cluster edges (shared keywords)
  for(let i=0;i<denseC.length;i++){
    for(let j=i+1;j<denseC.length;j++){
      const wA=new Set((denseC[i].top_words||[]).map(w=>w.toLowerCase()));
      const shared=(denseC[j].top_words||[]).filter(w=>wA.has(w.toLowerCase()));
      if(shared.length>0){
        edges.push({from:`c_${denseC[i].cluster_id}`,to:`c_${denseC[j].cluster_id}`,weight:shared.length});
      }
    }
  }

  const nodeMap=new Map(allNodes.map(n=>[n.id,n]));

  const isActive=(id:string)=>{
    if(!selectedCluster) return true;
    if(id===`c_${selectedCluster}`) return true;
    return edges.some(e=>(e.from===id||e.to===id)&&(e.from===`c_${selectedCluster}`||e.to===`c_${selectedCluster}`));
  };

  return (
    <div className="w-full">
      <p className="text-[10px] text-white/35 leading-4 mb-2">
        <span className="inline-block w-2 h-2 rounded-full bg-brand-400 mr-1 align-middle"/>Clusters (taille = nb articles)
        <span className="inline-block w-2 h-2 rounded-full bg-gold-400 ml-3 mr-1 align-middle"/>Concepts cliniques clés
        <span className="ml-3">· Connexions = co-occurrence dans le corpus</span>
      </p>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="bg-[#0a1410] rounded-2xl border border-white/5 overflow-visible" style={{maxHeight:420}}>
        <defs>
          <filter id="kglow2"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
          <filter id="kwglow"><feGaussianBlur stdDeviation="2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
          {clusterNodes.map(n=>(
            <radialGradient key={`rg-${n.id}`} id={`rg-${n.id}`} cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor={n.color} stopOpacity="1"/>
              <stop offset="100%" stopColor={n.color} stopOpacity="0.4"/>
            </radialGradient>
          ))}
        </defs>
        {/* Grid lines subtle */}
        {[0.25,0.5,0.75].map(f=>(
          <React.Fragment key={f}>
            <line x1={W*f} y1={0} x2={W*f} y2={H} stroke="rgba(255,255,255,0.02)" strokeWidth="1"/>
            <line x1={0} y1={H*f} x2={W} y2={H*f} stroke="rgba(255,255,255,0.02)" strokeWidth="1"/>
          </React.Fragment>
        ))}
        {/* Edges */}
        {edges.map((e,i)=>{
          const na=nodeMap.get(e.from),nb=nodeMap.get(e.to);
          if(!na||!nb) return null;
          const bothCluster=na.type==='cluster'&&nb.type==='cluster';
          const active=isActive(e.from)&&isActive(e.to);
          const selEdge=selectedCluster&&(e.from===`c_${selectedCluster}`||e.to===`c_${selectedCluster}`);
          return(
            <line key={i}
              x1={na.x} y1={na.y} x2={nb.x} y2={nb.y}
              stroke={bothCluster?(selEdge?"rgba(255,255,255,0.35)":"rgba(255,255,255,0.08)"):(selEdge?"rgba(227,172,59,0.5)":"rgba(227,172,59,0.12)")}
              strokeWidth={bothCluster?(selEdge?e.weight*1.5:e.weight*0.6):(selEdge?1.5:0.8)}
              strokeDasharray={bothCluster?"none":"3,2"}
              opacity={active?1:0.2}
            />
          );
        })}
        {/* Keyword nodes */}
        {kwNodes.map(n=>{
          const active=isActive(n.id);
          const hov=hoveredNode===n.id;
          return(
            <g key={n.id} className="cursor-pointer"
              onMouseEnter={()=>{setHoveredNode(n.id);setTooltip({x:n.x,y:n.y-n.r-8,text:n.label});}}
              onMouseLeave={()=>{setHoveredNode(null);setTooltip(null);}}
            >
              <circle cx={n.x} cy={n.y} r={hov?n.r+3:n.r}
                fill="rgba(227,172,59,0.15)"
                stroke={hov?"rgba(227,172,59,0.9)":"rgba(227,172,59,0.4)"}
                strokeWidth={hov?1.5:0.8}
                opacity={active?1:0.2}
                filter={hov?"url(#kwglow)":undefined}
              />
              {(n.r>12||hov)&&<text x={n.x} y={n.y} textAnchor="middle" dominantBaseline="middle"
                fontSize={hov?"8":"7"} fill="rgba(227,172,59,0.9)" fontWeight="600"
                opacity={active?1:0.3}
                className="pointer-events-none select-none"
              >{n.label}</text>}
            </g>
          );
        })}
        {/* Cluster nodes */}
        {clusterNodes.map(n=>{
          const sel=selectedCluster===n.cluster_id;
          const active=isActive(n.id);
          const hov=hoveredNode===n.id;
          return(
            <g key={n.id} className="cursor-pointer"
              onClick={()=>n.cluster_id!=null&&onSelectCluster(n.cluster_id)}
              onMouseEnter={()=>{setHoveredNode(n.id);setTooltip({x:n.x,y:n.y-n.r-10,text:`${n.label} · ${n.count} articles`});}}
              onMouseLeave={()=>{setHoveredNode(null);setTooltip(null);}}
            >
              {(sel||hov)&&<circle cx={n.x} cy={n.y} r={n.r+10} fill={n.color} opacity={0.12}/>}
              {sel&&<circle cx={n.x} cy={n.y} r={n.r+6} fill="none" stroke={n.color} strokeWidth="1.5" strokeDasharray="4,2" opacity={0.6}/>}
              <circle cx={n.x} cy={n.y} r={n.r}
                fill={`url(#rg-${n.id})`}
                stroke={sel?"#fff":n.color}
                strokeWidth={sel?2:1}
                strokeOpacity={sel?0.9:0.5}
                opacity={active?1:0.3}
                filter={(sel||hov)?"url(#kglow2)":undefined}
              />
              <text x={n.x} y={n.y-3} textAnchor="middle" dominantBaseline="middle"
                fontSize="8" fontWeight="800" fill="#fff" opacity={active?0.95:0.3}
                className="pointer-events-none select-none"
              >{n.label.replace("Cluster ","C")}</text>
              <text x={n.x} y={n.y+8} textAnchor="middle" dominantBaseline="middle"
                fontSize="7" fill="rgba(255,255,255,0.6)" opacity={active?1:0.3}
                className="pointer-events-none select-none"
              >{n.count}</text>
            </g>
          );
        })}
        {/* Tooltip */}
        {tooltip&&(
          <g>
            <rect x={tooltip.x-tooltip.text.length*2.8} y={tooltip.y-14} width={tooltip.text.length*5.6+8} height={16}
              rx="4" fill="rgba(10,20,16,0.92)" stroke="rgba(255,255,255,0.12)" strokeWidth="0.8"/>
            <text x={tooltip.x} y={tooltip.y-5} textAnchor="middle" fontSize="8" fill="rgba(255,255,255,0.9)"
              className="pointer-events-none select-none">{tooltip.text}</text>
          </g>
        )}
      </svg>
    </div>
  );
}

// Scatter plot UMAP moderne avec nuages pastel
function UmapScatterPlot({clusters,selectedCluster,onSelectCluster}:{clusters:ClusterResult[];selectedCluster:number|null;onSelectCluster:(id:number)=>void}) {
  const allPoints: Array<ClusterPoint&{cluster_id:number;is_noise:boolean}>=[];
  clusters.forEach(c=>{if(c.points) c.points.forEach(p=>allPoints.push({...p,cluster_id:c.cluster_id,is_noise:c.is_noise}));});
  if(!allPoints.length) return <span className="text-xs text-white/40">Aucun point.</span>;
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
  const [question, setQuestion] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamedText, setStreamedText] = useState("");
  const [sources, setSources] = useState<ScenarioRagResponse['sources']>([]);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const cancelRef = useRef<(() => void) | null>(null);
  const answerRef = useRef<HTMLDivElement>(null);

  const ask = (qText: string) => {
    if (!qText.trim() || streaming) return;
    // Cancel previous
    if (cancelRef.current) cancelRef.current();
    setStreaming(true);
    setStreamedText("");
    setSources([]);
    setError(null);
    setDone(false);

    const cancel = askScenarioRagStreamFiltered(scenarioId, qText, {
      onSources: (s) => setSources(s),
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
        title="Assistant Scientifique"
        subtitle="Réponses en temps réel avec streaming · Articles filtrés par seuil de pertinence + articles validés"
      />

      {/* Questions suggérées */}
      {suggestedQuestions.length > 0 && !streamedText && !streaming && (
        <div className="space-y-2">
          <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">Questions cliniques suggérées</p>
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
          placeholder="Posez votre question clinique ou opérationnelle..."
          disabled={streaming}
          className="flex-1 rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-xs text-white focus:outline-none focus:border-brand-500/50 transition disabled:opacity-50"
        />
        {streaming ? (
          <button onClick={reset}
            className="rounded-xl bg-rose-500/20 border border-rose-500/30 hover:bg-rose-500/30 text-rose-300 font-semibold px-4 text-xs transition shrink-0"
          >Arrêter</button>
        ) : (
          <button
            onClick={() => ask(question)}
            disabled={!question.trim()}
            className="rounded-xl bg-brand-500 hover:bg-brand-400 text-forest-950 font-semibold px-4 text-xs transition disabled:opacity-50 shrink-0"
          >Poser</button>
        )}
      </div>

      {error && <ErrorBox message={error} />}

      {(streaming || streamedText) && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3 border-t border-white/5 pt-5">
          {/* Réponse streaming */}
          <div className="lg:col-span-2 space-y-3">
            <div className="flex items-center justify-between">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/50">Réponse de l'Assistant</p>
              {streaming && (
                <div className="flex items-center gap-1.5 text-[10px] text-brand-300">
                  <span className="h-1.5 w-1.5 rounded-full bg-brand-400 animate-pulse"/>
                  Génération en cours...
                </div>
              )}
              {done && (
                <button onClick={reset} className="text-[10px] text-white/30 hover:text-white transition">Nouvelle question</button>
              )}
            </div>
            <div
              ref={answerRef}
              className="rounded-2xl border border-white/5 bg-white/2 p-4 text-xs text-white/80 leading-6 whitespace-pre-wrap max-h-[400px] overflow-y-auto"
            >
              {streamedText}
              {streaming && <span className="inline-block w-0.5 h-3 bg-brand-400 animate-pulse ml-0.5 align-middle"/>}
            </div>
          </div>

          {/* Sources citées */}
          {sources.length > 0 && (
            <div className="space-y-3">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/50">Sources scientifiques citées</p>
              <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1">
                {sources.map((src, i) => (
                  <div key={i} className="rounded-xl border border-white/5 bg-white/3 p-2.5 text-xs">
                    <div className="flex items-center justify-between gap-2">
                      <span className="rounded bg-brand-500/10 border border-brand-500/20 px-1.5 py-0.5 text-[9px] text-brand-300 font-mono">
                        SOURCE {i + 1}
                      </span>
                      <span className="text-[10px] text-white/35 font-mono">Pertinence: {(src.score * 100).toFixed(0)}%</span>
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
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/50">Sources scientifiques citées</p>
              <div className="rounded-xl border border-white/5 bg-white/2 p-3 text-center text-[10px] text-white/30">
                <Loader2 size={12} className="animate-spin mx-auto mb-1"/>
                Chargement des sources...
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
// Layout : colonne centrale (étapes) + boîtes d'exclusion à droite reliées
// Inspiré du template officiel PRISMA 2020 (Page, McKenzie et al. 2021)

const PRISMA_COLORS = {
  identification: { fill: "#0e2d1f", stroke: "#1a6640", text: "#6ee7a0", label: "#a7f3c4" },
  screening:      { fill: "#1a1a0a", stroke: "#7a6000", text: "#fbbf24", label: "#fde68a" },
  eligibility:    { fill: "#1a0a0a", stroke: "#7a2020", text: "#f87171", label: "#fca5a5" },
  included:       { fill: "#0a1a2a", stroke: "#1a4a7a", text: "#60a5fa", label: "#93c5fd" },
  exclusion:      { fill: "#1a0e0e", stroke: "#6b2020", text: "#fca5a5", label: "#fecaca" },
};

interface PrismaNode {
  id: string;
  phase: keyof typeof PRISMA_COLORS;
  title: string;
  count: number;
  lines: string[];
  exclusion?: { title: string; count: number; lines: string[] };
}

function PrismaSVGDiagram({ nodes }: { nodes: PrismaNode[] }) {
  // Dimensions
  const W = 780;
  const BOX_W = 300;
  const BOX_H = 80;
  const EXCL_W = 220;
  const EXCL_H = 70;
  const COL_X = 80;          // left edge of main column
  const EXCL_X = COL_X + BOX_W + 60; // left edge of exclusion column
  const ROW_GAP = 50;        // gap between boxes
  const START_Y = 20;

  // Compute y positions
  const positions = nodes.map((_, i) => START_Y + i * (BOX_H + ROW_GAP));
  const totalH = START_Y + nodes.length * (BOX_H + ROW_GAP) + 20;

  return (
    <svg
      viewBox={`0 0 ${W} ${totalH}`}
      width="100%"
      style={{ fontFamily: "ui-monospace, monospace", overflow: "visible" }}
      aria-label="Diagramme PRISMA 2020"
    >
      {nodes.map((node, i) => {
        const y = positions[i];
        const c = PRISMA_COLORS[node.phase];
        const cx = COL_X + BOX_W / 2; // center x of main box
        const nextY = positions[i + 1];
        const hasNext = i < nodes.length - 1;
        const excl = node.exclusion;
        const ec = PRISMA_COLORS.exclusion;

        // Exclusion box center y
        const exclCY = y + BOX_H / 2;

        return (
          <g key={node.id}>
            {/* ── Main box ── */}
            <rect
              x={COL_X} y={y}
              width={BOX_W} height={BOX_H}
              rx={10} ry={10}
              fill={c.fill} stroke={c.stroke} strokeWidth={1.5}
            />
            {/* Phase label */}
            <text x={COL_X + 12} y={y + 18} fontSize={9} fill={c.label}
              fontWeight="700" letterSpacing="1.5" textAnchor="start">
              {node.phase.toUpperCase()}
            </text>
            {/* Title */}
            <text x={COL_X + 12} y={y + 34} fontSize={11} fill={c.text}
              fontWeight="600" textAnchor="start">
              {node.title}
            </text>
            {/* Count */}
            <text x={COL_X + BOX_W - 12} y={y + 34} fontSize={20} fill={c.text}
              fontWeight="800" textAnchor="end" fontFamily="ui-monospace">
              {node.count.toLocaleString()}
            </text>
            {/* Sub-lines */}
            {node.lines.map((line, li) => (
              <text key={li} x={COL_X + 12} y={y + 50 + li * 13}
                fontSize={9} fill={c.label} opacity={0.75} textAnchor="start">
                {line}
              </text>
            ))}

            {/* ── Vertical connector to next box ── */}
            {hasNext && (
              <>
                <line
                  x1={cx} y1={y + BOX_H}
                  x2={cx} y2={nextY}
                  stroke="#2d5a3d" strokeWidth={1.5}
                  strokeDasharray={excl ? "0" : "4 2"}
                />
                {/* Arrow head */}
                <polygon
                  points={`${cx - 5},${nextY - 8} ${cx + 5},${nextY - 8} ${cx},${nextY}`}
                  fill="#2d5a3d"
                />
              </>
            )}

            {/* ── Exclusion box to the right ── */}
            {excl && (
              <>
                {/* Horizontal connector from main box right edge to exclusion box */}
                <line
                  x1={COL_X + BOX_W} y1={exclCY}
                  x2={EXCL_X} y2={exclCY}
                  stroke="#7a2020" strokeWidth={1.5}
                />
                {/* Arrow head pointing right */}
                <polygon
                  points={`${EXCL_X - 8},${exclCY - 5} ${EXCL_X - 8},${exclCY + 5} ${EXCL_X},${exclCY}`}
                  fill="#7a2020"
                />
                {/* Exclusion box */}
                <rect
                  x={EXCL_X} y={exclCY - EXCL_H / 2}
                  width={EXCL_W} height={EXCL_H}
                  rx={8} ry={8}
                  fill={ec.fill} stroke={ec.stroke} strokeWidth={1.5}
                />
                <text x={EXCL_X + 10} y={exclCY - EXCL_H / 2 + 16}
                  fontSize={9} fill={ec.label} fontWeight="700"
                  letterSpacing="1.2" textAnchor="start">
                  EXCLUS
                </text>
                <text x={EXCL_X + 10} y={exclCY - EXCL_H / 2 + 30}
                  fontSize={11} fill={ec.text} fontWeight="600" textAnchor="start">
                  {excl.title}
                </text>
                <text x={EXCL_X + EXCL_W - 10} y={exclCY - EXCL_H / 2 + 30}
                  fontSize={18} fill={ec.text} fontWeight="800"
                  textAnchor="end" fontFamily="ui-monospace">
                  {excl.count.toLocaleString()}
                </text>
                {excl.lines.map((line, li) => (
                  <text key={li}
                    x={EXCL_X + 10}
                    y={exclCY - EXCL_H / 2 + 44 + li * 12}
                    fontSize={8.5} fill={ec.label} opacity={0.7} textAnchor="start">
                    {line}
                  </text>
                ))}
              </>
            )}
          </g>
        );
      })}
    </svg>
  );
}

function PrismaSection({ scenarioId }: { scenarioId: string }) {
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

  if (loading) return <LoadingSpinner text="Calcul du flow PRISMA..." />;
  if (error || !data) return <ErrorBox message={error ?? "Erreur PRISMA"} />;

  const ident = data.identification;
  const screen = data.screening;
  const elig = data.eligibility;
  const inc = data.included;

  // Sources breakdown string
  const sourceLines = Object.entries(ident.by_source)
    .filter(([, v]) => v > 0)
    .sort(([, a], [, b]) => b - a)
    .map(([src, n]) => `${src.toUpperCase()}: ${n}`)
    .join("  ·  ");

  const nodes: PrismaNode[] = [
    {
      id: "identification",
      phase: "identification",
      title: "Enregistrements identifiés",
      count: ident.total_records_identified,
      lines: [
        sourceLines || "Sources non disponibles",
        `dont ${ident.duplicates_removed} doublons retirés`,
      ],
      exclusion: ident.duplicates_removed > 0 ? {
        title: "Doublons supprimés",
        count: ident.duplicates_removed,
        lines: ["Fusion par DOI / PMID / titre normalisé"],
      } : undefined,
    },
    {
      id: "screening",
      phase: "screening",
      title: "Enregistrements screenés",
      count: screen.records_screened,
      lines: [
        `Screening titre / résumé`,
        screen.records_awaiting_screening > 0
          ? `${screen.records_awaiting_screening} en attente d'évaluation manuelle`
          : `${screen.records_excluded_title_abstract} exclus`,
      ],
      exclusion: screen.records_excluded_title_abstract > 0 ? {
        title: "Exclus : titre / résumé",
        count: screen.records_excluded_title_abstract,
        lines: ["Hors sujet, doublons résiduels"],
      } : undefined,
    },
    {
      id: "eligibility",
      phase: "eligibility",
      title: "Textes intégraux évalués",
      count: elig.fulltext_assessed,
      lines: [
        `${elig.fulltext_retrieved} textes intégraux récupérés`,
        elig.fulltext_not_retrieved > 0
          ? `${elig.fulltext_not_retrieved} non récupérables (accès restreint)`
          : "Tous les textes récupérés",
      ],
      exclusion: (elig.fulltext_not_retrieved + elig.fulltext_excluded) > 0 ? {
        title: "Exclus : plein texte",
        count: elig.fulltext_not_retrieved + elig.fulltext_excluded,
        lines: [
          elig.fulltext_not_retrieved > 0 ? `${elig.fulltext_not_retrieved} non accessibles` : "",
          elig.fulltext_excluded > 0 ? `${elig.fulltext_excluded} hors critères` : "",
        ].filter(Boolean),
      } : undefined,
    },
    {
      id: "included",
      phase: "included",
      title: inc.screening_complete ? "Études incluses" : "Articles disponibles",
      count: inc.screening_complete ? inc.total_included : elig.fulltext_retrieved,
      lines: inc.screening_complete
        ? [`Inclus dans la synthèse qualitative`]
        : [
            `Screening manuel non encore effectué`,
            `${inc.awaiting_assessment} articles en attente d'évaluation`,
          ],
    },
  ];

  return (
    <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-5">
      <SectionHeader
        icon={<Shield size={14} className="text-brand-400" />}
        title="Diagramme PRISMA 2020 : Flow de Sélection"
        subtitle="Visualisation standardisée du processus de sélection systématique des articles"
      />

      {/* Legend */}
      <div className="flex flex-wrap gap-3 text-[10px]">
        {([
          { phase: "identification" as const, label: "Identification" },
          { phase: "screening" as const, label: "Screening" },
          { phase: "eligibility" as const, label: "Éligibilité" },
          { phase: "included" as const, label: "Inclus" },
          { phase: "exclusion" as const, label: "Exclusions" },
        ]).map(({ phase, label }) => (
          <span key={phase} className="flex items-center gap-1.5">
            <span
              className="inline-block w-3 h-3 rounded-sm border"
              style={{
                backgroundColor: PRISMA_COLORS[phase].fill,
                borderColor: PRISMA_COLORS[phase].stroke,
              }}
            />
            <span style={{ color: PRISMA_COLORS[phase].label }}>{label}</span>
          </span>
        ))}
      </div>

      {/* SVG Diagram */}
      <div className="overflow-x-auto">
        <PrismaSVGDiagram nodes={nodes} />
      </div>

      {/* Summary table */}
      <div className="border-t border-white/5 pt-4">
        <p className="text-[10px] text-white/35 uppercase tracking-wider mb-3">Résumé des étapes</p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: "Identifiés", value: ident.total_records_identified, sub: `${ident.duplicates_removed} doublons` },
            { label: "Screenés", value: screen.records_screened, sub: `${screen.records_excluded_title_abstract} exclus` },
            { label: "Éligibles", value: elig.fulltext_assessed, sub: `${elig.fulltext_retrieved} textes récupérés` },
            {
              label: inc.screening_complete ? "Inclus" : "Disponibles",
              value: inc.screening_complete ? inc.total_included : elig.fulltext_retrieved,
              sub: inc.screening_complete ? "synthèse qualitative" : "screening en attente",
            },
          ].map(({ label, value, sub }) => (
            <div key={label} className="rounded-xl border border-white/10 bg-white/3 p-3 text-center">
              <p className="text-[9px] uppercase tracking-wider text-white/50 mb-1">{label}</p>
              <p className="text-xl font-bold font-mono text-white">{value.toLocaleString()}</p>
              <p className="text-[9px] text-white/35 mt-0.5">{sub}</p>
            </div>
          ))}
        </div>
      </div>

      {!inc.screening_complete && (
        <div className="rounded-xl border border-gold-500/20 bg-gold-500/5 px-4 py-3">
          <p className="text-[10px] text-gold-400">
            <span className="font-semibold">Screening manuel non effectué</span> · Les articles sont disponibles pour évaluation dans l'onglet Corpus. Utilisez l'interface de screening pour inclure ou exclure chaque article.
          </p>
        </div>
      )}
    </div>
  );
}


// ─── Section: PICO Tableau Comparatif ─────────────────────────────────────────
function PicoSection({ scenarioId }: { scenarioId: string }) {
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
    const headers = ['ID','Titre','Année','Journal','Type étude','Confiance','Population (P)','Intervention (I)','Comparateur (C)','Outcome (O)','Notes'];
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

  if (loading) return <LoadingSpinner text="Chargement des données PICO..." />;
  if (error || !data) return <ErrorBox message={error ?? "Erreur PICO"} />;

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
        title="Tableau Comparatif PICO"
        subtitle="Vue synthétique de tous les articles avec extraction PICO structurée"
      />
      {/* Stats bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          {label:'Total articles',value:data.total,color:'text-white'},
          {label:'Avec PICO extrait',value:data.with_pico,color:'text-brand-300'},
          {label:'Couverture PICO',value:coverage+'%',color:coverage>70?'text-brand-300':coverage>40?'text-gold-400':'text-rose-300'},
          {label:'Sans PICO',value:data.total-data.with_pico,color:'text-white/50'},
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
          <span>Couverture PICO</span><span>{data.with_pico}/{data.total}</span>
        </div>
        <div className="h-2 bg-white/5 rounded-full overflow-hidden">
          <div className="h-full bg-brand-500 rounded-full transition-all" style={{width:`${coverage}%`}}/>
        </div>
      </div>
      {/* Controls */}
      <div className="flex flex-wrap gap-2 items-center">
        <input
          type="text" placeholder="Rechercher un article..." value={search}
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
              {f==='all'?'Tous':f==='with_pico'?'Avec PICO':'Sans PICO'}
            </button>
          ))}
        </div>
        <select value={sortBy} onChange={e=>setSortBy(e.target.value as any)}
          className="rounded-lg border border-white/10 bg-white/5 px-2 py-1.5 text-[10px] text-white/70 focus:outline-none"
        >
          <option value="year">Trier par année</option>
          <option value="confidence">Trier par confiance</option>
          <option value="design">Trier par type étude</option>
        </select>
        {data.with_pico > 0 && (
          <button onClick={exportCsv}
            className="flex items-center gap-1.5 rounded-xl border border-brand-500/30 bg-brand-500/10 px-3 py-1.5 text-[10px] text-brand-300 hover:bg-brand-500/20 transition"
          >
            <Download size={10}/>Export CSV
          </button>
        )}
      </div>
      {/* Table */}
      <div className="overflow-x-auto rounded-2xl border border-white/5">
        <table className="w-full text-[10px] border-collapse">
          <thead>
            <tr className="border-b border-white/5 bg-white/3">
              {['Titre','Année','Type étude','Confiance','P : Population','I : Intervention','C : Comparateur','O : Outcome'].map(h=>(
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
            Affichage des 100 premiers résultats sur {filtered.length}. Utilisez la recherche pour filtrer.
          </div>
        )}
        {filtered.length === 0 && (
          <div className="text-center py-8 text-xs text-white/35">Aucun article ne correspond aux filtres.</div>
        )}
      </div>
    </div>
  );
}

// ─── Section: Evidence Brief PDF ──────────────────────────────────────────────
function EvidenceBriefSection({ scenarioId, detail }: { scenarioId: string; detail: ScenarioDetail }) {
  const [data, setData] = React.useState<EvidenceBriefData | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [generating, setGenerating] = React.useState(false);

  React.useEffect(() => {
    setLoading(true);
    fetchEvidenceBrief(scenarioId)
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [scenarioId]);

  const generatePdf = async () => {
    if (!data) return;
    setGenerating(true);
    try {
      // Build HTML for PDF
      const html = `<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Evidence Brief : ${detail.title}</title>
<style>
  body{font-family:Georgia,serif;max-width:800px;margin:40px auto;color:#1a1a1a;line-height:1.6}
  h1{color:#0A3621;border-bottom:3px solid #E3AC3B;padding-bottom:8px}
  h2{color:#0A3621;margin-top:32px;font-size:1.1em;text-transform:uppercase;letter-spacing:.05em}
  h3{color:#2d7a52;font-size:.95em}
  .meta{color:#666;font-size:.85em;margin-bottom:24px}
  .stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}
  .stat{background:#f0f7f3;border:1px solid #aed4bc;border-radius:8px;padding:12px;text-align:center}
  .stat-val{font-size:1.8em;font-weight:700;color:#0A3621}
  .stat-label{font-size:.75em;color:#4d7461;margin-top:4px}
  table{width:100%;border-collapse:collapse;font-size:.82em;margin-top:12px}
  th{background:#0A3621;color:#fff;padding:8px 10px;text-align:left}
  td{padding:6px 10px;border-bottom:1px solid #e0e8e3}
  tr:nth-child(even) td{background:#f9fafb}
  .badge{display:inline-block;background:#d6eade;color:#0A3621;border-radius:4px;padding:2px 6px;font-size:.75em;font-weight:600}
  .bar-container{background:#e0e8e3;border-radius:4px;height:8px;margin-top:4px}
  .bar{background:#2d7a52;border-radius:4px;height:8px}
  .article-card{border:1px solid #e0e8e3;border-radius:8px;padding:12px;margin-bottom:10px}
  .article-title{font-weight:600;color:#0A3621;margin-bottom:4px}
  .article-meta{font-size:.8em;color:#666}
  .article-abstract{font-size:.82em;color:#444;margin-top:6px;font-style:italic}
  footer{margin-top:40px;padding-top:16px;border-top:1px solid #e0e8e3;font-size:.75em;color:#999;text-align:center}
</style></head><body>
<h1>Evidence Brief</h1>
<div class="meta">
  <strong>${detail.title}</strong> · Scénario LiteRev<br>
  Généré le ${new Date(data.generated_at).toLocaleDateString('fr-FR', {year:'numeric',month:'long',day:'numeric'})}<br>
  Projet : LiteRev Evidence
</div>

<h2>Résumé Exécutif</h2>
<p>${detail.description}</p>

<h2>Statistiques du Corpus</h2>
<div class="stat-grid">
  <div class="stat"><div class="stat-val">${data.corpus_stats.total}</div><div class="stat-label">Articles identifiés</div></div>
  <div class="stat"><div class="stat-val">${data.corpus_stats.total - data.corpus_stats.duplicates}</div><div class="stat-label">Articles uniques</div></div>
  <div class="stat"><div class="stat-val">${data.corpus_stats.included}</div><div class="stat-label">Inclus (screening)</div></div>
  <div class="stat"><div class="stat-val">${data.corpus_stats.with_pico}</div><div class="stat-label">Avec PICO extrait</div></div>
</div>
${data.corpus_stats.year_min && data.corpus_stats.year_max ? `<p class="meta">Couverture temporelle : ${data.corpus_stats.year_min} – ${data.corpus_stats.year_max}</p>` : ''}

<h2>Distribution par Type d'Étude</h2>
<table>
  <tr><th>Type d'étude</th><th>Nombre</th><th>Proportion</th></tr>
  ${data.study_design_distribution.slice(0,8).map(d=>`
  <tr>
    <td><span class="badge">${d.design}</span></td>
    <td>${d.count}</td>
    <td>
      <div class="bar-container"><div class="bar" style="width:${Math.round(d.count/(data.corpus_stats.total||1)*100)}%"></div></div>
      ${Math.round(d.count/(data.corpus_stats.total||1)*100)}%
    </td>
  </tr>`).join('')}
</table>

<h2>Articles Représentatifs</h2>
${data.top_articles.slice(0,8).map((a,i)=>`
<div class="article-card">
  <div class="article-title">${i+1}. ${a.title}</div>
  <div class="article-meta">
    ${a.year||'N/A'} • ${a.journal||'Journal non renseigné'}
    ${a.study_design?` • <span class="badge">${a.study_design}</span>`:''}
    ${a.citation_count?` • ${a.citation_count} citations`:''}
    ${a.doi?` • <a href="https://doi.org/${a.doi}">DOI</a>`:''}
  </div>
  ${a.abstract_excerpt?`<div class="article-abstract">"${a.abstract_excerpt}..."</div>`:''}
</div>`).join('')}

<footer>
  LiteRev Evidence | LiteRev Evidence | ${new Date().getFullYear()}<br>
  Ce document est généré automatiquement à partir de la base de données de littérature scientifique.
</footer>
</body></html>`;

      const blob = new Blob([html], {type:'text/html;charset=utf-8'});
      const url = URL.createObjectURL(blob);
      const win = window.open(url, '_blank');
      if (win) {
        setTimeout(() => { win.print(); }, 800);
      }
      URL.revokeObjectURL(url);
    } finally {
      setGenerating(false);
    }
  };

  if (loading) return <LoadingSpinner text="Chargement de l'Evidence Brief..." />;
  if (error || !data) return <ErrorBox message={error ?? "Erreur Evidence Brief"} />;

  const total = data.corpus_stats.total || 1;
  const picoRows = data.pico_table ?? [];
  const dbs = data.double_blind_stats ?? { reviewer_1_done:0, reviewer_2_done:0, both_done:0, agreements:0, conflicts:0 };
  const kappaRaw = dbs.both_done > 0 ? ((dbs.agreements / dbs.both_done) * 2 - 1) : null;
  const kappa = kappaRaw !== null ? Math.max(0, Math.min(1, kappaRaw)).toFixed(2) : 'N/A';

  return (
    <div className="space-y-6">
      <SectionHeader
        icon={<BookOpen size={14} className="text-brand-400" />}
        title="Evidence Brief"
        subtitle="Rapport synthétique complet : corpus, PICO, screening, niveau de preuve"
      />

      {/* Bannière sélection automatique */}
      {(data.corpus_stats.included === 0) && (
        <div className="rounded-2xl border border-gold-500/20 bg-gold-500/5 px-4 py-3 flex items-start gap-3">
          <AlertTriangle size={14} className="text-gold-400 shrink-0 mt-0.5" />
          <div className="text-xs text-gold-200/80 leading-relaxed">
            <strong className="text-gold-300">Aucun article validé par un relecteur humain</strong> — Les articles présentés dans cet Evidence Brief ont été sélectionnés automatiquement par recherche PubMed (lexicale) et/ou similarité sémantique. Pour une revue systématique formelle, un screening humain en double-aveugle est nécessaire avant d’utiliser ces résultats.
          </div>
        </div>
      )}

      {/* Header avec export */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h3 className="text-sm font-bold text-white">{detail.title}</h3>
          <p className="text-[10px] text-white/35 mt-0.5">Généré le {new Date(data.generated_at).toLocaleDateString('fr-FR',{year:'numeric',month:'long',day:'numeric'})}</p>
        </div>
        <button onClick={generatePdf} disabled={generating}
          className="flex items-center gap-2 rounded-2xl bg-brand-500 hover:bg-brand-400 text-white font-semibold px-4 py-2 text-xs transition disabled:opacity-50 shrink-0">
          {generating ? <Loader2 size={12} className="animate-spin"/> : <Download size={12}/>}
          {generating ? 'Génération...' : 'Exporter PDF'}
        </button>
      </div>

      {/* KPIs principaux */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2.5">
        {[
          {label:'Articles total',value:data.corpus_stats.total,color:'text-white',sub:null},
          {label:'Inclus',value:data.corpus_stats.included,color:'text-brand-300',sub:`${Math.round(data.corpus_stats.included/total*100)}%`},
          {label:'Exclus',value:data.corpus_stats.excluded,color:'text-red-400',sub:`${Math.round(data.corpus_stats.excluded/total*100)}%`},
          {label:'En attente',value:data.corpus_stats.pending ?? (total - data.corpus_stats.included - data.corpus_stats.excluded),color:'text-white/50',sub:null},
          {label:'PICO extraits',value:data.corpus_stats.with_pico,color:'text-gold-400',sub:`${Math.round(data.corpus_stats.pico_coverage_pct ?? data.corpus_stats.with_pico/total*100)}%`},
          {label:'Texte intégral',value:data.corpus_stats.with_fulltext ?? 0,color:'text-blue-300',sub:null},
        ].map(s=>(
          <div key={s.label} className="rounded-2xl border border-white/5 bg-white/2 p-3 text-center">
            <div className={`text-xl font-bold ${s.color}`}>{s.value}</div>
            {s.sub && <div className="text-[9px] text-white/30 font-mono">{s.sub}</div>}
            <div className="text-[9px] text-white/35 mt-0.5 leading-tight">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Couverture temporelle + citations */}
      {data.corpus_stats.year_min && data.corpus_stats.year_max && (
        <div className="flex flex-wrap gap-4 text-xs text-white/50">
          <span>Couverture : <span className="text-white/70 font-semibold">{data.corpus_stats.year_min} – {data.corpus_stats.year_max}</span></span>
          {data.corpus_stats.avg_citations != null && <span>Citations moy. : <span className="text-white/70 font-semibold">{data.corpus_stats.avg_citations.toFixed(1)}</span></span>}
          {data.corpus_stats.max_citations != null && <span>Max citations : <span className="text-white/70 font-semibold">{data.corpus_stats.max_citations}</span></span>}
        </div>
      )}

      {/* Double-aveugle stats */}
      {dbs.both_done > 0 && (
        <div className="rounded-2xl border border-white/5 bg-white/2 p-4 space-y-2">
          <p className="text-[10px] font-bold uppercase tracking-wider text-white/40">Revue Double-Aveugle</p>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
            {[
              {label:'R1 terminé',value:dbs.reviewer_1_done},
              {label:'R2 terminé',value:dbs.reviewer_2_done},
              {label:'Double-rev.',value:dbs.both_done},
              {label:'Accords',value:dbs.agreements,color:'text-brand-300'},
              {label:'Conflits',value:dbs.conflicts,color:'text-red-400'},
            ].map(s=>(
              <div key={s.label} className="text-center">
                <div className={`text-lg font-bold ${(s as {color?:string}).color ?? 'text-white'}`}>{s.value}</div>
                <div className="text-[9px] text-white/35">{s.label}</div>
              </div>
            ))}
          </div>
          <div className="text-[10px] text-white/40">
            Accord inter-reviewers (Kappa approx.) : <span className="text-white/60 font-mono font-semibold">{kappa}</span>
            {kappaRaw !== null && (
              <span className="ml-2">
                {kappaRaw >= 0.8 ? '(excellent)' : kappaRaw >= 0.6 ? '(bon)' : kappaRaw >= 0.4 ? '(modéré)' : '(faible)'}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Distributions : type d'étude + source + niveau de preuve */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Types d'étude */}
        <div className="rounded-2xl border border-white/5 bg-white/2 p-4 space-y-2">
          <p className="text-[10px] font-bold uppercase tracking-wider text-white/40">Types d'étude</p>
          <div className="space-y-1.5">
            {data.study_design_distribution.slice(0,6).map(d=>(
              <div key={d.design} className="flex items-center gap-2 text-[10px]">
                <span className="w-28 text-white/60 truncate">{d.design}</span>
                <div className="flex-1 h-1 bg-white/5 rounded-full overflow-hidden">
                  <div className="h-full bg-brand-500 rounded-full" style={{width:`${Math.round(d.count/total*100)}%`}}/>
                </div>
                <span className="w-7 text-right text-white/40 font-mono">{d.count}</span>
              </div>
            ))}
          </div>
        </div>
        {/* Sources */}
        <div className="rounded-2xl border border-white/5 bg-white/2 p-4 space-y-2">
          <p className="text-[10px] font-bold uppercase tracking-wider text-white/40">Sources</p>
          <div className="space-y-1.5">
            {(data.source_distribution ?? []).slice(0,6).map(d=>(
              <div key={d.source} className="flex items-center gap-2 text-[10px]">
                <span className="w-28 text-white/60 truncate capitalize">{d.source}</span>
                <div className="flex-1 h-1 bg-white/5 rounded-full overflow-hidden">
                  <div className="h-full bg-blue-500/60 rounded-full" style={{width:`${Math.round(d.count/total*100)}%`}}/>
                </div>
                <span className="w-7 text-right text-white/40 font-mono">{d.count}</span>
              </div>
            ))}
          </div>
        </div>
        {/* Niveaux de preuve */}
        <div className="rounded-2xl border border-white/5 bg-white/2 p-4 space-y-2">
          <p className="text-[10px] font-bold uppercase tracking-wider text-white/40">Niveaux de preuve</p>
          <div className="space-y-1.5">
            {(data.evidence_level_distribution ?? []).slice(0,6).map(d=>(
              <div key={d.level} className="flex items-center gap-2 text-[10px]">
                <span className="w-28 text-white/60 truncate capitalize">{d.level}</span>
                <div className="flex-1 h-1 bg-white/5 rounded-full overflow-hidden">
                  <div className="h-full bg-gold-400/60 rounded-full" style={{width:`${Math.round(d.count/total*100)}%`}}/>
                </div>
                <span className="w-7 text-right text-white/40 font-mono">{d.count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Top articles avec PICO */}
      <div className="space-y-3">
        <p className="text-[10px] font-bold uppercase tracking-wider text-white/40">Articles représentatifs avec extraction PICO</p>
        {data.top_articles.slice(0,5).map((a,i)=>(
          <div key={a.id} className="rounded-2xl border border-white/5 bg-white/2 p-4 space-y-2">
            <div className="flex items-start gap-2">
              <span className="text-[10px] font-mono text-brand-300 shrink-0 mt-0.5">#{i+1}</span>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-semibold text-white/85 leading-4 line-clamp-2">{a.title}</p>
                <div className="flex flex-wrap gap-2 mt-1 text-[9px] text-white/40">
                  <span>{a.year||'N/A'}</span>
                  {a.journal && <span className="truncate max-w-[160px]">{a.journal}</span>}
                  {a.study_design && <span className="rounded bg-brand-500/10 border border-brand-500/20 px-1 text-brand-300">{a.study_design}</span>}
                  {a.citation_count != null && <span>{a.citation_count} cit.</span>}
                  {a.similarity_score != null && <span className="text-gold-400/70">sim. {a.similarity_score.toFixed(2)}</span>}
                  {a.screening_status && (
                    <span className={`rounded px-1 ${
                      a.screening_status === 'included' ? 'bg-brand-500/15 text-brand-300' :
                      a.screening_status === 'excluded' ? 'bg-red-500/15 text-red-400' : 'bg-white/5 text-white/40'
                    }`}>{a.screening_status === 'included' ? 'Inclus' : a.screening_status === 'excluded' ? 'Exclu' : 'En attente'}</span>
                  )}
                </div>
              </div>
            </div>
            {a.pico_summary && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-1">
                {[
                  {label:'P',title:'Population',val:a.pico_summary.population},
                  {label:'I',title:'Intervention',val:a.pico_summary.intervention},
                  {label:'O',title:'Outcome',val:a.pico_summary.outcome},
                  {label:'Conclusion',title:'Conclusion clé',val:a.pico_summary.key_finding},
                ].map(p=>(
                  <div key={p.label} className="rounded-xl border border-white/5 bg-white/2 p-2">
                    <div className="text-[8px] font-bold uppercase text-white/30 mb-0.5">{p.title}</div>
                    <div className="text-[9px] text-white/65 leading-3.5 line-clamp-3">{p.val || '—'}</div>
                  </div>
                ))}
              </div>
            )}
            {a.abstract_excerpt && !a.pico_summary && (
              <p className="text-[9px] text-white/40 italic line-clamp-2">"{a.abstract_excerpt}..."</p>
            )}
          </div>
        ))}
      </div>

      {/* Tableau PICO comparatif */}
      {picoRows.length > 0 && (
        <div className="space-y-2">
          <p className="text-[10px] font-bold uppercase tracking-wider text-white/40">Tableau comparatif PICO ({picoRows.length} articles)</p>
          <div className="overflow-x-auto rounded-2xl border border-white/5">
            <table className="w-full text-[9px]">
              <thead>
                <tr className="border-b border-white/5">
                  {['Titre','Année','Design','P','I','C','O','Conclusion','Niveau'].map(h=>(
                    <th key={h} className="px-2 py-2 text-left text-[8px] font-bold uppercase text-white/30 whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {picoRows.slice(0,20).map((row,i)=>(
                  <tr key={row.id} className={i%2===0?'bg-white/1':''} >
                    <td className="px-2 py-1.5 text-white/70 max-w-[160px]"><span className="line-clamp-2 leading-3.5">{row.title}</span></td>
                    <td className="px-2 py-1.5 text-white/40 whitespace-nowrap">{row.year||'N/A'}</td>
                    <td className="px-2 py-1.5"><span className="rounded bg-brand-500/10 border border-brand-500/20 px-1 text-brand-300 whitespace-nowrap">{row.study_design||'N/A'}</span></td>
                    <td className="px-2 py-1.5 text-white/55 max-w-[100px]"><span className="line-clamp-2">{row.pico.population||'—'}</span></td>
                    <td className="px-2 py-1.5 text-white/55 max-w-[100px]"><span className="line-clamp-2">{row.pico.intervention||'—'}</span></td>
                    <td className="px-2 py-1.5 text-white/55 max-w-[80px]"><span className="line-clamp-2">{row.pico.comparator||'—'}</span></td>
                    <td className="px-2 py-1.5 text-white/55 max-w-[100px]"><span className="line-clamp-2">{row.pico.outcome||'—'}</span></td>
                    <td className="px-2 py-1.5 text-white/55 max-w-[120px]"><span className="line-clamp-2">{row.pico.key_finding||'—'}</span></td>
                    <td className="px-2 py-1.5 whitespace-nowrap">
                      <span className={`rounded px-1 ${
                        row.pico.evidence_level === 'forte' ? 'bg-brand-500/15 text-brand-300' :
                        row.pico.evidence_level === 'modérée' ? 'bg-gold-400/15 text-gold-400' : 'bg-white/5 text-white/40'
                      }`}>{row.pico.evidence_level||'N/A'}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {picoRows.length > 20 && <p className="text-[9px] text-white/30 text-right">{picoRows.length - 20} articles supplémentaires dans le PDF exporté</p>}
        </div>
      )}
    </div>
  );
}

// ─── Section: LLM Evidence Brief narratif ────────────────────────────────────

function LlmEvidenceBriefSection({ scenarioId }: { scenarioId: string }) {
  const [data, setData] = React.useState<LlmEvidenceBrief | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [regenerating, setRegenerating] = React.useState(false);
  const [genStatus, setGenStatus] = React.useState<string | null>(null);
  const pollRef = React.useRef<ReturnType<typeof setInterval> | null>(null);

  const load = React.useCallback(() => {
    setLoading(true);
    getLlmEvidenceBrief(scenarioId)
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [scenarioId]);

  React.useEffect(() => {
    load();
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [load]);

  // Polling si generation en cours
  React.useEffect(() => {
    if (!data || data.status !== 'generating') return;
    setGenStatus('Génération en cours...');
    pollRef.current = setInterval(() => {
      getBriefGenerationStatus(scenarioId).then(s => {
        if (s.status === 'done') {
          if (pollRef.current) clearInterval(pollRef.current);
          setGenStatus(null);
          load();
        } else if (s.status === 'error') {
          if (pollRef.current) clearInterval(pollRef.current);
          setGenStatus(null);
          setError(s.error ?? 'Erreur de génération');
        }
      });
    }, 5000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [data, scenarioId, load]);

  const handleRegenerate = async () => {
    setRegenerating(true);
    setGenStatus('Régénération lancée...');
    try {
      await generateEvidenceBrief(scenarioId, true);
      pollRef.current = setInterval(() => {
        getBriefGenerationStatus(scenarioId).then(s => {
          if (s.status === 'done') {
            if (pollRef.current) clearInterval(pollRef.current);
            setGenStatus(null);
            setRegenerating(false);
            load();
          } else if (s.status === 'error') {
            if (pollRef.current) clearInterval(pollRef.current);
            setGenStatus(null);
            setRegenerating(false);
            setError(s.error ?? 'Erreur de régénération');
          }
        });
      }, 5000);
    } catch (e: any) {
      setRegenerating(false);
      setGenStatus(null);
      setError(e.message);
    }
  };

  if (loading) return <LoadingSpinner text="Chargement du brief narratif LLM..." />;
  if (error) return <ErrorBox message={error} />;
  if (!data) return null;

  // Si aucun article disponible
  if (data.status === 'empty') {
    return (
      <div className="rounded-2xl border border-slate-500/20 bg-slate-500/5 px-5 py-4 flex items-start gap-3">
        <div className="text-xs text-slate-300/80">
          <strong className="text-slate-200">Aucun article disponible</strong> — {data.message ?? 'Ajoutez des articles ou abaissez le seuil de similarité.'}
        </div>
      </div>
    );
  }

  // Si génération en cours
  if (data.status === 'generating') {
    return (
      <div className="rounded-2xl border border-gold-500/20 bg-gold-500/5 px-5 py-4 flex items-start gap-3">
        <Loader2 size={14} className="text-gold-400 animate-spin shrink-0 mt-0.5" />
        <div className="text-xs text-gold-200/80">
          <strong className="text-gold-300">Brief LLM en cours de génération</strong> — {data.message ?? 'Réessayez dans 30 secondes.'}
        </div>
      </div>
    );
  }

  const meta = data._meta;
  const hasContent = !!(data.executive_summary || data.key_findings?.length);

  return (
    <div className="space-y-5">
      {/* Header + Régénérer */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <p className="text-[10px] text-white/35 mt-0.5">
            {meta ? (
              <>Généré le {new Date(meta.generated_at).toLocaleDateString('fr-FR', {year:'numeric',month:'long',day:'numeric'})} · {meta.articles_used} articles · seuil {meta.threshold?.toFixed(2)} · {meta.human_validated} validés humainement</>
            ) : (
              data._generated_at ? <>Généré le {new Date(data._generated_at).toLocaleDateString('fr-FR', {year:'numeric',month:'long',day:'numeric'})}</> : null
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {genStatus && (
            <span className="text-[10px] text-gold-400 flex items-center gap-1">
              <Loader2 size={10} className="animate-spin" />{genStatus}
            </span>
          )}
          <button onClick={handleRegenerate} disabled={regenerating}
            className="flex items-center gap-1.5 rounded-xl border border-brand-500/30 bg-brand-500/10 hover:bg-brand-500/20 text-brand-300 font-medium px-3 py-1.5 text-xs transition disabled:opacity-50">
            {regenerating ? (<Loader2 size={11} className="animate-spin" />) : (<RefreshCw size={11} />)}
            Régénérer
          </button>
        </div>
      </div>

      {!hasContent && (
        <div className="rounded-2xl border border-white/10 bg-white/3 px-4 py-6 text-center text-xs text-white/40">
          Aucun brief LLM disponible. Cliquez sur "Régénérer" pour lancer la génération.
        </div>
      )}

      {hasContent && (
        <>
          {/* Niveau de preuve + Grade */}
          {(data.evidence_level || data.grade_recommendation) && (
            <div className="flex flex-wrap gap-2">
              {data.evidence_level && (
                <span className={`rounded-xl px-3 py-1 text-xs font-semibold border ${
                  data.evidence_level.toLowerCase().includes('fort') ? 'bg-brand-500/15 border-brand-500/30 text-brand-300' :
                  data.evidence_level.toLowerCase().includes('mod') ? 'bg-gold-500/15 border-gold-500/30 text-gold-300' :
                  'bg-white/5 border-white/10 text-white/50'
                }`}>
                  Niveau : {data.evidence_level}
                </span>
              )}
              {data.grade_recommendation && (
                <span className="rounded-xl px-3 py-1 text-xs font-semibold border bg-brand-500/10 border-brand-500/20 text-brand-200">
                  Grade {data.grade_recommendation}
                </span>
              )}
            </div>
          )}

          {/* Résumé exécutif */}
          {data.executive_summary && (
            <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 p-4">
              <p className="text-[10px] font-bold uppercase tracking-wider text-brand-400 mb-2">Résumé exécutif</p>
              <p className="text-sm text-white/80 leading-relaxed">{data.executive_summary}</p>
            </div>
          )}

          {/* Contexte clinique */}
          {data.clinical_context && (
            <div className="space-y-1">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">Contexte clinique</p>
              <p className="text-xs text-white/65 leading-relaxed">{data.clinical_context}</p>
            </div>
          )}

          {/* Résultats clés */}
          {data.key_findings && data.key_findings.length > 0 && (
            <div className="space-y-2">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">Résultats clés</p>
              <ul className="space-y-1.5">
                {data.key_findings.map((f, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-white/70">
                    <span className="shrink-0 mt-0.5 h-4 w-4 rounded-full bg-brand-500/20 border border-brand-500/30 flex items-center justify-center text-[9px] font-bold text-brand-300">{i+1}</span>
                    {f}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Synthèse des évidences */}
          {data.evidence_synthesis && (
            <div className="space-y-1">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">Synthèse des évidences</p>
              <p className="text-xs text-white/65 leading-relaxed whitespace-pre-line">{data.evidence_synthesis}</p>
            </div>
          )}

          {/* PICO Summary */}
          {(data.population_summary || data.intervention_summary || data.outcome_summary) && (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {data.population_summary && (
                <div className="rounded-xl border border-white/8 bg-white/3 p-3">
                  <p className="text-[9px] font-bold uppercase tracking-wider text-white/30 mb-1">Population</p>
                  <p className="text-xs text-white/60 leading-relaxed">{data.population_summary}</p>
                </div>
              )}
              {data.intervention_summary && (
                <div className="rounded-xl border border-white/8 bg-white/3 p-3">
                  <p className="text-[9px] font-bold uppercase tracking-wider text-white/30 mb-1">Intervention</p>
                  <p className="text-xs text-white/60 leading-relaxed">{data.intervention_summary}</p>
                </div>
              )}
              {data.outcome_summary && (
                <div className="rounded-xl border border-white/8 bg-white/3 p-3">
                  <p className="text-[9px] font-bold uppercase tracking-wider text-white/30 mb-1">Outcome</p>
                  <p className="text-xs text-white/60 leading-relaxed">{data.outcome_summary}</p>
                </div>
              )}
            </div>
          )}

          {/* Actions recommandées */}
          {data.recommended_actions && data.recommended_actions.length > 0 && (
            <div className="rounded-2xl border border-gold-500/20 bg-gold-500/5 p-4 space-y-2">
              <p className="text-[10px] font-bold uppercase tracking-wider text-gold-400">Actions recommandées</p>
              <ul className="space-y-1.5">
                {data.recommended_actions.map((a, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-gold-200/80">
                    <Zap size={10} className="shrink-0 mt-0.5 text-gold-400" />
                    {a}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Implications cliniques */}
          {data.clinical_implications && (
            <div className="space-y-1">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">Implications cliniques</p>
              <p className="text-xs text-white/65 leading-relaxed">{data.clinical_implications}</p>
            </div>
          )}

          {/* Recommandations d'implémentation */}
          {data.implementation_recommendations && data.implementation_recommendations.length > 0 && (
            <div className="space-y-2">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">Recommandations d'implémentation</p>
              <ul className="space-y-1">
                {data.implementation_recommendations.map((r, i) => (
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
            {data.limitations && data.limitations.length > 0 && (
              <div className="space-y-1.5">
                <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">Limites</p>
                <ul className="space-y-1">
                  {data.limitations.map((l, i) => (
                    <li key={i} className="flex items-start gap-1.5 text-xs text-white/50">
                      <AlertCircle size={9} className="shrink-0 mt-0.5 text-rose-400" />
                      {l}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {data.research_gaps && data.research_gaps.length > 0 && (
              <div className="space-y-1.5">
                <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">Lacunes de recherche</p>
                <ul className="space-y-1">
                  {data.research_gaps.map((g, i) => (
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
          {data.future_research && (
            <div className="space-y-1">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">Directions de recherche futures</p>
              <p className="text-xs text-white/55 leading-relaxed">{data.future_research}</p>
            </div>
          )}

          {/* Références clés */}
          {data.key_references && data.key_references.length > 0 && (
            <div className="space-y-2">
              <p className="text-[10px] font-bold uppercase tracking-wider text-white/35">Références clés</p>
              <div className="space-y-1.5">
                {data.key_references.slice(0, 6).map((ref, i) => (
                  <div key={i} className="rounded-xl border border-white/8 bg-white/2 px-3 py-2">
                    <p className="text-xs font-medium text-white/70">{ref.title}</p>
                    <p className="text-[10px] text-white/35 mt-0.5">{ref.year} · {ref.journal}</p>
                    {ref.key_contribution && (
                      <p className="text-[10px] text-white/45 mt-0.5 italic">{ref.key_contribution}</p>
                    )}
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

// ─── Section: Knowledge Graph (co-citations) ─────────────────────────────────

function KnowledgeGraphSection({ scenarioId }: { scenarioId: string }) {
  const [data, setData] = React.useState<KnowledgeGraphData | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = React.useState<number | null>(null);
  const [selectedNode, setSelectedNode] = React.useState<number | null>(null);
  const [tooltip, setTooltip] = React.useState<{x:number;y:number;node:KGNode}|null>(null);
  const [minSim, setMinSim] = React.useState(0.35);

  React.useEffect(() => {
    setLoading(true);
    fetchKnowledgeGraph(scenarioId, 80, minSim)
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [scenarioId, minSim]);

  if (loading) return <LoadingSpinner text="Calcul du graphe de connaissances..." />;
  if (error || !data) return <ErrorBox message={error ?? "Erreur Knowledge Graph"} />;
  if (!data.nodes.length) return <div className="text-xs text-white/40 p-4">Aucun article avec embeddings disponible.</div>;

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
        title="Knowledge Graph : Réseau de Co-citations"
        subtitle={`${data.n_nodes} articles · ${data.n_edges} connexions · ${data.n_clusters} groupes thématiques · similarité cosinus sur embeddings`}
      />

      {/* Contrôles */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2 text-xs text-white/50">
          <span>Seuil similarité :</span>
          <input type="range" min={0.2} max={0.7} step={0.05} value={minSim}
            onChange={e => setMinSim(parseFloat(e.target.value))}
            className="w-28 accent-brand-500"
          />
          <span className="font-mono text-brand-300">{minSim.toFixed(2)}</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {uniqueClusters.slice(0,5).map((cid, i) => (
            <button key={cid}
              onClick={() => setSelectedNode(null)}
              className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/3 px-2 py-1 text-[10px] text-white/50 hover:text-white transition"
            >
              <span className="h-2 w-2 rounded-full" style={{background: CLUSTER_COLORS[i % CLUSTER_COLORS.length]}}/>
              Groupe {cid + 1} ({data.clusters.find(c => c.id === cid)?.size || 0} articles)
            </button>
          ))}
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
              const meta = `${n.year || 'N/A'} · ${n.design} · ${n.degree} connexions`;
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
                <div className="flex justify-between"><span>Année</span><span className="text-white/70">{selectedNodeData.year || 'N/A'}</span></div>
                <div className="flex justify-between"><span>Journal</span><span className="text-white/70 text-right max-w-[120px] truncate">{selectedNodeData.journal || '—'}</span></div>
                <div className="flex justify-between"><span>Type étude</span><span className="rounded bg-brand-500/10 border border-brand-500/20 px-1 text-brand-300">{selectedNodeData.design}</span></div>
                <div className="flex justify-between"><span>Connexions</span><span className="text-brand-300 font-semibold">{selectedNodeData.degree}</span></div>
                <div className="flex justify-between"><span>Qualité</span><span className="text-gold-400 font-semibold">{selectedNodeData.quality > 0 ? Math.round(selectedNodeData.quality * 100) + '%' : '—'}</span></div>
              </div>
            </div>
          ) : (
            <div className="rounded-2xl border border-white/5 bg-white/2 p-4 text-center">
              <Network size={20} className="text-white/20 mx-auto mb-2"/>
              <p className="text-[10px] text-white/35">Cliquez sur un nœud pour voir les détails de l'article</p>
            </div>
          )}

          {/* Légende clusters */}
          <div className="rounded-2xl border border-white/5 bg-white/2 p-3 space-y-2">
            <p className="text-[10px] font-bold uppercase tracking-wider text-white/40">Groupes thématiques</p>
            {data.clusters.slice(0, 6).map((c, i) => (
              <div key={c.id} className="flex items-center justify-between text-[10px]">
                <div className="flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full shrink-0" style={{background: CLUSTER_COLORS[i % CLUSTER_COLORS.length]}}/>
                  <span className="text-white/60">Groupe {c.id + 1}</span>
                </div>
                <span className="text-white/40 font-mono">{c.size} articles</span>
              </div>
            ))}
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 gap-2">
            {[
              {label: 'Articles', value: data.n_nodes},
              {label: 'Connexions', value: data.n_edges},
              {label: 'Groupes', value: data.n_clusters},
              {label: 'Seuil', value: data.min_similarity.toFixed(2)},
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
      setCodeError('Format invalide. Utilisez 4 chiffres (ex: 2847) ou R-2847');
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
    if (!reviewerCode) { alert('Veuillez d\'abord saisir votre code reviewer'); return; }
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

  if (loading) return <LoadingSpinner text="Chargement du screening double-aveugle..." />;

  const kappaColor = !kappa?.kappa ? 'text-white/40'
    : kappa.kappa >= 0.61 ? 'text-brand-300'
    : kappa.kappa >= 0.41 ? 'text-gold-400'
    : 'text-rose-300';

  return (
    <div className="space-y-5">
      <SectionHeader
        icon={<Users size={14} className="text-brand-400" />}
        title="Screening Double-Aveugle"
        subtitle="Évaluation indépendante par deux reviewers identifiés par code — score Kappa de Cohen"
      />

      {/* Identification reviewer */}
      {!reviewerCode ? (
        <div className="rounded-2xl border border-white/10 bg-white/3 p-5 space-y-3">
          <p className="text-xs font-semibold text-white/70">Identification Reviewer</p>
          <p className="text-[10px] text-white/40 leading-relaxed">
            Saisissez votre code reviewer à 4 chiffres pour commencer l'évaluation.
            Ce code vous identifie de manière unique et détermine si vous êtes Reviewer 1 ou Reviewer 2.
          </p>
          <div className="flex gap-2">
            <input
              type="text"
              value={codeInput}
              onChange={e => { setCodeInput(e.target.value); setCodeError(''); }}
              onKeyDown={e => e.key === 'Enter' && handleCodeSubmit()}
              placeholder="Ex: 2847 ou R-2847"
              maxLength={7}
              className="flex-1 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-white placeholder-white/25 focus:outline-none focus:border-brand-500/50"
            />
            <button onClick={handleCodeSubmit}
              className="rounded-xl bg-brand-500 hover:bg-brand-400 text-white font-semibold px-4 py-2 text-xs transition">
              Confirmer
            </button>
          </div>
          {codeError && <p className="text-[10px] text-red-400">{codeError}</p>}
          <p className="text-[9px] text-white/25 italic">Votre code est sauvegardé localement dans ce navigateur.</p>
        </div>
      ) : (
        <div className="flex items-center justify-between gap-3 rounded-2xl border border-brand-500/20 bg-brand-500/5 px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-xl bg-brand-500/20 flex items-center justify-center">
              <span className="text-xs font-bold text-brand-300">R{reviewer}</span>
            </div>
            <div>
              <p className="text-xs font-semibold text-white/80">Reviewer {reviewer} — <span className="font-mono text-brand-300">{reviewerCode}</span></p>
              <p className="text-[9px] text-white/35">Identité sauvegardée dans ce navigateur</p>
            </div>
          </div>
          <button onClick={handleResetCode} className="text-[9px] text-white/25 hover:text-white/50 transition">Changer</button>
        </div>
      )}

      {/* Score Kappa */}
      {kappa && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            {label: 'Articles évalués', value: kappa.n_evaluated, color: 'text-white'},
            {label: 'Score Kappa', value: kappa.kappa != null ? kappa.kappa.toFixed(3) : 'N/A', color: kappaColor},
            {label: 'Accord observé', value: kappa.po_observed != null ? Math.round(kappa.po_observed * 100) + '%' : 'N/A', color: 'text-brand-300'},
            {label: 'Conflits', value: kappa.conflicts, color: kappa.conflicts > 0 ? 'text-gold-400' : 'text-white/40'},
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
            <div className="text-[10px] text-white/35">Accord attendu par hasard : {Math.round((kappa.pe_expected || 0) * 100)}%</div>
          </div>
        </div>
      )}
      {kappa?.n_evaluated === 0 && (
        <div className="rounded-xl border border-gold-500/20 bg-gold-500/5 p-3 text-xs text-gold-300">
          Aucune évaluation double-aveugle n'a encore été soumise. Commencez à évaluer les articles dans la section Corpus.
        </div>
      )}

      {/* Conflits */}
      {conflicts.length > 0 && (
        <div className="space-y-3">
          <p className="text-[10px] font-bold uppercase tracking-wider text-gold-400">
            {conflicts.length} conflit(s) à résoudre
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
                  >Inclure (arbitrage)</button>
                  <button onClick={() => decide(art.id, 'excluded')} disabled={submitting === art.id}
                    className="flex-1 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-[10px] py-1.5 hover:bg-rose-500/20 transition disabled:opacity-50"
                  >Exclure (arbitrage)</button>
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
  const [email, setEmail] = React.useState('');
  const [frequency, setFrequency] = React.useState<'daily'|'weekly'|'immediate'>('weekly');
  const [subscribed, setSubscribed] = React.useState(false);
  const [subscribing, setSubscribing] = React.useState(false);
  const [lrStatus, setLrStatus] = React.useState<any>(null);
  const [lrLoading, setLrLoading] = React.useState(false);

  const handleSubscribe = async () => {
    if (!email.trim()) return;
    setSubscribing(true);
    try {
      await subscribeAlerts(email, scenarioId, frequency);
      setSubscribed(true);
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
        title="Alertes & Living Review"
        subtitle="Abonnez-vous aux nouvelles publications et gérez le pipeline de mise à jour automatique"
      />

      {/* Alertes email */}
      <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-4">
        <div className="flex items-center gap-2">
          <Bell size={13} className="text-gold-400"/>
          <h4 className="text-sm font-semibold text-white">Alertes email</h4>
        </div>
        <p className="text-xs text-white/50">Recevez une notification quand de nouveaux articles sont ajoutés à ce scénario.</p>
        {subscribed ? (
          <div className="rounded-xl border border-brand-500/20 bg-brand-500/5 p-3 flex items-center gap-2 text-xs text-brand-300">
            <CheckCircle2 size={13}/>
            Abonnement confirmé pour <strong>{email}</strong> · fréquence : {frequency}
          </div>
        ) : (
          <div className="space-y-3">
            <input
              type="email" value={email} onChange={e => setEmail(e.target.value)}
              placeholder="votre@email.com"
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
                >{f === 'immediate' ? 'Immédiat' : f === 'daily' ? 'Quotidien' : 'Hebdomadaire'}</button>
              ))}
            </div>
            <button onClick={handleSubscribe} disabled={subscribing || !email.trim()}
              className="flex items-center gap-2 rounded-xl bg-brand-500 hover:bg-brand-400 text-white font-semibold px-4 py-2 text-xs transition disabled:opacity-50"
            >
              {subscribing ? <Loader2 size={12} className="animate-spin"/> : <Bell size={12}/>}
              S'abonner aux alertes
            </button>
          </div>
        )}
      </div>

      {/* Living Review */}
      <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-4">
        <div className="flex items-center gap-2">
          <Rss size={13} className="text-brand-400"/>
          <h4 className="text-sm font-semibold text-white">Pipeline Living Review</h4>
        </div>
        <p className="text-xs text-white/50">
          Le pipeline Living Review interroge automatiquement PubMed avec la requête booléenne de ce scénario,
          insère les nouveaux articles, génère les embeddings et recalcule le clustering.
        </p>
        <div className="rounded-xl border border-white/5 bg-white/2 p-3 space-y-1.5 text-[10px] text-white/40">
          {['Interrogation PubMed avec requête booléenne','Insertion des nouveaux articles','Génération des embeddings (text-embedding-3-small)','Déduplication automatique','Recalcul du clustering UMAP + HDBSCAN','Mise à jour des résumés de clusters'].map((step, i) => (
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
          Simuler (dry run)
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

function EnrichmentSection({ scenarioId }: { scenarioId: string }) {
  const [status, setStatus] = React.useState<EnrichmentStatus | null>(null);
  const [loadingStatus, setLoadingStatus] = React.useState(false);
  const [running, setRunning] = React.useState<string | null>(null);
  const [lastResult, setLastResult] = React.useState<{ type: string; msg: string } | null>(null);
  const [limit, setLimit] = React.useState(50);

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
      else res = await fetchFulltextBatch(scenarioId, Math.min(limit, 20));
      setLastResult({ type, msg: res.message });
      await loadStatus();
    } catch (e: any) {
      setLastResult({ type, msg: `Erreur : ${e.message}` });
    } finally {
      setRunning(null);
    }
  };

  const JOBS = [
    {
      key: "pico" as const,
      label: "Extraction PICO",
      desc: "Extrait Population, Intervention, Comparateur, Outcome via LLM pour chaque article sans PICO.",
      icon: <Microscope size={15} className="text-brand-400" />,
      stat: status ? `${status.pico.count} / ${status.total} (${status.pico.pct}%)` : "—",
      pct: status ? status.pico.pct : 0,
      color: "bg-brand-500",
    },
    {
      key: "metadata" as const,
      label: "Enrichissement Métadonnées",
      desc: "Complète le type d'étude, la taille d'échantillon, le pays, le risque de biais via LLM.",
      icon: <Database size={15} className="text-gold-400" />,
      stat: status ? `${status.metadata.count} / ${status.total} (${status.metadata.pct}%)` : "—",
      pct: status ? status.metadata.pct : 0,
      color: "bg-gold-500",
    },
    {
      key: "fulltext" as const,
      label: "Récupération Full-Text",
      desc: "Recherche le texte intégral en accès libre via Unpaywall (DOI requis).",
      icon: <Globe size={15} className="text-forest-300" />,
      stat: status ? `${status.fulltext.count} / ${status.total} (${status.fulltext.pct}%)` : "—",
      pct: status ? status.fulltext.pct : 0,
      color: "bg-forest-400",
    },
  ];

  const userScenario = isUserScenario(scenarioId);

  return (
    <div className="space-y-5">
      <SectionHeader
        icon={<Zap size={16} className="text-gold-400" />}
        title="Enrichissement LLM"
        subtitle="Lancez les enrichissements automatiques sur les articles de ce scénario"
      />

      {userScenario && (
        <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 px-4 py-3 flex items-start gap-3">
          <Info size={14} className="text-brand-400 shrink-0 mt-0.5" />
          <div className="text-xs text-brand-200/80 leading-relaxed">
            <strong className="text-brand-300">Enrichissement automatique actif</strong> — Le pipeline de votre scénario lance automatiquement l'extraction PICO et l'enrichissement des métadonnées lors de l'ingéstion des articles. Utilisez les boutons ci-dessous uniquement pour compléter les articles qui auraient été manqués ou pour relancer un enrichissement spécifique.
          </div>
        </div>
      )}

      {/* Paramètre lot */}
      <div className="flex items-center gap-3 rounded-2xl border border-white/8 bg-white/3 px-4 py-3">
        <label className="text-xs text-white/50 shrink-0">Taille du lot :</label>
        <input
          type="number"
          min={5} max={200} step={5}
          value={limit}
          onChange={e => setLimit(Number(e.target.value))}
          className="w-20 rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-xs text-white text-center focus:outline-none focus:border-brand-500/50"
        />
        <span className="text-xs text-white/30">articles par exécution</span>
        <button
          onClick={loadStatus}
          disabled={loadingStatus}
          className="ml-auto flex items-center gap-1.5 rounded-xl border border-white/10 px-3 py-1.5 text-xs text-white/50 hover:text-white hover:bg-white/8 transition"
        >
          <RefreshCw size={11} className={loadingStatus ? "animate-spin" : ""} />
          Actualiser
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
                <span>Couverture</span>
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
                <><Loader2 size={12} className="animate-spin" /> En cours...</>
              ) : (
                <><Zap size={12} /> Lancer ({limit} articles)</>
              )}
            </button>
          </div>
        ))}
      </div>

      {/* Résultat dernier job */}
      {lastResult && (
        <div className={`flex items-start gap-2.5 rounded-2xl border px-4 py-3 text-xs ${
          lastResult.msg.startsWith("Erreur")
            ? "border-red-500/20 bg-red-500/5 text-red-300"
            : "border-brand-500/20 bg-brand-500/5 text-brand-200"
        }`}>
          {lastResult.msg.startsWith("Erreur") ? <AlertCircle size={13} className="mt-0.5 shrink-0" /> : <CheckCircle2 size={13} className="mt-0.5 shrink-0" />}
          <span><strong className="font-semibold capitalize">{lastResult.type}</strong> — {lastResult.msg}</span>
        </div>
      )}
    </div>
  );
}

// ─── Composite Tabs ──────────────────────────────────────────────────────────


// ─── Seuil de similarite ajustable ─────────────────────────────────────────

function SeuilSection({ scenarioId }: { scenarioId: string }) {
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
    } catch {}
    setSaving(false);
  };

  const handleRerank = async () => {
    setRerankStatus("Lancement du scoring...");
    try {
      await triggerRerank(scenarioId);
      pollRef.current = setInterval(() => {
        getRerankStatus(scenarioId).then(s => {
          if (s.status === "done") {
            if (pollRef.current) clearInterval(pollRef.current);
            setRerankStatus("Scoring termine : " + (s.updated ?? "?") + " articles mis a jour.");
            setTimeout(() => setRerankStatus(null), 4000);
          } else if (s.status === "error") {
            if (pollRef.current) clearInterval(pollRef.current);
            setRerankStatus("Erreur de scoring.");
          }
        });
      }, 3000);
    } catch (e: any) {
      setRerankStatus("Erreur : " + e.message);
    }
  };

  return (
    <div className="rounded-2xl border border-white/8 bg-white/2 px-4 py-3 flex flex-wrap items-center gap-4">
      <div className="flex items-center gap-2 text-xs text-white/60">
        <Zap size={12} className="text-brand-400 shrink-0" />
        <span className="font-medium text-white/70">Seuil de pertinence semantique :</span>
        <input
          type="range" min={0.1} max={0.9} step={0.05}
          value={threshold}
          onChange={e => setThreshold(parseFloat(e.target.value))}
          className="w-28 accent-brand-500"
        />
        <span className="font-mono text-brand-300 w-10 text-center">{threshold.toFixed(2)}</span>
      </div>
      <div className="flex items-center gap-2">
        <button onClick={handleSave} disabled={saving}
          className="flex items-center gap-1 rounded-lg bg-brand-500/15 hover:bg-brand-500/25 border border-brand-500/20 text-brand-300 px-2.5 py-1 text-[11px] font-medium transition disabled:opacity-50">
          {saving ? (<Loader2 size={10} className="animate-spin" />) : (<CheckCircle2 size={10} />)}
          {saved ? "Sauvegarde !" : "Sauvegarder"}
        </button>
        <button onClick={handleRerank}
          className="flex items-center gap-1 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-white/50 hover:text-white/70 px-2.5 py-1 text-[11px] font-medium transition">
          <RefreshCw size={10} />
          Recalculer scores
        </button>
        {rerankStatus && (
          <span className="text-[10px] text-gold-400">{rerankStatus}</span>
        )}
      </div>
      <p className="text-[10px] text-white/30 w-full">
        Les articles avec un score de similarite superieur a ce seuil (ou valides humainement) sont utilises dans l'Evidence Brief, l'Assistant IA et les Variables.
      </p>
    </div>
  );
}

/** ReviewTab : Corpus + PRISMA + Double-Aveugle (sous-tabs) */
function ReviewTab({ scenarioId, detail }: { scenarioId: string; detail: ScenarioDetail }) {
  const [sub, setSub] = React.useState<"corpus" | "prisma" | "screening">("corpus");
  const SUB = [
    { key: "corpus" as const,    label: "Corpus",         icon: <FileText size={12} /> },
    { key: "prisma" as const,    label: "PRISMA",         icon: <Shield size={12} /> },
    { key: "screening" as const, label: "Double-Aveugle", icon: <Users size={12} /> },
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
          <SeuilSection scenarioId={scenarioId} />
          <CorpusSection scenarioId={scenarioId} detail={detail} />
        </div>
      )}
      {sub === "prisma" && <PrismaSection scenarioId={scenarioId} />}
      {sub === "screening" && <DoubleBlindSection scenarioId={scenarioId} />}
    </div>
  );
}

/** EvidenceTab : PICO + Evidence Brief enrichi (sous-tabs) */
function EvidenceTab({ scenarioId, detail }: { scenarioId: string; detail: ScenarioDetail }) {
  const [sub, setSub] = React.useState<"llm" | "brief" | "pico">("llm");
  const SUB = [
    { key: "llm" as const,   label: "Brief Narratif LLM", icon: <Brain size={12} /> },
    { key: "brief" as const, label: "Evidence Brief",     icon: <BookOpen size={12} /> },
    { key: "pico" as const,  label: "Tableau PICO",       icon: <Table2 size={12} /> },
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
      {sub === "llm" && <LlmEvidenceBriefSection scenarioId={scenarioId} />}
      {sub === "brief" && <EvidenceBriefSection scenarioId={scenarioId} detail={detail} />}
      {sub === "pico" && <PicoSection scenarioId={scenarioId} />}
    </div>
  );
}

/** VizTab : Clustering + Knowledge Graph (sous-tabs) */
function VizTab({ scenarioId }: { scenarioId: string }) {
  const [sub, setSub] = React.useState<"clustering" | "kg">("clustering");
  const SUB = [
    { key: "clustering" as const, label: "Clustering & Topics", icon: <Layers size={12} /> },
    { key: "kg" as const,         label: "Knowledge Graph",     icon: <Network size={12} /> },
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
function VariablesModelTab({ scenarioId, detail }: { scenarioId: string; detail: ScenarioDetail }) {
  const userScenario = isUserScenario(scenarioId);
  const [sub, setSub] = React.useState<"variables" | "model">("variables");
  const SUB = [
    { key: "variables" as const, label: "Données & Variables", icon: <Database size={12} /> },
    ...(!userScenario ? [{ key: "model" as const, label: "Modèle Prédictif", icon: <Brain size={12} /> }] : []),
  ];
  return (
    <div className="space-y-4">
      <div className="flex gap-1.5 border-b border-white/5 pb-3">
        {SUB.map(s => (
          <button key={s.key} onClick={() => setSub(s.key as "variables" | "model")}
            className={`flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-medium transition ${
              sub === s.key ? "bg-brand-700 text-gold-400 font-semibold" : "text-white/60 hover:text-white hover:bg-white/8"
            }`}>
            {s.icon}{s.label}
          </button>
        ))}
      </div>
      {sub === "variables" && <VariablesSection detail={detail} scenarioId={scenarioId} />}
      {sub === "model" && !userScenario && <ModelSection scenarioId={scenarioId} />}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

type SectionKey = "review" | "evidence" | "assistant" | "viz" | "variables" | "queries" | "alerts" | "enrichment";

const SECTIONS: Array<{ key: SectionKey; label: string; icon: React.ReactNode }> = [
  { key: "review",      label: "Corpus & Revue",      icon: <FileText size={13} /> },
  { key: "evidence",    label: "PICO & Evidence",     icon: <BookOpen size={13} /> },
  { key: "assistant",   label: "Assistant IA",        icon: <MessageSquare size={13} /> },
  { key: "viz",         label: "Visualisation",       icon: <Layers size={13} /> },
  { key: "variables",   label: "Variables & Modèle",  icon: <Database size={13} /> },
  { key: "queries",     label: "Stratégie",          icon: <Search size={13} /> },
  { key: "enrichment",  label: "Enrichissement LLM", icon: <Zap size={13} className="text-gold-400" /> },
  { key: "alerts",      label: "Alertes",             icon: <Bell size={13} /> },
];

interface ScenarioDetailPageProps {
  scenarioId: string;
  onBack: () => void;
}

export function ScenarioDetailPage({ scenarioId, onBack }: ScenarioDetailPageProps) {
  const [detail, setDetail] = useState<ScenarioDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<SectionKey>("review");

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
        <span>Chargement du scénario...</span>
      </div>
    );
  }

  if (error || !detail) {
    return (
      <div className="space-y-4">
        <button onClick={onBack} className="flex items-center gap-2 text-sm text-white/50 hover:text-white transition">
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
          className="mt-1 flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-white/50 hover:text-white hover:bg-white/10 transition shrink-0"
        >
          <ArrowLeft size={12} /> Retour
        </button>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h2 className="text-xl font-semibold text-white">{detail.title}</h2>
            <span className="rounded-full border border-brand-500/20 bg-brand-500/10 px-2 py-0.5 text-xs text-brand-300">
              {detail.cluster}
            </span>
            <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-xs text-white/50 font-mono">
              {detail.corpus_stats.total} articles
            </span>
          </div>
          <p className="mt-1 text-sm text-white/50 leading-5">{detail.description}</p>
          
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
                <strong className="text-brand-300 font-semibold block mb-0.5">Enjeu Clinique & Décisionnel :</strong>
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
            {section.label}
          </button>
        ))}
      </div>

      {/* Contenu de la section active */}
      {activeSection === "review" && <ReviewTab scenarioId={scenarioId} detail={detail} />}
      {activeSection === "evidence" && <EvidenceTab scenarioId={scenarioId} detail={detail} />}
      {activeSection === "assistant" && <RagSection scenarioId={scenarioId} detail={detail} />}
      {activeSection === "viz" && <VizTab scenarioId={scenarioId} />}
      {activeSection === "variables" && <VariablesModelTab detail={detail} scenarioId={scenarioId} />}
      {activeSection === "queries" && <QueriesSection detail={detail} />}
      {activeSection === "enrichment" && <EnrichmentSection scenarioId={scenarioId} />}
      {activeSection === "alerts" && <AlertsSection scenarioId={scenarioId} />}
    </div>
  );
}
