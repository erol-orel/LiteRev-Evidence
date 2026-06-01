import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  ArrowLeft, Brain,
  ChevronDown, ChevronUp, Database, ExternalLink, FileText,
  Layers, MessageSquare, RefreshCw, RotateCcw, Search,
  Shield, Terminal, Zap, AlertTriangle,
  Globe, Upload, CheckCircle2, AlertCircle, Info,
  Microscope, Loader2
} from "lucide-react";
import {
  fetchScenarioDetail,
  fetchScenarioCorpus,
  fetchScenarioModelStatus,
  runScenarioModel,
  fetchScenarioClustering,
  askScenarioRag,
  fetchScenarioPrisma,
  uploadScenarioDataset,
  screenArticle,
  fetchArticlePico,
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
        {subtitle && <p className="text-xs text-forest-400 mt-0.5">{subtitle}</p>}
      </div>
    </div>
  );
}

function LoadingSpinner({ text }: { text?: string }) {
  return (
    <div className="flex items-center justify-center py-8 text-forest-400 gap-2">
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
            <p className="text-xs text-forest-500 italic">Aucune requête booléenne définie pour ce scénario.</p>
          )}
        </div>
      </div>
      {/* Natural Language Queries */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <MessageSquare size={12} className="text-brand-400" />
          <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">
            Requêtes Langage Naturel — Recherche Sémantique ({detail.nl_queries.length})
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
            <p className="text-xs text-forest-500 italic">Aucune requête NL définie pour ce scénario.</p>
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
                <tr className="border-b border-white/5 text-[10px] text-forest-400 uppercase tracking-wider">
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
                    <td className="py-3 px-3 text-forest-300 leading-5">{varInfo.definition}</td>
                    <td className="py-3 px-3 text-forest-400 font-mono text-[11px]">{varInfo.source}</td>
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
                <div key={i} className="flex items-center gap-2.5 rounded-xl border border-white/5 bg-white/3 px-3 py-2.5 text-xs text-forest-300">
                  <Database size={12} className="text-brand-400 shrink-0" />
                  <span>{db}</span>
                </div>
              ))
            ) : (
              <p className="text-xs text-forest-500 italic">Aucune base de données répertoriée.</p>
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
            <Upload size={24} className="text-forest-500" />
            <p className="text-xs text-forest-300 font-medium">Glissez-déposez votre fichier ici</p>
            <p className="text-[10px] text-forest-500">Formats acceptés : CSV, Excel (.xlsx, .xls)</p>
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
              <p className="text-[11px] text-forest-300 leading-4">
                {uploadResult.message}
              </p>
              <div className="rounded-lg bg-forest-900/50 p-2 text-[10px] font-mono text-forest-400 space-y-1">
                <div>Lignes détectées : <span className="text-brand-300">{uploadResult.detected_rows}</span></div>
                <div>Colonnes : <span className="text-brand-300">{uploadResult.detected_columns?.slice(0, 5).join(", ")}{uploadResult.detected_columns?.length > 5 ? "..." : ""}</span></div>
              </div>
              <div className="flex items-start gap-1 text-[10px] text-forest-500">
                <Info size={10} className="shrink-0 mt-0.5" />
                <span>Les variables manquantes du modèle seront automatiquement branchées lors du prochain recalcul.</span>
              </div>
            </div>
          )}
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
            <span className="text-[10px] font-bold uppercase tracking-wider text-forest-400">Statut Live du Modèle</span>
            <span className="flex h-2 w-2 rounded-full relative">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${colors.dot}`} />
              <span className={`relative inline-flex rounded-full h-2 w-2 ${colors.dot}`} />
            </span>
          </div>
          <p className="mt-4 text-3xl font-extrabold text-white">{data.status_label}</p>
          {data.model_result && data.model_result.value !== undefined && (
            <div className="mt-4 rounded-2xl bg-white/5 p-4 border border-white/5">
              <p className="text-[10px] text-forest-400 uppercase tracking-wider">Dernière valeur live calculée</p>
              <p className="text-3xl font-black text-brand-300 mt-1 font-mono">
                {typeof data.model_result.value === "number" ? data.model_result.value.toLocaleString() : data.model_result.value}
                {data.model_result.unit && <span className="text-sm font-normal ml-1 text-forest-400">{data.model_result.unit}</span>}
              </p>
              <p className="text-[10px] text-forest-500 mt-1.5 font-mono">Calculé le {new Date(data.timestamp).toLocaleString()}</p>
            </div>
          )}
        </div>

        <div className="space-y-3">
          <div className="flex items-center gap-2 text-xs text-forest-400">
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
              <span className="text-forest-500">Modèle mathématique</span>
              <p className="font-semibold text-white mt-1">{data.model_info.algorithm}</p>
            </div>
            <div className="rounded-xl border border-white/5 bg-white/2 px-3 py-2">
              <span className="text-forest-500">Fréquence de calcul</span>
              <p className="font-semibold text-white mt-1">{data.model_info.update_frequency}</p>
            </div>
            <div className="rounded-xl border border-white/5 bg-white/2 px-3 py-2 sm:col-span-2">
              <span className="text-forest-500">Indicateur de sortie (Outcome)</span>
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
              <p className="text-forest-400 mt-1 font-mono">{data.alert_thresholds.green.condition}</p>
              <p className="text-[10px] text-forest-500 mt-1">{data.alert_thresholds.green.label}</p>
            </div>
            <div className="rounded-xl border border-gold-500/10 bg-gold-500/5 px-3 py-2.5">
              <span className="font-semibold text-gold-300">Orange (Vigilance)</span>
              <p className="text-forest-400 mt-1 font-mono">{data.alert_thresholds.orange.condition}</p>
              <p className="text-[10px] text-forest-500 mt-1">{data.alert_thresholds.orange.label}</p>
            </div>
            <div className="rounded-xl border border-rose-500/10 bg-rose-500/5 px-3 py-2.5">
              <span className="font-semibold text-rose-300">Rouge (Alerte critique)</span>
              <p className="text-forest-400 mt-1 font-mono">{data.alert_thresholds.red.condition}</p>
              <p className="text-[10px] text-forest-500 mt-1">{data.alert_thresholds.red.label}</p>
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
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      {/* Liste des articles */}
      <div className="lg:col-span-2 space-y-4">
        <SectionHeader
          icon={<FileText size={14} className="text-brand-400" />}
          title={`Corpus d'évidences (${data.articles.length} articles validés)`}
          subtitle="Articles validés pour l'extraction de l'évidence"
        />
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
            <p className="text-xs text-forest-500 italic">Aucun article dans ce corpus.</p>
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
                <span className="w-10 text-forest-400 font-mono">{item.year}</span>
                <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-brand-500 rounded-full"
                    style={{ width: `${(item.count / data.total) * 100}%` }}
                  />
                </div>
                <span className="w-6 text-right text-forest-300 font-mono">{item.count}</span>
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
                <span className="w-20 text-forest-400 uppercase font-mono tracking-wider">{item.source}</span>
                <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-brand-500 rounded-full"
                    style={{ width: `${(item.count / data.total) * 100}%` }}
                  />
                </div>
                <span className="w-6 text-right text-forest-300 font-mono">{item.count}</span>
              </div>
            ))}
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
  article: CorpusArticle & { screening_status?: string };
  scenarioId: string;
  isExpanded: boolean;
  onToggle: () => void;
  onScreeningChange?: (id: number, status: string) => void;
}) {
  const [screeningStatus, setScreeningStatus] = React.useState<string>(article.screening_status ?? 'pending');
  const [screeningLoading, setScreeningLoading] = React.useState(false);
  const [pico, setPico] = React.useState<PicoData | null>(null);
  const [picoLoading, setPicoLoading] = React.useState(false);
  const [picoLoaded, setPicoLoaded] = React.useState(false);

  const handleScreen = async (status: 'included' | 'excluded' | 'pending') => {
    setScreeningLoading(true);
    try {
      await screenArticle(scenarioId, article.id, status);
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
  }[screeningStatus] ?? 'bg-white/5 text-forest-400 border border-white/10';

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
            <span className="rounded bg-white/5 px-1.5 py-0.5 text-[10px] font-mono text-forest-400 uppercase tracking-wider">
              {article.source}
            </span>
            {article.year && (
              <span className="text-[10px] font-mono text-forest-500">{article.year}</span>
            )}
            {article.has_fulltext && (
              <span className="rounded-full bg-brand-500/10 border border-brand-500/20 px-2 py-0.5 text-[9px] text-brand-300 font-medium">
                Full-text
              </span>
            )}
            <span className={`rounded-full px-2 py-0.5 text-[9px] font-medium ${statusBadge}`}>
              {statusLabel}
            </span>
          </div>
          <h4 className="text-sm font-semibold text-white mt-1.5 leading-5">{article.title}</h4>
          {article.authors && (
            <p className="text-xs text-forest-500 mt-1 truncate">{article.authors}</p>
          )}
        </div>
        <button className="text-forest-500 hover:text-white shrink-0 mt-1">
          {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>

      {isExpanded && (
        <div className="border-t border-white/5 bg-white/1 p-4 text-xs space-y-4">
          {/* Screening PRISMA */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-forest-400 font-medium mr-1">Screening PRISMA :</span>
            {screeningLoading ? (
              <Loader2 size={14} className="animate-spin text-forest-400" />
            ) : (
              <>
                <button
                  onClick={(e) => { e.stopPropagation(); handleScreen('included'); }}
                  className={`px-2.5 py-1 rounded-full text-[10px] font-semibold border transition ${
                    screeningStatus === 'included'
                      ? 'bg-brand-500/30 text-brand-200 border-brand-500/50'
                      : 'bg-white/5 text-forest-400 border-white/10 hover:bg-brand-500/10 hover:text-brand-300'
                  }`}
                >
                  <CheckCircle2 size={10} className="inline mr-1" />Inclure
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); handleScreen('excluded'); }}
                  className={`px-2.5 py-1 rounded-full text-[10px] font-semibold border transition ${
                    screeningStatus === 'excluded'
                      ? 'bg-rose-500/30 text-rose-200 border-rose-500/50'
                      : 'bg-white/5 text-forest-400 border-white/10 hover:bg-rose-500/10 hover:text-rose-300'
                  }`}
                >
                  <AlertCircle size={10} className="inline mr-1" />Exclure
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); handleScreen('pending'); }}
                  className={`px-2.5 py-1 rounded-full text-[10px] font-semibold border transition ${
                    screeningStatus === 'pending'
                      ? 'bg-gold-500/20 text-gold-300 border-gold-500/30'
                      : 'bg-white/5 text-forest-400 border-white/10 hover:bg-gold-500/10 hover:text-gold-400'
                  }`}
                >
                  En attente
                </button>
              </>
            )}
          </div>

          {/* PICO */}
          {picoLoading ? (
            <div className="flex items-center gap-2 text-forest-500">
              <Loader2 size={12} className="animate-spin" />
              <span>Chargement PICO...</span>
            </div>
          ) : pico ? (
            <div className="rounded-xl border border-white/5 bg-white/2 p-3 space-y-2">
              <p className="text-[10px] font-semibold text-forest-400 uppercase tracking-wider flex items-center gap-1">
                <Microscope size={10} />PICO
                {pico.pico_confidence != null && (
                  <span className="ml-auto font-mono text-forest-500">Confiance : {Math.round(pico.pico_confidence * 100)}%</span>
                )}
              </p>
              <div className="grid grid-cols-2 gap-2">
                {[['P', 'Population', pico.P], ['I', 'Intervention', pico.I], ['C', 'Comparateur', pico.C], ['O', 'Outcome', pico.O]].map(([key, label, val]) => val && (
                  <div key={key} className="rounded-lg bg-white/3 border border-white/5 p-2">
                    <span className="text-[9px] font-bold text-brand-400 uppercase">{key} — {label}</span>
                    <p className="text-forest-300 mt-0.5 leading-4">{val as string}</p>
                  </div>
                ))}
              </div>
              {pico.study_design && (
                <p className="text-[10px] text-forest-500">Type d'étude : <span className="text-forest-300">{pico.study_design}</span></p>
              )}
            </div>
          ) : picoLoaded ? (
            <p className="text-[10px] text-forest-600 italic">PICO non encore extrait pour cet article.</p>
          ) : null}

          {article.abstract && (
            <div>
              <p className="font-semibold text-forest-400 mb-1">Abstract</p>
              <p className="text-forest-300 leading-5">{article.abstract}</p>
            </div>
          )}
          <div className="flex items-center gap-4 flex-wrap text-forest-400 font-mono text-[10px] pt-1">
            {article.journal && <span>Journal: <span className="text-forest-300">{article.journal}</span></span>}
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
              <span className="text-[10px] text-forest-500 border border-white/5 rounded px-1.5 py-0.5">
                <Globe size={9} className="inline mr-0.5" />{article.country}
              </span>
            )}
            {article.keywords && (
              <div className="flex flex-wrap gap-1">
                {article.keywords.split(",").slice(0, 5).map((kw, i) => (
                  <span key={i} className="text-[10px] text-forest-500 bg-forest-800/50 px-1 rounded">
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
            const BASE = (import.meta as any).env?.VITE_API_BASE ?? "/api";
            const r = await fetch(`${BASE}/gesica/scenarios/${scenarioId}/clustering/status`);
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
                <div className="rounded-2xl border border-white/5 bg-white/2 p-4 flex flex-col justify-between">
                  <div>
                    <span className="text-[10px] font-bold uppercase tracking-wider text-forest-400">Visualisation 2D (UMAP)</span>
                    <p className="text-xs text-forest-500 mt-1 leading-4">Chaque point représente un article scientifique. Les articles proches traitent de sujets similaires.</p>
                  </div>

                  {/* Graphique de dispersion SVG interactif */}
                  <div className="relative aspect-square w-full bg-forest-950/40 rounded-xl border border-white/5 mt-4 overflow-hidden flex items-center justify-center">
                    <UmapScatterPlot
                      clusters={data.clusters}
                      selectedCluster={selectedCluster}
                      onSelectCluster={setSelectedCluster}
                    />
                  </div>

                  <div className="flex items-center justify-between text-[10px] text-forest-500 mt-3 font-mono">
                    <span>Axe X (UMAP 1)</span>
                    <span>Axe Y (UMAP 2)</span>
                  </div>
                </div>

                {/* Sélecteur de cluster de gauche */}
                <div className="space-y-2">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-forest-400 block px-1">Sélectionner un groupe</span>
                  <div className="space-y-1 max-h-64 overflow-y-auto pr-1">
                    {data.clusters.map((c) => (
                      <button
                        key={c.cluster_id}
                        onClick={() => setSelectedCluster(c.cluster_id)}
                        className={`w-full text-left rounded-xl px-3 py-2 text-xs transition flex items-center justify-between border ${
                          selectedCluster === c.cluster_id
                            ? "border-brand-500/30 bg-brand-500/10 text-brand-300"
                            : "border-transparent text-forest-400 hover:text-white hover:bg-white/3"
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
                          <p className="text-xs text-forest-400 mt-0.5">{activeClusterData.n_docs} articles scientifiques denses dans ce groupe</p>
                        </div>
                      </div>
                    </div>

                    {/* Résumé clinique LLM */}
                    <div className="space-y-2">
                      <div className="flex items-center gap-1.5 text-xs font-semibold text-brand-300 uppercase tracking-wider">
                        <Brain size={13} />
                        Synthèse du groupe
                      </div>
                      <div className="rounded-2xl border border-brand-500/15 bg-brand-500/5 p-4 text-xs text-forest-200 leading-6 italic">
                        "{activeClusterData.summary}"
                      </div>
                    </div>

                    {/* Mots-clés TF-IDF */}
                    <div className="space-y-2">
                      <p className="text-[10px] font-bold uppercase tracking-wider text-forest-400">Mots-clés prépondérants</p>
                      <div className="flex flex-wrap gap-1.5">
                        {activeClusterData.top_words.map((w, i) => (
                          <span
                            key={i}
                            className="rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-[10px] text-forest-300 font-mono"
                          >
                            {w}
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Article représentatif */}
                    <div className="space-y-2 border-t border-white/5 pt-4">
                      <p className="text-[10px] font-bold uppercase tracking-wider text-forest-400">Article le plus central / représentatif</p>
                      <div className="rounded-xl border border-white/5 bg-white/3 p-3">
                        <div className="flex items-center gap-1.5 text-[10px] text-forest-500 font-mono">
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

// Composant interne pour dessiner les points UMAP en SVG
function UmapScatterPlot({
  clusters,
  selectedCluster,
  onSelectCluster
}: {
  clusters: ClusterResult[];
  selectedCluster: number | null;
  onSelectCluster: (id: number) => void;
}) {
  // Aplatir tous les points pour trouver les min/max et normaliser les coordonnées
  const allPoints: Array<ClusterPoint & { cluster_id: number; is_noise: boolean }> = [];
  clusters.forEach((c) => {
    if (c.points) {
      c.points.forEach((p) => {
        allPoints.push({ ...p, cluster_id: c.cluster_id, is_noise: c.is_noise });
      });
    }
  });

  if (allPoints.length === 0) {
    return <span className="text-xs text-forest-500">Aucun point à afficher.</span>;
  }

  const xs = allPoints.map((p) => p.x);
  const ys = allPoints.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  // Dimensions de la boîte de visualisation SVG
  const width = 300;
  const height = 300;
  const padding = 20;

  return (
    <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`} className="overflow-visible">
      {/* Grille de fond discrète */}
      <line x1={padding} y1={height / 2} x2={width - padding} y2={height / 2} stroke="rgba(255,255,255,0.05)" strokeDasharray="3,3" />
      <line x1={width / 2} y1={padding} x2={width / 2} y2={height - padding} stroke="rgba(255,255,255,0.05)" strokeDasharray="3,3" />

      {/* Tracer tous les points */}
      {allPoints.map((p) => {
        // Normalisation
        const cx = padding + ((p.x - minX) / rangeX) * (width - 2 * padding);
        const cy = height - padding - ((p.y - minY) / rangeY) * (height - 2 * padding); // Inverser Y pour le SVG

        const isSelected = selectedCluster === p.cluster_id;
        const color = getClusterColor(p.cluster_id, p.is_noise);

        return (
          <circle
            key={p.id}
            cx={cx}
            cy={cy}
            r={isSelected ? 6 : p.is_noise ? 2 : 4}
            fill={color}
            opacity={isSelected ? 1.0 : p.is_noise ? 0.2 : 0.6}
            stroke={isSelected ? "#ffffff" : "transparent"}
            strokeWidth={1.5}
            className="cursor-pointer transition-all duration-200 hover:scale-150 hover:opacity-100"
            onClick={() => onSelectCluster(p.cluster_id)}
          >
            <title>{p.title} ({p.year || "Année inconnue"})</title>
          </circle>
        );
      })}
    </svg>
  );
}

// Palette de couleurs pour les clusters
function getClusterColor(clusterId: number, isNoise: boolean): string {
  if (isNoise) return "#64748b"; // Gris ardoise pour le bruit
  const colors = [
    "#10b981", // Émeraude (Cluster 0)
    "#06b6d4", // Cyan (Cluster 1)
    "#8b5cf6", // Violet (Cluster 2)
    "#f59e0b", // Ambre (Cluster 3)
    "#ec4899", // Rose (Cluster 4)
    "#3b82f6", // Bleu (Cluster 5)
    "#14b8a6", // Sarcelle (Cluster 6)
    "#a855f7", // Mauve (Cluster 7)
    "#f43f5e", // Rose vif (Cluster 8)
  ];
  return colors[clusterId % colors.length];
}

// ─── Section: RAG ─────────────────────────────────────────────────────────────

function RagSection({ scenarioId, detail }: { scenarioId: string; detail: ScenarioDetail }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScenarioRagResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const ask = async (qText: string) => {
    if (!qText.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await askScenarioRag(scenarioId, qText);
      setResult(res);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const suggestedQuestions = detail.nl_queries.slice(0, 3);

  return (
    <div className="rounded-3xl border border-white/10 bg-white/3 p-5 space-y-5">
      <SectionHeader
        icon={<MessageSquare size={14} className="text-brand-400" />}
        title="Assistant Scientifique RAG Dédié"
        subtitle="Posez des questions directement sur le corpus validé de ce scénario"
      />

      {/* Questions suggérées */}
      {suggestedQuestions.length > 0 && (
        <div className="space-y-2">
          <p className="text-[10px] font-bold uppercase tracking-wider text-forest-500">Questions cliniques suggérées</p>
          <div className="flex flex-wrap gap-2">
            {suggestedQuestions.map((q, i) => (
              <button
                key={i}
                onClick={() => {
                  setQuestion(q);
                  ask(q);
                }}
                disabled={loading}
                className="text-left rounded-xl border border-white/5 bg-white/2 hover:bg-white/5 px-3 py-2 text-xs text-forest-300 hover:text-white transition disabled:opacity-50"
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
          disabled={loading}
          className="flex-1 rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-xs text-white focus:outline-none focus:border-brand-500/50 transition disabled:opacity-50"
        />
        <button
          onClick={() => ask(question)}
          disabled={loading || !question.trim()}
          className="rounded-xl bg-brand-500 hover:bg-brand-400 text-forest-950 font-semibold px-4 text-xs transition disabled:opacity-50 shrink-0"
        >
          Poser
        </button>
      </div>

      {loading && <LoadingSpinner text="Recherche sémantique et génération de la réponse..." />}
      {error && <ErrorBox message={error} />}

      {result && !loading && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3 border-t border-white/5 pt-5">
          {/* Réponse */}
          <div className="lg:col-span-2 space-y-3">
            <p className="text-[10px] font-bold uppercase tracking-wider text-forest-400">Réponse de l'Assistant</p>
            <div className="rounded-2xl border border-white/5 bg-white/2 p-4 text-xs text-forest-200 leading-6 whitespace-pre-wrap">
              {result.answer}
            </div>
            {result.model && (
              <p className="text-[10px] text-forest-500 font-mono text-right">Généré via {result.model}</p>
            )}
          </div>

          {/* Sources citées */}
          <div className="space-y-3">
            <p className="text-[10px] font-bold uppercase tracking-wider text-forest-400">Sources scientifiques citées</p>
            <div className="space-y-2 max-h-[300px] overflow-y-auto pr-1">
              {result.sources.map((src, i) => (
                <div key={i} className="rounded-xl border border-white/5 bg-white/3 p-2.5 text-xs">
                  <div className="flex items-center justify-between gap-2">
                    <span className="rounded bg-brand-500/10 border border-brand-500/20 px-1.5 py-0.5 text-[9px] text-brand-300 font-mono">
                      SOURCE {i + 1}
                    </span>
                    <span className="text-[10px] text-forest-500 font-mono">Pertinence: {(src.score * 100).toFixed(0)}%</span>
                  </div>
                  <h5 className="font-semibold text-white mt-1.5 leading-4 line-clamp-2">{src.title}</h5>
                  <p className="text-[10px] text-forest-500 mt-1 truncate">
                    {src.authors} • {src.year || "N/A"}
                  </p>
                </div>
              ))}
            </div>
          </div>
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
        title: "Exclus — titre / résumé",
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
        title: "Exclus — plein texte",
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
        title="Diagramme PRISMA 2020 — Flow de Sélection"
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
        <p className="text-[10px] text-forest-500 uppercase tracking-wider mb-3">Résumé des étapes</p>
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
              <p className="text-[9px] uppercase tracking-wider text-forest-400 mb-1">{label}</p>
              <p className="text-xl font-bold font-mono text-white">{value.toLocaleString()}</p>
              <p className="text-[9px] text-forest-500 mt-0.5">{sub}</p>
            </div>
          ))}
        </div>
      </div>

      {!inc.screening_complete && (
        <div className="rounded-xl border border-gold-500/20 bg-gold-500/5 px-4 py-3">
          <p className="text-[10px] text-gold-400">
            <span className="font-semibold">Screening manuel non effectué</span> — Les articles sont disponibles pour évaluation dans l'onglet Corpus. Utilisez l'interface de screening pour inclure ou exclure chaque article.
          </p>
        </div>
      )}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

type SectionKey = "corpus" | "prisma" | "rag" | "variables" | "model" | "clustering" | "queries";

const SECTIONS: Array<{ key: SectionKey; label: string; icon: React.ReactNode }> = [
  { key: "corpus",     label: "Corpus",                   icon: <FileText size={13} /> },
  { key: "prisma",     label: "PRISMA",                   icon: <Shield size={13} /> },
  { key: "rag",        label: "Evidence (RAG)",            icon: <MessageSquare size={13} /> },
  { key: "variables", label: "Données & Variables",       icon: <Database size={13} /> },
  { key: "model",     label: "Modèle prédictif",          icon: <Brain size={13} /> },
  { key: "clustering",label: "Clustering & Topics",       icon: <Layers size={13} /> },
  { key: "queries",   label: "Stratégie de recherche",    icon: <Search size={13} /> },
];

interface ScenarioDetailPageProps {
  scenarioId: string;
  onBack: () => void;
}

export function ScenarioDetailPage({ scenarioId, onBack }: ScenarioDetailPageProps) {
  const [detail, setDetail] = useState<ScenarioDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<SectionKey>("corpus");

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
      <div className="flex items-center justify-center py-24 text-forest-400 gap-2">
        <RotateCcw size={18} className="animate-spin" />
        <span>Chargement du scénario...</span>
      </div>
    );
  }

  if (error || !detail) {
    return (
      <div className="space-y-4">
        <button onClick={onBack} className="flex items-center gap-2 text-sm text-forest-400 hover:text-white transition">
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
          className="mt-1 flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-forest-400 hover:text-white hover:bg-white/10 transition shrink-0"
        >
          <ArrowLeft size={12} /> Retour
        </button>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h2 className="text-xl font-semibold text-white">{detail.title}</h2>
            <span className="rounded-full border border-brand-500/20 bg-brand-500/10 px-2 py-0.5 text-xs text-brand-300">
              {detail.cluster}
            </span>
            <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-xs text-forest-400 font-mono">
              {detail.corpus_stats.total} articles
            </span>
          </div>
          <p className="mt-1 text-sm text-forest-400 leading-5">{detail.description}</p>
          {/* Actions recommandées */}
          {detail.recommended_actions.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {detail.recommended_actions.slice(0, 2).map((action, i) => (
                <span key={i} className="flex items-center gap-1.5 rounded-xl border border-white/5 bg-white/3 px-2.5 py-1 text-xs text-forest-300">
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
                ? "border border-brand-500/30 bg-brand-500/10 text-brand-300"
                : "border border-transparent text-forest-400 hover:text-white hover:bg-white/5"
            }`}
          >
            {section.icon}
            {section.label}
          </button>
        ))}
      </div>

      {/* Contenu de la section active */}
      {activeSection === "queries" && <QueriesSection detail={detail} />}
      {activeSection === "variables" && <VariablesSection detail={detail} scenarioId={scenarioId} />}
      {activeSection === "model" && <ModelSection scenarioId={scenarioId} />}
      {activeSection === "corpus" && <CorpusSection scenarioId={scenarioId} detail={detail} />}
      {activeSection === "clustering" && <ClusteringSection scenarioId={scenarioId} />}
      {activeSection === "rag" && <RagSection scenarioId={scenarioId} detail={detail} />}
      {activeSection === "prisma" && <PrismaSection scenarioId={scenarioId} />}
    </div>
  );
}
