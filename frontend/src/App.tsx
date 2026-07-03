import React, { useEffect, useMemo, useRef, useState } from "react";
import { ScenarioDetailPage, EnrichmentSection } from "./components/ScenarioDetailPage";
import { useI18n } from "./i18n/LanguageProvider";
import { Activity, BarChart2, BookOpen, Cloud, Download, ExternalLink, FolderOpen, MapPin, AlertTriangle, Users, Pill, Radio, RefreshCw, RotateCcw, ChevronDown, ChevronUp, Zap, Lock, KeyRound, Wrench, Trash2 } from "lucide-react";

import {
  fetchDocumentDetail,
  fetchEvidenceSummary,
  fetchGesicaScenarios,
  fetchCorpusStats,
  getFilterOptions,
  getReadableExcerpt,

  fetchTerrainMeteo,
  fetchTerrainGeo,
  fetchTerrainEpidemic,
  fetchTerrainDemographics,
  fetchTerrainPharmacies,
  fetchTerrainInformalSignals,
  fetchTerrainClimate,
  fetchFulltextStats,
  corpusMaintenance,
  embedPending,
  fetchCorpusStatsByYear,
  fetchCorpusStatsByYearNamed,
  type CorpusStats,
  type CorpusStatsByYear,
  type DocumentDetailResponse,
  type EvidenceSummaryResponse,
  type FilterOptions,
  type GesicaScenario,

  type TerrainMeteo,
  type TerrainGeo,
  type TerrainEpidemic,
  type TerrainDemographics,
  type TerrainPharmacies,
  type TerrainInformalSignals,
  type TerrainClimate,
  type FulltextStats,
  type CorpusMaintenanceReport,
  fetchSearchStrategy,
  type SearchStrategy,
  populateUserScenario,
  fetchUserScenarioPopulateStatus,
  fetchScenarioCorpus,
  type CorpusArticle,
  fetchUserScenarios,
  createUserScenario,
  deleteUserScenario,
  patchUserScenario,
  startUserScenarioPipeline,
  fetchUserScenarioPipelineStatus,
  fetchFolders,
  createFolder,
  updateFolder,
  deleteFolder,
  assignScenarioToFolder,
  getRecommendedActions,
  hasApiKey,
  setApiKey,
  clearApiKey,
  type UserScenario,
  type UserScenarioPipelineStatus,
  type ScenarioFolder,
  type SubQuery,
} from "./lib/api";
import type {
  ProjectContext,
  RelevanceLabel,
  SearchFilters,
  SearchMode,
  SearchResult,
} from "./types/search";

// The second tuple element is an i18n key resolved via t() at render time
// (this array lives at module level, where the t() hook is unavailable).
const FILTER_FIELDS: Array<[keyof FilterOptions, string]> = [
  ["sourceType", "search.filterSourceType"],
  ["diseaseOrCondition", "search.filterDiseaseOrCondition"],
  ["scenarioType", "search.filterScenarioType"],
  ["geographicScope", "search.filterGeographicScope"],
  ["evidenceCategory", "search.filterEvidenceCategory"],
];

const PAGE_SIZE = 20;

// Bornes du filtre Années : du plus ancien article réellement en base (en
// ignorant les années aberrantes < 1000) jusqu'à l'année courante (aujourd'hui).
function yearSliderBounds(yearOpts?: Array<{ value: string | number }> | null): { min: number; max: number } {
  const yrs = (yearOpts ?? []).map(y => Number(y.value)).filter(y => Number.isFinite(y) && y > 1000);
  return { min: yrs.length ? Math.min(...yrs) : 1990, max: new Date().getFullYear() };
}

type AppTab = "search" | "scenarios" | "stats" | "terrain";

interface SavedSearch {
  id: string;
  query: string;
  mode: SearchMode;
  projectContext: ProjectContext;
  timestamp: number;
  resultCount: number;
  name?: string;
  pinned?: boolean;
}

// localStorage supprimé : les recherches sauvegardées sont désormais persistées en backend (table user_scenarios)

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

// ─── TerrainView ─────────────────────────────────────────────────────────────
function TerrainView() {
  const { t } = useI18n();
  const [meteo, setMeteo] = useState<TerrainMeteo | null>(null);
  const [geo, setGeo] = useState<TerrainGeo | null>(null);
  const [epidemic, setEpidemic] = useState<TerrainEpidemic | null>(null);
  const [demographics, setDemographics] = useState<TerrainDemographics | null>(null);
  const [pharmacies, setPharmacies] = useState<TerrainPharmacies | null>(null);
  const [informalSignals, setInformalSignals] = useState<TerrainInformalSignals | null>(null);
  const [climate, setClimate] = useState<TerrainClimate | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  const loadAll = () => {
    setLoading(true);
    setError(null);
    Promise.all([
      fetchTerrainMeteo(),
      fetchTerrainGeo(),
      fetchTerrainEpidemic(),
      fetchTerrainDemographics(),
      fetchTerrainPharmacies(),
      fetchTerrainInformalSignals(),
      fetchTerrainClimate(),
    ])
      .then(([m, g, e, d, p, s, c]) => {
        setMeteo(m);
        setGeo(g);
        setEpidemic(e);
        setDemographics(d);
        setPharmacies(p);
        setInformalSignals(s);
        setClimate(c);
        setLastRefresh(new Date());
      })
      .catch((err) => setError(err.message || t("terrain.loadError")))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadAll();
  }, []);

  const alertColors: Record<string, string> = {
    none: "border-brand-500/30 bg-brand-500/10 text-brand-300",
    warning: "border-gold-500/30 bg-gold-500/10 text-gold-300",
    danger: "border-rose-500/30 bg-rose-500/10 text-rose-300",
  };
  const riskColors: Record<string, string> = {
    low: "text-brand-300",
    moderate: "text-gold-300",
    high: "text-rose-300",
  };
  const statusColors: Record<string, string> = {
    under_threshold: "bg-brand-500/10 text-brand-300 border-brand-500/30",
    warning: "bg-gold-500/10 text-gold-300 border-gold-500/30",
    epidemic: "bg-rose-500/10 text-rose-300 border-rose-500/30",
  };
  const trendIcons: Record<string, string> = {
    increasing: "↑",
    stable: "→",
    decreasing: "↓",
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16 text-forest-400">
        <RotateCcw size={18} className="mr-2 animate-spin" />
        {t("terrain.loading")}
      </div>
    );
  }
  if (error) {
    return (
      <div className="rounded-3xl border border-rose-500/30 bg-rose-500/10 p-6 text-rose-300">
        <AlertTriangle size={18} className="mb-2" />
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Cloud size={20} className="text-brand-400" />
          <div>
            <h2 className="text-xl font-semibold text-white">{t("terrain.title")}</h2>
            <p className="text-xs text-forest-400 mt-0.5">
              {t("terrain.subtitle")}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-forest-500">{t("terrain.refreshedAt")} {lastRefresh.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" })}</span>
          <button
            onClick={loadAll}
            className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-forest-300 hover:bg-white/10 transition"
          >
            <RefreshCw size={12} />
            {t("common.refresh")}
          </button>
        </div>
      </div>

      {/* Grille de KPIs sources */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
        {[
          { id: "meteo", label: t("terrain.kpi.meteo"), icon: <Cloud size={14} />, color: "text-brand-400", active: !!meteo },
          { id: "routing", label: t("terrain.kpi.routing"), icon: <MapPin size={14} />, color: "text-violet-400", active: !!geo },
          { id: "epidemic", label: t("terrain.kpi.epidemic"), icon: <Activity size={14} />, color: "text-brand-400", active: !!epidemic },
          { id: "demographics", label: t("terrain.kpi.demographics"), icon: <Users size={14} />, color: "text-gold-400", active: !!demographics },
          { id: "pharmacies", label: t("terrain.kpi.pharmacies"), icon: <Pill size={14} />, color: "text-rose-400", active: !!pharmacies },
          { id: "signals", label: t("terrain.kpi.signals"), icon: <Radio size={14} />, color: "text-brand-400", active: !!informalSignals },
          { id: "copernicus", label: t("terrain.kpi.copernicus"), icon: <Zap size={14} />, color: "text-gold-400", active: !!climate },
        ].map((s) => (
          <div key={s.id} className={`rounded-2xl border p-3 text-center transition ${
            s.active ? "border-white/10 bg-white/5" : "border-white/5 bg-white/2 opacity-40"
          }`}>
            <div className={`flex justify-center mb-1 ${s.color}`}>{s.icon}</div>
            <p className="text-xs text-forest-300 font-medium">{s.label}</p>
            <p className={`text-[10px] mt-0.5 ${s.active ? "text-brand-400" : "text-forest-500"}`}>
              {s.active ? t("terrain.active") : t("terrain.inactive")}
            </p>
          </div>
        ))}
      </div>

      {/* Météo */}
      {meteo && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <div className="flex items-center justify-between mb-4">
            <h3 className="flex items-center gap-2 text-lg font-semibold text-white">
              <Cloud size={16} className="text-brand-400" />
              {t("terrain.meteo.heading")} {meteo.station}
            </h3>
            <span className={`rounded-full border px-3 py-1 text-xs font-medium ${alertColors[meteo.alert_level] ?? alertColors.none}`}>
              {meteo.alert_level === "none" ? t("terrain.meteo.alertNone") : meteo.alert_level === "warning" ? t("terrain.meteo.alertWarning") : t("terrain.meteo.alertDanger")}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 mb-4">
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-brand-300">{meteo.temperature}°C</p>
              <p className="mt-1 text-xs text-forest-400">{t("terrain.meteo.temperature")}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-brand-300">{meteo.apparent_temperature}°C</p>
              <p className="mt-1 text-xs text-forest-400">{t("terrain.meteo.apparent")}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-brand-300">{meteo.humidity}%</p>
              <p className="mt-1 text-xs text-forest-400">{t("terrain.meteo.humidity")}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-brand-300">{meteo.wind_speed} km/h</p>
              <p className="mt-1 text-xs text-forest-400">{t("terrain.meteo.wind")}</p>
            </div>
          </div>
          <div className={`rounded-2xl border p-3 text-sm ${alertColors[meteo.alert_level] ?? alertColors.none}`}>
            <p className="font-medium">{meteo.alert_description}</p>
            <p className="mt-1 opacity-80">{meteo.impact_on_ems}</p>
          </div>
          <p className="mt-2 text-xs text-forest-500 italic">{meteo.architecture_note}</p>
        </div>
      )}

      {/* Géo / Routage */}
      {geo && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <h3 className="flex items-center gap-2 text-lg font-semibold text-white mb-4">
            <MapPin size={16} className="text-violet-400" />
            {t("terrain.geo.heading")} {geo.origin.label} → {geo.destination.label}
          </h3>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 mb-4">
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-violet-300">{geo.distance_km} km</p>
              <p className="mt-1 text-xs text-forest-400">{t("terrain.geo.distance")}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-violet-300">{geo.base_duration_min} min</p>
              <p className="mt-1 text-xs text-forest-400">{t("terrain.geo.baseDuration")}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-gold-300">{geo.cross_border_delay_min} min</p>
              <p className="mt-1 text-xs text-forest-400">{t("terrain.geo.customsDelay")}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-rose-300">{geo.total_estimated_response_time_min} min</p>
              <p className="mt-1 text-xs text-forest-400">{t("terrain.geo.totalEstimated")}</p>
            </div>
          </div>
          <div className="rounded-2xl border border-violet-500/30 bg-violet-500/10 p-3 text-sm text-violet-300">
            <p className="font-medium">{t("terrain.geo.coordinationAction")}</p>
            <p className="mt-1 opacity-80">{geo.coordination_action}</p>
          </div>
          <p className="mt-2 text-xs text-forest-500 italic">{geo.architecture_note}</p>
        </div>
      )}

      {/* Épidémie */}
      {epidemic && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <div className="flex items-center justify-between mb-4">
            <h3 className="flex items-center gap-2 text-lg font-semibold text-white">
              <Activity size={16} className="text-brand-400" />
              {t("terrain.epidemic.heading")} {epidemic.region}
            </h3>
            <span className={`text-lg font-bold ${riskColors[epidemic.global_ems_impact_risk] ?? "text-white"}`}>
              {t("terrain.epidemic.emsRisk")} {epidemic.global_ems_impact_risk.toUpperCase()}
            </span>
          </div>
          <div className="space-y-3 mb-4">
            {epidemic.diseases.map((d) => (
              <div key={d.name} className="rounded-2xl border border-white/10 bg-forest-900/60 p-3">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-white text-sm">{d.name}</span>
                  <div className="flex items-center gap-2">
                    <span className={`rounded-full border px-2 py-0.5 text-xs ${statusColors[d.status] ?? ""}`}>
                      {d.status === "under_threshold" ? t("terrain.epidemic.statusUnderThreshold") : d.status === "warning" ? t("terrain.epidemic.statusWarning") : t("terrain.epidemic.statusEpidemic")}
                    </span>
                    <span className={`text-sm font-bold ${
                      d.trend === "increasing" ? "text-rose-300" : d.trend === "decreasing" ? "text-brand-300" : "text-forest-300"
                    }`}>{trendIcons[d.trend]}</span>
                  </div>
                </div>
                <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-forest-400">
                  <span>{t("terrain.epidemic.france")} <span className="text-white font-medium">{d.incidence_per_100k_france}{t("terrain.epidemic.perThreshold")}</span></span>
                  <span>{t("terrain.epidemic.switzerland")} <span className="text-white font-medium">{d.incidence_per_100k_switzerland}{t("terrain.epidemic.perThreshold")}</span></span>
                  <span>{t("terrain.epidemic.threshold")} <span className="text-white font-medium">{d.epidemic_threshold}{t("terrain.epidemic.perThreshold")}</span></span>
                </div>
              </div>
            ))}
          </div>
          <div className="rounded-2xl border border-brand-500/30 bg-brand-500/10 p-3 text-sm text-brand-300">
            <p className="font-medium">{t("terrain.epidemic.recommendation")}</p>
            <p className="mt-1 opacity-80">{epidemic.recommended_action}</p>
          </div>
          <p className="mt-2 text-xs text-forest-500 italic">{epidemic.architecture_note}</p>
        </div>
      )}

      {/* Démographie */}
      {demographics && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <h3 className="flex items-center gap-2 text-lg font-semibold text-white mb-4">
            <Users size={16} className="text-gold-400" />
            {t("terrain.demographics.heading")} {demographics.commune} ({demographics.postal_code})
          </h3>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 mb-4">
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-gold-300">{demographics.population.toLocaleString("fr-FR")}</p>
              <p className="mt-1 text-xs text-forest-400">{t("terrain.demographics.population")}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-gold-300">{demographics.density_per_km2}</p>
              <p className="mt-1 text-xs text-forest-400">{t("terrain.demographics.density")}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-gold-300">{demographics.age_over_65_pct}%</p>
              <p className="mt-1 text-xs text-forest-400">{t("terrain.demographics.over65")}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-rose-300">×{demographics.ems_risk_multiplier}</p>
              <p className="mt-1 text-xs text-forest-400">{t("terrain.demographics.emsRiskMultiplier")}</p>
            </div>
          </div>
          <p className="text-xs text-forest-500 italic">{demographics.architecture_note}</p>
        </div>
      )}

      {/* Pharmacies & Médicaments */}
      {pharmacies && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <h3 className="flex items-center gap-2 text-lg font-semibold text-white mb-4">
            <Pill size={16} className="text-rose-400" />
            {t("terrain.pharmacies.heading")}
          </h3>
          {pharmacies.critical_medication_alerts.length > 0 && (
            <div className="mb-4 space-y-2">
              <h4 className="text-xs font-semibold uppercase tracking-wider text-forest-400">{t("terrain.pharmacies.criticalAlerts")}</h4>
              {pharmacies.critical_medication_alerts.map((alert, i) => (
                <div key={i} className={`rounded-2xl border p-3 text-sm ${
                  alert.status === "rupture" ? "border-rose-500/30 bg-rose-500/10 text-rose-300" :
                  alert.status === "tension" ? "border-gold-500/30 bg-gold-500/10 text-gold-300" :
                  "border-brand-500/30 bg-brand-500/10 text-brand-300"
                }`}>
                  <div className="flex items-center justify-between">
                    <span className="font-semibold">{alert.medication}</span>
                    <span className="text-xs uppercase font-bold">{alert.status}</span>
                  </div>
                  <p className="mt-1 text-xs opacity-80">{alert.recommendation}</p>
                </div>
              ))}
            </div>
          )}
          <div className="space-y-2">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-forest-400">{t("terrain.pharmacies.nearby")} ({pharmacies.pharmacies_nearby.length})</h4>
            {pharmacies.pharmacies_nearby.map((ph, i) => (
              <div key={i} className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-white">{ph.name}</p>
                  <p className="text-xs text-forest-400">{ph.street}, {ph.city}</p>
                </div>
                <div className="text-right">
                  <span className={`text-xs font-medium ${
                    ph.is_dispensary ? "text-brand-300" : "text-forest-400"
                  }`}>{ph.is_dispensary ? t("terrain.pharmacies.dispensary") : t("terrain.pharmacies.pharmacy")}</span>
                  <p className="text-xs text-forest-500">{ph.opening_hours}</p>
                </div>
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs text-forest-500 italic">{pharmacies.architecture_note}</p>
        </div>
      )}

      {/* Signaux Informels */}
      {informalSignals && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <h3 className="flex items-center gap-2 text-lg font-semibold text-white mb-4">
            <Radio size={16} className="text-brand-400" />
            {t("terrain.signals.heading")}
          </h3>
          <div className="space-y-3">
            {informalSignals.active_signals.map((sig) => (
              <div key={sig.id} className={`rounded-2xl border p-4 ${
                sig.severity === "high" ? "border-rose-500/30 bg-rose-500/5" :
                sig.severity === "moderate" ? "border-gold-500/30 bg-gold-500/5" :
                "border-white/10 bg-white/5"
              }`}>
                <div className="flex items-start justify-between gap-3 mb-2">
                  <div>
                    <p className="text-sm font-semibold text-white">{sig.title}</p>
                    <p className="text-xs text-forest-400 mt-0.5">{sig.source} · {sig.date} · {sig.geo_scope}</p>
                  </div>
                  <div className="flex flex-col items-end gap-1 shrink-0">
                    <span className={`rounded-full border px-2 py-0.5 text-xs font-medium ${
                      sig.severity === "high" ? "border-rose-500/30 bg-rose-500/10 text-rose-300" :
                      sig.severity === "moderate" ? "border-gold-500/30 bg-gold-500/10 text-gold-300" :
                      "border-brand-500/30 bg-brand-500/10 text-brand-300"
                    }`}>{sig.severity.toUpperCase()}</span>
                    <span className="text-xs text-forest-500">{t("terrain.signals.reliability")} {Math.round(sig.reliability_score * 100)}%</span>
                  </div>
                </div>
                <p className="text-sm text-forest-300 leading-5">{sig.content}</p>
                {sig.impact_on_ems && (
                  <p className="mt-2 text-xs text-brand-300 italic">{t("terrain.signals.impactEms")} {sig.impact_on_ems}</p>
                )}
                {sig.impact_on_hospital && (
                  <p className="mt-1 text-xs text-violet-300 italic">{t("terrain.signals.impactHospital")} {sig.impact_on_hospital}</p>
                )}
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs text-forest-500 italic">{informalSignals.architecture_note}</p>
        </div>
      )}

      {/* Copernicus Climate Data Store (CDS) */}
      {climate && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <div className="flex items-center justify-between mb-4">
            <h3 className="flex items-center gap-2 text-lg font-semibold text-white">
              <Zap size={16} className="text-gold-400" />
              {t("terrain.climate.heading")}
            </h3>
            <span className={`rounded-full border px-3 py-1 text-xs font-medium ${
              climate.api_status.includes("verified") ? "border-brand-500/30 bg-brand-500/10 text-brand-300" : "border-gold-500/30 bg-gold-500/10 text-gold-300"
            }`}>
              {climate.api_status === "connected_verified" ? t("terrain.climate.apiConnected") : t("terrain.climate.apiSimulated")}
            </span>
          </div>
          
          {climate.message && (
            <div className="mb-4 rounded-2xl border border-gold-500/20 bg-gold-500/5 p-3 text-xs text-gold-300">
              {climate.message}
            </div>
          )}

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 mb-4">
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-gold-300">{climate.climatology.historical_mean_temp_may_c}°C</p>
              <p className="mt-1 text-xs text-forest-400">{t("terrain.climate.historicalMeanMay")}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-gold-300">+{climate.climatology.current_anomaly_c}°C</p>
              <p className="mt-1 text-xs text-forest-400">{t("terrain.climate.thermalAnomaly")}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-gold-300">{climate.climatology.soil_moisture_deficit_percent}%</p>
              <p className="mt-1 text-xs text-forest-400">{t("terrain.climate.soilMoistureDeficit")}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-rose-300">{climate.climatology.heatwave_hazard_index.toUpperCase()}</p>
              <p className="mt-1 text-xs text-forest-400">{t("terrain.climate.heatwaveRisk")}</p>
            </div>
          </div>

          <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-4">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-forest-400 mb-2">{t("terrain.climate.projectionsTitle")}</h4>
            <ul className="space-y-1.5 text-xs text-forest-300">
              <li>• {t("terrain.climate.heatwaveDaysIncrease")} <span className="text-white font-medium">+{climate.projections_2030.expected_heatwave_days_increase_per_year} {t("terrain.climate.daysPerYear")}</span></li>
              <li>• {t("terrain.climate.heavyPrecipitationIncrease")} <span className="text-white font-medium">+{climate.projections_2030.expected_heavy_precipitation_increase_percent}%</span></li>
              <li>• {t("terrain.climate.emsVulnerabilityFactor")} <span className="text-white font-medium">{climate.projections_2030.ems_vulnerability_factor.replace(/_/g, " ")}</span></li>
            </ul>
          </div>
          
          <p className="mt-3 text-xs text-forest-500 italic">{climate.source}</p>
        </div>
      )}
    </div>
  );
}

function EvidenceStrengthBadge({ strength, showTooltip = false }: { strength: "weak" | "moderate" | "strong" | null; showTooltip?: boolean }) {
  const { t } = useI18n();
  if (!strength) return null;
  const config = {
    strong: { label: t("evidenceStrength.strong.label"), className: "bg-brand-500/20 text-brand-300 border-brand-500/30", tooltip: t("evidenceStrength.strong.tooltip") },
    moderate: { label: t("evidenceStrength.moderate.label"), className: "bg-gold-500/20 text-gold-300 border-gold-500/30", tooltip: t("evidenceStrength.moderate.tooltip") },
    weak: { label: t("evidenceStrength.weak.label"), className: "bg-rose-500/20 text-rose-300 border-rose-500/30", tooltip: t("evidenceStrength.weak.tooltip") },
  };
  const { label, className, tooltip } = config[strength];
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${className} ${showTooltip ? 'cursor-help' : ''}`}
      title={showTooltip ? tooltip : undefined}
    >
      <Zap size={10} />
      {label}
    </span>
  );
}

function SignalBadge({ label }: { label: string }) {
  return (
    <span className="rounded-full bg-brand-500/10 border border-brand-500/20 px-2 py-0.5 text-xs text-brand-300">
      {label}
    </span>
  );
}

function GesicaSignalsPanel({ summary }: { summary: EvidenceSummaryResponse }) {
  const { t } = useI18n();
  const s = summary.gesicaSignals;
  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="font-medium text-white flex items-center gap-2">
          <Zap size={14} className="text-brand-400" />
          Signaux EMS
        </h3>
        <EvidenceStrengthBadge strength={s.evidenceStrength} showTooltip={true} />
      </div>

      {s.forecastHorizon && (
        <div className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-forest-300">
          <span className="text-forest-400">{t("search.forecastHorizon")}</span>{" "}
          <span className="font-mono text-brand-300">{s.forecastHorizon}</span>
        </div>
      )}

      {s.demandSignals.length > 0 && (
        <div>
          <p className="mb-1 text-xs text-forest-400">Signaux de demande</p>
          <div className="flex flex-wrap gap-1">
            {s.demandSignals.slice(0, 8).map((sig) => (
              <SignalBadge key={sig} label={sig} />
            ))}
          </div>
        </div>
      )}

      {s.scenarioTags.length > 0 && (
        <div>
          <p className="mb-1 text-xs text-forest-400">{t("search.scenariosDetected")}</p>
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
          <p className="mb-1 text-xs text-forest-400">{t("search.reportedMetrics")}</p>
          <div className="flex flex-wrap gap-1">
            {s.reportedMetrics.map((m) => (
              <span key={m} className="rounded-full bg-forest-700/60 border border-white/10 px-2 py-0.5 text-xs text-forest-300 font-mono uppercase">
                {m}
              </span>
            ))}
          </div>
        </div>
      )}

      {s.crossBorder && (
        <div className="rounded-xl border border-gold-500/20 bg-gold-500/5 px-3 py-2 text-xs text-gold-300">
          Pertinence transfrontalière détectée (France / Suisse)
        </div>
      )}
    </section>
  );
}

function ForceIndexButton({ onDone }: { onDone?: () => void }) {
  const { t } = useI18n();
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  if (!hasApiKey()) return null;
  const run = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const r = await embedPending();
      setMsg(
        r.cooldown
          ? t("stats.forceIndexCooldown")
          : `${t("stats.forceIndexDone")}: ${r.embedded}${typeof r.remaining === "number" ? ` · ${r.remaining.toLocaleString()} ${t("stats.forceIndexRemaining")}` : ""}`
      );
      onDone?.();
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="mt-1.5 flex flex-wrap items-center gap-2">
      <button
        onClick={run}
        disabled={busy}
        className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] text-white/70 hover:bg-white/10 disabled:opacity-50"
      >
        <RefreshCw size={11} className={busy ? "animate-spin" : ""} />
        {busy ? t("stats.forceIndexing") : t("stats.forceIndex")}
      </button>
      {msg && <span className="text-[11px] text-forest-400">{msg}</span>}
    </div>
  );
}

function CorpusMaintenancePanel({ onRefresh }: { onRefresh?: () => void }) {
  const { t } = useI18n();
  const [busy, setBusy] = useState<false | "preview" | "apply">(false);
  const [report, setReport] = useState<CorpusMaintenanceReport | null>(null);
  const [applied, setApplied] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const keyed = hasApiKey();

  const actionable = (r: CorpusMaintenanceReport) =>
    r.duplicates.documents + r.legacy_chunks.junk_to_delete +
    r.legacy_chunks.redundant_to_delete + r.legacy_chunks.unique_to_reclassify;

  async function run(dryRun: boolean) {
    setBusy(dryRun ? "preview" : "apply");
    setError(null);
    try {
      const r = await corpusMaintenance(dryRun);
      setReport(r);
      setApplied(!dryRun);
      if (!dryRun) onRefresh?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mt-5 border-t border-white/5 pt-4">
      <h3 className="mb-1 text-sm font-medium text-forest-300 flex items-center gap-1.5">
        <Wrench size={13} className="text-brand-400" />
        {t("stats.maint.title")}
      </h3>
      <p className="mb-3 text-[11px] text-forest-400">{t("stats.maint.subtitle")}</p>

      {!keyed ? (
        <p className="text-[11px] text-gold-300/80 flex items-center gap-1.5"><Lock size={12} /> {t("stats.maint.needKey")}</p>
      ) : (
        <>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => run(true)}
              disabled={busy !== false}
              className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-white/80 hover:bg-white/10 disabled:opacity-50"
            >
              <RefreshCw size={12} className={busy === "preview" ? "animate-spin" : ""} />
              {busy === "preview" ? t("stats.maint.previewing") : t("stats.maint.preview")}
            </button>
            {report && !applied && actionable(report) > 0 && (
              <button
                onClick={() => { if (window.confirm(t("stats.maint.confirm"))) run(false); }}
                disabled={busy !== false}
                className="inline-flex items-center gap-1.5 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-1.5 text-xs text-red-200 hover:bg-red-500/20 disabled:opacity-50"
              >
                <Trash2 size={12} className={busy === "apply" ? "animate-spin" : ""} />
                {busy === "apply" ? t("stats.maint.applying") : t("stats.maint.apply")}
              </button>
            )}
          </div>

          {error && (
            <p className="mt-2 text-[11px] text-red-300 flex items-center gap-1.5"><AlertTriangle size={12} /> {t("stats.maint.error")}: {error}</p>
          )}

          {report && (
            <div className="mt-3 rounded-xl border border-white/10 bg-forest-900/40 p-3 text-[11px] text-forest-300 space-y-1">
              {applied ? (
                <p className="text-brand-300 font-medium">{t("stats.maint.done")}</p>
              ) : actionable(report) === 0 ? (
                <p className="text-forest-300">{t("stats.maint.nothingToDo")}</p>
              ) : null}

              {applied ? (
                <ul className="space-y-0.5">
                  {typeof report.duplicates.deleted_documents === "number" && report.duplicates.deleted_documents > 0 && (
                    <li><span className="font-mono text-white/80">{report.duplicates.deleted_documents}</span> {t("stats.maint.deletedDocs")} · <span className="font-mono text-white/60">{report.duplicates.chunks_cascade}</span> {t("stats.maint.dupChunks")} · <span className="font-mono text-white/60">{report.duplicates.article_scenarios}</span> {t("stats.maint.dupArs")}</li>
                  )}
                  {typeof report.legacy_chunks.deleted_redundant === "number" && report.legacy_chunks.deleted_redundant > 0 && (
                    <li><span className="font-mono text-white/80">{report.legacy_chunks.deleted_redundant}</span> {t("stats.maint.deletedRedundant")}</li>
                  )}
                  {typeof report.legacy_chunks.reclassified === "number" && report.legacy_chunks.reclassified > 0 && (
                    <li><span className="font-mono text-white/80">{report.legacy_chunks.reclassified}</span> {t("stats.maint.reclassified")}</li>
                  )}
                  {typeof report.legacy_chunks.deleted_junk === "number" && report.legacy_chunks.deleted_junk > 0 && (
                    <li><span className="font-mono text-white/80">{report.legacy_chunks.deleted_junk}</span> {t("stats.maint.deletedJunk")}</li>
                  )}
                </ul>
              ) : actionable(report) > 0 ? (
                <ul className="space-y-0.5">
                  {report.duplicates.documents > 0 && (
                    <li><span className="font-mono text-white/80">{report.duplicates.documents}</span> {t("stats.maint.dupDocs")} · <span className="font-mono text-white/60">{report.duplicates.chunks_cascade}</span> {t("stats.maint.dupChunks")} · <span className="font-mono text-white/60">{report.duplicates.article_scenarios}</span> {t("stats.maint.dupArs")}</li>
                  )}
                  {report.legacy_chunks.redundant_to_delete > 0 && (
                    <li><span className="font-mono text-white/80">{report.legacy_chunks.redundant_to_delete}</span> {t("stats.maint.redundantLabel")}</li>
                  )}
                  {report.legacy_chunks.unique_to_reclassify > 0 && (
                    <li><span className="font-mono text-white/80">{report.legacy_chunks.unique_to_reclassify}</span> {t("stats.maint.reclassifyLabel")}</li>
                  )}
                  {report.legacy_chunks.junk_to_delete > 0 && (
                    <li><span className="font-mono text-white/80">{report.legacy_chunks.junk_to_delete}</span> {t("stats.maint.junkLabel")}</li>
                  )}
                </ul>
              ) : null}

              {report.legacy_chunks.breakdown.length > 0 && (
                <p className="pt-1 mt-1 border-t border-white/5 text-forest-400 break-words">
                  {t("stats.maint.breakdownTitle")}: {report.legacy_chunks.breakdown.map(b => `${b.chunk_type} (${b.count})`).join(", ")}
                </p>
              )}

              {applied && report.backups.length > 0 && (
                <p className="pt-1 mt-1 border-t border-white/5 text-forest-400 break-words">
                  {t("stats.maint.backups")}: <span className="font-mono">{report.backups.join(", ")}</span>
                </p>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function StatsView({ corpusStats, fulltextStats, scenarios, statsByYear, onRefresh }: { corpusStats: CorpusStats | null; fulltextStats: FulltextStats | null; scenarios?: GesicaScenario[]; statsByYear?: CorpusStatsByYear | null; onRefresh?: () => void }) {
  const { t } = useI18n();
  if (!corpusStats) {
    return <div className="text-sm text-forest-400">{t("stats.loading")}</div>;
  }

  return (
    <div className="space-y-6">
      {corpusStats && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold text-white">
            <BarChart2 size={18} className="text-brand-400" />
            {t("stats.corpus")}
          </h2>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-4 text-center">
              <p className="text-2xl font-bold text-brand-300">{corpusStats.totalDocuments.toLocaleString()}</p>
              <p className="mt-1 text-xs text-forest-400">{t("stats.documents")}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-4 text-center">
              <p className="text-2xl font-bold text-brand-300">{corpusStats.totalChunks.toLocaleString()}</p>
              <p className="mt-1 text-xs text-forest-400">{t("stats.chunks")}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-4 text-center">
              <p className="text-2xl font-bold text-brand-300">{(scenarios ?? []).filter(s => !s.hidden).length}</p>
              <p className="mt-1 text-xs text-forest-400">{t("stats.scenarios")}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-4 text-center">
              <p className="text-2xl font-bold text-brand-300">{fulltextStats?.by_source?.length ?? Object.keys(corpusStats.bySource).length}</p>
              <p className="mt-1 text-xs text-forest-400">{t("stats.sources")}</p>
            </div>
          </div>

          {/* Couverture texte intégral (fusion de l'ancien panneau "Couverture textuelle") */}
          {fulltextStats && (
            <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-forest-400">
              <span>{t("stats.fulltextLabel")} <span className="text-blue-300 font-semibold">{fulltextStats.corpus.docs_with_fulltext.toLocaleString()}</span> / {corpusStats.totalDocuments.toLocaleString()} <span className="text-white/40">({fulltextStats.corpus.fulltext_coverage_pct.toFixed(1)}%)</span></span>
              <span>{t("stats.abstractOnlyLabel")} <span className="text-white/70 font-semibold">{fulltextStats.corpus.docs_abstract_only.toLocaleString()}</span></span>
              {typeof fulltextStats.corpus.duplicates === "number" && fulltextStats.corpus.duplicates > 0 && (
                <span>{t("stats.duplicatesLabel")} <span className="text-white/70 font-semibold">{fulltextStats.corpus.duplicates.toLocaleString()}</span></span>
              )}
            </div>
          )}

          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            {/* Détail des chunks (remplace "Par scénario" — redondant avec la heatmap) */}
            <div>
              <h3 className="mb-3 text-sm font-medium text-forest-300 flex items-center gap-1.5">
                <BarChart2 size={13} className="text-brand-400" />
                {t("stats.chunkDetail")}
              </h3>
              {fulltextStats?.chunks ? (() => {
                const ch = fulltextStats.chunks!;
                const emb = fulltextStats.embeddings;
                const tot = ch.total || 1;
                const rows = [
                  { label: t("stats.chunkFulltext"), value: ch.fulltext, cls: 'from-blue-500 to-blue-400' },
                  { label: t("stats.chunkAbstract"), value: ch.abstract, cls: 'from-gold-500 to-gold-400' },
                  ...(ch.other > 0 ? [{ label: t("stats.chunkOther"), value: ch.other, cls: 'from-forest-600 to-forest-500' }] : []),
                ];
                return (
                  <div className="space-y-2">
                    {rows.map(r => (
                      <div key={r.label}>
                        <div className="flex items-center justify-between text-xs mb-0.5">
                          <span className="text-white/70">{r.label}</span>
                          <span className="font-mono text-white/60">{r.value.toLocaleString()} <span className="text-white/30">({Math.round(r.value / tot * 100)}%)</span></span>
                        </div>
                        <div className="h-1.5 w-full rounded-full bg-white/5 overflow-hidden">
                          <div className={`h-full rounded-full bg-gradient-to-r ${r.cls}`} style={{ width: `${Math.round(r.value / tot * 100)}%` }} />
                        </div>
                      </div>
                    ))}
                    <p className="pt-1.5 mt-1 text-[11px] text-forest-400 border-t border-white/5">
                      <span className="text-white/70 font-semibold">{ch.total.toLocaleString()}</span> {t("stats.chunksTotalSuffix")}
                      {emb && <> · <span className="text-blue-300">{emb.chunks_with_embedding.toLocaleString()}</span> / {ch.total.toLocaleString()} {t("stats.indexedSuffix")} ({(Math.floor(emb.chunks_with_embedding / (ch.total || 1) * 1000) / 10).toFixed(1)}%)</>}
                      {/* Reliquat non embeddé volontairement (résumés de docs à texte intégral,
                          couverts par leurs sections) — affiché pour que total = indexés + couverts + en attente. */}
                      {(() => {
                        const covered = emb ? Math.max(0, ch.total - emb.chunks_with_embedding - (emb.chunks_pending ?? 0)) : 0;
                        return covered > 0 ? <> · <span className="text-forest-300">{covered.toLocaleString()}</span> {t("stats.coveredByFulltext")}</> : null;
                      })()}
                      {emb && typeof emb.chunks_pending === "number" && emb.chunks_pending > 0 && <> · <span className="text-gold-300">{emb.chunks_pending.toLocaleString()}</span> {t("stats.pendingEmbedding")}</>}
                    </p>
                    {emb && typeof emb.chunks_pending === "number" && emb.chunks_pending > 0 && (
                      <ForceIndexButton onDone={onRefresh} />
                    )}
                  </div>
                );
              })() : (
                <p className="text-xs text-forest-400">{t("common.loadingEllipsis")}</p>
              )}
            </div>
            {/* Par source */}
            <div>
              <h3 className="mb-3 text-sm font-medium text-forest-300 flex items-center gap-1.5">
                <BookOpen size={13} className="text-brand-400" />
                {t("stats.bySource")}
              </h3>
              <div className="space-y-1.5">
                {(() => {
                  // Fusion : par source, le texte intégral (bleu) sur le total (or).
                  // Le compte "texte intégral / total" reprend l'ancien panneau
                  // « Full Text par source ».
                  const rows = (fulltextStats?.by_source && fulltextStats.by_source.length > 0)
                    ? fulltextStats.by_source.map(s => ({ source: s.source, total: s.total, ft: s.with_fulltext }))
                    : Object.entries(corpusStats.bySource).map(([source, total]) => ({ source, total: total as number, ft: 0 }));
                  rows.sort((a, b) => b.total - a.total);
                  const maxVal = Math.max(1, ...rows.map(r => r.total));
                  return rows.map(r => (
                    <div key={r.source}>
                      <div className="flex items-center justify-between text-xs mb-0.5">
                        <span className="text-white/70 capitalize">{r.source}</span>
                        <span className="font-mono"><span className="text-blue-300">{r.ft.toLocaleString()}</span><span className="text-white/30"> / {r.total.toLocaleString()}</span></span>
                      </div>
                      <div className="h-1.5 w-full rounded-full bg-white/5 overflow-hidden flex">
                        <div className="h-full bg-blue-400 transition-all" style={{ width: `${Math.round((r.ft / maxVal) * 100)}%` }} title={t("stats.fulltextBarTitle")} />
                        <div className="h-full bg-gradient-to-r from-gold-500 to-gold-400 transition-all" style={{ width: `${Math.round((Math.max(0, r.total - r.ft) / maxVal) * 100)}%` }} title={t("stats.abstractBarTitle")} />
                      </div>
                    </div>
                  ));
                })()}
              </div>
            </div>
          </div>

          {/* Maintenance (admin) — purge doublons + normalisation des chunks « Autres » */}
          <CorpusMaintenancePanel onRefresh={onRefresh} />

          {/* Enrichissement LLM — global, corpus-wide (déplacé depuis l'onglet scénario :
              l'enrichissement est automatique et à l'échelle du corpus, pas par scénario). */}
          <div className="mt-5 border-t border-white/5 pt-4">
            <EnrichmentSection />
          </div>

          {/* Évolution temporelle — intégrée au bas du panneau Corpus */}
          {statsByYear && Object.keys(statsByYear.byYear).length > 0 && (
            <div className="mt-5 border-t border-white/5 pt-4">
              <h3 className="mb-3 text-sm font-medium text-forest-300 flex items-center gap-1.5">
                <BarChart2 size={13} className="text-brand-400" />
                {(() => {
                  const yrs = Object.keys(statsByYear.byYear).map((y) => parseInt(y)).filter((y) => !isNaN(y));
                  const lo = yrs.length ? Math.min(...yrs) : new Date().getFullYear();
                  const hi = yrs.length ? Math.max(...yrs) : new Date().getFullYear();
                  return `${t("stats.timeEvolution")} (${lo}–${hi})`;
                })()}
              </h3>
              {(() => {
                const counts: Record<number, number> = {};
                Object.entries(statsByYear.byYear).forEach(([y, c]) => {
                  const yr = parseInt(y);
                  if (!isNaN(yr)) counts[yr] = (counts[yr] ?? 0) + (c as number);
                });
                const years = Object.keys(counts).map(Number).sort((a, b) => a - b);
                if (years.length === 0) return null;
                const minY = years[0], maxY = years[years.length - 1];
                const maxVal = Math.max(...Object.values(counts));
                const peakYear = years.reduce((p, y) => (counts[y] > counts[p] ? y : p), years[0]);
                const allYears: number[] = [];
                for (let y = minY; y <= maxY; y++) allYears.push(y);
                return (
                  <>
                    <div className="flex items-end gap-px h-40 w-full">
                      {allYears.map((y) => {
                        const c = counts[y] ?? 0;
                        const pct = maxVal ? (c / maxVal) * 100 : 0;
                        return (
                          <div
                            key={y}
                            title={`${y} : ${c.toLocaleString()} ${c > 1 ? t("stats.articles") : t("stats.article")}`}
                            className={`flex-1 rounded-t transition-all hover:opacity-80 ${y >= 2020 ? 'bg-gradient-to-t from-brand-600 to-brand-400' : 'bg-forest-500/70'}`}
                            style={{ height: `${c > 0 ? Math.max(2, pct) : 0}%` }}
                          />
                        );
                      })}
                    </div>
                    <div className="mt-1.5 flex justify-between text-[10px] font-mono text-forest-400">
                      <span>{minY}</span>
                      <span>{Math.round((minY + maxY) / 2)}</span>
                      <span>{maxY}</span>
                    </div>
                    <p className="mt-2 text-xs text-forest-400 italic">
                      {t("stats.peak")} <span className="not-italic font-semibold text-brand-300">{peakYear}</span> ({counts[peakYear].toLocaleString()} {t("stats.articles")}) {t("stats.peakSuffix")}
                    </p>
                  </>
                );
              })()}
            </div>
          )}
        </div>
      )}

      {/* Heatmap scénario × source */}
      {statsByYear && Object.keys(statsByYear.heatmapScenarioSource).length > 0 && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold text-white">
            <Activity size={18} className="text-brand-400" />
            {t("stats.heatmapTitle")}
          </h2>
          {(() => {
            // Nouvelle forme : { scenario_id: { name, sources: { src: {total, fulltext} } } }.
            // Clé par id (plus de fusion de scénarios homonymes) + texte intégral par cellule.
            const entries = Object.entries(statsByYear.heatmapScenarioSource);
            const allSources = Array.from(
              new Set(entries.flatMap(([, v]) => Object.keys(v.sources)))
            ).sort();
            const rows = entries.map(([sid, v]) => {
              const total = Object.values(v.sources).reduce((s, c) => s + c.total, 0);
              const ftTotal = Object.values(v.sources).reduce((s, c) => s + c.fulltext, 0);
              return { sid, name: v.name, sources: v.sources, total, ftTotal };
            }).sort((a, b) => b.total - a.total);
            const globalMax = Math.max(1, ...rows.flatMap(e => Object.values(e.sources).map(c => c.total)));
            return (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr>
                      <th className="text-left text-forest-400 font-normal pb-2 pr-3 min-w-[160px]">{t("stats.heatmapScenario")} ({rows.length})</th>
                      {allSources.map(src => (
                        <th key={src} className="text-center text-forest-400 font-normal pb-2 px-1 capitalize">{src}</th>
                      ))}
                      <th className="text-right text-forest-400 font-normal pb-2 pl-2">{t("stats.heatmapTotal")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map(e => (
                      <tr key={e.sid} className="border-t border-white/5">
                        <td className="py-1.5 pr-3 text-white/70 truncate max-w-[240px]" title={e.name}>{e.name}</td>
                        {allSources.map(src => {
                          const cell = e.sources[src];
                          const val = cell?.total ?? 0;
                          const ft = cell?.fulltext ?? 0;
                          const intensity = globalMax > 0 ? val / globalMax : 0;
                          const bg = val > 0 ? `rgba(52, 211, 153, ${Math.max(0.08, intensity * 0.85)})` : 'transparent';
                          return (
                            <td
                              key={src}
                              className="text-center py-1 px-1 font-mono rounded leading-tight align-middle"
                              style={{ backgroundColor: bg, color: val > 0 ? '#d1fae5' : '#374151' }}
                              title={val > 0 ? `${val} ${t("stats.heatmapCellTooltip")} ${ft} ${t("stats.heatmapFulltextTooltip")}` : undefined}
                            >
                              {val > 0 ? (
                                <>
                                  <div>{val}</div>
                                  {ft > 0 && <div className="text-[9px] text-blue-200/90">{ft} ft</div>}
                                </>
                              ) : '—'}
                            </td>
                          );
                        })}
                        <td className="text-right py-1.5 pl-2 font-mono text-brand-300 font-semibold align-middle">
                          {e.total.toLocaleString()}
                          {e.ftTotal > 0 && <span className="block text-[9px] font-normal text-blue-200/80">{e.ftTotal} ft</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="mt-2 text-[10px] text-forest-400 italic">{t("stats.heatmapLegend")}</p>
              </div>
            );
          })()}
        </div>
      )}

      {/* Progression du Screening par Scénario */}
      {scenarios && scenarios.length > 0 && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl">
          <h3 className="mb-4 text-base font-semibold text-white flex items-center gap-2">
            <Activity size={16} className="text-brand-400" />
            {t("stats.screeningProgress")}
          </h3>
          <div className="space-y-2">
            {scenarios.filter(s => !s.hidden && s.articleCount > 0).map(s => {
              const total = s.articleCount ?? 0;
              const included = s.included_count ?? 0;
              const excluded = s.excluded_count ?? 0;
              const pending = Math.max(0, total - included - excluded);
              const pctIncluded = total > 0 ? Math.round(included / total * 100) : 0;
              const pctExcluded = total > 0 ? Math.round(excluded / total * 100) : 0;
              const pctPending = total > 0 ? Math.round(pending / total * 100) : 0;
              return (
                <div key={s.id} className="rounded-2xl border border-white/5 bg-white/2 p-3">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs font-semibold text-white/80 truncate max-w-[220px]">{s.title}</span>
                    <div className="flex items-center gap-3 text-[10px] shrink-0">
                      <span className="text-brand-300">{included} {t("stats.included")}</span>
                      <span className="text-red-400">{excluded} {t("stats.excluded")}</span>
                      <span className="text-white/35">{pending} {t("stats.pending")}</span>
                      <span className="text-white/50 font-mono">{total} {t("stats.total")}</span>
                    </div>
                  </div>
                  <div className="h-1.5 bg-white/5 rounded-full overflow-hidden flex">
                    <div className="h-full bg-brand-500 transition-all" style={{width:`${pctIncluded}%`}}/>
                    <div className="h-full bg-red-500/60 transition-all" style={{width:`${pctExcluded}%`}}/>
                    <div className="h-full bg-white/10 transition-all" style={{width:`${pctPending}%`}}/>
                  </div>
                  <div className="flex gap-3 mt-1 text-[9px] text-white/30">
                    <span className="flex items-center gap-1"><span className="h-1.5 w-1.5 rounded-full bg-brand-500 inline-block"/>{t("stats.includedPct")} {pctIncluded}%</span>
                    <span className="flex items-center gap-1"><span className="h-1.5 w-1.5 rounded-full bg-red-500/60 inline-block"/>{t("stats.excludedPct")} {pctExcluded}%</span>
                    <span className="flex items-center gap-1"><span className="h-1.5 w-1.5 rounded-full bg-white/10 inline-block"/>{t("stats.pendingPct")} {pctPending}%</span>
                    {s.kappa_score != null && <span className="ml-auto text-white/40">{t("stats.kappa")} <span className="font-mono">{s.kappa_score.toFixed(2)}</span></span>}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// Actions recommandées — composant HOISTÉ (niveau module) : identité stable, pas de
// remontage à chaque rendu. C'est le seul endroit qui porte des hooks, ce qui permet
// à renderScenarioCard d'être une simple fonction inline (cf. correctif double-clic).
function RecommendedActions({ scenario, isUser }: { scenario: GesicaScenario; isUser: boolean }) {
  const { t, lang } = useI18n();
  const [fetchedActions, setFetchedActions] = React.useState<string[] | null>(null);
  const [actionsGenerating, setActionsGenerating] = React.useState(false);
  React.useEffect(() => {
    if (!isUser) return;
    if (scenario.recommendedActions && scenario.recommendedActions.length > 0) return;
    let cancelled = false;
    const tick = (tries: number) => {
      getRecommendedActions(scenario.id).then(r => {
        if (cancelled) return;
        if (r.status === "ready") { setFetchedActions(r.actions); setActionsGenerating(false); }
        else if (r.status === "generating" && tries < 20) { setActionsGenerating(true); setTimeout(() => tick(tries + 1), 4000); }
        else { setFetchedActions([]); setActionsGenerating(false); }
      }).catch(() => { if (!cancelled) { setFetchedActions([]); setActionsGenerating(false); } });
    };
    // Réinitialiser avant de (re)charger : au changement de langue on ne veut pas
    // afficher les anciennes actions (autre langue) le temps de la régénération.
    setFetchedActions(null);
    tick(0);
    return () => { cancelled = true; };
    // `lang` dans les deps : changer la langue de l'UI régénère les actions.
  }, [isUser, scenario.id, lang]);
  const actions = (scenario.recommendedActions && scenario.recommendedActions.length > 0)
    ? scenario.recommendedActions
    : (fetchedActions ?? []);
  if (!(actions.length > 0 || actionsGenerating)) return null;
  return (
    <div>
      <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-forest-400">{t("common.recommendedActions")}</h4>
      {actions.length > 0 ? (
        <ul className="space-y-1.5">
          {actions.map((action, i) => (
            <li key={i} className="flex items-start gap-2 text-sm text-white/80">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-brand-400" />
              {action}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-xs text-forest-400 flex items-center gap-1.5">
          <RefreshCw size={11} className="animate-spin" /> {t("common.generatingRecommendedActions")}
        </p>
      )}
    </div>
  );
}

function ScenariosView({
  scenarios,
  loading,
  error,
  savedSearches = [],
  userScenarios = [],
  onReplaySearch,
  onDeleteSearch,
  onTogglePin,
  onPopulateUserScenario,
  populatingId,
  pipelineStatuses = {},
  folders = [],
  onCreateFolder,
  onDeleteFolder,
  onRenameFolder,
  onAssignFolder,
}: {
  scenarios: GesicaScenario[];
  loading?: boolean;
  error?: string | null;
  savedSearches?: SavedSearch[];
  userScenarios?: UserScenario[];
  onReplaySearch?: (s: SavedSearch) => void;
  onDeleteSearch?: (id: string) => void;
  onTogglePin?: (id: string) => void;
  onPopulateUserScenario?: (id: string) => void;
  populatingId?: string | null;
  pipelineStatuses?: Record<string, import('./lib/api').UserScenarioPipelineStatus>;
  folders?: ScenarioFolder[];
  onCreateFolder?: (name: string, color: string) => Promise<ScenarioFolder>;
  onDeleteFolder?: (folderId: string) => Promise<void>;
  onRenameFolder?: (folderId: string, name: string, color: string) => Promise<void>;
  onAssignFolder?: (scenarioId: string, folderId: string | null) => Promise<void>;
}) {
  const { t } = useI18n();
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [detailScenarioId, setDetailScenarioId] = useState<string | null>(null);
  const [detailInitialTab, setDetailInitialTab] = useState<"model" | undefined>(undefined);
  const [showNewFolderDialog, setShowNewFolderDialog] = useState(false);
  const [newFolderName, setNewFolderName] = useState('');
  const [newFolderColor, setNewFolderColor] = useState('#6366f1');
  const [assigningScenarioId, setAssigningScenarioId] = useState<string | null>(null);
  const [folderError, setFolderError] = useState<string | null>(null);
  const [editingFolderId, setEditingFolderId] = useState<string | null>(null);
  const [editFolderName, setEditFolderName] = useState('');
  const [editFolderColor, setEditFolderColor] = useState('#6366f1');

  // Page détail d'un scénario (GESICA ou utilisateur)
  if (detailScenarioId) {
    return <ScenarioDetailPage scenarioId={detailScenarioId} initialTab={detailInitialTab} onBack={() => { setDetailScenarioId(null); setDetailInitialTab(undefined); }} />;
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16 text-forest-400">
        <RotateCcw size={18} className="mr-2 animate-spin" />
        {t("scenarios.loading")}
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-16 gap-3">
        <div className="rounded-2xl border border-red-500/30 bg-red-500/10 px-6 py-4 text-red-300 max-w-xl text-center">
          <p className="font-semibold mb-1">{t("scenarios.loadErrorTitle")}</p>
          <p className="text-sm text-red-400 font-mono break-all">{error}</p>
          <p className="text-xs text-forest-400 mt-2">{t("scenarios.loadErrorHint1")} <code>/api/gesica/scenarios</code> {t("scenarios.loadErrorHint2")}</p>
        </div>
      </div>
    );
  }

  if (scenarios.length === 0) {
    return (
      <div className="flex items-center justify-center py-16 text-forest-400">
        <RotateCcw size={18} className="mr-2 animate-spin" />
        {t("scenarios.loading")}
      </div>
    );
  }

  // Extraire les clusters uniques

  const clusterColors: Record<string, string> = {
    "Prévention & Risques": "border-violet-500/30 bg-violet-500/10 text-violet-300",
    "Opérations EMS": "border-brand-500/30 bg-brand-500/10 text-brand-300",
    "Triage & Clinique": "border-brand-500/30 bg-brand-500/10 text-brand-300",
    "Soins Centrés Patient": "border-rose-500/30 bg-rose-500/10 text-rose-300",
    "Surveillance & Crise": "border-gold-500/30 bg-gold-500/10 text-gold-300",
    "Systèmes & IA": "border-brand-500/30 bg-brand-500/10 text-brand-300",
  };

  // Rendue comme FONCTION inline (et non composant imbriqué) : ainsi elle n'est pas
  // remontée à chaque rendu de ScenariosView — ce remont permanent faisait « perdre »
  // le 1er clic du bouton dossier (il fallait cliquer deux fois). Les hooks vivent
  // désormais dans le composant hoisté RecommendedActions.
  const renderScenarioCard = (scenario: GesicaScenario) => {
    const isExpanded = expandedId === scenario.id;
    const hasArticles = scenario.articleCount > 0;
    const isUser = (scenario as any).is_user_scenario === true || scenario.cluster === "user";

    return (
      <div key={scenario.id} className={`rounded-3xl border p-5 shadow-xl transition ${
        hasArticles ? "border-white/10 bg-white/5" : "border-white/5 bg-white/2 opacity-60"
      }`}>
        <div
          className="flex items-start gap-3 cursor-pointer"
          onClick={() => setExpandedId(isExpanded ? null : scenario.id)}
        >
          <div className="mt-1 rounded-xl border border-brand-500/20 bg-brand-500/10 p-2 shrink-0">
            <Activity size={14} className="text-brand-400" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="text-base font-semibold text-white">{scenario.title}</h3>
              <span className={`rounded-full border px-2 py-0.5 text-xs font-medium ${
                clusterColors[scenario.cluster] ?? "border-white/10 bg-white/5 text-forest-400"
              }`}>{scenario.cluster}</span>
              {hasArticles ? (
                <span className="rounded-full bg-brand-500/10 border border-brand-500/20 px-2 py-0.5 text-xs text-brand-300 font-mono">
                  {scenario.articleCount} {t("scenarios.articles")}
                </span>
              ) : (
                <span className="rounded-full bg-forest-700/40 border border-white/5 px-2 py-0.5 text-xs text-forest-500">
                  0 {t("scenarios.articles")}
                </span>
              )}
            </div>
            <p className="mt-1 text-sm leading-5 text-forest-400 line-clamp-2">
              {isUser && scenario.query ? `${t("scenarios.savedSearchPrefix")}${scenario.query}` : scenario.description}
            </p>
          </div>
          <div className="shrink-0 flex items-center gap-1.5">
            <button
              onClick={(e) => { e.stopPropagation(); setDetailInitialTab(undefined); setDetailScenarioId(scenario.id); }}
              className="rounded-xl border border-brand-500/20 bg-brand-500/10 px-2.5 py-1 text-[10px] text-brand-300 hover:bg-brand-500/20 transition font-medium"
              title={t("scenarios.detailPageTooltip")}
            >
              {t("scenarios.detailPage")}
            </button>
            {isUser && (
              <>
                {onPopulateUserScenario && (
                  <button type="button" onClick={(e) => { e.stopPropagation(); onPopulateUserScenario(scenario.id); }}
                    disabled={populatingId === scenario.id}
                    className="rounded-xl border border-forest-500/30 px-2 py-1 text-[10px] text-forest-300 hover:bg-forest-500/10 transition disabled:opacity-50"
                    title={t("scenarios.ingestArticlesTooltip")}>
                    {populatingId === scenario.id ? <RotateCcw size={11} className="animate-spin" /> : <Zap size={11} />}
                  </button>
                )}
                {onReplaySearch && (
                  <button type="button" onClick={(e) => { e.stopPropagation(); const saved = savedSearches.find(ss => ss.id === scenario.id); if (saved) onReplaySearch(saved); }}
                    className="rounded-xl border border-white/10 px-2 py-1 text-[10px] text-white/40 hover:text-white hover:bg-white/10 transition" title={t("scenarios.replaySearchTooltip")}>↻</button>
                )}
                {onAssignFolder && (
                  <button type="button" onClick={(e) => { e.stopPropagation(); setAssigningScenarioId(scenario.id); }}
                    className="rounded-xl border border-white/10 px-2 py-1 text-[10px] text-white/30 hover:text-white hover:bg-white/10 transition" title={t("scenarios.assignFolderTooltip")}><FolderOpen size={11} /></button>
                )}
                {onTogglePin && (
                  <button type="button" onClick={(e) => { e.stopPropagation(); onTogglePin(scenario.id); }}
                    className={`rounded-xl border px-2 py-1 text-[10px] transition ${(scenario as any).pinned ? "border-gold-400/30 text-gold-400 hover:bg-gold-500/10" : "border-white/10 text-white/30 hover:text-gold-300 hover:bg-white/10"}`}
                    title={(scenario as any).pinned ? t("scenarios.unpinTooltip") : t("scenarios.pinTooltip")}>★</button>
                )}
                {onDeleteSearch && (
                  <button type="button" onClick={(e) => { e.stopPropagation(); onDeleteSearch(scenario.id); }}
                    className="rounded-xl border border-white/10 px-2 py-1 text-[10px] text-white/30 hover:text-red-400 hover:border-red-500/20 transition" title={t("scenarios.deleteTooltip")}>✕</button>
                )}
              </>
            )}
            {!isUser && onDeleteSearch && (
              <button type="button" onClick={(e) => { e.stopPropagation(); onDeleteSearch(scenario.id); }}
                className="rounded-xl border border-white/10 px-2 py-1 text-[10px] text-white/30 hover:text-red-400 hover:border-red-500/20 transition" title={t("scenarios.deleteTooltip")}>✕</button>
            )}
            {isExpanded ? <ChevronUp size={16} className="text-forest-500" /> : <ChevronDown size={16} className="text-forest-500" />}
          </div>
        </div>
        {/* Assignation à un dossier — INLINE, directement sous CETTE carte (plus de
            carte unique en haut de liste qui obligeait à scroller bien au-dessus). */}
        {assigningScenarioId === scenario.id && (
          <div className="mt-3 rounded-2xl border border-brand-400/30 bg-brand-500/5 p-3 space-y-2">
            <h4 className="text-xs font-semibold text-white">{t("scenarios.placeInFolder")}</h4>
            <div className="flex flex-wrap gap-2">
              <button type="button" onClick={async (e) => { e.stopPropagation(); if (onAssignFolder) await onAssignFolder(scenario.id, null); setAssigningScenarioId(null); }}
                className="rounded-xl border border-white/10 px-3 py-1.5 text-xs text-white/50 hover:text-white hover:bg-white/10 transition">{t("scenarios.noFolder")}</button>
              {folders.map(f => (
                <button key={f.id} type="button" onClick={async (e) => { e.stopPropagation(); if (onAssignFolder) await onAssignFolder(scenario.id, f.id); setAssigningScenarioId(null); }}
                  className="rounded-xl border px-3 py-1.5 text-xs hover:opacity-80 transition"
                  style={{ borderColor: f.color + '60', backgroundColor: f.color + '20', color: f.color }}>{f.name}</button>
              ))}
              <button type="button" onClick={(e) => { e.stopPropagation(); setAssigningScenarioId(null); setFolderError(null); }}
                className="rounded-xl border border-white/10 px-3 py-1.5 text-xs text-white/30 hover:text-white transition">{t("common.cancel")}</button>
            </div>
            {/* Créer un dossier ET y placer ce scénario en un clic */}
            <div className="flex items-center gap-2 pt-2 border-t border-white/10">
              <input type="text" value={newFolderName} onChange={e => setNewFolderName(e.target.value)} onClick={e => e.stopPropagation()}
                placeholder={t("scenarios.newFolderPlaceholder")}
                className="flex-1 rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-white placeholder-white/30 focus:outline-none focus:border-brand-400" />
              <button type="button" disabled={!newFolderName.trim()}
                onClick={async (e) => {
                  e.stopPropagation();
                  if (!newFolderName.trim() || !onCreateFolder) return;
                  try {
                    const nf = await onCreateFolder(newFolderName.trim(), newFolderColor);
                    if (nf && onAssignFolder) await onAssignFolder(scenario.id, nf.id);
                    setNewFolderName(''); setFolderError(null); setAssigningScenarioId(null);
                  } catch (err) {
                    setFolderError(err instanceof Error ? err.message : t("scenarios.folderCreateError"));
                  }
                }}
                className="rounded-xl bg-brand-500/20 border border-brand-500/30 px-3 py-1.5 text-xs text-brand-300 hover:bg-brand-500/30 transition disabled:opacity-40 shrink-0">{t("scenarios.createAndAssign")}</button>
            </div>
            {folderError && <p className="text-[11px] text-rose-300">{folderError}</p>}
          </div>
        )}
        {isUser && pipelineStatuses[scenario.id] && pipelineStatuses[scenario.id].overall_status !== 'done' && (
          <div className="mt-2 flex items-center gap-2 pl-12">
            <RotateCcw size={10} className="text-brand-400 animate-spin shrink-0" />
            <span className="text-xs text-brand-300">
              {pipelineStatuses[scenario.id].overall_status === 'error' ? t("scenarios.pipelineError") : `${t("scenarios.pipelinePrefix")} ${pipelineStatuses[scenario.id].current_step ?? t("scenarios.pipelineInProgress")}…`}
            </span>
          </div>
        )}

        {isExpanded && (
          <div className="mt-4 space-y-4 border-t border-white/10 pt-4">
            {/* Actions recommandées (hooks isolés dans le composant hoisté) */}
            <RecommendedActions scenario={scenario} isUser={isUser} />

            {/* Modèle Prédictif — carte générique (scénarios utilisateur) */}
            {isUser && (
              <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Activity size={14} className="text-brand-400" />
                    <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">{t("scenarios.predictiveModel")}</span>
                  </div>
                  {scenario.model?.has_model && scenario.model.family && (
                    <span className="text-[10px] text-brand-400 bg-brand-500/10 border border-brand-500/20 px-2 py-0.5 rounded-full">
                      {scenario.model.family}{scenario.model.metric && scenario.model.metric_value != null ? ` · ${scenario.model.metric} ${scenario.model.metric_value.toFixed(3)}` : ""}
                    </span>
                  )}
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); setDetailInitialTab("model"); setDetailScenarioId(scenario.id); }}
                  className="w-full flex items-center justify-center gap-1.5 rounded-xl border border-brand-500/30 bg-brand-500/10 text-brand-300 font-semibold py-2 text-xs hover:bg-brand-500/20 transition"
                >
                  <Activity size={12} />
                  {scenario.model?.has_model ? t("scenarios.openModel") : t("scenarios.trainAndPredict")}
                </button>
              </div>
            )}

            {/* Liste d'articles retirée de la carte — disponible sur la page détail */}
            {false && scenario.relevantArticles.length > 0 && (
              <div>
                <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-forest-400">
                  Articles récents ({scenario.articleCount} total, 5 affichés)
                </h4>
                <div className="space-y-2">
                  {scenario.relevantArticles.map((article) => (
                    <div key={article.id} className="rounded-xl border border-white/10 bg-forest-900/40 px-3 py-2">
                      <p className="text-sm font-medium text-white/80 leading-5">{article.title}</p>
                      {article.authors && (
                        <p className="text-xs text-forest-400 mt-0.5 line-clamp-1">Par {article.authors}</p>
                      )}
                      <div className="mt-1.5 flex items-center gap-2 flex-wrap">
                        {article.journal && (
                          <span className="text-xs font-semibold text-forest-300 bg-forest-800 px-1.5 py-0.5 rounded">
                            {article.journal}
                          </span>
                        )}
                        <span className="text-xs text-forest-400">{article.source}</span>
                        {article.year && <span className="text-xs text-forest-500">{article.year}</span>}
                        
                        {article.study_design && (
                          <span className="text-xs text-gold-400 bg-gold-500/10 px-1.5 py-0.5 rounded font-mono">
                            {article.study_design}
                          </span>
                        )}
                        
                        {/* Badge couverture textuelle */}
                        <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded border ${
                          article.has_fulltext
                            ? 'text-brand-400 bg-brand-500/10 border-brand-500/20'
                            : 'text-forest-500 bg-forest-800/50 border-white/5'
                        }`} title={article.has_fulltext ? 'Texte intégral indexé' : 'Titre + résumé uniquement'}>
                          {article.has_fulltext ? 'Full Text' : 'Abstract'}
                        </span>

                        {article.doi && (
                          <a
                            href={`https://doi.org/${article.doi}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-[10px] text-forest-400 hover:text-brand-400 font-mono"
                            title={`DOI: ${article.doi}`}
                          >
                            DOI
                          </a>
                        )}

                        {article.open_access && (
                          <span className="text-xs text-brand-400 bg-brand-500/10 px-1.5 py-0.5 rounded">
                            OA
                          </span>
                        )}
                        
                        {article.citation_count !== null && article.citation_count > 0 && (
                          <span className="text-xs text-forest-400">
                            {article.citation_count} citation{article.citation_count > 1 ? 's' : ''}
                          </span>
                        )}

                        {article.url && (
                          <a
                            href={article.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs text-brand-400 hover:underline flex items-center gap-1 ml-auto"
                          >
                            Lien <ExternalLink size={10} />
                          </a>
                        )}
                      </div>
                      {article.keywords && (
                        <div className="mt-1 flex items-center gap-1 flex-wrap">
                          {article.keywords.split(',').slice(0, 4).map((kw, idx) => (
                            <span key={idx} className="text-[10px] text-forest-500 bg-forest-800/50 px-1 rounded">
                              #{kw.trim()}
                            </span>
                          ))}
                        </div>
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
  };

  // Construire les dossiers GESICA (un par cluster)
  const gesicaClusters = Array.from(new Set(scenarios.map(s => s.cluster))).sort();
  const totalArticles = scenarios.reduce((a, s) => a + s.articleCount, 0)
    + userScenarios.reduce((a, s) => a + (s.articleCount ?? 0), 0);
  const totalScenarios = scenarios.length + userScenarios.length;

  return (
    <div className="space-y-6">
      {/* En-tête unifié */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity size={20} className="text-brand-400" />
          <div>
            <h2 className="text-xl font-semibold text-white">{t("scenarios.allScenarios")}</h2>
            <p className="text-xs text-forest-400 mt-0.5">
              {totalScenarios} {t("scenarios.scenariosCount")} · {totalArticles.toLocaleString()} {t("scenarios.articles")}
            </p>
          </div>
        </div>
        {onCreateFolder && (
          <button
            type="button"
            onClick={() => setShowNewFolderDialog(true)}
            className="shrink-0 flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-white/60 hover:text-white hover:bg-white/10 transition"
          >
            <FolderOpen size={13} />
            {t("scenarios.newFolder")}
          </button>
        )}
      </div>

      {/* ── Dossiers GESICA (un par cluster) ── */}
      {gesicaClusters.map(cluster => {
        const clusterScenarios = scenarios.filter(s => s.cluster === cluster);
        return (
          <div key={cluster} className="rounded-2xl border border-brand-500/20 bg-brand-500/5 p-4 space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold flex items-center gap-1.5 text-brand-300">
                <FolderOpen size={14} />
                {cluster} <span className="text-xs font-normal opacity-60">({clusterScenarios.length})</span>
              </h3>
              <span className="rounded-full border border-brand-500/20 bg-brand-500/10 px-2 py-0.5 text-[10px] text-brand-400 font-medium">
                <RefreshCw size={8} className="inline mr-1" />{t("scenarios.livingReview")}
              </span>
            </div>
            {clusterScenarios.map(s => renderScenarioCard(s))}
          </div>
        );
      })}

      {/* ── Dossiers personnels + scénarios utilisateur ── */}
      <div className="space-y-4">

          {/* Dialog création dossier */}
          {showNewFolderDialog && (
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4 space-y-3">
              <h4 className="text-sm font-semibold text-white">{t("scenarios.newFolder")}</h4>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={newFolderName}
                  onChange={e => setNewFolderName(e.target.value)}
                  placeholder={t("scenarios.newFolderName")}
                  className="flex-1 rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-sm text-white placeholder-white/30 focus:outline-none focus:border-brand-400"
                />
                <input
                  type="color"
                  value={newFolderColor}
                  onChange={e => setNewFolderColor(e.target.value)}
                  className="w-10 h-9 rounded-xl border border-white/10 bg-white/5 cursor-pointer"
                  title={t("scenarios.folderColorTitle")}
                />
              </div>
              {folderError && <p className="text-[11px] text-rose-300">{folderError}</p>}
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={async () => {
                    if (!newFolderName.trim() || !onCreateFolder) return;
                    try {
                      await onCreateFolder(newFolderName.trim(), newFolderColor);
                      setNewFolderName('');
                      setNewFolderColor('#6366f1');
                      setFolderError(null);
                      setShowNewFolderDialog(false);
                    } catch (e) {
                      setFolderError(e instanceof Error ? e.message : t("scenarios.folderCreateError"));
                    }
                  }}
                  className="rounded-xl bg-brand-500/20 border border-brand-500/30 px-3 py-1.5 text-xs text-brand-300 hover:bg-brand-500/30 transition"
                >
                  {t("common.create")}
                </button>
                <button type="button" onClick={() => { setShowNewFolderDialog(false); setFolderError(null); }} className="rounded-xl border border-white/10 px-3 py-1.5 text-xs text-white/40 hover:text-white transition">
                  {t("common.cancel")}
                </button>
              </div>
            </div>
          )}

          {/* Dossiers */}
          {folders.map(folder => {
            const folderScenarios = userScenarios.filter(s => (s as any).folder_id === folder.id);
            return (
              <div key={folder.id} className="rounded-2xl border p-4 space-y-2" style={{ borderColor: folder.color + '40', backgroundColor: folder.color + '08' }}>
                <div className="flex items-center justify-between">
                  {editingFolderId === folder.id ? (
                    <div className="flex gap-2 flex-1">
                      <input type="text" value={editFolderName} onChange={e => setEditFolderName(e.target.value)} className="flex-1 rounded-xl border border-white/10 bg-white/5 px-2 py-1 text-sm text-white focus:outline-none focus:border-brand-400" />
                      <input type="color" value={editFolderColor} onChange={e => setEditFolderColor(e.target.value)} className="w-8 h-8 rounded border border-white/10 bg-white/5 cursor-pointer" />
                      <button type="button" onClick={async () => { if (onRenameFolder) await onRenameFolder(folder.id, editFolderName, editFolderColor); setEditingFolderId(null); }} className="rounded-xl bg-brand-500/20 border border-brand-500/30 px-2 py-1 text-xs text-brand-300">OK</button>
                      <button type="button" onClick={() => setEditingFolderId(null)} className="rounded-xl border border-white/10 px-2 py-1 text-xs text-white/30">✕</button>
                    </div>
                  ) : (
                    <h3 className="text-sm font-semibold flex items-center gap-1.5" style={{ color: folder.color }}>
                      <FolderOpen size={14} />
                      {folder.name} <span className="text-xs font-normal opacity-60">({folderScenarios.length})</span>
                    </h3>
                  )}
                  {editingFolderId !== folder.id && (
                    <div className="flex gap-1">
                      <button type="button" onClick={() => { setEditingFolderId(folder.id); setEditFolderName(folder.name); setEditFolderColor(folder.color); }} className="rounded-lg border border-white/10 px-2 py-1 text-xs text-white/30 hover:text-white transition" title={t("scenarios.renameTooltip")}>✏</button>
                      <button type="button" onClick={() => onDeleteFolder && onDeleteFolder(folder.id)} className="rounded-lg border border-white/10 px-2 py-1 text-xs text-white/30 hover:text-red-400 transition" title={t("scenarios.deleteTooltip")}>✕</button>
                    </div>
                  )}
                </div>
                {folderScenarios.length === 0 && (
                  <p className="text-xs text-white/30 italic">{t("scenarios.noScenarioInFolder")}</p>
                )}
                {folderScenarios.map(s => (
                  renderScenarioCard(s)
                ))}
              </div>
            );
          })}

          {/* Épinglés (hors dossier) */}
          {userScenarios.filter(s => s.pinned && !(s as any).folder_id).length > 0 && (
            <div className="space-y-2">
              <h3 className="text-xs font-semibold text-gold-400 uppercase tracking-widest">{t("scenarios.pinned")}</h3>
              {userScenarios.filter(s => s.pinned && !(s as any).folder_id).map(s => (
                renderScenarioCard(s)
              ))}
            </div>
          )}

          {/* Non épinglés (hors dossier) */}
          {userScenarios.filter(s => !s.pinned && !(s as any).folder_id).length > 0 && (
            <div className="space-y-2">
              <h3 className="text-xs font-semibold text-white/40 uppercase tracking-widest">{t("scenarios.recentSearches")}</h3>
              {userScenarios.filter(s => !s.pinned && !(s as any).folder_id).map(s => (
                renderScenarioCard(s)
              ))}
            </div>
          )}
        </div>
    </div>
  );
}

export default function App() {
  const [projectContext, setProjectContext] = useState<ProjectContext>("literev");
  const { t, lang, setLang } = useI18n();
  const [activeTab, setActiveTab] = useState<AppTab>("search");
  // Clé d'écriture admin (X-API-Key). Saisie une fois, stockée en localStorage côté
  // navigateur — jamais dans le bundle public (cf. authHeaders dans lib/api).
  const [apiKeySet, setApiKeySet] = useState<boolean>(() => hasApiKey());
  const handleManageApiKey = () => {
    if (apiKeySet) {
      if (window.confirm(t("header.removeKeyConfirm"))) {
        clearApiKey();
        setApiKeySet(false);
      }
      return;
    }
    const entered = window.prompt(t("header.setKeyPrompt"));
    if (entered === null) return;
    setApiKey(entered);
    setApiKeySet(hasApiKey());
  };
  const [mode, setMode] = useState<SearchMode>("boolean");
  const [includeLive, setIncludeLive] = useState(false);
  const [query, setQuery] = useState("");
  // Recherche multi-sous-requêtes (avancé) : requêtes ADDITIONNELLES combinées à
  // la requête principale ci-dessus. Vide = recherche mono-requête classique.
  const [extraQueries, setExtraQueries] = useState<SubQuery[]>([]);
  const [combinator, setCombinator] = useState<"union" | "intersection">("union");
  const [filters, setFilters] = useState<SearchFilters>({
    projectContext: "literev",
  });
  const [diseaseSearch, setDiseaseSearch] = useState<string>("");
  const [yearRange, setYearRange] = useState<[number, number]>([
    1990,
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
  const [fulltextStats, setFulltextStats] = useState<FulltextStats | null>(null);
  const [corpusStatsByYear, setCorpusStatsByYear] = useState<CorpusStatsByYear | null>(null);
  const [gesicaScenarios, setGesicaScenarios] = useState<GesicaScenario[]>([]);
  const [loadingScenarios, setLoadingScenarios] = useState(false);
  const [scenariosError, setScenariosError] = useState<string | null>(null);
  const [savedSearches, setSavedSearches] = useState<SavedSearch[]>([]);
  const [userScenarios, setUserScenarios] = useState<UserScenario[]>([]);
  const [saveSearchName, setSaveSearchName] = useState("");
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [populatingId, setPopulatingId] = useState<string | null>(null);
  const [pipelineStatuses, setPipelineStatuses] = useState<Record<string, UserScenarioPipelineStatus>>({});
  const pipelinePollRef = useRef<Record<string, ReturnType<typeof setInterval>>>({});
  const [searchSourceBreakdown, setSearchSourceBreakdown] = useState<Record<string, number> | null>(null);
  const [searchTotalMatching, setSearchTotalMatching] = useState<number | null>(null);
  const [searchFulltextDocs, setSearchFulltextDocs] = useState<number | null>(null);
  const [searchAbstractDocs, setSearchAbstractDocs] = useState<number | null>(null);
  const [searchLiveNewCount, setSearchLiveNewCount] = useState<number | null>(null);
  const [searchScoreType, setSearchScoreType] = useState<string | null>(null);
  const [searchScoreLabel, setSearchScoreLabel] = useState<string | null>(null);
  // Requête booléenne (mode booléen) : affichée à l'utilisateur et utilisée comme
  // base du corpus (même requête → compteur == taille du corpus).
  const [booleanStrategy, setBooleanStrategy] = useState<SearchStrategy | null>(null);
  const [translatingQuery, setTranslatingQuery] = useState(false);
  // Panneau de progression pendant la recherche (~30 s avec les APIs en direct) :
  // informe l'utilisateur de l'étape en cours pour qu'il patiente sereinement.
  const [searchPhase, setSearchPhase] = useState<'idle' | 'translating' | 'searching'>('idle');
  const [searchElapsed, setSearchElapsed] = useState(0);
  // Affichage local-first : les résultats de la base locale apparaissent
  // immédiatement ; `liveRefreshing` indique qu'un travail de fond (sources en
  // direct OU reranking) se poursuit et que la liste se rafraîchit toute seule.
  const [liveRefreshing, setLiveRefreshing] = useState(false);
  const [refreshLabel, setRefreshLabel] = useState<string | null>(null);
  // Phase RÉELLE du backend (renvoyée par /populate/status), pour piloter le
  // panneau de progression au lieu d'un minuteur approximatif côté client.
  const [searchBackendPhase, setSearchBackendPhase] =
    useState<'local' | 'federation' | 'scoring' | 'done' | null>(null);
  // Compteurs live par source pendant la fédération (depuis /populate/status),
  // affichés sur la page d'attente, + étape finale "Affinage (rerank)".
  const [searchSourceProgress, setSearchSourceProgress] = useState<Record<string, number> | null>(null);
  const [searchFinalizing, setSearchFinalizing] = useState(false);
  const [folders, setFolders] = useState<ScenarioFolder[]>([]);
  // Tri par défaut = pertinence (le score est désormais calculé pour TOUS les
  // documents affichés, la recherche attendant la fin du scoring). "relevance" =
  // rerank Cohere quand présent, sinon score sémantique (cosinus).
  const [sortBy, setSortBy] = useState<"relevance" | "semantic" | "lexical" | "year_desc" | "year_asc">("relevance");


  useEffect(() => {
    getFilterOptions()
      .then((opts) => {
        setFilterOptions(opts);
        const b = yearSliderBounds(opts.year);
        setYearRange([b.min, b.max]);
      })
      .catch((err) => console.error(err));
  }, []);

  // Nettoyage au démontage : stopper tous les pollings de pipeline encore actifs.
  // Sans cela, ils continuent d'émettre des requêtes et de tenter des setState sur
  // un arbre démonté (fuite mémoire + avertissements React). On lit .current dans
  // le cleanup à dessein, pour stopper les intervalles RÉELLEMENT en cours.
  useEffect(() => {
    return () => {
      Object.values(pipelinePollRef.current).forEach((id) => clearInterval(id));
      pipelinePollRef.current = {};
    };
  }, []);

  useEffect(() => {
    setFilters((prev) => ({ ...prev, projectContext }));
    setPage(1);
    setSelectedResult(null);
    setSelectedDocument(null);
    setEvidenceSummary(null);
  }, [projectContext]);

  // Compteur de secondes écoulées pendant une recherche (panneau de progression).
  useEffect(() => {
    if (searchPhase === 'idle') return;
    setSearchElapsed(0);
    const t = setInterval(() => setSearchElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [searchPhase]);

  useEffect(() => {
    if (activeTab === "stats") {
      fetchCorpusStats().then(setCorpusStats).catch(console.error);
      fetchFulltextStats().then(setFulltextStats).catch(console.error);
      fetchCorpusStatsByYearNamed().then(setCorpusStatsByYear).catch(() => fetchCorpusStatsByYear().then(setCorpusStatsByYear).catch(console.error));
      // Scénarios nécessaires au compteur "Scénarios" + "Par scénario" du tableau
      // de bord (sinon ils ne sont chargés que sur l'onglet Scénarios → 0).
      fetchGesicaScenarios().then(setGesicaScenarios).catch(console.error);
      fetchUserScenarios().then(setUserScenarios).catch(console.error);
    }
    if (activeTab === "scenarios") {
      setLoadingScenarios(true);
      setScenariosError(null);
      Promise.all([
        fetchGesicaScenarios(),
        fetchUserScenarios(),
        fetchFolders(),
      ])
        .then(([gesica, user, foldersData]) => {
          setGesicaScenarios(gesica);
          setUserScenarios(user);
          setFolders(foldersData);
          // Synchroniser savedSearches avec les user_scenarios backend
          setSavedSearches(user.map(u => ({
            id: u.id,
            query: u.query,
            mode: u.mode as SearchMode,
            projectContext: (u.filters?.projectContext ?? "literev") as ProjectContext,
            timestamp: u.created_at ? new Date(u.created_at).getTime() : Date.now(),
            resultCount: u.result_count ?? 0,
            name: u.title !== u.query ? u.title : undefined,
            pinned: u.pinned,
          })));
          setLoadingScenarios(false);
        })
        .catch((err) => { setScenariosError(String(err)); setLoadingScenarios(false); });
    }
  }, [activeTab]);

  // Deduplicate to ONE entry per document (keep the highest-scoring chunk per doc).
  // The backend returns one row per chunk; multiple chunks from the same document
  // must not produce multiple paginated entries — that causes the same paper to
  // appear on multiple pages and makes the page count meaningless.
  const dedupedResults = useMemo(() => {
    const byDoc = new Map<number, SearchResult>();
    for (const r of results) {
      const prev = byDoc.get(r.documentId);
      if (!prev || (r.score ?? 0) > (prev.score ?? 0)) {
        byDoc.set(r.documentId, r);
      }
    }
    const deduped = Array.from(byDoc.values());
    return deduped.sort((a, b) => {
      if (sortBy === "year_desc") return (b.year ?? 0) - (a.year ?? 0);
      if (sortBy === "year_asc") return (a.year ?? 0) - (b.year ?? 0);
      if (sortBy === "semantic") return (b.semanticScore ?? 0) - (a.semanticScore ?? 0);
      if (sortBy === "lexical") return (b.lexicalScore ?? 0) - (a.lexicalScore ?? 0);
      // "relevance" (défaut) = MÊME ordre de pertinence que le backend : rerank
      // Cohere quand présent (sous-ensemble pertinent), sinon cosinus. On TIERE
      // (rerankés d'abord) pour ne pas mélanger deux échelles de score distinctes.
      const aHas = a.rerankScore != null, bHas = b.rerankScore != null;
      if (aHas !== bHas) return aHas ? -1 : 1;
      if (aHas && bHas && a.rerankScore !== b.rerankScore) return (b.rerankScore as number) - (a.rerankScore as number);
      return (b.semanticScore ?? 0) - (a.semanticScore ?? 0);
    });
  }, [results, sortBy]);

  const uniqueDocCount = dedupedResults.length;

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

  // Construit la liste de sous-requêtes effective : la requête principale (langage
  // naturel) suivie des sous-requêtes additionnelles non vides. Renvoie null s'il
  // n'y a pas au moins une sous-requête additionnelle → recherche mono-requête.
  function buildSubQueries(): SubQuery[] | null {
    const extra = extraQueries
      .map((q) => ({ kind: q.kind, text: q.text.trim() }))
      .filter((q) => q.text);
    if (!extra.length || !query.trim()) return null;
    return [{ kind: "natural", text: query.trim() }, ...extra];
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
      // Mode booléen : on traduit d'abord la requête en langage naturel en une
      // requête booléenne. C'est CETTE requête qui définit le corpus, donc on la
      // recherche ET on la persiste sur le scénario → le compteur affiché ==
      // la taille du corpus (même requête booléenne des deux côtés).
      let strategy: SearchStrategy | null = null;
      if (mode === "boolean") {
        setSearchPhase('translating');
        setTranslatingQuery(true);
        try {
          strategy = await fetchSearchStrategy(query.trim());
          setBooleanStrategy(strategy);
        } catch (e) {
          console.warn("Boolean translation failed, using raw query:", e);
          setBooleanStrategy(null);
        } finally {
          setTranslatingQuery(false);
        }
      } else {
        setBooleanStrategy(null);
      }
      setSearchPhase('searching');
      // ── LA recherche CONSTRUIT le corpus, puis l'affiche ─────────────────────
      // Le nombre affiché == la taille du corpus du scénario (même opération, même
      // helper booléen). On crée le scénario, on construit le corpus (requête
      // booléenne sur base locale ∪ sources live, plafond 2000/source), puis on
      // lit le corpus et on l'affiche. Plus de "preview" divergente.
      const _sub = buildSubQueries();
      const newScenario = await createUserScenario({
        name: query.trim(),
        query: query.trim(),
        mode,
        filters: { projectContext },
        result_count: 0,
        pinned: false,
        search_strategy: strategy ?? undefined,
        ...(_sub ? { sub_queries: _sub, combinator } : {}),
      });
      const sid = newScenario.id;
      await populateUserScenario(sid, { includeLive, maxResults: 2000 });
      // ── CONSTRUCTION COMPLÈTE PUIS AFFICHAGE ─────────────────────────────────
      // On construit tout le corpus (base locale ∪ sources en direct), on le score,
      // puis on l'affiche EN UNE FOIS. `renderCorpus` lit le corpus FINAL ; il
      // n'est appelé qu'à la toute fin (plus de lecture à chaque tour de boucle,
      // plus d'affichage "local-first" qui montrait des compteurs intermédiaires).
      let lastTotal = 0;
      let firstDetailLoaded = false;
      const renderCorpus = async () => {
        const corpus = await fetchScenarioCorpus(sid, { limit: 10000 });
        const corpusResults: SearchResult[] = (corpus.articles || []).map((a: CorpusArticle) => ({
          id: `${a.id}-0`,
          documentId: a.id,
          chunkIndex: 0,
          content: a.abstract ?? '',
          title: a.title,
          abstract: a.abstract,
          source: a.source,
          year: a.year,
          url: a.url,
          highlight: (a.abstract ?? '').slice(0, 600),
          hasFulltext: a.has_fulltext,
          isNew: a.is_new ?? false,
          semanticScore: a.similarity_score ?? null,
          rerankScore: a.rerank_score ?? null,
          score: a.similarity_score ?? 0,
        }));
        setResults(corpusResults);
        setSearchTotalMatching(corpus.total);   // == taille du corpus
        // Répartition par source (base locale vs API en direct), nouvelles
        // références live, et texte intégral vs résumé seul — lus depuis le corpus.
        setSearchSourceBreakdown(corpus.source_breakdown ?? null);
        setSearchLiveNewCount(corpus.newly_fetched ?? 0);
        setSearchFulltextDocs(corpus.docs_with_fulltext ?? null);
        setSearchAbstractDocs(corpus.docs_abstract_only ?? null);
        // Le score sémantique (cosinus) est calculé pour tous les documents → on
        // l'affiche. L'ordre "Pertinence" l'affine via le rerank Cohere sur le
        // sous-ensemble pertinent.
        setSearchScoreType('semantic');
        setSearchScoreLabel('Score sémantique (cosinus) · l\'ordre « Pertinence » est affiné par le rerank Cohere sur le sous-ensemble pertinent');
        lastTotal = corpus.total;
        const first = corpusResults[0] ?? null;
        if (first && !firstDetailLoaded) {
          firstDetailLoaded = true;
          loadDocumentDetail(first).catch(() => { /* non bloquant */ });
        }
        return corpusResults.length;
      };
      // ── ATTENDRE LA FIN COMPLÈTE AVANT D'AFFICHER ────────────────────────────
      // Plus d'affichage local-first : on garde la page d'attente informative tant
      // que la récupération (base locale + sources en direct) ET le scoring ne sont
      // pas terminés. On sonde la phase RÉELLE du backend (local → federation →
      // scoring → done) et les compteurs par source, sans rien afficher encore.
      let reachedDone = false;
      for (let i = 0; i < 240; i++) {
        let status = 'pending';
        let phase: 'local' | 'federation' | 'scoring' | 'done' | null = null;
        try {
          const st = await fetchUserScenarioPopulateStatus(sid);
          status = st.status;
          phase = st.phase ?? null;
          if (st.sources) setSearchSourceProgress(st.sources);
        } catch { /* transient — keep polling */ }
        if (phase) setSearchBackendPhase(phase);
        if (status === 'done') { reachedDone = true; break; }
        if (status === 'error') throw new Error(t("search.corpusBuildFailed"));
        await new Promise((r) => setTimeout(r, 2000));
      }
      // Récupération + scoring (cosinus) terminés. Le cross-encoder Cohere réordonne
      // ENSUITE le sous-ensemble pertinent : on l'attend AUSSI pour que l'ordre soit
      // STABLE au moment de l'affichage (pas de réordonnancement sous les yeux de
      // l'utilisateur). Sans clé Cohere, l'étape est instantanée.
      setSearchBackendPhase('done');
      if (reachedDone) {
        setSearchFinalizing(true);
        for (let j = 0; j < 40; j++) {
          let rerank: string = 'done';
          try {
            const st = await fetchUserScenarioPopulateStatus(sid);
            rerank = st.rerank_status ?? 'done';
          } catch { rerank = 'done'; }
          if (rerank !== 'running') break;
          await new Promise((r) => setTimeout(r, 2000));
        }
        setSearchFinalizing(false);
      }
      // Tout est prêt (corpus final + scores + rerank) → on affiche UNE fois. Le
      // `loading` repasse à false dans le finally juste après : la page d'attente
      // disparaît et les résultats finaux (compteur == corpus) apparaissent ensemble.
      try {
        await renderCorpus();
      } catch {
        await new Promise((r) => setTimeout(r, 1500));
        await renderCorpus(); // une erreur ici remonte au catch externe (message lisible)
      }
      const corpus = { total: lastTotal };
      // Mettre à jour les listes locales (scénario désormais construit).
      setUserScenarios(prev => {
        const filtered = prev.filter(s => !(s.query === newScenario.query && s.mode === newScenario.mode && !s.pinned));
        return [{ ...newScenario, result_count: corpus.total, articleCount: corpus.total }, ...filtered].slice(0, 50);
      });
      setSavedSearches(prev => {
        const filtered = prev.filter(s => !(s.query === query.trim() && s.mode === mode && !s.pinned));
        return [{
          id: sid,
          query: newScenario.query,
          mode: newScenario.mode as SearchMode,
          projectContext: (newScenario.filters?.projectContext ?? projectContext) as ProjectContext,
          timestamp: newScenario.created_at ? new Date(newScenario.created_at).getTime() : Date.now(),
          resultCount: corpus.total,
          name: undefined,
          pinned: false,
        }, ...filtered].slice(0, 50);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : t("common.unknownError"));
      setResults([]);
    } finally {
      setLoading(false);
      setLiveRefreshing(false);
      setRefreshLabel(null);
      setSearchBackendPhase(null);
      setSearchFinalizing(false);
      setSearchSourceProgress(null);
      setSearchPhase('idle');
    }
  }

  function _launchPipelineForScenario(scenarioId: string) {
    // Déclencher le pipeline complet dès qu'un scénario est épinglé
    startUserScenarioPipeline(scenarioId, 500)
      .then(() => {
        // Démarrer le polling de statut
        const pollInterval = setInterval(() => {
          fetchUserScenarioPipelineStatus(scenarioId)
            .then(status => {
              setPipelineStatuses(prev => ({ ...prev, [scenarioId]: status }));
              if (status.overall_status === 'done' || status.overall_status === 'error') {
                clearInterval(pollInterval);
                delete pipelinePollRef.current[scenarioId];
                // Rafraîchir la liste des scénarios pour mettre à jour article_count
                fetchUserScenarios().then(data => {
                  setUserScenarios(data);
                  setSavedSearches(data.map(u => ({
                    id: u.id, query: u.query, mode: u.mode as SearchMode,
                    projectContext: (u.filters?.projectContext ?? 'literev') as ProjectContext,
                    timestamp: u.created_at ? new Date(u.created_at).getTime() : Date.now(),
                    // Afficher article_count (corpus réel) si disponible, sinon result_count (snapshot recherche)
                    resultCount: u.articleCount > 0 ? u.articleCount : (u.resultCount ?? u.result_count ?? 0),
                    name: u.title !== u.query ? u.title : undefined,
                    pinned: u.pinned,
                  })));
                }).catch(console.warn);
              }
            })
            .catch(console.warn);
        }, 5000);
        pipelinePollRef.current[scenarioId] = pollInterval;
      })
      .catch(err => console.warn('Pipeline launch failed:', err));
  }

  function handleSaveAsScenario() {
    if (!query.trim()) return;
    const name = saveSearchName.trim() || query.trim();
    const _sub = buildSubQueries();
    // Chercher si une entrée non-épinglée existe déjà pour cette requête. Les
    // recherches multi-sous-requêtes créent toujours une entrée neuve (pas de dédup
    // par query/mode : la même requête principale peut porter des sous-requêtes/un
    // combinateur différents).
    const existing = _sub ? undefined : userScenarios.find(s => s.query === query && s.mode === mode && !s.pinned);
    if (existing) {
      // Mettre à jour : renommer + épingler + lancer pipeline
      patchUserScenario(existing.id, { name, pinned: true })
        .then(updated => {
          setUserScenarios(prev => prev.map(s => s.id === updated.id ? updated : s));
          setSavedSearches(prev => prev.map(s => s.id === updated.id ? { ...s, name, pinned: true } : s));
          _launchPipelineForScenario(updated.id);
        })
        .catch(err => console.warn('Patch user_scenario failed:', err));
    } else {
      // Créer un nouveau scénario épinglé + lancer pipeline
      createUserScenario({
        name,
        query: query.trim(),
        mode,
        filters: { projectContext },
        result_count: searchTotalMatching ?? results.length,
        pinned: true,
        search_strategy: (mode === "boolean" ? booleanStrategy : null) ?? undefined,
        ...(_sub ? { sub_queries: _sub, combinator } : {}),
      }).then(newScenario => {
        setUserScenarios(prev => [newScenario, ...prev]);
        setSavedSearches(prev => [{
          id: newScenario.id,
          query: newScenario.query,
          mode: newScenario.mode as SearchMode,
          projectContext: (newScenario.filters?.projectContext ?? projectContext) as ProjectContext,
          timestamp: newScenario.created_at ? new Date(newScenario.created_at).getTime() : Date.now(),
          // result_count = snapshot de la recherche (sera remplacé par article_count après pipeline)
          resultCount: newScenario.result_count ?? results.length,
          name,
          pinned: true,
        }, ...prev]);
        _launchPipelineForScenario(newScenario.id);
      }).catch(err => console.warn('Create user_scenario failed:', err));
    }
    setShowSaveDialog(false);
    setSaveSearchName("");
  }

  function handleReplaySearch(s: SavedSearch) {
    setQuery(s.query);
    // Recherche unifiée : toujours en mode booléen (traduction automatique si besoin).
    setMode("boolean");
    setProjectContext(s.projectContext);
    setActiveTab("search");
    // Déclencher la recherche après mise à jour du state
    setTimeout(() => {
      const btn = document.getElementById("search-btn");
      if (btn) btn.click();
    }, 100);
  }

  function handleDeleteSavedSearch(id: string) {
    deleteUserScenario(id)
      .then(() => {
        setUserScenarios(prev => prev.filter(s => s.id !== id));
        setSavedSearches(prev => prev.filter(s => s.id !== id));
      })
      .catch(err => console.warn('Delete user_scenario failed:', err));
  }

  function handleTogglePin(id: string) {
    const current = userScenarios.find(s => s.id === id);
    if (!current) return;
    patchUserScenario(id, { pinned: !current.pinned })
      .then(updated => {
        setUserScenarios(prev => prev.map(s => s.id === updated.id ? updated : s));
        setSavedSearches(prev => prev.map(s => s.id === id ? { ...s, pinned: !s.pinned } : s));
      })
      .catch(err => console.warn('Toggle pin user_scenario failed:', err));
  }

  async function handlePopulateUserScenario(id: string) {
    // Lancer le pipeline complet (PubMed + PICO + metadata + fulltext + clustering)
    setPopulatingId(id);
    try {
      _launchPipelineForScenario(id);
    } catch (err) {
      console.warn('Pipeline launch failed:', err);
    } finally {
      setPopulatingId(null);
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
    const b = yearSliderBounds(filterOptions?.year);
    setYearRange([b.min, b.max]);
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
    { id: "search", label: t("nav.search"), icon: <BookOpen size={14} /> },
    { id: "scenarios", label: t("nav.scenarios"), icon: <Activity size={14} /> },
    { id: "terrain", label: t("nav.terrain"), icon: <Cloud size={14} className="text-brand-400" /> },
    { id: "stats", label: t("nav.stats"), icon: <BarChart2 size={14} /> },
  ];

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,rgba(10,54,33,0.18),transparent_35%),linear-gradient(180deg,#0a1410_0%,#121e19_100%)] text-white">
      <header className="border-b border-white/8 bg-[#0a1410]/80 backdrop-blur-xl">
        <div className="mx-auto max-w-[1380px] px-6 py-6">
          <div className="flex flex-col gap-6 xl:flex-row xl:items-center xl:justify-between">
            <div className="max-w-3xl">
              <div className="flex items-center gap-4 mb-2">
                <img src="/literev-logo.png" alt="LiteRev" className="h-14 w-auto object-contain" />
                <h1 className="text-3xl font-bold tracking-tight text-white">Evidence to Scenario</h1>
              </div>
            </div>

            <div className="flex items-center gap-6">
              <div className="flex items-center gap-1 rounded-xl border border-white/10 bg-white/5 p-0.5 text-xs" title={t("header.languageLabel")}>
                {(["fr", "en"] as const).map((l) => (
                  <button
                    key={l}
                    type="button"
                    onClick={() => setLang(l)}
                    className={`rounded-lg px-2 py-1 font-semibold transition ${
                      lang === l ? "bg-brand-700 text-gold-400" : "text-white/50 hover:text-white"
                    }`}
                  >
                    {l.toUpperCase()}
                  </button>
                ))}
              </div>
              <button
                type="button"
                onClick={handleManageApiKey}
                title={apiKeySet ? t("header.adminKeyActiveTooltip") : t("header.adminKeySetTooltip")}
                className={`flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-medium transition ${
                  apiKeySet
                    ? "border-brand-500/40 bg-brand-700/30 text-brand-200 hover:bg-brand-700/50"
                    : "border-white/10 bg-white/5 text-white/50 hover:text-white hover:bg-white/10"
                }`}
              >
                {apiKeySet ? <KeyRound size={13} /> : <Lock size={13} />}
                {apiKeySet ? t("header.adminKeyActive") : t("header.readOnly")}
              </button>
              <img src="/logo.jpg" alt="LiteRev arbre" className="h-20 w-20 rounded-2xl object-cover shadow-xl opacity-90" />
            </div>
          </div>

          <div className="mt-6 flex gap-1 rounded-2xl border border-white/10 bg-forest-900/60 p-1 w-fit">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 rounded-xl px-4 py-2 text-sm transition ${
                  activeTab === tab.id
                    ? "bg-brand-700 text-gold-400 font-semibold shadow-inner"
                    : "text-white/60 hover:text-white hover:bg-white/8"
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
          <StatsView corpusStats={corpusStats} fulltextStats={fulltextStats} scenarios={[...gesicaScenarios, ...userScenarios]} statsByYear={corpusStatsByYear} onRefresh={() => { fetchCorpusStats().then(setCorpusStats).catch(console.error); fetchFulltextStats().then(setFulltextStats).catch(console.error); }} />
        )}


        {activeTab === "terrain" && (
          <TerrainView />
        )}

        {activeTab === "scenarios" && (
          <ScenariosView
            scenarios={gesicaScenarios}
            loading={loadingScenarios}
            error={scenariosError}
            savedSearches={savedSearches}
            userScenarios={userScenarios}
            onReplaySearch={handleReplaySearch}
            onDeleteSearch={handleDeleteSavedSearch}
            onTogglePin={handleTogglePin}
            onPopulateUserScenario={handlePopulateUserScenario}
            populatingId={populatingId}
            pipelineStatuses={pipelineStatuses}
            folders={folders}
            onCreateFolder={async (name: string, color: string) => {
              const f = await createFolder(name, color);
              setFolders(prev => [f, ...prev]);   // nouveau dossier EN HAUT, visible immédiatement
              return f;
            }}
            onDeleteFolder={async (folderId: string) => {
              await deleteFolder(folderId);
              setFolders(prev => prev.filter(f => f.id !== folderId));
            }}
            onRenameFolder={async (folderId: string, name: string, color: string) => {
              const f = await updateFolder(folderId, name, color, 0);
              setFolders(prev => prev.map(x => x.id === folderId ? f : x));
            }}
            onAssignFolder={async (scenarioId: string, folderId: string | null) => {
              await assignScenarioToFolder(scenarioId, folderId);
              setUserScenarios(prev => prev.map(s => s.id === scenarioId ? { ...s, folder_id: folderId } : s));
            }}
          />
        )}



        {activeTab === "search" && (
          <div className="grid gap-6 xl:grid-cols-[280px_minmax(0,1fr)]">
            <aside className="xl:sticky xl:top-8 xl:self-start">
              <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold text-white">{t("search.filters")}</h2>
                  <button
                    type="button"
                    onClick={handleReset}
                    title={t("search.resetFiltersTooltip")}
                    className="flex items-center gap-1 rounded-xl border border-white/10 px-2 py-1 text-xs text-forest-400 transition hover:border-white/20 hover:text-white/80"
                  >
                    <RotateCcw size={12} />
                    {t("search.reset")}
                  </button>
                </div>

                <div className="mt-5 space-y-4">
                  {FILTER_FIELDS.map(([key, label]) => {
                    const options = filterOptions?.[key] ?? [];
                    const isDiseaseField = key === "diseaseOrCondition";
                    const filteredOptions = isDiseaseField && diseaseSearch
                      ? options.filter(o => o.label.toLowerCase().includes(diseaseSearch.toLowerCase()))
                      : options;
                    return (
                      <div key={key} className="block">
                        <span className="mb-2 block text-sm font-medium text-white/80">
                          {t(label)}
                        </span>
                        {isDiseaseField && (
                          <div className="relative mb-1">
                            <input
                              type="text"
                              placeholder={t("search.searchDiseasePlaceholder")}
                              value={diseaseSearch}
                              onChange={e => setDiseaseSearch(e.target.value)}
                              className="w-full rounded-xl border border-white/10 bg-forest-950/80 px-3 py-2 text-xs text-white placeholder-white/30 focus:border-brand-400 focus:outline-none"
                            />
                            {diseaseSearch.trim() && filteredOptions.length > 0 &&
                              !(filteredOptions.length === 1 && filteredOptions[0].label.toLowerCase() === diseaseSearch.trim().toLowerCase()) && (
                              <ul className="absolute z-20 mt-1 max-h-56 w-full overflow-auto rounded-xl border border-white/10 bg-forest-950 shadow-2xl">
                                {filteredOptions.slice(0, 10).map((opt) => (
                                  <li key={String(opt.value)}>
                                    <button
                                      type="button"
                                      onClick={() => {
                                        setFilters((prev) => ({ ...prev, [key]: String(opt.value) }));
                                        setDiseaseSearch(opt.label);
                                      }}
                                      className="block w-full px-3 py-2 text-left text-xs text-white/80 transition hover:bg-brand-500/20 hover:text-white"
                                    >
                                      {opt.label}
                                    </button>
                                  </li>
                                ))}
                              </ul>
                            )}
                          </div>
                        )}
                        <select
                          value={(filters as Record<string, string | undefined>)[key] ?? ""}
                          onChange={(e) =>
                            setFilters((prev) => ({
                              ...prev,
                              [key]: e.target.value || undefined,
                            }))
                          }
                          className="w-full appearance-none rounded-2xl border border-white/10 bg-forest-950/80 px-3 py-3 text-sm text-white focus:border-brand-400 focus:outline-none"
                        >
                          <option value="">{t("search.all")}</option>
                          {filteredOptions.map((opt) => (
                            <option key={String(opt.value)} value={String(opt.value)}>
                              {opt.label}
                            </option>
                          ))}
                        </select>
                      </div>
                    );
                  })}

                  {/* Rendu UNIQUEMENT quand les filtres sont chargés : sinon le
                      curseur Années affiche brièvement les bornes de repli (1990)
                      avant de sauter à l'année réelle du corpus (ex. 1845). React 19
                      regroupe setFilterOptions + setYearRange en un seul rendu, donc
                      le bloc apparaît directement aux bonnes bornes, sans clignotement. */}
                  {filterOptions && (
                  <div>
                    <span className="mb-2 block text-sm font-medium text-white/80">
                      {t("search.years")}{" "}
                      <span className="font-mono text-brand-300 text-xs">
                        {yearRange[0]} - {yearRange[1]}
                      </span>
                    </span>
                    {(() => {
                      const { min: minYear, max: maxYear } = yearSliderBounds(filterOptions?.year);
                      const range = maxYear - minYear || 1;
                      const leftPct = ((yearRange[0] - minYear) / range) * 100;
                      const rightPct = ((yearRange[1] - minYear) / range) * 100;
                      return (
                        <div className="relative h-8 flex items-center">
                          <div className="absolute w-full h-1.5 rounded-full bg-white/10" />
                          <div
                            className="absolute h-1.5 rounded-full bg-brand-500"
                            style={{ left: `${leftPct}%`, width: `${rightPct - leftPct}%` }}
                          />
                          <input
                            type="range"
                            min={minYear} max={maxYear}
                            value={yearRange[0]}
                            onChange={(e) => {
                              const v = Number(e.target.value);
                              if (v <= yearRange[1]) setYearRange([v, yearRange[1]]);
                            }}
                            className="range-dual absolute w-full h-1.5 bg-transparent cursor-pointer"
                            style={{ zIndex: 4 }}
                          />
                          <input
                            type="range"
                            min={minYear} max={maxYear}
                            value={yearRange[1]}
                            onChange={(e) => {
                              const v = Number(e.target.value);
                              if (v >= yearRange[0]) setYearRange([yearRange[0], v]);
                            }}
                            className="range-dual absolute w-full h-1.5 bg-transparent cursor-pointer"
                            style={{ zIndex: 5 }}
                          />
                        </div>
                      );
                    })()}
                    <div className="flex justify-between text-[10px] text-white/30 mt-1">
                      <span>{yearSliderBounds(filterOptions?.year).min}</span>
                      <span>{yearSliderBounds(filterOptions?.year).max}</span>
                    </div>
                  </div>
                  )}
                </div>
              </div>
            </aside>

            <section className="space-y-6">
              <section className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
                {/* Une seule recherche : l'utilisateur saisit librement, en langage
                    naturel ou en requête booléenne. La requête booléenne définit le
                    corpus. Le seuil sémantique n'intervient plus ici (uniquement sur
                    la page scénario pour sélectionner les articles pertinents). */}
                <div className="mb-3 flex items-start gap-2 text-xs text-forest-300">
                  <span>
                    {t("search.explainerPart1")} <span className="text-white/50">{t("search.explainerExample1")}</span>{" "}
                    {t("search.explainerPart2")} <span className="text-white/50">{t("search.explainerExample2")}</span>.
                  </span>
                </div>

                <div className="flex flex-col gap-3 lg:flex-row">
                  <input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                    placeholder={t("search.queryPlaceholder")}
                    className="min-h-14 flex-1 rounded-2xl border border-white/10 bg-forest-950/80 px-4 text-white outline-none placeholder:text-forest-500 focus:border-brand-400"
                  />
                  <button
                    id="search-btn"
                    type="button"
                    onClick={handleSearch}
                    disabled={loading}
                    className="min-h-14 rounded-2xl bg-brand-400 px-6 font-semibold text-forest-950 transition hover:bg-brand-300 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {loading ? t("search.searching") : t("search.searchButton")}
                  </button>
                </div>

                <label className="mt-3 flex items-start gap-2 text-xs text-forest-300 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={includeLive}
                    onChange={(e) => setIncludeLive(e.target.checked)}
                    className="mt-0.5 h-4 w-4 accent-brand-400 shrink-0"
                  />
                  <span>
                    <span>{t("search.includeLive")}</span>
                    <span className="block text-forest-500 mt-0.5">{t("search.includeLiveHint")}</span>
                  </span>
                </label>

                {/* Recherche multi-sous-requêtes (avancé) : combine la requête
                    principale ci-dessus avec des sous-requêtes additionnelles, par
                    union (OU) ou intersection (ET). */}
                <div className="mt-3">
                  {extraQueries.length === 0 ? (
                    <button
                      type="button"
                      onClick={() => setExtraQueries([{ kind: "natural", text: "" }])}
                      className="text-xs text-brand-300 hover:text-brand-200 transition"
                    >
                      + {t("search.addSubQuery")}
                    </button>
                  ) : (
                    <div className="rounded-2xl border border-white/10 bg-forest-950/40 p-3 space-y-2.5">
                      <p className="text-xs text-forest-300">{t("search.combineHint")}</p>
                      <div className="flex items-center gap-1.5 text-[11px]">
                        <span className="text-forest-500">{t("search.combinator")}</span>
                        {(["union", "intersection"] as const).map((c) => (
                          <button
                            key={c}
                            type="button"
                            onClick={() => setCombinator(c)}
                            className={`rounded-full border px-2 py-0.5 transition ${
                              combinator === c
                                ? "border-brand-400/60 bg-brand-500/20 text-brand-300"
                                : "border-white/10 bg-white/5 text-forest-400 hover:text-white"
                            }`}
                          >
                            {c === "union" ? t("search.combinatorUnion") : t("search.combinatorIntersection")}
                          </button>
                        ))}
                      </div>
                      {extraQueries.map((sq, i) => (
                        <div key={i} className="flex items-center gap-2">
                          <div className="flex shrink-0 overflow-hidden rounded-lg border border-white/10 text-[10px]">
                            {(["natural", "boolean"] as const).map((k) => (
                              <button
                                key={k}
                                type="button"
                                onClick={() =>
                                  setExtraQueries((prev) =>
                                    prev.map((r, j) => (j === i ? { ...r, kind: k } : r)),
                                  )
                                }
                                className={`px-2 py-1.5 transition ${
                                  sq.kind === k
                                    ? "bg-brand-500/20 text-brand-300"
                                    : "text-forest-400 hover:text-white"
                                }`}
                              >
                                {k === "boolean"
                                  ? t("search.subQueryKindBoolean")
                                  : t("search.subQueryKindNatural")}
                              </button>
                            ))}
                          </div>
                          <input
                            value={sq.text}
                            onChange={(e) =>
                              setExtraQueries((prev) =>
                                prev.map((r, j) => (j === i ? { ...r, text: e.target.value } : r)),
                              )
                            }
                            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                            placeholder={t("search.subQueryPlaceholder")}
                            className="min-w-0 flex-1 rounded-xl border border-white/10 bg-forest-950/80 px-3 py-2 text-sm text-white outline-none placeholder:text-forest-500 focus:border-brand-400"
                          />
                          <button
                            type="button"
                            onClick={() =>
                              setExtraQueries((prev) => prev.filter((_, j) => j !== i))
                            }
                            title={t("search.subQueryRemove")}
                            aria-label={t("search.subQueryRemove")}
                            className="shrink-0 px-1 text-sm text-forest-500 hover:text-rose-300 transition"
                          >
                            ✕
                          </button>
                        </div>
                      ))}
                      <button
                        type="button"
                        onClick={() =>
                          setExtraQueries((prev) => [...prev, { kind: "natural", text: "" }])
                        }
                        className="text-xs text-brand-300 hover:text-brand-200 transition"
                      >
                        + {t("search.addSubQuery")}
                      </button>
                    </div>
                  )}
                </div>

                {mode === "boolean" && (translatingQuery || booleanStrategy?.general) && (
                  <div className="mt-3 rounded-2xl border border-brand-400/20 bg-brand-400/5 p-3">
                    <div className="flex items-center gap-2 text-xs text-brand-300/80">
                      <span>{t("search.booleanQuery")}</span>
                      {translatingQuery && <span className="text-white/40">{t("search.translating")}</span>}
                    </div>
                    {booleanStrategy?.general && (
                      <code className="mt-1.5 block break-words font-mono text-xs text-white/80">
                        {booleanStrategy.general}
                      </code>
                    )}
                  </div>
                )}
              </section>

              {error && (
                <div className="rounded-2xl border border-rose-400/30 bg-rose-500/10 p-4 text-sm text-rose-100">
                  {error}
                </div>
              )}

              {loading && searchPhase !== 'idle' && (() => {
                // Étapes de la recherche pilotées par la PHASE RÉELLE du backend
                // (/populate/status), pas par un minuteur : le libellé affiché
                // correspond donc à ce que fait réellement le serveur.
                const liveSources = ["PubMed", "OpenAlex", "Crossref", "EuropePMC", "Preprints"];
                const translating = searchPhase === 'translating';
                const rankOf: Record<string, number> = { local: 1, federation: 2, scoring: 3, done: 4 };
                const rank = searchBackendPhase ? rankOf[searchBackendPhase] : 0;
                // Compteurs live par source (depuis /populate/status) pour rendre
                // l'attente concrète : on voit les références arriver source par source.
                const srcEntries = Object.entries(searchSourceProgress ?? {})
                  .filter(([, n]) => (n as number) > 0)
                  .sort((a, b) => (b[1] as number) - (a[1] as number));
                const fetchedSoFar = srcEntries.reduce((a, [, n]) => a + (n as number), 0);
                const steps: { label: string; done: boolean; active: boolean; hint?: string }[] = [
                  { label: t("search.stepTranslate"), done: !translating, active: translating },
                  { label: t("search.stepLocal"), done: !translating && rank > 1, active: !translating && rank <= 1 },
                  ...(includeLive ? [{
                    label: t("search.stepLive"),
                    done: rank > 2,
                    active: rank === 2,
                    hint: rank === 2
                      ? (fetchedSoFar > 0 ? `${fetchedSoFar} ${t("search.referencesFetched")}` : liveSources[searchElapsed % liveSources.length])
                      : undefined,
                  }] : []),
                  { label: t("search.stepScoring"), done: rank > 3 && !searchFinalizing, active: rank === 3 },
                  { label: t("search.stepRerank"), done: false, active: searchFinalizing },
                ];
                return (
                  <div className="rounded-3xl border border-brand-400/20 bg-white/5 p-6">
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-semibold text-white">{t("search.buildingCorpus")}</p>
                      <span className="font-mono text-xs text-white/40">{searchElapsed}s</span>
                    </div>
                    <ul className="mt-4 space-y-2.5">
                      {steps.map((s, i) => (
                        <li key={i} className="flex items-center gap-3 text-sm">
                          <span className="flex h-5 w-5 shrink-0 items-center justify-center">
                            {s.done ? (
                              <span className="text-brand-300">✓</span>
                            ) : s.active ? (
                              <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-brand-300/30 border-t-brand-300" />
                            ) : (
                              <span className="h-1.5 w-1.5 rounded-full bg-white/20" />
                            )}
                          </span>
                          <span className={s.done ? "text-white/40 line-through decoration-white/20" : s.active ? "text-white" : "text-white/40"}>
                            {s.label}
                            {s.hint && <span className="ml-1.5 text-emerald-300/70">· {s.hint}</span>}
                          </span>
                        </li>
                      ))}
                    </ul>
                    {/* Détail live par source pendant la fédération */}
                    {includeLive && rank === 2 && srcEntries.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {srcEntries.map(([src, n]) => (
                          <span key={src} className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-2 py-0.5 text-[10px] text-emerald-300/80">
                            {src}: <span className="font-semibold">{n as number}</span>
                          </span>
                        ))}
                      </div>
                    )}
                    <p className="mt-4 text-[11px] leading-snug text-white/30">
                      {t("search.resultsWillAppear")}
                    </p>
                  </div>
                );
              })()}

              {!loading && !error && !hasResults && (
                <div className="rounded-3xl border border-dashed border-white/10 bg-white/5 p-10 text-center text-forest-300">
                  {t("search.launchPrompt")}
                </div>
              )}

              {hasResults && (
                <>
                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <p className="text-sm text-forest-400">
                        {(() => {
                          // corpus.total inclut déjà les références live : pas d'addition.
                          const total = searchTotalMatching ?? uniqueDocCount;
                          return (
                            <>
                              <span className="font-semibold text-white">{total.toLocaleString()}</span>{" "}
                              {total > 1 ? t("search.documents") : t("search.document")} {total > 1 ? t("search.relevantPlural") : t("search.relevant")}
                              {" "}· {totalPages > 1 ? `${t("search.page")} ${page}/${totalPages}` : t("search.onePage")}
                            </>
                          );
                        })()}
                      </p>
                      {liveRefreshing && (
                        <p className="flex items-center gap-2 text-xs text-emerald-300/80">
                          <span className="h-3 w-3 animate-spin rounded-full border-2 border-emerald-300/30 border-t-emerald-300" />
                          {refreshLabel ?? t("search.refreshingResults")}
                        </p>
                      )}
                      {searchSourceBreakdown && Object.keys(searchSourceBreakdown).length > 0 && (() => {
                        const localEntries = Object.entries(searchSourceBreakdown).filter(([k]) => !k.endsWith(" (live)"));
                        const liveEntries = Object.entries(searchSourceBreakdown).filter(([k]) => k.endsWith(" (live)"));
                        return (
                          <>
                            {localEntries.length > 0 && (
                              <div className="flex flex-wrap items-center gap-1.5">
                                <span className="text-[10px] text-white/25 uppercase tracking-wider">{t("search.localBase")}</span>
                                {localEntries.map(([src, count]) => (
                                  <span key={src} className="rounded-lg border border-white/10 bg-white/5 px-2 py-0.5 text-xs text-white/50">
                                    {src}: <span className="font-semibold text-white/70">{count}</span>
                                  </span>
                                ))}
                              </div>
                            )}
                            {liveEntries.length > 0 && (
                              <div className="flex flex-wrap items-center gap-1.5">
                                <span className="text-[10px] text-emerald-500/60 uppercase tracking-wider">{t("search.liveApi")}</span>
                                {liveEntries.map(([src, count]) => (
                                  <span key={src} className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-2 py-0.5 text-xs text-emerald-400/70">
                                    {src.replace(" (live)", "")}: <span className="font-semibold text-emerald-300">{count}</span>
                                  </span>
                                ))}
                                {searchLiveNewCount != null && searchLiveNewCount > 0 && (
                                  <span className="text-[10px] text-emerald-500/50">+{searchLiveNewCount} {t("search.newReferences")}</span>
                                )}
                              </div>
                            )}
                          </>
                        );
                      })()}
                      {(searchFulltextDocs != null || searchAbstractDocs != null) && (
                        <div className="flex flex-wrap items-center gap-1.5">
                          <span className="text-[10px] text-white/25 uppercase tracking-wider">{t("search.content")}</span>
                          <span className="rounded-lg border border-brand-500/20 bg-brand-500/10 px-2 py-0.5 text-xs text-brand-300">
                            {t("search.fulltext")} <span className="font-semibold">{(searchFulltextDocs ?? 0).toLocaleString()}</span>
                          </span>
                          <span className="rounded-lg border border-white/10 bg-white/5 px-2 py-0.5 text-xs text-white/50">
                            {t("search.abstractOnly")} <span className="font-semibold text-white/70">{(searchAbstractDocs ?? 0).toLocaleString()}</span>
                          </span>
                        </div>
                      )}
                    </div>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={() => setShowSaveDialog(true)}
                        className="flex items-center gap-1.5 rounded-xl border border-gold-400/40 bg-gold-500/10 px-3 py-1.5 text-xs text-gold-300 transition hover:border-gold-400/70 hover:bg-gold-500/20"
                        title={t("search.saveAsScenarioTooltip")}
                      >
                        <BookOpen size={12} />
                        {t("search.saveAsScenario")}
                      </button>
                      <button
                        type="button"
                        onClick={() => handleExport("csv")}
                        className="flex items-center gap-1.5 rounded-xl border border-white/10 px-3 py-1.5 text-xs text-forest-300 transition hover:border-white/20 hover:text-white"
                      >
                        <Download size={12} />
                        CSV
                      </button>
                      <button
                        type="button"
                        onClick={() => handleExport("json")}
                        className="flex items-center gap-1.5 rounded-xl border border-white/10 px-3 py-1.5 text-xs text-forest-300 transition hover:border-white/20 hover:text-white"
                      >
                        <Download size={12} />
                        JSON
                      </button>
                    </div>
                  </div>

                  {/* Modal de sauvegarde comme scénario */}
                  {showSaveDialog && (
                    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
                      <div className="w-full max-w-md rounded-3xl border border-gold-400/30 bg-[#0a1410] p-6 shadow-2xl">
                        <h3 className="text-lg font-bold text-white mb-1">{t("search.saveAsScenario")}</h3>
                        <p className="text-sm text-white/60 mb-4">{t("search.saveDialogDesc")}</p>
                        <input
                          type="text"
                          value={saveSearchName}
                          onChange={e => setSaveSearchName(e.target.value)}
                          onKeyDown={e => e.key === "Enter" && handleSaveAsScenario()}
                          placeholder={query}
                          className="w-full rounded-2xl border border-white/10 bg-forest-950/80 px-4 py-3 text-white outline-none placeholder:text-forest-500 focus:border-gold-400 mb-4"
                          autoFocus
                        />
                        <div className="flex gap-3 justify-end">
                          <button
                            type="button"
                            onClick={() => { setShowSaveDialog(false); setSaveSearchName(""); }}
                            className="rounded-xl border border-white/10 px-4 py-2 text-sm text-white/60 hover:text-white transition"
                          >
                            {t("common.cancel")}
                          </button>
                          <button
                            type="button"
                            onClick={handleSaveAsScenario}
                            className="rounded-xl bg-gold-400 px-4 py-2 text-sm font-semibold text-forest-950 hover:bg-gold-300 transition"
                          >
                            {t("search.save")}
                          </button>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Sort controls */}
                  <div className="flex flex-wrap items-center gap-2 mb-2 mt-1">
                    <span className="text-xs text-forest-500">{t("search.sortBy")}</span>
                    {([
                      ["relevance", t("search.sortRelevance")],
                      ["year_desc", t("search.sortYearDesc")],
                      ["year_asc", t("search.sortYearAsc")],
                    ] as [typeof sortBy, string][]).map(([val, label]) => (
                      <button
                        key={val}
                        type="button"
                        onClick={() => { setSortBy(val); setPage(1); }}
                        className={`rounded-full border px-2.5 py-1 text-xs transition ${
                          sortBy === val
                            ? "border-brand-400/60 bg-brand-500/20 text-brand-300"
                            : "border-white/10 bg-white/5 text-forest-400 hover:border-white/20 hover:text-white"
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>

                  <div className="grid gap-6 2xl:grid-cols-[minmax(0,1fr)_380px]">
                    <div className="space-y-4">
                      {pagedResults.map((result) => (
                        <article
                          key={`${result.documentId}-${result.chunkIndex}-${result.content}`}
                          className={`rounded-3xl border bg-white/5 p-5 shadow-2xl transition ${
                            selectedResult?.id === result.id
                              ? "border-brand-400/60"
                              : "border-white/10 hover:border-brand-400/40"
                          }`}
                        >
                          <div className="flex items-start justify-between gap-4">
                            <button
                              type="button"
                              onClick={() => loadDocumentDetail(result)}
                              className="text-left"
                            >
                              <h3 className="text-xl font-semibold text-white hover:text-brand-300">
                                {result.title}
                              </h3>
                            </button>
                            {result.url && (
                              <a
                                href={result.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex shrink-0 items-center gap-2 rounded-xl border border-white/10 px-3 py-2 text-sm text-white/80 hover:bg-white/10"
                              >
                                {t("search.source")}
                                <ExternalLink size={14} />
                              </a>
                            )}
                          </div>

                          <div className="mt-3 flex flex-wrap gap-2 text-xs text-forest-400">
                            {/* Score sémantique (cosinus) — calculé pour TOUS les documents
                                affichés (la recherche attend la fin du scoring). */}
                            {searchScoreType && searchScoreType !== 'none' && (
                              <span className={`rounded-full px-2 py-1 ${
                                searchScoreType === 'hybrid' ? 'bg-violet-500/20 text-violet-300' :
                                searchScoreType === 'semantic' ? 'bg-blue-500/20 text-blue-300' :
                                'bg-white/5'
                              }`} title={searchScoreLabel ?? undefined}>
                                {searchScoreType === 'hybrid' ? '⊕' :
                                 searchScoreType === 'semantic' ? '◎' :
                                 '≡'} {(result.score ?? 0).toFixed(3)}
                              </span>
                            )}
                            {/* Score de reranking (cross-encoder Cohere) quand présent — c'est
                                LUI qui ordonne le sous-ensemble pertinent en tête de liste
                                (échelle distincte du cosinus, d'où l'affichage des deux). */}
                            {result.rerankScore != null && (
                              <span className="rounded-full bg-violet-500/15 px-2 py-1 text-violet-300 border border-violet-500/25"
                                title={t("search.rerankTooltip")}>
                                ⊕ Rerank {(result.rerankScore).toFixed(2)}
                              </span>
                            )}
                            {/* Décomposition (hybride uniquement : sinon redondant avec le score global) */}
                            {searchScoreType === 'hybrid' && result.semanticScore != null && (
                              <span className="rounded-full bg-blue-500/10 px-2 py-1 text-blue-300 border border-blue-500/20" title={t("search.semanticTooltip")}>
                                {t("search.semanticShort")} {(result.semanticScore).toFixed(2)}
                              </span>
                            )}
                            {searchScoreType === 'hybrid' && result.lexicalScore != null && (
                              <span className="rounded-full bg-amber-500/10 px-2 py-1 text-amber-300 border border-amber-500/20" title={t("search.lexicalTooltip")}>
                                {t("search.lexicalShort")} {(result.lexicalScore).toFixed(2)}
                              </span>
                            )}
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

                            {result.scenarioType && (
                              <span className="rounded-full bg-violet-500/10 px-2 py-1 text-violet-200">
                                {result.scenarioType}
                              </span>
                            )}
                            {result.evidenceCategory && (
                              <span className="rounded-full bg-forest-700/60 px-2 py-1 text-forest-300">
                                {result.evidenceCategory}
                              </span>
                            )}
                            {/* Badge couverture textuelle — basé sur hasFulltext
                                (présence d'un chunk plein texte), pas sur chunkType
                                qui n'est pas renseigné pour les résultats du corpus. */}
                            <span className={`rounded-full px-2 py-1 border text-[11px] font-semibold ${
                              result.hasFulltext
                                ? 'bg-brand-500/10 border-brand-500/20 text-brand-400'
                                : 'bg-forest-800/50 border-white/5 text-forest-500'
                            }`} title={result.hasFulltext ? t("search.fulltextIndexed") : t("search.titleAbstractOnly")}>
                              {result.hasFulltext ? t("search.fullTextBadge") : t("search.abstractBadge")}
                            </span>
                            {/* Provenance : récupéré en direct / nouvel ajout à ce corpus / déjà en base */}
                            {result.isLive ? (
                              <span className="rounded-full px-2 py-1 border text-[11px] bg-amber-500/10 border-amber-500/30 text-amber-300"
                                    title={t("search.liveApiBadgeTooltip")}>
                                {t("search.liveApiBadge")}
                              </span>
                            ) : result.isNew ? (
                              <span className="rounded-full px-2 py-1 border text-[11px] bg-brand-500/10 border-brand-500/30 text-brand-300"
                                    title={t("search.newBadgeTooltip")}>
                                {t("search.newBadge")}
                              </span>
                            ) : (
                              <span className="rounded-full px-2 py-1 border text-[11px] bg-emerald-500/10 border-emerald-500/20 text-emerald-300"
                                    title={t("search.localBaseBadgeTooltip")}>
                                {t("search.localBaseBadge")}
                              </span>
                            )}
                            {/* Statut d'embedding (vectorisation) */}
                            {result.isEmbedded != null && (
                              <span className={`rounded-full px-2 py-1 border text-[11px] ${
                                result.isEmbedded
                                  ? 'bg-violet-500/10 border-violet-500/20 text-violet-300'
                                  : 'bg-forest-800/50 border-white/5 text-forest-500'
                              }`} title={result.isEmbedded ? t("search.embeddedTooltip") : t("search.notEmbeddedTooltip")}>
                                {result.isEmbedded ? t("search.embedded") : t("search.notEmbedded")}
                              </span>
                            )}
                          </div>

                          <p className="mt-4 text-sm leading-6 text-white/80">
                            {result.highlight || result.content}
                          </p>

                          <div className="mt-5 flex flex-wrap gap-2">
                            {(["pertinent", "non-pertinent", "incertain"] as RelevanceLabel[]).map(
                              (tag) => {
                                const isSelected = relevanceMap[result.id] === tag;
                                const colorCls = tag === "pertinent"
                                  ? isSelected ? "border-emerald-400 bg-emerald-500/20 text-emerald-300" : "border-white/10 bg-white/5 text-forest-400 hover:border-emerald-400/50 hover:text-emerald-300"
                                  : tag === "non-pertinent"
                                  ? isSelected ? "border-red-400 bg-red-500/20 text-red-300" : "border-white/10 bg-white/5 text-forest-400 hover:border-red-400/50 hover:text-red-300"
                                  : isSelected ? "border-amber-400 bg-amber-500/20 text-amber-300" : "border-white/10 bg-white/5 text-forest-400 hover:border-amber-400/50 hover:text-amber-300";
                                return (
                                  <button
                                    key={tag}
                                    type="button"
                                    onClick={() =>
                                      setRelevanceMap((prev) => ({
                                        ...prev,
                                        [result.id]: tag,
                                      }))
                                    }
                                    className={`rounded-full border px-3 py-1 text-xs transition font-medium ${colorCls}`}
                                  >
                                    {tag === "pertinent" ? t("search.relevanceRelevant") : tag === "non-pertinent" ? t("search.relevanceNotRelevant") : t("search.relevanceUncertain")}
                                  </button>
                                );
                              },
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
                            className="rounded-xl border border-white/10 px-4 py-2 text-sm text-forest-300 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-30"
                          >
                            {t("search.previous")}
                          </button>
                          <span className="text-sm text-forest-400">
                            {page} / {totalPages}
                          </span>
                          <button
                            type="button"
                            disabled={page === totalPages}
                            onClick={() => setPage((p) => p + 1)}
                            className="rounded-xl border border-white/10 px-4 py-2 text-sm text-forest-300 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-30"
                          >
                            {t("search.next")}
                          </button>
                        </div>
                      )}
                    </div>

                    <aside className="2xl:sticky 2xl:top-8 2xl:self-start">
                      <div className="min-h-[220px] rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
                        {!selectedResult ? (
                          <div className="text-sm leading-6 text-forest-300">
                            {t("search.clickResult")}
                          </div>
                        ) : detailLoading ? (
                          <div className="text-sm leading-6 text-forest-300">
                            {t("search.loadingFullDocument")}
                          </div>
                        ) : (
                          <div className="space-y-5 text-sm text-white/80">
                            <div>
                              <p className="text-xs uppercase tracking-[0.2em] text-brand-300">
                                {t("search.selectedArticleDetail")}
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
                                  className="inline-flex items-center gap-2 rounded-xl border border-white/10 px-3 py-2 text-sm text-white/80 hover:bg-white/10"
                                >
                                  {t("search.openSource")}
                                  <ExternalLink size={16} />
                                </a>
                              </div>
                            )}

                            <section>
                              <h3 className="mb-2 font-medium text-white">{t("search.excerpt")}</h3>
                              <p className="rounded-2xl border border-white/10 bg-white/5 p-4 leading-6">
                                {detailView?.excerpt || "—"}
                              </p>
                            </section>

                            <section>
                              <h3 className="mb-2 font-medium text-white">{t("search.abstract")}</h3>
                              <p className="rounded-2xl border border-white/10 bg-white/5 p-4 leading-6">
                                {detailView?.abstract || "—"}
                              </p>
                            </section>

                            {evidenceSummary && (
                              <GesicaSignalsPanel summary={evidenceSummary} />
                            )}

                            <section>
                              <h3 className="mb-2 font-medium text-white">{t("search.metadata")}</h3>
                              <dl className="grid grid-cols-1 gap-3 rounded-2xl border border-white/10 bg-white/5 p-4">
                                <div><dt className="text-forest-400">{t("search.metaId")}</dt><dd>{detailView?.id ?? "—"}</dd></div>
                                <div><dt className="text-forest-400">{t("search.metaSource")}</dt><dd>{detailView?.source}</dd></div>
                                <div><dt className="text-forest-400">{t("search.metaYear")}</dt><dd>{detailView?.year}</dd></div>
                                <div><dt className="text-forest-400">{t("search.metaExternalId")}</dt><dd>{detailView?.externalId}</dd></div>
                                <div><dt className="text-forest-400">{t("search.metaProject")}</dt><dd>{detailView?.projectContext}</dd></div>
                                <div><dt className="text-forest-400">{t("search.metaType")}</dt><dd>{detailView?.sourceType}</dd></div>
                                <div><dt className="text-forest-400">{t("search.metaDisease")}</dt><dd>{detailView?.disease}</dd></div>
                                <div><dt className="text-forest-400">{t("search.metaScenario")}</dt><dd>{detailView?.scenario}</dd></div>
                                <div><dt className="text-forest-400">{t("search.metaGeography")}</dt><dd>{detailView?.geography}</dd></div>
                                <div><dt className="text-forest-400">{t("search.metaEvidence")}</dt><dd>{detailView?.evidence}</dd></div>
                                <div><dt className="text-forest-400">{t("search.metaChunks")}</dt><dd>{detailView?.chunkCount}</dd></div>
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
