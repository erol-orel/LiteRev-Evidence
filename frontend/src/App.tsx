import { useEffect, useMemo, useState } from "react";
import { Activity, BarChart2, BookOpen, Download, ExternalLink, RotateCcw, Zap, CheckSquare, XCircle, CheckCircle, HelpCircle, ArrowDown, Cloud, MapPin, AlertTriangle, Users, Pill, Radio, RefreshCw, ChevronDown, ChevronUp } from "lucide-react";

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
  fetchTerrainMeteo,
  fetchTerrainGeo,
  fetchTerrainEpidemic,
  fetchTerrainDemographics,
  fetchTerrainPharmacies,
  fetchTerrainInformalSignals,
  fetchTerrainClimate,
  fetchDemandForecast,
  fetchEpidemicEarlyWarning,
  fetchResponseTimeOptimization,
  fetchFulltextStats,
  type CorpusStats,
  type DocumentDetailResponse,
  type EvidenceSummaryResponse,
  type FilterOptions,
  type GesicaScenario,
  type GesicaStats,
  type AskResponse,
  type ScreeningDocument,
  type PrismaFlow,
  type TerrainMeteo,
  type TerrainGeo,
  type TerrainEpidemic,
  type TerrainDemographics,
  type TerrainPharmacies,
  type TerrainInformalSignals,
  type TerrainClimate,
  type DemandForecastResponse,
  type EpidemicEarlyWarningResponse,
  type EpidemicDiseaseResult,
  type ResponseTimeOptimizationResponse,
  type ResponseTimeAssignment,
  type FulltextStats,
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

type AppTab = "search" | "scenarios" | "stats" | "assistant" | "screening" | "terrain";

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
      .catch((err) => setError(err.message || "Erreur de chargement des données terrain."))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadAll();
  }, []);

  const alertColors: Record<string, string> = {
    none: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
    warning: "border-amber-500/30 bg-amber-500/10 text-amber-300",
    danger: "border-rose-500/30 bg-rose-500/10 text-rose-300",
  };
  const riskColors: Record<string, string> = {
    low: "text-emerald-300",
    moderate: "text-amber-300",
    high: "text-rose-300",
  };
  const statusColors: Record<string, string> = {
    under_threshold: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
    warning: "bg-amber-500/10 text-amber-300 border-amber-500/30",
    epidemic: "bg-rose-500/10 text-rose-300 border-rose-500/30",
  };
  const trendIcons: Record<string, string> = {
    increasing: "↑",
    stable: "→",
    decreasing: "↓",
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16 text-slate-400">
        <RotateCcw size={18} className="mr-2 animate-spin" />
        Chargement des données terrain en temps réel...
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
          <Cloud size={20} className="text-sky-400" />
          <div>
            <h2 className="text-xl font-semibold text-white">Données Terrain — Grand Genève</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              6 sources publiques actives — Météo, Routage, Épidémie, Démographie, Pharmacies, Signaux informels
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-500">Actualisé {lastRefresh.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" })}</span>
          <button
            onClick={loadAll}
            className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-300 hover:bg-white/10 transition"
          >
            <RefreshCw size={12} />
            Actualiser
          </button>
        </div>
      </div>

      {/* Grille de KPIs sources */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
        {[
          { label: "Météo", icon: <Cloud size={14} />, color: "text-sky-400", active: !!meteo },
          { label: "Routage", icon: <MapPin size={14} />, color: "text-violet-400", active: !!geo },
          { label: "Épidémie", icon: <Activity size={14} />, color: "text-emerald-400", active: !!epidemic },
          { label: "Démographie", icon: <Users size={14} />, color: "text-amber-400", active: !!demographics },
          { label: "Pharmacies", icon: <Pill size={14} />, color: "text-rose-400", active: !!pharmacies },
          { label: "Signaux", icon: <Radio size={14} />, color: "text-cyan-400", active: !!informalSignals },
          { label: "Copernicus", icon: <Zap size={14} />, color: "text-orange-400", active: !!climate },
        ].map((s) => (
          <div key={s.label} className={`rounded-2xl border p-3 text-center transition ${
            s.active ? "border-white/10 bg-white/5" : "border-white/5 bg-white/2 opacity-40"
          }`}>
            <div className={`flex justify-center mb-1 ${s.color}`}>{s.icon}</div>
            <p className="text-xs text-slate-300 font-medium">{s.label}</p>
            <p className={`text-[10px] mt-0.5 ${s.active ? "text-emerald-400" : "text-slate-500"}`}>
              {s.active ? "● Actif" : "○ Inactif"}
            </p>
          </div>
        ))}
      </div>

      {/* Météo */}
      {meteo && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <div className="flex items-center justify-between mb-4">
            <h3 className="flex items-center gap-2 text-lg font-semibold text-white">
              <Cloud size={16} className="text-sky-400" />
              Météo — {meteo.station}
            </h3>
            <span className={`rounded-full border px-3 py-1 text-xs font-medium ${alertColors[meteo.alert_level] ?? alertColors.none}`}>
              {meteo.alert_level === "none" ? "Aucune alerte" : meteo.alert_level === "warning" ? "Vigilance" : "Danger"}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 mb-4">
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-sky-300">{meteo.temperature}°C</p>
              <p className="mt-1 text-xs text-slate-400">Température</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-sky-300">{meteo.apparent_temperature}°C</p>
              <p className="mt-1 text-xs text-slate-400">Ressenti</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-sky-300">{meteo.humidity}%</p>
              <p className="mt-1 text-xs text-slate-400">Humidité</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-sky-300">{meteo.wind_speed} km/h</p>
              <p className="mt-1 text-xs text-slate-400">Vent</p>
            </div>
          </div>
          <div className={`rounded-2xl border p-3 text-sm ${alertColors[meteo.alert_level] ?? alertColors.none}`}>
            <p className="font-medium">{meteo.alert_description}</p>
            <p className="mt-1 opacity-80">{meteo.impact_on_ems}</p>
          </div>
          <p className="mt-2 text-xs text-slate-500 italic">{meteo.architecture_note}</p>
        </div>
      )}

      {/* Géo / Routage */}
      {geo && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <h3 className="flex items-center gap-2 text-lg font-semibold text-white mb-4">
            <MapPin size={16} className="text-violet-400" />
            Routage Transfrontalier — {geo.origin.label} → {geo.destination.label}
          </h3>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 mb-4">
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-violet-300">{geo.distance_km} km</p>
              <p className="mt-1 text-xs text-slate-400">Distance</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-violet-300">{geo.base_duration_min} min</p>
              <p className="mt-1 text-xs text-slate-400">Durée de base</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-amber-300">{geo.cross_border_delay_min} min</p>
              <p className="mt-1 text-xs text-slate-400">Délai douane</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-rose-300">{geo.total_estimated_response_time_min} min</p>
              <p className="mt-1 text-xs text-slate-400">Temps total estimé</p>
            </div>
          </div>
          <div className="rounded-2xl border border-violet-500/30 bg-violet-500/10 p-3 text-sm text-violet-300">
            <p className="font-medium">Action de coordination</p>
            <p className="mt-1 opacity-80">{geo.coordination_action}</p>
          </div>
          <p className="mt-2 text-xs text-slate-500 italic">{geo.architecture_note}</p>
        </div>
      )}

      {/* Épidémie */}
      {epidemic && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <div className="flex items-center justify-between mb-4">
            <h3 className="flex items-center gap-2 text-lg font-semibold text-white">
              <Activity size={16} className="text-emerald-400" />
              Surveillance Épidémique — {epidemic.region}
            </h3>
            <span className={`text-lg font-bold ${riskColors[epidemic.global_ems_impact_risk] ?? "text-white"}`}>
              Risque EMS : {epidemic.global_ems_impact_risk.toUpperCase()}
            </span>
          </div>
          <div className="space-y-3 mb-4">
            {epidemic.diseases.map((d) => (
              <div key={d.name} className="rounded-2xl border border-white/10 bg-slate-900/60 p-3">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-white text-sm">{d.name}</span>
                  <div className="flex items-center gap-2">
                    <span className={`rounded-full border px-2 py-0.5 text-xs ${statusColors[d.status] ?? ""}`}>
                      {d.status === "under_threshold" ? "Sous le seuil" : d.status === "warning" ? "Vigilance" : "Épidémie"}
                    </span>
                    <span className={`text-sm font-bold ${
                      d.trend === "increasing" ? "text-rose-300" : d.trend === "decreasing" ? "text-emerald-300" : "text-slate-300"
                    }`}>{trendIcons[d.trend]}</span>
                  </div>
                </div>
                <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-slate-400">
                  <span>France : <span className="text-white font-medium">{d.incidence_per_100k_france}/100k</span></span>
                  <span>Suisse : <span className="text-white font-medium">{d.incidence_per_100k_switzerland}/100k</span></span>
                  <span>Seuil : <span className="text-white font-medium">{d.epidemic_threshold}/100k</span></span>
                </div>
              </div>
            ))}
          </div>
          <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/10 p-3 text-sm text-emerald-300">
            <p className="font-medium">Recommandation opérationnelle</p>
            <p className="mt-1 opacity-80">{epidemic.recommended_action}</p>
          </div>
          <p className="mt-2 text-xs text-slate-500 italic">{epidemic.architecture_note}</p>
        </div>
      )}

      {/* Démographie */}
      {demographics && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <h3 className="flex items-center gap-2 text-lg font-semibold text-white mb-4">
            <Users size={16} className="text-amber-400" />
            Démographie — {demographics.commune} ({demographics.postal_code})
          </h3>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 mb-4">
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-amber-300">{demographics.population.toLocaleString("fr-FR")}</p>
              <p className="mt-1 text-xs text-slate-400">Population</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-amber-300">{demographics.density_per_km2}</p>
              <p className="mt-1 text-xs text-slate-400">Hab/km²</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-amber-300">{demographics.age_over_65_pct}%</p>
              <p className="mt-1 text-xs text-slate-400">&gt;65 ans</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-rose-300">×{demographics.ems_risk_multiplier}</p>
              <p className="mt-1 text-xs text-slate-400">Multiplicateur risque EMS</p>
            </div>
          </div>
          <p className="text-xs text-slate-500 italic">{demographics.architecture_note}</p>
        </div>
      )}

      {/* Pharmacies & Médicaments */}
      {pharmacies && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <h3 className="flex items-center gap-2 text-lg font-semibold text-white mb-4">
            <Pill size={16} className="text-rose-400" />
            Pharmacies de Garde & Alertes Médicaments
          </h3>
          {pharmacies.critical_medication_alerts.length > 0 && (
            <div className="mb-4 space-y-2">
              <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Alertes médicaments critiques</h4>
              {pharmacies.critical_medication_alerts.map((alert, i) => (
                <div key={i} className={`rounded-2xl border p-3 text-sm ${
                  alert.status === "rupture" ? "border-rose-500/30 bg-rose-500/10 text-rose-300" :
                  alert.status === "tension" ? "border-amber-500/30 bg-amber-500/10 text-amber-300" :
                  "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
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
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Pharmacies à proximité ({pharmacies.pharmacies_nearby.length})</h4>
            {pharmacies.pharmacies_nearby.map((ph, i) => (
              <div key={i} className="rounded-2xl border border-white/10 bg-slate-900/60 p-3 flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-white">{ph.name}</p>
                  <p className="text-xs text-slate-400">{ph.street}, {ph.city}</p>
                </div>
                <div className="text-right">
                  <span className={`text-xs font-medium ${
                    ph.is_dispensary ? "text-emerald-300" : "text-slate-400"
                  }`}>{ph.is_dispensary ? "✓ Dispensaire" : "Pharmacie"}</span>
                  <p className="text-xs text-slate-500">{ph.opening_hours}</p>
                </div>
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs text-slate-500 italic">{pharmacies.architecture_note}</p>
        </div>
      )}

      {/* Signaux Informels */}
      {informalSignals && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <h3 className="flex items-center gap-2 text-lg font-semibold text-white mb-4">
            <Radio size={16} className="text-cyan-400" />
            Signaux Informels — ProMED / GDELT
          </h3>
          <div className="space-y-3">
            {informalSignals.active_signals.map((sig) => (
              <div key={sig.id} className={`rounded-2xl border p-4 ${
                sig.severity === "high" ? "border-rose-500/30 bg-rose-500/5" :
                sig.severity === "moderate" ? "border-amber-500/30 bg-amber-500/5" :
                "border-white/10 bg-white/5"
              }`}>
                <div className="flex items-start justify-between gap-3 mb-2">
                  <div>
                    <p className="text-sm font-semibold text-white">{sig.title}</p>
                    <p className="text-xs text-slate-400 mt-0.5">{sig.source} — {sig.date} — {sig.geo_scope}</p>
                  </div>
                  <div className="flex flex-col items-end gap-1 shrink-0">
                    <span className={`rounded-full border px-2 py-0.5 text-xs font-medium ${
                      sig.severity === "high" ? "border-rose-500/30 bg-rose-500/10 text-rose-300" :
                      sig.severity === "moderate" ? "border-amber-500/30 bg-amber-500/10 text-amber-300" :
                      "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                    }`}>{sig.severity.toUpperCase()}</span>
                    <span className="text-xs text-slate-500">Fiabilité {Math.round(sig.reliability_score * 100)}%</span>
                  </div>
                </div>
                <p className="text-sm text-slate-300 leading-5">{sig.content}</p>
                {sig.impact_on_gesica && (
                  <p className="mt-2 text-xs text-cyan-300 italic">→ GESICA : {sig.impact_on_gesica}</p>
                )}
                {sig.impact_on_geoai4ei && (
                  <p className="mt-1 text-xs text-violet-300 italic">→ GeoAI4EI : {sig.impact_on_geoai4ei}</p>
                )}
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs text-slate-500 italic">{informalSignals.architecture_note}</p>
        </div>
      )}

      {/* Copernicus Climate Data Store (CDS) */}
      {climate && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <div className="flex items-center justify-between mb-4">
            <h3 className="flex items-center gap-2 text-lg font-semibold text-white">
              <Zap size={16} className="text-orange-400" />
              Copernicus Climate Data Store (CDS) — ERA5
            </h3>
            <span className={`rounded-full border px-3 py-1 text-xs font-medium ${
              climate.api_status.includes("verified") ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300" : "border-amber-500/30 bg-amber-500/10 text-amber-300"
            }`}>
              {climate.api_status === "connected_verified" ? "API Connectée" : "Mode simulé / Configuré"}
            </span>
          </div>
          
          {climate.message && (
            <div className="mb-4 rounded-2xl border border-orange-500/20 bg-orange-500/5 p-3 text-xs text-orange-300">
              {climate.message}
            </div>
          )}

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 mb-4">
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-orange-300">{climate.climatology.historical_mean_temp_may_c}°C</p>
              <p className="mt-1 text-xs text-slate-400">Moyenne historique (Mai)</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-orange-300">+{climate.climatology.current_anomaly_c}°C</p>
              <p className="mt-1 text-xs text-slate-400">Anomalie thermique</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-orange-300">{climate.climatology.soil_moisture_deficit_percent}%</p>
              <p className="mt-1 text-xs text-slate-400">Déficit d'humidité des sols</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-rose-300">{climate.climatology.heatwave_hazard_index.toUpperCase()}</p>
              <p className="mt-1 text-xs text-slate-400">Risque canicule</p>
            </div>
          </div>

          <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-4">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Projections climatiques horizon 2030 (Transfrontalier)</h4>
            <ul className="space-y-1.5 text-xs text-slate-300">
              <li>• Augmentation des jours de canicule extrême : <span className="text-white font-medium">+{climate.projections_2030.expected_heatwave_days_increase_per_year} jours/an</span></li>
              <li>• Augmentation des précipitations extrêmes : <span className="text-white font-medium">+{climate.projections_2030.expected_heavy_precipitation_increase_percent}%</span></li>
              <li>• Facteur de vulnérabilité EMS principal : <span className="text-white font-medium">{climate.projections_2030.ems_vulnerability_factor.replace(/_/g, " ")}</span></li>
            </ul>
          </div>
          
          <p className="mt-3 text-xs text-slate-500 italic">{climate.source}</p>
        </div>
      )}
    </div>
  );
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

function StatsView({ corpusStats, gesicaStats, fulltextStats }: { corpusStats: CorpusStats | null; gesicaStats: GesicaStats | null; fulltextStats: FulltextStats | null }) {
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

      {fulltextStats && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold text-white">
            <BookOpen size={18} className="text-emerald-400" />
            Couverture textuelle &amp; Hybrid Search
          </h2>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-4 text-center">
              <p className="text-2xl font-bold text-emerald-300">{fulltextStats.corpus.docs_with_fulltext.toLocaleString()}</p>
              <p className="mt-1 text-xs text-slate-400">Full Text</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-4 text-center">
              <p className="text-2xl font-bold text-slate-300">{fulltextStats.corpus.docs_abstract_only.toLocaleString()}</p>
              <p className="mt-1 text-xs text-slate-400">Abstract only</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-4 text-center">
              <p className={`text-2xl font-bold ${fulltextStats.corpus.fulltext_coverage_pct >= 20 ? 'text-emerald-300' : 'text-amber-300'}`}>
                {fulltextStats.corpus.fulltext_coverage_pct.toFixed(1)}%
              </p>
              <p className="mt-1 text-xs text-slate-400">Couverture</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-4 text-center">
              <p className={`text-2xl font-bold ${fulltextStats.hybrid_search.active ? 'text-cyan-300' : 'text-rose-300'}`}>
                {fulltextStats.hybrid_search.active ? 'HYBRID' : 'LEXICAL'}
              </p>
              <p className="mt-1 text-xs text-slate-400">Mode recherche</p>
            </div>
          </div>
          {fulltextStats.by_source && fulltextStats.by_source.length > 0 && (
            <div className="mt-4">
              <h3 className="mb-2 text-sm font-medium text-slate-300">Full Text par source</h3>
              <div className="space-y-1">
                {fulltextStats.by_source.slice(0, 8).map((s) => (
                  <div key={s.source} className="flex items-center justify-between rounded-xl border border-white/10 bg-slate-900/40 px-3 py-1.5 text-xs">
                    <span className="text-slate-300 capitalize">{s.source}</span>
                    <span className="font-mono text-emerald-300">{s.fulltext_count} / {s.total_count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
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

function ScenariosView({ scenarios, loading, error }: { scenarios: GesicaScenario[]; loading?: boolean; error?: string | null }) {
  const [selectedCluster, setSelectedCluster] = useState<string>("all");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16 text-slate-400">
        <RotateCcw size={18} className="mr-2 animate-spin" />
        Chargement des scénarios GESICA...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-16 gap-3">
        <div className="rounded-2xl border border-red-500/30 bg-red-500/10 px-6 py-4 text-red-300 max-w-xl text-center">
          <p className="font-semibold mb-1">Erreur de chargement des scénarios GESICA</p>
          <p className="text-sm text-red-400 font-mono break-all">{error}</p>
          <p className="text-xs text-slate-400 mt-2">Vérifiez que le service API est démarré sur app-01 et que <code>/api/gesica/scenarios</code> répond correctement.</p>
        </div>
      </div>
    );
  }

  if (scenarios.length === 0) {
    return (
      <div className="flex items-center justify-center py-16 text-slate-400">
        <RotateCcw size={18} className="mr-2 animate-spin" />
        Chargement des scénarios GESICA...
      </div>
    );
  }

  // Extraire les clusters uniques
  const clusters = ["all", ...Array.from(new Set(scenarios.map(s => s.cluster))).sort()];
  const filtered = selectedCluster === "all" ? scenarios : scenarios.filter(s => s.cluster === selectedCluster);
  const withArticles = filtered.filter(s => s.articleCount > 0);
  const withoutArticles = filtered.filter(s => s.articleCount === 0);

  const clusterColors: Record<string, string> = {
    "Prévention & Risques": "border-violet-500/30 bg-violet-500/10 text-violet-300",
    "Opérations EMS": "border-sky-500/30 bg-sky-500/10 text-sky-300",
    "Triage & Clinique": "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
    "Soins Centrés Patient": "border-rose-500/30 bg-rose-500/10 text-rose-300",
    "Surveillance & Crise": "border-amber-500/30 bg-amber-500/10 text-amber-300",
    "Systèmes & IA": "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
  };

  // Widget de prévision de la demande EMS (Scénario 1 : demand-forecasting)
  const DemandForecastWidget = () => {
    const [forecast, setForecast] = useState<DemandForecastResponse | null>(null);
    const [loadingForecast, setLoadingForecast] = useState(false);
    const [forecastError, setForecastError] = useState<string | null>(null);

    const loadForecast = () => {
      setLoadingForecast(true);
      setForecastError(null);
      fetchDemandForecast()
        .then(setForecast)
        .catch((e) => setForecastError(e.message))
        .finally(() => setLoadingForecast(false));
    };

    if (!forecast && !loadingForecast && !forecastError) {
      return (
        <button
          onClick={loadForecast}
          className="w-full rounded-xl border border-violet-500/30 bg-violet-500/10 px-4 py-3 text-sm text-violet-300 hover:bg-violet-500/20 transition flex items-center justify-center gap-2"
        >
          <BarChart2 size={14} />
          Lancer la prédiction J+7 (Prophet + LightGBM)
        </button>
      );
    }

    if (loadingForecast) {
      return (
        <div className="flex items-center justify-center py-4 text-violet-300 text-sm gap-2">
          <RotateCcw size={14} className="animate-spin" />
          Entraînement du modèle et calcul des prédictions...
        </div>
      );
    }

    if (forecastError) {
      return (
        <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">
          Erreur : {forecastError}
          <button onClick={loadForecast} className="ml-2 underline">Reessayer</button>
        </div>
      );
    }

    if (!forecast) return null;

    const riskColors = { NORMAL: "text-emerald-400", "ÉLEVÉ": "text-amber-400", CRITIQUE: "text-red-400" };
    const riskBg = { NORMAL: "bg-emerald-500/10 border-emerald-500/20", "ÉLEVÉ": "bg-amber-500/10 border-amber-500/20", CRITIQUE: "bg-red-500/10 border-red-500/20" };

    return (
      <div className="space-y-3">
        {/* Métadonnées du modèle */}
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-slate-400">
          <span>Modèle : <span className="text-slate-300">{forecast.model}</span></span>
          <span>Temp. actuelle : <span className="text-slate-300">{forecast.input_features.current_temperature}°C</span></span>
          <span>Index épidémique : <span className="text-slate-300">{forecast.input_features.epidemic_index}</span></span>
          {forecast.status === "fallback" && (
            <span className="text-amber-400 border border-amber-500/20 bg-amber-500/10 px-1.5 py-0.5 rounded">Fallback analytique</span>
          )}
          <button onClick={loadForecast} className="ml-auto text-violet-400 hover:text-violet-300 flex items-center gap-1">
            <RefreshCw size={10} /> Actualiser
          </button>
        </div>

        {/* Grille des 7 jours */}
        <div className="grid grid-cols-7 gap-1">
          {forecast.predictions.map((pred) => (
            <div
              key={pred.ds}
              className={`rounded-xl border p-2 text-center ${riskBg[pred.risk_level] ?? "bg-white/5 border-white/10"}`}
              title={pred.recommendation}
            >
              <p className="text-[9px] text-slate-400 truncate">{pred.date.split(' ')[0]}</p>
              <p className="text-[10px] text-slate-300">{pred.date.split(' ')[1]}/{pred.date.split(' ')[2]?.slice(0, 3)}</p>
              <p className={`text-sm font-bold mt-1 ${riskColors[pred.risk_level] ?? "text-white"}`}>{pred.demand}</p>
              <p className="text-[9px] text-slate-500">{pred.temp_estimated}°C</p>
              <p className={`text-[9px] font-semibold mt-0.5 ${riskColors[pred.risk_level] ?? "text-white"}`}>{pred.risk_level}</p>
            </div>
          ))}
        </div>

        {/* Recommandation du jour le plus à risque */}
        {forecast.predictions.some(p => p.risk_level !== "NORMAL") && (
          <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-xs text-amber-300">
            <AlertTriangle size={10} className="inline mr-1" />
            {forecast.predictions.find(p => p.risk_level !== "NORMAL")?.recommendation}
          </div>
        )}
      </div>
    );
  };

  // Widget de détection précoce d'épidémies (Scénario : epidemic-early-warning)
  const EpidemicEarlyWarningWidget = () => {
    const [data, setData] = useState<EpidemicEarlyWarningResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const load = () => {
      setLoading(true);
      setError(null);
      fetchEpidemicEarlyWarning()
        .then(setData)
        .catch((e) => setError(e.message))
        .finally(() => setLoading(false));
    };

    if (!data && !loading && !error) {
      return (
        <button onClick={load} className="w-full rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-300 hover:bg-emerald-500/20 transition flex items-center justify-center gap-2">
          <Activity size={14} />
          Lancer la surveillance épidémique J+14 (SARIMAX + Sentinelles)
        </button>
      );
    }
    if (loading) return <div className="flex items-center justify-center py-4 text-emerald-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Analyse des données Sentinelles FR...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;

    const alertColors = { NORMAL: "text-emerald-400", VIGILANCE: "text-amber-400", "ÉPIDÉMIE": "text-red-400" };
    const alertBg = { NORMAL: "bg-emerald-500/10 border-emerald-500/20", VIGILANCE: "bg-amber-500/10 border-amber-500/20", "ÉPIDÉMIE": "bg-red-500/10 border-red-500/20" };
    const diseases = Object.values(data.diseases) as EpidemicDiseaseResult[];

    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-slate-400">
          <span>Modèle : <span className="text-slate-300">{data.model}</span></span>
          <span>Région : <span className="text-slate-300">{data.region}</span></span>
          <span className={`font-semibold px-2 py-0.5 rounded border ${alertBg[data.overall_alert_level] ?? ""} ${alertColors[data.overall_alert_level] ?? ""}`}>
            Alerte globale : {data.overall_alert_level}
          </span>
          {data.status === "fallback" && <span className="text-amber-400 border border-amber-500/20 bg-amber-500/10 px-1.5 py-0.5 rounded">Fallback analytique</span>}
          <button onClick={load} className="ml-auto text-emerald-400 hover:text-emerald-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>

        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {diseases.map((d) => (
            <div key={d.disease} className={`rounded-xl border p-3 ${alertBg[d.max_alert_14d] ?? "bg-white/5 border-white/10"}`}>
              <p className="text-xs font-semibold text-white truncate">{d.label}</p>
              <p className={`text-lg font-bold mt-1 ${alertColors[d.max_alert_14d] ?? "text-white"}`}>{d.current_incidence}</p>
              <p className="text-[9px] text-slate-400">/100k — seuil {d.epidemic_threshold}</p>
              <p className={`text-[9px] font-semibold mt-1 ${alertColors[d.max_alert_14d] ?? "text-white"}`}>{d.max_alert_14d}</p>
            </div>
          ))}
        </div>

        <div className={`rounded-xl border px-3 py-2 text-xs ${alertBg[data.overall_alert_level] ?? "border-white/10 bg-white/5"} ${alertColors[data.overall_alert_level] ?? "text-slate-300"}`}>
          {data.global_recommendation}
        </div>
      </div>
    );
  };

  // Widget d'optimisation des temps de réponse (Scénario : response-time-optimization)
  const ResponseTimeWidget = () => {
    const [data, setData] = useState<ResponseTimeOptimizationResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const load = () => {
      setLoading(true);
      setError(null);
      fetchResponseTimeOptimization()
        .then(setData)
        .catch((e) => setError(e.message))
        .finally(() => setLoading(false));
    };

    if (!data && !loading && !error) {
      return (
        <button onClick={load} className="w-full rounded-xl border border-sky-500/30 bg-sky-500/10 px-4 py-3 text-sm text-sky-300 hover:bg-sky-500/20 transition flex items-center justify-center gap-2">
          <MapPin size={14} />
          Optimiser les temps de réponse EMS (OSRM + Open-Meteo)
        </button>
      );
    }
    if (loading) return <div className="flex items-center justify-center py-4 text-sky-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Calcul des itinéraires optimaux via OSRM...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;

    const statusColors = { OPTIMAL: "text-emerald-400", ACCEPTABLE: "text-amber-400", DÉGRADÉ: "text-red-400" };
    const statusBg = { OPTIMAL: "bg-emerald-500/10 border-emerald-500/20", ACCEPTABLE: "bg-amber-500/10 border-amber-500/20", DÉGRADÉ: "bg-red-500/10 border-red-500/20" };

    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-slate-400">
          <span>Temp. : <span className="text-slate-300">{data.weather.temperature}°C</span></span>
          <span>Facteur météo : <span className="text-slate-300">×{data.weather.weather_factor}</span></span>
          <span>Couverture : <span className="text-emerald-300 font-semibold">{data.metrics.coverage_rate_pct}%</span></span>
          <span>Temps moyen : <span className="text-sky-300 font-semibold">{data.metrics.mean_response_time_min} min</span></span>
          {data.metrics.degraded_zones > 0 && (
            <span className="text-red-400 border border-red-500/20 bg-red-500/10 px-1.5 py-0.5 rounded">{data.metrics.degraded_zones} zone(s) dégradée(s)</span>
          )}
          <button onClick={load} className="ml-auto text-sky-400 hover:text-sky-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>

        <div className="space-y-1.5 max-h-48 overflow-y-auto">
          {(data.assignments as ResponseTimeAssignment[]).map((a) => (
            <div key={a.zone_id} className={`rounded-xl border px-3 py-2 flex items-center justify-between gap-2 ${statusBg[a.response_status] ?? "bg-white/5 border-white/10"}`}>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-semibold text-white truncate">{a.zone_label}</p>
                <p className="text-[9px] text-slate-400 truncate">{a.base_label} {a.cross_border ? `→ via ${a.border_crossing}` : ""}</p>
              </div>
              <div className="text-right shrink-0">
                <p className={`text-sm font-bold ${statusColors[a.response_status] ?? "text-white"}`}>{a.total_response_time_min} min</p>
                <p className={`text-[9px] font-semibold ${statusColors[a.response_status] ?? "text-white"}`}>{a.response_status}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="rounded-xl border border-sky-500/20 bg-sky-500/5 px-3 py-2 text-xs text-sky-300">
          {data.global_recommendation}
        </div>
      </div>
    );
  };

  const ScenarioCard = ({ scenario }: { scenario: GesicaScenario }) => {
    const isExpanded = expandedId === scenario.id;
    const hasArticles = scenario.articleCount > 0;
    return (
      <div className={`rounded-3xl border p-5 shadow-xl transition ${
        hasArticles ? "border-white/10 bg-white/5" : "border-white/5 bg-white/2 opacity-60"
      }`}>
        <div
          className="flex items-start gap-3 cursor-pointer"
          onClick={() => setExpandedId(isExpanded ? null : scenario.id)}
        >
          <div className="mt-1 rounded-xl border border-cyan-500/20 bg-cyan-500/10 p-2 shrink-0">
            <Activity size={14} className="text-cyan-400" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="text-base font-semibold text-white">{scenario.title}</h3>
              <span className={`rounded-full border px-2 py-0.5 text-xs font-medium ${
                clusterColors[scenario.cluster] ?? "border-white/10 bg-white/5 text-slate-400"
              }`}>{scenario.cluster}</span>
              {hasArticles ? (
                <span className="rounded-full bg-cyan-500/10 border border-cyan-500/20 px-2 py-0.5 text-xs text-cyan-300 font-mono">
                  {scenario.articleCount} articles
                </span>
              ) : (
                <span className="rounded-full bg-slate-700/40 border border-white/5 px-2 py-0.5 text-xs text-slate-500">
                  0 articles
                </span>
              )}
            </div>
            <p className="mt-1 text-sm leading-5 text-slate-400 line-clamp-2">{scenario.description}</p>
          </div>
          <div className="shrink-0 text-slate-500">
            {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </div>
        </div>

        {isExpanded && (
          <div className="mt-4 space-y-4 border-t border-white/10 pt-4">
            {/* Living Evidence Note */}
            <div className={`rounded-2xl border px-3 py-2 text-xs ${
              hasArticles
                ? "border-cyan-500/20 bg-cyan-500/5 text-cyan-300"
                : "border-white/5 bg-white/2 text-slate-500"
            }`}>
              <RefreshCw size={10} className="inline mr-1" />
              {scenario.livingEvidenceNote}
            </div>

            {/* Actions recommandées */}
            <div>
              <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">Actions recommandées</h4>
              <ul className="space-y-1.5">
                {scenario.recommendedActions.map((action, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-slate-200">
                    <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-cyan-400" />
                    {action}
                  </li>
                ))}
              </ul>
            </div>

            {/* Bouton Modèle Prédictif pour demand-forecasting */}
            {scenario.id === "demand-forecasting" && (
              <div className="rounded-2xl border border-violet-500/20 bg-violet-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <BarChart2 size={14} className="text-violet-400" />
                    <span className="text-xs font-semibold text-violet-300 uppercase tracking-wider">Modèle Prédictif — Demande EMS J+7</span>
                  </div>
                  <span className="text-[10px] text-violet-400 bg-violet-500/10 border border-violet-500/20 px-2 py-0.5 rounded-full">Prophet + LightGBM</span>
                </div>
                <DemandForecastWidget />
              </div>
            )}

            {/* Widget Epidemic Early Warning */}
            {scenario.id === "epidemic-early-warning" && (
              <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Activity size={14} className="text-emerald-400" />
                    <span className="text-xs font-semibold text-emerald-300 uppercase tracking-wider">Modèle Prédictif — Surveillance Épidémique J+14</span>
                  </div>
                  <span className="text-[10px] text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full">SARIMAX + Sentinelles FR</span>
                </div>
                <EpidemicEarlyWarningWidget />
              </div>
            )}

            {/* Widget Response Time Optimization */}
            {scenario.id === "response-time-optimization" && (
              <div className="rounded-2xl border border-sky-500/20 bg-sky-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <MapPin size={14} className="text-sky-400" />
                    <span className="text-xs font-semibold text-sky-300 uppercase tracking-wider">Modèle Prédictif — Optimisation Temps de Réponse</span>
                  </div>
                  <span className="text-[10px] text-sky-400 bg-sky-500/10 border border-sky-500/20 px-2 py-0.5 rounded-full">OSRM + Open-Meteo</span>
                </div>
                <ResponseTimeWidget />
              </div>
            )}

            {/* Articles associés */}
            {scenario.relevantArticles.length > 0 && (
              <div>
                <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Articles récents ({scenario.articleCount} total, 5 affichés)
                </h4>
                <div className="space-y-2">
                  {scenario.relevantArticles.map((article) => (
                    <div key={article.id} className="rounded-xl border border-white/10 bg-slate-900/40 px-3 py-2">
                      <p className="text-sm font-medium text-slate-200 leading-5">{article.title}</p>
                      {article.authors && (
                        <p className="text-xs text-slate-400 mt-0.5 line-clamp-1">Par {article.authors}</p>
                      )}
                      <div className="mt-1.5 flex items-center gap-2 flex-wrap">
                        {article.journal && (
                          <span className="text-xs font-semibold text-slate-300 bg-slate-800 px-1.5 py-0.5 rounded">
                            {article.journal}
                          </span>
                        )}
                        <span className="text-xs text-slate-400">{article.source}</span>
                        {article.year && <span className="text-xs text-slate-500">{article.year}</span>}
                        
                        {article.study_design && (
                          <span className="text-xs text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded font-mono">
                            {article.study_design}
                          </span>
                        )}
                        
                        {/* Badge couverture textuelle */}
                        <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded border ${
                          article.has_fulltext
                            ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20'
                            : 'text-slate-500 bg-slate-800/50 border-white/5'
                        }`} title={article.has_fulltext ? 'Texte intégral indexé' : 'Titre + résumé uniquement'}>
                          {article.has_fulltext ? 'Full Text' : 'Abstract'}
                        </span>

                        {article.doi && (
                          <a
                            href={`https://doi.org/${article.doi}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-[10px] text-slate-400 hover:text-cyan-400 font-mono"
                            title={`DOI: ${article.doi}`}
                          >
                            DOI
                          </a>
                        )}

                        {article.open_access && (
                          <span className="text-xs text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                            OA
                          </span>
                        )}
                        
                        {article.citation_count !== null && article.citation_count > 0 && (
                          <span className="text-xs text-slate-400">
                            {article.citation_count} citation{article.citation_count > 1 ? 's' : ''}
                          </span>
                        )}

                        {article.url && (
                          <a
                            href={article.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs text-cyan-400 hover:underline flex items-center gap-1 ml-auto"
                          >
                            Lien <ExternalLink size={10} />
                          </a>
                        )}
                      </div>
                      {article.keywords && (
                        <div className="mt-1 flex items-center gap-1 flex-wrap">
                          {article.keywords.split(',').slice(0, 4).map((kw, idx) => (
                            <span key={idx} className="text-[10px] text-slate-500 bg-slate-800/50 px-1 rounded">
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

  return (
    <div className="space-y-6">
      {/* En-tête */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity size={20} className="text-cyan-400" />
          <div>
            <h2 className="text-xl font-semibold text-white">Scénarios GESICA — Living Evidence Review</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              {scenarios.length} scénarios · {scenarios.reduce((a, s) => a + s.articleCount, 0).toLocaleString()} articles indexés · Mis à jour automatiquement
            </p>
          </div>
        </div>
        <span className="rounded-full border border-cyan-500/20 bg-cyan-500/10 px-3 py-1 text-xs text-cyan-300 font-medium">
          <RefreshCw size={10} className="inline mr-1" />
          Living Review
        </span>
      </div>

      {/* Filtre par cluster */}
      <div className="flex flex-wrap gap-2">
        {clusters.map((cluster) => (
          <button
            key={cluster}
            onClick={() => setSelectedCluster(cluster)}
            className={`rounded-xl border px-3 py-1.5 text-xs font-medium transition ${
              selectedCluster === cluster
                ? "border-cyan-400 bg-cyan-500/20 text-white"
                : "border-white/10 bg-white/5 text-slate-400 hover:bg-white/10"
            }`}
          >
            {cluster === "all" ? `Tous (${scenarios.length})` : `${cluster} (${scenarios.filter(s => s.cluster === cluster).length})`}
          </button>
        ))}
      </div>

      {/* Scénarios avec articles */}
      {withArticles.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-cyan-400" />
            Scénarios actifs ({withArticles.length})
          </h3>
          {withArticles.map((s) => <ScenarioCard key={s.id} scenario={s} />)}
        </div>
      )}

      {/* Scénarios sans articles */}
      {withoutArticles.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-slate-500 flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-slate-600" />
            En attente d'articles ({withoutArticles.length})
          </h3>
          {withoutArticles.map((s) => <ScenarioCard key={s.id} scenario={s} />)}
        </div>
      )}
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
  const [fulltextStats, setFulltextStats] = useState<FulltextStats | null>(null);
  const [gesicaScenarios, setGesicaScenarios] = useState<GesicaScenario[]>([]);
  const [loadingScenarios, setLoadingScenarios] = useState(false);
  const [scenariosError, setScenariosError] = useState<string | null>(null);
  
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
      fetchFulltextStats().then(setFulltextStats).catch(console.error);
    }
    if (activeTab === "scenarios") {
      setLoadingScenarios(true);
      setScenariosError(null);
      fetchGesicaScenarios()
        .then((data) => { setGesicaScenarios(data); setLoadingScenarios(false); })
        .catch((err) => { setScenariosError(String(err)); setLoadingScenarios(false); });
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
    { id: "terrain", label: "Données Terrain", icon: <Cloud size={14} className="text-sky-400" /> },
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
          <StatsView corpusStats={corpusStats} gesicaStats={gesicaStats} fulltextStats={fulltextStats} />
        )}

        {activeTab === "terrain" && (
          <TerrainView />
        )}

        {activeTab === "scenarios" && (
          <ScenariosView scenarios={gesicaScenarios} loading={loadingScenarios} error={scenariosError} />
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
                            {/* Badge couverture textuelle */}
                            <span className={`rounded-full px-2 py-1 border text-[11px] font-semibold ${
                              result.chunkType === 'fulltext_section'
                                ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                                : 'bg-slate-800/50 border-white/5 text-slate-500'
                            }`} title={result.chunkType === 'fulltext_section' ? 'Texte intégral indexé' : 'Titre + résumé uniquement'}>
                              {result.chunkType === 'fulltext_section' ? 'Full Text' : 'Abstract'}
                            </span>
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
