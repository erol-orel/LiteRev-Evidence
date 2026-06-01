import { useEffect, useMemo, useState } from "react";
import { ScenarioDetailPage } from "./components/ScenarioDetailPage";
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
  fetchCardiacArrestPrediction,
  fetchHeatwaveEMSImpact,
  fetchStrokeDetection,
  fetchTriageSupport,
  fetchUndertriageRisk,
  fetchTraumaCare,
  fetchMassCasualty,
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
  type CardiacArrestPredictionResponse,
  type OHCAForecastDay,
  type HeatwaveEMSImpactResponse,
  type HeatwaveForecastDay,
  type StrokeDetectionResponse,
  type TriageSupportResponse,
  type UndertriageRiskResponse,
  type TraumaCareResponse,
  type MassCasualtyResponse,
  type FulltextStats,
  fetchClinicalDeterioration,
  fetchCallQualification,
  fetchDispatchDecision,
  fetchPatientPathway,
  fetchAmbulanceDispatch,
  fetchHospitalCapacity,
  fetchSurveillance,
  fetchSurgeManagement,
  fetchResourceAllocation,
  fetchEnvironmentalRisk,
  fetchPandemicPreparedness,
  fetchCrossBorder,
  fetchSituationalAwareness,
  fetchDisasterRisk,
  fetchMCIVictim,
  type ClinicalDeteriorationResponse,
  type CallQualificationResponse,
  type DispatchDecisionResponse,
  type PatientPathwayResponse,
  type AmbulanceDispatchResponse,
  type HospitalCapacityResponse,
  type SurveillanceResponse,
  type SurgeManagementResponse,
  type ResourceAllocationResponse,
  type EnvironmentalRiskResponse,
  type PandemicPreparednessResponse,
  type CrossBorderResponse,
  type SituationalAwarenessResponse,
  type DisasterRiskResponse,
  type MCIVictimResponse,
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
          <Cloud size={20} className="text-brand-400" />
          <div>
            <h2 className="text-xl font-semibold text-white">Données Terrain — Grand Genève</h2>
            <p className="text-xs text-forest-400 mt-0.5">
              6 sources publiques actives — Météo, Routage, Épidémie, Démographie, Pharmacies, Signaux informels
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-forest-500">Actualisé {lastRefresh.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" })}</span>
          <button
            onClick={loadAll}
            className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-forest-300 hover:bg-white/10 transition"
          >
            <RefreshCw size={12} />
            Actualiser
          </button>
        </div>
      </div>

      {/* Grille de KPIs sources */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
        {[
          { label: "Météo", icon: <Cloud size={14} />, color: "text-brand-400", active: !!meteo },
          { label: "Routage", icon: <MapPin size={14} />, color: "text-violet-400", active: !!geo },
          { label: "Épidémie", icon: <Activity size={14} />, color: "text-brand-400", active: !!epidemic },
          { label: "Démographie", icon: <Users size={14} />, color: "text-gold-400", active: !!demographics },
          { label: "Pharmacies", icon: <Pill size={14} />, color: "text-rose-400", active: !!pharmacies },
          { label: "Signaux", icon: <Radio size={14} />, color: "text-brand-400", active: !!informalSignals },
          { label: "Copernicus", icon: <Zap size={14} />, color: "text-gold-400", active: !!climate },
        ].map((s) => (
          <div key={s.label} className={`rounded-2xl border p-3 text-center transition ${
            s.active ? "border-white/10 bg-white/5" : "border-white/5 bg-white/2 opacity-40"
          }`}>
            <div className={`flex justify-center mb-1 ${s.color}`}>{s.icon}</div>
            <p className="text-xs text-forest-300 font-medium">{s.label}</p>
            <p className={`text-[10px] mt-0.5 ${s.active ? "text-brand-400" : "text-forest-500"}`}>
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
              <Cloud size={16} className="text-brand-400" />
              Météo — {meteo.station}
            </h3>
            <span className={`rounded-full border px-3 py-1 text-xs font-medium ${alertColors[meteo.alert_level] ?? alertColors.none}`}>
              {meteo.alert_level === "none" ? "Aucune alerte" : meteo.alert_level === "warning" ? "Vigilance" : "Danger"}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 mb-4">
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-brand-300">{meteo.temperature}°C</p>
              <p className="mt-1 text-xs text-forest-400">Température</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-brand-300">{meteo.apparent_temperature}°C</p>
              <p className="mt-1 text-xs text-forest-400">Ressenti</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-brand-300">{meteo.humidity}%</p>
              <p className="mt-1 text-xs text-forest-400">Humidité</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-brand-300">{meteo.wind_speed} km/h</p>
              <p className="mt-1 text-xs text-forest-400">Vent</p>
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
            Routage Transfrontalier — {geo.origin.label} → {geo.destination.label}
          </h3>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 mb-4">
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-violet-300">{geo.distance_km} km</p>
              <p className="mt-1 text-xs text-forest-400">Distance</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-violet-300">{geo.base_duration_min} min</p>
              <p className="mt-1 text-xs text-forest-400">Durée de base</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-gold-300">{geo.cross_border_delay_min} min</p>
              <p className="mt-1 text-xs text-forest-400">Délai douane</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-rose-300">{geo.total_estimated_response_time_min} min</p>
              <p className="mt-1 text-xs text-forest-400">Temps total estimé</p>
            </div>
          </div>
          <div className="rounded-2xl border border-violet-500/30 bg-violet-500/10 p-3 text-sm text-violet-300">
            <p className="font-medium">Action de coordination</p>
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
              Surveillance Épidémique — {epidemic.region}
            </h3>
            <span className={`text-lg font-bold ${riskColors[epidemic.global_ems_impact_risk] ?? "text-white"}`}>
              Risque EMS : {epidemic.global_ems_impact_risk.toUpperCase()}
            </span>
          </div>
          <div className="space-y-3 mb-4">
            {epidemic.diseases.map((d) => (
              <div key={d.name} className="rounded-2xl border border-white/10 bg-forest-900/60 p-3">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-white text-sm">{d.name}</span>
                  <div className="flex items-center gap-2">
                    <span className={`rounded-full border px-2 py-0.5 text-xs ${statusColors[d.status] ?? ""}`}>
                      {d.status === "under_threshold" ? "Sous le seuil" : d.status === "warning" ? "Vigilance" : "Épidémie"}
                    </span>
                    <span className={`text-sm font-bold ${
                      d.trend === "increasing" ? "text-rose-300" : d.trend === "decreasing" ? "text-brand-300" : "text-forest-300"
                    }`}>{trendIcons[d.trend]}</span>
                  </div>
                </div>
                <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-forest-400">
                  <span>France : <span className="text-white font-medium">{d.incidence_per_100k_france}/100k</span></span>
                  <span>Suisse : <span className="text-white font-medium">{d.incidence_per_100k_switzerland}/100k</span></span>
                  <span>Seuil : <span className="text-white font-medium">{d.epidemic_threshold}/100k</span></span>
                </div>
              </div>
            ))}
          </div>
          <div className="rounded-2xl border border-brand-500/30 bg-brand-500/10 p-3 text-sm text-brand-300">
            <p className="font-medium">Recommandation opérationnelle</p>
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
            Démographie — {demographics.commune} ({demographics.postal_code})
          </h3>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 mb-4">
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-gold-300">{demographics.population.toLocaleString("fr-FR")}</p>
              <p className="mt-1 text-xs text-forest-400">Population</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-gold-300">{demographics.density_per_km2}</p>
              <p className="mt-1 text-xs text-forest-400">Hab/km²</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-gold-300">{demographics.age_over_65_pct}%</p>
              <p className="mt-1 text-xs text-forest-400">&gt;65 ans</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-rose-300">×{demographics.ems_risk_multiplier}</p>
              <p className="mt-1 text-xs text-forest-400">Multiplicateur risque EMS</p>
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
            Pharmacies de Garde & Alertes Médicaments
          </h3>
          {pharmacies.critical_medication_alerts.length > 0 && (
            <div className="mb-4 space-y-2">
              <h4 className="text-xs font-semibold uppercase tracking-wider text-forest-400">Alertes médicaments critiques</h4>
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
            <h4 className="text-xs font-semibold uppercase tracking-wider text-forest-400">Pharmacies à proximité ({pharmacies.pharmacies_nearby.length})</h4>
            {pharmacies.pharmacies_nearby.map((ph, i) => (
              <div key={i} className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-white">{ph.name}</p>
                  <p className="text-xs text-forest-400">{ph.street}, {ph.city}</p>
                </div>
                <div className="text-right">
                  <span className={`text-xs font-medium ${
                    ph.is_dispensary ? "text-brand-300" : "text-forest-400"
                  }`}>{ph.is_dispensary ? "✓ Dispensaire" : "Pharmacie"}</span>
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
            Signaux Informels — ProMED / GDELT
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
                    <p className="text-xs text-forest-400 mt-0.5">{sig.source} — {sig.date} — {sig.geo_scope}</p>
                  </div>
                  <div className="flex flex-col items-end gap-1 shrink-0">
                    <span className={`rounded-full border px-2 py-0.5 text-xs font-medium ${
                      sig.severity === "high" ? "border-rose-500/30 bg-rose-500/10 text-rose-300" :
                      sig.severity === "moderate" ? "border-gold-500/30 bg-gold-500/10 text-gold-300" :
                      "border-brand-500/30 bg-brand-500/10 text-brand-300"
                    }`}>{sig.severity.toUpperCase()}</span>
                    <span className="text-xs text-forest-500">Fiabilité {Math.round(sig.reliability_score * 100)}%</span>
                  </div>
                </div>
                <p className="text-sm text-forest-300 leading-5">{sig.content}</p>
                {sig.impact_on_gesica && (
                  <p className="mt-2 text-xs text-brand-300 italic">→ GESICA : {sig.impact_on_gesica}</p>
                )}
                {sig.impact_on_geoai4ei && (
                  <p className="mt-1 text-xs text-violet-300 italic">→ GeoAI4EI : {sig.impact_on_geoai4ei}</p>
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
              Copernicus Climate Data Store (CDS) — ERA5
            </h3>
            <span className={`rounded-full border px-3 py-1 text-xs font-medium ${
              climate.api_status.includes("verified") ? "border-brand-500/30 bg-brand-500/10 text-brand-300" : "border-gold-500/30 bg-gold-500/10 text-gold-300"
            }`}>
              {climate.api_status === "connected_verified" ? "API Connectée" : "Mode simulé / Configuré"}
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
              <p className="mt-1 text-xs text-forest-400">Moyenne historique (Mai)</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-gold-300">+{climate.climatology.current_anomaly_c}°C</p>
              <p className="mt-1 text-xs text-forest-400">Anomalie thermique</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-gold-300">{climate.climatology.soil_moisture_deficit_percent}%</p>
              <p className="mt-1 text-xs text-forest-400">Déficit d'humidité des sols</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-3 text-center">
              <p className="text-2xl font-bold text-rose-300">{climate.climatology.heatwave_hazard_index.toUpperCase()}</p>
              <p className="mt-1 text-xs text-forest-400">Risque canicule</p>
            </div>
          </div>

          <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-4">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-forest-400 mb-2">Projections climatiques horizon 2030 (Transfrontalier)</h4>
            <ul className="space-y-1.5 text-xs text-forest-300">
              <li>• Augmentation des jours de canicule extrême : <span className="text-white font-medium">+{climate.projections_2030.expected_heatwave_days_increase_per_year} jours/an</span></li>
              <li>• Augmentation des précipitations extrêmes : <span className="text-white font-medium">+{climate.projections_2030.expected_heavy_precipitation_increase_percent}%</span></li>
              <li>• Facteur de vulnérabilité EMS principal : <span className="text-white font-medium">{climate.projections_2030.ems_vulnerability_factor.replace(/_/g, " ")}</span></li>
            </ul>
          </div>
          
          <p className="mt-3 text-xs text-forest-500 italic">{climate.source}</p>
        </div>
      )}
    </div>
  );
}

function EvidenceStrengthBadge({ strength }: { strength: "weak" | "moderate" | "strong" | null }) {
  if (!strength) return null;
  const config = {
    strong: { label: "Forte", className: "bg-brand-500/20 text-brand-300 border-brand-500/30" },
    moderate: { label: "Modérée", className: "bg-gold-500/20 text-gold-300 border-gold-500/30" },
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
    <span className="rounded-full bg-brand-500/10 border border-brand-500/20 px-2 py-0.5 text-xs text-brand-300">
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
          <Zap size={14} className="text-brand-400" />
          Signaux GESICA
        </h3>
        <EvidenceStrengthBadge strength={s.evidenceStrength} />
      </div>

      {s.forecastHorizon && (
        <div className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-forest-300">
          <span className="text-forest-400">Horizon prévisionnel :</span>{" "}
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
          <p className="mb-1 text-xs text-forest-400">Scénarios détectés</p>
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
          <p className="mb-1 text-xs text-forest-400">Métriques rapportées</p>
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

function StatsView({ corpusStats, gesicaStats, fulltextStats }: { corpusStats: CorpusStats | null; gesicaStats: GesicaStats | null; fulltextStats: FulltextStats | null }) {
  if (!corpusStats && !gesicaStats) {
    return <div className="text-sm text-forest-400">Chargement des statistiques...</div>;
  }

  return (
    <div className="space-y-6">
      {corpusStats && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold text-white">
            <BarChart2 size={18} className="text-brand-400" />
            Corpus global
          </h2>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-4 text-center">
              <p className="text-2xl font-bold text-brand-300">{corpusStats.totalDocuments.toLocaleString()}</p>
              <p className="mt-1 text-xs text-forest-400">Documents</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-4 text-center">
              <p className="text-2xl font-bold text-brand-300">{corpusStats.totalChunks.toLocaleString()}</p>
              <p className="mt-1 text-xs text-forest-400">Chunks</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-4 text-center">
              <p className="text-2xl font-bold text-brand-300">{Object.keys(corpusStats.byProject).length}</p>
              <p className="mt-1 text-xs text-forest-400">Projets</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-4 text-center">
              <p className="text-2xl font-bold text-brand-300">{Object.keys(corpusStats.bySource).length}</p>
              <p className="mt-1 text-xs text-forest-400">Sources</p>
            </div>
          </div>

          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <div>
              <h3 className="mb-2 text-sm font-medium text-forest-300">Par projet</h3>
              <div className="space-y-2">
                {Object.entries(corpusStats.byProject).map(([proj, count]) => (
                  <div key={proj} className="flex items-center justify-between rounded-xl border border-white/10 bg-forest-900/40 px-3 py-2 text-sm">
                    <span className="text-forest-200 capitalize">{proj}</span>
                    <span className="font-mono text-brand-300">{count}</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h3 className="mb-2 text-sm font-medium text-forest-300">Par source</h3>
              <div className="space-y-2">
                {Object.entries(corpusStats.bySource).map(([src, count]) => (
                  <div key={src} className="flex items-center justify-between rounded-xl border border-white/10 bg-forest-900/40 px-3 py-2 text-sm">
                    <span className="text-forest-200">{src}</span>
                    <span className="font-mono text-brand-300">{count}</span>
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
            <BookOpen size={18} className="text-brand-400" />
            Couverture textuelle &amp; Hybrid Search
          </h2>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-4 text-center">
              <p className="text-2xl font-bold text-brand-300">{fulltextStats.corpus.docs_with_fulltext.toLocaleString()}</p>
              <p className="mt-1 text-xs text-forest-400">Full Text</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-4 text-center">
              <p className="text-2xl font-bold text-forest-300">{fulltextStats.corpus.docs_abstract_only.toLocaleString()}</p>
              <p className="mt-1 text-xs text-forest-400">Abstract only</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-4 text-center">
              <p className={`text-2xl font-bold ${fulltextStats.corpus.fulltext_coverage_pct >= 20 ? 'text-brand-300' : 'text-gold-300'}`}>
                {fulltextStats.corpus.fulltext_coverage_pct.toFixed(1)}%
              </p>
              <p className="mt-1 text-xs text-forest-400">Couverture</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-4 text-center">
              <p className={`text-2xl font-bold ${fulltextStats.hybrid_search.active ? 'text-brand-300' : 'text-rose-300'}`}>
                {fulltextStats.hybrid_search.active ? 'HYBRID' : 'LEXICAL'}
              </p>
              <p className="mt-1 text-xs text-forest-400">Mode recherche</p>
            </div>
          </div>
          {fulltextStats.by_source && fulltextStats.by_source.length > 0 && (
            <div className="mt-4">
              <h3 className="mb-2 text-sm font-medium text-forest-300">Full Text par source</h3>
              <div className="space-y-1">
                {fulltextStats.by_source.slice(0, 8).map((s) => (
                  <div key={s.source} className="flex items-center justify-between rounded-xl border border-white/10 bg-forest-900/40 px-3 py-1.5 text-xs">
                    <span className="text-forest-300 capitalize">{s.source}</span>
                    <span className="font-mono text-brand-300">{s.with_fulltext} / {s.total}</span>
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
            <Activity size={18} className="text-brand-400" />
            Corpus GESICA
          </h2>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            {Object.entries(gesicaStats.evidenceStrengthDistribution).map(([strength, count]) => {
              const colors: Record<string, string> = {
                strong: "text-brand-300",
                moderate: "text-gold-300",
                weak: "text-rose-300",
              };
              return (
                <div key={strength} className="rounded-2xl border border-white/10 bg-forest-900/60 p-4 text-center">
                  <p className={`text-2xl font-bold ${colors[strength] ?? "text-white"}`}>{count}</p>
                  <p className="mt-1 text-xs text-forest-400 capitalize">Preuve {strength}</p>
                </div>
              );
            })}
          </div>

          {Object.keys(gesicaStats.forecastHorizons).length > 0 && (
            <div className="mt-4">
              <h3 className="mb-2 text-sm font-medium text-forest-300">Horizons prévisionnels</h3>
              <div className="flex flex-wrap gap-2">
                {Object.entries(gesicaStats.forecastHorizons).slice(0, 10).map(([h, count]) => (
                  <span key={h} className="rounded-full border border-brand-500/20 bg-brand-500/10 px-3 py-1 text-xs text-brand-300">
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
  const [detailScenarioId, setDetailScenarioId] = useState<string | null>(null);

  // Page détail d'un scénario
  if (detailScenarioId) {
    return <ScenarioDetailPage scenarioId={detailScenarioId} onBack={() => setDetailScenarioId(null)} />;
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16 text-forest-400">
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
          <p className="text-xs text-forest-400 mt-2">Vérifiez que le service API est démarré sur app-01 et que <code>/api/gesica/scenarios</code> répond correctement.</p>
        </div>
      </div>
    );
  }

  if (scenarios.length === 0) {
    return (
      <div className="flex items-center justify-center py-16 text-forest-400">
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
    "Opérations EMS": "border-brand-500/30 bg-brand-500/10 text-brand-300",
    "Triage & Clinique": "border-brand-500/30 bg-brand-500/10 text-brand-300",
    "Soins Centrés Patient": "border-rose-500/30 bg-rose-500/10 text-rose-300",
    "Surveillance & Crise": "border-gold-500/30 bg-gold-500/10 text-gold-300",
    "Systèmes & IA": "border-brand-500/30 bg-brand-500/10 text-brand-300",
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

    const riskColors = { NORMAL: "text-brand-400", "ÉLEVÉ": "text-gold-400", CRITIQUE: "text-red-400" };
    const riskBg = { NORMAL: "bg-brand-500/10 border-brand-500/20", "ÉLEVÉ": "bg-gold-500/10 border-gold-500/20", CRITIQUE: "bg-red-500/10 border-red-500/20" };

    return (
      <div className="space-y-3">
        {/* Métadonnées du modèle */}
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>Modèle : <span className="text-forest-300">{forecast.model}</span></span>
          <span>Temp. actuelle : <span className="text-forest-300">{forecast.input_features.current_temperature}°C</span></span>
          <span>Index épidémique : <span className="text-forest-300">{forecast.input_features.epidemic_index}</span></span>
          {forecast.status === "fallback" && (
            <span className="text-gold-400 border border-gold-500/20 bg-gold-500/10 px-1.5 py-0.5 rounded">Fallback analytique</span>
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
              <p className="text-[9px] text-forest-400 truncate">{pred.date.split(' ')[0]}</p>
              <p className="text-[10px] text-forest-300">{pred.date.split(' ')[1]}/{pred.date.split(' ')[2]?.slice(0, 3)}</p>
              <p className={`text-sm font-bold mt-1 ${riskColors[pred.risk_level] ?? "text-white"}`}>{pred.demand}</p>
              <p className="text-[9px] text-forest-500">{pred.temp_estimated}°C</p>
              <p className={`text-[9px] font-semibold mt-0.5 ${riskColors[pred.risk_level] ?? "text-white"}`}>{pred.risk_level}</p>
            </div>
          ))}
        </div>

        {/* Recommandation du jour le plus à risque */}
        {forecast.predictions.some(p => p.risk_level !== "NORMAL") && (
          <div className="rounded-xl border border-gold-500/20 bg-gold-500/5 px-3 py-2 text-xs text-gold-300">
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
        <button onClick={load} className="w-full rounded-xl border border-brand-500/30 bg-brand-500/10 px-4 py-3 text-sm text-brand-300 hover:bg-brand-500/20 transition flex items-center justify-center gap-2">
          <Activity size={14} />
          Lancer la surveillance épidémique J+14 (SARIMAX + Sentinelles)
        </button>
      );
    }
    if (loading) return <div className="flex items-center justify-center py-4 text-brand-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Analyse des données Sentinelles FR...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;

    const alertColors = { NORMAL: "text-brand-400", VIGILANCE: "text-gold-400", "ÉPIDÉMIE": "text-red-400" };
    const alertBg = { NORMAL: "bg-brand-500/10 border-brand-500/20", VIGILANCE: "bg-gold-500/10 border-gold-500/20", "ÉPIDÉMIE": "bg-red-500/10 border-red-500/20" };
    const diseases = Object.values(data.diseases) as EpidemicDiseaseResult[];

    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>Modèle : <span className="text-forest-300">{data.model}</span></span>
          <span>Région : <span className="text-forest-300">{data.region}</span></span>
          <span className={`font-semibold px-2 py-0.5 rounded border ${alertBg[data.overall_alert_level] ?? ""} ${alertColors[data.overall_alert_level] ?? ""}`}>
            Alerte globale : {data.overall_alert_level}
          </span>
          {data.status === "fallback" && <span className="text-gold-400 border border-gold-500/20 bg-gold-500/10 px-1.5 py-0.5 rounded">Fallback analytique</span>}
          <button onClick={load} className="ml-auto text-brand-400 hover:text-brand-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>

        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {diseases.map((d) => (
            <div key={d.disease} className={`rounded-xl border p-3 ${alertBg[d.max_alert_14d] ?? "bg-white/5 border-white/10"}`}>
              <p className="text-xs font-semibold text-white truncate">{d.label}</p>
              <p className={`text-lg font-bold mt-1 ${alertColors[d.max_alert_14d] ?? "text-white"}`}>{d.current_incidence}</p>
              <p className="text-[9px] text-forest-400">/100k — seuil {d.epidemic_threshold}</p>
              <p className={`text-[9px] font-semibold mt-1 ${alertColors[d.max_alert_14d] ?? "text-white"}`}>{d.max_alert_14d}</p>
            </div>
          ))}
        </div>

        <div className={`rounded-xl border px-3 py-2 text-xs ${alertBg[data.overall_alert_level] ?? "border-white/10 bg-white/5"} ${alertColors[data.overall_alert_level] ?? "text-forest-300"}`}>
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
        <button onClick={load} className="w-full rounded-xl border border-brand-500/30 bg-brand-500/10 px-4 py-3 text-sm text-brand-300 hover:bg-brand-500/20 transition flex items-center justify-center gap-2">
          <MapPin size={14} />
          Optimiser les temps de réponse EMS (OSRM + Open-Meteo)
        </button>
      );
    }
    if (loading) return <div className="flex items-center justify-center py-4 text-brand-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Calcul des itinéraires optimaux via OSRM...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;

    const statusColors = { OPTIMAL: "text-brand-400", ACCEPTABLE: "text-gold-400", DÉGRADÉ: "text-red-400" };
    const statusBg = { OPTIMAL: "bg-brand-500/10 border-brand-500/20", ACCEPTABLE: "bg-gold-500/10 border-gold-500/20", DÉGRADÉ: "bg-red-500/10 border-red-500/20" };

    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>Temp. : <span className="text-forest-300">{data.weather.temperature}°C</span></span>
          <span>Facteur météo : <span className="text-forest-300">×{data.weather.weather_factor}</span></span>
          <span>Couverture : <span className="text-brand-300 font-semibold">{data.metrics.coverage_rate_pct}%</span></span>
          <span>Temps moyen : <span className="text-brand-300 font-semibold">{data.metrics.mean_response_time_min} min</span></span>
          {data.metrics.degraded_zones > 0 && (
            <span className="text-red-400 border border-red-500/20 bg-red-500/10 px-1.5 py-0.5 rounded">{data.metrics.degraded_zones} zone(s) dégradée(s)</span>
          )}
          <button onClick={load} className="ml-auto text-brand-400 hover:text-brand-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>

        <div className="space-y-1.5 max-h-48 overflow-y-auto">
          {(data.assignments as ResponseTimeAssignment[]).map((a) => (
            <div key={a.zone_id} className={`rounded-xl border px-3 py-2 flex items-center justify-between gap-2 ${statusBg[a.response_status] ?? "bg-white/5 border-white/10"}`}>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-semibold text-white truncate">{a.zone_label}</p>
                <p className="text-[9px] text-forest-400 truncate">{a.base_label} {a.cross_border ? `→ via ${a.border_crossing}` : ""}</p>
              </div>
              <div className="text-right shrink-0">
                <p className={`text-sm font-bold ${statusColors[a.response_status] ?? "text-white"}`}>{a.total_response_time_min} min</p>
                <p className={`text-[9px] font-semibold ${statusColors[a.response_status] ?? "text-white"}`}>{a.response_status}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="rounded-xl border border-brand-500/20 bg-brand-500/5 px-3 py-2 text-xs text-brand-300">
          {data.global_recommendation}
        </div>
      </div>
    );
  };

  // Widget Cardiac Arrest Prediction (OHCA)
  const CardiacArrestWidget = () => {
    const [data, setData] = useState<CardiacArrestPredictionResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const load = () => {
      setLoading(true);
      setError(null);
      fetchCardiacArrestPrediction()
        .then(setData)
        .catch((e) => setError(e.message))
        .finally(() => setLoading(false));
    };

    if (!data && !loading && !error) {
      return (
        <button onClick={load} className="w-full rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300 hover:bg-rose-500/20 transition flex items-center justify-center gap-2">
          <Activity size={14} />
          Prédire le risque OHCA (LightGBM + météo)
        </button>
      );
    }
    if (loading) return <div className="flex items-center justify-center py-4 text-rose-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Calcul du risque OHCA en cours...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;

    const alertColors: Record<string, string> = { NORMAL: "text-brand-400", VIGILANCE: "text-gold-400", "ÉLEVÉ": "text-gold-400", CRITIQUE: "text-red-400" };
    const alertBg: Record<string, string> = { NORMAL: "bg-brand-500/10 border-brand-500/20", VIGILANCE: "bg-gold-500/10 border-gold-500/20", "ÉLEVÉ": "bg-gold-500/10 border-gold-500/20", CRITIQUE: "bg-red-500/10 border-red-500/20" };

    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>Alerte : <span className={`font-bold ${alertColors[data.overall_alert_level] ?? "text-white"}`}>{data.overall_alert_level}</span></span>
          <span>Risque max 3j : <span className="text-rose-300 font-semibold">×{data.max_risk_multiplier_3d}</span></span>
          <span>T° max : <span className="text-forest-300">{data.current_weather.temp_max}°C</span></span>
          <span>Saison : <span className="text-forest-300">{data.current_weather.season}</span></span>
          {data.flu_epidemic_active && <span className="text-gold-400 border border-gold-500/20 bg-gold-500/10 px-1.5 py-0.5 rounded">Grippe active (+12%)</span>}
          <button onClick={load} className="ml-auto text-rose-400 hover:text-rose-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>
        <div className="space-y-1.5">
          {(data.forecast_3d as OHCAForecastDay[]).map((d) => (
            <div key={d.date} className={`rounded-xl border px-3 py-2 flex items-center justify-between gap-2 ${alertBg[d.alert_level] ?? "bg-white/5 border-white/10"}`}>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-semibold text-white">{d.date} ({d.day_name})</p>
                <p className="text-[9px] text-forest-400 truncate">{d.active_risk_factors.length > 0 ? d.active_risk_factors[0] : "Pas de facteur de risque actif"}</p>
              </div>
              <div className="text-right shrink-0">
                <p className={`text-sm font-bold ${alertColors[d.alert_level] ?? "text-white"}`}>{d.ohca_absolute_predicted} OHCA/j</p>
                <p className={`text-[9px] font-semibold ${alertColors[d.alert_level] ?? "text-white"}`}>{d.risk_pct_above_baseline > 0 ? `+${d.risk_pct_above_baseline}%` : "Baseline"}</p>
              </div>
            </div>
          ))}
        </div>
        {data.recommendations.length > 0 && (
          <div className={`rounded-xl border px-3 py-2 text-xs ${alertBg[data.overall_alert_level] ?? "border-white/10 bg-white/5"} ${alertColors[data.overall_alert_level] ?? "text-forest-300"}`}>
            {data.recommendations[0]}
          </div>
        )}
      </div>
    );
  };

  // Widget Heatwave EMS Impact (DLNM + UTCI)
  const HeatwaveEMSWidget = () => {
    const [data, setData] = useState<HeatwaveEMSImpactResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const load = () => {
      setLoading(true);
      setError(null);
      fetchHeatwaveEMSImpact()
        .then(setData)
        .catch((e) => setError(e.message))
        .finally(() => setLoading(false));
    };

    if (!data && !loading && !error) {
      return (
        <button onClick={load} className="w-full rounded-xl border border-gold-500/30 bg-gold-500/10 px-4 py-3 text-sm text-gold-300 hover:bg-gold-500/20 transition flex items-center justify-center gap-2">
          <Cloud size={14} />
          Analyser l'impact canicule sur les EMS (DLNM + UTCI)
        </button>
      );
    }
    if (loading) return <div className="flex items-center justify-center py-4 text-gold-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Calcul de l'impact thermique en cours...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;

    const alertColors: Record<string, string> = { NORMAL: "text-brand-400", VIGILANCE: "text-gold-400", ALERTE: "text-gold-400", URGENCE: "text-red-400" };
    const alertBg: Record<string, string> = { NORMAL: "bg-brand-500/10 border-brand-500/20", VIGILANCE: "bg-gold-500/10 border-gold-500/20", ALERTE: "bg-gold-500/10 border-gold-500/20", URGENCE: "bg-red-500/10 border-red-500/20" };

    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>Alerte : <span className={`font-bold ${alertColors[data.overall_alert_level] ?? "text-white"}`}>{data.overall_alert_level}</span></span>
          <span>UTCI : <span className="text-gold-300 font-semibold">{data.current_weather.utci}°C</span></span>
          <span>T° max : <span className="text-forest-300">{data.current_weather.temp_max}°C</span></span>
          <span>EMS aujourd'hui : <span className="text-gold-300 font-semibold">{data.dlnm_analysis.ems_calls_today} appels</span></span>
          {data.heatwave_status.active && <span className="text-red-400 border border-red-500/20 bg-red-500/10 px-1.5 py-0.5 rounded">Vague de chaleur ({data.heatwave_status.duration_days}j)</span>}
          <button onClick={load} className="ml-auto text-gold-400 hover:text-gold-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>
        <div className="space-y-1.5 max-h-48 overflow-y-auto">
          {(data.forecast_7d as HeatwaveForecastDay[]).map((d) => (
            <div key={d.date} className={`rounded-xl border px-3 py-2 flex items-center justify-between gap-2 ${alertBg[d.alert_level] ?? "bg-white/5 border-white/10"}`}>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-semibold text-white">{d.date}</p>
                <p className="text-[9px] text-forest-400">UTCI {d.utci}°C — {d.utci_category.replace(/_/g, " ")} {d.is_heatwave_day ? "🔥" : ""}</p>
              </div>
              <div className="text-right shrink-0">
                <p className={`text-sm font-bold ${alertColors[d.alert_level] ?? "text-white"}`}>{d.ems_calls_predicted} appels</p>
                <p className={`text-[9px] font-semibold ${alertColors[d.alert_level] ?? "text-white"}`}>{d.ems_excess_pct > 0 ? `+${d.ems_excess_pct}%` : "Baseline"}</p>
              </div>
            </div>
          ))}
        </div>
        {data.recommendations.length > 0 && (
          <div className={`rounded-xl border px-3 py-2 text-xs ${alertBg[data.overall_alert_level] ?? "border-white/10 bg-white/5"} ${alertColors[data.overall_alert_level] ?? "text-forest-300"}`}>
            {data.recommendations[0]}
          </div>
        )}
      </div>
    );
  };

  // Widget Stroke Detection
  const StrokeDetectionWidget = () => {
    const [data, setData] = useState<StrokeDetectionResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = () => { setLoading(true); setError(null); fetchStrokeDetection().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return <button onClick={load} className="w-full rounded-xl border border-brand-500/30 bg-brand-500/10 px-4 py-3 text-sm text-brand-300 hover:bg-brand-500/20 transition flex items-center justify-center gap-2"><Activity size={14} />Analyser les délais AVC — Door-to-Needle (XGBoost)</button>;
    if (loading) return <div className="flex items-center justify-center py-4 text-brand-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Calcul des délais AVC en cours...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    const alertColors: Record<string, string> = { NORMAL: "text-brand-400", VIGILANCE: "text-gold-400", ALERTE: "text-gold-400", CRITIQUE: "text-red-400" };
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>Alerte : <span className={`font-bold ${alertColors[data.overall_alert_level] ?? "text-white"}`}>{data.overall_alert_level}</span></span>
          <span>Risque circadien : <span className="text-brand-300 font-semibold">{data.circadian_risk?.risk_level}</span></span>
          <button onClick={load} className="ml-auto text-brand-400 hover:text-brand-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>
        <div className="space-y-1.5 max-h-48 overflow-y-auto">
          {data.stroke_units?.map((u) => (
            <div key={u.name} className={`rounded-xl border px-3 py-2 flex items-center justify-between gap-2 ${u.dtn_ok ? "bg-brand-500/5 border-brand-500/20" : "bg-gold-500/5 border-gold-500/20"}`}>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-semibold text-white">{u.name} — {u.city} ({u.country})</p>
                <p className="text-[9px] text-forest-400">{u.distance_km} km · {u.transport_time_min} min transport</p>
              </div>
              <div className="text-right shrink-0">
                <p className={`text-sm font-bold ${u.dtn_ok ? "text-brand-400" : "text-gold-400"}`}>{u.estimated_dtn_min} min DTN</p>
                <p className="text-[9px] text-forest-400">{u.tpa_eligible ? "tPA ✓" : "tPA ✗"} {u.thrombectomy_eligible ? "· Thrombect. ✓" : ""}</p>
              </div>
            </div>
          ))}
        </div>
        {data.recommendations?.length > 0 && <div className="rounded-xl border border-brand-500/20 bg-brand-500/5 px-3 py-2 text-xs text-brand-300">{data.recommendations[0]}</div>}
      </div>
    );
  };

  // Widget Triage Support
  const TriageSupportWidget = () => {
    const [data, setData] = useState<TriageSupportResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = () => { setLoading(true); setError(null); fetchTriageSupport().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return <button onClick={load} className="w-full rounded-xl border border-gold-500/30 bg-gold-500/10 px-4 py-3 text-sm text-gold-300 hover:bg-gold-500/20 transition flex items-center justify-center gap-2"><CheckSquare size={14} />Charger l'aide au triage CCMU / NEWS2</button>;
    if (loading) return <div className="flex items-center justify-center py-4 text-gold-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Chargement du référentiel triage...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    const loadColors: Record<string, string> = { NORMAL: "text-brand-400", MODÉRÉ: "text-gold-400", ÉLEVÉ: "text-gold-400", SATURÉ: "text-red-400" };
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>Charge : <span className={`font-bold ${loadColors[data.current_load?.level] ?? "text-white"}`}>{data.current_load?.label}</span></span>
          <span>Attente moy. : <span className="text-gold-300 font-semibold">{data.current_load?.mean_wait_min} min</span></span>
          <button onClick={load} className="ml-auto text-gold-400 hover:text-gold-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>
        <div className="grid grid-cols-5 gap-1.5">
          {Object.entries(data.ccmu_levels ?? {}).map(([k, v]) => (
            <div key={k} className="rounded-xl border border-white/10 bg-forest-900/40 px-2 py-2 text-center">
              <p className="text-xs font-bold text-white">{k}</p>
              <p className="text-[9px] text-forest-400 mt-0.5 leading-tight">{v.label}</p>
              <p className="text-[9px] text-forest-500 mt-0.5">{v.target_time_min} min</p>
            </div>
          ))}
        </div>
        {data.recommendations?.length > 0 && <div className="rounded-xl border border-gold-500/20 bg-gold-500/5 px-3 py-2 text-xs text-gold-300">{data.recommendations[0]}</div>}
      </div>
    );
  };

  // Widget Undertriage Risk
  const UndertriageRiskWidget = () => {
    const [data, setData] = useState<UndertriageRiskResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = () => { setLoading(true); setError(null); fetchUndertriageRisk().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return <button onClick={load} className="w-full rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300 hover:bg-rose-500/20 transition flex items-center justify-center gap-2"><AlertTriangle size={14} />Analyser les risques de sous-triage EMS</button>;
    if (loading) return <div className="flex items-center justify-center py-4 text-rose-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Analyse des risques de sous-triage...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    const riskColors: Record<string, string> = { FAIBLE: "text-brand-400", MODÉRÉ: "text-gold-400", ÉLEVÉ: "text-gold-400", CRITIQUE: "text-red-400" };
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>Alerte : <span className={`font-bold ${riskColors[data.overall_alert_level] ?? "text-white"}`}>{data.overall_alert_level}</span></span>
          <span>Cible sous-triage : <span className="text-rose-300 font-semibold">≤{data.undertriage_rate_target_pct}%</span></span>
          <button onClick={load} className="ml-auto text-rose-400 hover:text-rose-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>
        <div className="space-y-1.5 max-h-48 overflow-y-auto">
          {data.high_risk_scenarios?.slice(0, 4).map((s, i) => (
            <div key={i} className={`rounded-xl border px-3 py-2 ${s.risk_level === "CRITIQUE" || s.risk_level === "ÉLEVÉ" ? "border-rose-500/20 bg-rose-500/5" : "border-gold-500/20 bg-gold-500/5"}`}>
              <div className="flex items-center justify-between">
                <p className="text-xs font-semibold text-white">{s.scenario}</p>
                <span className={`text-xs font-bold ${riskColors[s.risk_level] ?? "text-white"}`}>{s.undertriage_risk_pct}%</span>
              </div>
              <p className="text-[9px] text-forest-400 mt-0.5">{s.recommended_action}</p>
            </div>
          ))}
        </div>
        {data.recommendations?.length > 0 && <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 px-3 py-2 text-xs text-rose-300">{data.recommendations[0]}</div>}
      </div>
    );
  };

  // Widget Trauma Care
  const TraumaCareWidget = () => {
    const [data, setData] = useState<TraumaCareResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = () => { setLoading(true); setError(null); fetchTraumaCare().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return <button onClick={load} className="w-full rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300 hover:bg-red-500/20 transition flex items-center justify-center gap-2"><Activity size={14} />Calculer ISS / RTS / TRISS — Cas Trauma</button>;
    if (loading) return <div className="flex items-center justify-center py-4 text-red-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Calcul des scores trauma...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>Cas analysés : <span className="text-red-300 font-semibold">{data.cohort_summary?.n_cases}</span></span>
          <span>Survie moy. : <span className="text-brand-300 font-semibold">{data.cohort_summary?.mean_survival_pct}%</span></span>
          <span>Damage Control : <span className="text-gold-300 font-semibold">{data.cohort_summary?.damage_control_cases}/{data.cohort_summary?.n_cases}</span></span>
          <button onClick={load} className="ml-auto text-red-400 hover:text-red-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>
        <div className="space-y-1.5 max-h-48 overflow-y-auto">
          {data.case_examples?.map((c, i) => (
            <div key={i} className={`rounded-xl border px-3 py-2 flex items-center justify-between gap-2 ${c.damage_control_indicated ? "border-gold-500/20 bg-gold-500/5" : "border-white/10 bg-white/5"}`}>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-semibold text-white">{c.case_name}</p>
                <p className="text-[9px] text-forest-400">ISS {c.scores.iss} — {c.scores.iss_level} {c.damage_control_indicated ? "· DC ⚠" : ""}</p>
              </div>
              <div className="text-right shrink-0">
                <p className={`text-sm font-bold ${c.scores.triss_survival_pct >= 75 ? "text-brand-400" : c.scores.triss_survival_pct >= 50 ? "text-gold-400" : "text-red-400"}`}>{c.scores.triss_survival_pct}%</p>
                <p className="text-[9px] text-forest-400">TRISS</p>
              </div>
            </div>
          ))}
        </div>
        {data.recommendations?.length > 0 && <div className="rounded-xl border border-red-500/20 bg-red-500/5 px-3 py-2 text-xs text-red-300">{data.recommendations[0]}</div>}
      </div>
    );
  };

  // Widget Mass Casualty
  const MassCasualtyWidget = () => {
    const [data, setData] = useState<MassCasualtyResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [nVictims, setNVictims] = useState(50);
    const [eventType, setEventType] = useState("transport_accident");
    const load = () => { setLoading(true); setError(null); fetchMassCasualty(nVictims, eventType).then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return (
      <div className="space-y-3">
        <div className="flex items-center gap-2 flex-wrap">
          <select value={eventType} onChange={(e) => setEventType(e.target.value)} className="rounded-lg border border-white/10 bg-forest-900/60 px-2 py-1.5 text-xs text-forest-300">
            <option value="transport_accident">Accident transport</option>
            <option value="explosion">Explosion</option>
            <option value="chemical">Intoxication chimique</option>
            <option value="building_collapse">Effondrement</option>
            <option value="mass_shooting">Fusillade</option>
            <option value="industrial_accident">Accident industriel</option>
          </select>
          <input type="number" min={1} max={500} value={nVictims} onChange={(e) => setNVictims(parseInt(e.target.value) || 50)} className="w-20 rounded-lg border border-white/10 bg-forest-900/60 px-2 py-1.5 text-xs text-forest-300" placeholder="Victimes" />
          <button onClick={load} className="rounded-xl border border-violet-500/30 bg-violet-500/10 px-4 py-1.5 text-sm text-violet-300 hover:bg-violet-500/20 transition flex items-center gap-2"><Users size={14} />Simuler</button>
        </div>
      </div>
    );
    if (loading) return <div className="flex items-center justify-center py-4 text-violet-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Simulation Monte-Carlo en cours...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    const alertColors: Record<string, string> = { VIGILANCE: "text-gold-400", MODÉRÉ: "text-gold-400", ÉLEVÉ: "text-gold-400", CRITIQUE: "text-red-400" };
    const saltColors: Record<string, string> = { immediate: "text-red-400", delayed: "text-gold-400", minimal: "text-brand-400", expectant: "text-forest-400", deceased: "text-forest-600" };
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>Alerte : <span className={`font-bold ${alertColors[data.overall_alert_level] ?? "text-white"}`}>{data.overall_alert_level}</span></span>
          <span>Victimes : <span className="text-violet-300 font-semibold">{data.scenario?.n_victims}</span></span>
          <span>Renforts : <span className={data.resource_needs?.mutual_aid_required ? "text-red-400 font-bold" : "text-brand-400"}>{data.resource_needs?.mutual_aid_required ? "REQUIS" : "Non requis"}</span></span>
          <button onClick={load} className="ml-auto text-violet-400 hover:text-violet-300 flex items-center gap-1"><RefreshCw size={10} /> Recalculer</button>
        </div>
        <div className="grid grid-cols-5 gap-1.5">
          {Object.entries(data.salt_distribution ?? {}).map(([k, v]) => (
            <div key={k} className="rounded-xl border border-white/10 bg-forest-900/40 px-2 py-2 text-center">
              <p className={`text-sm font-bold ${saltColors[k] ?? "text-white"}`}>{v.mean}</p>
              <p className="text-[9px] text-forest-400 mt-0.5 leading-tight">{v.label.split(" ")[0]}</p>
            </div>
          ))}
        </div>
        <div className="space-y-1 max-h-32 overflow-y-auto">
          {data.hospital_distribution?.map((h, i) => (
            <div key={i} className="rounded-xl border border-white/10 bg-forest-900/40 px-3 py-1.5 flex items-center justify-between gap-2">
              <p className="text-xs text-forest-300">{h.hospital} ({h.country})</p>
              <div className="flex items-center gap-3 text-[10px]">
                <span className="text-red-400">{h.assigned_immediate} 🔴</span>
                <span className="text-gold-400">{h.assigned_delayed} 🟡</span>
                <span className="text-forest-400">{h.transport_time_min} min</span>
              </div>
            </div>
          ))}
        </div>
        {data.recommendations?.length > 0 && <div className="rounded-xl border border-violet-500/20 bg-violet-500/5 px-3 py-2 text-xs text-violet-300">{data.recommendations[0]}</div>}
      </div>
    );
  };

  // ─── Widget Clinical Deterioration ─────────────────────────────────────────
  const ClinicalDeteriorationWidget = () => {
    const [data, setData] = useState<ClinicalDeteriorationResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = () => { setLoading(true); setError(null); fetchClinicalDeterioration().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return <button onClick={load} className="w-full rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300 hover:bg-rose-500/20 transition flex items-center justify-center gap-2"><Activity size={14} />Analyser les signes vitaux — NEWS2 / MEWS</button>;
    if (loading) return <div className="flex items-center justify-center py-4 text-rose-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Calcul des scores de dégradation...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    const alertC: Record<string, string> = { NORMAL: "text-brand-400", VIGILANCE: "text-gold-400", ALERTE: "text-gold-400", CRITIQUE: "text-red-400" };
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>Alerte : <span className={`font-bold ${alertC[data.overall_alert] ?? "text-white"}`}>{data.overall_alert}</span></span>
          <span>NEWS2 : <span className="text-rose-300 font-semibold">{data.news2_score}</span></span>
          <span>MEWS : <span className="text-rose-300 font-semibold">{data.mews_score}</span></span>
          <button onClick={load} className="ml-auto text-rose-400 hover:text-rose-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(data.vital_signs ?? {}).map(([k, v]) => (
            <div key={k} className="rounded-xl border border-white/10 bg-forest-900/40 px-2 py-1.5">
              <p className="text-[9px] text-forest-500 uppercase">{k.replace(/_/g, " ")}</p>
              <p className={`text-xs font-bold ${(v as {status: string; value: number; unit: string}).status === "NORMAL" ? "text-brand-400" : "text-gold-400"}`}>{(v as {status: string; value: number; unit: string}).value} {(v as {status: string; value: number; unit: string}).unit}</p>
            </div>
          ))}
        </div>
        {data.recommendations?.length > 0 && <div className={`rounded-xl border px-3 py-2 text-xs ${alertC[data.overall_alert] === "text-red-400" ? "border-red-500/20 bg-red-500/5 text-red-300" : "border-gold-500/20 bg-gold-500/5 text-gold-300"}`}>{data.recommendations[0]}</div>}
      </div>
    );
  };

  // ─── Widget Call Qualification ────────────────────────────────────────────────
  const CallQualificationWidget = () => {
    const [data, setData] = useState<CallQualificationResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = () => { setLoading(true); setError(null); fetchCallQualification().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return <button onClick={load} className="w-full rounded-xl border border-brand-500/30 bg-brand-500/10 px-4 py-3 text-sm text-brand-300 hover:bg-brand-500/20 transition flex items-center justify-center gap-2"><Radio size={14} />Analyser la qualification des appels (NLP)</button>;
    if (loading) return <div className="flex items-center justify-center py-4 text-brand-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Analyse NLP des appels en cours...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    const prioC: Record<number, string> = { 1: "text-red-400", 2: "text-gold-400", 3: "text-gold-400", 4: "text-brand-400", 5: "text-forest-400" };
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>Priorité globale : <span className={`font-bold ${prioC[data.overall_priority] ?? "text-white"}`}>{data.overall_label}</span></span>
          <span>{data.calls_analyzed} appels analysés</span>
          <button onClick={load} className="ml-auto text-brand-400 hover:text-brand-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>
        <div className="space-y-1.5 max-h-40 overflow-y-auto">
          {data.sample_calls?.map((c) => (
            <div key={c.call_id} className="rounded-xl border border-white/10 bg-forest-900/40 px-3 py-2 flex items-center justify-between gap-2">
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-white truncate">{c.chief_complaint}</p>
                <p className="text-[9px] text-forest-400">{c.recommended_resource} · {c.confidence_pct}% confiance</p>
              </div>
              <span className={`text-sm font-bold shrink-0 ${prioC[c.priority] ?? "text-white"}`}>P{c.priority}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // ─── Widget Dispatch Decision ─────────────────────────────────────────────────
  const DispatchDecisionWidget = () => {
    const [data, setData] = useState<DispatchDecisionResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = () => { setLoading(true); setError(null); fetchDispatchDecision().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return <button onClick={load} className="w-full rounded-xl border border-brand-500/30 bg-brand-500/10 px-4 py-3 text-sm text-brand-300 hover:bg-brand-500/20 transition flex items-center justify-center gap-2"><MapPin size={14} />Analyser les décisions de dispatch</button>;
    if (loading) return <div className="flex items-center justify-center py-4 text-brand-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Calcul des recommandations dispatch...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>Statut : <span className="font-bold text-white">{data.overall_status}</span></span>
          <span>{data.pending_incidents} incidents · {data.available_resources} ressources dispo</span>
          <button onClick={load} className="ml-auto text-brand-400 hover:text-brand-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>
        <div className="space-y-1.5 max-h-40 overflow-y-auto">
          {data.dispatch_recommendations?.map((r) => (
            <div key={r.incident_id} className="rounded-xl border border-white/10 bg-forest-900/40 px-3 py-2 flex items-center justify-between gap-2">
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-white truncate">{r.category}</p>
                <p className="text-[9px] text-forest-400">{r.recommended_resource} · ETA {r.eta_min} min</p>
              </div>
              <span className={`text-xs font-bold shrink-0 ${r.priority === 1 ? "text-red-400" : r.priority === 2 ? "text-gold-400" : "text-brand-400"}`}>P{r.priority}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // ─── Widget Patient Pathway ───────────────────────────────────────────────────
  const PatientPathwayWidget = () => {
    const [data, setData] = useState<PatientPathwayResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = () => { setLoading(true); setError(null); fetchPatientPathway().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return <button onClick={load} className="w-full rounded-xl border border-brand-500/30 bg-brand-500/10 px-4 py-3 text-sm text-brand-300 hover:bg-brand-500/20 transition flex items-center justify-center gap-2"><MapPin size={14} />Optimiser le parcours patient transfrontalier</button>;
    if (loading) return <div className="flex items-center justify-center py-4 text-brand-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Calcul des parcours optimaux...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>{data.summary?.total_cases} cas · {data.summary?.cross_border_cases} transfrontaliers · ETA moy. {data.summary?.mean_eta_min} min</span>
          <button onClick={load} className="ml-auto text-brand-400 hover:text-brand-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>
        <div className="space-y-1.5 max-h-40 overflow-y-auto">
          {data.cases?.map((c) => (
            <div key={c.case_id} className={`rounded-xl border px-3 py-2 flex items-center justify-between gap-2 ${c.cross_border ? "border-gold-500/20 bg-gold-500/5" : "border-white/10 bg-forest-900/40"}`}>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-white truncate">{c.condition}</p>
                <p className="text-[9px] text-forest-400">{c.recommended} {c.cross_border ? "🌍 Transfrontalier" : ""}</p>
              </div>
              <span className="text-sm font-bold shrink-0 text-brand-300">{c.eta_min} min</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // ─── Widget Ambulance Dispatch ────────────────────────────────────────────────
  const AmbulanceDispatchWidget = () => {
    const [data, setData] = useState<AmbulanceDispatchResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = () => { setLoading(true); setError(null); fetchAmbulanceDispatch().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return <button onClick={load} className="w-full rounded-xl border border-brand-500/30 bg-brand-500/10 px-4 py-3 text-sm text-brand-300 hover:bg-brand-500/20 transition flex items-center justify-center gap-2"><MapPin size={14} />Analyser la couverture ambulancière (VRP)</button>;
    if (loading) return <div className="flex items-center justify-center py-4 text-brand-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Optimisation VRP en cours...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>Couverture : <span className={`font-bold ${data.coverage.coverage_pct >= 80 ? "text-brand-400" : data.coverage.coverage_pct >= 60 ? "text-gold-400" : "text-red-400"}`}>{data.coverage.coverage_pct}%</span></span>
          <span>{data.coverage.uncovered_zones} zones non couvertes · {data.coverage.degraded_zones} dégradées</span>
          <button onClick={load} className="ml-auto text-brand-400 hover:text-brand-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>
        <div className="space-y-1.5 max-h-40 overflow-y-auto">
          {data.zone_details?.map((z) => (
            <div key={z.zone_id} className={`rounded-xl border px-3 py-2 flex items-center justify-between gap-2 ${z.covered ? "border-brand-500/20 bg-brand-500/5" : "border-red-500/20 bg-red-500/5"}`}>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-white truncate">{z.zone_name}</p>
                <p className="text-[9px] text-forest-400">{z.best_base} · redondance ×{z.redundancy}</p>
              </div>
              <span className={`text-sm font-bold shrink-0 ${z.covered ? "text-brand-400" : "text-red-400"}`}>{z.eta_min} min</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // ─── Widget Hospital Capacity ─────────────────────────────────────────────────
  const HospitalCapacityWidget = () => {
    const [data, setData] = useState<HospitalCapacityResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = () => { setLoading(true); setError(null); fetchHospitalCapacity().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return <button onClick={load} className="w-full rounded-xl border border-brand-500/30 bg-brand-500/10 px-4 py-3 text-sm text-brand-300 hover:bg-brand-500/20 transition flex items-center justify-center gap-2"><BarChart2 size={14} />Analyser la capacité hospitalière et le staffing</button>;
    if (loading) return <div className="flex items-center justify-center py-4 text-brand-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Calcul capacité & staffing...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    return (
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-xl border border-white/10 bg-forest-900/40 px-3 py-2 text-center">
            <p className="text-lg font-bold text-brand-300">{data.current_status?.ed_occupancy_pct}%</p>
            <p className="text-[9px] text-forest-400">Occupation Urgences</p>
          </div>
          <div className="rounded-xl border border-white/10 bg-forest-900/40 px-3 py-2 text-center">
            <p className="text-lg font-bold text-brand-300">{data.current_status?.icu_occupancy_pct}%</p>
            <p className="text-[9px] text-forest-400">Occupation Réa</p>
          </div>
        </div>
        <div className={`rounded-xl border px-3 py-2 text-xs ${data.staffing_now?.status === "DÉFICIT" ? "border-red-500/20 bg-red-500/5 text-red-300" : "border-brand-500/20 bg-brand-500/5 text-brand-300"}`}>
          Staffing : {data.staffing_now?.current_crews}/{data.staffing_now?.required_crews} équipes — {data.staffing_now?.action}
        </div>
        <button onClick={load} className="text-[10px] text-brand-400 hover:text-brand-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
      </div>
    );
  };

  // ─── Widget Surveillance ──────────────────────────────────────────────────────
  const SurveillanceWidget = () => {
    const [data, setData] = useState<SurveillanceResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = () => { setLoading(true); setError(null); fetchSurveillance().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return <button onClick={load} className="w-full rounded-xl border border-brand-500/30 bg-brand-500/10 px-4 py-3 text-sm text-brand-300 hover:bg-brand-500/20 transition flex items-center justify-center gap-2"><Activity size={14} />Lancer la surveillance des anomalies EMS</button>;
    if (loading) return <div className="flex items-center justify-center py-4 text-brand-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Détection d'anomalies en cours...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>Statut : <span className="font-bold text-white">{data.overall_status}</span></span>
          <span>{data.active_alerts?.length ?? 0} alertes actives</span>
          <button onClick={load} className="ml-auto text-brand-400 hover:text-brand-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>
        {data.active_alerts?.length > 0 && (
          <div className="space-y-1.5">
            {data.active_alerts.map((a, i) => (
              <div key={i} className="rounded-xl border border-gold-500/20 bg-gold-500/5 px-3 py-2 text-xs text-gold-300">
                <span className="font-semibold">{a.indicator}</span> — z={a.zscore.toFixed(2)} — {a.message}
              </div>
            ))}
          </div>
        )}
        {data.active_alerts?.length === 0 && <div className="rounded-xl border border-brand-500/20 bg-brand-500/5 px-3 py-2 text-xs text-brand-300">Aucune anomalie détectée — activité normale</div>}
      </div>
    );
  };

  // ─── Widget Surge Management ──────────────────────────────────────────────────
  const SurgeManagementWidget = () => {
    const [data, setData] = useState<SurgeManagementResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = () => { setLoading(true); setError(null); fetchSurgeManagement().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return <button onClick={load} className="w-full rounded-xl border border-gold-500/30 bg-gold-500/10 px-4 py-3 text-sm text-gold-300 hover:bg-gold-500/20 transition flex items-center justify-center gap-2"><Zap size={14} />Analyser le surge et la file d'attente EMS</button>;
    if (loading) return <div className="flex items-center justify-center py-4 text-gold-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Modèle M/M/c en cours...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    return (
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-xl border border-white/10 bg-forest-900/40 px-3 py-2 text-center">
            <p className="text-lg font-bold text-gold-300">{data.queue_metrics?.utilization_pct}%</p>
            <p className="text-[9px] text-forest-400">Utilisation</p>
          </div>
          <div className="rounded-xl border border-white/10 bg-forest-900/40 px-3 py-2 text-center">
            <p className="text-lg font-bold text-gold-300">{data.queue_metrics?.mean_wait_min} min</p>
            <p className="text-[9px] text-forest-400">Attente moy.</p>
          </div>
        </div>
        <div className={`rounded-xl border px-3 py-2 text-xs ${data.surge_status === "SURGE CRITIQUE" ? "border-red-500/20 bg-red-500/5 text-red-300" : data.surge_status === "SURGE MODÉRÉ" ? "border-gold-500/20 bg-gold-500/5 text-gold-300" : "border-brand-500/20 bg-brand-500/5 text-brand-300"}`}>
          {data.surge_status} — {data.staffing?.additional_needed > 0 ? `+${data.staffing?.additional_needed} équipes nécessaires` : "Staffing suffisant"}
        </div>
        <button onClick={load} className="text-[10px] text-gold-400 hover:text-gold-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
      </div>
    );
  };

  // ─── Widget Resource Allocation ───────────────────────────────────────────────
  const ResourceAllocationWidget = () => {
    const [data, setData] = useState<ResourceAllocationResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = () => { setLoading(true); setError(null); fetchResourceAllocation().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return <button onClick={load} className="w-full rounded-xl border border-violet-500/30 bg-violet-500/10 px-4 py-3 text-sm text-violet-300 hover:bg-violet-500/20 transition flex items-center justify-center gap-2"><BarChart2 size={14} />Optimiser l'allocation des ressources</button>;
    if (loading) return <div className="flex items-center justify-center py-4 text-violet-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Optimisation PL en cours...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>{data.summary?.allocated}/{data.summary?.total_incidents} incidents alloués</span>
          {data.summary?.unmet > 0 && <span className="text-red-400">{data.summary?.unmet} non couverts</span>}
          <button onClick={load} className="ml-auto text-violet-400 hover:text-violet-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>
        <div className="space-y-1.5 max-h-40 overflow-y-auto">
          {data.allocations?.slice(0, 5).map((a) => (
            <div key={a.incident_id} className="rounded-xl border border-white/10 bg-forest-900/40 px-3 py-2 flex items-center justify-between gap-2">
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-white truncate">{a.category}</p>
                <p className="text-[9px] text-forest-400">{a.allocated}</p>
              </div>
              <span className={`text-xs font-bold shrink-0 ${a.status === "ALLOUÉ" ? "text-brand-400" : "text-red-400"}`}>{a.status}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // ─── Widget Environmental Risk ────────────────────────────────────────────────
  const EnvironmentalRiskWidget = () => {
    const [data, setData] = useState<EnvironmentalRiskResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = () => { setLoading(true); setError(null); fetchEnvironmentalRisk().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return <button onClick={load} className="w-full rounded-xl border border-green-500/30 bg-green-500/10 px-4 py-3 text-sm text-green-300 hover:bg-green-500/20 transition flex items-center justify-center gap-2"><Cloud size={14} />Analyser la qualité de l'air et le risque EMS</button>;
    if (loading) return <div className="flex items-center justify-center py-4 text-green-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Analyse qualité de l'air...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    return (
      <div className="space-y-3">
        <div className="grid grid-cols-3 gap-2">
          <div className="rounded-xl border border-white/10 bg-forest-900/40 px-2 py-1.5 text-center">
            <p className="text-sm font-bold text-green-300">{data.air_quality?.pm2_5_ugm3}</p>
            <p className="text-[9px] text-forest-400">PM2.5 µg/m³</p>
          </div>
          <div className="rounded-xl border border-white/10 bg-forest-900/40 px-2 py-1.5 text-center">
            <p className="text-sm font-bold text-green-300">{data.air_quality?.ozone_ugm3}</p>
            <p className="text-[9px] text-forest-400">O₃ µg/m³</p>
          </div>
          <div className="rounded-xl border border-white/10 bg-forest-900/40 px-2 py-1.5 text-center">
            <p className="text-sm font-bold text-green-300">{data.air_quality?.no2_ugm3}</p>
            <p className="text-[9px] text-forest-400">NO₂ µg/m³</p>
          </div>
        </div>
        <div className={`rounded-xl border px-3 py-2 text-xs ${data.ems_impact?.risk_level === "ÉLEVÉ" ? "border-red-500/20 bg-red-500/5 text-red-300" : "border-brand-500/20 bg-brand-500/5 text-brand-300"}`}>
          IQA : {data.air_quality?.iqa_level} — Impact EMS : {data.ems_impact?.estimated_call_increase_pct > 0 ? `+${data.ems_impact?.estimated_call_increase_pct}%` : "Baseline"}
        </div>
        <button onClick={load} className="text-[10px] text-green-400 hover:text-green-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
      </div>
    );
  };

  // ─── Widget Pandemic Preparedness ────────────────────────────────────────────
  const PandemicPreparednessWidget = () => {
    const [data, setData] = useState<PandemicPreparednessResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = () => { setLoading(true); setError(null); fetchPandemicPreparedness().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return <button onClick={load} className="w-full rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300 hover:bg-red-500/20 transition flex items-center justify-center gap-2"><AlertTriangle size={14} />Simuler la préparation pandémique (SEIR)</button>;
    if (loading) return <div className="flex items-center justify-center py-4 text-red-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Simulation SEIR + Monte-Carlo...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    const forecast = data["30d_forecast"];
    return (
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-xl border border-white/10 bg-forest-900/40 px-3 py-2 text-center">
            <p className="text-lg font-bold text-red-300">{forecast?.peak_infected?.toLocaleString()}</p>
            <p className="text-[9px] text-forest-400">Pic infectés (J+{forecast?.peak_day})</p>
          </div>
          <div className="rounded-xl border border-white/10 bg-forest-900/40 px-3 py-2 text-center">
            <p className={`text-lg font-bold ${forecast?.peak_icu_required > forecast?.icu_capacity ? "text-red-400" : "text-brand-400"}`}>{forecast?.peak_icu_required}</p>
            <p className="text-[9px] text-forest-400">Lits réa requis / {forecast?.icu_capacity} dispo</p>
          </div>
        </div>
        <div className={`rounded-xl border px-3 py-2 text-xs ${data.preparedness_assessment?.includes("CRITIQUE") ? "border-red-500/20 bg-red-500/5 text-red-300" : "border-gold-500/20 bg-gold-500/5 text-gold-300"}`}>
          R0={data.parameters?.R0} — {data.preparedness_assessment}
        </div>
        <button onClick={load} className="text-[10px] text-red-400 hover:text-red-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
      </div>
    );
  };

  // ─── Widget Cross-Border ──────────────────────────────────────────────────────
  const CrossBorderWidget = () => {
    const [data, setData] = useState<CrossBorderResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = () => { setLoading(true); setError(null); fetchCrossBorder().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return <button onClick={load} className="w-full rounded-xl border border-brand-500/30 bg-brand-500/10 px-4 py-3 text-sm text-brand-300 hover:bg-brand-500/20 transition flex items-center justify-center gap-2"><MapPin size={14} />Analyser la coordination transfrontalière CH/FR</button>;
    if (loading) return <div className="flex items-center justify-center py-4 text-brand-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Analyse des accords bilatéraux...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>Statut : <span className="font-bold text-white">{data.coordination_status}</span></span>
          <span>{data.active_agreements} accords actifs · {data.total_daily_capacity} interventions/j</span>
          <button onClick={load} className="ml-auto text-brand-400 hover:text-brand-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-xl border border-white/10 bg-forest-900/40 px-3 py-2 text-center">
            <p className="text-lg font-bold text-brand-300">{data.available_resources?.CH}</p>
            <p className="text-[9px] text-forest-400">Ressources CH</p>
          </div>
          <div className="rounded-xl border border-white/10 bg-forest-900/40 px-3 py-2 text-center">
            <p className="text-lg font-bold text-brand-300">{data.available_resources?.FR}</p>
            <p className="text-[9px] text-forest-400">Ressources FR</p>
          </div>
        </div>
      </div>
    );
  };

  // ─── Widget Situational Awareness ────────────────────────────────────────────
  const SituationalAwarenessWidget = () => {
    const [data, setData] = useState<SituationalAwarenessResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = () => { setLoading(true); setError(null); fetchSituationalAwareness().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return <button onClick={load} className="w-full rounded-xl border border-brand-500/30 bg-brand-500/10 px-4 py-3 text-sm text-brand-300 hover:bg-brand-500/20 transition flex items-center justify-center gap-2"><Activity size={14} />Afficher la conscience situationnelle temps réel</button>;
    if (loading) return <div className="flex items-center justify-center py-4 text-brand-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Agrégation des sources temps réel...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    const ind = data.real_time_indicators;
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-[10px] text-forest-400">
          <span>Statut : <span className="font-bold text-white">{data.overall_status}</span></span>
          <button onClick={load} className="ml-auto text-brand-400 hover:text-brand-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>
        <div className="grid grid-cols-3 gap-2">
          {[
            { label: "Incidents actifs", value: ind?.active_incidents, color: "text-brand-300" },
            { label: "Équipes dispo", value: ind?.available_ems_crews, color: "text-brand-300" },
            { label: "Occupation URG", value: `${ind?.ed_occupancy_pct}%`, color: ind?.ed_occupancy_pct > 80 ? "text-red-400" : "text-gold-300" },
            { label: "Appels en attente", value: ind?.pending_calls_in_queue, color: ind?.pending_calls_in_queue > 5 ? "text-red-400" : "text-forest-300" },
            { label: "Trans. actifs", value: ind?.cross_border_active, color: "text-brand-300" },
            { label: "Risque météo", value: ind?.weather_risk, color: "text-gold-300" },
          ].map((item) => (
            <div key={item.label} className="rounded-xl border border-white/10 bg-forest-900/40 px-2 py-1.5 text-center">
              <p className={`text-sm font-bold ${item.color}`}>{item.value}</p>
              <p className="text-[9px] text-forest-400">{item.label}</p>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // ─── Widget Disaster Risk ─────────────────────────────────────────────────────
  const DisasterRiskWidget = () => {
    const [data, setData] = useState<DisasterRiskResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = () => { setLoading(true); setError(null); fetchDisasterRisk().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return <button onClick={load} className="w-full rounded-xl border border-gold-500/30 bg-gold-500/10 px-4 py-3 text-sm text-gold-300 hover:bg-gold-500/20 transition flex items-center justify-center gap-2"><AlertTriangle size={14} />Évaluer les risques de catastrophes</button>;
    if (loading) return <div className="flex items-center justify-center py-4 text-gold-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Analyse des risques géospatiaux...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    const riskC: Record<string, string> = { FAIBLE: "text-brand-400", MODÉRÉ: "text-gold-400", ÉLEVÉ: "text-gold-400", CRITIQUE: "text-red-400" };
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>Risque global : <span className={`font-bold ${riskC[data.overall_risk_level] ?? "text-white"}`}>{data.overall_risk_level}</span></span>
          <button onClick={load} className="ml-auto text-gold-400 hover:text-gold-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>
        <div className="space-y-1.5 max-h-40 overflow-y-auto">
          {data.all_risks?.map((r, i) => (
            <div key={i} className="rounded-xl border border-white/10 bg-forest-900/40 px-3 py-2 flex items-center justify-between gap-2">
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-white truncate">{r.type} — {r.zone}</p>
                <p className="text-[9px] text-forest-400">{(r.probability_annual * 100).toFixed(0)}% annuel · {r.population_at_risk?.toLocaleString()} pers.</p>
              </div>
              <span className={`text-xs font-bold shrink-0 ${riskC[r.risk_level] ?? "text-white"}`}>{r.risk_level}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // ─── Widget MCI Victim ────────────────────────────────────────────────────────
  const MCIVictimWidget = () => {
    const [data, setData] = useState<MCIVictimResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const load = () => { setLoading(true); setError(null); fetchMCIVictim().then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false)); };
    if (!data && !loading && !error) return <button onClick={load} className="w-full rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300 hover:bg-rose-500/20 transition flex items-center justify-center gap-2"><Users size={14} />Estimer le nombre de victimes AME</button>;
    if (loading) return <div className="flex items-center justify-center py-4 text-rose-300 text-sm gap-2"><RotateCcw size={14} className="animate-spin" />Estimation des victimes en cours...</div>;
    if (error) return <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-300">Erreur : {error} <button onClick={load} className="ml-2 underline">Réessayer</button></div>;
    if (!data) return null;
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-forest-400">
          <span>Incident : <span className="text-white font-semibold">{data.incident?.type}</span> — {data.incident?.location}</span>
          <button onClick={load} className="ml-auto text-rose-400 hover:text-rose-300 flex items-center gap-1"><RefreshCw size={10} /> Actualiser</button>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 px-3 py-2 text-center">
            <p className="text-2xl font-bold text-rose-300">{data.estimated_victims}</p>
            <p className="text-[9px] text-forest-400">Victimes estimées</p>
          </div>
          <div className="rounded-xl border border-white/10 bg-forest-900/40 px-3 py-2 text-center">
            <p className="text-xs font-bold text-red-400">T1: {data.triage_distribution?.T1_critical}</p>
            <p className="text-xs font-bold text-gold-400">T2: {data.triage_distribution?.T2_serious}</p>
            <p className="text-xs font-bold text-brand-400">T3: {data.triage_distribution?.T3_minor}</p>
          </div>
        </div>
        <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 px-3 py-2 text-xs text-rose-300">
          SMUR: {data.recommended_resources?.SMUR} · AMB: {data.recommended_resources?.AMBULANCE} · Médecins: {data.recommended_resources?.MÉDECINS}
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
                  {scenario.articleCount} articles
                </span>
              ) : (
                <span className="rounded-full bg-forest-700/40 border border-white/5 px-2 py-0.5 text-xs text-forest-500">
                  0 articles
                </span>
              )}
            </div>
            <p className="mt-1 text-sm leading-5 text-forest-400 line-clamp-2">{scenario.description}</p>
          </div>
          <div className="shrink-0 flex items-center gap-2">
            <button
              onClick={(e) => { e.stopPropagation(); setDetailScenarioId(scenario.id); }}
              className="rounded-xl border border-brand-500/20 bg-brand-500/10 px-2.5 py-1 text-[10px] text-brand-300 hover:bg-brand-500/20 transition font-medium"
              title="Ouvrir la page détail du scénario"
            >
              Page détail
            </button>
            {isExpanded ? <ChevronUp size={16} className="text-forest-500" /> : <ChevronDown size={16} className="text-forest-500" />}
          </div>
        </div>

        {isExpanded && (
          <div className="mt-4 space-y-4 border-t border-white/10 pt-4">
            {/* Living Evidence Note */}
            <div className={`rounded-2xl border px-3 py-2 text-xs ${
              hasArticles
                ? "border-brand-500/20 bg-brand-500/5 text-brand-300"
                : "border-white/5 bg-white/2 text-forest-500"
            }`}>
              <RefreshCw size={10} className="inline mr-1" />
              {scenario.livingEvidenceNote}
            </div>

            {/* Actions recommandées */}
            <div>
              <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-forest-400">Actions recommandées</h4>
              <ul className="space-y-1.5">
                {scenario.recommendedActions.map((action, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-forest-200">
                    <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-brand-400" />
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
              <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Activity size={14} className="text-brand-400" />
                    <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">Modèle Prédictif — Surveillance Épidémique J+14</span>
                  </div>
                  <span className="text-[10px] text-brand-400 bg-brand-500/10 border border-brand-500/20 px-2 py-0.5 rounded-full">SARIMAX + Sentinelles FR</span>
                </div>
                <EpidemicEarlyWarningWidget />
              </div>
            )}

            {/* Widget Response Time Optimization */}
            {scenario.id === "response-time-optimization" && (
              <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <MapPin size={14} className="text-brand-400" />
                    <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">Modèle Prédictif — Optimisation Temps de Réponse</span>
                  </div>
                  <span className="text-[10px] text-brand-400 bg-brand-500/10 border border-brand-500/20 px-2 py-0.5 rounded-full">OSRM + Open-Meteo</span>
                </div>
                <ResponseTimeWidget />
              </div>
            )}

            {/* Widget Cardiac Arrest Prediction (OHCA) */}
            {scenario.id === "cardiac-arrest-prediction" && (
              <div className="rounded-2xl border border-rose-500/20 bg-rose-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Activity size={14} className="text-rose-400" />
                    <span className="text-xs font-semibold text-rose-300 uppercase tracking-wider">Modèle Prédictif — Arrêts Cardiaques OHCA J+3</span>
                  </div>
                  <span className="text-[10px] text-rose-400 bg-rose-500/10 border border-rose-500/20 px-2 py-0.5 rounded-full">LightGBM + Météo</span>
                </div>
                <CardiacArrestWidget />
              </div>
            )}

            {/* Widget Heatwave EMS Impact */}
            {scenario.id === "heatwave-ems-impact" && (
              <div className="rounded-2xl border border-gold-500/20 bg-gold-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Cloud size={14} className="text-gold-400" />
                    <span className="text-xs font-semibold text-gold-300 uppercase tracking-wider">Modèle Prédictif — Impact Canicule EMS J+7</span>
                  </div>
                  <span className="text-[10px] text-gold-400 bg-gold-500/10 border border-gold-500/20 px-2 py-0.5 rounded-full">DLNM + UTCI</span>
                </div>
                <HeatwaveEMSWidget />
              </div>
            )}

            {/* Widget Stroke Detection */}
            {scenario.id === "stroke-detection" && (
              <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Activity size={14} className="text-brand-400" />
                    <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">Modèle Prédictif — AVC Door-to-Needle</span>
                  </div>
                  <span className="text-[10px] text-brand-400 bg-brand-500/10 border border-brand-500/20 px-2 py-0.5 rounded-full">XGBoost + NIHSS</span>
                </div>
                <StrokeDetectionWidget />
              </div>
            )}

            {/* Widget Triage Support */}
            {scenario.id === "triage-support" && (
              <div className="rounded-2xl border border-gold-500/20 bg-gold-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <CheckSquare size={14} className="text-gold-400" />
                    <span className="text-xs font-semibold text-gold-300 uppercase tracking-wider">Aide au Triage — CCMU / NEWS2</span>
                  </div>
                  <span className="text-[10px] text-gold-400 bg-gold-500/10 border border-gold-500/20 px-2 py-0.5 rounded-full">CCMU + NEWS2</span>
                </div>
                <TriageSupportWidget />
              </div>
            )}

            {/* Widget Undertriage Risk */}
            {scenario.id === "undertriage-risk" && (
              <div className="rounded-2xl border border-rose-500/20 bg-rose-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <AlertTriangle size={14} className="text-rose-400" />
                    <span className="text-xs font-semibold text-rose-300 uppercase tracking-wider">Détection Sous-Triage EMS</span>
                  </div>
                  <span className="text-[10px] text-rose-400 bg-rose-500/10 border border-rose-500/20 px-2 py-0.5 rounded-full">RF + Régression logistique</span>
                </div>
                <UndertriageRiskWidget />
              </div>
            )}

            {/* Widget Trauma Care */}
            {scenario.id === "trauma-care" && (
              <div className="rounded-2xl border border-red-500/20 bg-red-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Activity size={14} className="text-red-400" />
                    <span className="text-xs font-semibold text-red-300 uppercase tracking-wider">Scores Trauma — ISS / RTS / TRISS</span>
                  </div>
                  <span className="text-[10px] text-red-400 bg-red-500/10 border border-red-500/20 px-2 py-0.5 rounded-full">Cox Survival + ISS</span>
                </div>
                <TraumaCareWidget />
              </div>
            )}

            {/* Widget Mass Casualty */}
            {scenario.id === "mass-casualty" && (
              <div className="rounded-2xl border border-violet-500/20 bg-violet-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Users size={14} className="text-violet-400" />
                    <span className="text-xs font-semibold text-violet-300 uppercase tracking-wider">Simulation AME — SALT Triage</span>
                  </div>
                  <span className="text-[10px] text-violet-400 bg-violet-500/10 border border-violet-500/20 px-2 py-0.5 rounded-full">Monte-Carlo + SALT</span>
                </div>
                <MassCasualtyWidget />
              </div>
            )}

            {/* Widget Clinical Deterioration */}
            {scenario.id === "clinical-deterioration-prediction" && (
              <div className="rounded-2xl border border-rose-500/20 bg-rose-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Activity size={14} className="text-rose-400" />
                    <span className="text-xs font-semibold text-rose-300 uppercase tracking-wider">Détection Dégradation Clinique — NEWS2 / MEWS</span>
                  </div>
                  <span className="text-[10px] text-rose-400 bg-rose-500/10 border border-rose-500/20 px-2 py-0.5 rounded-full">NEWS2 + MEWS + LSTM</span>
                </div>
                <ClinicalDeteriorationWidget />
              </div>
            )}

            {/* Widget Emergency Call Qualification */}
            {(scenario.id === "emergency-call-qualification" || scenario.id === "call-prioritization") && (
              <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Radio size={14} className="text-brand-400" />
                    <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">Qualification Appels — NLP + Prioritisation</span>
                  </div>
                  <span className="text-[10px] text-brand-400 bg-brand-500/10 border border-brand-500/20 px-2 py-0.5 rounded-full">NLP + Scoring</span>
                </div>
                <CallQualificationWidget />
              </div>
            )}

            {/* Widget Dispatch Decision Support */}
            {scenario.id === "dispatch-decision-support" && (
              <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <MapPin size={14} className="text-brand-400" />
                    <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">Aide à la Décision Dispatch</span>
                  </div>
                  <span className="text-[10px] text-brand-400 bg-brand-500/10 border border-brand-500/20 px-2 py-0.5 rounded-full">Arbre de décision + VRP</span>
                </div>
                <DispatchDecisionWidget />
              </div>
            )}

            {/* Widget Patient Pathway */}
            {scenario.id === "patient-pathway-optimization" && (
              <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <MapPin size={14} className="text-brand-400" />
                    <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">Optimisation Parcours Patient Transfrontalier</span>
                  </div>
                  <span className="text-[10px] text-brand-400 bg-brand-500/10 border border-brand-500/20 px-2 py-0.5 rounded-full">OSRM + PL</span>
                </div>
                <PatientPathwayWidget />
              </div>
            )}

            {/* Widget Ambulance Dispatch Optimization */}
            {scenario.id === "ambulance-dispatch-optimization" && (
              <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <MapPin size={14} className="text-brand-400" />
                    <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">Optimisation Couverture Ambulancière — VRP</span>
                  </div>
                  <span className="text-[10px] text-brand-400 bg-brand-500/10 border border-brand-500/20 px-2 py-0.5 rounded-full">VRP + Couverture spatiale</span>
                </div>
                <AmbulanceDispatchWidget />
              </div>
            )}

            {/* Widget Hospital Capacity */}
            {(scenario.id === "hospital-capacity-forecasting" || scenario.id === "staffing-level-prediction") && (
              <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <BarChart2 size={14} className="text-brand-400" />
                    <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">Capacité Hospitalière & Staffing EMS</span>
                  </div>
                  <span className="text-[10px] text-brand-400 bg-brand-500/10 border border-brand-500/20 px-2 py-0.5 rounded-full">Prophet + NEDOCS</span>
                </div>
                <HospitalCapacityWidget />
              </div>
            )}

            {/* Widget Surveillance */}
            {scenario.id === "surveillance" && (
              <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Activity size={14} className="text-brand-400" />
                    <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">Surveillance Anomalies EMS</span>
                  </div>
                  <span className="text-[10px] text-brand-400 bg-brand-500/10 border border-brand-500/20 px-2 py-0.5 rounded-full">Isolation Forest + Z-score</span>
                </div>
                <SurveillanceWidget />
              </div>
            )}

            {/* Widget Surge Management */}
            {scenario.id === "surge-management" && (
              <div className="rounded-2xl border border-gold-500/20 bg-gold-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Zap size={14} className="text-gold-400" />
                    <span className="text-xs font-semibold text-gold-300 uppercase tracking-wider">Gestion de Surge — File d'Attente</span>
                  </div>
                  <span className="text-[10px] text-gold-400 bg-gold-500/10 border border-gold-500/20 px-2 py-0.5 rounded-full">M/M/c + Erlang</span>
                </div>
                <SurgeManagementWidget />
              </div>
            )}

            {/* Widget Resource Allocation */}
            {scenario.id === "resource-allocation" && (
              <div className="rounded-2xl border border-violet-500/20 bg-violet-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <BarChart2 size={14} className="text-violet-400" />
                    <span className="text-xs font-semibold text-violet-300 uppercase tracking-wider">Allocation Optimale des Ressources</span>
                  </div>
                  <span className="text-[10px] text-violet-400 bg-violet-500/10 border border-violet-500/20 px-2 py-0.5 rounded-full">Optimisation PL</span>
                </div>
                <ResourceAllocationWidget />
              </div>
            )}

            {/* Widget Environmental Risk */}
            {scenario.id === "environmental-risk-forecasting" && (
              <div className="rounded-2xl border border-green-500/20 bg-green-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Cloud size={14} className="text-green-400" />
                    <span className="text-xs font-semibold text-green-300 uppercase tracking-wider">Risque Environnemental — Qualité de l'Air</span>
                  </div>
                  <span className="text-[10px] text-green-400 bg-green-500/10 border border-green-500/20 px-2 py-0.5 rounded-full">XGBoost + PM2.5/O3</span>
                </div>
                <EnvironmentalRiskWidget />
              </div>
            )}

            {/* Widget Pandemic Preparedness */}
            {scenario.id === "pandemic-preparedness" && (
              <div className="rounded-2xl border border-red-500/20 bg-red-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <AlertTriangle size={14} className="text-red-400" />
                    <span className="text-xs font-semibold text-red-300 uppercase tracking-wider">Préparation Pandémique — Modèle SEIR</span>
                  </div>
                  <span className="text-[10px] text-red-400 bg-red-500/10 border border-red-500/20 px-2 py-0.5 rounded-full">SEIR + Monte-Carlo</span>
                </div>
                <PandemicPreparednessWidget />
              </div>
            )}

            {/* Widget Cross-Border Coordination */}
            {scenario.id === "cross-border-coordination" && (
              <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <MapPin size={14} className="text-brand-400" />
                    <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">Coordination Transfrontalière CH/FR</span>
                  </div>
                  <span className="text-[10px] text-brand-400 bg-brand-500/10 border border-brand-500/20 px-2 py-0.5 rounded-full">Accords bilatéraux</span>
                </div>
                <CrossBorderWidget />
              </div>
            )}

            {/* Widget Situational Awareness */}
            {scenario.id === "situational-awareness" && (
              <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Activity size={14} className="text-brand-400" />
                    <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">Conscience Situationnelle Temps Réel</span>
                  </div>
                  <span className="text-[10px] text-brand-400 bg-brand-500/10 border border-brand-500/20 px-2 py-0.5 rounded-full">Dashboard multi-sources</span>
                </div>
                <SituationalAwarenessWidget />
              </div>
            )}

            {/* Widget Disaster Risk */}
            {scenario.id === "disaster-risk-assessment" && (
              <div className="rounded-2xl border border-gold-500/20 bg-gold-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <AlertTriangle size={14} className="text-gold-400" />
                    <span className="text-xs font-semibold text-gold-300 uppercase tracking-wider">Évaluation Risques Catastrophes</span>
                  </div>
                  <span className="text-[10px] text-gold-400 bg-gold-500/10 border border-gold-500/20 px-2 py-0.5 rounded-full">Risque géospatial</span>
                </div>
                <DisasterRiskWidget />
              </div>
            )}

            {/* Widget MCI Victim Estimation */}
            {scenario.id === "mci-victim-estimation" && (
              <div className="rounded-2xl border border-rose-500/20 bg-rose-500/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Users size={14} className="text-rose-400" />
                    <span className="text-xs font-semibold text-rose-300 uppercase tracking-wider">Estimation Victimes AME — Régression spatiale</span>
                  </div>
                  <span className="text-[10px] text-rose-400 bg-rose-500/10 border border-rose-500/20 px-2 py-0.5 rounded-full">Régression spatiale</span>
                </div>
                <MCIVictimWidget />
              </div>
            )}

            {/* Articles associés */}
            {scenario.relevantArticles.length > 0 && (
              <div>
                <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-forest-400">
                  Articles récents ({scenario.articleCount} total, 5 affichés)
                </h4>
                <div className="space-y-2">
                  {scenario.relevantArticles.map((article) => (
                    <div key={article.id} className="rounded-xl border border-white/10 bg-forest-900/40 px-3 py-2">
                      <p className="text-sm font-medium text-forest-200 leading-5">{article.title}</p>
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

  return (
    <div className="space-y-6">
      {/* En-tête */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity size={20} className="text-brand-400" />
          <div>
            <h2 className="text-xl font-semibold text-white">Scénarios GESICA — LiteRev - Evidence to Scenario</h2>
            <p className="text-xs text-forest-400 mt-0.5">
              {scenarios.length} scénarios · {scenarios.reduce((a, s) => a + s.articleCount, 0).toLocaleString()} articles indexés · Mis à jour automatiquement
            </p>
          </div>
        </div>
        <span className="rounded-full border border-brand-500/20 bg-brand-500/10 px-3 py-1 text-xs text-brand-300 font-medium">
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
                ? "border-brand-400 bg-brand-500/20 text-white"
                : "border-white/10 bg-white/5 text-forest-400 hover:bg-white/10"
            }`}
          >
            {cluster === "all" ? `Tous (${scenarios.length})` : `${cluster} (${scenarios.filter(s => s.cluster === cluster).length})`}
          </button>
        ))}
      </div>

      {/* Scénarios avec articles */}
      {withArticles.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-forest-300 flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-brand-400" />
            Scénarios actifs ({withArticles.length})
          </h3>
          {withArticles.map((s) => <ScenarioCard key={s.id} scenario={s} />)}
        </div>
      )}

      {/* Scénarios sans articles */}
      {withoutArticles.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-forest-500 flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-forest-600" />
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
    return <div className="text-sm text-forest-400">Chargement du module de screening...</div>;
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
            <CheckSquare size={18} className="text-brand-400" />
            Diagramme de Flux PRISMA — {projectContext.toUpperCase()}
          </h3>
          <div className="grid gap-4 md:grid-cols-4 text-center">
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-4">
              <div className="text-2xl font-bold text-brand-300">{prismaFlow.recordsIdentified}</div>
              <div className="mt-1 text-xs text-forest-400 uppercase tracking-wider">Identifiés (Stage 1)</div>
            </div>
            <div className="flex flex-col justify-center items-center">
              <ArrowDown size={16} className="text-forest-500 mb-2" />
              <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-4 w-full">
                <div className="text-2xl font-bold text-gold-400">{prismaFlow.recordsScreened}</div>
                <div className="mt-1 text-xs text-forest-400 uppercase tracking-wider">Screenés (Titre/Abstract)</div>
              </div>
            </div>
            <div className="flex flex-col justify-center items-center">
              <div className="text-xs text-rose-400 font-semibold mb-1">-{prismaFlow.recordsExcluded} Exclus</div>
              <ArrowDown size={16} className="text-forest-500 mb-2" />
              <div className="rounded-2xl border border-brand-500/20 bg-brand-500/5 p-4 w-full">
                <div className="text-2xl font-bold text-brand-400">{prismaFlow.recordsIncluded}</div>
                <div className="mt-1 text-xs text-forest-400 uppercase tracking-wider">Inclus (Full-text)</div>
              </div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-forest-900/60 p-4 flex flex-col justify-center">
              <div className="text-xs text-forest-400">Taux d'Inclusion</div>
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
              <span className="text-xs text-forest-400 font-normal">{pendingDocs.length} en attente</span>
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
                      ? "border-brand-400 bg-brand-500/10"
                      : "border-white/5 bg-white/5 hover:border-white/20"
                  }`}
                >
                  <span className="text-xs font-semibold text-forest-200 line-clamp-2">{d.title}</span>
                  <div className="flex items-center justify-between text-[10px] text-forest-400 w-full">
                    <span>{d.source} · {d.year ?? "—"}</span>
                    {d.screeningStatus === "included" && (
                      <span className="flex items-center gap-0.5 text-brand-400 font-semibold">
                        <CheckCircle size={10} /> Inclus
                      </span>
                    )}
                    {d.screeningStatus === "excluded" && (
                      <span className="flex items-center gap-0.5 text-rose-400 font-semibold">
                        <XCircle size={10} /> Exclu
                      </span>
                    )}
                    {(!d.screeningStatus || d.screeningStatus === "pending") && (
                      <span className="flex items-center gap-0.5 text-gold-400 font-semibold">
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
                <span className="rounded bg-white/5 px-2 py-1 text-xs text-forest-400">
                  {selectedDoc.source} · {selectedDoc.year ?? "—"}
                </span>
                <h3 className="text-2xl font-semibold text-white mt-3">{selectedDoc.title}</h3>
                {selectedDoc.url && (
                  <a
                    href={selectedDoc.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 text-xs text-brand-400 hover:text-brand-300 mt-2"
                  >
                    Voir l'article d'origine <ExternalLink size={10} />
                  </a>
                )}
              </div>

              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-forest-400 mb-2">Abstract</h4>
                <p className="rounded-2xl border border-white/10 bg-forest-950/60 p-4 leading-6 text-sm text-forest-200">
                  {selectedDoc.abstract || "Aucun abstract disponible."}
                </p>
              </div>

              {/* Formulaire de Décision */}
              <div className="border-t border-white/10 pt-5 space-y-4">
                <h4 className="text-sm font-semibold text-white">Décision de Screening (Inclusion / Exclusion)</h4>
                
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <label className="block text-xs text-forest-400 mb-1">Raison de l'exclusion (obligatoire si exclu)</label>
                    <select
                      value={reason}
                      onChange={(e) => setReason(e.target.value)}
                      className="w-full rounded-xl border border-white/10 bg-forest-950/80 p-3 text-sm text-white outline-none focus:border-brand-400"
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
                    <label className="block text-xs text-forest-400 mb-1">Notes de screening</label>
                    <input
                      value={notes}
                      onChange={(e) => setNotes(e.target.value)}
                      placeholder="Ex. Excellente étude de prévision par LSTM à Genève"
                      className="w-full rounded-xl border border-white/10 bg-forest-950/80 p-3 text-sm text-white outline-none focus:border-brand-400"
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
                    className="rounded-xl border border-brand-500/30 bg-brand-500/10 hover:bg-brand-500/20 px-5 py-3 text-sm font-semibold text-brand-200 transition flex items-center gap-2"
                  >
                    <CheckCircle size={16} />
                    Inclure dans le corpus final
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl text-center text-forest-400 py-20">
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
          <Zap size={18} className="text-brand-400" />
          Assistant Scientifique RAG — {projectContext.toUpperCase()}
        </h2>
        <p className="mb-4 text-sm text-forest-300 leading-6">
          Posez une question complexe à l'assistant. Il va interroger les chunks de la base de données les plus pertinents pour votre projet, puis synthétiser une réponse scientifiquement étayée et citer ses sources.
        </p>

        <div className="flex flex-col gap-3 sm:flex-row">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onAsk()}
            placeholder="Ex. Quelles sont les meilleures méthodes d'IA pour prédire l'afflux de patients aux urgences ?"
            className="min-h-14 flex-1 rounded-2xl border border-white/10 bg-forest-950/80 px-4 text-white outline-none placeholder:text-forest-500 focus:border-brand-400"
          />
          <button
            type="button"
            onClick={onAsk}
            disabled={loading || !question.trim()}
            className="min-h-14 rounded-2xl bg-brand-400 px-6 font-semibold text-forest-950 transition hover:bg-brand-300 disabled:cursor-not-allowed disabled:opacity-60 flex items-center justify-center gap-2"
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
        <div className="rounded-3xl border border-dashed border-white/10 bg-white/5 p-10 text-center text-forest-300 flex flex-col items-center justify-center gap-3">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-brand-400 border-t-transparent" />
          <p className="text-sm">L'assistant analyse les articles scientifiques et rédige sa synthèse...</p>
        </div>
      )}

      {response && (
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
          <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl space-y-4">
            <h3 className="text-lg font-semibold text-white">Synthèse de l'Assistant</h3>
            <div className="prose prose-invert max-w-none text-sm leading-7 text-forest-200 whitespace-pre-wrap">
              {response.answer}
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl space-y-4 h-fit">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              <BookOpen size={16} className="text-brand-400" />
              Sources utilisées ({response.sources.length})
            </h3>
            <div className="space-y-3">
              {response.sources.map((s, i) => (
                <div key={s.documentId} className="rounded-2xl border border-white/10 bg-forest-900/60 p-4 space-y-2">
                  <div className="flex items-start justify-between gap-2">
                    <span className="inline-flex shrink-0 h-5 w-5 items-center justify-center rounded-full bg-brand-400/20 text-xs font-bold text-brand-300">
                      {i + 1}
                    </span>
                    {s.url && (
                      <a
                        href={s.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-forest-400 hover:text-white"
                      >
                        <ExternalLink size={12} />
                      </a>
                    )}
                  </div>
                  <h4 className="text-sm font-semibold text-white leading-5">{s.title}</h4>
                  <div className="flex flex-wrap gap-1 text-[10px]">
                    <span className="rounded bg-white/5 px-1.5 py-0.5 text-forest-400">{s.source}</span>
                    {s.year && <span className="rounded bg-white/5 px-1.5 py-0.5 text-forest-400">{s.year}</span>}
                    {s.evidenceStrength && (
                      <span className="rounded bg-brand-500/10 px-1.5 py-0.5 text-brand-300 capitalize">
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
    { id: "assistant", label: "Assistant RAG", icon: <Zap size={14} className="text-brand-400" /> },
    { id: "screening", label: "Screening PRISMA", icon: <CheckSquare size={14} className="text-brand-400" /> },
    { id: "scenarios", label: "Scénarios GESICA", icon: <Activity size={14} /> },
    { id: "terrain", label: "Données Terrain", icon: <Cloud size={14} className="text-brand-400" /> },
    { id: "stats", label: "Statistiques", icon: <BarChart2 size={14} /> },
  ];

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,rgba(10,54,33,0.18),transparent_35%),linear-gradient(180deg,#0a1410_0%,#121e19_100%)] text-white">
      <header className="border-b border-white/8 bg-[#0a1410]/80 backdrop-blur-xl">
        <div className="mx-auto max-w-[1380px] px-6 py-6">
          <div className="flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
            <div className="max-w-3xl">
              <div className="flex items-center gap-3 mb-2">
                <img src="/logo.jpg" alt="LiteRev arbre" className="h-14 w-14 rounded-xl object-cover" />
              </div>
              <h1 className="mt-3 flex items-center gap-3">
                <img src="/literev-logo.png" alt="LiteRev" className="h-10 w-auto object-contain" />
                <span className="text-4xl font-semibold tracking-tight text-white">- Evidence to Scenario</span>
              </h1>
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
                      ? "border-gold-400 bg-brand-700/60 text-gold-400 font-semibold shadow-2xl"
                      : "border-white/20 bg-white/5 text-white/90 hover:border-gold-400/40 hover:bg-white/10 hover:text-white"
                  }`}
                >
                  <div className="text-sm font-semibold">{label}</div>
                </button>
              ))}
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
                    : "text-white hover:text-gold-300 hover:bg-white/8"
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
                    className="flex items-center gap-1 rounded-xl border border-white/10 px-2 py-1 text-xs text-forest-400 transition hover:border-white/20 hover:text-forest-200"
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
                        <span className="mb-2 block text-sm font-medium text-forest-200">
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
                          className="w-full appearance-none rounded-2xl border border-white/10 bg-forest-950/80 px-3 py-3 text-sm text-white focus:border-brand-400 focus:outline-none"
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
                    <span className="mb-2 block text-sm font-medium text-forest-200">
                      Année{" "}
                      <span className="font-mono text-brand-300">
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
                        className="w-full accent-gold-400"
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
                        className="w-full accent-gold-400"
                      />
                    </div>
                  </div>
                </div>
              </div>
            </aside>

            <section className="space-y-6">
              <section className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
                <div className="mb-4 flex items-center gap-2 rounded-2xl border border-white/10 bg-forest-900/80 p-1 text-sm">
                  {(["semantic", "boolean"] as SearchMode[]).map((item) => (
                    <button
                      key={item}
                      type="button"
                      onClick={() => setMode(item)}
                      className={`rounded-xl px-4 py-2 capitalize transition ${
                        mode === item
                          ? "bg-brand-500 text-forest-950"
                          : "text-forest-300 hover:bg-white/10"
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
                    className="min-h-14 flex-1 rounded-2xl border border-white/10 bg-forest-950/80 px-4 text-white outline-none placeholder:text-forest-500 focus:border-brand-400"
                  />
                  <button
                    type="button"
                    onClick={handleSearch}
                    disabled={loading}
                    className="min-h-14 rounded-2xl bg-brand-400 px-6 font-semibold text-forest-950 transition hover:bg-brand-300 disabled:cursor-not-allowed disabled:opacity-60"
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
                <div className="rounded-3xl border border-dashed border-white/10 bg-white/5 p-10 text-center text-forest-300">
                  Lancez une recherche pour afficher les résultats.
                </div>
              )}

              {hasResults && (
                <>
                  <div className="flex items-center justify-between">
                    <p className="text-sm text-forest-400">
                      <span className="font-semibold text-white">{dedupedResults.length}</span>{" "}
                      résultat{dedupedResults.length > 1 ? "s" : ""} · {totalPages > 1 ? `page ${page}/${totalPages}` : "1 page"}
                    </p>
                    <div className="flex gap-2">
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
                                className="inline-flex shrink-0 items-center gap-2 rounded-xl border border-white/10 px-3 py-2 text-sm text-forest-200 hover:bg-white/10"
                              >
                                Source
                                <ExternalLink size={14} />
                              </a>
                            )}
                          </div>

                          <div className="mt-3 flex flex-wrap gap-2 text-xs text-forest-400">
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
                              <span className="rounded-full bg-brand-500/10 px-2 py-1 text-brand-200">
                                {result.projectContext}
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
                            {/* Badge couverture textuelle */}
                            <span className={`rounded-full px-2 py-1 border text-[11px] font-semibold ${
                              result.chunkType === 'fulltext_section'
                                ? 'bg-brand-500/10 border-brand-500/20 text-brand-400'
                                : 'bg-forest-800/50 border-white/5 text-forest-500'
                            }`} title={result.chunkType === 'fulltext_section' ? 'Texte intégral indexé' : 'Titre + résumé uniquement'}>
                              {result.chunkType === 'fulltext_section' ? 'Full Text' : 'Abstract'}
                            </span>
                          </div>

                          <p className="mt-4 text-sm leading-6 text-forest-200">
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
                                      ? "border-brand-400 bg-brand-500/15 text-brand-200"
                                      : "border-white/10 bg-white/5 text-forest-400 hover:border-white/20 hover:text-forest-200"
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
                            className="rounded-xl border border-white/10 px-4 py-2 text-sm text-forest-300 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-30"
                          >
                            Précédent
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
                            Suivant
                          </button>
                        </div>
                      )}
                    </div>

                    <aside className="2xl:sticky 2xl:top-8 2xl:self-start">
                      <div className="min-h-[220px] rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
                        {!selectedResult ? (
                          <div className="text-sm leading-6 text-forest-300">
                            Cliquez sur un résultat pour afficher le détail du document.
                          </div>
                        ) : detailLoading ? (
                          <div className="text-sm leading-6 text-forest-300">
                            Chargement du document complet...
                          </div>
                        ) : (
                          <div className="space-y-5 text-sm text-forest-200">
                            <div>
                              <p className="text-xs uppercase tracking-[0.2em] text-brand-300">
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
                                  className="inline-flex items-center gap-2 rounded-xl border border-white/10 px-3 py-2 text-sm text-forest-200 hover:bg-white/10"
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
                                <div><dt className="text-forest-400">ID</dt><dd>{detailView?.id ?? "—"}</dd></div>
                                <div><dt className="text-forest-400">Source</dt><dd>{detailView?.source}</dd></div>
                                <div><dt className="text-forest-400">Année</dt><dd>{detailView?.year}</dd></div>
                                <div><dt className="text-forest-400">External ID</dt><dd>{detailView?.externalId}</dd></div>
                                <div><dt className="text-forest-400">Projet</dt><dd>{detailView?.projectContext}</dd></div>
                                <div><dt className="text-forest-400">Type</dt><dd>{detailView?.sourceType}</dd></div>
                                <div><dt className="text-forest-400">Pathologie</dt><dd>{detailView?.disease}</dd></div>
                                <div><dt className="text-forest-400">Scénario</dt><dd>{detailView?.scenario}</dd></div>
                                <div><dt className="text-forest-400">Zone</dt><dd>{detailView?.geography}</dd></div>
                                <div><dt className="text-forest-400">Preuve</dt><dd>{detailView?.evidence}</dd></div>
                                <div><dt className="text-forest-400">Chunks</dt><dd>{detailView?.chunkCount}</dd></div>
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
