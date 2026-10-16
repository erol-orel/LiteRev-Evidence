import React from "react";
import { AlertTriangle, Loader2, Upload } from "lucide-react";
import { useI18n } from "../i18n/LanguageProvider";
import {
  applyScenarioSubset, fetchGeography, fetchNutsStatus, hasApiKey, importNutsCsv,
  previewScenarioSubset,
} from "../lib/api";
import type { GeographyResponse, GeoUnresolvedReason, SubsetPlan } from "../lib/api";
import { share } from "../lib/extraction";

const TOP = 15;
const REASONS: GeoUnresolvedReason[] = ["no_place", "several_countries", "multi_country_area", "not_recognised"];
const NO_PLACE_KEY = "(no place stated)";

/** Where the studies took place: a country per bar, its regions beneath, and the places that could
 *  not be resolved. Counted over ALL the extracted relevant articles. The NUTS list in use is said,
 *  and an administrator can load Eurostat's official file to resolve the rest of Europe.
 *
 *  A country can also be TICKED. Reading the bars, the next question is always "and now give me
 *  only those", so the panel narrows the corpus to the countries chosen, through the same
 *  mechanism as the clusters: a preview first, then an exclusion carrying its reason into PRISMA,
 *  undoable from the Corpus tab. Everything downstream then speaks of that sub-corpus. */
export function GeographyPanel({ scenarioId, onNarrowed }: {
  scenarioId: string;
  /** Called once a narrowing is applied: the corpus just changed under every panel. */
  onNarrowed?: () => void;
}) {
  const { t } = useI18n();
  const T = (k: string) => t(`scenarioDetail.extraction.geo.${k}`);
  const [g, setG] = React.useState<GeographyResponse | null>(null);
  const [regions, setRegions] = React.useState<number | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [msg, setMsg] = React.useState<string | null>(null);
  const [showUnres, setShowUnres] = React.useState(false);
  const [showAll, setShowAll] = React.useState(false);   // every country, every region
  const fileRef = React.useRef<HTMLInputElement>(null);
  const keyed = hasApiKey();
  // The countries ticked, the preview they produce, and what the apply said.
  const [picked, setPicked] = React.useState<string[]>([]);
  const [plan, setPlan] = React.useState<SubsetPlan | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [narrowNote, setNarrowNote] = React.useState<string | null>(null);
  const toggle = (iso: string) =>
    setPicked((p) => (p.includes(iso) ? p.filter((x) => x !== iso) : [...p, iso]));

  const load = React.useCallback(
    () => Promise.all([fetchGeography(scenarioId), fetchNutsStatus()]),
    [scenarioId],
  );
  React.useEffect(() => {
    let alive = true;
    load()
      .then(([geo, st]) => { if (alive) { setG(geo); setRegions(st.n_regions); } })
      .catch((e: Error) => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [load]);

  // The preview follows the ticks, and never writes anything.
  React.useEffect(() => {
    if (!picked.length) { setPlan(null); return; }
    let alive = true;
    setNarrowNote(null);
    previewScenarioSubset(scenarioId, { countries: picked })
      .then((p) => { if (alive) setPlan(p); })
      .catch((e: Error) => { if (alive) { setPlan(null); setNarrowNote(e.message); } });
    return () => { alive = false; };
  }, [scenarioId, picked]);

  const applyNarrowing = async () => {
    if (!picked.length) return;
    setBusy(true); setNarrowNote(null);
    try {
      const out = await applyScenarioSubset(scenarioId, { countries: picked });
      setNarrowNote(T("narrowApplied").replace("{n}", String(out.applied ?? out.exclude ?? 0)));
      setPicked([]); setPlan(null);
      const [geo, st] = await load();
      setG(geo); setRegions(st.n_regions);
      onNarrowed?.();
    } catch (e) {
      setNarrowNote((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const upload = async (file: File | undefined) => {
    if (!file) return;
    setMsg(null);
    try {
      const out = await importNutsCsv(await file.text());
      setMsg(T("loaded").replace("{n}", String(out.loaded)).replace("{c}", String(out.countries)));
      const [geo, st] = await load();
      setG(geo); setRegions(st.n_regions);
    } catch (e) {
      setMsg((e as Error).message);
    }
  };

  if (error) return <p className="text-[11px] text-rose-300"><AlertTriangle size={11} className="inline mr-1" />{error}</p>;
  if (!g || g.n_papers === 0) return null;
  const top = showAll ? g.countries : g.countries.slice(0, TOP);
  const max = Math.max(...top.map((c) => c.n_papers), 1);

  return (
    <div className="space-y-3">
      <div>
        <h4 className="text-xs font-semibold text-white/80 uppercase tracking-wider">{T("title")}</h4>
        <p className="text-[10px] text-white/40 mt-0.5 max-w-[680px] leading-4">
          {T("subtitle").replace("{n}", String(g.n_resolved)).replace("{total}", String(g.n_papers))}
        </p>
        {/* La question qu'un relecteur pose en premier, et qu'un compte par pays ne sait pas
            rendre : « Europe » ou « Germany and the Netherlands » sont européens sans nommer
            un pays, donc ils ne tiennent dans aucune barre. Le compte européen le dit lui-même
            plutôt que de laisser additionner les barres. */}
        {g.n_in_europe != null && g.n_in_europe > 0 && (
          <p className="text-[10px] text-brand-300/80 mt-1 max-w-[680px] leading-4">
            {T("inEurope").replace("{n}", String(g.n_in_europe)).replace("{total}", String(g.n_papers))}
            {g.n_in_europe_without_country ? " " + T("inEuropeNoCountry")
              .replace("{n}", String(g.n_in_europe_without_country)) : ""}
            {/* The one selection nobody wants to assemble by hand, country by country. */}
            <button onClick={() => setPicked(["EUROPE"])}
              className="ml-2 underline underline-offset-2 text-brand-300/90 hover:text-brand-300">
              {T("narrowEurope")}
            </button>
          </p>
        )}
      </div>

      {top.length === 0 ? <p className="text-[11px] text-white/40">{T("none")}</p> : (
        <div className="space-y-2 rounded-2xl border border-white/5 bg-white/2 p-4">
          {top.map((c) => (
            <div key={c.iso2}>
              <div className="flex items-center gap-3 text-[11px]">
                {/* Ticking a country is the start of a narrowing, so the whole label is the
                    control: a 11px checkbox alone is a target nobody hits. */}
                <label className="flex w-36 shrink-0 items-center gap-1.5 cursor-pointer" title={c.name}>
                  <input type="checkbox" checked={picked.includes(c.iso2)}
                    onChange={() => toggle(c.iso2)}
                    aria-label={T("narrowPick").replace("{country}", c.name)}
                    className="h-3 w-3 shrink-0 accent-[rgb(var(--brand-500))]" />
                  <span className={`truncate ${picked.includes(c.iso2) ? "text-brand-300" : "text-white/80"}`}>{c.name}</span>
                </label>
                <div className="h-3 flex-1 overflow-hidden rounded-sm bg-white/5">
                  <div className="h-full rounded-sm bg-brand-500" style={{ width: `${share(c.n_papers, max)}%` }} />
                </div>
                <span className="w-28 shrink-0 text-right font-mono text-white/60">
                  {c.n_papers} <span className="text-white/35">{T("papers")}</span>
                </span>
                <span className="hidden w-14 shrink-0 text-right font-mono text-[10px] text-white/35 sm:block" title="NUTS 0">{c.nuts0 ?? "-"}</span>
              </div>
              {c.regions.length > 0 && (
                <div className="ml-[10.5rem] mt-1 flex flex-wrap gap-1">
                  {(showAll ? c.regions : c.regions.slice(0, 10)).map((r) => (
                    <span key={r.code} className="rounded bg-white/5 px-1.5 py-px text-[9px] text-white/55" title={r.code}>
                      {r.name} <span className="font-mono text-white/35">{r.code}</span> {r.n_papers}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
          {/* The narrowing bar: what the ticks would do, then the one button that does it.
              It only appears once something is ticked, so the panel stays a panel. */}
          {picked.length > 0 && (
            <div className="mt-1 rounded-xl border border-brand-500/25 bg-brand-500/5 px-3 py-2 space-y-1.5">
              <p className="text-[11px] text-brand-200">
                {T("narrowPicked").replace("{n}", String(picked.length))}
              </p>
              {plan && (
                <p className="text-[10px] text-white/55">
                  {T("narrowPlan")
                    .replace("{keep}", plan.keep.toLocaleString())
                    .replace("{relevant}", plan.relevant.toLocaleString())
                    .replace("{exclude}", plan.exclude.toLocaleString())}
                  {plan.undecided > 0 && " " + T("narrowUnjudged").replace("{n}", plan.undecided.toLocaleString())}
                </p>
              )}
              <div className="flex flex-wrap items-center gap-2">
                {keyed ? (
                  <button onClick={() => void applyNarrowing()} disabled={busy || !plan || plan.keep === 0}
                    className="flex items-center gap-1.5 rounded-lg border border-brand-500/40 bg-brand-500/15 px-2.5 py-1 text-[11px] font-medium text-brand-300 hover:bg-brand-500/25 transition disabled:opacity-50">
                    {busy && <Loader2 size={11} className="animate-spin" />}{T("narrowApply")}
                  </button>
                ) : <span className="text-[10px] text-white/40">{T("narrowNeedsKey")}</span>}
                <button onClick={() => { setPicked([]); setPlan(null); }}
                  className="text-[10px] text-white/45 underline hover:text-white">{T("narrowClear")}</button>
              </div>
              <p className="text-[10px] text-white/35 leading-4">{T("narrowHint")}</p>
            </div>
          )}
          {narrowNote && <p className="text-[10px] text-gold-300/80">{narrowNote}</p>}
          {(g.countries.length > TOP || g.countries.some((c) => c.regions.length > 10)) && (
            <p className="text-[10px] text-white/35">
              {!showAll && g.countries.length > TOP && <>{T("more").replace("{n}", String(g.countries.length - TOP))} </>}
              <button onClick={() => setShowAll((v) => !v)} className="text-brand-300/80 underline hover:text-brand-300">
                {showAll ? t("lists.showLess") : t("lists.showAll").replace("{n}", String(g.countries.length))}
              </button>
            </p>
          )}
        </div>
      )}

      {/* La liste NUTS est chargée UNE FOIS pour le serveur (table geo_nuts) : elle survit
          aux redémarrages et sert tous les scénarios. Le bouton restait pourtant affiché en
          grand, à côté du message « 1 932 régions chargées », comme s'il fallait recommencer.
          Une fois chargée, elle tient en une ligne, et le remplacement est un lien discret. */}
      {g.nuts_source === "loaded" ? (
        <div className="flex flex-wrap items-center gap-2 text-[10px] text-white/40">
          <span className="text-white/55">{T("sourceLoaded").replace("{n}", String(regions ?? 0))}</span>
          <span title={T("loadedHint")} className="cursor-help text-white/30">ⓘ</span>
          {keyed && (
            <>
              <button onClick={() => fileRef.current?.click()}
                className="text-white/35 underline underline-offset-2 hover:text-white/70">
                {T("replace")}
              </button>
              <input ref={fileRef} type="file" accept=".csv,text/csv" className="hidden"
                onChange={(e) => { void upload(e.target.files?.[0]); e.target.value = ""; }} />
            </>
          )}
          {msg && <span className="text-gold-300/80">{msg}</span>}
        </div>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-3 text-[10px] text-white/40">
            <span className="rounded-md border border-white/10 bg-white/5 px-2 py-1 text-white/60">{T("sourceBuiltin")}</span>
            {keyed ? (
              <>
                <button onClick={() => fileRef.current?.click()}
                  className="flex items-center gap-1.5 rounded-lg border border-brand-500/30 bg-brand-500/10 px-2.5 py-1 text-brand-300 hover:bg-brand-500/20 transition">
                  <Upload size={10} />{T("upload")}
                </button>
                <input ref={fileRef} type="file" accept=".csv,text/csv" className="hidden"
                  onChange={(e) => { void upload(e.target.files?.[0]); e.target.value = ""; }} />
              </>
            ) : <span>{T("needsKey")}</span>}
            {msg && <span className="text-gold-300/80">{msg}</span>}
          </div>
          <p className="text-[10px] text-white/35 max-w-[680px] leading-4">{T("builtinHint")}</p>
        </>
      )}

      {g.n_unresolved > 0 && (
        <div>
          <button onClick={() => setShowUnres((v) => !v)} className="text-[10px] text-white/45 underline hover:text-white">
            {T("unresolved").replace("{n}", String(g.n_unresolved))}
          </button>
          {showUnres && (
            <>
              <p className="mt-2 text-[10px] text-white/40">
                {REASONS.filter((r) => g.unresolved_by_reason?.[r]).map((r) => `${T(`reason.${r}`)} ${g.unresolved_by_reason[r]}`).join(" · ")}
              </p>
              <ul className="mt-2 space-y-0.5 text-[10px] text-white/50">
                {g.unresolved.map((u) => {
                  // The API's "(no place stated)" stands for the empty ones: say it in the interface's language.
                  const empty = u.reason === "no_place" && u.location === NO_PLACE_KEY;
                  return (
                    <li key={u.location}>
                      <span className="text-white/70">{empty ? T("reason.no_place") : u.location}</span>{" "}
                      <span className="text-white/35">{u.n}{u.reason && !empty ? ` · ${T(`reason.${u.reason}`)}` : ""}</span>
                    </li>
                  );
                })}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}
