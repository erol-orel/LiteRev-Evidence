import React from "react";
import { AlertTriangle, Upload } from "lucide-react";
import { useI18n } from "../i18n/LanguageProvider";
import { fetchGeography, fetchNutsStatus, hasApiKey, importNutsCsv } from "../lib/api";
import type { GeographyResponse } from "../lib/api";
import { share } from "../lib/extraction";

const TOP = 15;

/** Where the studies took place: a country per bar, its regions beneath, and the places that could
 *  not be resolved. Counted over ALL the extracted relevant articles. The NUTS list in use is said,
 *  and an administrator can load Eurostat's official file to resolve the rest of Europe. */
export function GeographyPanel({ scenarioId }: { scenarioId: string }) {
  const { t } = useI18n();
  const T = (k: string) => t(`scenarioDetail.extraction.geo.${k}`);
  const [g, setG] = React.useState<GeographyResponse | null>(null);
  const [regions, setRegions] = React.useState<number | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [msg, setMsg] = React.useState<string | null>(null);
  const [showUnres, setShowUnres] = React.useState(false);
  const fileRef = React.useRef<HTMLInputElement>(null);
  const keyed = hasApiKey();

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
  const top = g.countries.slice(0, TOP);
  const max = Math.max(...top.map((c) => c.n_papers), 1);

  return (
    <div className="space-y-3">
      <div>
        <h4 className="text-xs font-semibold text-white/80 uppercase tracking-wider">{T("title")}</h4>
        <p className="text-[10px] text-white/40 mt-0.5 max-w-[680px] leading-4">
          {T("subtitle").replace("{n}", String(g.n_resolved)).replace("{total}", String(g.n_papers))}
        </p>
      </div>

      {top.length === 0 ? <p className="text-[11px] text-white/40">{T("none")}</p> : (
        <div className="space-y-2 rounded-2xl border border-white/5 bg-white/2 p-4">
          {top.map((c) => (
            <div key={c.iso2}>
              <div className="flex items-center gap-3 text-[11px]">
                <span className="w-36 shrink-0 truncate text-white/80" title={c.name}>{c.name}</span>
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
                  {c.regions.slice(0, 10).map((r) => (
                    <span key={r.code} className="rounded bg-white/5 px-1.5 py-px text-[9px] text-white/55" title={r.code}>
                      {r.name} <span className="font-mono text-white/35">{r.code}</span> {r.n_papers}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
          {g.countries.length > TOP && <p className="text-[10px] text-white/35">{T("more").replace("{n}", String(g.countries.length - TOP))}</p>}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3 text-[10px] text-white/40">
        <span className="rounded-md border border-white/10 bg-white/5 px-2 py-1 text-white/60">
          {g.nuts_source === "loaded" ? T("sourceLoaded").replace("{n}", String(regions ?? 0)) : T("sourceBuiltin")}
        </span>
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
      {g.nuts_source === "builtin" && <p className="text-[10px] text-white/35 max-w-[680px] leading-4">{T("builtinHint")}</p>}

      {g.n_unresolved > 0 && (
        <div>
          <button onClick={() => setShowUnres((v) => !v)} className="text-[10px] text-white/45 underline hover:text-white">
            {T("unresolved").replace("{n}", String(g.n_unresolved))}
          </button>
          {showUnres && (
            <ul className="mt-2 space-y-0.5 text-[10px] text-white/50">
              {g.unresolved.map((u) => <li key={u.location}><span className="text-white/70">{u.location}</span> <span className="text-white/35">{u.n}</span></li>)}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
