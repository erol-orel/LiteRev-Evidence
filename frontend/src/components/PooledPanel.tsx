import React from "react";
import { AlertTriangle, ChevronDown, ChevronUp, Download, Info } from "lucide-react";
import { useI18n } from "../i18n/LanguageProvider";
import { fetchPooled } from "../lib/api";
import type { PooledComparison, PooledGroup, PooledResponse, PooledStudy } from "../lib/api";
import { bandTone, csvOfPooled, fmtOr, forestScale, intervalText, orDomain, pct, proportionDomain } from "../lib/pooling";

const W = 360;            // width of the plotting area
const ROW = 20;

function studyLabel(s: PooledStudy): string {
  return `${s.first_author || "?"} ${s.year ?? ""}`.trim();
}

/** A forest plot: one row per study (square sized by weight, line = its 95% interval), then the
 *  pooled value (diamond) with the prediction interval below it. `log` for odds ratios. */
function Forest({ studies, pooled, domain, log, nullLine, T }: {
  studies: { label: string; est: number; lo: number; hi: number; w: number; n: string }[];
  pooled: { est: number; lo: number; hi: number; pLo: number | null; pHi: number | null };
  domain: [number, number]; log: boolean; nullLine: number | null; T: (k: string) => string;
}) {
  const x = forestScale(domain[0], domain[1], W, log);
  const rows = studies.length;
  const H = (rows + 3) * ROW + 24;
  const fmt = (v: number) => (log ? fmtOr(v) : pct(v));
  const ticks = log ? [0.25, 0.5, 1, 2, 4].filter((t) => t > domain[0] && t < domain[1]) : [0, 0.25, 0.5, 0.75, 1].filter((t) => t <= domain[1] + 1e-9);
  const wmax = Math.max(...studies.map((s) => s.w), 1);
  return (
    <svg viewBox={`0 0 ${W + 250} ${H}`} className="w-full max-w-[640px]" role="img" aria-label={T("forest")}>
      <g transform="translate(130,6)">
        {ticks.map((t) => (
          <g key={t}>
            {t !== nullLine && <line x1={x(t)} x2={x(t)} y1={0} y2={(rows + 2) * ROW} stroke="currentColor" className="text-white/10" />}
            <text x={x(t)} y={(rows + 2) * ROW + 12} textAnchor="middle" className="fill-white/35" fontSize="8">{fmt(t)}</text>
          </g>
        ))}
        {nullLine != null && <line x1={x(nullLine)} x2={x(nullLine)} y1={0} y2={(rows + 2) * ROW} stroke="currentColor" className="text-white/40" strokeDasharray="3 3" />}
        {studies.map((s, i) => {
          const y = i * ROW + ROW / 2;
          const side = 3 + (s.w / wmax) * 4;
          return (
            <g key={i}>
              <text x={-6} y={y + 3} textAnchor="end" className="fill-white/60" fontSize="9">{s.label}</text>
              <line x1={x(s.lo)} x2={x(s.hi)} y1={y} y2={y} stroke="currentColor" className="text-white/45" />
              <rect x={x(s.est) - side} y={y - side} width={side * 2} height={side * 2} className="fill-brand-400" />
              <text x={W + 8} y={y + 3} className="fill-white/50" fontSize="8.5">{fmt(s.est)} ({fmt(s.lo)} to {fmt(s.hi)})</text>
              <text x={W + 140} y={y + 3} className="fill-white/35" fontSize="8.5">{s.n}</text>
            </g>
          );
        })}
        {(() => {
          const y = rows * ROW + ROW / 2 + 4;
          return (
            <g>
              <text x={-6} y={y + 3} textAnchor="end" className="fill-white font-semibold" fontSize="9">{T("pooledLabel")}</text>
              <polygon points={`${x(pooled.lo)},${y} ${x(pooled.est)},${y - 6} ${x(pooled.hi)},${y} ${x(pooled.est)},${y + 6}`} className="fill-gold-400" />
              <text x={W + 8} y={y + 3} className="fill-white" fontSize="8.5" fontWeight="600">{fmt(pooled.est)} ({fmt(pooled.lo)} to {fmt(pooled.hi)})</text>
              {pooled.pLo != null && pooled.pHi != null && (
                <g>
                  <text x={-6} y={y + ROW + 3} textAnchor="end" className="fill-white/50" fontSize="9">{T("predictionLabel")}</text>
                  <line x1={x(pooled.pLo)} x2={x(pooled.pHi)} y1={y + ROW} y2={y + ROW} stroke="currentColor" className="text-gold-400/70" strokeWidth="2" strokeLinecap="round" />
                  <text x={W + 8} y={y + ROW + 3} className="fill-white/50" fontSize="8.5">{fmt(pooled.pLo)} to {fmt(pooled.pHi)}</text>
                </g>
              )}
            </g>
          );
        })()}
      </g>
    </svg>
  );
}

function HetChip({ g, T }: { g: { I2: number; band: string; tau2: number }; T: (k: string) => string }) {
  return (
    <span className={`font-mono ${bandTone(g.band)}`} title={`tau2 ${g.tau2}`}>
      I2 {Math.round(g.I2)}% <span className="font-sans text-[9px]">({T(`band.${g.band}`)})</span>
    </span>
  );
}

/** What the studies say together, with the working shown. Counted over ALL the relevant rows. */
export function PooledPanel({ scenarioId, tick }: { scenarioId: string; tick: number }) {
  const { t } = useI18n();
  const T = (k: string) => t(`scenarioDetail.extraction.pooled.${k}`);
  const [opts, setOpts] = React.useState({ reviewedOnly: false, verifiedOnly: true, splitDisease: true });
  const [data, setData] = React.useState<PooledResponse | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [open, setOpen] = React.useState<string | null>(null);
  const [showSmall, setShowSmall] = React.useState(false);

  React.useEffect(() => {
    let alive = true;
    fetchPooled(scenarioId, opts)
      .then((d) => { if (alive) { setData(d); setError(null); } })
      .catch((e: Error) => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [scenarioId, opts, tick]);

  const download = () => {
    if (!data) return;
    const blob = new Blob([csvOfPooled(data.pooled)], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `pooled_${scenarioId}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const big = (data?.pooled ?? []).filter((g) => g.pooled);
  const small = (data?.pooled ?? []).filter((g) => !g.pooled);
  const name = (g: { label: string; label_path?: string | null }) => (g.label_path ?? g.label).replace(/_/g, " ");
  const excluded = data ? Object.entries(data.excluded).filter(([, n]) => n > 0) : [];

  const toggles: { k: keyof typeof opts; label: string }[] = [
    { k: "verifiedOnly", label: T("optVerified") }, { k: "reviewedOnly", label: T("optReviewed") }, { k: "splitDisease", label: T("optDisease") },
  ];

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h4 className="text-xs font-semibold text-white/80 uppercase tracking-wider">{T("title")}</h4>
          <p className="text-[10px] text-white/40 mt-0.5 max-w-[680px] leading-4">{T("subtitle")}</p>
        </div>
        {big.length > 0 && (
          <button onClick={download} className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-2.5 py-1 text-[10px] text-white/70 hover:bg-white/10 transition">
            <Download size={10} />{T("download")}
          </button>
        )}
      </div>

      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {toggles.map((o) => (
          <label key={o.k} className="flex items-center gap-1.5 text-[10px] text-white/55 cursor-pointer">
            <input type="checkbox" checked={opts[o.k]} onChange={(e) => setOpts((p) => ({ ...p, [o.k]: e.target.checked }))} className="accent-brand-500" />
            {o.label}
          </label>
        ))}
      </div>

      <div className="flex items-start gap-2 rounded-xl border border-gold-400/20 bg-gold-400/5 px-3 py-2 text-[11px] text-gold-400/90">
        <Info size={13} className="mt-0.5 shrink-0" /><span>{T("caution")}</span>
      </div>

      {error && <p className="text-[11px] text-rose-300"><AlertTriangle size={11} className="inline mr-1" />{error}</p>}

      {data && big.length === 0 && <p className="text-[11px] text-white/40">{T("none").replace("{n}", String(data.filters.min_studies))}</p>}

      {big.length > 0 && (
        <div className="overflow-x-auto rounded-2xl border border-white/5">
          <table className="w-full text-[10px] border-collapse">
            <thead>
              <tr className="border-b border-white/5 bg-white/3">
                {["colLabel", "colDisease", "colStudies", "colPeople", "colPooled", "colPrediction", "colHet"].map((h) => (
                  <th key={h} className="text-left px-3 py-2 text-white/40 font-semibold uppercase tracking-wider whitespace-nowrap">{T(h)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {big.map((g: PooledGroup) => {
                const id = `${g.sheet}|${g.group}|${g.label}|${g.disease}`;
                const isOpen = open === id;
                return (
                  <React.Fragment key={id}>
                    <tr onClick={() => setOpen(isOpen ? null : id)} className="border-b border-white/5 cursor-pointer hover:bg-white/4">
                      <td className="px-3 py-2 text-white/80">
                        <span className="inline-flex items-center gap-1.5">
                          {isOpen ? <ChevronUp size={11} className="text-brand-300" /> : <ChevronDown size={11} className="text-white/40" />}
                          {name(g)}{!g.mapped && <span className="text-gold-400/70"> ({t("scenarioDetail.extraction.unmapped")})</span>}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-white/50">{g.disease ?? T("allDiseases")}</td>
                      <td className="px-3 py-2 font-mono text-white/65">{g.k}</td>
                      <td className="px-3 py-2 font-mono text-white/55">{g.events_total}/{g.n_total}</td>
                      <td className="px-3 py-2 font-mono text-white whitespace-nowrap">{pct(g.pooled!.p)} <span className="text-white/40">({intervalText(g.pooled!.ci_low, g.pooled!.ci_high, pct)})</span></td>
                      <td className="px-3 py-2 font-mono text-white/45 whitespace-nowrap">{intervalText(g.pooled!.pi_low, g.pooled!.pi_high, pct)}</td>
                      <td className="px-3 py-2 whitespace-nowrap">{g.heterogeneity && <HetChip g={g.heterogeneity} T={T} />}</td>
                    </tr>
                    {isOpen && (
                      <tr className="border-b border-white/5 bg-white/2">
                        <td colSpan={7} className="px-4 py-3">
                          <Forest T={T} log={false} nullLine={null} domain={proportionDomain(g)}
                            studies={g.studies.map((s) => ({ label: studyLabel(s), est: s.p ?? 0, lo: s.ci_low ?? 0, hi: s.ci_high ?? 0, w: s.weight_pct ?? 1, n: `${s.x}/${s.n}` }))}
                            pooled={{ est: g.pooled!.p, lo: g.pooled!.ci_low, hi: g.pooled!.ci_high, pLo: g.pooled!.pi_low, pHi: g.pooled!.pi_high }} />
                          {g.heterogeneity && (
                            <p className="mt-1 text-[10px] text-white/40">
                              {T("hetDetail").replace("{q}", String(g.heterogeneity.Q)).replace("{df}", String(g.heterogeneity.df))
                                .replace("{p}", String(g.heterogeneity.p)).replace("{tau}", String(g.heterogeneity.tau2))}
                            </p>
                          )}
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {(data?.comparisons.length ?? 0) > 0 && (
        <div className="space-y-2">
          <h5 className="text-[10px] font-semibold text-white/60 uppercase tracking-wider">{T("comparisonsTitle")}</h5>
          <p className="text-[10px] text-white/35">{T("comparisonsHint")}</p>
          <div className="space-y-2">
            {data!.comparisons.map((c: PooledComparison) => {
              const id = `cmp|${c.sheet}|${c.group}|${c.a}|${c.b}|${c.disease}`;
              const isOpen = open === id;
              return (
                <div key={id} className="rounded-xl border border-white/5">
                  <button onClick={() => setOpen(isOpen ? null : id)} className="flex w-full flex-wrap items-center justify-between gap-2 px-3 py-2 text-left text-[11px]">
                    <span className="text-white/80">{c.a.replace(/_/g, " ")} <span className="text-white/35">{T("versus")}</span> {c.b.replace(/_/g, " ")}
                      <span className="ml-2 text-white/40">{c.disease ?? T("allDiseases")}, {T("papers").replace("{n}", String(c.k))}</span></span>
                    <span className="font-mono text-white">OR {fmtOr(c.pooled.or)} <span className="text-white/40">({intervalText(c.pooled.ci_low, c.pooled.ci_high, fmtOr)})</span>
                      {" "}<HetChip g={c.heterogeneity} T={T} /></span>
                  </button>
                  {isOpen && (
                    <div className="border-t border-white/5 px-3 py-3">
                      <Forest T={T} log nullLine={1} domain={orDomain(c)}
                        studies={c.studies.map((s) => ({ label: studyLabel(s), est: s.or ?? 1, lo: s.ci_low ?? 1, hi: s.ci_high ?? 1, w: s.weight_pct ?? 1, n: `${s.x1}/${s.n1} vs ${s.x2}/${s.n2}` }))}
                        pooled={{ est: c.pooled.or, lo: c.pooled.ci_low, hi: c.pooled.ci_high, pLo: c.pooled.pi_low, pHi: c.pooled.pi_high }} />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {small.length > 0 && (
        <div>
          <button onClick={() => setShowSmall((v) => !v)} className="text-[10px] text-white/45 underline hover:text-white">
            {T("tooFew").replace("{n}", String(small.length)).replace("{min}", String(data?.filters.min_studies ?? 3))}
          </button>
          {showSmall && (
            <ul className="mt-2 space-y-1 text-[10px] text-white/50">
              {small.slice(0, 40).map((g) => (
                <li key={`${g.sheet}|${g.group}|${g.label}|${g.disease}`}>
                  <span className="text-white/70">{name(g)}</span> ({g.disease ?? T("allDiseases")}):{" "}
                  {g.studies.map((s) => `${studyLabel(s)} ${s.x}/${s.n} (${pct(s.p)})`).join("; ")}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {data && (
        <p className="text-[10px] text-white/35">
          {T("rowsUsed").replace("{n}", String(data.n_rows_used))}
          {data.n_duplicate_rows_dropped > 0 && ` ${T("duplicates").replace("{n}", String(data.n_duplicate_rows_dropped))}`}
          {excluded.length > 0 && ` ${T("leftOut")} ${excluded.map(([k, n]) => `${T(`excl.${k}`)} ${n}`).join(", ")}.`}
        </p>
      )}
    </div>
  );
}
