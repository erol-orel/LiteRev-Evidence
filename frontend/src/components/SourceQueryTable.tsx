import { useState } from "react";
import { Copy, Loader2, RefreshCw, ChevronDown, ChevronUp } from "lucide-react";
import { useI18n } from "../i18n/LanguageProvider";
import type { SourcePlanRow, SourceQueryPlan } from "../lib/api";

/** The query each database actually receives, one row per database, in its own syntax.
 *
 *  `SourcePlanPanel` above says WHICH sources are adapted; this says WHAT each one gets.
 *  PRISMA-S asks for the strategy as submitted to each database, and until now LiteRev
 *  could only report that something had been adapted: "openalex: 4 parts", "doaj: reduced".
 *  On the COVID exposure strategy of production, 4,519 characters, OpenAIRE receives three
 *  terms, because its API refuses more than four logical operators. That is the kind of
 *  thing a reviewer has to be able to read, copy, and check against the database's own
 *  site, and it is why every row carries its exact string here.
 *
 *  `onRerun` adds the other half: a reviewer who finds one database short asks for that
 *  one alone, instead of a whole federation of thirteen. */
export function SourceQueryTable({ plan, onRerun, running, className }: {
  plan: SourceQueryPlan;
  /** Absent on the search page, where no scenario exists to re-run yet. */
  onRerun?: (source: string) => void;
  /** The source a re-run was just asked for, or null. */
  running?: string | null;
  className?: string;
}) {
  const { t } = useI18n();
  const T = (k: string) => t(`scenarioDetail.queries.sourceQuery${k}`);
  const [open, setOpen] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  if (!plan.plan?.length) return null;

  function copy(key: string, text: string) {
    try {
      void navigator.clipboard?.writeText(text);
      setCopied(key);
      window.setTimeout(() => setCopied(c => (c === key ? null : c)), 1600);
    } catch { /* a clipboard the browser refuses is not an error worth a banner */ }
  }

  /** The carriage in one phrase, and the hue that says whether anything was lost. Only
   *  `reduced` and `keywords` drop terms; `parts` and `body` keep every one. */
  function carriage(row: SourcePlanRow): { label: string; tone: string } {
    if (row.how === "parts") return { label: T("Parts").replace("{n}", String(row.parts ?? row.sent.length)), tone: "text-sky-300" };
    if (row.how === "reduced") return { label: T("Reduced"), tone: "text-gold-300" };
    if (row.how === "keywords") return { label: T("Keywords"), tone: "text-rose-300" };
    if (row.how === "body") return { label: T("Body"), tone: "text-emerald-300" };
    return { label: T("Whole"), tone: "text-emerald-300" };
  }

  return (
    <div className={`rounded-xl border border-white/10 bg-white/[0.03] ${className ?? ""}`}
      data-testid="source-query-table">
      <div className="px-3 py-2 border-b border-white/5">
        <p className="text-[10px] font-semibold text-white/60 uppercase tracking-wider">{T("Title")}</p>
        <p className="text-[11px] text-white/45 leading-5">{T("Intro")}</p>
        {plan.planned_from === "query" && (
          <p className="mt-1 text-[11px] text-gold-300/90 leading-5">{T("FromQuery")}</p>
        )}
      </div>
      <ul className="divide-y divide-white/5">
        {plan.plan.map(row => {
          const { label, tone } = carriage(row);
          const isOpen = open === row.source;
          const first = row.sent[0] ?? "";
          return (
            <li key={row.source} className="px-3 py-2">
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => setOpen(o => (o === row.source ? null : row.source))}
                  className="flex items-center gap-1.5 text-[11px] font-semibold text-white/80 hover:text-white transition"
                  aria-expanded={isOpen}
                >
                  {isOpen ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
                  {row.source}
                </button>
                <span className={`text-[10px] ${tone}`}>{label}</span>
                {typeof row.encoded === "number" && typeof row.limit_value === "number" && (
                  <span className="text-[10px] text-white/30">
                    {T("Measured").replace("{encoded}", row.encoded.toLocaleString())
                      .replace("{limit}", row.limit_value.toLocaleString())}
                  </span>
                )}
                {onRerun && (
                  <button
                    type="button"
                    onClick={() => onRerun(row.source)}
                    disabled={!!running}
                    title={T("RerunHint")}
                    className="ml-auto flex items-center gap-1 rounded-lg border border-white/15 bg-white/5
                               px-2 py-0.5 text-[10px] text-white/70 hover:bg-white/10 transition
                               disabled:opacity-40"
                  >
                    {running === row.source
                      ? <Loader2 size={9} className="animate-spin" />
                      : <RefreshCw size={9} />}
                    {T("Rerun")}
                  </button>
                )}
              </div>
              {row.note && (
                <p className="mt-1 text-[10px] text-white/40 leading-4">{row.note}</p>
              )}
              {!isOpen && first && (
                <code className="mt-1 block truncate font-mono text-[10px] text-white/40">{first}</code>
              )}
              {isOpen && (
                <div className="mt-2 space-y-2">
                  {row.sent.map((q, i) => (
                    <div key={i} className="rounded-lg border border-white/10 bg-white/[0.02] p-2">
                      <div className="mb-1 flex items-center gap-2">
                        <span className="text-[9px] uppercase tracking-wider text-white/35">
                          {row.sent.length > 1
                            ? T("Part").replace("{i}", String(i + 1)).replace("{n}", String(row.sent.length))
                            : T("Sent")}
                        </span>
                        <button
                          type="button"
                          onClick={() => copy(`${row.source}-${i}`, q)}
                          className="ml-auto flex items-center gap-1 rounded border border-white/15 bg-white/5
                                     px-1.5 py-0.5 text-[9px] text-white/60 hover:bg-white/10 transition"
                        >
                          <Copy size={8} />
                          {copied === `${row.source}-${i}` ? T("Copied") : T("Copy")}
                        </button>
                      </div>
                      <code className="block max-h-40 overflow-auto whitespace-pre-wrap break-words
                                       font-mono text-[10px] leading-5 text-white/70">{q}</code>
                    </div>
                  ))}
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
