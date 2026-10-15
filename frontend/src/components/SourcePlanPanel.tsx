import { useI18n } from "../i18n/LanguageProvider";
import type { SourceQueryPlan } from "../lib/api";

/** What each database will receive of a query, said BEFORE anything is queried.
 *
 *  A reviewer used to learn this afterwards, from the PRISMA method notes, and only for
 *  the sources that were reduced or split: a database that refused the query outright
 *  simply appeared as failed. One panel for both places that need it, the search page
 *  (a query, no scenario yet) and a scenario's strategy tab, so the two never drift.
 *
 *  `reduced` is the only carriage that drops terms, and it is named as such. */
export function SourcePlanPanel({ plan, className }: {
  plan: SourceQueryPlan;
  className?: string;
}) {
  const { t } = useI18n();
  const T = (k: string) => t(`scenarioDetail.queries.sourcePlan${k}`);
  if (!plan.plan?.length) return null;
  const parts = Object.entries(plan.parts || {});
  return (
    <div className={`rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2 space-y-1 ${className ?? ""}`}
      data-testid="source-plan">
      <p className="text-[10px] font-semibold text-white/60 uppercase tracking-wider">{T("Title")}</p>
      <p className="text-[11px] text-white/50 leading-5">
        {T("Intro").replace("{chars}", String(plan.query_chars))}
      </p>
      {plan.whole.length > 0 && (
        <p className="text-[11px] text-emerald-300/80">
          {T("Whole").replace("{sources}", plan.whole.join(", "))}
        </p>
      )}
      {plan.body.length > 0 && (
        <p className="text-[11px] text-emerald-300/80">
          {T("Body").replace("{sources}", plan.body.join(", "))}
        </p>
      )}
      {parts.length > 0 && (
        <p className="text-[11px] text-sky-300/80">
          {T("Parts").replace("{sources}", parts.map(([s, n]) => `${s} (${n})`).join(", "))}
        </p>
      )}
      {plan.reduced.length > 0 && (
        <p className="text-[11px] text-gold-300/90">
          {T("Reduced").replace("{sources}", plan.reduced.join(", "))}
        </p>
      )}
      {plan.keywords.length > 0 && (
        <p className="text-[11px] text-white/40">
          {T("Keywords").replace("{sources}", plan.keywords.join(", "))}
        </p>
      )}
    </div>
  );
}
