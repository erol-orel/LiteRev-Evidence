import { AlertTriangle } from "lucide-react";
import { useI18n } from "../i18n/LanguageProvider";
import { changedSpan } from "../lib/queryProposal";

/** A correction PROPOSED for the query the user entered: the query as entered, the proposed
 *  one with the change highlighted in each, why, and two buttons. Nothing changes unless
 *  the user accepts: the search waits for the choice, and "keep my query" runs it exactly as
 *  it was typed. */
export function QueryProposal({ entered, proposed, reason, onAccept, onKeep }: {
  entered: string;
  proposed: string;
  reason?: string | null;
  onAccept: () => void;
  onKeep: () => void;
}) {
  const { t } = useI18n();
  const T = (k: string) => t(`search.proposal.${k}`);
  const span = changedSpan(entered, proposed);
  const box = "mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-white/10 bg-forest-950/60 p-2 font-mono text-[11px] leading-relaxed";
  return (
    <section role="region" aria-label={T("title")}
      className="mt-3 rounded-2xl border border-amber-400/30 bg-amber-500/10 p-3 text-xs text-amber-100">
      <p className="flex items-center gap-1.5 font-semibold text-amber-200">
        <AlertTriangle size={13} className="shrink-0" />
        {T("title")}
      </p>
      {reason && <p className="mt-1 leading-snug" data-testid="proposal-reason">{reason}</p>}
      <div className="mt-2">
        <span className="text-forest-300">{T("entered")}</span>
        <div className={`${box} text-white/70`} data-testid="proposal-entered">
          {span.prefix}
          {span.removed && <mark className="rounded bg-rose-400/30 px-0.5 text-rose-100">{span.removed}</mark>}
          {span.suffix}
        </div>
      </div>
      <div className="mt-2">
        <span className="text-forest-300">{T("proposed")}</span>
        <div className={`${box} text-white/85`} data-testid="proposal-proposed">
          {span.prefix}
          {span.added && <mark className="rounded bg-emerald-400/30 px-0.5 text-emerald-100">{span.added}</mark>}
          {span.suffix}
        </div>
      </div>
      <p className="mt-2 text-[11px] text-amber-100/70">{T("nothingChanges")}</p>
      <div className="mt-2 flex flex-wrap gap-2">
        <button type="button" onClick={onAccept}
          className="rounded-lg border border-brand-400/40 bg-brand-500/20 px-3 py-1.5 text-[11px] font-semibold text-brand-200 transition hover:bg-brand-500/30">
          {T("accept")}
        </button>
        <button type="button" onClick={onKeep}
          className="rounded-lg border border-white/15 bg-white/5 px-3 py-1.5 text-[11px] font-semibold text-white/80 transition hover:bg-white/10">
          {T("keep")}
        </button>
      </div>
    </section>
  );
}
