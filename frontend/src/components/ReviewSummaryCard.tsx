import React from "react";
import { Download } from "lucide-react";
import { useI18n } from "../i18n/LanguageProvider";
import { extractionDatasetUrl, fetchReviewSummary } from "../lib/api";
import type { ReviewStatus, ReviewSummary } from "../lib/api";
import { kappaBand, share } from "../lib/extraction";

const BAR: Record<ReviewStatus, string> = {
  accepted: "bg-brand-500", edited: "bg-sky-400", rejected: "bg-rose-400/70", conflict: "bg-gold-400", unreviewed: "bg-white/10",
};
const ORDER: ReviewStatus[] = ["accepted", "edited", "rejected", "conflict", "unreviewed"];
/** Below this many rows in common, a kappa is too noisy to lean on. */
const MIN_COMMON = 20;

/** Who has reviewed what, how far the review has got, and how well two reviewers agree.
 *  Counted over ALL the extracted rows of the relevant articles. */
export function ReviewSummaryCard({ scenarioId, reviewer, onReviewer, tick }: {
  scenarioId: string; reviewer: string; onReviewer: (v: string) => void; tick: number;
}) {
  const { t } = useI18n();
  const T = (k: string) => t(`scenarioDetail.extraction.review.${k}`);
  const [s, setS] = React.useState<ReviewSummary | null>(null);

  React.useEffect(() => {
    let alive = true;
    fetchReviewSummary(scenarioId).then((r) => { if (alive) setS(r); }).catch(() => undefined);
    return () => { alive = false; };
  }, [scenarioId, tick]);

  return (
    <div className="rounded-2xl border border-white/5 bg-white/2 p-4 space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h4 className="text-xs font-semibold text-white/80 uppercase tracking-wider">{T("title")}</h4>
          <p className="text-[10px] text-white/40 mt-0.5 max-w-[640px] leading-4">{T("subtitle")}</p>
        </div>
        <label className="flex items-center gap-2 text-[10px] text-white/50">
          {T("yourName")}
          <input value={reviewer} onChange={(e) => onReviewer(e.target.value)} maxLength={40} placeholder={T("namePlaceholder")}
            className="w-36 rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-brand-500/50" />
        </label>
      </div>

      {s && s.n_observations > 0 && (
        <>
          <div>
            <div className="flex justify-between text-[10px] text-white/40 mb-1.5">
              <span>{T("progress").replace("{n}", String(s.n_reviewed)).replace("{total}", String(s.n_observations))}</span>
              <span>{Math.round(s.share_reviewed * 100)}%</span>
            </div>
            <div className="flex h-2 overflow-hidden rounded-full bg-white/5">
              {ORDER.filter((k) => k !== "unreviewed").map((k) => (
                <div key={k} className={`h-full ${BAR[k]}`} style={{ width: `${share(s.counts[k], s.n_observations)}%` }} title={`${T(`status.${k}`)}: ${s.counts[k]}`} />
              ))}
            </div>
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-white/55">
              {ORDER.map((k) => (
                <span key={k} className="flex items-center gap-1.5">
                  <span className={`h-2 w-2 rounded-sm ${BAR[k]}`} />{T(`status.${k}`)} <b className="font-mono text-white/80">{s.counts[k]}</b>
                </span>
              ))}
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <h5 className="text-[10px] font-semibold text-white/60 uppercase tracking-wider mb-1">{T("reviewersTitle")}</h5>
              {s.reviewers.length === 0 ? <p className="text-[10px] text-white/35">-</p> : (
                <ul className="space-y-0.5 text-[11px] text-white/65">
                  {s.reviewers.map((r) => <li key={r.reviewer}>{r.reviewer} <span className="text-white/35">{T("decisions").replace("{n}", String(r.n_decisions))}</span></li>)}
                </ul>
              )}
            </div>
            <div>
              <h5 className="text-[10px] font-semibold text-white/60 uppercase tracking-wider mb-1">{T("agreementTitle")}</h5>
              {s.agreement.length === 0 ? <p className="text-[10px] text-white/35">{T("noAgreement")}</p> : (
                <ul className="space-y-1 text-[11px] text-white/65">
                  {s.agreement.map((a) => (
                    <li key={a.reviewers.join("|")}>
                      {T("agreementRow").replace("{a}", a.reviewers[0]).replace("{b}", a.reviewers[1])
                        .replace("{n}", String(a.n_common)).replace("{pct}", String(Math.round((a.observed ?? 0) * 100)))
                        .replace("{k}", a.kappa == null ? "-" : a.kappa.toFixed(2))}
                      {" "}<span className="text-white/40">({T(`kappaBand.${kappaBand(a.kappa)}`)})</span>
                      {a.n_common < MIN_COMMON && <div className="text-[10px] text-gold-400/80">{T("fewRows")}</div>}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
          {s.n_stale_decisions > 0 && <p className="text-[10px] text-gold-400/80">{T("stale").replace("{n}", String(s.n_stale_decisions))}</p>}
        </>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <a href={extractionDatasetUrl(scenarioId)} download
          className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-2.5 py-1 text-[10px] text-white/70 hover:bg-white/10 transition">
          <Download size={10} />{T("dataset")}
        </a>
        <span className="text-[10px] text-white/35">{T("datasetHint")}</span>
      </div>
    </div>
  );
}
