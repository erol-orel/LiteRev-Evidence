import React from "react";
import { AlertTriangle, Check, CheckCheck, CheckCircle2, Pencil, RotateCcw, Trash2, X } from "lucide-react";
import { useI18n } from "../i18n/LanguageProvider";
import { hasApiKey, reviewBulk, reviewObservation } from "../lib/api";
import type { ArticleExtraction, ExtractionObservation, ReviewStatus } from "../lib/api";
import { editIsValid, editPayload } from "../lib/extraction";

export type Detail = { loading: boolean; error?: string; data?: ArticleExtraction };

const ROW_STYLE: Record<ReviewStatus, string> = {
  unreviewed: "",
  accepted: "border-l-2 border-l-brand-500/60",
  edited: "border-l-2 border-l-sky-400/60",
  rejected: "border-l-2 border-l-rose-400/50 opacity-55",
  conflict: "border-l-2 border-l-gold-400/70 bg-gold-400/5",
};

/** The observations of one article, with the review controls. A reviewer accepts, edits or
 *  rejects each row; the model's own values are never overwritten (a correction is shown as
 *  the corrected value, with the original kept behind it). */
export function ObservationPanel({ scenarioId, articleId, detail, reviewer, onUpdated }: {
  scenarioId: string; articleId: number; detail: Detail | undefined; reviewer: string;
  onUpdated: (updated: ExtractionObservation | null) => void;
}) {
  const { t } = useI18n();
  const T = (k: string) => t(`scenarioDetail.extraction.${k}`);
  const [editing, setEditing] = React.useState<string | null>(null);
  const [form, setForm] = React.useState({ value: "", n_cases: "", pop_risk: "", covariate: "" });
  const [busy, setBusy] = React.useState(false);
  const [msg, setMsg] = React.useState<string | null>(null);
  const keyed = hasApiKey();
  const named = reviewer.trim().length >= 2;
  const can = keyed && named;

  if (!detail || detail.loading) {
    return <div className="flex items-center gap-2 text-white/50 text-[11px]"><RotateCcw size={12} className="animate-spin" />{T("detailLoading")}</div>;
  }
  if (detail.error || !detail.data?.extraction) {
    return <p className="text-[11px] text-rose-300">{detail.error ?? T("detailError")}</p>;
  }
  const ex = detail.data.extraction;
  const obs = ex.observations ?? [];
  const nVerified = obs.filter((o) => o.quote_verified && (o.reviews ?? []).every((r) => r.reviewer !== reviewer.trim())).length;

  const act = async (o: ExtractionObservation, status: "accepted" | "rejected" | "clear" | "edited", edits?: Record<string, unknown>) => {
    if (!o.obs_key) return;
    setBusy(true); setMsg(null);
    try {
      const out = await reviewObservation(scenarioId, articleId, { obs_key: o.obs_key, reviewer: reviewer.trim(), status, edits });
      onUpdated(out.observation);
      setEditing(null);
    } catch (e) {
      setMsg((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const acceptVerified = async () => {
    setBusy(true); setMsg(null);
    try {
      await reviewBulk(scenarioId, articleId, reviewer.trim(), "accepted");
      onUpdated(null);                                              // the list and the row states are re-read
    } catch (e) {
      setMsg((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const startEdit = (o: ExtractionObservation) => {
    const e = o.effective ?? o;
    setForm({ value: e.value == null ? "" : String(e.value), n_cases: e.n_cases == null ? "" : String(e.n_cases),
              pop_risk: e.pop_risk == null ? "" : String(e.pop_risk), covariate: e.covariate ?? "" });
    setEditing(o.obs_key ?? null);
    setMsg(null);
  };

  const input = "w-full min-w-[56px] rounded-md border border-white/15 bg-white/5 px-1.5 py-1 text-[10px] text-white focus:outline-none focus:border-brand-500/60";
  const hint = !keyed ? T("review.needsKey") : !named ? T("review.needsName") : "";

  return (
    <div className="space-y-3">
      {ex.ref?.description && <p className="text-[11px] text-white/60 leading-4">{ex.ref.description}</p>}
      {ex.model && (
        <p className="text-[10px] text-white/35">
          {T("provenance").replace("{model}", ex.model).replace("{prompt}", ex.prompt_sha ?? "?")
            .replace("{date}", (ex.extracted_at ?? "").slice(0, 10) || "?")}
        </p>
      )}
      {obs.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <button disabled={!can || busy || nVerified === 0} onClick={() => void acceptVerified()} title={hint}
            className="flex items-center gap-1.5 rounded-lg border border-brand-500/30 bg-brand-500/10 px-2.5 py-1 text-[10px] text-brand-300 hover:bg-brand-500/20 transition disabled:opacity-40">
            <CheckCheck size={11} />{T("review.acceptVerified").replace("{n}", String(nVerified))}
          </button>
          {hint && <span className="text-[10px] text-white/35">{hint}</span>}
          {msg && <span className="text-[10px] text-rose-300">{msg}</span>}
        </div>
      )}
      {(detail.data.stale_reviews?.length ?? 0) > 0 && (
        <p className="text-[10px] text-gold-400/80">
          <AlertTriangle size={10} className="inline mr-1" />
          {T("review.stale").replace("{n}", String(detail.data.stale_reviews!.length))}
        </p>
      )}
      {obs.length === 0 ? (
        <p className="text-[11px] text-white/40">{T("detailEmpty")}</p>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-white/5">
          <table className="w-full text-[10px] border-collapse">
            <thead>
              <tr className="border-b border-white/5 bg-white/3">
                {["obsSheet", "obsGroup", "obsCovariate", "obsValue", "obsCases", "obsPopulation", "obsWhere", "obsQuote", "obsReview"].map((h) => (
                  <th key={h} className="text-left px-2.5 py-1.5 text-white/40 font-semibold uppercase tracking-wider whitespace-nowrap">{T(h)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {obs.map((o, i) => {
                const st: ReviewStatus = o.review_status ?? "unreviewed";
                const eff = o.effective ?? o;
                const isEdit = editing !== null && editing === o.obs_key;
                const changed = (k: "value" | "n_cases" | "pop_risk") => st === "edited" && eff[k] !== o[k];
                const num = (k: "value" | "n_cases" | "pop_risk") => (
                  <span className="font-mono">
                    {eff[k] ?? "-"}
                    {changed(k) && <span className="ml-1 text-white/30 line-through">{o[k] ?? "-"}</span>}
                  </span>
                );
                return (
                  <tr key={o.obs_key ?? i} className={`border-b border-white/5 align-top ${ROW_STYLE[st]}`}>
                    <td className="px-2.5 py-1.5 whitespace-nowrap text-white/55">{t(`scenarioDetail.extraction.sheet.${o.sheet}`)}</td>
                    <td className="px-2.5 py-1.5 text-white/60">{eff.group || "-"}</td>
                    <td className="px-2.5 py-1.5 text-white/80">
                      {isEdit
                        ? <input aria-label={T("obsCovariate")} className={input} value={form.covariate} onChange={(e) => setForm({ ...form, covariate: e.target.value })} />
                        : <>
                            <span className={st === "rejected" ? "line-through" : ""}>{eff.covariate}</span>
                            {st === "edited" && eff.covariate !== o.covariate && <span className="ml-1 text-white/30 line-through">{o.covariate}</span>}
                            {eff.label_path
                              ? <div className="text-[9px] text-brand-300/70">{eff.label_path.replace(/_/g, " ")}</div>
                              : <div className="text-[9px] text-gold-400/70">{T("unmapped")}</div>}
                          </>}
                    </td>
                    {(["value", "n_cases", "pop_risk"] as const).map((k) => (
                      <td key={k} className="px-2.5 py-1.5 text-white/65">
                        {isEdit
                          ? <input aria-label={T(k === "value" ? "obsValue" : k === "n_cases" ? "obsCases" : "obsPopulation")} className={input}
                              value={form[k]} onChange={(e) => setForm({ ...form, [k]: e.target.value })} inputMode="decimal" />
                          : num(k)}
                      </td>
                    ))}
                    <td className="px-2.5 py-1.5 text-white/50 whitespace-nowrap">
                      {[o.page_section, o.source_kind].filter(Boolean).join(" · ") || "-"}
                    </td>
                    <td className="px-2.5 py-1.5 max-w-[300px]">
                      <div className="flex items-start gap-1.5">
                        {o.quote_verified
                          ? <CheckCircle2 size={11} className="mt-0.5 text-brand-300 shrink-0" aria-label={T("quoteFound")} />
                          : <AlertTriangle size={11} className="mt-0.5 text-gold-400 shrink-0" aria-label={T("quoteNotFound")} />}
                        <span className={`leading-4 italic ${o.quote_verified ? "text-white/55" : "text-gold-400/80"}`}>{o.quote || "-"}</span>
                      </div>
                    </td>
                    <td className="px-2.5 py-1.5 whitespace-nowrap">
                      <div className="mb-1 text-[9px] uppercase tracking-wider text-white/45">
                        {T(`review.status.${st}`)}
                        {(o.reviews?.length ?? 0) > 0 && <span className="ml-1 normal-case tracking-normal text-white/30">{o.reviews!.map((r) => r.reviewer).join(", ")}</span>}
                      </div>
                      {isEdit ? (
                        <div className="flex gap-1">
                          <button disabled={busy || !editIsValid(form)} onClick={() => void act(o, "edited", editPayload(form))}
                            className="rounded-md border border-brand-500/40 bg-brand-500/15 px-2 py-1 text-brand-300 disabled:opacity-40">{T("review.save")}</button>
                          <button onClick={() => setEditing(null)} className="rounded-md border border-white/10 px-2 py-1 text-white/60">{T("review.cancel")}</button>
                        </div>
                      ) : (
                        <div className="flex gap-1">
                          {([
                            { k: "accepted", icon: <Check size={11} />, label: T("review.accept"), run: () => void act(o, "accepted") },
                            { k: "edit", icon: <Pencil size={11} />, label: T("review.edit"), run: () => startEdit(o) },
                            { k: "rejected", icon: <X size={11} />, label: T("review.reject"), run: () => void act(o, "rejected") },
                            { k: "clear", icon: <Trash2 size={11} />, label: T("review.clear"), run: () => void act(o, "clear") },
                          ] as const).map((b) => (
                            <button key={b.k} disabled={!can || busy} onClick={b.run} title={hint || b.label} aria-label={b.label}
                              className="rounded-md border border-white/10 bg-white/5 p-1.5 text-white/60 hover:bg-white/10 hover:text-white transition disabled:opacity-30">
                              {b.icon}
                            </button>
                          ))}
                        </div>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
