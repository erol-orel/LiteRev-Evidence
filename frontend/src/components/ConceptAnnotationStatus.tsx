import React from "react";
import { Loader2, Sparkles } from "lucide-react";
import { useI18n } from "../i18n/LanguageProvider";
import { hasApiKey, startConceptAnnotation } from "../lib/api";
import type { ConceptGraphData } from "../lib/api";

/** Where the concept annotation stands: how many relevant articles are not annotated yet, how many
 *  cannot be, and a button to run the pass. Reading the map never starts it: it is paid, and it
 *  used to start on every visit. */
export function ConceptAnnotationStatus({ scenarioId, data, onStarted }: {
  scenarioId: string; data: ConceptGraphData; onStarted: () => void;
}) {
  const { t } = useI18n();
  const T = (k: string) => t(`scenarioDetail.knowledgeGraph.annotation.${k}`);
  const [busy, setBusy] = React.useState(false);
  const [msg, setMsg] = React.useState<string | null>(null);
  const missing = data.n_missing_concepts ?? 0;
  const unannotatable = data.n_unannotatable ?? 0;
  const givenUp = data.n_concepts_given_up ?? 0;
  const paused = !data.enriching && data.annotation?.status === "paused";
  if (!missing && !unannotatable && !paused && !msg) return null;
  const canRun = hasApiKey() && !data.enriching;

  const run = async (retryGivenUp: boolean) => {
    const n = retryGivenUp ? givenUp + missing : missing;
    if (!window.confirm(T("confirm").replace("{n}", String(n)))) return;
    setBusy(true);
    setMsg(null);
    try {
      const r = await startConceptAnnotation(scenarioId, retryGivenUp);
      if (r.status === "no_llm") setMsg(T("noLlm"));
      else if (r.status === "paused") setMsg(T("paused").replace("{reason}", r.reason ?? ""));
      else onStarted();
    } catch (e) {
      setMsg(`${T("failed")} ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 rounded-xl border border-white/10 bg-white/3 px-3 py-2 text-xs text-white/55">
      {missing > 0 && <span>{T("missing").replace("{n}", String(missing))}</span>}
      {unannotatable > 0 && (
        <span>
          {T("unannotatable").replace("{n}", String(unannotatable))
            .replace("{given_up}", String(givenUp)).replace("{no_text}", String(data.n_without_text ?? 0))}
        </span>
      )}
      {paused && <span className="text-amber-300/80">{T("lastPaused").replace("{reason}", data.annotation?.reason ?? "")}</span>}
      {canRun && missing > 0 && (
        <button type="button" disabled={busy} onClick={() => run(false)}
          className="flex items-center gap-1.5 rounded-lg border border-brand-500/40 bg-brand-500/15 px-2.5 py-1 text-[11px] font-medium text-brand-300 hover:bg-brand-500/25 transition disabled:opacity-50">
          {busy ? <Loader2 size={11} className="animate-spin" /> : <Sparkles size={11} />}
          {T("run").replace("{n}", String(missing))}
        </button>
      )}
      {canRun && givenUp > 0 && (
        <button type="button" disabled={busy} onClick={() => run(true)} className="text-[11px] underline text-brand-300 hover:text-brand-200 disabled:opacity-50">
          {T("retry")}
        </button>
      )}
      {!hasApiKey() && missing > 0 && <span className="text-white/35">{T("needsKey")}</span>}
      {msg && <span className="text-rose-300">{msg}</span>}
    </div>
  );
}
