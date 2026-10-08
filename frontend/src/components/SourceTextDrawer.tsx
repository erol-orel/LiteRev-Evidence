import React from "react";
import { AlertTriangle, CheckCircle2, X } from "lucide-react";
import { useI18n } from "../i18n/LanguageProvider";
import { fetchTextWindow } from "../lib/api";
import type { TextWindow } from "../lib/api";

const STEP = 1500;
const MAX_CONTEXT = 6000;

/** The paper's own text around a quote, with the quote highlighted: the quickest way for a
 *  reviewer to see whether a value was read correctly. It shows the text the model read, cut
 *  at the same size, and says plainly when the quote was found, only partly found, or not found. */
export function SourceTextDrawer({ scenarioId, articleId, quote, onClose }: {
  scenarioId: string; articleId: number; quote: string; onClose: () => void;
}) {
  const { t } = useI18n();
  const T = (k: string) => t(`scenarioDetail.extraction.source.${k}`);
  const [context, setContext] = React.useState(700);
  const [w, setW] = React.useState<TextWindow | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const mark = React.useRef<HTMLElement>(null);

  React.useEffect(() => {
    let alive = true;
    fetchTextWindow(scenarioId, articleId, quote, context)
      .then((d) => { if (alive) { setW(d); setError(null); } })
      .catch((e: Error) => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [scenarioId, articleId, quote, context]);

  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  React.useEffect(() => { mark.current?.scrollIntoView({ block: "center" }); }, [w?.start, w?.window_start]);

  const more = !!w && (w.window_start > 0 || w.window_end < w.n_chars) && context < MAX_CONTEXT;
  const status = !w ? null : w.found ? "found" : w.partial ? "partial" : "notFound";

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/50" onClick={onClose}>
      <aside role="dialog" aria-modal="true" aria-label={T("title")} onClick={(e) => e.stopPropagation()}
        className="flex h-full w-full max-w-[600px] flex-col border-l border-white/10 bg-[#0b1020] shadow-2xl">
        <header className="flex items-start justify-between gap-3 border-b border-white/10 px-4 py-3">
          <div className="min-w-0">
            <h3 className="text-xs font-semibold text-white uppercase tracking-wider">{T("title")}</h3>
            {w && <p className="mt-0.5 truncate text-[11px] text-white/50" title={w.title}>{w.title}</p>}
          </div>
          <button onClick={onClose} aria-label={T("close")} className="rounded-lg border border-white/10 p-1.5 text-white/60 hover:bg-white/10 hover:text-white transition">
            <X size={14} />
          </button>
        </header>

        <div className="space-y-2 border-b border-white/5 px-4 py-3">
          <p className="rounded-lg border border-white/10 bg-white/3 px-3 py-2 text-[11px] italic leading-4 text-white/65">{quote}</p>
          {w && (
            <div className="flex flex-wrap items-center gap-2 text-[10px]">
              <span className={`rounded-md border px-1.5 py-0.5 ${w.source === "fulltext" ? "border-brand-500/20 bg-brand-500/10 text-brand-300" : "border-gold-400/20 bg-gold-400/10 text-gold-400"}`}>
                {w.source === "fulltext" ? T("fromFulltext") : T("fromAbstract")}
              </span>
              {status === "found" && <span className="flex items-center gap-1 text-brand-300"><CheckCircle2 size={11} />{T("found")}</span>}
              {status === "partial" && <span className="flex items-center gap-1 text-gold-400"><AlertTriangle size={11} />{T("partial")}</span>}
              {status === "notFound" && <span className="flex items-center gap-1 text-rose-300"><AlertTriangle size={11} />{T("notFound")}</span>}
            </div>
          )}
          {w?.text_truncated && <p className="text-[10px] text-white/40">{T("truncated")}</p>}
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-4">
          {error && <p className="text-[11px] text-rose-300">{error}</p>}
          {!w && !error && <p className="text-[11px] text-white/40">{T("loading")}</p>}
          {w && (
            <p className="whitespace-pre-wrap break-words text-[12px] leading-5 text-white/70">
              {w.window_start > 0 && <span className="text-white/30">... </span>}
              {w.before}
              {w.match && <mark ref={mark} className="rounded bg-gold-400/30 px-0.5 text-white">{w.match}</mark>}
              {w.after}
              {w.window_end < w.n_chars && <span className="text-white/30"> ...</span>}
            </p>
          )}
          {more && (
            <button onClick={() => setContext((c) => Math.min(MAX_CONTEXT, c + STEP))}
              className="mt-3 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-[10px] text-white/70 hover:bg-white/10 transition">
              {T("showMore")}
            </button>
          )}
        </div>
      </aside>
    </div>
  );
}
