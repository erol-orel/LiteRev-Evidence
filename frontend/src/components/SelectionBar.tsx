import React from "react";
import { AlertTriangle, Loader2, X } from "lucide-react";
import { useI18n } from "../i18n/LanguageProvider";
import {
  applyScenarioSubset, downloadScenarioSubset, fetchScenarioSubsetState, hasApiKey,
  previewScenarioSubset, undoScenarioSubset,
} from "../lib/api";
import type { SubsetCondition, SubsetPlan, SubsetState } from "../lib/api";
import { ExportButton } from "./ExportButton";

/** La sélection partagée : une liste de conditions sur les variables du registre, ce
 *  qu'elles feraient, et le bouton qui le fait.
 *
 *  Le mécanisme dessous est celui des clusters et des pays, inchangé : les articles sortent
 *  en `excluded` sur la ligne de liaison du scénario, avec leur motif, ce qui les fait
 *  apparaître dans la PRISMA et retire des extractions suivantes ; l'annulation ne défait
 *  que ces lignes-là. Ce composant n'ajoute donc pas une deuxième notion de « corpus », il
 *  donne une porte d'entrée à toutes les variables au lieu de cinq.
 *
 *  Un article qu'AUCUNE condition ne sait juger est gardé, sauf demande explicite : une
 *  absence de valeur n'est pas un « non ». */
export function SelectionBar({ scenarioId, conditions, combine, onCombine, onRemove, onApplied }: {
  scenarioId: string;
  conditions: SubsetCondition[];
  combine: "all" | "any";
  onCombine: (c: "all" | "any") => void;
  onRemove: (variable: string) => void;
  /** Le corpus vient de changer sous tous les panneaux : le registre se recharge. */
  onApplied: () => void;
}) {
  const { t } = useI18n();
  const T = (k: string) => t(`scenarioDetail.register.sel.${k}`);
  const [plan, setPlan] = React.useState<SubsetPlan | null>(null);
  const [state, setState] = React.useState<SubsetState | null>(null);
  const [dropUnjudged, setDropUnjudged] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [note, setNote] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const keyed = hasApiKey();

  // Les conditions COMPLÈTES : une variable sans valeur ni borne ne se prévisualise pas.
  const ready = React.useMemo(
    () => conditions.filter((c) => (c.values && c.values.length) || c.min != null || c.max != null),
    [conditions],
  );

  const loadState = React.useCallback(() => {
    fetchScenarioSubsetState(scenarioId).then(setState).catch(() => setState(null));
  }, [scenarioId]);
  React.useEffect(loadState, [loadState]);

  React.useEffect(() => {
    if (!ready.length) { setPlan(null); setError(null); return; }
    let alive = true;
    previewScenarioSubset(scenarioId, {
      conditions: ready, combine, unassigned: dropUnjudged ? "exclude" : "keep",
    })
      .then((p) => { if (alive) { setPlan(p); setError(null); } })
      .catch((e: Error) => { if (alive) { setPlan(null); setError(e.message); } });
    return () => { alive = false; };
  }, [scenarioId, ready, combine, dropUnjudged]);

  const apply = async () => {
    setBusy(true);
    setNote(null);
    try {
      const out = await applyScenarioSubset(scenarioId, {
        conditions: ready, combine, unassigned: dropUnjudged ? "exclude" : "keep",
      });
      setNote(T("applied").replace("{n}", String(out.applied ?? out.exclude)));
      loadState();
      onApplied();
    } catch (e) {
      setNote((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const undo = async () => {
    setBusy(true);
    setNote(null);
    try {
      const out = await undoScenarioSubset(scenarioId);
      setNote(T("undone").replace("{n}", String(out.restored)));
      loadState();
      onApplied();
    } catch (e) {
      setNote((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const inForce = state?.narrowed ? state : null;
  if (!conditions.length && !inForce) return null;

  return (
    <div className="rounded-2xl border border-brand-500/25 bg-brand-500/5 px-3 py-2.5 space-y-2">
      {conditions.length > 0 && (
        <>
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] font-semibold text-brand-200">{T("title")}</span>
            {conditions.map((c) => (
              <span key={c.variable}
                className="inline-flex items-center gap-1 rounded-lg border border-white/10 bg-white/[0.06] px-2 py-0.5 text-[11px] text-white/80">
                <span className="font-mono text-[10px] text-white/50">{c.variable}</span>
                {c.values?.length ? c.values.join(", ") : `${c.min ?? ""}..${c.max ?? ""}`}
                <button onClick={() => onRemove(c.variable)} aria-label={T("remove")}
                  className="text-white/40 hover:text-rose-300"><X size={10} /></button>
              </span>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-3 text-[10.5px] text-white/55">
            {conditions.length > 1 && (
              <span className="inline-flex items-center gap-1">
                {(["all", "any"] as const).map((m) => (
                  <button key={m} onClick={() => onCombine(m)}
                    className={`rounded px-1.5 py-px ${combine === m
                      ? "bg-pill text-onpill font-semibold" : "text-white/50 hover:text-white"}`}>
                    {T(m)}
                  </button>
                ))}
              </span>
            )}
            <label className="inline-flex items-center gap-1.5 cursor-pointer">
              <input type="checkbox" checked={dropUnjudged}
                onChange={(e) => setDropUnjudged(e.target.checked)}
                className="h-3 w-3 accent-[rgb(var(--brand-500))]" />
              {T("dropUnjudged")}
            </label>
          </div>
        </>
      )}

      {error && (
        <p className="text-[10.5px] text-rose-300"><AlertTriangle size={10} className="inline mr-1" />{error}</p>
      )}

      {plan && (
        <p className="text-[10.5px] leading-4 text-white/70">
          {T("plan").replace("{keep}", plan.keep.toLocaleString())
            .replace("{relevant}", plan.relevant.toLocaleString())
            .replace("{exclude}", plan.exclude.toLocaleString())}
          {plan.undecided > 0 && " " + T("unjudged").replace("{n}", plan.undecided.toLocaleString())}
          {plan.kept_manual ? " " + T("keptManual").replace("{n}", String(plan.kept_manual)) : ""}
        </p>
      )}

      <div className="flex flex-wrap items-center gap-2">
        {/* L'export vient AVANT appliquer, et ne demande pas de clé : c'est ce qu'un
            relecteur veut d'abord, la liste des articles que sa sélection garde. Appliquer
            exclut le reste du corpus et se cumule ; télécharger ne touche à rien. */}
        {plan && plan.keep > 0 && (
          <ExportButton
            onClick={() => downloadScenarioSubset(scenarioId, {
              conditions: ready, combine, unassigned: dropUnjudged ? "exclude" : "keep",
            })}
            label={T("export").replace("{n}", plan.keep.toLocaleString())}
            title={T("exportHint")} />
        )}
        {plan && plan.exclude > 0 && (
          <button onClick={apply} disabled={busy || !keyed}
            title={keyed ? undefined : T("needsKey")}
            className="rounded-lg bg-brand-600 px-2.5 py-1 text-[11px] font-semibold text-onsolid disabled:opacity-40">
            {busy ? <Loader2 size={11} className="inline animate-spin mr-1" /> : null}
            {T("apply").replace("{n}", plan.exclude.toLocaleString())}
          </button>
        )}
        {inForce && (
          <button onClick={undo} disabled={busy || !keyed}
            className="rounded-lg border border-white/15 px-2.5 py-1 text-[11px] text-white/75 disabled:opacity-40">
            {T("undo").replace("{n}", String(inForce.excluded_by_scope))}
          </button>
        )}
      </div>

      {inForce && (
        <div className="space-y-0.5 border-t border-white/5 pt-1.5">
          <p className="text-[10px] text-white/45">{T("inForce").replace("{n}", String(inForce.excluded_by_scope))}</p>
          {inForce.steps.slice(0, 4).map((s) => (
            <p key={s.reason} className="truncate font-mono text-[9.5px] text-white/35" title={s.reason}>
              {s.articles} &middot; {s.reason}
            </p>
          ))}
        </div>
      )}

      <p className="text-[10px] leading-4 text-white/40">{T("caveat")}</p>
      {note && <p className="text-[10.5px] text-gold-300/90">{note}</p>}
    </div>
  );
}
