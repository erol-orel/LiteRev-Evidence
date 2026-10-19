import React from "react";
import { AlertTriangle, Check, Clock, Minus } from "lucide-react";
import { useI18n, currentLang } from "../i18n/LanguageProvider";
import { fetchScenarioOutputs } from "../lib/api";
import type { ScenarioOutput, ScenarioOutputs } from "../lib/api";

/** Ce que ce scénario devrait contenir, et ce qu'il contient.
 *
 *  Chaque résultat avait son rangement, sa règle de fraîcheur et sa panne silencieuse, et
 *  aucun écran ne disait ce qu'un scénario fini porte. C'est ainsi que le rapport de
 *  preuves a pu être détruit par l'onglet qui l'affiche pendant des semaines : personne
 *  ne demandait jamais s'il était là. Le détecteur était un relecteur devant des
 *  partenaires, ce qui est le détecteur le plus cher qui soit.
 *
 *  Une pastille par résultat, son état, et pour une couverture le nombre d'articles sur
 *  le nombre attendu. Lecture seule : rien n'est lancé d'ici, et ce qui manque est chiffré
 *  par classe (gratuit, un appel, un appel PAR ARTICLE) pour que la décision de dépenser
 *  soit prise en connaissance de cause. */
export function ReadinessStrip({ scenarioId, onGoTo }: {
  scenarioId: string;
  /** Aller à l'onglet où le résultat se lit. */
  onGoTo?: (tab: string) => void;
}) {
  const { t, lang } = useI18n();
  const T = (k: string) => t(`scenarioDetail.readiness.${k}`);
  const [data, setData] = React.useState<ScenarioOutputs | null>(null);
  const [open, setOpen] = React.useState(false);

  React.useEffect(() => {
    let alive = true;
    setData(null);
    fetchScenarioOutputs(scenarioId, currentLang())
      .then((d) => { if (alive) setData(d); })
      .catch(() => {});
    return () => { alive = false; };
  }, [scenarioId, lang]);

  if (!data) return null;

  // Les états « sans objet » et « hors nature » ne sont pas des manques : les montrer
  // comme tels apprendrait à ignorer le panneau. Ils restent lisibles dans le détail.
  const shown = data.outputs.filter((o) => o.state !== "not_applicable");
  const priced = data.cost_of_todo.one_pass || 0;

  return (
    <div className="rounded-2xl border border-white/5 bg-white/[0.02] px-3 py-2">
      <button onClick={() => setOpen((o) => !o)} aria-expanded={open}
        className="flex w-full flex-wrap items-center gap-2 text-left">
        <span className="text-[10.5px] font-semibold uppercase tracking-wider text-white/55">
          {T("title")}
        </span>
        {data.complete ? (
          <span className="flex items-center gap-1 text-[11px] text-brand-300">
            <Check size={11} />{T("complete")}
          </span>
        ) : (
          <span className="text-[11px] text-gold-300/90">
            {T("todo").replace("{n}", String(data.todo.length))}
            {priced > 0 && " " + T("priced").replace("{n}", String(priced))}
          </span>
        )}
        <span className="ml-auto flex flex-wrap items-center gap-1">
          {shown.map((o) => <Dot key={o.key} o={o} />)}
        </span>
      </button>

      {open && (
        <div className="mt-2 space-y-1 border-t border-white/5 pt-2">
          <p className="text-[10px] leading-4 text-white/40">
            {T("note").replace("{relevant}", data.n_relevant.toLocaleString())
              .replace("{corpus}", data.n_corpus.toLocaleString())}
          </p>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-[10px]">
              <tbody>
                {data.outputs.map((o) => (
                  <tr key={o.key} className="border-b border-white/5">
                    <td className="px-1.5 py-1 whitespace-nowrap">
                      <Dot o={o} />
                    </td>
                    <td className="px-1.5 py-1 text-white/75">
                      {onGoTo && o.state !== "not_applicable" ? (
                        <button onClick={() => onGoTo(o.tab)} className="underline-offset-2 hover:underline">
                          {o.name}
                        </button>
                      ) : o.name}
                    </td>
                    <td className="px-1.5 py-1 font-mono text-white/55">
                      {o.shape === "coverage" && o.of != null
                        ? `${(o.n ?? 0).toLocaleString()} / ${o.of.toLocaleString()}`
                        : o.generated_at
                        ? new Date(o.generated_at).toLocaleDateString(lang === "fr" ? "fr-FR" : "en-US")
                        : ""}
                    </td>
                    <td className="px-1.5 py-1 text-white/40">
                      {T(`state.${o.state}`)}
                      {o.stale_because && ` (${o.stale_because})`}
                    </td>
                    <td className="px-1.5 py-1 text-white/35">{T(`cost.${o.cost}`)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-[10px] leading-4 text-white/35">{T("footer")}</p>
        </div>
      )}
    </div>
  );
}

/** La pastille d'un résultat : couleur ET forme, jamais la couleur seule. */
function Dot({ o }: { o: ScenarioOutput }) {
  const { t } = useI18n();
  const T = (k: string) => t(`scenarioDetail.readiness.${k}`);
  const label = `${o.name}: ${T(`state.${o.state}`)}`
    + (o.shape === "coverage" && o.of != null ? ` (${o.n ?? 0}/${o.of})` : "");
  if (o.state === "ready") {
    return <span title={label} className="inline-flex h-4 w-4 items-center justify-center rounded-full bg-brand-500/15 text-brand-300"><Check size={9} /></span>;
  }
  if (o.state === "stale") {
    return <span title={label} className="inline-flex h-4 w-4 items-center justify-center rounded-full bg-gold-500/15 text-gold-300"><Clock size={9} /></span>;
  }
  if (o.state === "partial") {
    return <span title={label} className="inline-flex h-4 min-w-4 items-center justify-center rounded-full bg-gold-500/10 px-1 font-mono text-[8px] text-gold-200/90">{Math.round(((o.n ?? 0) / Math.max(o.of ?? 1, 1)) * 100)}</span>;
  }
  if (o.state === "missing") {
    return <span title={label} className="inline-flex h-4 w-4 items-center justify-center rounded-full bg-rose-500/15 text-rose-300"><AlertTriangle size={9} /></span>;
  }
  // « sans objet » et « hors nature » : gris, discrets, et nommés dans l'infobulle.
  return <span title={label} className="inline-flex h-4 w-4 items-center justify-center rounded-full bg-white/5 text-white/30"><Minus size={9} /></span>;
}
