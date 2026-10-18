import React from "react";
import { AlertTriangle, ChevronDown, ChevronRight } from "lucide-react";
import { useI18n, currentLang } from "../i18n/LanguageProvider";
import {
  fetchVariableGap, fetchVariablesOfInterest, hasApiKey, runExtractionForNewVariables,
  variablesOfInterestExportUrl,
} from "../lib/api";
import type { SubsetCondition, VariableGap, VoiRegister, VoiVariable } from "../lib/api";
import { share } from "../lib/extraction";
import { SelectionBar } from "./SelectionBar";
import { CrosstabPanel } from "./CrosstabPanel";

/** Le registre des variables d'intérêt : ce que cette revue demande à CHAQUE article, en une
 *  seule liste, compté sur tout le sous-ensemble pertinent.
 *
 *  Les mêmes faits existent déjà, éparpillés : l'année et la revue dans le Corpus, le devis et
 *  le niveau de preuve dans les barres du profil de preuve, le pays dans Géographie, le thème
 *  dans Clustering, les concepts dans le graphe, le PICO dans son onglet, les lignes par
 *  covariable dans l'Extraction. Aucun écran ne disait « voilà les vingt-neuf questions que
 *  nous posons, et combien d'articles y ont répondu ». Celui-ci le dit, et rien n'est retiré
 *  ailleurs : tant que les deux comptes cohabitent, un écart se voit.
 *
 *  Lecture seule. Aucune extraction n'est lancée, aucun modèle appelé, rien n'est écrit.
 *  Les pastilles FILTRER et CROISER sont affichées et inertes : la page ne promet pas ce
 *  qu'elle ne fait pas encore. */
export function VariablesOfInterestPanel({ scenarioId }: { scenarioId: string }) {
  const { t, lang } = useI18n();
  const T = (k: string) => t(`scenarioDetail.register.${k}`);
  const [reg, setReg] = React.useState<VoiRegister | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [open, setOpen] = React.useState<Record<string, boolean>>({});
  // La sélection en cours de composition : une condition par variable, dans l'ordre où
  // elles ont été ouvertes. Elle ne devient une exclusion qu'au clic sur le bouton.
  const [conds, setConds] = React.useState<SubsetCondition[]>([]);
  const [combine, setCombine] = React.useState<"all" | "any">("all");
  const [reload, setReload] = React.useState(0);
  // Le croisement : la première variable cochée devient les lignes, la seconde les
  // colonnes, une troisième repart des lignes. Deux clics suffisent, sans menu.
  const [cross, setCross] = React.useState<string[]>([]);
  // La facture d'une relecture : ce que coûterait de poser aux anciens articles une
  // variable ajoutée depuis. Lue séparément du registre, parce qu'elle parle de dépense
  // et non de contenu, et parce qu'elle doit pouvoir manquer sans vider la page.
  const [gap, setGap] = React.useState<VariableGap | null>(null);
  const [asking, setAsking] = React.useState(false);
  const [askNote, setAskNote] = React.useState<string | null>(null);

  React.useEffect(() => {
    let alive = true;
    setReg(null);
    setError(null);
    fetchVariablesOfInterest(scenarioId, currentLang())
      .then((r) => { if (alive) setReg(r); })
      .catch((e: Error) => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [scenarioId, lang, reload]);
  // Un scénario différent, une sélection différente : garder les conditions ferait porter
  // sur un corpus des valeurs lues dans un autre.
  React.useEffect(() => { setConds([]); }, [scenarioId]);

  React.useEffect(() => {
    let alive = true;
    setGap(null);
    fetchVariableGap(scenarioId)
      .then((g) => { if (alive) setGap(g); })
      .catch(() => {});
    return () => { alive = false; };
  }, [scenarioId, reload]);

  const askNewVariables = async () => {
    setAsking(true);
    setAskNote(null);
    try {
      const out = await runExtractionForNewVariables(scenarioId);
      setAskNote(out.status === "running" ? T("gapRunning")
        : out.status === "paused" ? (out.reason || T("gapPaused"))
        : out.status === "no_llm" ? T("gapNoLlm") : T("gapStarted"));
    } catch (e) {
      setAskNote((e as Error).message);
    } finally {
      setAsking(false);
    }
  };

  const condOf = (key: string) => conds.find((c) => c.variable === key);
  const toggleCond = (v: VoiVariable) => setConds((cs) => (
    cs.some((c) => c.variable === v.key)
      ? cs.filter((c) => c.variable !== v.key)
      : [...cs, { variable: v.key, ...(v.type === "number" ? { min: null, max: null } : { values: [] })}]
  ));
  const setValues = (key: string, values: string[]) =>
    setConds((cs) => cs.map((c) => (c.variable === key ? { ...c, values } : c)));
  const setBound = (key: string, which: "min" | "max", raw: string) =>
    setConds((cs) => cs.map((c) => (c.variable === key
      ? { ...c, [which]: raw.trim() === "" ? null : Number(raw) } : c)));
  const removeCond = (key: string) => setConds((cs) => cs.filter((c) => c.variable !== key));
  const toggleCross = (key: string) => setCross((xs) => (
    xs.includes(key) ? xs.filter((x) => x !== key) : [...xs, key].slice(-2)));

  if (error) {
    return (
      <p className="text-[11px] text-rose-300">
        <AlertTriangle size={11} className="inline mr-1" />{error}
      </p>
    );
  }
  if (!reg) return <p className="text-[11px] text-white/40">{t("common.loadingEllipsis")}</p>;

  /** Le nom d'une variable : celui du codebook quand elle en a un, sinon la chaîne
   *  d'interface de son nom standard. Une clé sans traduction retombe sur elle-même
   *  plutôt que d'afficher le chemin de la locale. */
  const name = (v: VoiVariable) => {
    if (v.name) return v.name;
    const s = t(`scenarioDetail.register.var.${v.key}`);
    return s.startsWith("scenarioDetail.") ? v.key.replace(/_/g, " ") : s;
  };
  const label = (k: string, fallback: string) => {
    const s = t(`scenarioDetail.register.${k}`);
    return s.startsWith("scenarioDetail.") ? fallback : s;
  };

  const families = reg.families.filter((f) => reg.variables.some((v) => v.family === f));

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-white/90">{T("title")}</h3>
          <p className="mt-1 max-w-[760px] text-[11px] leading-5 text-white/55">
            {T("subtitle").replace("{relevant}", reg.n_relevant.toLocaleString())}
          </p>
        </div>
        {/* La carte des manques dans la forme qu'un partenaire colle dans son tableur. */}
        <a href={variablesOfInterestExportUrl(scenarioId, currentLang())}
          className="shrink-0 rounded-lg border border-white/10 px-2 py-1 text-[10.5px] text-white/65 hover:text-white hover:border-white/25">
          {T("exportCsv")}
        </a>
      </div>

      <div className="flex flex-wrap gap-2">
        {[
          { v: reg.n_standard, l: T("tileStandard") },
          { v: reg.n_own, l: T("tileOwn") },
          { v: reg.n_relevant, l: T("tileRelevant") },
          { v: reg.n_extracted, l: T("tileRead") },
        ].map((tile) => (
          <div key={tile.l} className="min-w-[120px] flex-1 rounded-xl border border-white/5 bg-white/2 px-3 py-2">
            <div className="font-mono text-lg leading-tight text-white/90">{tile.v.toLocaleString()}</div>
            <div className="text-[10px] text-white/40">{tile.l}</div>
          </div>
        ))}
      </div>

      {/* La facture d'une relecture. Elle n'apparaît que s'il y a quelque chose à relire,
          et le bouton est le SEUL chemin qui dépense : ni le pipeline ni l'extraction
          ordinaire ne relisent un article parce qu'une variable a été ajoutée. */}
      {gap?.complete && gap.n_papers_missing_a_variable > 0 && (
        <div className="rounded-2xl border border-gold-400/25 bg-gold-400/[0.06] px-3 py-2.5 space-y-1.5">
          <p className="text-[11px] leading-4 text-gold-200">
            {T("gapTitle")
              .replace("{papers}", gap.n_papers_missing_a_variable.toLocaleString())
              .replace("{calls}", gap.n_calls_estimate.toLocaleString())}
          </p>
          <div className="flex flex-wrap gap-1">
            {gap.by_variable.slice(0, 8).map((v) => (
              <span key={v.key} className="rounded border border-white/10 bg-white/[0.05] px-1.5 py-px text-[10px] text-white/75">
                {v.name}<span className="ml-1 font-mono text-white/40">{v.n_papers}</span>
              </span>
            ))}
          </div>
          {gap.n_papers_unknown > 0 && (
            <p className="text-[10px] leading-4 text-white/45">
              {T("gapUnknown").replace("{n}", gap.n_papers_unknown.toLocaleString())}
            </p>
          )}
          <div className="flex flex-wrap items-center gap-2">
            <button onClick={askNewVariables} disabled={asking || !hasApiKey()}
              title={hasApiKey() ? undefined : T("gapNeedsKey")}
              className="rounded-lg bg-gold-500/80 px-2.5 py-1 text-[11px] font-semibold text-onsolid disabled:opacity-40">
              {T("gapAsk").replace("{papers}", gap.n_papers_missing_a_variable.toLocaleString())}
            </button>
            {askNote && <span className="text-[10.5px] text-gold-200">{askNote}</span>}
          </div>
        </div>
      )}

      {/* Ce que les chiffres veulent dire, dit une fois, en haut : un taux de couverture sans
          son dénominateur est le défaut que cette page existe pour corriger. */}
      <p className="max-w-[760px] text-[10px] leading-4 text-white/40">{T("denominatorNote")}</p>

      <SelectionBar scenarioId={scenarioId} conditions={conds} combine={combine}
        onCombine={setCombine} onRemove={removeCond}
        onApplied={() => { setConds([]); setReload((n) => n + 1); }} />

      {cross.length === 2 && (
        <CrosstabPanel scenarioId={scenarioId} rows={cross[0]} cols={cross[1]}
          nameOf={(key) => {
            const v = reg.variables.find((x) => x.key === key);
            return v ? name(v) : key;
          }}
          onClear={() => setCross([])} />
      )}

      {Object.entries(reg.incomplete).map(([fam, why]) => (
        <p key={fam} className="text-[10px] text-gold-300/80">
          <AlertTriangle size={10} className="inline mr-1" />
          {T("familyUnread").replace("{family}", label(`family.${fam}`, fam))} {why}
        </p>
      ))}

      {families.map((fam) => {
        const vars = reg.variables.filter((v) => v.family === fam);
        const isOpen = open[fam] !== false;        // ouvertes par défaut : la page EST la liste
        return (
          <div key={fam} className="overflow-hidden rounded-2xl border border-white/5">
            <button onClick={() => setOpen((o) => ({ ...o, [fam]: !isOpen }))}
              className="flex w-full items-center justify-between gap-3 bg-white/[0.03] px-3 py-2 text-left">
              <span className="flex items-center gap-1.5 text-[11px] font-semibold text-white/80">
                {isOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                {label(`family.${fam}`, fam)}
                <span className="font-mono text-[10px] font-normal text-white/35">{vars.length}</span>
              </span>
              <span className="truncate text-[10px] text-white/35">{label(`familyHint.${fam}`, "")}</span>
            </button>
            {isOpen && (
              <div>
                {vars.filter((v) => !v.unmapped).map((v) => (
                  <VariableRow key={v.key} v={v} name={name(v)} T={T} label={label} note={reg.notes[v.key]}
                    cond={condOf(v.key)} onToggle={() => toggleCond(v)}
                    onValues={(vals) => setValues(v.key, vals)}
                    onBound={(w, raw) => setBound(v.key, w, raw)}
                    crossAs={cross.indexOf(v.key)} onCross={() => toggleCross(v.key)} />
                ))}
                {/* La traîne des libellés de groupe que le codebook ne sait pas placer. Elle
                    est à part, sous son propre compte VRAI : sur un vrai corpus elle est plus
                    longue que la liste des questions de la revue, et mêlée à elles, elle ferait
                    passer la carte des manques pour la liste de ce qui est présent. */}
                {vars.some((v) => v.unmapped) && (
                  <div className="border-t border-gold-400/20 bg-gold-400/[0.04] px-3 py-2">
                    <p className="text-[10.5px] leading-4 text-gold-300/90">
                      {T("unmappedHeader")
                        .replace("{n}", reg.n_unmapped.toLocaleString())
                        .replace("{papers}", reg.n_papers_unmapped.toLocaleString())}
                      {reg.n_unmapped > reg.n_unmapped_shown && " " + T("unmappedShown")
                        .replace("{shown}", reg.n_unmapped_shown.toLocaleString())}
                    </p>
                    <p className="mt-0.5 text-[10px] text-white/40">{T("unmappedHow")}</p>
                  </div>
                )}
                {vars.filter((v) => v.unmapped).map((v) => (
                  <VariableRow key={v.key} v={v} name={name(v)} T={T} label={label} note={reg.notes[v.key]}
                    cond={condOf(v.key)} onToggle={() => toggleCond(v)}
                    onValues={(vals) => setValues(v.key, vals)}
                    onBound={(w, raw) => setBound(v.key, w, raw)}
                    crossAs={cross.indexOf(v.key)} onCross={() => toggleCross(v.key)} />
                ))}
              </div>
            )}
          </div>
        );
      })}

      <p className="max-w-[760px] text-[10px] leading-4 text-white/35">{T("footer")}</p>
    </div>
  );
}

function VariableRow({ v, name, T, label, note, cond, onToggle, onValues, onBound,
                      crossAs, onCross }: {
  v: VoiVariable;
  name: string;
  T: (k: string) => string;
  label: (k: string, fallback: string) => string;
  note?: string;
  cond?: SubsetCondition;
  onToggle: () => void;
  onValues: (values: string[]) => void;
  onBound: (which: "min" | "max", raw: string) => void;
  /** -1 quand la variable n'est pas croisée, 0 pour les lignes, 1 pour les colonnes. */
  crossAs: number;
  onCross: () => void;
}) {
  const known = v.n_with_value != null;
  const picked = new Set(cond?.values || []);
  const toggleValue = (value: string) => onValues(
    picked.has(value) ? [...picked].filter((x) => x !== value) : [...picked, value]);
  const pct = known && v.denominator > 0 ? share(v.n_with_value as number, v.denominator) : 0;
  return (
    <div className="grid gap-x-3 gap-y-1 border-t border-white/5 px-3 py-2 md:grid-cols-[1.4fr_1.1fr_2fr]">
      <div className="min-w-0">
        <div className="flex items-center gap-1.5">
          <span className="truncate text-[12px] text-white/85">{name}</span>
          {v.unmapped && (
            <span className="shrink-0 rounded border border-gold-400/40 px-1 font-mono text-[9px] uppercase text-gold-300">
              {T("unmapped")}
            </span>
          )}
        </div>
        <div className="truncate font-mono text-[9.5px] text-white/35" title={v.key}>{v.key}</div>
        <div className="text-[10px] text-white/40">
          {label(`source.${v.source}`, v.source.replace(/_/g, " "))}
          <span className="mx-1 text-white/20">&middot;</span>
          <span className={v.cost === "one_pass" ? "text-gold-300/80" : "text-brand-300/70"}>
            {label(`cost.${v.cost}`, v.cost)}
          </span>
        </div>
      </div>

      <div className="flex min-w-0 items-center gap-2">
        <div className={`h-1.5 flex-1 overflow-hidden rounded-sm ${known ? "bg-white/5" : "bg-transparent"}`}
          style={known ? undefined : {
            backgroundImage: "repeating-linear-gradient(90deg, rgb(var(--ink) / 0.08) 0 4px, transparent 4px 8px)",
          }}>
          {known && <div className="h-full rounded-sm bg-brand-500" style={{ width: `${pct}%` }} />}
        </div>
        <span className="shrink-0 whitespace-nowrap font-mono text-[10px] text-white/55">
          {known
            ? T("coverage").replace("{n}", (v.n_with_value as number).toLocaleString())
              .replace("{of}", v.denominator.toLocaleString())
              .replace("{set}", label(`denominator.${v.denominator_of}`, v.denominator_of))
            : T("notCounted")}
        </span>
      </div>

      <div className="min-w-0">
        {v.values.length > 0 ? (
          <div className="flex flex-wrap gap-1">
            {/* Une modalité devient CLIQUABLE dès que la variable est une condition : la
                question qui suit la lecture d'un compte est toujours « et maintenant,
                donne-moi seulement ceux-là ». */}
            {v.values.map((val) => (cond && v.type !== "number" ? (
              <button key={val.value} onClick={() => toggleValue(val.value)} title={val.value}
                className={`rounded border px-1.5 py-px text-[10px] transition ${picked.has(val.value)
                  ? "border-brand-400/60 bg-brand-500/20 text-brand-200"
                  : "border-white/5 bg-white/[0.04] text-white/70 hover:border-white/20"}`}>
                {val.label}
                <span className="ml-1 font-mono text-white/40">{val.n.toLocaleString()}</span>
              </button>
            ) : (
              <span key={val.value}
                className="rounded border border-white/5 bg-white/[0.04] px-1.5 py-px text-[10px] text-white/70"
                title={val.value}>
                {val.label}
                <span className="ml-1 font-mono text-white/40">{val.n.toLocaleString()}</span>
              </span>
            )))}
            {v.n_values > v.values.length && (
              <span className="rounded border border-dashed border-white/10 px-1.5 py-px text-[10px] text-white/35">
                {T("moreValues").replace("{n}", String(v.n_values - v.values.length))}
              </span>
            )}
          </div>
        ) : (
          <p className="text-[10px] text-white/30">{note || (v.covers ? T("covers").replace("{keys}", v.covers.join(", ")) : T("noValues"))}</p>
        )}

        {cond && v.type === "number" && (
          <div className="mt-1 flex items-center gap-1.5 text-[10px] text-white/50">
            <label className="inline-flex items-center gap-1">
              {T("from")}
              <input type="number" value={cond.min ?? ""} onChange={(e) => onBound("min", e.target.value)}
                className="w-20 rounded border border-white/10 bg-white/5 px-1 py-px font-mono text-[10px] text-white/85" />
            </label>
            <label className="inline-flex items-center gap-1">
              {T("to")}
              <input type="number" value={cond.max ?? ""} onChange={(e) => onBound("max", e.target.value)}
                className="w-20 rounded border border-white/10 bg-white/5 px-1 py-px font-mono text-[10px] text-white/85" />
            </label>
          </div>
        )}

        <div className="mt-1 flex gap-1">
          {v.filterable ? (
            <button onClick={onToggle}
              className={`rounded border px-1.5 font-mono text-[9px] uppercase transition ${cond
                ? "border-brand-400/60 bg-brand-500/20 text-brand-200"
                : "border-white/10 text-white/40 hover:border-white/25 hover:text-white/70"}`}>
              {cond ? T("actFiltering") : T("actFilter")}
            </button>
          ) : (
            <span className="rounded border border-dashed border-white/[0.07] px-1.5 font-mono text-[9px] uppercase text-white/20"
              title={T("notFilterable")}>{T("actFilter")}</span>
          )}
          {v.crossable && (
            <button onClick={onCross}
              className={`rounded border px-1.5 font-mono text-[9px] uppercase transition ${crossAs >= 0
                ? "border-gold-400/60 bg-gold-400/15 text-gold-300"
                : "border-white/10 text-white/40 hover:border-white/25 hover:text-white/70"}`}>
              {crossAs === 0 ? T("actCrossRows") : crossAs === 1 ? T("actCrossCols") : T("actCross")}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
