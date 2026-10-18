import React from "react";
import { AlertTriangle, ChevronDown, ChevronRight } from "lucide-react";
import { useI18n, currentLang } from "../i18n/LanguageProvider";
import { fetchVariablesOfInterest } from "../lib/api";
import type { SubsetCondition, VoiRegister, VoiVariable } from "../lib/api";
import { share } from "../lib/extraction";
import { SelectionBar } from "./SelectionBar";

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
      <div>
        <h3 className="text-sm font-semibold text-white/90">{T("title")}</h3>
        <p className="mt-1 max-w-[760px] text-[11px] leading-5 text-white/55">
          {T("subtitle").replace("{relevant}", reg.n_relevant.toLocaleString())}
        </p>
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

      {/* Ce que les chiffres veulent dire, dit une fois, en haut : un taux de couverture sans
          son dénominateur est le défaut que cette page existe pour corriger. */}
      <p className="max-w-[760px] text-[10px] leading-4 text-white/40">{T("denominatorNote")}</p>

      <SelectionBar scenarioId={scenarioId} conditions={conds} combine={combine}
        onCombine={setCombine} onRemove={removeCond}
        onApplied={() => { setConds([]); setReload((n) => n + 1); }} />

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
                    onBound={(w, raw) => setBound(v.key, w, raw)} />
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
                    onBound={(w, raw) => setBound(v.key, w, raw)} />
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

function VariableRow({ v, name, T, label, note, cond, onToggle, onValues, onBound }: {
  v: VoiVariable;
  name: string;
  T: (k: string) => string;
  label: (k: string, fallback: string) => string;
  note?: string;
  cond?: SubsetCondition;
  onToggle: () => void;
  onValues: (values: string[]) => void;
  onBound: (which: "min" | "max", raw: string) => void;
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
          {v.crossable && <span className="rounded border border-dashed border-white/10 px-1.5 font-mono text-[9px] uppercase text-white/30">{T("actCross")}</span>}
        </div>
      </div>
    </div>
  );
}
