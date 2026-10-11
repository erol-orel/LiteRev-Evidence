// Pure helpers of the search page: query text, facet kinds, display names. They live
// outside App.tsx so they can be unit-tested without rendering the application.
import type { SubQuery } from "./api";

// Bornes du filtre Années : du plus ancien article réellement en base (en
// ignorant les années aberrantes < 1000) jusqu'à l'année courante (aujourd'hui).
export function yearSliderBounds(yearOpts?: Array<{ value: string | number }> | null): { min: number; max: number } {
  const yrs = (yearOpts ?? []).map(y => Number(y.value)).filter(y => Number.isFinite(y) && y > 1000);
  return { min: yrs.length ? Math.min(...yrs) : 1990, max: new Date().getFullYear() };
}

// Les bornes d'années qui partent avec la recherche, dans les filtres du scénario : celles
// que le curseur a resserrées, et elles seules. Le curseur était affiché sans être envoyé
// nulle part. Une borne écarte aussi les articles SANS année (`d.year >= x` est faux sur
// NULL) : le curseur laissé à ses extrémités ne doit donc rien borner.
export function yearBoundsFilter(
  range: [number, number],
  yearOpts?: Array<{ value: string | number }> | null,
): { year_min?: number; year_max?: number } {
  const b = yearSliderBounds(yearOpts);
  return {
    ...(range[0] > b.min ? { year_min: range[0] } : {}),
    ...(range[1] < b.max ? { year_max: range[1] } : {}),
  };
}

// Le curseur d'une recherche rejouée : ses bornes enregistrées, ramenées dans celles du
// curseur ; une borne absente le laisse à son extrémité.
export function yearRangeOf(
  filters: Record<string, unknown> | null | undefined,
  yearOpts?: Array<{ value: string | number }> | null,
): [number, number] {
  const b = yearSliderBounds(yearOpts);
  const at = (v: unknown, fallback: number) => {
    const n = typeof v === "number" ? v : typeof v === "string" && v.trim() ? Number(v) : NaN;
    return Number.isFinite(n) ? Math.min(Math.max(Math.round(n), b.min), b.max) : fallback;
  };
  const lo = at(filters?.year_min, b.min);
  const hi = at(filters?.year_max, b.max);
  return lo <= hi ? [lo, hi] : [hi, lo];
}

export function csvEscape(value: unknown): string {
  return JSON.stringify(value ?? "");
}

// Libellé court et lisible pour un scénario créé depuis une requête. Une requête
// booléenne (ou multi-sous-requêtes) dépasse souvent la colonne `name` VARCHAR(255) ;
// le nom n'est qu'un AFFICHAGE - la requête complète reste dans `query`. On tronque
// sur une frontière de mot quand c'est possible, avec « … ».
export function scenarioDisplayName(q: string, limit = 140): string {
  const s = (q ?? "").trim();
  if (s.length <= limit) return s;
  const cut = s.slice(0, limit);
  const lastSpace = cut.lastIndexOf(" ");
  return (lastSpace > limit * 0.6 ? cut.slice(0, lastSpace) : cut).trimEnd() + "…";
}

// Miroir CLIENT de main.py:_combined_query_text - expression COMPLÈTE d'une recherche
// multi-facettes, parenthésée selon le fold gauche→droite réellement appliqué :
// « (A) AND (B) », « ((A) OR (B)) AND (C) ». Sert de nom par défaut au scénario :
// avant, seul le texte de la requête principale était utilisé et le ET/OU entre
// les facettes n'apparaissait nulle part.
export function combinedQueryText(sub: SubQuery[], combinator: "union" | "intersection"): string {
  const facets = sub.map((q) => ({ ...q, text: q.text.trim() })).filter((q) => q.text);
  if (facets.length < 2) return facets[0]?.text ?? "";
  const defaultOp = combinator === "intersection" ? "and" : "or";
  let expr = facets[0].text;
  for (const f of facets.slice(1)) {
    expr = `(${expr}) ${(f.op ?? defaultOp).toUpperCase()} (${f.text})`;
  }
  return expr;
}

// Miroir CLIENT de main.py:_looks_boolean - détecte une SYNTAXE booléenne (opérateurs
// AND/OR/NOT en majuscules, tags [dp]/[tiab]…, guillemets doubles, parenthèses) pour
// afficher le type détecté sans que l'utilisateur ait à le taguer. Le backend refait
// la même détection (source de vérité) ; ceci n'est que l'indicateur d'UI.
export function looksBoolean(text: string): boolean {
  const t = (text ?? "").trim();
  if (!t) return false;
  if (/\[(dp|tiab|ti|ab|mesh|majr|au|tw|la|pt)\]/i.test(t)) return true;
  if (/\b(AND|OR|NOT)\b/.test(t)) return true;
  if ((t.match(/"/g)?.length ?? 0) >= 2) return true;
  if (t.includes("(") && t.includes(")")) return true;
  return false;
}

// Type EFFECTIF d'une facette : override explicite (boolean|natural) sinon détecté.
export function effectiveKind(sq: SubQuery): "boolean" | "natural" {
  if (sq.kind === "boolean" || sq.kind === "natural") return sq.kind;
  return looksBoolean(sq.text) ? "boolean" : "natural";
}
