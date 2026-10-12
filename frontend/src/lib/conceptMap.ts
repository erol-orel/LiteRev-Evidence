// La carte des concepts telle que l'utilisateur la COMPOSE, séparée de ce que l'écran a
// la place de dessiner.
//
// Deux choses différentes vivaient dans le même ensemble `shown` : les choix de
// l'utilisateur (catégories masquées, concepts retirés, nombre minimal d'articles) et
// une coupe d'affichage (quatorze concepts par colonne, sur les soixante que le serveur
// dessine). L'export « carte telle que composée » et la restriction du corpus lisaient
// cet ensemble : les concepts coupés par l'affichage, et leurs articles, sortaient de
// l'export et de la sélection « garder », qui écrit des exclusions. Ici les deux sont
// distincts : l'écran dessine `byType`, l'export et la restriction envoient au serveur
// les FILTRES (`composedFilter`), qu'il applique à l'index complet des concepts.

import type { ConceptGraphData, ConceptNode, ConceptType } from "./api";

/** Concepts dessinés au plus par colonne : une coupe d'affichage, jamais une sélection. */
export const CONCEPTS_DRAWN_PER_TYPE = 14;

export interface ConceptMapFilters {
  hiddenTypes: Set<ConceptType>;
  hiddenNodes: Set<number>;
  minArticles: number;
}

export interface ComposedConceptMap {
  /** Les colonnes dessinées, dans l'ordre. */
  types: ConceptType[];
  /** Les noeuds dessinés par colonne (au plus CONCEPTS_DRAWN_PER_TYPE). */
  byType: Partial<Record<ConceptType, ConceptNode[]>>;
  /** Les identifiants des noeuds dessinés. */
  shown: Set<number>;
  /** Vrai dès que l'utilisateur a choisi quelque chose : seul ce choix ouvre la restriction. */
  filtersActive: boolean;
  /** Par colonne : combien sont dessinés sur combien l'index complet en retient. */
  perType: Array<{ type: ConceptType; shown: number; total: number }>;
}

/** Combien de concepts de l'index COMPLET passent les filtres, pour un type.
 *  `concept_counts` ({type: {nb d'articles: nb de concepts}}) vient du serveur ; une
 *  carte mise en cache avant lui n'en a pas, et l'on compte alors ses seuls noeuds. */
export function indexTotal(data: ConceptGraphData, type: ConceptType, f: ConceptMapFilters): number {
  const removedHere = data.nodes.filter(n => n.type === type && f.hiddenNodes.has(n.id) && n.count >= f.minArticles).length;
  const hist = data.concept_counts?.[type];
  if (!hist) {
    return data.nodes.filter(n => n.type === type && !f.hiddenNodes.has(n.id) && n.count >= f.minArticles).length;
  }
  let total = 0;
  for (const [count, n] of Object.entries(hist)) if (Number(count) >= f.minArticles) total += Number(n) || 0;
  return Math.max(0, total - removedHere);
}

/** Combien de concepts de l'index la carte composée retient, tous types confondus, y
 *  compris ceux d'un type que la carte ne dessine pas du tout : c'est ce que l'export et
 *  la restriction liront. */
export function composedConceptCount(data: ConceptGraphData, f: ConceptMapFilters): number {
  if (!data.concept_counts) {
    return data.nodes.filter(n => !f.hiddenTypes.has(n.type) && !f.hiddenNodes.has(n.id)
                                  && n.count >= f.minArticles).length;
  }
  let total = 0;
  for (const ty of Object.keys(data.concept_counts) as ConceptType[]) {
    if (!f.hiddenTypes.has(ty)) total += indexTotal(data, ty, f);
  }
  return total;
}

/** Ce que l'écran dessine, et ce qu'il ne dessine pas. Pur. */
export function composeConceptMap(
  data: ConceptGraphData, f: ConceptMapFilters, order: readonly ConceptType[],
  perColumn: number = CONCEPTS_DRAWN_PER_TYPE,
): ComposedConceptMap {
  const keep = (n: ConceptNode) => !f.hiddenNodes.has(n.id) && n.count >= f.minArticles;
  const present = order.filter(ty => data.nodes.some(n => n.type === ty));
  const types = present.filter(ty => !f.hiddenTypes.has(ty) && data.nodes.some(n => n.type === ty && keep(n)));
  const byType: Partial<Record<ConceptType, ConceptNode[]>> = {};
  const shown = new Set<number>();
  const perType: ComposedConceptMap["perType"] = [];
  for (const ty of types) {
    const drawn = data.nodes.filter(n => n.type === ty && keep(n))
      .sort((a, b) => b.count - a.count).slice(0, perColumn);
    byType[ty] = drawn;
    for (const n of drawn) shown.add(n.id);
    perType.push({ type: ty, shown: drawn.length, total: Math.max(drawn.length, indexTotal(data, ty, f)) });
  }
  // Un identifiant retiré qui ne désigne plus aucun noeud (la carte a été recalculée
  // depuis) n'est pas un choix en vigueur.
  const removed = data.nodes.some(n => f.hiddenNodes.has(n.id));
  const filtersActive = f.hiddenTypes.size > 0 || removed || f.minArticles > 1;
  return { types, byType, shown, filtersActive, perType };
}

/** La carte composée, dite comme le serveur la lit : les types masqués, le minimum
 *  d'articles et les concepts retirés (par `type:label`, pas par identifiant de noeud,
 *  qui n'est qu'une position dans un calcul donné). */
export function composedFilter(
  data: ConceptGraphData, f: ConceptMapFilters,
): { exclude_types: string[]; min_count: number; exclude: string[] } {
  return {
    exclude_types: [...f.hiddenTypes].sort(),
    min_count: Math.max(1, Math.floor(f.minArticles) || 1),
    exclude: data.nodes.filter(n => f.hiddenNodes.has(n.id)).map(n => `${n.type}:${n.label.en}`).sort(),
  };
}
