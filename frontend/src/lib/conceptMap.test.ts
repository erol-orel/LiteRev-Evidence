import { describe, expect, it } from "vitest";
import type { ConceptGraphData, ConceptNode, ConceptType } from "./api";
import {
  CONCEPTS_DRAWN_PER_TYPE, composeConceptMap, composedConceptCount, composedFilter, indexTotal,
} from "./conceptMap";

const ORDER: ConceptType[] = ["pathogen", "vector", "outcome", "topic"];

function node(id: number, type: ConceptType, en: string, count: number): ConceptNode {
  return { id, type, label: { en, fr: en }, count, new_count: 0, articles: [], articles_listed: 0 };
}

/** 20 outcomes (the map drew 20 of the 35 the index holds), 2 vectors, 1 pathogen. */
function map(withCounts = true): ConceptGraphData {
  const nodes: ConceptNode[] = [node(0, "pathogen", "dengue virus", 9),
    node(1, "vector", "Aedes aegypti", 6), node(2, "vector", "Aedes albopictus", 2)];
  for (let i = 0; i < 20; i++) nodes.push(node(10 + i, "outcome", `outcome ${i}`, 30 - i));
  return {
    kind: "concepts", version: 2, scenario_id: "usr-x", n_articles: 100, n_total: 100,
    n_with_concepts: 90, n_missing_concepts: 0, source: "llm", latest_year: 2025,
    types: [], nodes, edges: [], triples: [], gaps: [], articles: {},
    ...(withCounts ? {
      max_nodes: 23, min_count: 2, concepts_total: 41,
      // the index: 35 outcomes, 4 vectors, 2 pathogens
      concept_counts: {
        outcome: { "1": 10, "2": 5, ...Object.fromEntries(Array.from({ length: 20 }, (_, i) => [String(30 - i), 1])) },
        vector: { "1": 2, "2": 1, "6": 1 },
        pathogen: { "1": 1, "9": 1 },
      },
    } : {}),
  };
}

const none = { hiddenTypes: new Set<ConceptType>(), hiddenNodes: new Set<number>(), minArticles: 1 };

describe("composeConceptMap", () => {
  it("draws at most fourteen per type and says how many the index holds", () => {
    const m = composeConceptMap(map(), none, ORDER);
    expect(m.byType.outcome).toHaveLength(CONCEPTS_DRAWN_PER_TYPE);
    expect(m.byType.outcome![0].label.en).toBe("outcome 0");          // the most cited first
    expect(m.perType.find(p => p.type === "outcome")).toEqual({ type: "outcome", shown: 14, total: 35 });
    expect(m.perType.find(p => p.type === "vector")).toEqual({ type: "vector", shown: 2, total: 4 });
    expect(m.shown.size).toBe(14 + 2 + 1);
  });

  it("is not a selection: the display cut alone does not open the narrowing", () => {
    expect(composeConceptMap(map(), none, ORDER).filtersActive).toBe(false);
    expect(composeConceptMap(map(), { ...none, minArticles: 3 }, ORDER).filtersActive).toBe(true);
    expect(composeConceptMap(map(), { ...none, hiddenTypes: new Set<ConceptType>(["topic"]) }, ORDER).filtersActive).toBe(true);
    expect(composeConceptMap(map(), { ...none, hiddenNodes: new Set([1]) }, ORDER).filtersActive).toBe(true);
    // A removed id that no longer names a node (the map was recomputed) is not a choice.
    expect(composeConceptMap(map(), { ...none, hiddenNodes: new Set([999]) }, ORDER).filtersActive).toBe(false);
  });

  it("applies the minimum and the removals to the index totals too", () => {
    const f = { ...none, minArticles: 2, hiddenNodes: new Set([1]) };
    expect(indexTotal(map(), "vector", f)).toBe(1);        // 2 cited twice or more, minus Aedes aegypti
    expect(indexTotal(map(), "outcome", f)).toBe(25);
    const m = composeConceptMap(map(), f, ORDER);
    expect(m.byType.vector!.map(n => n.id)).toEqual([2]);
  });

  it("falls back on the drawn nodes for a map cached before the index counts", () => {
    const m = composeConceptMap(map(false), none, ORDER);
    expect(m.perType.find(p => p.type === "outcome")).toEqual({ type: "outcome", shown: 14, total: 20 });
  });
});

describe("composedConceptCount", () => {
  it("counts what the export will read, types the map does not draw included", () => {
    expect(composedConceptCount(map(), none)).toBe(41);
    expect(composedConceptCount(map(), { ...none, hiddenTypes: new Set<ConceptType>(["outcome"]) })).toBe(6);
    expect(composedConceptCount(map(), { ...none, minArticles: 2, hiddenNodes: new Set([1]) })).toBe(25 + 1 + 1);
    expect(composedConceptCount(map(false), none)).toBe(23);
  });
});

describe("composedFilter", () => {
  it("sends the user's filters, by label, never the drawn set", () => {
    const f = { hiddenTypes: new Set<ConceptType>(["topic", "place"]), hiddenNodes: new Set([2, 999]), minArticles: 3 };
    expect(composedFilter(map(), f)).toEqual({
      exclude_types: ["place", "topic"], min_count: 3, exclude: ["vector:Aedes albopictus"],
    });
    expect(composedFilter(map(), none)).toEqual({ exclude_types: [], min_count: 1, exclude: [] });
  });
});
