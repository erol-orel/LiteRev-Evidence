import type { CodebookNode } from "./api";

export interface NodeOption { value: string; l1: string; l2: string; label: string }

/** The values a label of this sheet can be mapped to: the level-2 nodes, grouped by level 1,
 *  those of `preferL1` first (a label already filed under a known group most likely belongs
 *  there). `value` is "l1|l2". */
export function nodeOptions(nodes: CodebookNode[], sheet: string, lang: string, preferL1?: string | null): NodeOption[] {
  const fr = lang.toLowerCase().startsWith("fr");
  const out = nodes
    .filter((n) => n.sheet === sheet && n.l2)
    .map((n) => ({
      value: `${n.l1}|${n.l2}`, l1: n.l1, l2: n.l2 as string,
      label: (fr ? n.label_fr : n.label_en) || (n.l2 as string).replace(/_/g, " "),
    }));
  return out.sort((a, b) => {
    const pa = a.l1 === preferL1 ? 0 : 1;
    const pb = b.l1 === preferL1 ? 0 : 1;
    return pa - pb || a.l1.localeCompare(b.l1) || a.label.localeCompare(b.label);
  });
}

/** The share of extracted rows that carry a codebook label, as a whole percentage. */
export function mappedShare(mapped: number, unmapped: number): number {
  const total = mapped + unmapped;
  return total > 0 ? Math.round((mapped / total) * 100) : 0;
}
