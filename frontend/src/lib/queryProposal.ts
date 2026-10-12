import type { SearchStrategy } from "./api";

/** The fields of a pending proposal: never kept once the user has chosen. */
const PROPOSAL_KEYS = ["proposed_query", "proposal_reason", "proposal_reasons", "proposal_issues"] as const;

function withoutProposal(s: SearchStrategy): SearchStrategy {
  const out: SearchStrategy = { ...s };
  for (const k of PROPOSAL_KEYS) delete out[k];
  return out;
}

/** The strategy used once the user ACCEPTS the proposed correction: the proposal becomes the
 *  query, for PubMed as for every other source. `entered_query` keeps what was typed. */
export function acceptProposal(s: SearchStrategy): SearchStrategy {
  const proposed = s.proposed_query ?? s.general;
  return { ...withoutProposal(s), general: proposed, pubmed: proposed };
}

/** The strategy used when the user keeps the query as entered: unchanged, nothing pending. */
export function keepEnteredQuery(s: SearchStrategy): SearchStrategy {
  return withoutProposal(s);
}

/** Where two versions of a query differ: the common start and end, and what lies between
 *  in each. A fix in a strategy of several thousand characters is often one character; the
 *  screen highlights it instead of asking the user to compare two walls of text. */
export function changedSpan(before: string, after: string): {
  prefix: string; removed: string; added: string; suffix: string;
} {
  const max = Math.min(before.length, after.length);
  let p = 0;
  while (p < max && before[p] === after[p]) p++;
  let s = 0;
  while (s < max - p && before[before.length - 1 - s] === after[after.length - 1 - s]) s++;
  return {
    prefix: before.slice(0, p),
    removed: before.slice(p, before.length - s),
    added: after.slice(p, after.length - s),
    suffix: before.slice(before.length - s),
  };
}
