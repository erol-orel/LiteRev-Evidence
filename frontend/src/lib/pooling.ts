import type { PooledComparison, PooledGroup } from "./api";

/** A percentage with one decimal, trailing zero dropped: 0.2236 -> "22.4%". */
export function pct(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return "-";
  return `${(p * 100).toFixed(1).replace(/\.0$/, "")}%`;
}

/** An odds ratio with two decimals, or three significant digits when it is large. */
export function fmtOr(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "-";
  return v >= 100 ? String(Math.round(v)) : String(Number(v.toPrecision(3)));
}

export function intervalText(lo: number | null | undefined, hi: number | null | undefined, f: (n: number | null | undefined) => string): string {
  return lo == null || hi == null ? "-" : `${f(lo)} to ${f(hi)}`;
}

/** The horizontal scale of a forest plot: value -> x in [0, width]. A log scale for odds ratios. */
export function forestScale(lo: number, hi: number, width: number, log: boolean): (v: number) => number {
  const a = log ? Math.log(lo) : lo;
  const b = log ? Math.log(hi) : hi;
  const span = b - a || 1;
  return (v: number) => {
    const t = ((log ? Math.log(Math.max(v, 1e-9)) : v) - a) / span;
    return Math.max(0, Math.min(width, t * width));
  };
}

/** The domain of a proportion plot: 0 up to a rounded-up maximum of everything drawn, never above 1. */
export function proportionDomain(g: PooledGroup): [number, number] {
  const top = Math.max(...g.studies.map((s) => s.ci_high ?? s.p ?? 0), g.pooled?.pi_high ?? 0, g.pooled?.ci_high ?? 0, 0.05);
  return [0, Math.min(1, Math.ceil(top * 10) / 10)];
}

/** The domain of an odds-ratio plot (log scale), with room on both sides and the null value 1 inside. */
export function orDomain(c: PooledComparison): [number, number] {
  const los = [...c.studies.map((s) => s.ci_low ?? 1), c.pooled.pi_low ?? c.pooled.ci_low, 1];
  const his = [...c.studies.map((s) => s.ci_high ?? 1), c.pooled.pi_high ?? c.pooled.ci_high, 1];
  return [Math.max(Math.min(...los) / 1.15, 1e-3), Math.min(Math.max(...his) * 1.15, 1e3)];
}

/** What a reader should be told next to I2, in the order of how worried to be. */
export function bandTone(band: string): string {
  return band === "low" ? "text-brand-300" : band === "moderate" ? "text-sky-300"
    : band === "substantial" ? "text-gold-400" : "text-rose-300";
}

export function csvOfPooled(groups: PooledGroup[]): string {
  const q = (s: string | null) => `"${(s ?? "").replace(/"/g, '""')}"`;
  const head = ["sheet", "group", "label", "disease", "k", "n_total", "events_total", "pooled_p", "ci_low", "ci_high",
    "pi_low", "pi_high", "I2", "tau2", "heterogeneity"];
  const rows = groups.map((g) => [
    g.sheet, g.group, g.label, g.disease ?? "", g.k, g.n_total, g.events_total,
    g.pooled?.p ?? "", g.pooled?.ci_low ?? "", g.pooled?.ci_high ?? "", g.pooled?.pi_low ?? "", g.pooled?.pi_high ?? "",
    g.heterogeneity?.I2 ?? "", g.heterogeneity?.tau2 ?? "", g.heterogeneity?.band ?? "",
  ].map((v) => (typeof v === "string" && v !== "" ? q(v) : v)).join(","));
  return [head.join(","), ...rows].join("\n");
}
