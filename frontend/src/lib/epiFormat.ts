/** How a pooled epidemiological parameter is shown. The API returns raw floats
 *  (2.0859409266027855), which read as precision the data do not have. */

/** A number for display: a proportion as a percentage with one decimal, a duration in days
 *  with one decimal, anything else (ratios) with three significant digits. */
export function formatEpiNumber(v: number | null | undefined, unit?: string | null): string {
  if (v == null || !Number.isFinite(v)) return "-";
  const u = (unit ?? "").toLowerCase();
  if (u === "proportion") return `${(v * 100).toFixed(1).replace(/\.0$/, "")} %`;
  if (u === "days") return String(Number(v.toFixed(1)));
  const s = Number(v.toPrecision(3));
  return String(s);
}

/** "[low \u2013 high]" in the same unit as the value, or "-" when the interval is incomplete. */
export function formatEpiInterval(
  low: number | null | undefined, high: number | null | undefined, unit?: string | null,
): string {
  if (low == null || high == null || !Number.isFinite(low) || !Number.isFinite(high)) return "-";
  const u = (unit ?? "").toLowerCase();
  const fmt = (x: number) => formatEpiNumber(x, unit).replace(u === "proportion" ? " %" : "", "");
  return u === "proportion" ? `[${fmt(low)} \u2013 ${fmt(high)}] %` : `[${fmt(low)} \u2013 ${fmt(high)}]`;
}
