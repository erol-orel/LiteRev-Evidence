import React from "react";
import { AlertTriangle, X } from "lucide-react";
import { useI18n, currentLang } from "../i18n/LanguageProvider";
import { fetchVariablesCrosstab, variablesCrosstabExportUrl } from "../lib/api";
import type { VoiCrosstab } from "../lib/api";
import { ExportButton } from "./ExportButton";

/** Deux variables du registre croisées : qui rapporte quoi, et où.
 *
 *  Un total d'axe compte des ARTICLES, pas des cases : un article qui rapporte deux
 *  professions et un pays compte une fois pour ce pays. La somme des cases est donc plus
 *  grande que le total de l'axe, et c'est normal ; ce qui ne le serait pas, c'est qu'un
 *  nombre d'ici contredise la barre qui montre la même chose ailleurs.
 *
 *  Une case vide ne veut dire quelque chose que DANS une population connue : la grille dit
 *  donc combien d'articles les deux variables savent juger, et combien une seule sait. */
export function CrosstabPanel({ scenarioId, rows, cols, nameOf, onClear }: {
  scenarioId: string;
  rows: string;
  cols: string;
  nameOf: (key: string) => string;
  onClear: () => void;
}) {
  const { t, lang } = useI18n();
  const T = (k: string) => t(`scenarioDetail.register.cross.${k}`);
  const [grid, setGrid] = React.useState<VoiCrosstab | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let alive = true;
    setGrid(null);
    setError(null);
    fetchVariablesCrosstab(scenarioId, rows, cols, currentLang())
      .then((g) => { if (alive) setGrid(g); })
      .catch((e: Error) => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [scenarioId, rows, cols, lang]);

  const head = (
    <div className="flex items-start justify-between gap-3">
      <div>
        <h4 className="text-[12px] font-semibold text-white/85">
          {T("title").replace("{rows}", nameOf(rows)).replace("{cols}", nameOf(cols))}
        </h4>
        {grid?.complete && (
          <p className="mt-0.5 text-[10px] leading-4 text-white/45">
            {T("population").replace("{n}", grid.n_both.toLocaleString())}
            {grid.n_rows_only + grid.n_cols_only > 0 && " " + T("oneSided")
              .replace("{rows}", grid.n_rows_only.toLocaleString())
              .replace("{cols}", grid.n_cols_only.toLocaleString())}
            {(grid.rows_total > grid.rows.length || grid.cols_total > grid.cols.length) &&
              " " + T("truncated").replace("{rows}", String(grid.rows_total))
                .replace("{cols}", String(grid.cols_total))}
          </p>
        )}
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {/* La grille est la figure qu'un partenaire recopie dans son rapport : la recopier
            à la main est précisément le travail que cet outil doit lui épargner. */}
        <ExportButton href={variablesCrosstabExportUrl(scenarioId, rows, cols, currentLang())}
          label={T("exportCsv")} title={T("exportHint")} disabled={!grid?.complete} />
        <button onClick={onClear} aria-label={T("clear")}
          className="text-white/35 hover:text-white/80"><X size={12} /></button>
      </div>
    </div>
  );

  if (error) {
    return (
      <div className="rounded-2xl border border-white/5 bg-white/[0.02] px-3 py-2.5 space-y-1">
        {head}
        <p className="text-[10.5px] text-rose-300">
          <AlertTriangle size={10} className="inline mr-1" />{error}
        </p>
      </div>
    );
  }
  if (!grid) {
    return (
      <div className="rounded-2xl border border-white/5 bg-white/[0.02] px-3 py-2.5 space-y-1">
        {head}
        <p className="text-[10.5px] text-white/40">{t("common.loadingEllipsis")}</p>
      </div>
    );
  }
  if (!grid.complete) {
    return (
      <div className="rounded-2xl border border-white/5 bg-white/[0.02] px-3 py-2.5 space-y-1">
        {head}
        <p className="text-[10.5px] text-gold-300/90">{grid.note || grid.error || T("empty")}</p>
      </div>
    );
  }

  const at = (r: string, c: string) => grid.cells.find((x) => x.row === r && x.col === c)?.n ?? 0;
  const max = Math.max(1, ...grid.cells.map((c) => c.n));

  return (
    <div className="rounded-2xl border border-white/5 bg-white/[0.02] px-3 py-2.5 space-y-2">
      {head}
      {/* Une grille large défile dans son propre cadre : la page, elle, ne défile jamais
          de côté. Le fond d'une case suit son compte, d'une seule teinte, du clair au
          foncé : une seule grandeur, donc une seule couleur. */}
      <div className="overflow-x-auto">
        <table className="border-collapse text-[10.5px]">
          <thead>
            <tr>
              <th className="sticky left-0 z-10 bg-[rgb(var(--surface))] px-2 py-1 text-left font-normal text-white/35">
                {nameOf(rows)}
              </th>
              {grid.cols.map((c) => (
                <th key={c.value} title={`${c.label} (${c.n})`}
                  className="max-w-[88px] px-1.5 py-1 text-left align-bottom font-normal text-white/55">
                  <span className="block truncate">{c.label}</span>
                  <span className="font-mono text-[9px] text-white/30">{c.n}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {grid.rows.map((r) => (
              <tr key={r.value}>
                <th scope="row" title={`${r.label} (${r.n})`}
                  className="sticky left-0 z-10 max-w-[150px] bg-[rgb(var(--surface))] px-2 py-1 text-left font-normal">
                  <span className="block truncate text-white/80">{r.label}</span>
                  <span className="font-mono text-[9px] text-white/30">{r.n}</span>
                </th>
                {grid.cols.map((c) => {
                  const n = at(r.value, c.value);
                  return (
                    <td key={c.value}
                      className={`px-1.5 py-1 text-center font-mono tabular-nums ${n ? "text-white/90" : "text-white/20"}`}
                      style={n ? { backgroundColor: `rgb(var(--c-brand-500) / ${0.12 + 0.55 * (n / max)})` } : undefined}>
                      {n || "."}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[10px] leading-4 text-white/40">
        {T("readAs").replace("{observed}", String(grid.pairs_observed))
          .replace("{shown}", String(grid.cells_shown))}
      </p>
    </div>
  );
}
