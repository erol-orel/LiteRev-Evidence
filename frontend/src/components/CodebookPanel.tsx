import React from "react";
import { AlertTriangle, ChevronDown, ChevronUp, Download, FileText, Upload } from "lucide-react";
import { useI18n, currentLang } from "../i18n/LanguageProvider";
import {
  addCodebookSynonym, codebookExportUrl, fetchCodebook, fetchCodebookTemplates,
  fetchUnmappedLabelsUpTo, hasApiKey, importCodebookCsv, loadCodebookTemplate,
  readCodebookAnnex, resetCodebook, writeCodebook,
} from "../lib/api";
import type {
  CodebookProposal, CodebookResponse, CodebookTemplate, UnmappedResponse,
} from "../lib/api";
import { mappedShare, nodeOptions } from "../lib/codebook";

// Lignes de la proposition montrées d'un coup. Une annexe de deux cents valeurs se lit,
// mais pas dans un panneau : au-delà, le tableau défile.
const PROPOSAL_ROWS = 60;

// Labels listed per step. The table stopped at the first 50 under "N labels still to
// map", and the endpoint at 500: past them a label could not be mapped from the screen.
const UNMAPPED_PAGE = 50;

/** The labels of the extraction against the codebook: how many rows are mapped, the labels
 *  that are not (most frequent first), and a way to map them, to load the reviewers' own
 *  hierarchy, or to go back to the default. Counted over ALL the extracted rows. */
export function CodebookPanel({ scenarioId, onChanged }: { scenarioId: string; onChanged: () => void }) {
  const { t } = useI18n();
  const T = (k: string) => t(`scenarioDetail.extraction.codebook.${k}`);
  const [open, setOpen] = React.useState(false);
  const [cb, setCb] = React.useState<CodebookResponse | null>(null);
  const [un, setUn] = React.useState<UnmappedResponse | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [msg, setMsg] = React.useState<string | null>(null);
  const [pick, setPick] = React.useState<Record<string, string>>({});
  const [loadingMore, setLoadingMore] = React.useState(false);
  const fileRef = React.useRef<HTMLInputElement>(null);
  const annexRef = React.useRef<HTMLInputElement>(null);
  // La proposition lue dans un document : elle n'est encore écrite nulle part. Tant
  // qu'elle est là, le relecteur regarde ; c'est son approbation qui écrit.
  const [proposal, setProposal] = React.useState<CodebookProposal | null>(null);
  const [reading, setReading] = React.useState(false);
  const [approving, setApproving] = React.useState(false);
  // Les vocabulaires de départ. Chargés à l'ouverture du panneau seulement : ils ne
  // dépendent pas du scénario, et une revue qui n'ouvre pas le codebook n'en a pas besoin.
  const [templates, setTemplates] = React.useState<CodebookTemplate[] | null>(null);
  const keyed = hasApiKey();
  // How many labels the table lists; a mapping reloads that many, not the first 50.
  const wanted = React.useRef(UNMAPPED_PAGE);

  const load = React.useCallback(() => (
    Promise.all([fetchCodebook(scenarioId), fetchUnmappedLabelsUpTo(scenarioId, wanted.current)])
  ), [scenarioId]);

  const showMore = async () => {
    setLoadingMore(true);
    wanted.current += UNMAPPED_PAGE;
    try { setUn(await fetchUnmappedLabelsUpTo(scenarioId, wanted.current)); }
    catch (e) { setMsg((e as Error).message); }
    finally { setLoadingMore(false); }
  };

  React.useEffect(() => {
    let alive = true;
    load()
      .then(([c, u]) => { if (alive) { setCb(c); setUn(u); } })
      .catch((e: Error) => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [load]);

  const refresh = async () => {
    const [c, u] = await load();
    setCb(c); setUn(u);
    onChanged();
  };
  const guard = async (fn: () => Promise<void>) => {
    setMsg(null);
    try { await fn(); } catch (e) { setMsg((e as Error).message); }
  };

  const map = (key: string, sheet: string, label: string | null) => guard(async () => {
    const v = pick[key];
    if (!v || !label) return;
    const [l1, l2] = v.split("|");
    await addCodebookSynonym(scenarioId, { sheet, l1, l2: l2 || null, label });
    await refresh();
  });

  const upload = (file: File | undefined) => guard(async () => {
    if (!file) return;
    const out = await importCodebookCsv(scenarioId, await file.text());
    setMsg(T("uploaded").replace("{n}", String(out.n_nodes)));
    await refresh();
  });

  React.useEffect(() => {
    if (!open || templates) return;
    fetchCodebookTemplates(currentLang()).then((t) => setTemplates(t.templates)).catch(() => {});
  }, [open, templates]);

  const loadTemplate = (key: string, name: string) => guard(async () => {
    if (!window.confirm(T("confirmTemplate").replace("{name}", name))) return;
    const out = await loadCodebookTemplate(scenarioId, key);
    setMsg(T("templateLoaded").replace("{name}", name).replace("{n}", String(out.n_nodes)));
    await refresh();
  });

  const readAnnex = (file: File | undefined) => guard(async () => {
    if (!file) return;
    setProposal(null);
    setReading(true);
    try { setProposal(await readCodebookAnnex(scenarioId, file, currentLang())); }
    finally { setReading(false); }
  });

  const approve = (mode: "add" | "replace") => guard(async () => {
    if (!proposal) return;
    if (mode === "replace" && !window.confirm(T("annexConfirmReplace"))) return;
    setApproving(true);
    try {
      const out = await writeCodebook(scenarioId, proposal.nodes, mode);
      setMsg(T(mode === "add" ? "annexAdded" : "annexReplaced")
        .replace("{added}", String(out.n_added)).replace("{merged}", String(out.n_merged))
        .replace("{n}", String(out.n_nodes)));
      setProposal(null);
      await refresh();
    } finally { setApproving(false); }
  });

  const reset = () => guard(async () => {
    if (!window.confirm(T("confirmReset"))) return;
    await resetCodebook(scenarioId);
    await refresh();
  });

  const share = un ? mappedShare(un.rows_mapped, un.rows_unmapped) : 0;
  const lang = currentLang();

  return (
    <div className="rounded-2xl border border-white/5 bg-white/2">
      <button onClick={() => setOpen((o) => !o)} aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left">
        <div>
          <h4 className="text-xs font-semibold text-white/80 uppercase tracking-wider">{T("title")}</h4>
          <p className="text-[10px] text-white/40 mt-0.5">
            {un ? T("summary").replace("{pct}", String(share)).replace("{n}", String(un.n_distinct_unmapped)) : T("subtitle")}
          </p>
        </div>
        {open ? <ChevronUp size={14} className="text-white/40" /> : <ChevronDown size={14} className="text-white/40" />}
      </button>

      {open && (
        <div className="space-y-3 border-t border-white/5 px-4 py-3">
          {error && <p className="text-[11px] text-rose-300"><AlertTriangle size={11} className="inline mr-1" />{error}</p>}
          <p className="text-[11px] text-white/45 leading-4">{T("subtitle")}</p>

          {cb && (
            <div className="flex flex-wrap items-center gap-2 text-[10px]">
              <span className="rounded-md border border-white/10 bg-white/5 px-2 py-1 text-white/60">
                {cb.source === "custom" ? T("sourceCustom").replace("{n}", String(cb.n_nodes)) : T("sourceDefault")}
              </span>
              <a href={codebookExportUrl(scenarioId)} download
                className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-2.5 py-1 text-white/70 hover:bg-white/10 transition">
                <Download size={10} />{T("download")}
              </a>
              {keyed ? (
                <>
                  <button onClick={() => fileRef.current?.click()}
                    className="flex items-center gap-1.5 rounded-lg border border-brand-500/30 bg-brand-500/10 px-2.5 py-1 text-brand-300 hover:bg-brand-500/20 transition">
                    <Upload size={10} />{T("upload")}
                  </button>
                  <input ref={fileRef} type="file" accept=".csv,text/csv" className="hidden"
                    onChange={(e) => { void upload(e.target.files?.[0]); e.target.value = ""; }} />
                  {cb.source === "custom" && (
                    <button onClick={reset} className="text-white/40 hover:text-white underline">{T("reset")}</button>
                  )}
                </>
              ) : (
                <span className="text-white/35">{T("needsKey")}</span>
              )}
            </div>
          )}
          {/* Les vocabulaires de départ. Le vocabulaire « Une seule santé » n'est plus le
              point de départ imposé : c'est l'un des quatre, et une revue sur la capacité
              hospitalière ou sur la persistance d'un virus part du sien. Charger remplace,
              et le dit avant : un vocabulaire posé sur un codebook rattaché à la main
              perdrait le rattachement. */}
          {templates && templates.length > 0 && keyed && (
            <div className="space-y-1.5 rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2">
              <p className="text-[10.5px] font-semibold text-white/70">{T("templatesTitle")}</p>
              <p className="text-[10px] leading-4 text-white/40">{T("templatesNote")}</p>
              <div className="flex flex-wrap gap-1.5">
                {templates.map((tpl) => (
                  <button key={tpl.key} onClick={() => loadTemplate(tpl.key, tpl.name)}
                    title={`${tpl.note} (${tpl.n_groups} ${T("groups")}, ${tpl.n_nodes} ${T("nodes")})`}
                    className="rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-left text-[10px] text-white/75 hover:border-brand-400/50 hover:text-white transition">
                    {tpl.name}
                    <span className="ml-1 font-mono text-[9px] text-white/35">{tpl.n_groups}</span>
                    {tpl.is_default_for_new && (
                      <span className="ml-1 font-mono text-[9px] uppercase text-brand-300/80">{T("templateDefault")}</span>
                    )}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Lire l'annexe telle qu'elle a été écrite. Un partenaire envoie un tableau
              dans un Word, un PDF ou un classeur ; le retaper est l'après-midi qui empêche
              une revue d'utiliser son propre vocabulaire. La lecture N'ÉCRIT RIEN : elle
              propose, le relecteur regarde, et c'est son approbation qui écrit. */}
          {keyed && (
            <div className="space-y-1.5 rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2">
              <p className="text-[10.5px] font-semibold text-white/70">{T("annexTitle")}</p>
              <p className="text-[10px] leading-4 text-white/40">{T("annexNote")}</p>
              <div className="flex flex-wrap items-center gap-2">
                <button onClick={() => annexRef.current?.click()} disabled={reading}
                  className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-2.5 py-1 text-[10px] text-white/75 hover:bg-white/10 transition disabled:opacity-40">
                  <FileText size={10} />{reading ? T("annexReading") : T("annexRead")}
                </button>
                <input ref={annexRef} type="file" className="hidden"
                  accept=".docx,.xlsx,.xlsm,.pdf,.csv,.tsv,.txt,.md"
                  onChange={(e) => { void readAnnex(e.target.files?.[0]); e.target.value = ""; }} />
                <span className="text-[9.5px] text-white/30">{T("annexFormats")}</span>
              </div>
            </div>
          )}

          {proposal && (
            <div className="space-y-2 rounded-xl border border-gold-400/25 bg-gold-400/[0.04] px-3 py-2.5">
              <p className="text-[10.5px] font-semibold text-gold-200/90">
                {T("annexProposalTitle").replace("{n}", String(proposal.n_nodes))
                  .replace("{groups}", String(proposal.n_groups))}
              </p>
              <p className="text-[10px] leading-4 text-white/45">
                {T("annexReadFrom").replace("{format}", proposal.format)
                  .replace("{chars}", proposal.n_chars_read.toLocaleString())}
                {proposal.truncated && ` ${T("annexTruncated")}`}
              </p>
              {proposal.model_note && (
                <p className="text-[10px] leading-4 text-white/40">
                  {T("annexSkipped")} {proposal.model_note}
                </p>
              )}
              {proposal.warnings.length > 0 && (
                <ul className="space-y-0.5">
                  {proposal.warnings.map((w, i) => (
                    <li key={i} className="text-[10px] text-rose-300/90">
                      <AlertTriangle size={10} className="inline mr-1" />{w}
                    </li>
                  ))}
                </ul>
              )}
              <div className="max-h-56 overflow-y-auto rounded-lg border border-white/5">
                <table className="w-full border-collapse text-[10px]">
                  <thead className="sticky top-0 bg-[rgb(var(--surface))]">
                    <tr className="border-b border-white/5">
                      {["colSheet", "colGroup", "annexColValue", "annexColSynonyms"].map((h) => (
                        <th key={h} className="whitespace-nowrap px-2 py-1 text-left font-semibold uppercase tracking-wider text-white/40">{T(h)}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {proposal.nodes.slice(0, PROPOSAL_ROWS).map((n, i) => (
                      <tr key={`${n.sheet}|${n.l1}|${n.l2 ?? ""}|${i}`} className="border-b border-white/5">
                        <td className="px-2 py-1 whitespace-nowrap text-white/50">{t(`scenarioDetail.extraction.sheet.${n.sheet}`)}</td>
                        <td className="px-2 py-1 text-white/60">{n.l1.replace(/_/g, " ")}</td>
                        <td className="px-2 py-1 text-white/80">
                          {n.l2 ? ((lang === "fr" ? n.label_fr : n.label_en) || n.l2.replace(/_/g, " ")) : T("annexTheVariable")}
                        </td>
                        <td className="px-2 py-1 text-white/40">{n.synonyms.join(", ") || "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {proposal.nodes.length > PROPOSAL_ROWS && (
                  <p className="border-t border-white/5 px-2 py-1 text-[9.5px] text-white/35">
                    {T("annexMoreRows").replace("{n}", String(proposal.nodes.length - PROPOSAL_ROWS))}
                  </p>
                )}
              </div>
              <details className="text-[10px] text-white/40">
                <summary className="cursor-pointer hover:text-white/60">{T("annexSeeText")}</summary>
                <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap rounded-lg bg-white/[0.03] px-2 py-1.5 text-[9.5px] leading-4 text-white/45">{proposal.text_preview}</pre>
              </details>
              <div className="flex flex-wrap items-center gap-2 pt-0.5">
                <button disabled={approving} onClick={() => void approve("add")}
                  className="rounded-lg border border-brand-500/30 bg-brand-500/10 px-2.5 py-1 text-[10px] text-brand-300 hover:bg-brand-500/20 transition disabled:opacity-40">
                  {T("annexApproveAdd").replace("{n}", String(proposal.n_nodes))}
                </button>
                <button disabled={approving} onClick={() => void approve("replace")}
                  className="rounded-lg border border-white/10 bg-white/5 px-2.5 py-1 text-[10px] text-white/70 hover:bg-white/10 transition disabled:opacity-40">
                  {T("annexApproveReplace")}
                </button>
                <button disabled={approving} onClick={() => setProposal(null)}
                  className="text-[10px] text-white/40 underline hover:text-white">{T("annexDiscard")}</button>
              </div>
              <p className="text-[9.5px] leading-4 text-white/35">{T("annexNothingStored")}</p>
            </div>
          )}

          {msg && <p className="text-[11px] text-gold-300/80">{msg}</p>}

          {un && un.unmapped.length === 0 ? (
            <p className="text-[11px] text-brand-300">{T("allMapped")}</p>
          ) : un && cb && (
            <div className="overflow-x-auto rounded-xl border border-white/5">
              <table className="w-full text-[10px] border-collapse">
                <thead>
                  <tr className="border-b border-white/5 bg-white/3">
                    {["colSheet", "colGroup", "colLabel", "colRows", "colArticles", "colMapTo"].map((h) => (
                      <th key={h} className="text-left px-2.5 py-1.5 text-white/40 font-semibold uppercase tracking-wider whitespace-nowrap">{T(h)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {un.unmapped.map((r, i) => {
                    const key = `${r.sheet}|${r.group}|${r.covariate}|${i}`;
                    const opts = nodeOptions(cb.nodes, r.sheet, lang, r.l1);
                    return (
                      <tr key={key} className="border-b border-white/5 align-middle">
                        <td className="px-2.5 py-1.5 text-white/50 whitespace-nowrap">{t(`scenarioDetail.extraction.sheet.${r.sheet}`)}</td>
                        <td className="px-2.5 py-1.5 text-white/55">{r.group || "-"}</td>
                        <td className="px-2.5 py-1.5 text-white/80">{r.covariate || "-"}</td>
                        <td className="px-2.5 py-1.5 font-mono text-white/60">{r.n_rows}</td>
                        <td className="px-2.5 py-1.5 font-mono text-white/60">{r.n_articles}</td>
                        <td className="px-2.5 py-1.5">
                          {keyed ? (
                            <div className="flex items-center gap-1.5">
                              <select value={pick[key] ?? ""} onChange={(e) => setPick((p) => ({ ...p, [key]: e.target.value }))}
                                className="max-w-[180px] rounded-md border border-white/10 bg-white/5 px-1.5 py-1 text-[10px] text-white/70 focus:outline-none">
                                <option value="">{T("mapPlaceholder")}</option>
                                {opts.map((o) => <option key={o.value} value={o.value}>{o.l1.replace(/_/g, " ")} &gt; {o.label}</option>)}
                              </select>
                              <button disabled={!pick[key]} onClick={() => void map(key, r.sheet, r.covariate)}
                                className="rounded-md border border-brand-500/30 bg-brand-500/10 px-2 py-1 text-brand-300 hover:bg-brand-500/20 transition disabled:opacity-40">
                                {T("map")}
                              </button>
                            </div>
                          ) : <span className="text-white/25">-</span>}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {un.n_distinct_unmapped > un.unmapped.length && (
                <div className="flex items-center justify-between gap-2 border-t border-white/5 px-2.5 py-1.5 text-[10px] text-white/45">
                  <span>{T("shownOf").replace("{shown}", String(un.unmapped.length)).replace("{total}", String(un.n_distinct_unmapped))}</span>
                  <button disabled={loadingMore} onClick={() => void showMore()}
                    className="text-brand-300 underline hover:text-brand-200 disabled:opacity-40">
                    {t("lists.loadMore").replace("{n}", String(Math.min(UNMAPPED_PAGE, un.n_distinct_unmapped - un.unmapped.length)))}
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
