import React from "react";
import { Download, Loader2 } from "lucide-react";

/** Le bouton d'export, LE MÊME PARTOUT.
 *
 *  Chaque tableau avait le sien, ou n'en avait pas : un lien gris en haut à droite ici,
 *  rien là, un bouton ailleurs. Un relecteur a donc cherché l'export du registre et
 *  conclu qu'il n'existait pas, alors qu'il était à deux centimètres. Une icône de
 *  téléchargement, un libellé, la même place au-dessus du tableau qu'il exporte : si
 *  quelque chose se lit à l'écran, le fichier correspondant se prend ici.
 *
 *  Deux formes, parce que les deux existent côté serveur : `href` pour un export qui
 *  tient dans une URL (un GET), `onClick` pour celui dont les conditions partent dans un
 *  corps (un POST, donc un blob). Le libellé et la place ne changent pas pour autant. */
export function ExportButton({ href, onClick, label, title, disabled, className = "" }: {
  /** Un GET : le navigateur télécharge lui-même. */
  href?: string;
  /** Un POST : la fonction télécharge, et le bouton montre son attente. */
  onClick?: () => Promise<void> | void;
  label: string;
  title?: string;
  disabled?: boolean;
  className?: string;
}) {
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const base = "inline-flex shrink-0 items-center gap-1.5 rounded-lg border px-2.5 py-1"
    + " text-[11px] transition-colors";
  const live = "border-brand-400/30 bg-brand-400/[0.07] text-brand-200"
    + " hover:border-brand-400/60 hover:bg-brand-400/15";
  const off = "border-white/10 text-white/30 cursor-not-allowed";

  const run = async () => {
    if (!onClick || busy) return;
    setBusy(true);
    setError(null);
    try {
      await onClick();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const inner = (
    <>
      {busy ? <Loader2 size={12} className="animate-spin" /> : <Download size={12} />}
      {label}
    </>
  );

  if (disabled) {
    return <span title={title} className={`${base} ${off} ${className}`}>{inner}</span>;
  }
  if (href) {
    return (
      <a href={href} title={title} className={`${base} ${live} ${className}`}>{inner}</a>
    );
  }
  return (
    <span className="inline-flex flex-col items-end gap-0.5">
      <button type="button" onClick={run} disabled={busy} title={title}
        className={`${base} ${live} ${className}`}>{inner}</button>
      {/* L'échec se dit là où on a cliqué : un fichier qui n'arrive pas sans un mot est
          indistinguable d'un bouton mort. */}
      {error && <span className="max-w-[240px] text-right text-[10px] text-rose-300">{error}</span>}
    </span>
  );
}
