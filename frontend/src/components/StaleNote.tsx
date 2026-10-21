import { Clock } from "lucide-react";
import { useI18n } from "../i18n/LanguageProvider";

/** « Ceci décrit le corpus d'avant. »
 *
 *  Un résultat payé n'est plus effacé quand le corpus bouge : le clustering porte un
 *  résumé écrit par groupe, la carte des concepts une annotation par article, et un seul
 *  article recevant son score les détruisait tous d'un coup. Ils sont donc gardés.
 *
 *  Les garder n'est honnête qu'à cette condition : l'écran DIT qu'ils décrivent
 *  l'ensemble d'avant, avec les deux dates. Servir l'ancien corpus en le présentant comme
 *  l'actuel serait pire que de l'avoir effacé, et c'est exactement le bug pour lequel
 *  l'effacement avait été écrit. */
export function StaleNote({ builtAt, movedAt, className = "" }: {
  /** Quand ce résultat a été construit. */
  builtAt?: string | null;
  /** Quand le corpus a bougé pour la dernière fois. */
  movedAt?: string | null;
  className?: string;
}) {
  const { t, lang } = useI18n();
  const day = (iso?: string | null) => (
    iso ? new Date(iso).toLocaleDateString(lang === "fr" ? "fr-FR" : "en-US") : "?"
  );
  return (
    <div className={`rounded-xl border border-gold-400/25 bg-gold-400/[0.05] px-3 py-2 ${className}`}>
      <p className="flex items-start gap-1.5 text-[11px] leading-5 text-gold-200/90">
        <Clock size={12} className="mt-0.5 shrink-0" />
        <span>
          {t("scenarioDetail.staleNote.text")
            .replace("{built}", day(builtAt))
            .replace("{moved}", day(movedAt))}
        </span>
      </p>
      <p className="mt-0.5 pl-[18px] text-[10px] leading-4 text-white/45">
        {t("scenarioDetail.staleNote.how")}
      </p>
    </div>
  );
}
