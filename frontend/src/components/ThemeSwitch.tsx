import { Monitor, Moon, Sun } from "lucide-react";
import { useI18n } from "../i18n/LanguageProvider";
import { useTheme, type ThemeChoice } from "../theme/ThemeProvider";

/**
 * Le choix du thème, dessiné comme le sélecteur de langue juste à côté : trois boutons,
 * « système » compris, parce que le réglage par défaut suit le système et qu'un utilisateur
 * qui a cliqué doit pouvoir y revenir.
 *
 * Le bouton marqué est celui du CHOIX, pas du thème résolu : sur « système », c'est
 * l'icône système qui est allumée, et l'infobulle dit ce qui est appliqué.
 */
export function ThemeSwitch() {
  const { t } = useI18n();
  const { choice, theme, setChoice } = useTheme();

  const items: Array<{ key: ThemeChoice; icon: typeof Sun; label: string }> = [
    { key: "system", icon: Monitor, label: t("header.themeSystem") },
    { key: "light", icon: Sun, label: t("header.themeLight") },
    { key: "dark", icon: Moon, label: t("header.themeDark") },
  ];

  return (
    <div
      className="flex items-center gap-1 rounded-xl border border-white/10 bg-white/5 p-0.5 text-xs"
      title={`${t("header.themeLabel")} · ${theme === "light" ? t("header.themeLight") : t("header.themeDark")}`}
      role="group"
      aria-label={t("header.themeLabel")}
      data-testid="theme-switch"
    >
      {items.map(({ key, icon: Icon, label }) => (
        <button
          key={key}
          type="button"
          onClick={() => setChoice(key)}
          aria-pressed={choice === key}
          aria-label={label}
          title={label}
          data-theme-choice={key}
          className={`rounded-lg px-2 py-1 transition ${
            choice === key
              ? "bg-pill text-onpill"
              : "text-white/50 hover:text-white"
          }`}
        >
          <Icon size={13} />
        </button>
      ))}
    </div>
  );
}
