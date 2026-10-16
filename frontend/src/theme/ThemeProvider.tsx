import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

/** Le choix de l'utilisateur. « system » suit le réglage du système d'exploitation. */
export type ThemeChoice = "system" | "dark" | "light";
/** Le thème réellement appliqué, une fois « system » résolu. */
export type ResolvedTheme = "dark" | "light";

const STORAGE_KEY = "literev-theme";
const QUERY = "(prefers-color-scheme: light)";

/**
 * Le choix persisté, lisible HORS de React (comme `currentLang`), pour que le thème
 * puisse être posé sur `<html>` avant le premier rendu.
 */
export function currentThemeChoice(): ThemeChoice {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "dark" || saved === "light" || saved === "system") return saved;
  } catch {
    /* mode privé, stockage refusé : on retombe sur le système */
  }
  return "system";
}

/** Ce que le système demande. Sombre par défaut : c'est l'interface d'origine. */
export function systemTheme(): ResolvedTheme {
  try {
    return window.matchMedia?.(QUERY).matches ? "light" : "dark";
  } catch {
    return "dark";
  }
}

export function resolveTheme(choice: ThemeChoice): ResolvedTheme {
  return choice === "system" ? systemTheme() : choice;
}

/**
 * Pose le thème sur l'élément racine. `data-theme` est l'interrupteur que lit
 * `theme.css` : sans attribut, ce sont les couleurs sombres, celles d'avant le mode clair.
 * L'attribut est écrit dans les deux cas, pour qu'une capture d'écran ou un test puisse
 * lire ce qui est appliqué au lieu de le déduire d'une absence.
 */
export function applyTheme(resolved: ResolvedTheme): void {
  try {
    const root = document.documentElement;
    root.setAttribute("data-theme", resolved);
    // La couleur de la barre du navigateur sur mobile suit le fond de page.
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", resolved === "light" ? "#f7faf8" : "#0A3621");
  } catch {
    /* pas de DOM (tests hors navigateur) : rien à poser */
  }
}

type ThemeContextValue = {
  /** Ce que l'utilisateur a choisi, « system » compris. */
  choice: ThemeChoice;
  /** Le thème appliqué, « system » déjà résolu : à lire pour dessiner un graphique. */
  theme: ResolvedTheme;
  setChoice: (c: ThemeChoice) => void;
  /** Passe de l'un à l'autre, en partant de ce qui est affiché. */
  toggle: () => void;
};

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [choice, setChoiceState] = useState<ThemeChoice>(currentThemeChoice);
  const [resolved, setResolved] = useState<ResolvedTheme>(() => resolveTheme(currentThemeChoice()));

  // Un seul endroit applique le thème, pour que l'attribut ne puisse pas mentir.
  useEffect(() => {
    applyTheme(resolved);
  }, [resolved]);

  // Tant que le choix est « system », un changement de réglage du système est suivi en
  // direct : l'utilisateur n'a pas à recharger la page à la tombée de la nuit.
  useEffect(() => {
    if (choice !== "system") return;
    let mql: MediaQueryList;
    try {
      mql = window.matchMedia(QUERY);
    } catch {
      return;
    }
    const onChange = () => setResolved(systemTheme());
    onChange();
    mql.addEventListener?.("change", onChange);
    return () => mql.removeEventListener?.("change", onChange);
  }, [choice]);

  const setChoice = useCallback((c: ThemeChoice) => {
    setChoiceState(c);
    setResolved(resolveTheme(c));
    try {
      localStorage.setItem(STORAGE_KEY, c);
    } catch {
      /* le thème s'applique quand même, il ne survivra pas au rechargement */
    }
  }, []);

  // Le bouton part de ce qui est AFFICHÉ, pas du choix : depuis « system » résolu en
  // sombre, un clic donne clair, ce que l'utilisateur attend.
  const toggle = useCallback(() => {
    setChoice(resolved === "dark" ? "light" : "dark");
  }, [resolved, setChoice]);

  const value = useMemo<ThemeContextValue>(
    () => ({ choice, theme: resolved, setChoice, toggle }),
    [choice, resolved, setChoice, toggle],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

/**
 * Le thème appliqué. Hors d'un ThemeProvider, répond « dark » au lieu de lever : un
 * graphique isolé dans un test doit pouvoir se dessiner.
 */
export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (ctx) return ctx;
  return {
    choice: "system",
    theme: "dark",
    setChoice: () => {},
    toggle: () => {},
  };
}

/** Pratique quand seul le thème appliqué compte (couleurs d'un graphique). */
export function useResolvedTheme(): ResolvedTheme {
  return useTheme().theme;
}

/**
 * Les couleurs que le SVG ne peut pas lire dans une variable CSS : un attribut de
 * présentation (`stroke="..."`) n'accepte pas `var()`, donc les graphiques choisissent
 * leur valeur en JavaScript, à partir du thème appliqué.
 *
 * `ink` est le liseré d'une sélection, `edge` un lien entre deux noeuds, `noise` le groupe
 * des restes. Les couleurs des séries, elles, ne changent pas : ce sont des nuances
 * moyennes, lisibles sur fond sombre comme sur fond clair, et les changer casserait la
 * correspondance avec les légendes déjà publiées.
 */
export function chartColors(theme: ResolvedTheme) {
  return theme === "light"
    ? { ink: "#0b1a12", edge: "#2d7a52", noise: "#64748b" }
    : { ink: "#ffffff", edge: "#9be7c4", noise: "#94a3b8" };
}
