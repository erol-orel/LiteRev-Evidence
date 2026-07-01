import React, {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";
import { fr, type Translations } from "./locales/fr";
import { en } from "./locales/en";

export type Lang = "fr" | "en";

const RESOURCES: Record<Lang, Translations> = { fr, en };
const STORAGE_KEY = "literev-lang";

function detectInitial(): Lang {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "fr" || saved === "en") return saved;
    const nav = (navigator.language || "fr").toLowerCase();
    return nav.startsWith("en") ? "en" : "fr";
  } catch {
    return "fr";
  }
}

// Dotted-path lookup, e.g. lookup(fr, "nav.search").
function lookup(res: Translations, path: string): unknown {
  return path
    .split(".")
    .reduce<unknown>(
      (o, k) =>
        o && typeof o === "object" ? (o as Record<string, unknown>)[k] : undefined,
      res,
    );
}

type I18nContextValue = {
  lang: Lang;
  setLang: (l: Lang) => void;
  /** Translate a dotted key. Falls back to French, then the key itself. */
  t: (path: string) => string;
};

const I18nContext = createContext<I18nContextValue | null>(null);

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLangState] = useState<Lang>(detectInitial);

  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    try {
      localStorage.setItem(STORAGE_KEY, l);
    } catch {
      /* ignore persistence errors (private mode, etc.) */
    }
    try {
      document.documentElement.lang = l;
    } catch {
      /* ignore */
    }
  }, []);

  const t = useCallback(
    (path: string) => {
      const val = lookup(RESOURCES[lang], path) ?? lookup(fr, path);
      return typeof val === "string" ? val : path;
    },
    [lang],
  );

  const value = useMemo<I18nContextValue>(
    () => ({ lang, setLang, t }),
    [lang, setLang, t],
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

/**
 * Standalone translator usable OUTSIDE React (e.g. in lib/api.ts or class
 * components). Reads the persisted language directly from localStorage and
 * falls back to French, then the key itself — mirroring the hook's `t`.
 */
export function tStandalone(path: string): string {
  let lang: Lang = "fr";
  try {
    const s = localStorage.getItem(STORAGE_KEY);
    if (s === "fr" || s === "en") lang = s;
  } catch {
    /* ignore */
  }
  const val = lookup(RESOURCES[lang], path) ?? lookup(fr, path);
  return typeof val === "string" ? val : path;
}

export function useI18n(): I18nContextValue {
  const ctx = useContext(I18nContext);
  if (!ctx) {
    throw new Error("useI18n must be used within a LanguageProvider");
  }
  return ctx;
}

/** Convenience hook when only the translate function is needed. */
export function useT(): (path: string) => string {
  return useI18n().t;
}
