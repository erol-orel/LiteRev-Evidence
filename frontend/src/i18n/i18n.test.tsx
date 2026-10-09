import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { LanguageProvider, currentLang, tStandalone, useI18n } from "./LanguageProvider";
import { en } from "./locales/en";
import { fr } from "./locales/fr";

const STORAGE_KEY = "literev-lang";

/** Flatten a nested locale object into "a.b.c" → value. */
function flatten(obj: unknown, prefix = "", out: Record<string, unknown> = {}): Record<string, unknown> {
  if (obj && typeof obj === "object" && !Array.isArray(obj)) {
    for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
      flatten(v, prefix ? `${prefix}.${k}` : k, out);
    }
  } else {
    out[prefix] = obj;
  }
  return out;
}

const placeholders = (s: string) => [...s.matchAll(/\{[a-zA-Z0-9_]+\}/g)].map((m) => m[0]).sort();

describe("locale files", () => {
  const flatFr = flatten(fr);
  const flatEn = flatten(en);

  it("carry the same keys in French and in English", () => {
    expect(Object.keys(flatEn).sort()).toEqual(Object.keys(flatFr).sort());
  });

  it("have a non-empty string for every key", () => {
    for (const [lang, flat] of [["fr", flatFr], ["en", flatEn]] as const) {
      const bad = Object.entries(flat).filter(([, v]) => typeof v !== "string" || !v.trim());
      expect(bad, `${lang}: empty or non-string values`).toEqual([]);
    }
  });

  it("use the same {placeholders} in both languages", () => {
    const mismatched = Object.keys(flatFr).filter(
      (k) => placeholders(String(flatFr[k])).join(",") !== placeholders(String(flatEn[k] ?? "")).join(","),
    );
    expect(mismatched).toEqual([]);
  });
});

describe("currentLang", () => {
  it("prefers the persisted choice", () => {
    localStorage.setItem(STORAGE_KEY, "fr");
    expect(currentLang()).toBe("fr");
    localStorage.setItem(STORAGE_KEY, "en");
    expect(currentLang()).toBe("en");
  });

  it("falls back to the browser language, then French", () => {
    localStorage.setItem(STORAGE_KEY, "de");                     // not a supported value
    const lang = vi.spyOn(navigator, "language", "get");
    lang.mockReturnValue("en-GB");
    expect(currentLang()).toBe("en");
    lang.mockReturnValue("fr-CH");
    expect(currentLang()).toBe("fr");
    lang.mockReturnValue("de-DE");
    expect(currentLang()).toBe("fr");
  });

  it("returns French when storage is not accessible", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("storage blocked");
    });
    expect(currentLang()).toBe("fr");
  });
});

describe("tStandalone", () => {
  it("translates in the persisted language and falls back to the key", () => {
    localStorage.setItem(STORAGE_KEY, "en");
    expect(tStandalone("nav.search")).toBe(en.nav.search);
    localStorage.setItem(STORAGE_KEY, "fr");
    expect(tStandalone("nav.search")).toBe(fr.nav.search);
    expect(tStandalone("does.not.exist")).toBe("does.not.exist");
    expect(tStandalone("nav")).toBe("nav");                       // a branch, not a string
  });
});

function Consumer() {
  const { lang, setLang, t } = useI18n();
  return (
    <div>
      <span data-testid="lang">{lang}</span>
      <span data-testid="label">{t("nav.search")}</span>
      <button onClick={() => setLang("en")}>en</button>
      <button onClick={() => setLang("fr")}>fr</button>
    </div>
  );
}

describe("LanguageProvider", () => {
  it("switches the strings, persists the choice and tags the document", () => {
    localStorage.setItem(STORAGE_KEY, "fr");
    render(
      <LanguageProvider>
        <Consumer />
      </LanguageProvider>,
    );
    expect(screen.getByTestId("lang")).toHaveTextContent("fr");
    expect(screen.getByTestId("label")).toHaveTextContent(fr.nav.search);

    fireEvent.click(screen.getByText("en"));
    expect(screen.getByTestId("lang")).toHaveTextContent("en");
    expect(screen.getByTestId("label")).toHaveTextContent(en.nav.search);
    expect(localStorage.getItem(STORAGE_KEY)).toBe("en");
    expect(document.documentElement.lang).toBe("en");

    fireEvent.click(screen.getByText("fr"));
    expect(screen.getByTestId("label")).toHaveTextContent(fr.nav.search);
    expect(localStorage.getItem(STORAGE_KEY)).toBe("fr");
  });

  it("refuses to be used outside the provider", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => render(<Consumer />)).toThrow(/LanguageProvider/);
  });
});

/** Every literal key a component asks for must exist. `t()` falls back to French and
 *  then to the PATH ITSELF, so a key that exists nowhere renders as the literal string
 *  "common.loading" on the page, silently. Three call sites did exactly that.
 *
 *  The sources are read through Vite's `import.meta.glob` rather than node:fs, so this
 *  file type-checks under the app's own tsconfig (which has no @types/node). */
const SOURCES = import.meta.glob("../**/*.{ts,tsx}", {
  query: "?raw", import: "default", eager: true,
}) as Record<string, string>;

describe("every translation key a component asks for exists", () => {
  const flatFr = flatten(fr);

  it("has no t(\"a.b.c\") pointing at a key that is not in the locales", () => {
    const missing: string[] = [];
    for (const [file, src] of Object.entries(SOURCES)) {
      if (file.includes(".test.")) continue;
      // `t("a.b.c")` with a LITERAL key, which is a FULL locale path. `T(...)` is
      // excluded on purpose: several components define it as a prefixed helper whose
      // argument is only the tail of a path. Template literals and computed keys are
      // out of reach of a static check and are left alone.
      for (const m of src.matchAll(/(?<![a-zA-Z0-9_.$])t\(\s*"([a-zA-Z0-9_.]+)"\s*\)/g)) {
        const key = m[1];
        if (!key.includes(".")) continue;
        if (key in flatFr) continue;
        missing.push(`${file}: ${key}`);
      }
    }
    expect(missing).toEqual([]);
  });
});
