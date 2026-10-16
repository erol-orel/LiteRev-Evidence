import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ThemeProvider, chartColors, currentThemeChoice, resolveTheme, useTheme } from "./ThemeProvider";
import { ThemeSwitch } from "../components/ThemeSwitch";
import { LanguageProvider } from "../i18n/LanguageProvider";

/** Le réglage du système, que le composant lit avec matchMedia. */
function systemPrefers(light: boolean) {
  vi.stubGlobal("matchMedia", (q: string) => ({
    matches: light && q.includes("light"),
    media: q,
    addEventListener: () => {},
    removeEventListener: () => {},
  }));
}

function Probe() {
  const { choice, theme, toggle } = useTheme();
  return (
    <div>
      <span data-testid="choice">{choice}</span>
      <span data-testid="theme">{theme}</span>
      <button type="button" onClick={toggle}>flip</button>
    </div>
  );
}

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("the theme the app starts in", () => {
  it("follows the system, and the system is what decides", () => {
    systemPrefers(true);
    render(<ThemeProvider><Probe /></ThemeProvider>);
    expect(screen.getByTestId("choice").textContent).toBe("system");
    expect(screen.getByTestId("theme").textContent).toBe("light");
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
  });

  it("is dark when the system says nothing, because dark is the interface as it was", () => {
    systemPrefers(false);
    render(<ThemeProvider><Probe /></ThemeProvider>);
    expect(screen.getByTestId("theme").textContent).toBe("dark");
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
  });

  it("obeys a stored choice over the system", () => {
    systemPrefers(true);
    localStorage.setItem("literev-theme", "dark");
    render(<ThemeProvider><Probe /></ThemeProvider>);
    expect(screen.getByTestId("theme").textContent).toBe("dark");
  });

  it("ignores a stored value it does not understand", () => {
    localStorage.setItem("literev-theme", "sepia");
    expect(currentThemeChoice()).toBe("system");
  });

  it("still reads a choice when the storage throws, as in a private window", () => {
    vi.stubGlobal("localStorage", {
      getItem: () => { throw new Error("denied"); },
      setItem: () => { throw new Error("denied"); },
    });
    expect(currentThemeChoice()).toBe("system");
  });
});

describe("changing the theme", () => {
  it("keeps the choice for the next visit", () => {
    systemPrefers(false);
    render(<ThemeProvider><Probe /></ThemeProvider>);
    fireEvent.click(screen.getByText("flip"));
    expect(screen.getByTestId("theme").textContent).toBe("light");
    expect(localStorage.getItem("literev-theme")).toBe("light");
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
  });

  it("flips from what is SHOWN, not from the word system", () => {
    // Sur « système » résolu en clair, un clic doit donner sombre : partir du choix
    // donnerait « clair », c'est-à-dire aucun changement visible.
    systemPrefers(true);
    render(<ThemeProvider><Probe /></ThemeProvider>);
    expect(screen.getByTestId("theme").textContent).toBe("light");
    fireEvent.click(screen.getByText("flip"));
    expect(screen.getByTestId("theme").textContent).toBe("dark");
  });

  it("goes back to following the system from the switch", () => {
    systemPrefers(true);
    localStorage.setItem("literev-theme", "dark");
    render(
      <ThemeProvider>
        <LanguageProvider><ThemeSwitch /></LanguageProvider>
      </ThemeProvider>,
    );
    const auto = document.querySelector('[data-theme-choice="system"]') as HTMLElement;
    fireEvent.click(auto);
    expect(localStorage.getItem("literev-theme")).toBe("system");
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
  });

  it("marks the chosen button for a screen reader", () => {
    systemPrefers(false);
    render(
      <ThemeProvider>
        <LanguageProvider><ThemeSwitch /></LanguageProvider>
      </ThemeProvider>,
    );
    fireEvent.click(document.querySelector('[data-theme-choice="light"]') as HTMLElement);
    expect(document.querySelector('[data-theme-choice="light"]')!.getAttribute("aria-pressed"))
      .toBe("true");
    expect(document.querySelector('[data-theme-choice="dark"]')!.getAttribute("aria-pressed"))
      .toBe("false");
  });
});

describe("what the charts read", () => {
  it("gives a stroke that contrasts with each theme's chart background", () => {
    expect(chartColors("dark").ink).toBe("#ffffff");
    expect(chartColors("light").ink).not.toBe("#ffffff");
    // Le gris du groupe « restes » est plus foncé sur fond clair, sinon il s'efface.
    expect(chartColors("light").noise).not.toBe(chartColors("dark").noise);
  });

  it("resolves a plain choice without asking the system", () => {
    expect(resolveTheme("light")).toBe("light");
    expect(resolveTheme("dark")).toBe("dark");
  });
});
