import { describe, expect, it } from "vitest";
import { en } from "../i18n/locales/en";
import { fr } from "../i18n/locales/fr";
import { levelBadge, levelKey, levelText } from "./gradeLevels";

/** A `t` reading one locale object, as the provider does. */
function tOf(locale: unknown) {
  return (key: string) => {
    const v = key.split(".").reduce<unknown>((o, k) => (o as Record<string, unknown> | undefined)?.[k], locale);
    return typeof v === "string" ? v : key;
  };
}

describe("GRADE levels of the evidence brief", () => {
  const server = ["Élevée", "Modérée", "Faible", "Très faible", "Non applicable", "Non évaluée"];

  it("shows each server level in English to an English reader, never the French value", () => {
    const t = tOf(en);
    expect(server.map((l) => levelText(l, t))).toEqual(
      ["High", "Moderate", "Low", "Very low", "Not applicable", "Not assessed"]);
  });

  it("shows the French label to a French reader", () => {
    const t = tOf(fr);
    expect(server.map((l) => levelText(l, t))).toEqual(server);
  });

  it("reads the former four strengths of older briefs", () => {
    const t = tOf(en);
    expect(["Fort", "Modéré", "Faible", "Insuffisant"].map((l) => levelText(l, t)))
      .toEqual(["Strong", "Moderate", "Low", "Insufficient"]);
  });

  it("gives every known level a label in both languages and a coloured badge", () => {
    for (const l of [...server, "Fort", "Modéré", "Insuffisant"]) {
      const key = levelKey(l);
      expect(key, l).not.toBeNull();
      expect(tOf(en)(`scenarioDetail.evidences.levels.${key}`)).not.toContain("scenarioDetail");
      expect(tOf(fr)(`scenarioDetail.evidences.levels.${key}`)).not.toContain("scenarioDetail");
      expect(levelBadge(l)).not.toBe("bg-white/10 text-white/50");
    }
  });

  it("shows an unknown value as it came, with a neutral badge", () => {
    expect(levelText("Something new", tOf(en))).toBe("Something new");
    expect(levelText(null, tOf(en))).toBe("");
    expect(levelBadge("Something new")).toBe("bg-white/10 text-white/50");
  });
});
