import { describe, expect, it } from "vitest";
import type { CodebookNode } from "./api";
import { mappedShare, nodeOptions } from "./codebook";

const node = (sheet: string, l1: string, l2: string | null, en?: string, fr?: string): CodebookNode => ({
  sheet, l1, l2, l3: null, synonyms: [], label_en: en ?? null, label_fr: fr ?? null,
});

describe("nodeOptions", () => {
  const nodes = [
    node("human_susc", "age", "child", "Children", "Enfants"),
    node("human_susc", "sex_gender", null),
    node("human_susc", "sex_gender", "male", "Male", "Homme"),
    node("human_susc", "sex_gender", "female"),
    node("env", "setting", "backyard", "Backyard"),
  ];

  it("lists only the level-2 values of the sheet", () => {
    expect(nodeOptions(nodes, "human_susc", "en").map((o) => o.value)).toEqual(
      ["age|child", "sex_gender|female", "sex_gender|male"]);
    expect(nodeOptions(nodes, "env", "en")).toHaveLength(1);
  });

  it("puts the group the label is already filed under first", () => {
    expect(nodeOptions(nodes, "human_susc", "en", "sex_gender")[0].l1).toBe("sex_gender");
  });

  it("uses the label in the language, and the key when there is none", () => {
    const fr = nodeOptions(nodes, "human_susc", "fr", "sex_gender");
    expect(fr.find((o) => o.l2 === "male")?.label).toBe("Homme");
    expect(fr.find((o) => o.l2 === "female")?.label).toBe("female");
  });
});

describe("mappedShare", () => {
  it("is a whole percentage and 0 with no rows", () => {
    expect(mappedShare(3, 1)).toBe(75);
    expect(mappedShare(0, 0)).toBe(0);
  });
});
