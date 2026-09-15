import { describe, expect, it } from "vitest";
import {
  combinedQueryText,
  csvEscape,
  effectiveKind,
  looksBoolean,
  scenarioDisplayName,
  yearSliderBounds,
} from "./searchText";

describe("combinedQueryText", () => {
  it("returns the only facet unchanged", () => {
    expect(combinedQueryText([{ kind: "auto", text: " influenza AND surveillance " }], "union"))
      .toBe("influenza AND surveillance");
  });

  it("returns an empty string when every facet is blank", () => {
    expect(combinedQueryText([{ kind: "auto", text: "  " }, { kind: "auto", text: "" }], "intersection")).toBe("");
  });

  it("joins the facets with the global combinator", () => {
    const sub = [{ kind: "boolean" as const, text: "A" }, { kind: "auto" as const, text: " B " }];
    expect(combinedQueryText(sub, "intersection")).toBe("(A) AND (B)");
    expect(combinedQueryText(sub, "union")).toBe("(A) OR (B)");
  });

  it("folds left to right and lets a facet override the operator", () => {
    const sub = [
      { kind: "auto" as const, text: "A" },
      { kind: "auto" as const, text: "B", op: "or" as const },
      { kind: "auto" as const, text: "C", op: "and" as const },
    ];
    expect(combinedQueryText(sub, "intersection")).toBe("((A) OR (B)) AND (C)");
  });

  it("skips blank facets so the operators stay attached to real text", () => {
    const sub = [
      { kind: "auto" as const, text: "A" },
      { kind: "auto" as const, text: "   " },
      { kind: "auto" as const, text: "C" },
    ];
    expect(combinedQueryText(sub, "intersection")).toBe("(A) AND (C)");
  });
});

describe("looksBoolean", () => {
  it("detects upper-case operators, field tags, quoted phrases and parentheses", () => {
    expect(looksBoolean("influenza AND surveillance")).toBe(true);
    expect(looksBoolean("influenza NOT vaccine")).toBe(true);
    expect(looksBoolean("influenza[tiab]")).toBe(true);
    expect(looksBoolean('"case counts"')).toBe(true);
    expect(looksBoolean("(influenza) surveillance")).toBe(true);
  });

  it("treats plain language as natural", () => {
    expect(looksBoolean("ambulance demand forecasting")).toBe(false);
    expect(looksBoolean("forecasting and surveillance of influenza")).toBe(false);
    expect(looksBoolean("a single \" quote")).toBe(false);
    expect(looksBoolean("")).toBe(false);
    expect(looksBoolean("   ")).toBe(false);
  });
});

describe("effectiveKind", () => {
  it("keeps an explicit kind", () => {
    expect(effectiveKind({ kind: "natural", text: "influenza AND surveillance" })).toBe("natural");
    expect(effectiveKind({ kind: "boolean", text: "ambulance demand" })).toBe("boolean");
  });

  it("detects the kind when it is automatic", () => {
    expect(effectiveKind({ kind: "auto", text: "influenza AND surveillance" })).toBe("boolean");
    expect(effectiveKind({ kind: "auto", text: "ambulance demand" })).toBe("natural");
  });
});

describe("scenarioDisplayName", () => {
  it("keeps a short query as it is", () => {
    expect(scenarioDisplayName("  influenza AND surveillance ")).toBe("influenza AND surveillance");
  });

  it("truncates a long query on a word boundary with an ellipsis", () => {
    const q = Array.from({ length: 40 }, (_, i) => `word${i}`).join(" ");
    const name = scenarioDisplayName(q, 60);
    expect(name.length).toBeLessThanOrEqual(61);
    expect(name.endsWith("…")).toBe(true);
    expect(name).not.toMatch(/ …$/);
    expect(q.startsWith(name.slice(0, -1))).toBe(true);
    expect(name.slice(0, -1).split(" ").every((w) => /^word\d+$/.test(w))).toBe(true);
  });

  it("cuts hard when there is no usable space", () => {
    const q = "x".repeat(200);
    expect(scenarioDisplayName(q, 50)).toBe("x".repeat(50) + "…");
  });
});

describe("yearSliderBounds", () => {
  it("ignores implausible years and ends at the current year", () => {
    const b = yearSliderBounds([{ value: "2005" }, { value: 12 }, { value: "n/a" }, { value: 1998 }]);
    expect(b).toEqual({ min: 1998, max: new Date().getFullYear() });
  });

  it("falls back to 1990 without any year", () => {
    expect(yearSliderBounds(null).min).toBe(1990);
    expect(yearSliderBounds([]).min).toBe(1990);
  });
});

describe("csvEscape", () => {
  it("quotes values and turns null into an empty string", () => {
    expect(csvEscape('a "b"')).toBe('"a \\"b\\""');
    expect(csvEscape(null)).toBe('""');
    expect(csvEscape(undefined)).toBe('""');
    expect(csvEscape(42)).toBe("42");
  });
});
