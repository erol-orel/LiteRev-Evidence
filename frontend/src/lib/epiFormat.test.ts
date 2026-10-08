import { describe, expect, it } from "vitest";
import { formatEpiInterval, formatEpiNumber } from "./epiFormat";

describe("formatEpiNumber", () => {
  it("rounds a ratio to three significant digits", () => {
    expect(formatEpiNumber(2.0859409266027855, "ratio")).toBe("2.09");
    expect(formatEpiNumber(12.3456, "ratio")).toBe("12.3");
    expect(formatEpiNumber(0.9999, null)).toBe("1");
  });
  it("shows a proportion as a percentage and a duration in days with one decimal", () => {
    expect(formatEpiNumber(0.5015984724495364, "proportion")).toBe("50.2 %");
    expect(formatEpiNumber(0.5, "proportion")).toBe("50 %");
    expect(formatEpiNumber(4.04, "days")).toBe("4");
    expect(formatEpiNumber(5.26, "days")).toBe("5.3");
  });
  it("never prints NaN or undefined", () => {
    expect(formatEpiNumber(null, "ratio")).toBe("-");
    expect(formatEpiNumber(undefined)).toBe("-");
    expect(formatEpiNumber(Number.NaN, "ratio")).toBe("-");
  });
});

describe("formatEpiInterval", () => {
  it("formats both bounds in the unit of the value", () => {
    expect(formatEpiInterval(1.2007645909669, 2.9711172622386712, "ratio")).toBe("[1.2 \u2013 2.97]");
    expect(formatEpiInterval(0.1571699902352, 0.8460269546638, "proportion")).toBe("[15.7 \u2013 84.6] %");
  });
  it("is a dash when the interval is incomplete", () => {
    expect(formatEpiInterval(1, null, "ratio")).toBe("-");
    expect(formatEpiInterval(null, null)).toBe("-");
  });
});
