// Shared setup for the vitest unit tests: jest-dom matchers and a clean DOM,
// storage and mocks between tests.
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  try {
    localStorage.clear();
    sessionStorage.clear();
  } catch {
    /* storage unavailable in this environment */
  }
});
