import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ErrorBoundary } from "./ErrorBoundary";
import { en } from "../i18n/locales/en";

function Boom({ explode }: { explode: boolean }) {
  if (explode) throw new Error("kaboom");
  return <p>rendered fine</p>;
}

describe("ErrorBoundary", () => {
  it("isolates a rendering error behind a localized message with the section name", () => {
    localStorage.setItem("literev-lang", "en");
    vi.spyOn(console, "error").mockImplementation(() => {});
    render(
      <ErrorBoundary label="nav.search">
        <Boom explode />
      </ErrorBoundary>,
    );
    expect(screen.getByText(`${en.errorBoundary.title} ${en.errorBoundary.ofSection} « ${en.nav.search} ».`)).toBeInTheDocument();
    expect(screen.getByText("kaboom")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: en.errorBoundary.retry })).toBeInTheDocument();
  });

  it("retries on demand and resets when the reset key changes", () => {
    localStorage.setItem("literev-lang", "en");
    vi.spyOn(console, "error").mockImplementation(() => {});
    const onReset = vi.fn();
    const { rerender } = render(
      <ErrorBoundary resetKey="a" onReset={onReset}>
        <Boom explode />
      </ErrorBoundary>,
    );
    expect(screen.getByText("kaboom")).toBeInTheDocument();

    // Retry re-renders the children (which still throw) and notifies the parent.
    fireEvent.click(screen.getByRole("button", { name: en.errorBoundary.retry }));
    expect(onReset).toHaveBeenCalledTimes(1);
    expect(screen.getByText("kaboom")).toBeInTheDocument();

    // A new reset key (tab or scenario change) clears the error once the child recovers.
    rerender(
      <ErrorBoundary resetKey="b" onReset={onReset}>
        <Boom explode={false} />
      </ErrorBoundary>,
    );
    expect(screen.getByText("rendered fine")).toBeInTheDocument();
  });

  it("renders a custom fallback when one is given", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    render(
      <ErrorBoundary fallback={<p>custom fallback</p>}>
        <Boom explode />
      </ErrorBoundary>,
    );
    expect(screen.getByText("custom fallback")).toBeInTheDocument();
  });
});
