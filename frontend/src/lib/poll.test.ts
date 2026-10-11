import { afterEach, describe, expect, it, vi } from "vitest";
import { POLL_MAX_ERRORS, pollDelay, pollJob } from "./poll";

afterEach(() => {
  vi.useRealTimers();
});

describe("pollDelay", () => {
  it("polls at the given pace, then every 15 s after five minutes", () => {
    expect(pollDelay(0, { intervalMs: 2000 })).toBe(2000);
    expect(pollDelay(4 * 60_000, { intervalMs: 2000 })).toBe(2000);
    expect(pollDelay(5 * 60_000, { intervalMs: 2000 })).toBe(15_000);
    expect(pollDelay(3 * 3600_000, { intervalMs: 2000 })).toBe(15_000);
  });

  it("never slows down to a faster pace than the base one", () => {
    expect(pollDelay(10 * 60_000, { intervalMs: 30_000 })).toBe(30_000);
    expect(pollDelay(61_000, { intervalMs: 1000, slowAfterMs: 60_000, slowIntervalMs: 5000 })).toBe(5000);
  });
});

/** A server whose job runs for `runningReads` reads, then reports `done`. */
function server(runningReads: number, fail: Set<number> = new Set()) {
  let n = 0;
  const read = vi.fn(async () => {
    n += 1;
    if (fail.has(n)) throw new Error("network");
    return { status: n <= runningReads ? "running" : "done", n };
  });
  return read;
}

describe("pollJob", () => {
  it("follows the job until the server stops reporting it running", async () => {
    vi.useFakeTimers();
    const read = server(3);
    const seen: number[] = [];
    const h = pollJob({ intervalMs: 1000, read, running: (s) => s.status === "running",
                        onStatus: (s) => seen.push(s.n) });
    await vi.runAllTimersAsync();
    const out = await h.promise;
    expect(out).toEqual({ kind: "done", status: { status: "done", n: 4 } });
    expect(seen).toEqual([1, 2, 3, 4]);
  });

  it("has no fixed cap: a job running for hours is still followed", async () => {
    vi.useFakeTimers();
    const read = server(2000);
    const h = pollJob({ intervalMs: 3000, read, running: (s) => s.status === "running" });
    await vi.advanceTimersByTimeAsync(4 * 3600_000);
    expect(read.mock.calls.length).toBeGreaterThan(900);
    h.cancel();
    expect(await h.promise).toEqual({ kind: "cancelled" });
  });

  it("slows down after five minutes", async () => {
    vi.useFakeTimers();
    const read = server(100_000);
    const h = pollJob({ intervalMs: 2000, read, running: () => true });
    await vi.advanceTimersByTimeAsync(5 * 60_000);
    const fast = read.mock.calls.length;
    expect(fast).toBe(150);                          // one read every 2 s
    await vi.advanceTimersByTimeAsync(5 * 60_000);
    expect(read.mock.calls.length - fast).toBe(20);  // then one every 15 s
    h.cancel();
  });

  it("tolerates a few transient errors", async () => {
    vi.useFakeTimers();
    const read = server(3, new Set([2, 3, 5]));
    const h = pollJob({ intervalMs: 1000, read, running: (s) => s.status === "running" });
    await vi.runAllTimersAsync();
    expect((await h.promise).kind).toBe("done");
  });

  it("says the contact was lost, not that the job failed, after several errors in a row", async () => {
    vi.useFakeTimers();
    const read = vi.fn(async () => { throw new Error("502"); });
    const h = pollJob({ intervalMs: 1000, read, running: () => true });
    await vi.runAllTimersAsync();
    const out = await h.promise;
    expect(out.kind).toBe("lost");
    expect(read).toHaveBeenCalledTimes(POLL_MAX_ERRORS);
  });

  it("stops reading once cancelled", async () => {
    vi.useFakeTimers();
    const read = server(1000);
    const h = pollJob({ intervalMs: 1000, read, running: () => true, firstDelayMs: 0 });
    await vi.advanceTimersByTimeAsync(3500);
    h.cancel();
    const calls = read.mock.calls.length;
    await vi.advanceTimersByTimeAsync(60_000);
    expect(read.mock.calls.length).toBe(calls);
    expect(await h.promise).toEqual({ kind: "cancelled" });
  });
});
