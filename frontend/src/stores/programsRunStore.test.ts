import { describe, it, expect, vi, beforeEach } from "vitest";

const api = vi.hoisted(() => ({ runPrograms: vi.fn() }));
vi.mock("@/api/programs", async (orig) => ({ ...(await orig<typeof import("@/api/programs")>()), ...api }));

import { useProgramsRunStore } from "./programsRunStore";

describe("programs run store", () => {
  beforeEach(() => {
    api.runPrograms.mockReset();
    useProgramsRunStore.getState().reset();
    useProgramsRunStore.setState({ version: 0 });
  });

  it("keeps progress, the sweep and memory warnings, and bumps the version when it saves", async () => {
    api.runPrograms.mockImplementation(async (_id, _body, h) => {
      h.onK({ k: 5, silhouette: 0.9, mean_correlation: 0.99, n_stable: 5, error: 0.8, seconds: 3 });
      h.onWatchdog({ type: "watchdog", kind: "slow_step", step: "gene_programs", elapsed_s: 60 });
      h.onWatchdog({ type: "watchdog", kind: "memory", threshold_pct: 80, used_gb: 4.9, limit_gb: 6, pct: 81, limited: true, step: "gene_programs" });
      return { available: true, k: 10 };
    });
    await useProgramsRunStore.getState().run("d1", { k: 10 });
    const s = useProgramsRunStore.getState();
    expect(s.running).toBe(false);
    expect(s.kResults.map((k) => k.k)).toEqual([5]);
    expect(s.warnings).toHaveLength(1);
    expect(s.warnings[0]).toMatch(/Memory at 81% of the 6\.0 GB limit/);
    expect(s.result?.k).toBe(10);
    expect(s.version).toBe(1);
  });

  it("a cancel aborts the stream, saves nothing and says so", async () => {
    api.runPrograms.mockImplementation(
      (_id, _body, h) =>
        new Promise((_resolve, reject) => h.signal.addEventListener("abort", () => reject(new Error("aborted")))),
    );
    const done = useProgramsRunStore.getState().run("d1", {});
    expect(useProgramsRunStore.getState().running).toBe(true);
    useProgramsRunStore.getState().cancel();
    await done;
    const s = useProgramsRunStore.getState();
    expect(s.running).toBe(false);
    expect(s.error).toBe("Cancelled: nothing was saved.");
    expect(s.version).toBe(0);
  });
});
