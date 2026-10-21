import { describe, it, expect } from "vitest";
import type { GeneProgram, ProgramsByGroup } from "@/api/programs";
import {
  errorText,
  heatmapData,
  parseSweep,
  prettyTerm,
  programLabel,
  stabilityBadge,
  stabilityText,
  sweepTraces,
  topGroup,
  usageLabel,
} from "./programs";

const prog = (index: number, extra: Partial<GeneProgram> = {}): GeneProgram => ({
  index,
  name: `P${index}`,
  n_members: 5,
  runs_found: 5,
  correlation: 0.97,
  silhouette: 0.9,
  stable: true,
  mean_usage: 0.2,
  pct_top: 0.3,
  top_genes: [
    { gene: `A${index}`, score: 3 },
    { gene: `B${index}`, score: 2 },
    { gene: `C${index}`, score: 1 },
    { gene: `D${index}`, score: 0.5 },
  ],
  ...extra,
});

const BY: ProgramsByGroup = {
  column: "cell_type",
  groups: ["B", "NK", "T"],
  n_cells: [10, 20, 30],
  mean: [
    [0.1, 0.7, 0.2],
    [0.5, 0.05, 0.45],
  ],
};

describe("programs helpers", () => {
  it("labels a program by its first genes", () => {
    expect(programLabel(prog(2))).toBe("P2 (A2, B2, C2)");
    expect(programLabel(prog(2), 2)).toBe("P2 (A2, B2)");
    expect(programLabel(prog(1, { top_genes: [] }))).toBe("P1");
    expect(usageLabel("P3", ["ISG15", "IFI6", "MX1", "RSAD2"])).toBe("P3 usage share (ISG15, IFI6, MX1)");
  });

  it("builds the program x group heatmap and finds each program's top group", () => {
    const h = heatmapData(BY, [prog(1), prog(2)]);
    expect(h.x).toEqual(["B", "NK", "T"]);
    expect(h.y).toEqual(["P1 (A1, B1)", "P2 (A2, B2)"]);
    expect(h.z).toEqual(BY.mean);
    expect(h.zmax).toBe(0.7);
    expect(h.index).toEqual([1, 2]);
    expect(topGroup(BY, 1)).toEqual({ group: "NK", mean: 0.7 });
    expect(topGroup(BY, 2)).toEqual({ group: "B", mean: 0.5 });
    expect(topGroup(BY, 3)).toBeNull();
  });

  it("explains why a program is or is not stable", () => {
    expect(stabilityText(prog(1), 5, 0.9)).toMatch(/Found in 5 of 5 runs .* 0\.97; silhouette 0\.90: stable\./);
    expect(stabilityText(prog(1, { stable: false, runs_found: 3 }), 5, 0.9)).toMatch(/missing from 2 of the 5 runs/);
    expect(stabilityText(prog(1, { stable: false, correlation: 0.8 }), 5, 0.9)).toMatch(/correlate below 0\.9/);
    expect(stabilityText(prog(1), 1, 0.9)).toMatch(/needs at least 2 seeds/);
    expect(stabilityBadge(prog(1), 5)).toBe("stable");
    expect(stabilityBadge(prog(1, { stable: false, runs_found: 3 }), 5)).toBe("3/5 runs");
    expect(stabilityBadge(prog(1, { stable: false, correlation: 0.881 }), 5)).toBe("r = 0.88");
    expect(stabilityBadge(prog(1), 1)).toBe("1 run");
  });

  it("reads a sweep from text", () => {
    expect(parseSweep("20, 5 10;15 10", 2, 30, 8)).toEqual([5, 10, 15, 20]);
    expect(parseSweep("5, 40", 2, 30, 8)).toMatch(/from 2 to 30/);
    expect(parseSweep("5.5", 2, 30, 8)).toMatch(/whole number/);
    expect(parseSweep("2 3 4 5 6 7 8 9 10", 2, 30, 8)).toMatch(/at most 8/);
    expect(parseSweep("", 2, 30, 8)).toEqual([]);
  });

  it("plots the sweep on two axes", () => {
    const t = sweepTraces([
      { k: 5, silhouette: 0.99, mean_correlation: 1, n_stable: 5, error: 0.81, seconds: 3 },
      { k: 10, silhouette: 0.83, mean_correlation: 0.98, n_stable: 6, error: 0.8, seconds: 12 },
    ]);
    expect(t[0]!.x).toEqual([5, 10]);
    expect(t[0]!.y).toEqual([0.99, 0.83]);
    expect(t[1]!.y).toEqual([0.81, 0.8]);
    expect(t[1]!.yaxis).toBe("y2");
  });

  it("tidies Hallmark names and server errors", () => {
    expect(prettyTerm("HALLMARK_INTERFERON_ALPHA_RESPONSE")).toBe("Interferon alpha response");
    expect(prettyTerm("GOBP_X")).toBe("Gobp x");
    expect(errorText(new Error('API error 404: {"detail":"There are 10 programs; no P12."}'))).toBe(
      "There are 10 programs; no P12.",
    );
    expect(errorText(new Error("network down"))).toBe("network down");
  });
});
