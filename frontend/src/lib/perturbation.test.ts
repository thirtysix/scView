import { describe, it, expect } from "vitest";
import type { MixscapeTarget } from "@/api/perturbation";
import { deHandoff, ldaTraces, mappingChanges, posteriorTraces, sortTargets } from "./perturbation";

const row = (target: string, n_cells: number, ko_fraction: number, n_de_genes: number): MixscapeTarget => ({
  target, n_cells, ko_fraction, n_de_genes, n_ko: Math.round(n_cells * ko_fraction), n_np: 0, rounds: 1, top_de_genes: [],
});

describe("perturbation helpers", () => {
  it("sorts the per-target table, ties by name", () => {
    const rows = [row("B", 10, 0.5, 3), row("A", 20, 0.5, 9), row("C", 5, 0.9, 0)];
    expect(sortTargets(rows, "ko_fraction", true).map((r) => r.target)).toEqual(["C", "A", "B"]);
    expect(sortTargets(rows, "n_cells", false).map((r) => r.target)).toEqual(["C", "B", "A"]);
    expect(sortTargets(rows, "target", false).map((r) => r.target)).toEqual(["A", "B", "C"]);
  });

  it("builds the posterior bars at the bin centres", () => {
    const [np, ko] = posteriorTraces({ edges: [0, 0.5, 1], ko: [0, 7], np: [3, 1] }) as { x: number[]; y: number[]; name: string }[];
    expect(np?.name).toBe("NP");
    expect(np?.x).toEqual([0.25, 0.75]);
    expect(ko?.y).toEqual([0, 7]);
  });

  it("puts the control first in the LDA view", () => {
    const traces = ldaTraces({ x: [1, 2, 3], y: [0, 0, 0], label: ["B", "NT", "A"], n_cells: 3, n_shown: 3, dims: 2 }) as {
      name: string;
      x: number[];
    }[];
    expect(traces.map((t) => t.name)).toEqual(["NT (control)", "A KO", "B KO"]);
    expect(traces[0]?.x).toEqual([2]);
  });

  it("hands a target's KO cells vs the controls to DE, and keeps only edited mappings", () => {
    expect(deHandoff("STAT1")).toEqual({
      mode: "groups", splitBy: "mixscape_class", groupA: "STAT1 KO", groupB: "NT", withinCol: "", withinVal: "",
    });
    expect(mappingChanges({ a: "A", b: " B2 ", c: "" }, { a: "A", b: "B", c: "C" })).toEqual({ b: "B2" });
  });
});
