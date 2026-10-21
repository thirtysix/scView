import { describe, expect, it } from "vitest";
import type { MergePair } from "@/api/clustering";
import { clustersAfter, defaultSelection, describePair, pairKey } from "./mergeCheck";

const pair = (a: string, b: string, extra: Partial<MergePair> = {}): MergePair => ({
  a,
  b,
  adjacency: 0.2,
  n_cells_a: 100,
  n_cells_b: 80,
  evidence: "held_out",
  n_genes: 3,
  up: [],
  down: [],
  train_pairs: ["4+7"],
  candidate: true,
  ...extra,
});

describe("mergeCheck helpers", () => {
  it("ticks the candidates only", () => {
    const sel = defaultSelection([pair("1", "2"), pair("2", "3", { candidate: false, n_genes: 80 })]);
    expect([...sel]).toEqual([pairKey(pair("1", "2"))]);
  });

  it("counts the clusters left once chained merges join", () => {
    expect(clustersAfter(10, [])).toBe(10);
    expect(clustersAfter(10, [pair("1", "2"), pair("2", "3")])).toBe(8);
    expect(clustersAfter(10, [pair("1", "2"), pair("2", "1"), pair("3", "4")])).toBe(8);
  });

  it("says why a pair is or is not a candidate", () => {
    expect(describePair(pair("1", "2"), 10)).toMatch(/3 genes .*training pairs 4\+7.*fewer than 10, a merge candidate/);
    expect(describePair(pair("1", "2", { candidate: false, n_genes: 1 }), 10)).toMatch(/1 gene distinguish.*kept apart/);
    expect(describePair(pair("1", "2", { evidence: "together", n_genes: null, candidate: false }), 10)).toMatch(
      /never split/,
    );
  });
});
