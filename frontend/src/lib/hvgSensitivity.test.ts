import { describe, it, expect } from "vitest";
import type { HvgClusterMatch, HvgRecorded, HvgRow } from "@/api/clustering";
import {
  clusterStability,
  controlMismatchReason,
  describeMatch,
  formatDuration,
  jaccardTone,
  otherSettings,
  perCellValues,
  secondsLeft,
  shortSettingLabel,
  verdictLabel,
} from "./hvgSensitivity";

const match = (cluster: string, jaccard: number, extra: Partial<HvgClusterMatch> = {}): HvgClusterMatch => ({
  cluster,
  n_cells: 100,
  jaccard,
  match: "0",
  contained: jaccard,
  purity: 1,
  parts: 1,
  status: jaccard >= 0.75 ? "stable" : "splits",
  ...extra,
});

const row = (key: string, clusters: HvgClusterMatch[], isCurrent = false): HvgRow => ({
  flavor: "seurat",
  n_top_genes: 2000,
  key,
  label: key,
  is_current: isCurrent,
  n_hvgs: 2000,
  shared_hvgs: 2000,
  ari: 1,
  ami: 1,
  n_clusters: clusters.length,
  sizes: [],
  clusters,
  seconds: 10,
  column: `col_${key}`,
  exists: false,
});

describe("clusterStability", () => {
  const rows = [
    row("control", [match("0", 1), match("1", 1), match("2", 1)], true),
    row("a", [match("0", 0.95), match("1", 0.9), match("2", 0.3, { status: "dissolves" })]),
    row("b", [match("0", 0.9), match("1", 0.6, { status: "merges" }), match("2", 0.4)]),
  ];

  it("leaves the control out and orders the least stable first", () => {
    const out = clusterStability(rows);
    expect(out.map((s) => s.cluster)).toEqual(["2", "1", "0"]);
    const [two, one, zero] = out;
    expect(two!.mean).toBeCloseTo(0.35);
    expect(two!.verdict).toBe("unstable");
    expect(one!.verdict).toBe("depends");
    expect(one!.nStable).toBe(1);
    expect(zero!.verdict).toBe("robust");
    expect(zero!.matches.map((m) => m?.jaccard)).toEqual([0.95, 0.9]);
  });

  it("has no verdict before any other setting has run", () => {
    const out = clusterStability([rows[0]!]);
    expect(out.map((s) => s.verdict)).toEqual([null, null, null]);
    expect(out.map(verdictLabel)).toEqual(["—", "—", "—"]);
    expect(otherSettings([rows[0]!])).toEqual([]);
  });

  it("says in how many settings a cluster that depends on them is stable", () => {
    expect(clusterStability(rows).map(verdictLabel)).toEqual(["unstable", "stable in 1 of 2", "robust"]);
  });

  it("marks a cluster a setting lacks as missing", () => {
    const out = clusterStability([rows[0]!, row("c", [match("0", 0.9)])]);
    const one = out.find((s) => s.cluster === "1")!;
    expect(one.matches).toEqual([null]);
    expect(one.mean).toBeNull();
  });
});

describe("secondsLeft", () => {
  it("uses the estimate before the first setting finishes", () => {
    expect(secondsLeft([], 5, 12)).toBe(60);
    expect(secondsLeft([], 5, null)).toBeNull();
  });

  it("leaves out the first setting, which also compiles, once there are others", () => {
    expect(secondsLeft([45], 5, 12)).toBe(48); // capped by the estimate
    expect(secondsLeft([45, 10, 12], 5, 12)).toBe(22);
    expect(secondsLeft([45, 10, 12, 11, 10], 5, 12)).toBe(0);
  });
});

describe("formatting", () => {
  it("formats durations", () => {
    expect(formatDuration(2)).toBe("5 s");
    expect(formatDuration(43)).toBe("45 s");
    expect(formatDuration(71)).toBe("1 min 10 s");
    expect(formatDuration(180)).toBe("3 min");
    expect(formatDuration(1234)).toBe("21 min");
  });

  it("labels settings briefly", () => {
    expect(shortSettingLabel({ flavor: "seurat", n_top_genes: 1000 })).toBe("Seurat 1k");
    expect(shortSettingLabel({ flavor: "cell_ranger", n_top_genes: 1500 })).toBe("CR 1.5k");
    expect(shortSettingLabel({ flavor: "pearson_residuals", n_top_genes: 800 })).toBe("Pearson 800");
  });

  it("tones Jaccard values and says what became of a cluster", () => {
    expect(jaccardTone(0.8)).toContain("emerald");
    expect(jaccardTone(0.6)).toContain("amber");
    expect(jaccardTone(0.5)).toContain("red");
    expect(jaccardTone(null)).toContain("slate");
    const text = describeMatch(match("0", 0.42, { contained: 0.53, purity: 0.91, parts: 2 }));
    expect(text).toBe(
      "Jaccard 0.42: splits. 53% of its cells are in one new cluster, which is 91% this cluster, spread over 2 clusters.",
    );
  });
});

describe("perCellValues", () => {
  it("maps each cell's cluster to its value, NaN for missing labels or values", () => {
    const out = perCellValues([0, 1, 2, -1, 1], ["a", "b", "c"], new Map([["a", 0.9], ["b", 0.4]]));
    expect(Array.from(out.slice(0, 2))).toEqual([Math.fround(0.9), Math.fround(0.4)]);
    expect(Number.isNaN(out[2]!)).toBe(true);
    expect(Number.isNaN(out[3]!)).toBe(true);
    expect(out[4]).toBeCloseTo(0.4);
  });
});

describe("controlMismatchReason", () => {
  const rec: HvgRecorded = {
    reference: "cluster", method: "leiden", resolution: 1, flavor: "seurat", n_top_genes: 2000,
    hvg_batch_key: null, target_sum: 1e4, n_comps: 50, n_neighbors: 15, batch_key: "stim", notes: [],
    reference_from_file: false, pcs_made_by: "0.16.1", pcs_made_on: "2026-10-17", pcs_older: false,
  };
  it("falls back to the generic reason", () => {
    expect(controlMismatchReason(rec)).toMatch(/data or the graph changed/);
    expect(controlMismatchReason(null)).toMatch(/data or the graph changed/);
  });
  it("names older processing and a reference from the file", () => {
    const old = { ...rec, pcs_older: true, pcs_made_by: "0.1.0", pcs_made_on: "2026-06-10" };
    expect(controlMismatchReason(old)).toMatch(/older scView \(scView 0\.1\.0, 2026-06-10\)/);
    const both = controlMismatchReason({ ...old, reference_from_file: true });
    expect(both).toMatch(/^“cluster” came with the file/);
    expect(both).toMatch(/; and the dataset was processed by an older scView/);
  });
});
