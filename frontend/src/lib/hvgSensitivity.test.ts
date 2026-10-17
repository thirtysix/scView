import { describe, it, expect } from "vitest";
import type { HvgClusterMatch, HvgRow } from "@/api/clustering";
import {
  clusterStability,
  describeMatch,
  formatDuration,
  jaccardTone,
  otherSettings,
  perCellValues,
  secondsLeft,
  shortSettingLabel,
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
    expect(otherSettings([rows[0]!])).toEqual([]);
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
