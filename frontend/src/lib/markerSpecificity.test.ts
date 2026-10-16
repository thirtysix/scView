import { describe, it, expect } from "vitest";
import type { MarkerGene } from "@/api/markers";
import {
  MARKER_CSV_HEADERS,
  detectionRatio,
  formatPct,
  formatRatio,
  hasSpecificity,
  markerCsvRow,
  sortMarkers,
} from "./markerSpecificity";

const m = (gene: string, extra: Partial<MarkerGene> = {}): MarkerGene => ({
  gene, group: "T", logfoldchange: 1, pval: 1e-5, pval_adj: 1e-3, ...extra,
});

describe("detectionRatio", () => {
  it("divides the group's detection by the next-best group's", () => {
    expect(detectionRatio({ pct_in: 0.8, pct_max_other: 0.2 })).toBeCloseTo(4);
  });

  it("is infinite when no other group detects the gene (the backend sends null)", () => {
    expect(detectionRatio({ pct_in: 0.5, pct_max_other: 0 })).toBe(Infinity);
    expect(formatRatio(Infinity)).toBe("∞");
  });

  it("is unknown without both fractions, or when nobody detects it", () => {
    expect(detectionRatio({ pct_in: 0.5 })).toBeNull();
    expect(detectionRatio({ pct_in: 0, pct_max_other: 0 })).toBeNull();
    expect(formatRatio(null)).toBe("–");
  });
});

describe("sortMarkers", () => {
  const rows = [
    m("A", { tau: 0.2, pct_in: 0.9, pct_max_other: 0.9 }),
    m("B", { tau: 0.95, pct_in: 0.4, pct_max_other: 0 }),
    m("C", { tau: null, logfoldchange: null }),
    m("D", { tau: 0.6, pct_in: 0.6, pct_max_other: 0.2 }),
  ];

  it("sorts by tau, missing last in both directions", () => {
    expect(sortMarkers(rows, "tau", "desc").map((r) => r.gene)).toEqual(["B", "D", "A", "C"]);
    expect(sortMarkers(rows, "tau", "asc").map((r) => r.gene)).toEqual(["A", "D", "B", "C"]);
  });

  it("sorts by detection ratio with an infinite ratio on top", () => {
    expect(sortMarkers(rows, "pct_ratio", "desc").map((r) => r.gene)).toEqual(["B", "D", "A", "C"]);
  });

  it("keeps a missing fold change last", () => {
    expect(sortMarkers(rows, "logfoldchange", "asc").at(-1)!.gene).toBe("C");
  });
});

describe("formatPct", () => {
  it("shows a few cells as <1% rather than 0%, next to a large ratio", () => {
    expect(formatPct(0.0035)).toBe("<1%");
    expect(formatPct(0)).toBe("0%");
    expect(formatPct(0.935)).toBe("94%");
    expect(formatPct(null)).toBe("–");
  });
});

describe("hasSpecificity", () => {
  it("is false for markers from a backend without the columns", () => {
    expect(hasSpecificity([m("A"), m("B")])).toBe(false);
    expect(hasSpecificity([m("A", { tau: 0.5 })])).toBe(true);
  });
});

describe("markerCsvRow", () => {
  it("writes the specificity columns, an infinite ratio as inf", () => {
    const row = markerCsvRow(m("CD79A", {
      tau: 0.945, pct_in: 0.8, pct_max_other: 0, max_other_group: "B activated", score: 30,
    }));
    const byName = Object.fromEntries(MARKER_CSV_HEADERS.map((h, i) => [h, row[i]]));
    expect(byName.tau).toBe("0.945");
    expect(byName.pct_max_other).toBe("0.000");
    expect(byName.max_other_group).toBe("B activated");
    expect(byName.pct_ratio).toBe("inf");
    expect(markerCsvRow(m("X")).slice(-4)).toEqual(["", "", "", ""]);
  });
});
