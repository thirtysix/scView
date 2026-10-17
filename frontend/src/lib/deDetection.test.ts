import { describe, it, expect } from "vitest";
import {
  DEFAULT_MIN_PCT,
  describePrefilter,
  detectionBins,
  formatPct,
  genesPassing,
  minPctChoices,
  MIN_PCT_CHOICES,
} from "./deDetection";

/** A 0.01 % grid histogram with `n` genes at each of the given fractions. */
function hist(at: Record<number, number>): number[] {
  const h = new Array<number>(10_001).fill(0);
  for (const [frac, n] of Object.entries(at)) h[Math.floor(Number(frac) * 10_000)]! += n;
  return h;
}

describe("genesPassing", () => {
  const h = hist({ 0: 7, 0.005: 3, 0.05: 2, 0.0999: 4, 0.1: 5, 0.25: 1, 1: 6 });

  it("counts the genes at or above a cut-off, exactly on the grid", () => {
    expect(genesPassing(h, 0)).toBe(28);
    expect(genesPassing(h, 0.01)).toBe(18);
    expect(genesPassing(h, 0.05)).toBe(18);
    expect(genesPassing(h, 0.1)).toBe(12); // 9.99% stays out, exactly 10% is in
    expect(genesPassing(h, 0.25)).toBe(7);
    expect(genesPassing(h, 1)).toBe(6);
  });
});

describe("detectionBins", () => {
  it("regroups the grid into log bins, with the genes under 0.1 % apart", () => {
    const b = detectionBins(hist({ 0: 7, 0.0005: 2, 0.001: 1, 0.01: 3, 0.1: 4, 1: 5 }));
    expect(b.counts).toHaveLength(30);
    expect(b.below).toBe(9);
    expect(b.lo[0]).toBeCloseTo(-3);
    expect(b.counts[0]).toBe(1); // 0.1 % opens the first bin
    expect(b.counts[10]).toBe(3); // 1 %
    expect(b.counts[20]).toBe(4); // 10 %
    expect(b.counts[29]).toBe(5); // all cells land in the last bin
    expect(b.counts.reduce((s, c) => s + c, 0) + b.below).toBe(22);
  });
});

describe("cut-off labels", () => {
  it("formats fractions and offers Seurat's defaults", () => {
    expect([formatPct(0.1), formatPct(0.025), formatPct(0.001), formatPct(1)]).toEqual(["10%", "2.5%", "0.1%", "100%"]);
    expect(MIN_PCT_CHOICES.map((c) => c.value)).toEqual([0, 0.01, 0.05, 0.1, 0.25]);
    expect(MIN_PCT_CHOICES.find((c) => c.value === DEFAULT_MIN_PCT)!.label).toMatch(/1% of either group \(Seurat v5\)/);
    expect(MIN_PCT_CHOICES.find((c) => c.value === 0.1)!.label).toMatch(/10% of either group \(Seurat v4\)/);
    expect(MIN_PCT_CHOICES[0]!.label).toMatch(/no detection cut-off/);
  });

  it("keeps a restored cut-off that is not one of the choices", () => {
    expect(minPctChoices(0.1)).toBe(MIN_PCT_CHOICES);
    expect(minPctChoices(0.02).map((c) => c.value)).toEqual([0, 0.01, 0.02, 0.05, 0.1, 0.25]);
  });
});

describe("describePrefilter", () => {
  it("says what was tested and what was left out", () => {
    expect(describePrefilter({ min_pct: 0.1, n_genes_tested: 8123, n_genes_skipped: 5930 })).toBe(
      "8,123 genes tested; 5,930 detected in under 10% of the cells of both groups were left out.",
    );
    expect(describePrefilter({ min_pct: 0, n_genes_tested: 14053, n_genes_skipped: 0 })).toMatch(
      /^All 14,053 genes tested \(no detection cut-off\)/,
    );
    expect(describePrefilter({ min_pct: null, n_genes_tested: 500, n_genes_skipped: 0 })).toMatch(/scaled/);
    expect(describePrefilter({})).toBe(""); // an older backend
  });
});
