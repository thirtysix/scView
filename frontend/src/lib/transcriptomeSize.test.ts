import { describe, it, expect } from "vitest";
import type { TranscriptomeSize } from "@/api/de";
import { describeSize, equalCopyLog2FC, sizeCaution, sizeCautionForCopilot } from "./transcriptomeSize";

const SIZE: TranscriptomeSize = {
  source: "obs['n_counts']",
  median_a: 2518,
  median_b: 1403,
  ratio: 1.795,
  caution: true,
  threshold: 1.5,
  values_a: [2400, 2600],
  values_b: [1300, 1500],
};

describe("transcriptome size", () => {
  it("gives the fold change a gene with equal transcripts per cell reads as", () => {
    expect(equalCopyLog2FC(SIZE)).toBeCloseTo(-0.844, 3); // bigger A cells: equal-copy genes look down in A
    expect(equalCopyLog2FC({ ...SIZE, median_a: 1000, median_b: 2000 })).toBeCloseTo(1);
    expect(equalCopyLog2FC({ ...SIZE, median_b: 0 })).toBe(0);
  });

  it("describes the medians in one line", () => {
    expect(describeSize(SIZE, "CD14 Mono", "CD4 Naive T")).toBe(
      "Median total counts per cell: CD14 Mono 2,518, CD4 Naive T 1,403 (1.79x; obs['n_counts']).",
    );
  });

  it("warns with the bigger cells named first, whichever side they are", () => {
    const text = sizeCaution(SIZE, "CD14 Mono", "CD4 Naive T");
    expect(text).toMatch(/^CD14 Mono cells hold 1\.79x the RNA of CD4 Naive T cells \(median total counts 2,518 vs 1,403\)/);
    expect(text).toContain("reads as log2FC -0.84 here");
    expect(text).toContain('genes "down" in CD14 Mono');
    const flipped = sizeCaution({ ...SIZE, median_a: 1403, median_b: 2518 }, "CD4 Naive T", "CD14 Mono");
    expect(flipped).toMatch(/^CD14 Mono cells hold 1\.79x the RNA of CD4 Naive T cells/);
    expect(flipped).toContain("reads as log2FC +0.84 here");
  });

  it("tells the co-pilot how far fold changes shift", () => {
    expect(sizeCautionForCopilot(SIZE, "CD14 Mono", "CD4 Naive T")).toBe(
      "median total counts differ 1.79x (CD14 Mono 2,518 vs CD4 Naive T 1,403), so library-size " +
        "normalisation shifts fold changes of broadly expressed genes by about -0.84 log2",
    );
  });
});
