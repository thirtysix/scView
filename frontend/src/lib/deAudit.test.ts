import { describe, expect, it } from "vitest";
import type { DEResponse } from "@/api/de";
import { auditBlocker, concordanceText, fewDonorsNote, nullBins, nullRange, shareText } from "./deAudit";

const RES: DEResponse = {
  n_selected: 2031,
  n_rest: 2194,
  label: "STIM (CD14 Mono)",
  genes: [],
  method: "wilcoxon",
  result_id: "r1",
  replicate_key: "Donor",
  n_donors_a: 8,
  n_donors_b: 8,
};

describe("auditBlocker", () => {
  it("allows a kept between-groups result with p-values and donors", () => {
    expect(auditBlocker(RES, true, 3)).toBeNull();
    expect(auditBlocker({ ...RES, replicate_key: null }, true, 1)).toBeNull();
  });

  it("names the reason it cannot run", () => {
    expect(auditBlocker(null, true, 3)).toMatch(/Run a DE/);
    expect(auditBlocker(RES, false, 3)).toMatch(/between-groups/);
    expect(auditBlocker({ ...RES, result_id: null }, true, 3)).toMatch(/not kept/);
    expect(auditBlocker({ ...RES, method: "logreg" }, true, 3)).toMatch(/no p-values/);
    expect(auditBlocker({ ...RES, replicate_key: null }, true, 0)).toMatch(/donor or sample column/);
  });
});

describe("fewDonorsNote", () => {
  it("warns that the shuffles will be skipped below 3 donors a side", () => {
    expect(fewDonorsNote(RES)).toBeNull();
    expect(fewDonorsNote({ ...RES, n_donors_b: 2 })).toMatch(/Only 8 and 2 donors.*3 a side/);
    expect(fewDonorsNote({ ...RES, n_donors_a: null, n_donors_b: null })).toBeNull();
  });
});

describe("nullBins", () => {
  it("bins every shuffle and reaches past the observed count", () => {
    const b = nullBins([0, 3, 12, 590, 663], 1836, 12);
    expect(b.y.reduce((s, v) => s + v, 0)).toBe(5);
    expect(b.range[1]).toBeGreaterThan(1836);
    expect(b.x.length).toBeLessThanOrEqual(12);
    expect(b.y[0]).toBe(3);
  });

  it("handles all-zero counts", () => {
    const b = nullBins([0, 0], 0);
    expect(b.y[0]).toBe(2);
    expect(b.width).toBe(1);
  });
});

describe("text", () => {
  it("states the range and median of the shuffles", () => {
    expect(nullRange([663, 10, 310, 17])).toBe("10-663 (median 164)");
    expect(nullRange([4, 4])).toBe("4");
    expect(nullRange([])).toBe("none");
  });

  it("reads the concordance and the top-donor share", () => {
    const t = concordanceText(
      {
        method: "deseq2", name: "DESeq2", n_tested: 3454, n_samples_a: 8, n_samples_b: 8, design: "~Donor + condition",
        n_sig_cell: 1836, n_sig_pseudobulk: 2265, n_shared: 1565, jaccard: 0.6171, n_genes_common: 3454,
        spearman_lfc: 0.9946,
      },
      0.05,
    );
    expect(t).toBe(
      "Adj. p < 0.05: 1,836 at cell level, 2,265 in pseudobulk (DESeq2, 8 vs 8 donor samples), 1,565 in both (Jaccard 0.62); Spearman 0.99 of the log2 fold changes over 3,454 genes in both.",
    );
    expect(shareText(0.0417, "SNG-1039")).toBe("4% (SNG-1039)");
    expect(shareText(null, null)).toBe("–");
  });
});
