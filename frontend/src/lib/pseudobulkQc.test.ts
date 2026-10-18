import { describe, it, expect } from "vitest";
import { result } from "@/test/pseudobulkQcFixture";
import { defaultColourBy, fmtP, pcaTraces, pcLabel, r2Shade } from "./pseudobulkQc";

describe("pseudobulk QC helpers", () => {
  it("labels PCs with their variance and formats p-values", () => {
    expect(pcLabel(1, [0.75, 0.049])).toBe("PC1 (75%)");
    expect(pcLabel(2, [0.75, 0.049])).toBe("PC2 (4.9%)");
    expect(pcLabel(3, [0.75])).toBe("PC3");
    expect(fmtP(2.6e-13)).toBe("3e-13");
    expect(fmtP(0.0058)).toBe("0.006");
    expect(fmtP(0.42)).toBe("0.42");
    expect(fmtP(null)).toBe("–");
    expect(r2Shade(null)).toBeUndefined();
    expect(r2Shade(1)).toBe("rgba(37, 99, 235, 0.540)");
  });

  it("colours by PC1's categorical driver, else the split, else the first category", () => {
    expect(defaultColourBy(result())).toBe("stim");
    expect(defaultColourBy(result({ drivers: { PC1: { covariate: "n_cells", r2: 0.9, p: 0.01 } } }))).toBe("stim");
    expect(defaultColourBy(result({ drivers: {}, split_by: null }))).toBe("stim");
    expect(defaultColourBy(result({ covariates: [] }))).toBeNull();
  });

  it("draws a trace per category with outliers ringed and named, one trace for a number", () => {
    const colorOf = () => [1, 2, 3] as const;
    const t = pcaTraces(result(), "stim", colorOf, "Viridis");
    expect(t.map((x) => x.name)).toEqual(["CTRL (2)", "STIM (2)"]);
    expect(t[0]!.x).toEqual([-30, -28]);
    expect(t[0]!.text).toEqual(["", "D2 | CTRL"]);
    expect((t[0]!.marker.line as { color: string[] }).color).toEqual(["#ffffff", "#dc2626"]);
    expect(t[0]!.hovertext[1]).toContain("90 cells");
    const n = pcaTraces(result(), "n_cells", colorOf, "Viridis");
    expect(n).toHaveLength(1);
    expect(n[0]!.marker.color).toEqual([300, 310, 90, 280]);
    expect(n[0]!.marker.showscale).toBe(true);
    const third = pcaTraces(result(), "stim", colorOf, "Viridis", [1, 3]);
    expect(third[0]!.y).toEqual([0.1, 4]);
  });
});
