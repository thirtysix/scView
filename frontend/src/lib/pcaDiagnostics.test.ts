import { describe, it, expect } from "vitest";
import type { PcaDiagnostics } from "@/api/pca";
import { elbowLayout, elbowTraces, fmtPct, loadingTraces, NEGATIVE_COLOR, pcLabel, POSITIVE_COLOR } from "./pcaDiagnostics";

const D: Extract<PcaDiagnostics, { available: true }> = {
  available: true,
  n_cells: 100,
  n_pcs: 4,
  n_genes_total: 50,
  n_genes_used: 30,
  params: { n_comps: 4, use_highly_variable: true, mask_var: "highly_variable", zero_center: true },
  variance: [4, 2, 1, 0.5],
  variance_ratio: [0.2, 0.1, 0.05, 0.025],
  cumulative: [0.2, 0.3, 0.35, 0.375],
  ratio_basis: "total",
  retained: 0.375,
  elbow: 2,
  pcs_90: 3,
  neighbors: { use_rep: "X_pca_harmony", n_pcs: 3, n_neighbors: 15, from_pca: true },
  loadings: [
    { pc: 1, positive: [{ gene: "A", loading: 0.5 }, { gene: "B", loading: 0.3 }], negative: [{ gene: "Z", loading: -0.4 }] },
  ],
  method: "…",
};

describe("elbowTraces / elbowLayout", () => {
  it("draws percentages, highlights the elbow bar and marks elbow + neighbour PCs", () => {
    const [bars, cum] = elbowTraces(D) as { y: number[]; marker: { color: string[] } }[];
    expect(bars!.y).toEqual([20, 10, 5, 2.5]);
    expect(cum!.y[3]).toBeCloseTo(37.5);
    expect(bars!.marker.color[1]).not.toBe(bars!.marker.color[0]); // the elbow bar stands out
    const layout = elbowLayout(D);
    expect(layout.shapes!.map((s) => s.x0)).toEqual([2, 3]);
    expect(layout.annotations!.map((a) => a.text)).toEqual(["elbow PC2", "neighbours: 3 PCs"]);
  });

  it("marks only the elbow when the graph came from another representation", () => {
    const layout = elbowLayout({ ...D, neighbors: { use_rep: "X_scvi", n_pcs: 10, n_neighbors: 15, from_pca: false } });
    expect(layout.shapes).toHaveLength(1);
    expect(elbowLayout({ ...D, neighbors: null }).shapes).toHaveLength(1);
  });
});

describe("loadingTraces", () => {
  it("puts negatives at the bottom and the largest loadings at the outer ends, coloured by sign", () => {
    const [t] = loadingTraces(D.loadings[0]!) as { y: string[]; x: number[]; marker: { color: string[] } }[];
    expect(t!.y).toEqual(["Z", "B", "A"]);
    expect(t!.x).toEqual([-0.4, 0.3, 0.5]);
    expect(t!.marker.color).toEqual([NEGATIVE_COLOR, POSITIVE_COLOR, POSITIVE_COLOR]);
  });
});

describe("labels", () => {
  it("formats shares to a sensible precision", () => {
    expect(fmtPct(0.2)).toBe("20%");
    expect(fmtPct(0.029)).toBe("2.9%");
    expect(fmtPct(0.0062)).toBe("0.62%");
    expect(pcLabel(D, 1)).toBe("PC1 (20%)");
    expect(pcLabel(D, 9)).toBe("PC9");
  });
});
