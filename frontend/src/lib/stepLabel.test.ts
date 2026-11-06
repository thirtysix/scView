import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { STEP_LABELS, stepLabel } from "./stepLabel";

/** The step kinds the backend can record (backend core/step_kinds.py RECORDED_STEPS). */
function recordedSteps(): string[] {
  const py = readFileSync(resolve(__dirname, "../../../backend/src/scview/core/step_kinds.py"), "utf8");
  const start = py.indexOf("RECORDED_STEPS");
  const body = py.slice(start, py.indexOf("\n)\n", start));
  return [...body.matchAll(/^\s+"([a-z0-9_]+)",/gm)].map((m) => m[1]!);
}

describe("stepLabel", () => {
  it("keeps abbreviations in capitals", () => {
    expect(stepLabel("qc_metrics")).toBe("QC metrics");
    expect(stepLabel("pca")).toBe("PCA");
    expect(stepLabel("highly_variable_genes")).toBe("Highly variable genes");
    expect(stepLabel("umap_3d")).toBe("UMAP 3d");
    expect(stepLabel("tsne")).toBe("t-SNE");
  });

  it("names the spatial steps in words", () => {
    expect(stepLabel("spatial_autocorr")).toBe("Spatially variable genes (Moran's I)");
    expect(stepLabel("doublet_detection_skipped")).toBe("Doublet detection skipped");
  });

  it("has its own words for every step kind the backend records", () => {
    const kinds = recordedSteps();
    expect(kinds.length).toBeGreaterThan(60); // the file was read
    expect(kinds).toContain("spatial_autocorr");
    expect(kinds.filter((k) => !STEP_LABELS[k])).toEqual([]);
    // and none the backend no longer records
    expect(Object.keys(STEP_LABELS).filter((k) => !kinds.includes(k))).toEqual([]);
  });

  it("names a skipped step by the step's own label, and spells out a kind from a newer version", () => {
    expect(stepLabel("cell_cycle_skipped")).toBe("Cell cycle skipped");
    expect(stepLabel("ambient_correction_skipped")).toBe("Ambient RNA correction skipped");
    expect(stepLabel("atac_peak_calling")).toBe("ATAC peak calling");
    expect(stepLabel("vdj_attach")).toBe("TCR/BCR contigs attached");
    expect(stepLabel("ingest")).toBe("Imported");
  });
});
