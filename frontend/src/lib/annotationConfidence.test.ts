import { describe, it, expect } from "vitest";
import type { ConfidenceLabelRow, ConfidenceResult } from "@/api/annotation";
import type { DatasetInfo } from "@/api/types";
import {
  confidenceColumns,
  confusedWithText,
  coverageText,
  defaultConfidenceColumn,
  isConfidenceOutput,
  rejectedShare,
  shrinkHint,
  smallerAlpha,
  sortLabelRows,
} from "./annotationConfidence";

const col = (name: string, n_unique: number, dtype = "category") => ({ name, dtype, n_unique });

function dataset(cols: ReturnType<typeof col>[]): Pick<DatasetInfo, "obs_columns" | "n_cells" | "active_clustering" | "user_annotation_columns"> {
  return { obs_columns: cols, n_cells: 1000, active_clustering: "scview_leiden_r0.5" };
}

function row(label: string, shares: Partial<ConfidenceLabelRow> = {}): ConfidenceLabelRow {
  return {
    label, n_cells: 100, confident: 0.9, ambiguous: 0, unsure: 0.05, not_fit: 0.05,
    coverage_held_out: 0.9, confused_with: [], ...shares,
  };
}

describe("confidenceColumns", () => {
  it("offers label columns best first, without the check's outputs or flags", () => {
    const d = dataset([
      col("scview_leiden_r0.5", 10),
      col("cluster", 13),
      col("predicted_doublet", 2, "bool"),
      col("scview_conf_cluster_flag", 4),
      col("scview_conf_cluster_alt", 12),
      col("total_counts", 900, "float32"),
    ]);
    expect(confidenceColumns(d).map((c) => c.name)).toEqual(["scview_leiden_r0.5", "cluster"]);
    expect(isConfidenceOutput("scview_conf_cluster_flag")).toBe(true);
    expect(isConfidenceOutput("cluster")).toBe(false);
    expect(confidenceColumns(null)).toEqual([]);
  });

  it("opens on a cell-type column when there is one, else the best grouping", () => {
    expect(defaultConfidenceColumn(dataset([col("scview_leiden_r0.5", 10), col("cell_type", 13)]))).toBe("cell_type");
    expect(defaultConfidenceColumn(dataset([col("scview_leiden_r0.5", 10), col("Donor", 8)]))).toBe("scview_leiden_r0.5");
    expect(defaultConfidenceColumn(dataset([]))).toBe("");
  });
});

describe("label rows", () => {
  it("puts the labels that fit worst first", () => {
    const rows = [row("B", { unsure: 0.01, not_fit: 0.02 }), row("CD4 Memory T", { unsure: 0.12, not_fit: 0.09 }),
      row("DC", { unsure: 0.04, not_fit: 0.11 })];
    expect(sortLabelRows(rows).map((r) => r.label)).toEqual(["CD4 Memory T", "DC", "B"]);
    expect(rejectedShare(rows[1]!)).toBeCloseTo(0.21);
    expect(rows[0]!.label).toBe("B"); // not sorted in place
  });

  it("lists the nearest other labels with their shares", () => {
    const r = row("DC", { confused_with: [{ label: "CD14 Mono", n_cells: 63, share: 0.136 }, { label: "CD8 T", n_cells: 3, share: 0.007 }] });
    expect(confusedWithText(r)).toBe("CD14 Mono 13.6%, CD8 T 0.7%");
  });
});

describe("coverageText", () => {
  it("states held-out coverage against the target with its sampling spread", () => {
    const r = { coverage: { target: 0.9, held_out: 0.8941, n_held_out: 6911, calibration: 0.9, n_calibration: 6925, sd: 0.0051, within_sampling_error: true } };
    expect(coverageText(r)).toBe("89.4% of 6,911 held-out cells kept their label in their set (target 90.0%; sampling spread ±1.0%).");
  });
});

describe("shrinkHint", () => {
  const counts = (ambiguous: number, unsure: number): ConfidenceResult["counts"] => ({
    confident: 1000, ambiguous, unsure, "doesn't fit": 50,
  });

  it("explains an alpha above the classifier's error rate, and suggests a smaller one", () => {
    const hint = shrinkHint({ alpha: 0.1, accuracy: 0.9219, counts: counts(0, 673) });
    expect(hint).toMatch(/No cell is ambiguous at α = 0.1/);
    expect(hint).toMatch(/92.2% of cells/);
    expect(hint).toMatch(/Try α = 0.05/);
    expect(smallerAlpha({ accuracy: 0.995 })).toBe(0.01);
  });

  it("says nothing when sets grew or nothing is unsure", () => {
    expect(shrinkHint({ alpha: 0.05, accuracy: 0.92, counts: counts(891, 0) })).toBeNull();
    expect(shrinkHint({ alpha: 0.1, accuracy: 0.99, counts: counts(0, 0) })).toBeNull();
  });

  it("suggests no alpha that is not smaller", () => {
    expect(shrinkHint({ alpha: 0.01, accuracy: 0.995, counts: counts(0, 3) })).not.toMatch(/Try/);
  });
});
