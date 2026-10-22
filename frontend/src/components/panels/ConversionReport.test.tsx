import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import type { ConversionReport as Report } from "@/api/types";
import { ConversionReport } from "./ConversionReport";

const REPORT: Report = {
  source_class: "SingleCellExperiment",
  n_cells: 60,
  n_genes: 40,
  x: "logcounts",
  kept: ["X: the logcounts assay (sparse)", "altExp ADT (5 features) as the protein modality (obsm protein_counts)"],
  skipped: ["altExp ERCC (4 features): not converted (only antibody and hashtag altExps are)"],
};

describe("ConversionReport", () => {
  it("says what was converted, what was kept and what was left out", () => {
    render(<ConversionReport report={REPORT} />);
    expect(screen.getByText("Converted from a SingleCellExperiment: 60 cells × 40 genes.")).toBeTruthy();
    expect(screen.getByText("2 kept, 1 not converted")).toBeTruthy();
    expect(screen.getByText(REPORT.kept[1]!)).toBeTruthy();
    expect(screen.getByText(REPORT.skipped[0]!)).toBeTruthy();
    // something was left out: the list starts open
    expect(screen.getByTestId("conversion-report").querySelector("details")?.open).toBe(true);
  });

  it("keeps the list folded when nothing was left out", () => {
    render(<ConversionReport report={{ ...REPORT, skipped: [] }} />);
    expect(screen.getByText("2 kept, nothing left out")).toBeTruthy();
    expect(screen.queryByText("Not converted")).toBeNull();
    expect(screen.getByTestId("conversion-report").querySelector("details")?.open).toBe(false);
  });

  it("shows a Seurat conversion (no item lists) as one line", () => {
    render(<ConversionReport report={{ source_class: "Seurat", detail: "Seurat v5", kept: [], skipped: [] }} />);
    expect(screen.getByText("Converted from a Seurat object (Seurat v5).")).toBeTruthy();
    expect(screen.getByTestId("conversion-report").querySelector("details")).toBeNull();
  });
});
