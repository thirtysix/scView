import { describe, expect, it } from "vitest";
import type { ConversionReport } from "@/api/types";
import { commitConversionNote, conversionHeadline, conversionNote, isRFile, objectPhrase } from "./conversion";

describe("commitConversionNote", () => {
  it("names the file, which of how many, and the converter's stage", () => {
    expect(commitConversionNote(null)).toBeNull();
    expect(commitConversionNote({ file: "obj.rds", index: 1, of: 1, progress: null }))
      .toBe("obj.rds: Reading the R file… a large object takes a few minutes.");
    expect(commitConversionNote({
      file: "b/sce.rds", index: 2, of: 3,
      progress: { stage: "converting", message: "Converting a Seurat object", source_class: "Seurat" },
    })).toBe("b/sce.rds (2 of 3): Converting a Seurat object…");
  });
});

describe("isRFile", () => {
  it("takes .rds and .RData in any case, nothing else", () => {
    expect(isRFile("kang.rds")).toBe(true);
    expect(isRFile("objects.RData")).toBe(true);
    expect(isRFile("kang.RDS")).toBe(true);
    expect(isRFile("kang.h5ad")).toBe(false);
    expect(isRFile("rds")).toBe(false);
  });
});

describe("objectPhrase", () => {
  it("picks the article and names Seurat as an object", () => {
    expect(objectPhrase("SingleCellExperiment")).toBe("a SingleCellExperiment");
    expect(objectPhrase("Seurat")).toBe("a Seurat object");
    expect(objectPhrase("RangedSummarizedExperiment")).toBe("a RangedSummarizedExperiment");
    expect(objectPhrase("AnnDataLike")).toBe("an AnnDataLike");
  });
});

describe("conversionNote", () => {
  it("says the file is being read before the class is known", () => {
    expect(conversionNote(null)).toBe("Reading the R file… a large object takes a few minutes.");
    expect(conversionNote({ stage: "reading", message: "Reading the R file" })).toMatch(/^Reading the R file/);
  });

  it("names the class and size once the converter has read it", () => {
    expect(
      conversionNote({
        stage: "converting", message: "Converting a SingleCellExperiment", source_class: "SingleCellExperiment",
        n_cells: 13836, n_genes: 14053,
      }),
    ).toBe("Converting a SingleCellExperiment (13,836 cells × 14,053 features)…");
  });

  it("adds what is being written", () => {
    expect(
      conversionNote({
        stage: "writing", message: "Writing the counts assay", source_class: "SingleCellExperiment",
        n_cells: 60, n_genes: 40,
      }),
    ).toBe("Converting a SingleCellExperiment (60 cells × 40 features): writing the counts assay…");
  });

  it("reads a Seurat conversion (no sizes, no stages)", () => {
    expect(conversionNote({ stage: "converting", message: "Converting a Seurat object", source_class: "Seurat" }))
      .toBe("Converting a Seurat object…");
  });
});

describe("conversionHeadline", () => {
  const base: ConversionReport = { source_class: "SingleCellExperiment", kept: [], skipped: [] };

  it("gives the class and the size", () => {
    expect(conversionHeadline({ ...base, n_cells: 13836, n_genes: 14053 }))
      .toBe("Converted from a SingleCellExperiment: 13,836 cells × 14,053 genes.");
  });

  it("names the object from an .RData file and Seurat's version", () => {
    expect(conversionHeadline({ ...base, object_name: "sce", n_cells: 25, n_genes: 12 }))
      .toBe("Converted from a SingleCellExperiment (object sce in the .RData file): 25 cells × 12 genes.");
    expect(conversionHeadline({ ...base, source_class: "Seurat", detail: "Seurat v5" }))
      .toBe("Converted from a Seurat object (Seurat v5).");
  });
});
