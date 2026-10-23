import { describe, it, expect } from "vitest";
import { largeDatasetTitle } from "./largeDataset";

describe("largeDatasetTitle", () => {
  it("says the matrices stay on disk", () => {
    const t = largeDatasetTitle({ matrices_on_disk: true, gene_store: "ready", counts: true });
    expect(t).toMatch(/stay on disk/);
    expect(t).not.toMatch(/slow|not available/);
  });

  it("names a rebuild, a missing store and missing counts", () => {
    expect(
      largeDatasetTitle({ matrices_on_disk: true, gene_store: "rebuilding", gene_store_progress: 0.42, counts: true }),
    ).toMatch(/being rebuilt \(42%\)/);
    const t = largeDatasetTitle({
      matrices_on_disk: true,
      gene_store: "missing",
      gene_store_note: "data.genes.h5 is missing",
      counts: false,
      counts_note: "data.counts.h5 holds other cells",
    });
    expect(t).toMatch(/Gene views are slow: data.genes.h5 is missing\./);
    expect(t).toMatch(/Counts are not available: data.counts.h5 holds other cells\./);
  });
});
