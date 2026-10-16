import { describe, it, expect } from "vitest";
import type { ObsColumnInfo } from "@/api/types";
import { replicateColumns, suggestReplicateColumn } from "./de";

const col = (name: string, n_unique: number, dtype = "category"): ObsColumnInfo => ({
  name,
  dtype,
  n_unique,
});

describe("replicateColumns", () => {
  it("keeps low-cardinality non-float columns that are not already in use", () => {
    const cols = [
      col("stim", 2),
      col("ind", 8, "int64"),
      col("cell_type", 13),
      col("total_counts", 5000, "float32"),
      col("barcode", 13836, "object"),
      col("constant", 1),
    ];
    expect(replicateColumns(cols, ["stim", "cell_type"]).map((c) => c.name)).toEqual(["ind"]);
  });
});

describe("suggestReplicateColumn", () => {
  it("prefers a donor column over a sample column", () => {
    expect(suggestReplicateColumn([col("sample_id", 16), col("donor_id", 8)])).toBe("donor_id");
  });

  it("matches ind exactly, not inside other names", () => {
    expect(suggestReplicateColumn([col("kind", 3), col("ind", 8)])).toBe("ind");
    expect(suggestReplicateColumn([col("kind", 3), col("index_group", 4)])).toBe("");
  });

  it("falls back to Seurat's orig.ident", () => {
    expect(suggestReplicateColumn([col("seurat_clusters", 10), col("orig.ident", 6)])).toBe(
      "orig.ident",
    );
  });
});
