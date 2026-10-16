import { describe, it, expect } from "vitest";
import type { ObsColumnInfo } from "@/api/types";
import { defaultColorBy, isCategoricalDtype, isGroupingColumn, rankGroupingColumns } from "./obsColumns";

const cat = (name: string, n: number, dtype = "category"): ObsColumnInfo => ({
  name,
  dtype,
  n_unique: n,
  values: Array.from({ length: Math.min(n, 100) }, (_, i) => `${name}${i}`),
});
const num = (name: string): ObsColumnInfo => ({ name, dtype: "float32", n_unique: 900 });
const names = (cols: ObsColumnInfo[], opts = {}) => rankGroupingColumns(cols, { nCells: 1000, ...opts }).map((r) => r.name);

describe("isCategoricalDtype", () => {
  it("treats category, object and bool columns as groupings", () => {
    expect(["category", "object", "bool"].map(isCategoricalDtype)).toEqual([true, true, true]);
    expect(["float32", "int64", undefined].map(isCategoricalDtype)).toEqual([false, false, false]);
  });
});

describe("isGroupingColumn", () => {
  const all = ["leiden", "scview_leiden_r1.0", "barcode"];
  it("needs 2–100 groups and a categorical dtype", () => {
    expect(isGroupingColumn(cat("sample", 1), all, 1000)).toBe(false);
    expect(isGroupingColumn(cat("sample", 2), all, 1000)).toBe(true);
    expect(isGroupingColumn(cat("sample", 100), all, 1000)).toBe(true);
    expect(isGroupingColumn(cat("sample", 101), all, 1000)).toBe(false);
    expect(isGroupingColumn(num("n_genes"), all, 1000)).toBe(false);
  });

  it("never takes an id: one value per cell, or a barcode / cell id by name", () => {
    expect(isGroupingColumn(cat("anything", 95), all, 100)).toBe(false);
    expect(isGroupingColumn(cat("barcode", 50, "object"), all, 1000)).toBe(false);
    expect(isGroupingColumn(cat("cell_id", 50, "object"), all, 1000)).toBe(false);
  });

  it("skips a bare leiden column hidden in favour of its scview_ copy", () => {
    expect(isGroupingColumn(cat("leiden", 10), all, 1000)).toBe(false);
    expect(isGroupingColumn(cat("scview_leiden_r1.0", 10), all, 1000)).toBe(true);
  });
});

describe("rankGroupingColumns", () => {
  it("puts hand labels and cell types first, then clusterings, then other groupings", () => {
    const cols = [
      cat("sample", 8),
      cat("seurat_clusters", 15),
      num("pct_counts_mt"),
      cat("cell_type", 13),
      cat("manual_labels", 3),
      cat("stim", 2),
    ];
    expect(names(cols)).toEqual(["manual_labels", "cell_type", "seurat_clusters", "sample", "stim"]);
  });

  it("recognises the usual cell-type and clustering names", () => {
    const celltypes = [
      "celltype",
      "cell_type",
      "Cell Type",
      "predicted.celltype.l2",
      "broad_celltype",
      "seurat_annotations",
      "majority_voting",
      "predicted_labels",
      "predicted.id",
      "cell_ontology_class",
      "author_annotation",
    ];
    for (const n of celltypes) {
      expect(rankGroupingColumns([cat(n, 5)], { nCells: 1000 })[0]?.kind, n).toBe("cell type");
    }
    const clusterings = ["leiden_r0.5", "louvain", "seurat_clusters", "RNA_snn_res.0.8", "over_clustering", "cluster", "scview_leiden_r1.0_sub"];
    for (const n of clusterings) {
      expect(rankGroupingColumns([cat(n, 5)], { nCells: 1000 })[0]?.kind, n).toBe("clustering");
    }
  });

  it("does not read ontology term ids or *_id columns as cell types", () => {
    const cols = [cat("cell_type_ontology_term_id", 13), cat("donor_id", 8), cat("cell_type", 13)];
    expect(rankGroupingColumns(cols, { nCells: 1000 })).toEqual([
      { name: "cell_type", kind: "cell type" },
      { name: "cell_type_ontology_term_id", kind: "other" },
      { name: "donor_id", kind: "other" },
    ]);
  });

  it("puts the active clustering first among the clusterings", () => {
    const cols = [cat("seurat_clusters", 15), cat("scview_leiden_r1.0", 10)];
    expect(names(cols, { activeClustering: "scview_leiden_r1.0" })).toEqual(["scview_leiden_r1.0", "seurat_clusters"]);
  });

  it("counts the user annotation columns the backend lists as hand labels", () => {
    const cols = [cat("cell_type", 13), cat("my_tcells", 4)];
    expect(names(cols, { userColumns: ["my_tcells"] })).toEqual(["my_tcells", "cell_type"]);
  });

  it("puts boolean flags after other groupings, keeping the dataset's order otherwise", () => {
    const cols = [cat("predicted_doublet", 2, "bool"), cat("sample", 4), cat("phase", 3)];
    expect(names(cols)).toEqual(["sample", "phase", "predicted_doublet"]);
  });
});

describe("defaultColorBy", () => {
  const ds = (cols: ObsColumnInfo[], active?: string) => ({ obs_columns: cols, n_cells: 1000, active_clustering: active });

  it("opens Kang coloured by cell type rather than by the clustering", () => {
    const kang = [cat("orig.ident", 2), cat("label", 2), num("nCount_RNA"), cat("cell_type", 13), cat("scview_leiden_r1.0", 14)];
    expect(defaultColorBy(ds(kang, "scview_leiden_r1.0"))).toBe("cell_type");
  });

  it("falls back to the first grouping, then to the active clustering, then to nothing", () => {
    expect(defaultColorBy(ds([num("n_genes"), cat("sample", 4)]))).toBe("sample");
    // 150 clusters is over the cap, but the previous default coloured by it.
    expect(defaultColorBy(ds([num("n_genes"), cat("leiden_fine", 150)], "leiden_fine"))).toBe("leiden_fine");
    expect(defaultColorBy(ds([num("n_genes")]))).toBe("");
    expect(defaultColorBy(null)).toBe("");
  });

  it("opens on the hand-label columns the dataset info lists, whatever their name", () => {
    const cols = [cat("cell_type", 13), cat("my_tcells", 4)];
    expect(defaultColorBy({ ...ds(cols), user_annotation_columns: ["my_tcells"] })).toBe("my_tcells");
    expect(defaultColorBy(ds(cols))).toBe("cell_type");
  });
});
