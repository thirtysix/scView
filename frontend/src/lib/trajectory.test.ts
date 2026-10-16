import { describe, it, expect } from "vitest";
import type { ObsColumnInfo } from "@/api/types";
import { defaultRootColumn, rootGroupColumns } from "./trajectory";

const col = (name: string, values: string[] | undefined, dtype = "category"): ObsColumnInfo => ({
  name,
  dtype,
  n_unique: values?.length ?? 5000,
  values,
});

describe("root group pickers", () => {
  const cols = [
    col("scview_leiden_r1.0", ["0", "1", "2"]),
    col("predicted_doublet", ["False", "True"], "bool"),
    col("barcode", undefined, "object"),
    col("cell_type", ["HSC", "Ery", "Mono"]),
  ];

  it("offers categorical columns with listed values, not booleans", () => {
    expect(rootGroupColumns(cols).map((c) => c.name)).toEqual(["scview_leiden_r1.0", "cell_type"]);
  });

  it("prefers a cell-type column, then the active clustering", () => {
    const usable = rootGroupColumns(cols);
    expect(defaultRootColumn(usable, "scview_leiden_r1.0")).toBe("cell_type");
    expect(defaultRootColumn(usable.slice(0, 1), "scview_leiden_r1.0")).toBe("scview_leiden_r1.0");
    expect(defaultRootColumn([], null)).toBe("");
  });
});
