import { describe, expect, it } from "vitest";
import type { ObsColumnInfo } from "@/api/types";
import { labelOverlap, suggestColumnPair } from "./compareColumns";

const col = (name: string, values: string[]): ObsColumnInfo => ({
  name,
  dtype: "category",
  n_unique: values.length,
  values,
});

describe("suggestColumnPair", () => {
  it("pairs a transferred label column with the reference's cell types", () => {
    const a = [col("cell_type", ["B", "CD14 Mono", "NK"]), col("stim", ["CTRL", "STIM"])];
    const b = [col("leiden", ["0", "1"]), col("ref_transfer", ["B", "CD14 Mono", "pDC"])];
    expect(suggestColumnPair(["cell_type", "stim"], ["ref_transfer", "leiden"], a, b)).toEqual([
      "cell_type",
      "ref_transfer",
    ]);
  });

  it("falls back to a shared name, then to each dataset's best column", () => {
    const a = [col("cell_type", ["x", "y"])];
    const b = [col("cell_type", ["p", "q"]), col("other", ["m", "n"])];
    expect(suggestColumnPair(["cell_type"], ["other", "cell_type"], a, b)).toEqual(["cell_type", "cell_type"]);
    expect(suggestColumnPair(["cell_type"], ["other"], a, [b[1]!])).toEqual(["cell_type", "other"]);
  });

  it("ignores case and needs labels on both sides", () => {
    expect(labelOverlap(col("a", ["B", "nk"]), col("b", ["b", "NK"]))).toBe(1);
    expect(labelOverlap(col("a", ["B"]), { name: "b", dtype: "category", n_unique: 3 })).toBe(0);
  });
});
