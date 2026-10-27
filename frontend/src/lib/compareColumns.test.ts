import { describe, expect, it } from "vitest";
import type { ObsColumnInfo } from "@/api/types";
import { rankGroupingColumns } from "./obsColumns";
import { labelOverlap, suggestColumnPair } from "./compareColumns";

const col = (name: string, values: string[], dtype = "category"): ObsColumnInfo => ({
  name,
  dtype,
  n_unique: values.length,
  values,
});
const pair = (a: ObsColumnInfo[], b: ObsColumnInfo[]) =>
  suggestColumnPair(rankGroupingColumns(a), rankGroupingColumns(b), a, b);

describe("suggestColumnPair", () => {
  it("pairs a transferred label column with the reference's cell types", () => {
    const a = [col("cell_type", ["B", "CD14 Mono", "NK", "pDC"]), col("stim", ["CTRL", "STIM"])];
    const b = [col("leiden", ["0", "1"]), col("ref_transfer", ["B", "CD14 Mono", "pDC", "DC"])];
    expect(pair(a, b)).toEqual(["cell_type", "ref_transfer"]);
  });

  it("never pairs true/false flags or a two-label condition", () => {
    // Live on Kang vs paul15 (b26): predicted_doublet was paired with itself.
    const flags = [col("predicted_doublet", ["False", "True"], "bool"), col("stim", ["CTRL", "STIM"])];
    const a = [col("cluster", ["B", "CD14 Mono", "NK"]), ...flags];
    const b = [col("paul15_clusters", ["1Ery", "2Ery", "7MEP"]), ...flags];
    a.push(col("scview_leiden_r0.5", ["0", "1", "2", "3"]));
    b.push(col("scview_leiden_r0.5", ["0", "1", "2", "3", "4"]));
    expect(pair(a, b)).toEqual(["cluster", "paul15_clusters"]);
  });

  it("falls back to a same-named cell-type column, then to each dataset's best column", () => {
    const a = [col("cell_type", ["x", "y"])];
    const b = [col("cell_type", ["p", "q"]), col("leiden", ["0", "1"])];
    expect(pair(a, b)).toEqual(["cell_type", "cell_type"]);
    expect(pair(a, [b[1]!])).toEqual(["cell_type", "leiden"]);
  });

  it("ignores case and needs labels on both sides", () => {
    expect(labelOverlap(col("a", ["B", "nk"]), col("b", ["b", "NK"]))).toEqual({ shared: 2, share: 1 });
    expect(labelOverlap(col("a", ["B"]), { name: "b", dtype: "category", n_unique: 3 }).shared).toBe(0);
  });
});
