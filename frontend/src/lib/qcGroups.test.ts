import { describe, it, expect } from "vitest";
import type { QcByGroup } from "@/api/qc";
import type { ObsColumnInfo } from "@/api/types";
import { groupScatterTraces, qcGroupOptions, qcGroupReason } from "./qcGroups";

const col = (name: string, n_unique: number, dtype = "category"): ObsColumnInfo => ({
  name,
  dtype,
  n_unique,
});

describe("qcGroupOptions", () => {
  it("lists grouping columns, disabling those over the cap", () => {
    const opts = qcGroupOptions([
      col("Donor", 8),
      col("ind", 8, "int64"),
      col("barcode", 13836, "object"),
      col("total_counts", 4000, "float32"),
      col("constant", 1),
      col("fine_clusters", 73),
    ]);
    expect(opts).toEqual([
      { name: "Donor", n: 8, disabled: false },
      { name: "ind", n: 8, disabled: false },
      { name: "fine_clusters", n: 73, disabled: true },
    ]);
  });
});

describe("qcGroupReason", () => {
  it("names why the backend picked the column, and nothing for a user's pick", () => {
    expect(qcGroupReason("donor")).toBe("the donor / sample column");
    expect(qcGroupReason("clustering")).toBe("the clustering");
    expect(qcGroupReason("requested")).toBe("");
  });
});

describe("groupScatterTraces", () => {
  const groups = [
    { name: "b", n_cells: 2 },
    { name: "a", n_cells: 1 },
    { name: "empty", n_cells: 0 },
  ];
  const scatter: NonNullable<QcByGroup["scatter"]> = {
    x: [10, 20, 30],
    y: [1, 2, 3],
    group: [0, 1, 0],
    x_label: "total_counts",
    y_label: "n_genes_by_counts",
    n_shown: 3,
  };
  const colours: Record<string, readonly [number, number, number]> = {
    a: [255, 0, 0],
    b: [0, 0, 255],
  };

  it("makes one trace per non-empty group, in category order and colour", () => {
    const traces = groupScatterTraces(scatter, groups, ["a", "b", "empty"], (n) => colours[n] ?? [0, 0, 0]);
    expect(traces.map((t) => t.name)).toEqual(["a", "b"]);
    expect(traces[0]).toMatchObject({ x: [20], y: [2], marker: { color: "rgb(255,0,0)" } });
    expect(traces[1]).toMatchObject({ x: [10, 30], y: [1, 3], marker: { color: "rgb(0,0,255)" } });
    // The name is read by Plotly, never pasted into its template syntax.
    expect(traces[0]!.hovertemplate).toContain("%{fullData.name}");
  });
});
