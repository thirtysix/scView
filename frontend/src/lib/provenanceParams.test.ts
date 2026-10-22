import { describe, it, expect } from "vitest";
import { paramText } from "./provenanceParams";

describe("paramText", () => {
  it("shows plain values and lists as before", () => {
    expect(paramText("min_cells", 3)).toBe("min_cells=3");
    expect(paramText("marker_columns", ["leiden", "cell_type"])).toBe("marker_columns=leiden, cell_type");
  });

  it("summarises the filtering step's per-sample thresholds instead of [object Object]", () => {
    expect(paramText("qc_thresholds", { column: "Donor", samples: [{}, {}, {}] })).toBe(
      "per-sample thresholds: Donor (3 samples)",
    );
    expect(paramText("qc_thresholds", { column: null, samples: [] })).toBe("thresholds: all cells pooled");
  });

  it("cuts a long list to its first 12 items and the count", () => {
    const genes = Array.from({ length: 2000 }, (_, i) => `G${i}`);
    expect(paramText("genes", genes)).toBe(`genes=${genes.slice(0, 12).join(", ")}, ... (2,000)`);
    expect(paramText("seeds", [0, 1, 2])).toBe("seeds=0, 1, 2");
  });

  it("keeps any other object short", () => {
    const text = paramText("extra", { a: "x".repeat(200) });
    expect(text.startsWith('extra={"a":"xxx')).toBe(true);
    expect(text.length).toBeLessThanOrEqual(86);
  });
});
