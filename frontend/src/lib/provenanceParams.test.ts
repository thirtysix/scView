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

  it("summarises the doublet step's channels instead of [object Object]", () => {
    const wells = Array.from({ length: 133 }, (_, i) => ({
      name: String(i + 1), n_cells: 640 + i, expected_doublet_rate: 0.00512 + i * 1.6e-5, n_called: 5, status: "scored",
    }));
    expect(paramText("channels", wells)).toBe("Scrublet per channel: 133 channels, 640-772 cells, expected 0.5-0.7%, 665 called");
    expect(paramText("channels", [{ n_cells: 50, expected_doublet_rate: 0.004, n_called: 0, status: "too_small" }])).toBe(
      "Scrublet per channel: 1 channel, 50 cells, expected 0.4%, 0 called, 1 too small to score",
    );
  });

  it("keeps any other object short", () => {
    const text = paramText("extra", { a: "x".repeat(200) });
    expect(text.startsWith('extra={"a":"xxx')).toBe(true);
    expect(text.length).toBeLessThanOrEqual(86);
  });
});
