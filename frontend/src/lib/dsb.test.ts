import { describe, expect, it } from "vitest";
import type { ProteinInfo } from "@/api/types";
import { dropletsInWindow, dsbNote, normLabel, proteinLayerFor } from "./dsb";

const BASE: ProteinInfo = {
  names: ["CD3", "IgG1"],
  controls: ["IgG1"],
  layer: "protein",
  counts_layer: "protein_counts",
  normalisation: "CLR per cell",
};

describe("dsb helpers", () => {
  it("reads DSB only once it is computed", () => {
    expect(proteinLayerFor(BASE, "dsb")).toBe("protein");
    expect(proteinLayerFor({ ...BASE, dsb_layer: "protein_dsb" }, "dsb")).toBe("protein_dsb");
    expect(proteinLayerFor({ ...BASE, dsb_layer: "protein_dsb" }, "clr")).toBe("protein");
    expect(proteinLayerFor(null, "dsb")).toBeUndefined();
    expect(normLabel("protein_dsb")).toBe("DSB");
    expect(normLabel("protein")).toBe("CLR");
  });

  it("counts the droplets of a window as the backend does: lower edge in, upper out", () => {
    // bins of 0.1: RNA bin 3 = [0.3, 0.4), protein bins 15 and 29 = [1.5, 1.6), [2.9, 3.0)
    const density = { rna_bin: [3, 3, 25, 3, 0], prot_bin: [15, 29, 20, 30, 20], n: [10, 20, 5, 7, 3], bin_width: 0.1 };
    expect(dropletsInWindow(density, [0.3, 2.5], [1.5, 3])).toBe(30);
    expect(dropletsInWindow(density, [0.3, 2.6], [1.5, 3.1])).toBe(42);
    expect(dropletsInWindow(density, [0, 2.5], [1.5, 3])).toBe(33);
  });

  it("says how the values were made", () => {
    expect(dsbNote({ mode: "empty_droplets", isotypes: ["IgG1"], n_droplets: 69162, denoise: true, method: "" })).toBe(
      "DSB against 69,162 empty droplets; per-cell noise removed with isotypes IgG1.",
    );
    expect(dsbNote({ mode: "model_negative", isotypes: [], n_droplets: 0, denoise: true, method: "" })).toMatch(
      /estimated from the cells .* background mean alone \(no isotype controls\)/,
    );
  });
});
