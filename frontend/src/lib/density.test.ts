import { describe, it, expect, vi, beforeEach } from "vitest";
import { RecordBatch, Schema, Table, tableFromArrays, tableToIPC } from "apache-arrow";

const apiFetchBinary = vi.fn();
vi.mock("@/api/client", () => ({ apiFetchBinary: (...a: unknown[]) => apiFetchBinary(...a) }));

import {
  clearDensityCache,
  decodeDensity,
  densityLabel,
  densityLegendNote,
  fetchDensity,
  overlayGene,
  overlayWord,
  sameDensityView,
  showDensity,
  type DensityMeta,
} from "./density";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

const META: DensityMeta = {
  genes: ["CD3D", "CD8A"], missing_genes: ["NOPE"], embedding: "X_umap", dims: [0, 1], layer: "raw",
  joint: true, n_cells: 13836, bandwidth: [0.5, 0.6], grid: 256, adjust: 1, n_expressing: [4168, 1200],
  n_negative: 0, n_no_coords: 0,
};

function densityIpc(meta: DensityMeta, values: number[]): ArrayBuffer {
  const t = tableFromArrays({ density: Float32Array.from(values) });
  const schema = new Schema(t.schema.fields, new Map([["scview", JSON.stringify(meta)]]));
  const bytes = tableToIPC(new Table(schema, t.batches.map((b) => new RecordBatch(schema, b.data))), "stream");
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

beforeEach(() => {
  apiFetchBinary.mockReset();
  clearDensityCache();
  useSettingsStore.setState({ embedding: "X_umap", pcaDims: [0, 1], expressionLayer: "raw" });
  useDatasetStore.setState({ currentDataset: null, currentDatasetId: "ds1" });
  useUnifiedViewStore.getState().clearOverlayState();
});

describe("decodeDensity", () => {
  it("reads one value per cell and the facts", () => {
    const out = decodeDensity(densityIpc(META, [0.1, 0.5, 0]));
    expect(Array.from(out.values)).toEqual([Math.fround(0.1), 0.5, 0]);
    expect(out.meta.genes).toEqual(["CD3D", "CD8A"]);
    expect(out.meta.bandwidth).toEqual([0.5, 0.6]);
  });
});

describe("labels", () => {
  it("names one gene's density and several genes' joint density", () => {
    expect(densityLabel(["CD3D"])).toBe("CD3D density");
    expect(densityLabel(["CD3D", "CD8A"])).toBe("CD3D + CD8A joint density");
  });

  it("says what the colour is, with the n", () => {
    const note = densityLegendNote(META);
    expect(note).toContain("weighted density (Nebulosa-style)");
    expect(note).toContain("product of each gene's density over its peak");
    expect(note).toContain("13,836 cells");
    expect(note).toContain("CD3D in 4,168");
    expect(densityLegendNote(null)).toBe("Colour: weighted density (Nebulosa-style).");
  });

  it("words each overlay kind, and finds the one gene shown", () => {
    const d1 = { type: "density" as const, genes: ["CD3D"], embedding: "X_umap", layer: "raw" };
    const d2 = { ...d1, genes: ["CD3D", "CD8A"] };
    expect(overlayWord(d1)).toBe("(Nebulosa-style)");
    expect(overlayWord({ type: "expression", gene: "CD3D" })).toBe("expression");
    expect(overlayWord({ type: "expression", gene: "CD3", layer: "protein" })).toBe("protein");
    expect(overlayWord({ type: "program", index: 1, name: "P1" })).toBe("program");
    expect(overlayWord({ type: "score", name: "set" })).toBe("score");
    expect(overlayGene(d1, "CD3D density")).toBe("CD3D");
    expect(overlayGene(d2, "CD3D + CD8A joint density")).toBeNull();
    expect(overlayGene({ type: "expression", gene: "MS4A1" }, "MS4A1")).toBe("MS4A1");
    expect(overlayGene(null, "")).toBeNull();
  });

  it("compares the plot and layer a density was computed for", () => {
    const o = { embedding: "X_pca", dims: [2, 3] as [number, number], layer: "raw" };
    expect(sameDensityView(o, { embedding: "X_pca", dims: [2, 3], layer: "raw" })).toBe(true);
    expect(sameDensityView(o, { embedding: "X_pca", layer: "raw" })).toBe(false);
    expect(sameDensityView({ embedding: "X_umap", layer: "raw" }, { embedding: "X_umap", dims: [0, 1], layer: "raw" })).toBe(true);
    expect(sameDensityView({ embedding: "X_umap", layer: "raw" }, { embedding: "X_umap", layer: "X" })).toBe(false);
  });
});

describe("fetchDensity", () => {
  it("asks for the genes on the plot and caches the answer", async () => {
    apiFetchBinary.mockResolvedValue(densityIpc(META, [1, 2]));
    const r = { genes: ["CD3D", "CD8A"], embedding: "X_pca", dims: [2, 3] as [number, number], layer: "raw" };
    await fetchDensity("ds1", r);
    await fetchDensity("ds1", r);
    expect(apiFetchBinary).toHaveBeenCalledTimes(1);
    const url = String(apiFetchBinary.mock.calls[0]![0]);
    expect(url).toContain("/datasets/ds1/expression/density?");
    const qs = new URLSearchParams(url.split("?")[1]);
    expect(qs.get("genes")).toBe("CD3D,CD8A");
    expect(qs.get("embedding")).toBe("X_pca");
    expect(qs.get("dims")).toBe("2,3");
    expect(qs.get("layer")).toBe("raw");
    await fetchDensity("ds1", { ...r, embedding: "X_umap", dims: undefined });
    expect(apiFetchBinary).toHaveBeenCalledTimes(2);
  });
});

describe("showDensity", () => {
  it("puts the density on the main plot as the overlay", async () => {
    apiFetchBinary.mockResolvedValue(densityIpc({ ...META, genes: ["CD3D"], joint: false }, [0.2, 0.4]));
    await showDensity("ds1", ["CD3D"]);
    const s = useUnifiedViewStore.getState();
    expect(s.scatterOverlay).toEqual({ type: "density", genes: ["CD3D"], embedding: "X_umap", layer: "raw" });
    expect(s.overlayLabel).toBe("CD3D density");
    expect(s.overlayDatasetId).toBe("ds1");
    expect(Array.from(s.overlayValues!)).toEqual([Math.fround(0.2), Math.fround(0.4)]);
  });

  it("leaves the overlay alone when the server refuses", async () => {
    apiFetchBinary.mockRejectedValue(new Error('API error 404: {"detail":"None of the genes"}'));
    await expect(showDensity("ds1", ["NOPE"])).rejects.toThrow("404");
    expect(useUnifiedViewStore.getState().scatterOverlay).toBeNull();
  });
});
