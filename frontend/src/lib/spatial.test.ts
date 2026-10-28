import { describe, it, expect } from "vitest";
import type { DatasetInfo, SpatialInfo, SpatialLibrary } from "@/api/types";
import {
  activeLibrary,
  imageBounds,
  imageResFor,
  listedEmbeddings,
  maskToLibrary,
  referenceEmbedding,
  spotRadius,
  tissueSpatial,
  tissueWorld,
  unitWord,
  wantsHires,
} from "./spatial";

const LIB: SpatialLibrary = {
  id: "V1_Human_Lymph_Node",
  n_obs: 4035,
  images: {
    lowres: { width: 600, height: 591, scalef: 0.0177 },
    hires: { width: 2000, height: 1970, scalef: 0.0590 },
  },
  spot_diameter_fullres: 89.4,
  bounds: [1000, 1200, 30000, 31000],
};
const SPATIAL: SpatialInfo = {
  technology: "visium",
  unit: "spot",
  library_column: "library_id",
  libraries: [LIB, { ...LIB, id: "second", n_obs: 10 }],
};
const ds = (over: Partial<DatasetInfo> = {}) =>
  ({ id: "d", available_embeddings: ["X_pca", "X_umap"], spatial: SPATIAL, ...over }) as unknown as DatasetInfo;

describe("tissue plot helpers", () => {
  it("lists Tissue after the real embeddings, only with spatial data", () => {
    expect(listedEmbeddings(ds())).toEqual(["X_pca", "X_umap", "spatial"]);
    expect(listedEmbeddings(ds({ spatial: null }))).toEqual(["X_pca", "X_umap"]);
    expect(listedEmbeddings(null)).toEqual([]);
    expect(tissueSpatial("spatial", ds())).toBe(SPATIAL);
    expect(tissueSpatial("X_umap", ds())).toBeNull();
    expect(tissueSpatial("spatial", ds({ spatial: null }))).toBeNull();
  });

  it("keeps the UMAP for the cluster map beside the tissue", () => {
    expect(referenceEmbedding(ds())).toBe("X_umap");
    expect(referenceEmbedding(ds({ available_embeddings: ["X_pca"] }))).toBe("X_pca");
    expect(referenceEmbedding(ds({ available_embeddings: [] }))).toBeNull();
  });

  it("says spots for Visium", () => {
    expect(unitWord(SPATIAL)).toBe("spots");
    expect(unitWord(SPATIAL, false)).toBe("spot");
    expect(unitWord({ ...SPATIAL, unit: "cell" })).toBe("cells");
    expect(unitWord(null)).toBe("cells");
  });

  it("places the lowres and the hires image over the same full-res extent, y down", () => {
    const low = imageBounds(LIB.images.lowres!);
    const high = imageBounds(LIB.images.hires!);
    expect(low[0]).toBe(0);
    expect(low[3]).toBe(0);
    expect(low[2]).toBeCloseTo(600 / 0.0177);
    expect(low[1]).toBeCloseTo(-591 / 0.0177);
    expect(high[2]).toBeCloseTo(2000 / 0.059);
    expect(high[1]).toBeCloseTo(-1970 / 0.059);
    // A spot at full-res (x, y) lands on the image pixel (y * scalef, x * scalef).
    const [x, y] = [15000, 20000];
    const [wx, wy] = tissueWorld(Float32Array.from([x, y]));
    const col = ((wx! - low[0]) / (low[2] - low[0])) * 600;
    const row = ((low[3] - wy!) / (low[3] - low[1])) * 591;
    expect(col).toBeCloseTo(x * 0.0177, 1);
    expect(row).toBeCloseTo(y * 0.0177, 1);
  });

  it("loads the hires image once zoomed in past the lowres pixels", () => {
    // One lowres pixel = 1 / 0.0177 world units; on screen 2^zoom of them per px.
    const edge = Math.log2(0.0177);
    expect(wantsHires(edge - 0.5, LIB.images)).toBe(false);
    expect(wantsHires(edge + 0.5, LIB.images)).toBe(true);
    expect(imageResFor(LIB.images, false)).toBe("lowres");
    expect(imageResFor(LIB.images, true)).toBe("hires");
    expect(imageResFor({ hires: LIB.images.hires }, false)).toBe("hires");
    expect(wantsHires(-20, { hires: LIB.images.hires })).toBe(true);
    expect(wantsHires(10, { lowres: LIB.images.lowres })).toBe(false);
    expect(imageResFor({}, true)).toBeNull();
  });

  it("draws a spot at its real radius at size 2, scaled by the size control", () => {
    expect(spotRadius(LIB, 2)).toBeCloseTo(44.7);
    expect(spotRadius(LIB, 4)).toBeCloseTo(89.4);
    expect(spotRadius({ ...LIB, spot_diameter_fullres: null }, 2)).toBeNull();
    expect(spotRadius(null, 2)).toBeNull();
  });

  it("shows one section at a time: the others' cells get no position", () => {
    const pos = Float32Array.from([1, -1, 2, -2, 3, -3]);
    const masked = maskToLibrary(pos, Int32Array.from([0, 1, 0]), 0);
    expect(Array.from(masked.slice(0, 2))).toEqual([1, -1]);
    expect(Number.isNaN(masked[2])).toBe(true);
    expect(Number.isNaN(masked[3])).toBe(true);
    expect(Array.from(masked.slice(4))).toEqual([3, -3]);
    expect(pos[2]).toBe(2); // the input is left as it is
    expect(activeLibrary(SPATIAL, "second")?.id).toBe("second");
    expect(activeLibrary(SPATIAL, "gone")?.id).toBe(LIB.id);
    expect(activeLibrary(SPATIAL, null)?.id).toBe(LIB.id);
  });

  it("turns image coordinates (y down) into the plot's world (y up)", () => {
    expect(Array.from(tissueWorld(Float32Array.from([10, 20, 30, 40])))).toEqual([10, -20, 30, -40]);
  });
});
