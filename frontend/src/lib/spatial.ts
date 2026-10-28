/**
 * The tissue plot (Unified View, Embedding "Tissue"): obsm "spatial" drawn over
 * the section's H&E image. Coordinates arrive in full-resolution image pixels
 * with y growing downwards; the scatter's OrthographicView has y growing
 * upwards, so the tissue world is (x, -y): the image then reads the right way
 * up, and the lasso, labels, fit and exports need no tissue case of their own.
 */
import type { DatasetInfo, SpatialImageInfo, SpatialInfo, SpatialLibrary } from "@/api/types";
import { SPATIAL_EMBEDDING, type SpatialImageRes } from "@/api/spatial";

/** Why Split, Compare and the gene grid are off on the tissue plot (they draw the embedding). */
export const TISSUE_GRID_REASON =
  "Split, Compare and the gene grid show the embedding, not the tissue: pick UMAP under Embedding to use them";

/** The embeddings the chooser offers: the real ones, then "spatial" when the dataset has it. */
export function listedEmbeddings(dataset: DatasetInfo | null | undefined): string[] {
  if (!dataset) return [];
  const real = dataset.available_embeddings;
  return dataset.spatial && !real.includes(SPATIAL_EMBEDDING) ? [...real, SPATIAL_EMBEDDING] : real;
}

/** The dataset's spatial info when `embedding` is the tissue plot, else null. */
export function tissueSpatial(embedding: string, dataset: DatasetInfo | null | undefined): SpatialInfo | null {
  return embedding === SPATIAL_EMBEDDING && dataset?.spatial ? dataset.spatial : null;
}

/** The embedding the cluster map shows beside the tissue: the first real one (UMAP first). */
export function referenceEmbedding(dataset: DatasetInfo | null | undefined): string | null {
  const real = (dataset?.available_embeddings ?? []).filter((e) => e !== SPATIAL_EMBEDDING);
  return ["X_umap", "X_tsne", "X_pca"].find((e) => real.includes(e)) ?? real[0] ?? null;
}

/** "spot" / "spots" for Visium, "cell" / "cells" otherwise. */
export function unitWord(spatial: SpatialInfo | null | undefined, plural = true): string {
  const w = spatial?.unit === "spot" ? "spot" : "cell";
  return plural ? `${w}s` : w;
}

/** The section shown: `wanted` when the dataset has it, else the first. */
export function activeLibrary(spatial: SpatialInfo, wanted: string | null | undefined): SpatialLibrary | null {
  return spatial.libraries.find((l) => l.id === wanted) ?? spatial.libraries[0] ?? null;
}

/** API (x, y down) -> tissue world (x, -y), interleaved. */
export function tissueWorld(raw: Float32Array): Float32Array {
  const out = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i += 2) {
    out[i] = raw[i]!;
    out[i + 1] = -raw[i + 1]!;
  }
  return out;
}

/**
 * Only one library's cells: the others get NaN positions, so they are not
 * drawn, not fitted, not lassoed and not labelled (their colours stay theirs).
 * `codes` are the library column's category codes per cell.
 */
export function maskToLibrary(positions: Float32Array, codes: ArrayLike<number>, code: number): Float32Array {
  const out = new Float32Array(positions);
  const n = Math.min(codes.length, positions.length / 2);
  for (let i = 0; i < n; i++) {
    if (codes[i] !== code) {
      out[i * 2] = NaN;
      out[i * 2 + 1] = NaN;
    }
  }
  return out;
}

/**
 * Where an image sits in the tissue world, as deck.gl BitmapLayer bounds
 * [left, bottom, right, top]: image pixel (r, c) is full-res (c / scalef,
 * r / scalef), so the image spans width / scalef by height / scalef, its top
 * row at y = 0.
 */
export function imageBounds(img: SpatialImageInfo): [number, number, number, number] {
  return [0, -img.height / img.scalef, img.width / img.scalef, 0];
}

/**
 * Whether the camera has zoomed in past the lowres image's own pixels (one
 * lowres pixel wider than one screen pixel): then the hires image is worth
 * loading. Without a lowres image the hires one is the only one.
 */
export function wantsHires(zoom: number, images: SpatialLibrary["images"]): boolean {
  if (!images.hires) return false;
  if (!images.lowres) return true;
  return 2 ** zoom / images.lowres.scalef > 1;
}

/**
 * The image to draw: lowres, hires once it is wanted (or when it is the only
 * one), none without either. deck.gl keeps drawing the lowres image while the
 * hires one loads (an async prop keeps its last value), so there is no gap.
 */
export function imageResFor(images: SpatialLibrary["images"], hires: boolean): SpatialImageRes | null {
  if (images.hires && (hires || !images.lowres)) return "hires";
  return images.lowres ? "lowres" : null;
}

/**
 * A spot's radius in world units (full-res pixels): half the Visium spot
 * diameter at the default point size 2, scaled by the size control; null
 * without a known diameter (cells are drawn in screen pixels, like the scatter).
 */
export function spotRadius(library: SpatialLibrary | null, pointSize: number): number | null {
  const d = library?.spot_diameter_fullres;
  return d != null && d > 0 ? (d / 2) * (pointSize / 2) : null;
}
