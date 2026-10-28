import { API_BASE } from "@/lib/constants";
import { fetchEmbeddingBinary } from "./embeddings";

/** The obsm key of spatial coordinates; served by the embedding endpoint by this name. */
export const SPATIAL_EMBEDDING = "spatial";

export type SpatialImageRes = "lowres" | "hires";

/** The PNG of one library's tissue image (404 when the library has none at `res`). */
export function spatialImageUrl(datasetId: string, libraryId: string, res: SpatialImageRes): string {
  return `${API_BASE}/datasets/${datasetId}/spatial/${encodeURIComponent(libraryId)}/image?res=${res}`;
}

/** Spatial coordinates (x, y in full-resolution pixels, y down) as Arrow, optionally with a column. */
export function fetchSpatialCoords(datasetId: string, colorBy?: string): Promise<ArrayBuffer> {
  return fetchEmbeddingBinary(datasetId, SPATIAL_EMBEDDING, colorBy);
}
