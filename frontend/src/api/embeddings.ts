import { apiFetchBinary } from "./client";

/**
 * Fetch embedding coordinates (and optional color-by column) as an Arrow IPC
 * binary stream from the API.
 */
export async function fetchEmbeddingBinary(
  datasetId: string,
  embeddingName: string,
  colorBy?: string,
): Promise<ArrayBuffer> {
  const params = colorBy
    ? `?color_by=${encodeURIComponent(colorBy)}`
    : "";
  return apiFetchBinary(
    `/datasets/${datasetId}/embeddings/${embeddingName}${params}`,
  );
}
