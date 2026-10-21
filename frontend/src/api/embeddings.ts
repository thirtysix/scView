import { apiFetchBinary } from "./client";

/**
 * Fetch embedding coordinates (and optional color-by column) as an Arrow IPC
 * binary stream from the API.
 */
export async function fetchEmbeddingBinary(
  datasetId: string,
  embeddingName: string,
  colorBy?: string,
  /** Two 0-based columns of a many-column embedding (a PCA) to serve as x and y. */
  dims?: [number, number],
): Promise<ArrayBuffer> {
  const q = new URLSearchParams();
  if (colorBy) q.set("color_by", colorBy);
  if (dims) q.set("dims", `${dims[0]},${dims[1]}`);
  const params = q.size ? `?${q}` : "";
  return apiFetchBinary(
    `/datasets/${datasetId}/embeddings/${embeddingName}${params}`,
  );
}
