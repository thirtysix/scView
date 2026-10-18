import { apiFetchBinary } from "./client";
import { decodeArrowBuffer } from "@/lib/arrow";

/**
 * Per-cell expression of several genes in one request (`GET /expression`,
 * Arrow IPC, one float column per gene). Genes the layer lacks are left out of
 * the result; the request fails (404) when none are found.
 */
export async function fetchGeneExpression(
  datasetId: string,
  genes: string[],
  layer?: string,
): Promise<Record<string, Float32Array>> {
  const layerQuery = layer ? `&layer=${encodeURIComponent(layer)}` : "";
  const buffer = await apiFetchBinary(
    `/datasets/${datasetId}/expression?genes=${genes.map(encodeURIComponent).join(",")}${layerQuery}`,
  );
  const decoded = decodeArrowBuffer(buffer);
  const out: Record<string, Float32Array> = {};
  for (const [gene, values] of Object.entries(decoded)) {
    out[gene] = values instanceof Float32Array ? values : Float32Array.from(values as ArrayLike<number>);
  }
  return out;
}
