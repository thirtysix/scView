import { apiFetchBinary } from "./client";
import { decodeArrowBuffer } from "@/lib/arrow";
import { decodeHeatmapCells, type CellScale, type HeatmapCells } from "@/lib/cellHeatmap";

/**
 * Per-cell values of up to 100 genes in a seeded, stratified sample of cells
 * (`GET /expression/heatmap-cells`, Arrow IPC), for the marker heatmap's
 * "Per cell" view. Fails (400) when none of the genes are in the layer.
 */
export async function fetchHeatmapCells(
  datasetId: string,
  params: { genes: string[]; groupby: string; perGroup: number; scale: CellScale; layer?: string; seed?: number },
): Promise<HeatmapCells> {
  const qs = new URLSearchParams({
    genes: params.genes.join(","),
    groupby: params.groupby,
    per_group: String(params.perGroup),
    scale: params.scale,
    seed: String(params.seed ?? 0),
    ...(params.layer ? { layer: params.layer } : {}),
  });
  return decodeHeatmapCells(await apiFetchBinary(`/datasets/${datasetId}/expression/heatmap-cells?${qs}`));
}

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
