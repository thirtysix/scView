/**
 * Expression-weighted kernel density on the embedding (C087, Nebulosa-style):
 * GET /expression/density gives one value per cell, the joint density for
 * several genes (core/density.py). The bandwidth comes from the coordinates of
 * ALL cells, so every panel of a split view reads one global density on one
 * bandwidth and the panels compare like for like.
 */
import { tableFromIPC } from "apache-arrow";
import { apiFetchBinary } from "@/api/client";
import { embeddingDims } from "@/hooks/useEmbedding";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

/** Most genes in one joint density (the server's MAX_GENES). */
export const DENSITY_MAX_GENES = 9;

/** The colour scale's name in legends and captions. */
export const DENSITY_SCALE_NAME = "weighted density (Nebulosa-style)";

/** What the density is, for the InfoTips by the toggle and the legend. */
export const DENSITY_TIP =
  "Each cell is coloured by a kernel density of the cells around it on this plot, weighted by their " +
  "expression (Nebulosa, Alquicira-Hernandez & Powell 2021). It smooths: a gene caught in only a few " +
  "cells of a population (dropouts) still lights up the whole region, and one stray expressing cell " +
  "fades. So it can show signal where few cells express, but it also blurs small groups into their " +
  "neighbours and depends on the plot's layout. Joint density of several genes = the product of each " +
  "gene's density over its peak: high only where all of them are. Computed once over all cells, so " +
  "the panels of a split view share one bandwidth.";

export interface DensityMeta {
  genes: string[];
  missing_genes: string[];
  embedding: string;
  dims: [number, number];
  layer: string;
  joint: boolean;
  n_cells: number;
  bandwidth: [number, number];
  grid: number;
  adjust: number;
  /** Cells with a value above 0, per gene */
  n_expressing: number[];
  n_negative: number;
  n_no_coords: number;
}

export interface DensityResult {
  values: Float32Array;
  meta: DensityMeta;
}

/** Decode the endpoint's Arrow stream: one float column, the facts in schema metadata. */
export function decodeDensity(buffer: ArrayBuffer | Uint8Array): DensityResult {
  const table = tableFromIPC(buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer));
  const raw = table.schema.metadata.get("scview");
  if (!raw) throw new Error("The density response has no description.");
  const col = table.getChildAt(0)?.toArray();
  const values = col instanceof Float32Array ? col : Float32Array.from((col ?? []) as ArrayLike<number>);
  return { values, meta: JSON.parse(raw) as DensityMeta };
}

export interface DensityRequest {
  genes: string[];
  embedding: string;
  /** The PCA pair plotted, when not the first two columns */
  dims?: [number, number];
  layer?: string;
}

const CACHE_SIZE = 8;
const CACHE_MS = 5 * 60 * 1000;
const cache = new Map<string, { at: number; result: DensityResult }>();

function cacheKey(datasetId: string, r: DensityRequest): string {
  return JSON.stringify([datasetId, r.genes, r.embedding, r.dims ?? null, r.layer ?? ""]);
}

/** Forget cached densities (a dataset was reprocessed; tests). */
export function clearDensityCache(): void {
  cache.clear();
}

/** GET /expression/density, the last few answers cached (5 min) like the gene columns. */
export async function fetchDensity(datasetId: string, r: DensityRequest): Promise<DensityResult> {
  const key = cacheKey(datasetId, r);
  const hit = cache.get(key);
  if (hit && Date.now() - hit.at < CACHE_MS) {
    cache.delete(key); // most recent last
    cache.set(key, hit);
    return hit.result;
  }
  const qs = new URLSearchParams({ genes: r.genes.join(","), embedding: r.embedding });
  if (r.dims) qs.set("dims", r.dims.join(","));
  if (r.layer) qs.set("layer", r.layer);
  const result = decodeDensity(await apiFetchBinary(`/datasets/${datasetId}/expression/density?${qs}`));
  cache.set(key, { at: Date.now(), result });
  while (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value!);
  return result;
}

/** "CD3D density", "CD3D + CD8A joint density". */
export function densityLabel(genes: string[]): string {
  return `${genes.join(" + ")} ${genes.length > 1 ? "joint density" : "density"}`;
}

/** The legend's line under the scale: what the values are, with the n. */
export function densityLegendNote(meta: DensityMeta | null): string {
  if (!meta) return `Colour: ${DENSITY_SCALE_NAME}.`;
  const expressing = meta.genes
    .map((g, i) => `${g} in ${(meta.n_expressing[i] ?? 0).toLocaleString("en-US")}`)
    .join(", ");
  return (
    `Colour: ${DENSITY_SCALE_NAME}${meta.joint ? ", the product of each gene's density over its peak" : ""}; ` +
    `over all ${meta.n_cells.toLocaleString("en-US")} cells (${expressing} expressing).`
  );
}

type Overlay = ReturnType<typeof useUnifiedViewStore.getState>["scatterOverlay"];

/** The word after an overlay's name in the chip, hover and info row. */
export function overlayWord(o: Overlay): string {
  if (!o) return "";
  if (o.type === "program") return "program";
  if (o.type === "density") return "(Nebulosa-style)";
  if (o.type !== "expression") return "score";
  return o.layer ? "protein" : "expression";
}

/** The one gene an overlay shows (expression or a single gene's density), else null. */
export function overlayGene(o: Overlay, overlayLabel: string): string | null {
  if (o?.type === "expression") return overlayLabel || null;
  if (o?.type === "density" && o.genes.length === 1) return o.genes[0]!;
  return null;
}

/** The plot and layer a density request is for, now (settings + dataset stores). */
export function currentDensityView(): { embedding: string; dims?: [number, number]; layer: string } {
  const { embedding, pcaDims, expressionLayer } = useSettingsStore.getState();
  const nDims = useDatasetStore.getState().currentDataset?.embedding_dimensions?.[embedding];
  return { embedding, dims: embeddingDims(embedding, pcaDims, nDims), layer: expressionLayer };
}

/** Whether a density overlay was computed for this plot and layer. */
export function sameDensityView(
  o: { embedding: string; dims?: [number, number]; layer: string },
  view: { embedding: string; dims?: [number, number]; layer: string },
): boolean {
  return (
    o.embedding === view.embedding &&
    o.layer === view.layer &&
    (o.dims?.[0] ?? 0) === (view.dims?.[0] ?? 0) &&
    (o.dims?.[1] ?? 1) === (view.dims?.[1] ?? 1)
  );
}

/** The last density drawn on the main plot (for its legend note). */
let lastMeta: DensityMeta | null = null;
export function lastDensityMeta(): DensityMeta | null {
  return lastMeta;
}

/**
 * The density of `genes` on the plot on screen, onto the main scatter as the
 * overlay. Returns the facts; throws when the server refuses (none of the
 * genes in the layer, a large dataset without its gene store, ...).
 */
export async function showDensity(datasetId: string, genes: string[]): Promise<DensityMeta> {
  const view = currentDensityView();
  const res = await fetchDensity(datasetId, { genes, ...view, layer: view.layer || undefined });
  const store = useUnifiedViewStore.getState();
  lastMeta = res.meta;
  store.setOverlayValues(res.values);
  store.setOverlayLabel(densityLabel(res.meta.genes));
  store.setScatterOverlay({
    type: "density",
    genes: res.meta.genes,
    embedding: view.embedding,
    ...(view.dims ? { dims: view.dims } : {}),
    layer: view.layer,
  });
  store.setOverlayDatasetId(datasetId);
  return res.meta;
}
