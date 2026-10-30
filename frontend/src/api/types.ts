import type { PublicSource } from "./publicImport";

export interface ExpressionLayerInfo {
  key: string;
  label: string;
  n_genes: number;
}

export interface DatasetInfo {
  id: string;
  name: string;
  filename: string;
  n_cells: number | null;
  n_genes: number | null;
  status: "pending" | "converting" | "ready" | "error";
  available_embeddings: string[];
  embedding_dimensions?: Record<string, number>;
  obs_columns: ObsColumnInfo[];
  active_clustering?: string;
  /** Hand-label columns made in the app (uns scview_user_annotations) */
  user_annotation_columns?: string[];
  /** Cell-type columns the app's annotator wrote: column -> celltypist | llm | marker_score */
  app_annotation_columns?: Record<string, string>;
  expression_layers?: ExpressionLayerInfo[];
  default_expression_layer?: string;
  /** CITE-seq antibodies (protein modality), or null/absent without them */
  protein?: ProteinInfo | null;
  /** A CRISPR screen's guides (backend core/crispr.available), or null/absent without them */
  perturbation?: { n_guides: number; cellranger: boolean; assigned: boolean; columns: string[] } | null;
  /** Hash of the cell barcodes in order: equal for the same cells (saved sessions) */
  cells_fingerprint?: string;
  created_at: string;
  error_message?: string;
  /** What the R converter kept and left out (an .rds / .RData upload) */
  conversion?: ConversionReport | null;
  /** The converter's latest stage, while status is "converting" */
  conversion_progress?: ConversionProgress | null;
  /** A large dataset (backend core/large_mode.py): its matrices stay on disk; null otherwise */
  large?: LargeDatasetInfo | null;
  /** Where a public dataset came from (a CELLxGENE Discover import), to cite it */
  public_source?: PublicSource | null;
  /** Spatial coordinates (obsm "spatial", tissue images), or null/absent without them */
  spatial?: SpatialInfo | null;
}

/** One tissue image of a spatial library (uns spatial[library].images[res]). */
export interface SpatialImageInfo {
  width: number;
  height: number;
  /** Image pixels per full-resolution pixel (Space Ranger tissue_*_scalef) */
  scalef: number;
}

/** One tissue section (a key of uns spatial; a value of obs library_id). */
export interface SpatialLibrary {
  id: string;
  n_obs: number;
  images: { lowres?: SpatialImageInfo; hires?: SpatialImageInfo };
  /** Spot diameter in full-resolution pixels (Visium), else null */
  spot_diameter_fullres: number | null;
  /** [xmin, ymin, xmax, ymax] of this library's obsm spatial, full-res pixels */
  bounds: [number, number, number, number];
}

/** A dataset's spatial data (backend GET /datasets/{id}, batch 27 contract). */
export interface SpatialInfo {
  technology: "visium" | "visium_hd" | "xenium" | "cells";
  /** What one observation is: a Visium spot holds several cells, a Visium HD bin is a grid square */
  unit: "spot" | "bin" | "cell";
  library_column: string;
  libraries: SpatialLibrary[];
}

/** How a large dataset's on-disk files are (backend AnnDataAdaptor.large_info). */
export interface LargeDatasetInfo {
  matrices_on_disk: boolean;
  /** "ready" | "rebuilding" (in the background) | "missing" (gene views are slow meanwhile) */
  gene_store: "ready" | "rebuilding" | "missing";
  gene_store_note?: string | null;
  /** 0..1 while rebuilding */
  gene_store_progress?: number | null;
  /** Whether the counts sidecar can be used */
  counts: boolean;
  counts_note?: string | null;
}

/** The R converter's report (backend core/conversion.py; also uns scview_conversion). */
export interface ConversionReport {
  /** "SingleCellExperiment", "Seurat", "SpatialExperiment", ... */
  source_class: string;
  source_package?: string;
  /** The class it was converted as, when that differs (a subclass from a package not installed) */
  read_as?: string;
  /** "Seurat v5" */
  detail?: string;
  /** The object's name inside an .RData file */
  object_name?: string;
  n_cells?: number;
  n_genes?: number;
  /** The assay that became X */
  x?: string;
  layers?: string[];
  obsm?: string[];
  /** Sentences: what went where */
  kept: string[];
  /** Sentences: what was left out, and why */
  skipped: string[];
}

export interface ConversionProgress {
  stage: "reading" | "converting" | "writing" | "done" | "failed";
  message: string;
  source_class?: string;
  n_cells?: number;
  n_genes?: number;
  updated_at?: string;
}

export interface ObsColumnInfo {
  name: string;
  dtype: string;
  n_unique: number;
  values?: string[];
  /** Palette slot per value, once a saved order pinned colours to labels (core/category_order.py). */
  color_slots?: number[];
}

export interface EmbeddingInfo {
  name: string;
  dimensions: number;
}

export interface PreprocessingState {
  qc_computed: boolean;
  filtered: boolean;
  normalized: boolean;
  log_transformed: boolean;
  hvgs_selected: boolean;
  scaled: boolean;
  pca_computed: boolean;
  neighbors_computed: boolean;
  clustered: boolean;
  embeddings_computed: boolean;
  markers_computed: boolean;
  cell_cycle_scored: boolean;
  details: Record<string, string>;
}

/** A dataset's protein modality: antibody targets in panel order, read as the
 *  expression layer `layer` (CLR per cell) or `counts_layer` (raw counts). */
export interface ProteinInfo {
  names: string[];
  /** Isotype controls among the names (background, not targets) */
  controls: string[];
  layer: string;
  counts_layer: string;
  normalisation: string;
  /** The DSB layer once computed (backend core/dsb.py), else null */
  dsb_layer?: string | null;
  /** How the DSB values were made, once computed */
  dsb?: {
    mode: "empty_droplets" | "model_negative";
    isotypes: string[];
    n_droplets: number;
    denoise: boolean;
    method: string;
  } | null;
}
