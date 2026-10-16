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
  expression_layers?: ExpressionLayerInfo[];
  default_expression_layer?: string;
  /** Hash of the cell barcodes in order: equal for the same cells (saved sessions) */
  cells_fingerprint?: string;
  created_at: string;
  error_message?: string;
}

export interface ObsColumnInfo {
  name: string;
  dtype: string;
  n_unique: number;
  values?: string[];
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
