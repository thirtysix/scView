import { apiFetch } from "./client";

// --- What the AnnData object holds (backend core/structure.py) ---

export interface ValueSample {
  n_sampled: number;
  min: number | null;
  max: number | null;
  integer: boolean;
  negative: boolean;
  n_nan: number;
}

export interface MatrixSummary {
  shape: [number, number];
  dtype: string;
  /** "csr" | "csc" | "dense" | ... */
  format: string;
  on_disk: boolean;
  /** Stored values (sparse); null for a dense matrix */
  nnz: number | null;
  density: number | null;
  sample: ValueSample;
  looks_like_counts: boolean;
  /** .raw only */
  n_vars?: number;
  var_columns?: string[];
}

export interface ColumnSummary {
  name: string;
  dtype: string;
  kind: "categorical" | "numeric" | "bool" | "text";
  n_unique: number;
  n_missing: number;
  all_missing: boolean;
  n_categories?: number;
  min?: number | null;
  max?: number | null;
  examples: (string | number | boolean | null)[];
}

export interface TableSummary {
  n_rows: number;
  n_columns: number;
  columns: ColumnSummary[];
  truncated: boolean;
}

export interface NamesSummary {
  n: number;
  n_duplicated: number;
  duplicated_examples: string[];
  examples: string[];
}

export interface KeyedArray {
  key: string;
  shape: number[] | null;
  dtype: string | null;
  kind: "sparse" | "DataFrame" | "array";
}

export interface UnsNode {
  key: string;
  type: string;
  size?: number;
  shape?: number[] | null;
  dtype?: string | null;
  value?: unknown;
  fields?: string[];
  children?: UnsNode[];
  more?: number;
}

export interface StructureFlag {
  level: "warn" | "info";
  where: string;
  text: string;
}

export interface DatasetStructure {
  n_obs: number;
  n_vars: number;
  mode: "memory" | "backed" | "large";
  file_bytes: number | null;
  X: MatrixSummary | null;
  raw: MatrixSummary | null;
  layers: Record<string, MatrixSummary>;
  sidecars: Record<string, MatrixSummary>;
  obs: TableSummary;
  var: TableSummary;
  obs_names: NamesSummary;
  var_names: NamesSummary;
  obsm: KeyedArray[];
  obsp: KeyedArray[];
  varm: KeyedArray[];
  varp: KeyedArray[];
  uns: UnsNode[];
  sampling: { values: number; rows: number; seed: number };
  flags: StructureFlag[];
}

export function getStructure(datasetId: string): Promise<DatasetStructure> {
  return apiFetch(`/datasets/${datasetId}/structure`);
}
