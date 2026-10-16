/** Obs column helpers shared by the scatter's pickers, tooltip and labels. */
import type { DatasetInfo, ObsColumnInfo } from "@/api/types";
import { isRedundantClusterCol } from "@/lib/formatting";

/** A column the scatter colours by category (codes), not by value. */
export function isCategoricalDtype(dtype: string | undefined): boolean {
  return dtype === "category" || dtype === "object" || dtype === "bool";
}

// ---------------------------------------------------------------------------
// Default colour-by: which grouping a dataset opens coloured by
// ---------------------------------------------------------------------------

export const GROUPING_MIN_VALUES = 2;
export const GROUPING_MAX_VALUES = 100;

/** Barcodes and cell ids by name; one value per cell is caught by count. */
const ID_NAME = /^(_?index|barcodes?|cell_?(ids?|names?|barcodes?)|obs_?names?)$/i;

/**
 * A categorical column with 2–100 groups that is not an id (one value per
 * cell, or named like barcodes) nor a bare leiden/louvain column hidden in
 * favour of its scview_<method>_r<res> copy: a column worth colouring by.
 */
export function isGroupingColumn(col: ObsColumnInfo, allNames: readonly string[], nCells: number): boolean {
  if (!isCategoricalDtype(col.dtype)) return false;
  const n = col.n_unique;
  if (n < GROUPING_MIN_VALUES || n > GROUPING_MAX_VALUES) return false;
  if (nCells > 0 && n / nCells >= 0.9) return false;
  if (ID_NAME.test(col.name)) return false;
  return !isRedundantClusterCol(col.name, allNames as string[]);
}

export type GroupingKind = "user labels" | "cell type" | "clustering" | "other";

export interface RankedGrouping {
  name: string;
  kind: GroupingKind;
}

// Cell-type annotations: cell_type / celltype / *_celltype / predicted.celltype.l2
// (Azimuth), seurat_annotations (pbmc3k), CellTypist's majority_voting and
// predicted_labels, Seurat label transfer's predicted.id, cell_ontology_class.
const CELL_TYPE = [
  /cell[\s._-]?types?/i,
  /^seurat_annotations?$/i,
  /^majority_voting$/i,
  /^predicted_labels$/i,
  /^predicted[._]id$/i,
  /^cell_ontology_class$/i,
  /annotations?$/i,
];
// Ontology term ids and other *_id columns are codes, not names to read.
const ID_SUFFIX = /(^|[._\s-])(ontology[._\s-]?(term[._\s-]?)?)?ids?$/i;
const CLUSTERING = [/leiden|louvain/i, /^seurat_clusters$/i, /snn_res/i, /cluster/i, /_sub$/];
// Hand labels the app makes are named manual_labels, manual_labels_2, ...
const USER_LABELS = /^manual_labels(_\d+)?$/i;

function kindOf(name: string, userColumns: ReadonlySet<string>, activeClustering: string): {
  kind: GroupingKind;
  /** Order within the kind: lower first */
  priority: number;
} {
  if (userColumns.has(name) || USER_LABELS.test(name)) return { kind: "user labels", priority: 0 };
  const isId = ID_SUFFIX.test(name) && !/^predicted[._]id$/i.test(name);
  if (!isId && CELL_TYPE.some((re) => re.test(name))) return { kind: "cell type", priority: 0 };
  if (name === activeClustering) return { kind: "clustering", priority: 0 };
  if (!isId && CLUSTERING.some((re) => re.test(name))) return { kind: "clustering", priority: 1 };
  return { kind: "other", priority: 0 };
}

const KIND_ORDER: Record<GroupingKind, number> = { "user labels": 0, "cell type": 1, clustering: 2, other: 3 };

export interface RankOptions {
  nCells?: number | null;
  /** The dataset's active clustering: first among the clusterings */
  activeClustering?: string | null;
  /** User annotation columns (uns scview_user_annotations), when known */
  userColumns?: readonly string[];
}

/**
 * The columns a dataset can be coloured by, best first: hand labels and
 * cell-type annotations, then clusterings (the active one first), then any
 * other grouping. Booleans (flags such as predicted doublets) come last.
 * Numeric and id-like columns are left out. Ties keep the dataset's order.
 */
export function rankGroupingColumns(columns: readonly ObsColumnInfo[], opts: RankOptions = {}): RankedGrouping[] {
  const allNames = columns.map((c) => c.name);
  const nCells = opts.nCells ?? 0;
  const users = new Set(opts.userColumns ?? []);
  const active = opts.activeClustering ?? "";
  return columns
    .map((col, index) => ({ col, index }))
    .filter(({ col }) => isGroupingColumn(col, allNames, nCells))
    .map(({ col, index }) => {
      const { kind, priority } = kindOf(col.name, users, active);
      const flag = kind === "other" && col.dtype === "bool" ? 1 : 0;
      return { name: col.name, kind, key: [KIND_ORDER[kind], priority + flag, index] as const };
    })
    .sort((a, b) => a.key[0] - b.key[0] || a.key[1] - b.key[1] || a.key[2] - b.key[2])
    .map(({ name, kind }) => ({ name, kind }));
}

/**
 * What a newly opened dataset is coloured by: the best grouping (see
 * rankGroupingColumns), else its active clustering (the previous default,
 * kept for a clustering the ranking leaves out, e.g. over 100 clusters), else
 * nothing ("").
 */
export function defaultColorBy(
  dataset: Pick<DatasetInfo, "obs_columns" | "n_cells" | "active_clustering" | "user_annotation_columns"> | null,
): string {
  if (!dataset) return "";
  const best = rankGroupingColumns(dataset.obs_columns ?? [], {
    nCells: dataset.n_cells,
    activeClustering: dataset.active_clustering,
    userColumns: dataset.user_annotation_columns,
  })[0];
  if (best) return best.name;
  const active = dataset.active_clustering ?? "";
  return active && (dataset.obs_columns ?? []).some((c) => c.name === active) ? active : "";
}
