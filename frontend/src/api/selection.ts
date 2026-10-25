import { apiFetch } from "./client";

/** A selection's mean, mean over expressing cells and fraction expressing, one per gene. */
export interface SelectionStats {
  layer: string;
  genes: string[];
  n_cells: number;
  mean: number[];
  mean_expressing: number[];
  fraction: number[];
}

/**
 * C035: the marker dot plot's / heatmap's "Selection" row, from POST
 * /expression/by-group with the selected cells and `include_groups` false (the
 * groups come from the GET the plot already made).
 */
export async function getSelectionByGroup(
  datasetId: string,
  genes: string[],
  cells: ArrayLike<number>,
  layer?: string,
): Promise<SelectionStats> {
  const body = await apiFetch<{ selection: SelectionStats }>(`/datasets/${datasetId}/expression/by-group`, {
    method: "POST",
    body: JSON.stringify({ genes, cells: Array.from(cells), include_groups: false, ...(layer ? { layer } : {}) }),
  });
  return body.selection;
}
