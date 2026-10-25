import { streamEvents } from "./clustering";
import { apiFetch } from "./client";

// --- Minimal marker panel: the fewest genes that still tell the groups apart
// (backend core/marker_panel.py, api/v1/marker_panel.py) ---

export interface PanelStart {
  groupby: string;
  groups: string[];
  n_cells: number;
  n_per_group: Record<string, number>;
  group_sizes: Record<string, number>;
  cells_per_group: number;
  n_candidates: number;
  per_group: number;
  size: number;
  required: string[];
  excluded: string[];
  missing: string[];
  dropped_groups: string[];
  restricted: number | null;
  folds: number;
  seed: number;
  screen: number;
  /** "stratified 3-fold CV, seed 0" */
  cv: string;
  estimate_seconds: number;
  notes: string[];
}

export interface PanelStep {
  gene: string;
  n_genes: number;
  balanced_accuracy: number;
  recall: Record<string, number>;
  required: boolean;
  /** Candidates cross-validated at this step */
  tried: number;
}

export interface BaselinePoint {
  gene: string;
  n_genes: number;
  balanced_accuracy: number;
}

export interface PanelGene {
  gene: string;
  required: boolean;
  /** Groups whose odds it raises most (coefficient >= half its largest) */
  helps: string[];
  /** Groups it is a candidate marker of */
  marker_of: string[];
  /** Standardised coefficient per group */
  coef: Record<string, number>;
}

export interface PanelResult extends PanelStart {
  panel: string[];
  path: PanelStep[];
  baseline: BaselinePoint[];
  genes: PanelGene[];
  confusion: { groups: string[]; counts: number[][] };
  balanced_accuracy: number;
  baseline_accuracy: number | null;
  best_n: number;
  best_accuracy: number;
  within: number;
  importance_note: string;
  seconds: number;
}

export interface PanelRequest {
  groupby: string;
  size?: number;
  per_group?: number;
  required?: string[];
  excluded?: string[];
  restrict?: string[] | null;
}

export const PANEL_SIZE_MAX = 50;
export const PANEL_PER_GROUP_MAX = 50;

/** The latest panel made for this dataset since it was loaded (null: none). */
export function getMarkerPanel(datasetId: string): Promise<{ result: PanelResult | null; running: boolean }> {
  return apiFetch(`/datasets/${datasetId}/markers/panel`);
}

/**
 * Choose the panel, calling `onStart` once, `onStep` per gene added, `onBaseline`
 * and `onResult`. Rejects with the server's reason; aborting `signal` stops
 * reading and, on the server, the run before its next gene.
 */
export async function runMarkerPanel(
  datasetId: string,
  body: PanelRequest,
  handlers: {
    onStart?: (s: PanelStart) => void;
    onStep?: (s: PanelStep) => void;
    onBaseline?: (b: BaselinePoint[]) => void;
    onResult?: (r: PanelResult) => void;
    signal?: AbortSignal;
  },
): Promise<void> {
  await streamEvents(
    `/datasets/${datasetId}/markers/panel`,
    body,
    (event, data) => {
      if (event === "start") handlers.onStart?.(data as PanelStart);
      else if (event === "step") handlers.onStep?.(data as PanelStep);
      else if (event === "baseline") handlers.onBaseline?.((data as { path: BaselinePoint[] }).path);
      else if (event === "result") handlers.onResult?.(data as PanelResult);
    },
    "The marker panel",
    handlers.signal,
  );
}
