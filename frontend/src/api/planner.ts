import { apiFetch } from "./client";
import { streamEvents } from "./clustering";

/** The pilot columns every planner request names. */
export interface PilotRequest {
  /** Cell types (or clusters). */
  group_key: string;
  /** Left out: every cell is one condition (donors only). */
  condition_key?: string;
  /** Donor / sample column; left out, the backend picks the most donor-like one. */
  sample_key?: string;
  reference?: string;
  /** [reference, tested]; default: the column's first two values. */
  conditions?: string[];
}

export interface PilotGroup {
  group: string;
  share: number;
  n_cells: number;
}

export interface PilotInfo {
  group_key: string;
  condition_key: string | null;
  sample_key: string | null;
  sample_key_detected: boolean;
  conditions: string[];
  /** "paired" | "unpaired" | "single" (one condition) | "none" (no replicates). */
  design: "paired" | "unpaired" | "single" | "none";
  replicated: boolean;
  n_samples: Record<string, number>;
  n_donors: number;
  cells_per_sample: { median: number; min: number; max: number };
  groups: PilotGroup[];
  /** Abundant, mid, rare: the default cell types to plan for. */
  suggested: string[];
  /** Why there is nothing to plan from, when not replicated. */
  message: string | null;
  assumed_sd_default: number;
  assumed_sd_source: string;
  dropped_samples: string[];
}

export interface CompositionPlanRequest extends PilotRequest {
  groups?: string[];
  fold_changes?: number[];
  donors?: number[];
  cells?: number[];
  alpha?: number;
  target_power?: number;
  design?: "paired" | "unpaired";
  reps?: number;
  /** A donor-to-donor SD of the log-odds share to plan with instead of the pilot's. */
  assumed_sd?: number;
}

export interface DEPlanRequest extends PilotRequest {
  group: string;
  log2fc?: number[];
  donors?: number[];
  /** Cells of that type per donor. */
  cells?: number[];
  de_share?: number;
  alpha?: number;
  target_power?: number;
  design?: "paired" | "unpaired";
  reps?: number;
}

export interface PlanPilotSummary {
  design: PilotInfo["design"];
  n_samples: Record<string, number>;
  n_donors: number;
  cells_per_sample: { median: number; min: number; max: number };
  dropped_samples: string[];
}

export interface CompositionType {
  group: string;
  share: number;
  share_tested: number | null;
  n_cells: number;
  sd_within: number;
  sd_donor: number;
  sd_total: number;
  /** The spread the simulated design uses (within-donor when paired, total when not). */
  sd_used: number;
  /** 95% interval of the within-donor SD (pilot spreads only). */
  sd_low: number | null;
  sd_high: number | null;
  /** 95% interval of the spread the design uses; the dashed curves use its upper end. */
  sd_low_used: number | null;
  sd_upper_used: number | null;
  targeted: boolean;
}

export interface CompositionPoint {
  group: string;
  fold_change: number;
  donors: number;
  cells: number;
  /** null: the fold change would push the type past 98% of the cells. */
  power: number | null;
  se: number | null;
  power_raw: number | null;
  /** At the upper end of the spread's 95% interval (the dashed curve). */
  pessimistic: boolean;
}

export interface NeededRow {
  cells: number;
  donors_needed: number | null;
}

export interface CompositionNeeded extends NeededRow {
  group: string;
  fold_change: number;
  pessimistic: boolean;
  feasible: boolean;
}

export interface CompositionPlan {
  kind: "composition";
  group_key: string;
  condition_key: string | null;
  sample_key: string | null;
  sample_key_detected: boolean;
  conditions: string[];
  reference: string | null;
  /** "pilot": spreads estimated from the dataset; "assumed": an entered one. */
  source: "pilot" | "assumed";
  assumed_sd: number | null;
  assumed_sd_default: number;
  assumed_sd_source: string;
  pilot: PlanPilotSummary & { df: number | null };
  design: "paired" | "unpaired";
  can_pair: boolean;
  alpha: number;
  target_power: number;
  reps: number;
  donors: number[];
  cells: number[];
  fold_changes: number[];
  middle_cells: number;
  types: CompositionType[];
  points: CompositionPoint[];
  needed: CompositionNeeded[];
  assumptions: string[];
  notes: string[];
  elapsed_s: number;
}

export interface DEPoint {
  log2fc: number;
  donors: number;
  cells: number;
  power: number;
  se: number;
  /** False discoveries / discoveries in the simulated studies. */
  fdp: number | null;
  /** Share of the changed genes passing the app's count filter. */
  tested_share: number;
  /** Power for the lowest, middle and highest third of genes by mean count per cell. */
  power_by_expression: number[];
}

export interface DEPlan {
  kind: "de";
  group_key: string;
  group: string;
  condition_key: string | null;
  sample_key: string | null;
  sample_key_detected: boolean;
  conditions: string[];
  reference: string | null;
  pilot: PlanPilotSummary & {
    n_tested: number;
    n_simulated: number;
    counts_source: string;
    n_significant: number | null;
  };
  dispersion: { pilot_median: number; bio_median: number; cell_median: number; donor_sd_median: number };
  expression_tertiles: number[];
  design: "paired" | "unpaired";
  can_pair: boolean;
  log2fc: number[];
  donors: number[];
  cells: number[];
  de_share: number;
  alpha: number;
  target_power: number;
  reps: number;
  points: DEPoint[];
  needed: (NeededRow & { log2fc: number })[];
  assumptions: string[];
  notes: string[];
  elapsed_s: number;
}

export interface PlanStart {
  kind: "composition" | "de";
  n_points: number;
  reps: number;
  design: string;
}

export interface PlanProgress {
  done: number;
  total: number;
  donors: number;
  cells: number;
  power: number | null;
}

/** What a plan would use: design, samples, shares, and whether there are replicates. */
export function planPilot(datasetId: string, body: PilotRequest): Promise<PilotInfo> {
  return apiFetch<PilotInfo>(`/datasets/${datasetId}/planner/pilot`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

type Handlers = {
  onStart?: (s: PlanStart) => void;
  onProgress?: (p: PlanProgress) => void;
  signal?: AbortSignal;
};

async function runPlan<T>(path: string, body: unknown, what: string, h: Handlers): Promise<T> {
  let result: T | null = null;
  await streamEvents(
    path,
    body,
    (event, data) => {
      if (event === "start") h.onStart?.(data as PlanStart);
      else if (event === "progress") h.onProgress?.(data as PlanProgress);
      else if (event === "result") result = data as T;
    },
    what,
    h.signal,
  );
  if (!result) throw new Error(`${what} finished without a result.`);
  return result;
}

/** Simulate the composition power grid; resolves with the plan. Aborting stops reading. */
export function runCompositionPlan(datasetId: string, body: CompositionPlanRequest, h: Handlers = {}) {
  return runPlan<CompositionPlan>(`/datasets/${datasetId}/planner/composition-stream`, body, "The study plan", h);
}

/** Simulate the pseudobulk DE power grid for one cell type. */
export function runDEPlan(datasetId: string, body: DEPlanRequest, h: Handlers = {}) {
  return runPlan<DEPlan>(`/datasets/${datasetId}/planner/de-stream`, body, "The study plan", h);
}
