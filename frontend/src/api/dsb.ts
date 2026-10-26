import { apiFetch } from "./client";

/** Droplets per (log10 RNA, log10 protein) bin of a sample's raw matrix (not cells). */
export interface DsbDensity {
  rna_bin: number[];
  prot_bin: number[];
  n: number[];
  bin_width: number;
}

export interface DsbSample {
  sample: string;
  n_cells: number;
  /** Empty droplets inside the default window */
  n_droplets_default: number;
  n_cells_excluded: number;
  /** Panel antibodies the raw matrix does not have (DSB refuses then) */
  missing_antibodies: string[];
  density: DsbDensity;
}

/** The last DSB run (backend core/dsb.py record). */
export interface DsbRecord {
  mode: "empty_droplets" | "model_negative";
  denoise: boolean;
  isotypes: string[];
  rna_window: [number, number] | null;
  prot_window: [number, number] | null;
  n_droplets: number;
  n_cells: number;
  samples: { sample: string; n_cells: number; n_droplets: number }[];
  notes: string[];
  method: string;
}

export interface DsbStatus {
  available: boolean;
  reason: string | null;
  computed: boolean;
  empty_droplets?: boolean;
  empty_droplets_reason?: string | null;
  samples?: DsbSample[];
  isotypes?: string[];
  proteins?: string[];
  defaults?: {
    rna_window: [number, number];
    prot_window: [number, number];
    bin_width: number;
    rna_max: number;
    prot_max: number;
    min_droplets: number;
  };
  record?: DsbRecord | null;
  help?: Record<string, string>;
}

export interface DsbRequest {
  mode?: "auto" | "empty_droplets" | "model_negative";
  rna_window?: [number, number];
  prot_window?: [number, number];
  isotypes?: string[];
  denoise?: boolean;
}

export function getDsbStatus(datasetId: string): Promise<DsbStatus> {
  return apiFetch<DsbStatus>(`/datasets/${datasetId}/protein/dsb`);
}

export function runDsb(datasetId: string, body: DsbRequest): Promise<DsbRecord> {
  return apiFetch<DsbRecord>(`/datasets/${datasetId}/protein/dsb`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
