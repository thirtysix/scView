import { apiFetch } from "./client";

/** One gene's share of a sample's soup (0..1). */
export interface SoupGene {
  gene: string;
  share: number;
}

/** One sample's soup and contamination fraction (backend core/ambient.py). */
export interface AmbientSample {
  sample: string;
  column: string | null;
  n_cells: number;
  /** "empty_droplets": from the raw matrix; "cells": the cells' own counts (approximate). */
  source: "empty_droplets" | "cells";
  top_genes: SoupGene[];
  genes_covered: number;
  n_droplets?: number;
  n_umis?: number;
  umi_max?: number;
  raw_files?: string[];
  rho: number | null;
  rho_low: number | null;
  rho_high: number | null;
  /** ok | no_clusters | few_clusters | no_markers | no_soup */
  rho_status: string;
  n_markers: number;
  n_estimates: number;
  rho_genes: string[];
  notes: string[];
}

/** A top marker whose signal outside its cluster looks ambient. */
export interface AmbientMarker {
  gene: string;
  /** Its rank in the soup (1 = the soup's top gene). */
  soup_rank: number;
  /** Observed / expected (rho x soup) counts outside the cluster; null when approximate. */
  ratio: number | null;
}

/** One sample's record of the ambient_correction step. */
export interface AmbientCorrectionSample {
  sample: string;
  rho: number;
  rho_source: "estimated" | "set";
  removed_share?: number;
  n_cells?: number;
}

export interface AmbientResponse {
  available: boolean;
  approximate: boolean;
  column: string | null;
  counts_from: string;
  samples: AmbientSample[];
  markers: {
    column: string | null;
    approximate: boolean;
    groups: Record<string, AmbientMarker[]>;
    n_flagged: number;
  };
  /** A targeted panel: no empty droplets to measure a soup in; `notes` says so. */
  not_applicable?: boolean;
  hb_note: string | null;
  notes: string[];
  /** The last correction's record when the counts were corrected, else null. */
  correction: { samples?: AmbientCorrectionSample[]; removed_share?: number; ambient_rho?: number | null } | null;
  help: Record<string, string>;
  method: string;
  url: string;
}

/** GET /datasets/{id}/qc/ambient (a large dataset: LargeDatasetRefusal). */
export function getAmbient(datasetId: string, column?: string): Promise<AmbientResponse> {
  const q = column ? `?column=${encodeURIComponent(column)}` : "";
  return apiFetch<AmbientResponse>(`/datasets/${datasetId}/qc/ambient${q}`);
}
