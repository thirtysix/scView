import { apiFetch } from "./client";

export type HashingMethod = "htodemux" | "hashedDrops";
export type HashClass = "Singlet" | "Doublet" | "Negative";

/** One tag of a saved call (core/hashing.py): HTODemux has a count cutoff, hashedDrops an ambient level. */
export interface HashingTagFit {
  tag: string;
  /** HTODemux: a cell is positive for the tag above this many counts (the background fit's quantile). */
  cutoff?: number;
  background_group?: number;
  background_cells?: number;
  background_mean?: number;
  /** Negative binomial size of the background fit; null = Poisson (not over-dispersed). */
  nb_size?: number | null;
  n_positive?: number;
  /** hashedDrops: the tag's ambient level (mean count of its lower mode). */
  ambient?: number;
  n_singlets?: number;
}

export interface HashingCall {
  method: HashingMethod;
  label: string;
  tags: string[];
  params: Record<string, unknown>;
  thresholds: { positive_quantile?: number; doublet_lfc2?: number | null; confident_lfc?: number };
  per_tag: HashingTagFit[];
  when: string;
  n_cells_at_call: number;
  /** On the current cells (filtering after the call removes some). */
  counts: Record<HashClass, number>;
  /** Cells per hash_call value: each tag's singlets, "Doublet", "Negative". */
  per_call: Record<string, number>;
  n_uncalled: number;
}

export interface HashingSamples {
  column: string;
  mapping: Record<string, string>;
  counts: Record<string, number>;
  n_unassigned: number;
}

export interface HashingSummary {
  n_cells: number;
  /** Tag counts or a protein modality exist (the card is shown). */
  available: boolean;
  /** Tag counts kept (obsm hto_counts, or another tool's table). */
  tags: string[];
  tag_sources: Record<string, string>;
  /** Antibody features still in the protein modality (any can be marked as a hashtag). */
  proteins: string[];
  /** Those whose names look like hashtags. */
  suggested: string[];
  moved_from_protein: string[];
  methods: { id: HashingMethod; label: string }[];
  defaults: { method: HashingMethod; quantile: number };
  limits: { min_tags: number; max_tags: number; min_cells: number };
  default_sample_column: string;
  call: HashingCall | null;
  samples: HashingSamples | null;
  /** Hashing class x Scrublet's call, when the dataset has both. */
  scrublet: Record<HashClass, { scrublet_doublet: number; scrublet_singlet: number }> | null;
  /** The obs column the doublet views read (Scrublet's first, then text calls such as hash_class). */
  doublet_source: string | null;
}

export interface HashingTagHist {
  tag: string;
  bin_edges: number[];
  /** Per group: "tag" (singlets called to it), "doublet", "other"; "all" before a call. */
  counts: Record<string, number[]>;
  cutoff: number | null;
  /** The count cutoff in CLR units (the CLR is monotone in a tag's counts). */
  cutoff_clr: number | null;
  n_above: number | null;
  median_count: number;
}

export interface HashingLfcHist {
  bin_edges: number[];
  counts: number[];
  threshold: number | null;
}

export interface HashingPlotData {
  available: boolean;
  tags: string[];
  n_cells: number;
  per_tag?: HashingTagHist[];
  method?: HashingMethod | null;
  heatmap?: { tags: string[]; calls: string[]; n: number[]; mean_clr: number[][] } | null;
  lfc?: { lfc: HashingLfcHist | null; lfc2: HashingLfcHist | null } | null;
}

/** GET /hashing: tags found, hashtag suggestions, the saved call and sample column. */
export function getHashing(datasetId: string): Promise<HashingSummary> {
  return apiFetch(`/datasets/${datasetId}/hashing`);
}

/** GET /hashing/plot-data: per-tag CLR histograms with cutoffs, tag x call heatmap. */
export function getHashingPlotData(datasetId: string): Promise<HashingPlotData> {
  return apiFetch(`/datasets/${datasetId}/hashing/plot-data`);
}

/** POST /hashing/demux: call every cell from `tags` (antibodies among them leave the protein modality). */
export function runHashingDemux(
  datasetId: string,
  body: { tags: string[]; method: HashingMethod; quantile?: number },
): Promise<HashingSummary> {
  return apiFetch(`/datasets/${datasetId}/hashing/demux`, { method: "POST", body: JSON.stringify(body) });
}

/** POST /hashing/samples: write `column` with each singlet's sample name (doublets / negatives empty). */
export function saveHashingSamples(
  datasetId: string,
  mapping: Record<string, string>,
  column: string,
): Promise<HashingSummary> {
  return apiFetch(`/datasets/${datasetId}/hashing/samples`, {
    method: "POST",
    body: JSON.stringify({ mapping, column }),
  });
}
