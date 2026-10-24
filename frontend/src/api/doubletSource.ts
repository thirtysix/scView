/** Whose doublet calls the app uses (backend core/doublet_source.py): cell hashing when
 *  it was run, else Scrublet, else another tool's text column. */
export type DoubletSource = "auto" | "hashing" | "scrublet" | "either";

export const DOUBLET_SOURCES: readonly DoubletSource[] = ["auto", "hashing", "scrublet", "either"];

/** What the doublet views use, with the hashing x Scrublet overlap when both exist. */
export interface DoubletSourceInfo {
  /** hashing | scrublet | either | a text column's name */
  source: string;
  label: string;
  columns: string[];
  n_cells: number;
  n_doublets: number;
  fraction: number;
  hashing: { n_doublets: number; fraction: number; n_negatives: number } | null;
  scrublet: { n_doublets: number; fraction: number } | null;
  both: { n: number; fraction: number } | null;
  options: DoubletSource[];
}

export interface AgreementCounts {
  n_cells: number;
  hashing: number;
  scrublet: number;
  both: number;
}

/** Cell hashing x Scrublet's calls (at the threshold shown), overall and per sample. */
export interface DoubletAgreement {
  n_cells: number;
  hashing: number;
  scrublet: number;
  both: number;
  hashing_only: number;
  scrublet_only: number;
  neither: number;
  /** Hashing negatives (no tag), and how many of them Scrublet calls doublets. */
  negatives: number;
  negatives_scrublet: number;
  samples: (AgreementCounts & { name: string })[];
  /** Cells without a value in the sample column (hashing doublets have no hashing sample). */
  unassigned: AgreementCounts | null;
  /** The Scrublet threshold the counts are for; null = its saved call. */
  threshold: number | null;
  column: string | null;
}

/** The filter preview's doublet block: the sources that mean something for this
 *  dataset, the one chosen (and what it stood for), and hashing negatives. */
export interface DoubletChoices {
  source: DoubletSource;
  used: { source: string; label: string; n: number } | null;
  auto: { source: string; label: string };
  options: DoubletSource[];
  n_by_source: Partial<Record<DoubletSource, number | null>>;
  both: number | null;
  n_cells: number;
  /** Cells hashing calls Negative; null without hashing. */
  hash_negatives: number | null;
  drop_hash_negatives: boolean;
  negative_note: string;
  /** Cells the chosen calls leave unscored (Scrublet's channels too small for it): kept. */
  unscored?: number | null;
}
