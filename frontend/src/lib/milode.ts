/** Helpers for neighbourhood DE, miloDE-style (Compare > Composition > Neighbourhood DE). */
import type { MiloDEGeneDetail, MiloDEGeneRow, MiloDEResult } from "@/api/milode";
import type { CsvCell } from "@/lib/csv";

export interface DENhoodRow {
  /** Position in the result's columns (the id /milode/nhood and /milode/cells take). */
  id: number;
  indexCell: string;
  size: number;
  nSamples: number;
  tested: boolean;
  design: string;
  nDe: number;
  nDeWithin: number;
  label: string | null;
  /** Cells of each condition, in `conditions` order. */
  cells: number[];
  minorityShare: number | null;
  /** Under `one_sided` of its cells from one condition: its log2FC rest on few cells. */
  oneSided: boolean;
}

export function deNhoodRows(r: MiloDEResult): DENhoodRow[] {
  const n = r.nhoods;
  return n.index_cell.map((cell, i) => {
    const minority = n.minority_share[i] ?? null;
    return {
      id: i,
      indexCell: cell,
      size: n.size[i] ?? 0,
      nSamples: n.n_samples[i] ?? 0,
      tested: n.tested[i] ?? false,
      design: n.design[i] ?? "",
      nDe: n.n_de[i] ?? 0,
      nDeWithin: n.n_de_within[i] ?? 0,
      label: n.label?.[i] ?? null,
      cells: r.conditions.map((k) => n.cells[k]?.[i] ?? 0),
      minorityShare: minority,
      oneSided: (n.tested[i] ?? false) && (minority == null || minority < r.one_sided),
    };
  });
}

export type GeneSortField = "n_de" | "share" | "median_logfc" | "gene";

/** Gene rows matching `query` (case-insensitive substring), sorted. */
export function sortGenes(
  rows: MiloDEGeneRow[],
  field: GeneSortField,
  dir: "asc" | "desc",
  query = "",
): MiloDEGeneRow[] {
  const q = query.trim().toLowerCase();
  const out = q ? rows.filter((g) => g.gene.toLowerCase().includes(q)) : [...rows];
  const sign = dir === "asc" ? 1 : -1;
  out.sort((a, b) => {
    if (field === "gene") return sign * a.gene.localeCompare(b.gene);
    const x = a[field];
    const y = b[field];
    if (x == null && y == null) return 0;
    if (x == null) return 1; // missing last either way
    if (y == null) return -1;
    return sign * (x - y) || b.n_de - a.n_de || a.gene.localeCompare(b.gene);
  });
  return out;
}

/** Symmetric colour range for a gene's log2FC: its largest |log2FC| where
 *  significant (at least 1, rounded up to 0.5). */
export function geneRange(d: MiloDEGeneDetail): number {
  let m = 0;
  d.log_fc.forEach((v, i) => {
    if (d.significant[i] && v != null) m = Math.max(m, Math.abs(v));
  });
  return Math.max(1, Math.ceil(m * 2) / 2);
}

/** The embedding to show neighbourhoods on: the scatter's, else UMAP, t-SNE, then any. */
export function pickDELayout(r: MiloDEResult, preferred?: string | null): string | null {
  const keys = Object.keys(r.layouts);
  if (preferred && keys.includes(preferred)) return preferred;
  for (const k of ["X_umap", "X_tsne"]) if (keys.includes(k)) return k;
  return keys.find((k) => !/pca/i.test(k)) ?? keys[0] ?? null;
}

/** Points of one gene on a layout: significant neighbourhoods coloured by log2FC,
 *  the other tested ones grey, untested ones left out. */
export function genePoints(
  r: MiloDEResult,
  d: MiloDEGeneDetail,
  layout: string,
): { sig: number[]; grey: number[] } {
  const pos = r.layouts[layout] ?? [];
  const sig: number[] = [];
  const grey: number[] = [];
  d.log_fc.forEach((v, i) => {
    const p = pos[i];
    if (v == null || !p || p[0] == null || p[1] == null) return;
    (d.significant[i] ? sig : grey).push(i);
  });
  return { sig, grey };
}

/** One line on what was run, for the footnote. */
export function describeMiloDE(r: MiloDEResult): string {
  const reps = r.conditions.map((k) => `${k} ${r.n_samples[k] ?? 0}`).join(", ");
  return (
    `${r.n_nhoods.toLocaleString()} neighbourhoods: cells within 2 steps of an index cell on a ` +
    `${r.k}-nearest-neighbour graph of ${r.representation} (miloDE's second-order neighbourhoods, ` +
    `${Math.round(r.prop * 100)}% of cells sampled, then a set cover). Per neighbourhood, raw counts ` +
    `summed per sample (${r.min_cells_per_sample}+ cells), edgeR's expression filter and TMM, then ` +
    `limma-voom ${r.formula}, ${r.design} by ${r.sample_key} (${reps} samples). Per gene, spatial FDR ` +
    `across neighbourhoods, weighted by their overlap, as in miloDE (Missarova et al. 2024).`
  );
}

/** The gene summary as CSV. */
export function miloDEGeneCsv(r: MiloDEResult): { headers: string[]; rows: CsvCell[][] } {
  const headers = [
    "gene", "nhoods_tested", "nhoods_de", "share_de", "nhoods_up", "nhoods_down",
    `median_log2FC_${r.conditions[1]}_vs_${r.conditions[0]}`, "min_spatial_fdr",
  ];
  const rows = r.genes.map((g) => [
    g.gene, g.n_tested, g.n_de, g.share, g.n_up, g.n_down, g.median_logfc, g.min_spatial_fdr,
  ]);
  return { headers, rows };
}

/** A question for the co-pilot about this result. */
export function miloDEQuestion(r: MiloDEResult): string {
  const [ref, test] = r.conditions;
  const top = r.genes
    .filter((g) => g.n_de > 0)
    .slice(0, 6)
    .map((g) => `${g.gene} ${g.n_de}/${g.n_tested}`)
    .join(", ");
  return (
    `Interpret the neighbourhood DE result (miloDE-style), ${test} vs ${ref} with ${r.sample_key} as ` +
    `replicates: ${r.n_genes_de} of ${r.n_genes} genes are DE in at least one of ${r.n_tested} tested ` +
    `neighbourhoods` + (top ? ` (most widespread: ${top})` : "") +
    `. Which changes are shared across cell types and which are confined to part of one?`
  );
}
