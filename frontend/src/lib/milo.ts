/** Helpers for Milo neighbourhood differential abundance (Compare > Composition). */
import type { MiloResult } from "@/api/milo";
import type { CsvCell } from "@/lib/csv";

export interface NhoodRow {
  /** Position in the result's columns (the id the backend's /milo/cells takes). */
  id: number;
  indexCell: string;
  size: number;
  label: string;
  labelFraction: number | null;
  logFC: number | null;
  pval: number | null;
  spatialFdr: number | null;
  significant: boolean;
  /** Cells of each condition, in `conditions` order. */
  counts: number[];
}

/** Label for neighbourhoods when the run had no label column. */
export const NO_LABEL = "All neighbourhoods";

/** The result's per-neighbourhood columns as rows. */
export function nhoodRows(r: MiloResult): NhoodRow[] {
  const n = r.nhoods;
  return n.index_cell.map((cell, i) => {
    const fdr = n.spatial_fdr[i] ?? null;
    return {
      id: i,
      indexCell: cell,
      size: n.size[i] ?? 0,
      label: n.label?.[i] ?? NO_LABEL,
      labelFraction: n.label_fraction?.[i] ?? null,
      logFC: n.log_fc[i] ?? null,
      pval: n.pval[i] ?? null,
      spatialFdr: fdr,
      significant: fdr != null && fdr < r.alpha,
      counts: r.conditions.map((k) => n.counts[k]?.[i] ?? 0),
    };
  });
}

function median(values: number[]): number {
  const v = [...values].sort((a, b) => a - b);
  if (!v.length) return 0;
  const m = Math.floor(v.length / 2);
  return v.length % 2 ? v[m]! : (v[m - 1]! + v[m]!) / 2;
}

/** Labels bottom to top by median logFC, as Milo's beeswarm orders them;
 *  "Mixed" always last (top). */
export function labelOrder(rows: NhoodRow[]): string[] {
  const by = new Map<string, number[]>();
  for (const r of rows) {
    if (r.logFC == null) continue;
    const list = by.get(r.label) ?? [];
    list.push(r.logFC);
    by.set(r.label, list);
  }
  for (const r of rows) if (!by.has(r.label)) by.set(r.label, []);
  const labels = [...by.keys()].filter((l) => l !== "Mixed");
  labels.sort((a, b) => median(by.get(a)!) - median(by.get(b)!) || a.localeCompare(b));
  return by.has("Mixed") ? [...labels, "Mixed"] : labels;
}

/**
 * Vertical offsets for a beeswarm: points (x values) on one row are placed in
 * x order at the free slot nearest the row's centre (0, +1, -1, +2, ...) times
 * `step`, where a slot is taken by any point within `radius` in x. Beyond
 * `maxOffset` the slots wrap, so a crowded row stays within its band.
 */
export function beeswarmOffsets(xs: number[], radius: number, step: number, maxOffset = 0.42): number[] {
  const order = xs.map((x, i) => [x, i] as const).sort((a, b) => a[0] - b[0]);
  const out = new Array<number>(xs.length).fill(0);
  const maxSlot = Math.max(0, Math.floor(maxOffset / step));
  const slots = [0];
  for (let k = 1; k <= maxSlot; k++) slots.push(k, -k);
  // Points still within `radius` of the current x, with their slot.
  let window: { x: number; slot: number }[] = [];
  for (const [x, i] of order) {
    window = window.filter((p) => x - p.x < radius);
    const used = new Set(window.map((p) => p.slot));
    let slot = slots.find((s) => !used.has(s));
    if (slot === undefined) {
      // Every slot taken: reuse the one used longest ago.
      slot = window[0]!.slot;
      window = window.filter((p) => p.slot !== slot);
    }
    window.push({ x, slot });
    out[i] = slot * step;
  }
  return out;
}

/** The embedding to lay the neighbourhood graph out on: the scatter's, else
 *  UMAP, t-SNE, then whatever the result has (PCA last). */
export function pickLayout(r: MiloResult, preferred?: string | null): string | null {
  const keys = Object.keys(r.layouts);
  if (preferred && keys.includes(preferred)) return preferred;
  for (const k of ["X_umap", "X_tsne"]) if (keys.includes(k)) return k;
  return keys.find((k) => !/pca/i.test(k)) ?? keys[0] ?? null;
}

/** Edge segments for one Plotly line trace: x / y with null between edges. */
export function edgeSegments(r: MiloResult, layout: string): { x: (number | null)[]; y: (number | null)[] } {
  const pos = r.layouts[layout] ?? [];
  const x: (number | null)[] = [];
  const y: (number | null)[] = [];
  for (const [i, j] of r.edges) {
    const a = pos[i];
    const b = pos[j];
    if (!a || !b || a[0] == null || a[1] == null || b[0] == null || b[1] == null) continue;
    x.push(a[0], b[0], null);
    y.push(a[1], b[1], null);
  }
  return { x, y };
}

/** Marker diameter (px) for a neighbourhood of `size` cells, by area. */
export function nodeSize(size: number, maxSize: number): number {
  if (maxSize <= 0) return 6;
  return 4 + 14 * Math.sqrt(Math.max(size, 0) / maxSize);
}

/** Symmetric colour range for logFC: the largest |logFC| among significant
 *  neighbourhoods (at least 1). */
export function logFcRange(rows: NhoodRow[]): number {
  let m = 0;
  for (const r of rows) if (r.significant && r.logFC != null) m = Math.max(m, Math.abs(r.logFC));
  return Math.max(1, Math.ceil(m * 2) / 2);
}

/** Neighbourhoods that shift, in one direction. */
export function shifted(rows: NhoodRow[], direction: "up" | "down"): NhoodRow[] {
  return rows.filter((r) => r.significant && r.logFC != null && (direction === "up" ? r.logFC > 0 : r.logFC < 0));
}

/** One line on what was run, for the result's footnote. */
export function describeMilo(r: MiloResult): string {
  const reps = r.conditions.map((k) => `${k} ${r.n_samples[k] ?? 0}`).join(", ");
  return (
    `${r.n_nhoods.toLocaleString()} neighbourhoods (${Math.round(r.prop * 100)}% of cells sampled, ` +
    `refined; mean ${Math.round(r.mean_nhood_size)} cells) on the ${r.representation} graph. ` +
    `Per neighbourhood a negative binomial GLM (DESeq2) ${r.formula}, ${r.design} by ` +
    `${r.sample_key} (${reps} samples); spatial FDR = Benjamini-Hochberg weighted by each ` +
    `neighbourhood's density, as in Milo (Dann et al. 2022).`
  );
}

/** One row per neighbourhood. */
export function miloCsv(r: MiloResult): { headers: string[]; rows: CsvCell[][] } {
  const headers = [
    "nhood", "index_cell", "n_cells", r.label_key ?? "label", "label_fraction",
    `log2FC_${r.conditions[1]}_vs_${r.conditions[0]}`, "p_value", "spatial_fdr",
    ...r.conditions.map((k) => `cells_${k}`),
  ];
  const rows = nhoodRows(r).map((n) => [
    n.id, n.indexCell, n.size, n.label === NO_LABEL ? "" : n.label, n.labelFraction,
    n.logFC, n.pval, n.spatialFdr, ...n.counts,
  ]);
  return { headers, rows };
}

/** A question for the co-pilot about this result. */
export function miloQuestion(r: MiloResult): string {
  const [ref, test] = r.conditions;
  const top = r.labels
    .filter((l) => l.n_up + l.n_down > 0)
    .sort((a, b) => b.n_up + b.n_down - (a.n_up + a.n_down))
    .slice(0, 5)
    .map((l) => `${l.label}: ${l.n_up} up, ${l.n_down} down of ${l.n_nhoods}`)
    .join("; ");
  return (
    `Interpret the Milo differential abundance result, ${test} vs ${ref} with ${r.sample_key} as ` +
    `replicates: ${r.n_significant} of ${r.n_nhoods} neighbourhoods shift at spatial FDR < ${r.alpha}` +
    (top ? ` (${top})` : "") +
    `. Which shifts look like real changes in abundance, and which could be a change of cell state?`
  );
}
