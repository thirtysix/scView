/**
 * The DE gene prefilter (backend core/de_prefilter.py): the cell-level test runs
 * only on genes detected in at least `min_pct` of the cells of either group.
 * The DE response carries every gene's larger detection fraction as a histogram
 * on a 0.01 % grid, so the subtab can draw it and tell exactly how many genes
 * another cut-off would keep.
 */

/** Seurat v5's FindMarkers default (v4 used 0.1); the backend's reasons are in
 *  core/de_prefilter.py. */
export const DEFAULT_MIN_PCT = 0.01;

const choice = (value: number, note = "") => ({
  value,
  label: value > 0 ? `detected in ≥ ${formatPct(value)} of either group${note}` : "all (no detection cut-off)",
});

export const MIN_PCT_CHOICES: { value: number; label: string }[] = [
  choice(0),
  choice(0.01, " (Seurat v5)"),
  choice(0.05),
  choice(0.1, " (Seurat v4)"),
  choice(0.25),
];

/** The choices, with `current` among them when a session restored another value. */
export function minPctChoices(current: number): { value: number; label: string }[] {
  if (MIN_PCT_CHOICES.some((c) => c.value === current)) return MIN_PCT_CHOICES;
  return [...MIN_PCT_CHOICES, choice(current)].sort((a, b) => a.value - b.value);
}

/** "10%", "2.5%", "0.1%". */
export function formatPct(fraction: number): string {
  return `${Number((fraction * 100).toPrecision(3))}%`;
}

const fmt = (n: number | null | undefined) => (n ?? 0).toLocaleString("en-US");

/** What the prefilter did to a cell-level DE result, for the line under it. */
export function describePrefilter(r: {
  min_pct?: number | null;
  n_genes_tested?: number | null;
  n_genes_skipped?: number | null;
}): string {
  if (r.n_genes_tested == null) return ""; // an older backend
  if (r.min_pct == null) {
    return `All ${fmt(r.n_genes_tested)} genes tested: the expression is scaled, so detection cannot be read.`;
  }
  if (r.min_pct === 0) return `All ${fmt(r.n_genes_tested)} genes tested (no detection cut-off).`;
  return (
    `${fmt(r.n_genes_tested)} genes tested; ${fmt(r.n_genes_skipped)} detected in under ` +
    `${formatPct(r.min_pct)} of the cells of both groups were left out.`
  );
}

/** How many genes a cut-off keeps: exact for any cut-off on the 0.01 % grid. */
export function genesPassing(detection: number[], minPct: number): number {
  const grid = detection.length - 1;
  const from = Math.max(0, Math.ceil(minPct * grid - 1e-6));
  let n = 0;
  for (let k = from; k <= grid; k++) n += detection[k] ?? 0;
  return n;
}

export interface DetectionBins {
  /** Lower edge of each bin, as log10 of the fraction of cells. */
  lo: number[];
  counts: number[];
  /** Bin width in log10 units. */
  width: number;
  /** Genes in under 10^minLog of either group's cells (none detected included). */
  below: number;
  minLog: number;
}

/** The histogram regrouped for a log axis: `perDecade` bins a decade from
 *  10^minLog (0.1 % of cells) to all cells. */
export function detectionBins(detection: number[], perDecade = 10, minLog = -3): DetectionBins {
  const grid = detection.length - 1;
  const n = Math.round(-minLog * perDecade);
  const counts = new Array<number>(n).fill(0);
  let below = detection[0] ?? 0;
  for (let k = 1; k <= grid; k++) {
    const c = detection[k] ?? 0;
    if (!c) continue;
    const pos = (Math.log10(k / grid) - minLog) * perDecade;
    if (pos < -1e-9) below += c;
    else counts[Math.min(n - 1, Math.floor(pos + 1e-9))]! += c;
  }
  return {
    lo: counts.map((_, i) => minLog + i / perDecade),
    counts,
    width: 1 / perDecade,
    below,
    minLog,
  };
}
