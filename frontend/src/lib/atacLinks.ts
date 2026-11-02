import type { LinkArc, LinksBody, LinksEstimate, LinksPage } from "@/api/atacLinks";
import { formatDuration } from "@/lib/hvgSensitivity";

// Peak-gene links (backend core/atac_links.py): the card's text and the arcs' geometry.

const gb = (bytes: number) => `${(bytes / 1024 ** 3).toFixed(bytes >= 10 * 1024 ** 3 ? 0 : 1)} GB`;

const POSITIVE = [37, 99, 235];
const NEGATIVE = [220, 38, 38];
const DEEPEST_BELOW = 0.14; // the row's bottom margin: its one line of text never sits under the deepest arc

/** Genes typed or pasted, split on commas and white space, each once. */
export function parseGeneList(text: string): string[] {
  return [...new Set(text.split(/[\s,;]+/).map((g) => g.trim()).filter(Boolean))];
}

export function linksBody(genes: LinksBody["genes"], listText: string, column: string, values: string[]): LinksBody {
  return {
    genes,
    ...(genes === "list" ? { gene_list: parseGeneList(listText) } : {}),
    cells_column: column || null,
    cells_value: column ? values : null,
  };
}

/** "2,000 genes, 11,208 cells: about 1 min, 0.9 GB beyond the open dataset and 0.8 GB of disk while it runs. ..." */
export function estimateText(e: LinksEstimate): string {
  const download = e.genome.downloaded
    ? ""
    : ` The first use of ${e.genome.genome} downloads the ${e.genome.source} sequence (${(e.genome.size / 1e6).toFixed(0)} MB, once; included in the time).`;
  return `${e.n_genes.toLocaleString()} genes, ${e.n_cells.toLocaleString()} cells: about ${formatDuration(e.estimate_seconds)}, ` +
    `${gb(e.estimate_bytes)} beyond the open dataset and ${gb(e.estimate_disk_bytes)} of disk while it runs.${download}`;
}

/** What a result counted, in one sentence for its caption. */
export function countsText(r: LinksPage): string {
  const c = r.counts;
  const parts = [
    `${c.n_links.toLocaleString()} links at p < 0.05 from ${c.n_pairs_tested.toLocaleString()} pairs tested on ` +
      `${c.n_genes_tested.toLocaleString()} genes; ${c.n_links_padj.toLocaleString()} also at BH FDR 5%.`,
  ];
  const skipped = c.n_genes_few_cells + c.n_genes_few_peaks;
  if (skipped > 0) {
    parts.push(`${skipped.toLocaleString()} genes not tested (${c.n_genes_few_cells.toLocaleString()} in 10 cells or fewer, ` +
      `${c.n_genes_few_peaks.toLocaleString()} with fewer than 2 peaks within 500 kb).`);
  }
  if (c.n_peaks_unknown > 0) {
    parts.push(`${c.n_peaks_unknown.toLocaleString()} peaks left out: their contig has no bases under that name in the genome file, so no GC.`);
  }
  return parts.join(" ");
}

export function cellsText(r: LinksPage): string {
  const p = r.params;
  return p.cells_column ? `${p.cells_column}: ${(p.cells_value ?? []).join(", ")}` : "all cells";
}

/** An arc's colour: blue for a positive score, red for a negative, deeper with |score|. */
export function arcColor(score: number, largest: number): string {
  const [r, g, b] = score >= 0 ? POSITIVE : NEGATIVE;
  const share = largest > 0 ? Math.min(1, Math.abs(score) / largest) : 1;
  return `rgba(${r},${g},${b},${(0.35 + 0.65 * share).toFixed(2)})`;
}

/** Depth of an arc below the row's top (0-1), by |score| among those in view. */
export function arcDepth(score: number, largest: number): number {
  const share = largest > 0 ? Math.min(1, Math.abs(score) / largest) : 1;
  return (0.22 + 0.72 * share) * (1 - DEEPEST_BELOW);
}

export interface ArcShape {
  path: string;
  apexX: number;
  apexY: number;
  color: string;
  dash: "solid" | "dash";
  width: number;
}

/** One quadratic curve per link, peak middle to TSS, hanging from the row's top (y = 1): the curve's
 *  lowest point is half way to its control point. A negative score is dashed as well as red, so the sign
 *  survives greyscale. The strongest are drawn last, on top. */
export function arcShapes(arcs: LinkArc[], selectedPeak?: string | null): ArcShape[] {
  const largest = arcs.reduce((m, a) => Math.max(m, Math.abs(a.score)), 0);
  return [...arcs]
    .sort((a, b) => Math.abs(a.score) - Math.abs(b.score))
    .map((a) => {
      const depth = arcDepth(a.score, largest);
      const apexX = (a.middle + a.tss) / 2;
      return {
        path: `M ${a.middle} 1 Q ${apexX} ${1 - 2 * depth} ${a.tss} 1`,
        apexX,
        apexY: 1 - depth,
        color: arcColor(a.score, largest),
        dash: a.score < 0 ? ("dash" as const) : ("solid" as const),
        width: a.peak === selectedPeak ? 3 : 1.5,
      };
    });
}
