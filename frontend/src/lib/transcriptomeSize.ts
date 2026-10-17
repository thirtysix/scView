import type { TranscriptomeSize } from "@/api/de";

/**
 * Transcriptome-size caution for a DE result (backend core/transcriptome_size.py).
 * Library-size normalisation scales every cell to the same total, so when side A's
 * cells hold r times the RNA of side B's, a gene with the same number of
 * transcripts per cell on both sides reads as log2FC -log2(r).
 */

const fmt = (n: number) => Math.round(n).toLocaleString("en-US");

/** The log2 fold change (A vs B) a gene with equal transcripts per cell gets. */
export function equalCopyLog2FC(s: TranscriptomeSize): number {
  return s.median_a > 0 && s.median_b > 0 ? Math.log2(s.median_b / s.median_a) : 0;
}

function ratioText(s: TranscriptomeSize): string {
  return s.ratio != null ? `${s.ratio.toFixed(2)}x` : "far apart";
}

/** One line for the result summary. */
export function describeSize(s: TranscriptomeSize, a: string, b: string): string {
  return (
    `Median total counts per cell: ${a} ${fmt(s.median_a)}, ${b} ${fmt(s.median_b)} ` +
    `(${ratioText(s)}; ${s.source}).`
  );
}

/** The warning shown when the medians are `threshold`-fold apart or more. */
export function sizeCaution(s: TranscriptomeSize, a: string, b: string): string {
  const [big, small] = s.median_a >= s.median_b ? [a, b] : [b, a];
  const shift = equalCopyLog2FC(s);
  return (
    `${big} cells hold ${ratioText(s)} the RNA of ${small} cells (median total counts ` +
    `${fmt(Math.max(s.median_a, s.median_b))} vs ${fmt(Math.min(s.median_a, s.median_b))}). ` +
    `Normalising every cell to the same total hides that: a gene with the same number of ` +
    `transcripts per cell in both groups reads as log2FC ${shift > 0 ? "+" : ""}${shift.toFixed(2)} ` +
    `here, and fold changes of broadly expressed genes shift together. Read small fold changes, ` +
    `and genes "down" in ${big}, with care.`
  );
}

/** For the co-pilot's description of the comparison (only when cautioned). */
export function sizeCautionForCopilot(s: TranscriptomeSize, a: string, b: string): string {
  return (
    `median total counts differ ${ratioText(s)} (${a} ${fmt(s.median_a)} vs ${b} ${fmt(s.median_b)}), ` +
    `so library-size normalisation shifts fold changes of broadly expressed genes by about ` +
    `${equalCopyLog2FC(s).toFixed(2)} log2`
  );
}
