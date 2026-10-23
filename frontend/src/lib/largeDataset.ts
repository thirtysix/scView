import type { LargeDatasetInfo } from "@/api/types";

/** What a large dataset's badge says on hover: where its matrices are, and anything
 *  slow or missing about its gene-major copy or counts. */
export function largeDatasetTitle(info: LargeDatasetInfo): string {
  const lines = [
    "This dataset's expression and counts matrices stay on disk; genes are read from a gene-major " +
      "copy. Views that would need a whole matrix in memory say so instead of running.",
  ];
  if (info.gene_store === "rebuilding")
    lines.push(
      `The gene-major copy is being rebuilt${info.gene_store_progress != null ? ` (${Math.round(info.gene_store_progress * 100)}%)` : ""}: gene views are slow until it is done.`,
    );
  else if (info.gene_store === "missing")
    lines.push(`Gene views are slow: ${info.gene_store_note ?? "the gene-major copy is missing"}.`);
  if (!info.counts) lines.push(`Counts are not available: ${info.counts_note ?? "their file is missing"}.`);
  return lines.join("\n");
}
