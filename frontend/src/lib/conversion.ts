import type { ConversionProgress, ConversionReport } from "@/api/types";

/** Files the R converter takes (a lone one goes to it, not the ingest wizard). */
export function isRFile(name: string): boolean {
  return /\.(rds|rdata)$/i.test(name);
}

/** "a SingleCellExperiment", "an AnnData", "a Seurat object". */
export function objectPhrase(cls: string): string {
  const noun = cls === "Seurat" ? "Seurat object" : cls;
  return `${/^[aeiou]/i.test(noun) ? "an" : "a"} ${noun}`;
}

function size(cells?: number, features?: number, unit = "features"): string {
  return cells && features ? ` (${cells.toLocaleString()} cells × ${features.toLocaleString()} ${unit})` : "";
}

/**
 * The busy line while an R file converts, from the converter's progress file:
 * "Converting a SingleCellExperiment (13,836 cells × 14,053 features): writing
 * the counts assay…". Before the converter has read the file it cannot know
 * the class, so it says so.
 */
export function conversionNote(progress?: ConversionProgress | null): string {
  if (!progress?.source_class) return "Reading the R file… a large object takes a few minutes.";
  const what = objectPhrase(progress.source_class);
  const stage =
    progress.stage === "writing" && progress.message
      ? `: ${progress.message.charAt(0).toLowerCase()}${progress.message.slice(1)}`
      : "";
  return `Converting ${what}${size(progress.n_cells, progress.n_genes)}${stage}…`;
}

/** "Converted from a SingleCellExperiment (object sce in the .RData file): 13,836 cells × 14,053 genes." */
export function conversionHeadline(report: ConversionReport): string {
  const parts = [report.detail, report.object_name ? `object ${report.object_name} in the .RData file` : null]
    .filter(Boolean)
    .join("; ");
  const counts =
    report.n_cells && report.n_genes
      ? `: ${report.n_cells.toLocaleString()} cells × ${report.n_genes.toLocaleString()} genes`
      : "";
  return `Converted from ${objectPhrase(report.source_class)}${parts ? ` (${parts})` : ""}${counts}.`;
}
