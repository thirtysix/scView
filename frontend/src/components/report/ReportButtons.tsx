import { FileText } from "lucide-react";
import { useDatasetStore } from "@/stores/datasetStore";
import { openClusterReport, openGeneReport } from "@/stores/reportStore";

/** Icon button on a gene row (Markers table): opens the gene's report in the Report tab. */
export function GeneReportButton({ gene, groupby }: { gene: string; groupby?: string }) {
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  if (!datasetId) return null;
  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        openGeneReport(datasetId, gene, groupby || undefined);
      }}
      title={`Gene report for ${gene}: expression per group, marker status, databases, per donor, co-expression`}
      aria-label={`Gene report for ${gene}`}
      className="flex-shrink-0 text-slate-300 opacity-0 transition-opacity hover:text-primary focus:opacity-100 group-hover:opacity-100"
    >
      <FileText className="h-3 w-3" />
    </button>
  );
}

/** Small text button next to a highlighted cluster: opens its report in the Report tab. */
export function ClusterReportButton({ column, value }: { column: string; value: string }) {
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  if (!datasetId) return null;
  return (
    <button
      type="button"
      onClick={() => openClusterReport(datasetId, column, value)}
      title={`Report for ${value}: size, labels, markers, enrichment, QC and composition, with a download`}
      className="ml-1 inline-flex items-center gap-0.5 rounded px-1 text-[10px] font-medium text-primary hover:bg-blue-50"
    >
      <FileText className="h-3 w-3" />
      Cluster report
    </button>
  );
}
