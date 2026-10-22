import { FileText } from "lucide-react";
import type { BundleFile } from "@/api/ingest";

/** What a side file is, from its detected kind. */
export function sideFileRole(f: Pick<BundleFile, "kind">): string {
  if (f.kind === "tenx_metrics") return "Cell Ranger metrics";
  if (f.kind === "tenx_h5") return "raw (unfiltered) matrix";
  return "raw matrix part";
}

/**
 * The ingest wizard's list of a sample's side files: a Cell Ranger metrics_summary.csv
 * or raw matrix that is attached for sequencing QC (core/seq_metrics.py), never loaded.
 */
export function SequencingQcSideFiles({ files }: { files?: BundleFile[] }) {
  if (!files?.length) return null;
  return (
    <ul className="mt-2 space-y-1 pl-6 text-xs" data-testid="seqqc-side-files">
      {files.map((f) => (
        <li
          key={f.name}
          className="flex items-center gap-1.5 text-slate-500"
          title="Read for the Sequencing QC card on Data Assessment (metrics, flags and the barcode-rank plot); not loaded as cells."
        >
          <FileText className="h-3 w-3 text-sky-600" />
          <span className="text-slate-600">{f.name}</span>
          <span>({sideFileRole(f)}): will be attached</span>
        </li>
      ))}
    </ul>
  );
}
