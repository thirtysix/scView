import { FileText } from "lucide-react";
import { filePath, type BundleFile } from "@/api/ingest";

const SPATIAL_PART: Record<string, string> = {
  positions: "spot positions",
  scalefactors: "image scale factors",
  hires: "high-resolution tissue image",
  lowres: "low-resolution tissue image",
  archive: "Space Ranger spatial folder",
};

const XENIUM_PART: Record<string, string> = {
  experiment: "panel and run details",
  cells: "cell table: positions and per-cell counts",
  image: "morphology image",
  metrics: "Xenium's run metrics, kept with the dataset's details",
};

/** What a side file is, from its detected kind. */
export function sideFileRole(f: Pick<BundleFile, "kind" | "side_kind" | "spatial_part">): string {
  if (f.side_kind === "spatial") return SPATIAL_PART[f.spatial_part ?? ""] ?? "Space Ranger spatial file";
  if (f.side_kind === "xenium") return XENIUM_PART[f.spatial_part ?? ""] ?? "Xenium file";
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
          key={f.path}
          className="flex items-center gap-1.5 text-slate-500"
          title={
            f.side_kind === "spatial" || f.side_kind === "xenium"
              ? "Places the spots or cells on the tissue image (Unified View tissue plot); not loaded as cells."
              : "Read for the Sequencing QC card on Data Assessment (metrics, flags and the barcode-rank plot); not loaded as cells."
          }
        >
          <FileText className="h-3 w-3 text-sky-600" />
          <span className="break-all text-slate-600">{filePath(f)}</span>
          <span>({sideFileRole(f)}): will be attached</span>
        </li>
      ))}
    </ul>
  );
}
