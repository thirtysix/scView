import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Loader2, X } from "lucide-react";

import { downloadUrl } from "@/api/export";
import {
  getRExportInfo,
  getRExportStatus,
  rExportFileUrl,
  startRExport,
  type RExportInfo,
  type RExportJob,
  type RFormat,
} from "@/api/rdsExport";
import { InfoTip } from "@/components/common/InfoTip";
import { useDialogFocus } from "@/hooks/useDialogFocus";
import { formatBytes } from "@/lib/formatting";

const TITLES: Record<RFormat, string> = {
  seurat: "Seurat object (.rds)",
  sce: "SingleCellExperiment (.rds)",
};
const READ_IN_R: Record<RFormat, string> = {
  seurat: 'obj <- readRDS("file.rds")  # library(Seurat) or library(SeuratObject), version 5 or later',
  sce: 'sce <- readRDS("file.rds")  # library(SingleCellExperiment)',
};

interface Props {
  datasetId: string;
  format: RFormat;
  onClose: () => void;
  /** How often the job is checked while R writes the file (ms) */
  pollMs?: number;
}

/**
 * The dataset as an R object: what goes in and what is left out, then the file is
 * built by the R converter on the server (progress shown) and the browser saves it.
 */
export function RExportDialog({ datasetId, format, onClose, pollMs = 1500 }: Props) {
  const [info, setInfo] = useState<RExportInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<RExportJob | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  useDialogFocus(dialogRef, onClose);

  useEffect(() => {
    let live = true;
    getRExportInfo(datasetId, format)
      .then((i) => {
        if (!live) return;
        setInfo(i);
        if (i.running_job) setJobId(i.running_job); // started earlier: follow it again
      })
      .catch((e: unknown) => live && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, [datasetId, format]);

  // Follow the job until R has written the file, then download it once.
  useEffect(() => {
    if (!jobId) return;
    let live = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const st = await getRExportStatus(datasetId, jobId);
        if (!live) return;
        setJob(st);
        if (st.status === "done") {
          downloadUrl(rExportFileUrl(datasetId, jobId), st.filename);
          return;
        }
        if (st.status === "error") {
          setError(st.error ?? "The export failed.");
          return;
        }
        timer = setTimeout(poll, pollMs);
      } catch (e) {
        if (live) setError(e instanceof Error ? e.message : String(e));
      }
    };
    void poll();
    return () => {
      live = false;
      if (timer) clearTimeout(timer);
    };
  }, [datasetId, jobId, pollMs]);

  const start = async () => {
    setError(null);
    setJob(null);
    try {
      const j = await startRExport(datasetId, format);
      setJob(j);
      setJobId(j.job);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const busy = job !== null && (job.status === "writing" || job.status === "converting");
  const done = job?.status === "done";
  const blocked = !info || !info.fits || info.converter === "down";
  const leftOut = Object.entries(info?.dropped ?? {});

  return createPortal(
    <div className="fixed inset-0 z-[1100] flex items-center justify-center bg-slate-900/40 p-4" role="presentation">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label={TITLES[format]}
        className="max-h-[90vh] w-full max-w-xl overflow-y-auto rounded-xl bg-white p-5 shadow-xl"
      >
        <div className="mb-3 flex items-start justify-between gap-3">
          <div>
            <h2 className="flex items-center gap-1 text-base font-semibold text-slate-800">
              {TITLES[format]}
              <InfoTip label="How to open it in R">
                For R: readRDS(). It needs {format === "seurat" ? "SeuratObject 5 or later (Seurat 5)" : "the SingleCellExperiment package (Bioconductor)"}.
                The file is built by scView&apos;s R converter and keeps the counts and log-normalised values
                sparse.
              </InfoTip>
            </h2>
            <p className="text-xs text-slate-500">
              {format === "seurat"
                ? "Counts and log-normalised data layers, cell metadata, layouts, clusters as Idents."
                : "counts and logcounts assays, colData, reducedDims (PCA, UMAP, …)."}
            </p>
          </div>
          <button type="button" aria-label="Close" onClick={onClose} className="rounded p-1 text-slate-400 hover:bg-slate-100">
            <X className="h-4 w-4" />
          </button>
        </div>

        {!info && !error && <p className="text-sm text-slate-500">Checking the dataset…</p>}

        {info && !done && (
          <div className="space-y-3 text-sm">
            <p className="text-xs text-slate-600" data-testid="rexport-contents">
              {info.n_cells.toLocaleString()} cells × {info.n_genes.toLocaleString()} genes
              {info.counts_from ? ` · raw counts from ${info.counts_from}` : " · no raw counts"}
              {info.data_from ? ` · log-normalised values from ${info.data_from}` : " · no log-normalised values"}
              {info.embeddings.length > 0 && ` · layouts: ${info.embeddings.join(", ")}`}
              {info.n_proteins > 0 && ` · ${info.n_proteins} proteins (ADT)`}
              {info.cluster_column && (format === "seurat" ? ` · Idents: ${info.cluster_column}` : "")}
            </p>
            {(info.notes.length > 0 || leftOut.length > 0) && (
              <details className="text-xs text-slate-600">
                <summary className="cursor-pointer">What is left out</summary>
                <ul className="ml-4 mt-1 list-disc space-y-0.5">
                  {info.notes.map((n) => (
                    <li key={n}>{n}</li>
                  ))}
                  {leftOut.map(([where, keys]) => (
                    <li key={where}>
                      {where}: {keys.join(", ")}
                    </li>
                  ))}
                  {format === "seurat" && <li>Scaled values (scale.data): run ScaleData() in R.</li>}
                </ul>
              </details>
            )}
            {info.converter === "busy" && (
              <p className="text-xs text-amber-700">The R converter is busy with another file; this export waits for it.</p>
            )}
            {info.converter === "down" && (
              <p role="alert" className="text-xs text-rose-700">
                The R converter is not running, so .rds files cannot be written. Start it (docker compose up -d
                converter) and open this again.
              </p>
            )}
            {!info.fits && info.too_big && (
              <p role="alert" className="text-xs text-rose-700">
                {info.too_big}
              </p>
            )}
            {info.fits && (
              <p className="text-xs text-slate-500">
                Needs about {formatBytes(info.needs_bytes)} of memory in the R converter.
              </p>
            )}
          </div>
        )}

        {done && job?.report && (
          <section aria-label="Export report" className="space-y-2 text-sm">
            <p className="text-slate-700">
              Downloaded {job.filename}
              {job.size_bytes !== undefined && ` (${formatBytes(job.size_bytes)})`}.
            </p>
            <code className="block rounded bg-slate-50 px-2 py-1 text-xs text-slate-600">{READ_IN_R[format]}</code>
            <details className="text-xs text-slate-600" open>
              <summary className="cursor-pointer">What went in ({job.report.kept.length})</summary>
              <ul className="ml-4 mt-1 list-disc space-y-0.5">
                {job.report.kept.map((k) => (
                  <li key={k}>{k}</li>
                ))}
              </ul>
            </details>
            {job.report.skipped.length > 0 && (
              <details className="text-xs text-slate-600" open>
                <summary className="cursor-pointer">Left out ({job.report.skipped.length})</summary>
                <ul className="ml-4 mt-1 list-disc space-y-0.5">
                  {job.report.skipped.map((k) => (
                    <li key={k}>{k}</li>
                  ))}
                </ul>
              </details>
            )}
          </section>
        )}

        {error && <p role="alert" className="mt-3 text-sm text-rose-700">{error}</p>}

        <div className="mt-4 flex items-center justify-end gap-2">
          {busy && (
            <span className="mr-auto flex items-center gap-1 text-xs text-slate-500" role="status">
              <Loader2 className="h-3.5 w-3.5 animate-spin" /> {job?.message}
            </span>
          )}
          {busy && (
            <span className="text-xs text-slate-400">Closing keeps it running; open this again to download it.</span>
          )}
          <button type="button" onClick={onClose} className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm">
            Close
          </button>
          {!done && (
            <button
              type="button"
              onClick={start}
              disabled={blocked || busy}
              className="rounded-lg bg-blue-600 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
            >
              Download
            </button>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
