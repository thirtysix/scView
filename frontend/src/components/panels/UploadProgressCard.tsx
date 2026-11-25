import { useState } from "react";
import { Check, FileUp, Loader2, X } from "lucide-react";

import type { PreflightVerdict } from "@/api/preflight";
import { Help } from "@/components/common/Help";
import { PreflightNote } from "@/components/common/PreflightNote";
import {
  formatRate,
  formatSize,
  formatTimeLeft,
  percentSent,
  sentOfTotal,
  type UploadProgress,
} from "@/lib/uploadProgress";

export type UploadPhase = "uploading" | "converting";

/**
 * The Data panel's upload card, in place of the drop zone while files go up: a bar with the
 * percent, MB sent, speed and time left, and Cancel. An R object then shows its conversion here
 * (the converter's stage and memory estimate). A drop on the card starts nothing.
 */
export function UploadProgressCard({
  names,
  phase,
  progress,
  rate,
  left,
  rObject = false,
  stage,
  preflight,
  onCancel,
}: {
  /** What is going up: a file name, or "12 files". */
  names: string;
  phase: UploadPhase;
  progress: UploadProgress | null;
  rate: number | null;
  left: number | null;
  /** An .rds / .RData: the card has an Upload and a Convert step. */
  rObject?: boolean;
  /** The converter's stage while it converts. */
  stage?: string | null;
  preflight?: PreflightVerdict | null;
  onCancel?: () => void;
}) {
  const [refused, setRefused] = useState(false);
  const sent = phase === "converting" || (progress != null && progress.loaded >= progress.total);
  const percent = phase === "converting" ? 100 : progress ? percentSent(progress.loaded, progress.total) : 0;
  const statusText =
    phase === "converting"
      ? (stage ?? "Starting the R converter…")
      : sent
        ? rObject
          ? "Sent. Saving the file and starting the R converter…"
          : "Sent. Checking the files…"
        : "Uploading. Keep this page open until it is sent.";

  return (
    <section
      aria-label="Upload progress"
      data-testid="upload-card"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        setRefused(true);
      }}
      className="rounded-xl border-2 border-primary/40 bg-white p-5 shadow-sm"
    >
      <div className="flex items-start gap-3">
        <FileUp className="mt-0.5 h-6 w-6 shrink-0 text-primary" />
        <div className="min-w-0 flex-1">
          <p className="text-base font-semibold text-slate-800" data-testid="upload-title">
            {phase === "converting" ? "Converting " : "Uploading "}
            <span className="break-all">{names}</span>
          </p>
          {rObject && <Steps phase={phase} progress={progress} sent={sent} />}
        </div>
        {phase === "uploading" && !sent && onCancel && (
          <button
            type="button"
            onClick={onCancel}
            className="flex shrink-0 items-center gap-1 rounded-lg border border-slate-300 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50"
          >
            <X className="h-4 w-4" /> Cancel
          </button>
        )}
      </div>

      {phase === "uploading" ? (
        <div className="mt-4">
          <div
            role="progressbar"
            aria-label={`Upload of ${names}`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={percent}
            aria-valuetext={progress ? `${percent}%, ${sentOfTotal(progress.loaded, progress.total)}` : "Starting"}
            className="h-3 w-full overflow-hidden rounded-full bg-slate-200"
          >
            <div
              className="h-full rounded-full bg-primary transition-[width] duration-200"
              style={{ width: `${percent}%` }}
            />
          </div>
          <p className="mt-2 flex flex-wrap gap-x-3 text-sm tabular-nums text-slate-600" data-testid="upload-numbers">
            <span className="font-semibold text-slate-800">{percent}%</span>
            {progress && <span>{sentOfTotal(progress.loaded, progress.total)}</span>}
            {!sent && rate != null && <span>{formatRate(rate)}</span>}
            {!sent && (
              <Help id="upload.time-left">
                <span>{left != null ? `${formatTimeLeft(left)} left` : "measuring speed…"}</span>
              </Help>
            )}
          </p>
          {progress && progress.fileCount > 1 && !sent && (
            <p className="mt-1 truncate text-xs text-slate-500" data-testid="upload-file">
              File {progress.fileIndex} of {progress.fileCount}: {progress.file}
            </p>
          )}
        </div>
      ) : (
        // the converter gives stages, not a fraction: a full bar that pulses, never a part-filled one
        <div className="mt-4 h-3 w-full overflow-hidden rounded-full bg-slate-200" aria-hidden>
          <div className="h-full w-full animate-pulse rounded-full bg-primary/40" />
        </div>
      )}

      <p role="status" aria-live="polite" className="mt-3 flex items-center gap-2 text-sm text-slate-600">
        {sent && <Loader2 className="h-4 w-4 shrink-0 animate-spin text-primary" />}
        <span>{statusText}</span>
      </p>
      {phase === "converting" && preflight && (
        <div className="mt-2">
          <PreflightNote verdict={preflight} />
        </div>
      )}
      {refused && (
        <p className="mt-2 text-xs text-amber-700" role="alert">
          One upload at a time: add more files once this one is done.
        </p>
      )}
    </section>
  );
}

/** Upload, then Convert: which is done and which is running. */
function Steps({ phase, progress, sent }: { phase: UploadPhase; progress: UploadProgress | null; sent: boolean }) {
  const uploadDone = phase === "converting";
  return (
    <ol className="mt-1 flex flex-wrap gap-x-4 text-xs text-slate-500">
      <li className={`flex items-center gap-1 ${uploadDone ? "text-emerald-700" : "font-medium text-slate-700"}`}>
        {uploadDone ? <Check className="h-3.5 w-3.5" /> : <span aria-hidden>1.</span>}
        Upload{uploadDone && progress ? ` (${formatSize(progress.total)})` : ""}
      </li>
      <li className={`flex items-center gap-1 ${uploadDone ? "font-medium text-slate-700" : ""}`}>
        <span aria-hidden>2.</span>
        <Help id="upload.convert">Convert</Help>
        {!uploadDone && sent && <span>(next)</span>}
      </li>
    </ol>
  );
}
