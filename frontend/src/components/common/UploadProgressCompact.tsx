import { useEffect, useState } from "react";
import { Loader2, X } from "lucide-react";

import { Help } from "@/components/common/Help";
import type { UploadTask } from "@/hooks/useUploadTask";
import { formatRate, formatTimeLeft, percentSent, sentOfTotal, type UploadProgress } from "@/lib/uploadProgress";

/** Classes for a file chooser's label: greyed and not clickable while an upload runs. */
export function lockedLabel(locked: boolean): string {
  return locked ? "cursor-not-allowed opacity-50" : "cursor-pointer hover:bg-slate-50";
}

/** Shown only once an upload has run this long: a sub-second upload flashes nothing. */
export const SHOW_AFTER_MS = 500;

/**
 * The upload card for cards inside a subtab (the Data panel has the large one): a bar, the
 * percent, MB of the total, speed and time left once known, and Cancel. Once every byte is
 * sent, `sentText` with a spinner (the server reads the file). `minimal`: the bar and percent
 * only, for small files (a gene set, a labels table).
 */
export function UploadProgressCompact({
  name,
  progress,
  rate,
  left,
  onCancel,
  sentText = "Sent. Checking the file…",
  minimal = false,
  delayMs = SHOW_AFTER_MS,
}: {
  name: string;
  progress: UploadProgress | null;
  rate: number | null;
  left: number | null;
  onCancel?: () => void;
  sentText?: string;
  minimal?: boolean;
  delayMs?: number;
}) {
  const [shown, setShown] = useState(delayMs <= 0);
  useEffect(() => {
    if (delayMs <= 0) return;
    const id = setTimeout(() => setShown(true), delayMs);
    return () => clearTimeout(id);
  }, [delayMs]);
  if (!shown) return null;

  const sent = progress != null && progress.loaded >= progress.total;
  const percent = progress ? percentSent(progress.loaded, progress.total) : 0;
  const bar = (
    <div
      role="progressbar"
      aria-label={`Upload of ${name}`}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={percent}
      aria-valuetext={progress ? `${percent}%, ${sentOfTotal(progress.loaded, progress.total)}` : "Starting"}
      className="h-1.5 w-full overflow-hidden rounded-full bg-slate-200"
    >
      <div className="h-full rounded-full bg-primary transition-[width] duration-200" style={{ width: `${percent}%` }} />
    </div>
  );
  const status = (
    <p
      role="status"
      aria-live="polite"
      className={sent ? "flex items-center gap-1 text-[11px] text-slate-600" : "sr-only"}
    >
      {sent && <Loader2 className="h-3 w-3 shrink-0 animate-spin text-primary" />}
      <span>{sent ? sentText : `Uploading ${name}.`}</span>
    </p>
  );

  if (minimal) {
    return (
      <div className="space-y-0.5" data-testid="upload-compact">
        <p className="truncate text-[11px] tabular-nums text-slate-500">
          Uploading {name}: {percent}%
        </p>
        {bar}
        {status}
      </div>
    );
  }

  return (
    <div className="space-y-1 rounded border border-slate-200 bg-slate-50 px-2 py-1.5" data-testid="upload-compact">
      <div className="flex items-center gap-2">
        <p className="min-w-0 flex-1 truncate text-[11px] font-medium text-slate-700">Uploading {name}</p>
        {!sent && onCancel && (
          <button
            type="button"
            onClick={onCancel}
            className="flex shrink-0 items-center gap-0.5 rounded border border-slate-300 bg-white px-1.5 py-0.5 text-[11px] text-slate-700 hover:bg-slate-100"
          >
            <X className="h-3 w-3" /> Cancel
          </button>
        )}
      </div>
      {bar}
      <p className="flex flex-wrap gap-x-2 text-[11px] tabular-nums text-slate-500" data-testid="upload-compact-numbers">
        <span className="font-semibold text-slate-700">{percent}%</span>
        {progress && <span>{sentOfTotal(progress.loaded, progress.total)}</span>}
        {!sent && rate != null && <span>{formatRate(rate)}</span>}
        {!sent && left != null && (
          <Help id="upload.time-left">
            <span>{formatTimeLeft(left)} left</span>
          </Help>
        )}
      </p>
      {progress && progress.fileCount > 1 && !sent && (
        <p className="truncate text-[11px] text-slate-500" data-testid="upload-compact-file">
          File {progress.fileIndex} of {progress.fileCount}: {progress.file}
        </p>
      )}
      {status}
    </div>
  );
}

/** A card's upload while it runs (`UploadProgressCompact`), and a note once one is cancelled.
 *  `keptNothing`: the note says nothing was kept (true when one request is the whole upload). */
export function UploadTaskProgress({
  task,
  keptNothing = true,
  ...rest
}: { task: UploadTask; keptNothing?: boolean } & Pick<
  Parameters<typeof UploadProgressCompact>[0],
  "sentText" | "minimal" | "delayMs"
>) {
  if (task.name) {
    return (
      <UploadProgressCompact
        name={task.name}
        progress={task.progress}
        rate={task.rate}
        left={task.left}
        onCancel={task.cancel}
        {...rest}
      />
    );
  }
  if (!task.cancelled) return null;
  return (
    <p className="text-[11px] text-slate-500" data-testid="upload-cancelled">
      Upload of {task.cancelled} cancelled.{keptNothing ? " Nothing was kept." : ""}
    </p>
  );
}
