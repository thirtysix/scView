import { useState } from "react";
import { formatBytes, hdDefaultBin, type HdBinOption } from "@/lib/ingestUpload";

/** What a bin size is good for, measured on a mouse brain section (batch 32). */
function binNote(um: number): string {
  if (um === 16) return "2 to 4 cells a bin; the fast path, processed in minutes";
  if (um === 8) return "about one cell a bin; a processing run takes much longer, mostly its UMAP";
  return "";
}

/**
 * A Visium HD folder holds several bin sizes, and each is its own dataset: asked
 * before anything is uploaded (the wizard knows them from the folder names), so the
 * 2 um matrix is never sent. How many bins are in the tissue is only known from the
 * matrix, so the upper bound of the capture area is shown; the estimate after the
 * upload says the real count.
 */
export function HdBinChoice({
  options,
  onChoose,
  onCancel,
}: {
  options: HdBinOption[];
  onChoose: (um: number) => void;
  onCancel: () => void;
}) {
  const [um, setUm] = useState<number | null>(hdDefaultBin(options));
  return (
    <div className="rounded-lg border border-slate-200 p-3 text-sm" data-testid="hd-bin-choice">
      <p className="font-medium text-slate-700">Visium HD: which bin size?</p>
      <p className="mt-0.5 text-xs text-slate-500">
        One dataset per bin size; only that folder&apos;s files are uploaded.
      </p>
      <div className="mt-2 space-y-1">
        {options.map((o) => (
          <label
            key={o.um}
            className={`flex items-baseline gap-2 text-xs ${o.refused ? "text-slate-400" : "text-slate-700"}`}
          >
            <input
              type="radio"
              name="hd-bin"
              checked={um === o.um}
              disabled={o.refused}
              onChange={() => setUm(o.um)}
              className="accent-blue-500"
            />
            <span className="font-medium">{o.um} µm</span>
            <span className="text-slate-500">
              {o.refused
                ? "millions of bins, too many to read here: use 8 or 16 µm"
                : [
                    o.matrixBytes > 0 ? `matrix ${formatBytes(o.matrixBytes)}` : null,
                    `at most ${o.maxBins.toLocaleString()} bins on the slide`,
                    binNote(o.um) || null,
                  ]
                    .filter(Boolean)
                    .join("; ")}
            </span>
          </label>
        ))}
      </div>
      <div className="mt-2 flex gap-2">
        <button
          type="button"
          disabled={um == null}
          onClick={() => um != null && onChoose(um)}
          className="rounded bg-primary px-2.5 py-1 text-xs font-medium text-white hover:opacity-90 disabled:opacity-50"
        >
          Upload {um != null ? `${um} µm bins` : ""}
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="rounded border border-slate-300 px-2.5 py-1 text-xs text-slate-600 hover:bg-slate-50"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
