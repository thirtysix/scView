import { useState } from "react";
import { AlertTriangle } from "lucide-react";
import { canCreateWebGL } from "@/lib/webgl";

/**
 * Shown over a deck.gl plot whose WebGL context was lost. Retry remounts the
 * plot only when the browser will still create a context; when it will not
 * (Chrome blocks WebGL for a page after a GPU reset it blames on that page),
 * remounting would fail again, so the overlay asks for a page reload instead.
 */
export function GpuLostOverlay({
  onRetry,
  blocked: blockedFromParent = false,
}: {
  /** Remount the plot (a new canvas and context) */
  onRetry: () => void;
  /** The last attempt already failed to create a context */
  blocked?: boolean;
}) {
  const [probedBlocked, setProbedBlocked] = useState(false);
  const blocked = blockedFromParent || probedBlocked;

  const retry = () => {
    if (canCreateWebGL()) {
      setProbedBlocked(false);
      onRetry();
    } else {
      setProbedBlocked(true);
    }
  };

  return (
    <div
      role="alert"
      className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-2 rounded-lg bg-slate-900/80 px-4 text-white"
    >
      <AlertTriangle className="h-6 w-6 text-amber-400" />
      <p className="text-sm font-medium">GPU context lost</p>
      <p className="max-w-xs text-center text-xs text-slate-300">
        {blocked
          ? "The browser has stopped this page from using the GPU after a graphics reset. Reload the page to draw the plots again."
          : "The browser dropped this plot's graphics context (a GPU reset, or the GPU ran short of memory)."}
      </p>
      {blocked ? (
        <button
          onClick={() => window.location.reload()}
          className="mt-1 rounded-md border border-slate-500 bg-slate-700 px-4 py-1.5 text-xs font-medium text-white hover:bg-slate-600"
        >
          Reload page
        </button>
      ) : (
        <button
          onClick={retry}
          className="mt-1 rounded-md border border-slate-500 bg-slate-700 px-4 py-1.5 text-xs font-medium text-white hover:bg-slate-600"
        >
          Retry
        </button>
      )}
    </div>
  );
}
