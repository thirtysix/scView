import { create } from "zustand";
import { watchdogMessage, type WatchdogEvent } from "@/api/pipelineRun";
import type { RunHandlers, RunProgress, RunStage, StabilityCurve } from "@/api/spatialDomains";

export type RunKind = "domains" | "enrichment" | "niches";

/**
 * A spatial domains, enrichment or niches run in flight (Domains subtab). Kept here, not
 * in the tab, so switching tabs does not abort it; Cancel aborts the stream, which stops
 * the run before its next piece of work without saving.
 */
interface SpatialDomainsRunState {
  datasetId: string | null;
  kind: RunKind | null;
  running: boolean;
  stage: RunStage | null;
  progress: RunProgress | null;
  curve: StabilityCurve | null;
  warnings: string[];
  error: string | null;
  /** The column a finished run wrote; the tab colours the tissue by it, then clears this. */
  finished: string | null;
  /** Bumped when a run saves: the tab reloads the results and the dataset. */
  version: number;
  start: <T>(
    datasetId: string,
    kind: RunKind,
    what: string,
    go: (handlers: RunHandlers) => Promise<T>,
    columnOf: (result: T) => string | null,
  ) => Promise<T | null>;
  cancel: () => void;
  reset: () => void;
}

let controller: AbortController | null = null;

const idle = {
  datasetId: null,
  kind: null,
  running: false,
  stage: null,
  progress: null,
  curve: null,
  warnings: [] as string[],
  error: null,
};

export const useSpatialDomainsRunStore = create<SpatialDomainsRunState>((set, get) => ({
  ...idle,
  finished: null,
  version: 0,
  start: async (datasetId, kind, what, go, columnOf) => {
    if (get().running) return null;
    controller = new AbortController();
    const signal = controller.signal;
    set({ ...idle, datasetId, kind, running: true });
    try {
      const result = await go({
        signal,
        onStage: (stage) => set({ stage, progress: null }),
        onProgress: (progress) => set({ progress }),
        onCurve: (curve) => set({ curve }),
        onWatchdog: (w: WatchdogEvent) => {
          if (w.kind === "memory") set((s) => ({ warnings: [...s.warnings, watchdogMessage(w, () => what)] }));
        },
      });
      const column = columnOf(result);
      set((s) => ({
        running: false,
        version: s.version + 1,
        finished: column ?? s.finished,
      }));
      return result;
    } catch (e) {
      const aborted = signal.aborted;
      set({ running: false, error: aborted ? "Cancelled: nothing was saved." : e instanceof Error ? e.message : String(e) });
      return null;
    } finally {
      if (controller?.signal === signal) controller = null;
    }
  },
  cancel: () => controller?.abort(),
  reset: () => {
    controller?.abort();
    set({ ...idle });
  },
}));
