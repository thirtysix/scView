import { create } from "zustand";
import { runCnv, type CnvProgress, type CnvRunBody, type CnvStart, type CnvSummary } from "@/api/cnv";
import { watchdogMessage, type WatchdogEvent } from "@/api/pipelineRun";

/**
 * A copy-number run in flight (Copy number subtab). Kept here, not in the subtab,
 * so switching tabs does not abort it; Cancel aborts the stream, which stops the
 * run after its current block of cells without saving.
 */
interface CnvRunState {
  datasetId: string | null;
  running: boolean;
  start: CnvStart | null;
  progress: CnvProgress | null;
  warnings: string[];
  error: string | null;
  result: CnvSummary | null;
  /** Bumped when a run saves: the subtab reloads the summary and the dataset's columns. */
  version: number;
  run: (datasetId: string, body: CnvRunBody) => Promise<void>;
  cancel: () => void;
  reset: () => void;
}

let controller: AbortController | null = null;

const idle = { datasetId: null, running: false, start: null, progress: null, warnings: [], error: null, result: null };

export const useCnvRunStore = create<CnvRunState>((set, get) => ({
  ...idle,
  version: 0,
  run: async (datasetId, body) => {
    if (get().running) return;
    controller = new AbortController();
    const signal = controller.signal;
    set({ ...idle, datasetId, running: true });
    try {
      const result = await runCnv(datasetId, body, {
        signal,
        onStart: (start) => set({ start }),
        onProgress: (progress) => set({ progress }),
        onWatchdog: (w: WatchdogEvent) => {
          if (w.kind === "memory")
            set((s) => ({ warnings: [...s.warnings, watchdogMessage(w, () => "Copy number")] }));
        },
      });
      set((s) => ({ result, running: false, version: s.version + 1 }));
    } catch (e) {
      const aborted = signal.aborted;
      set({
        running: false,
        error: aborted ? "Cancelled: nothing was saved." : e instanceof Error ? e.message : String(e),
      });
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
