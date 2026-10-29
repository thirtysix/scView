import { create } from "zustand";
import { runWnn, type WnnInfo, type WnnRunBody, type WnnStage, type WnnStart } from "@/api/wnn";
import { watchdogMessage, type WatchdogEvent } from "@/api/pipelineRun";

/**
 * A WNN run in flight (Protein subtab). Kept here, not in the card, so switching
 * tabs does not abort it; Cancel aborts the stream, which stops the run before its
 * next stage without saving.
 */
interface WnnRunState {
  datasetId: string | null;
  running: boolean;
  start: WnnStart | null;
  stage: WnnStage | null;
  warnings: string[];
  error: string | null;
  /** Bumped when a run saves: the card reloads the result and the dataset. */
  version: number;
  run: (datasetId: string, body: WnnRunBody) => Promise<WnnInfo | null>;
  cancel: () => void;
  reset: () => void;
}

let controller: AbortController | null = null;

const idle = { datasetId: null, running: false, start: null, stage: null, warnings: [], error: null };

export const useWnnRunStore = create<WnnRunState>((set, get) => ({
  ...idle,
  version: 0,
  run: async (datasetId, body) => {
    if (get().running) return null;
    controller = new AbortController();
    const signal = controller.signal;
    set({ ...idle, datasetId, running: true });
    try {
      const result = await runWnn(datasetId, body, {
        signal,
        onStart: (start) => set({ start }),
        onStage: (stage) => set({ stage }),
        onWatchdog: (w: WatchdogEvent) => {
          if (w.kind === "memory") set((s) => ({ warnings: [...s.warnings, watchdogMessage(w, () => "WNN")] }));
        },
      });
      set((s) => ({ running: false, version: s.version + 1 }));
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
