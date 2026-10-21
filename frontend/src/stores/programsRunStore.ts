import { create } from "zustand";
import {
  runPrograms,
  type ProgramsKStats,
  type ProgramsProgress,
  type ProgramsRunBody,
  type ProgramsStart,
  type ProgramsSummary,
} from "@/api/programs";
import { watchdogMessage, type WatchdogEvent } from "@/api/pipelineRun";

/**
 * A gene programs run in flight (Programs subtab). Kept here, not in the subtab,
 * so switching tabs does not abort it; Cancel aborts the stream, which stops the
 * run after its current replicate without saving.
 */
interface ProgramsRunState {
  datasetId: string | null;
  running: boolean;
  start: ProgramsStart | null;
  progress: ProgramsProgress | null;
  kResults: ProgramsKStats[];
  warnings: string[];
  error: string | null;
  /** The saved programs of the last run that finished, for that dataset. */
  result: ProgramsSummary | null;
  /** Bumped when a run saves: the subtab reloads the summary. */
  version: number;
  run: (datasetId: string, body: ProgramsRunBody) => Promise<void>;
  cancel: () => void;
  reset: () => void;
}

let controller: AbortController | null = null;

const idle = {
  datasetId: null,
  running: false,
  start: null,
  progress: null,
  kResults: [],
  warnings: [],
  error: null,
  result: null,
};

export const useProgramsRunStore = create<ProgramsRunState>((set, get) => ({
  ...idle,
  version: 0,
  run: async (datasetId, body) => {
    if (get().running) return;
    controller = new AbortController();
    const signal = controller.signal;
    set({ ...idle, datasetId, running: true });
    try {
      const result = await runPrograms(datasetId, body, {
        signal,
        onStart: (start) => set({ start }),
        onProgress: (progress) => set({ progress }),
        onK: (k) => set((s) => ({ kResults: [...s.kResults, k] })),
        onWatchdog: (w: WatchdogEvent) =>
          set((s) => ({ warnings: [...s.warnings, watchdogMessage(w, () => "Gene programs")] })),
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
