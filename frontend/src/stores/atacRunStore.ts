import { create } from "zustand";
import {
  runAtac,
  runAtacDa,
  runAtacWnn,
  type AtacDa,
  type AtacDaBody,
  type AtacRunBody,
  type AtacStage,
  type AtacWnnBody,
} from "@/api/atac";
import { watchdogMessage, type WatchdogEvent } from "@/api/pipelineRun";

export type AtacRunKind = "run" | "wnn" | "differential";

const WHAT: Record<AtacRunKind, string> = { run: "ATAC analysis", wnn: "WNN", differential: "Differential accessibility" };

/**
 * An ATAC run in flight (Accessibility subtab). Kept here, not in a card, so switching
 * tabs does not abort it; Cancel aborts the stream, which stops the run before its next
 * block without saving. One at a time: the backend runs one per dataset.
 */
interface AtacRunState {
  datasetId: string | null;
  kind: AtacRunKind | null;
  running: boolean;
  stage: AtacStage | null;
  progress: string | null;
  warnings: string[];
  error: string | null;
  /** Bumped when a run saves: the tab reloads its state and the dataset. */
  version: number;
  /** The last differential result, for the table. */
  da: AtacDa | null;
  run: (datasetId: string, body: AtacRunBody) => Promise<void>;
  runWnn: (datasetId: string, body: AtacWnnBody) => Promise<void>;
  runDa: (datasetId: string, body: AtacDaBody) => Promise<void>;
  cancel: () => void;
}

let controller: AbortController | null = null;

const idle = { datasetId: null, kind: null, running: false, stage: null, progress: null, warnings: [], error: null };

export const useAtacRunStore = create<AtacRunState>((set, get) => {
  const start = async (datasetId: string, kind: AtacRunKind, go: (h: {
    signal: AbortSignal;
    onStage: (s: AtacStage) => void;
    onProgress: (m: string) => void;
    onWatchdog: (w: WatchdogEvent) => void;
  }) => Promise<void>) => {
    if (get().running) return;
    controller = new AbortController();
    const signal = controller.signal;
    set({ ...idle, datasetId, kind, running: true });
    try {
      await go({
        signal,
        onStage: (stage) => set({ stage }),
        onProgress: (progress) => set({ progress }),
        onWatchdog: (w) => {
          if (w.kind === "memory") set((s) => ({ warnings: [...s.warnings, watchdogMessage(w, () => WHAT[kind])] }));
        },
      });
      set((s) => ({ running: false, version: s.version + 1 }));
    } catch (e) {
      const aborted = signal.aborted;
      set({ running: false, error: aborted ? "Cancelled: nothing was saved." : e instanceof Error ? e.message : String(e) });
    } finally {
      if (controller?.signal === signal) controller = null;
    }
  };
  return {
    ...idle,
    version: 0,
    da: null,
    run: (datasetId, body) => start(datasetId, "run", async (h) => void (await runAtac(datasetId, body, h))),
    runWnn: (datasetId, body) => start(datasetId, "wnn", async (h) => void (await runAtacWnn(datasetId, body, h))),
    runDa: (datasetId, body) =>
      start(datasetId, "differential", async (h) => {
        const da = await runAtacDa(datasetId, body, h);
        set({ da });
      }),
    cancel: () => controller?.abort(),
  };
});
