import { create } from "zustand";
import { useUnifiedViewStore } from "./unifiedViewStore";

export type ReportMode = "cluster" | "gene";

/**
 * What the Report subtab (components/report/ReportSubtab.tsx) shows. A target is
 * null until the user picks one there or an entry point sets it (Markers row,
 * cluster map, a link inside a report); the subtab then follows the view instead
 * (the highlighted group, the gene on the plot). Targets belong to `datasetId`, so
 * a dataset switch falls back to the view.
 */
interface ReportState {
  mode: ReportMode;
  datasetId: string | null;
  cluster: { column: string; value: string } | null;
  gene: string | null;
  /** The gene report's grouping column; null = the Unified View's group-by. */
  groupby: string | null;
  setMode: (mode: ReportMode) => void;
  setCluster: (datasetId: string, cluster: { column: string; value: string } | null) => void;
  setGene: (datasetId: string, gene: string | null) => void;
  setGroupby: (datasetId: string, groupby: string | null) => void;
}

export const useReportStore = create<ReportState>((set, get) => {
  // Targets of another dataset are dropped when a new one is set.
  const own = (datasetId: string) =>
    get().datasetId === datasetId ? {} : { datasetId, cluster: null, gene: null, groupby: null };
  return {
    mode: "cluster",
    datasetId: null,
    cluster: null,
    gene: null,
    groupby: null,
    setMode: (mode) => set({ mode }),
    setCluster: (datasetId, cluster) => set({ ...own(datasetId), cluster }),
    setGene: (datasetId, gene) => set({ ...own(datasetId), gene }),
    setGroupby: (datasetId, groupby) => set({ ...own(datasetId), groupby }),
  };
});

/** Open the Report subtab on one group's report. */
export function openClusterReport(datasetId: string, column: string, value: string): void {
  const s = useReportStore.getState();
  s.setCluster(datasetId, { column, value });
  s.setMode("cluster");
  useUnifiedViewStore.getState().setActiveSubtab("report");
}

/** Open the Report subtab on one gene's report (grouped by `groupby` when given). */
export function openGeneReport(datasetId: string, gene: string, groupby?: string): void {
  const s = useReportStore.getState();
  s.setGene(datasetId, gene);
  if (groupby) s.setGroupby(datasetId, groupby);
  s.setMode("gene");
  useUnifiedViewStore.getState().setActiveSubtab("report");
}
