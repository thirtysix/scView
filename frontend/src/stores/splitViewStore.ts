import { create } from "zustand";

// Split-by small multiples in the Unified View (lib/splitView.ts), the
// "compare colourings" grid (the same cells coloured by 2–4 columns), and the
// gene grids (several genes side by side, or a two-gene blend). Kept in a
// store so the toolbar controls and the scatter area agree, and so a grid
// survives the remount PanelContainer does on every tab switch. One grid at a
// time: turning one on turns the others off.

/** The embedding coloured by each of several genes, or by two genes and their blend. */
export interface GeneGrid {
  datasetId: string;
  kind: "features" | "blend";
  genes: string[];
}

interface SplitViewState {
  /** Obs column the scatter is split by; null = the single plot. */
  splitBy: string | null;
  setSplitBy: (column: string | null) => void;
  /** Draw every other cell as faint grey behind each panel's own. */
  showContext: boolean;
  setShowContext: (show: boolean) => void;
  /** Columns shown side by side, one panel each; null = off. */
  compareColumns: string[] | null;
  setCompareColumns: (columns: string[] | null) => void;
  /** Genes shown side by side (or blended); null = off. */
  geneGrid: GeneGrid | null;
  setGeneGrid: (grid: GeneGrid | null) => void;
  /** Split view lasso: the panel's own cells under the outline, or every panel's. */
  lassoScope: "panel" | "all";
  setLassoScope: (scope: "panel" | "all") => void;
}

export const useSplitViewStore = create<SplitViewState>((set) => ({
  splitBy: null,
  setSplitBy: (column) =>
    set(column ? { splitBy: column, compareColumns: null, geneGrid: null } : { splitBy: null }),
  showContext: true,
  setShowContext: (show) => set({ showContext: show }),
  compareColumns: null,
  setCompareColumns: (columns) =>
    set(
      columns && columns.length > 0
        ? { compareColumns: columns, splitBy: null, geneGrid: null }
        : { compareColumns: null },
    ),
  geneGrid: null,
  setGeneGrid: (grid) =>
    set(grid && grid.genes.length > 0 ? { geneGrid: grid, splitBy: null, compareColumns: null } : { geneGrid: null }),
  lassoScope: "panel",
  setLassoScope: (scope) => set({ lassoScope: scope }),
}));

/** A grid covers the single plot (its panels take the lasso while it does). */
export const isGridActive = (
  s: Pick<SplitViewState, "splitBy" | "compareColumns"> & Partial<Pick<SplitViewState, "geneGrid">>,
): boolean => s.splitBy != null || s.compareColumns != null || s.geneGrid != null;
