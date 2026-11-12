import { create } from "zustand";
import { DEFAULT_PANELS_PER_PAGE, type PanelsPerPage } from "@/lib/gridPaging";

// Split-by small multiples in the Unified View (lib/splitView.ts), the
// "compare colourings" grid (the same cells coloured by 2–12 columns), and the
// gene grids (several genes side by side, or a two-gene blend). Kept in a
// store so the toolbar controls and the scatter area agree, and so a grid
// survives the remount PanelContainer does on every tab switch. One grid at a
// time: turning one on turns the others off. The grids share one page size and
// page (lib/gridPaging.ts); a different split, comparison or gene grid starts
// on page 1.

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
  /** Panels a grid page shows (4 / 6 / 9 / 12 / all) */
  panelsPerPage: PanelsPerPage;
  /** Page size, and the page to show with it (GridPager passes `repage`, which keeps the panels in view) */
  setPanelsPerPage: (perPage: PanelsPerPage, page?: number) => void;
  /** Current page, 0-based (the grid clamps it to its panel count) */
  page: number;
  setPage: (page: number) => void;
}

const sameList = (a: readonly string[] | null | undefined, b: readonly string[] | null | undefined) =>
  !!a && !!b && a.length === b.length && a.every((v, i) => v === b[i]);

export const useSplitViewStore = create<SplitViewState>((set) => ({
  splitBy: null,
  setSplitBy: (column) =>
    set((s) =>
      column
        ? { splitBy: column, compareColumns: null, geneGrid: null, page: column === s.splitBy ? s.page : 0 }
        : { splitBy: null },
    ),
  showContext: true,
  setShowContext: (show) => set({ showContext: show }),
  compareColumns: null,
  setCompareColumns: (columns) =>
    set((s) =>
      columns && columns.length > 0
        ? {
            compareColumns: columns,
            splitBy: null,
            geneGrid: null,
            page: sameList(columns, s.compareColumns) ? s.page : 0,
          }
        : { compareColumns: null },
    ),
  geneGrid: null,
  // Adding or removing a gene keeps the page (the grid clamps it); another kind
  // of grid, or another dataset's, starts on page 1.
  setGeneGrid: (grid) =>
    set((s) =>
      grid && grid.genes.length > 0
        ? {
            geneGrid: grid,
            splitBy: null,
            compareColumns: null,
            page: s.geneGrid && s.geneGrid.kind === grid.kind && s.geneGrid.datasetId === grid.datasetId ? s.page : 0,
          }
        : { geneGrid: null },
    ),
  lassoScope: "panel",
  setLassoScope: (scope) => set({ lassoScope: scope }),
  panelsPerPage: DEFAULT_PANELS_PER_PAGE,
  setPanelsPerPage: (panelsPerPage, page) => set((s) => ({ panelsPerPage, page: page ?? s.page })),
  page: 0,
  setPage: (page) => set({ page: Math.max(0, Math.floor(page)) }),
}));

/** A grid covers the single plot (its panels take the lasso while it does). */
export const isGridActive = (
  s: Pick<SplitViewState, "splitBy" | "compareColumns"> & Partial<Pick<SplitViewState, "geneGrid">>,
): boolean => s.splitBy != null || s.compareColumns != null || s.geneGrid != null;
