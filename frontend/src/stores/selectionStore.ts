import { create } from "zustand";
import type { CellFilter } from "@/lib/cellFilter";

interface SelectionState {
  selectedCellIndices: Set<number> | null;
  /** The filter the selection came from (CellFilterControl); null for a lasso,
   *  a legend click or any other selection. */
  selectionFilter: CellFilter | null;
  selectionMode: "lasso" | "click" | "none";
  highlightedGroup: { column: string; value: string } | null;
  setSelection: (indices: Set<number> | null, filter?: CellFilter | null) => void;
  setSelectionMode: (mode: "lasso" | "click" | "none") => void;
  clearSelection: () => void;
  setHighlight: (group: { column: string; value: string } | null) => void;
}

export const useSelectionStore = create<SelectionState>((set) => ({
  selectedCellIndices: null,
  selectionFilter: null,
  selectionMode: "none",
  highlightedGroup: null,
  setSelection: (indices, filter = null) =>
    set({ selectedCellIndices: indices, selectionFilter: indices ? filter : null }),
  setSelectionMode: (mode) => set({ selectionMode: mode }),
  clearSelection: () => set({ selectedCellIndices: null, selectionFilter: null, selectionMode: "none" }),
  setHighlight: (group) => set({ highlightedGroup: group }),
}));
