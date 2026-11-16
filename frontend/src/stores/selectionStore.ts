import { create } from "zustand";
import type { CellFilter } from "@/lib/cellFilter";

interface SelectionState {
  selectedCellIndices: Set<number> | null;
  /** The filter the selection came from (CellFilterControl); null for a lasso,
   *  a legend click or any other selection. */
  selectionFilter: CellFilter | null;
  /** The groups the selection is (a legend click, legend picks, "By group"); null otherwise.
   *  A new dataset or a label made from it records them (backend core/subset.py). */
  selectionGroups: { column: string; values: string[] } | null;
  selectionMode: "lasso" | "click" | "none";
  highlightedGroup: { column: string; value: string } | null;
  /** Whether the highlight should also select its cells on the main plot (as a
   *  legend click does); false for a highlight restored from a session or link,
   *  whose own selection is restored separately. */
  highlightSelects: boolean;
  setSelection: (indices: Set<number> | null, filter?: CellFilter | null) => void;
  /** The selection as whole groups of one column */
  setGroupSelection: (indices: Set<number>, column: string, values: string[]) => void;
  setSelectionMode: (mode: "lasso" | "click" | "none") => void;
  clearSelection: () => void;
  setHighlight: (group: { column: string; value: string } | null, opts?: { select?: boolean }) => void;
}

export const useSelectionStore = create<SelectionState>((set) => ({
  selectedCellIndices: null,
  selectionFilter: null,
  selectionGroups: null,
  selectionMode: "none",
  highlightedGroup: null,
  highlightSelects: true,
  setSelection: (indices, filter = null) =>
    set({ selectedCellIndices: indices, selectionFilter: indices ? filter : null, selectionGroups: null }),
  setGroupSelection: (indices, column, values) =>
    set({ selectedCellIndices: indices, selectionFilter: null, selectionGroups: { column, values: [...values] } }),
  setSelectionMode: (mode) => set({ selectionMode: mode }),
  clearSelection: () =>
    set({ selectedCellIndices: null, selectionFilter: null, selectionGroups: null, selectionMode: "none" }),
  setHighlight: (group, opts) => set({ highlightedGroup: group, highlightSelects: opts?.select ?? true }),
}));
