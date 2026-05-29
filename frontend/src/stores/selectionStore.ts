import { create } from "zustand";

interface SelectionState {
  selectedCellIndices: Set<number> | null;
  selectionMode: "lasso" | "click" | "none";
  highlightedGroup: { column: string; value: string } | null;
  setSelection: (indices: Set<number> | null) => void;
  setSelectionMode: (mode: "lasso" | "click" | "none") => void;
  clearSelection: () => void;
  setHighlight: (group: { column: string; value: string } | null) => void;
}

export const useSelectionStore = create<SelectionState>((set) => ({
  selectedCellIndices: null,
  selectionMode: "none",
  highlightedGroup: null,
  setSelection: (indices) => set({ selectedCellIndices: indices }),
  setSelectionMode: (mode) => set({ selectionMode: mode }),
  clearSelection: () => set({ selectedCellIndices: null, selectionMode: "none" }),
  setHighlight: (group) => set({ highlightedGroup: group }),
}));
