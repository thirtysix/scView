import { create } from "zustand";

// Split-by small multiples in the Unified View (lib/splitView.ts). Kept in a
// store so the toolbar control and the scatter area agree, and so a split
// survives the remount PanelContainer does on every tab switch.
interface SplitViewState {
  /** Obs column the scatter is split by; null = the single plot. */
  splitBy: string | null;
  setSplitBy: (column: string | null) => void;
  /** Draw every other cell as faint grey behind each panel's own. */
  showContext: boolean;
  setShowContext: (show: boolean) => void;
}

export const useSplitViewStore = create<SplitViewState>((set) => ({
  splitBy: null,
  setSplitBy: (column) => set({ splitBy: column || null }),
  showContext: true,
  setShowContext: (show) => set({ showContext: show }),
}));
