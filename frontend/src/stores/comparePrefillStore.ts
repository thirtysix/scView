import { create } from "zustand";

/** A Compare tab, as the co-pilot names it: Milo and miloDE are modes of Composition. */
export type CompareTarget = "composition" | "milo" | "milode" | "joint" | "planner";

/** What the co-pilot asked a Compare tab to open with (open_compare_tab). The tab
 *  applies the columns it can use and falls back to its own defaults for the rest;
 *  `nonce` makes the same request twice reapply. */
export interface ComparePrefill {
  tab: CompareTarget;
  condition?: string | null;
  group?: string | null;
  sample?: string | null;
  nonce: number;
}

interface ComparePrefillState {
  prefill: ComparePrefill | null;
  /** The nonce the Compare panel last opened, so a later visit does not reopen it. */
  applied: number;
  setPrefill: (p: Omit<ComparePrefill, "nonce">) => void;
  markApplied: (nonce: number) => void;
}

export const useComparePrefillStore = create<ComparePrefillState>((set) => ({
  prefill: null,
  applied: 0,
  setPrefill: (p) => set((s) => ({ prefill: { ...p, nonce: (s.prefill?.nonce ?? 0) + 1 } })),
  markApplied: (applied) => set({ applied }),
}));
