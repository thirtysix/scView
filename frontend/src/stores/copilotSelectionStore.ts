import { create } from "zustand";

/**
 * The co-pilot's view of the cell selection (lib/copilotSelection.ts). Keyed by
 * the selection store's Set itself: every new selection is a new Set, so a
 * selection left out of the chat's context comes back when the cells change, and
 * a selection is "fresh" until a question has been sent with it.
 */
interface CopilotSelectionState {
  /** The selection the user took out of the chat's context (the chip's remove) */
  excluded: Set<number> | null;
  /** The selection the last question carried */
  lastSent: Set<number> | null;
  exclude: (cells: Set<number> | null) => void;
  include: () => void;
  markSent: (cells: Set<number> | null) => void;
}

export const useCopilotSelectionStore = create<CopilotSelectionState>((set) => ({
  excluded: null,
  lastSent: null,
  exclude: (cells) => set({ excluded: cells }),
  include: () => set({ excluded: null }),
  markSent: (cells) => set({ lastSent: cells }),
}));
