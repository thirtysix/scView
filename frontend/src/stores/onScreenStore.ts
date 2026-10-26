import { create } from "zustand";

/** The results a tab is showing, kept for the co-pilot's view context
 *  (lib/copilotView.ts): "Explain this result" names what is on screen. Each tab
 *  sets its own slot and clears it when it unmounts or its result goes. */
export interface OnScreenDE {
  id: string;
  label: string;
}
export interface OnScreenEnrichment {
  column: string;
  group: string;
}

interface OnScreenState {
  de: OnScreenDE | null;
  enrichment: OnScreenEnrichment | null;
  compareTab: string | null;
  setDE: (de: OnScreenDE | null) => void;
  setEnrichment: (e: OnScreenEnrichment | null) => void;
  setCompareTab: (tab: string | null) => void;
}

export const useOnScreenStore = create<OnScreenState>((set) => ({
  de: null,
  enrichment: null,
  compareTab: null,
  setDE: (de) => set({ de }),
  setEnrichment: (enrichment) => set({ enrichment }),
  setCompareTab: (compareTab) => set({ compareTab }),
}));
