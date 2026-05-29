import { create } from "zustand";

type ScatterOverlay =
  | { type: "obs"; column: string }
  | { type: "expression"; gene: string }
  | { type: "score"; name: string }
  | null;

type Subtab = "markers" | "expression" | "genesets" | "enrichment";

interface UnifiedViewState {
  activeSubtab: Subtab;
  setActiveSubtab: (tab: Subtab) => void;

  scatterOverlay: ScatterOverlay;
  setScatterOverlay: (overlay: ScatterOverlay) => void;

  bottomPanelOpen: boolean;
  toggleBottomPanel: () => void;
  openBottomPanel: () => void;

  activeGene: string | null;
  setActiveGene: (gene: string | null) => void;

  groupByColumn: string;
  setGroupByColumn: (col: string) => void;
}

export const useUnifiedViewStore = create<UnifiedViewState>((set) => ({
  activeSubtab: "markers",
  setActiveSubtab: (tab) => set({ activeSubtab: tab }),

  scatterOverlay: null,
  setScatterOverlay: (overlay) => set({ scatterOverlay: overlay }),

  bottomPanelOpen: false,
  toggleBottomPanel: () => set((s) => ({ bottomPanelOpen: !s.bottomPanelOpen })),
  openBottomPanel: () => set({ bottomPanelOpen: true }),

  activeGene: null,
  setActiveGene: (gene) => set({ activeGene: gene }),

  groupByColumn: "",
  setGroupByColumn: (col) => set({ groupByColumn: col }),
}));
