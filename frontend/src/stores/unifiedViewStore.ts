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

  // Overlay payload — lifted out of UnifiedViewPanel so it survives the
  // unmount/remount that PanelContainer triggers on every tab switch
  // (see PanelContainer's key={activePanel}). overlayDatasetId guards against
  // painting one dataset's values onto another after a dataset switch.
  overlayValues: Float32Array | null;
  setOverlayValues: (values: Float32Array | null) => void;
  overlayLabel: string;
  setOverlayLabel: (label: string) => void;
  overlayDatasetId: string | null;
  setOverlayDatasetId: (id: string | null) => void;

  // Violin payload (set alongside the overlay) — persisted for the same reason
  violinData: Record<string, number[]>;
  setViolinData: (data: Record<string, number[]>) => void;
  violinTitle: string;
  setViolinTitle: (title: string) => void;

  // Clear the overlay + violin in one shot (revert to obs coloring)
  clearOverlayState: () => void;

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

  overlayValues: null,
  setOverlayValues: (values) => set({ overlayValues: values }),
  overlayLabel: "",
  setOverlayLabel: (label) => set({ overlayLabel: label }),
  overlayDatasetId: null,
  setOverlayDatasetId: (id) => set({ overlayDatasetId: id }),

  violinData: {},
  setViolinData: (data) => set({ violinData: data }),
  violinTitle: "",
  setViolinTitle: (title) => set({ violinTitle: title }),

  clearOverlayState: () =>
    set({
      scatterOverlay: null,
      overlayValues: null,
      overlayLabel: "",
      overlayDatasetId: null,
      violinData: {},
      violinTitle: "",
    }),

  bottomPanelOpen: false,
  toggleBottomPanel: () => set((s) => ({ bottomPanelOpen: !s.bottomPanelOpen })),
  openBottomPanel: () => set({ bottomPanelOpen: true }),

  activeGene: null,
  setActiveGene: (gene) => set({ activeGene: gene }),

  groupByColumn: "",
  setGroupByColumn: (col) => set({ groupByColumn: col }),
}));
