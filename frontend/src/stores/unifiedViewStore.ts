import { create } from "zustand";
import type { LabelChoice } from "@/lib/centroidLabels";
import type { UnifiedSubtab } from "@/lib/constants";
import type { ClipSetting } from "@/lib/colorScale";
import type { Camera, SessionOverlay } from "@/lib/session";
import type { ReliabilityRef } from "@/lib/embeddingReliability";
import type { ScoreMethod } from "@/api/genesetScore";

type ScatterOverlay =
  | { type: "obs"; column: string }
  // layer: set for a protein (CITE-seq), whose values come from its own layer
  | { type: "expression"; gene: string; layer?: string }
  // genes: the set scored, so a saved session can score it again; activity: the
  // TF / pathway whose per-cell activity this is, for the same reason; reliability:
  // the plot whose neighbour preservation this is (a fixed 0-1 scale); method: how
  // the gene set was scored (absent = score_genes)
  | {
      type: "score";
      name: string;
      genes?: string[];
      activity?: { network: "collectri" | "progeny"; source: string };
      reliability?: ReliabilityRef;
      method?: ScoreMethod;
    }
  // One gene program's per-cell usage share (Programs subtab; 1-based index, P1..Pk)
  | { type: "program"; index: number; name: string }
  // Expression-weighted kernel density (lib/density.ts) of one gene, or the joint
  // density of several, on the embedding (and PCA pair) and layer it was computed for
  | { type: "density"; genes: string[]; embedding: string; dims?: [number, number]; layer: string }
  | null;

type Subtab = UnifiedSubtab;

export type BottomView = "distribution" | "distances";

/** A camera on one dataset's embedding. */
export interface PlacedCamera {
  datasetId: string;
  embedding: string;
  camera: Camera;
}

interface UnifiedViewState {
  activeSubtab: Subtab;
  setActiveSubtab: (tab: Subtab) => void;

  scatterOverlay: ScatterOverlay;
  setScatterOverlay: (overlay: ScatterOverlay) => void;
  // A gene picked anywhere is drawn as its weighted density, not its expression
  densityMode: boolean;
  setDensityMode: (on: boolean) => void;

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
  // This view's colour-range clipping, over the settings default (null = the
  // default); saved with the session, dropped with the overlay on a dataset switch.
  colorClip: ClipSetting | null;
  setColorClip: (clip: ClipSetting | null) => void;

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
  // What the panel under the scatter shows: the violin, or the distances
  // between groups in PC space (the UMAP / t-SNE caveat opens it there).
  bottomView: BottomView;
  setBottomView: (view: BottomView) => void;
  showDistances: () => void;

  activeGene: string | null;
  setActiveGene: (gene: string | null) => void;

  groupByColumn: string;
  setGroupByColumn: (col: string) => void;

  // Group labels on the plot: the user's on/off for one colouring (a column, or
  // a compare set); any other colouring gets the default (lib/centroidLabels.ts).
  centroidLabels: LabelChoice | null;
  setCentroidLabels: (choice: LabelChoice | null) => void;

  // The main plot's camera once the user has moved it (null = fitted), kept
  // for "Save session"; nothing renders from it.
  camera: PlacedCamera | null;
  setCamera: (camera: PlacedCamera | null) => void;
  // From "Open session", applied by the Unified View when it shows that
  // dataset (and, for the camera, that embedding).
  pendingCamera: PlacedCamera | null;
  setPendingCamera: (camera: PlacedCamera | null) => void;
  pendingOverlay: { datasetId: string; overlay: SessionOverlay } | null;
  setPendingOverlay: (pending: { datasetId: string; overlay: SessionOverlay } | null) => void;
}

export const useUnifiedViewStore = create<UnifiedViewState>((set) => ({
  centroidLabels: null,
  setCentroidLabels: (choice) => set({ centroidLabels: choice }),

  camera: null,
  setCamera: (camera) => set({ camera }),
  pendingCamera: null,
  setPendingCamera: (camera) => set({ pendingCamera: camera }),
  pendingOverlay: null,
  setPendingOverlay: (pending) => set({ pendingOverlay: pending }),

  activeSubtab: "markers",
  setActiveSubtab: (tab) => set({ activeSubtab: tab }),

  scatterOverlay: null,
  setScatterOverlay: (overlay) => set({ scatterOverlay: overlay }),
  densityMode: false,
  setDensityMode: (on) => set({ densityMode: on }),

  overlayValues: null,
  setOverlayValues: (values) => set({ overlayValues: values }),
  overlayLabel: "",
  setOverlayLabel: (label) => set({ overlayLabel: label }),
  overlayDatasetId: null,
  setOverlayDatasetId: (id) => set({ overlayDatasetId: id }),
  colorClip: null,
  setColorClip: (clip) => set({ colorClip: clip }),

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
      colorClip: null,
      violinData: {},
      violinTitle: "",
    }),

  bottomPanelOpen: false,
  toggleBottomPanel: () => set((s) => ({ bottomPanelOpen: !s.bottomPanelOpen })),
  // A new gene or score is shown in the violin, so it brings that view back.
  openBottomPanel: () => set({ bottomPanelOpen: true, bottomView: "distribution" }),
  bottomView: "distribution",
  setBottomView: (view) => set({ bottomView: view }),
  showDistances: () => set({ bottomPanelOpen: true, bottomView: "distances" }),

  activeGene: null,
  setActiveGene: (gene) => set({ activeGene: gene }),

  groupByColumn: "",
  setGroupByColumn: (col) => set({ groupByColumn: col }),
}));
