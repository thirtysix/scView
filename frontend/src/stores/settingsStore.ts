import { create } from "zustand";
import type { CategoricalPaletteName, ColormapName, DivergingColormapName } from "@/lib/colors";
import { DEFAULT_CLIP, type ClipSetting, type SignedColormapMode } from "@/lib/colorScale";

interface SettingsState {
  embedding: string;
  /** Which two columns of a PCA embedding the scatter shows (0-based); [0, 1] = PC1 vs PC2. */
  pcaDims: [number, number];
  colorBy: string;
  /** colorBy was set by the user (or on their behalf), not picked as a dataset's
   *  default: opening another dataset keeps it when that dataset has the column. */
  colorByChosen: boolean;
  expressionLayer: string; // units for expression overlay/violin (obs layer key)
  pointSize: number;
  opacity: number;
  plotBackground: "white" | "dark";
  categoricalPalette: CategoricalPaletteName;
  continuousColormap: ColormapName;
  /** For signed scores (lib/colorScale.ts decides when) */
  divergingColormap: DivergingColormapName;
  signedColormapMode: SignedColormapMode;
  /** Default colour-range clipping; a view may override it (unifiedViewStore.colorClip) */
  colorClip: ClipSetting;
  maxRenderedCells: number;
  pipelineRunning: boolean;
  setEmbedding: (embedding: string) => void;
  setPcaDims: (dims: [number, number]) => void;
  setColorBy: (colorBy: string) => void;
  /** Colour a newly opened dataset by its default grouping (lib/obsColumns.ts) */
  setDefaultColorBy: (colorBy: string) => void;
  setExpressionLayer: (layer: string) => void;
  setPointSize: (size: number) => void;
  setOpacity: (opacity: number) => void;
  setPlotBackground: (bg: "white" | "dark") => void;
  setCategoricalPalette: (p: CategoricalPaletteName) => void;
  setContinuousColormap: (m: ColormapName) => void;
  setDivergingColormap: (m: DivergingColormapName) => void;
  setSignedColormapMode: (m: SignedColormapMode) => void;
  setColorClip: (c: ClipSetting) => void;
  setMaxRenderedCells: (max: number) => void;
  setPipelineRunning: (running: boolean) => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  embedding: "X_umap",
  pcaDims: [0, 1],
  colorBy: "",
  colorByChosen: false,
  expressionLayer: "",
  pointSize: 2,
  opacity: 0.8,
  plotBackground: "white",
  categoricalPalette: "cvd",
  continuousColormap: "viridis",
  divergingColormap: "RdBu",
  signedColormapMode: "auto",
  colorClip: DEFAULT_CLIP,
  maxRenderedCells: 100_000,
  pipelineRunning: false,
  setEmbedding: (embedding) => set({ embedding }),
  setPcaDims: (pcaDims) => set({ pcaDims }),
  setColorBy: (colorBy) => set({ colorBy, colorByChosen: true }),
  setDefaultColorBy: (colorBy) => set({ colorBy, colorByChosen: false }),
  setExpressionLayer: (layer) => set({ expressionLayer: layer }),
  setPointSize: (size) => set({ pointSize: size }),
  setOpacity: (opacity) => set({ opacity }),
  setPlotBackground: (bg) => set({ plotBackground: bg }),
  setCategoricalPalette: (categoricalPalette) => set({ categoricalPalette }),
  setContinuousColormap: (continuousColormap) => set({ continuousColormap }),
  setDivergingColormap: (divergingColormap) => set({ divergingColormap }),
  setSignedColormapMode: (signedColormapMode) => set({ signedColormapMode }),
  setColorClip: (colorClip) => set({ colorClip }),
  setMaxRenderedCells: (max) => set({ maxRenderedCells: max }),
  setPipelineRunning: (running) => set({ pipelineRunning: running }),
}));
