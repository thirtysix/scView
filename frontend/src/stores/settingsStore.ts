import { create } from "zustand";
import type { CategoricalPaletteName, ColormapName } from "@/lib/colors";

interface SettingsState {
  embedding: string;
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
  maxRenderedCells: number;
  pipelineRunning: boolean;
  setEmbedding: (embedding: string) => void;
  setColorBy: (colorBy: string) => void;
  /** Colour a newly opened dataset by its default grouping (lib/obsColumns.ts) */
  setDefaultColorBy: (colorBy: string) => void;
  setExpressionLayer: (layer: string) => void;
  setPointSize: (size: number) => void;
  setOpacity: (opacity: number) => void;
  setPlotBackground: (bg: "white" | "dark") => void;
  setCategoricalPalette: (p: CategoricalPaletteName) => void;
  setContinuousColormap: (m: ColormapName) => void;
  setMaxRenderedCells: (max: number) => void;
  setPipelineRunning: (running: boolean) => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  embedding: "X_umap",
  colorBy: "",
  colorByChosen: false,
  expressionLayer: "",
  pointSize: 2,
  opacity: 0.8,
  plotBackground: "white",
  categoricalPalette: "cvd",
  continuousColormap: "viridis",
  maxRenderedCells: 100_000,
  pipelineRunning: false,
  setEmbedding: (embedding) => set({ embedding }),
  setColorBy: (colorBy) => set({ colorBy, colorByChosen: true }),
  setDefaultColorBy: (colorBy) => set({ colorBy, colorByChosen: false }),
  setExpressionLayer: (layer) => set({ expressionLayer: layer }),
  setPointSize: (size) => set({ pointSize: size }),
  setOpacity: (opacity) => set({ opacity }),
  setPlotBackground: (bg) => set({ plotBackground: bg }),
  setCategoricalPalette: (categoricalPalette) => set({ categoricalPalette }),
  setContinuousColormap: (continuousColormap) => set({ continuousColormap }),
  setMaxRenderedCells: (max) => set({ maxRenderedCells: max }),
  setPipelineRunning: (running) => set({ pipelineRunning: running }),
}));
