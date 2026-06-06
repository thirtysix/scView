import { create } from "zustand";

interface SettingsState {
  embedding: string;
  colorBy: string;
  expressionLayer: string; // units for expression overlay/violin (obs layer key)
  pointSize: number;
  opacity: number;
  plotBackground: "white" | "dark";
  maxRenderedCells: number;
  pipelineRunning: boolean;
  setEmbedding: (embedding: string) => void;
  setColorBy: (colorBy: string) => void;
  setExpressionLayer: (layer: string) => void;
  setPointSize: (size: number) => void;
  setOpacity: (opacity: number) => void;
  setPlotBackground: (bg: "white" | "dark") => void;
  setMaxRenderedCells: (max: number) => void;
  setPipelineRunning: (running: boolean) => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  embedding: "X_umap",
  colorBy: "",
  expressionLayer: "",
  pointSize: 2,
  opacity: 0.8,
  plotBackground: "white",
  maxRenderedCells: 100_000,
  pipelineRunning: false,
  setEmbedding: (embedding) => set({ embedding }),
  setColorBy: (colorBy) => set({ colorBy }),
  setExpressionLayer: (layer) => set({ expressionLayer: layer }),
  setPointSize: (size) => set({ pointSize: size }),
  setOpacity: (opacity) => set({ opacity }),
  setPlotBackground: (bg) => set({ plotBackground: bg }),
  setMaxRenderedCells: (max) => set({ maxRenderedCells: max }),
  setPipelineRunning: (running) => set({ pipelineRunning: running }),
}));
