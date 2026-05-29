import { create } from "zustand";

interface SettingsState {
  embedding: string;
  colorBy: string;
  pointSize: number;
  opacity: number;
  plotBackground: "white" | "dark";
  maxRenderedCells: number;
  pipelineRunning: boolean;
  setEmbedding: (embedding: string) => void;
  setColorBy: (colorBy: string) => void;
  setPointSize: (size: number) => void;
  setOpacity: (opacity: number) => void;
  setPlotBackground: (bg: "white" | "dark") => void;
  setMaxRenderedCells: (max: number) => void;
  setPipelineRunning: (running: boolean) => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  embedding: "X_umap",
  colorBy: "",
  pointSize: 2,
  opacity: 0.8,
  plotBackground: "white",
  maxRenderedCells: 100_000,
  pipelineRunning: false,
  setEmbedding: (embedding) => set({ embedding }),
  setColorBy: (colorBy) => set({ colorBy }),
  setPointSize: (size) => set({ pointSize: size }),
  setOpacity: (opacity) => set({ opacity }),
  setPlotBackground: (bg) => set({ plotBackground: bg }),
  setMaxRenderedCells: (max) => set({ maxRenderedCells: max }),
  setPipelineRunning: (running) => set({ pipelineRunning: running }),
}));
