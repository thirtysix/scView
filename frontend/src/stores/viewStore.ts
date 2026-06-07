import { create } from "zustand";
import type { PanelId } from "@/lib/constants";

interface ViewState {
  activePanel: PanelId;
  sidebarCollapsed: boolean;
  pendingGene: string | null;
  copilotOpen: boolean;
  setPanel: (panel: PanelId) => void;
  toggleSidebar: () => void;
  setPendingGene: (gene: string | null) => void;
  toggleCopilot: () => void;
  setCopilotOpen: (open: boolean) => void;
}

export const useViewStore = create<ViewState>((set) => ({
  activePanel: "load",
  sidebarCollapsed: false,
  pendingGene: null,
  copilotOpen: false,
  setPanel: (panel) => set({ activePanel: panel }),
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
  setPendingGene: (gene) => set({ pendingGene: gene }),
  toggleCopilot: () => set((s) => ({ copilotOpen: !s.copilotOpen })),
  setCopilotOpen: (open) => set({ copilotOpen: open }),
}));
