import { create } from "zustand";
import type { PanelId } from "@/lib/constants";

interface ViewState {
  activePanel: PanelId;
  sidebarCollapsed: boolean;
  pendingGene: string | null;
  setPanel: (panel: PanelId) => void;
  toggleSidebar: () => void;
  setPendingGene: (gene: string | null) => void;
}

export const useViewStore = create<ViewState>((set) => ({
  activePanel: "load",
  sidebarCollapsed: false,
  pendingGene: null,
  setPanel: (panel) => set({ activePanel: panel }),
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
  setPendingGene: (gene) => set({ pendingGene: gene }),
}));
