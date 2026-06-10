import { create } from "zustand";
import type { PanelId } from "@/lib/constants";

interface ViewState {
  activePanel: PanelId;
  sidebarCollapsed: boolean;
  pendingGene: string | null;
  copilotOpen: boolean;
  copilotWidth: number;
  /** A question queued for the co-pilot by an "ask about this" affordance. */
  pendingAsk: string | null;
  setPanel: (panel: PanelId) => void;
  toggleSidebar: () => void;
  setPendingGene: (gene: string | null) => void;
  toggleCopilot: () => void;
  setCopilotOpen: (open: boolean) => void;
  setCopilotWidth: (width: number) => void;
  /** Open the co-pilot and queue a contextual question for it to answer. */
  askCopilot: (question: string) => void;
  clearPendingAsk: () => void;
}

export const useViewStore = create<ViewState>((set) => ({
  activePanel: "load",
  sidebarCollapsed: false,
  pendingGene: null,
  copilotOpen: false,
  copilotWidth: 400,
  pendingAsk: null,
  setPanel: (panel) => set({ activePanel: panel }),
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
  setPendingGene: (gene) => set({ pendingGene: gene }),
  toggleCopilot: () => set((s) => ({ copilotOpen: !s.copilotOpen })),
  setCopilotOpen: (open) => set({ copilotOpen: open }),
  setCopilotWidth: (width) => set({ copilotWidth: width }),
  askCopilot: (question) => set({ copilotOpen: true, pendingAsk: question }),
  clearPendingAsk: () => set({ pendingAsk: null }),
}));
