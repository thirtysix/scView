import { create } from "zustand";
import { createJSONStorage, persist, type StateStorage } from "zustand/middleware";
import type { ScoreMethod } from "@/api/genesetScore";

// Never throws: blocked storage leaves the choice working for this page.
const storage: StateStorage = {
  getItem: (key) => {
    try { return window.localStorage.getItem(key); } catch { return null; }
  },
  setItem: (key, value) => {
    try { window.localStorage.setItem(key, value); } catch { /* not kept */ }
  },
  removeItem: (key) => {
    try { window.localStorage.removeItem(key); } catch { /* nothing to do */ }
  },
};

/**
 * The gene-set scoring method the user picked (score_genes or AUCell), shared by
 * every scoring control and kept in localStorage: it is the user's choice, not a
 * dataset's. A score already on the plot keeps the method it was computed with.
 */
interface ScoringMethodState {
  method: ScoreMethod;
  setMethod: (method: ScoreMethod) => void;
}

export const useScoringMethodStore = create<ScoringMethodState>()(
  persist(
    (set) => ({
      method: "score_genes",
      setMethod: (method) => set({ method }),
    }),
    {
      name: "scview.scoringMethod.v1",
      storage: createJSONStorage(() => storage),
      partialize: (s) => ({ method: s.method }),
      // Anything but a known method (an edited or future value) falls back to the default.
      merge: (persisted, current) => {
        const m = (persisted as { method?: unknown } | null)?.method;
        return { ...current, method: m === "aucell" || m === "score_genes" ? m : current.method };
      },
    },
  ),
);
