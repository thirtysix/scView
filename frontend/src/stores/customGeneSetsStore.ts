import { create } from "zustand";

/**
 * Bumped whenever a custom gene-set collection is uploaded, renamed or deleted,
 * so every list of collections (the enrichment tree, the gene-set browsers)
 * fetches again instead of showing a stale or deleted one.
 */
interface CustomGeneSetsState {
  version: number;
  bump: () => void;
}

export const useCustomGeneSetsStore = create<CustomGeneSetsState>((set) => ({
  version: 0,
  bump: () => set((s) => ({ version: s.version + 1 })),
}));
