import { create } from "zustand";
import { MAX_GRID_GENES, type BlendScheme } from "@/lib/geneCoexpression";

// The genes picked in the Expression subtab's "Several genes" section, and how
// the gene views show them. A store, so the picks survive the remount
// PanelContainer does on every tab switch; the picks belong to one dataset.
interface GeneGridState {
  /** Dataset the picks were made on (other datasets start empty) */
  datasetId: string | null;
  genes: string[];
  setGenes: (datasetId: string, genes: string[]) => void;
  /** The pair for the blend and the gene-vs-gene view (two of `genes`) */
  pair: [string, string] | null;
  setPair: (pair: [string, string] | null) => void;
  /** Feature grid: one colour scale for every gene (else each gene its own) */
  sharedScale: boolean;
  setSharedScale: (shared: boolean) => void;
  /** Blend: quantile of the expressing cells each gene's scale tops out at */
  blendClip: number;
  setBlendClip: (clip: number) => void;
  /** Blend colours: Seurat red/green, or magenta/green for red-green colour blindness */
  blendScheme: BlendScheme;
  setBlendScheme: (scheme: BlendScheme) => void;
  /** The stacked violin / gene-vs-gene views are open below the picks */
  showViolins: boolean;
  setShowViolins: (show: boolean) => void;
  showScatter: boolean;
  setShowScatter: (show: boolean) => void;
}

export const useGeneGridStore = create<GeneGridState>((set) => ({
  datasetId: null,
  genes: [],
  setGenes: (datasetId, genes) =>
    set((s) => {
      const kept = genes.slice(0, MAX_GRID_GENES);
      // A pair that lost a gene falls back to the first two picks.
      const pairOk = s.pair && s.datasetId === datasetId && s.pair.every((g) => kept.includes(g));
      return {
        datasetId,
        genes: kept,
        pair: pairOk ? s.pair : kept.length >= 2 ? [kept[0]!, kept[1]!] : null,
      };
    }),
  pair: null,
  setPair: (pair) => set({ pair }),
  sharedScale: false,
  setSharedScale: (sharedScale) => set({ sharedScale }),
  blendClip: 0.99,
  setBlendClip: (blendClip) => set({ blendClip }),
  blendScheme: "seurat",
  setBlendScheme: (blendScheme) => set({ blendScheme }),
  showViolins: false,
  setShowViolins: (showViolins) => set({ showViolins }),
  showScatter: false,
  setShowScatter: (showScatter) => set({ showScatter }),
}));
