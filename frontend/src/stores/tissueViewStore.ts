import { create } from "zustand";
import { DEFAULT_DOT_SIZE, DOT_SIZE_RANGE } from "@/lib/transcripts";

/** What a cell is drawn as on the tissue plot: the points, or its segmented shape (lib/cellOutlines.ts). */
export type TissueDraw = "points" | "cell" | "nucleus";

export interface TranscriptsShown {
  shown: number;
  total: number;
  perGene: number[];
}

/**
 * The tissue plot's own settings (Unified View, Embedding "Tissue"): the
 * section shown (one library at a time), and the H&E image under the spots.
 */
interface TissueViewState {
  /** Library id shown; null = the dataset's first */
  library: string | null;
  showImage: boolean;
  /** 0..1 */
  imageOpacity: number;
  draw: TissueDraw;
  /** Genes whose transcripts are drawn over the tissue (Transcripts tab) */
  transcriptGenes: string[];
  transcriptDotSize: number;
  /** What the view holds of them, once answered */
  transcriptShown: TranscriptsShown | null;
  /** The dataset last opened in Unified View: a spatial one opens on its tissue plot, once */
  openedOn: string | null;
  setOpenedOn: (datasetId: string | null) => void;
  setLibrary: (library: string | null) => void;
  setShowImage: (show: boolean) => void;
  setImageOpacity: (opacity: number) => void;
  setDraw: (draw: TissueDraw) => void;
  setTranscriptGenes: (genes: string[]) => void;
  setTranscriptDotSize: (size: number) => void;
  setTranscriptShown: (shown: TranscriptsShown | null) => void;
}

export const useTissueViewStore = create<TissueViewState>((set) => ({
  library: null,
  showImage: true,
  imageOpacity: 1,
  draw: "points",
  transcriptGenes: [],
  transcriptDotSize: DEFAULT_DOT_SIZE,
  transcriptShown: null,
  openedOn: null,
  setOpenedOn: (openedOn) => set({ openedOn }),
  setLibrary: (library) => set({ library }),
  setShowImage: (showImage) => set({ showImage }),
  setImageOpacity: (imageOpacity) => set({ imageOpacity: Math.min(1, Math.max(0, imageOpacity)) }),
  setDraw: (draw) => set({ draw }),
  setTranscriptGenes: (transcriptGenes) => set({ transcriptGenes }),
  setTranscriptDotSize: (size) =>
    set({ transcriptDotSize: Math.min(DOT_SIZE_RANGE[1], Math.max(DOT_SIZE_RANGE[0], size)) }),
  setTranscriptShown: (transcriptShown) => set({ transcriptShown }),
}));
