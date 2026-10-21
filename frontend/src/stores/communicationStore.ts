import { create } from "zustand";
import type { CommunicationResult } from "@/api/communication";

/** What the Communication tab runs: kept here so a tab switch keeps the pickers
 *  and the last result, and a saved session can restore the settings. */
export interface CommunicationSettings {
  /** Sender / receiver groups ("" = the Unified View's group-by column) */
  groupby: string;
  /** One condition's cells only ("" = all cells) */
  subsetColumn: string;
  subsetValue: string;
  /** Share of a group's cells each gene (subunit) must be detected in */
  exprProp: number;
}

export const COMMUNICATION_DEFAULTS: CommunicationSettings = {
  groupby: "",
  subsetColumn: "",
  subsetValue: "",
  exprProp: 0.1,
};

interface CommunicationState extends CommunicationSettings {
  /** The last result, with the dataset it was run on */
  result: { datasetId: string; result: CommunicationResult } | null;
  update: (patch: Partial<CommunicationSettings>) => void;
  setResult: (datasetId: string, result: CommunicationResult | null) => void;
  reset: () => void;
}

export const useCommunicationStore = create<CommunicationState>((set) => ({
  ...COMMUNICATION_DEFAULTS,
  result: null,
  update: (patch) => set(patch),
  setResult: (datasetId, result) => set({ result: result ? { datasetId, result } : null }),
  reset: () => set({ ...COMMUNICATION_DEFAULTS, result: null }),
}));
