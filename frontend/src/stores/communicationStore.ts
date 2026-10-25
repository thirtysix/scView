import { create } from "zustand";
import type { CommunicationResult, CompareResult } from "@/api/communication";

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
  /** One run, or two conditions compared */
  mode: "single" | "compare";
  /** Compare mode: the condition column and its two values (A = reference) */
  compareColumn: string;
  compareA: string;
  compareB: string;
}

export const COMMUNICATION_DEFAULTS: CommunicationSettings = {
  groupby: "",
  subsetColumn: "",
  subsetValue: "",
  exprProp: 0.1,
  mode: "single",
  compareColumn: "",
  compareA: "",
  compareB: "",
};

interface CommunicationState extends CommunicationSettings {
  /** The last result, with the dataset it was run on */
  result: { datasetId: string; result: CommunicationResult } | null;
  /** The last comparison, with its dataset */
  compareResult: { datasetId: string; result: CompareResult } | null;
  update: (patch: Partial<CommunicationSettings>) => void;
  setResult: (datasetId: string, result: CommunicationResult | null) => void;
  setCompareResult: (datasetId: string, result: CompareResult | null) => void;
  reset: () => void;
}

export const useCommunicationStore = create<CommunicationState>((set) => ({
  ...COMMUNICATION_DEFAULTS,
  result: null,
  compareResult: null,
  update: (patch) => set(patch),
  setResult: (datasetId, result) => set({ result: result ? { datasetId, result } : null }),
  setCompareResult: (datasetId, result) => set({ compareResult: result ? { datasetId, result } : null }),
  reset: () => set({ ...COMMUNICATION_DEFAULTS, result: null, compareResult: null }),
}));
