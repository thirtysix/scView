import { create } from "zustand";
import { DEFAULT_MIN_PCT } from "@/lib/deDetection";

export type DEMode = "selection" | "groups";
export type DETest = "wilcoxon" | "pseudobulk";

/** What the DE subtab compares and how: kept here, not in the subtab, so the
 *  pickers survive a tab switch and a saved session can restore them. */
export interface DESettings {
  mode: DEMode;
  /** Between groups: the column compared, its two groups, an optional scope */
  splitBy: string;
  groupA: string;
  groupB: string;
  withinCol: string;
  withinVal: string;
  test: DETest;
  /** Donor / sample column for pseudobulk */
  replicateKey: string;
  /** |log2FC| a gene needs to be listed */
  fcThreshold: number;
  /** Cell-level test: fraction of either group's cells a gene must be detected in */
  minPct: number;
}

export const DE_DEFAULTS: DESettings = {
  mode: "selection",
  splitBy: "",
  groupA: "",
  groupB: "",
  withinCol: "",
  withinVal: "",
  test: "wilcoxon",
  replicateKey: "",
  fcThreshold: 1,
  minPct: DEFAULT_MIN_PCT,
};

interface DESettingsState extends DESettings {
  /** The highlighted group last turned into the "within" scope, so a remount
   *  of the subtab does not undo a scope picked by hand since. */
  scopedHighlight: { column: string; value: string } | null;
  update: (patch: Partial<DESettings & { scopedHighlight: DESettingsState["scopedHighlight"] }>) => void;
  reset: () => void;
}

export const useDESettingsStore = create<DESettingsState>((set) => ({
  ...DE_DEFAULTS,
  scopedHighlight: null,
  update: (patch) => set(patch),
  reset: () => set({ ...DE_DEFAULTS, scopedHighlight: null }),
}));
