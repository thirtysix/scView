import { create } from "zustand";
import type { PreviousValues } from "@/api/labels";

/** One hand labelling that can be undone (this session only). */
export interface LabelUndoEntry {
  datasetId: string;
  column: string;
  label: string;
  nCells: number;
  created: boolean;
  previous: PreviousValues[];
  /** Cell Ontology terms the labelling replaced (label -> id or null) */
  previousTerms?: Record<string, string | null>;
  /** The dataset's cell count when labelled: a later filter renumbers cells,
   *  and the saved indices would then point at other cells. */
  datasetCells: number;
}

/** Undo depth per dataset; each entry holds the indices of the cells it changed. */
export const MAX_UNDO = 20;

interface LabelUndoState {
  stacks: Record<string, LabelUndoEntry[]>;
  /** The column each dataset was last labelled into, offered first next time */
  lastColumn: Record<string, string>;
  push: (entry: LabelUndoEntry) => void;
  /** Remove and return the dataset's latest entry */
  pop: (datasetId: string) => LabelUndoEntry | undefined;
  clear: (datasetId: string) => void;
}

export const useLabelUndoStore = create<LabelUndoState>((set, get) => ({
  stacks: {},
  lastColumn: {},
  push: (entry) =>
    set((s) => ({
      stacks: {
        ...s.stacks,
        [entry.datasetId]: [...(s.stacks[entry.datasetId] ?? []), entry].slice(-MAX_UNDO),
      },
      lastColumn: { ...s.lastColumn, [entry.datasetId]: entry.column },
    })),
  pop: (datasetId) => {
    const stack = get().stacks[datasetId] ?? [];
    const top = stack[stack.length - 1];
    if (top) set((s) => ({ stacks: { ...s.stacks, [datasetId]: stack.slice(0, -1) } }));
    return top;
  },
  clear: (datasetId) => set((s) => ({ stacks: { ...s.stacks, [datasetId]: [] } })),
}));

/** The entry an undo would revert for `datasetId`, if any. */
export function latestUndo(
  stacks: Record<string, LabelUndoEntry[]>,
  datasetId: string | null,
): LabelUndoEntry | undefined {
  const stack = datasetId ? stacks[datasetId] : undefined;
  return stack?.[stack.length - 1];
}
