import { create } from "zustand";

/**
 * A counter per (dataset, obs column), bumped when the column's values change in
 * place (hand labels). A refreshed dataset shows a changed set of categories,
 * but not cells moving between existing ones, so views that fetched a column's
 * values (the cluster map, marker tables) also refetch on its revision.
 */
interface ObsRevisionState {
  revisions: Record<string, number>;
  bump: (datasetId: string, column: string) => void;
}

const key = (datasetId: string, column: string) => `${datasetId}\u0000${column}`;

export const useObsRevisionStore = create<ObsRevisionState>((set) => ({
  revisions: {},
  bump: (datasetId, column) =>
    set((s) => ({ revisions: { ...s.revisions, [key(datasetId, column)]: (s.revisions[key(datasetId, column)] ?? 0) + 1 } })),
}));

/** The revision of one column (0 until it changes). */
export function useObsRevision(datasetId: string | null | undefined, column: string | null | undefined): number {
  return useObsRevisionStore((s) => (datasetId && column ? (s.revisions[key(datasetId, column)] ?? 0) : 0));
}
