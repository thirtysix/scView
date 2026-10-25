import { useCallback, useMemo } from "react";
import { selectionMask } from "@/lib/linkedSelection";
import { useSelectionStore } from "@/stores/selectionStore";

export interface LinkedSelection {
  /** The selected cells, or null when nothing is selected (an empty Set counts as none). */
  selected: ReadonlySet<number> | null;
  size: number;
  /** 1 per selected cell of the `n` asked for; null without a selection or `n`. */
  mask: Uint8Array | null;
  /** Make `cells` the selection (a set of cells, not a group, as a lasso); null or empty clears it. */
  select: (cells: Set<number> | null) => void;
  clear: () => void;
}

/**
 * C035: the one hook a panel uses to show the current cell selection (and to
 * set it from its own box / lasso). `n` is how many cells the panel's per-cell
 * arrays hold: the mask is built once per selection and shared by every panel
 * that asks for the same `n` (lib/linkedSelection.ts).
 */
export function useLinkedSelection(n?: number | null): LinkedSelection {
  const raw = useSelectionStore((s) => s.selectedCellIndices);
  const setSelection = useSelectionStore((s) => s.setSelection);
  const clearSelection = useSelectionStore((s) => s.clearSelection);
  const setHighlight = useSelectionStore((s) => s.setHighlight);
  const selected = raw && raw.size > 0 ? raw : null;
  const mask = useMemo(() => (selected && n ? selectionMask(selected, n) : null), [selected, n]);
  const select = useCallback(
    (cells: Set<number> | null) => {
      setHighlight(null);
      if (cells && cells.size > 0) setSelection(cells);
      else clearSelection();
    },
    [setSelection, clearSelection, setHighlight],
  );
  const clear = useCallback(() => {
    setHighlight(null);
    clearSelection();
  }, [clearSelection, setHighlight]);
  return { selected, size: selected?.size ?? 0, mask, select, clear };
}
