import type { SelectionContext } from "@/api/assistant";
import { encodeSelection } from "@/lib/session";
import { useCopilotSelectionStore } from "@/stores/copilotSelectionStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { useGatesStore } from "@/stores/gatesStore";
import { useSelectionStore } from "@/stores/selectionStore";

/** How the current selection was made: a shown gate, a cell filter, the lasso or a
 *  legend/group click. */
export function selectionSource(): string {
  const sel = useSelectionStore.getState();
  const cells = sel.selectedCellIndices;
  const datasetId = useDatasetStore.getState().currentDatasetId;
  const { shown, records } = useGatesStore.getState();
  if (cells && shown && shown.datasetId === datasetId && shown.cells === cells) {
    const gate = records[shown.datasetId]?.gates.find((g) => g.id === shown.gateId);
    if (gate) return `gate:${gate.name}`;
  }
  if (sel.selectionFilter) return "filter";
  if (sel.selectionMode === "lasso") return "lasso";
  return sel.highlightedGroup ? "group" : "lasso";
}

/** The selected cells for the co-pilot's view context, or undefined when nothing is
 *  selected or the user took the selection out of the chat. The cells go as session
 *  files keep them (indices or a bitmask, whichever is shorter). */
export function selectionContext(): SelectionContext | undefined {
  const cells = useSelectionStore.getState().selectedCellIndices;
  const nTotal = useDatasetStore.getState().currentDataset?.n_cells ?? 0;
  if (!cells || cells.size === 0 || nTotal <= 0) return undefined;
  const { excluded, lastSent } = useCopilotSelectionStore.getState();
  if (excluded === cells) return undefined;
  const encoded = encodeSelection(cells, nTotal);
  if ("filter" in encoded) return undefined;
  return { n_cells: cells.size, source: selectionSource(), cells: encoded, fresh: lastSent !== cells };
}

/** Short words for the chip: "lasso", "gate T cells", "filter". */
export function describeSource(source: string): string {
  if (source.startsWith("gate:")) return `gate ${source.slice(5)}`;
  if (source === "group") return "group click";
  return source;
}
