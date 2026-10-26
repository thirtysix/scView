import { Sparkles } from "lucide-react";
import { useCopilotSelectionStore } from "@/stores/copilotSelectionStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useViewStore } from "@/stores/viewStore";

/** In the toolbar's selection chip: opens the co-pilot with this selection in its
 *  context (the chat shows it as a chip, with suggested questions). Nothing is asked
 *  until the user picks or types a question. */
export function AskSelectionButton() {
  const cells = useSelectionStore((s) => s.selectedCellIndices);
  if (!cells || cells.size === 0) return null;
  const open = () => {
    useCopilotSelectionStore.getState().include();
    const view = useViewStore.getState();
    if (view.activePanel !== "assistant") view.setCopilotOpen(true);
  };
  return (
    <button
      type="button"
      onClick={open}
      aria-label="Ask the co-pilot about this selection"
      title="Ask the co-pilot about this selection"
      className="flex items-center gap-0.5 rounded px-1 py-0.5 text-orange-700 hover:bg-orange-100 dark:text-orange-300 dark:hover:bg-orange-900"
    >
      <Sparkles className="h-3 w-3" aria-hidden="true" />
      Ask
    </button>
  );
}
