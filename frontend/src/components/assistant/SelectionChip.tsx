import { Lasso, Plus, X } from "lucide-react";
import { describeSource, selectionSource } from "@/lib/copilotSelection";
import { useCopilotSelectionStore } from "@/stores/copilotSelectionStore";
import { useSelectionStore } from "@/stores/selectionStore";

export const SELECTION_QUESTIONS = [
  "What are these cells?",
  "How do they differ from the rest?",
  "Run DE for this selection vs the rest",
];

/**
 * Above the composer while cells are selected: "Selection: 1,234 cells", sent with
 * the next question (the co-pilot then gets the selection's facts). Removing it
 * leaves this selection out of the chat until the cells change. Suggested questions
 * ask about it directly.
 */
export function SelectionChip({ onAsk, busy }: { onAsk: (q: string) => void; busy: boolean }) {
  const cells = useSelectionStore((s) => s.selectedCellIndices);
  // Re-render when the source changes (a gate shown, a filter, the lasso).
  useSelectionStore((s) => s.selectionFilter);
  const excluded = useCopilotSelectionStore((s) => s.excluded);
  const exclude = useCopilotSelectionStore((s) => s.exclude);
  const include = useCopilotSelectionStore((s) => s.include);
  if (!cells || cells.size === 0) return null;
  const n = cells.size.toLocaleString("en-US");
  if (excluded === cells) {
    return (
      <div className="flex items-center gap-2 px-3 pt-2 text-xs text-muted-foreground">
        <span>Selection left out of the chat.</span>
        <button
          type="button"
          onClick={include}
          aria-label={`Add the selection (${n} cells) to the co-pilot's context`}
          className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 hover:bg-accent hover:text-foreground"
        >
          <Plus className="h-3 w-3" />
          Add it back
        </button>
      </div>
    );
  }
  return (
    <div className="space-y-1.5 px-3 pt-2">
      <div className="flex items-center gap-1.5 text-xs">
        <span
          className="inline-flex items-center gap-1 rounded-full bg-orange-50 py-0.5 pl-2 pr-0.5 text-orange-700 ring-1 ring-orange-200"
          title={`The co-pilot gets these cells' facts with your next question (${describeSource(selectionSource())})`}
        >
          <Lasso className="h-3 w-3" aria-hidden="true" />
          Selection: {n} cells
          <button
            type="button"
            onClick={() => exclude(cells)}
            aria-label="Leave the selection out of the co-pilot's context"
            className="rounded-full p-0.5 hover:bg-orange-100"
          >
            <X className="h-3 w-3" />
          </button>
        </span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {SELECTION_QUESTIONS.map((q) => (
          <button
            key={q}
            type="button"
            disabled={busy}
            onClick={() => onAsk(q)}
            className="rounded-full border px-2.5 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-accent-foreground disabled:opacity-40"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
