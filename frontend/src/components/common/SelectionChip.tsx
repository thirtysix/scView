import { useLinkedSelection } from "@/hooks/useLinkedSelection";
import { SELECTION_COLOR } from "@/lib/linkedSelection";

/**
 * "Selection: n cells [Clear]" on every panel that shows the selection, so the
 * link between panels is visible. Nothing while no cells are selected.
 */
export function SelectionChip({ note, className = "" }: { note?: string; className?: string }) {
  const { size, clear } = useLinkedSelection();
  if (size === 0) return null;
  return (
    <span
      data-testid="selection-chip"
      className={`inline-flex flex-shrink-0 items-center gap-1 rounded-full border border-orange-300 bg-orange-50 px-2 py-0.5 text-[11px] text-orange-800 ${className}`}
      title={`The ${size.toLocaleString()} selected cells (lasso, legend, gate or filter) are shown here${note ? `: ${note}` : ""}. Clear (or Esc on the plot) drops the selection.`}
    >
      <span className="h-2 w-2 rounded-full" style={{ background: SELECTION_COLOR }} aria-hidden />
      Selection: {size.toLocaleString()} cells
      <button
        type="button"
        onClick={clear}
        className="ml-0.5 rounded px-1 font-medium text-orange-700 hover:bg-orange-100"
        aria-label="Clear selection"
      >
        Clear
      </button>
    </span>
  );
}
