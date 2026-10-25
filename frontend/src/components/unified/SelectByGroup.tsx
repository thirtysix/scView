import { useEffect, useRef, useState } from "react";
import { ListChecks } from "lucide-react";
import { DialogBox } from "@/components/common/DialogBox";
import { prettyObsLabel } from "@/lib/formatting";

export interface GroupSelectSource {
  /** The colour-by column the groups come from */
  column: string;
  categories: readonly string[];
  /** Cells per group, in `categories` order */
  counts?: readonly number[];
  /** Make the cells of these groups the selection (as a legend pick does) */
  onSelect: (groups: string[]) => void;
}

/**
 * "By group": the keyboard path to a lasso-like selection. Pick groups of the
 * current colour-by column as checkboxes; Select makes their cells the
 * selection, through the same store a lasso or legend click uses, so DE
 * (selection vs rest) and the linked plots follow.
 */
export function SelectByGroupControl({ source }: { source: GroupSelectSource | null }) {
  const [open, setOpen] = useState(false);
  const [picked, setPicked] = useState<string[]>([]);
  const ref = useRef<HTMLDivElement>(null);

  // A click outside closes it, as the other toolbar popovers do.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);
  // Another colouring: the picks meant the old column.
  const column = source?.column;
  useEffect(() => {
    setPicked([]);
    setOpen(false);
  }, [column]);

  const disabled = !source || source.categories.length === 0;
  const toggle = (g: string) => setPicked((p) => (p.includes(g) ? p.filter((x) => x !== g) : [...p, g]));
  const total = source?.counts
    ? picked.reduce((n, g) => n + (source.counts![source.categories.indexOf(g)] ?? 0), 0)
    : null;

  return (
    <div ref={ref} className="relative flex items-center">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        disabled={disabled}
        aria-expanded={open}
        aria-haspopup="dialog"
        title={
          disabled
            ? "Colour the plot by a grouping column to select by group"
            : `Select the cells of chosen ${prettyObsLabel(source.column)} groups (no mouse needed)`
        }
        className="flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40"
      >
        <ListChecks className="h-3.5 w-3.5" />
        By group
      </button>
      {open && source && (
        <DialogBox
          label="Select cells by group"
          onClose={() => setOpen(false)}
          className="absolute left-0 top-full z-50 mt-1 w-64 rounded-lg border border-slate-200 bg-white p-3 text-xs shadow-lg"
        >
          <fieldset>
            <legend className="mb-2 font-semibold text-slate-700">
              Select cells by {prettyObsLabel(source.column)}
            </legend>
            <div className="max-h-56 space-y-0.5 overflow-y-auto">
              {source.categories.map((g, i) => (
                <label key={g} className="flex cursor-pointer items-center gap-2 rounded px-1 py-0.5 hover:bg-slate-50">
                  <input
                    type="checkbox"
                    checked={picked.includes(g)}
                    onChange={() => toggle(g)}
                    className="h-3 w-3 accent-blue-500"
                  />
                  <span className="truncate text-slate-700">{g}</span>
                  {source.counts && (
                    <span className="ml-auto flex-shrink-0 tabular-nums text-slate-500">
                      {(source.counts[i] ?? 0).toLocaleString("en-US")}
                    </span>
                  )}
                </label>
              ))}
            </div>
          </fieldset>
          <div className="mt-3 flex items-center justify-end gap-2">
            {total !== null && picked.length > 0 && (
              <span className="mr-auto text-slate-500">{total.toLocaleString("en-US")} cells</span>
            )}
            <button type="button" onClick={() => setOpen(false)} className="rounded px-2 py-1 text-slate-600 hover:bg-slate-100">
              Cancel
            </button>
            <button
              type="button"
              onClick={() => {
                source.onSelect(picked);
                setOpen(false);
              }}
              disabled={picked.length === 0}
              className="rounded bg-blue-500 px-2.5 py-1 font-medium text-white hover:bg-blue-600 disabled:opacity-50"
            >
              Select
            </button>
          </div>
        </DialogBox>
      )}
    </div>
  );
}
