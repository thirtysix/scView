import { useEffect, useMemo, useRef, useState } from "react";
import { DialogBox } from "@/components/common/DialogBox";
import { Columns2 } from "lucide-react";
import { prettyObsLabel } from "@/lib/formatting";
import {
  COMPARE_MAX_COLUMNS,
  COMPARE_MIN_COLUMNS,
  compareColumnOptions,
  initialCompareSelection,
} from "@/lib/splitView";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSplitViewStore } from "@/stores/splitViewStore";

/**
 * Toolbar "Compare" button: pick 2–4 grouping columns (cell types, clusterings,
 * hand labels...) to see the same embedding coloured by each, side by side.
 * Sits next to "Split"; one grid at a time.
 */
export function CompareColoringsControl({ disabledReason = null }: { disabledReason?: string | null } = {}) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const colorBy = useSettingsStore((s) => s.colorBy);
  const compareColumns = useSplitViewStore((s) => s.compareColumns);
  const setCompareColumns = useSplitViewStore((s) => s.setCompareColumns);
  const [open, setOpen] = useState(false);
  const [picked, setPicked] = useState<string[]>([]);
  const ref = useRef<HTMLDivElement>(null);

  const options = useMemo(
    () =>
      compareColumnOptions(
        dataset?.obs_columns ?? [],
        dataset?.n_cells,
        dataset?.active_clustering,
        dataset?.user_annotation_columns,
      ),
    [dataset],
  );

  // A comparison carried over from another dataset (or a removed column) keeps
  // only the columns this dataset has, and ends when fewer than two are left.
  useEffect(() => {
    if (!compareColumns) return;
    const kept = compareColumns.filter((c) => options.some((o) => o.name === c));
    if (kept.length < COMPARE_MIN_COLUMNS) setCompareColumns(null);
    else if (kept.length !== compareColumns.length) setCompareColumns(kept);
  }, [compareColumns, options, setCompareColumns]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  const toggleOpen = () => {
    if (!open) {
      setPicked(compareColumns ?? initialCompareSelection(options.map((o) => o.name), colorBy));
    }
    setOpen((o) => !o);
  };
  const toggle = (name: string) =>
    setPicked((p) => (p.includes(name) ? p.filter((c) => c !== name) : [...p, name]));
  const full = picked.length >= COMPARE_MAX_COLUMNS;
  const active = compareColumns != null;

  return (
    <div ref={ref} className="relative flex items-center">
      <button
        onClick={toggleOpen}
        disabled={options.length < COMPARE_MIN_COLUMNS || !!disabledReason}
        aria-expanded={open}
        aria-haspopup="dialog"
        title={
          disabledReason
            ? disabledReason
            : options.length < COMPARE_MIN_COLUMNS
            ? "Needs two or more grouping columns"
            : `Show the plot coloured by ${COMPARE_MIN_COLUMNS}–${COMPARE_MAX_COLUMNS} columns side by side`
        }
        className={`flex items-center gap-1 rounded border px-2 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
          active
            ? "border-blue-300 bg-blue-50 text-blue-700"
            : "border-slate-300 bg-white text-slate-600 hover:bg-slate-100"
        }`}
      >
        <Columns2 className="h-3.5 w-3.5" />
        Compare
      </button>
      {open && (
        <DialogBox
          label="Compare colourings"
          onClose={() => setOpen(false)}
          className="absolute left-0 top-full z-40 mt-1 w-64 rounded-lg border border-slate-200 bg-white p-3 text-xs shadow-lg"
        >
          <p className="mb-2 font-semibold text-slate-700">Compare colourings side by side</p>
          <p className="mb-2 text-[11px] text-slate-500">
            Pick {COMPARE_MIN_COLUMNS}–{COMPARE_MAX_COLUMNS} columns: one panel each, same camera.
          </p>
          <div className="max-h-56 space-y-0.5 overflow-y-auto">
            {options.map((o) => {
              const checked = picked.includes(o.name);
              return (
                <label
                  key={o.name}
                  className={`flex cursor-pointer items-center gap-2 rounded px-1 py-0.5 hover:bg-slate-50 ${
                    !checked && full ? "cursor-not-allowed opacity-50" : ""
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={!checked && full}
                    onChange={() => toggle(o.name)}
                    className="h-3 w-3 accent-blue-500"
                  />
                  <span className="truncate text-slate-700">{prettyObsLabel(o.name)}</span>
                  <span className="ml-auto flex-shrink-0 text-slate-400">{o.nValues}</span>
                </label>
              );
            })}
          </div>
          <div className="mt-3 flex items-center justify-end gap-2">
            {active && (
              <button
                onClick={() => {
                  setCompareColumns(null);
                  setOpen(false);
                }}
                className="rounded px-2 py-1 text-slate-600 hover:bg-slate-100"
              >
                Off
              </button>
            )}
            <button
              onClick={() => {
                setCompareColumns(picked);
                setOpen(false);
              }}
              disabled={picked.length < COMPARE_MIN_COLUMNS}
              className="rounded bg-blue-500 px-2.5 py-1 font-medium text-white hover:bg-blue-600 disabled:opacity-50"
            >
              Show {picked.length > 0 ? picked.length : ""} side by side
            </button>
          </div>
        </DialogBox>
      )}
    </div>
  );
}
