import { useCallback, useEffect, useRef, useState } from "react";
import { useDialogFocus } from "@/hooks/useDialogFocus";
import { Loader2, Tag, Undo2 } from "lucide-react";
import { ApiDetailError, labelCells, listLabelColumns, undoLabel } from "@/api/labels";
import type { DatasetInfo } from "@/api/types";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { columnNameError, labelError, suggestColumnName, UNASSIGNED } from "@/lib/labels";
import { defaultColorBy } from "@/lib/obsColumns";
import { formatNumber, prettyObsLabel } from "@/lib/formatting";
import { useDatasetStore } from "@/stores/datasetStore";
import { latestUndo, useLabelUndoStore } from "@/stores/labelUndoStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

const NEW = "\u0000new";

/**
 * Toolbar controls for hand labels: "Label selection…" while cells are selected
 * (a lasso or a legend group), and "Undo label" for the latest labelling of this
 * dataset in this session.
 */
export function LabelSelectionControls() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = dataset?.id ?? null;
  const selected = useSelectionStore((s) => s.selectedCellIndices);
  const undoEntry = useLabelUndoStore((s) => latestUndo(s.stacks, datasetId));
  const refresh = useDatasetRefresh();
  const [open, setOpen] = useState(false);
  const [undoing, setUndoing] = useState(false);
  const [undoError, setUndoError] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  const hasSelection = !!selected && selected.size > 0;
  useEffect(() => {
    if (!hasSelection) setOpen(false);
  }, [hasSelection]);
  useEffect(() => {
    setUndoError(null);
  }, [datasetId]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  const handleUndo = useCallback(async () => {
    const current = useDatasetStore.getState().currentDataset;
    const entry = latestUndo(useLabelUndoStore.getState().stacks, current?.id ?? null);
    if (!current || !entry) return;
    setUndoError(null);
    if (current.n_cells != null && entry.datasetCells > 0 && current.n_cells !== entry.datasetCells) {
      useLabelUndoStore.getState().clear(current.id);
      setUndoError("The dataset's cells changed since labelling, so it can't be undone.");
      return;
    }
    setUndoing(true);
    try {
      const res = await undoLabel(current.id, {
        column: entry.column,
        previous: entry.previous,
        created: entry.created,
      });
      useLabelUndoStore.getState().pop(current.id);
      const fresh = await refresh(current.id, entry.column);
      if (res.removed) {
        // Nothing may point at the removed column.
        const settings = useSettingsStore.getState();
        if (settings.colorBy === entry.column) settings.setDefaultColorBy(defaultColorBy(fresh));
        const unified = useUnifiedViewStore.getState();
        if (unified.groupByColumn === entry.column) unified.setGroupByColumn("");
        const selection = useSelectionStore.getState();
        if (selection.highlightedGroup?.column === entry.column) selection.setHighlight(null);
      }
    } catch (e) {
      // A refused undo (column gone, cells changed) will never succeed: drop it.
      if (e instanceof ApiDetailError && e.status !== null && e.status < 500) {
        useLabelUndoStore.getState().pop(current.id);
      }
      setUndoError(e instanceof Error ? e.message : "Undo failed.");
    } finally {
      setUndoing(false);
    }
  }, [refresh]);

  if (!dataset || (!hasSelection && !undoEntry && !undoError)) return null;

  return (
    <div ref={ref} className="relative flex items-center gap-1">
      {hasSelection && (
        <button
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          aria-haspopup="dialog"
          title="Give the selected cells a label in an annotation column"
          className={`flex items-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors ${
            open ? "bg-orange-50 text-orange-700 ring-1 ring-orange-300" : "text-slate-600 hover:bg-slate-100"
          }`}
        >
          <Tag className="h-3.5 w-3.5" />
          Label selection…
        </button>
      )}
      {undoEntry && (
        <button
          onClick={() => void handleUndo()}
          disabled={undoing}
          title={`Undo: "${undoEntry.label}" on ${undoEntry.nCells.toLocaleString()} cells in ${prettyObsLabel(undoEntry.column)}`}
          className="flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-100 disabled:cursor-wait disabled:opacity-60"
        >
          {undoing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Undo2 className="h-3.5 w-3.5" />}
          Undo label
        </button>
      )}
      {undoError && (
        <span role="alert" className="max-w-[220px] truncate text-[11px] text-red-600" title={undoError}>
          {undoError}
        </span>
      )}
      {open && hasSelection && selected && (
        <LabelSelectionForm dataset={dataset} indices={selected} onClose={() => setOpen(false)} />
      )}
    </div>
  );
}

interface LabelSelectionFormProps {
  dataset: DatasetInfo;
  indices: Set<number>;
  onClose: () => void;
}

function LabelSelectionForm({ dataset, indices, onClose }: LabelSelectionFormProps) {
  const dialogRef = useRef<HTMLFormElement>(null);
  useDialogFocus(dialogRef, onClose);
  const refresh = useDatasetRefresh();
  const lastColumn = useLabelUndoStore((s) => s.lastColumn[dataset.id]);
  const existingNames = dataset.obs_columns.map((c) => c.name);
  const [columns, setColumns] = useState<string[] | null>(null);
  const [target, setTarget] = useState<string>(NEW);
  const [newName, setNewName] = useState(() => suggestColumnName(existingNames));
  const [label, setLabel] = useState("");
  const [touched, setTouched] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listLabelColumns(dataset.id)
      .then((r) => {
        if (cancelled) return;
        setColumns(r.columns);
        const pick = lastColumn && r.columns.includes(lastColumn) ? lastColumn : r.columns[0];
        if (pick) setTarget(pick);
      })
      .catch(() => {
        if (!cancelled) setColumns([]);
      });
    return () => {
      cancelled = true;
    };
  }, [dataset.id, lastColumn]);

  const isNew = target === NEW;
  const labelProblem = labelError(label);
  const nameProblem = isNew ? columnNameError(newName, existingNames) : null;
  const knownLabels = isNew
    ? []
    : (dataset.obs_columns.find((c) => c.name === target)?.values ?? []).filter((v) => v !== UNASSIGNED);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setTouched(true);
    if (labelProblem || nameProblem || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await labelCells(dataset.id, {
        column: isNew ? newName.trim() : target,
        label: label.trim(),
        indices: Array.from(indices),
        newColumn: isNew,
      });
      useLabelUndoStore.getState().push({
        datasetId: dataset.id,
        column: res.column,
        label: res.label,
        nCells: res.n_cells,
        created: res.created,
        previous: res.previous,
        datasetCells: dataset.n_cells ?? 0,
      });
      await refresh(dataset.id, res.column);
      useSettingsStore.getState().setColorBy(res.column); // show the label straight away
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Labelling failed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <form
      ref={dialogRef}
      role="dialog"
      aria-label="Label selection"
      onSubmit={(e) => void submit(e)}
      className="absolute left-0 top-full z-50 mt-1 w-72 space-y-2.5 rounded-lg border border-slate-200 bg-white p-3 text-xs shadow-lg"
    >
      <div className="font-semibold text-slate-700">
        Label {formatNumber(indices.size)} selected cell{indices.size === 1 ? "" : "s"}
      </div>

      <label className="block space-y-1">
        <span className="text-[11px] font-medium text-slate-500">Label</span>
        <input
          autoFocus
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          list={knownLabels.length ? "label-selection-known" : undefined}
          placeholder="e.g. activated T cells"
          className="w-full rounded border border-slate-300 px-2 py-1 text-xs text-slate-700 focus:border-blue-400 focus:outline-none"
        />
        {knownLabels.length > 0 && (
          <datalist id="label-selection-known">
            {knownLabels.map((v) => (
              <option key={v} value={v} />
            ))}
          </datalist>
        )}
      </label>

      <label className="block space-y-1">
        <span className="text-[11px] font-medium text-slate-500">Column</span>
        <select
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          disabled={columns === null}
          className="w-full rounded border border-slate-300 bg-white px-2 py-1 text-xs text-slate-700 focus:border-blue-400 focus:outline-none"
        >
          {(columns ?? []).map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
          <option value={NEW}>New column…</option>
        </select>
      </label>

      {isNew && (
        <label className="block space-y-1">
          <span className="text-[11px] font-medium text-slate-500">New column name</span>
          <input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            className="w-full rounded border border-slate-300 px-2 py-1 text-xs text-slate-700 focus:border-blue-400 focus:outline-none"
          />
          <span className="block text-[10px] text-slate-400">
            Every other cell starts as &ldquo;{UNASSIGNED}&rdquo;.
          </span>
        </label>
      )}

      {touched && (labelProblem || nameProblem) && (
        <p role="alert" className="rounded bg-amber-50 px-2 py-1 text-[11px] text-amber-800">
          {labelProblem ?? nameProblem}
        </p>
      )}
      {error && (
        <p role="alert" className="rounded bg-red-50 px-2 py-1 text-[11px] text-red-700">
          {error}
        </p>
      )}

      <div className="flex justify-end gap-2 pt-1">
        <button
          type="button"
          onClick={onClose}
          className="rounded px-2 py-1 text-xs text-slate-500 hover:bg-slate-100"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={busy}
          className="flex items-center gap-1 rounded bg-blue-500 px-3 py-1 text-xs font-medium text-white hover:bg-blue-600 disabled:opacity-60"
        >
          {busy && <Loader2 className="h-3 w-3 animate-spin" />}
          Apply label
        </button>
      </div>
    </form>
  );
}
