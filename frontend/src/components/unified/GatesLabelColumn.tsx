import { useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import { labelCells, type PreviousValues } from "@/api/labels";
import type { DatasetInfo } from "@/api/types";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { planLabelColumn, suggestGateColumn, type Gate, type GateEval, type LabelGroup } from "@/lib/gates";
import { columnNameError, UNASSIGNED } from "@/lib/labels";
import { useLabelUndoStore } from "@/stores/labelUndoStore";
import { useSettingsStore } from "@/stores/settingsStore";

/** The backend's cap on labels in one column (core/user_labels.py MAX_LABELS), Unassigned included. */
const MAX_LABELS = 100;

/** Previous values of several labellings of disjoint cells, one group per value. */
export function mergePrevious(lists: PreviousValues[][]): PreviousValues[] {
  const byValue = new Map<string | null, number[]>();
  for (const list of lists) {
    for (const g of list) byValue.set(g.value, [...(byValue.get(g.value) ?? []), ...g.indices]);
  }
  return [...byValue].map(([value, indices]) => ({ value, indices }));
}

/**
 * "Label column…": the leaf gates written into one new annotation column with
 * the existing `POST /labels`, one call per gate, each on cells no other call
 * touches (a cell in several leaf gates goes to the deepest, then the later
 * one). All the calls make one undo step for the toolbar's "Undo label".
 */
export function GatesLabelColumn({
  dataset,
  gates,
  evals,
  onDone,
  onCancel,
}: {
  dataset: DatasetInfo;
  gates: Gate[];
  evals: Map<string, GateEval>;
  onDone: (message: string) => void;
  onCancel: () => void;
}) {
  const refresh = useDatasetRefresh();
  const nCells = dataset.n_cells ?? 0;
  const existing = dataset.obs_columns.map((c) => c.name);
  const [column, setColumn] = useState(() => suggestGateColumn(existing));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const plan = useMemo(() => planLabelColumn(gates, evals, nCells), [gates, evals, nCells]);
  const nameProblem = columnNameError(column, existing);
  const tooMany = plan && plan.groups.length >= MAX_LABELS;

  const write = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!plan || nameProblem || tooMany || busy || plan.groups.length === 0) return;
    setBusy(true);
    setError(null);
    const name = column.trim();
    const previous: PreviousValues[][] = [];
    const written: LabelGroup[] = [];
    let failure: string | null = null;
    for (const group of plan.groups) {
      try {
        const res = await labelCells(dataset.id, {
          column: name,
          label: group.label,
          indices: group.indices,
          newColumn: written.length === 0,
        });
        previous.push(res.previous);
        written.push(group);
      } catch (err) {
        failure = `“${group.label}” failed: ${err instanceof Error ? err.message : String(err)}`;
        break;
      }
    }
    if (written.length > 0) {
      useLabelUndoStore.getState().push({
        datasetId: dataset.id,
        column: name,
        label: `${written.length} gate label${written.length === 1 ? "" : "s"}`,
        nCells: written.reduce((n, g) => n + g.indices.length, 0),
        created: true,
        previous: mergePrevious(previous),
        datasetCells: nCells,
      });
      try {
        await refresh(dataset.id, name);
        useSettingsStore.getState().setColorBy(name);
      } catch {
        /* the column is saved; the pickers catch up on the next refresh */
      }
    }
    setBusy(false);
    if (failure) {
      setError(
        written.length > 0
          ? `Wrote ${written.length} of ${plan.groups.length} labels; ${failure}. “Undo label” removes what was written.`
          : failure,
      );
      return;
    }
    onDone(`Wrote “${name}”: ${written.length} labels from the leaf gates; “Undo label” in the toolbar takes it back.`);
  };

  return (
    <form onSubmit={(e) => void write(e)} aria-label="Gates as a label column" className="space-y-1 rounded border border-slate-200 bg-slate-50 px-2 py-1.5">
      <div className="flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-1">
          <span className="text-[11px] text-slate-500">New column</span>
          <input
            autoFocus
            value={column}
            onChange={(e) => setColumn(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape") onCancel();
            }}
            className="w-36 rounded border border-slate-300 bg-white px-1.5 py-0.5 text-[11px] text-slate-700 focus:border-blue-400 focus:outline-none"
          />
        </label>
        <button
          type="submit"
          disabled={!plan || !!nameProblem || !!tooMany || busy || plan.groups.length === 0}
          className="flex items-center gap-1 rounded bg-blue-500 px-2 py-0.5 text-[11px] font-medium text-white hover:bg-blue-600 disabled:opacity-50"
        >
          {busy && <Loader2 className="h-3 w-3 animate-spin" />}
          Write labels
        </button>
        <button type="button" onClick={onCancel} className="rounded px-1.5 py-0.5 text-[11px] text-slate-500 hover:bg-slate-100">
          Cancel
        </button>
      </div>
      {!plan ? (
        <p className="text-[11px] text-slate-500">Every leaf gate (a gate with none inside it) has to be counted first.</p>
      ) : (
        <p className="text-[11px] text-slate-500" data-testid="gate-label-caption">
          One label per leaf gate: {plan.groups.map((g) => `${g.label} (${g.indices.length.toLocaleString()})`).join(", ")}
          ; {plan.unassigned.toLocaleString()} cells in no leaf gate are &ldquo;{UNASSIGNED}&rdquo;.{" "}
          {plan.overlap > 0
            ? `${plan.overlap.toLocaleString()} cells are in more than one leaf gate; each takes the deepest, then the later one in the list.`
            : "No cell is in more than one leaf gate."}
          {plan.inParentOnly > 0 &&
            ` ${plan.inParentOnly.toLocaleString()} of the Unassigned cells are in a gate that has gates inside it, but in none of those (a NOT gate would label them).`}
          {plan.empty.length > 0 && ` ${plan.empty.map((n) => `“${n}”`).join(", ")}: no cells of its own left, so no label.`}
        </p>
      )}
      {tooMany && (
        <p role="alert" className="text-[11px] text-amber-700">
          A column takes at most {MAX_LABELS - 1} labels besides &ldquo;{UNASSIGNED}&rdquo;.
        </p>
      )}
      {nameProblem && (
        <p role="alert" className="text-[11px] text-amber-700">
          {nameProblem}
        </p>
      )}
      {error && (
        <p role="alert" className="text-[11px] text-red-700">
          {error}
        </p>
      )}
    </form>
  );
}
