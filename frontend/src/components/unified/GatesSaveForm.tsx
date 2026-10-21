import { useMemo, useState } from "react";
import type { DatasetInfo } from "@/api/types";
import { describeFilter } from "@/lib/cellFilter";
import {
  bitsetAnd,
  bitsetCount,
  bitsetFromIndices,
  bitsetToSelection,
  bitsetToSet,
  gateNameError,
  nextGateName,
  type Gate,
  type GateEval,
  type TreeRow,
} from "@/lib/gates";
import { useGatesStore } from "@/stores/gatesStore";
import { useSelectionStore } from "@/stores/selectionStore";

const input =
  "rounded border border-slate-300 bg-white px-1.5 py-0.5 text-[11px] text-slate-700 focus:border-blue-400 focus:outline-none";

/** "Top level" and every counted gate, indented by depth. */
export function GateParentSelect({
  value,
  onChange,
  tree,
  evals,
  exclude = [],
  label = "Inside",
}: {
  value: string | null;
  onChange: (id: string | null) => void;
  tree: TreeRow[];
  evals: Map<string, GateEval>;
  exclude?: string[];
  label?: string;
}) {
  return (
    <label className="flex items-center gap-1">
      <span className="text-[11px] text-slate-500">{label}</span>
      <select
        aria-label="Parent gate"
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value || null)}
        className={`${input} max-w-[220px]`}
      >
        <option value="">Top level (all cells)</option>
        {tree
          .filter(({ gate }) => evals.get(gate.id)?.status === "ok" && !exclude.includes(gate.id))
          .map(({ gate, depth }) => (
            <option key={gate.id} value={gate.id}>
              {"  ".repeat(depth)}
              {gate.name}
            </option>
          ))}
      </select>
    </label>
  );
}

/**
 * "Save selection as gate": a name and a parent (the active gate, else the top
 * level). A lasso or legend selection is saved by cell position, intersected
 * with the parent; a Filter's selection is saved as the filter itself.
 */
export function GatesSaveForm({
  dataset,
  gates,
  tree,
  evals,
  defaultParent,
  onDone,
  onCancel,
}: {
  dataset: DatasetInfo;
  gates: Gate[];
  tree: TreeRow[];
  evals: Map<string, GateEval>;
  defaultParent: string | null;
  onDone: (message: string) => void;
  onCancel: () => void;
}) {
  const selected = useSelectionStore((s) => s.selectedCellIndices);
  const filter = useSelectionStore((s) => s.selectionFilter);
  const nCells = dataset.n_cells ?? 0;
  const [name, setName] = useState(() => nextGateName(gates));
  const [parentId, setParentId] = useState<string | null>(() =>
    defaultParent && evals.get(defaultParent)?.status === "ok" ? defaultParent : null,
  );
  const [touched, setTouched] = useState(false);

  const cells = useMemo(() => bitsetFromIndices(selected ?? [], nCells), [selected, nCells]);
  const nSelected = useMemo(() => bitsetCount(cells), [cells]);
  const parentEval = parentId ? evals.get(parentId) : undefined;
  const parentCells = parentEval?.status === "ok" ? parentEval.cells : null;
  const inside = useMemo(() => (parentCells ? bitsetAnd(cells, parentCells) : cells), [cells, parentCells]);
  const nInside = useMemo(() => bitsetCount(inside), [inside]);
  const parentName = gates.find((g) => g.id === parentId)?.name ?? "";
  const problem = gateNameError(name, gates);
  const outside = nSelected - nInside;

  const save = (e: React.FormEvent) => {
    e.preventDefault();
    setTouched(true);
    if (problem || nInside === 0) return;
    const store = useGatesStore.getState();
    const id = store.addGate(
      dataset,
      {
        name,
        parentId,
        def: filter ? { kind: "filter", filter } : { kind: "cells", selection: bitsetToSelection(inside) },
      },
      filter ? cells : undefined,
    );
    if (!id) return;
    // The new gate is the selection and the one the next gate goes inside.
    const shownCells = bitsetToSet(inside);
    useSelectionStore.getState().setSelection(shownCells, filter && !parentId ? filter : null);
    useGatesStore.setState({ active: { datasetId: dataset.id, gateId: id }, shown: { datasetId: dataset.id, gateId: id, cells: shownCells } });
    onDone(
      `Saved “${name.trim()}”: ${nInside.toLocaleString()} cells${parentId ? ` inside “${parentName}”` : ""}.` +
        (outside > 0 ? ` ${outside.toLocaleString()} selected cells outside it were left out.` : ""),
    );
  };

  return (
    <form
      onSubmit={save}
      onKeyDown={(e) => {
        if (e.key === "Escape") onCancel();
      }}
      aria-label="Save selection as gate"
      className="space-y-1 rounded border border-slate-200 bg-slate-50 px-2 py-1.5"
    >
      <div className="flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-1">
          <span className="text-[11px] text-slate-500">Name</span>
          <input autoFocus value={name} onChange={(e) => setName(e.target.value)} className={`${input} w-40`} />
        </label>
        <GateParentSelect value={parentId} onChange={setParentId} tree={tree} evals={evals} />
        <button
          type="submit"
          disabled={nInside === 0}
          className="rounded bg-blue-500 px-2 py-0.5 text-[11px] font-medium text-white hover:bg-blue-600 disabled:opacity-50"
        >
          Save gate
        </button>
        <button type="button" onClick={onCancel} className="rounded px-1.5 py-0.5 text-[11px] text-slate-500 hover:bg-slate-100">
          Cancel
        </button>
      </div>
      <p className="text-[11px] text-slate-500" data-testid="gate-save-caption">
        {parentId
          ? nInside === 0
            ? `None of the ${nSelected.toLocaleString()} selected cells are inside “${parentName}”.`
            : `Saves ${nInside.toLocaleString()} of the ${nSelected.toLocaleString()} selected cells, the ones inside “${parentName}”` +
              (outside > 0 ? `; ${outside.toLocaleString()} outside it are left out.` : " (all of them are).")
          : `Saves the ${nSelected.toLocaleString()} selected cells at the top level.`}{" "}
        {filter
          ? `Kept as the filter “${describeFilter(filter)}”, asked again each time, so it follows label and clustering changes.`
          : "Kept by cell position."}
      </p>
      {touched && problem && (
        <p role="alert" className="text-[11px] text-amber-700">
          {problem}
        </p>
      )}
    </form>
  );
}
