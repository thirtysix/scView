import { useMemo, useState } from "react";
import type { DatasetInfo } from "@/api/types";
import { GateParentSelect } from "@/components/unified/GatesSaveForm";
import {
  bitsetAnd,
  bitsetAndNot,
  bitsetCount,
  bitsetToSet,
  combineCells,
  combineName,
  defaultCombineParent,
  formatGatePct,
  gateNameError,
  pctOf,
  type Bitset,
  type CombineOp,
  type Gate,
  type GateEval,
  type TreeRow,
} from "@/lib/gates";
import { useGatesStore } from "@/stores/gatesStore";
import { useSelectionStore } from "@/stores/selectionStore";

const input =
  "rounded border border-slate-300 bg-white px-1.5 py-0.5 text-[11px] text-slate-700 focus:border-blue-400 focus:outline-none";

const OPS: { value: CombineOp; label: string; title: string }[] = [
  { value: "and", label: "AND", title: "Cells in every picked gate" },
  { value: "or", label: "OR", title: "Cells in any picked gate" },
  { value: "not", label: "NOT", title: "Cells in the first gate and in none of the others" },
];

/** "Combine…": a new gate from two or more gates with AND, OR or NOT ("A not B"). */
export function GatesCombineForm({
  dataset,
  gates,
  tree,
  evals,
  onDone,
  onCancel,
}: {
  dataset: DatasetInfo;
  gates: Gate[];
  tree: TreeRow[];
  evals: Map<string, GateEval>;
  onDone: (message: string) => void;
  onCancel: () => void;
}) {
  const nCells = dataset.n_cells ?? 0;
  const [op, setOp] = useState<CombineOp>("and");
  const [first, setFirst] = useState<string>(() => tree[0]?.gate.id ?? "");
  const [picked, setPicked] = useState<string[]>([]);
  const [customName, setCustomName] = useState<string | null>(null);
  const [customParent, setCustomParent] = useState<{ id: string | null } | null>(null);
  const [touched, setTouched] = useState(false);

  const operands = useMemo(() => {
    const order = tree.map((r) => r.gate.id);
    const others = picked
      .filter((id) => order.includes(id) && (op !== "not" || id !== first))
      .sort((a, b) => order.indexOf(a) - order.indexOf(b));
    return op === "not" ? (first ? [first, ...others] : others) : others;
  }, [tree, picked, op, first]);
  const nameOf = (id: string | null) => gates.find((g) => g.id === id)?.name ?? "";
  const autoName = combineName(op, operands.map(nameOf));
  const name = customName ?? autoName;
  const parentId = customParent ? customParent.id : defaultCombineParent(gates, op, operands);
  const enough = operands.length >= 2;

  const preview = useMemo(() => {
    if (!enough) return null;
    const parts: Bitset[] = [];
    for (const id of operands) {
      const e = evals.get(id);
      if (e?.status !== "ok") return null;
      parts.push(e.cells);
    }
    const own = combineCells(op, parts);
    const parent = parentId ? evals.get(parentId) : undefined;
    if (parentId && parent?.status !== "ok") return null;
    const cells = parent?.status === "ok" ? bitsetAnd(own, parent.cells) : own;
    const n = bitsetCount(cells);
    const parentN = parent?.status === "ok" ? parent.n : nCells;
    const outside = parent?.status === "ok" ? bitsetCount(bitsetAndNot(own, parent.cells)) : 0;
    return { cells, n, pct: pctOf(n, parentN), outside };
  }, [enough, operands, op, parentId, evals, nCells]);

  const problem = !enough
    ? op === "not"
      ? "Pick a gate to start from and at least one to take away."
      : "Pick at least two gates."
    : gateNameError(name, gates);

  const togglePick = (id: string) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));

  const create = (e: React.FormEvent) => {
    e.preventDefault();
    setTouched(true);
    if (problem || !preview) return;
    const id = useGatesStore.getState().addGate(dataset, {
      name,
      parentId,
      def: { kind: "combine", op, operands },
    });
    if (!id) return;
    if (preview.n > 0) {
      const cells = bitsetToSet(preview.cells);
      useSelectionStore.getState().setHighlight(null);
      useSelectionStore.getState().setSelection(cells);
      useGatesStore.setState({ active: { datasetId: dataset.id, gateId: id }, shown: { datasetId: dataset.id, gateId: id, cells } });
    }
    onDone(`Made “${name.trim()}”: ${preview.n.toLocaleString()} cells.`);
  };

  return (
    <form
      onSubmit={create}
      onKeyDown={(e) => {
        if (e.key === "Escape") onCancel();
      }}
      aria-label="Combine gates"
      className="space-y-1 rounded border border-slate-200 bg-slate-50 px-2 py-1.5"
    >
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex overflow-hidden rounded border border-slate-300" role="radiogroup" aria-label="Combine with">
          {OPS.map((o) => (
            <button
              key={o.value}
              type="button"
              role="radio"
              aria-checked={op === o.value}
              title={o.title}
              onClick={() => setOp(o.value)}
              className={`px-1.5 py-0.5 text-[10px] font-medium ${op === o.value ? "bg-blue-500 text-white" : "bg-white text-slate-600 hover:bg-slate-100"}`}
            >
              {o.label}
            </button>
          ))}
        </div>
        {op === "not" && (
          <label className="flex items-center gap-1">
            <span className="text-[11px] text-slate-500">From</span>
            <select aria-label="Start from gate" value={first} onChange={(e) => setFirst(e.target.value)} className={`${input} max-w-[180px]`}>
              {tree.map(({ gate, depth }) => (
                <option key={gate.id} value={gate.id}>
                  {"  ".repeat(depth)}
                  {gate.name}
                </option>
              ))}
            </select>
          </label>
        )}
        <span className="text-[11px] text-slate-500">{op === "not" ? "take away" : "gates"}</span>
        <div className="flex flex-wrap gap-x-2 gap-y-0.5">
          {tree
            .filter(({ gate }) => op !== "not" || gate.id !== first)
            .map(({ gate }) => (
              <label key={gate.id} className="flex items-center gap-0.5 text-[11px] text-slate-700">
                <input type="checkbox" checked={picked.includes(gate.id)} onChange={() => togglePick(gate.id)} className="h-3 w-3" />
                {gate.name}
              </label>
            ))}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-1">
          <span className="text-[11px] text-slate-500">Name</span>
          <input
            value={name}
            onChange={(e) => setCustomName(e.target.value)}
            placeholder="Pick gates first"
            className={`${input} w-48`}
          />
        </label>
        <GateParentSelect value={parentId} onChange={(id) => setCustomParent({ id })} tree={tree} evals={evals} />
        <button
          type="submit"
          disabled={!preview}
          className="rounded bg-blue-500 px-2 py-0.5 text-[11px] font-medium text-white hover:bg-blue-600 disabled:opacity-50"
        >
          Make gate
        </button>
        <button type="button" onClick={onCancel} className="rounded px-1.5 py-0.5 text-[11px] text-slate-500 hover:bg-slate-100">
          Cancel
        </button>
      </div>
      {preview && (
        <p className="text-[11px] text-slate-500" data-testid="gate-combine-caption">
          &ldquo;{name.trim()}&rdquo;: {preview.n.toLocaleString()} cells, {formatGatePct(preview.pct)} of{" "}
          {parentId ? `“${nameOf(parentId)}”` : `all ${nCells.toLocaleString()} cells`}.
          {preview.outside > 0 &&
            ` ${preview.outside.toLocaleString()} cells of the combination are outside “${nameOf(parentId)}” and are left out.`}{" "}
          It follows its gates: when one changes, so does this.
        </p>
      )}
      {touched && problem && (
        <p role="alert" className="text-[11px] text-amber-700">
          {problem}
        </p>
      )}
    </form>
  );
}
