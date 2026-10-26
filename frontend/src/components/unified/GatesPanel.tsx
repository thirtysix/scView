import { useMemo, useState } from "react";
import { Combine, Download, Eye, Loader2, Pencil, Plus, Tags, Trash2, X } from "lucide-react";
import { obsColumnValues } from "@/api/gates";
import type { DatasetInfo } from "@/api/types";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { GatesCombineForm } from "@/components/unified/GatesCombineForm";
import { GatesLabelColumn } from "@/components/unified/GatesLabelColumn";
import { GatesSaveForm } from "@/components/unified/GatesSaveForm";
import { useGates, useResolveFilterGates } from "@/hooks/useGates";
import { getPalette } from "@/lib/colors";
import { downloadCsv } from "@/lib/csv";
import { replicateColumns, suggestReplicateColumn } from "@/lib/de";
import { exportFilename } from "@/lib/exportFilename";
import { prettyObsLabel } from "@/lib/formatting";
import {
  dependents,
  describeGate,
  formatGatePct,
  gateCountsTable,
  gateNameError,
  labelSlots,
  MAX_GATES,
  planLabelColumn,
  subtree,
  type DeleteMode,
  type Gate,
  type GateEval,
} from "@/lib/gates";
import { GATES_MAX_CHARS, selectGate, useGatesStorageStatus, useGatesStore } from "@/stores/gatesStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { errorMessage } from "@/lib/errorMessage";

type Form = "save" | "combine" | "label" | null;

const btn =
  "flex items-center gap-1 rounded px-2 py-0.5 text-[11px] font-medium text-slate-600 transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40";

/**
 * The gate tree: name, colour, n, % of parent and % of total per gate; a click
 * makes a gate the selection (shown on the plot, used by DE "selection vs
 * rest", "Label selection…" and the histogram) and the gate new ones go
 * inside. Saving, combining, a label column and the counts CSV start here.
 */
export function GatesPanel({ dataset, onClose }: { dataset: DatasetInfo; onClose: () => void }) {
  const { record, gates, tree, stale, evals } = useGates(dataset);
  useResolveFilterGates(dataset, gates);
  const nCells = dataset.n_cells ?? 0;
  const activeId = useGatesStore((s) => (s.active?.datasetId === dataset.id ? s.active.gateId : null));
  const shown = useGatesStore((s) => (s.shown?.datasetId === dataset.id ? s.shown : null));
  const selected = useSelectionStore((s) => s.selectedCellIndices);
  const storage = useGatesStorageStatus();
  const palette = getPalette(useSettingsStore((s) => s.categoricalPalette));
  const [form, setForm] = useState<Form>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [renaming, setRenaming] = useState<{ id: string; name: string } | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  const slots = useMemo(() => labelSlots(tree, planLabelColumn(gates, evals, nCells)), [tree, gates, evals, nCells]);
  const shownId = shown && shown.cells === selected ? shown.gateId : null;
  const hasSelection = !!selected && selected.size > 0;
  const activeGate = gates.find((g) => g.id === activeId) ?? null;
  const full = gates.length >= MAX_GATES;
  const nameOf = (id: string | null) => gates.find((g) => g.id === id)?.name ?? "";

  const done = (text: string) => {
    setForm(null);
    setMessage(text);
  };
  const toggleForm = (f: Exclude<Form, null>) => {
    setMessage(null);
    setForm((cur) => (cur === f ? null : f));
  };

  const remove = (gate: Gate, mode: DeleteMode) => {
    const result = useGatesStore.getState().deleteGate(dataset.id, gate.id, mode, evals);
    setDeleting(null);
    if (!result) return;
    const parts = [`Deleted ${result.removed.map((n) => `“${n}”`).join(", ")}.`];
    if (result.frozen.length > 0) {
      parts.push(
        `${result.frozen.map((n) => `“${n}”`).join(", ")} used it and now ${result.frozen.length === 1 ? "keeps its" : "keep their"} cells, saved by position.`,
      );
    }
    setMessage(parts.join(" "));
  };
  const askDelete = (gate: Gate) => {
    const hasChildren = gates.some((g) => g.parentId === gate.id);
    if (!hasChildren && dependents(gates, gate.id).length === 0) remove(gate, "subtree");
    else setDeleting(gate.id);
  };

  const commitRename = () => {
    if (!renaming) return;
    if (gateNameError(renaming.name, gates, renaming.id)) return;
    useGatesStore.getState().renameGate(dataset.id, renaming.id, renaming.name);
    setRenaming(null);
  };
  const renameProblem = renaming ? gateNameError(renaming.name, gates, renaming.id) : null;

  return (
    <div className="space-y-1.5 text-xs" aria-label="Gates">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="flex items-center gap-1 text-[11px] font-semibold text-slate-700">
          Gates
          <InfoTip label="About gates" width={300}>
            A gate is a named set of cells: a lasso or legend selection (saved by cell position), a Filter (asked again
            each time, so it follows label changes), or an AND / OR / NOT of other gates. A gate drawn inside another
            keeps only the cells that are also in its parent, so % of parent is always a share of the parent&rsquo;s
            cells. Click a gate to make it the selection: DE &ldquo;selection vs rest&rdquo;, &ldquo;Label
            selection&hellip;&rdquo; and the histogram then use it, and new gates go inside it.
          </InfoTip>
        </span>
        <button
          className={btn}
          onClick={() => toggleForm("save")}
          disabled={!hasSelection || !!stale || full}
          aria-expanded={form === "save"}
          title={
            stale
              ? "Discard the stale gates first: they were drawn on other cells"
              : full
                ? `At most ${MAX_GATES} gates are kept per dataset`
                : hasSelection
                  ? "Save the selected cells as a named gate"
                  : "Select cells first (lasso, a legend click or Filter)"
          }
        >
          <Plus className="h-3 w-3" />
          Save selection as gate…
        </button>
        <button
          className={btn}
          onClick={() => toggleForm("combine")}
          disabled={gates.length < 2 || full}
          aria-expanded={form === "combine"}
          title={gates.length < 2 ? "Needs two gates" : "A new gate from two or more gates: AND, OR or NOT"}
        >
          <Combine className="h-3 w-3" />
          Combine…
        </button>
        <button
          className={btn}
          onClick={() => toggleForm("label")}
          disabled={gates.length === 0}
          aria-expanded={form === "label"}
          title="Write the leaf gates into one annotation column (cells in no gate: Unassigned)"
        >
          <Tags className="h-3 w-3" />
          Label column…
        </button>
        <CountsExport dataset={dataset} gates={gates} evals={evals} />
        <div className="flex-1" />
        <button
          onClick={onClose}
          aria-label="Close gates"
          title="Close the gates list (the gates are kept)"
          className="rounded p-0.5 text-slate-400 hover:bg-slate-100"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      {activeGate && (
        <p className="text-[11px] text-slate-500" data-testid="gates-active">
          New gates go inside <span className="font-medium text-orange-700">{activeGate.name}</span> (the gate clicked
          last); the save form can put them elsewhere.
        </p>
      )}

      {stale && record && (
        <div role="alert" className="flex flex-wrap items-center gap-2 rounded bg-amber-50 px-2 py-1 text-[11px] text-amber-800">
          <span>
            {record.gates.length} saved gate{record.gates.length === 1 ? " is" : "s are"} not applied. {stale}
          </span>
          <button
            onClick={() => {
              useGatesStore.getState().discard(dataset.id);
              setMessage("Discarded the stale gates.");
            }}
            className="rounded border border-amber-300 bg-white px-1.5 py-0.5 text-[11px] text-amber-800 hover:bg-amber-100"
          >
            Discard them
          </button>
        </div>
      )}
      {storage.blocked && (
        <p className="text-[11px] text-amber-700">
          This browser won&rsquo;t keep the gates (its storage is blocked or full): they last until the page is
          closed. Save a session file to keep them.
        </p>
      )}
      {!storage.blocked && storage.notKept.includes(dataset.id) && (
        <p className="text-[11px] text-amber-700">
          These gates are too large to keep in this browser (over {(GATES_MAX_CHARS / 1e6).toFixed(0)} MB for all
          gates): they last until the page is closed. Save a session file to keep them.
        </p>
      )}
      {!storage.blocked && storage.notKept.some((id) => id !== dataset.id) && (
        <p className="text-[11px] text-amber-700">
          Gates of {storage.notKept.filter((id) => id !== dataset.id).length} other dataset(s) no longer fit in the{" "}
          {(GATES_MAX_CHARS / 1e6).toFixed(0)} MB this browser keeps for gates and will be gone after a reload.
        </p>
      )}

      {form === "save" && hasSelection && (
        <GatesSaveForm
          dataset={dataset}
          gates={gates}
          tree={tree}
          evals={evals}
          defaultParent={activeId}
          onDone={done}
          onCancel={() => setForm(null)}
        />
      )}
      {form === "combine" && (
        <GatesCombineForm dataset={dataset} gates={gates} tree={tree} evals={evals} onDone={done} onCancel={() => setForm(null)} />
      )}
      {form === "label" && (
        <GatesLabelColumn dataset={dataset} gates={gates} evals={evals} onDone={done} onCancel={() => setForm(null)} />
      )}
      {message && (
        <p role="status" className="text-[11px] text-slate-600">
          {message}
        </p>
      )}

      {tree.length === 0 ? (
        !stale && (
          <p className="text-[11px] text-slate-500">
            No gates yet. Select cells with the lasso, a legend click or Filter, then &ldquo;Save selection as
            gate&rdquo;.
          </p>
        )
      ) : (
        <>
          <div className="max-h-56 overflow-y-auto">
            <table className="w-full text-[11px]">
              <thead className="sticky top-0 bg-white">
                <tr className="text-left text-[10px] font-medium text-slate-500">
                  <th className="py-0.5 font-medium">Gate</th>
                  <th
                    className="py-0.5 text-right font-medium"
                    title="Cells in the gate: its own cells (saved, from its filter, or combined from other gates) that are also in its parent gate"
                  >
                    n
                  </th>
                  <th
                    className="py-0.5 text-right font-medium"
                    title={`n divided by the parent gate's n; for a top-level gate, by all ${nCells.toLocaleString()} cells`}
                  >
                    % parent
                  </th>
                  <th
                    className="py-0.5 text-right font-medium"
                    title={`n divided by all ${nCells.toLocaleString()} cells in the dataset`}
                  >
                    % total
                  </th>
                  <th className="w-16 py-0.5">
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {tree.map(({ gate, depth, hasChildren }) => {
                  const ev = evals.get(gate.id);
                  const slot = slots.get(gate.id);
                  const rgb = slot !== undefined ? palette[slot % palette.length] : undefined;
                  const isActive = gate.id === activeId;
                  const isShown = gate.id === shownId;
                  const canSelect = ev?.status === "ok" && ev.n > 0;
                  return (
                    <GateRows
                      key={gate.id}
                      gate={gate}
                      depth={depth}
                      ev={ev}
                      swatch={rgb ? `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})` : null}
                      swatchTitle={
                        rgb
                          ? "The colour this gate takes in a label column made with “Label column…” (Unassigned takes the first)"
                          : hasChildren
                            ? "Has gates inside it: a label column names its cells by those gates"
                            : "No cells of its own left (a deeper or later gate takes them), so no label"
                      }
                      active={isActive}
                      shown={isShown}
                      canSelect={canSelect}
                      describe={describeGate(gate, gates)}
                      onSelect={() => {
                        setMessage(null);
                        selectGate(dataset.id, gate, ev);
                      }}
                      renaming={renaming?.id === gate.id ? renaming.name : null}
                      renameProblem={renaming?.id === gate.id ? renameProblem : null}
                      onRenameStart={() => setRenaming({ id: gate.id, name: gate.name })}
                      onRenameChange={(name) => setRenaming({ id: gate.id, name })}
                      onRenameCommit={commitRename}
                      onRenameCancel={() => setRenaming(null)}
                      onDelete={() => askDelete(gate)}
                      confirmDelete={
                        deleting === gate.id ? (
                          <DeleteConfirm
                            gate={gate}
                            gates={gates}
                            parentName={nameOf(gate.parentId)}
                            onDelete={(mode) => remove(gate, mode)}
                            onCancel={() => setDeleting(null)}
                          />
                        ) : null
                      }
                    />
                  );
                })}
              </tbody>
            </table>
          </div>
          <Caption>
            n = cells in the gate and in its parent gate; % parent = n / the parent&rsquo;s n (all{" "}
            {nCells.toLocaleString()} cells for a top-level gate); % total = n / all {nCells.toLocaleString()} cells.
          </Caption>
        </>
      )}
    </div>
  );
}

interface GateRowsProps {
  gate: Gate;
  depth: number;
  ev: GateEval | undefined;
  /** Its label-column colour, or null (hollow) when it gets no label */
  swatch: string | null;
  swatchTitle: string;
  active: boolean;
  shown: boolean;
  canSelect: boolean;
  describe: string;
  onSelect: () => void;
  renaming: string | null;
  renameProblem: string | null;
  onRenameStart: () => void;
  onRenameChange: (name: string) => void;
  onRenameCommit: () => void;
  onRenameCancel: () => void;
  onDelete: () => void;
  confirmDelete: React.ReactNode;
}

function GateRows(p: GateRowsProps) {
  const { gate, ev } = p;
  const kind =
    gate.def.kind === "cells" ? "" : gate.def.kind === "filter" ? "filter" : gate.def.op.toUpperCase();
  return (
    <>
      <tr className={`border-t border-slate-100 ${p.active ? "bg-orange-50" : ""}`} data-testid={`gate-row-${gate.name}`}>
        <td className="py-0.5">
          <div className="flex min-w-0 items-center gap-1" style={{ paddingLeft: p.depth * 14 }}>
            <span
              className={`h-2.5 w-2.5 flex-shrink-0 rounded-sm ${p.swatch ? "" : "border border-slate-400"}`}
              style={p.swatch ? { background: p.swatch } : undefined}
              title={p.swatchTitle}
              data-testid="gate-swatch"
            />
            {p.renaming !== null ? (
              <span className="flex min-w-0 items-center gap-1">
                <input
                  autoFocus
                  aria-label={`Rename ${gate.name}`}
                  value={p.renaming}
                  onChange={(e) => p.onRenameChange(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") p.onRenameCommit();
                    if (e.key === "Escape") p.onRenameCancel();
                  }}
                  onBlur={p.onRenameCancel}
                  className="w-40 rounded border border-slate-300 px-1 py-0 text-[11px] focus:border-blue-400 focus:outline-none"
                />
                {p.renameProblem && (
                  <span role="alert" className="truncate text-[10px] text-amber-700">
                    {p.renameProblem}
                  </span>
                )}
              </span>
            ) : (
              <button
                onClick={p.onSelect}
                disabled={!p.canSelect}
                title={`${gate.name}: ${p.describe}. ${
                  p.canSelect
                    ? p.shown
                      ? "Click to clear the selection."
                      : "Click to make it the selection (shown on the plot) and the gate new ones go inside."
                    : ev?.status === "ok"
                      ? "No cells in this gate."
                      : ""
                }`}
                className={`truncate text-left font-medium disabled:cursor-default ${
                  p.active ? "text-orange-700" : "text-slate-700 hover:text-blue-600"
                }`}
              >
                {gate.name}
              </button>
            )}
            {kind && <span className="flex-shrink-0 rounded bg-slate-100 px-1 text-[9px] text-slate-500">{kind}</span>}
          </div>
        </td>
        <EvalCells ev={ev} />
        <td className="py-0.5">
          <div className="flex items-center justify-end gap-0.5">
            <button
              onClick={p.onSelect}
              disabled={!p.canSelect}
              aria-label={`Show ${gate.name} on the plot`}
              aria-pressed={p.shown}
              title={p.shown ? "Shown on the plot as the selection; click to clear" : "Show on the plot (as the selection; other cells fade)"}
              className={`rounded p-0.5 disabled:opacity-30 ${p.shown ? "text-orange-600" : "text-slate-400 hover:text-slate-600"}`}
            >
              <Eye className="h-3 w-3" />
            </button>
            <button
              onClick={p.onRenameStart}
              aria-label={`Rename ${gate.name}`}
              title="Rename"
              className="rounded p-0.5 text-slate-400 hover:text-slate-600"
            >
              <Pencil className="h-3 w-3" />
            </button>
            <button
              onClick={p.onDelete}
              aria-label={`Delete ${gate.name}`}
              title="Delete"
              className="rounded p-0.5 text-slate-400 hover:text-red-600"
            >
              <Trash2 className="h-3 w-3" />
            </button>
          </div>
        </td>
      </tr>
      {p.confirmDelete && (
        <tr>
          <td colSpan={5} className="pb-1">
            {p.confirmDelete}
          </td>
        </tr>
      )}
    </>
  );
}

function EvalCells({ ev }: { ev: GateEval | undefined }) {
  if (!ev || ev.status === "pending") {
    return (
      <td colSpan={3} className="py-0.5 text-right text-slate-400">
        <span className="inline-flex items-center gap-1">
          <Loader2 className="h-3 w-3 animate-spin" />
          counting
        </span>
      </td>
    );
  }
  if (ev.status === "error") {
    return (
      <td colSpan={3} className="max-w-[260px] truncate py-0.5 text-right text-red-600" title={`Not counted: ${ev.reason}`}>
        Not counted: {ev.reason}
      </td>
    );
  }
  return (
    <>
      <td className="py-0.5 text-right tabular-nums text-slate-700">{ev.n.toLocaleString()}</td>
      <td className="py-0.5 text-right tabular-nums text-slate-600">{formatGatePct(ev.pctParent)}</td>
      <td className="py-0.5 text-right tabular-nums text-slate-600">{formatGatePct(ev.pctTotal)}</td>
    </>
  );
}

function DeleteConfirm({
  gate,
  gates,
  parentName,
  onDelete,
  onCancel,
}: {
  gate: Gate;
  gates: Gate[];
  parentName: string;
  onDelete: (mode: DeleteMode) => void;
  onCancel: () => void;
}) {
  const children = gates.filter((g) => g.parentId === gate.id);
  const users = dependents(gates, gate.id);
  const below = subtree(gates, gate.id).length - 1;
  const where = parentName ? `“${parentName}”` : "the top level";
  return (
    <div role="dialog" aria-label={`Delete ${gate.name}`} className="space-y-1 rounded border border-red-200 bg-red-50 px-2 py-1 text-[11px] text-red-800">
      {children.length > 0 && (
        <p>
          &ldquo;{gate.name}&rdquo; has {children.length} gate{children.length === 1 ? "" : "s"} directly inside it.
          Moved up to {where}, a gate saved by position keeps its cells; a filter or combined gate then counts its
          cells anywhere in {parentName ? `“${parentName}”` : "the dataset"}.
        </p>
      )}
      {users.length > 0 && (
        <p>
          {users.map((u) => `“${u.name}”`).join(", ")} {users.length === 1 ? "is" : "are"} built from it and will keep
          the cells {users.length === 1 ? "it has" : "they have"} now, saved by position.
        </p>
      )}
      <div className="flex flex-wrap gap-1.5">
        {children.length > 0 && (
          <button onClick={() => onDelete("reparent")} className="rounded border border-red-300 bg-white px-1.5 py-0.5 hover:bg-red-100">
            Delete it, move {children.length === 1 ? "its gate" : `its ${children.length} gates`} up
          </button>
        )}
        <button onClick={() => onDelete("subtree")} className="rounded border border-red-300 bg-white px-1.5 py-0.5 hover:bg-red-100">
          {below > 0 ? `Delete it and the ${below} gate${below === 1 ? "" : "s"} inside` : "Delete it"}
        </button>
        <button onClick={onCancel} className="rounded px-1.5 py-0.5 text-slate-600 hover:bg-white">
          Cancel
        </button>
      </div>
    </div>
  );
}

/** "Export counts (CSV)", with per-sample columns when a sample/donor column is picked. */
function CountsExport({ dataset, gates, evals }: { dataset: DatasetInfo; gates: Gate[]; evals: Map<string, GateEval> }) {
  const options = useMemo(() => replicateColumns(dataset.obs_columns, []), [dataset.obs_columns]);
  const [column, setColumn] = useState(() => suggestReplicateColumn(options));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nCells = dataset.n_cells ?? 0;
  const counted = gates.length > 0 && gates.every((g) => evals.get(g.id)?.status === "ok");

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      let sample = null;
      if (column) {
        const values = await obsColumnValues(dataset.id, column);
        if (values.length !== nCells) throw new Error(`“${column}” has ${values.length} values for ${nCells} cells.`);
        sample = { column, values };
      }
      const { headers, rows } = gateCountsTable(gates, evals, nCells, sample);
      downloadCsv(`${exportFilename(dataset.name, "gates")}.csv`, headers, rows);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <span className="flex items-center gap-1">
      <button
        className={btn}
        onClick={() => void run()}
        disabled={!counted || busy}
        title={
          counted
            ? "Gate, parent, definition, n, % parent and % total; per sample, n and % of the parent's cells in that sample"
            : "Every gate must be counted first"
        }
      >
        {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Download className="h-3 w-3" />}
        Export counts (CSV)
      </button>
      {options.length > 0 && (
        <select
          aria-label="Per-sample columns in the CSV"
          value={column}
          onChange={(e) => setColumn(e.target.value)}
          title="Adds n and % of the parent gate's cells per value of this column (a donor or sample)"
          className="max-w-[130px] rounded border border-slate-300 bg-white px-1 py-0.5 text-[10px] text-slate-700 focus:border-blue-400 focus:outline-none"
        >
          <option value="">per sample: none</option>
          {options.map((c) => (
            <option key={c.name} value={c.name}>
              per {prettyObsLabel(c.name)}
            </option>
          ))}
        </select>
      )}
      {error && (
        <span role="alert" className="max-w-[220px] truncate text-[11px] text-red-600" title={error}>
          {error}
        </span>
      )}
    </span>
  );
}
