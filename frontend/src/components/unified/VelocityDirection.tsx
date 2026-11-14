import { useState } from "react";
import { Loader2 } from "lucide-react";
import {
  checkVelocityDirection,
  type VelocityDirection as Direction,
  type VelocityDirectionSpec,
  type VelocityRun,
  type VelocityTransition,
} from "@/api/velocity";
import type { ObsColumnInfo } from "@/api/types";
import { Caption } from "@/components/common/Caption";
import { Help } from "@/components/common/Help";
import { errorMessage } from "@/lib/errorMessage";

export type DirectionMode = "auto" | "order" | "root" | "column";

export interface DirectionChoice {
  mode: DirectionMode;
  /** Groups, earliest first, separated by ">" */
  order: string;
  root: string;
  column: string;
  /** A text column's values, earliest first, separated by ">" */
  columnOrder: string;
}

export const DEFAULT_DIRECTION: DirectionChoice = { mode: "auto", order: "", root: "", column: "", columnOrder: "" };

const split = (text: string) => text.split(">").map((x) => x.trim()).filter(Boolean);

/** The order to send with a run or a check; null when the choice is not filled in yet. */
export function directionSpec(choice: DirectionChoice, groupBy: string | null): VelocityDirectionSpec | null {
  const base = groupBy ? { group_by: groupBy } : {};
  if (choice.mode === "auto") return base;
  if (choice.mode === "order") {
    const order = split(choice.order);
    return order.length >= 2 ? { ...base, order } : null;
  }
  if (choice.mode === "root") return choice.root ? { ...base, root: choice.root } : null;
  if (!choice.column) return null;
  const columnOrder = split(choice.columnOrder);
  return { ...base, column: choice.column, ...(columnOrder.length ? { column_order: columnOrder } : {}) };
}

/** The choice a saved check was made with, so the controls show it. */
export function choiceFrom(direction: Direction | null | undefined): DirectionChoice {
  const spec = direction?.spec;
  if (!direction?.checked || !spec) return DEFAULT_DIRECTION;
  if (spec.order) return { ...DEFAULT_DIRECTION, mode: "order", order: spec.order.join(" > ") };
  if (spec.root) return { ...DEFAULT_DIRECTION, mode: "root", root: spec.root };
  if (spec.column && direction.kind !== "pseudotime") {
    return { ...DEFAULT_DIRECTION, mode: "column", column: spec.column, columnOrder: (spec.column_order ?? []).join(" > ") };
  }
  return DEFAULT_DIRECTION;
}

const STATUS: Record<VelocityTransition["status"], { text: string; cls: string }> = {
  forward: { text: "forward", cls: "text-emerald-700" },
  backward: { text: "BACKWARD", cls: "font-semibold text-red-700" },
  unclear: { text: "unclear", cls: "text-slate-500" },
  few_cells: { text: "too few cells", cls: "text-slate-500" },
  no_boundary: { text: "groups do not meet", cls: "text-slate-500" },
};

const signed = (x: number) => `${x >= 0 ? "+" : ""}${x.toFixed(2)}`;

function TransitionTable({ rows, k }: { rows: VelocityTransition[]; k: number }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-left">
        <thead className="text-slate-500">
          <tr>
            <th className="pr-2 font-medium">Transition</th>
            <th className="pr-2 font-medium" title={`Cells of the first group with cells of the second among their ${k} nearest on the plot`}>n</th>
            <th className="pr-2 font-medium" title="Mean cosine between each boundary cell's arrow and the way to its neighbours in the next group, with a 95% interval: +1 points straight at it, -1 straight away">Cosine</th>
            <th className="pr-2 font-medium" title="Boundary cells whose arrow points towards the next group">Forward</th>
            <th className="font-medium" title="Backward: the whole interval is below 0. Forward: above 0. Unclear: it spans 0.">Verdict</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.from}>${r.to}`} className="border-t border-slate-100">
              <td className="pr-2">{r.from} {"->"} {r.to}</td>
              <td className="pr-2">{r.n_cells.toLocaleString()}</td>
              <td className="pr-2">
                {r.mean_cos != null ? `${signed(r.mean_cos)} (${signed(r.low ?? 0)} to ${signed(r.high ?? 0)})` : "-"}
              </td>
              <td className="pr-2">{r.share_forward != null ? `${Math.round(r.share_forward * 100)}%` : "-"}</td>
              <td className={STATUS[r.status].cls}>{STATUS[r.status].text}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * The direction check of a velocity run (backend core/velocity_direction.py): its arrows against an
 * order the user trusts, per transition and per cell, and the controls to check against another order
 * or on the plot on screen.
 */
export function VelocityDirection({
  datasetId,
  run,
  embedding,
  embeddingOk,
  columns,
  choice,
  setChoice,
  onChecked,
}: {
  datasetId: string;
  run: VelocityRun;
  /** The plot on screen, and whether arrows can be drawn on it */
  embedding: string;
  embeddingOk: boolean;
  columns: readonly ObsColumnInfo[];
  choice: DirectionChoice;
  setChoice: (c: DirectionChoice) => void;
  onChecked: (run: VelocityRun) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const d = run.direction;
  const groups = run.clusters.map((c) => c.group);
  const spec = directionSpec(choice, run.group_column);

  const check = async (on: string | null) => {
    if (!spec) return;
    setBusy(true);
    setError(null);
    try {
      const res = await checkVelocityDirection(datasetId, { ...spec, embedding: on });
      onChecked(res.run);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const rows = d?.transitions ?? [];
  const backward = rows.filter((r) => r.status === "backward");
  const otherPlot = !!d?.embedding && embeddingOk && d.embedding !== embedding;
  const cells = d?.cells;

  return (
    <div className="space-y-2 rounded-md border border-slate-200 p-2">
      <p className="font-medium">
        <Help id="velocity.direction">Direction check</Help>
      </p>
      {d && !d.checked && <p className="text-slate-600">Direction unchecked: {d.reason}</p>}
      {d?.checked && (
        <>
          <p className="text-slate-600">
            Against {d.label}, by {d.group_column}, on {d.embedding}
            {d.derived ? ` (transitions: groups that meet on the plot and that the order separates in ${Math.round((d.order_auc ?? 0) * 100)}% of cell pairs)` : ""}.
          </p>
          {backward.length > 0 && (
            <p className="rounded-md border border-red-300 bg-red-50 p-2 text-red-800">
              Backward at {backward.map((r) => `${r.from} -> ${r.to} (${signed(r.mean_cos ?? 0)}, n = ${r.n_cells.toLocaleString()})`).join("; ")}.
              Velocity confidence does not show this; read the arrows there with care.
            </p>
          )}
          {rows.length > 0 ? <TransitionTable rows={rows} k={d.k ?? 30} /> : <p className="text-slate-500">No transitions to check.</p>}
          {!!d.skipped?.length && (
            <p className="text-slate-500">
              Not compared (they meet, but the order does not separate them): {d.skipped.map((s) => s.groups.join(" / ")).join(", ")}.
            </p>
          )}
          {cells && cells.mean_cos != null && (
            <p className="text-slate-600">
              Per cell, against the way the order rises among each cell's neighbours: mean cosine {signed(cells.mean_cos)},{" "}
              {Math.round((cells.share_forward ?? 0) * 100)}% forward ({cells.n_cells.toLocaleString()} of{" "}
              {cells.n_run_cells.toLocaleString()} cells have neighbours that differ in order).
            </p>
          )}
          <Caption>{d.note}</Caption>
        </>
      )}
      {otherPlot && (
        <p className="text-amber-700">
          Checked on {d?.embedding}; the plot shows {embedding}.{" "}
          <button type="button" disabled={busy || !spec} onClick={() => void check(embedding)} className="text-blue-600 hover:underline disabled:opacity-50">
            Check on this plot
          </button>
        </p>
      )}

      <div className="flex flex-wrap items-end gap-2">
        <label className="flex flex-col gap-0.5">
          <span className="text-slate-500">
            <Help id="velocity.direction_against">Check against</Help>
          </span>
          <select
            aria-label="Check against"
            value={choice.mode}
            onChange={(e) => setChoice({ ...choice, mode: e.target.value as DirectionMode,
              order: choice.order || groups.join(" > "), root: choice.root || groups[0] || "" })}
            className="rounded border border-slate-300 bg-white px-1 py-0.5"
          >
            <option value="auto">scView's pseudotime</option>
            <option value="order">Groups in an order</option>
            <option value="root">A root group (PAGA tree)</option>
            <option value="column">A column (pseudotime, time, stage)</option>
          </select>
        </label>
        {choice.mode === "order" && (
          <label className="flex min-w-[14rem] flex-1 flex-col gap-0.5">
            <span className="text-slate-500">{run.group_column ?? "Groups"}, earliest first</span>
            <input
              aria-label="Group order"
              value={choice.order}
              onChange={(e) => setChoice({ ...choice, order: e.target.value })}
              placeholder="A > B > C"
              className="rounded border border-slate-300 bg-white px-1 py-0.5"
            />
          </label>
        )}
        {choice.mode === "root" && (
          <select
            aria-label="Root group"
            value={choice.root}
            onChange={(e) => setChoice({ ...choice, root: e.target.value })}
            className="rounded border border-slate-300 bg-white px-1 py-0.5"
          >
            {groups.map((g) => (
              <option key={g} value={g}>{g}</option>
            ))}
          </select>
        )}
        {choice.mode === "column" && (
          <>
            <select
              aria-label="Order column"
              value={choice.column}
              onChange={(e) => setChoice({ ...choice, column: e.target.value })}
              className="max-w-[12rem] rounded border border-slate-300 bg-white px-1 py-0.5"
            >
              <option value="">Pick a column</option>
              {columns.map((c) => (
                <option key={c.name} value={c.name}>{c.name}</option>
              ))}
            </select>
            <input
              aria-label="Column values in order"
              value={choice.columnOrder}
              onChange={(e) => setChoice({ ...choice, columnOrder: e.target.value })}
              placeholder="Values in order, if text: day 1 > day 3"
              className="min-w-[12rem] flex-1 rounded border border-slate-300 bg-white px-1 py-0.5"
            />
          </>
        )}
        <button
          type="button"
          disabled={busy || !spec}
          onClick={() => void check(embeddingOk ? embedding : null)}
          className="rounded-md border border-slate-300 px-2 py-0.5 hover:bg-slate-50 disabled:opacity-50"
        >
          Check direction
        </button>
        {busy && <Loader2 className="h-3 w-3 animate-spin text-slate-500" />}
      </div>
      {error && <p className="text-red-600">{error}</p>}
    </div>
  );
}
