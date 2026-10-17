import { Loader2, Network } from "lucide-react";
import type { ObsColumnInfo } from "@/api/types";
import type { PagaView } from "@/api/trajectory";
import { prettyObsLabel } from "@/lib/formatting";

interface PagaCardProps {
  /** Columns whose groups can be the graph's nodes */
  columns: ObsColumnInfo[];
  groupby: string;
  onGroupbyChange: (column: string) => void;
  view: PagaView | null;
  computing: boolean;
  error: string | null;
  onCompute: () => void;
  show: boolean;
  onShowChange: (show: boolean) => void;
  threshold: number;
  onThresholdChange: (threshold: number) => void;
  treeOnly: boolean;
  onTreeOnlyChange: (treeOnly: boolean) => void;
  /** Edges drawn at the current settings */
  nVisible: number;
  /** The embedding has no positions for the graph (still loading, or none) */
  unplaced?: boolean;
}

const selectCls =
  "rounded-lg border border-slate-300 bg-white px-2 py-1.5 text-sm text-slate-700 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100";

/** Compute PAGA for a grouping and set how its graph is drawn over the embedding. */
export function PagaCard({
  columns,
  groupby,
  onGroupbyChange,
  view,
  computing,
  error,
  onCompute,
  show,
  onShowChange,
  threshold,
  onThresholdChange,
  treeOnly,
  onTreeOnlyChange,
  nVisible,
  unplaced = false,
}: PagaCardProps) {
  const record = view?.paga ?? null;
  return (
    <div className="flex-shrink-0 rounded-xl border border-slate-200 bg-white px-4 py-3 text-left shadow-sm">
      {columns.length === 0 ? (
        <p className="text-xs text-slate-500">No categorical column (cluster or cell type) to build a PAGA graph from.</p>
      ) : (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium text-slate-600">PAGA graph of</span>
          <select
            aria-label="PAGA grouping"
            value={groupby}
            onChange={(e) => onGroupbyChange(e.target.value)}
            className={selectCls}
          >
            {columns.map((c) => (
              <option key={c.name} value={c.name}>
                {prettyObsLabel(c.name)}
              </option>
            ))}
          </select>
          <button
            onClick={onCompute}
            disabled={computing || !groupby}
            className="flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-sm font-medium text-white transition hover:bg-primary/90 disabled:opacity-50"
          >
            {computing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Network className="h-3.5 w-3.5" />}
            {computing ? "Computing..." : record?.groupby === groupby ? "Recompute PAGA" : "Compute PAGA"}
          </button>
          {record && (
            <>
              <label className="ml-2 flex items-center gap-1.5 text-sm text-slate-600">
                <input type="checkbox" checked={show} onChange={(e) => onShowChange(e.target.checked)} />
                Show on plot
              </label>
              <label className="flex items-center gap-1.5 text-sm text-slate-600">
                Edges &ge;
                <input
                  type="range"
                  aria-label="PAGA edge threshold"
                  min={0}
                  max={1}
                  step={0.01}
                  value={threshold}
                  onChange={(e) => onThresholdChange(Number(e.target.value))}
                  className="w-28"
                />
                <span className="w-9 tabular-nums">{threshold.toFixed(2)}</span>
              </label>
              <label className="flex items-center gap-1.5 text-sm text-slate-600">
                <input type="checkbox" checked={treeOnly} onChange={(e) => onTreeOnlyChange(e.target.checked)} />
                Spanning tree only
              </label>
            </>
          )}
        </div>
      )}
      <p className="mt-1.5 text-xs text-slate-500">
        PAGA (scanpy) scores how strongly each pair of groups is connected in the neighbour graph,
        against random wiring (0 to 1). Nodes sit at each group&apos;s median on this embedding, sized by
        cell count; wider, darker edges are stronger.
      </p>
      {record && (
        <p className="mt-1 text-xs text-slate-500" data-testid="paga-summary">
          Current: {prettyObsLabel(record.groupby)}, {record.groups.length} groups,{" "}
          {nVisible} of {record.edges.length} edges shown; {record.graph}.
          {unplaced && " This embedding has no positions for it."}
        </p>
      )}
      {view?.stale && <p className="mt-1 text-xs text-amber-700">{view.stale}</p>}
      {error && <p className="mt-1.5 text-xs text-red-600">{error}</p>}
    </div>
  );
}
