import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ChevronDown, ChevronUp, Loader2, Waypoints, X } from "lucide-react";
import { fetchEmbeddingBinary } from "@/api/embeddings";
import {
  getEmbeddingReliability,
  getReliabilitySummary,
  type ReliabilityGroup,
  type ReliabilitySummary,
} from "@/api/reliability";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { embeddingDims } from "@/hooks/useEmbedding";
import { useEmbeddingReady } from "@/hooks/useEmbeddingReady";
import { decodeArrowBuffer } from "@/lib/arrow";
import {
  applyReliability,
  errorText,
  fmtScore,
  fmtShare,
  groupCells,
  reliabilityCaption,
  sameView,
  sortGroups,
  summaryMatches,
  summaryOf,
  type GroupSortKey,
} from "@/lib/embeddingReliability";
import { prettyEmbeddingLabel, prettyObsLabel } from "@/lib/formatting";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

/** A summary with the dataset it belongs to. */
interface Shown {
  datasetId: string;
  summary: ReliabilitySummary;
}

const COLUMN_TIPS: Record<GroupSortKey, string> = {
  group: "The group of the column chosen above (default: the clustering)",
  n: "Cells of the group that were scored",
  median:
    "Median over the group's cells of the share of each cell's graph neighbours that are also among its nearest cells on the plot (1 = all kept); the tooltip on each value gives the interquartile range",
  share_low:
    "Share of the group's cells below the low threshold: fewer than 1 in 5 of their graph neighbours are beside them on the plot",
};

/**
 * "Plot reliability" in the Unified View toolbar: colours each cell by the share
 * of its graph neighbours (the kNN graph the clustering used) that the plot keeps
 * around it, on a fixed 0-1 scale, with a floating per-group table. On while the
 * main plot's overlay is this score; switching the embedding refetches it.
 */
export function EmbeddingReliabilityControl() {
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const dataset = useDatasetStore((s) => s.currentDataset);
  const embedding = useSettingsStore((s) => s.embedding);
  const pcaDims = useSettingsStore((s) => s.pcaDims);
  const nDims = dataset?.embedding_dimensions?.[embedding];
  const dims = embeddingDims(embedding, pcaDims, nDims);
  const d0 = dims?.[0];
  const d1 = dims?.[1];
  const ready = useEmbeddingReady(datasetId, embedding);
  const overlay = useUnifiedViewStore((s) => s.scatterOverlay);
  const overlayDatasetId = useUnifiedViewStore((s) => s.overlayDatasetId);
  const ref = overlay?.type === "score" && overlayDatasetId === datasetId ? overlay.reliability : undefined;

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shown, setShown] = useState<Shown | null>(null);
  const [groupBy, setGroupBy] = useState<string | null>(null);
  const latest = useRef(0);

  // Another dataset: nothing carried over.
  useEffect(() => {
    setShown(null);
    setGroupBy(null);
    setError(null);
  }, [datasetId]);

  const clearOverlay = useCallback(() => {
    const store = useUnifiedViewStore.getState();
    if (store.scatterOverlay?.type === "score" && store.scatterOverlay.reliability) {
      store.setScatterOverlay(null);
      store.setOverlayValues(null);
      store.setOverlayLabel("");
    }
  }, []);

  /** Scores for the plot on screen, onto the scatter. */
  const load = useCallback(async () => {
    if (!datasetId || !embedding) return;
    const req = ++latest.current;
    const view: [number, number] | undefined = d0 !== undefined && d1 !== undefined ? [d0, d1] : undefined;
    setBusy(true);
    setError(null);
    try {
      const res = await getEmbeddingReliability(datasetId, embedding, { dims: view, groupBy: groupBy ?? undefined });
      if (req !== latest.current) return; // a newer request (another embedding) won
      applyReliability(datasetId, res, view);
      setShown({ datasetId, summary: summaryOf(res) });
    } catch (e) {
      if (req !== latest.current) return;
      setError(`Plot reliability: ${errorText(e)}`);
      clearOverlay();
    } finally {
      if (req === latest.current) setBusy(false);
    }
  }, [datasetId, embedding, d0, d1, groupBy, clearOverlay]);

  // Switching the embedding (or the PCA pair) while it is on fetches it for the new plot.
  useEffect(() => {
    if (!ref || !ready || sameView(ref, embedding, dims)) return;
    void load();
  }, [ref, ready, embedding, d0, d1]); // eslint-disable-line react-hooks/exhaustive-deps

  // On without a summary (restored from a session): fetch the table only.
  const summary = shown && shown.datasetId === datasetId ? shown.summary : null;
  const matches = !!ref && !!summary && summaryMatches(ref, summary);
  useEffect(() => {
    if (!ref || !datasetId || matches || busy) return;
    const req = ++latest.current;
    getReliabilitySummary(datasetId, ref.embedding, { dims: ref.dims, groupBy: groupBy ?? undefined })
      .then((s) => {
        if (req === latest.current) setShown({ datasetId, summary: s });
      })
      .catch((e) => {
        if (req === latest.current) setError(`Plot reliability: ${errorText(e)}`);
      });
  }, [ref, datasetId, matches]); // eslint-disable-line react-hooks/exhaustive-deps

  const regroup = async (column: string) => {
    if (!ref || !datasetId) return;
    setGroupBy(column);
    const req = ++latest.current;
    try {
      const s = await getReliabilitySummary(datasetId, ref.embedding, { dims: ref.dims, groupBy: column });
      if (req === latest.current) setShown({ datasetId, summary: s });
    } catch (e) {
      if (req === latest.current) setError(`Plot reliability: ${errorText(e)}`);
    }
  };

  const toggle = () => {
    if (ref) {
      clearOverlay();
      return;
    }
    void load();
  };

  if (!datasetId) return null;
  return (
    <>
      <button
        onClick={toggle}
        disabled={!ready || busy}
        aria-pressed={!!ref}
        title={
          ready
            ? "Colour each cell by how many of its graph neighbours (the kNN graph the clustering used) the plot keeps around it, on a fixed 0-1 scale; a low value means its position should not be read closely"
            : "No embedding to score"
        }
        className={`flex items-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
          ref ? "bg-blue-50 text-blue-700 ring-1 ring-blue-300" : "text-slate-600 hover:bg-slate-100"
        }`}
      >
        {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Waypoints className="h-3.5 w-3.5" />}
        Plot reliability
      </button>
      {error && !ref && (
        <span role="alert" title={error} className="max-w-[240px] truncate text-[11px] text-red-600">
          {error}
        </span>
      )}
      {ref &&
        createPortal(
          <EmbeddingReliabilityPanel
            summary={matches ? summary : null}
            error={error}
            busy={busy}
            onGroupBy={regroup}
            onClose={clearOverlay}
          />,
          document.body,
        )}
    </>
  );
}

/** The floating table: what the score means, overall numbers, per group. */
export function EmbeddingReliabilityPanel({
  summary,
  error,
  busy,
  onGroupBy,
  onClose,
}: {
  summary: ReliabilitySummary | null;
  error: string | null;
  busy: boolean;
  onGroupBy: (column: string) => void;
  onClose: () => void;
}) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const highlighted = useSelectionStore((s) => s.highlightedGroup);
  const [collapsed, setCollapsed] = useState(false);
  const [sort, setSort] = useState<{ key: GroupSortKey; desc: boolean } | null>(null);
  const [rowError, setRowError] = useState<string | null>(null);
  const codes = useRef<{ datasetId: string; column: string; codes: ArrayLike<number> } | null>(null);

  const groupColumns = useMemo(
    () =>
      (dataset?.obs_columns ?? []).filter(
        (c) => (c.dtype === "category" || c.dtype === "object" || c.dtype === "bool") && (c.values?.length ?? 0) > 0,
      ),
    [dataset],
  );
  const rows = useMemo(
    () => (summary ? sortGroups(summary.groups, sort?.key ?? null, sort?.desc ?? false) : []),
    [summary, sort],
  );

  const column = summary?.group_by ?? null;
  const categories = column ? dataset?.obs_columns.find((c) => c.name === column)?.values : undefined;

  // A row click selects the group's cells and highlights it, as a legend click does.
  const highlight = async (group: string) => {
    if (!column || !datasetId || !summary) return;
    setRowError(null);
    const sel = useSelectionStore.getState();
    if (sel.highlightedGroup?.column === column && sel.highlightedGroup.value === group) {
      sel.setHighlight(null);
      sel.clearSelection();
      return;
    }
    if (!categories) {
      setRowError(`Can't select by ${prettyObsLabel(column)}: it has too many groups to list.`);
      return;
    }
    try {
      let c = codes.current;
      if (!c || c.datasetId !== datasetId || c.column !== column) {
        const buffer = await fetchEmbeddingBinary(datasetId, summary.embedding, column);
        const got = decodeArrowBuffer(buffer)[column];
        if (!got) throw new Error("no labels came back");
        c = { datasetId, column, codes: got };
        codes.current = c;
      }
      sel.setSelection(groupCells(c.codes, categories, group));
      sel.setHighlight({ column, value: group });
    } catch (e) {
      setRowError(`Could not select ${group}: ${errorText(e)}`);
    }
  };

  const header = (key: GroupSortKey, label: string, align: "left" | "right" = "right") => {
    const active = sort?.key === key;
    return (
      <th className={`px-1.5 py-1 font-medium ${align === "right" ? "text-right" : "text-left"}`} title={COLUMN_TIPS[key]}>
        <button
          type="button"
          onClick={() => setSort({ key, desc: active ? !sort!.desc : key === "share_low" || key === "n" })}
          className={`inline-flex items-center gap-0.5 hover:text-slate-800 ${active ? "text-slate-800" : ""}`}
        >
          {label}
          {active && (sort!.desc ? <ChevronDown className="h-3 w-3" /> : <ChevronUp className="h-3 w-3" />)}
        </button>
      </th>
    );
  };

  const plot = summary
    ? `${prettyEmbeddingLabel(summary.embedding)}${summary.n_dims === 3 ? " 3D" : ""}${
        /pca/i.test(summary.embedding) ? `, PC${summary.dims[0]! + 1} vs PC${summary.dims[1]! + 1}` : ""
      }`
    : "";

  return (
    <div
      role="dialog"
      aria-label="Plot reliability"
      className="fixed right-4 top-20 z-40 flex max-h-[calc(100vh-6rem)] w-[24rem] max-w-[calc(100vw-2rem)] flex-col gap-2 overflow-y-auto rounded-xl border border-slate-200 bg-white p-3 text-xs shadow-xl"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <h3 className="text-sm font-semibold text-slate-800">Plot reliability{plot ? `: ${plot}` : ""}</h3>
          {busy && <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-500" aria-label="Loading" />}
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setCollapsed((c) => !c)}
            aria-label={collapsed ? "Show the table" : "Hide the table"}
            className="rounded p-0.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          >
            {collapsed ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
          </button>
          <button
            type="button"
            onClick={onClose}
            aria-label="Turn plot reliability off"
            title="Turn plot reliability off (clears the colouring)"
            className="rounded p-0.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      {error && (
        <p role="alert" className="text-[11px] text-red-600">
          {error}
        </p>
      )}
      {!summary && !error && <p className="text-[11px] text-slate-400">Loading…</p>}

      {summary && !collapsed && (
        <>
          <Caption testId="reliability-caption">{reliabilityCaption(summary)}</Caption>
          <dl className="grid grid-cols-3 gap-2 rounded-lg bg-slate-50 px-2 py-1.5">
            <div title="Median over the scored cells; IQR = interquartile range">
              <dt className="text-[10px] text-slate-500">Median (IQR)</dt>
              <dd className="font-semibold tabular-nums text-slate-800">
                {fmtScore(summary.overall.median)}{" "}
                <span className="font-normal text-slate-500">
                  ({fmtScore(summary.overall.q1)}–{fmtScore(summary.overall.q3)})
                </span>
              </dd>
            </div>
            <div title={`k / (n - 1) = ${summary.k} / ${(summary.n_cells - 1).toLocaleString()}: the score when a cell's plot neighbours are a random draw`}>
              <dt className="text-[10px] text-slate-500">Chance</dt>
              <dd className="font-semibold tabular-nums text-slate-800">{fmtScore(summary.chance)}</dd>
            </div>
            <div title={`Mean score after one seeded shuffle of the plot coordinates (seed ${summary.shuffled.seed})`}>
              <dt className="text-[10px] text-slate-500">Random placement</dt>
              <dd className="font-semibold tabular-nums text-slate-800">{fmtScore(summary.shuffled.mean)}</dd>
            </div>
          </dl>
          {summary.notes.length > 0 && (
            <ul className="list-disc space-y-0.5 pl-4 text-[11px] text-slate-500">
              {summary.notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          )}

          <div className="flex items-center gap-1.5">
            <label htmlFor="reliability-group-by" className="text-[11px] font-medium text-slate-500">
              By
            </label>
            <select
              id="reliability-group-by"
              value={summary.group_by ?? ""}
              onChange={(e) => onGroupBy(e.target.value)}
              className="max-w-[200px] rounded border border-slate-300 bg-white px-1.5 py-0.5 text-[11px] text-slate-700 focus:border-blue-400 focus:outline-none"
            >
              {!summary.group_by && <option value="">None</option>}
              {groupColumns.map((c) => (
                <option key={c.name} value={c.name}>
                  {prettyObsLabel(c.name)}
                </option>
              ))}
            </select>
            {sort && (
              <button type="button" onClick={() => setSort(null)} className="text-[11px] text-primary hover:underline">
                Worst first
              </button>
            )}
            <InfoTip label="How plot reliability is computed" align="right" width={300}>
              {summary.method}
            </InfoTip>
          </div>

          {rows.length > 0 ? (
            <table className="w-full border-collapse text-[11px]">
              <thead className="text-slate-500">
                <tr className="border-b border-slate-200">
                  {header("group", "Group", "left")}
                  {header("n", "Cells")}
                  {header("median", "Median")}
                  {header("share_low", "Low")}
                </tr>
              </thead>
              <tbody>
                {rows.map((r: ReliabilityGroup) => {
                  const on = highlighted?.column === column && highlighted.value === r.group;
                  return (
                    <tr
                      key={r.group}
                      onClick={() => void highlight(r.group)}
                      aria-selected={on}
                      className={`cursor-pointer border-b border-slate-100 hover:bg-slate-50 ${on ? "bg-orange-50" : ""}`}
                      title="Select and highlight this group's cells"
                    >
                      <td className="max-w-[10rem] truncate px-1.5 py-0.5 text-slate-700">{r.group}</td>
                      <td className="px-1.5 py-0.5 text-right tabular-nums text-slate-600">{r.n.toLocaleString()}</td>
                      <td
                        className="px-1.5 py-0.5 text-right tabular-nums text-slate-800"
                        title={`IQR ${fmtScore(r.q1)} to ${fmtScore(r.q3)}; mean ${fmtScore(r.mean)}`}
                      >
                        {fmtScore(r.median)}
                      </td>
                      <td className="px-1.5 py-0.5 text-right tabular-nums text-slate-600">{fmtShare(r.share_low)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : (
            <p className="text-[11px] text-slate-400">No group column to summarise by.</p>
          )}
          {rowError && (
            <p role="alert" className="text-[11px] text-red-600">
              {rowError}
            </p>
          )}
          {rows.length > 0 && (
            <Caption n={{ value: rows.length, unit: "groups" }}>
              {sort ? "Sorted as chosen." : "Worst first (lowest median)."} Click a row to select its cells. Small
              groups score higher by chance alone (fewer cells to land among), so compare groups of similar size.
            </Caption>
          )}
        </>
      )}
    </div>
  );
}
