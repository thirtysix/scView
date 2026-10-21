import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Layers, Loader2, X } from "lucide-react";
import {
  getSweepInfo,
  keepResolution,
  runSweep,
  type ClusterTree,
  type SweepInfo,
  type SweepRow,
  type SweepStart,
} from "@/api/clustering";
import { ClusterTreePlot } from "@/components/unified/ClusterTreePlot";
import { MergeCheck } from "@/components/unified/MergeCheck";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { prettyEmbeddingLabel } from "@/lib/formatting";
import { bestSilhouette, formatResolutions, parseResolutions } from "@/lib/resolutionSweep";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";

/**
 * "Resolutions" in the Unified View toolbar: a floating panel that clusters the
 * dataset's neighbour graph at several resolutions without changing the dataset,
 * compares them, draws the clustering tree across them, and keeps the chosen
 * one as a clustering column.
 */
export function ResolutionSweepControl() {
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const [open, setOpen] = useState(false);
  if (!datasetId) return null;
  return (
    <>
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        title="Try several clustering resolutions and keep one"
        className={`flex items-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors ${
          open ? "bg-blue-50 text-blue-700 ring-1 ring-blue-300" : "text-slate-600 hover:bg-slate-100"
        }`}
      >
        <Layers className="h-3.5 w-3.5" />
        Resolutions
      </button>
      {open &&
        createPortal(
          <ResolutionSweepPanel key={datasetId} datasetId={datasetId} onClose={() => setOpen(false)} />,
          document.body,
        )}
    </>
  );
}

const METHOD_LABEL = { leiden: "Leiden", louvain: "Louvain" } as const;

export function ResolutionSweepPanel({ datasetId, onClose }: { datasetId: string; onClose: () => void }) {
  const refresh = useDatasetRefresh();
  const [info, setInfo] = useState<SweepInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [text, setText] = useState("");
  const [start, setStart] = useState<SweepStart | null>(null);
  const [rows, setRows] = useState<SweepRow[]>([]);
  const [tree, setTree] = useState<ClusterTree | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [keeping, setKeeping] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    getSweepInfo(datasetId)
      .then((i) => {
        if (cancelled) return;
        setInfo(i);
        setText(formatResolutions(i.default_resolutions));
      })
      .catch((e) => {
        if (!cancelled) setInfoError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
      abortRef.current?.abort(); // closing the panel stops a running sweep
    };
  }, [datasetId]);

  const parsed = info
    ? parseResolutions(text, { min: info.min_resolution, max: info.max_resolution, maxCount: info.max_resolutions })
    : { values: [], error: null };

  const run = useCallback(async () => {
    if (!info || parsed.error || running) return;
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setRunning(true);
    setError(null);
    setRows([]);
    setTree(null);
    setSelected(null);
    setStart(null);
    try {
      await runSweep(
        datasetId,
        { resolutions: parsed.values, method: info.method },
        {
          onStart: setStart,
          onResult: (r) => setRows((prev) => [...prev, r]),
          onTree: setTree,
          signal: ctrl.signal,
        },
      );
    } catch (e) {
      if (!ctrl.signal.aborted) setError(e instanceof Error ? e.message : "The sweep failed.");
    } finally {
      if (abortRef.current === ctrl) abortRef.current = null;
      setRunning(false);
    }
  }, [datasetId, info, parsed.error, parsed.values, running]);

  const show = (column: string) => useSettingsStore.getState().setColorBy(column);

  const keep = async (row: SweepRow) => {
    setKeeping(row.resolution);
    setError(null);
    try {
      const res = await keepResolution(datasetId, row.resolution, start?.method ?? info?.method);
      if (res.created) await refresh(datasetId);
      setRows((prev) => prev.map((r) => (r.column === res.column ? { ...r, exists: true } : r)));
      show(res.column);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not keep that resolution.");
    } finally {
      setKeeping(null);
    }
  };

  const best = bestSilhouette(rows);
  const total = start?.resolutions.length ?? parsed.values.length;
  const method = METHOD_LABEL[start?.method ?? info?.method ?? "leiden"];

  return (
    <div
      role="dialog"
      aria-label="Try clustering resolutions"
      className="fixed right-4 top-20 z-50 flex max-h-[calc(100vh-6rem)] w-[36rem] max-w-[calc(100vw-2rem)] flex-col gap-2 overflow-y-auto rounded-xl border border-slate-200 bg-white p-4 text-xs shadow-xl"
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="text-sm font-semibold text-slate-800">Try clustering resolutions</div>
          <p className="mt-0.5 text-[11px] text-slate-500">
            {method} on the dataset&rsquo;s neighbour graph at each resolution. Nothing is saved until you keep one.
          </p>
        </div>
        <button onClick={onClose} aria-label="Close" className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600">
          <X className="h-4 w-4" />
        </button>
      </div>

      {infoError && <p role="alert" className="rounded bg-red-50 px-2 py-1.5 text-red-700">{infoError}</p>}
      {info && !info.available && (
        <p role="alert" className="rounded bg-amber-50 px-2 py-1.5 text-amber-800">{info.reason}</p>
      )}
      {!info && !infoError && (
        <div className="flex items-center gap-1.5 text-slate-400">
          <Loader2 className="h-3.5 w-3.5 animate-spin" /> Checking the dataset…
        </div>
      )}

      {info?.available && (
        <>
          <form
            className="flex items-end gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              void run();
            }}
          >
            <label className="flex-1 space-y-1">
              <span className="text-[11px] font-medium text-slate-500">
                Resolutions (up to {info.max_resolutions})
              </span>
              <input
                value={text}
                onChange={(e) => setText(e.target.value)}
                disabled={running}
                className="w-full rounded border border-slate-300 px-2 py-1 text-xs text-slate-700 focus:border-blue-400 focus:outline-none"
              />
            </label>
            {running ? (
              <button
                type="button"
                onClick={() => abortRef.current?.abort()}
                className="rounded border border-slate-300 px-3 py-1 text-xs text-slate-600 hover:bg-slate-50"
              >
                Stop
              </button>
            ) : (
              <button
                type="submit"
                disabled={!!parsed.error}
                className="rounded bg-blue-500 px-3 py-1 text-xs font-medium text-white hover:bg-blue-600 disabled:opacity-50"
              >
                Run
              </button>
            )}
          </form>
          {parsed.error && <p className="text-[11px] text-amber-700">{parsed.error}</p>}
          {running && (
            <div className="flex items-center gap-1.5 text-slate-500">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Clustering {Math.min(rows.length + 1, total)} of {total}…
            </div>
          )}
        </>
      )}

      {rows.length > 0 && (
        <table className="w-full border-collapse text-[11px]">
          <thead>
            <tr className="border-b border-slate-200 text-left text-slate-500">
              <th className="py-1 pr-2 font-medium">Res.</th>
              <th className="py-1 pr-2 font-medium">Clusters</th>
              <th className="py-1 pr-2 font-medium">Sizes</th>
              <th className="py-1 pr-2 font-medium" title="Mean silhouette in the graph's PCA space: higher means better-separated clusters">
                Silhouette
              </th>
              <th className="py-1 pr-2 font-medium" title="Adjusted Rand index to the row above: 1 means the same partition">
                ARI ↑
              </th>
              <th className="py-1" />
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr
                key={r.resolution}
                onClick={() => setSelected(r.resolution)}
                aria-selected={selected === r.resolution}
                title="Pick this resolution in the tree"
                className={`cursor-pointer border-b border-slate-100 ${selected === r.resolution ? "bg-blue-50" : "hover:bg-slate-50"}`}
              >
                <td className="py-1 pr-2 font-medium tabular-nums text-slate-700">{formatResolutions([r.resolution])}</td>
                <td className="py-1 pr-2 tabular-nums text-slate-700">{r.n_clusters}</td>
                <td className="py-1 pr-2">
                  <SizeBars sizes={r.sizes} />
                </td>
                <td className={`py-1 pr-2 tabular-nums ${i === best ? "font-semibold text-emerald-700" : "text-slate-600"}`}>
                  {r.silhouette == null ? "—" : r.silhouette.toFixed(3)}
                </td>
                <td className="py-1 pr-2 tabular-nums text-slate-600">
                  {r.ari_previous == null ? "—" : r.ari_previous.toFixed(2)}
                </td>
                <td className="py-1 text-right">
                  {r.exists ? (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        show(r.column);
                      }}
                      title={`Saved as ${r.column}: colour the plot by it`}
                      className="rounded px-1.5 py-0.5 text-blue-600 hover:bg-blue-50"
                    >
                      Show
                    </button>
                  ) : (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        void keep(r);
                      }}
                      disabled={keeping !== null}
                      title={`Save these clusters as ${r.column}`}
                      className="flex items-center gap-1 rounded border border-slate-300 px-1.5 py-0.5 text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                    >
                      {keeping === r.resolution && <Loader2 className="h-3 w-3 animate-spin" />}
                      Keep this one
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {tree && tree.nodes.length > 0 && (
        <div className="space-y-1">
          <div className="text-[11px] font-medium text-slate-600">Clustering tree</div>
          <ClusterTreePlot tree={tree} selected={selected} onSelect={setSelected} />
          <p className="text-[10px] leading-snug text-slate-400">
            Each row is a resolution; an edge carries the cells a cluster shares with one at the next resolution
            (thicker: more cells; darker: a bigger share of the child; under {Math.round(tree.prop_filter * 100)}% not
            drawn). Colour: SC3 stability, high when a cluster&rsquo;s cells stay together at the other resolutions.{" "}
            {tree.n_mixed > 0
              ? `${tree.n_mixed} cluster${tree.n_mixed === 1 ? " draws" : "s draw"} at least ${Math.round(tree.mixed_prop * 100)}% of ${tree.n_mixed === 1 ? "its" : "their"} cells from each of two or more parents (red outline): cells move between branches there, a sign the resolution is past what the data support.`
              : "No cluster draws its cells from two parents: the tree only splits."}
          </p>
        </div>
      )}

      {error && <p role="alert" className="rounded bg-red-50 px-2 py-1.5 text-red-700">{error}</p>}

      {start && (
        <p className="text-[10px] leading-snug text-slate-400">
          {start.representation && start.silhouette_cells
            ? `Silhouette on ${start.silhouette_cells.toLocaleString()} of ${start.n_cells.toLocaleString()} cells (same cells for every row) in ${prettyEmbeddingLabel(start.representation)}. `
            : "No PCA to score silhouettes in. "}
          ARI ↑ compares each row with the one above. &ldquo;Keep this one&rdquo; adds a clustering column; the
          dataset&rsquo;s active clustering stays as it is.
        </p>
      )}

      {info?.available && (
        <MergeCheck datasetId={datasetId} rows={running ? [] : rows} method={start?.method ?? info.method} selected={selected} />
      )}
    </div>
  );
}

/** Cluster sizes as tiny bars, largest first, with the range as text. */
function SizeBars({ sizes }: { sizes: number[] }) {
  const max = sizes[0] ?? 1;
  const shown = sizes.slice(0, 30);
  return (
    <div className="flex items-center gap-1.5" title={sizes.map((s) => s.toLocaleString()).join(", ")}>
      <div className="flex h-4 items-end gap-px" aria-hidden>
        {shown.map((s, i) => (
          <div key={i} className="w-1 rounded-sm bg-blue-400" style={{ height: `${Math.max(8, (s / max) * 100)}%` }} />
        ))}
        {sizes.length > shown.length && <span className="text-[9px] text-slate-400">…</span>}
      </div>
      <span className="whitespace-nowrap tabular-nums text-slate-500">
        {max.toLocaleString()}–{(sizes[sizes.length - 1] ?? 0).toLocaleString()}
      </span>
    </div>
  );
}
