import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { fetchEmbeddingBinary } from "@/api/embeddings";
import { getWnn, getWnnWeights, type WnnInfo, type WnnWeights } from "@/api/wnn";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import Plot from "@/components/plots/Plot";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { embeddingReady } from "@/hooks/useEmbeddingReady";
import { decodeArrowBuffer } from "@/lib/arrow";
import { mapCategoryToColor } from "@/lib/colors";
import { errorMessage } from "@/lib/errorMessage";
import { prettyObsLabel } from "@/lib/formatting";
import { drawThumbnail } from "@/lib/layoutStability";
import { plotConfig } from "@/lib/plotConfig";
import { useScatterBackground } from "@/lib/theme";
import {
  RNA_WEIGHT_LABEL,
  WNN_CLUSTERS,
  WNN_LAYOUTS,
  extremes,
  planText,
  sampleCells,
  stageText,
  thumbOf,
} from "@/lib/wnn";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useWnnRunStore } from "@/stores/wnnRunStore";

const THUMB_PX = 120;
type RGB = readonly [number, number, number];

/**
 * "Joint RNA + protein (WNN)" (backend core/wnn.py): Seurat's weighted nearest
 * neighbours on a CITE-seq dataset. Run it, then see the RNA, protein and joint
 * layouts side by side in the current colouring, colour the cells by their RNA
 * weight (fixed 0-1 scale, violin by group) or by the WNN clusters.
 */
export function WnnCard({ datasetId, groupByColumn }: { datasetId: string; groupByColumn: string }) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const colorBy = useSettingsStore((s) => s.colorBy);
  const setColorBy = useSettingsStore((s) => s.setColorBy);
  const run = useWnnRunStore();
  const live = run.datasetId === datasetId;
  const refreshDataset = useDatasetRefresh();
  const [info, setInfo] = useState<WnnInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [weights, setWeights] = useState<WnnWeights | null>(null);
  const [busy, setBusy] = useState(false);

  const groupby = useMemo(() => {
    const col = dataset?.obs_columns.find((c) => c.name === groupByColumn);
    return col && col.n_unique <= 100 ? groupByColumn : "";
  }, [dataset, groupByColumn]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getWnn(datasetId, groupby || undefined)
      .then((i) => !cancelled && setInfo(i))
      .catch((e) => !cancelled && setError(errorMessage(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [datasetId, groupby, run.version]);

  // A saved run's columns and embeddings the dataset's listing does not show yet (the
  // run just finished, here or while the tab was closed): refresh it, once per run.
  const listed = !!dataset?.obs_columns.some((c) => c.name === WNN_CLUSTERS) &&
    !!dataset?.available_embeddings?.includes("X_umap_wnn");
  const refreshed = useRef<string | null>(null);
  useEffect(() => {
    const id = info?.available && !info.stale ? info.run_id ?? null : null;
    if (!id || listed || refreshed.current === id) return;
    refreshed.current = id;
    void refreshDataset(datasetId, WNN_CLUSTERS).catch(() => undefined);
  }, [info, listed, datasetId, refreshDataset]);

  // The RNA weight by group, for the box plot (and the violin under the scatter).
  const available = !!info?.available;
  useEffect(() => {
    if (!available || !groupby) {
      setWeights(null);
      return;
    }
    let cancelled = false;
    getWnnWeights(datasetId, groupby)
      .then((w) => !cancelled && setWeights(w))
      .catch(() => !cancelled && setWeights(null));
    return () => {
      cancelled = true;
    };
  }, [datasetId, groupby, available, info?.run_id]);

  const showWeight = useCallback(async () => {
    setBusy(true);
    try {
      const w = weights && weights.column === (groupby || null) ? weights : await getWnnWeights(datasetId, groupby || undefined);
      const ui = useUnifiedViewStore.getState();
      ui.setOverlayValues(new Float32Array(w.values));
      ui.setOverlayLabel(RNA_WEIGHT_LABEL);
      ui.setScatterOverlay({ type: "score", name: RNA_WEIGHT_LABEL, share: true });
      ui.setOverlayDatasetId(datasetId);
      if (w.violin) {
        ui.setViolinData(w.violin);
        ui.setViolinTitle(RNA_WEIGHT_LABEL);
        ui.openBottomPanel();
      }
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }, [datasetId, groupby, weights]);

  const hasClusters = !!dataset?.obs_columns.some((c) => c.name === WNN_CLUSTERS);
  const plan = info?.runnable.plan ?? null;
  const saved = info?.plan ?? null;
  const running = (live && run.running) || (!!info?.running && !run.running);
  const rows = info?.by_group?.rows ?? [];
  const ends = extremes(rows);

  return (
    <section className="space-y-2 rounded-lg border border-slate-200 p-2" aria-label="Joint RNA + protein (WNN)">
      <h4 className="flex items-center gap-1 text-xs font-semibold text-slate-600">
        <span>Joint RNA + protein (WNN)</span>
        <InfoTip label="About WNN" width={300}>
          Weighted nearest neighbours (Hao et al. 2021, Cell): for each cell, how well its RNA neighbours and its
          protein neighbours predict it decides how much each modality counts; one neighbour graph is then built from
          both. The same method as Seurat&rsquo;s FindMultiModalNeighbors.
        </InfoTip>
      </h4>

      {loading && !info && (
        <p className="flex items-center gap-1.5 text-[11px] text-slate-400">
          <Loader2 className="h-3 w-3 animate-spin" /> Loading…
        </p>
      )}
      <RequestError message={error} />

      {info && !info.runnable.ok && !available && (
        <p className="text-[11px] leading-snug text-slate-500">{info.runnable.reason}</p>
      )}

      {plan && (
        <p className="text-[11px] leading-snug text-slate-500" data-testid="wnn-plan">
          {planText(plan)}
        </p>
      )}
      {plan?.warnings.map((w) => (
        <p key={w} className="text-[11px] leading-snug text-amber-700">{w}</p>
      ))}
      {plan?.notes.map((n) => (
        <p key={n} className="text-[11px] leading-snug text-slate-500">{n}</p>
      ))}

      {running ? (
        <div className="space-y-1" data-testid="wnn-progress">
          <p className="flex items-center gap-1.5 text-[11px] text-slate-600">
            <Loader2 className="h-3 w-3 animate-spin" />
            {live ? stageText(run.stage) : "WNN is running on this dataset."}
          </p>
          {live && run.stage && (
            <div className="h-1 w-full rounded bg-slate-100">
              <div
                className="h-1 rounded bg-primary"
                style={{ width: `${Math.round(((run.stage.index - 1) / run.stage.total) * 100)}%` }}
              />
            </div>
          )}
          {live && (
            <button
              type="button"
              onClick={() => run.cancel()}
              className="rounded border border-slate-300 px-2 py-0.5 text-[11px] text-slate-600 hover:border-slate-400"
            >
              Cancel
            </button>
          )}
        </div>
      ) : (
        info?.runnable.ok && (
          <button
            type="button"
            onClick={() => void run.run(datasetId, {})}
            className="rounded bg-primary px-2.5 py-1 text-xs font-medium text-white hover:bg-primary/90"
          >
            {available ? "Run WNN again" : "Run WNN"}
          </button>
        )
      )}
      {live && run.error && <RequestError message={run.error} />}
      {live && run.warnings.map((w) => (
        <p key={w} className="text-[11px] text-amber-700">{w}</p>
      ))}

      {available && info && (
        <div className="space-y-2">
          {info.stale && (
            <p className="text-[11px] leading-snug text-amber-700" data-testid="wnn-stale">{info.stale_reason}</p>
          )}
          <p className="text-[11px] leading-snug text-slate-500">
            Saved run: mean RNA weight {info.mean_rna_weight?.toFixed(2)} over {info.plan?.n_cells?.toLocaleString()}{" "}
            cells, {info.n_clusters} WNN clusters
            {saved ? ` (${saved.n_proteins} proteins, ${saved.d_protein} protein PCs, ${saved.d_rna} RNA PCs)` : ""}.
          </p>
          <WnnLayouts datasetId={datasetId} colorBy={colorBy} groupBy={groupby} />
          <div className="flex flex-wrap gap-1.5">
            <button
              type="button"
              onClick={() => void showWeight()}
              disabled={busy}
              className="rounded border border-slate-300 px-2 py-0.5 text-[11px] text-slate-700 hover:border-primary hover:text-primary disabled:opacity-50"
            >
              Colour by RNA weight
            </button>
            <button
              type="button"
              onClick={() => setColorBy(WNN_CLUSTERS)}
              disabled={!hasClusters}
              title={hasClusters ? "Colour the cells by the clusters of the WNN graph" : "Loading the WNN clusters…"}
              className="rounded border border-slate-300 px-2 py-0.5 text-[11px] text-slate-700 hover:border-primary hover:text-primary disabled:opacity-50"
            >
              Colour by WNN clusters
            </button>
            <InfoTip label="About the RNA weight" width={280}>
              Hao 2021&rsquo;s per-cell modality weight: 0 to 1, RNA and protein weights sum to 1. Drawn on a fixed
              0-1 scale.
            </InfoTip>
          </div>
          {weights?.violin && Object.keys(weights.violin).length > 0 && (
            <WeightBoxes violin={weights.violin} column={groupby} />
          )}
          {groupby && rows.length > 0 && (
            <Caption n={{ value: rows.reduce((a, r) => a + r.n_cells, 0), unit: "cells" }}>
              RNA weight by {prettyObsLabel(groupby)}
              {ends ? `: highest in ${ends.rna}, lowest (protein counts most) in ${ends.protein}` : ""}.
            </Caption>
          )}
          <p className="text-[11px] leading-snug text-slate-500">
            RNA weight is high wherever the antibody panel has no marker for a cell type, so read it against the panel.
            The weights are relative, not quality scores of either modality.
          </p>
        </div>
      )}
      <p className="text-[10px] leading-snug text-slate-400">
        Method: Hao et al. 2021, Cell 184:3573 (weighted nearest neighbours); algorithm as Seurat&rsquo;s
        FindMultiModalNeighbors (Seurat, MIT licence).
      </p>
    </section>
  );
}

function WeightBoxes({ violin, column }: { violin: Record<string, number[]>; column: string }) {
  const groups = Object.keys(violin);
  return (
    <Plot
      data={groups.map((g) => ({
        type: "box" as const,
        name: g,
        x: violin[g],
        orientation: "h" as const,
        boxpoints: false as const,
        marker: { color: "#2563eb" },
        line: { width: 1 },
        hovertemplate: `${g}<br>RNA weight %{x:.2f}<extra></extra>`,
      }))}
      layout={{
        height: Math.max(140, 18 * groups.length + 50),
        margin: { t: 6, r: 10, b: 32, l: 110 },
        xaxis: { range: [0, 1], title: { text: "RNA weight", font: { size: 11, color: "#64748b" } },
                 tickfont: { size: 10, color: "#64748b" }, gridcolor: "#f1f5f9", zeroline: false },
        // Category, every group named: cluster numbers ("0", "1", ...) made it a numeric axis
        // that labelled every other row.
        yaxis: { type: "category" as const, tickmode: "array" as const, tickvals: groups, ticktext: groups,
                 tickfont: { size: 10, color: "#64748b" }, autorange: "reversed" as const, automargin: true },
        showlegend: false,
        paper_bgcolor: "white",
        plot_bgcolor: "white",
        font: { family: "Inter, system-ui, sans-serif" },
      }}
      config={plotConfig(`wnn_rna_weight_by_${column}`)}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}

/** RNA | Protein | WNN layouts, the same sampled cells, coloured by a group column. */
function WnnLayouts({ datasetId, colorBy, groupBy }: { datasetId: string; colorBy: string; groupBy: string }) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const palette = useSettingsStore((s) => s.categoricalPalette);
  const categorical = (name: string) => {
    const c = dataset?.obs_columns.find((x) => x.name === name);
    return !!c && (c.dtype === "category" || c.dtype === "object" || c.dtype === "bool") && c.n_unique <= 100;
  };
  const column = categorical(colorBy) ? colorBy : categorical(groupBy) ? groupBy : "";
  const colInfo = dataset?.obs_columns.find((c) => c.name === column);
  const slots = colInfo?.color_slots;
  const colorOf = useCallback(
    (code: number): RGB => mapCategoryToColor(code, palette, slots).slice(0, 3) as unknown as RGB,
    [palette, slots],
  );
  const cells = useMemo(() => sampleCells(dataset?.n_cells ?? 0), [dataset?.n_cells]);
  const shown = WNN_LAYOUTS.filter((l) => embeddingReady(dataset, datasetId, l.key));
  return (
    <figure className="space-y-1">
      <div className="grid grid-cols-3 gap-1">
        {shown.map((l) => (
          <LayoutThumb key={l.key} datasetId={datasetId} embedding={l.key} title={l.title} column={column}
                       cells={cells} colorOf={colorOf} />
        ))}
      </div>
      <Caption n={{ value: cells.length, unit: "cells", of: dataset?.n_cells ?? cells.length }}>
        UMAPs of the RNA, the proteins and the joint WNN graph
        {column ? `, coloured by ${prettyObsLabel(column)}` : ""}. Pick them in the embedding menu to see them full size.
      </Caption>
    </figure>
  );
}

function LayoutThumb({
  datasetId,
  embedding,
  title,
  column,
  cells,
  colorOf,
}: {
  datasetId: string;
  embedding: string;
  title: string;
  column: string;
  cells: number[];
  colorOf: (code: number) => RGB;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const background = useScatterBackground() === "dark" ? "#1e293b" : "#ffffff";
  const q = useQuery({
    queryKey: ["embedding", datasetId, embedding, column],
    queryFn: () => fetchEmbeddingBinary(datasetId, embedding, column || undefined),
    staleTime: 5 * 60 * 1000,
    gcTime: 60 * 1000,
  });
  const decoded = useMemo(
    () => (q.data ? (decodeArrowBuffer(q.data) as Record<string, ArrayLike<number>>) : null),
    [q.data],
  );
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !decoded) return;
    const thumb = thumbOf(decoded, cells);
    if (!thumb) return;
    const ratio = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
    canvas.width = THUMB_PX * ratio;
    canvas.height = THUMB_PX * ratio;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const codes = column ? decoded[column] ?? null : null;
    drawThumbnail(ctx, { thumb, codes, colorOf, width: THUMB_PX, height: THUMB_PX, background, pointSize: 1.5 });
  }, [decoded, cells, column, colorOf, background]);
  return (
    <div className="flex flex-col items-center gap-0.5 rounded border border-slate-200 p-1" data-testid={`wnn-thumb-${embedding}`}>
      <span className="text-[10px] font-medium text-slate-600">{title}</span>
      <div className="relative" style={{ width: THUMB_PX, height: THUMB_PX }}>
        <canvas ref={ref} style={{ width: THUMB_PX, height: THUMB_PX }} aria-label={`${title} UMAP`} />
        {!decoded && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-50 text-[10px] text-slate-400">
            {q.isError ? "not available" : <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          </div>
        )}
      </div>
    </div>
  );
}
