import { useEffect, useMemo, useState } from "react";
import { AlertCircle, Loader2 } from "lucide-react";
import type { Data, Layout } from "plotly.js";
import Plot from "@/components/plots/Plot";
import { plotConfig } from "@/lib/plotConfig";
import { plotlyColorscale } from "@/lib/colors";
import { rgbCss } from "@/lib/categoryColors";
import { useCategoryColors } from "@/hooks/useCategoryColors";
import { useSettingsStore } from "@/stores/settingsStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { useObsRevision } from "@/stores/obsRevisionStore";
import { fetchHeatmapCells } from "@/api/expression";
import {
  CELLS_PER_GROUP,
  DEFAULT_CELLS_PER_GROUP,
  HEATMAP_MAX_GENES,
  cellHeatmapCaption,
  cellHeatmapFigure,
  cellScaleCaption,
  type CellScale,
  type HeatmapCells,
} from "@/lib/cellHeatmap";
import { pickMarkerGenes, type MarkerRow } from "@/lib/markerPlot";

const selectClass = "rounded border border-slate-300 bg-white px-1.5 py-0.5 text-xs text-slate-700";

/** The server's reason ("API error 400: {detail}") as a sentence. */
function errorText(err: unknown): string {
  const msg = err instanceof Error ? err.message.replace(/^API error \d+: /, "") : String(err);
  try {
    return String(JSON.parse(msg).detail ?? msg);
  } catch {
    return msg;
  }
}

/**
 * The marker heatmap's "Per cell" view (C034): the same genes as the group
 * means heatmap (top markers per group, or the user's), drawn for a seeded,
 * stratified sample of cells grouped in the column's stored order.
 */
export function CellHeatmap({
  datasetId,
  groupby,
  groups,
  markers,
  perGroup,
  genes: chosenGenes,
  layer,
}: {
  datasetId: string;
  groupby: string;
  groups: string[];
  markers: Record<string, MarkerRow[]>;
  /** Top markers per group (when no genes are chosen) */
  perGroup: number;
  genes?: string[];
  layer?: string;
}) {
  const colormapName = useSettingsStore((s) => s.continuousColormap);
  const divergingName = useSettingsStore((s) => s.divergingColormap);
  const layers = useDatasetStore((s) => s.currentDataset?.expression_layers);
  const colorOf = useCategoryColors(groupby);
  const revision = useObsRevision(datasetId, groupby); // a saved order or hand labels move cells
  const custom = (chosenGenes?.length ?? 0) > 0;
  const allGenes = useMemo(
    () => (custom ? chosenGenes! : pickMarkerGenes(groups, markers, perGroup).map((p) => p.gene)),
    [custom, chosenGenes, groups, markers, perGroup],
  );
  const genes = useMemo(() => allGenes.slice(0, HEATMAP_MAX_GENES), [allGenes]);
  const [cellsPerGroup, setCellsPerGroup] = useState<number>(DEFAULT_CELLS_PER_GROUP);
  const [scale, setScale] = useState<CellScale>("zscore");
  const [data, setData] = useState<HeatmapCells | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (genes.length === 0 || !groupby) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchHeatmapCells(datasetId, { genes, groupby, perGroup: cellsPerGroup, scale, layer })
      .then((d) => { if (!cancelled) setData(d); })
      .catch((e) => { if (!cancelled) setError(errorText(e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [datasetId, groupby, genes, cellsPerGroup, scale, layer, revision]);

  const figure = useMemo(() => {
    if (!data) return null;
    const colours = data.meta.groups.map((g) => rgbCss(colorOf(g)));
    return cellHeatmapFigure(data, colours, plotlyColorscale(colormapName), plotlyColorscale(divergingName));
  }, [data, colorOf, colormapName, divergingName]);

  if (genes.length === 0) {
    return <p className="py-8 text-center text-sm text-slate-400">No up-regulated marker genes to plot.</p>;
  }

  const controls = (
    <div className="mb-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate-600">
      <label htmlFor="cell-heatmap-per-group" className="font-medium">Cells per group:</label>
      <select
        id="cell-heatmap-per-group"
        value={cellsPerGroup}
        onChange={(e) => setCellsPerGroup(Number(e.target.value))}
        className={selectClass}
      >
        {CELLS_PER_GROUP.map((n) => (
          <option key={n} value={n}>{n}</option>
        ))}
      </select>
      <label htmlFor="cell-heatmap-scale" className="ml-2 font-medium">Colour:</label>
      <select
        id="cell-heatmap-scale"
        value={scale}
        onChange={(e) => setScale(e.target.value as CellScale)}
        className={selectClass}
      >
        <option value="zscore">z-score per gene</option>
        <option value="none">expression (fixed range)</option>
      </select>
      {loading && data && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
    </div>
  );

  if (error) {
    const noGenes = error.startsWith("None of the genes");
    return (
      <div>
        {controls}
        <div
          className={`flex items-center gap-2 rounded-lg border px-4 py-3 text-sm ${
            noGenes ? "border-amber-200 bg-amber-50 text-amber-800" : "border-red-200 bg-red-50 text-red-700"
          }`}
        >
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          {noGenes ? `${error} (names are case-sensitive)` : error}
        </div>
      </div>
    );
  }
  if (!data || !figure) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
        <span className="ml-3 text-sm text-slate-500">Sampling cells...</span>
      </div>
    );
  }

  const meta = data.meta;
  const layerLabel = layers?.find((l) => l.key === meta.layer)?.label ?? meta.layer;
  const left = allGenes.length - genes.length;
  return (
    <div>
      {controls}
      <Plot
        data={figure.data as Data[]}
        layout={figure.layout as Partial<Layout>}
        config={plotConfig("markers_heatmap_cells")}
        useResizeHandler
        style={{ width: "100%" }}
      />
      {meta.missing_genes.length > 0 && (
        <p className="mt-1 text-xs text-amber-700">
          Not in this dataset&apos;s {layerLabel} layer, so not plotted (names are case-sensitive):{" "}
          {meta.missing_genes.join(", ")}.
        </p>
      )}
      {left > 0 && (
        <p className="mt-1 text-xs text-amber-700">
          The first {HEATMAP_MAX_GENES} genes are drawn per cell; {left} more are in the group means heatmap.
        </p>
      )}
      <p className="mt-1 text-xs text-slate-500" data-testid="cell-heatmap-caption">
        {custom
          ? `Your ${meta.genes.length} gene${meta.genes.length === 1 ? "" : "s"} in single cells of each ${groupby} group.`
          : `Top ${perGroup} up-regulated markers of each ${groupby} group, in single cells.`}{" "}
        {cellHeatmapCaption(meta)} {cellScaleCaption(meta.scale, meta.value_range)} Layer: {layerLabel}.
      </p>
    </div>
  );
}
