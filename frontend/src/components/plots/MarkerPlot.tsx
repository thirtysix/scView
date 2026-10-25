import { useEffect, useMemo, useState } from "react";
import { plotlyColorscale } from "@/lib/colors";
import { useSettingsStore } from "@/stores/settingsStore";
import { AlertCircle, Loader2 } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { plotConfig } from "@/lib/plotConfig";
import { apiFetch } from "@/api/client";
import { useDatasetStore } from "@/stores/datasetStore";
import { missingGenes } from "@/lib/geneList";
import { getGroupDistances, type GroupDistances } from "@/api/clustering";
import { columnDendrogram, dendrogramRowOrder, dendrogramSegments, rowDendrogram } from "@/lib/dendrogram";
import { clusterGenes } from "@/lib/geneClustering";
import { discreteColorscale } from "@/lib/cellHeatmap";
import { rgbCss } from "@/lib/categoryColors";
import { useCategoryColors } from "@/hooks/useCategoryColors";
import { useObsRevision } from "@/stores/obsRevisionStore";
import { useSelectionByGroup } from "@/hooks/useSelectionByGroup";
import { selectionName, selectionRow } from "@/lib/linkedTraces";
import { SelectionChip } from "@/components/common/SelectionChip";
import {
  DOT_LEGEND,
  MIN_FRACTION,
  dotSize,
  heatmapBands,
  markerColours,
  pickMarkerGenes,
  type ColourScale,
  type MarkerRow,
  type MeanOf,
} from "@/lib/markerPlot";

export { pickMarkerGenes, scalePerGene } from "@/lib/markerPlot";

interface ByGroup {
  groupby: string;
  layer: string;
  groups: string[];
  genes: string[];
  mean: number[][]; // [group][gene]
  /** Mean over the expressing cells only (absent from older servers) */
  mean_expressing?: number[][];
  fraction: number[][]; // [group][gene]
  /** The layer's fixed colour range, the same for any genes and groups */
  scale_range?: [number, number];
}

const pct = (f: number) => `${Math.round(f * 100)}%`;

/** Fixed dot-size key: the same sizes as the plot's dots, for every plot. */
function DotSizeLegend() {
  const h = dotSize(1) + 4;
  const ring = Math.max(dotSize(MIN_FRACTION / 2), 5);
  let x = 0;
  const items = DOT_LEGEND.map((f) => {
    const d = dotSize(f);
    const cx = x + d / 2;
    x += d + 36; // room for the "100%" label
    return { f, d, cx };
  });
  const ringX = x + ring / 2;
  return (
    <div className="mt-1 flex items-center gap-2 text-[11px] text-slate-500" data-testid="dot-size-legend">
      <span>Cells expressing:</span>
      <svg width={ringX + ring / 2 + 34} height={h} role="img" aria-label="Dot size legend">
        {items.map(({ f, d, cx }) => (
          <g key={f}>
            <circle cx={cx} cy={h / 2} r={d / 2} fill="#94a3b8" stroke="#64748b" strokeWidth={0.5} />
            <text x={cx + d / 2 + 2} y={h / 2 + 3.5} fontSize={10} fill="#64748b">{pct(f)}</text>
          </g>
        ))}
        <circle cx={ringX} cy={h / 2} r={ring / 2} fill="none" stroke="#94a3b8" strokeWidth={1.5} />
        <text x={ringX + ring / 2 + 2} y={h / 2 + 3.5} fontSize={10} fill="#64748b">
          &lt;{pct(MIN_FRACTION)}
        </text>
      </svg>
    </div>
  );
}

const selectClass = "rounded border border-slate-300 bg-white px-1.5 py-0.5 text-xs text-slate-700";
/** The plot's share of the width when the dendrogram is drawn on its right. */
const PLOT_DOMAIN: [number, number] = [0, 0.88];
const TREE_DOMAIN: [number, number] = [0.9, 1];
/** Heights (px) of the marker-group strip and the gene dendrogram above a heatmap. */
const STRIP_PX = 12;
const GENE_TREE_PX = 60;

/** The server's reason ("API error 400: {detail}") as a sentence. */
function errorText(err: unknown): string {
  const msg = err instanceof Error ? err.message.replace(/^API error \d+: /, "") : String(err);
  try {
    return String(JSON.parse(msg).detail ?? msg);
  } catch {
    return msg;
  }
}

export function MarkerPlot({
  datasetId,
  groupby,
  groups,
  markers,
  kind,
  perGroup,
  genes: chosenGenes,
  layer,
}: {
  datasetId: string;
  groupby: string;
  groups: string[];
  markers: Record<string, MarkerRow[]>;
  kind: "dotplot" | "heatmap";
  perGroup: number;
  /** Plot these genes instead of the top markers (empty or absent: top markers). */
  genes?: string[];
  /** Expression layer to read (default: the dataset's), e.g. "protein" */
  layer?: string;
}) {
  const colormapName = useSettingsStore((s) => s.continuousColormap);
  const custom = (chosenGenes?.length ?? 0) > 0;
  const picked = useMemo(
    () =>
      custom
        ? chosenGenes!.map((gene) => ({ gene, group: "" }))
        : pickMarkerGenes(groups, markers, perGroup),
    [custom, chosenGenes, groups, markers, perGroup],
  );
  const [data, setData] = useState<ByGroup | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [meanOf, setMeanOf] = useState<MeanOf>("all");
  const [scale, setScale] = useState<ColourScale>("gene");
  // The user's own top of the fixed range, kept for the layer it was set on.
  const [fixedMax, setFixedMax] = useState<{ layer: string; value: number } | null>(null);
  const layers = useDatasetStore((s) => s.currentDataset?.expression_layers);
  // Order the groups by the clustering of their centroid distances (C043).
  const [byTree, setByTree] = useState(false);
  const revision = useObsRevision(datasetId, groupby); // hand labels move cells between groups
  const treeKey = `${datasetId}\u0000${groupby}\u0000${revision}`;
  const [tree, setTree] = useState<{ key: string; d: GroupDistances | null; error: string | null } | null>(null);
  // Order the heatmap's genes by clustering their profiles across groups (C034).
  const [byGeneTree, setByGeneTree] = useState(false);
  const colorOf = useCategoryColors(groupby);
  // On the mean over all cells of the plotted groups; correlation ignores both
  // the group order and each gene's scale, so the tree is the same for every
  // colour option and group order.
  const geneTree = useMemo(() => {
    if (!byGeneTree || kind !== "heatmap" || !data || data.genes.length < 2) return null;
    const idx = custom
      ? data.groups.map((_, i) => i)
      : groups.map((g) => data.groups.indexOf(g)).filter((i) => i >= 0);
    if (idx.length < 2) return null;
    return clusterGenes(data.genes.map((_, c) => idx.map((i) => data.mean[i]?.[c] ?? 0)));
  }, [byGeneTree, kind, data, custom, groups]);

  useEffect(() => {
    if (!byTree || !groupby || tree?.key === treeKey) return;
    let cancelled = false;
    getGroupDistances(datasetId, groupby)
      .then((d) => { if (!cancelled) setTree({ key: treeKey, d, error: null }); })
      .catch((e) => { if (!cancelled) setTree({ key: treeKey, d: null, error: errorText(e) }); });
    return () => { cancelled = true; };
  }, [byTree, datasetId, groupby, treeKey, tree?.key]);

  // C035: the current selection as an extra row, over the same genes.
  const pickedGenes = useMemo(() => picked.map((p) => p.gene), [picked]);
  const selectionStats = useSelectionByGroup(datasetId, pickedGenes, layer);

  useEffect(() => {
    if (picked.length === 0) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    const qs = new URLSearchParams({ groupby, ...(layer ? { layer } : {}) });
    for (const p of picked) qs.append("genes", p.gene);
    apiFetch<ByGroup>(`/datasets/${datasetId}/expression/by-group?${qs}`)
      .then((d) => { if (!cancelled) setData(d); })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : String(e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [datasetId, groupby, picked, layer, revision]); // revision: a saved category order or label move

  if (picked.length === 0) {
    return <p className="py-8 text-center text-sm text-slate-400">No up-regulated marker genes to plot.</p>;
  }
  if (error && custom && error.includes("404")) {
    return (
      <p className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
        None of these genes are in this dataset (names are case-sensitive):{" "}
        {chosenGenes!.join(", ")}.
      </p>
    );
  }
  if (error) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
        <AlertCircle className="h-4 w-4 flex-shrink-0" />
        {error}
      </div>
    );
  }
  if (loading || !data) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
        <span className="ml-3 text-sm text-slate-500">Computing expression per group...</span>
      </div>
    );
  }

  // Rows in the marker table's group order (chosen genes: the column's own
  // order, as the legend lists it); genes in the order they were picked.
  const baseIdx = custom
    ? data.groups.map((_, i) => i)
    : groups.map((g) => data.groups.indexOf(g)).filter((i) => i >= 0);
  if (baseIdx.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-slate-400">
        These markers were computed for a different column than {groupby}. Choose the column
        under "Compute for" to recompute them.
      </p>
    );
  }
  // Dendrogram order when asked for and fetched; the tree is drawn when it has
  // exactly the plotted groups.
  const treeData = byTree && tree?.key === treeKey ? tree.d : null;
  const baseRows = baseIdx.map((i) => data.groups[i]!);
  const groupRows = treeData ? dendrogramRowOrder(treeData.groups, treeData.order, baseRows) : baseRows;
  const rowIdx = groupRows.map((g) => data.groups.indexOf(g));
  const sameGroups =
    !!treeData && treeData.groups.length === groupRows.length && treeData.groups.every((g) => groupRows.includes(g));
  const lines = sameGroups && treeData?.tree ? dendrogramSegments(treeData.tree, treeData.order) : null;
  const treeLoading = byTree && tree?.key !== treeKey;
  const treeError = byTree && tree?.key === treeKey ? tree.error : null;
  const layerLabel = layers?.find((l) => l.key === data.layer)?.label ?? data.layer;
  const markerOf = new Map(picked.map((p) => [p.gene, p.group]));
  // Genes in the order picked (top markers: in blocks by marker group), or as
  // the gene tree orders them.
  const colIdx = geneTree ? geneTree.order : data.genes.map((_, c) => c);
  const cols = <T,>(row: T[]) => colIdx.map((c) => row[c]!);
  const genes = cols(data.genes);
  const missing = custom ? missingGenes(chosenGenes!, genes) : [];
  // The selection's row goes last (C035), but not under a dendrogram whose leaves are the groups.
  const selRow = lines ? null : selectionRow(selectionStats, data.genes, colIdx);
  const selRowName = selRow && selectionStats ? selectionName(selectionStats.n_cells) : null;
  const rows = selRowName ? [...groupRows, selRowName] : groupRows;
  const mean = rowIdx.map((i) => cols(data.mean[i]!));
  const meanExpressing = data.mean_expressing ? rowIdx.map((i) => cols(data.mean_expressing![i]!)) : null;
  const fraction = rowIdx.map((i) => cols(data.fraction[i]!));
  if (selRow) {
    mean.push(selRow.mean);
    meanExpressing?.push(selRow.meanExpressing);
    fraction.push(selRow.fraction);
  }
  // Bands above a heatmap: which group each top marker came from, then the gene tree.
  const geneLines = geneTree ? dendrogramSegments(geneTree.tree, geneTree.order) : null;
  const showStrip = kind === "heatmap" && !custom;
  const baseHeight = Math.max(260, rows.length * 28 + 150);
  const margin = { l: 120, r: 20, t: showStrip || geneLines ? 28 : 10, b: 110 };
  const bands = heatmapBands(baseHeight - 10 - margin.b, showStrip ? STRIP_PX : 0, geneLines ? GENE_TREE_PX : 0);
  const hasBands = bands.extraPx > 0;
  const dendro = lines ? rowDendrogram(lines, TREE_DOMAIN, hasBands ? bands.main : undefined) : null;
  const xDomain: [number, number] = dendro ? PLOT_DOMAIN : [0, 1];
  const geneDendro = geneLines && bands.tree ? columnDendrogram(geneLines, xDomain, bands.tree) : null;
  const stripGroups = showStrip ? rows.filter((g) => genes.some((gene) => markerOf.get(gene) === g)) : [];
  const strip =
    showStrip && bands.strip
      ? {
          type: "heatmap" as const,
          x: genes,
          y: ["marker of"],
          z: [genes.map((gene) => stripGroups.indexOf(markerOf.get(gene) ?? ""))],
          zmin: -0.5,
          zmax: stripGroups.length - 0.5,
          colorscale: discreteColorscale(stripGroups.map((g) => rgbCss(colorOf(g)))),
          showscale: false,
          text: [genes.map((gene) => `<b>${gene}</b>: top marker of ${markerOf.get(gene) ?? "?"}`)] as unknown as string[],
          hoverinfo: "text" as const,
          xgap: 1,
          yaxis: "y4",
        }
      : null;
  const isProtein = layer === "protein";
  const noun = isProtein ? "protein" : "gene";
  const serverRange: [number, number] = data.scale_range ?? [0, Math.max(1, ...mean.flat())];
  const range: [number, number] = [
    serverRange[0],
    fixedMax && fixedMax.layer === data.layer ? fixedMax.value : serverRange[1],
  ];
  const colours = markerColours({ mean, meanExpressing, fraction }, meanOf, scale, range, noun);
  const hover = rows.map((g, r) =>
    genes.map(
      (gene, c) =>
        `<b>${gene}</b>${custom ? "" : ` (marker of ${markerOf.get(gene) ?? "?"})`}<br>${g === selRowName ? g : `${groupby}: ${g}`}` +
        `<br>mean ${(mean[r]?.[c] ?? 0).toFixed(2)}` +
        (meanExpressing ? `<br>mean in expressing cells ${(meanExpressing[r]?.[c] ?? 0).toFixed(2)}` : "") +
        `<br>${Math.round((fraction[r]?.[c] ?? 0) * 100)}% of cells express it`,
    ),
  );
  const colorbar = {
    title: { text: colours.title },
    thickness: 12,
    ...(hasBands ? { len: bands.main[1], y: 0, yanchor: "bottom" as const } : {}),
  };

  // Dot plot: dots with too few expressing cells are hollow rings without a colour.
  const points = rows.flatMap((g, r) => genes.map((gene, c) => ({ g, gene, r, c })));
  const solid = points.filter(({ r, c }) => (fraction[r]?.[c] ?? 0) >= MIN_FRACTION && colours.z[r]?.[c] != null);
  const hollow = points.filter(({ r, c }) => !((fraction[r]?.[c] ?? 0) >= MIN_FRACTION && colours.z[r]?.[c] != null));
  const traces =
    kind === "heatmap"
      ? [
          {
            type: "heatmap" as const,
            x: genes,
            y: rows,
            z: colours.z,
            zmin: colours.cmin,
            zmax: colours.cmax,
            text: hover as unknown as string[],
            hoverinfo: "text" as const,
            colorscale: plotlyColorscale(colormapName),
            colorbar,
          },
        ]
      : [
          {
            type: "scatter" as const,
            mode: "markers" as const,
            x: solid.map((p) => p.gene),
            y: solid.map((p) => p.g),
            text: solid.map((p) => hover[p.r]![p.c]!),
            hoverinfo: "text" as const,
            showlegend: false,
            marker: {
              size: solid.map((p) => dotSize(fraction[p.r]![p.c]!)),
              color: solid.map((p) => colours.z[p.r]![p.c]!),
              colorscale: "Reds",
              cmin: colours.cmin,
              cmax: colours.cmax,
              line: { width: 0.5, color: "#94a3b8" },
              colorbar,
            },
          },
          {
            type: "scatter" as const,
            mode: "markers" as const,
            x: hollow.map((p) => p.gene),
            y: hollow.map((p) => p.g),
            text: hollow.map((p) => hover[p.r]![p.c]!),
            hoverinfo: "text" as const,
            showlegend: false,
            marker: {
              symbol: "circle-open",
              size: hollow.map((p) => Math.max(dotSize(fraction[p.r]![p.c]!), 5)),
              color: "#94a3b8",
              line: { width: 1.5 },
            },
          },
        ];
  const treeNote = treeData
    ? ` Groups ordered by the complete-linkage clustering of their centroid distances in PCA space${dendro ? " (tree on the right)" : ""}.`
    : "";
  const blankNote =
    kind === "heatmap" && meanOf === "expressing" && meanExpressing
      ? ` Blank: under ${Math.round(MIN_FRACTION * 100)}% of the group's cells express it.`
      : "";
  const geneTreeNote = geneDendro
    ? ` ${isProtein ? "Proteins" : "Genes"} ordered by the average-linkage clustering of their mean over all cells across the groups (correlation distance; tree above).`
    : "";
  const stripNote = strip ? " Strip above: the group each gene is a top marker of." : "";
  const selectionNote = selRowName
    ? " Last row: the selected cells (any group)."
    : selectionStats && lines
      ? " The selection's row is left out while the groups follow the dendrogram."
      : "";

  return (
    <div>
      <div className="mb-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate-600">
        <SelectionChip note="the last row" />
        <label htmlFor="marker-colour-mean" className="font-medium">Colour:</label>
        <select
          id="marker-colour-mean"
          value={meanOf}
          onChange={(e) => setMeanOf(e.target.value as MeanOf)}
          className={selectClass}
        >
          <option value="all">mean over all cells</option>
          <option value="expressing" disabled={!meanExpressing}>mean in expressing cells</option>
        </select>
        <select
          aria-label="Colour scale"
          value={scale}
          onChange={(e) => setScale(e.target.value as ColourScale)}
          className={selectClass}
        >
          <option value="gene">scaled per {noun}</option>
          <option value="fixed">fixed range</option>
        </select>
        {scale === "fixed" && (
          <span className="flex items-center gap-1">
            <span>{range[0]} to</span>
            <input
              type="number"
              aria-label="Top of the fixed colour range"
              step={0.5}
              value={range[1]}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (Number.isFinite(v) && v > range[0]) setFixedMax({ layer: data.layer, value: v });
              }}
              className="w-16 rounded border border-slate-300 bg-white px-1 py-0.5 text-xs text-slate-700"
            />
            {range[1] !== serverRange[1] && (
              <button
                onClick={() => setFixedMax(null)}
                className="text-xs font-medium text-primary hover:underline"
                title="The layer's 99th percentile of non-zero values"
              >
                reset ({serverRange[1]})
              </button>
            )}
          </span>
        )}
        {kind === "heatmap" && (
          <label
            className="ml-auto flex items-center gap-1"
            title="Order the genes by the average-linkage clustering of their mean expression across the groups (correlation distance)"
          >
            <input type="checkbox" checked={byGeneTree} onChange={(e) => setByGeneTree(e.target.checked)} />
            Cluster genes
          </label>
        )}
        <label
          className={`${kind === "heatmap" ? "" : "ml-auto "}flex items-center gap-1`}
          title="Order the groups by the complete-linkage clustering of their centroid distances in PCA space"
        >
          <input type="checkbox" checked={byTree} onChange={(e) => setByTree(e.target.checked)} />
          Order groups by dendrogram
        </label>
        {treeLoading && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
      </div>
      {treeError && (
        <p role="alert" className="mb-1 rounded-md bg-amber-50 px-2 py-1 text-xs text-amber-800">
          Can&apos;t order by dendrogram: {treeError}
        </p>
      )}
      <Plot
        data={[
          ...traces,
          ...(strip ? [strip] : []),
          ...(dendro ? [dendro.trace] : []),
          ...(geneDendro ? [geneDendro.trace] : []),
        ]}
        layout={{
          height: baseHeight + bands.extraPx + (margin.t - 10),
          margin,
          xaxis: {
            type: "category",
            tickangle: -60,
            automargin: true,
            categoryarray: genes,
            categoryorder: "array",
            ...(dendro ? { domain: PLOT_DOMAIN } : {}),
            // An explicit range so the gene tree's leaves meet the columns.
            ...(geneDendro ? { range: [-0.5, genes.length - 0.5] } : {}),
          },
          yaxis: {
            type: "category",
            automargin: true,
            categoryarray: rows,
            categoryorder: "array",
            ...(hasBands ? { domain: bands.main } : {}),
            // An explicit range so the dendrogram's leaves meet the rows.
            ...(dendro ? { range: [rows.length - 0.5, -0.5] } : { autorange: "reversed" as const }),
          },
          ...(strip && bands.strip
            ? {
                yaxis4: {
                  domain: bands.strip,
                  anchor: "x",
                  type: "category",
                  fixedrange: true,
                  tickfont: { size: 10, color: "#64748b" },
                  automargin: true,
                },
              }
            : {}),
          ...(dendro ? dendro.layout : {}),
          ...(geneDendro ? geneDendro.layout : {}),
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
        }}
        config={plotConfig(`markers_${kind}`)}
        useResizeHandler
        style={{ width: "100%" }}
      />
      {kind === "dotplot" && <DotSizeLegend />}
      {missing.length > 0 && (
        <p className="mt-1 text-xs text-amber-700">
          Not in this dataset&apos;s {layerLabel} layer, so not plotted (names are case-sensitive):{" "}
          {missing.join(", ")}.
        </p>
      )}
      {isProtein ? (
        <p className="mt-1 text-xs text-slate-500">
          {genes.length} protein{genes.length === 1 ? "" : "s"} across the {groupby} groups. Protein
          level: CLR. {colours.caption}
          {blankNote}
          {treeNote}
          {geneTreeNote}
          {selectionNote}
        </p>
      ) : (
        <p className="mt-1 text-xs text-slate-500">
          {custom
            ? `Your ${genes.length} gene${genes.length === 1 ? "" : "s"} across the ${groupby} groups.`
            : `Top ${perGroup} up-regulated markers of each ${groupby} group.`}{" "}
          {colours.caption}
          {kind === "dotplot" &&
            ` Dot size: share of the group's cells that express the gene (area, same scale in every plot); hollow: under ${Math.round(MIN_FRACTION * 100)}%, no colour.`}
          {blankNote}
          {treeNote}
          {geneTreeNote}
          {stripNote}
          {selectionNote}
          {` Layer: ${layerLabel}.`}
        </p>
      )}
    </div>
  );
}
