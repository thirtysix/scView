import { useCallback, useMemo } from "react";
import { ColorLegend } from "@/components/plots/ColorLegend";
import { BlendLegend } from "@/components/plots/BlendLegend";
import { MultiPanelScatter, type GridHover, type GridPanel } from "@/components/unified/MultiPanelScatter";
import { useGeneExpression } from "@/hooks/useGeneExpression";
import { exportFilename } from "@/lib/exportFilename";
import { prettyEmbeddingLabel } from "@/lib/formatting";
import {
  BLEND_CORNERS,
  blendColors,
  blendScale,
  scaled,
  sharedRange,
  valueOrder,
} from "@/lib/geneCoexpression";
import { colorByLine, type HoverLine } from "@/lib/hoverInfo";
import type { ScatterLegend } from "@/lib/scatterExport";
import { drawnIndices, gatherPositions, panelColors, sampleCellIndices, valueRange } from "@/lib/splitView";
import { selectedLast, shuffledOrder } from "@/lib/drawOrder";
import { useDatasetStore } from "@/stores/datasetStore";
import { useGeneGridStore } from "@/stores/geneGridStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSplitViewStore, type GeneGrid } from "@/stores/splitViewStore";

interface GeneFeatureGridProps {
  datasetId: string;
  embedding: string;
  grid: GeneGrid;
  positions: Float32Array;
  dimensions: 2 | 3;
  selectedIndices?: Set<number> | null;
  /** Hovered cell with x/y in the scatter area's pixels, and its value of every gene */
  onHover?: (info: { index: number; x: number; y: number; lines: HoverLine[] } | null) => void;
}

const CLIP_OPTIONS = [
  { value: 0.99, label: "99th pct" },
  { value: 0.95, label: "95th pct" },
  { value: 1, label: "max" },
];

/**
 * The embedding coloured by several genes side by side (feature plots), or by
 * two genes and their blend (Seurat FeaturePlot blend: A red, B green, both
 * yellow). Built on MultiPanelScatter like the split and compare grids: one
 * camera, one WebGL context, lasso off while it shows. Each panel draws its
 * expressing cells on top of the rest.
 */
export function GeneFeatureGrid({
  datasetId,
  embedding,
  grid,
  positions,
  dimensions,
  selectedIndices = null,
  onHover,
}: GeneFeatureGridProps) {
  const opacity = useSettingsStore((s) => s.opacity);
  const colormap = useSettingsStore((s) => s.continuousColormap);
  const maxRenderedCells = useSettingsStore((s) => s.maxRenderedCells);
  const layer = useSettingsStore((s) => s.expressionLayer);
  const pipelineRunning = useSettingsStore((s) => s.pipelineRunning);
  const datasetName = useDatasetStore((s) => s.currentDataset?.name);
  const setGeneGrid = useSplitViewStore((s) => s.setGeneGrid);
  const sharedScale = useGeneGridStore((s) => s.sharedScale);
  const setSharedScale = useGeneGridStore((s) => s.setSharedScale);
  const clip = useGeneGridStore((s) => s.blendClip);
  const setClip = useGeneGridStore((s) => s.setBlendClip);
  const blend = grid.kind === "blend";

  const { values, missing, isLoading, error } = useGeneExpression(datasetId, grid.genes, layer, !pipelineRunning);
  const stride = dimensions === 3 ? 3 : 2;
  const total = positions.length / stride;
  const genes = useMemo(
    () => (values ? grid.genes.filter((g) => values[g]?.length === total) : []),
    [values, grid.genes, total],
  );
  const ready = total > 0 && !!values && genes.length >= (blend ? 2 : 1);

  // --- Every panel draws the single plot's sample, in its seeded order ---
  const sample = useMemo(() => sampleCellIndices(total, maxRenderedCells), [total, maxRenderedCells]);
  const shuffled = useMemo(() => shuffledOrder(drawnIndices(total, sample)), [total, sample]);

  const ranges = useMemo(() => {
    if (!ready) return [];
    const own = genes.map((g) => valueRange(values![g]!));
    if (!sharedScale || blend) return own;
    const shared = sharedRange(genes.map((g) => values![g]!));
    return genes.map(() => shared);
  }, [ready, genes, values, sharedScale, blend]);

  const panels = useMemo<GridPanel[]>(() => {
    if (!ready) return [];
    const make = (key: string, title: string, subtitle: string, order: Uint32Array, colors: Uint8Array): GridPanel => ({
      key,
      title,
      subtitle,
      indices: order,
      positions: gatherPositions(positions, order, stride),
      colors,
    });
    const shown = blend ? genes.slice(0, 2) : genes;
    // A feature plot: expressing cells on top, then a selection over everything.
    const out = shown.map((g, i) => {
      const v = values![g]!;
      const order = selectedLast(valueOrder(shuffled, (c) => v[c]!), selectedIndices);
      const colors = panelColors(order, {
        colorValues: v,
        colorType: "continuous",
        range: ranges[i]!,
        opacity,
        selected: selectedIndices,
        colormap,
      });
      return make(g, g, expressingText(v), order, colors);
    });
    if (blend) {
      const a = values![shown[0]!]!;
      const b = values![shown[1]!]!;
      const sa = blendScale(a, clip);
      const sb = blendScale(b, clip);
      const order = selectedLast(
        valueOrder(shuffled, (c) => Math.max(scaled(a[c]!, sa), scaled(b[c]!, sb))),
        selectedIndices,
      );
      out.push(
        make("blend", `${shown[0]} + ${shown[1]}`, "blend", order, blendColors(order, a, b, { clip, opacity, selected: selectedIndices })),
      );
    }
    return out;
  }, [ready, blend, genes, values, shuffled, selectedIndices, ranges, opacity, colormap, positions, stride, clip]);

  // Memoised: the hover handler depends on it, and MultiPanelScatter clears the
  // tooltip whenever that handler changes (a new array per render would clear it on every hover).
  const legendGenes = useMemo(() => (blend ? genes.slice(0, 2) : genes), [blend, genes]);
  const oneLegend = sharedScale && !blend && legendGenes.length > 1;
  const exportLegend = useMemo<ScatterLegend[]>(() => {
    if (!ready) return [];
    const continuous: ScatterLegend[] = oneLegend
      ? [{ type: "continuous", title: "Expression (all genes)", ...ranges[0]! }]
      : legendGenes.map((g, i) => ({ type: "continuous", title: g, ...ranges[i]! }));
    if (!blend) return continuous;
    const [a, b] = legendGenes;
    return [
      ...continuous,
      {
        type: "categorical",
        title: "Blend",
        categories: [`${a} only`, `${b} only`, "both", "neither"],
        colors: [BLEND_CORNERS.a, BLEND_CORNERS.b, BLEND_CORNERS.both, BLEND_CORNERS.neither],
      },
    ];
  }, [ready, oneLegend, legendGenes, ranges, blend]);

  // The tooltip leads with the cell's value of every gene shown.
  const handleHover = useCallback(
    (info: GridHover | null) => {
      if (!info || !values) {
        onHover?.(null);
        return;
      }
      const lines = legendGenes
        .map((g) => colorByLine({ label: g, values: values[g] ?? null, index: info.index, type: "continuous" }))
        .filter((l): l is HoverLine => l != null);
      onHover?.({ index: info.index, x: info.x, y: info.y, lines });
    },
    [legendGenes, values, onHover],
  );

  const sampledNote = sample
    ? `Showing ${sample.length.toLocaleString()} of ${total.toLocaleString()} cells (sampled)`
    : undefined;
  const failed =
    error != null
      ? `Could not load ${grid.genes.join(", ")}.`
      : values && genes.length < (blend ? 2 : 1)
        ? `${missing.join(", ")} not found in this expression layer.`
        : null;

  return (
    <MultiPanelScatter
      positions={positions}
      dimensions={dimensions}
      panels={panels}
      ready={ready}
      loadingText={isLoading ? `Loading ${grid.genes.join(", ")}…` : "Preparing…"}
      errorText={failed}
      header={
        <>
          {blend ? "Blend" : "Genes"}{" "}
          <span className="font-semibold text-slate-700">{legendGenes.join(" · ")}</span>
          {missing.length > 0 && ready && (
            <span className="ml-1 text-amber-600">· not in this layer: {missing.join(", ")}</span>
          )}
        </>
      }
      headerControls={
        blend ? (
          <label className="flex flex-shrink-0 items-center gap-1" title="Each gene's colour tops out at this quantile of its expressing cells">
            Top at
            <select
              aria-label="Blend scale top"
              value={clip}
              onChange={(e) => setClip(Number(e.target.value))}
              className="rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px]"
            >
              {CLIP_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
        ) : (
          <label
            className="flex flex-shrink-0 cursor-pointer items-center gap-1"
            title="One colour scale for every gene (from the lowest to the highest value of any of them), so panels compare levels; off: each gene spans the full scale"
          >
            <input
              type="checkbox"
              checked={sharedScale}
              onChange={(e) => setSharedScale(e.target.checked)}
              className="h-3 w-3 accent-blue-500"
            />
            Shared scale
          </label>
        )
      }
      sampledNote={sampledNote}
      sampledBadge={
        sample ? `${(sample.length / 1000).toFixed(0)}K of ${(total / 1000).toFixed(0)}K cells (sampled)` : undefined
      }
      onClose={() => setGeneGrid(null)}
      closeTitle={blend ? "Close the blend" : "Close the gene grid"}
      onHover={handleHover}
      exportLegend={exportLegend}
      exportFilename={exportFilename(
        datasetName,
        `${prettyEmbeddingLabel(embedding || "embedding")}_${blend ? "blend" : "genes"}_${legendGenes.join("_")}`,
      )}
      embedding={embedding}
      legend={
        ready ? (
          <div className="flex flex-col gap-3 p-2">
            {oneLegend ? (
              <ColorLegend type="continuous" min={ranges[0]!.min} max={ranges[0]!.max} label="Expression (all genes)" />
            ) : (
              legendGenes.map((g, i) => (
                <ColorLegend key={g} type="continuous" min={ranges[i]!.min} max={ranges[i]!.max} label={g} />
              ))
            )}
            {blend && <BlendLegend geneA={legendGenes[0]!} geneB={legendGenes[1]!} />}
          </div>
        ) : null
      }
    />
  );
}

/** "12% expressing": the share of cells above 0. */
function expressingText(values: Float32Array): string {
  let n = 0;
  for (let i = 0; i < values.length; i++) if (values[i]! > 0) n++;
  const pct = values.length ? (100 * n) / values.length : 0;
  return `${pct < 1 && pct > 0 ? "<1" : Math.round(pct)}% expressing`;
}
