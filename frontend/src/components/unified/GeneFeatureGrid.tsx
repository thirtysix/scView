import { useCallback, useMemo } from "react";
import { ColorLegend } from "@/components/plots/ColorLegend";
import { BlendLegend } from "@/components/plots/BlendLegend";
import { MultiPanelScatter, type GridHover, type GridPanel } from "@/components/unified/MultiPanelScatter";
import { useSameCellsGridLasso } from "@/components/unified/GridLasso";
import { SelectionChip } from "@/components/common/SelectionChip";
import { useGeneExpression } from "@/hooks/useGeneExpression";
import { exportFilename } from "@/lib/exportFilename";
import { prettyEmbeddingLabel } from "@/lib/formatting";
import {
  BLEND_SCHEME_OPTIONS,
  BLEND_SCHEMES,
  blendColors,
  blendScale,
  scaled,
  valueOrder,
} from "@/lib/geneCoexpression";
import { colorByLine, type HoverLine } from "@/lib/hoverInfo";
import type { ScatterLegend } from "@/lib/scatterExport";
import { drawnIndices, gatherPositions, panelColors, sampleCellIndices } from "@/lib/splitView";
import { mergedStats, overlayStats, resolveColorScale, type ResolvedColorScale } from "@/lib/colorScale";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
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
  /** A lasso closed in any panel: the cells under it */
  onLassoComplete?: (cells: Set<number>) => void;
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
 * camera, one WebGL context; a lasso in any panel selects the cells under it.
 * Each panel draws its expressing cells on top of the rest.
 */
export function GeneFeatureGrid({
  datasetId,
  embedding,
  grid,
  positions,
  dimensions,
  selectedIndices = null,
  onHover,
  onLassoComplete,
}: GeneFeatureGridProps) {
  const lasso = useSameCellsGridLasso(positions, dimensions, onLassoComplete);
  const opacity = useSettingsStore((s) => s.opacity);
  const sequentialColormap = useSettingsStore((s) => s.continuousColormap);
  const divergingColormap = useSettingsStore((s) => s.divergingColormap);
  const signedColormapMode = useSettingsStore((s) => s.signedColormapMode);
  const defaultClip = useSettingsStore((s) => s.colorClip);
  const viewClip = useUnifiedViewStore((s) => s.colorClip);
  const setViewClip = useUnifiedViewStore((s) => s.setColorClip);
  const clipSetting = viewClip ?? defaultClip;
  const maxRenderedCells = useSettingsStore((s) => s.maxRenderedCells);
  const layer = useSettingsStore((s) => s.expressionLayer);
  const pipelineRunning = useSettingsStore((s) => s.pipelineRunning);
  const datasetName = useDatasetStore((s) => s.currentDataset?.name);
  const setGeneGrid = useSplitViewStore((s) => s.setGeneGrid);
  const sharedScale = useGeneGridStore((s) => s.sharedScale);
  const setSharedScale = useGeneGridStore((s) => s.setSharedScale);
  const clip = useGeneGridStore((s) => s.blendClip);
  const setClip = useGeneGridStore((s) => s.setBlendClip);
  const scheme = useGeneGridStore((s) => s.blendScheme);
  const setScheme = useGeneGridStore((s) => s.setBlendScheme);
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

  // Each gene's values sorted once (lib/colorScale.ts); the shared scale pools them.
  const stats = useMemo(() => (ready ? genes.map((g) => overlayStats(values![g]!)) : []), [ready, genes, values]);
  const pooled = useMemo(() => (ready && sharedScale && !blend ? mergedStats(stats) : null), [ready, sharedScale, blend, stats]);
  const ranges = useMemo<ResolvedColorScale[]>(() => {
    if (!ready) return [];
    const settings = { sequential: sequentialColormap, diverging: divergingColormap, signedMode: signedColormapMode, clip: clipSetting };
    if (pooled) {
      const shared = resolveColorScale("expression", pooled, settings);
      return genes.map(() => shared);
    }
    return stats.map((st) => resolveColorScale("expression", st, settings));
  }, [ready, genes, stats, pooled, sequentialColormap, divergingColormap, signedColormapMode, clipSetting]);

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
      const scale = ranges[i]!;
      const colors = panelColors(order, {
        colorValues: v,
        colorType: "continuous",
        range: scale,
        opacity,
        selected: selectedIndices,
        colormap: scale.colormap,
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
        make("blend", `${shown[0]} + ${shown[1]}`, "blend", order, blendColors(order, a, b, { clip, opacity, selected: selectedIndices, scheme })),
      );
    }
    return out;
  }, [ready, blend, genes, values, shuffled, selectedIndices, ranges, opacity, positions, stride, clip, scheme]);

  // Memoised: the hover handler depends on it, and MultiPanelScatter clears the
  // tooltip whenever that handler changes (a new array per render would clear it on every hover).
  const legendGenes = useMemo(() => (blend ? genes.slice(0, 2) : genes), [blend, genes]);
  const oneLegend = sharedScale && !blend && legendGenes.length > 1;
  const exportLegend = useMemo<ScatterLegend[]>(() => {
    if (!ready) return [];
    const entry = (title: string, r: ResolvedColorScale): ScatterLegend => ({
      type: "continuous",
      title,
      min: r.min,
      max: r.max,
      colormap: r.colormap,
      centre: r.centre,
    });
    const continuous: ScatterLegend[] = oneLegend
      ? [entry("Expression (all genes)", ranges[0]!)]
      : legendGenes.map((g, i) => entry(g, ranges[i]!));
    if (!blend) return continuous;
    const [a, b] = legendGenes;
    const corners = BLEND_SCHEMES[scheme];
    return [
      ...continuous,
      {
        type: "categorical",
        title: "Blend",
        categories: [`${a} only`, `${b} only`, "both", "neither"],
        colors: [corners.a, corners.b, corners.both, corners.neither],
      },
    ];
  }, [ready, oneLegend, legendGenes, ranges, blend, scheme]);

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
        <>
        <SelectionChip />
        {blend ? (
          <>
            <select
              aria-label="Blend colours"
              value={scheme}
              onChange={(e) => setScheme(e.target.value as typeof scheme)}
              title={BLEND_SCHEME_OPTIONS.find((o) => o.value === scheme)?.title}
              className="flex-shrink-0 rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px]"
            >
              {BLEND_SCHEME_OPTIONS.map((o) => (
                <option key={o.value} value={o.value} title={o.title}>
                  {o.label}
                </option>
              ))}
            </select>
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
          </>
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
        )}
        </>
      }
      sampledNote={sampledNote}
      sampledBadge={
        sample ? `${(sample.length / 1000).toFixed(0)}K of ${(total / 1000).toFixed(0)}K cells (sampled)` : undefined
      }
      onClose={() => setGeneGrid(null)}
      closeTitle={blend ? "Close the blend" : "Close the gene grid"}
      onHover={handleHover}
      lassoActive={lasso.lassoActive}
      onLasso={lasso.onLasso}
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
              <ColorLegend
                type="continuous"
                scale={ranges[0]!}
                clip={clipSetting}
                onClipChange={setViewClip}
                clipOverridden={viewClip !== null}
                label="Expression (all genes)"
              />
            ) : (
              legendGenes.map((g, i) => (
                <ColorLegend
                  key={g}
                  type="continuous"
                  scale={ranges[i]!}
                  label={g}
                  // The clip control once, under the first legend; it applies to every panel.
                  {...(i === 0 ? { clip: clipSetting, onClipChange: setViewClip, clipOverridden: viewClip !== null } : {})}
                />
              ))
            )}
            {blend && <BlendLegend geneA={legendGenes[0]!} geneB={legendGenes[1]!} scheme={scheme} />}
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
