import { useMemo } from "react";
import { Dna } from "lucide-react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { GeneSearch } from "@/components/common/GeneSearch";
import { AddToGeneListMenu } from "@/components/common/AddToGeneListMenu";
import { ColorLegend } from "@/components/plots/ColorLegend";
import type { ResolvedColorScale } from "@/lib/colorScale";
import { MultiGenePanel } from "@/components/unified/MultiGenePanel";
import { CoexpressedGenes } from "@/components/unified/CoexpressedGenes";
import { DensityToggle } from "@/components/unified/DensityToggle";
import { DENSITY_SCALE_NAME, DENSITY_TIP, densityLegendNote, lastDensityMeta } from "@/lib/density";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

interface UnifiedExpressionSubtabProps {
  onGeneSelect: (gene: string) => void;
  activeGene: string | null;
  groupByColumn: string;
  setGroupByColumn: (col: string) => void;
  /** The active gene's resolved colour scale (the plot's legend, in the subtab) */
  expressionScale: ResolvedColorScale | null;
}

export function UnifiedExpressionSubtab({
  onGeneSelect,
  activeGene,
  groupByColumn,
  setGroupByColumn,
  expressionScale,
}: UnifiedExpressionSubtabProps) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const showsDensity = useUnifiedViewStore((s) => s.scatterOverlay?.type === "density");

  const categoricalColumns = useMemo(() => {
    if (!dataset) return [];
    const nCells = dataset.n_cells ?? 0;
    return dataset.obs_columns.filter((c) => {
      if (c.dtype !== "category" && c.dtype !== "object" && c.dtype !== "bool") return false;
      if (c.n_unique > 100) return false;
      if (nCells > 0 && c.n_unique / nCells >= 0.9) return false;
      return true;
    });
  }, [dataset]);

  const expressionLayers = dataset?.expression_layers ?? [];
  const expressionLayer = useSettingsStore((s) => s.expressionLayer);
  const setExpressionLayer = useSettingsStore((s) => s.setExpressionLayer);

  if (!dataset || !datasetId) {
    return <div className="text-xs text-slate-400">No dataset loaded.</div>;
  }

  return (
    <div className="space-y-3">
      {/* Gene search */}
      <div>
        <label className="mb-1 block text-[11px] font-medium text-slate-500">Search gene</label>
        <GeneSearch
          datasetId={datasetId}
          onSelect={onGeneSelect}
          placeholder="e.g. CD3D, MS4A1..."
          className="w-full"
        />
      </div>

      {/* Active gene info */}
      {activeGene && (
        <div className="rounded-lg border border-blue-100 bg-blue-50/50 p-2.5">
          <div className="flex items-center gap-2">
            <Dna className="h-3.5 w-3.5 text-blue-500" />
            <span className="text-sm font-semibold text-blue-700">{activeGene}</span>
            <AddToGeneListMenu
              genes={[activeGene]}
              defaultName="My genes"
              source="Expression search"
              className="ml-auto"
              title={`Add ${activeGene} to a gene list`}
            />
          </div>
          {expressionScale && (
            <div className="mt-2">
              <ColorLegend
                type="continuous"
                scale={expressionScale}
                label={showsDensity ? `${activeGene} ${DENSITY_SCALE_NAME}` : `${activeGene} expression`}
                note={showsDensity ? densityLegendNote(lastDensityMeta()) : undefined}
                noteTip={showsDensity ? DENSITY_TIP : undefined}
              />
            </div>
          )}
          <DensityToggle datasetId={datasetId} gene={activeGene} onGeneSelect={onGeneSelect} />
        </div>
      )}

      {/* Layer selector */}
      {expressionLayers.length > 1 && (
        <div>
          <label className="mb-1 block text-[11px] font-medium text-slate-500">
            Expression layer
          </label>
          {/* The same setting as the toolbar's Layer menu. */}
          <select
            value={expressionLayer}
            onChange={(e) => setExpressionLayer(e.target.value)}
            className="w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700"
          >
            {expressionLayers.map((l) => (
              <option key={l.key} value={l.key}>{l.label}</option>
            ))}
          </select>
        </div>
      )}

      {/* Group by */}
      <div>
        <label className="mb-1 block text-[11px] font-medium text-slate-500">
          Group by (violin)
        </label>
        <select
          aria-label="Group by (violin)"
          value={groupByColumn}
          onChange={(e) => setGroupByColumn(e.target.value)}
          className="w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700"
        >
          {categoricalColumns.map((col) => (
            <option key={col.name} value={col.name}>
              {col.name} ({col.n_unique} groups)
            </option>
          ))}
        </select>
      </div>

      {/* Genes that rise and fall with the overlaid one */}
      {activeGene && (
        <CoexpressedGenes datasetId={datasetId} activeGene={activeGene} onGeneSelect={onGeneSelect} />
      )}

      {/* Several genes: feature grid, stacked violin */}
      <MultiGenePanel datasetId={datasetId} groupByColumn={groupByColumn} />

      {/* Instructions */}
      {!activeGene && (
        <div className="rounded-lg border border-dashed border-slate-200 bg-slate-50 p-4 text-center">
          <Dna className="mx-auto mb-2 h-6 w-6 text-slate-300" />
          <p className="text-xs text-slate-400">
            Search for a gene above to color the scatter plot by its expression
          </p>
        </div>
      )}
    </div>
  );
}
