import { useMemo } from "react";
import { Dna } from "lucide-react";
import { useDatasetStore } from "@/stores/datasetStore";
import { GeneSearch } from "@/components/common/GeneSearch";
import { ColorLegend } from "@/components/plots/ColorLegend";

interface UnifiedExpressionSubtabProps {
  onGeneSelect: (gene: string) => void;
  activeGene: string | null;
  groupByColumn: string;
  setGroupByColumn: (col: string) => void;
  expressionMinMax: { min: number; max: number } | null;
}

export function UnifiedExpressionSubtab({
  onGeneSelect,
  activeGene,
  groupByColumn,
  setGroupByColumn,
  expressionMinMax,
}: UnifiedExpressionSubtabProps) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);

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
          </div>
          {expressionMinMax && (
            <div className="mt-2">
              <ColorLegend
                type="continuous"
                min={expressionMinMax.min}
                max={expressionMinMax.max}
                label={`${activeGene} expression`}
              />
            </div>
          )}
        </div>
      )}

      {/* Layer selector */}
      {expressionLayers.length > 1 && (
        <div>
          <label className="mb-1 block text-[11px] font-medium text-slate-500">
            Expression layer
          </label>
          <select className="w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700">
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
