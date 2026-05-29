import { useState, useCallback, useMemo, useEffect } from "react";
import { Dna, Loader2 } from "lucide-react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useViewStore } from "@/stores/viewStore";
import { useEmbedding } from "@/hooks/useEmbedding";
import { apiFetch, apiFetchBinary } from "@/api/client";
import { decodeArrowBuffer } from "@/lib/arrow";
import { EmbeddingScatter } from "@/components/plots/EmbeddingScatter";
import { GeneSearch } from "@/components/common/GeneSearch";
import { ViolinPlot } from "@/components/plots/ViolinPlot";
import { ColorLegend } from "@/components/plots/ColorLegend";
import { formatNumber } from "@/lib/formatting";

interface ViolinResponse {
  gene: string;
  groups: Record<string, number[]>;
}

export function GeneExpressionPanel() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const pointSize = useSettingsStore((s) => s.pointSize);
  const opacity = useSettingsStore((s) => s.opacity);
  const plotBackground = useSettingsStore((s) => s.plotBackground);
  const maxRenderedCells = useSettingsStore((s) => s.maxRenderedCells);
  const selectedIndices = useSelectionStore((s) => s.selectedCellIndices);

  const { positions, numCells, dimensions, isLoading: embeddingLoading } = useEmbedding();

  const [selectedGene, setSelectedGene] = useState<string | null>(null);
  const [expressionValues, setExpressionValues] = useState<Float32Array | null>(null);
  const [isLoadingExpression, setIsLoadingExpression] = useState(false);
  const [expressionError, setExpressionError] = useState<string | null>(null);

  const [violinData, setViolinData] = useState<Record<string, number[]>>({});
  const [isLoadingViolin, setIsLoadingViolin] = useState(false);

  const [currentGroupBy, setCurrentGroupBy] = useState<string>("");
  const [currentLayer, setCurrentLayer] = useState<string>("");

  // Expression layers from dataset metadata
  const expressionLayers = useMemo(
    () => dataset?.expression_layers ?? [],
    [dataset],
  );

  // Initialize currentLayer from dataset default
  useEffect(() => {
    if (!currentLayer && dataset?.default_expression_layer) {
      setCurrentLayer(dataset.default_expression_layer);
    }
  }, [dataset, currentLayer]);

  // Auto-detect categorical columns for group-by, excluding non-informative ones
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

  // Set default group-by: prefer active clustering, then first categorical column
  useEffect(() => {
    if (!currentGroupBy) {
      if (dataset?.active_clustering) {
        const found = categoricalColumns.find((c) => c.name === dataset.active_clustering);
        if (found) {
          setCurrentGroupBy(found.name);
          return;
        }
      }
      if (categoricalColumns.length > 0) {
        setCurrentGroupBy(categoricalColumns[0]!.name);
      }
    }
  }, [categoricalColumns, currentGroupBy, dataset]);

  // Compute min/max for color legend
  const expressionMinMax = useMemo(() => {
    if (!expressionValues) return { min: 0, max: 1 };
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < expressionValues.length; i++) {
      const v = expressionValues[i]!;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    return { min, max };
  }, [expressionValues]);

  // Fetch expression data when gene is selected
  const fetchExpression = useCallback(
    async (gene: string, layer?: string) => {
      if (!datasetId) return;

      setIsLoadingExpression(true);
      setExpressionError(null);

      const layerParam = layer || currentLayer;
      const layerQuery = layerParam ? `&layer=${encodeURIComponent(layerParam)}` : "";

      try {
        const buffer = await apiFetchBinary(
          `/datasets/${datasetId}/expression?genes=${encodeURIComponent(gene)}${layerQuery}`,
        );
        const decoded = decodeArrowBuffer(buffer);
        // The expression column should be the gene name or the first column
        const values =
          decoded[gene] ??
          decoded[Object.keys(decoded)[0]!];
        if (values instanceof Float32Array) {
          setExpressionValues(values);
        } else if (values) {
          // Convert to Float32Array if needed
          const f32 = new Float32Array(values.length);
          for (let i = 0; i < values.length; i++) {
            f32[i] = Number(values[i]);
          }
          setExpressionValues(f32);
        } else {
          setExpressionValues(null);
          setExpressionError("No expression data returned");
        }
      } catch (err) {
        setExpressionError(err instanceof Error ? err.message : String(err));
        setExpressionValues(null);
      } finally {
        setIsLoadingExpression(false);
      }
    },
    [datasetId, currentLayer],
  );

  // Fetch violin data when gene or groupBy changes
  const fetchViolin = useCallback(
    async (gene: string, groupBy: string, layer?: string) => {
      if (!datasetId || !groupBy) return;

      setIsLoadingViolin(true);

      const layerParam = layer || currentLayer;
      const layerQuery = layerParam ? `&layer=${encodeURIComponent(layerParam)}` : "";

      try {
        const data = await apiFetch<ViolinResponse>(
          `/datasets/${datasetId}/expression/violin?gene=${encodeURIComponent(gene)}&groupby=${encodeURIComponent(groupBy)}${layerQuery}`,
        );
        setViolinData(data.groups ?? {});
      } catch {
        setViolinData({});
      } finally {
        setIsLoadingViolin(false);
      }
    },
    [datasetId, currentLayer],
  );

  // Handle gene selection
  const handleGeneSelect = useCallback(
    (gene: string) => {
      setSelectedGene(gene);
      fetchExpression(gene);
      if (currentGroupBy) {
        fetchViolin(gene, currentGroupBy);
      }
    },
    [fetchExpression, fetchViolin, currentGroupBy],
  );

  // Consume pendingGene from view store (e.g. from Marker Genes "View" button)
  const pendingGene = useViewStore((s) => s.pendingGene);
  const setPendingGene = useViewStore((s) => s.setPendingGene);
  useEffect(() => {
    if (pendingGene && datasetId) {
      handleGeneSelect(pendingGene);
      setPendingGene(null);
    }
  }, [pendingGene, datasetId, handleGeneSelect, setPendingGene]);

  // Re-fetch violin when groupBy changes
  const handleGroupByChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const groupBy = e.target.value;
      setCurrentGroupBy(groupBy);
      if (selectedGene) {
        fetchViolin(selectedGene, groupBy);
      }
    },
    [selectedGene, fetchViolin],
  );

  // Re-fetch expression + violin when layer changes
  const handleLayerChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const newLayer = e.target.value;
      setCurrentLayer(newLayer);
      if (selectedGene) {
        fetchExpression(selectedGene, newLayer);
        if (currentGroupBy) {
          fetchViolin(selectedGene, currentGroupBy, newLayer);
        }
      }
    },
    [selectedGene, currentGroupBy, fetchExpression, fetchViolin],
  );

  // No dataset
  if (!dataset || !datasetId) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <Dna className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">Gene Expression</h2>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-12 text-center text-slate-400">
          Load a dataset to explore gene expression.
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-4">
      {/* Header */}
      <div className="flex flex-shrink-0 items-center justify-between">
        <div className="flex items-center gap-3">
          <Dna className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">Gene Expression</h2>
          {selectedGene && (
            <span className="rounded-full bg-blue-100 px-3 py-0.5 text-sm font-semibold text-blue-700">
              {selectedGene}
            </span>
          )}
        </div>
        {numCells > 0 && (
          <span className="text-sm text-slate-500">
            {formatNumber(numCells)} cells
          </span>
        )}
      </div>

      {/* Gene search bar */}
      <div className="flex flex-shrink-0 items-center gap-4">
        <GeneSearch
          datasetId={datasetId}
          onSelect={handleGeneSelect}
          placeholder="Search for a gene (e.g. CD3D, MS4A1)..."
          className="flex-1 max-w-md"
        />

        {/* Layer selector */}
        <div className="flex items-center gap-2">
          <label htmlFor="layer-select" className="text-sm font-medium text-slate-600">
            Layer:
          </label>
          <select
            id="layer-select"
            value={currentLayer}
            onChange={handleLayerChange}
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
          >
            {expressionLayers.length === 0 && (
              <option value="">Loading...</option>
            )}
            {expressionLayers.map((l) => (
              <option key={l.key} value={l.key}>
                {l.label}
              </option>
            ))}
          </select>
        </div>

        {/* Group-by selector */}
        <div className="flex items-center gap-2">
          <label htmlFor="groupby-select" className="text-sm font-medium text-slate-600">
            Group by:
          </label>
          <select
            id="groupby-select"
            value={currentGroupBy}
            onChange={handleGroupByChange}
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
          >
            {categoricalColumns.map((col) => (
              <option key={col.name} value={col.name}>
                {col.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* No gene selected */}
      {!selectedGene && (
        <div className="flex flex-1 items-center justify-center rounded-xl border border-dashed border-slate-300 bg-slate-50">
          <div className="text-center">
            <Dna className="mx-auto mb-3 h-10 w-10 text-slate-300" />
            <p className="text-sm text-slate-500">
              Search for a gene above to visualize its expression
            </p>
          </div>
        </div>
      )}

      {/* Expression error */}
      {expressionError && (
        <div className="flex-shrink-0 rounded-lg border border-red-200 bg-red-50 px-4 py-2.5 text-sm text-red-700">
          {expressionError}
        </div>
      )}

      {/* Split layout when gene is selected */}
      {selectedGene && (
        <div className="flex min-h-0 flex-1 gap-4">
          {/* Left: Embedding scatter colored by expression */}
          <div className="flex min-w-0 flex-1 flex-col gap-2">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-700">
                Expression on Embedding
              </h3>
              {isLoadingExpression && (
                <span className="flex items-center gap-1 text-xs text-blue-500">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Loading expression...
                </span>
              )}
            </div>
            <div className="relative min-h-0 flex-1 rounded-xl border border-slate-200 bg-white shadow-sm">
              {embeddingLoading && !positions ? (
                <div className="flex h-full items-center justify-center">
                  <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
                </div>
              ) : (
                <EmbeddingScatter
                  positions={positions}
                  colorValues={expressionValues}
                  colorType="continuous"
                  pointSize={pointSize}
                  opacity={opacity}
                  selectedIndices={selectedIndices}
                  background={plotBackground}
                  maxRenderedCells={maxRenderedCells}
                  dimensions={dimensions}
                />
              )}
            </div>
            {/* Color legend */}
            {expressionValues && (
              <div className="flex-shrink-0 px-2">
                <ColorLegend
                  type="continuous"
                  min={expressionMinMax.min}
                  max={expressionMinMax.max}
                  label={selectedGene}
                />
              </div>
            )}
          </div>

          {/* Right: Violin plot */}
          <div className="flex min-w-[350px] flex-1 flex-col gap-2">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-700">
                Expression by {currentGroupBy || "Group"}
              </h3>
              {isLoadingViolin && (
                <span className="flex items-center gap-1 text-xs text-blue-500">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Loading...
                </span>
              )}
            </div>
            <div className="flex-1 overflow-auto rounded-xl border border-slate-200 bg-white p-2 shadow-sm">
              {isLoadingViolin && Object.keys(violinData).length === 0 ? (
                <div className="flex h-full items-center justify-center">
                  <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
                </div>
              ) : (
                <ViolinPlot
                  data={violinData}
                  title={`${selectedGene} expression`}
                  xLabel={currentGroupBy}
                  yLabel="Expression"
                  height={Math.max(350, Object.keys(violinData).length * 30 + 100)}
                />
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
