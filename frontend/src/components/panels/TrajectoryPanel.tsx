import { useState, useCallback, useEffect } from "react";
import { GitBranch, Loader2, AlertCircle } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useEmbedding } from "@/hooks/useEmbedding";
import { apiFetch } from "@/api/client";
import { EmbeddingScatter } from "@/components/plots/EmbeddingScatter";
import { Panel } from "@/components/common/Panel";
import { GeneSearch } from "@/components/common/GeneSearch";
import { ColorLegend } from "@/components/plots/ColorLegend";
import { formatNumber } from "@/lib/formatting";

interface TrajectoryColumnsResponse {
  pseudotime_columns: string[];
  n_columns: number;
}

interface PseudotimeValuesResponse {
  column: string;
  n_cells: number;
  values: number[];
  min: number;
  max: number;
}

interface GeneAlongPseudotime {
  found: boolean;
  scatter_pseudotime?: number[];
  scatter_expression?: number[];
  binned_pseudotime?: number[];
  binned_expression?: number[];
  binned_counts?: number[];
}

interface TrajectoryGenesResponse {
  pseudotime_column: string;
  genes: Record<string, GeneAlongPseudotime>;
}

export function TrajectoryPanel() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const pointSize = useSettingsStore((s) => s.pointSize);
  const opacity = useSettingsStore((s) => s.opacity);
  const plotBackground = useSettingsStore((s) => s.plotBackground);
  const maxRenderedCells = useSettingsStore((s) => s.maxRenderedCells);
  const selectedIndices = useSelectionStore((s) => s.selectedCellIndices);

  const { positions, numCells, isLoading: embeddingLoading } = useEmbedding();

  // Pseudotime state
  const [availableColumns, setAvailableColumns] = useState<string[]>([]);
  const [selectedColumn, setSelectedColumn] = useState("");
  const [pseudotimeValues, setPseudotimeValues] = useState<Float32Array | null>(null);
  const [pseudotimeMinMax, setPseudotimeMinMax] = useState({ min: 0, max: 1 });
  const [isLoadingColumns, setIsLoadingColumns] = useState(false);
  const [isLoadingPseudotime, setIsLoadingPseudotime] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Gene expression along pseudotime
  const [selectedGenes, setSelectedGenes] = useState<string[]>([]);
  const [geneTrajectories, setGeneTrajectories] = useState<
    Record<string, GeneAlongPseudotime>
  >({});
  const [isLoadingGenes, setIsLoadingGenes] = useState(false);

  // Fetch available pseudotime columns
  useEffect(() => {
    if (!datasetId) return;

    setIsLoadingColumns(true);
    setError(null);

    apiFetch<TrajectoryColumnsResponse>(
      `/datasets/${datasetId}/trajectory`,
    )
      .then((data) => {
        setAvailableColumns(data.pseudotime_columns);
        if (data.pseudotime_columns.length > 0 && !selectedColumn) {
          setSelectedColumn(data.pseudotime_columns[0]!);
        }
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : String(err));
        setAvailableColumns([]);
      })
      .finally(() => setIsLoadingColumns(false));
  }, [datasetId]);

  // Fetch pseudotime values when column changes
  useEffect(() => {
    if (!datasetId || !selectedColumn) return;

    setIsLoadingPseudotime(true);

    apiFetch<PseudotimeValuesResponse>(
      `/datasets/${datasetId}/trajectory/${encodeURIComponent(selectedColumn)}`,
    )
      .then((data) => {
        const f32 = new Float32Array(data.values.length);
        for (let i = 0; i < data.values.length; i++) {
          f32[i] = data.values[i]!;
        }
        setPseudotimeValues(f32);
        setPseudotimeMinMax({ min: data.min, max: data.max });
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : String(err));
        setPseudotimeValues(null);
      })
      .finally(() => setIsLoadingPseudotime(false));
  }, [datasetId, selectedColumn]);

  // Handle gene selection — add to list and fetch trajectory data
  const handleGeneSelect = useCallback(
    async (gene: string) => {
      if (!datasetId || !selectedColumn) return;

      // Add gene if not already selected (up to 5 genes)
      setSelectedGenes((prev) => {
        if (prev.includes(gene)) return prev;
        if (prev.length >= 5) return [...prev.slice(1), gene];
        return [...prev, gene];
      });

      setIsLoadingGenes(true);

      try {
        // Fetch including the new gene plus existing ones
        const allGenes = [...new Set([...selectedGenes, gene])];
        const data = await apiFetch<TrajectoryGenesResponse>(
          `/datasets/${datasetId}/trajectory/${encodeURIComponent(selectedColumn)}/genes?genes=${encodeURIComponent(allGenes.join(","))}`,
        );
        setGeneTrajectories(data.genes);
      } catch {
        // Ignore — we keep whatever we had
      } finally {
        setIsLoadingGenes(false);
      }
    },
    [datasetId, selectedColumn, selectedGenes],
  );

  // Remove a gene from the selection
  const removeGene = useCallback((gene: string) => {
    setSelectedGenes((prev) => prev.filter((g) => g !== gene));
    setGeneTrajectories((prev) => {
      const next = { ...prev };
      delete next[gene];
      return next;
    });
  }, []);

  // Color palette for gene lines
  const GENE_COLORS = [
    "#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed",
  ];

  // No dataset
  if (!dataset || !datasetId) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <GitBranch className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">Trajectory</h2>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-12 text-center text-slate-400">
          Load a dataset to explore trajectory and pseudotime.
        </div>
      </div>
    );
  }

  // No pseudotime columns available
  if (!isLoadingColumns && availableColumns.length === 0) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <GitBranch className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">Trajectory</h2>
        </div>
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-8 text-center">
          <AlertCircle className="mx-auto mb-3 h-8 w-8 text-amber-500" />
          <p className="mb-1 text-sm font-medium text-amber-800">
            No pseudotime columns found
          </p>
          <p className="text-xs text-amber-600">
            This dataset does not contain pseudotime or trajectory data.
            Run trajectory inference (e.g. diffusion pseudotime, Monocle) on
            your data and re-upload to use this feature.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-4">
      {/* Header */}
      <div className="flex flex-shrink-0 items-center justify-between">
        <div className="flex items-center gap-3">
          <GitBranch className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">Trajectory</h2>
          {selectedColumn && (
            <span className="rounded-full bg-teal-100 px-3 py-0.5 text-sm font-semibold text-teal-700">
              {selectedColumn}
            </span>
          )}
        </div>
        {numCells > 0 && (
          <span className="text-sm text-slate-500">
            {formatNumber(numCells)} cells
          </span>
        )}
      </div>

      {/* Controls row */}
      <div className="flex flex-shrink-0 items-center gap-4">
        {/* Pseudotime column selector */}
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-slate-600">
            Pseudotime:
          </label>
          <select
            value={selectedColumn}
            onChange={(e) => setSelectedColumn(e.target.value)}
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
          >
            {availableColumns.map((col) => (
              <option key={col} value={col}>
                {col}
              </option>
            ))}
          </select>
        </div>

        {/* Gene search */}
        <GeneSearch
          datasetId={datasetId}
          onSelect={handleGeneSelect}
          placeholder="Add gene to trajectory plot..."
          className="flex-1 max-w-md"
        />

        {isLoadingPseudotime && (
          <span className="flex items-center gap-1 text-xs text-blue-500">
            <Loader2 className="h-3 w-3 animate-spin" />
            Loading pseudotime...
          </span>
        )}
      </div>

      {/* Selected genes chips */}
      {selectedGenes.length > 0 && (
        <div className="flex flex-shrink-0 flex-wrap items-center gap-2">
          <span className="text-xs font-medium text-slate-500">Genes:</span>
          {selectedGenes.map((gene, i) => (
            <span
              key={gene}
              className="inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold text-white"
              style={{ backgroundColor: GENE_COLORS[i % GENE_COLORS.length] }}
            >
              {gene}
              <button
                onClick={() => removeGene(gene)}
                className="ml-0.5 rounded-full p-0.5 transition-colors hover:bg-white/30"
              >
                <span className="sr-only">Remove</span>
                &times;
              </button>
            </span>
          ))}
          {isLoadingGenes && (
            <Loader2 className="h-3 w-3 animate-spin text-blue-500" />
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex flex-shrink-0 items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-2.5 text-sm text-red-700">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          {error}
        </div>
      )}

      {/* Main content: embedding + gene trajectory plot */}
      <div className="flex min-h-0 flex-1 gap-4">
        {/* Left: Embedding scatter colored by pseudotime */}
        <div className="flex min-w-0 flex-1 flex-col gap-2">
          <h3 className="text-sm font-semibold text-slate-700">
            Embedding Colored by Pseudotime
          </h3>
          <Panel className="min-h-0 flex-1" bodyClassName="relative min-h-0 p-0">
            {(embeddingLoading || isLoadingPseudotime) && !positions ? (
              <div className="flex h-full items-center justify-center">
                <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
              </div>
            ) : (
              <EmbeddingScatter
                positions={positions}
                colorValues={pseudotimeValues}
                colorType="continuous"
                pointSize={pointSize}
                opacity={opacity}
                selectedIndices={selectedIndices}
                background={plotBackground}
                maxRenderedCells={maxRenderedCells}
              />
            )}
          </Panel>
          {pseudotimeValues && (
            <div className="flex-shrink-0 px-2">
              <ColorLegend
                type="continuous"
                min={pseudotimeMinMax.min}
                max={pseudotimeMinMax.max}
                label={selectedColumn}
              />
            </div>
          )}
        </div>

        {/* Right: Gene expression along pseudotime */}
        <div className="flex w-[480px] flex-shrink-0 flex-col gap-2">
          <h3 className="text-sm font-semibold text-slate-700">
            Gene Expression Along Pseudotime
          </h3>
          <div className="flex-1 overflow-auto rounded-xl border border-slate-200 bg-white p-2 shadow-sm">
            {selectedGenes.length === 0 ? (
              <div className="flex h-full items-center justify-center">
                <div className="text-center">
                  <GitBranch className="mx-auto mb-2 h-8 w-8 text-slate-300" />
                  <p className="text-sm text-slate-400">
                    Search for genes above to plot expression along pseudotime
                  </p>
                </div>
              </div>
            ) : (
              <Plot
                data={selectedGenes.flatMap((gene, i) => {
                  const traj = geneTrajectories[gene];
                  if (!traj || !traj.found) return [];

                  const color = GENE_COLORS[i % GENE_COLORS.length]!;
                  const traces = [];

                  // Scatter: raw expression points
                  if (traj.scatter_pseudotime && traj.scatter_expression) {
                    traces.push({
                      type: "scattergl" as const,
                      mode: "markers" as const,
                      x: traj.scatter_pseudotime,
                      y: traj.scatter_expression,
                      name: `${gene} (cells)`,
                      marker: {
                        color: color,
                        size: 2,
                        opacity: 0.15,
                      },
                      showlegend: false,
                      hovertemplate:
                        `<b>${gene}</b><br>Pseudotime: %{x:.3f}<br>Expression: %{y:.3f}<extra></extra>`,
                    });
                  }

                  // Line: binned smoothed expression
                  if (traj.binned_pseudotime && traj.binned_expression) {
                    traces.push({
                      type: "scatter" as const,
                      mode: "lines" as const,
                      x: traj.binned_pseudotime,
                      y: traj.binned_expression,
                      name: gene,
                      line: {
                        color: color,
                        width: 2.5,
                        shape: "spline" as const,
                        smoothing: 1.3,
                      },
                      hovertemplate:
                        `<b>${gene}</b><br>Pseudotime: %{x:.3f}<br>Mean expr: %{y:.3f}<extra></extra>`,
                    });
                  }

                  return traces;
                })}
                layout={{
                  height: 450,
                  margin: { t: 20, r: 20, b: 60, l: 60 },
                  xaxis: {
                    title: {
                      text: selectedColumn || "Pseudotime",
                      font: { size: 12, color: "#64748b" },
                    },
                    tickfont: { size: 10, color: "#64748b" },
                    gridcolor: "#f1f5f9",
                  },
                  yaxis: {
                    title: {
                      text: "Expression",
                      font: { size: 12, color: "#64748b" },
                    },
                    tickfont: { size: 10, color: "#64748b" },
                    gridcolor: "#f1f5f9",
                    zeroline: false,
                  },
                  showlegend: selectedGenes.length > 1,
                  legend: {
                    x: 1,
                    y: 1,
                    xanchor: "right" as const,
                    bgcolor: "rgba(255,255,255,0.8)",
                    font: { size: 11 },
                  },
                  paper_bgcolor: "white",
                  plot_bgcolor: "white",
                  font: { family: "Inter, system-ui, sans-serif" },
                }}
                config={{ responsive: true, displayModeBar: false }}
                useResizeHandler
                className="w-full"
                style={{ width: "100%", height: "100%" }}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
