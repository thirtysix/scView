import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChartScatter, AlertCircle, Loader2, X } from "lucide-react";
import { useEmbedding } from "@/hooks/useEmbedding";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useViewStore } from "@/stores/viewStore";
import { EmbeddingScatter } from "@/components/plots/EmbeddingScatter";
import { Panel } from "@/components/common/Panel";
import { PlotControls } from "@/components/plots/PlotControls";
import { ColorLegend } from "@/components/plots/ColorLegend";
import { LassoSelector } from "@/components/plots/LassoSelector";
import { mapCategoryToColor } from "@/lib/colors";
import { formatNumber } from "@/lib/formatting";
import { apiFetch } from "@/api/client";

export function OverviewPanel() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const embedding = useSettingsStore((s) => s.embedding);
  const colorBy = useSettingsStore((s) => s.colorBy);
  const pointSize = useSettingsStore((s) => s.pointSize);
  const opacity = useSettingsStore((s) => s.opacity);
  const plotBackground = useSettingsStore((s) => s.plotBackground);
  const maxRenderedCells = useSettingsStore((s) => s.maxRenderedCells);
  const selectedIndices = useSelectionStore((s) => s.selectedCellIndices);
  const selectionMode = useSelectionStore((s) => s.selectionMode);
  const setSelection = useSelectionStore((s) => s.setSelection);
  const clearSelection = useSelectionStore((s) => s.clearSelection);

  const setPointSize = useSettingsStore((s) => s.setPointSize);
  const setOpacity = useSettingsStore((s) => s.setOpacity);

  const { positions, colors, numCells, colorColumnName, dimensions, isLoading, error } =
    useEmbedding();

  // Auto-adjust rendering defaults based on cell count (once per dataset)
  const autoAdjustedRef = useRef<string | null>(null);
  useEffect(() => {
    const dsId = dataset?.id ?? null;
    if (numCells > 0 && dsId && autoAdjustedRef.current !== dsId) {
      autoAdjustedRef.current = dsId;
      if (numCells > 100_000) {
        setPointSize(1);
        setOpacity(0.5);
      } else if (numCells > 50_000) {
        setPointSize(1.5);
        setOpacity(0.6);
      }
    }
  }, [numCells, dataset, setPointSize, setOpacity]);

  // Auto-select first available embedding when dataset changes
  const setEmbedding = useSettingsStore((s) => s.setEmbedding);
  useEffect(() => {
    if (!dataset) return;
    const available = dataset.available_embeddings;
    if (available.length === 0) {
      setEmbedding("");
    } else if (!available.includes(embedding)) {
      // Current selection isn't available — pick the best default
      const preferred = ["X_umap", "X_tsne", "X_pca"];
      const pick = preferred.find((e) => available.includes(e)) ?? available[0]!;
      setEmbedding(pick);
    }
  }, [dataset, embedding, setEmbedding]);

  // Track the deck.gl view state for lasso coordinate mapping
  const [viewState, setViewState] = useState<Record<string, unknown> | null>(
    null,
  );
  // Determine color type from the current colorBy obs column
  const colorType = useMemo<"categorical" | "continuous">(() => {
    if (!colorBy || !dataset) return "categorical";
    const col = dataset.obs_columns.find((c) => c.name === colorBy);
    if (!col) return "categorical";
    if (
      col.dtype === "category" ||
      col.dtype === "object" ||
      col.dtype === "bool"
    ) {
      return "categorical";
    }
    return "continuous";
  }, [colorBy, dataset]);

  // Get category names from the obs_columns metadata
  const obsColumn = useMemo(() => {
    if (!colorBy || !dataset) return null;
    return dataset.obs_columns.find((c) => c.name === colorBy) ?? null;
  }, [colorBy, dataset]);

  const categories = useMemo(() => {
    if (colorType !== "categorical" || !obsColumn?.values) return undefined;
    return obsColumn.values;
  }, [colorType, obsColumn]);

  const categoryColors = useMemo<[number, number, number][] | undefined>(() => {
    if (!categories) return undefined;
    return categories.map((_, i) => {
      const [r, g, b] = mapCategoryToColor(i);
      return [r, g, b] as [number, number, number];
    });
  }, [categories]);

  // Min/max for continuous color legend
  const colorMinMax = useMemo(() => {
    if (colorType !== "continuous" || !colors) return { min: 0, max: 1 };
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < colors.length; i++) {
      const v = colors[i]!;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    return { min, max };
  }, [colorType, colors]);

  // Handle lasso selection completion
  const handleSelectionComplete = useCallback(
    (indices: Set<number>) => {
      if (indices.size === 0) {
        clearSelection();
      } else {
        setSelection(indices);
      }
    },
    [setSelection, clearSelection],
  );

  // Keyboard shortcut: L to toggle lasso
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "l" || e.key === "L") {
        const store = useSelectionStore.getState();
        store.setSelectionMode(
          store.selectionMode === "lasso" ? "none" : "lasso",
        );
      }
      if (e.key === "Escape") {
        clearSelection();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [clearSelection]);

  // Hover tooltip state
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const [hoveredCell, setHoveredCell] = useState<{
    x: number;
    y: number;
    index: number;
    metadata: Record<string, unknown> | null;
  } | null>(null);
  const hoverDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleHover = useCallback(
    (info: { index: number; x: number; y: number } | null) => {
      if (hoverDebounceRef.current) {
        clearTimeout(hoverDebounceRef.current);
        hoverDebounceRef.current = null;
      }

      if (!info) {
        setHoveredCell(null);
        return;
      }

      // Show position immediately with null metadata
      setHoveredCell({ x: info.x, y: info.y, index: info.index, metadata: null });

      // Debounce the API fetch
      if (datasetId) {
        hoverDebounceRef.current = setTimeout(() => {
          apiFetch<Record<string, unknown>>(
            `/datasets/${datasetId}/metadata/cell/${info.index}`,
          )
            .then((data) => {
              setHoveredCell((prev) =>
                prev && prev.index === info.index
                  ? { ...prev, metadata: data }
                  : prev,
              );
            })
            .catch(() => {
              // Ignore — tooltip stays without metadata
            });
        }, 50);
      }
    },
    [datasetId],
  );

  // Cleanup hover debounce on unmount
  useEffect(() => {
    return () => {
      if (hoverDebounceRef.current) clearTimeout(hoverDebounceRef.current);
    };
  }, []);

  const setPanel = useViewStore((s) => s.setPanel);

  const hasEmbeddings =
    dataset != null && dataset.available_embeddings.length > 0;

  // No dataset loaded
  if (!dataset) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <ChartScatter className="h-6 w-6 text-blue-500" />
          <h2 className="text-2xl font-bold text-slate-900">Visualizations</h2>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-12 text-center text-slate-400">
          Load a dataset to view the embedding scatter plot.
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-4">
      {/* Header */}
      <div className="flex flex-shrink-0 items-center justify-between">
        <div className="flex items-center gap-3">
          <ChartScatter className="h-6 w-6 text-blue-500" />
          <h2 className="text-2xl font-bold text-slate-900">Visualizations</h2>
        </div>
        <div className="flex items-center gap-3 text-sm text-slate-500">
          {numCells > 0 && (
            <span>
              {formatNumber(numCells)} cells
            </span>
          )}
          {embedding && (
            <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-medium text-slate-600">
              {embedding.replace("X_", "").toUpperCase()}
            </span>
          )}
          {isLoading && (
            <span className="flex items-center gap-1 text-blue-500">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Loading...
            </span>
          )}
        </div>
      </div>

      {/* No embeddings banner */}
      {!hasEmbeddings && (
        <div className="flex flex-shrink-0 items-center gap-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          <AlertCircle className="h-5 w-5 flex-shrink-0 text-amber-500" />
          <div className="flex-1">
            <span className="font-medium">No embeddings available.</span>{" "}
            This dataset needs UMAP, tSNE, or PCA coordinates.
            Run the preprocessing pipeline in{" "}
            <button
              onClick={() => setPanel("assessment")}
              className="inline font-semibold text-primary underline decoration-primary/40 hover:decoration-primary"
            >
              Data Assessment
            </button>{" "}
            to compute them.
          </div>
          <button
            onClick={() => setPanel("assessment")}
            className="flex-shrink-0 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-white transition hover:bg-primary/90"
          >
            Go to Assessment
          </button>
        </div>
      )}

      {/* Error banner */}
      {error && hasEmbeddings && (
        <div className="flex flex-shrink-0 items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-2.5 text-sm text-red-700">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          <span className="truncate">{error}</span>
        </div>
      )}

      {/* Selection info bar */}
      {selectedIndices && selectedIndices.size > 0 && (
        <div className="flex flex-shrink-0 items-center justify-between rounded-lg border border-orange-200 bg-orange-50 px-4 py-2">
          <span className="text-sm font-medium text-orange-800">
            {formatNumber(selectedIndices.size)} of{" "}
            {formatNumber(numCells)} cells selected
          </span>
          <button
            onClick={clearSelection}
            className="flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-orange-600 transition-colors hover:bg-orange-100"
          >
            <X className="h-3 w-3" />
            Clear
          </button>
        </div>
      )}

      {/* Main content: scatter + controls */}
      <div className="flex min-h-0 flex-1 gap-4">
        {/* Scatter plot area — 70% */}
        <div className="relative min-w-0 flex-[7]">
          <Panel
            title={`${embedding ? embedding.replace(/^X_/, "").toUpperCase() : "Embedding"}${colorColumnName ? ` · ${colorColumnName}` : ""}`}
            bodyClassName="relative min-h-0 p-0"
          >
            <EmbeddingScatter
              positions={positions}
              colorValues={colors}
              colorType={colorType}
              pointSize={pointSize}
              opacity={opacity}
              selectedIndices={selectedIndices}
              background={plotBackground}
              maxRenderedCells={maxRenderedCells}
              onViewStateChange={setViewState}
              onHover={handleHover}
              dimensions={dimensions}
            />
            {dimensions === 2 && (
              <LassoSelector
                active={selectionMode === "lasso"}
                positions={positions}
                viewState={viewState}
                onSelectionComplete={handleSelectionComplete}
              />
            )}

            {/* Hover tooltip */}
            {hoveredCell && (
              <div
                className="pointer-events-none absolute z-20 max-w-xs rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs shadow-lg"
                style={{ left: hoveredCell.x + 14, top: hoveredCell.y - 14 }}
              >
                <div className="mb-1 font-semibold text-slate-700">
                  Cell {hoveredCell.index.toLocaleString()}
                </div>
                {hoveredCell.metadata ? (() => {
                  const isEmbeddingCol = (key: string) => {
                    const k = key.toLowerCase();
                    return k.startsWith("x_") ||
                      /^(umap|tsne|t-?sne|pca|phate|diffmap|draw_graph|harmony|pc)[-_]?\d*$/i.test(key);
                  };
                  const entries = Object.entries(hoveredCell.metadata).filter(([key]) => !isEmbeddingCol(key));
                  return (
                  <dl className="space-y-0.5">
                    {entries
                      .slice(0, 12)
                      .map(([key, val]) => (
                      <div key={key} className="flex gap-2">
                        <dt className="flex-shrink-0 text-slate-400">{key}:</dt>
                        <dd className="truncate font-medium text-slate-600">
                          {val == null ? "N/A" : typeof val === "number" ? val.toFixed(3) : String(val)}
                        </dd>
                      </div>
                    ))}
                    {entries.length > 12 && (
                      <div className="text-slate-400">
                        +{entries.length - 12} more
                      </div>
                    )}
                  </dl>
                  );
                })() : (
                  <div className="text-slate-400">Loading...</div>
                )}
              </div>
            )}
          </Panel>
        </div>

        {/* Controls panel — 30% */}
        <div className="flex w-64 flex-shrink-0 flex-col gap-4 overflow-y-auto">
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <h3 className="mb-3 text-sm font-semibold text-slate-700">
              Plot settings
            </h3>
            <PlotControls />
          </div>

          {/* Color legend */}
          {colorBy && (
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <h3 className="mb-3 text-sm font-semibold text-slate-700">
                Legend
              </h3>
              <ColorLegend
                type={colorType}
                categories={categories}
                categoryColors={categoryColors}
                min={colorMinMax.min}
                max={colorMinMax.max}
                label={colorColumnName ?? colorBy}
                onCategoryClick={(cat) => {
                  // Highlight cells of this category
                  if (!colors || !categories) return;
                  const catIndex = categories.indexOf(cat);
                  if (catIndex < 0) return;
                  const indices = new Set<number>();
                  for (let i = 0; i < colors.length; i++) {
                    if (colors[i] === catIndex) {
                      indices.add(i);
                    }
                  }
                  if (selectedIndices && selectedIndices.size === indices.size) {
                    // Toggle off if already selected
                    let same = true;
                    for (const idx of indices) {
                      if (!selectedIndices.has(idx)) {
                        same = false;
                        break;
                      }
                    }
                    if (same) {
                      clearSelection();
                      return;
                    }
                  }
                  setSelection(indices);
                }}
              />
            </div>
          )}

          {/* Dataset info summary */}
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <h3 className="mb-2 text-sm font-semibold text-slate-700">
              Dataset
            </h3>
            <dl className="space-y-1 text-xs">
              <div className="flex justify-between">
                <dt className="text-slate-500">Name</dt>
                <dd className="truncate pl-2 font-medium text-slate-700">
                  {dataset.name}
                </dd>
              </div>
              {dataset.n_cells != null && (
                <div className="flex justify-between">
                  <dt className="text-slate-500">Cells</dt>
                  <dd className="font-medium tabular-nums text-slate-700">
                    {dataset.n_cells.toLocaleString()}
                  </dd>
                </div>
              )}
              {dataset.n_genes != null && (
                <div className="flex justify-between">
                  <dt className="text-slate-500">Genes</dt>
                  <dd className="font-medium tabular-nums text-slate-700">
                    {dataset.n_genes.toLocaleString()}
                  </dd>
                </div>
              )}
              <div className="flex justify-between">
                <dt className="text-slate-500">Embeddings</dt>
                <dd className="font-medium text-slate-700">
                  {dataset.available_embeddings.length}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-500">Obs columns</dt>
                <dd className="font-medium text-slate-700">
                  {dataset.obs_columns.length}
                </dd>
              </div>
            </dl>
          </div>
        </div>
      </div>
    </div>
  );
}
