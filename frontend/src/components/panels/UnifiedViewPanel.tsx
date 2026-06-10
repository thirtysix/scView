import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LayoutDashboard, AlertCircle, Loader2 } from "lucide-react";
import { useEmbedding } from "@/hooks/useEmbedding";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useViewStore } from "@/stores/viewStore";
import { EmbeddingScatter } from "@/components/plots/EmbeddingScatter";
import { ColorLegend } from "@/components/plots/ColorLegend";
import { LassoSelector } from "@/components/plots/LassoSelector";
import { UnifiedToolbar } from "@/components/unified/UnifiedToolbar";
import { UnifiedBottomPanel } from "@/components/unified/UnifiedBottomPanel";
import { UnifiedMarkersSubtab } from "@/components/unified/UnifiedMarkersSubtab";
import { UnifiedExpressionSubtab } from "@/components/unified/UnifiedExpressionSubtab";
import { UnifiedGeneSetsSubtab } from "@/components/unified/UnifiedGeneSetsSubtab";
import { UnifiedEnrichmentSubtab } from "@/components/unified/UnifiedEnrichmentSubtab";
import { ClusterReference } from "@/components/unified/ClusterReference";
import { mapCategoryToColor } from "@/lib/colors";
import { prettyObsLabel } from "@/lib/formatting";
import { apiFetch, apiFetchBinary } from "@/api/client";
import { fetchEmbeddingBinary } from "@/api/embeddings";
import { getDataset, renameObsCategory } from "@/api/datasets";
import { decodeArrowBuffer } from "@/lib/arrow";
import { useQueryClient } from "@tanstack/react-query";

const SUBTABS = [
  { id: "markers" as const, label: "Markers" },
  { id: "expression" as const, label: "Expression" },
  { id: "genesets" as const, label: "Gene Sets" },
  { id: "enrichment" as const, label: "Enrichment" },
] as const;

interface ViolinResponse {
  gene: string;
  groups: Record<string, number[]>;
}

export function UnifiedViewPanel() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const setCurrentDataset = useDatasetStore((s) => s.setCurrentDataset);
  const queryClient = useQueryClient();
  const pointSize = useSettingsStore((s) => s.pointSize);
  const opacity = useSettingsStore((s) => s.opacity);
  const plotBackground = useSettingsStore((s) => s.plotBackground);
  const maxRenderedCells = useSettingsStore((s) => s.maxRenderedCells);
  const colorBy = useSettingsStore((s) => s.colorBy);
  const embedding = useSettingsStore((s) => s.embedding);
  const expressionLayer = useSettingsStore((s) => s.expressionLayer);
  const setExpressionLayer = useSettingsStore((s) => s.setExpressionLayer);
  const selectedIndices = useSelectionStore((s) => s.selectedCellIndices);
  const selectionMode = useSelectionStore((s) => s.selectionMode);
  const setSelection = useSelectionStore((s) => s.setSelection);
  const clearSelection = useSelectionStore((s) => s.clearSelection);
  const setHighlight = useSelectionStore((s) => s.setHighlight);
  const setPanel = useViewStore((s) => s.setPanel);
  const askCopilot = useViewStore((s) => s.askCopilot);

  const activeSubtab = useUnifiedViewStore((s) => s.activeSubtab);
  const setActiveSubtab = useUnifiedViewStore((s) => s.setActiveSubtab);
  const scatterOverlay = useUnifiedViewStore((s) => s.scatterOverlay);
  const setScatterOverlay = useUnifiedViewStore((s) => s.setScatterOverlay);
  const openBottomPanel = useUnifiedViewStore((s) => s.openBottomPanel);
  const groupByColumn = useUnifiedViewStore((s) => s.groupByColumn);
  const setGroupByColumn = useUnifiedViewStore((s) => s.setGroupByColumn);

  // Overlay + violin payload live in the store so they survive the
  // unmount/remount PanelContainer does on every tab switch.
  const overlayValues = useUnifiedViewStore((s) => s.overlayValues);
  const setOverlayValues = useUnifiedViewStore((s) => s.setOverlayValues);
  const overlayLabel = useUnifiedViewStore((s) => s.overlayLabel);
  const setOverlayLabel = useUnifiedViewStore((s) => s.setOverlayLabel);
  const setOverlayDatasetId = useUnifiedViewStore((s) => s.setOverlayDatasetId);
  const overlayDatasetId = useUnifiedViewStore((s) => s.overlayDatasetId);
  const violinData = useUnifiedViewStore((s) => s.violinData);
  const setViolinData = useUnifiedViewStore((s) => s.setViolinData);
  const violinTitle = useUnifiedViewStore((s) => s.violinTitle);
  const setViolinTitle = useUnifiedViewStore((s) => s.setViolinTitle);
  const clearOverlayState = useUnifiedViewStore((s) => s.clearOverlayState);

  // --- Embedding data (obs-colored) ---
  const { positions, colors, numCells, colorColumnName, dimensions, isLoading, error } =
    useEmbedding();

  // --- Overlay loading flag (transient; values/label persist in the store) ---
  const [isLoadingOverlay, setIsLoadingOverlay] = useState(false);

  // --- Violin loading flag (transient; data/title persist in the store) ---
  const [isLoadingViolin, setIsLoadingViolin] = useState(false);

  // --- Auto-select embedding ---
  const setEmbedding = useSettingsStore((s) => s.setEmbedding);
  const setPointSize = useSettingsStore((s) => s.setPointSize);
  const setOpacity = useSettingsStore((s) => s.setOpacity);

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

  useEffect(() => {
    if (!dataset) return;
    const available = dataset.available_embeddings;
    if (available.length === 0) {
      setEmbedding("");
    } else if (!available.includes(embedding)) {
      const preferred = ["X_umap", "X_tsne", "X_pca"];
      const pick = preferred.find((e) => available.includes(e)) ?? available[0]!;
      setEmbedding(pick);
    }
  }, [dataset, embedding, setEmbedding]);

  // --- Drop a persisted overlay if it belongs to a different dataset ---
  // (overlay payload lives in the store, so a dataset switch could otherwise
  // paint the previous dataset's values onto the new one.)
  useEffect(() => {
    if (!datasetId) return;
    if (overlayDatasetId && overlayDatasetId !== datasetId) {
      clearOverlayState();
    }
  }, [datasetId, overlayDatasetId, clearOverlayState]);

  // --- Auto-select groupBy column ---
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

  useEffect(() => {
    if (groupByColumn || categoricalColumns.length === 0) return;
    if (dataset?.active_clustering) {
      const found = categoricalColumns.find((c) => c.name === dataset.active_clustering);
      if (found) { setGroupByColumn(found.name); return; }
    }
    const clusterPatterns = ["leiden", "louvain", "cluster", "seurat_clusters", "cell_type", "celltype"];
    const match = categoricalColumns.find((c) =>
      clusterPatterns.some((p) => c.name.toLowerCase().includes(p)),
    );
    setGroupByColumn(match?.name ?? categoricalColumns[0]!.name);
  }, [categoricalColumns, groupByColumn, dataset, setGroupByColumn]);

  // --- View state for lasso ---
  const [viewState, setViewState] = useState<Record<string, unknown> | null>(null);

  // --- Resizable split pane ---
  const [splitFraction, setSplitFraction] = useState(0.6); // left panel fraction (0.3–0.8)
  const splitContainerRef = useRef<HTMLDivElement>(null);
  const isDraggingRef = useRef(false);

  const handleDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDraggingRef.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    const onMove = (ev: MouseEvent) => {
      if (!isDraggingRef.current || !splitContainerRef.current) return;
      const rect = splitContainerRef.current.getBoundingClientRect();
      const fraction = (ev.clientX - rect.left) / rect.width;
      setSplitFraction(Math.min(0.8, Math.max(0.3, fraction)));
    };

    const onUp = () => {
      isDraggingRef.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, []);

  // --- Color type from obs column ---
  const colorType = useMemo<"categorical" | "continuous">(() => {
    if (scatterOverlay?.type === "expression" || scatterOverlay?.type === "score")
      return "continuous";
    if (!colorBy || !dataset) return "categorical";
    const col = dataset.obs_columns.find((c) => c.name === colorBy);
    if (!col) return "categorical";
    if (col.dtype === "category" || col.dtype === "object" || col.dtype === "bool")
      return "categorical";
    return "continuous";
  }, [colorBy, dataset, scatterOverlay]);

  // --- Category data for legend ---
  const obsColumn = useMemo(() => {
    if (!colorBy || !dataset) return null;
    return dataset.obs_columns.find((c) => c.name === colorBy) ?? null;
  }, [colorBy, dataset]);

  const categories = useMemo(() => {
    if (scatterOverlay?.type === "expression" || scatterOverlay?.type === "score") return undefined;
    if (colorType !== "categorical" || !obsColumn?.values) return undefined;
    return obsColumn.values;
  }, [colorType, obsColumn, scatterOverlay]);

  const categoryColors = useMemo<[number, number, number][] | undefined>(() => {
    if (!categories) return undefined;
    // ColorLegend wants RGB; mapCategoryToColor returns RGBA — drop the alpha.
    return categories.map((_, i) => mapCategoryToColor(i).slice(0, 3) as [number, number, number]);
  }, [categories]);

  // --- Determine what to pass to scatter ---
  const effectiveColorValues = scatterOverlay ? overlayValues : colors;
  const effectiveColorType = scatterOverlay ? "continuous" : colorType;

  // --- Min/max for continuous legend ---
  const colorMinMax = useMemo(() => {
    const vals = effectiveColorType === "continuous" ? effectiveColorValues : null;
    if (!vals) return { min: 0, max: 1 };
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i]!;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    return { min, max };
  }, [effectiveColorType, effectiveColorValues]);

  // --- Fetch expression ---
  const fetchExpression = useCallback(
    async (gene: string) => {
      if (!datasetId) return;
      setIsLoadingOverlay(true);
      try {
        const layerQuery = expressionLayer
          ? `&layer=${encodeURIComponent(expressionLayer)}`
          : "";
        const buffer = await apiFetchBinary(
          `/datasets/${datasetId}/expression?genes=${encodeURIComponent(gene)}${layerQuery}`,
        );
        const decoded = decodeArrowBuffer(buffer);
        const values = decoded[gene] ?? decoded[Object.keys(decoded)[0]!];
        if (values instanceof Float32Array) {
          setOverlayValues(values);
        } else if (values) {
          const f32 = new Float32Array(values.length);
          for (let i = 0; i < values.length; i++) f32[i] = Number(values[i]);
          setOverlayValues(f32);
        }
        setOverlayLabel(gene);
        setScatterOverlay({ type: "expression", gene });
        setOverlayDatasetId(datasetId);
      } catch (err) {
        console.error("Failed to fetch expression:", err);
      } finally {
        setIsLoadingOverlay(false);
      }
    },
    [datasetId, expressionLayer, setScatterOverlay, setOverlayValues, setOverlayLabel, setOverlayDatasetId],
  );

  // --- Fetch violin ---
  const fetchViolin = useCallback(
    async (gene: string, group: string) => {
      if (!datasetId || !group) return;
      setIsLoadingViolin(true);
      try {
        const layerQuery = expressionLayer
          ? `&layer=${encodeURIComponent(expressionLayer)}`
          : "";
        const data = await apiFetch<ViolinResponse>(
          `/datasets/${datasetId}/expression/violin?gene=${encodeURIComponent(gene)}&groupby=${encodeURIComponent(group)}${layerQuery}`,
        );
        setViolinData(data.groups ?? {});
        setViolinTitle(`${gene} expression`);
        openBottomPanel();
      } catch {
        setViolinData({});
      } finally {
        setIsLoadingViolin(false);
      }
    },
    [datasetId, expressionLayer, openBottomPanel, setViolinData, setViolinTitle],
  );

  // --- Gene click handler (shared across Markers + Expression subtabs) ---
  const handleGeneClick = useCallback(
    (gene: string) => {
      fetchExpression(gene);
      if (groupByColumn) fetchViolin(gene, groupByColumn);
    },
    [fetchExpression, fetchViolin, groupByColumn],
  );

  // Default the expression layer to the dataset's default; re-default when the
  // current layer key isn't valid for the (possibly newly selected) dataset.
  useEffect(() => {
    if (!dataset) return;
    const keys = (dataset.expression_layers ?? []).map((l) => l.key);
    if (!expressionLayer || !keys.includes(expressionLayer)) {
      setExpressionLayer(dataset.default_expression_layer ?? keys[0] ?? "");
    }
  }, [dataset, expressionLayer, setExpressionLayer]);

  // Re-fetch the active gene overlay + violin when the expression layer changes.
  const prevLayerRef = useRef(expressionLayer);
  useEffect(() => {
    if (prevLayerRef.current === expressionLayer) return;
    prevLayerRef.current = expressionLayer;
    if (scatterOverlay?.type === "expression" && overlayLabel) {
      fetchExpression(overlayLabel);
      if (groupByColumn) fetchViolin(overlayLabel, groupByColumn);
    }
  }, [expressionLayer]); // eslint-disable-line react-hooks/exhaustive-deps

  // --- Gene set score handler ---
  const handleScoreComplete = useCallback(
    (scores: Float32Array, name: string) => {
      setOverlayValues(scores);
      setOverlayLabel(name);
      setScatterOverlay({ type: "score", name });
      if (datasetId) setOverlayDatasetId(datasetId);

      // Build violin from scores grouped by the active groupBy column
      if (groupByColumn && dataset && datasetId) {
        // We need to fetch the groupBy column values to build the violin
        fetchEmbeddingBinary(datasetId, useSettingsStore.getState().embedding, groupByColumn)
          .then((buffer) => {
            const decoded = decodeArrowBuffer(buffer);
            const groupValues = decoded[groupByColumn];
            if (groupValues) {
              const groups: Record<string, number[]> = {};
              for (let i = 0; i < scores.length && i < groupValues.length; i++) {
                const groupIdx = groupValues[i]!;
                const obsCol = dataset.obs_columns.find((c) => c.name === groupByColumn);
                const groupName = obsCol?.values?.[groupIdx] ?? String(groupIdx);
                if (!groups[groupName]) groups[groupName] = [];
                groups[groupName].push(scores[i]!);
              }
              setViolinData(groups);
              setViolinTitle(`${name} score`);
              openBottomPanel();
            }
          })
          .catch(() => { /* ignore */ });
      }
    },
    [setScatterOverlay, setOverlayValues, setOverlayLabel, setOverlayDatasetId,
     setViolinData, setViolinTitle, groupByColumn, dataset, datasetId, openBottomPanel],
  );

  // --- Clear overlay (revert to obs coloring) ---
  const handleClearOverlay = useCallback(() => {
    setScatterOverlay(null);
    setOverlayValues(null);
    setOverlayLabel("");
  }, [setScatterOverlay, setOverlayValues, setOverlayLabel]);

  // --- Clear overlay when colorBy changes in toolbar ---
  const prevColorByRef = useRef(colorBy);
  useEffect(() => {
    if (prevColorByRef.current !== colorBy) {
      prevColorByRef.current = colorBy;
      if (scatterOverlay) {
        handleClearOverlay();
      }
    }
  }, [colorBy]); // eslint-disable-line react-hooks/exhaustive-deps

  // --- Lasso handler ---
  const handleSelectionComplete = useCallback(
    (indices: Set<number>) => {
      if (indices.size === 0) clearSelection();
      else setSelection(indices);
    },
    [setSelection, clearSelection],
  );

  // --- Keyboard shortcuts ---
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Don't intercept when typing in inputs
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.key === "l" || e.key === "L") {
        const store = useSelectionStore.getState();
        store.setSelectionMode(store.selectionMode === "lasso" ? "none" : "lasso");
      }
      if (e.key === "Escape") {
        clearSelection();
        handleClearOverlay();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [clearSelection, handleClearOverlay]);

  // --- Hover tooltip ---
  const [hoveredCell, setHoveredCell] = useState<{
    x: number; y: number; index: number;
    metadata: Record<string, unknown> | null;
  } | null>(null);
  const hoverDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleHover = useCallback(
    (info: { index: number; x: number; y: number } | null) => {
      if (hoverDebounceRef.current) {
        clearTimeout(hoverDebounceRef.current);
        hoverDebounceRef.current = null;
      }
      if (!info) { setHoveredCell(null); return; }
      setHoveredCell({ x: info.x, y: info.y, index: info.index, metadata: null });
      if (datasetId) {
        hoverDebounceRef.current = setTimeout(() => {
          apiFetch<Record<string, unknown>>(
            `/datasets/${datasetId}/metadata/cell/${info.index}`,
          ).then((data) => {
            setHoveredCell((prev) =>
              prev && prev.index === info.index ? { ...prev, metadata: data } : prev,
            );
          }).catch(() => {});
        }, 50);
      }
    },
    [datasetId],
  );

  useEffect(() => {
    return () => { if (hoverDebounceRef.current) clearTimeout(hoverDebounceRef.current); };
  }, []);

  // --- Legend click (category selection) ---
  const handleCategoryClick = useCallback(
    (cat: string) => {
      if (!colors || !categories) return;
      const catIndex = categories.indexOf(cat);
      if (catIndex < 0) return;
      const indices = new Set<number>();
      for (let i = 0; i < colors.length; i++) {
        if (colors[i] === catIndex) indices.add(i);
      }
      if (selectedIndices && selectedIndices.size === indices.size) {
        let same = true;
        for (const idx of indices) {
          if (!selectedIndices.has(idx)) { same = false; break; }
        }
        if (same) { clearSelection(); return; }
      }
      setSelection(indices);
    },
    [colors, categories, selectedIndices, setSelection, clearSelection],
  );

  // Inline-rename a categorical label (e.g. correct a cell-type annotation).
  // Persists to the derived layer, then refreshes the dataset metadata + the
  // embedding color buffer so the legend and plot pick up the new label.
  const handleRenameCategory = useCallback(
    async (cat: string, newName: string) => {
      if (!datasetId || !colorBy) return;
      try {
        await renameObsCategory(datasetId, colorBy, cat, newName);
        const fresh = await getDataset(datasetId);
        setCurrentDataset(fresh);
        await queryClient.invalidateQueries({ queryKey: ["embedding", datasetId] });
      } catch (err) {
        console.error("Failed to rename category:", err);
      }
    },
    [datasetId, colorBy, setCurrentDataset, queryClient],
  );

  // "Ask about this" — highlight the group and pose a contextual question to the
  // co-pilot. The question embeds the specifics so it resolves even without view context.
  const handleAskAboutCategory = useCallback(
    (cat: string) => {
      if (!colorBy) return;
      setHighlight({ column: colorBy, value: cat });
      const label = prettyObsLabel(colorBy);
      askCopilot(
        `What is the "${cat}" group in ${label}? Summarize its marker genes and likely cell-type identity.`,
      );
    },
    [colorBy, setHighlight, askCopilot],
  );

  const handleAskAboutOverlay = useCallback(() => {
    const gene = overlayLabel;
    if (!gene) return;
    askCopilot(
      `What is ${gene}, and what does its expression pattern across these cells suggest?`,
    );
  }, [overlayLabel, askCopilot]);

  const hasEmbeddings = dataset != null && dataset.available_embeddings.length > 0;

  // No dataset
  if (!dataset) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <LayoutDashboard className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">Unified View</h2>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-12 text-center text-slate-400">
          Load a dataset to explore in the unified view.
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-2">
      {/* No embeddings warning */}
      {!hasEmbeddings && (
        <div className="flex flex-shrink-0 items-center gap-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800">
          <AlertCircle className="h-4 w-4 flex-shrink-0 text-amber-500" />
          <span>No embeddings available. Run the pipeline in{" "}
            <button onClick={() => setPanel("assessment")} className="font-semibold text-primary underline">
              Data Assessment
            </button>.
          </span>
        </div>
      )}

      {/* Error */}
      {error && hasEmbeddings && (
        <div className="flex flex-shrink-0 items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          <span className="truncate">{error}</span>
        </div>
      )}

      {/* Main area: left column (toolbar + scatter + distribution) | right panel */}
      <div ref={splitContainerRef} className="flex min-h-0 flex-1">
        {/* Left column — UMAP width: toolbar, scatter, and distribution stacked */}
        <div className="flex min-w-0 flex-col gap-2" style={{ width: `${splitFraction * 100}%` }}>
          <UnifiedToolbar numCells={numCells} />
          <div className="relative min-h-0 flex-1">
            <div className="h-full w-full rounded-xl border border-slate-200 bg-white shadow-sm">
            {isLoading && !positions ? (
              <div className="flex h-full items-center justify-center">
                <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
              </div>
            ) : (
              <EmbeddingScatter
                positions={positions}
                colorValues={effectiveColorValues}
                colorType={effectiveColorType}
                pointSize={pointSize}
                opacity={opacity}
                selectedIndices={selectedIndices}
                background={plotBackground}
                maxRenderedCells={maxRenderedCells}
                externalViewState={viewState}
                onViewStateChange={setViewState}
                onHover={handleHover}
                dimensions={dimensions}
              />
            )}
            {dimensions === 2 && (
              <LassoSelector
                active={selectionMode === "lasso"}
                positions={positions}
                viewState={viewState}
                onSelectionComplete={handleSelectionComplete}
              />
            )}

            {/* Overlay indicator */}
            {scatterOverlay && (
              <div className="absolute left-3 bottom-14 z-10 flex items-center gap-1.5 rounded-md border border-blue-200 bg-blue-50/90 px-2 py-1 text-xs text-blue-700">
                <span className="font-medium">{overlayLabel}</span>
                <span className="text-blue-500">
                  {scatterOverlay.type === "expression" ? "expression" : "score"}
                </span>
                <button
                  onClick={handleClearOverlay}
                  className="ml-1 rounded px-1 text-blue-400 hover:bg-blue-100 hover:text-blue-600"
                >
                  Clear
                </button>
              </div>
            )}

            {/* Loading overlay indicator */}
            {isLoadingOverlay && (
              <div className="absolute right-12 top-3 z-10 flex items-center gap-1.5 rounded-md bg-white/90 px-2 py-1 text-xs text-blue-500 shadow">
                <Loader2 className="h-3 w-3 animate-spin" />
                Loading...
              </div>
            )}

            {/* Legend overlay (bottom-left) */}
            <div className="absolute left-3 bottom-3 z-10 max-h-48 max-w-[200px] overflow-y-auto rounded-lg bg-white/90 p-2 shadow">
              {effectiveColorType === "continuous" && overlayValues ? (
                <ColorLegend
                  type="continuous"
                  min={colorMinMax.min}
                  max={colorMinMax.max}
                  label={overlayLabel || colorColumnName || colorBy}
                  onAsk={
                    scatterOverlay?.type === "expression" && overlayLabel
                      ? handleAskAboutOverlay
                      : undefined
                  }
                />
              ) : colorBy && categories ? (
                <ColorLegend
                  type="categorical"
                  categories={categories}
                  categoryColors={categoryColors}
                  label={colorColumnName ?? colorBy}
                  onCategoryClick={handleCategoryClick}
                  onRenameCategory={handleRenameCategory}
                  onAskAbout={handleAskAboutCategory}
                />
              ) : null}
            </div>

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
                      {entries.slice(0, 8).map(([key, val]) => (
                        <div key={key} className="flex gap-2">
                          <dt className="flex-shrink-0 text-slate-400">{key}:</dt>
                          <dd className="truncate font-medium text-slate-600">
                            {val == null ? "N/A" : typeof val === "number" ? val.toFixed(3) : String(val)}
                          </dd>
                        </div>
                      ))}
                      {entries.length > 8 && (
                        <div className="text-slate-400">+{entries.length - 8} more</div>
                      )}
                    </dl>
                  );
                })() : (
                  <div className="text-slate-400">Loading...</div>
                )}
              </div>
            )}
            </div>
          </div>
          {/* Distribution — matches the UMAP width; frees the right column to extend down */}
          <UnifiedBottomPanel
            violinData={violinData}
            violinTitle={violinTitle}
            violinGroupLabel={groupByColumn ? prettyObsLabel(groupByColumn) : groupByColumn}
            isLoading={isLoadingViolin}
          />
        </div>

        {/* Drag handle */}
        <div
          onMouseDown={handleDragStart}
          className="group flex w-2 flex-shrink-0 cursor-col-resize items-center justify-center"
        >
          <div className="h-8 w-1 rounded-full bg-slate-300 transition-colors group-hover:bg-primary group-active:bg-primary" />
        </div>

        {/* Right: tabbed side panel */}
        <div className="flex min-w-[260px] flex-1 flex-col rounded-xl border border-slate-200 bg-white shadow-sm">
          {/* Cluster reference map (half width) + summary card — keeps cluster
              layout + key facts visible while the main plot is recoloured. */}
          {groupByColumn && positions && datasetId && (
            <div className="flex flex-shrink-0 items-start gap-2 p-2">
              <div className="w-1/2 min-w-0">
                <ClusterReference
                  datasetId={datasetId}
                  embedding={embedding}
                  column={groupByColumn}
                  positions={positions}
                  dimensions={dimensions}
                  categories={
                    dataset?.obs_columns.find((c) => c.name === groupByColumn)?.values ?? []
                  }
                  viewState={viewState}
                  onViewStateChange={setViewState}
                />
              </div>
              <div className="flex w-1/2 min-w-0 flex-col gap-1.5 rounded-lg border border-slate-200 bg-slate-50/60 p-2.5 text-xs">
                <div className="text-[11px] font-semibold text-slate-600">Summary</div>
                <InfoRow label="Cells" value={numCells.toLocaleString()} />
                <InfoRow label="Grouping" value={prettyObsLabel(groupByColumn)} />
                <InfoRow
                  label="Groups"
                  value={String(
                    dataset?.obs_columns.find((c) => c.name === groupByColumn)?.values?.length ?? 0,
                  )}
                />
                <InfoRow label="Embedding" value={embedding.replace(/^X_/, "").toUpperCase()} />
                <InfoRow
                  label="Overlay"
                  value={scatterOverlay ? `${overlayLabel} (${scatterOverlay.type})` : "—"}
                />
              </div>
            </div>
          )}
          {/* Tab bar */}
          <div className="flex flex-shrink-0 border-b border-slate-200">
            {SUBTABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveSubtab(tab.id)}
                className={`flex-1 px-2 py-2 text-xs font-medium transition-colors ${
                  activeSubtab === tab.id
                    ? "border-b-2 border-primary text-primary"
                    : "text-slate-500 hover:text-slate-700"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-y-auto p-3">
            {activeSubtab === "markers" && (
              <UnifiedMarkersSubtab
                onGeneClick={handleGeneClick}
                groupByColumn={groupByColumn}
                setGroupByColumn={setGroupByColumn}
              />
            )}
            {activeSubtab === "expression" && (
              <UnifiedExpressionSubtab
                onGeneSelect={handleGeneClick}
                activeGene={scatterOverlay?.type === "expression" ? overlayLabel : null}
                groupByColumn={groupByColumn}
                setGroupByColumn={setGroupByColumn}
                expressionMinMax={scatterOverlay?.type === "expression" ? colorMinMax : null}
              />
            )}
            {activeSubtab === "genesets" && (
              <UnifiedGeneSetsSubtab
                onScoreComplete={handleScoreComplete}
                groupByColumn={groupByColumn}
                setGroupByColumn={setGroupByColumn}
              />
            )}
            {activeSubtab === "enrichment" && (
              <UnifiedEnrichmentSubtab
                onTermClick={handleGeneClick}
                onScoreGeneSet={handleScoreComplete}
                groupByColumn={groupByColumn}
                setGroupByColumn={setGroupByColumn}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-slate-400">{label}</span>
      <span className="truncate font-medium text-slate-600" title={value}>
        {value}
      </span>
    </div>
  );
}
