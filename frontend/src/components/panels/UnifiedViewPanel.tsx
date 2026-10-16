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
import { UnifiedDESubtab } from "@/components/unified/UnifiedDESubtab";
import { ClusterReference } from "@/components/unified/ClusterReference";
import { SplitScatterGrid } from "@/components/unified/SplitScatterGrid";
import { CompareColoringsGrid } from "@/components/unified/CompareColoringsGrid";
import { isGridActive, useSplitViewStore } from "@/stores/splitViewStore";
import type { ScatterLegend } from "@/lib/scatterExport";
import { mapCategoryToColor } from "@/lib/colors";
import { isCategoricalDtype } from "@/lib/obsColumns";
import { colorByLine, type HoverLine } from "@/lib/hoverInfo";
import { HoverLineRow } from "@/components/plots/HoverLineRow";
import { useCategoryColors } from "@/hooks/useCategoryColors";
import { labelsShown } from "@/lib/centroidLabels";
import { cameraFromViewState } from "@/lib/session";
import { prettyEmbeddingLabel, prettyObsLabel } from "@/lib/formatting";
import { apiFetch, apiFetchBinary } from "@/api/client";
import { fetchEmbeddingBinary } from "@/api/embeddings";
import { getDataset, renameObsCategory } from "@/api/datasets";
import { subclusterGroup } from "@/api/clustering";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { decodeArrowBuffer } from "@/lib/arrow";
import { useQueryClient } from "@tanstack/react-query";

const SUBTABS = [
  { id: "markers" as const, label: "Markers" },
  { id: "expression" as const, label: "Expression" },
  { id: "genesets" as const, label: "Gene Sets" },
  { id: "enrichment" as const, label: "Enrichment" },
  { id: "de" as const, label: "DE" },
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
  const paletteName = useSettingsStore((s) => s.categoricalPalette);
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

  // --- Split-by small multiples (SplitScatterGrid covers the single plot,
  // which stays mounted underneath so leaving the split shows it as it was) ---
  const splitBy = useSplitViewStore((s) => s.splitBy);
  const splitActive = !!(splitBy && datasetId && positions);
  // ...or the same cells coloured by 2-4 columns side by side (same machinery)
  const compareColumns = useSplitViewStore((s) => s.compareColumns);
  const compareActive = !!(compareColumns && datasetId && positions);
  const gridActive = splitActive || compareActive;

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

  // The main plot's camera, kept in the store for "Save session"; a camera from
  // "Open session" is shown once this dataset and embedding are on screen.
  const setCamera = useUnifiedViewStore((s) => s.setCamera);
  const handleCameraChange = useCallback(
    (vs: Record<string, unknown>, fitted: boolean) => {
      const camera = fitted ? null : cameraFromViewState(vs);
      setCamera(camera && datasetId ? { datasetId, embedding, camera } : null);
    },
    [datasetId, embedding, setCamera],
  );
  const pendingCamera = useUnifiedViewStore((s) => s.pendingCamera);
  const setPendingCamera = useUnifiedViewStore((s) => s.setPendingCamera);
  const restoreCamera =
    pendingCamera && pendingCamera.datasetId === datasetId && pendingCamera.embedding === embedding
      ? (pendingCamera.camera as unknown as Record<string, unknown>)
      : null;
  const handleCameraRestored = useCallback(() => {
    setViewState(restoreCamera); // the lasso and the cluster map follow it
    setPendingCamera(null);
  }, [restoreCamera, setPendingCamera]);

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
    return categories.map((_, i) => mapCategoryToColor(i, paletteName).slice(0, 3) as [number, number, number]);
  }, [categories, paletteName]);

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
  // `genes`: the set scored, so a saved session can score it again.
  const handleScoreComplete = useCallback(
    (scores: Float32Array, name: string, genes?: string[]) => {
      setOverlayValues(scores);
      setOverlayLabel(name);
      setScatterOverlay({ type: "score", name, genes });
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

  // --- An overlay from "Open session", drawn once this dataset is on screen ---
  const pendingOverlay = useUnifiedViewStore((s) => s.pendingOverlay);
  const setPendingOverlay = useUnifiedViewStore((s) => s.setPendingOverlay);
  useEffect(() => {
    if (!pendingOverlay || !datasetId || pendingOverlay.datasetId !== datasetId) return;
    const o = pendingOverlay.overlay;
    setPendingOverlay(null);
    if (o.type === "expression") {
      handleGeneClick(o.gene);
      return;
    }
    apiFetch<{ scores: number[] }>(`/datasets/${datasetId}/genesets/score`, {
      method: "POST",
      body: JSON.stringify({ gene_set: o.genes, score_name: o.name }),
    })
      .then((res) => handleScoreComplete(new Float32Array(res.scores), o.name, o.genes))
      .catch((err) => console.error("Failed to restore the gene-set overlay:", err));
  }, [pendingOverlay, datasetId, handleGeneClick, handleScoreComplete, setPendingOverlay]);

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
  // A lasso selection is not a group, so it drops any highlighted group.
  const handleSelectionComplete = useCallback(
    (indices: Set<number>) => {
      setHighlight(null);
      if (indices.size === 0) clearSelection();
      else setSelection(indices);
    },
    [setSelection, clearSelection, setHighlight],
  );

  // --- Keyboard shortcuts ---
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Don't intercept when typing in inputs
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if ((e.key === "l" || e.key === "L") && !isGridActive(useSplitViewStore.getState())) {
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
  // `lines`: the cell's value in each compared column, from the compare grid.
  const [hoveredCell, setHoveredCell] = useState<{
    x: number; y: number; index: number;
    metadata: Record<string, unknown> | null;
    lines?: HoverLine[];
  } | null>(null);
  const hoverDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleHover = useCallback(
    (info: { index: number; x: number; y: number; lines?: HoverLine[] } | null) => {
      if (hoverDebounceRef.current) {
        clearTimeout(hoverDebounceRef.current);
        hoverDebounceRef.current = null;
      }
      if (!info) { setHoveredCell(null); return; }
      setHoveredCell({ x: info.x, y: info.y, index: info.index, metadata: null, lines: info.lines });
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

  // The tooltip's first line: what the plot is coloured by, and the cell's value
  // (from the arrays the plot draws, so it needs no request). The column is the
  // one the colour buffer belongs to, which lags colorBy while a new one loads.
  const colorColumnInfo = useMemo(
    () => dataset?.obs_columns.find((c) => c.name === colorColumnName) ?? null,
    [dataset, colorColumnName],
  );
  const colorOfCategory = useCategoryColors(colorColumnName);

  // Group labels at their centroids, for a categorical colouring (no overlay):
  // on by default up to 40 groups, else as toggled in the toolbar.
  const labelChoice = useUnifiedViewStore((s) => s.centroidLabels);
  const labelCategories =
    !scatterOverlay && colorColumnInfo && isCategoricalDtype(colorColumnInfo.dtype)
      ? colorColumnInfo.values ?? null
      : null;
  const showLabels =
    !!labelCategories && !!colorColumnName && labelsShown(labelChoice, colorColumnName, labelCategories.length);
  const hoverColorLine = (index: number, metadata: Record<string, unknown> | null): HoverLine | null => {
    if (scatterOverlay && overlayValues) {
      return colorByLine({
        label: `${overlayLabel} ${scatterOverlay.type === "expression" ? "expression" : "score"}`,
        values: overlayValues,
        index,
        type: "continuous",
      });
    }
    if (!colorColumnName || !colors) return null;
    return colorByLine({
      label: prettyObsLabel(colorColumnName),
      values: colors,
      index,
      type: colorColumnInfo && !isCategoricalDtype(colorColumnInfo.dtype) ? "continuous" : "categorical",
      categories: colorColumnInfo?.values,
      colorOf: (_code, category) => colorOfCategory(category),
      fallback: metadata?.[colorColumnName],
    });
  };

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
        if (same) { clearSelection(); setHighlight(null); return; }
      }
      setSelection(indices);
      // Shared with the cluster map and the Markers / DE subtabs.
      if (colorBy) setHighlight({ column: colorBy, value: cat });
    },
    [colors, categories, selectedIndices, setSelection, clearSelection, colorBy, setHighlight],
  );

  // A group highlighted elsewhere (an Observations row, the co-pilot's "highlight
  // the NK cluster") selects its cells here once the plot is coloured by its
  // column, as a legend click does. Applied once per highlight.
  const highlightedGroup = useSelectionStore((s) => s.highlightedGroup);
  const appliedHighlight = useRef<typeof highlightedGroup>(null);
  useEffect(() => {
    const hl = highlightedGroup;
    if (!hl || hl === appliedHighlight.current) return;
    if (hl.column !== colorBy || colorColumnName !== colorBy || !colors || !categories) return;
    const catIndex = categories.indexOf(hl.value);
    if (catIndex < 0) return;
    const indices = new Set<number>();
    for (let i = 0; i < colors.length; i++) if (colors[i] === catIndex) indices.add(i);
    appliedHighlight.current = hl;
    setSelection(indices);
  }, [highlightedGroup, colorBy, colorColumnName, colors, categories, setSelection]);
  // ...and keeps its label while the others fade.
  const labelHighlight = highlightedGroup?.column === colorColumnName ? highlightedGroup.value : null;

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

  // Split one group into sub-clusters on its own graph (saved as <column>_sub),
  // then colour by the result.
  const refreshDataset = useDatasetRefresh();
  const [split, setSplit] = useState<{ cat: string; status: string; error?: boolean } | null>(null);
  const handleSubcluster = useCallback(
    async (cat: string) => {
      if (!datasetId || !colorBy) return;
      setSplit({ cat, status: `Splitting "${cat}" on its own graph…` });
      try {
        const res = await subclusterGroup(datasetId, colorBy, cat);
        await refreshDataset(datasetId, res.column);
        setHighlight(null);
        useSettingsStore.getState().setColorBy(res.column);
        setSplit({
          cat: "",
          status: `"${cat}" split into ${res.n_subclusters} (${res.groups.join(", ")}); colour: ${prettyObsLabel(res.column)}.`,
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message.replace(/^API error \d+: /, "") : String(err);
        let detail = msg;
        try { detail = String(JSON.parse(msg).detail ?? msg); } catch { /* plain text */ }
        setSplit({ cat: "", status: `Could not split "${cat}": ${detail}`, error: true });
      }
    },
    [datasetId, colorBy, refreshDataset, setHighlight],
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

  // "Download PNG" name + legend, shared by the single plot and the split grid
  const scatterExportName = `${prettyEmbeddingLabel(embedding || "embedding")}_${overlayLabel || colorColumnName || colorBy}`;
  const scatterExportLegend: ScatterLegend | null =
    effectiveColorType === "continuous" && effectiveColorValues
      ? { type: "continuous", title: overlayLabel || colorColumnName || colorBy, ...colorMinMax }
      : categories && categoryColors
        ? { type: "categorical", title: colorColumnName ?? colorBy, categories, colors: categoryColors }
        : null;

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
                restoreViewState={restoreCamera}
                onRestoreViewState={handleCameraRestored}
                cameraKey={`${datasetId}:${embedding}`}
                onCameraChange={handleCameraChange}
                onHover={handleHover}
                dimensions={dimensions}
                exportName={scatterExportName}
                exportLegend={scatterExportLegend}
                categories={labelCategories}
                showLabels={showLabels}
                highlightCategory={labelHighlight}
              />
            )}
            {dimensions === 2 && !gridActive && (
              <LassoSelector
                active={selectionMode === "lasso"}
                positions={positions}
                viewState={viewState}
                onSelectionComplete={handleSelectionComplete}
              />
            )}
            {splitBy && datasetId && positions && (
              <SplitScatterGrid
                datasetId={datasetId}
                embedding={embedding}
                column={splitBy}
                positions={positions}
                dimensions={dimensions}
                colorValues={effectiveColorValues}
                colorType={effectiveColorType}
                selectedIndices={selectedIndices}
                onHover={handleHover}
                exportName={scatterExportName}
                exportLegend={scatterExportLegend}
                labelCategories={labelCategories}
                showLabels={showLabels}
                highlightCategory={labelHighlight}
              />
            )}
            {compareColumns && datasetId && positions && (
              <CompareColoringsGrid
                datasetId={datasetId}
                embedding={embedding}
                columns={compareColumns}
                positions={positions}
                dimensions={dimensions}
                selectedIndices={selectedIndices}
                onHover={handleHover}
              />
            )}

            {/* Loading overlay indicator */}
            {isLoadingOverlay && (
              <div className="absolute right-12 top-3 z-10 flex items-center gap-1.5 rounded-md bg-white/90 px-2 py-1 text-xs text-blue-500 shadow">
                <Loader2 className="h-3 w-3 animate-spin" />
                Loading...
              </div>
            )}

            {/* Overlay chip stacked above the legend (bottom-left): one flex
                column, so a tall legend pushes the chip up instead of covering it.
                While split, it sits in the legend column the grid keeps free;
                the compare grid draws a legend per column there instead. */}
            {!compareActive && (
            <div className={`pointer-events-none absolute z-10 flex flex-col items-start gap-1.5 ${
              splitActive ? "right-2 top-10 max-h-[calc(100%-3rem)]" : "left-3 bottom-3 max-h-[calc(100%-1.5rem)]"
            }`}>
            {scatterOverlay && (
              <div className="pointer-events-auto flex flex-shrink-0 items-center gap-1.5 rounded-md border border-blue-200 bg-blue-50/90 px-2 py-1 text-xs text-blue-700">
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

            {/* Legend overlay */}
            <div className="pointer-events-auto max-h-48 min-h-0 max-w-[200px] overflow-y-auto rounded-lg bg-white/90 p-2 shadow">
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
                  onSubcluster={handleSubcluster}
                  splittingCategory={split?.cat || null}
                />
              ) : null}
              {split?.status && (
                <p
                  role={split.error ? "alert" : "status"}
                  className={`mt-1 max-w-[220px] text-[11px] ${split.error ? "text-red-600" : "text-slate-500"}`}
                >
                  {split.status}
                </p>
              )}
            </div>
            </div>
            )}

            {/* Hover tooltip: the colour-by value first, then the cell's metadata */}
            {hoveredCell && (() => {
              const compared = !!hoveredCell.lines;
              const colorLine = compared ? null : hoverColorLine(hoveredCell.index, hoveredCell.metadata);
              const lines = hoveredCell.lines ?? (colorLine ? [colorLine] : []);
              return (
              <div
                className="pointer-events-none absolute z-20 max-w-xs rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs shadow-lg"
                style={{ left: hoveredCell.x + 14, top: hoveredCell.y - 14 }}
              >
                {lines.length > 0 && (
                  <div className="mb-1 space-y-0.5">
                    {lines.map((line) => (
                      <HoverLineRow key={line.label} line={line} />
                    ))}
                  </div>
                )}
                <div className={`mb-1 ${lines.length > 0 ? "text-[11px] text-slate-400" : "font-semibold text-slate-700"}`}>
                  Cell {hoveredCell.index.toLocaleString()}
                </div>
                {hoveredCell.metadata ? (() => {
                  const isEmbeddingCol = (key: string) => {
                    const k = key.toLowerCase();
                    return k.startsWith("x_") ||
                      /^(umap|tsne|t-?sne|pca|phate|diffmap|draw_graph|harmony|pc)[-_]?\d*$/i.test(key);
                  };
                  // The colour-by (or compared) columns are already listed first.
                  const shown = new Set<string>(
                    compared
                      ? compareColumns ?? []
                      : colorLine && !(scatterOverlay && overlayValues) && colorColumnName
                        ? [colorColumnName]
                        : [],
                  );
                  const entries = Object.entries(hoveredCell.metadata).filter(
                    ([key]) => !isEmbeddingCol(key) && !shown.has(key),
                  );
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
              );
            })()}
            </div>
          </div>
          {/* Distribution — matches the UMAP width; frees the right column to extend down */}
          <UnifiedBottomPanel
            violinData={violinData}
            violinTitle={violinTitle}
            violinGroupLabel={groupByColumn ? prettyObsLabel(groupByColumn) : groupByColumn}
            violinGroupColumn={groupByColumn}
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
                <InfoRow label="Embedding" value={prettyEmbeddingLabel(embedding)} />
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
            {activeSubtab === "de" && (
              <UnifiedDESubtab onGeneClick={handleGeneClick} onScoreGeneSet={handleScoreComplete} />
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
