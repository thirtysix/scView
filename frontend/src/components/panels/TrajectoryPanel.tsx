import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { GitBranch, Loader2, AlertCircle, Play, X } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { plotConfig } from "@/lib/plotConfig";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useEmbedding } from "@/hooks/useEmbedding";
import { apiFetch, apiFetchBinary } from "@/api/client";
import { getDataset } from "@/api/datasets";
import { computePaga, getPaga, rankTrendGenes } from "@/api/trajectory";
import { EmbeddingScatter } from "@/components/plots/EmbeddingScatter";
import { Panel } from "@/components/common/Panel";
import { GeneSearch } from "@/components/common/GeneSearch";
import { ColorLegend } from "@/components/plots/ColorLegend";
import { PagaCard } from "@/components/trajectory/PagaCard";
import { TrendGenesView } from "@/components/trajectory/TrendGenesView";
import { decodeArrowBuffer } from "@/lib/arrow";
import { formatNumber, prettyObsLabel } from "@/lib/formatting";
import { DEFAULT_PAGA_THRESHOLD, branchEnds as pagaBranchEnds, pagaLayers, treePath, visibleEdges } from "@/lib/paga";
import { DPT_COLUMN, defaultRootColumn, rootGroupColumns } from "@/lib/trajectory";

/** How the in-app pseudotime column was computed (POST /trajectory/dpt). */
interface DPTParams {
  column: string;
  root_column: string;
  root_value: string;
  root_cell: string;
  graph: string;
  n_unreachable: number;
}

interface TrajectoryColumnsResponse {
  pseudotime_columns: string[];
  n_columns: number;
  computed?: DPTParams | null;
  /** The dataset has spliced + unspliced layers (RNA velocity input) */
  velocity_layers?: boolean;
}

interface PseudotimeValuesResponse {
  column: string;
  n_cells: number;
  values: (number | null)[];
  min: number;
  max: number;
}

interface GeneAlongPseudotime {
  found: boolean;
  scatter_pseudotime?: number[];
  scatter_expression?: number[];
  binned_pseudotime?: number[];
  /** null for a bin no cell falls in: drawn as a gap */
  binned_expression?: (number | null)[];
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
  const highlighted = useSelectionStore((s) => s.highlightedGroup);
  const setCurrentDataset = useDatasetStore((s) => s.setCurrentDataset);
  const embedding = useSettingsStore((s) => s.embedding);
  const queryClient = useQueryClient();

  const { positions, numCells, dimensions, isLoading: embeddingLoading } = useEmbedding();

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

  // In-app pseudotime: a root group picked from a categorical column.
  const [computed, setComputed] = useState<DPTParams | null>(null);
  const [velocityLayers, setVelocityLayers] = useState(false);
  // Bumped after pseudotime is (re)computed, so columns, values and trends refetch.
  const [refreshKey, setRefreshKey] = useState(0);
  const [isComputing, setIsComputing] = useState(false);
  const [computeError, setComputeError] = useState<string | null>(null);
  const rootCols = useMemo(() => rootGroupColumns(dataset?.obs_columns), [dataset]);
  const [rootColumn, setRootColumn] = useState("");
  const [rootValue, setRootValue] = useState("");
  const rootValues = useMemo(
    () => rootCols.find((c) => c.name === rootColumn)?.values ?? [],
    [rootCols, rootColumn],
  );

  useEffect(() => {
    if (!rootCols.some((c) => c.name === rootColumn)) {
      setRootColumn(defaultRootColumn(rootCols, dataset?.active_clustering));
    }
  }, [rootCols, rootColumn, dataset]);

  useEffect(() => {
    if (!rootValues.includes(rootValue)) setRootValue(rootValues[0] ?? "");
  }, [rootValues, rootValue]);

  // A group clicked in a legend elsewhere becomes the suggested root.
  useEffect(() => {
    if (highlighted && rootCols.some((c) => c.name === highlighted.column)) {
      setRootColumn(highlighted.column);
      setRootValue(highlighted.value);
    }
  }, [highlighted, rootCols]);

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
        setComputed(data.computed ?? null);
        setVelocityLayers(!!data.velocity_layers);
        // Keep the current column if this dataset has it (after a recompute it does).
        setSelectedColumn((prev) =>
          data.pseudotime_columns.includes(prev) ? prev : (data.pseudotime_columns[0] ?? ""),
        );
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : String(err));
        setAvailableColumns([]);
      })
      .finally(() => setIsLoadingColumns(false));
  }, [datasetId, refreshKey]);

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
          f32[i] = data.values[i] ?? NaN; // null: no pseudotime for this cell
        }
        setPseudotimeValues(f32);
        setPseudotimeMinMax({ min: data.min, max: data.max });
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : String(err));
        setPseudotimeValues(null);
      })
      .finally(() => setIsLoadingPseudotime(false));
  }, [datasetId, selectedColumn, refreshKey]);

  // Plotted genes follow the pseudotime: refetch their trends when it changes.
  const genesRef = useRef<string[]>([]);
  genesRef.current = selectedGenes;
  useEffect(() => {
    const genes = genesRef.current;
    if (!datasetId || !selectedColumn || genes.length === 0) return;
    apiFetch<TrajectoryGenesResponse>(
      `/datasets/${datasetId}/trajectory/${encodeURIComponent(selectedColumn)}/genes?genes=${encodeURIComponent(genes.join(","))}`,
    )
      .then((data) => setGeneTrajectories(data.genes))
      .catch(() => setGeneTrajectories({}));
  }, [datasetId, selectedColumn, refreshKey]);

  // Compute diffusion pseudotime from the chosen root group, then show it.
  const computePseudotime = useCallback(async () => {
    if (!datasetId || !rootColumn || !rootValue) return;
    setIsComputing(true);
    setComputeError(null);
    try {
      const res = await apiFetch<DPTParams>(`/datasets/${datasetId}/trajectory/dpt`, {
        method: "POST",
        body: JSON.stringify({ root_column: rootColumn, root_value: rootValue }),
      });
      // The new obs column joins the colour-by lists, and any cached colouring
      // by it is stale. Skip if the user has moved to another dataset meanwhile.
      if (useDatasetStore.getState().currentDatasetId === datasetId) {
        setCurrentDataset(await getDataset(datasetId));
      }
      await queryClient.invalidateQueries({ queryKey: ["embedding", datasetId] });
      setSelectedColumn(res.column);
      setRefreshKey((k) => k + 1);
    } catch (err) {
      setComputeError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsComputing(false);
    }
  }, [datasetId, rootColumn, rootValue, setCurrentDataset, queryClient]);

  // ---- PAGA: a graph of one column's groups, drawn over the embedding ----
  const [pagaGroupby, setPagaGroupby] = useState("");
  const [showPaga, setShowPaga] = useState(true);
  const [pagaThreshold, setPagaThreshold] = useState(DEFAULT_PAGA_THRESHOLD);
  const [treeOnly, setTreeOnly] = useState(false);
  const [pagaComputing, setPagaComputing] = useState(false);
  const [pagaError, setPagaError] = useState<string | null>(null);

  const pagaQuery = useQuery({
    queryKey: ["paga", datasetId, embedding, refreshKey],
    queryFn: () => getPaga(datasetId!, embedding),
    enabled: !!datasetId && !!embedding,
    retry: false,
  });
  const pagaView = pagaQuery.data ?? null;
  const paga = pagaView?.paga ?? null;
  const pagaUsable = !!paga && !pagaView?.stale;

  // The active clustering by default; once a graph exists, the column it was built on.
  const pagaColumn = paga?.groupby;
  useEffect(() => {
    if (pagaColumn && rootCols.some((c) => c.name === pagaColumn)) setPagaGroupby(pagaColumn);
  }, [pagaColumn, rootCols]);
  useEffect(() => {
    if (rootCols.some((c) => c.name === pagaGroupby)) return;
    const active = dataset?.active_clustering;
    setPagaGroupby(
      active && rootCols.some((c) => c.name === active) ? active : defaultRootColumn(rootCols, active),
    );
  }, [rootCols, pagaGroupby, dataset]);

  const runPaga = useCallback(async () => {
    if (!datasetId || !pagaGroupby) return;
    setPagaComputing(true);
    setPagaError(null);
    try {
      await computePaga(datasetId, pagaGroupby);
      setShowPaga(true);
      await queryClient.invalidateQueries({ queryKey: ["paga", datasetId] });
    } catch (err) {
      setPagaError(err instanceof Error ? err.message : String(err));
    } finally {
      setPagaComputing(false);
    }
  }, [datasetId, pagaGroupby, queryClient]);

  // ---- Genes changing along pseudotime, on all cells or one PAGA branch ----
  const [rightTab, setRightTab] = useState<"changing" | "trends">("changing");
  const [branchEnd, setBranchEnd] = useState("");
  const rootGroup = pagaView?.root_group ?? null;
  const ends = useMemo(
    () => (paga && pagaUsable && rootGroup ? pagaBranchEnds(paga, rootGroup) : []),
    [paga, pagaUsable, rootGroup],
  );
  useEffect(() => {
    if (branchEnd && !ends.includes(branchEnd)) setBranchEnd("");
  }, [ends, branchEnd]);
  const branchPath = useMemo(
    () => (paga && rootGroup && branchEnd ? treePath(paga, rootGroup, branchEnd) : null),
    [paga, rootGroup, branchEnd],
  );
  const branchNote = !paga
    ? "Compute a PAGA graph to rank genes along one branch only."
    : pagaView?.stale
      ? "The PAGA graph is out of date; recompute it to pick a branch."
      : !rootGroup
        ? "Compute pseudotime from a start group to pick a branch from it."
        : null;

  const trendQuery = useQuery({
    queryKey: ["trend-genes", datasetId, selectedColumn, branchPath && paga ? [paga.groupby, ...branchPath] : null, refreshKey],
    queryFn: ({ signal }) =>
      rankTrendGenes(
        datasetId!,
        selectedColumn,
        branchPath && paga ? { groups_column: paga.groupby, groups: branchPath } : {},
        signal,
      ),
    enabled: !!datasetId && !!selectedColumn && rightTab === "changing",
    staleTime: 5 * 60 * 1000,
    retry: false,
    // The last ranking of this pseudotime stays up while another branch's loads.
    placeholderData: (prev, prevQuery) =>
      prevQuery?.queryKey[1] === datasetId && prevQuery?.queryKey[2] === selectedColumn ? prev : undefined,
  });

  const overlayLayers = useMemo(() => {
    if (!showPaga || !paga || !pagaUsable || !pagaView?.positions) return null;
    return pagaLayers({
      record: paga,
      positions: pagaView.positions,
      threshold: pagaThreshold,
      treeOnly,
      path: branchPath,
      dark: plotBackground === "dark",
    });
  }, [showPaga, paga, pagaUsable, pagaView, pagaThreshold, treeOnly, branchPath, plotBackground]);
  const nVisibleEdges = paga ? visibleEdges(paga, pagaThreshold, treeOnly).length : 0;

  // A gene clicked in the heatmap colours the embedding by its expression.
  const [focusGene, setFocusGene] = useState<string | null>(null);
  const [geneColour, setGeneColour] = useState<{ gene: string; values: Float32Array; min: number; max: number } | null>(null);
  useEffect(() => {
    setFocusGene(null);
  }, [datasetId]);
  useEffect(() => {
    if (!datasetId || !focusGene) {
      setGeneColour(null);
      return;
    }
    let cancelled = false;
    apiFetchBinary(`/datasets/${datasetId}/expression?genes=${encodeURIComponent(focusGene)}`)
      .then((buffer) => {
        const decoded = decodeArrowBuffer(buffer);
        const col = decoded[focusGene] ?? decoded[Object.keys(decoded)[0]!];
        if (cancelled || !col) return;
        const values = col instanceof Float32Array ? col : Float32Array.from(col);
        let min = Infinity;
        let max = -Infinity;
        for (const v of values) {
          if (v < min) min = v;
          if (v > max) max = v;
        }
        setGeneColour({ gene: focusGene, values, min, max: max > min ? max : min + 1 });
      })
      .catch(() => {
        if (!cancelled) setGeneColour(null);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, focusGene]);
  const colourByGene = geneColour && geneColour.gene === focusGene ? geneColour : null;

  // Handle gene selection — add to list and fetch trajectory data
  const handleGeneSelect = useCallback(
    async (gene: string) => {
      if (!datasetId || !selectedColumn) return;
      setRightTab("trends");

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

  const selectCls =
    "rounded-lg border border-slate-300 bg-white px-2 py-1.5 text-sm text-slate-700 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100";

  const computeCard = (
    <div className="flex-shrink-0 rounded-xl border border-slate-200 bg-white px-4 py-3 text-left shadow-sm">
      {rootCols.length === 0 ? (
        <p className="text-xs text-slate-500">
          No categorical column (cluster or cell type) to pick a start group from.
        </p>
      ) : (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium text-slate-600">Compute pseudotime from</span>
          <select value={rootColumn} onChange={(e) => setRootColumn(e.target.value)} className={selectCls}>
            {rootCols.map((c) => (
              <option key={c.name} value={c.name}>
                {prettyObsLabel(c.name)}
              </option>
            ))}
          </select>
          <select value={rootValue} onChange={(e) => setRootValue(e.target.value)} className={selectCls}>
            {rootValues.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
          <button
            onClick={computePseudotime}
            disabled={isComputing || !rootValue}
            className="flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-sm font-medium text-white transition hover:bg-primary/90 disabled:opacity-50"
          >
            {isComputing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
            {isComputing ? "Computing..." : "Compute"}
          </button>
        </div>
      )}
      <p className="mt-1.5 text-xs text-slate-500">
        Diffusion pseudotime (scanpy DPT) on the neighbour graph, measured from the start
        group&apos;s most central cell. Saved to the dataset as <code>{DPT_COLUMN}</code>; cells the
        start group cannot reach are left grey.
      </p>
      {computeError && <p className="mt-1.5 text-xs text-red-600">{computeError}</p>}
    </div>
  );

  const velocityNote = velocityLayers ? (
    <p className="flex-shrink-0 text-xs text-slate-500" data-testid="velocity-note">
      This dataset has <code>spliced</code> and <code>unspliced</code> layers, the input for RNA
      velocity. scView does not compute velocity yet: run scVelo on the exported .h5ad, and its{" "}
      <code>latent_time</code> column, uploaded back, is listed here as a pseudotime.
    </p>
  ) : null;

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
            This dataset does not contain pseudotime or trajectory data yet.
            Compute diffusion pseudotime below (needs a processed dataset with a
            PCA), or upload a file that already has a pseudotime column.
          </p>
        </div>
        {computeCard}
        {velocityNote}
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

      {computeCard}

      {selectedColumn === DPT_COLUMN && computed && (
        <p className="-mt-2 flex-shrink-0 text-xs text-slate-500">
          Current <code>{DPT_COLUMN}</code>: started from {computed.root_value} (
          {prettyObsLabel(computed.root_column)}), {computed.graph}.
          {computed.n_unreachable > 0 &&
            ` ${formatNumber(computed.n_unreachable)} cells are unreachable from it (grey).`}
        </p>
      )}

      <PagaCard
        columns={rootCols}
        groupby={pagaGroupby}
        onGroupbyChange={setPagaGroupby}
        view={pagaView}
        computing={pagaComputing}
        error={pagaError}
        onCompute={runPaga}
        show={showPaga}
        onShowChange={setShowPaga}
        threshold={pagaThreshold}
        onThresholdChange={setPagaThreshold}
        treeOnly={treeOnly}
        onTreeOnlyChange={setTreeOnly}
        nVisible={nVisibleEdges}
        unplaced={!!paga && !pagaQuery.isFetching && !pagaView?.positions}
      />

      {velocityNote}

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
      <div className="flex min-h-[520px] flex-1 gap-4">
        {/* Left: Embedding scatter colored by pseudotime (or a clicked gene) */}
        <div className="flex min-w-0 flex-1 flex-col gap-2">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-slate-700">
              {colourByGene ? `Embedding Colored by ${colourByGene.gene}` : "Embedding Colored by Pseudotime"}
            </h3>
            {colourByGene && (
              <button
                onClick={() => setFocusGene(null)}
                className="flex items-center gap-1 rounded-full border border-slate-300 px-2 py-0.5 text-xs text-slate-600 hover:bg-slate-50"
              >
                <X className="h-3 w-3" /> Back to pseudotime
              </button>
            )}
          </div>
          <Panel className="min-h-0 flex-1" bodyClassName="relative min-h-0 p-0">
            {(embeddingLoading || isLoadingPseudotime) && !positions ? (
              <div className="flex h-full items-center justify-center">
                <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
              </div>
            ) : (
              <EmbeddingScatter
                positions={positions}
                colorValues={colourByGene ? colourByGene.values : pseudotimeValues}
                colorType="continuous"
                pointSize={pointSize}
                opacity={opacity}
                selectedIndices={selectedIndices}
                background={plotBackground}
                maxRenderedCells={maxRenderedCells}
                dimensions={dimensions}
                exportName={colourByGene ? `trajectory_${colourByGene.gene}` : `pseudotime_${selectedColumn}`}
                colorLabel={
                  colourByGene ? `${colourByGene.gene} expression` : prettyObsLabel(selectedColumn) || "Pseudotime"
                }
                exportLegend={
                  colourByGene
                    ? { type: "continuous", title: colourByGene.gene, min: colourByGene.min, max: colourByGene.max }
                    : pseudotimeValues
                      ? { type: "continuous", title: selectedColumn, ...pseudotimeMinMax }
                      : null
                }
                overlayLayers={overlayLayers}
              />
            )}
          </Panel>
          {colourByGene ? (
            <div className="flex-shrink-0 px-2">
              <ColorLegend type="continuous" min={colourByGene.min} max={colourByGene.max} label={colourByGene.gene} />
            </div>
          ) : (
            pseudotimeValues && (
              <div className="flex-shrink-0 px-2">
                <ColorLegend
                  type="continuous"
                  min={pseudotimeMinMax.min}
                  max={pseudotimeMinMax.max}
                  label={selectedColumn}
                />
              </div>
            )
          )}
        </div>

        {/* Right: genes changing along pseudotime | chosen genes' trends */}
        <div className="flex w-[480px] flex-shrink-0 flex-col gap-2">
          <div className="flex items-center gap-1" role="tablist">
            {(
              [
                ["changing", "Genes changing along pseudotime"],
                ["trends", `Gene trends${selectedGenes.length ? ` (${selectedGenes.length})` : ""}`],
              ] as const
            ).map(([tab, label]) => (
              <button
                key={tab}
                role="tab"
                aria-selected={rightTab === tab}
                onClick={() => setRightTab(tab)}
                className={`rounded-md px-2.5 py-1 text-sm font-semibold transition ${
                  rightTab === tab ? "bg-slate-200 text-slate-800" : "text-slate-500 hover:bg-slate-100"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="flex-1 overflow-auto rounded-xl border border-slate-200 bg-white p-2 shadow-sm">
            {rightTab === "changing" ? (
              <TrendGenesView
                result={trendQuery.data ?? null}
                loading={trendQuery.isFetching}
                error={trendQuery.error ? (trendQuery.error as Error).message : null}
                focusGene={focusGene}
                onFocusGene={setFocusGene}
                branchEnds={ends}
                branchEnd={branchEnd}
                onBranchEndChange={setBranchEnd}
                path={branchPath}
                branchNote={branchNote}
                groupby={paga?.groupby ?? null}
                exportName={selectedColumn}
              />
            ) : selectedGenes.length === 0 ? (
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
                config={plotConfig("trajectory_expression")}
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
