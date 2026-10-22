import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import {
  ListTree,
  Loader2,
  AlertCircle,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Download,
  Search,
  ExternalLink,
} from "lucide-react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";
import { useDelayedFlag } from "@/hooks/useDelayedFlag";
import { apiFetch } from "@/api/client";
import { MARKERS_PER_GROUP, NO_MARKERS_REASON, noMarkersReason, type MarkersResponse } from "@/api/markers";
import { formatFoldChange, formatPValue } from "@/lib/formatting";
import { downloadCsv } from "@/lib/csv";
import {
  MARKER_CSV_HEADERS,
  SPECIFICITY_HELP,
  detectionRatio,
  formatPct,
  formatRatio,
  formatTau,
  hasSpecificity,
  markerCsvRow,
  sortMarkers,
  type MarkerSortField,
} from "@/lib/markerSpecificity";
import { MarkerPlot } from "@/components/plots/MarkerPlot";
import { CellHeatmap } from "@/components/plots/CellHeatmap";
import { MarkerSpecificityPlot } from "@/components/plots/MarkerSpecificityPlot";
import { GeneSearch } from "@/components/common/GeneSearch";
import { MAX_PLOT_GENES, mergeGeneLists, parseGeneList } from "@/lib/geneList";

type SortField = MarkerSortField;
type SortDir = "asc" | "desc";

export function MarkerGenesPanel() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const setPanel = useViewStore((s) => s.setPanel);

  const [markersData, setMarkersData] = useState<MarkersResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Why there are no markers (an unprocessed dataset: 200 with no groups), or null.
  const [noMarkers, setNoMarkers] = useState<string | null>(null);
  // Stored markers arrive at once; a request still running after 1.5 s is
  // computing them for this column, which takes a while on a large dataset.
  const computingMarkers = useDelayedFlag(isLoading, 1500);

  const [selectedGroup, setSelectedGroup] = useState<string>("");
  const [sortField, setSortField] = useState<SortField>("logfoldchange");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [filterQuery, setFilterQuery] = useState("");
  const [groupbyColumn, setGroupbyColumn] = useState<string>("");
  const [perGroup, setPerGroup] = useState(5);
  // Heatmap of group means, or of single cells sampled per group (C034).
  const [heatmapOf, setHeatmapOf] = useState<"means" | "cells">("means");
  const markerView = useViewStore((s) => s.markerView);
  const setMarkerView = useViewStore((s) => s.setMarkerView);
  const markerGenes = useViewStore((s) => s.markerGenes);
  const setMarkerGenes = useViewStore((s) => s.setMarkerGenes);
  // Typed, picked or pasted genes for the dot plot / heatmap (see GeneSearch `multiple`).
  const addMarkerGenes = useCallback(
    (text: string) => setMarkerGenes(mergeGeneLists(markerGenes, parseGeneList(text))),
    [markerGenes, setMarkerGenes],
  );
  const initialFetchDone = useRef(false);

  // Categorical columns suitable for marker gene grouping
  const groupbyColumns = useMemo(() => {
    if (!dataset) return [];
    return dataset.obs_columns.filter((c) => {
      if (c.dtype !== "category" && c.dtype !== "object" && c.dtype !== "bool") return false;
      if (c.n_unique < 2 || c.n_unique > 100) return false;
      return true;
    });
  }, [dataset]);

  // Set default groupby column
  useEffect(() => {
    if (groupbyColumn || groupbyColumns.length === 0) return;
    // Prefer active clustering, then common cluster columns
    if (dataset?.active_clustering) {
      const found = groupbyColumns.find((c) => c.name === dataset.active_clustering);
      if (found) { setGroupbyColumn(found.name); return; }
    }
    const clusterPatterns = ["leiden", "louvain", "cluster", "seurat_clusters", "cell_type", "celltype"];
    const match = groupbyColumns.find((c) =>
      clusterPatterns.some((p) => c.name.toLowerCase().includes(p)),
    );
    if (match) { setGroupbyColumn(match.name); return; }
    setGroupbyColumn(groupbyColumns[0]!.name);
  }, [groupbyColumns, groupbyColumn, dataset]);

  // Fetch markers — first try pre-computed, then on-demand with groupbyColumn
  useEffect(() => {
    if (!datasetId) return;
    // A request made before the default column was picked can answer after the
    // column's own markers: only the latest run may set state.
    let cancelled = false;

    setIsLoading(true);
    setError(null);
    setNoMarkers(null);

    const show = (data: MarkersResponse) => {
      if (cancelled) return;
      const none = noMarkersReason(data);
      if (none) {
        setMarkersData(null);
        setNoMarkers(none);
        return;
      }
      setMarkersData(data);
      setNoMarkers(null);
      setSelectedGroup(data.groups.length > 0 ? data.groups[0]! : "");
      initialFetchDone.current = true;
    };
    const fail = (err: unknown) => {
      if (cancelled) return;
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes("404")) setNoMarkers(NO_MARKERS_REASON);
      else setError(msg);
    };
    const onDemand = () =>
      apiFetch<MarkersResponse>(
        `/datasets/${datasetId}/markers?format=json&groupby_column=${encodeURIComponent(groupbyColumn)}`,
      );

    // On initial load, try pre-computed markers first (no groupby_column)
    // On subsequent changes to groupbyColumn, compute on-the-fly
    const useOnDemand = initialFetchDone.current && groupbyColumn;
    const first = useOnDemand ? onDemand() : apiFetch<MarkersResponse>(`/datasets/${datasetId}/markers?format=json`);

    first
      .then((data) => {
        // No stored markers (200, no groups): compute the default column's on demand.
        if (noMarkersReason(data) && !useOnDemand && groupbyColumn && !cancelled) {
          return onDemand().then(show, fail);
        }
        show(data);
      })
      .catch((err) => {
        const msg = err instanceof Error ? err.message : String(err);
        // No pre-computed markers — try on-demand if we have a groupby column
        if (msg.includes("404") && !useOnDemand && groupbyColumn && !cancelled) {
          return onDemand().then(show, fail);
        }
        fail(err);
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, groupbyColumn]);

  // Current markers for selected group
  const currentMarkers = useMemo(() => {
    if (!markersData || !selectedGroup) return [];
    return markersData.markers[selectedGroup] ?? [];
  }, [markersData, selectedGroup]);

  // Filtered markers
  const filteredMarkers = useMemo(() => {
    if (!filterQuery.trim()) return currentMarkers;
    const q = filterQuery.trim().toLowerCase();
    return currentMarkers.filter((m) => m.gene.toLowerCase().includes(q));
  }, [currentMarkers, filterQuery]);

  // Sorted markers; a missing value sorts last either way.
  const sortedMarkers = useMemo(
    () => sortMarkers(filteredMarkers, sortField, sortDir),
    [filteredMarkers, sortField, sortDir],
  );
  const specificity = useMemo(() => hasSpecificity(currentMarkers), [currentMarkers]);

  // Handle sort click
  const handleSort = useCallback(
    (field: SortField) => {
      if (sortField === field) {
        setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
      } else {
        setSortField(field);
        setSortDir(field === "pval" || field === "pval_adj" ? "asc" : "desc");
      }
    },
    [sortField],
  );

  // Sort icon helper
  const sortIcon = useCallback(
    (field: SortField) => {
      if (sortField !== field) {
        return <ArrowUpDown className="h-3 w-3 text-slate-400" />;
      }
      return sortDir === "asc" ? (
        <ArrowUp className="h-3 w-3 text-blue-500" />
      ) : (
        <ArrowDown className="h-3 w-3 text-blue-500" />
      );
    },
    [sortField, sortDir],
  );

  const setPendingGene = useViewStore((s) => s.setPendingGene);

  // Navigate to gene expression panel with the selected gene
  const handleGeneClick = useCallback(
    (gene: string) => {
      setPendingGene(gene);
      setPanel("expression");
    },
    [setPanel, setPendingGene],
  );

  // Export as CSV
  const handleExportCSV = useCallback(() => {
    if (sortedMarkers.length === 0) return;
    downloadCsv(
      `markers_${selectedGroup}_${datasetId}.csv`,
      MARKER_CSV_HEADERS,
      sortedMarkers.map(markerCsvRow),
    );
  }, [sortedMarkers, selectedGroup, datasetId]);

  // No dataset
  if (!dataset || !datasetId) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <ListTree className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">Marker Genes</h2>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-12 text-center text-slate-400">
          Load a dataset to view marker genes.
        </div>
      </div>
    );
  }

  // Loading
  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <ListTree className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">Marker Genes</h2>
        </div>
        <div className="flex items-center justify-center py-16">
          <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
          <span className="ml-3 text-sm text-slate-500">
            {computingMarkers
              ? "Computing marker genes… this can take ~30 s on a large dataset"
              : "Loading marker genes..."}
          </span>
        </div>
      </div>
    );
  }

  // No markers computed yet (404) — friendly warning
  if (noMarkers || (!error && !markersData && !isLoading)) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <ListTree className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">Marker Genes</h2>
        </div>
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-8 text-center">
          <AlertCircle className="mx-auto mb-3 h-8 w-8 text-amber-500" />
          <p className="mb-1 text-sm font-medium text-amber-800">
            No marker genes available
          </p>
          {noMarkers && <p className="mb-1 text-xs text-amber-700" data-testid="no-markers-reason">{noMarkers}</p>}
          <p className="mb-4 text-xs text-amber-600">
            This dataset does not have marker genes computed yet. Run the
            preprocessing pipeline in{" "}
            <button
              onClick={() => setPanel("assessment")}
              className="inline font-semibold text-primary underline decoration-primary/40 hover:decoration-primary"
            >
              Data Assessment
            </button>{" "}
            to compute differential expression and marker genes for each cluster.
          </p>
          <button
            onClick={() => setPanel("assessment")}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white transition hover:bg-primary/90"
          >
            Go to Data Assessment
          </button>
        </div>
      </div>
    );
  }

  // Real error (non-404)
  if (error) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <ListTree className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">Marker Genes</h2>
        </div>
        <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          {error}
        </div>
      </div>
    );
  }

  // Shouldn't reach here without markersData, but guard for TypeScript
  if (!markersData) return null;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <ListTree className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">Marker Genes</h2>
        </div>
      </div>

      {/* Controls row */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex overflow-hidden rounded-lg border border-slate-300 text-sm shadow-sm" role="group" aria-label="Marker view">
          {([["table", "Table"], ["dotplot", "Dot plot"], ["heatmap", "Heatmap"]] as const).map(([v, label]) => (
            <button
              key={v}
              onClick={() => setMarkerView(v)}
              aria-pressed={markerView === v}
              className={`px-3 py-1.5 font-medium transition-colors ${
                markerView === v ? "bg-primary text-white" : "bg-white text-slate-700 hover:bg-slate-50"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Groupby column selector (compute for which obs column) */}
        {groupbyColumns.length > 0 && (
          <div className="flex items-center gap-2">
            <label htmlFor="marker-groupby-select" className="text-sm font-medium text-slate-600">
              Compute for:
            </label>
            <select
              id="marker-groupby-select"
              value={groupbyColumn}
              onChange={(e) => {
                setGroupbyColumn(e.target.value);
                setFilterQuery("");
              }}
              className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
            >
              {groupbyColumns.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.name} ({col.n_unique} groups)
                </option>
              ))}
            </select>
          </div>
        )}

        {markerView === "heatmap" && (
          <div className="flex overflow-hidden rounded-lg border border-slate-300 text-sm shadow-sm" role="group" aria-label="Heatmap of">
            {([["means", "Group means"], ["cells", "Per cell"]] as const).map(([v, label]) => (
              <button
                key={v}
                onClick={() => setHeatmapOf(v)}
                aria-pressed={heatmapOf === v}
                title={v === "cells" ? "Single cells, a seeded sample of each group" : "One row per group: the group's mean"}
                className={`px-3 py-1.5 font-medium transition-colors ${
                  heatmapOf === v ? "bg-slate-700 text-white" : "bg-white text-slate-700 hover:bg-slate-50"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        )}

        {markerView !== "table" && markerGenes.length === 0 && (
          <div className="flex items-center gap-2">
            <label htmlFor="marker-pergroup-select" className="text-sm font-medium text-slate-600">
              Genes per group:
            </label>
            <select
              id="marker-pergroup-select"
              value={perGroup}
              onChange={(e) => setPerGroup(Number(e.target.value))}
              className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
            >
              {[3, 5, 10].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </div>
        )}

        {markerView === "table" && (<>
        {/* Group selector (filter within computed results) */}
        <div className="flex items-center gap-2">
          <label htmlFor="marker-group-select" className="text-sm font-medium text-slate-600">
            Cluster / Group:
          </label>
          <select
            id="marker-group-select"
            value={selectedGroup}
            onChange={(e) => {
              setSelectedGroup(e.target.value);
              setFilterQuery("");
            }}
            className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
          >
            {markersData.groups.map((g) => (
              <option key={g} value={g}>
                {g} ({(markersData.markers[g] ?? []).length} genes)
              </option>
            ))}
          </select>
        </div>

        {/* Filter */}
        <div className="relative flex-1 max-w-xs">
          <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            placeholder="Filter genes..."
            className="w-full rounded-lg border border-slate-300 bg-white py-1.5 pl-8 pr-3 text-sm text-slate-700 placeholder-slate-400 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
          />
        </div>

        {/* Export */}
        <button
          onClick={handleExportCSV}
          disabled={sortedMarkers.length === 0}
          className="flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Download className="h-3.5 w-3.5" />
          Export CSV
        </button>

        {/* Count info */}
        <span
          className="text-xs text-slate-500"
          title={`The backend keeps each group's top ${MARKERS_PER_GROUP} markers by Wilcoxon score; every gene is tested, the rest are cut`}
        >
          {sortedMarkers.length} of the group&apos;s top {currentMarkers.length} markers by Wilcoxon score
        </span>
        </>)}
      </div>

      {markerView !== "table" ? (
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          {/* Your own genes instead of the top markers */}
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <GeneSearch
              datasetId={datasetId}
              onSelect={addMarkerGenes}
              multiple
              placeholder="Plot your own genes: type one, or paste a list..."
              className="w-full max-w-sm"
            />
            {markerGenes.length === 0 ? (
              <span className="text-xs text-slate-400">Showing the top markers of each group.</span>
            ) : (
              <>
                {markerGenes.map((g) => (
                  <span
                    key={g}
                    className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2.5 py-0.5 font-mono text-xs text-slate-700"
                  >
                    {g}
                    <button
                      onClick={() => setMarkerGenes(markerGenes.filter((x) => x !== g))}
                      title={`Remove ${g}`}
                      className="rounded-full px-0.5 text-slate-400 hover:bg-slate-200 hover:text-slate-600"
                    >
                      &times;
                    </button>
                  </span>
                ))}
                <button
                  onClick={() => setMarkerGenes([])}
                  className="text-xs font-medium text-primary hover:underline"
                >
                  Back to top markers
                </button>
                {markerGenes.length >= MAX_PLOT_GENES && (
                  <span className="text-xs text-amber-700">Up to {MAX_PLOT_GENES} genes.</span>
                )}
              </>
            )}
          </div>
          {markerView === "heatmap" && heatmapOf === "cells" ? (
            <CellHeatmap
              datasetId={datasetId}
              groupby={groupbyColumn}
              groups={markersData.groups}
              markers={markersData.markers}
              perGroup={perGroup}
              genes={markerGenes}
            />
          ) : (
            <MarkerPlot
              datasetId={datasetId}
              groupby={groupbyColumn}
              groups={markersData.groups}
              markers={markersData.markers}
              kind={markerView}
              perGroup={perGroup}
              genes={markerGenes}
            />
          )}
        </div>
      ) : (
      <div className="space-y-4">
      {specificity && (
        <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
          <div className="mb-1 flex items-baseline gap-2">
            <h3 className="text-sm font-semibold text-slate-700">
              Specificity vs fold change: {selectedGroup}
            </h3>
            <span className="text-xs text-slate-400">
              One point per marker ({currentMarkers.length.toLocaleString()}; the 10 strongest labelled). Top right:
              strong and found in this group only. A strong marker low on the plot is shared with other groups.
            </span>
          </div>
          <MarkerSpecificityPlot
            markers={currentMarkers}
            name={selectedGroup}
            onGeneClick={handleGeneClick}
            height={260}
            maxLabels={10}
          />
        </div>
      )}
      {/* Marker genes table */}
      <div className="rounded-xl border border-slate-200 bg-white shadow-sm">
        <div className="max-h-[600px] overflow-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 z-10">
              <tr className="border-b border-slate-200 bg-slate-50">
                <th
                  onClick={() => handleSort("gene")}
                  className="cursor-pointer px-4 py-2.5 text-left font-medium text-slate-600 hover:text-slate-800"
                >
                  <span className="flex items-center gap-1">
                    Gene {sortIcon("gene")}
                  </span>
                </th>
                <th
                  onClick={() => handleSort("logfoldchange")}
                  title="log2 fold change of this group's mean expression over all other cells (scanpy rank_genes_groups: expm1 of the mean log-normalised values on each side)"
                  className="cursor-pointer px-4 py-2.5 text-right font-medium text-slate-600 hover:text-slate-800"
                >
                  <span className="flex items-center justify-end gap-1">
                    Log FC {sortIcon("logfoldchange")}
                  </span>
                </th>
                <th
                  onClick={() => handleSort("pval")}
                  title="Wilcoxon rank-sum p, this group vs all other cells, unadjusted"
                  className="cursor-pointer px-4 py-2.5 text-right font-medium text-slate-600 hover:text-slate-800"
                >
                  <span className="flex items-center justify-end gap-1">
                    P-value {sortIcon("pval")}
                  </span>
                </th>
                <th
                  onClick={() => handleSort("pval_adj")}
                  title="For ranking only: the clusters were found from these same cells, so marker p-values are inflated."
                  className="cursor-pointer px-4 py-2.5 text-right font-medium text-slate-600 hover:text-slate-800"
                >
                  <span className="flex items-center justify-end gap-1">
                    Adj. P-value {sortIcon("pval_adj")}
                  </span>
                </th>
                <th
                  onClick={() => handleSort("pct_in")}
                  title="Share of the group's cells with any count of the gene"
                  className="cursor-pointer px-4 py-2.5 text-right font-medium text-slate-600 hover:text-slate-800"
                >
                  <span className="flex items-center justify-end gap-1">
                    % In {sortIcon("pct_in")}
                  </span>
                </th>
                {specificity && (
                  <>
                    <th
                      onClick={() => handleSort("tau")}
                      title={SPECIFICITY_HELP.tau}
                      className="cursor-pointer px-4 py-2.5 text-right font-medium text-slate-600 hover:text-slate-800"
                    >
                      <span className="flex items-center justify-end gap-1">Tau {sortIcon("tau")}</span>
                    </th>
                    <th
                      title={SPECIFICITY_HELP.detection}
                      className="px-4 py-2.5 text-right font-medium text-slate-600"
                    >
                      % Next group
                    </th>
                    <th
                      onClick={() => handleSort("pct_ratio")}
                      title={SPECIFICITY_HELP.ratio}
                      className="cursor-pointer px-4 py-2.5 text-right font-medium text-slate-600 hover:text-slate-800"
                    >
                      <span className="flex items-center justify-end gap-1">
                        Detection ratio {sortIcon("pct_ratio")}
                      </span>
                    </th>
                  </>
                )}
                {currentMarkers.some((m) => m.score != null) && (
                  <th
                    onClick={() => handleSort("score")}
                    title="scanpy rank_genes_groups score: the Wilcoxon rank-sum z-score of this group vs all other cells (the ranking the top-100 cut uses)"
                    className="cursor-pointer px-4 py-2.5 text-right font-medium text-slate-600 hover:text-slate-800"
                  >
                    <span className="flex items-center justify-end gap-1">
                      Score {sortIcon("score")}
                    </span>
                  </th>
                )}
                <th className="px-4 py-2.5 text-center font-medium text-slate-600">
                  Action
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedMarkers.map((marker) => {
                const isSignificant = marker.pval_adj < 0.05;
                return (
                  <tr
                    key={`${marker.group}-${marker.gene}`}
                    className="border-b border-slate-50 transition-colors hover:bg-slate-50"
                  >
                    <td className="px-4 py-2">
                      <span className="font-mono font-medium text-slate-800">
                        {marker.gene}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      <span
                        className={
                          (marker.logfoldchange ?? 0) > 0
                            ? "text-emerald-600"
                            : (marker.logfoldchange ?? 0) < 0
                              ? "text-red-600"
                              : "text-slate-600"
                        }
                      >
                        {formatFoldChange(marker.logfoldchange, 3)}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums text-slate-600">
                      {formatPValue(marker.pval)}
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      <span
                        className={isSignificant ? "font-medium text-blue-600" : "text-slate-600"}
                      >
                        {formatPValue(marker.pval_adj)}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums text-slate-600">
                      {marker.pct_in != null ? `${(marker.pct_in * 100).toFixed(1)}%` : "-"}
                    </td>
                    {specificity && (
                      <>
                        <td className="px-4 py-2 text-right tabular-nums text-slate-600">
                          {formatTau(marker.tau)}
                        </td>
                        <td className="px-4 py-2 text-right tabular-nums text-slate-600">
                          {formatPct(marker.pct_max_other)}
                          {marker.max_other_group && (
                            <span className="ml-1 text-xs text-slate-400">{marker.max_other_group}</span>
                          )}
                        </td>
                        <td className="px-4 py-2 text-right tabular-nums text-slate-600">
                          {formatRatio(detectionRatio(marker))}
                        </td>
                      </>
                    )}
                    {currentMarkers.some((m) => m.score != null) && (
                      <td className="px-4 py-2 text-right tabular-nums text-slate-600">
                        {marker.score?.toFixed(3) ?? "-"}
                      </td>
                    )}
                    <td className="px-4 py-2 text-center">
                      <button
                        onClick={() => handleGeneClick(marker.gene)}
                        title={`View ${marker.gene} expression`}
                        className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-blue-600 transition-colors hover:bg-blue-50"
                      >
                        <ExternalLink className="h-3 w-3" />
                        View
                      </button>
                    </td>
                  </tr>
                );
              })}
              {sortedMarkers.length === 0 && (
                <tr>
                  <td colSpan={specificity ? 10 : 7} className="px-4 py-8 text-center text-slate-400">
                    {filterQuery
                      ? "No genes match the current filter"
                      : "No marker genes for this group"}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
      </div>
      )}
    </div>
  );
}
