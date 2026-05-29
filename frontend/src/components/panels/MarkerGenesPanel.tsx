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
import { apiFetch } from "@/api/client";
import { formatPValue } from "@/lib/formatting";

interface MarkerGene {
  gene: string;
  group: string;
  logfoldchange: number;
  pval: number;
  pval_adj: number;
  pct_in?: number;
  pct_out?: number;
  score?: number;
}

interface MarkersResponse {
  groups: string[];
  markers: Record<string, MarkerGene[]>;
}

type SortField = "gene" | "logfoldchange" | "pval" | "pval_adj" | "pct_in" | "score";
type SortDir = "asc" | "desc";

export function MarkerGenesPanel() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const setPanel = useViewStore((s) => s.setPanel);

  const [markersData, setMarkersData] = useState<MarkersResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [noMarkers, setNoMarkers] = useState(false);

  const [selectedGroup, setSelectedGroup] = useState<string>("");
  const [sortField, setSortField] = useState<SortField>("logfoldchange");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [filterQuery, setFilterQuery] = useState("");
  const [groupbyColumn, setGroupbyColumn] = useState<string>("");
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

    setIsLoading(true);
    setError(null);
    setNoMarkers(false);

    // On initial load, try pre-computed markers first (no groupby_column)
    // On subsequent changes to groupbyColumn, compute on-the-fly
    const useOnDemand = initialFetchDone.current && groupbyColumn;
    const url = useOnDemand
      ? `/datasets/${datasetId}/markers?format=json&groupby_column=${encodeURIComponent(groupbyColumn)}`
      : `/datasets/${datasetId}/markers?format=json`;

    apiFetch<MarkersResponse>(url)
      .then((data) => {
        setMarkersData(data);
        setSelectedGroup(data.groups.length > 0 ? data.groups[0]! : "");
        initialFetchDone.current = true;
      })
      .catch((err) => {
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes("404") && !useOnDemand) {
          // No pre-computed markers — try on-demand if we have a groupby column
          if (groupbyColumn) {
            apiFetch<MarkersResponse>(
              `/datasets/${datasetId}/markers?format=json&groupby_column=${encodeURIComponent(groupbyColumn)}`,
            )
              .then((data) => {
                setMarkersData(data);
                setSelectedGroup(data.groups.length > 0 ? data.groups[0]! : "");
                initialFetchDone.current = true;
              })
              .catch((innerErr) => {
                const innerMsg = innerErr instanceof Error ? innerErr.message : String(innerErr);
                if (innerMsg.includes("404")) {
                  setNoMarkers(true);
                } else {
                  setError(innerMsg);
                }
              })
              .finally(() => setIsLoading(false));
            return;
          }
          setNoMarkers(true);
        } else if (msg.includes("404")) {
          setNoMarkers(true);
        } else {
          setError(msg);
        }
        setIsLoading(false);
        return;
      })
      .then(() => setIsLoading(false));
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

  // Sorted markers
  const sortedMarkers = useMemo(() => {
    const sorted = [...filteredMarkers];
    sorted.sort((a, b) => {
      let aVal: number;
      let bVal: number;

      switch (sortField) {
        case "gene":
          return sortDir === "asc"
            ? a.gene.localeCompare(b.gene)
            : b.gene.localeCompare(a.gene);
        case "logfoldchange":
          aVal = a.logfoldchange;
          bVal = b.logfoldchange;
          break;
        case "pval":
          aVal = a.pval;
          bVal = b.pval;
          break;
        case "pval_adj":
          aVal = a.pval_adj;
          bVal = b.pval_adj;
          break;
        case "pct_in":
          aVal = a.pct_in ?? 0;
          bVal = b.pct_in ?? 0;
          break;
        case "score":
          aVal = a.score ?? 0;
          bVal = b.score ?? 0;
          break;
        default:
          return 0;
      }

      return sortDir === "asc" ? aVal - bVal : bVal - aVal;
    });

    return sorted;
  }, [filteredMarkers, sortField, sortDir]);

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

    const headers = ["gene", "group", "log_fold_change", "p_value", "p_value_adjusted", "pct_in", "pct_out", "score"];
    const rows = sortedMarkers.map((m) =>
      [
        m.gene,
        m.group,
        m.logfoldchange.toFixed(4),
        m.pval.toExponential(4),
        m.pval_adj.toExponential(4),
        m.pct_in?.toFixed(3) ?? "",
        m.pct_out?.toFixed(3) ?? "",
        m.score?.toFixed(4) ?? "",
      ].join(","),
    );

    const csv = [headers.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `markers_${selectedGroup}_${datasetId}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [sortedMarkers, selectedGroup, datasetId]);

  // No dataset
  if (!dataset || !datasetId) {
    return (
      <div className="mx-auto max-w-6xl space-y-6">
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
      <div className="mx-auto max-w-6xl space-y-6">
        <div className="flex items-center gap-3">
          <ListTree className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">Marker Genes</h2>
        </div>
        <div className="flex items-center justify-center py-16">
          <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
          <span className="ml-3 text-sm text-slate-500">Loading marker genes...</span>
        </div>
      </div>
    );
  }

  // No markers computed yet (404) — friendly warning
  if (noMarkers || (!error && !markersData && !isLoading)) {
    return (
      <div className="mx-auto max-w-6xl space-y-6">
        <div className="flex items-center gap-3">
          <ListTree className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">Marker Genes</h2>
        </div>
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-8 text-center">
          <AlertCircle className="mx-auto mb-3 h-8 w-8 text-amber-500" />
          <p className="mb-1 text-sm font-medium text-amber-800">
            No marker genes available
          </p>
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
      <div className="mx-auto max-w-6xl space-y-6">
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
    <div className="mx-auto max-w-6xl space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <ListTree className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">Marker Genes</h2>
        </div>
      </div>

      {/* Controls row */}
      <div className="flex flex-wrap items-center gap-4">
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
        <span className="text-xs text-slate-500">
          {sortedMarkers.length} of {currentMarkers.length} genes
        </span>
      </div>

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
                  className="cursor-pointer px-4 py-2.5 text-right font-medium text-slate-600 hover:text-slate-800"
                >
                  <span className="flex items-center justify-end gap-1">
                    Log FC {sortIcon("logfoldchange")}
                  </span>
                </th>
                <th
                  onClick={() => handleSort("pval")}
                  className="cursor-pointer px-4 py-2.5 text-right font-medium text-slate-600 hover:text-slate-800"
                >
                  <span className="flex items-center justify-end gap-1">
                    P-value {sortIcon("pval")}
                  </span>
                </th>
                <th
                  onClick={() => handleSort("pval_adj")}
                  className="cursor-pointer px-4 py-2.5 text-right font-medium text-slate-600 hover:text-slate-800"
                >
                  <span className="flex items-center justify-end gap-1">
                    Adj. P-value {sortIcon("pval_adj")}
                  </span>
                </th>
                <th
                  onClick={() => handleSort("pct_in")}
                  className="cursor-pointer px-4 py-2.5 text-right font-medium text-slate-600 hover:text-slate-800"
                >
                  <span className="flex items-center justify-end gap-1">
                    % In {sortIcon("pct_in")}
                  </span>
                </th>
                {currentMarkers.some((m) => m.score != null) && (
                  <th
                    onClick={() => handleSort("score")}
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
                          marker.logfoldchange > 0
                            ? "text-emerald-600"
                            : marker.logfoldchange < 0
                              ? "text-red-600"
                              : "text-slate-600"
                        }
                      >
                        {marker.logfoldchange.toFixed(3)}
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
                  <td colSpan={7} className="px-4 py-8 text-center text-slate-400">
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
  );
}
