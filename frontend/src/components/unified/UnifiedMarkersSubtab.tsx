import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { Loader2, Search, ArrowUpDown, ArrowUp, ArrowDown, Download } from "lucide-react";
import { useDatasetStore } from "@/stores/datasetStore";
import { apiFetch } from "@/api/client";
import { formatPValue } from "@/lib/formatting";
import { downloadCsv } from "@/lib/csv";

interface MarkerGene {
  gene: string;
  group: string;
  logfoldchange: number;
  pval: number;
  pval_adj: number;
  pct_in?: number;
  score?: number;
}

interface MarkersResponse {
  groups: string[];
  markers: Record<string, MarkerGene[]>;
}

type SortField = "gene" | "logfoldchange" | "pval_adj";
type SortDir = "asc" | "desc";

interface UnifiedMarkersSubtabProps {
  onGeneClick: (gene: string) => void;
  groupByColumn: string;
  setGroupByColumn: (col: string) => void;
}

export function UnifiedMarkersSubtab({
  onGeneClick,
  groupByColumn,
  setGroupByColumn,
}: UnifiedMarkersSubtabProps) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);

  const [markersData, setMarkersData] = useState<MarkersResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [noMarkers, setNoMarkers] = useState(false);

  const [selectedGroup, setSelectedGroup] = useState<string>("");
  const [sortField, setSortField] = useState<SortField>("logfoldchange");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [filterQuery, setFilterQuery] = useState("");
  const [minAbsFC, setMinAbsFC] = useState(0);
  const initialFetchDone = useRef(false);

  const groupbyColumns = useMemo(() => {
    if (!dataset) return [];
    return dataset.obs_columns.filter((c) => {
      if (c.dtype !== "category" && c.dtype !== "object" && c.dtype !== "bool") return false;
      if (c.n_unique < 2 || c.n_unique > 100) return false;
      return true;
    });
  }, [dataset]);

  // Fetch markers
  useEffect(() => {
    if (!datasetId) return;
    // Cancellation guard: when groupByColumn auto-selects, this effect re-runs.
    // Without it, a stale first run (empty column → 404 → "no markers") can
    // resolve *after* the second run succeeds and clobber it.
    let cancelled = false;
    setIsLoading(true);
    setError(null);
    setNoMarkers(false);

    const useOnDemand = initialFetchDone.current && groupByColumn;
    const url = useOnDemand
      ? `/datasets/${datasetId}/markers?format=json&groupby_column=${encodeURIComponent(groupByColumn)}`
      : `/datasets/${datasetId}/markers?format=json`;

    const applyData = (data: MarkersResponse) => {
      if (cancelled) return;
      setMarkersData(data);
      setSelectedGroup(data.groups.length > 0 ? data.groups[0]! : "");
      initialFetchDone.current = true;
      setIsLoading(false);
    };

    apiFetch<MarkersResponse>(url)
      .then(applyData)
      .catch((err) => {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : String(err);
        // No default markers — fall back to on-demand markers for the
        // auto-selected grouping column.
        if (msg.includes("404") && !useOnDemand && groupByColumn) {
          apiFetch<MarkersResponse>(
            `/datasets/${datasetId}/markers?format=json&groupby_column=${encodeURIComponent(groupByColumn)}`,
          )
            .then(applyData)
            .catch((innerErr) => {
              if (cancelled) return;
              const innerMsg = innerErr instanceof Error ? innerErr.message : String(innerErr);
              if (innerMsg.includes("404")) setNoMarkers(true);
              else setError(innerMsg);
              setIsLoading(false);
            });
          return;
        }
        if (msg.includes("404")) setNoMarkers(true);
        else setError(msg);
        setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [datasetId, groupByColumn]);

  const currentMarkers = useMemo(() => {
    if (!markersData || !selectedGroup) return [];
    return markersData.markers[selectedGroup] ?? [];
  }, [markersData, selectedGroup]);

  const filteredMarkers = useMemo(() => {
    let filtered = currentMarkers;
    if (minAbsFC > 0) {
      filtered = filtered.filter((m) => Math.abs(m.logfoldchange) >= minAbsFC);
    }
    if (filterQuery.trim()) {
      const q = filterQuery.trim().toLowerCase();
      filtered = filtered.filter((m) => m.gene.toLowerCase().includes(q));
    }
    return filtered;
  }, [currentMarkers, filterQuery, minAbsFC]);

  const sortedMarkers = useMemo(() => {
    const sorted = [...filteredMarkers];
    sorted.sort((a, b) => {
      if (sortField === "gene")
        return sortDir === "asc" ? a.gene.localeCompare(b.gene) : b.gene.localeCompare(a.gene);
      const aVal = sortField === "logfoldchange" ? a.logfoldchange : a.pval_adj;
      const bVal = sortField === "logfoldchange" ? b.logfoldchange : b.pval_adj;
      return sortDir === "asc" ? aVal - bVal : bVal - aVal;
    });
    return sorted;
  }, [filteredMarkers, sortField, sortDir]);

  const handleSort = useCallback(
    (field: SortField) => {
      if (sortField === field) setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
      else { setSortField(field); setSortDir(field === "pval_adj" ? "asc" : "desc"); }
    },
    [sortField],
  );

  const sortIcon = useCallback(
    (field: SortField) => {
      if (sortField !== field) return <ArrowUpDown className="h-2.5 w-2.5 text-slate-400" />;
      return sortDir === "asc"
        ? <ArrowUp className="h-2.5 w-2.5 text-blue-500" />
        : <ArrowDown className="h-2.5 w-2.5 text-blue-500" />;
    },
    [sortField, sortDir],
  );

  const handleExportCSV = useCallback(() => {
    if (sortedMarkers.length === 0) return;
    downloadCsv(
      `markers_${selectedGroup}_${datasetId}.csv`,
      ["gene", "group", "log_fold_change", "p_value_adjusted"],
      sortedMarkers.map((m) => [
        m.gene,
        m.group,
        m.logfoldchange.toFixed(4),
        m.pval_adj.toExponential(4),
      ]),
    );
  }, [sortedMarkers, selectedGroup, datasetId]);

  if (!dataset || !datasetId) {
    return <div className="text-xs text-slate-400">No dataset loaded.</div>;
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
        <span className="ml-2 text-xs text-slate-500">Loading markers...</span>
      </div>
    );
  }

  if (noMarkers) {
    return (
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-center text-xs text-amber-700">
        No marker genes available. Run the preprocessing pipeline to compute markers.
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-xs text-red-700">{error}</div>
    );
  }

  if (!markersData) return null;

  return (
    <div className="flex flex-col gap-2">
      {/* Groupby + group selectors */}
      <div className="flex flex-wrap gap-2">
        {groupbyColumns.length > 0 && (
          <select
            value={groupByColumn}
            onChange={(e) => { setGroupByColumn(e.target.value); setFilterQuery(""); }}
            className="flex-1 min-w-0 rounded border border-slate-300 bg-white px-2 py-1 text-xs text-slate-700"
          >
            {groupbyColumns.map((col) => (
              <option key={col.name} value={col.name}>
                {col.name} ({col.n_unique})
              </option>
            ))}
          </select>
        )}
        <select
          value={selectedGroup}
          onChange={(e) => { setSelectedGroup(e.target.value); setFilterQuery(""); }}
          className="flex-1 min-w-0 rounded border border-slate-300 bg-white px-2 py-1 text-xs text-slate-700"
        >
          {markersData.groups.map((g) => (
            <option key={g} value={g}>
              {g} ({(markersData.markers[g] ?? []).length})
            </option>
          ))}
        </select>
      </div>

      {/* Filter + count */}
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            placeholder="Filter genes..."
            className="w-full rounded border border-slate-300 bg-white py-1 pl-7 pr-2 text-xs text-slate-700 placeholder-slate-400"
          />
        </div>
        <span className="text-[10px] text-slate-400 whitespace-nowrap">
          {sortedMarkers.length}/{currentMarkers.length}
        </span>
        <button
          onClick={handleExportCSV}
          disabled={sortedMarkers.length === 0}
          title="Export CSV"
          className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600 disabled:opacity-50"
        >
          <Download className="h-3 w-3" />
        </button>
      </div>

      {/* |FC| threshold filter */}
      <div className="flex items-center gap-2">
        <label className="text-[11px] font-medium text-slate-500 whitespace-nowrap">|FC| &ge;</label>
        <input
          type="range"
          min={0}
          max={5}
          step={0.25}
          value={minAbsFC}
          onChange={(e) => setMinAbsFC(parseFloat(e.target.value))}
          className="h-1 flex-1 cursor-pointer appearance-none rounded-lg bg-slate-200 accent-primary"
        />
        <span className="w-8 text-right text-[11px] tabular-nums text-slate-600">
          {minAbsFC.toFixed(minAbsFC % 1 === 0 ? 0 : 2)}
        </span>
      </div>

      {/* Compact table */}
      <div className="max-h-[calc(100vh-340px)] overflow-auto rounded-lg border border-slate-200">
        <table className="w-full text-xs">
          <thead className="sticky top-0 z-10">
            <tr className="border-b border-slate-200 bg-slate-50">
              <th
                onClick={() => handleSort("gene")}
                className="cursor-pointer px-2 py-1.5 text-left font-medium text-slate-600 hover:text-slate-800"
              >
                <span className="flex items-center gap-1">Gene {sortIcon("gene")}</span>
              </th>
              <th
                onClick={() => handleSort("logfoldchange")}
                className="cursor-pointer px-2 py-1.5 text-right font-medium text-slate-600 hover:text-slate-800"
              >
                <span className="flex items-center justify-end gap-1">LogFC {sortIcon("logfoldchange")}</span>
              </th>
              <th
                onClick={() => handleSort("pval_adj")}
                className="cursor-pointer px-2 py-1.5 text-right font-medium text-slate-600 hover:text-slate-800"
              >
                <span className="flex items-center justify-end gap-1">Adj.P {sortIcon("pval_adj")}</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedMarkers.map((marker) => (
              <tr
                key={`${marker.group}-${marker.gene}`}
                onClick={() => onGeneClick(marker.gene)}
                className="cursor-pointer border-b border-slate-50 transition-colors hover:bg-blue-50"
              >
                <td className="px-2 py-1.5">
                  <span className="font-mono font-medium text-slate-800">{marker.gene}</span>
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums">
                  <span className={marker.logfoldchange > 0 ? "text-emerald-600" : marker.logfoldchange < 0 ? "text-red-600" : "text-slate-600"}>
                    {marker.logfoldchange.toFixed(2)}
                  </span>
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums">
                  <span className={marker.pval_adj < 0.05 ? "font-medium text-blue-600" : "text-slate-600"}>
                    {formatPValue(marker.pval_adj)}
                  </span>
                </td>
              </tr>
            ))}
            {sortedMarkers.length === 0 && (
              <tr>
                <td colSpan={3} className="px-2 py-4 text-center text-slate-400">
                  {filterQuery ? "No genes match filter" : "No markers for this group"}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <p className="text-[10px] text-slate-400">Click a gene to view its expression on the scatter plot</p>
    </div>
  );
}
