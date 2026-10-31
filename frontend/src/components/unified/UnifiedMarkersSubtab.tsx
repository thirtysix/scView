import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { SortableTh, ariaSort } from "@/components/common/SortableTh";
import {
  Loader2, Search, ArrowUpDown, ArrowUp, ArrowDown, Download, Sparkles, ChevronDown, ChevronRight,
} from "lucide-react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useObsRevision } from "@/stores/obsRevisionStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useViewStore } from "@/stores/viewStore";
import { useDelayedFlag } from "@/hooks/useDelayedFlag";
import { apiFetch } from "@/api/client";
import type { MarkersResponse } from "@/api/markers";
import { AmbientFlag, useAmbientFlags } from "@/components/common/AmbientFlag";
import { MarkerSpecificityPlot } from "@/components/plots/MarkerSpecificityPlot";
import { CountSplitCheck } from "@/components/unified/CountSplitCheck";
import type { SplitMarkersResult } from "@/api/splitMarkers";
import { splitPValueMap } from "@/lib/splitMarkers";
import { MARKERS_PER_GROUP, NO_MARKERS_REASON, noMarkersReason } from "@/api/markers";
import { AddToGeneListMenu } from "@/components/common/AddToGeneListMenu";
import { GeneReportButton } from "@/components/report/ReportButtons";
import { MinimalPanel } from "@/components/unified/MinimalPanel";
import { OffTargetPanel } from "@/components/unified/OffTargetPanel";
import { ProteinNetwork } from "@/components/unified/ProteinNetwork";
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

type SortField = MarkerSortField;

/** The highlighted group's name when it is a group of the markers' column, else null. */
export function highlightedMarkerGroup(
  groups: string[],
  groupByColumn: string,
  highlighted: { column: string; value: string } | null,
): string | null {
  if (!highlighted || highlighted.column !== groupByColumn) return null;
  return groups.includes(highlighted.value) ? highlighted.value : null;
}

/** What a newly highlighted group does to the Markers subtab: open it when it is a
 * group of the markers' column, or switch to its own column when that is one the
 * subtab can group by (the plot coloured by cell type, say); else nothing. */
export function followHighlight(
  highlighted: { column: string; value: string } | null,
  groupByColumn: string,
  groupableColumns: string[],
  groups: string[] | null,
): { column: string } | { group: string } | null {
  if (!highlighted) return null;
  if (highlighted.column !== groupByColumn) {
    return groupableColumns.includes(highlighted.column) ? { column: highlighted.column } : null;
  }
  const group = groups ? highlightedMarkerGroup(groups, groupByColumn, highlighted) : null;
  return group ? { group } : null;
}
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
  const askCopilot = useViewStore((s) => s.askCopilot);
  const highlighted = useSelectionStore((s) => s.highlightedGroup);
  const revision = useObsRevision(datasetId, groupByColumn); // refetch after hand labels

  const [markersData, setMarkersData] = useState<MarkersResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Why there are no markers (an unprocessed dataset: 200 with no groups), or null.
  const [noMarkers, setNoMarkers] = useState<string | null>(null);
  // Still loading after 1.5 s: the backend is computing this column's markers.
  const computingMarkers = useDelayedFlag(isLoading, 1500);

  const [selectedGroup, setSelectedGroup] = useState<string>("");
  const [sortField, setSortField] = useState<SortField>("logfoldchange");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [filterQuery, setFilterQuery] = useState("");
  const [minAbsFC, setMinAbsFC] = useState(0);
  const [showPlot, setShowPlot] = useState(false);
  // Count-split check (CountSplitCheck): its result, and its p-values as a table column.
  const [splitResult, setSplitResult] = useState<SplitMarkersResult | null>(null);
  const [showSplitP, setShowSplitP] = useState(false);

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
    setNoMarkers(null);

    // Always the grouping column's own markers: the backend serves stored ones
    // when it has them and computes them otherwise. Asking for the default
    // markers first showed the clustering's markers under another grouping.
    const url = groupByColumn
      ? `/datasets/${datasetId}/markers?format=json&groupby_column=${encodeURIComponent(groupByColumn)}`
      : `/datasets/${datasetId}/markers?format=json`;

    apiFetch<MarkersResponse>(url)
      .then((data) => {
        if (cancelled) return;
        const none = noMarkersReason(data);
        if (none) {
          setMarkersData(null);
          setNoMarkers(none);
          setIsLoading(false);
          return;
        }
        setMarkersData(data);
        const hl = useSelectionStore.getState().highlightedGroup;
        setSelectedGroup(highlightedMarkerGroup(data.groups, groupByColumn, hl) ?? data.groups[0] ?? "");
        setIsLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes("404")) setNoMarkers(NO_MARKERS_REASON);
        else setError(msg);
        setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [datasetId, groupByColumn, revision]);

  const { flags: ambientFlags, approximate: ambientApprox } = useAmbientFlags(datasetId, groupByColumn, markersData, dataset?.large);

  // A group clicked in the legend, the cluster map or an Observations row (or
  // highlighted by the co-pilot) opens its markers, switching to its column when
  // the markers are for another one. Each highlight is followed once, so picking
  // another grouping by hand afterwards sticks.
  const followed = useRef<typeof highlighted>(null);
  useEffect(() => {
    if (!highlighted || highlighted === followed.current) return;
    const action = followHighlight(
      highlighted, groupByColumn, groupbyColumns.map((c) => c.name), markersData?.groups ?? null,
    );
    if (!action) return; // markers still loading: the fetch opens the group
    followed.current = highlighted;
    setFilterQuery("");
    if ("column" in action) setGroupByColumn(action.column); // the fetch then opens the group
    else setSelectedGroup(action.group);
  }, [highlighted, markersData, groupByColumn, groupbyColumns, setGroupByColumn]);

  const currentMarkers = useMemo(() => {
    if (!markersData || !selectedGroup) return [];
    return markersData.markers[selectedGroup] ?? [];
  }, [markersData, selectedGroup]);

  const filteredMarkers = useMemo(() => {
    let filtered = currentMarkers;
    if (minAbsFC > 0) {
      filtered = filtered.filter((m) => Math.abs(m.logfoldchange ?? 0) >= minAbsFC);
    }
    if (filterQuery.trim()) {
      const q = filterQuery.trim().toLowerCase();
      filtered = filtered.filter((m) => m.gene.toLowerCase().includes(q));
    }
    return filtered;
  }, [currentMarkers, filterQuery, minAbsFC]);

  // A missing value (a NaN fold change, no specificity) sorts last either way.
  const sortedMarkers = useMemo(
    () => sortMarkers(filteredMarkers, sortField, sortDir),
    [filteredMarkers, sortField, sortDir],
  );
  const specificity = useMemo(() => hasSpecificity(currentMarkers), [currentMarkers]);
  const splitP = useMemo(
    () => (showSplitP && splitResult ? splitPValueMap(splitResult.groups.find((g) => g.group === selectedGroup)) : null),
    [showSplitP, splitResult, selectedGroup],
  );

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
      MARKER_CSV_HEADERS,
      sortedMarkers.map(markerCsvRow),
    );
  }, [sortedMarkers, selectedGroup, datasetId]);

  const askAboutGene = useCallback(
    (gene: string) => {
      const where = groupByColumn ? ` in the ${selectedGroup} group of ${groupByColumn}` : "";
      askCopilot(
        `What is ${gene}, a top marker${where}? What cell types is it associated with?`,
      );
    },
    [askCopilot, groupByColumn, selectedGroup],
  );

  if (!dataset || !datasetId) {
    return <div className="text-xs text-slate-400">No dataset loaded.</div>;
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
        <span className="ml-2 text-xs text-slate-500">
          {computingMarkers
            ? "Computing marker genes… this can take ~30 s on a large dataset"
            : "Loading markers..."}
        </span>
      </div>
    );
  }

  if (noMarkers) {
    return (
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-center text-xs text-amber-700" data-testid="no-markers">
        {noMarkers}
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
        <AddToGeneListMenu
          genes={() => sortedMarkers.map((m) => m.gene)}
          defaultName={`Markers: ${selectedGroup}`}
          source={`Markers of ${selectedGroup}${groupByColumn ? ` (${groupByColumn})` : ""}`}
          disabled={sortedMarkers.length === 0}
          title="Add the markers shown (after the filters, in table order) to a gene list"
        />
      </div>

      {/* |FC| threshold filter */}
      <div className="flex items-center gap-2">
        <label className="text-[11px] font-medium text-slate-500 whitespace-nowrap">|FC| &ge;</label>
        <input
          aria-label="Minimum |FC|"
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

      {/* Effect size vs specificity for the group's markers */}
      {specificity && (
        <div>
          <button
            onClick={() => setShowPlot((v) => !v)}
            aria-expanded={showPlot}
            className="flex items-center gap-1 text-[11px] text-slate-500 hover:text-slate-700"
          >
            {showPlot ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            Specificity vs fold change
          </button>
          {showPlot && (
            <div className="mt-1 rounded-lg border border-slate-200 p-1">
              <MarkerSpecificityPlot
                markers={currentMarkers}
                name={selectedGroup}
                onGeneClick={onGeneClick}
                height={220}
              />
              <p className="px-1 text-[10px] leading-snug text-slate-400">
                One point per marker ({currentMarkers.length.toLocaleString()} of {selectedGroup}&apos;s; the 6 strongest
                labelled). Top right: strong and found in this group only. A strong marker low on the plot is
                shared with other groups. Click a point to view the gene.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Compact table */}
      <div className="max-h-[calc(100vh-340px)] overflow-auto rounded-lg border border-slate-200">
        <table className="w-full text-xs">
          <thead className="sticky top-0 z-10">
            <tr className="border-b border-slate-200 bg-slate-50">
              <SortableTh
                onSort={() => handleSort("gene")}
                sort={ariaSort("gene", sortField, sortDir)}
                title="Gene symbol; click a row to colour the scatter by its expression"
                className="px-2 py-1.5 text-left font-medium text-slate-600"
              >
                Gene {sortIcon("gene")}
              </SortableTh>
              <SortableTh
                onSort={() => handleSort("logfoldchange")}
                sort={ariaSort("logfoldchange", sortField, sortDir)}
                title="log2 fold change of this group's mean expression over all other cells (scanpy rank_genes_groups: expm1 of the mean log-normalised values on each side)"
                className="px-2 py-1.5 text-right font-medium text-slate-600"
                align="right"
              >
                LogFC {sortIcon("logfoldchange")}
              </SortableTh>
              <SortableTh
                onSort={() => handleSort("pval_adj")}
                sort={ariaSort("pval_adj", sortField, sortDir)}
                title="For ranking only: the clusters were found from these same cells, so marker p-values are inflated."
                className="px-2 py-1.5 text-right font-medium text-slate-600"
                align="right"
              >
                Adj.P {sortIcon("pval_adj")}
              </SortableTh>
              {splitP && (
                <th
                  title="Adjusted p on held-out counts (count splitting): the group's training cluster vs the rest, tested on the half of the counts the clustering never saw. Blank: not among the check's top genes."
                  className="whitespace-nowrap px-2 py-1.5 text-right font-medium text-slate-600"
                >
                  Split adj.P
                </th>
              )}
              {specificity && (
                <>
                  <SortableTh
                    onSort={() => handleSort("tau")}
                    sort={ariaSort("tau", sortField, sortDir)}
                    title={SPECIFICITY_HELP.tau}
                    className="px-2 py-1.5 text-right font-medium text-slate-600"
                    align="right"
                  >
                    Tau {sortIcon("tau")}
                  </SortableTh>
                  <SortableTh
                    onSort={() => handleSort("pct_in")}
                    sort={ariaSort("pct_in", sortField, sortDir)}
                    title={SPECIFICITY_HELP.detection}
                    className="whitespace-nowrap px-2 py-1.5 text-right font-medium text-slate-600"
                    align="right"
                  >
                    In / next {sortIcon("pct_in")}
                  </SortableTh>
                  <SortableTh
                    onSort={() => handleSort("pct_ratio")}
                    sort={ariaSort("pct_ratio", sortField, sortDir)}
                    title={SPECIFICITY_HELP.ratio}
                    className="px-2 py-1.5 text-right font-medium text-slate-600"
                    align="right"
                  >
                    Ratio {sortIcon("pct_ratio")}
                  </SortableTh>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            {sortedMarkers.map((marker) => (
              <tr
                key={`${marker.group}-${marker.gene}`}
                onClick={() => onGeneClick(marker.gene)}
                className="group cursor-pointer border-b border-slate-50 transition-colors hover:bg-blue-50"
              >
                <td className="px-2 py-1.5">
                  <div className="flex items-center justify-between gap-1">
                    <span className="flex min-w-0 items-center gap-1">
                      <span className="font-mono font-medium text-slate-800">{marker.gene}</span>
                      <AmbientFlag flag={ambientFlags.get(marker.group)?.get(marker.gene)} group={marker.group} approximate={ambientApprox} />
                    </span>
                    <span className="flex items-center gap-1">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          askAboutGene(marker.gene);
                        }}
                        title="Ask the co-pilot about this gene"
                        className="flex-shrink-0 text-slate-300 opacity-0 transition-opacity hover:text-primary group-hover:opacity-100"
                      >
                        <Sparkles className="h-3 w-3" />
                      </button>
                      <GeneReportButton gene={marker.gene} groupby={groupByColumn} />
                    </span>
                  </div>
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums">
                  <span className={marker.logfoldchange == null ? "text-slate-400" : marker.logfoldchange > 0 ? "text-emerald-600" : marker.logfoldchange < 0 ? "text-red-600" : "text-slate-600"}>
                    {formatFoldChange(marker.logfoldchange, 2)}
                  </span>
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums">
                  <span className={marker.pval_adj < 0.05 ? "font-medium text-blue-600" : "text-slate-600"}>
                    {formatPValue(marker.pval_adj)}
                  </span>
                </td>
                {splitP && (
                  <td className="px-2 py-1.5 text-right tabular-nums" data-testid="split-p">
                    {splitP.get(marker.gene) == null ? (
                      <span className="text-slate-400">–</span>
                    ) : (
                      <span className={(splitP.get(marker.gene) as number) < 0.05 ? "font-medium text-blue-600" : "text-orange-700"}>
                        {formatPValue(splitP.get(marker.gene) as number)}
                      </span>
                    )}
                  </td>
                )}
                {specificity && (
                  <>
                    <td className="px-2 py-1.5 text-right tabular-nums text-slate-600">
                      {formatTau(marker.tau)}
                    </td>
                    <td
                      className="whitespace-nowrap px-2 py-1.5 text-right tabular-nums text-slate-600"
                      title={marker.max_other_group ? `Next-best group: ${marker.max_other_group}` : undefined}
                    >
                      {formatPct(marker.pct_in)} / {formatPct(marker.pct_max_other)}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-slate-600">
                      {formatRatio(detectionRatio(marker))}
                    </td>
                  </>
                )}
              </tr>
            ))}
            {sortedMarkers.length === 0 && (
              <tr>
                <td colSpan={(specificity ? 6 : 3) + (splitP ? 1 : 0)} className="px-2 py-4 text-center text-slate-400">
                  {filterQuery ? "No genes match filter" : "No markers for this group"}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <p className="text-[10px] text-slate-400" data-testid="markers-caption">
        {sortedMarkers.length.toLocaleString()} of the group&apos;s top {currentMarkers.length.toLocaleString()} markers by
        Wilcoxon score (every gene is tested; only the highest-ranked {MARKERS_PER_GROUP} per group are kept)
        {currentMarkers.length > sortedMarkers.length ? " after the filters" : ""}. Click a gene to view its
        expression on the scatter plot. Use p-values to rank markers, not as evidence: the clusters were found
        from these same cells, so markers always test as significant.
      </p>

      <CountSplitCheck
        datasetId={datasetId}
        column={groupByColumn}
        group={selectedGroup}
        onGroupChange={(g) => { setSelectedGroup(g); setFilterQuery(""); }}
        showSplitP={showSplitP}
        onShowSplitPChange={setShowSplitP}
        onResult={setSplitResult}
      />
      {datasetId && groupByColumn && (
        <OffTargetPanel datasetId={datasetId} column={groupByColumn} group={selectedGroup} onGeneClick={onGeneClick} />
      )}
      {datasetId && groupByColumn && <MinimalPanel datasetId={datasetId} groupby={groupByColumn} />}
      {datasetId && groupByColumn && selectedGroup && (
        <ProteinNetwork
          datasetId={datasetId}
          sources={[{ value: `markers:${groupByColumn}/${selectedGroup}`, label: `${selectedGroup} markers` }]}
          onGeneClick={onGeneClick}
        />
      )}
    </div>
  );
}
