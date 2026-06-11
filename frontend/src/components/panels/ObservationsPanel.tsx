import { useState, useEffect, useMemo, useCallback } from "react";
import { Table2, Loader2, AlertCircle, MousePointerClick, Download, Sparkles } from "lucide-react";
import Plot from "react-plotly.js";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useViewStore } from "@/stores/viewStore";
import { apiFetch } from "@/api/client";
import { CATEGORICAL_COLORS } from "@/lib/colors";
import { formatNumber } from "@/lib/formatting";
import { downloadCsv } from "@/lib/csv";

interface MetadataSummaryRaw {
  groupby: string | null;
  counts: Record<string, number>;
}

interface MetadataSummary {
  groupby: string;
  groups: Record<string, { count: number }>;
  total: number;
}

interface CrosstabResponse {
  row_values: string[];
  col_values: string[];
  counts: number[][];
}

export function ObservationsPanel() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const setSelection = useSelectionStore((s) => s.setSelection);
  const setHighlight = useSelectionStore((s) => s.setHighlight);
  const askCopilot = useViewStore((s) => s.askCopilot);

  const [summary, setSummary] = useState<MetadataSummary | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [primaryCol, setPrimaryCol] = useState<string>("");
  const [colorByCol, setColorByCol] = useState<string>("");
  const [selectedGroup, setSelectedGroup] = useState<string | null>(null);

  const [crosstab, setCrosstab] = useState<CrosstabResponse | null>(null);
  const [isLoadingCrosstab, setIsLoadingCrosstab] = useState(false);

  // All categorical columns, excluding non-informative ones
  const allCategoricalCols = useMemo(() => {
    if (!dataset) return [];
    const nCells = dataset.n_cells ?? 0;
    return dataset.obs_columns.filter((c) => {
      if (c.dtype !== "category" && c.dtype !== "object" && c.dtype !== "bool") return false;
      if (c.n_unique > 100) return false;
      if (nCells > 0 && c.n_unique / nCells >= 0.9) return false;
      return true;
    });
  }, [dataset]);

  // Set defaults
  useEffect(() => {
    if (!primaryCol && allCategoricalCols.length > 0) {
      // Prefer active clustering, then first categorical column
      if (dataset?.active_clustering) {
        const found = allCategoricalCols.find((c) => c.name === dataset.active_clustering);
        if (found) { setPrimaryCol(found.name); return; }
      }
      setPrimaryCol(allCategoricalCols[0]!.name);
    }
  }, [allCategoricalCols, primaryCol, dataset]);

  useEffect(() => {
    if (!colorByCol && allCategoricalCols.length > 1) {
      const other = allCategoricalCols.find((c) => c.name !== primaryCol);
      if (other) setColorByCol(other.name);
    }
  }, [allCategoricalCols, colorByCol, primaryCol]);

  // Fetch summary
  useEffect(() => {
    if (!datasetId || !primaryCol) return;

    setIsLoading(true);
    setError(null);

    apiFetch<MetadataSummaryRaw>(
      `/datasets/${datasetId}/metadata/summary?groupby=${encodeURIComponent(primaryCol)}`,
    )
      .then((raw) => {
        if (!raw.groupby || !raw.counts) {
          setSummary(null);
          return;
        }
        const groups: Record<string, { count: number }> = {};
        let total = 0;
        for (const [key, count] of Object.entries(raw.counts)) {
          groups[key] = { count };
          total += count;
        }
        setSummary({ groupby: raw.groupby, groups, total });
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setIsLoading(false));
  }, [datasetId, primaryCol]);

  // Fetch crosstab when both columns are set and different
  useEffect(() => {
    if (!datasetId || !primaryCol || !colorByCol || primaryCol === colorByCol) {
      setCrosstab(null);
      return;
    }

    setIsLoadingCrosstab(true);
    apiFetch<CrosstabResponse>(
      `/datasets/${datasetId}/metadata/crosstab?row=${encodeURIComponent(primaryCol)}&col=${encodeURIComponent(colorByCol)}`,
    )
      .then((data) => setCrosstab(data))
      .catch(() => setCrosstab(null))
      .finally(() => setIsLoadingCrosstab(false));
  }, [datasetId, primaryCol, colorByCol]);

  // Handle row click: highlight cells on embedding
  const handleGroupClick = useCallback(
    (groupName: string) => {
      if (selectedGroup === groupName) {
        setSelectedGroup(null);
        setHighlight(null);
        setSelection(null);
      } else {
        setSelectedGroup(groupName);
        setHighlight({ column: primaryCol, value: groupName });
        setSelection(null);
      }
    },
    [selectedGroup, primaryCol, setSelection, setHighlight],
  );

  // Export the per-group summary (name, count, percentage).
  const handleExportSummary = useCallback(() => {
    if (!summary) return;
    const rows = Object.entries(summary.groups)
      .sort(([, a], [, b]) => b.count - a.count)
      .map(([name, grp]) => [
        name,
        grp.count,
        summary.total > 0 ? ((grp.count / summary.total) * 100).toFixed(2) : "0",
      ]);
    downloadCsv(`observations_${primaryCol}_${datasetId}.csv`, [primaryCol, "cell_count", "percentage"], rows);
  }, [summary, primaryCol, datasetId]);

  // Export the full row×column composition matrix when a crosstab is available.
  const handleExportCrosstab = useCallback(() => {
    if (!crosstab) return;
    const headers = [primaryCol, ...crosstab.col_values];
    const rows = crosstab.row_values.map((rv, ri) => [
      rv,
      ...crosstab.col_values.map((_, ci) => crosstab.counts[ri]?.[ci] ?? 0),
    ]);
    downloadCsv(`composition_${primaryCol}_by_${colorByCol}_${datasetId}.csv`, headers, rows);
  }, [crosstab, primaryCol, colorByCol, datasetId]);

  // Build bar chart traces — stacked bars when crosstab is available
  const compositionTraces = useMemo(() => {
    // Stacked bars from crosstab
    if (crosstab && crosstab.row_values.length > 0 && crosstab.col_values.length > 0) {
      return crosstab.col_values.map((colVal, ci) => {
        const c = CATEGORICAL_COLORS[ci % CATEGORICAL_COLORS.length]!;
        return {
          type: "bar" as const,
          x: crosstab.row_values,
          y: crosstab.row_values.map((_, ri) => crosstab.counts[ri]?.[ci] ?? 0),
          name: colVal,
          marker: {
            color: `rgb(${c[0]},${c[1]},${c[2]})`,
          },
        };
      });
    }

    // Fallback: simple bars from summary
    if (!summary) return [];

    const groupNames = Object.keys(summary.groups).sort(
      (a, b) => (summary.groups[b]?.count ?? 0) - (summary.groups[a]?.count ?? 0),
    );

    return [
      {
        type: "bar" as const,
        x: groupNames,
        y: groupNames.map((s) => summary.groups[s]?.count ?? 0),
        name: "Cell count",
        marker: {
          color: groupNames.map((_, i) => {
            const c = CATEGORICAL_COLORS[i % CATEGORICAL_COLORS.length]!;
            return `rgb(${c[0]},${c[1]},${c[2]})`;
          }),
        },
      },
    ];
  }, [summary, crosstab]);

  const compositionLayout = useMemo(
    () => ({
      barmode: "stack" as const,
      title: {
        text: crosstab
          ? `${colorByCol} Composition per ${primaryCol}`
          : `Cell Counts per ${primaryCol}`,
        font: { size: 14, color: "#334155" },
      },
      xaxis: {
        title: { text: primaryCol, font: { size: 12, color: "#64748b" } },
        tickfont: { size: 10, color: "#64748b" },
        tickangle: -45,
      },
      yaxis: {
        title: { text: "Count", font: { size: 12, color: "#64748b" } },
        tickfont: { size: 11, color: "#64748b" },
        gridcolor: "#f1f5f9",
      },
      legend: {
        font: { size: 10 },
        orientation: "h" as const,
        y: -0.3,
      },
      margin: { t: 40, r: 20, b: 100, l: 60 },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      autosize: true,
      height: 420,
    }),
    [primaryCol, colorByCol, crosstab],
  );

  // No dataset
  if (!dataset || !datasetId) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <Table2 className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">Observations</h2>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-12 text-center text-slate-400">
          Load a dataset to view observation metadata.
        </div>
      </div>
    );
  }

  // No categorical columns
  if (allCategoricalCols.length === 0) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <Table2 className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">Observations</h2>
        </div>
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-8 text-center">
          <AlertCircle className="mx-auto mb-2 h-8 w-8 text-amber-500" />
          <p className="text-sm text-amber-700">
            No categorical observation columns detected in this dataset.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Table2 className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">Observations</h2>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label htmlFor="primary-col-select" className="text-sm font-medium text-slate-600">
              Group by:
            </label>
            <select
              id="primary-col-select"
              value={primaryCol}
              onChange={(e) => {
                setPrimaryCol(e.target.value);
                setSelectedGroup(null);
                setHighlight(null);
              }}
              className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
            >
              {allCategoricalCols.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.name} ({col.n_unique} unique)
                </option>
              ))}
            </select>
          </div>
          {allCategoricalCols.length > 1 && (
            <div className="flex items-center gap-2">
              <label htmlFor="color-by-select" className="text-sm font-medium text-slate-600">
                Color by:
              </label>
              <select
                id="color-by-select"
                value={colorByCol}
                onChange={(e) => setColorByCol(e.target.value)}
                className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
              >
                {allCategoricalCols
                  .filter((c) => c.name !== primaryCol)
                  .map((col) => (
                    <option key={col.name} value={col.name}>
                      {col.name}
                    </option>
                  ))}
              </select>
            </div>
          )}
        </div>
      </div>

      {/* Loading */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
          <span className="ml-3 text-sm text-slate-500">Loading observation data...</span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Loading crosstab */}
      {isLoadingCrosstab && (
        <div className="flex items-center gap-2 text-xs text-blue-500">
          <Loader2 className="h-3 w-3 animate-spin" />
          Loading composition data...
        </div>
      )}

      {summary && !isLoading && (
        <>
          {/* Summary table */}
          <div className="rounded-xl border border-slate-200 bg-white shadow-sm">
            <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
              <h3 className="text-sm font-semibold text-slate-700">
                {primaryCol} Summary
              </h3>
              <div className="flex items-center gap-3">
                <span className="flex items-center gap-1 text-xs text-slate-400">
                  <MousePointerClick className="h-3.5 w-3.5" />
                  Click to highlight on embedding
                </span>
                <button
                  onClick={handleExportSummary}
                  title="Export summary as CSV"
                  className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-xs text-slate-600 hover:bg-slate-50"
                >
                  <Download className="h-3.5 w-3.5" />
                  CSV
                </button>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-100 bg-slate-50">
                    <th className="px-4 py-2.5 text-left font-medium text-slate-600">
                      {primaryCol}
                    </th>
                    <th className="px-4 py-2.5 text-right font-medium text-slate-600">
                      Cell Count
                    </th>
                    <th className="px-4 py-2.5 text-right font-medium text-slate-600">
                      Percentage
                    </th>
                    <th className="px-4 py-2.5 text-left font-medium text-slate-600">
                      Distribution
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(summary.groups)
                    .sort(([, a], [, b]) => b.count - a.count)
                    .map(([name, grp], idx) => {
                      const pct = summary.total > 0 ? (grp.count / summary.total) * 100 : 0;
                      const color = CATEGORICAL_COLORS[idx % CATEGORICAL_COLORS.length]!;
                      const isSelected = selectedGroup === name;

                      return (
                        <tr
                          key={name}
                          onClick={() => handleGroupClick(name)}
                          className={`group cursor-pointer border-b border-slate-50 transition-colors ${
                            isSelected
                              ? "bg-blue-50 ring-1 ring-inset ring-blue-200"
                              : "hover:bg-slate-50"
                          }`}
                        >
                          <td className="px-4 py-2.5">
                            <div className="flex items-center gap-2">
                              <span
                                className="inline-block h-3 w-3 flex-shrink-0 rounded-full"
                                style={{
                                  backgroundColor: `rgb(${color[0]},${color[1]},${color[2]})`,
                                }}
                              />
                              <span className="font-medium text-slate-800">{name}</span>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setHighlight({ column: primaryCol, value: name });
                                  askCopilot(
                                    `What is the "${name}" group in ${primaryCol}? Summarize its marker genes and likely identity.`,
                                  );
                                }}
                                title="Ask the co-pilot about this group"
                                className="flex-shrink-0 text-slate-300 opacity-0 transition-opacity hover:text-primary group-hover:opacity-100"
                              >
                                <Sparkles className="h-3.5 w-3.5" />
                              </button>
                            </div>
                          </td>
                          <td className="px-4 py-2.5 text-right tabular-nums text-slate-600">
                            {formatNumber(grp.count)}
                          </td>
                          <td className="px-4 py-2.5 text-right tabular-nums text-slate-600">
                            {pct.toFixed(1)}%
                          </td>
                          <td className="px-4 py-2.5">
                            <div className="h-2.5 w-full max-w-[200px] overflow-hidden rounded-full bg-slate-100">
                              <div
                                className="h-full rounded-full"
                                style={{
                                  width: `${pct}%`,
                                  backgroundColor: `rgb(${color[0]},${color[1]},${color[2]})`,
                                }}
                              />
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
            <div className="border-t border-slate-100 px-4 py-2 text-xs text-slate-500">
              Total: {formatNumber(summary.total)} cells across{" "}
              {Object.keys(summary.groups).length} groups
            </div>
          </div>

          {/* Composition bar chart */}
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            {crosstab && (
              <div className="mb-2 flex justify-end">
                <button
                  onClick={handleExportCrosstab}
                  title="Export composition matrix as CSV"
                  className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-xs text-slate-600 hover:bg-slate-50"
                >
                  <Download className="h-3.5 w-3.5" />
                  CSV
                </button>
              </div>
            )}
            <Plot
              data={compositionTraces}
              layout={compositionLayout}
              config={{ responsive: true, displayModeBar: false }}
              useResizeHandler
              className="w-full"
              style={{ width: "100%" }}
            />
          </div>
        </>
      )}
    </div>
  );
}
