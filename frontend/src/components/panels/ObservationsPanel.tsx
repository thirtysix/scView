import { useState, useEffect, useMemo, useCallback } from "react";
import { Table2, Loader2, AlertCircle, MousePointerClick, Download, Sparkles } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { plotConfig } from "@/lib/plotConfig";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";
import { apiFetch } from "@/api/client";
import { useCategoryColors, useCategoryOrder } from "@/hooks/useCategoryColors";
import { rgbCss, sortByCategoryOrder } from "@/lib/categoryColors";
import { formatNumber } from "@/lib/formatting";
import { downloadCsv } from "@/lib/csv";
import { suggestReplicateColumn } from "@/lib/de";
import { conditionColumns, suggestConditionColumn } from "@/lib/composition";
import {
  barLabel,
  barValue,
  crosstabBars,
  sortBars,
  type Bar,
  type BarSort,
  type Crosstab,
} from "@/lib/compositionBars";
import { WaffleChart } from "@/components/plots/WaffleChart";

/** The waffle's "every cell" option, beside one bar per row. */
const ALL_CELLS = "\u0000all";
const ctlCls = "rounded-md border border-slate-300 bg-white px-2 py-1 text-xs text-slate-700";

interface MetadataSummaryRaw {
  groupby: string | null;
  counts: Record<string, number>;
}

interface MetadataSummary {
  groupby: string;
  groups: Record<string, { count: number }>;
  total: number;
}

type CrosstabResponse = Crosstab;

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
  // Composition bars: a third column to split the bars by (e.g. the condition,
  // with one bar per donor), their order, counts or shares, and a waffle.
  const [splitCol, setSplitCol] = useState("");
  const [barSort, setBarSort] = useState<BarSort>("category");
  const [shareOf, setShareOf] = useState("");
  const [percent, setPercent] = useState(false);
  const [waffleKey, setWaffleKey] = useState(""); // a bar's label, ALL_CELLS, or "" = off

  // Each column's categories keep the colours they have on the embedding.
  const primaryColor = useCategoryColors(primaryCol);
  const colorByColor = useCategoryColors(colorByCol);
  // ...and the order they are stored in (bars, stacks and legend follow it).
  const primaryOrder = useCategoryOrder(primaryCol);
  const colorByOrder = useCategoryOrder(colorByCol);

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
    // After the group-by default, or both could land on the same column.
    if (!colorByCol && primaryCol && allCategoricalCols.length > 1) {
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

  // The split only counts when it is a third column.
  const facetCol = splitCol && splitCol !== primaryCol && splitCol !== colorByCol ? splitCol : "";

  // Fetch crosstab when both columns are set and different
  useEffect(() => {
    if (!datasetId || !primaryCol || !colorByCol || primaryCol === colorByCol) {
      setCrosstab(null);
      return;
    }

    let cancelled = false;
    setIsLoadingCrosstab(true);
    apiFetch<CrosstabResponse>(
      `/datasets/${datasetId}/metadata/crosstab?row=${encodeURIComponent(primaryCol)}&col=${encodeURIComponent(colorByCol)}` +
        (facetCol ? `&facet=${encodeURIComponent(facetCol)}` : ""),
    )
      .then((data) => !cancelled && setCrosstab(data))
      .catch(() => !cancelled && setCrosstab(null))
      .finally(() => !cancelled && setIsLoadingCrosstab(false));
    return () => {
      cancelled = true;
    };
  }, [datasetId, primaryCol, colorByCol, facetCol]);

  // A waffle and a share-of choice belong to the columns they were made for.
  useEffect(() => setWaffleKey(""), [primaryCol, colorByCol, facetCol]);
  useEffect(() => setShareOf(""), [colorByCol]);

  // One bar per donor / sample, split by condition, when the dataset has both.
  const perSample = useMemo(() => {
    const donor = suggestReplicateColumn(allCategoricalCols);
    if (!donor) return null;
    const condition = suggestConditionColumn(conditionColumns(allCategoricalCols, [donor]));
    return { donor, condition };
  }, [allCategoricalCols]);
  const showPerSample = useCallback(() => {
    if (!perSample) return;
    // The stack keeps the cell types (or whatever the bars were before).
    const stack =
      primaryCol !== perSample.donor && primaryCol !== perSample.condition ? primaryCol : colorByCol;
    setPrimaryCol(perSample.donor);
    if (stack && stack !== perSample.donor) setColorByCol(stack);
    setSplitCol(perSample.condition);
    setSelectedGroup(null);
    setHighlight(null);
  }, [perSample, primaryCol, colorByCol, setHighlight]);

  // Handle row click: highlight cells on embedding
  const handleGroupClick = useCallback(
    (groupName: string) => {
      if (selectedGroup === groupName) {
        setSelectedGroup(null);
        setHighlight(null);
        setSelection(null);
      } else {
        setSelectedGroup(groupName);
        // Colour by this column so Unified View can select the group's cells.
        useSettingsStore.getState().setColorBy(primaryCol);
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

  // The bars in display order: one per row, or per (split value, row) with cells.
  const split = !!(facetCol && crosstab?.facet_values);
  const bars = useMemo<Bar[]>(() => {
    if (!crosstab) return [];
    return sortBars(crosstabBars(crosstab, split), barSort, {
      rowOrder: primaryOrder,
      colValues: crosstab.col_values,
      shareOf: shareOf || crosstab.col_values[0],
      facetOrder: crosstab.facet_values,
    });
  }, [crosstab, split, barSort, primaryOrder, shareOf]);
  // Stacked bottom-up, and listed, in the stacked column's stored order.
  const stackOrder = useMemo(
    () => (crosstab ? sortByCategoryOrder(crosstab.col_values, colorByOrder) : []),
    [crosstab, colorByOrder],
  );

  // Export the composition matrix (one row per bar, as shown) when a crosstab is available.
  const handleExportCrosstab = useCallback(() => {
    if (!crosstab) return;
    const headers = [...(split ? [facetCol] : []), primaryCol, ...stackOrder];
    const idx = stackOrder.map((v) => crosstab.col_values.indexOf(v));
    const rows = bars.map((b) => [
      ...(split ? [b.facet ?? ""] : []),
      b.row,
      ...idx.map((ci) => (percent ? barValue(b, ci, true).toFixed(2) : barValue(b, ci, false))),
    ]);
    const what = percent ? "percent" : "counts";
    const by = split ? `_split_${facetCol}` : "";
    downloadCsv(`composition_${primaryCol}_by_${colorByCol}${by}_${what}_${datasetId}.csv`, headers, rows);
  }, [crosstab, bars, stackOrder, split, facetCol, percent, primaryCol, colorByCol, datasetId]);

  // Batch-integration quality: how evenly each cluster mixes across the colorBy
  // groups (e.g. batches/conditions), via normalized Shannon entropy. 1.0 = a
  // cluster is split evenly across all groups (well-integrated); 0 = single-group.
  // The overall score is weighted by cluster size. Only meaningful when colorBy is
  // a batch/condition with ≥2 groups.
  const mixing = useMemo(() => {
    if (!crosstab || crosstab.col_values.length < 2 || crosstab.row_values.length === 0) return null;
    const k = crosstab.col_values.length;
    const logK = Math.log(k);
    const perCluster: { name: string; score: number; n: number }[] = [];
    let weightedSum = 0;
    let totalN = 0;
    crosstab.row_values.forEach((name, ri) => {
      const counts = crosstab.col_values.map((_, ci) => crosstab.counts[ri]?.[ci] ?? 0);
      const n = counts.reduce((a, b) => a + b, 0);
      if (n === 0) return;
      let h = 0;
      for (const c of counts) {
        if (c > 0) {
          const p = c / n;
          h -= p * Math.log(p);
        }
      }
      const score = logK > 0 ? h / logK : 0;
      perCluster.push({ name, score, n });
      weightedSum += score * n;
      totalN += n;
    });
    if (totalN === 0) return null;
    perCluster.sort((a, b) => a.score - b.score);
    return { overall: weightedSum / totalN, perCluster };
  }, [crosstab]);

  // Build bar chart traces — stacked bars when crosstab is available
  // Without a second column: one bar per group, in the chosen order.
  const summaryBars = useMemo<Bar[]>(() => {
    if (!summary) return [];
    const plain = Object.entries(summary.groups).map(([row, g]) => ({
      row,
      facet: null,
      counts: [g.count],
      total: g.count,
    }));
    return sortBars(plain, barSort === "share" ? "size" : barSort, { rowOrder: primaryOrder, colValues: [] });
  }, [summary, barSort, primaryOrder]);

  const compositionTraces = useMemo(() => {
    // Stacked bars from crosstab
    if (crosstab && bars.length > 0 && crosstab.col_values.length > 0) {
      const x = split ? [bars.map((b) => b.facet ?? ""), bars.map((b) => b.row)] : bars.map((b) => b.row);
      return stackOrder.map((colVal) => {
        const ci = crosstab.col_values.indexOf(colVal);
        return {
          type: "bar" as const,
          x,
          y: bars.map((b) => barValue(b, ci, percent)),
          customdata: bars.map((b) => [b.counts[ci] ?? 0, barValue(b, ci, true), barLabel(b)]),
          hovertemplate: `%{customdata[2]}<br>${colVal}: %{customdata[0]:,} cells (%{customdata[1]:.1f}%)<extra></extra>`,
          name: colVal,
          marker: {
            color: rgbCss(colorByColor(colVal)),
          },
        };
      });
    }

    // Fallback: simple bars from summary
    if (summaryBars.length === 0) return [];
    const total = summary?.total ?? 0;
    const groupNames = summaryBars.map((b) => b.row);
    return [
      {
        type: "bar" as const,
        x: groupNames,
        y: summaryBars.map((b) => (percent ? (total > 0 ? (b.total / total) * 100 : 0) : b.total)),
        name: "Cell count",
        marker: {
          color: groupNames.map((name) => rgbCss(primaryColor(name))),
        },
      },
    ];
  }, [crosstab, bars, split, stackOrder, percent, summaryBars, summary, primaryColor, colorByColor]);

  const compositionLayout = useMemo(
    () => ({
      barmode: "stack" as const,
      title: {
        text: crosstab
          ? `${colorByCol} Composition per ${primaryCol}${split ? ` by ${facetCol}` : ""}`
          : `Cell Counts per ${primaryCol}`,
        font: { size: 14, color: "#334155" },
      },
      xaxis: {
        title: { text: split ? `${facetCol} / ${primaryCol}` : primaryCol, font: { size: 12, color: "#64748b" } },
        tickfont: { size: 10, color: "#64748b" },
        tickangle: -45,
        ...(split ? { type: "multicategory" as const } : {}),
      },
      yaxis: {
        title: { text: percent ? "% of cells" : "Count", font: { size: 12, color: "#64748b" } },
        tickfont: { size: 11, color: "#64748b" },
        gridcolor: "#f1f5f9",
        ...(percent ? { range: [0, 100] } : {}),
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
    [primaryCol, colorByCol, crosstab, split, facetCol, percent],
  );

  // The waffle: one bar's make-up, or every cell's, by the stacked column.
  const waffle = useMemo(() => {
    if (!crosstab || !waffleKey) return null;
    const idx = stackOrder.map((v) => crosstab.col_values.indexOf(v));
    let counts: number[];
    let title: string;
    if (waffleKey === ALL_CELLS) {
      counts = idx.map((ci) => crosstab.counts.reduce((a, row) => a + (row[ci] ?? 0), 0));
      title = `All cells by ${colorByCol}`;
    } else {
      const bar = bars.find((b) => barLabel(b) === waffleKey);
      if (!bar) return null;
      counts = idx.map((ci) => bar.counts[ci] ?? 0);
      title = `${primaryCol} ${barLabel(bar)} by ${colorByCol}`;
    }
    return {
      title,
      categories: stackOrder.map((name, i) => ({
        name,
        count: counts[i] ?? 0,
        color: rgbCss(colorByColor(name)),
      })),
    };
  }, [crosstab, waffleKey, stackOrder, bars, primaryCol, colorByCol, colorByColor]);

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
                    .map(([name, grp]) => {
                      const pct = summary.total > 0 ? (grp.count / summary.total) * 100 : 0;
                      const color = primaryColor(name);
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

          {/* Batch-integration quality */}
          {mixing && (
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="mb-2 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-700">
                  Batch integration quality
                  <span className="ml-2 font-normal text-slate-400">
                    how evenly {primaryCol} clusters mix across {colorByCol}
                  </span>
                </h3>
                <button
                  onClick={() =>
                    askCopilot(
                      `Are my ${primaryCol} clusters well-integrated across ${colorByCol}? ` +
                        `The overall batch-mixing score is ${mixing.overall.toFixed(2)} (1.0 = fully mixed). ` +
                        `Should I run or revisit batch integration?`,
                    )
                  }
                  title="Ask the co-pilot about integration"
                  className="inline-flex items-center gap-1 rounded-full border border-primary/30 px-2 py-0.5 text-[11px] font-medium text-primary hover:bg-primary/10"
                >
                  <Sparkles className="h-3 w-3" />
                  Ask
                </button>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-2xl font-bold tabular-nums text-slate-800">
                  {mixing.overall.toFixed(2)}
                </span>
                <div className="flex-1">
                  <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.round(mixing.overall * 100)}%`,
                        backgroundColor:
                          mixing.overall >= 0.66
                            ? "#22c55e"
                            : mixing.overall >= 0.33
                              ? "#f59e0b"
                              : "#ef4444",
                      }}
                    />
                  </div>
                  <div className="mt-1 text-[11px] text-slate-400">
                    0 = each cluster is one {colorByCol} group · 1 = evenly mixed across all{" "}
                    {colorByCol} groups
                  </div>
                </div>
              </div>
              {mixing.perCluster.length > 1 && (
                <div className="mt-2 text-[11px] text-slate-500">
                  Least mixed: {mixing.perCluster.slice(0, 3).map((c) => `${c.name} (${c.score.toFixed(2)})`).join(", ")}
                </div>
              )}
            </div>
          )}

          {/* Composition bar chart */}
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-2 text-xs text-slate-500">
              <label className="flex items-center gap-1.5">
                Sort
                <select
                  aria-label="Sort bars"
                  value={barSort}
                  onChange={(e) => setBarSort(e.target.value as BarSort)}
                  className={ctlCls}
                >
                  <option value="category">Stored order</option>
                  <option value="name">Name</option>
                  <option value="size">Size</option>
                  {crosstab && <option value="share">Share of…</option>}
                </select>
              </label>
              {crosstab && barSort === "share" && (
                <select
                  aria-label="Share of"
                  value={shareOf || stackOrder[0] || ""}
                  onChange={(e) => setShareOf(e.target.value)}
                  className={ctlCls}
                >
                  {stackOrder.map((v) => (
                    <option key={v} value={v}>
                      % {v}
                    </option>
                  ))}
                </select>
              )}
              <div className="flex overflow-hidden rounded-md border border-slate-300" role="group" aria-label="Bar values">
                {(["Counts", "%"] as const).map((label) => {
                  const on = (label === "%") === percent;
                  return (
                    <button
                      key={label}
                      aria-pressed={on}
                      onClick={() => setPercent(label === "%")}
                      className={`px-2 py-1 ${on ? "bg-primary/10 font-medium text-primary" : "bg-white text-slate-600 hover:bg-slate-50"}`}
                    >
                      {label}
                    </button>
                  );
                })}
              </div>
              {crosstab && (
                <label className="flex items-center gap-1.5">
                  Split by
                  <select
                    aria-label="Split bars by"
                    value={facetCol}
                    onChange={(e) => setSplitCol(e.target.value)}
                    className={ctlCls}
                  >
                    <option value="">None</option>
                    {allCategoricalCols
                      .filter((c) => c.name !== primaryCol && c.name !== colorByCol && c.n_unique <= 20)
                      .map((c) => (
                        <option key={c.name} value={c.name}>
                          {c.name}
                        </option>
                      ))}
                  </select>
                </label>
              )}
              {perSample && !(primaryCol === perSample.donor && facetCol === perSample.condition) && (
                <button
                  onClick={showPerSample}
                  title={`One bar per ${perSample.donor}${perSample.condition ? `, split by ${perSample.condition}` : ""}`}
                  className="rounded-md border border-primary/30 px-2 py-1 font-medium text-primary hover:bg-primary/10"
                >
                  Per {perSample.donor}
                  {perSample.condition ? ` by ${perSample.condition}` : ""}
                </button>
              )}
              {crosstab && (
                <label className="flex items-center gap-1.5">
                  Waffle
                  <select
                    aria-label="Waffle of"
                    value={waffleKey}
                    onChange={(e) => setWaffleKey(e.target.value)}
                    className={`${ctlCls} max-w-[12rem]`}
                  >
                    <option value="">Off</option>
                    <option value={ALL_CELLS}>All cells</option>
                    {bars.map((b) => (
                      <option key={barLabel(b)} value={barLabel(b)}>
                        {barLabel(b)}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {crosstab && (
                <button
                  onClick={handleExportCrosstab}
                  title="Export the composition as shown (counts or %) as CSV"
                  className="ml-auto flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-xs text-slate-600 hover:bg-slate-50"
                >
                  <Download className="h-3.5 w-3.5" />
                  CSV
                </button>
              )}
            </div>
            <Plot
              data={compositionTraces}
              layout={compositionLayout}
              config={plotConfig(`composition_${primaryCol}`)}
              onClick={(e) => {
                // A bar's click shows its waffle.
                const i = e.points?.[0]?.pointIndex;
                const bar = typeof i === "number" ? bars[i] : undefined;
                if (crosstab && bar) setWaffleKey(barLabel(bar));
              }}
              useResizeHandler
              className="w-full"
              style={{ width: "100%" }}
            />
            {crosstab && !waffle && (
              <p className="mt-1 text-[11px] text-slate-400">Click a bar to see its make-up as a 10 x 10 waffle.</p>
            )}
            {waffle && (
              <div className="mt-3 border-t border-slate-100 pt-3">
                <WaffleChart categories={waffle.categories} title={waffle.title} />
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
