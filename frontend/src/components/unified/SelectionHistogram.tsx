import { useEffect, useMemo, useState } from "react";
import Plot from "@/components/plots/Plot";
import { useObsCodes } from "@/hooks/useObsCodes";
import { useObsValues } from "@/hooks/useObsValues";
import { pseudoLogTicks } from "@/lib/logAxis";
import { isCategoricalDtype } from "@/lib/obsColumns";
import {
  compareSides,
  densities,
  effectLines,
  fmtValue,
  histograms,
  sidesFromGroups,
  sidesFromSelection,
  splitSides,
  toScale,
} from "@/lib/overlayHistogram";
import { plotConfig } from "@/lib/plotConfig";
import { prettyObsLabel } from "@/lib/formatting";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

// Blue and orange: told apart with any common colour-vision deficiency.
const COLOR_A = "#2563eb";
const COLOR_B = "#f97316";
const FILL: Record<string, string> = { [COLOR_A]: "rgba(37,99,235,0.18)", [COLOR_B]: "rgba(249,115,22,0.18)" };
const HEIGHT = 220;
// QC columns offered first when nothing is on the plot.
const PREFERRED = ["total_counts", "n_genes_by_counts", "pct_counts_mt", "n_counts", "n_genes"];

type Compare = "selection" | "group_rest" | "group_group";
const OVERLAY = "__overlay__";

interface SelectionHistogramProps {
  datasetId: string;
  /** Obs column the groups come from (the violin's grouping). */
  groupColumn?: string;
}

/**
 * N18: the cells of a selection (lasso, legend picks, a cell filter) or of a
 * group against the rest, or against another group, as two overlaid
 * normalised histograms or density curves of one value per cell: the gene or
 * score on the plot, or a numeric metadata / QC column. With n, medians and a
 * descriptive effect summary.
 */
export function SelectionHistogram({ datasetId, groupColumn }: SelectionHistogramProps) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const embedding = useSettingsStore((s) => s.embedding);
  const selected = useSelectionStore((s) => s.selectedCellIndices);
  const overlay = useUnifiedViewStore((s) => s.scatterOverlay);
  const overlayValues = useUnifiedViewStore((s) => s.overlayValues);
  const overlayLabel = useUnifiedViewStore((s) => s.overlayLabel);
  const overlayDatasetId = useUnifiedViewStore((s) => s.overlayDatasetId);
  const hasOverlay = !!overlay && !!overlayValues && overlayDatasetId === datasetId;

  const numericColumns = useMemo(() => {
    const cols = (dataset?.obs_columns ?? []).filter((c) => !isCategoricalDtype(c.dtype) && c.n_unique > 1);
    const rank = (n: string) => {
      const i = PREFERRED.indexOf(n);
      return i < 0 ? PREFERRED.length : i;
    };
    return [...cols].sort((a, b) => rank(a.name) - rank(b.name));
  }, [dataset]);

  // What to show: the overlay when there is one (it follows the plot), else a column.
  const [source, setSource] = useState<string>(OVERLAY);
  const effectiveSource =
    source === OVERLAY ? (hasOverlay ? OVERLAY : numericColumns[0]?.name ?? "") : source;
  const obsColumn = effectiveSource !== OVERLAY ? effectiveSource : null;
  const { values: obsValues, isLoading } = useObsValues(datasetId, embedding, obsColumn);
  const values = effectiveSource === OVERLAY ? overlayValues : obsValues;
  const isGene = effectiveSource === OVERLAY && overlay?.type === "expression" && !overlay.layer;
  const valueLabel = effectiveSource === OVERLAY ? overlayLabel : prettyObsLabel(effectiveSource);

  // Who is compared with whom.
  const groups = useMemo(
    () => dataset?.obs_columns.find((c) => c.name === groupColumn)?.values ?? [],
    [dataset, groupColumn],
  );
  const { codes } = useObsCodes(datasetId, embedding, groupColumn || null);
  const hasSelection = !!selected && selected.size > 0;
  const [compare, setCompare] = useState<Compare>(hasSelection ? "selection" : "group_rest");
  // A new selection is what the user wants to look at.
  useEffect(() => {
    if (hasSelection) setCompare("selection");
  }, [selected, hasSelection]);
  const effectiveCompare: Compare = compare === "selection" && !hasSelection ? "group_rest" : compare;
  const [groupA, setGroupA] = useState(0);
  const [groupB, setGroupB] = useState(1);
  useEffect(() => {
    setGroupA(0);
    setGroupB(1);
  }, [groupColumn]);

  const [density, setDensity] = useState(false);
  const [log, setLog] = useState(false);

  const n = values?.length ?? 0;
  const sides = useMemo(() => {
    if (!n) return null;
    if (effectiveCompare === "selection") return selected ? sidesFromSelection(n, selected) : null;
    if (!codes || codes.length !== n || !groups.length) return null;
    return sidesFromGroups(codes, groupA, effectiveCompare === "group_rest" ? "rest" : groupB);
  }, [n, effectiveCompare, selected, codes, groups.length, groupA, groupB]);

  const split = useMemo(() => (values && sides ? splitSides(values, sides) : null), [values, sides]);
  const stats = useMemo(() => (split ? compareSides(split.a, split.b) : null), [split]);
  const curves = useMemo(
    () => (split ? (density ? densities(split.a, split.b, 200, log) : histograms(split.a, split.b, 40, log)) : null),
    [split, density, log],
  );

  const nameA =
    effectiveCompare === "selection" ? "selected" : String(groups[groupA] ?? "group");
  const nameB =
    effectiveCompare === "group_group" ? String(groups[groupB] ?? "other group") : effectiveCompare === "selection" ? "other cells" : "rest";

  const ticks = useMemo(() => {
    if (!log || !split) return null;
    const ends = [split.a[0], split.a[split.a.length - 1], split.b[0], split.b[split.b.length - 1]].filter(
      (v): v is number => v !== undefined,
    );
    return ends.length ? pseudoLogTicks(Math.min(...ends), Math.max(...ends)) : null;
  }, [log, split]);

  const traces = useMemo(() => {
    if (!curves) return [];
    const one = (y: number[], name: string, color: string) =>
      density
        ? { type: "scatter" as const, mode: "lines" as const, x: curves.x, y, name, line: { color, width: 2 },
            fill: "tozeroy" as const, fillcolor: FILL[color], hovertemplate: "%{y:.3f}<extra>%{fullData.name}</extra>" }
        : { type: "bar" as const, x: curves.x, y, name, width: curves.width, marker: { color, opacity: 0.55 },
            hovertemplate: "%{y:.1%} of cells<extra>%{fullData.name}</extra>" };
    return [one(curves.a, `${nameA} (n = ${stats?.a.n.toLocaleString()})`, COLOR_A),
            one(curves.b, `${nameB} (n = ${stats?.b.n.toLocaleString()})`, COLOR_B)];
  }, [curves, density, nameA, nameB, stats]);

  const layout = useMemo(() => {
    const median = (v: number | undefined, color: string) =>
      v === undefined || !Number.isFinite(v)
        ? null
        : { type: "line" as const, xref: "x" as const, yref: "paper" as const, x0: toScale(v, log), x1: toScale(v, log),
            y0: 0, y1: 1, line: { color, width: 1.5, dash: "dash" as const } };
    return {
      barmode: "overlay" as const,
      bargap: 0,
      height: HEIGHT,
      autosize: true,
      margin: { t: 28, r: 12, b: 44, l: 52 },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      font: { family: "Inter, system-ui, sans-serif" },
      legend: { orientation: "h" as const, x: 0, y: 1.15, font: { size: 11 } },
      xaxis: {
        title: { text: log ? `${valueLabel} (log scale)` : valueLabel, font: { size: 12, color: "#64748b" } },
        tickfont: { size: 11, color: "#64748b" },
        zeroline: false,
        ...(ticks ? { tickmode: "array" as const, tickvals: ticks.tickvals, ticktext: ticks.ticktext } : {}),
      },
      yaxis: {
        title: { text: density ? "Density" : "Fraction of cells", font: { size: 12, color: "#64748b" } },
        tickfont: { size: 11, color: "#64748b" },
        gridcolor: "#f1f5f9",
        rangemode: "tozero" as const,
        ...(density ? {} : { tickformat: ".0%" }),
      },
      shapes: [median(stats?.a.median, COLOR_A), median(stats?.b.median, COLOR_B)].filter(
        (s): s is NonNullable<typeof s> => s !== null,
      ),
    };
  }, [log, ticks, valueLabel, density, stats]);

  const select = "rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700";
  const seg = (on: boolean) =>
    `px-1.5 py-0.5 ${on ? "bg-primary/10 font-medium text-primary" : "bg-white text-slate-600 hover:bg-slate-50"}`;

  let empty: string | null = null;
  if (!hasOverlay && !numericColumns.length) empty = "Pick a gene or score, or load a dataset with numeric metadata.";
  else if (!values) empty = isLoading ? "Loading values…" : "No values for this choice.";
  else if (effectiveCompare !== "selection" && !groupColumn) empty = "Choose a grouping for the violin first, or select cells.";
  else if (!stats || !stats.a.n || !stats.b.n) empty = `No cells with a value on one side (${nameA} or ${nameB}).`;

  return (
    <div className="space-y-1">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-slate-600">
        <label className="flex items-center gap-1">
          Values
          <select aria-label="Values" className={select} value={effectiveSource} onChange={(e) => setSource(e.target.value)}>
            {hasOverlay && <option value={OVERLAY}>{overlayLabel} (on the plot)</option>}
            {numericColumns.length > 0 && (
              <optgroup label="Cell metadata">
                {numericColumns.map((c) => (
                  <option key={c.name} value={c.name}>
                    {prettyObsLabel(c.name)}
                  </option>
                ))}
              </optgroup>
            )}
          </select>
        </label>
        <label className="flex items-center gap-1">
          Compare
          <select aria-label="Compare" className={select} value={effectiveCompare} onChange={(e) => setCompare(e.target.value as Compare)}>
            <option value="selection" disabled={!hasSelection}>
              {hasSelection ? `Selection (${selected!.size.toLocaleString()}) vs rest` : "Selection vs rest (select cells first)"}
            </option>
            <option value="group_rest" disabled={!groups.length}>One group vs rest</option>
            <option value="group_group" disabled={groups.length < 2}>Group vs group</option>
          </select>
        </label>
        {effectiveCompare !== "selection" && groups.length > 0 && (
          <>
            <select aria-label="Group" className={select} value={groupA} onChange={(e) => {
                const a = Number(e.target.value);
                setGroupA(a);
                if (a === groupB) setGroupB(a === 0 ? 1 : 0);
              }}>
              {groups.map((g, i) => (
                <option key={g} value={i}>{g}</option>
              ))}
            </select>
            {effectiveCompare === "group_group" && (
              <>
                <span>vs</span>
                <select aria-label="Other group" className={select} value={groupB} onChange={(e) => setGroupB(Number(e.target.value))}>
                  {groups.map((g, i) => (
                    <option key={g} value={i} disabled={i === groupA}>{g}</option>
                  ))}
                </select>
              </>
            )}
          </>
        )}
        <div className="ml-auto flex gap-1">
          <div className="flex overflow-hidden rounded border border-slate-300" role="group" aria-label="Curve">
            <button aria-pressed={!density} onClick={() => setDensity(false)} className={seg(!density)}>Histogram</button>
            <button aria-pressed={density} onClick={() => setDensity(true)} className={seg(density)}>Density</button>
          </div>
          <div
            className="flex overflow-hidden rounded border border-slate-300"
            role="group"
            aria-label="Value axis"
            title="Log value axis: sign(v) x log10(1 + |v|), so zeros and negative values stay; ticks in the original units"
          >
            <button aria-pressed={!log} onClick={() => setLog(false)} className={seg(!log)}>Linear</button>
            <button aria-pressed={log} onClick={() => setLog(true)} className={seg(log)}>Log</button>
          </div>
        </div>
      </div>

      {empty || !stats ? (
        <div className="flex items-center justify-center py-8 text-xs text-slate-400">{empty}</div>
      ) : (
        <>
          <Plot
            data={traces}
            layout={layout}
            config={plotConfig("selection_histogram")}
            useResizeHandler
            className="w-full"
            style={{ width: "100%", height: HEIGHT }}
          />
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 text-[11px] text-slate-600" aria-label="Summary">
            {([[nameA, stats.a, COLOR_A], [nameB, stats.b, COLOR_B]] as const).map(([name, s, color]) => (
              <div key={name} className="contents">
                <dt className="font-medium" style={{ color }}>{name}</dt>
                <dd>
                  n = {s.n.toLocaleString()} · median {fmtValue(s.median)} · mean {fmtValue(s.mean)}
                  {isGene && ` · ${Math.round(s.fracPositive * 100)}% expressing (> 0)`}
                </dd>
              </div>
            ))}
          </dl>
          <ul className="text-[11px] text-slate-500" aria-label="Effect summary">
            {effectLines(stats, nameA, nameB).map((line) => (
              <li key={line}>{line}</li>
            ))}
            <li className="text-slate-400">
              Descriptive only: cells from the same samples are not independent, so these are not tests. Use DE or the
              composition test for inference.
            </li>
          </ul>
        </>
      )}
    </div>
  );
}
