import { useMemo } from "react";
import { Loader2 } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { useCategoryColors, useCategoryOrder } from "@/hooks/useCategoryColors";
import { useGeneExpression } from "@/hooks/useGeneExpression";
import { useObsCodes } from "@/hooks/useObsCodes";
import { rgbCss, sortByCategoryOrder } from "@/lib/categoryColors";
import { prettyObsLabel } from "@/lib/formatting";
import {
  cellsWithCode,
  pearson,
  quadrantCells,
  quadrantCounts,
  spearmanOf,
  strideSample,
  type Quadrant,
  type QuadrantCounts,
} from "@/lib/geneCoexpression";
import { plotConfig } from "@/lib/plotConfig";
import { SelectionChip } from "@/components/common/SelectionChip";
import { useLinkedSelection } from "@/hooks/useLinkedSelection";
import { SELECTION_COLOR, cellsInPlotlySelection, type PlotlySelection } from "@/lib/linkedSelection";
import { selectedPoints, selectionName } from "@/lib/linkedTraces";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";

/** Points drawn (the statistics use every cell). */
const MAX_POINTS = 20000;
const AXIS = { tickfont: { size: 10, color: "#64748b" }, gridcolor: "#f1f5f9", zeroline: false };

interface GeneVsGeneProps {
  datasetId: string;
  geneA: string;
  geneB: string;
  /** Colour the cells by this column (the current group-by) */
  groupColumn: string;
}

interface Stats {
  pearson: number | null;
  spearman: number | null;
  quadrants: QuadrantCounts;
  n: number;
}

function statsOf(a: Float32Array, b: Float32Array, subset: Uint32Array | null): Stats {
  return {
    pearson: pearson(a, b, subset),
    spearman: spearmanOf(a, b, subset),
    quadrants: quadrantCounts(a, b, subset),
    n: subset ? subset.length : a.length,
  };
}

const fmtR = (r: number | null) => (r == null ? "n/a" : r.toFixed(2));

/**
 * Gene A against gene B, one point per cell coloured by the current group-by,
 * with Pearson and Spearman r (all cells, and within the highlighted group)
 * and how many cells express both, one or neither. A quadrant's count selects
 * its cells, as a lasso would.
 */
export function GeneVsGene({ datasetId, geneA, geneB, groupColumn }: GeneVsGeneProps) {
  const layer = useSettingsStore((s) => s.expressionLayer);
  const embedding = useSettingsStore((s) => s.embedding);
  const highlighted = useSelectionStore((s) => s.highlightedGroup);
  const setSelection = useSelectionStore((s) => s.setSelection);
  const setHighlight = useSelectionStore((s) => s.setHighlight);
  const colorOf = useCategoryColors(groupColumn);
  const order = useCategoryOrder(groupColumn);
  const hlOrder = useCategoryOrder(highlighted?.column);

  const linked = useLinkedSelection();
  const selected = linked.selected;
  const genes = useMemo(() => [geneA, geneB], [geneA, geneB]);
  const { values, missing, isLoading, error } = useGeneExpression(datasetId, genes, layer);
  const { codes } = useObsCodes(datasetId, embedding, groupColumn);
  const { codes: hlCodes } = useObsCodes(datasetId, embedding, highlighted?.column);
  const a = values?.[geneA] ?? null;
  const b = values?.[geneB] ?? null;

  const all = useMemo(() => (a && b ? statsOf(a, b, null) : null), [a, b]);
  const inGroup = useMemo(() => {
    if (!a || !b || !highlighted || !hlCodes || hlCodes.length !== a.length) return null;
    const code = hlOrder.indexOf(highlighted.value);
    if (code < 0) return null;
    return statsOf(a, b, cellsWithCode(hlCodes, code));
  }, [a, b, highlighted, hlCodes, hlOrder]);

  const traces = useMemo(() => {
    if (!a || !b) return [];
    const drawn = strideSample(a.length, MAX_POINTS);
    const n = drawn ? drawn.length : a.length;
    const byGroup = new Map<string, { x: number[]; y: number[] }>();
    for (let k = 0; k < n; k++) {
      const i = drawn ? drawn[k]! : k;
      const group = codes && codes.length === a.length ? order[codes[i]!] ?? String(codes[i]) : "cells";
      let t = byGroup.get(group);
      if (!t) byGroup.set(group, (t = { x: [], y: [] }));
      t.x.push(a[i]!);
      t.y.push(b[i]!);
    }
    const groupTraces = sortByCategoryOrder([...byGroup.keys()], order).map((group) => ({
      type: "scattergl" as const,
      mode: "markers" as const,
      name: group,
      x: byGroup.get(group)!.x,
      y: byGroup.get(group)!.y,
      // Faded under a selection, which is drawn on top (C035).
      marker: { size: 3, color: group === "cells" ? "#2563eb" : rgbCss(colorOf(group)), opacity: selected ? 0.15 : 0.5 },
      hovertemplate: `${group}<br>${geneA} %{x:.2f}<br>${geneB} %{y:.2f}<extra></extra>`,
    }));
    if (!selected) return groupTraces;
    const s = selectedPoints(a, b, selected, MAX_POINTS);
    return [
      ...groupTraces,
      {
        type: "scattergl" as const,
        mode: "markers" as const,
        name: selectionName(s.n),
        x: s.x,
        y: s.y,
        marker: { size: 4, color: SELECTION_COLOR, opacity: 0.85 },
        hovertemplate: `selected<br>${geneA} %{x:.2f}<br>${geneB} %{y:.2f}<extra></extra>`,
      },
    ];
  }, [a, b, codes, order, colorOf, geneA, geneB, selected]);

  // Reverse direction: a box or lasso drawn here selects every cell inside it
  // (all cells, not only the points drawn).
  const handleSelected = (e: Readonly<PlotlySelection> | undefined) => {
    if (!a || !b) return;
    const cells = cellsInPlotlySelection(a, b, e);
    if (cells) linked.select(cells);
  };

  const select = (q: Quadrant) => {
    if (!a || !b) return;
    const cells = quadrantCells(a, b, q);
    setHighlight(null); // a set of cells, not a group (as a lasso)
    setSelection(cells.size > 0 ? cells : null);
  };

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 py-6 text-xs text-slate-400">
        <Loader2 className="h-4 w-4 animate-spin" /> Loading {geneA} and {geneB}…
      </div>
    );
  }
  if (error) return <p className="text-xs text-red-600">Could not load {geneA} and {geneB}.</p>;
  if (!a || !b || !all) {
    return (
      <p className="text-xs text-amber-700">
        {missing.join(" and ")} not found in this expression layer.
      </p>
    );
  }

  const pct = (k: number, n: number) => (n ? `${((100 * k) / n).toFixed(1)}%` : "0%");
  const q = all.quadrants;
  const cell = (quadrant: Quadrant, label: string) => (
    <button
      type="button"
      onClick={() => select(quadrant)}
      title={`Select the ${q[quadrant].toLocaleString()} ${label} cells`}
      className="rounded border border-slate-200 px-1.5 py-1 text-left hover:border-primary hover:bg-blue-50"
    >
      <div className="text-[10px] text-slate-500">{label}</div>
      <div className="font-semibold text-slate-700">
        {q[quadrant].toLocaleString()} <span className="font-normal text-slate-400">{pct(q[quadrant], all.n)}</span>
      </div>
    </button>
  );

  return (
    <div className="space-y-2">
      <Plot
        data={traces}
        layout={{
          height: 260,
          margin: { t: 8, r: 10, b: 40, l: 46 },
          xaxis: { ...AXIS, title: { text: geneA, font: { size: 11, color: "#64748b" } } },
          yaxis: { ...AXIS, title: { text: geneB, font: { size: 11, color: "#64748b" } } },
          showlegend: false,
          paper_bgcolor: "white",
          plot_bgcolor: "white",
          font: { family: "Inter, system-ui, sans-serif" },
        }}
        config={plotConfig(`gene_vs_gene_${geneA}_${geneB}`)}
        onSelected={handleSelected}
        useResizeHandler
        style={{ width: "100%" }}
      />
      <div className="flex flex-wrap items-center gap-1.5">
        <SelectionChip note="orange, over the faded rest" />
        <span className="text-[10px] text-slate-400">Box or lasso select (plot toolbar) to select cells.</span>
      </div>
      <p className="text-[11px] leading-snug text-slate-500" data-testid="gene-vs-gene-stats">
        All {all.n.toLocaleString()} cells: Pearson r {fmtR(all.pearson)}, Spearman ρ {fmtR(all.spearman)}
        {traces.reduce((s, t) => s + t.x.length, 0) < all.n ? ` (${MAX_POINTS.toLocaleString()} or fewer drawn)` : ""}.
        {inGroup && highlighted && (
          <>
            {" "}
            In {highlighted.value} ({prettyObsLabel(highlighted.column)}, {inGroup.n.toLocaleString()} cells): r{" "}
            {fmtR(inGroup.pearson)}, ρ {fmtR(inGroup.spearman)}; both expressed in{" "}
            {pct(inGroup.quadrants.both, inGroup.n)}.
          </>
        )}
      </p>
      <div className="grid grid-cols-2 gap-1 text-xs" aria-label="Quadrant counts">
        {cell("bOnly", `${geneB} only`)}
        {cell("both", "both")}
        {cell("neither", "neither")}
        {cell("aOnly", `${geneA} only`)}
      </div>
      <button
        type="button"
        onClick={() => select("both")}
        disabled={q.both === 0}
        className="w-full rounded bg-primary px-2 py-1 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-50"
      >
        Select co-expressing cells ({q.both.toLocaleString()})
      </button>
      <p className="text-[10px] text-slate-400">
        Expressed = value above 0 in the current layer. Coloured by {prettyObsLabel(groupColumn)}.
      </p>
    </div>
  );
}
