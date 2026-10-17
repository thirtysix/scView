import { useMemo } from "react";
import { Download, Loader2 } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { plotConfig } from "@/lib/plotConfig";
import type { TrendGenesResult } from "@/api/trajectory";
import { downloadCsv } from "@/lib/csv";
import { prettyObsLabel } from "@/lib/formatting";
import { PATTERN_LABEL, heatmapOrder, scaleRow, tickBins, trendGenesCsv } from "@/lib/trendGenes";

interface TrendGenesViewProps {
  result: TrendGenesResult | null;
  loading: boolean;
  error: string | null;
  /** The gene clicked last: its trend is plotted and the embedding coloured by it */
  focusGene: string | null;
  onFocusGene: (gene: string) => void;
  /** Where a branch can end (groups the PAGA tree reaches from the root group) */
  branchEnds: string[];
  branchEnd: string;
  onBranchEndChange: (group: string) => void;
  /** The groups on the chosen branch, root first */
  path: string[] | null;
  /** Why no branch can be picked, when none can */
  branchNote: string | null;
  /** The PAGA grouping the branch groups belong to */
  groupby: string | null;
  exportName: string;
}

const AXIS = { tickfont: { size: 10, color: "#64748b" }, gridcolor: "#f1f5f9" };
const fmt = (v: number | null, d = 2) => (v == null ? "–" : v.toFixed(d));

/** Genes changing along pseudotime: a heatmap of the top genes (rows ordered by
 *  where their trend peaks), and the clicked gene's binned trend with its fit. */
export function TrendGenesView({
  result,
  loading,
  error,
  focusGene,
  onFocusGene,
  branchEnds,
  branchEnd,
  onBranchEndChange,
  path,
  branchNote,
  groupby,
  exportName,
}: TrendGenesViewProps) {
  const ordered = useMemo(() => (result ? heatmapOrder(result.genes) : []), [result]);
  const focus = result?.genes.find((g) => g.gene === focusGene) ?? null;
  const bins = useMemo(() => result?.bins.pseudotime ?? [], [result]);
  const ticks = tickBins(bins.length);

  const heatmap = useMemo(() => {
    if (!result || ordered.length === 0) return null;
    return {
      type: "heatmap" as const,
      z: ordered.map((g) => scaleRow(g.mean)),
      x: bins.map((_, i) => i),
      y: ordered.map((g) => g.gene),
      customdata: ordered.map((g) =>
        g.mean.map((m, b) => [bins[b] ?? 0, m, PATTERN_LABEL[g.pattern]]),
      ) as unknown as number[][],
      colorscale: "Viridis",
      zmin: 0,
      zmax: 1,
      colorbar: { title: { text: "scaled", side: "right" as const }, thickness: 10, len: 0.5 },
      hovertemplate:
        "<b>%{y}</b> (%{customdata[2]})<br>pseudotime %{customdata[0]:.3f}<br>mean %{customdata[1]:.3f}<extra></extra>",
    };
  }, [result, ordered, bins]);

  return (
    <div className="flex flex-col gap-2 text-left">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <label className="font-medium text-slate-600" htmlFor="trend-cells">
          Cells:
        </label>
        <select
          id="trend-cells"
          value={branchEnd}
          onChange={(e) => onBranchEndChange(e.target.value)}
          disabled={branchEnds.length === 0}
          className="rounded-lg border border-slate-300 bg-white px-2 py-1 text-sm text-slate-700 shadow-sm disabled:opacity-60"
        >
          <option value="">All cells with a pseudotime</option>
          {branchEnds.map((g) => (
            <option key={g} value={g}>
              Branch to {g}
            </option>
          ))}
        </select>
        {loading && <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-500" />}
        {result && result.genes.length > 0 && (
          <button
            onClick={() => {
              const { headers, rows } = trendGenesCsv(result);
              downloadCsv(`${exportName}_pseudotime_genes`, headers, rows);
            }}
            title="Download the ranked genes as CSV"
            className="ml-auto flex items-center gap-1 rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-600 hover:bg-slate-50"
          >
            <Download className="h-3 w-3" /> CSV
          </button>
        )}
      </div>
      {path && groupby && (
        <p className="text-xs text-slate-500" data-testid="trend-path">
          Branch on PAGA&apos;s spanning tree ({prettyObsLabel(groupby)}): {path.join(" → ")}
        </p>
      )}
      {!path && branchNote && <p className="text-xs text-slate-400">{branchNote}</p>}
      {error && <p className="text-xs text-red-600">{error}</p>}

      {result && (
        <p className="text-xs text-slate-500">
          Top {result.genes.length} of {result.n_genes_tested.toLocaleString()} genes tested on{" "}
          {result.n_cells.toLocaleString()} cells ({result.expression}; genes detected in &ge;{" "}
          {result.min_detected} cells). Ranked by a spline F-test along the cells&apos; pseudotime
          order; p-values are indicative only. Click a gene to plot it and colour the embedding.
        </p>
      )}

      {focus && (
        <div className="rounded-lg border border-slate-200 p-1" data-testid="trend-focus">
          <p className="px-1 text-xs text-slate-600">
            <b>{focus.gene}</b> {PATTERN_LABEL[focus.pattern]}, peak at pseudotime{" "}
            {fmt(focus.peak_pseudotime, 3)}; change {fmt(focus.amplitude)} (log), F {fmt(focus.f_stat, 1)},
            R² {fmt(focus.r2)}
          </p>
          <Plot
            data={[
              {
                type: "scatter" as const,
                mode: "markers" as const,
                x: bins,
                y: focus.mean,
                name: "bin mean",
                marker: { size: 5, color: "#94a3b8" },
                hovertemplate: "pseudotime %{x:.3f}<br>mean %{y:.3f}<extra></extra>",
              },
              {
                type: "scatter" as const,
                mode: "lines" as const,
                x: bins,
                y: focus.fitted,
                name: "spline fit",
                line: { color: "#2563eb", width: 2.5 },
                hovertemplate: "pseudotime %{x:.3f}<br>fit %{y:.3f}<extra></extra>",
              },
            ]}
            layout={{
              height: 200,
              margin: { t: 8, r: 10, b: 36, l: 44 },
              xaxis: { ...AXIS, title: { text: "pseudotime", font: { size: 11, color: "#64748b" } } },
              yaxis: { ...AXIS, title: { text: "expression", font: { size: 11, color: "#64748b" } }, zeroline: false },
              showlegend: false,
              paper_bgcolor: "white",
              plot_bgcolor: "white",
              font: { family: "Inter, system-ui, sans-serif" },
            }}
            config={plotConfig(`pseudotime_trend_${focus.gene}`)}
            useResizeHandler
            style={{ width: "100%" }}
          />
        </div>
      )}

      {heatmap && (
        <Plot
          data={[heatmap]}
          layout={{
            height: 70 + 13 * ordered.length,
            margin: { t: 8, r: 10, b: 50, l: 90 },
            xaxis: {
              ...AXIS,
              tickvals: ticks,
              ticktext: ticks.map((b) => (bins[b] ?? 0).toFixed(2)),
              title: { text: "pseudotime (bins of equal cell count)", font: { size: 11, color: "#64748b" } },
            },
            yaxis: { ...AXIS, autorange: "reversed" as const, tickfont: { size: 10, color: "#334155" } },
            paper_bgcolor: "white",
            plot_bgcolor: "white",
            font: { family: "Inter, system-ui, sans-serif" },
          }}
          config={plotConfig("pseudotime_genes_heatmap")}
          onClick={(ev) => {
            const gene = ev.points?.[0]?.y;
            if (gene != null) onFocusGene(String(gene));
          }}
          useResizeHandler
          style={{ width: "100%" }}
        />
      )}
    </div>
  );
}
