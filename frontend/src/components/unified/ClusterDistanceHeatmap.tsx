import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { getGroupDistances } from "@/api/clustering";
import { nearestGroups, orderedHeatmap, spaceLabel } from "@/lib/clusterDistances";
import { dendrogramSegments, rowDendrogram } from "@/lib/dendrogram";
import { plotlyColorscale } from "@/lib/colors";
import { plotConfig } from "@/lib/plotConfig";
import { prettyObsLabel } from "@/lib/formatting";
import { useObsRevision } from "@/stores/obsRevisionStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";

/** Cell values are written in the heatmap up to this many groups. */
const MAX_LABELLED = 15;
const MIN_HEIGHT = 240;
const MAX_HEIGHT = 420;
/** The heatmap's share of the width; the dendrogram takes the strip right of it. */
const HEAT_DOMAIN: [number, number] = [0, 0.84];
const TREE_DOMAIN: [number, number] = [0.86, 1];

/**
 * Distances between the groups of a column in the PCs the neighbour graph was
 * built on, as a heatmap ordered so similar groups sit together: the view a
 * UMAP or t-SNE plot can't give (its caveat links here). The complete-linkage
 * dendrogram that orders it is drawn on the right, and the same view can show
 * the Pearson correlation of the centroids instead (same order). Clicking a
 * cell highlights its row's group, as the legend and cluster map do.
 */
export function ClusterDistanceHeatmap({
  datasetId,
  defaultColumn,
  columns,
}: {
  datasetId: string;
  /** The column shown until the user picks another (the plot's grouping) */
  defaultColumn: string;
  /** Categorical columns to choose from */
  columns: readonly { name: string; n_unique: number }[];
}) {
  const [picked, setPicked] = useState<{ datasetId: string; column: string } | null>(null);
  const column =
    picked && picked.datasetId === datasetId && columns.some((c) => c.name === picked.column)
      ? picked.column
      : defaultColumn;
  const revision = useObsRevision(datasetId, column); // hand labels move cells between groups
  const { data, error, isLoading } = useQuery({
    queryKey: ["group-distances", datasetId, column, revision],
    queryFn: () => getGroupDistances(datasetId, column),
    enabled: !!column,
    staleTime: 5 * 60 * 1000,
    retry: false,
  });
  const colormap = useSettingsStore((s) => s.continuousColormap);
  const [measure, setMeasure] = useState<"distance" | "correlation">("distance");
  const highlighted = useSelectionStore((s) => s.highlightedGroup);
  const setHighlight = useSelectionStore((s) => s.setHighlight);

  const correlation = measure === "correlation" && data?.correlation ? data.correlation : null;
  const heat = useMemo(
    () => (data ? orderedHeatmap(data, correlation ?? data.distances) : null),
    [data, correlation],
  );
  const tree = useMemo(
    () => (data?.tree ? dendrogramSegments(data.tree, data.order) : null),
    [data],
  );
  const hover = useMemo(
    () =>
      heat?.z.map((row, r) =>
        row.map((v, c) =>
          v === null
            ? ""
            : `<b>${heat.labels[r]}</b> to <b>${heat.labels[c]}</b><br>` +
              (correlation ? `correlation ${v.toFixed(2)}` : `distance ${v.toFixed(2)}`) +
              `<br>${heat.sizes[r]!.toLocaleString()} and ${heat.sizes[c]!.toLocaleString()} cells`,
        ),
      ) ?? [],
    [heat, correlation],
  );
  const dendro = tree ? rowDendrogram(tree, TREE_DOMAIN) : null;
  const focus = highlighted && data && highlighted.column === column ? highlighted.value : null;
  const nearest = focus && data ? nearestGroups(data, focus) : [];
  // Tall enough for a row label per group, but it shares the column with the plot.
  const height = Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, (heat?.labels.length ?? 0) * 16 + 110));

  return (
    <div className="space-y-1 text-[11px] text-slate-500">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <label htmlFor="distance-column" className="font-medium text-slate-600">
          Distances between
        </label>
        <select
          id="distance-column"
          value={column}
          onChange={(e) => setPicked({ datasetId, column: e.target.value })}
          className="min-w-0 max-w-[50%] rounded border border-slate-200 bg-white px-1.5 py-0.5 text-xs text-slate-600"
        >
          {!columns.some((c) => c.name === column) && <option value={column}>{prettyObsLabel(column)}</option>}
          {columns.map((c) => (
            <option key={c.name} value={c.name}>
              {prettyObsLabel(c.name)} ({c.n_unique})
            </option>
          ))}
        </select>
        <span>groups, in PCA space</span>
        {data?.correlation && (
          <span role="group" aria-label="Measure" className="ml-auto inline-flex overflow-hidden rounded border border-slate-200">
            {(["distance", "correlation"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMeasure(m)}
                aria-pressed={measure === m}
                className={`px-1.5 py-0.5 text-[11px] ${measure === m ? "bg-slate-600 text-white" : "bg-white text-slate-600 hover:bg-slate-50"}`}
              >
                {m === "distance" ? "Distance" : "Correlation"}
              </button>
            ))}
          </span>
        )}
        {isLoading && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
      </div>
      {error ? (
        <p role="alert" className="rounded-md bg-amber-50 px-2 py-1.5 text-amber-800">
          {errorText(error)}
        </p>
      ) : data && heat ? (
        <>
          <Plot
            data={[
              {
                type: "heatmap",
                x: heat.labels,
                y: heat.labels,
                z: heat.z,
                text: hover as unknown as string[],
                hoverinfo: "text",
                ...(correlation
                  ? { colorscale: "RdBu", zmin: -1, zmax: 1 } // Plotly's RdBu: blue low, red high
                  : { colorscale: plotlyColorscale(colormap) }),
                colorbar: {
                  title: { text: correlation ? "correlation" : "distance", font: { size: 10 } },
                  thickness: 10,
                },
                xgap: 1,
                ygap: 1,
                ...(heat.labels.length <= MAX_LABELLED
                  ? { texttemplate: correlation ? "%{z:.2f}" : "%{z:.1f}", textfont: { size: 9 } }
                  : {}),
              },
              ...(dendro ? [dendro.trace] : []),
            ]}
            layout={{
              height,
              margin: { l: 10, r: 10, t: 6, b: 10 },
              xaxis: {
                type: "category",
                tickangle: -45,
                automargin: true,
                tickfont: { size: 10 },
                ...(dendro ? { domain: HEAT_DOMAIN } : {}),
              },
              yaxis: {
                type: "category",
                automargin: true,
                tickfont: { size: 10 },
                // An explicit range so the dendrogram's leaves meet the rows.
                ...(dendro ? { range: [heat.labels.length - 0.5, -0.5] } : { autorange: "reversed" as const }),
              },
              ...(dendro ? dendro.layout : {}),
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
            }}
            config={plotConfig(`distances_${column}`)}
            useResizeHandler
            style={{ width: "100%", height }}
            onClick={(ev: { points?: { y?: unknown }[] }) => {
              const group = ev.points?.[0]?.y;
              if (typeof group === "string") setHighlight(focus === group ? null : { column, value: group });
            }}
          />
          {nearest.length > 0 && (
            <p data-testid="nearest-groups">
              Nearest to <span className="font-semibold text-slate-700">{focus}</span>:{" "}
              {nearest.map((n) => `${n.group} (${n.distance.toFixed(1)})`).join(", ")}
            </p>
          )}
          <p>
            {correlation ? (
              <>
                Pearson correlation between group centroids across {spaceLabel(data)}. The PCs are
                centred on the average cell, so groups that differ from it in the same way correlate
                (correlating mean expression instead puts nearly every pair high, as all groups share
                most of their expression).
              </>
            ) : (
              <>
                Euclidean distance between group centroids in {spaceLabel(data)}. Unlike gaps on a
                UMAP or t-SNE plot, these distances follow how different the groups&apos; expression is.
              </>
            )}{" "}
            Groups ordered by the complete-linkage clustering of the distances
            {tree ? ", drawn on the right" : ""}.
            {data.n_left_out > 0 ? ` ${data.n_left_out.toLocaleString()} cells without a group are left out.` : ""}
          </p>
        </>
      ) : null}
    </div>
  );
}

/** The server's reason ("API error 400: {detail}") as a sentence. */
function errorText(err: unknown): string {
  const msg = err instanceof Error ? err.message.replace(/^API error \d+: /, "") : String(err);
  try {
    return String(JSON.parse(msg).detail ?? msg);
  } catch {
    return msg;
  }
}
