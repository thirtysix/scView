import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { getGroupDistances } from "@/api/clustering";
import { nearestGroups, orderedHeatmap, spaceLabel } from "@/lib/clusterDistances";
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

/**
 * Distances between the groups of a column in the PCs the neighbour graph was
 * built on, as a heatmap ordered so similar groups sit together: the view a
 * UMAP or t-SNE plot can't give (its caveat links here). Clicking a cell
 * highlights its row's group, as the legend and cluster map do.
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
  const highlighted = useSelectionStore((s) => s.highlightedGroup);
  const setHighlight = useSelectionStore((s) => s.setHighlight);

  const heat = useMemo(() => (data ? orderedHeatmap(data) : null), [data]);
  const hover = useMemo(
    () =>
      heat?.z.map((row, r) =>
        row.map((v, c) =>
          v === null
            ? ""
            : `<b>${heat.labels[r]}</b> to <b>${heat.labels[c]}</b><br>distance ${v.toFixed(2)}` +
              `<br>${heat.sizes[r]!.toLocaleString()} and ${heat.sizes[c]!.toLocaleString()} cells`,
        ),
      ) ?? [],
    [heat],
  );
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
                colorscale: plotlyColorscale(colormap),
                colorbar: { title: { text: "distance", font: { size: 10 } }, thickness: 10 },
                xgap: 1,
                ygap: 1,
                ...(heat.labels.length <= MAX_LABELLED
                  ? { texttemplate: "%{z:.1f}", textfont: { size: 9 } }
                  : {}),
              },
            ]}
            layout={{
              height,
              margin: { l: 10, r: 10, t: 6, b: 10 },
              xaxis: { type: "category", tickangle: -45, automargin: true, tickfont: { size: 10 } },
              yaxis: { type: "category", autorange: "reversed", automargin: true, tickfont: { size: 10 } },
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
            Euclidean distance between group centroids in {spaceLabel(data)}; groups ordered by
            complete-linkage clustering. Unlike gaps on a UMAP or t-SNE plot, these distances follow
            how different the groups&apos; expression is.
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
