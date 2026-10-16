import { useEffect, useMemo, useState } from "react";
import { plotlyColorscale } from "@/lib/colors";
import { useSettingsStore } from "@/stores/settingsStore";
import { AlertCircle, Loader2 } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { plotConfig } from "@/lib/plotConfig";
import { apiFetch } from "@/api/client";
import { useDatasetStore } from "@/stores/datasetStore";
import { missingGenes } from "@/lib/geneList";

interface MarkerRow {
  gene: string;
  logfoldchange: number;
}

interface ByGroup {
  groupby: string;
  layer: string;
  groups: string[];
  genes: string[];
  mean: number[][]; // [group][gene]
  fraction: number[][]; // [group][gene]
}

/** The top `perGroup` up-regulated markers of each group, in group order, each gene once. */
export function pickMarkerGenes(
  groups: string[],
  markers: Record<string, MarkerRow[]>,
  perGroup: number,
): { gene: string; group: string }[] {
  const seen = new Set<string>();
  const out: { gene: string; group: string }[] = [];
  for (const group of groups) {
    let n = 0;
    for (const m of markers[group] ?? []) {
      if (n >= perGroup) break;
      if (!(m.logfoldchange > 0) || seen.has(m.gene)) continue;
      seen.add(m.gene);
      out.push({ gene: m.gene, group });
      n++;
    }
  }
  return out;
}

/** Scale each gene (column) to 0-1 across groups, as scanpy's standard_scale="var". */
export function scalePerGene(mean: number[][]): number[][] {
  const nGenes = mean[0]?.length ?? 0;
  const lo: number[] = [];
  const hi: number[] = [];
  for (let j = 0; j < nGenes; j++) {
    const col = mean.map((row) => row[j] ?? 0);
    lo.push(Math.min(...col));
    hi.push(Math.max(...col));
  }
  return mean.map((row) =>
    row.map((v, j) => {
      const range = (hi[j] ?? 0) - (lo[j] ?? 0);
      return range > 0 ? (v - (lo[j] ?? 0)) / range : 0;
    }),
  );
}

export function MarkerPlot({
  datasetId,
  groupby,
  groups,
  markers,
  kind,
  perGroup,
  genes: chosenGenes,
}: {
  datasetId: string;
  groupby: string;
  groups: string[];
  markers: Record<string, MarkerRow[]>;
  kind: "dotplot" | "heatmap";
  perGroup: number;
  /** Plot these genes instead of the top markers (empty or absent: top markers). */
  genes?: string[];
}) {
  const colormapName = useSettingsStore((s) => s.continuousColormap);
  const custom = (chosenGenes?.length ?? 0) > 0;
  const picked = useMemo(
    () =>
      custom
        ? chosenGenes!.map((gene) => ({ gene, group: "" }))
        : pickMarkerGenes(groups, markers, perGroup),
    [custom, chosenGenes, groups, markers, perGroup],
  );
  const [data, setData] = useState<ByGroup | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const layers = useDatasetStore((s) => s.currentDataset?.expression_layers);

  useEffect(() => {
    if (picked.length === 0) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    const qs = new URLSearchParams({ groupby });
    for (const p of picked) qs.append("genes", p.gene);
    apiFetch<ByGroup>(`/datasets/${datasetId}/expression/by-group?${qs}`)
      .then((d) => { if (!cancelled) setData(d); })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : String(e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [datasetId, groupby, picked]);

  if (picked.length === 0) {
    return <p className="py-8 text-center text-sm text-slate-400">No up-regulated marker genes to plot.</p>;
  }
  if (error && custom && error.includes("404")) {
    return (
      <p className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
        None of these genes are in this dataset (names are case-sensitive):{" "}
        {chosenGenes!.join(", ")}.
      </p>
    );
  }
  if (error) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
        <AlertCircle className="h-4 w-4 flex-shrink-0" />
        {error}
      </div>
    );
  }
  if (loading || !data) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
        <span className="ml-3 text-sm text-slate-500">Computing expression per group...</span>
      </div>
    );
  }

  // Rows in the marker table's group order (chosen genes: the column's own
  // order, as the legend lists it); genes in the order they were picked.
  const rowIdx = custom
    ? data.groups.map((_, i) => i)
    : groups.map((g) => data.groups.indexOf(g)).filter((i) => i >= 0);
  const rows = rowIdx.map((i) => data.groups[i]!);
  if (rows.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-slate-400">
        These markers were computed for a different column than {groupby}. Choose the column
        under "Compute for" to recompute them.
      </p>
    );
  }
  const layerLabel = layers?.find((l) => l.key === data.layer)?.label ?? data.layer;
  const markerOf = new Map(picked.map((p) => [p.gene, p.group]));
  const genes = data.genes;
  const missing = custom ? missingGenes(chosenGenes!, genes) : [];
  const mean = rowIdx.map((i) => data.mean[i]!);
  const fraction = rowIdx.map((i) => data.fraction[i]!);
  const scaled = scalePerGene(mean);
  const hover = rows.map((g, r) =>
    genes.map(
      (gene, c) =>
        `<b>${gene}</b>${custom ? "" : ` (marker of ${markerOf.get(gene) ?? "?"})`}<br>${groupby}: ${g}` +
        `<br>mean ${(mean[r]?.[c] ?? 0).toFixed(2)}` +
        `<br>${Math.round((fraction[r]?.[c] ?? 0) * 100)}% of cells express it`,
    ),
  );

  const trace =
    kind === "heatmap"
      ? {
          type: "heatmap" as const,
          x: genes,
          y: rows,
          z: scaled,
          text: hover as unknown as string[],
          hoverinfo: "text" as const,
          colorscale: plotlyColorscale(colormapName),
          colorbar: { title: { text: "Mean<br>(scaled)" }, thickness: 12 },
        }
      : {
          type: "scatter" as const,
          mode: "markers" as const,
          x: rows.flatMap(() => genes),
          y: rows.flatMap((g) => genes.map(() => g)),
          text: hover.flat(),
          hoverinfo: "text" as const,
          marker: {
            size: fraction.flat().map((f) => Math.max(f * 22, 1)),
            color: scaled.flat(),
            colorscale: "Reds",
            cmin: 0,
            cmax: 1,
            line: { width: 0.5, color: "#94a3b8" },
            colorbar: { title: { text: "Mean<br>(scaled)" }, thickness: 12 },
          },
        };

  return (
    <div>
      <Plot
        data={[trace]}
        layout={{
          height: Math.max(260, rows.length * 28 + 150),
          margin: { l: 120, r: 20, t: 10, b: 110 },
          xaxis: { type: "category", tickangle: -60, automargin: true, categoryarray: genes, categoryorder: "array" },
          yaxis: { type: "category", autorange: "reversed", automargin: true },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
        }}
        config={plotConfig(`markers_${kind}`)}
        useResizeHandler
        style={{ width: "100%" }}
      />
      {missing.length > 0 && (
        <p className="mt-1 text-xs text-amber-700">
          Not in this dataset&apos;s {layerLabel} layer, so not plotted (names are case-sensitive):{" "}
          {missing.join(", ")}.
        </p>
      )}
      <p className="mt-1 text-xs text-slate-500">
        {custom
          ? `Your ${genes.length} gene${genes.length === 1 ? "" : "s"} across the ${groupby} groups.`
          : `Top ${perGroup} up-regulated markers of each ${groupby} group.`}{" "}
        Colour: mean expression, scaled to 0-1 per gene.
        {kind === "dotplot" && " Dot size: share of cells in the group that express the gene."}
        {` Layer: ${layerLabel}.`}
      </p>
    </div>
  );
}
