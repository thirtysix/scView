import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { getDiversity, getOverlap, getUsage, type DiversityMetric, type Overlap, type Usage } from "@/api/vdj";
import { Caption, capText } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { plotlyColorscale } from "@/lib/colors";
import { prettyObsLabel } from "@/lib/formatting";
import { plotConfig } from "@/lib/plotConfig";
import { errorText } from "@/lib/programs";
import { diversityBars, METRICS, overlapHover, usageHover } from "@/lib/vdj";
import { useSettingsStore } from "@/stores/settingsStore";

const LABELLED = 12;

function heatHeight(rows: number): number {
  return Math.min(460, Math.max(200, rows * 16 + 110));
}

/**
 * Repertoire views per group of one column (default: the samples): diversity,
 * clonotype overlap between groups, and V/J gene usage.
 */
export function ClonotypeRepertoire({
  datasetId,
  groupColumns,
  groupByColumn,
}: {
  datasetId: string;
  groupColumns: string[];
  /** The Unified View's group-by: offered first when it is a grouping */
  groupByColumn: string;
}) {
  const [groupBy, setGroupBy] = useState("");
  const options = useMemo(
    () => (groupByColumn && groupColumns.includes(groupByColumn) ? [groupByColumn, ...groupColumns.filter((c) => c !== groupByColumn)] : groupColumns),
    [groupColumns, groupByColumn],
  );
  const diversity = useQuery({
    queryKey: ["vdj-view", datasetId, "diversity-default", groupBy],
    queryFn: () => getDiversity(datasetId, groupBy, false),
    retry: false,
  });
  const column = groupBy || diversity.data?.group_by || "";

  return (
    <section aria-label="Repertoire per group" className="space-y-2 border-t border-slate-200 pt-2">
      <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-slate-600">
        <span className="font-medium">Repertoire per</span>
        <select
          aria-label="Repertoire grouping"
          value={column}
          onChange={(e) => setGroupBy(e.target.value)}
          className="max-w-[10rem] rounded border border-slate-300 px-1 py-0.5 text-xs"
        >
          {column && !options.includes(column) && <option value={column}>{prettyObsLabel(column)}</option>}
          {!column && <option value="">pick a column…</option>}
          {options.map((c) => (
            <option key={c} value={c}>
              {prettyObsLabel(c)}
            </option>
          ))}
        </select>
      </div>
      {diversity.error && !column && (
        <p role="alert" className="rounded-md bg-amber-50 px-2 py-1.5 text-amber-800">
          {errorText(diversity.error)}
        </p>
      )}
      {column && (
        <>
          <DiversityChart datasetId={datasetId} column={column} />
          <OverlapHeatmap datasetId={datasetId} column={column} />
          <UsageHeatmap datasetId={datasetId} column={column} />
        </>
      )}
    </section>
  );
}

function DiversityChart({ datasetId, column }: { datasetId: string; column: string }) {
  const [metric, setMetric] = useState<DiversityMetric>("shannon");
  const [rarefy, setRarefy] = useState(false);
  const { data, error, isFetching } = useQuery({
    queryKey: ["vdj-view", datasetId, "diversity", column, rarefy],
    queryFn: () => getDiversity(datasetId, column, rarefy),
    retry: false,
  });
  const bars = useMemo(() => (data ? diversityBars(data, metric) : null), [data, metric]);
  const info = METRICS.find((m) => m.id === metric);
  const withCells = data?.groups.filter((g) => g.n_with_clonotype > 0) ?? [];

  return (
    <div className="space-y-1" data-testid="vdj-diversity">
      <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-slate-600">
        <span className="font-medium">Diversity</span>
        <select
          aria-label="Diversity measure"
          value={metric}
          onChange={(e) => setMetric(e.target.value as DiversityMetric)}
          className="rounded border border-slate-300 px-1 py-0.5 text-xs"
        >
          {METRICS.map((m) => (
            <option key={m.id} value={m.id} title={m.info}>
              {m.label}
            </option>
          ))}
        </select>
        {info && <InfoTip label={`About ${info.label}`}>{info.info}</InfoTip>}
        <label
          className="flex items-center gap-1"
          title={`Subsample every group to the smallest group's cells (groups with ${data?.min_cells ?? 10}+ cells), 20 seeded draws, and show the mean and SD`}
        >
          <input type="checkbox" checked={rarefy} onChange={(e) => setRarefy(e.target.checked)} />
          rarefy
        </label>
        {isFetching && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
      </div>
      {error && (
        <p role="alert" className="rounded-md bg-amber-50 px-2 py-1.5 text-amber-800">
          {errorText(error)}
        </p>
      )}
      {data && bars && bars.x.length > 0 && (
        <Plot
          data={[
            {
              type: "bar",
              x: bars.x,
              y: bars.y,
              text: bars.hover,
              hoverinfo: "text",
              textposition: "none",
              marker: { color: "rgb(71,85,105)" },
              ...(data.rarefy ? { error_y: { type: "data", array: bars.sd, visible: true, thickness: 1, width: 3 } } : {}),
            },
          ]}
          layout={{
            height: 180,
            margin: { l: 40, r: 8, t: 6, b: 10 },
            xaxis: { type: "category", automargin: true, tickangle: -45, tickfont: { size: 9 } },
            yaxis: { title: { text: info?.label ?? metric, font: { size: 10 } }, tickfont: { size: 9 }, rangemode: "tozero" },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
          }}
          config={plotConfig(`clonotype_diversity_${column}`)}
          useResizeHandler
          style={{ width: "100%", height: 180 }}
        />
      )}
      {data && (
        <Caption
          n={withCells.map((g) => ({ value: g.n_with_clonotype, unit: `cells in ${g.name}` })).slice(0, 6)}
          testId="vdj-diversity-caption"
        >
          {data.caption}
          {bars && bars.left_out.length > 0 && ` Not shown (under ${data.min_cells} cells with a clonotype): ${bars.left_out.join(", ")}.`}
          {withCells.length > 6 && ` ${withCells.length} groups; the first six are counted here.`}
        </Caption>
      )}
    </div>
  );
}

function OverlapHeatmap({ datasetId, column }: { datasetId: string; column: string }) {
  const [metric, setMetric] = useState<Overlap["metric"]>("morisita_horn");
  const colormap = useSettingsStore((s) => s.continuousColormap);
  const { data, error, isFetching } = useQuery({
    queryKey: ["vdj-view", datasetId, "overlap", column, metric],
    queryFn: () => getOverlap(datasetId, column, metric),
    retry: false,
  });
  const hover = useMemo(() => (data ? overlapHover(data) : []), [data]);
  const height = heatHeight(data?.groups.length ?? 0);

  return (
    <div className="space-y-1" data-testid="vdj-overlap">
      <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-slate-600">
        <span className="font-medium">Clonotype overlap</span>
        <span role="group" aria-label="Overlap measure" className="inline-flex overflow-hidden rounded border border-slate-200">
          {(["morisita_horn", "jaccard"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMetric(m)}
              aria-pressed={metric === m}
              className={`px-1.5 py-0.5 text-[11px] ${metric === m ? "bg-slate-600 text-white" : "bg-white text-slate-600 hover:bg-slate-50"}`}
            >
              {m === "morisita_horn" ? "Morisita-Horn" : "Jaccard"}
            </button>
          ))}
        </span>
        <InfoTip label="About the overlap measures" width={300}>
          Morisita-Horn (Horn 1966) compares clone sizes: 1 when two groups hold the same clonotypes in the same
          proportions, 0 when they share none. It is weighted towards the expanded clonotypes and changes little
          with how many cells each group has, so it is the default. Jaccard counts shared clonotypes over all
          clonotypes of the two groups, every singleton equally, so it falls as groups get larger.
        </InfoTip>
        {isFetching && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
      </div>
      {error && (
        <p role="alert" className="rounded-md bg-amber-50 px-2 py-1.5 text-amber-800">
          {errorText(error)}
        </p>
      )}
      {data && data.groups.length > 1 && (
        <Plot
          data={[
            {
              type: "heatmap",
              x: data.groups,
              y: data.groups,
              z: data.matrix,
              zmin: 0,
              zmax: 1,
              text: hover as unknown as string[],
              hoverinfo: "text",
              colorscale: plotlyColorscale(colormap),
              colorbar: { thickness: 10, title: { text: metric === "morisita_horn" ? "MH" : "Jaccard", font: { size: 10 } } },
              xgap: 1,
              ygap: 1,
              ...(data.groups.length <= LABELLED ? { texttemplate: "%{z:.2f}", textfont: { size: 9 } } : {}),
            },
          ]}
          layout={{
            height,
            margin: { l: 10, r: 10, t: 6, b: 10 },
            xaxis: { type: "category", tickangle: -45, automargin: true, tickfont: { size: 9 } },
            yaxis: { type: "category", automargin: true, autorange: "reversed", tickfont: { size: 9 } },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
          }}
          config={plotConfig(`clonotype_overlap_${column}`)}
          useResizeHandler
          style={{ width: "100%", height }}
        />
      )}
      {data && (
        <Caption n={{ value: data.n_groups_shown, unit: "groups", of: data.n_groups_total }}>
          {metric === "morisita_horn" ? "Morisita-Horn" : "Jaccard"} overlap of the clonotypes between groups of{" "}
          {prettyObsLabel(column)}; hover for shared clonotypes.
          {data.n_groups_shown < data.n_groups_total && " The largest groups by cells are shown."}
          {data.groups.length < 2 && " Overlap needs two groups with clonotypes."}
        </Caption>
      )}
    </div>
  );
}

function UsageHeatmap({ datasetId, column }: { datasetId: string; column: string }) {
  const [chain, setChain] = useState("");
  const [gene, setGene] = useState<Usage["gene"]>("v");
  const [count, setCount] = useState<Usage["count"]>("clonotypes");
  const colormap = useSettingsStore((s) => s.continuousColormap);
  const { data, error, isFetching } = useQuery({
    queryKey: ["vdj-view", datasetId, "usage", column, chain, gene, count],
    queryFn: () => getUsage(datasetId, column, { chain: chain || undefined, gene, count }),
    retry: false,
  });
  const hover = useMemo(() => (data ? usageHover(data) : []), [data]);
  const height = heatHeight(data?.groups.length ?? 0);
  const cap = data ? capText({ shown: data.genes.length, total: data.n_genes_total, by: `${count} using them` }) : null;

  return (
    <div className="space-y-1" data-testid="vdj-usage">
      <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-slate-600">
        <span className="font-medium">Gene usage</span>
        <select
          aria-label="Chain"
          value={data?.chain ?? chain}
          onChange={(e) => setChain(e.target.value)}
          className="rounded border border-slate-300 px-1 py-0.5 text-xs"
        >
          {Object.keys(data?.chains_present ?? {}).map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
          {data && !(data.chain in data.chains_present) && <option value={data.chain}>{data.chain}</option>}
        </select>
        <select
          aria-label="Gene segment"
          value={gene}
          onChange={(e) => setGene(e.target.value as Usage["gene"])}
          className="rounded border border-slate-300 px-1 py-0.5 text-xs"
        >
          <option value="v">V genes</option>
          <option value="j">J genes</option>
        </select>
        <select
          aria-label="Count"
          value={count}
          onChange={(e) => setCount(e.target.value as Usage["count"])}
          title="Clonotypes: each clonotype once per group (an expanded clone does not dominate). Cells: every cell."
          className="rounded border border-slate-300 px-1 py-0.5 text-xs"
        >
          <option value="clonotypes">per clonotype</option>
          <option value="cells">per cell</option>
        </select>
        {isFetching && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
      </div>
      {error && (
        <p role="alert" className="rounded-md bg-amber-50 px-2 py-1.5 text-amber-800">
          {errorText(error)}
        </p>
      )}
      {data && data.genes.length > 0 && data.groups.length > 0 && (
        <Plot
          data={[
            {
              type: "heatmap",
              x: data.genes,
              y: data.groups,
              z: data.share,
              zmin: 0,
              text: hover as unknown as string[],
              hoverinfo: "text",
              colorscale: plotlyColorscale(colormap),
              colorbar: { thickness: 10, title: { text: "share", font: { size: 10 } } },
              xgap: 1,
              ygap: 1,
            },
          ]}
          layout={{
            height,
            margin: { l: 10, r: 10, t: 6, b: 10 },
            xaxis: { type: "category", tickangle: -60, automargin: true, tickfont: { size: 8 } },
            yaxis: { type: "category", automargin: true, autorange: "reversed", tickfont: { size: 9 } },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
          }}
          config={plotConfig(`clonotype_${data.chain}_${gene}_usage_${column}`)}
          useResizeHandler
          style={{ width: "100%", height }}
        />
      )}
      {data && (
        <Caption n={data.groups.map((g, i) => ({ value: data.n[i]!, unit: `${count} in ${g}` })).slice(0, 4)}>
          Share of each group&apos;s {count} with a primary {data.chain} chain that use each {gene.toUpperCase()} gene.
          {cap && ` ${cap}`}
          {data.groups.length > 4 && ` ${data.groups.length} groups; the first four are counted here.`}
        </Caption>
      )}
    </div>
  );
}
