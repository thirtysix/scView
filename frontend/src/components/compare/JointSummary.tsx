import { useEffect, useMemo, useState } from "react";
import { apiFetch } from "@/api/client";
import { getJointSummary, type BatchMixing, type JointSummary as Summary } from "@/api/integration";
import { getPalette } from "@/lib/colors";
import { prettyObsLabel } from "@/lib/formatting";
import { useSettingsStore } from "@/stores/settingsStore";
import { CompositionTest } from "./CompositionTest";

/** A score as a share of the ideal, for bar widths: 0 = datasets apart, 1 = ideal. */
export function shareOfIdeal(value: number | null | undefined, ideal: number): number {
  if (value == null || !(ideal > 0)) return 0;
  return Math.max(0, Math.min(1, value / ideal));
}

/** `/metadata/crosstab` response: counts[row][col]. */
export interface Crosstab {
  row_values: string[];
  col_values: string[];
  counts: number[][];
}

export interface CompositionRow {
  group: string;
  n: number;
  counts: number[];
  shares: number[];
}

/** Per row (cluster) of a crosstab, each column's (dataset's) share of its cells,
 *  plus the same over all cells, which is what an evenly mixed cluster looks like. */
export function compositionRows(ct: Crosstab): { all: CompositionRow; rows: CompositionRow[] } {
  const row = (group: string, counts: number[]): CompositionRow => {
    const n = counts.reduce((a, b) => a + b, 0);
    return { group, n, counts, shares: counts.map((c) => (n > 0 ? c / n : 0)) };
  };
  const rows = ct.row_values
    .map((g, ri) => row(g, ct.col_values.map((_, ci) => ct.counts[ri]?.[ci] ?? 0)))
    .filter((r) => r.n > 0);
  const totals = ct.col_values.map((_, ci) => rows.reduce((a, r) => a + (r.counts[ci] ?? 0), 0));
  return { all: row("All cells", totals), rows };
}

/**
 * Quality checks for a joint dataset: how well its source datasets mix before and
 * after batch correction, and what each cluster is made of. Renders nothing for a
 * dataset that isn't joint.
 */
export function JointSummary({ datasetId, title }: { datasetId: string; title: string }) {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [crosstab, setCrosstab] = useState<Crosstab | null>(null);

  useEffect(() => {
    let cancelled = false;
    setSummary(null);
    setCrosstab(null);
    getJointSummary(datasetId)
      .then((s) => {
        if (cancelled) return;
        setSummary(s);
        if (!s.joint || !s.cluster_column || !s.label_column) return;
        return apiFetch<Crosstab>(
          `/datasets/${datasetId}/metadata/crosstab?row=${encodeURIComponent(s.cluster_column)}` +
            `&col=${encodeURIComponent(s.label_column)}`,
        ).then((ct) => !cancelled && setCrosstab(ct));
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  if (!summary?.joint) return null;
  return (
    <div className="space-y-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <h3 className="text-sm font-semibold text-slate-700">{title}</h3>
      {summary.mixing ? (
        <MixingScore mixing={summary.mixing} />
      ) : (
        <p className="text-sm text-slate-400">No batch-mixing score for this dataset.</p>
      )}
      {crosstab && summary.cluster_column && (
        <ClusterComposition crosstab={crosstab} clusterColumn={summary.cluster_column} />
      )}
      {crosstab && summary.cluster_column && (
        <ReplicateTest
          datasetId={datasetId}
          labelColumn={summary.label_column}
          clusterColumn={summary.cluster_column}
        />
      )}
    </div>
  );
}

/** The bars count cells; testing a cluster's share needs samples as replicates. */
function ReplicateTest({
  datasetId,
  labelColumn,
  clusterColumn,
}: {
  datasetId: string;
  labelColumn: string | null | undefined;
  clusterColumn: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="space-y-3">
      <p className="text-xs text-slate-500">
        These bars count cells. Whether a cluster&apos;s share really differs between datasets or
        conditions needs the donors or samples as replicates: a chi-square or Fisher test on the cell
        counts treats every cell as independent and overstates significance.{" "}
        <button onClick={() => setOpen((o) => !o)} className="font-medium text-primary hover:underline">
          {open ? "Hide the test" : "Test cluster shares across samples"}
        </button>
      </p>
      {open && (
        <CompositionTest datasetId={datasetId} defaultCondition={labelColumn} defaultGroup={clusterColumn} />
      )}
    </div>
  );
}

function ClusterComposition({ crosstab, clusterColumn }: { crosstab: Crosstab; clusterColumn: string }) {
  const palette = getPalette(useSettingsStore((s) => s.categoricalPalette));
  const { all, rows } = useMemo(() => compositionRows(crosstab), [crosstab]);
  const color = (i: number) => {
    const c = palette[i % palette.length]!;
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  };
  if (rows.length === 0) return null;

  const bar = (r: CompositionRow, strong: boolean) => (
    <div key={r.group} className="flex items-center gap-3 text-xs">
      <span
        className={`w-28 flex-shrink-0 truncate ${strong ? "font-medium text-slate-700" : "text-slate-600"}`}
        title={r.group}
      >
        {r.group}
      </span>
      <div className="flex h-3 flex-1 overflow-hidden rounded bg-slate-100">
        {r.shares.map((s, i) =>
          s > 0 ? (
            <div
              key={crosstab.col_values[i]}
              style={{ width: `${s * 100}%`, backgroundColor: color(i) }}
              title={`${crosstab.col_values[i]}: ${(s * 100).toFixed(1)}% (${(r.counts[i] ?? 0).toLocaleString()} cells)`}
            />
          ) : null,
        )}
      </div>
      <span className="w-14 text-right font-mono text-slate-500">{r.n.toLocaleString()}</span>
    </div>
  );

  return (
    <div>
      <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">
        Datasets per cluster ({prettyObsLabel(clusterColumn)})
      </h4>
      <div className="mb-2 flex flex-wrap gap-3 text-xs text-slate-600">
        {crosstab.col_values.map((v, i) => (
          <span key={v} className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: color(i) }} />
            {v}
          </span>
        ))}
      </div>
      <div className="max-h-80 space-y-1 overflow-y-auto pr-1">
        {bar(all, true)}
        {rows.map((r) => bar(r, false))}
      </div>
      <p className="mt-2 text-xs text-slate-500">
        A cluster close to the &quot;All cells&quot; split is shared by the datasets; one made of a
        single dataset is either specific to it or a batch effect left uncorrected.
      </p>
    </div>
  );
}

function MixingScore({ mixing }: { mixing: BatchMixing }) {
  const rows: { label: string; value: number | null }[] = [
    { label: "Before correction", value: mixing.uncorrected },
  ];
  if (mixing.corrected != null) rows.push({ label: "After correction", value: mixing.corrected });
  const fmt = (v: number | null) => (v == null ? "–" : v.toFixed(3));

  return (
    <div>
      <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">
        Batch mixing
      </h4>
      <div className="space-y-1.5">
        {rows.map((r) => (
          <div key={r.label} className="flex items-center gap-3 text-sm">
            <span className="w-36 flex-shrink-0 text-slate-600">{r.label}</span>
            <div className="relative h-2.5 flex-1 overflow-hidden rounded-full bg-slate-100">
              <div
                className="h-full rounded-full bg-primary"
                style={{ width: `${shareOfIdeal(r.value, mixing.ideal) * 100}%` }}
              />
            </div>
            <span className="w-14 text-right font-mono text-slate-800">{fmt(r.value)}</span>
          </div>
        ))}
        <div className="flex items-center gap-3 text-sm">
          <span className="w-36 flex-shrink-0 text-slate-400">Ideal</span>
          <div className="flex-1" />
          <span className="w-14 text-right font-mono text-slate-500">{fmt(mixing.ideal)}</span>
        </div>
      </div>
      <p className="mt-2 text-xs text-slate-500">
        Share of each cell&apos;s {mixing.k} nearest neighbours that come from another dataset, averaged
        over {mixing.n_scored.toLocaleString()}
        {mixing.n_scored < mixing.n_cells ? ` of ${mixing.n_cells.toLocaleString()}` : ""} cells. Near 0,
        the datasets sit apart; the ideal is what perfect mixing gives for these dataset sizes. High
        mixing is only expected where the datasets share cell types.
      </p>
      {mixing.ilisi && (
        <p className="mt-2 text-xs text-slate-600">
          Datasets per neighbourhood (iLISI): <span className="font-mono">{fmt(mixing.ilisi.uncorrected)}</span>{" "}
          before, <span className="font-mono">{fmt(mixing.ilisi.corrected)}</span> after; ideal{" "}
          <span className="font-mono">{fmt(mixing.ilisi.ideal)}</span> (1 = each cell sees one dataset).
        </p>
      )}
      {mixing.conservation && <Conservation cons={mixing.conservation} warning={mixing.overcorrection} />}
      {mixing.datasets.length > 0 && (
        <table className="mt-2 w-full text-xs">
          <thead>
            <tr className="text-left text-slate-400">
              <th className="py-1 font-medium">Dataset</th>
              <th className="py-1 text-right font-medium" title="Cells from this dataset in the joint embedding">Cells</th>
              <th className="py-1 text-right font-medium" title={`Share of this dataset's cells' ${mixing.k} nearest neighbours from another dataset, on the uncorrected PCA`}>
                Before
              </th>
              {mixing.corrected != null && (
                <th className="py-1 text-right font-medium" title="The same share after batch correction">After</th>
              )}
              <th className="py-1 text-right font-medium" title="The share perfect mixing would give this dataset: the other datasets' fraction of all cells">
                Ideal
              </th>
            </tr>
          </thead>
          <tbody className="font-mono text-slate-700">
            {mixing.datasets.map((d) => (
              <tr key={d.label} className="border-t border-slate-100">
                <td className="py-1 font-sans">{d.label}</td>
                <td className="py-1 text-right">{d.n_cells.toLocaleString()}</td>
                <td className="py-1 text-right">{fmt(d.uncorrected)}</td>
                {mixing.corrected != null && <td className="py-1 text-right">{fmt(d.corrected)}</td>}
                <td className="py-1 text-right text-slate-500">{fmt(d.ideal)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

/** Bio-conservation: does correction keep cell types apart? */
export function Conservation({
  cons,
  warning,
}: {
  cons: NonNullable<BatchMixing["conservation"]>;
  warning?: string | null;
}) {
  const fmt = (v: number | null) => (v == null ? "–" : v.toFixed(2));
  const what = cons.kind === "shared"
    ? `its '${cons.label_column}' label`
    : "its own dataset's cluster (neighbours from the same dataset)";
  return (
    <div className="mt-3">
      <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
        Cell types kept apart
      </h4>
      <p className="text-xs text-slate-600">
        Share of a cell&apos;s neighbours with {what}: <span className="font-mono">{fmt(cons.uncorrected)}</span>{" "}
        before, <span className="font-mono">{fmt(cons.corrected)}</span> after correction. A big drop means the
        correction is merging distinct cell types.
      </p>
      {warning && (
        <p role="alert" className="mt-1.5 rounded border border-red-200 bg-red-50 px-2 py-1 text-xs text-red-800">
          {warning}
        </p>
      )}
    </div>
  );
}
