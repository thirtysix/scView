import { useEffect, useRef, useState } from "react";
import { attachAtacMetrics, getAtacQc, type AtacQc } from "@/api/atac";
import { Caption } from "@/components/common/Caption";
import { RequestError } from "@/components/common/RequestError";
import Plot from "@/components/plots/Plot";
import { QC_MAX_CELLS, sampled, splitByGroup } from "@/lib/atac";
import { histogramFigure } from "@/lib/atacPlots";
import { errorMessage } from "@/lib/errorMessage";
import { prettyObsLabel } from "@/lib/formatting";
import { plotConfig } from "@/lib/plotConfig";

const AXIS = { tickfont: { size: 10, color: "#64748b" }, gridcolor: "#f1f5f9", zeroline: false };

/** The ATAC QC columns per cell: a histogram each, and a violin per group when one is chosen. */
export function AtacQcCard({
  datasetId,
  groupby,
  qcColumns,
  metricsFiles,
  onAttached,
}: {
  datasetId: string;
  groupby: string;
  qcColumns: string[];
  metricsFiles: string[];
  onAttached: () => void;
}) {
  const [qc, setQc] = useState<AtacQc | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attached, setAttached] = useState<string | null>(null);
  const file = useRef<HTMLInputElement | null>(null);
  const columnsKey = qcColumns.join(",");

  useEffect(() => {
    let cancelled = false;
    setError(null);
    getAtacQc(datasetId, groupby || undefined)
      .then((q) => !cancelled && setQc(q))
      .catch((e) => !cancelled && setError(errorMessage(e)));
    return () => {
      cancelled = true;
    };
  }, [datasetId, groupby, columnsKey]);

  const attach = async (f: File) => {
    setError(null);
    try {
      const r = await attachAtacMetrics(datasetId, f);
      setAttached(`${r.file}: ${r.n_matched.toLocaleString()} of ${r.n_cells.toLocaleString()} cells matched.`);
      onAttached();
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  const cells = Math.min(qc?.n_cells ?? 0, QC_MAX_CELLS);
  return (
    <section className="space-y-2 rounded-lg border border-slate-200 p-2" aria-label="ATAC quality">
      <h4 className="text-xs font-semibold text-slate-600">ATAC quality</h4>
      <RequestError message={error} />
      {qc?.columns.map((col) => (
        <div key={col.key} data-testid={`atac-qc-${col.key}`}>
          <Plot
            {...figureOf(col.label, col.values)}
            config={plotConfig(`atac_${col.key}`)}
            useResizeHandler
            style={{ width: "100%" }}
          />
          {qc.groups && qc.group_order && (
            <GroupViolin label={col.label} name={col.key} column={groupby}
                         byGroup={splitByGroup(col.values, qc.groups, qc.group_order, QC_MAX_CELLS)} />
          )}
        </div>
      ))}
      {qc && (
        <Caption n={{ value: cells, unit: "cells", of: qc.n_cells }}>
          Per-cell ATAC quality{groupby ? `, by ${prettyObsLabel(groupby)}` : ""}. Axes end at the 99.5th percentile.
        </Caption>
      )}
      <div className="flex flex-wrap items-center gap-1.5">
        <input ref={file} type="file" accept=".csv,.gz" className="hidden" aria-label="Per-barcode metrics file"
               onChange={(e) => {
                 const f = e.target.files?.[0];
                 if (f) void attach(f);
                 e.target.value = "";
               }} />
        <button type="button" onClick={() => file.current?.click()}
                className="rounded border border-slate-300 px-2 py-0.5 text-[11px] text-slate-700 hover:border-primary hover:text-primary">
          {metricsFiles.length ? "Replace per-barcode metrics" : "Attach per-barcode metrics"}
        </button>
        <span className="text-[11px] text-slate-500">
          {attached ?? (metricsFiles.length
            ? metricsFiles.join(", ")
            : "per_barcode_metrics.csv adds the fractions in peaks (FRiP), at TSS and mitochondrial.")}
        </span>
      </div>
    </section>
  );
}

function figureOf(label: string, values: (number | null)[]) {
  const { data, layout } = histogramFigure(label, sampled(values.filter((v): v is number => v !== null), QC_MAX_CELLS));
  return { data, layout };
}

function GroupViolin({ label, name, column, byGroup }: { label: string; name: string; column: string; byGroup: Record<string, number[]> }) {
  const groups = Object.keys(byGroup);
  // One odd cell (668k peak counts on 10x PBMC, median near 20k) would squash every violin.
  const values = Object.values(byGroup).flat().sort((x, y) => x - y);
  const top = values[Math.min(values.length - 1, Math.floor(values.length * 0.995))] ?? 1;
  return (
    <Plot
      data={groups.map((g) => ({
        type: "violin" as const, name: g, y: byGroup[g], x0: g, points: false as const, box: { visible: true },
        line: { width: 1, color: "#2563eb" }, hovertemplate: `${g}<br>%{y}<extra></extra>`,
      }))}
      layout={{
        height: 170, margin: { t: 6, r: 8, b: 44, l: 40 },
        // Category: cluster numbers would make it a numeric axis labelling every other group.
        xaxis: { type: "category" as const, tickmode: "array" as const, tickvals: groups, ticktext: groups,
                 tickfont: { size: 9, color: "#64748b" } },
        yaxis: { ...AXIS, range: [0, top * 1.05], title: { text: label, font: { size: 9, color: "#94a3b8" } } },
        showlegend: false, paper_bgcolor: "white", plot_bgcolor: "white", font: { family: "Inter, system-ui, sans-serif" },
      }}
      config={plotConfig(`atac_${name}_by_${column}`)}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}
