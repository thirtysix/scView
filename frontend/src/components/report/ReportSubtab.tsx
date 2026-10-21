import { useEffect, useMemo, useState } from "react";
import { Download, Loader2, RefreshCw } from "lucide-react";
import { InfoTip } from "@/components/common/InfoTip";
import {
  clusterReportHtmlUrl,
  fetchClusterReport,
  fetchGeneReport,
  geneReportHtmlUrl,
  ReportError,
  type Report,
} from "@/api/report";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { openClusterReport, openGeneReport, useReportStore, type ReportMode } from "@/stores/reportStore";
import type { ObsColumnInfo } from "@/api/types";
import { ReportCard } from "./ReportCard";
import { ClusterSectionBody } from "./ClusterSections";
import { GeneSectionBody } from "./GeneSections";

/** Columns a report can be about: labels with 2 to 100 listed values. */
function groupColumns(columns: ObsColumnInfo[] | undefined): ObsColumnInfo[] {
  return (columns ?? []).filter(
    (c) => (c.dtype === "category" || c.dtype === "object" || c.dtype === "bool") &&
      c.n_unique >= 2 && (c.values?.length ?? 0) >= 2,
  );
}

const sel = "w-full rounded border border-slate-300 bg-white px-2 py-1 text-xs text-slate-700";

/**
 * Report subtab: one page per cluster or per gene that collects the evidence the
 * other tabs hold (backend core/report*.py), each section with its n, how it was
 * computed, and a note when its input was never computed, plus a self-contained
 * HTML download. The cluster defaults to the highlighted group, else the first
 * group of the colouring; the gene to the gene on the plot.
 */
export function ReportSubtab({ groupByColumn }: { groupByColumn: string }) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const highlighted = useSelectionStore((s) => s.highlightedGroup);
  const colorBy = useSettingsStore((s) => s.colorBy);
  const overlay = useUnifiedViewStore((s) => s.scatterOverlay);
  const store = useReportStore();
  const own = store.datasetId === datasetId;
  const columns = useMemo(() => groupColumns(dataset?.obs_columns), [dataset]);
  const has = (name: string | null | undefined) => !!name && columns.some((c) => c.name === name);
  const valuesOf = (name: string) => columns.find((c) => c.name === name)?.values ?? [];

  const cluster = useMemo(() => {
    const valid = (c: { column: string; value: string } | null) =>
      !!c && columns.some((col) => col.name === c.column && (col.values ?? []).includes(c.value));
    if (own && valid(store.cluster)) return store.cluster;
    if (valid(highlighted)) return highlighted;
    const col = [colorBy, groupByColumn, dataset?.active_clustering]
      .map((n) => columns.find((c) => c.name === n))
      .find((c) => c) ?? columns[0];
    return col?.values?.length ? { column: col.name, value: col.values[0]! } : null;
  }, [own, store.cluster, highlighted, columns, colorBy, groupByColumn, dataset]);

  const overlayGene = overlay?.type === "expression" && !overlay.layer ? overlay.gene : null;
  const gene = (own && store.gene) || overlayGene || "";
  const groupby = (own && has(store.groupby) && store.groupby) ||
    (has(groupByColumn) ? groupByColumn : has(dataset?.active_clustering) ? dataset!.active_clustering! : columns[0]?.name ?? "");
  const mode = store.mode;

  const [draft, setDraft] = useState(gene);
  useEffect(() => setDraft(gene), [gene]);

  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<ReportError | null>(null);
  const [loading, setLoading] = useState(false);
  const [refresh, setRefresh] = useState(0);

  const clusterColumn = cluster?.column ?? "";
  const clusterValue = cluster?.value ?? "";
  useEffect(() => {
    setReport(null);
    setError(null);
    if (!datasetId) return;
    if (mode === "cluster" ? !clusterColumn : !gene) return;
    const ctrl = new AbortController();
    setLoading(true);
    const run = mode === "cluster"
      ? fetchClusterReport(datasetId, clusterColumn, clusterValue, ctrl.signal)
      : fetchGeneReport(datasetId, gene, groupby || undefined, ctrl.signal);
    run
      .then((r) => {
        if (!ctrl.signal.aborted) setReport(r);
      })
      .catch((e: unknown) => {
        if (ctrl.signal.aborted) return;
        setError(e instanceof ReportError ? e : new ReportError(e instanceof Error ? e.message : String(e), 0));
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false);
      });
    return () => ctrl.abort();
  }, [datasetId, mode, clusterColumn, clusterValue, gene, groupby, refresh]);

  if (!dataset || !datasetId) return <div className="text-xs text-slate-400">No dataset loaded.</div>;

  const setMode = (m: ReportMode) => store.setMode(m);
  const onGene = (g: string) =>
    openGeneReport(datasetId, g, report?.kind === "cluster" ? report.column : groupby || undefined);
  const onGroup = (g: string) => openClusterReport(datasetId, report?.kind === "gene" ? report.groupby : groupby, g);
  const submitGene = () => {
    const g = draft.trim();
    if (g) store.setGene(datasetId, g);
  };
  const downloadUrl = mode === "cluster"
    ? cluster && clusterReportHtmlUrl(datasetId, cluster.column, cluster.value)
    : gene && geneReportHtmlUrl(datasetId, gene, groupby || undefined);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex rounded border border-slate-300 text-xs" role="group" aria-label="Report on">
          {(["cluster", "gene"] as const).map((m) => (
            <button
              key={m}
              type="button"
              aria-pressed={mode === m}
              onClick={() => setMode(m)}
              className={`px-2.5 py-1 font-medium ${mode === m ? "bg-primary text-white" : "text-slate-600 hover:bg-slate-50"}`}
            >
              {m === "cluster" ? "Cluster" : "Gene"}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setRefresh((n) => n + 1)}
            disabled={loading || !downloadUrl}
            title="Build the report again (after computing something it was missing)"
            aria-label="Refresh the report"
            className="text-slate-400 hover:text-slate-600 disabled:opacity-40"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
          {downloadUrl && (
            <a
              href={downloadUrl}
              download
              className="inline-flex items-center gap-1 rounded border border-slate-300 px-2 py-1 text-[11px] font-medium text-slate-600 hover:bg-slate-50"
              title="One self-contained HTML file (tables and plots inline) to email or print"
            >
              <Download className="h-3 w-3" />
              Download HTML
            </a>
          )}
          <InfoTip label="About reports" align="right">
            A report collects what scView already knows about one cluster or one gene: sizes per donor and
            condition, other labels, marker genes and the marker databases, HALLMARK enrichment, QC, composition,
            co-expression and the recorded methods. Each section says how many cells it rests on and how it was
            computed. A section whose input was never computed says where to compute it; nothing slow is run here.
          </InfoTip>
        </div>
      </div>

      {mode === "cluster" ? (
        <div className="grid grid-cols-2 gap-2">
          <label className="text-[11px] font-medium text-slate-500">
            Column
            <select
              aria-label="Report column"
              className={sel}
              value={cluster?.column ?? ""}
              onChange={(e) => {
                const v = valuesOf(e.target.value);
                if (v.length) store.setCluster(datasetId, { column: e.target.value, value: v[0]! });
              }}
            >
              {columns.map((c) => (
                <option key={c.name} value={c.name}>{c.name} ({c.n_unique})</option>
              ))}
            </select>
          </label>
          <label className="text-[11px] font-medium text-slate-500">
            Group
            <select
              aria-label="Report group"
              className={sel}
              value={cluster?.value ?? ""}
              onChange={(e) => cluster && store.setCluster(datasetId, { column: cluster.column, value: e.target.value })}
            >
              {(cluster ? valuesOf(cluster.column) : []).map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </label>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-2">
          <form
            className="text-[11px] font-medium text-slate-500"
            onSubmit={(e) => {
              e.preventDefault();
              submitGene();
            }}
          >
            <label htmlFor="report-gene">Gene</label>
            <div className="flex gap-1">
              <input
                id="report-gene"
                aria-label="Report gene"
                className={sel}
                value={draft}
                placeholder="e.g. a marker gene"
                onChange={(e) => setDraft(e.target.value)}
              />
              <button type="submit" className="rounded bg-primary px-2 text-[11px] font-medium text-white disabled:opacity-50" disabled={!draft.trim()}>
                Show
              </button>
            </div>
          </form>
          <label className="text-[11px] font-medium text-slate-500">
            Group by
            <select
              aria-label="Report group by"
              className={sel}
              value={groupby}
              onChange={(e) => store.setGroupby(datasetId, e.target.value)}
            >
              {columns.map((c) => (
                <option key={c.name} value={c.name}>{c.name} ({c.n_unique})</option>
              ))}
            </select>
          </label>
        </div>
      )}

      {mode === "gene" && !gene && (
        <p className="text-xs text-slate-400">
          Name a gene, or open one from the Markers table (the report icon on a row) or the plot.
        </p>
      )}
      {mode === "cluster" && !cluster && (
        <p className="text-xs text-slate-400">This dataset has no label column to report on.</p>
      )}

      {loading && (
        <p className="flex items-center gap-1.5 text-xs text-slate-500" role="status">
          <Loader2 className="h-3.5 w-3.5 animate-spin" /> Building the report…
        </p>
      )}
      {error && (
        <div className="rounded border border-red-200 bg-red-50 p-2 text-xs text-red-700" role="alert">
          {error.message}
          {error.suggestion && (
            <button
              type="button"
              className="ml-2 font-medium text-primary underline"
              onClick={() => store.setGene(datasetId, error.suggestion!)}
            >
              Show {error.suggestion}
            </button>
          )}
        </div>
      )}

      {report && !loading && (
        <div className="space-y-2">
          <div>
            <h3 className="text-sm font-semibold text-slate-800">{report.title}</h3>
            <p className="text-[11px] text-slate-500">
              {report.kind === "cluster"
                ? `${report.n_cells.toLocaleString()} of ${report.dataset.n_cells.toLocaleString()} cells in '${report.column}'.`
                : `Cells grouped by '${report.groupby}' (${report.dataset.n_cells.toLocaleString()} cells).`}{" "}
              Built in {report.elapsed_seconds.toFixed(1)} s.
            </p>
          </div>
          {report.sections.map((s) => (
            <ReportCard key={s.id} section={s}>
              {report.kind === "cluster" ? (
                <ClusterSectionBody section={s} report={report} onGene={onGene} />
              ) : (
                <GeneSectionBody section={s} report={report} onGene={onGene} onGroup={onGroup} />
              )}
            </ReportCard>
          ))}
        </div>
      )}
    </div>
  );
}
