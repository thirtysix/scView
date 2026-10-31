import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Loader2, Upload } from "lucide-react";
import type { Data, Layout } from "plotly.js";
import {
  deleteFragments,
  getFragmentLengths,
  getFragments,
  getQcValues,
  runFragmentQc,
  uploadFragmentFile,
  type FragmentLengths,
  type FragmentQc,
  type FragmentStatus,
  type QcValues,
} from "@/api/atacFragments";
import type { WatchdogEvent } from "@/api/pipelineRun";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import Plot from "@/components/plots/Plot";
import { useCategoryColors } from "@/hooks/useCategoryColors";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import {
  histogramFigure,
  lengthByGroupFigure,
  lengthFigure,
  tssProfileFigure,
  violinFigure,
} from "@/lib/atacPlots";
import { rgbCss } from "@/lib/categoryColors";
import { aboutGb } from "@/lib/labelTransfer";
import { plotConfig } from "@/lib/plotConfig";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";

// Common cut-offs, from Signac's PBMC vignette (https://stuartlab.org/signac/articles/pbmc_vignette).
const TSS_LOW = 3;
const NUCLEOSOME_HIGH = 4;

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

const fmt = (v: number | null | undefined, digits = 2) => (v == null ? "n/a" : v.toLocaleString(undefined, { maximumFractionDigits: digits }));

/** Files in the order they upload: the index last, so the status settles on the pair. */
export function uploadOrder(files: File[]): File[] {
  return [...files].sort((a, b) => Number(a.name.toLowerCase().endsWith(".tbi")) - Number(b.name.toLowerCase().endsWith(".tbi")));
}

/**
 * ATAC fragment QC (backend core/atac_fragments.py): attach atac_fragments.tsv.gz + .tbi, run
 * one pass over the file, and see each cell's TSS enrichment, nucleosome signal and the
 * fragment length ladder, overall and by group. Signac's definitions.
 */
export function AtacFragmentQc({ datasetId }: { datasetId: string }) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const colorBy = useSettingsStore((s) => s.colorBy);
  const refreshDataset = useDatasetRefresh();
  const [status, setStatus] = useState<FragmentStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [upload, setUpload] = useState<{ name: string; fraction: number } | null>(null);
  const [progress, setProgress] = useState<{ fraction: number; message: string } | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [groupBy, setGroupBy] = useState("");
  const [values, setValues] = useState<QcValues | null>(null);
  const [lengths, setLengths] = useState<FragmentLengths | null>(null);
  const uploadAbort = useRef<AbortController | null>(null);
  const runAbort = useRef<AbortController | null>(null);

  const columns = useMemo(
    () => (dataset?.obs_columns ?? []).filter(
      (c) => (c.dtype === "category" || c.dtype === "object" || c.dtype === "bool") && c.n_unique >= 1 && c.n_unique <= 24,
    ),
    [dataset],
  );
  useEffect(() => {
    setGroupBy((cur) => {
      if (cur && columns.some((c) => c.name === cur)) return cur;
      return columns.some((c) => c.name === colorBy) ? colorBy : "";
    });
  }, [columns, colorBy]);

  const refresh = useCallback(async () => {
    try {
      setStatus(await getFragments(datasetId));
    } catch (e) {
      setError(errorMessage(e));
    }
  }, [datasetId]);

  useEffect(() => {
    setStatus(null);
    setValues(null);
    setLengths(null);
    setError(null);
    void refresh();
    return () => {
      uploadAbort.current?.abort();
      runAbort.current?.abort();
    };
  }, [datasetId, refresh]);

  const qc: FragmentQc | null = status?.qc ?? null;
  const createdAt = qc?.created_at;

  useEffect(() => {
    if (!createdAt) return;
    let cancelled = false;
    Promise.all([getQcValues(datasetId, groupBy || null), getFragmentLengths(datasetId, groupBy || null)])
      .then(([v, l]) => {
        if (cancelled) return;
        setValues(v);
        setLengths(l);
      })
      .catch((e) => !cancelled && setError(errorMessage(e)));
    return () => {
      cancelled = true;
    };
  }, [datasetId, groupBy, createdAt]);

  const colors = useCategoryColors(groupBy || null);
  const colorOf = useCallback((g: string) => (groupBy ? rgbCss(colors(g)) : "#2563eb"), [groupBy, colors]);

  const onFiles = async (list: FileList | null) => {
    const files = uploadOrder(Array.from(list ?? []));
    if (!files.length) return;
    setError(null);
    const ctl = new AbortController();
    uploadAbort.current = ctl;
    try {
      for (const file of files) {
        setUpload({ name: file.name, fraction: 0 });
        const next = await uploadFragmentFile(datasetId, file, (fraction) => setUpload({ name: file.name, fraction }), ctl.signal);
        setStatus((cur) => ({ ...cur, ...next }));
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError(errorMessage(e));
      void refresh();
    } finally {
      setUpload(null);
    }
  };

  const onRun = async () => {
    setError(null);
    setWarning(null);
    setRunning(true);
    setProgress({ fraction: 0, message: "Starting" });
    const ctl = new AbortController();
    runAbort.current = ctl;
    try {
      await runFragmentQc(datasetId, {
        progress: setProgress,
        watchdog: (w: WatchdogEvent) => {
          if (w.kind === "memory") setWarning("Memory is running short while this runs.");
        },
        error: setError,
      }, ctl.signal);
      await refresh();
      // The new per-cell columns: colour the embedding by them.
      void refreshDataset(datasetId, "atac_tss_enrichment").catch(() => undefined);
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError(errorMessage(e));
    } finally {
      setRunning(false);
      setProgress(null);
    }
  };

  const onRemove = async () => {
    setError(null);
    try {
      await deleteFragments(datasetId);
      await refresh();
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  const est = status?.estimate;
  const cut = values?.groups.some((g) => g.n_shown < g.n);
  const allValues = useMemo(() => ({
    tss: values?.groups.flatMap((g) => g.tss_enrichment) ?? [],
    nuc: values?.groups.flatMap((g) => g.nucleosome_signal) ?? [],
  }), [values]);
  const shown = values?.groups.reduce((a, g) => a + g.n_shown, 0) ?? 0;
  const total = values?.groups.reduce((a, g) => a + g.n, 0) ?? 0;

  return (
    <section className="space-y-2 rounded-lg border border-slate-200 p-2" aria-label="Fragment QC">
      <h4 className="flex items-center gap-1 text-xs font-semibold text-slate-600">
        <span>Fragment QC</span>
        <InfoTip label="About fragment QC" width={320}>
          The fragments file lists every ATAC fragment with its position and cell. One pass over it scores each cell&rsquo;s
          TSS enrichment and nucleosome signal, with Signac&rsquo;s definitions, and gives the coverage tracks their scale.
        </InfoTip>
      </h4>

      {!status?.attached && (
        <p className="text-[11px] leading-snug text-slate-500">
          Attach <code>atac_fragments.tsv.gz</code> and <code>atac_fragments.tsv.gz.tbi</code>, which Cell Ranger ARC writes
          side by side in <code>outs/</code>. They stay with this dataset.
        </p>
      )}
      {status?.missing && <p className="text-[11px] leading-snug text-amber-700" data-testid="atac-missing">{status.missing}</p>}
      {!status?.attached && (
        <label className="inline-flex cursor-pointer items-center gap-1 rounded border border-slate-200 px-2 py-1 text-xs text-slate-600 hover:bg-slate-50">
          <Upload className="h-3 w-3" /> Choose files
          <input
            type="file"
            multiple
            accept=".gz,.bgz,.tbi"
            className="sr-only"
            aria-label="Fragments file and index"
            disabled={!!upload}
            onChange={(e) => {
              void onFiles(e.target.files);
              e.target.value = "";
            }}
          />
        </label>
      )}
      {upload && (
        <div className="space-y-0.5" role="status">
          <p className="text-[11px] text-slate-500">Uploading {upload.name}: {Math.round(upload.fraction * 100)}%</p>
          <div className="h-1 rounded bg-slate-100"><div className="h-1 rounded bg-blue-500" style={{ width: `${upload.fraction * 100}%` }} /></div>
        </div>
      )}

      {status?.attached && (
        <p className="text-[11px] leading-snug text-slate-500" data-testid="atac-attached">
          {status.name ?? "Fragments file"} attached
          {status.barcode_share != null && (
            <>, {(status.barcode_share * 100).toFixed(1)}% of its first {(status.lines_checked ?? 0).toLocaleString()} fragments in this dataset&rsquo;s cells</>
          )}
          {status.genome && <>, {status.genome}</>}.{" "}
          <button type="button" className="text-slate-400 underline hover:text-slate-600" onClick={() => void onRemove()}>
            Remove
          </button>
        </p>
      )}

      {status?.attached && (
        <div className="flex flex-wrap items-center gap-2">
          {running ? (
            <button
              type="button"
              className="rounded border border-slate-300 px-2 py-1 text-xs text-slate-600 hover:bg-slate-50"
              onClick={() => runAbort.current?.abort()}
            >
              Cancel
            </button>
          ) : (
            <button
              type="button"
              className="rounded bg-blue-600 px-2 py-1 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-50"
              disabled={status.running}
              onClick={() => void onRun()}
            >
              {qc ? "Run fragment QC again" : "Run fragment QC"}
            </button>
          )}
          {est && !running && (
            <span className="text-[11px] leading-snug text-slate-500" data-testid="atac-estimate">
              Needs {aboutGb(est.peak_bytes)} of memory. {est.note}
            </span>
          )}
        </div>
      )}
      {progress && (
        <div className="space-y-0.5" role="status">
          <p className="flex items-center gap-1 text-[11px] text-slate-500">
            <Loader2 className="h-3 w-3 animate-spin" /> {progress.message}
          </p>
          <div className="h-1 rounded bg-slate-100"><div className="h-1 rounded bg-blue-500" style={{ width: `${progress.fraction * 100}%` }} /></div>
        </div>
      )}
      {warning && <p className="text-[11px] text-amber-700">{warning}</p>}
      <RequestError message={error} />

      {qc && (
        <div className="space-y-2" data-testid="atac-qc">
          <dl className="grid grid-cols-3 gap-2 text-[11px]">
            <div>
              <dt className="flex items-center gap-1 text-slate-400">
                TSS enrichment
                <InfoTip label="About TSS enrichment" width={320}>
                  Tn5 cuts per base within 500 bp of a protein-coding TSS, over cuts per base in the 100 bp at each end of
                  +-1 kb. Open promoters make it high; a cell with little signal over background has it near 1. Signac&rsquo;s
                  PBMC vignette keeps cells above {TSS_LOW}; ENCODE&rsquo;s bulk ATAC standard calls 5-7 acceptable for GRCh38 with a
                  slightly different measure.
                </InfoTip>
              </dt>
              <dd className="font-medium tabular-nums text-slate-700">{fmt(qc.medians.tss_enrichment)} (median)</dd>
            </div>
            <div>
              <dt className="flex items-center gap-1 text-slate-400">
                Nucleosome signal
                <InfoTip label="About nucleosome signal" width={320}>
                  Fragments spanning one nucleosome (147-294 bp) over nucleosome-free ones (under 147 bp). Signac&rsquo;s PBMC
                  vignette keeps cells below {NUCLEOSOME_HIGH}; high values mean few short fragments, a sign of over-digested or
                  poorly transposed nuclei.
                </InfoTip>
              </dt>
              <dd className="font-medium tabular-nums text-slate-700">{fmt(qc.medians.nucleosome_signal)} (median)</dd>
            </div>
            <div>
              <dt className="text-slate-400">Fragments per cell</dt>
              <dd className="font-medium tabular-nums text-slate-700">{fmt(qc.medians.n_fragments, 0)} (median)</dd>
            </div>
          </dl>
          <Caption n={[{ value: qc.n_cells, unit: "cells" }, { value: qc.n_fragments_in_cells, unit: "fragments in cells", of: qc.n_lines }]}>
            Read in {Math.round(qc.seconds)} s; TSSs: {qc.params.annotation ?? "gene table"}, {qc.tss_profile.n_tss.toLocaleString()} protein-coding genes.
          </Caption>
          {qc.notes.map((n) => (
            <p key={n} className="text-[11px] leading-snug text-amber-700">{n}</p>
          ))}

          <label className="flex items-center gap-1 text-[11px] text-slate-500">
            Group by
            <select
              aria-label="Group by"
              className="rounded border border-slate-200 bg-white px-1 py-0.5 text-xs text-slate-700"
              value={groupBy}
              onChange={(e) => setGroupBy(e.target.value)}
            >
              <option value="">All cells</option>
              {columns.map((c) => (
                <option key={c.name} value={c.name}>{c.name}</option>
              ))}
            </select>
          </label>

          {values && (
            <figure className="space-y-1">
              <div className="grid gap-2 md:grid-cols-2">
                <QcPlot figure={histogramFigure("TSS enrichment", allValues.tss, { low: TSS_LOW })} name="atac_tss_enrichment_histogram" />
                <QcPlot figure={histogramFigure("Nucleosome signal", allValues.nuc, { high: NUCLEOSOME_HIGH })} name="atac_nucleosome_signal_histogram" />
                {groupBy && (
                  <>
                    <QcPlot figure={violinFigure("TSS enrichment", values.groups, "tss_enrichment", colorOf)} name={`atac_tss_enrichment_by_${groupBy}`} />
                    <QcPlot figure={violinFigure("Nucleosome signal", values.groups, "nucleosome_signal", colorOf)} name={`atac_nucleosome_signal_by_${groupBy}`} />
                  </>
                )}
              </div>
              <Caption n={{ value: shown, unit: "cells", of: total }}>
                Dotted lines: Signac&rsquo;s PBMC vignette cut-offs (TSS enrichment {TSS_LOW}, nucleosome signal {NUCLEOSOME_HIGH}).
                {cut && ` A random ${values.max_per_group.toLocaleString()} per group.`}
              </Caption>
            </figure>
          )}

          <figure className="space-y-1">
            <div className="grid gap-2 md:grid-cols-2">
              {lengths && lengths.groups.length > 1
                ? <QcPlot figure={lengthByGroupFigure(lengths, colorOf)} name={`atac_fragment_lengths_by_${groupBy}`} />
                : <QcPlot figure={lengthFigure(qc.length_histogram)} name="atac_fragment_lengths" />}
              <QcPlot figure={tssProfileFigure(qc.tss_profile)} name="atac_tss_profile" />
            </div>
            <Caption n={{ value: qc.length_histogram.n, unit: "fragments" }}>
              Left: on a good library the lengths form a ladder, a nucleosome-free peak under 147 bp, then steps at about one
              and two nucleosomes (near 200 and 400 bp). Right: cuts around {qc.tss_profile.n_tss.toLocaleString()} TSSs, read
              5&prime; to 3&prime;, as a fold over the flanks.
            </Caption>
          </figure>
        </div>
      )}
    </section>
  );
}

function QcPlot({ figure, name }: { figure: { data: Data[]; layout: Partial<Layout> }; name: string }) {
  return <Plot data={figure.data} layout={figure.layout} config={plotConfig(name)} useResizeHandler style={{ width: "100%" }} />;
}
