import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, Upload, X } from "lucide-react";
import {
  deleteTranscripts,
  getTranscriptStatus,
  runTranscriptIndex,
  uploadOutlines,
  uploadTranscripts,
  type TranscriptStatus,
} from "@/api/transcripts";
import type { WatchdogEvent } from "@/api/pipelineRun";
import { Caption } from "@/components/common/Caption";
import { Help } from "@/components/common/Help";
import { InfoTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { errorMessage } from "@/lib/errorMessage";
import { aboutGb } from "@/lib/labelTransfer";
import { DOT_SIZE_RANGE, shownText, sizeText, timeText, transcriptColorCss } from "@/lib/transcripts";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useTissueViewStore } from "@/stores/tissueViewStore";

const fmt = (v: number | null | undefined, digits = 0) =>
  v == null ? "n/a" : v.toLocaleString(undefined, { maximumFractionDigits: digits });

/**
 * Transcripts of a Xenium run on the tissue plot: attach its transcripts.parquet (indexed here, by
 * gene and place, then deleted), choose genes, see them as dots. Also gives a dataset imported
 * without its boundaries files their outlines.
 */
export function TranscriptsSubtab() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = dataset?.id ?? null;
  const spatial = dataset?.spatial ?? null;
  const merged = (spatial?.libraries.length ?? 0) > 1;
  const embedding = useSettingsStore((s) => s.embedding);
  const setEmbedding = useSettingsStore((s) => s.setEmbedding);
  const genes = useTissueViewStore((s) => s.transcriptGenes);
  const setGenes = useTissueViewStore((s) => s.setTranscriptGenes);
  const dotSize = useTissueViewStore((s) => s.transcriptDotSize);
  const setDotSize = useTissueViewStore((s) => s.setTranscriptDotSize);
  const shown = useTissueViewStore((s) => s.transcriptShown);
  const refreshDataset = useDatasetRefresh();
  const [status, setStatus] = useState<TranscriptStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [upload, setUpload] = useState<{ name: string; fraction: number } | null>(null);
  const [progress, setProgress] = useState<{ fraction: number; message: string } | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [draft, setDraft] = useState("");
  const uploadAbort = useRef<AbortController | null>(null);
  const runAbort = useRef<AbortController | null>(null);

  const refresh = useCallback(async () => {
    if (!datasetId) return;
    try {
      setStatus(await getTranscriptStatus(datasetId));
    } catch (e) {
      setError(errorMessage(e));
    }
  }, [datasetId]);

  useEffect(() => {
    setStatus(null);
    setError(null);
    void refresh();
    return () => {
      uploadAbort.current?.abort();
      runAbort.current?.abort();
    };
  }, [refresh]);

  const send = async (file: File) => {
    if (!datasetId) return;
    setError(null);
    const ctl = new AbortController();
    uploadAbort.current = ctl;
    setUpload({ name: file.name, fraction: 0 });
    try {
      setStatus(await uploadTranscripts(datasetId, file, (fraction) => setUpload({ name: file.name, fraction }), ctl.signal));
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError(errorMessage(e));
      void refresh();
    } finally {
      setUpload(null);
    }
  };

  const index = async () => {
    if (!datasetId) return;
    setError(null);
    setWarning(null);
    setRunning(true);
    setProgress({ fraction: 0, message: "Starting" });
    const ctl = new AbortController();
    runAbort.current = ctl;
    try {
      await runTranscriptIndex(datasetId, {
        progress: (p) => setProgress({ fraction: p.done / p.total, message: p.message }),
        watchdog: (w: WatchdogEvent) => {
          if (w.kind === "memory") setWarning("Memory is running short while this runs.");
        },
        error: setError,
      }, ctl.signal);
      await refresh();
      await refreshDataset(datasetId);
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError(errorMessage(e));
    } finally {
      setRunning(false);
      setProgress(null);
    }
  };

  const remove = async () => {
    if (!datasetId) return;
    setError(null);
    try {
      await deleteTranscripts(datasetId);
      setGenes([]);
      await refresh();
      await refreshDataset(datasetId);
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  const sendOutlines = async (list: FileList | null) => {
    if (!datasetId) return;
    setError(null);
    const ctl = new AbortController();
    uploadAbort.current = ctl;
    try {
      for (const file of Array.from(list ?? [])) {
        setUpload({ name: file.name, fraction: 0 });
        await uploadOutlines(datasetId, file, (fraction) => setUpload({ name: file.name, fraction }), ctl.signal);
      }
      await refreshDataset(datasetId);
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError(errorMessage(e));
    } finally {
      setUpload(null);
    }
  };

  const panel = status?.genes ?? [];
  const max = status?.max_genes ?? 8;
  const addGene = (raw: string) => {
    const name = panel.find((g) => g.name.toLowerCase() === raw.trim().toLowerCase())?.name;
    if (!name || genes.includes(name) || genes.length >= max) return false;
    setGenes([...genes, name]);
    return true;
  };

  const summary = status?.summary;
  const check = summary?.check;
  const est = status?.estimate;
  const missingOutlines = (["cell", "nucleus"] as const).filter((k) => !(spatial?.outlines ?? []).includes(k));

  if (!spatial) return null;

  return (
    <div className="space-y-3 text-xs" data-testid="transcripts-subtab">
      <section className="space-y-2 rounded-lg border border-slate-200 p-2" aria-label="Transcripts">
        <h4 className="flex items-center gap-1 text-xs font-semibold text-slate-600">
          <span>Transcripts</span>
          <InfoTip label="About transcripts" width={320}>
            Every decoded molecule of the run, by position. Kept: quality q &ge; {status?.min_qv ?? 20} (10x&rsquo;s own default) and
            genes only, without the control features. The index stays with this dataset; the uploaded file is deleted once indexed.
          </InfoTip>
        </h4>

        {merged && (
          <p className="text-[11px] leading-snug text-amber-700">
            This dataset merges {spatial.libraries.length} sections and a transcripts file holds one: import the run on its own to attach its transcripts.
          </p>
        )}

        {!merged && !status?.attached && !status?.uploaded && (
          <>
            <p className="text-[11px] leading-snug text-slate-500">
              Upload <code>transcripts.parquet</code> from the Xenium output folder (beside <code>cells.parquet</code>; about
              0.7 GB for a 300-gene run). It is indexed here by gene and place, which needs about half its size on disk and takes
              under a minute.
            </p>
            <label className="inline-flex cursor-pointer items-center gap-1 rounded border border-slate-200 px-2 py-1 text-xs text-slate-600 hover:bg-slate-50">
              <Upload className="h-3 w-3" /> Choose transcripts.parquet
              <input
                type="file"
                accept=".parquet"
                className="sr-only"
                aria-label="transcripts.parquet"
                disabled={!!upload}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void send(f);
                  e.target.value = "";
                }}
              />
            </label>
          </>
        )}

        {upload && (
          <div className="space-y-0.5" role="status">
            <p className="text-[11px] text-slate-500">Uploading {upload.name}: {Math.round(upload.fraction * 100)}%</p>
            <div className="h-1 rounded bg-slate-100"><div className="h-1 rounded bg-blue-500" style={{ width: `${upload.fraction * 100}%` }} /></div>
          </div>
        )}

        {status?.uploaded && (
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
                onClick={() => void index()}
              >
                Index transcripts
              </button>
            )}
            {est && !running && (
              <span className="text-[11px] leading-snug text-slate-500" data-testid="transcripts-estimate">
                {est.rows.toLocaleString()} rows: needs {sizeText(est.disk_bytes)} of disk, {aboutGb(est.peak_bytes)} of memory, about{" "}
                {timeText(est.seconds_range[0])} to {timeText(est.seconds_range[1])}.
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

        {summary && (
          <div className="space-y-1" data-testid="transcripts-summary">
            <Caption n={{ value: summary.n_genes, unit: "genes" }}>
              {fmt(summary.n_kept)} transcripts kept of {fmt(summary.n_rows)} read: {fmt(summary.n_low_quality)} under q{" "}
              {summary.min_qv} and {fmt(summary.n_control)} of control features left out.{" "}
              {status?.disk_bytes != null && <>{sizeText(status.disk_bytes)} on disk. </>}
              <button type="button" className="text-slate-400 underline hover:text-slate-600" onClick={() => void remove()}>
                Remove
              </button>
            </Caption>
            {check?.available ? (
              <Caption testId="transcripts-check">
                Against the count matrix, per gene (q &ge; {summary.min_qv} with a cell id): r = {check.r != null ? check.r.toFixed(4) : "n/a"},
                median difference {fmt(check.median_abs_diff)}, largest {fmt(check.max_abs_diff)}
                {check.max_abs_diff ? ` (${check.max_diff_gene})` : ""}; {fmt(check.n_equal)} of {fmt(check.n_genes)} genes equal.
                {check.only_in_transcripts?.length ? ` Only among the transcripts: ${check.only_in_transcripts.join(", ")}.` : ""}
                {check.only_in_matrix?.length ? ` Only in the matrix: ${check.only_in_matrix.join(", ")}.` : ""}{" "}
                <Help id="transcripts.check" label="About the check against the count matrix" />
              </Caption>
            ) : (
              check?.note && <Caption testId="transcripts-check">{check.note}</Caption>
            )}
          </div>
        )}

        {status?.attached && (
          <div className="space-y-2">
            <form
              className="flex items-center gap-1"
              onSubmit={(e) => {
                e.preventDefault();
                if (addGene(draft)) setDraft("");
              }}
            >
              <input
                list="transcript-genes"
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                placeholder={`Add a gene (up to ${max})`}
                aria-label="Add a gene"
                disabled={genes.length >= max}
                className="w-40 rounded border border-slate-300 bg-white px-2 py-1 text-xs text-slate-700 focus:border-blue-400 focus:outline-none"
              />
              <datalist id="transcript-genes">
                {panel.map((g) => (
                  <option key={g.name} value={g.name}>{g.n.toLocaleString()}</option>
                ))}
              </datalist>
              <button
                type="submit"
                className="rounded border border-slate-300 px-2 py-1 text-xs text-slate-600 hover:bg-slate-50 disabled:opacity-50"
                disabled={genes.length >= max}
              >
                Add
              </button>
              <Help id="transcripts.genes" label="About drawing transcripts" />
            </form>
            {genes.length > 0 && (
              <ul className="flex flex-wrap gap-1" aria-label="Genes drawn">
                {genes.map((g, i) => (
                  <li key={g} className="flex items-center gap-1 rounded border border-slate-200 px-1.5 py-0.5 text-[11px] text-slate-700">
                    <span className="inline-block h-2.5 w-2.5 rounded-full border border-slate-400" style={{ background: transcriptColorCss(i) }} />
                    {g}
                    {shown?.perGene[i] != null && <span className="text-slate-400">{shown.perGene[i]!.toLocaleString()}</span>}
                    <button type="button" aria-label={`Remove ${g}`} title={`Remove ${g}`} className="text-slate-400 hover:text-slate-600" onClick={() => setGenes(genes.filter((x) => x !== g))}>
                      <X className="h-3 w-3" />
                    </button>
                  </li>
                ))}
              </ul>
            )}
            <div className="flex items-center gap-1">
              <label className="flex items-center gap-1 text-[11px] font-medium text-slate-500">
                Dot size
                <input
                  type="range"
                  min={DOT_SIZE_RANGE[0]}
                  max={DOT_SIZE_RANGE[1]}
                  step={0.5}
                  value={dotSize}
                  onChange={(e) => setDotSize(parseFloat(e.target.value))}
                  className="w-20 accent-blue-500"
                />
              </label>
              <Help id="transcripts.dot-size" label="About the dot size" />
            </div>
            {genes.length > 0 && shown && <Caption testId="transcripts-shown">{shownText(shown.shown, shown.total)}</Caption>}
            {genes.length > 0 && embedding !== "spatial" && (
              <p className="text-[11px] leading-snug text-slate-500">
                The dots are drawn on the tissue plot.{" "}
                <button type="button" className="text-blue-600 underline" onClick={() => setEmbedding("spatial")}>
                  Show the tissue
                </button>
              </p>
            )}
            <Caption>
              The positions of every molecule of the run, whichever cells are filtered. An h5ad export carries neither outlines nor
              transcripts: they stay with this dataset here.
            </Caption>
          </div>
        )}
      </section>

      {!merged && missingOutlines.length > 0 && (
        <section className="space-y-2 rounded-lg border border-slate-200 p-2" aria-label="Cell outlines">
          <h4 className="flex items-center gap-1 text-xs font-semibold text-slate-600">
            <span>Cell outlines</span>
            <Help id="transcripts.outlines" label="About cell outlines" />
          </h4>
          <p className="text-[11px] leading-snug text-slate-500">
            Upload {missingOutlines.map((k) => `${k === "cell" ? "cell" : "nucleus"}_boundaries.parquet`).join(" and ")} from the Xenium
            output folder (about 9 MB each) to draw cells instead of points: the plot&rsquo;s Draw choice appears with them. A new import of
            the folder brings them along.
          </p>
          {spatial.outlines_note && <p className="text-[11px] leading-snug text-amber-700">{spatial.outlines_note}</p>}
          <label className="inline-flex cursor-pointer items-center gap-1 rounded border border-slate-200 px-2 py-1 text-xs text-slate-600 hover:bg-slate-50">
            <Upload className="h-3 w-3" /> Choose boundaries files
            <input
              type="file"
              multiple
              accept=".parquet,.gz"
              className="sr-only"
              aria-label="Boundaries files"
              disabled={!!upload}
              onChange={(e) => {
                void sendOutlines(e.target.files);
                e.target.value = "";
              }}
            />
          </label>
        </section>
      )}
    </div>
  );
}
