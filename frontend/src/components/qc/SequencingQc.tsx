import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { AlertTriangle, Info, Loader2, Paperclip, XCircle } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { Panel } from "@/components/common/Panel";
import { InfoTip } from "@/components/common/InfoTip";
import { Caption } from "@/components/common/Caption";
import {
  attachSeqMetrics,
  getSeqMetrics,
  removeSeqMetrics,
  type SeqFlag,
  type SeqMetricsResponse,
  type SeqSample,
} from "@/api/seqqc";
import {
  ATTACH_ACCEPT,
  curveCaption,
  flagFor,
  flagTip,
  formatMetric,
  kneeSentence,
  kneeShapes,
  LEVEL_STYLE,
  LEVEL_WORD,
  metricLabel,
  metricTip,
  rankTraces,
  sampleColumns,
  TABLE_METRICS,
} from "@/lib/seqqc";
import { plotConfig } from "@/lib/plotConfig";
import { useDatasetStore } from "@/stores/datasetStore";

const FILL_STYLE = { width: "100%", height: "100%" } as const;
const TH = "whitespace-nowrap px-1.5 py-1 text-left font-medium";
const NUM = "whitespace-nowrap px-1.5 py-1 text-right tabular-nums";
const LEVEL_ICON = { error: XCircle, warn: AlertTriangle, info: Info } as const;
const CELL_TONE = { error: "text-red-700 font-semibold", warn: "text-amber-700 font-semibold", info: "text-sky-700" } as const;

function FlagBadge({ flag }: { flag: SeqFlag }) {
  const Icon = LEVEL_ICON[flag.level];
  return (
    <span
      className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[10px] font-medium ${LEVEL_STYLE[flag.level]}`}
      title={flagTip(flag)}
      data-testid="seqqc-flag-badge"
    >
      <Icon className="h-3 w-3" />
      {LEVEL_WORD[flag.level]}: {flag.title}
    </span>
  );
}

function Th({ children, tip, right = false }: { children: ReactNode; tip: string; right?: boolean }) {
  return (
    <th className={right ? `${TH} text-right` : TH} title={tip}>
      {children}
    </th>
  );
}

/** Upload form: metrics_summary.csv and/or the raw matrix, for the whole dataset or one sample. */
function AttachForm({
  datasetId,
  onAttached,
  onCancel,
}: {
  datasetId: string;
  onAttached: (r: SeqMetricsResponse) => void;
  onCancel?: () => void;
}) {
  const obsColumns = useDatasetStore((s) => s.currentDataset?.obs_columns);
  const columns = useMemo(() => sampleColumns(obsColumns), [obsColumns]);
  const [files, setFiles] = useState<File[]>([]);
  const [column, setColumn] = useState("");
  const [sample, setSample] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const values = columns.find((c) => c.name === column)?.values ?? [];
  const canSend = files.length > 0 && !busy && (!column || !!sample);

  const send = async () => {
    setBusy(true);
    setError(null);
    try {
      onAttached(await attachSeqMetrics(datasetId, files, column ? { column, sample } : {}));
      setFiles([]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-2 rounded-md bg-slate-50 p-2 text-[11px] text-slate-600" data-testid="seqqc-attach">
      <p>
        Attach the Cell Ranger run&apos;s <code>metrics_summary.csv</code> and/or its raw matrix (
        <code>raw_feature_bc_matrix.h5</code>, or the raw <code>matrix.mtx</code>, <code>barcodes.tsv</code> and{" "}
        <code>features.tsv</code>) from the <code>outs/</code> folder. The raw matrix gives the barcode-rank plot.
      </p>
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => input.current?.click()}
          disabled={busy}
          className="flex items-center gap-1 rounded border border-slate-300 bg-white px-2 py-0.5 text-[11px] font-medium text-slate-600 hover:bg-slate-100 disabled:opacity-50"
        >
          <Paperclip className="h-3 w-3" /> Choose files
        </button>
        <input
          ref={input}
          type="file"
          multiple
          accept={ATTACH_ACCEPT}
          className="hidden"
          data-testid="seqqc-file-input"
          onChange={(e) => {
            setFiles(Array.from(e.target.files ?? []));
            e.target.value = "";
          }}
        />
        <span className="min-w-0 truncate" data-testid="seqqc-file-names">
          {files.length ? files.map((f) => f.name).join(", ") : "No files chosen"}
        </span>
      </div>
      {columns.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <label htmlFor="seqqc-column">For</label>
          <select
            id="seqqc-column"
            value={column}
            onChange={(e) => {
              setColumn(e.target.value);
              setSample("");
            }}
            title="A merged dataset: the column that tells its samples apart, then the sample these files belong to. Leave at 'the whole dataset' for one sample."
            className="rounded border border-slate-200 bg-white px-1.5 py-0.5 text-xs text-slate-600"
          >
            <option value="">the whole dataset</option>
            {columns.map((c) => (
              <option key={c.name} value={c.name}>
                one {c.name} ({c.n_unique})
              </option>
            ))}
          </select>
          {column && (
            <select
              aria-label="Sample"
              value={sample}
              onChange={(e) => setSample(e.target.value)}
              className="max-w-[45%] rounded border border-slate-200 bg-white px-1.5 py-0.5 text-xs text-slate-600"
            >
              <option value="">choose…</option>
              {values.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          )}
        </div>
      )}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => void send()}
          disabled={!canSend}
          className="flex items-center gap-1 rounded bg-primary px-2 py-0.5 text-[11px] font-medium text-white hover:bg-primary/90 disabled:opacity-50"
        >
          {busy && <Loader2 className="h-3 w-3 animate-spin" />}
          {busy ? "Reading…" : "Attach"}
        </button>
        {onCancel && (
          <button type="button" onClick={onCancel} disabled={busy} className="text-slate-400 hover:text-slate-600 hover:underline">
            Cancel
          </button>
        )}
        {busy && <span className="text-slate-400">A raw matrix of millions of barcodes takes a few seconds.</span>}
      </div>
      {error && (
        <p role="alert" className="rounded-md bg-amber-50 px-2 py-1.5 text-xs text-amber-800">
          {error}
        </p>
      )}
    </div>
  );
}

function SampleTable({ data, focus, onFocus }: { data: SeqMetricsResponse; focus: number; onFocus: (i: number) => void }) {
  const info = data.metrics_info;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[11px] text-slate-600" data-testid="seqqc-table">
        <thead className="border-b border-slate-200 text-slate-500">
          <tr>
            <Th tip="The sample these metrics belong to: a value of the dataset's sample column for a merged dataset, else the dataset.">Sample</Th>
            {TABLE_METRICS.map((m) => (
              <Th key={m.key} right tip={`${metricLabel(info, m.key)}: ${metricTip(info, m.key)}`}>
                {m.short}
              </Th>
            ))}
            <Th tip="Cell Ranger web-summary alerts this sample's metrics cross (warning or error), and 10x recommendations (note).">Flags</Th>
          </tr>
        </thead>
        <tbody>
          {data.samples.map((s, i) => {
            const vals = s.metrics?.values ?? {};
            const worst = s.flags.filter((f) => f.level !== "info").length;
            return (
              <tr
                key={s.sample}
                onClick={() => onFocus(i)}
                className={`cursor-pointer border-b border-slate-100 ${i === focus ? "bg-slate-50" : "hover:bg-slate-50/60"}`}
              >
                <td className="max-w-[9rem] truncate px-1.5 py-1 font-medium text-slate-700" title={s.sample}>
                  {s.sample}
                </td>
                {TABLE_METRICS.map((m) => {
                  const f = flagFor(s.flags, m.key);
                  return (
                    <td key={m.key} className={`${NUM} ${f ? CELL_TONE[f.level] : ""}`} title={f ? flagTip(f) : undefined}>
                      {formatMetric(m.key, vals[m.key])}
                    </td>
                  );
                })}
                <td className="whitespace-nowrap px-1.5 py-1">
                  {!s.metrics ? "no metrics file" : worst ? `${worst} alert${worst > 1 ? "s" : ""}` : "none"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function RankPlot({ sample }: { sample: SeqSample }) {
  const k = sample.knee!;
  const traces = useMemo(
    () => rankTraces(k, sample.cells_source === "ingest" ? "Cells (filtered matrix)" : "Cells in the dataset"),
    [k, sample.cells_source],
  );
  const { shapes, annotations } = useMemo(() => kneeShapes(k), [k]);
  return (
    <>
      <Panel title={`Barcode ranks: ${sample.sample}`} className="h-[320px]">
        <Plot
          data={traces}
          layout={{
            autosize: true,
            margin: { l: 56, r: 10, t: 10, b: 40 },
            // dtick 1: a label per power of ten only (no 2 / 5 minor labels).
            xaxis: { type: "log", dtick: 1, title: { text: "Barcode rank", font: { size: 10 } }, tickfont: { size: 9 } },
            yaxis: { type: "log", dtick: 1, title: { text: "UMI count", font: { size: 10 } }, tickfont: { size: 9 } },
            shapes,
            annotations,
            showlegend: true,
            legend: { font: { size: 9 }, x: 0, y: 0, xanchor: "left", yanchor: "bottom", bgcolor: "rgba(255,255,255,0.7)" },
          }}
          config={plotConfig(`barcode_ranks_${sample.sample}`)}
          style={FILL_STYLE}
          useResizeHandler
        />
      </Panel>
      <Caption n={{ value: k.n_barcodes, unit: "barcodes with UMIs" }} testId="seqqc-curve-caption">
        {curveCaption(k)}
      </Caption>
    </>
  );
}

/** The sample attached last (a re-attached sample keeps its place in the list). */
function newestSample(r: SeqMetricsResponse): number {
  let best = 0;
  r.samples.forEach((s, i) => {
    if (s.attached_at > (r.samples[best]?.attached_at ?? "")) best = i;
  });
  return best;
}

/**
 * Sequencing QC (N22): the Cell Ranger run's metrics_summary.csv per sample, flagged
 * against Cell Ranger's own web-summary alerts, and the raw matrix's barcode-rank
 * curve with the knee and inflection (DropletUtils barcodeRanks) and the dataset's
 * cells marked (core/seq_metrics.py). Files come with the upload or are attached here.
 */
export function SequencingQc({ datasetId }: { datasetId: string }) {
  const [data, setData] = useState<SeqMetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [attaching, setAttaching] = useState(false);
  const [focus, setFocus] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);
    setAttaching(false);
    setFocus(0);
    getSeqMetrics(datasetId)
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  const sample = data?.samples[Math.min(focus, (data?.samples.length ?? 1) - 1)];
  const [removing, setRemoving] = useState(false);

  const remove = async (name: string) => {
    setRemoving(true);
    setError(null);
    try {
      setData(await removeSeqMetrics(datasetId, name));
      setFocus(0);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRemoving(false);
    }
  };
  const notes = sample ? [...(sample.metrics?.notes ?? []), ...sample.notes] : [];

  return (
    <div className="space-y-2 rounded-xl border border-slate-200 bg-white p-3" data-testid="sequencing-qc">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <h3 className="text-sm font-semibold text-slate-700">Sequencing QC (Cell Ranger)</h3>
          <InfoTip label="About sequencing QC" width={320}>
            Read from the Cell Ranger run&apos;s metrics_summary.csv and raw (unfiltered) barcode matrix. Flags are Cell
            Ranger&apos;s own web-summary alerts (version 10.1 thresholds; error and warning lines), plus 10x&apos;s
            recommended depth of 20,000 read pairs per cell as a note. The barcode-rank plot ranks every barcode with
            UMIs by its total; the knee and inflection are computed as DropletUtils barcodeRanks does (curve tracing),
            and the cells of the dataset are coloured.
          </InfoTip>
        </div>
        <span className="flex items-center gap-2">
          {loading && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
          {data && !attaching && (
            <button
              type="button"
              onClick={() => setAttaching(true)}
              className="rounded border border-slate-200 px-2 py-0.5 text-[11px] font-medium text-slate-600 hover:bg-slate-100"
              title="Attach metrics or a raw matrix for another sample, or replace this one's."
            >
              Attach files
            </button>
          )}
        </span>
      </div>
      {error && (
        <p role="alert" className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
          {error}
        </p>
      )}
      {!loading && !error && !data && !attaching && (
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500" data-testid="seqqc-none">
          <span>No Cell Ranger metrics are attached to this dataset, so sequencing depth, mapping and cell calling can&apos;t be checked.</span>
          <button
            type="button"
            onClick={() => setAttaching(true)}
            className="flex items-center gap-1 rounded border border-slate-300 px-2 py-0.5 font-medium text-slate-600 hover:bg-slate-100"
          >
            <Paperclip className="h-3 w-3" /> Attach Cell Ranger metrics
          </button>
        </div>
      )}
      {attaching && (
        <AttachForm
          datasetId={datasetId}
          onAttached={(r) => {
            setData(r);
            setAttaching(false);
            setFocus(newestSample(r));
          }}
          onCancel={() => setAttaching(false)}
        />
      )}
      {data && sample && (
        <>
          <SampleTable data={data} focus={focus} onFocus={setFocus} />
          <Caption n={{ value: data.n_samples, unit: data.n_samples === 1 ? "sample" : "samples" }}>
            One row per sample, as Cell Ranger reported it
            {data.n_samples > 1 ? "; click a row for its flags and barcode ranks" : ""}. Coloured values cross a Cell
            Ranger alert line (hover for the line and its source).
          </Caption>
          <p className="text-[11px] text-slate-600" data-testid="seqqc-summary">
            <span className="font-medium text-slate-700">
              {sample.sample}
              {sample.column ? ` (${sample.column})` : ""}:
            </span>{" "}
            {sample.summary}{" "}
            <button
              type="button"
              onClick={() => void remove(sample.sample)}
              disabled={removing}
              className="text-slate-400 underline hover:text-slate-600 disabled:opacity-50"
              title={`Remove ${sample.sample}'s metrics and barcode ranks from the dataset (to attach the right files instead).`}
            >
              Remove
            </button>
          </p>
          {sample.flags.length > 0 && (
            <ul className="space-y-1" data-testid="seqqc-flags">
              {sample.flags.map((f) => (
                <li key={`${f.metric}-${f.level}`} className="space-y-0.5">
                  <FlagBadge flag={f} />
                  <p className="text-[11px] text-slate-600">
                    {f.message}{" "}
                    <a href={f.url} target="_blank" rel="noreferrer" className="text-slate-400 underline hover:text-slate-600">
                      {f.source}
                    </a>
                  </p>
                </li>
              ))}
            </ul>
          )}
          {sample.knee ? (
            <>
              {sample.knee.curve.rank.length > 0 && <RankPlot sample={sample} />}
              <p className="text-[11px] text-slate-600" data-testid="seqqc-knee">
                {kneeSentence(sample)}
              </p>
            </>
          ) : (
            <p className="text-[11px] text-slate-500">
              No raw (unfiltered) matrix is attached for {sample.sample}, so there is no barcode-rank plot.
            </p>
          )}
          {notes.map((n, i) => (
            <p key={i} className="rounded-md bg-slate-50 px-2 py-1.5 text-[11px] text-slate-600">
              {n}
            </p>
          ))}
          {sample.metrics && (
            <details className="text-[11px] text-slate-600">
              <summary className="cursor-pointer text-slate-500 hover:text-slate-700">
                All {sample.metrics.raw.length} rows of {sample.files.metrics ?? "the metrics file"}
              </summary>
              <table className="mt-1 w-full">
                <tbody>
                  {sample.metrics.raw.map(([name, value], i) => (
                    <tr key={i} className="border-b border-slate-100">
                      <td className="px-1.5 py-0.5">{name}</td>
                      <td className="px-1.5 py-0.5 text-right tabular-nums">{value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          )}
        </>
      )}
    </div>
  );
}
