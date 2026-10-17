import { useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Loader2 } from "lucide-react";
import {
  getFilterPreview,
  type FilterGrouping,
  type FilterPreview,
  type FilterThresholds,
  type MadThresholds,
  type QcThresholdMode,
} from "@/api/qc";
import { useDatasetStore } from "@/stores/datasetStore";
import { qcGroupOptions } from "@/lib/qcGroups";
import { bound, pct, thresholdMode } from "@/lib/qcThresholds";

const DEBOUNCE_MS = 150;
const ROWS = 6; // per table, more behind "show all"
const WARNINGS = 4; // whole groups first, then the worst; more behind "show all"
const MAX_MAD_SAMPLES = 200; // backend qc_filter.MAX_SAMPLES

export const MODE_LABELS: Record<QcThresholdMode, string> = {
  mad: "Per-sample MAD",
  fixed: "Fixed cut-offs",
};

/** One table of groups (samples or clusters): most affected first. */
function LossTable({ title, grouping }: { title: string; grouping: FilterGrouping }) {
  const [all, setAll] = useState(false);
  if (grouping.note) return <p className="text-[11px] text-slate-400">{grouping.note}</p>;
  const rows = [...grouping.groups].sort((a, b) => b.fraction - a.fraction || b.n_cells - a.n_cells);
  const shown = all ? rows : rows.slice(0, ROWS);
  return (
    <div>
      <p className="mb-0.5 text-[11px] font-medium text-slate-600">
        {title} <span className="font-normal text-slate-400">({grouping.column})</span>
      </p>
      <table className="w-full text-[11px]">
        <thead>
          <tr className="text-left text-slate-400">
            <th className="font-medium">{grouping.column}</th>
            <th className="text-right font-medium">Cells</th>
            <th className="text-right font-medium">Removed</th>
            <th className="text-right font-medium">Share</th>
          </tr>
        </thead>
        <tbody className="text-slate-600">
          {shown.map((g) => (
            <tr key={g.name}>
              <td className="max-w-[10rem] truncate pr-2" title={g.name}>
                {g.name}
              </td>
              <td className="text-right">{g.n_cells.toLocaleString()}</td>
              <td className="text-right">{g.n_removed.toLocaleString()}</td>
              <td
                className={`text-right ${
                  g.fraction >= 1 ? "font-semibold text-red-700" : g.fraction > 0.5 ? "font-semibold text-amber-700" : ""
                }`}
              >
                {pct(g.fraction)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > ROWS && (
        <button type="button" onClick={() => setAll((v) => !v)} className="text-[11px] text-primary hover:underline">
          {all ? "Show fewer" : `Show all ${rows.length}`}
        </button>
      )}
    </div>
  );
}

/** A count range a sample keeps: "312–9,870"; a side without a bound reads "any". */
function keptRange(lo: number | null, hi: number | null): string {
  if (lo === null && hi === null) return "any";
  return `${lo === null ? "any" : bound(lo)}–${hi === null ? "any" : bound(hi)}`;
}

/** MAD mode: each sample's bounds and what they remove, in the samples' own order. */
export function SampleThresholds({ mad }: { mad: MadThresholds }) {
  const [all, setAll] = useState(false);
  if (!mad.column) {
    const p = mad.pooled;
    return (
      <p className="text-[11px] text-slate-600" data-testid="mad-pooled">
        No sample column found, so all cells are one group: kept with {keptRange(p.min_counts, p.max_counts)} counts
        and {keptRange(p.min_genes, p.max_genes)} genes per cell
        {p.max_pct_mt !== null ? `, under ${bound(p.max_pct_mt, 1)}% mito` : ""}.
      </p>
    );
  }
  const shown = all ? mad.samples : mad.samples.slice(0, ROWS);
  const hasMt = mad.samples.some((s) => s.max_pct_mt !== null);
  const skipped = (s: MadThresholds["samples"][number], metric: string) =>
    s.skipped.includes(metric) ? "Its MAD is 0 (most cells alike): no bound from it here." : undefined;
  return (
    <div>
      <p className="mb-0.5 text-[11px] font-medium text-slate-600">
        Thresholds per sample <span className="font-normal text-slate-400">({mad.column}; cells kept within)</span>
      </p>
      <table className="w-full text-[11px]" aria-label="Thresholds per sample">
        <thead>
          <tr className="text-left text-slate-400">
            <th className="font-medium">{mad.column}</th>
            <th className="text-right font-medium">Cells</th>
            <th className="text-right font-medium">Counts</th>
            <th className="text-right font-medium">Genes</th>
            {hasMt && <th className="text-right font-medium">Max % mito</th>}
            <th className="text-right font-medium">Removed</th>
          </tr>
        </thead>
        <tbody className="text-slate-600">
          {shown.map((s) => (
            <tr key={s.name}>
              <td
                className="max-w-[8rem] truncate pr-2"
                title={
                  s.pooled
                    ? `${s.name}: fewer than ${mad.min_sample_cells} cells, so judged against all cells pooled`
                    : s.name
                }
              >
                {s.name}
                {s.pooled && <span className="text-slate-400"> (pooled)</span>}
              </td>
              <td className="text-right">{s.n_cells.toLocaleString()}</td>
              <td className="text-right" title={skipped(s, "total_counts")}>
                {keptRange(s.min_counts, s.max_counts)}
              </td>
              <td className="text-right" title={skipped(s, "n_genes")}>
                {keptRange(s.min_genes, s.max_genes)}
              </td>
              {hasMt && (
                <td className="text-right" title={skipped(s, "pct_mt")}>
                  {s.max_pct_mt === null ? "any" : `< ${bound(s.max_pct_mt, 1)}`}
                </td>
              )}
              <td className="text-right">
                {s.n_removed === undefined
                  ? "—"
                  : `${s.n_removed.toLocaleString()} (${pct(s.n_cells ? s.n_removed / s.n_cells : 0)})`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {mad.samples.length > ROWS && (
        <button type="button" onClick={() => setAll((v) => !v)} className="text-[11px] text-primary hover:underline">
          {all ? "Show fewer" : `Show all ${mad.samples.length}`}
        </button>
      )}
      {mad.n_unlabelled > 0 && (
        <p className="text-[11px] text-slate-400">
          {mad.n_unlabelled.toLocaleString()} cells without a {mad.column} value are judged against all cells pooled.
        </p>
      )}
    </div>
  );
}

/** MAD vs fixed cut-offs: one choice, the step's `qc_threshold_mode`. */
function ModeSwitch({ mode, onChange }: { mode: QcThresholdMode; onChange: (m: QcThresholdMode) => void }) {
  return (
    <div role="group" aria-label="Threshold mode" className="inline-flex rounded-md border border-slate-200 p-0.5">
      {(["mad", "fixed"] as const).map((m) => (
        <button
          key={m}
          type="button"
          aria-pressed={mode === m}
          onClick={() => onChange(m)}
          className={`rounded px-2 py-0.5 text-[11px] font-medium ${
            mode === m ? "bg-primary text-white" : "text-slate-600 hover:bg-slate-100"
          }`}
        >
          {MODE_LABELS[m]}
        </button>
      ))}
    </div>
  );
}

/** A small number input that commits a positive number (anything else is ignored). */
function KInput({ label, value, onCommit }: { label: string; value: number; onCommit: (v: number) => void }) {
  return (
    <input
      type="number"
      min={0.5}
      step={0.5}
      value={value}
      aria-label={label}
      onChange={(e) => {
        const v = Number(e.target.value);
        if (e.target.value.trim() !== "" && isFinite(v) && v > 0) onCommit(v);
      }}
      className="mx-0.5 w-12 rounded border border-slate-200 px-1 py-0 text-[11px] text-slate-700"
    />
  );
}

/** What MAD mode does, with its multipliers and the samples it judges by. */
function MadControls({
  thresholds,
  preview,
  onChange,
}: {
  thresholds: FilterThresholds;
  preview: FilterPreview | null;
  onChange?: (patch: Partial<FilterThresholds>) => void;
}) {
  const obsColumns = useDatasetStore((s) => s.currentDataset?.obs_columns);
  const options = useMemo(() => qcGroupOptions(obsColumns, MAX_MAD_SAMPLES), [obsColumns]);
  const k = thresholds.qc_mad_k ?? 5;
  const kMt = thresholds.qc_mad_k_mt ?? 3;
  const chosen = thresholds.qc_sample_column ?? "";
  const detected = !chosen && preview?.mad ? preview.mad.column : null;
  const mito = preview?.has_mt !== false;
  return (
    <div className="space-y-1 text-[11px] leading-relaxed text-slate-600" data-testid="mad-controls">
      <p>
        Removes each sample&apos;s outliers: counts and genes per cell outside the median ±
        {onChange ? <KInput label="MADs for counts and genes" value={k} onCommit={(v) => onChange({ qc_mad_k: v })} /> : ` ${k} `}
        MADs (log scale)
        {mito && (
          <>
            , % mito above the median +
            {onChange ? (
              <KInput label="MADs for % mito" value={kMt} onCommit={(v) => onChange({ qc_mad_k_mt: v })} />
            ) : (
              ` ${kMt} `
            )}
            MADs, and never {bound(thresholds.max_pct_mt, 0)}% or more (the ceiling on the % mito plot)
          </>
        )}
        .
      </p>
      {onChange && (
        <label className="flex flex-wrap items-center gap-1.5">
          <span className="font-medium text-slate-500">Samples</span>
          <select
            aria-label="Sample column for MAD thresholds"
            value={chosen}
            onChange={(e) => onChange({ qc_sample_column: e.target.value })}
            className="min-w-0 max-w-[60%] rounded border border-slate-200 bg-white px-1.5 py-0.5 text-[11px] text-slate-600"
          >
            <option value="">
              {detected ? `Detected: ${detected}` : preview?.mad ? "None found (all cells pooled)" : "Detect"}
            </option>
            {options.map((o) => (
              <option key={o.name} value={o.name} disabled={o.disabled}>
                {o.name} ({o.n})
              </option>
            ))}
          </select>
        </label>
      )}
    </div>
  );
}

/** A dataset scView filtered before: what a new run does with it. */
function PreviousNote({ preview, mode }: { preview: FilterPreview; mode: QcThresholdMode }) {
  const prev = preview.previous;
  if (!prev) return null;
  const when = prev.timestamp ? ` on ${prev.timestamp.slice(0, 10)}` : "";
  const reused = mode === "mad" && preview.mad?.reused_from;
  return (
    <p className="rounded-md bg-slate-50 px-2 py-1 text-[11px] text-slate-600" data-testid="qc-previous">
      Filtered before by scView{when} ({prev.mode === "mad" ? "per-sample MAD" : "fixed cut-offs"}).{" "}
      {reused
        ? "With the same settings its per-sample thresholds are used again, so filtering again removes no more cells; other settings work them out afresh from the cells left."
        : mode === "mad"
          ? "MAD thresholds now come from the cells left, whose spread is narrower, so this pass trims the tails a little further."
          : ""}
    </p>
  );
}

/**
 * What the QC thresholds set on the histograms would remove, asked of the backend
 * on every change (debounced): overall, by each threshold on its own, per sample and
 * per cluster, with a warning for a sample or cluster removed whole or more than half.
 * In MAD mode (the default for new runs) each sample's own thresholds are listed.
 */
export function QcFilterSummary({
  datasetId,
  thresholds,
  refreshKey = 0,
  onChange,
}: {
  datasetId: string;
  thresholds: FilterThresholds;
  refreshKey?: number;
  /** Change the filtering step's params (mode, multipliers, samples, doublets). */
  onChange?: (patch: Partial<FilterThresholds>) => void;
}) {
  const [preview, setPreview] = useState<FilterPreview | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [allWarnings, setAllWarnings] = useState(false);
  const request = useRef(0);
  const key = JSON.stringify(thresholds);
  const mode = thresholdMode(thresholds);

  useEffect(() => {
    const id = ++request.current;
    setLoading(true);
    const timer = setTimeout(() => {
      getFilterPreview(datasetId, JSON.parse(key) as FilterThresholds)
        .then((p) => {
          if (id !== request.current) return; // a newer threshold was asked for
          setPreview(p);
          setError(null);
        })
        .catch((e) => {
          if (id === request.current) setError(String(e?.message ?? e));
        })
        .finally(() => {
          if (id === request.current) setLoading(false);
        });
    }, DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [datasetId, key, refreshKey]);

  const header = (
    <div className="flex flex-wrap items-center justify-between gap-2">
      <h4 className="text-xs font-semibold text-slate-700">Cells these thresholds remove</h4>
      <span className="flex items-center gap-2">
        {loading && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
        {onChange && <ModeSwitch mode={mode} onChange={(m) => onChange({ qc_threshold_mode: m })} />}
      </span>
    </div>
  );

  if (!preview) {
    return error ? (
      <div className="space-y-2 rounded-xl border border-slate-200 bg-white p-3">
        {header}
        <p role="alert" className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
          Cell-loss summary unavailable: {error}
        </p>
      </div>
    ) : null;
  }

  const parts = Object.values(preview.by_criterion).map(
    (c) => `${c!.label.toLowerCase()} ${c!.n_removed.toLocaleString()}`,
  );
  const mad = mode === "mad" ? (preview.mad ?? null) : null;
  return (
    <div className="space-y-2 rounded-xl border border-slate-200 bg-white p-3" data-testid="qc-loss-summary">
      {header}
      {mode === "mad" && <MadControls thresholds={thresholds} preview={preview} onChange={onChange} />}
      {error && (
        <p role="alert" className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
          {error}
        </p>
      )}
      <p className="text-[11px] text-slate-700">
        <span className="font-semibold">
          {preview.n_removed.toLocaleString()} of {preview.n_cells.toLocaleString()} cells (
          {pct(preview.fraction_removed)})
        </span>
        {parts.length > 0 ? `; each threshold on its own: ${parts.join(", ")}` : ": no threshold is on"}.
      </p>
      <PreviousNote preview={preview} mode={mode} />
      {preview.warnings.length > 0 && (
        <ul className="space-y-1" aria-label="Cell-loss warnings">
          {(allWarnings ? preview.warnings : preview.warnings.slice(0, WARNINGS)).map((w) => (
            <li
              key={`${w.kind}:${w.group}`}
              className={`flex items-start gap-1.5 rounded-md px-2 py-1 text-[11px] ${
                w.whole ? "bg-red-50 text-red-800" : "bg-amber-50 text-amber-800"
              }`}
            >
              <AlertTriangle className="mt-0.5 h-3 w-3 flex-shrink-0" />
              <span>{w.message}</span>
            </li>
          ))}
        </ul>
      )}
      {preview.warnings.length > WARNINGS && (
        <button
          type="button"
          onClick={() => setAllWarnings((v) => !v)}
          className="text-[11px] text-primary hover:underline"
        >
          {allWarnings ? "Fewer warnings" : `Show all ${preview.warnings.length} warnings`}
        </button>
      )}
      {mad && <SampleThresholds mad={mad} />}
      <div className="grid gap-3 sm:grid-cols-2">
        {preview.samples && !mad?.column && <LossTable title="Per sample" grouping={preview.samples} />}
        {preview.clusters && <LossTable title="Per cluster" grouping={preview.clusters} />}
      </div>
      {preview.doublet_calls && onChange && (
        <label className="flex items-center gap-1.5 text-[11px] text-slate-600">
          <input
            type="checkbox"
            checked={thresholds.drop_doublets}
            onChange={(e) => onChange({ drop_doublets: e.target.checked })}
            className="h-3 w-3"
          />
          Also drop predicted doublets
        </label>
      )}
      <p className="text-[11px] text-slate-400">
        {preview.has_mt ? "" : "% mito can't be measured here, so no mito threshold applies. "}
        {mode === "mad" ? "The fixed genes and counts cut-offs are not used in MAD mode. " : ""}
        These are the Cell/Gene Filtering step&apos;s parameters: they apply when that step runs (tick it,
        or Reprocess from Scratch).
      </p>
    </div>
  );
}
