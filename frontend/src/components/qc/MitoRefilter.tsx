import { useState } from "react";
import { AlertTriangle, Loader2, RotateCcw } from "lucide-react";
import { getDataset } from "@/api/datasets";
import { getMitoRefilter, runMitoRefilter, type RefilterPreview, type RefilterType } from "@/api/mitoFilter";
import { Caption } from "@/components/common/Caption";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";

const STEP_LABELS: Record<string, string> = {
  qc_metrics: "QC metrics",
  doublet_detection: "doublet detection",
  filtering: "filtering",
  normalization: "normalization",
  log_transform: "log transform",
  highly_variable_genes: "variable genes",
  scaling: "scaling",
  pca: "PCA",
  batch_correction: "batch correction",
  neighbors: "neighbour graph",
  clustering: "clustering",
  embeddings: "UMAP",
  marker_genes: "marker genes",
  enrichment: "enrichment",
  cell_cycle: "cell cycle",
  dissociation_stress: "dissociation stress",
};

function pct(v: number | null | undefined): string {
  return v === null || v === undefined || !isFinite(v) ? "—" : `${v.toFixed(1)}%`;
}

/** How the cells filtering removed got a type, in one sentence. */
export function typedText(p: RefilterPreview): string {
  const t = p.typed;
  const parts: string[] = [];
  if (t.column) parts.push(`${t.column.toLocaleString()} from the original upload's '${p.column}'`);
  if (t.expression) {
    const a = p.assignment;
    parts.push(
      `${t.expression.toLocaleString()} by expression (the '${p.column}' group whose mean profile is most ` +
        `correlated with the cell's own, over ${a?.genes.toLocaleString() ?? "?"} variable genes without MT- genes` +
        (a?.median_r !== undefined && a?.median_r !== null ? `; median r ${a.median_r.toFixed(2)}` : "") +
        (a?.n_low ? `; ${a.n_low.toLocaleString()} with r under 0.2` : "") +
        ")",
    );
  }
  const head = `The ${p.removed_before.toLocaleString()} cells the last filtering removed were typed: `;
  const none = t.none ? ` ${t.none.toLocaleString()} cells have no type and keep the global rule.` : "";
  return parts.length ? `${head}${parts.join("; ")}.${none}` : `Nothing was removed before.${none}`;
}

function Row({ t }: { t: RefilterType }) {
  return (
    <tr className="border-t border-slate-100" data-testid={`refilter-row-${t.name}`}>
      <td className="max-w-[9rem] truncate px-1.5 py-0.5" title={[t.note, t.warning].filter(Boolean).join(" ") || t.name}>
        {t.name}
      </td>
      <td className="px-1.5 py-0.5 text-right tabular-nums">{t.n_all.toLocaleString()}</td>
      <td className="px-1.5 py-0.5 text-right tabular-nums">{pct(t.median)}</td>
      <td className="px-1.5 py-0.5 text-right tabular-nums" title={t.note || undefined}>
        {t.bound !== null && t.bound !== undefined ? pct(t.bound) : <span className="text-slate-400">global</span>}
      </td>
      <td className="px-1.5 py-0.5 text-right tabular-nums">{t.removed_before.toLocaleString()}</td>
      <td className="px-1.5 py-0.5 text-right tabular-nums">{t.removed_after.toLocaleString()}</td>
      <td className={`px-1.5 py-0.5 text-right tabular-nums ${t.brought_back ? "font-medium text-emerald-700" : ""}`}>
        {t.brought_back.toLocaleString()}
      </td>
      <td className={`px-1.5 py-0.5 text-right tabular-nums ${t.newly_removed ? "font-medium text-amber-700" : ""}`}>
        {t.newly_removed.toLocaleString()}
      </td>
    </tr>
  );
}

/** The preview: per type what changes, how removed cells were typed, what it can and cannot bring back. */
export function RefilterPreviewView({ p }: { p: RefilterPreview }) {
  return (
    <div className="space-y-1.5 text-[11px] text-slate-600" data-testid="refilter-preview">
      <p className="text-slate-700">
        <span className="font-semibold">
          {p.n_after.toLocaleString()} of {p.n_original.toLocaleString()} cells kept
        </span>{" "}
        (now {p.n_current.toLocaleString()}): {p.brought_back.toLocaleString()} brought back,{" "}
        {p.newly_removed.toLocaleString()} newly removed.
      </p>
      <p>{typedText(p)}</p>
      <div className="max-h-56 overflow-auto rounded border border-slate-100">
        <table className="w-full text-[11px]">
          <thead className="sticky top-0 bg-slate-50 text-left text-slate-500">
            <tr>
              <th className="px-1.5 py-0.5 font-medium" title={`Groups of '${p.column}'`}>Type</th>
              <th className="px-1.5 py-0.5 text-right font-medium" title="Cells of the type in the original upload">Cells</th>
              <th className="px-1.5 py-0.5 text-right font-medium" title="Median % mito over all the type's cells, before any filtering">Median</th>
              <th className="px-1.5 py-0.5 text-right font-medium" title="Its own bound (median + 3 MADs, at most 50%), or the global rule when it has none">Bound</th>
              <th className="px-1.5 py-0.5 text-right font-medium" title="Cells of the type the last filtering removed (any rule)">Removed now</th>
              <th className="px-1.5 py-0.5 text-right font-medium" title="Cells of the type the re-filter removes (any rule)">After</th>
              <th className="px-1.5 py-0.5 text-right font-medium" title="Removed now, kept after">Back</th>
              <th className="px-1.5 py-0.5 text-right font-medium" title="Kept now, removed after (a bound below the global rule)">New</th>
            </tr>
          </thead>
          <tbody>
            {p.types.map((t) => (
              <Row key={t.name} t={t} />
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-emerald-800">{p.can}</p>
      <p>{p.cannot}</p>
      {p.warnings.map((w) => (
        <p key={w} className="flex items-start gap-1 rounded bg-amber-50 px-1.5 py-0.5 text-amber-800">
          <AlertTriangle className="mt-0.5 h-3 w-3 flex-shrink-0" />
          {w}
        </p>
      ))}
      <Caption n={[{ value: p.n_original, unit: "cells" }, { value: p.types.length, unit: `types of '${p.column}'` }]}>
        Starts again from the original upload ({p.source_file}) and re-runs {p.steps.length} step
        {p.steps.length === 1 ? "" : "s"}: {p.steps.map((s) => STEP_LABELS[s] ?? s).join(", ")}, with the settings the
        dataset was processed with and % mito judged per type.
        {p.notes.length ? ` ${p.notes.join(" ")}` : ""}
      </Caption>
    </div>
  );
}

/**
 * "Re-filter with per-cluster bounds": preview what re-filtering from the original
 * upload with each type's own % mito bound would do, then run it (a new filtering
 * and the dataset's steps again), with progress.
 */
export function MitoRefilter({
  datasetId,
  column,
  perCluster,
  onDone,
}: {
  datasetId: string;
  column: string;
  /** The column is a clustering (the label says "per-cluster"). */
  perCluster: boolean;
  /** After the new dataset was saved and the library entry refreshed (views reload). */
  onDone?: () => void;
}) {
  const setPipelineRunning = useSettingsStore((s) => s.setPipelineRunning);
  const [preview, setPreview] = useState<RefilterPreview | null>(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);
  const label = `Re-filter with per-${perCluster ? "cluster" : "type"} bounds`;

  const ask = () => {
    setLoading(true);
    setError(null);
    setDone(null);
    getMitoRefilter(datasetId, column)
      .then(setPreview)
      .catch((e) => setError(String(e?.message ?? e)))
      .finally(() => setLoading(false));
  };

  const run = async () => {
    if (!preview) return;
    setRunning(true);
    setPipelineRunning(true);
    setError(null);
    let total = preview.steps.length;
    try {
      await runMitoRefilter(datasetId, column, ({ event, data }) => {
        if (event === "refilter") setProgress(String(data.message ?? ""));
        else if (event === "step_start") {
          total = Number(data.total ?? total);
          const step = String(data.step);
          setProgress(`Step ${Number(data.index) + 1}/${total}: ${STEP_LABELS[step] ?? step}`);
        } else if (event === "substep") setProgress(String(data.message ?? ""));
      });
      setDone(`Re-filtered: ${preview.n_after.toLocaleString()} cells kept (was ${preview.n_current.toLocaleString()}).`);
      setPreview(null);
      try {
        useDatasetStore.getState().setCurrentDataset(await getDataset(datasetId));
      } catch {
        /* best-effort: the library refreshes on its own */
      }
      onDone?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
      setPipelineRunning(false);
      setProgress(null);
    }
  };

  return (
    <div className="space-y-1.5 rounded-md border border-amber-200 bg-amber-50/40 p-2" data-testid="mito-refilter">
      {!preview && (
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={ask}
            disabled={loading || running}
            className="inline-flex items-center gap-1 rounded bg-amber-100 px-2 py-1 text-[11px] font-medium text-amber-800 hover:bg-amber-200 disabled:opacity-50"
            title="Preview first: starts again from the original upload, with each type's own % mito bound"
          >
            {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <RotateCcw className="h-3 w-3" />}
            {label}
          </button>
          <span className="text-[11px] text-slate-500">
            {loading ? "Reading the original upload…" : "Shows what it would change before anything runs."}
          </span>
        </div>
      )}
      {preview && (
        <>
          <RefilterPreviewView p={preview} />
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={run}
              disabled={running}
              className="inline-flex items-center gap-1 rounded bg-amber-600 px-2 py-1 text-[11px] font-medium text-white hover:bg-amber-700 disabled:opacity-50"
            >
              {running ? <Loader2 className="h-3 w-3 animate-spin" /> : <RotateCcw className="h-3 w-3" />}
              {running ? "Re-filtering…" : `Re-filter (${preview.n_after.toLocaleString()} cells)`}
            </button>
            {!running && (
              <button type="button" onClick={() => setPreview(null)} className="text-[11px] text-slate-500 hover:underline">
                Cancel
              </button>
            )}
            {progress && <span className="text-[11px] text-blue-700">{progress}</span>}
          </div>
        </>
      )}
      {done && <p className="text-[11px] text-emerald-700">{done}</p>}
      {error && (
        <p role="alert" className="rounded bg-red-50 px-2 py-1 text-[11px] text-red-700">
          {error}
        </p>
      )}
    </div>
  );
}
