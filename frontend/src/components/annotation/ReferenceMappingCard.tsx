import { useEffect, useRef, useState } from "react";
import { Loader2, Map as MapIcon } from "lucide-react";
import {
  checkReference,
  getLabelTransfer,
  runLabelTransfer,
  type LabelTransferInfo,
  type LabelTransferMethod,
  type LabelTransferSummary,
  type ReferenceCheck,
} from "@/api/labelTransfer";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { aboutGb, METHOD_LABELS, referenceChoices, unmappedLevel } from "@/lib/labelTransfer";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";
import { errorMessage } from "@/lib/errorMessage";

const pct = (v: number) => `${(100 * v).toFixed(v < 0.1 ? 1 : 0)}%`;

const colourBy = (column: string) => {
  useSettingsStore.getState().setColorBy(column);
  useViewStore.getState().setPanel("unified");
};

/** A shift of the query from the reference this large gets a note (Kang vs Kang: 1.0, another study: 2.1). */
const SHIFT_NOTE = 1.3;

/**
 * Reference mapping (C042): label this dataset's cells from another library dataset's
 * cell-type column, and flag cells the reference does not contain.
 */
export function ReferenceMappingCard({
  datasetId,
  onCrosscheck,
}: {
  datasetId: string;
  /** Open the annotation cross-check on these columns */
  onCrosscheck?: (columns: string[]) => void;
}) {
  const datasets = useDatasetStore((s) => s.availableDatasets);
  const refresh = useDatasetRefresh();
  const [info, setInfo] = useState<LabelTransferInfo | null>(null);
  const [refId, setRefId] = useState("");
  const [check, setCheck] = useState<ReferenceCheck | null>(null);
  const [column, setColumn] = useState("");
  const [method, setMethod] = useState<LabelTransferMethod>("projection");
  const [k, setK] = useState(15);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState("");
  const [result, setResult] = useState<LabelTransferSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    setInfo(null);
    setResult(null);
    setRefId("");
    setError(null);
    let live = true;
    getLabelTransfer(datasetId)
      .then((i) => {
        if (!live || !i?.methods) return;
        setInfo(i);
        setK(i.k);
        setResult(i.saved[i.saved.length - 1] ?? null);
      })
      .catch(() => {}); // an older backend without the route: the card stays hidden
    return () => {
      live = false;
      abort.current?.abort();
    };
  }, [datasetId]);

  useEffect(() => {
    setCheck(null);
    setColumn("");
    if (!refId) return;
    let live = true;
    checkReference(datasetId, refId, method)
      .then((c) => {
        if (!live) return;
        setCheck(c);
        const cols = c.label_columns ?? [];
        const guess = cols.find((x) => /cell[ ._-]?type|label|annot/i.test(x.column)) ?? cols[0];
        setColumn(guess?.column ?? "");
      })
      .catch((e) => live && setCheck({ ok: false, reason: errorMessage(e) }));
    return () => {
      live = false;
    };
  }, [datasetId, refId, method]);

  if (!info) return null;
  const choices = referenceChoices(datasets, datasetId);

  const run = async () => {
    setRunning(true);
    setError(null);
    setProgress("Starting");
    abort.current = new AbortController();
    try {
      const r = await runLabelTransfer(
        datasetId,
        { reference_id: refId, label_column: column, method, k },
        { onProgress: setProgress, signal: abort.current.signal },
      );
      setResult(r);
      await refresh(datasetId, r.columns.label);
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) setError(errorMessage(e));
    } finally {
      setRunning(false);
      setProgress("");
    }
  };

  const fewGenes =
    check?.shared_genes_by_name != null && check.min_shared_genes != null && check.shared_genes_by_name < check.min_shared_genes;
  const maxHist = result ? Math.max(1, ...result.score_hist.counts) : 1;

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4" data-testid="reference-mapping">
      <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-800">
        <MapIcon className="h-4 w-4" /> Label from a reference dataset
        <InfoTip width={320}>
          Pick another dataset in your library whose cells are already annotated. Each cell here takes the labels of its
          nearest reference cells, and cells unlike any reference cell are flagged &quot;not in reference&quot;.
        </InfoTip>
      </div>
      {!info.available ? (
        <p className="text-xs text-slate-500">{info.reason}</p>
      ) : (
        <div className="space-y-2 text-xs">
          <div className="flex flex-wrap items-center gap-2">
            <label className="flex items-center gap-1">
              Reference
              <select
                className="rounded border px-1 py-0.5"
                value={refId}
                disabled={running}
                onChange={(e) => setRefId(e.target.value)}
                aria-label="Reference dataset"
              >
                <option value="">choose a dataset</option>
                {choices.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
            </label>
            {check?.label_columns && check.label_columns.length > 0 && (
              <label className="flex items-center gap-1">
                Labels
                <select
                  className="rounded border px-1 py-0.5"
                  value={column}
                  disabled={running}
                  onChange={(e) => setColumn(e.target.value)}
                  aria-label="Reference label column"
                >
                  {check.label_columns.map((c) => (
                    <option key={c.column} value={c.column}>
                      {c.column} ({c.n_labels})
                    </option>
                  ))}
                </select>
              </label>
            )}
            <label className="flex items-center gap-1">
              Method
              <select
                className="rounded border px-1 py-0.5"
                value={method}
                disabled={running}
                onChange={(e) => setMethod(e.target.value as LabelTransferMethod)}
                aria-label="Mapping method"
              >
                {info.methods.map((m) => (
                  <option key={m} value={m}>
                    {METHOD_LABELS[m]}
                  </option>
                ))}
              </select>
              <InfoTip width={340}>
                Project onto reference (like scanpy&apos;s ingest): the reference is left as it is and this dataset is placed
                in its PCA. It assumes no batch effect between the two: use it for data from the same lab and protocol.
                Integrate first: both are corrected together with Harmony, for a reference from another lab or protocol;
                it can also blend a state found only here into its nearest reference type. Train CellTypist: a classifier
                trained on the reference (about a minute).
              </InfoTip>
            </label>
            <label className="flex items-center gap-1">
              Neighbours
              <input
                type="number"
                className="w-14 rounded border px-1 py-0.5"
                min={info.min_k}
                max={info.max_k}
                value={k}
                disabled={running}
                onChange={(e) => setK(Number(e.target.value))}
                aria-label="Neighbours"
              />
            </label>
          </div>
          {check && !check.ok && <p className="text-amber-700">{check.reason}</p>}
          {check?.ok && (
            <p className="text-slate-500">
              {check.n_cells?.toLocaleString()} reference cells, {check.shared_genes_by_name?.toLocaleString()} genes shared
              by name{check.has_umap ? "; cells here are also placed on its UMAP" : ""}.
              {check.estimate &&
                ` Needs ${aboutGb(check.estimate.peak_bytes)}, ${check.estimate.seconds_range[0]}-${check.estimate.seconds_range[1]} s.`}
              {fewGenes && " Few genes match by name; Ensembl ids are matched to symbols when a gene-id column exists."}
              {check.label_columns?.length === 0 && " It has no label column (2-200 categories)."}
            </p>
          )}
          <div className="flex items-center gap-2">
            <button
              className="rounded bg-primary px-3 py-1 text-white disabled:opacity-50"
              disabled={running || !refId || !column || !check?.ok}
              onClick={run}
            >
              {running ? <Loader2 className="inline h-3 w-3 animate-spin" /> : null} Map cells
            </button>
            {running && (
              <>
                <span className="text-slate-600">{progress}</span>
                <button className="text-slate-500 hover:underline" onClick={() => abort.current?.abort()}>
                  Cancel
                </button>
              </>
            )}
          </div>
          {error && <p className="text-red-600">{error}</p>}
          {result && !running && (
            <div className="space-y-2 border-t pt-2" data-testid="reference-mapping-result">
              <p className="text-slate-700">
                {result.labels.length} labels from <b>{result.reference.name}</b> ({result.reference.label_column},{" "}
                {result.method_text}); median score {result.median_score.toFixed(2)}; <b>{pct(result.unmapped_share)}</b>{" "}
                not in reference
                <InfoTip width={320}>{result.note}</InfoTip>
              </p>
              {(result.shift ?? 1) >= SHIFT_NOTE && (
                <p className="text-amber-700" data-testid="reference-mapping-shift">
                  This dataset sits {result.shift!.toFixed(1)}x farther from the reference than its own cells do (another
                  study, chemistry or depth). The &quot;not in reference&quot; cut was scaled to match, so it flags cells
                  unlike the rest of this dataset, not the shift itself.
                </p>
              )}
              <div className="flex flex-wrap gap-2">
                {[result.columns.label, result.columns.score, result.columns.unmapped].map((c) => (
                  <button key={c} className="rounded border px-2 py-0.5 hover:bg-slate-50" onClick={() => colourBy(c)}>
                    Colour by {c}
                  </button>
                ))}
                {onCrosscheck && result.cluster_column && (
                  <button
                    className="rounded border px-2 py-0.5 hover:bg-slate-50"
                    onClick={() => onCrosscheck([result.columns.label, result.cluster_column!])}
                  >
                    Cross-check against clusters
                  </button>
                )}
              </div>
              <div>
                <div className="flex h-12 items-end gap-px" aria-label="Score histogram">
                  {result.score_hist.counts.map((c, i) => (
                    <div
                      key={i}
                      className="flex-1 bg-sky-400"
                      style={{ height: `${(100 * c) / maxHist}%` }}
                      title={`${result.score_hist.edges[i]}-${result.score_hist.edges[i + 1]}: ${c} cells`}
                    />
                  ))}
                </div>
                <Caption n={{ value: result.n_cells, unit: "cells" }}>
                  Score: share of a cell&apos;s {result.k} nearest reference cells (weighted by distance) carrying its label, 0
                  to 1.
                </Caption>
              </div>
              {result.clusters.length > 0 && (
                <table className="w-full">
                  <thead>
                    <tr className="text-left text-slate-500">
                      <th title={`Clusters of ${result.cluster_column}`}>Cluster</th>
                      <th className="text-right">Cells</th>
                      <th className="text-right" title="Share of the cluster's cells not in the reference">
                        Not in ref.
                      </th>
                      <th className="pl-3 text-left" title="The cluster's commonest transferred label">Label</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.clusters.map((c) => {
                      const level = unmappedLevel(c);
                      return (
                        <tr
                          key={c.cluster}
                          className={level === "candidate" ? "bg-red-50" : level === "elevated" ? "bg-amber-50" : ""}
                        >
                          <td>
                            {c.cluster}
                            {level === "candidate" && <span className="ml-1 text-red-700">absent from reference?</span>}
                          </td>
                          <td className="text-right">{c.n_cells.toLocaleString()}</td>
                          <td className="text-right">{pct(c.unmapped_share)}</td>
                          <td className="pl-3">
                            {c.top_label} ({pct(c.top_label_share)})
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
