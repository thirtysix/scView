import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertCircle, AlertTriangle, CheckCircle2, Circle, Layers, Loader2, XCircle } from "lucide-react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";
import { getDataset, listDatasets } from "@/api/datasets";
import {
  planIntegration,
  runIntegration,
  type IntegrationEvent,
  type IntegrationMethod,
  type IntegrationPlan,
} from "@/api/integration";
import { formatNumber, prettyObsLabel } from "@/lib/formatting";
import type { DatasetInfo } from "@/api/types";
import { JointSummary } from "./JointSummary";

/** Pipeline steps of a joint run, in order, with display labels. */
const STEP_LABELS: Record<string, string> = {
  qc_metrics: "QC metrics",
  normalization: "Normalize",
  log_transform: "Log-transform",
  highly_variable_genes: "Batch-aware variable genes",
  scaling: "Scale",
  pca: "PCA",
  uncorrected_embedding: "Uncorrected UMAP (for comparison)",
  batch_correction: "Harmony integration",
  neighbors: "Neighbor graph",
  clustering: "Joint clustering",
  embeddings: "Joint UMAP",
  marker_genes: "Marker genes",
};

type StepStatus = "pending" | "running" | "done" | "error";

interface Progress {
  stage: string;
  steps: string[];
  status: Record<string, StepStatus>;
  errors: Record<string, string>;
  substep: string | null;
}

/**
 * Build a joint dataset from 2+ library datasets: their raw counts are merged
 * (shared genes), labelled by a `dataset` column, and processed together with
 * Harmony so all of them share one UMAP, one clustering and one marker set.
 */
export function JointEmbedding() {
  const current = useDatasetStore((s) => s.currentDataset);
  const setCurrentDataset = useDatasetStore((s) => s.setCurrentDataset);
  const setAvailableDatasets = useDatasetStore((s) => s.setAvailableDatasets);
  const setPanel = useViewStore((s) => s.setPanel);

  const [all, setAll] = useState<DatasetInfo[]>([]);
  const [listing, setListing] = useState(true);
  const [selected, setSelected] = useState<string[]>(current ? [current.id] : []);
  const [labels, setLabels] = useState<Record<string, string>>({});
  const [name, setName] = useState("");
  const [method, setMethod] = useState<IntegrationMethod>("harmony");
  const [resolution, setResolution] = useState(0.5);

  const [plan, setPlan] = useState<IntegrationPlan | null>(null);
  const [planning, setPlanning] = useState(false);
  const [planError, setPlanError] = useState<string | null>(null);

  const [progress, setProgress] = useState<Progress | null>(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [resultId, setResultId] = useState<string | null>(null);

  useEffect(() => {
    listDatasets()
      .then((ds) => setAll(ds.filter((d) => d.status === "ready")))
      .catch(() => setAll([]))
      .finally(() => setListing(false));
  }, []);

  const byId = useMemo(() => new Map(all.map((d) => [d.id, d])), [all]);
  const labelFor = useCallback(
    (id: string) => (labels[id] ?? byId.get(id)?.name ?? id).trim(),
    [labels, byId],
  );
  const labelList = useMemo(() => selected.map(labelFor), [selected, labelFor]);
  const labelKey = labelList.join("\u0000");

  // Re-check compatibility whenever the selection (or a label) settles.
  useEffect(() => {
    setPlan(null);
    setPlanError(null);
    if (selected.length < 2) return;
    let cancelled = false;
    const t = setTimeout(() => {
      setPlanning(true);
      planIntegration(selected, labelList)
        .then((p) => !cancelled && setPlan(p))
        .catch((e) => !cancelled && setPlanError(e instanceof Error ? e.message : String(e)))
        .finally(() => !cancelled && setPlanning(false));
    }, 400);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
    // labelKey stands in for labelList (a new array every render).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, labelKey]);

  const toggle = (id: string) =>
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));

  const defaultName = labelList.length >= 2 ? `joint: ${labelList.join(" + ")}` : "";

  const onEvent = useCallback((e: IntegrationEvent) => {
    setProgress((p) => {
      if (!p) return p;
      const status = { ...p.status };
      const step = String(e.data.step ?? "");
      switch (e.event) {
        case "stage":
          return { ...p, stage: String(e.data.message ?? "") };
        case "created":
          return { ...p, stage: "Processing the joint dataset" };
        case "step_start":
          status[step] = "running";
          return { ...p, status, substep: null };
        case "step_done":
          status[step] = "done";
          return { ...p, status, substep: null };
        case "step_error":
          status[step] = "error";
          return { ...p, status, substep: null, errors: { ...p.errors, [step]: String(e.data.error) } };
        case "substep":
          return { ...p, substep: String(e.data.message ?? "") };
        case "complete":
          return { ...p, stage: "Done" };
        default:
          return p;
      }
    });
  }, []);

  const build = async () => {
    if (!plan?.ok) return;
    const steps = Object.keys(STEP_LABELS).filter(
      (s) => method === "harmony" || (s !== "batch_correction" && s !== "uncorrected_embedding"),
    );
    setRunning(true);
    setRunError(null);
    setResultId(null);
    setProgress({ stage: "Starting", steps, status: {}, errors: {}, substep: null });
    try {
      const id = await runIntegration(
        {
          dataset_ids: selected,
          labels: labelList,
          name: name.trim() || defaultName,
          method,
          params: { clustering_resolution: resolution },
        },
        onEvent,
      );
      setResultId(id);
      listDatasets()
        .then(setAvailableDatasets)
        .catch(() => undefined);
    } catch (e) {
      setRunError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  const openResult = async () => {
    if (!resultId) return;
    const ds = await getDataset(resultId);
    setCurrentDataset(ds);
    useSettingsStore.getState().setEmbedding("X_umap");
    useSettingsStore.getState().setColorBy("dataset");
    setPanel("unified");
  };

  const m = plan?.merge;
  const sharedGenes = m ? (m.reconciliation?.feasible ? m.reconciliation.overlap_after : m.intersection) : 0;

  return (
    <div className="space-y-4">
      <p className="text-sm text-slate-500">
        Merge datasets on their shared genes and process them together, so every cell sits in one
        UMAP with one clustering. Harmony removes the dataset-to-dataset batch effect; a
        <code className="mx-1 rounded bg-slate-100 px-1 text-xs">dataset</code>column records
        where each cell came from. The source datasets are left untouched.
      </p>

      {current && current.id !== resultId && (
        <JointSummary datasetId={current.id} title={`Open joint dataset: ${current.name}`} />
      )}

      {/* Dataset picker */}
      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <h3 className="mb-2 text-sm font-semibold text-slate-700">Datasets to integrate</h3>
        {listing ? (
          <p className="flex items-center gap-2 text-sm text-slate-400">
            <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading datasets…
          </p>
        ) : (
          all.length === 0 && <p className="text-sm text-slate-400">No ready datasets in the library.</p>
        )}
        <div className="space-y-1.5">
          {all.map((d) => {
            const on = selected.includes(d.id);
            return (
              <div
                key={d.id}
                className={`flex flex-wrap items-center gap-3 rounded-lg border px-3 py-2 ${
                  on ? "border-primary/40 bg-primary/5" : "border-slate-200"
                }`}
              >
                <label className="flex min-w-0 flex-1 cursor-pointer items-center gap-2.5">
                  <input
                    type="checkbox"
                    checked={on}
                    onChange={() => toggle(d.id)}
                    disabled={running}
                    className="h-4 w-4"
                  />
                  <span className="truncate text-sm font-medium text-slate-800">{d.name}</span>
                  <span className="text-xs text-slate-400">
                    {d.n_cells != null ? `${formatNumber(d.n_cells)} cells` : ""}
                  </span>
                </label>
                {on && (
                  <label className="flex items-center gap-1.5 text-xs text-slate-500">
                    label
                    <input
                      value={labels[d.id] ?? d.name}
                      onChange={(e) => setLabels((l) => ({ ...l, [d.id]: e.target.value }))}
                      disabled={running}
                      className="w-36 rounded border border-slate-300 px-2 py-1 text-xs text-slate-700"
                    />
                  </label>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Compatibility */}
      {selected.length >= 2 && (
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-700">
            Compatibility
            {planning && <Loader2 className="h-3.5 w-3.5 animate-spin text-slate-400" />}
          </h3>
          {planError && <ErrorLine text={planError} />}
          {plan && (
            <div className="space-y-2 text-sm">
              <ul className="space-y-1">
                {plan.datasets.map((d) => (
                  <li key={d.id} className="flex items-start gap-2">
                    {d.error ? (
                      <XCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-500" />
                    ) : (
                      <CheckCircle2 className="mt-0.5 h-4 w-4 flex-shrink-0 text-emerald-500" />
                    )}
                    <span className="text-slate-700">
                      <span className="font-medium">{d.label}</span>
                      {d.error ? (
                        <span className="text-red-600"> — {d.error}</span>
                      ) : (
                        <span className="text-slate-500">
                          {" "}
                          — {d.n_cells.toLocaleString()} cells, raw counts from {d.counts_source}
                        </span>
                      )}
                    </span>
                  </li>
                ))}
              </ul>
              {m && (
                <p className="text-slate-600">
                  <span className="font-medium">{sharedGenes.toLocaleString()}</span> shared genes (of{" "}
                  {m.union.toLocaleString()} in any dataset) ·{" "}
                  <span className="font-medium">{plan.est_cells.toLocaleString()}</span> cells in total
                </p>
              )}
              {m?.reconciliation?.message && <WarnLine text={m.reconciliation.message} />}
              {m?.warnings.map((w) => <WarnLine key={w} text={w} />)}
              {plan.shared_columns.length > 0 && (
                <p className="text-xs text-slate-500">
                  Annotations in every dataset (comparable across them):{" "}
                  {plan.shared_columns.map((c) => (
                    <span key={c} className="mr-1 inline-block rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[11px] text-slate-600">
                      {prettyObsLabel(c)}
                    </span>
                  ))}
                </p>
              )}
              {plan.blocking.map((b) => <ErrorLine key={b} text={b} />)}
            </div>
          )}
        </div>
      )}

      {/* Settings + run */}
      {selected.length >= 2 && (
        <div className="flex flex-wrap items-end gap-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="min-w-[220px] flex-1">
            <label className="mb-1 block text-xs font-medium text-slate-500">Name</label>
            <input
              value={name}
              placeholder={defaultName}
              onChange={(e) => setName(e.target.value)}
              disabled={running}
              className="w-full rounded-lg border border-slate-300 px-3 py-1.5 text-sm text-slate-700"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-500">Batch correction</label>
            <select
              value={method}
              onChange={(e) => setMethod(e.target.value as IntegrationMethod)}
              disabled={running}
              className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700"
            >
              <option value="harmony">Harmony (recommended)</option>
              <option value="none">None — concatenate only</option>
            </select>
          </div>
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-500">Cluster resolution</label>
            <input
              type="number"
              min={0.1}
              max={3}
              step={0.1}
              value={resolution}
              onChange={(e) => setResolution(parseFloat(e.target.value) || 0.5)}
              disabled={running}
              className="w-24 rounded-lg border border-slate-300 px-3 py-1.5 text-sm text-slate-700"
            />
          </div>
          <button
            onClick={build}
            disabled={!plan?.ok || running || planning}
            className="ml-auto inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary/90 disabled:opacity-50"
          >
            {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Layers className="h-4 w-4" />}
            Build joint embedding
          </button>
        </div>
      )}

      {/* Progress */}
      {progress && (
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <h3 className="mb-2 text-sm font-semibold text-slate-700">{progress.stage}</h3>
          <ul className="grid gap-1 text-sm sm:grid-cols-2">
            {progress.steps.map((s) => (
              <li key={s} className="flex items-center gap-2">
                <StepIcon status={progress.status[s] ?? "pending"} />
                <span className={progress.status[s] === "error" ? "text-red-600" : "text-slate-600"}>
                  {STEP_LABELS[s] ?? s}
                </span>
              </li>
            ))}
          </ul>
          {progress.substep && <p className="mt-2 text-xs text-slate-400">{progress.substep}</p>}
          {Object.entries(progress.errors).map(([s, err]) => (
            <ErrorLine key={s} text={`${STEP_LABELS[s] ?? s}: ${err}`} />
          ))}
          {runError && <ErrorLine text={runError} />}
          {resultId && (
            <div className="mt-3 flex flex-wrap items-center gap-3 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2.5">
              <span className="text-sm text-emerald-800">
                Joint dataset ready. It opens colored by <code>dataset</code>; switch the embedding to{" "}
                <em>UMAP (uncorrected)</em> to see the layout before correction, and use the DE tab's{" "}
                <em>Between groups</em> mode to compare datasets within a cluster or cell type.
              </span>
              <button
                onClick={openResult}
                className="ml-auto rounded-lg bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-700"
              >
                Open in Unified View
              </button>
            </div>
          )}
        </div>
      )}

      {resultId && <JointSummary datasetId={resultId} title="How well the datasets mixed" />}
    </div>
  );
}

function StepIcon({ status }: { status: StepStatus }) {
  if (status === "running") return <Loader2 className="h-4 w-4 animate-spin text-primary" />;
  if (status === "done") return <CheckCircle2 className="h-4 w-4 text-emerald-500" />;
  if (status === "error") return <XCircle className="h-4 w-4 text-red-500" />;
  return <Circle className="h-4 w-4 text-slate-300" />;
}

function ErrorLine({ text }: { text: string }) {
  return (
    <div className="mt-1 flex items-start gap-2 text-sm text-red-700">
      <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0" />
      <span>{text}</span>
    </div>
  );
}

function WarnLine({ text }: { text: string }) {
  return (
    <div className="flex items-start gap-2 text-sm text-amber-700">
      <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" />
      <span>{text}</span>
    </div>
  );
}
