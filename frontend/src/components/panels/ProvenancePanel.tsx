import { useCallback, useEffect, useState } from "react";
import {
  AlertTriangle,
  ClipboardList,
  Clock,
  GitMerge,
  Loader2,
  RotateCcw,
} from "lucide-react";

import {
  getProvenance,
  getRerunPlan,
  rerunStep,
  type Provenance,
  type ProvStep,
  type RerunPlan,
} from "@/api/provenance";
import { getDataset } from "@/api/datasets";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";

// Steps that can be re-run in place (mirror backend rerun.INPLACE_RERUNNABLE).
const RERUNNABLE = new Set([
  "clustering",
  "embeddings",
  "marker_genes",
  "enrichment",
  "cell_cycle",
]);

// Steps with a single high-value editable parameter.
const RERUN_PARAM: Record<string, { key: string; label: string; step: number; min: number; def: number }> = {
  clustering: { key: "clustering_resolution", label: "Resolution", step: 0.1, min: 0.1, def: 0.5 },
  embeddings: { key: "umap_min_dist", label: "Min distance", step: 0.05, min: 0.0, def: 0.5 },
};

function prettyStep(step: string): string {
  return step.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

function fmtTime(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export function ProvenancePanel() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const setCurrentDataset = useDatasetStore((s) => s.setCurrentDataset);
  const setPanel = useViewStore((s) => s.setPanel);
  const [prov, setProv] = useState<Provenance | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Re-run editor state
  const [editingIdx, setEditingIdx] = useState<number | null>(null);
  const [plan, setPlan] = useState<RerunPlan | null>(null);
  const [paramValue, setParamValue] = useState("");
  const [rerunning, setRerunning] = useState(false);
  const [rerunMsg, setRerunMsg] = useState<string | null>(null);

  const load = useCallback(async (id: string) => {
    setLoading(true);
    setError(null);
    try {
      setProv(await getProvenance(id));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load history.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (dataset?.id) void load(dataset.id);
  }, [dataset?.id, load]);

  const openEditor = useCallback(
    async (idx: number, step: ProvStep) => {
      if (!dataset) return;
      setEditingIdx(idx);
      setPlan(null);
      setRerunMsg(null);
      const cfg = RERUN_PARAM[step.step];
      const cur = cfg ? (step.params?.[cfg.key] as number | undefined) ?? cfg.def : 0;
      setParamValue(String(cur));
      try {
        setPlan(await getRerunPlan(dataset.id, step.step));
      } catch {
        /* show button-less message */
      }
    },
    [dataset]
  );

  const confirmRerun = useCallback(
    async (step: string) => {
      if (!dataset) return;
      setRerunning(true);
      setRerunMsg(null);
      try {
        const cfg = RERUN_PARAM[step];
        const params = cfg ? { [cfg.key]: Number(paramValue) } : {};
        const { result } = await rerunStep(dataset.id, step, params);
        await load(dataset.id);
        setCurrentDataset(await getDataset(dataset.id)); // propagate new state to other panels
        setEditingIdx(null);
        const ran = (result?.steps_run as string[]) ?? [];
        setRerunMsg(`Re-ran: ${ran.map(prettyStep).join(", ")}.`);
      } catch (e) {
        setRerunMsg(e instanceof Error ? e.message : "Re-run failed.");
      } finally {
        setRerunning(false);
      }
    },
    [dataset, paramValue, load, setCurrentDataset]
  );

  if (!dataset) {
    return <div className="p-6 text-sm text-slate-500">Load a dataset to see its history.</div>;
  }

  const source = prov?.recorded.source;
  const history = prov?.recorded.history ?? [];
  const current = prov?.recorded.current ?? {};

  return (
    <div className="space-y-6">
      <div>
        <h2 className="flex items-center gap-2 text-2xl font-bold text-slate-800">
          <ClipboardList className="h-6 w-6 text-primary" /> History &amp; Provenance
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          What scView has recorded about how <span className="font-medium">{dataset.name}</span>{" "}
          was created and transformed.
        </p>
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <Loader2 className="h-4 w-4 animate-spin text-primary" /> Loading history…
        </div>
      )}
      {error && <div className="rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</div>}
      {rerunMsg && (
        <div className="rounded-lg bg-emerald-50 p-3 text-sm text-emerald-700">{rerunMsg}</div>
      )}

      {prov && !prov.has_history && (
        <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600">
          No scView history is recorded for this dataset — it was likely created outside scView.
          You can still see <em>inferred</em> processing state in{" "}
          <button
            onClick={() => setPanel("assessment")}
            className="font-medium text-primary hover:underline"
          >
            Data Assessment
          </button>
          . Anything scView does from here on will be recorded automatically.
        </div>
      )}

      {prov && prov.reconcile_issues.length > 0 && (
        <div className="space-y-2">
          {prov.reconcile_issues.map((msg, i) => (
            <div
              key={i}
              className="flex gap-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800"
            >
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
              <span>The recorded history disagrees with the data: {msg}.</span>
            </div>
          ))}
        </div>
      )}

      {source && Object.keys(source).length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <h3 className="text-sm font-semibold text-slate-700">Origin</h3>
          <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
            {source.original_filename && (
              <>
                <dt className="text-slate-400">File</dt>
                <dd className="text-slate-700">{source.original_filename}</dd>
              </>
            )}
            {source.format && (
              <>
                <dt className="text-slate-400">Format</dt>
                <dd className="text-slate-700">{source.format}</dd>
              </>
            )}
            {source.origin && (
              <>
                <dt className="text-slate-400">Added via</dt>
                <dd className="text-slate-700">{source.origin}</dd>
              </>
            )}
            {source.n_cells != null && (
              <>
                <dt className="text-slate-400">At ingest</dt>
                <dd className="text-slate-700">
                  {source.n_cells.toLocaleString()} cells × {source.n_genes?.toLocaleString()} genes
                </dd>
              </>
            )}
            {source.ingested_at && (
              <>
                <dt className="text-slate-400">When</dt>
                <dd className="text-slate-700">{fmtTime(source.ingested_at)}</dd>
              </>
            )}
          </dl>
          {source.merged_from && source.merged_from.length > 0 && (
            <div className="mt-3 rounded-lg bg-slate-50 p-2 text-xs text-slate-600">
              <span className="flex items-center gap-1 font-medium text-slate-700">
                <GitMerge className="h-3.5 w-3.5" /> Merged from {source.merged_from.length} samples
              </span>
              {source.merged_from.map((s) => (
                <span key={s.sample} className="ml-5 block">
                  {s.sample} ({s.format})
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {history.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <h3 className="mb-3 text-sm font-semibold text-slate-700">
            What scView has done ({history.length} step{history.length > 1 ? "s" : ""})
          </h3>
          <ol className="space-y-3">
            {history.map((h, i) => (
              <li key={i}>
                <div className="flex items-start justify-between gap-2">
                  <StepRow step={h} />
                  {RERUNNABLE.has(h.step) && (
                    <button
                      onClick={() => (editingIdx === i ? setEditingIdx(null) : openEditor(i, h))}
                      title="Edit & re-run from here"
                      className="flex shrink-0 items-center gap-1 rounded-lg border border-slate-200 px-2 py-1 text-xs text-slate-500 hover:bg-slate-50 hover:text-primary"
                    >
                      <RotateCcw className="h-3 w-3" /> Re-run
                    </button>
                  )}
                </div>
                {editingIdx === i && (
                  <RerunEditor
                    step={h.step}
                    plan={plan}
                    paramValue={paramValue}
                    setParamValue={setParamValue}
                    rerunning={rerunning}
                    onConfirm={() => confirmRerun(h.step)}
                    onCancel={() => setEditingIdx(null)}
                    onGoAssessment={() => setPanel("assessment")}
                  />
                )}
              </li>
            ))}
          </ol>
        </div>
      )}

      {Object.keys(current).length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <h3 className="mb-2 text-sm font-semibold text-slate-700">Current state</h3>
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(current).map(([k, v]) => (
              <span
                key={k}
                className="rounded bg-primary/10 px-2 py-0.5 text-xs text-primary"
                title={typeof v === "object" ? JSON.stringify(v) : undefined}
              >
                {k}
                {typeof v !== "boolean" &&
                  `: ${typeof v === "object" ? JSON.stringify(v) : String(v)}`}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function StepRow({ step }: { step: ProvStep }) {
  const params = Object.entries(step.params ?? {}).filter(
    ([, v]) => v !== null && v !== "" && !(Array.isArray(v) && v.length === 0)
  );
  return (
    <div className="flex min-w-0 flex-1 gap-3">
      <Clock className="mt-0.5 h-4 w-4 shrink-0 text-slate-300" />
      <div className="min-w-0">
        <div className="flex flex-wrap items-baseline gap-x-2">
          <span className="text-sm font-medium text-slate-800">{prettyStep(step.step)}</span>
          <span className="text-xs text-slate-400">{step.tool}</span>
          <span className="text-xs text-slate-300">{fmtTime(step.timestamp)}</span>
        </div>
        {params.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1">
            {params.map(([k, v]) => (
              <span key={k} className="rounded bg-slate-100 px-1.5 py-0.5 text-xs text-slate-600">
                {k}={Array.isArray(v) ? v.join(", ") : String(v)}
              </span>
            ))}
          </div>
        )}
        {step.effect && (
          <div className="mt-0.5 text-xs text-slate-400">
            → {step.effect.n_cells.toLocaleString()} cells ×{" "}
            {step.effect.n_genes.toLocaleString()} genes
          </div>
        )}
      </div>
    </div>
  );
}

function RerunEditor({
  step,
  plan,
  paramValue,
  setParamValue,
  rerunning,
  onConfirm,
  onCancel,
  onGoAssessment,
}: {
  step: string;
  plan: RerunPlan | null;
  paramValue: string;
  setParamValue: (v: string) => void;
  rerunning: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  onGoAssessment: () => void;
}) {
  const cfg = RERUN_PARAM[step];
  return (
    <div className="ml-7 mt-2 rounded-lg border border-primary/30 bg-primary/5 p-3 text-sm">
      {!plan ? (
        <span className="flex items-center gap-2 text-slate-400">
          <Loader2 className="h-3.5 w-3.5 animate-spin" /> Planning…
        </span>
      ) : plan.requires_reprocess ? (
        <div className="text-amber-700">
          {plan.message}{" "}
          <button onClick={onGoAssessment} className="font-medium underline">
            Open Data Assessment
          </button>
          .
        </div>
      ) : (
        <>
          <p className="text-slate-600">{plan.message}</p>
          {cfg && (
            <label className="mt-2 flex items-center gap-2 text-slate-700">
              {cfg.label}:
              <input
                type="number"
                value={paramValue}
                step={cfg.step}
                min={cfg.min}
                onChange={(e) => setParamValue(e.target.value)}
                className="w-24 rounded border border-slate-300 px-2 py-1 text-sm focus:border-primary focus:outline-none"
              />
            </label>
          )}
          <div className="mt-2 flex gap-2">
            <button
              onClick={onConfirm}
              disabled={rerunning}
              className="flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-50"
            >
              {rerunning ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <RotateCcw className="h-3.5 w-3.5" />
              )}
              Re-run
            </button>
            <button
              onClick={onCancel}
              disabled={rerunning}
              className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs text-slate-600 hover:bg-slate-50 disabled:opacity-50"
            >
              Cancel
            </button>
          </div>
        </>
      )}
    </div>
  );
}
