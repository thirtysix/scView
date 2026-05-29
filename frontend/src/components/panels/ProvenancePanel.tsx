import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, ClipboardList, Clock, Loader2, GitMerge } from "lucide-react";

import { getProvenance, type Provenance, type ProvStep } from "@/api/provenance";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";

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
  const setPanel = useViewStore((s) => s.setPanel);
  const [prov, setProv] = useState<Provenance | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  if (!dataset) {
    return <div className="p-6 text-sm text-slate-500">Load a dataset to see its history.</div>;
  }

  const source = prov?.recorded.source;
  const history = prov?.recorded.history ?? [];
  const current = prov?.recorded.current ?? {};

  return (
    <div className="mx-auto max-w-3xl space-y-6">
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

      {/* Reconciliation warnings */}
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

      {/* Source */}
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

      {/* History timeline */}
      {history.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <h3 className="mb-3 text-sm font-semibold text-slate-700">
            What scView has done ({history.length} step{history.length > 1 ? "s" : ""})
          </h3>
          <ol className="space-y-3">
            {history.map((h, i) => (
              <StepRow key={i} step={h} />
            ))}
          </ol>
        </div>
      )}

      {/* Current state summary */}
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
    <li className="flex gap-3">
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
            → {step.effect.n_cells.toLocaleString()} cells × {step.effect.n_genes.toLocaleString()} genes
          </div>
        )}
      </div>
    </li>
  );
}
