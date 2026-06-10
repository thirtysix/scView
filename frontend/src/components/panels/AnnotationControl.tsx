import { useEffect, useMemo, useState } from "react";
import { Tags, Loader2, Check } from "lucide-react";
import { API_BASE } from "@/lib/constants";
import { prettyObsLabel, isRedundantClusterCol } from "@/lib/formatting";
import { useDatasetStore } from "@/stores/datasetStore";

interface CellTypistModel {
  model: string;
  description: string;
}

interface Props {
  /** Run the cell_type_annotation pipeline step with these PipelineParams. Resolves on completion. */
  onAnnotate: (params: Record<string, unknown>) => Promise<void>;
  running: boolean;
}

/**
 * Cell-type annotation control for the Data Assessment panel.
 *
 * Pick the clustering to base predictions on (so you can annotate, e.g., several
 * Leiden resolutions into separate columns), choose the method (default: any-tissue
 * LLM, no reference model to pick), and the output obs column is shown explicitly.
 */
export function AnnotationControl({ onAnnotate, running }: Props) {
  const dataset = useDatasetStore((s) => s.currentDataset);

  // Candidate grouping (clustering) columns: categorical, 2..100 groups.
  const groupings = useMemo(() => {
    const cols = dataset?.obs_columns ?? [];
    const allNames = cols.map((c) => c.name);
    return cols
      .filter(
        (c) =>
          !isRedundantClusterCol(c.name, allNames) &&
          (c.dtype === "category" || c.dtype === "object" || c.dtype === "bool") &&
          c.n_unique >= 2 &&
          c.n_unique <= 100,
      )
      .map((c) => c.name);
  }, [dataset?.obs_columns]);
  const primary = dataset?.active_clustering ?? (groupings.includes("cluster") ? "cluster" : groupings[0]);

  const [method, setMethod] = useState<"llm" | "celltypist" | "marker_score">("llm");
  const [tissue, setTissue] = useState("");
  const [model, setModel] = useState("Immune_All_Low.pkl");
  const [models, setModels] = useState<CellTypistModel[]>([]);
  const [groupby, setGroupby] = useState<string>("");
  const [target, setTarget] = useState("cell_type");
  const [targetEdited, setTargetEdited] = useState(false);
  const [annotating, setAnnotating] = useState(false);
  const [lastWritten, setLastWritten] = useState<string | null>(null);

  // Default the grouping once the dataset's columns are known.
  useEffect(() => {
    if (!groupby && primary) setGroupby(primary);
  }, [primary, groupby]);

  // Suggest an output column from the grouping (until the user edits it), so
  // annotating multiple clusterings writes to distinct columns.
  useEffect(() => {
    if (targetEdited || !groupby) return;
    setTarget(`${groupby}_celltypeAnno`);
  }, [groupby, targetEdited]);

  // Lazily load the CellTypist catalog only when that method is selected.
  useEffect(() => {
    if (method !== "celltypist" || models.length > 0) return;
    fetch(`${API_BASE}/annotation/celltypist-models`)
      .then((r) => r.json())
      .then((d: { default?: string; models?: CellTypistModel[] }) => {
        setModels(d.models ?? []);
        if (d.default) setModel(d.default);
      })
      .catch(() => {
        /* offline: keep the default model */
      });
  }, [method, models.length]);

  const handleAnnotate = async () => {
    const tgt = target.trim() || "cell_type";
    const params: Record<string, unknown> = {
      annotation_method: method,
      annotation_groupby: groupby,
      annotation_target: tgt,
    };
    if (method === "llm") params.annotation_tissue = tissue.trim();
    else if (method === "celltypist") params.celltypist_model = model;
    setAnnotating(true);
    setLastWritten(null);
    try {
      await onAnnotate(params);
      setLastWritten(tgt);
    } finally {
      setAnnotating(false);
    }
  };

  const busy = annotating || running;
  const tab = (key: "llm" | "celltypist" | "marker_score", label: string) => (
    <button
      type="button"
      onClick={() => setMethod(key)}
      className={`rounded-md px-3 py-1.5 font-medium transition-colors ${
        method === key ? "bg-primary text-white" : "text-slate-600 hover:bg-slate-50"
      }`}
    >
      {label}
    </button>
  );

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <div className="mb-2 flex items-center gap-2">
        <Tags className="h-5 w-5 text-primary" />
        <h3 className="text-sm font-semibold text-slate-900">Cell-type annotation</h3>
      </div>
      <p className="mb-3 text-xs text-slate-500">
        Label clusters with cell types. The result is written to the obs column shown below, which
        you can then color or group by.
      </p>

      {/* Grouping + output column */}
      <div className="mb-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div>
          <label className="mb-1 block text-xs font-medium text-slate-600">
            Base predictions on (clustering)
          </label>
          <select
            value={groupby}
            onChange={(e) => setGroupby(e.target.value)}
            className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-sm focus:border-primary focus:outline-none"
          >
            {groupings.length === 0 && <option value="">(run clustering first)</option>}
            {groupings.map((g) => (
              <option key={g} value={g}>
                {prettyObsLabel(g)}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-slate-600">Output column</label>
          <input
            type="text"
            value={target}
            onChange={(e) => {
              setTarget(e.target.value);
              setTargetEdited(true);
            }}
            className="w-full rounded-lg border border-slate-200 px-3 py-1.5 font-mono text-sm focus:border-primary focus:outline-none"
          />
        </div>
      </div>

      {/* Method */}
      <div className="mb-3 inline-flex rounded-lg border border-slate-200 p-0.5 text-xs">
        {tab("llm", "AI (any tissue)")}
        {tab("celltypist", "CellTypist (reference)")}
        {tab("marker_score", "Marker score (offline)")}
      </div>

      {method === "marker_score" ? (
        <p className="mb-3 text-[11px] text-slate-400">
          Offline and deterministic: scores curated marker sets per cluster and assigns the top cell
          type. No model or network; best for common immune/PBMC types.
        </p>
      ) : method === "llm" ? (
        <div className="mb-3">
          <label className="mb-1 block text-xs font-medium text-slate-600">Tissue (optional)</label>
          <input
            type="text"
            value={tissue}
            onChange={(e) => setTissue(e.target.value)}
            placeholder="e.g. human PBMC, mouse brain"
            className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-sm focus:border-primary focus:outline-none"
          />
          <p className="mt-1 text-[11px] text-slate-400">
            No reference model needed: names cell types from each cluster's top marker genes.
          </p>
        </div>
      ) : (
        <div className="mb-3">
          <label className="mb-1 block text-xs font-medium text-slate-600">
            Reference model (match your tissue)
          </label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-sm focus:border-primary focus:outline-none"
          >
            {models.length === 0 && <option value={model}>{model.replace(/\.pkl$/, "")}</option>}
            {models.map((m) => (
              <option key={m.model} value={m.model}>
                {m.model.replace(/\.pkl$/, "")} — {m.description.slice(0, 64)}
              </option>
            ))}
          </select>
          <p className="mt-1 text-[11px] text-slate-400">
            CellTypist models are tissue-specific; a mismatch mislabels cells.
          </p>
        </div>
      )}

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={handleAnnotate}
          disabled={busy || !groupby}
          className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-primary/90 disabled:opacity-50"
        >
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Tags className="h-4 w-4" />}
          Annotate cell types
        </button>
        {lastWritten && !busy && (
          <span className="inline-flex items-center gap-1 text-xs text-emerald-700">
            <Check className="h-4 w-4" />
            Wrote{" "}
            <code className="rounded bg-emerald-50 px-1 font-mono">{lastWritten}</code> — color or
            group by it in Unified View.
          </span>
        )}
      </div>
    </div>
  );
}
