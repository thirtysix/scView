import { useEffect, useState } from "react";
import { Tags, Loader2 } from "lucide-react";
import { API_BASE } from "@/lib/constants";

interface CellTypistModel {
  model: string;
  description: string;
}

interface Props {
  /** Run the cell_type_annotation pipeline step with these PipelineParams. */
  onAnnotate: (params: Record<string, unknown>) => void;
  running: boolean;
}

/**
 * Cell-type annotation control for the Data Assessment panel.
 *
 * Default method is the any-tissue LLM (no reference model to pick — names cell
 * types from each cluster's marker genes). CellTypist is the opt-in reference
 * method, which then needs a tissue-matched model from the catalog.
 */
export function AnnotationControl({ onAnnotate, running }: Props) {
  const [method, setMethod] = useState<"llm" | "celltypist">("llm");
  const [tissue, setTissue] = useState("");
  const [model, setModel] = useState("Immune_All_Low.pkl");
  const [models, setModels] = useState<CellTypistModel[]>([]);

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

  const handleAnnotate = () => {
    const params: Record<string, unknown> = {
      annotation_method: method,
      annotation_target: "cell_type",
    };
    if (method === "llm") params.annotation_tissue = tissue.trim();
    else params.celltypist_model = model;
    onAnnotate(params);
  };

  const tab = (key: "llm" | "celltypist", label: string) => (
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
        Label your clusters with cell types, writing a{" "}
        <code className="rounded bg-slate-100 px-1">cell_type</code> column you can color by.
        Requires a clustering.
      </p>

      <div className="mb-3 inline-flex rounded-lg border border-slate-200 p-0.5 text-xs">
        {tab("llm", "AI (any tissue)")}
        {tab("celltypist", "CellTypist (reference)")}
      </div>

      {method === "llm" ? (
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

      <button
        type="button"
        onClick={handleAnnotate}
        disabled={running}
        className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-primary/90 disabled:opacity-50"
      >
        {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Tags className="h-4 w-4" />}
        Annotate cell types
      </button>
    </div>
  );
}
