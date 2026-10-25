import { useEffect, useId, useMemo, useState } from "react";
import { AlertTriangle, Tags, Loader2, Check } from "lucide-react";
import { getCellTypistModels, getCellTypistSummary, type CellTypistModel } from "@/api/annotation";
import { getLlmStatus } from "@/api/assistant";
import { CellTypistSummary } from "@/components/panels/CellTypistSummary";
import { defaultModel, orderModels, speciesMismatch, type DatasetSpecies } from "@/lib/celltypistModels";
import { prettyObsLabel, isRedundantClusterCol } from "@/lib/formatting";
import { useDatasetStore } from "@/stores/datasetStore";

interface Props {
  /** Run the cell_type_annotation pipeline step with these PipelineParams. Resolves on completion. */
  onAnnotate: (params: Record<string, unknown>) => Promise<void>;
  running: boolean;
  /** The dataset's species from its gene names (assessment), to order and check models */
  species?: DatasetSpecies;
}

const SPECIES_GROUP: Record<CellTypistModel["species"], string> = {
  human: "Human models",
  mouse: "Mouse models",
  other: "Other species",
};

/**
 * Cell-type annotation control for the Data Assessment panel.
 *
 * Pick the clustering to base predictions on (so you can annotate, e.g., several
 * Leiden resolutions into separate columns), choose the method (default: any-tissue
 * LLM, no reference model to pick; marker scores when no API key is set), and the
 * output obs column is shown explicitly.
 */
export function AnnotationControl({ onAnnotate, running, species }: Props) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);

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

  const fieldId = useId();
  const [method, setMethod] = useState<"llm" | "celltypist" | "marker_score">("llm");
  const [methodPicked, setMethodPicked] = useState(false);
  // null until known; without a key the server runs CellTypist for "llm".
  const [llmAvailable, setLlmAvailable] = useState<boolean | null>(null);
  const [tissue, setTissue] = useState("");
  const [markerSets, setMarkerSets] = useState<"databases" | "builtin">("databases");
  const [model, setModel] = useState("Immune_All_Low.pkl");
  const [modelPicked, setModelPicked] = useState(false);
  const [models, setModels] = useState<CellTypistModel[]>([]);
  const [serverDefault, setServerDefault] = useState("Immune_All_Low.pkl");
  // Columns with a CellTypist run (their summary shows under the button), and a
  // counter bumped by each run so the summary refetches.
  const [ctColumns, setCtColumns] = useState<string[]>([]);
  const [ctRuns, setCtRuns] = useState(0);
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

  useEffect(() => {
    let live = true;
    getLlmStatus()
      .then((s) => live && setLlmAvailable(s.available))
      .catch(() => {
        /* unknown: keep the AI default */
      });
    return () => {
      live = false;
    };
  }, []);

  // Without an API key start on the offline marker scores, until the user picks.
  useEffect(() => {
    if (llmAvailable === false && !methodPicked) setMethod("marker_score");
  }, [llmAvailable, methodPicked]);

  // Lazily load the CellTypist catalog only when that method is selected.
  useEffect(() => {
    if (method !== "celltypist" || models.length > 0) return;
    getCellTypistModels()
      .then((d) => {
        setModels(d.models ?? []);
        if (d.default) setServerDefault(d.default);
      })
      .catch(() => {
        /* offline: keep the default model */
      });
  }, [method, models.length]);

  // Start from a model of the dataset's species, until the user picks one.
  useEffect(() => {
    if (!modelPicked && models.length > 0) setModel(defaultModel(models, species, serverDefault));
  }, [models, species, serverDefault, modelPicked]);

  const groups = useMemo(() => {
    const out: { species: CellTypistModel["species"]; models: CellTypistModel[] }[] = [];
    for (const m of orderModels(models, species)) {
      const last = out[out.length - 1];
      if (last && last.species === m.species) last.models.push(m);
      else out.push({ species: m.species, models: [m] });
    }
    return out;
  }, [models, species]);
  const mismatch = speciesMismatch(models.find((m) => m.model === model), species);

  // Which columns have a CellTypist run, for showing a stored run's summary.
  useEffect(() => {
    if (!datasetId) return;
    let live = true;
    getCellTypistSummary(datasetId)
      .then((info) => live && setCtColumns(info.columns))
      .catch(() => live && setCtColumns([]));
    return () => {
      live = false;
    };
  }, [datasetId, ctRuns]);
  const summaryColumn =
    lastWritten && ctColumns.includes(lastWritten)
      ? lastWritten
      : ctColumns.includes(target.trim())
        ? target.trim()
        : null;

  const handleAnnotate = async () => {
    const tgt = target.trim() || "cell_type";
    const params: Record<string, unknown> = {
      annotation_method: method,
      annotation_groupby: groupby,
      annotation_target: tgt,
    };
    if (method === "llm") params.annotation_tissue = tissue.trim();
    else if (method === "celltypist") params.celltypist_model = model;
    else params.annotation_marker_sets = markerSets;
    setAnnotating(true);
    setLastWritten(null);
    try {
      await onAnnotate(params);
      setLastWritten(tgt);
    } finally {
      setAnnotating(false);
      setCtRuns((n) => n + 1); // another method's run forgets the column's CellTypist run
    }
  };

  const busy = annotating || running;
  const tab = (key: "llm" | "celltypist" | "marker_score", label: string) => (
    <button
      type="button"
      onClick={() => {
        setMethod(key);
        setMethodPicked(true);
      }}
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
          <label htmlFor={`${fieldId}-groupby`} className="mb-1 block text-xs font-medium text-slate-600">
            Base predictions on (clustering)
          </label>
          <select
            id={`${fieldId}-groupby`}
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
          <label htmlFor={`${fieldId}-target`} className="mb-1 block text-xs font-medium text-slate-600">Output column</label>
          <input
            id={`${fieldId}-target`}
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
      <p className="-mt-1 mb-3 text-[11px] text-slate-400">
        To compare methods, run each into its own output column (e.g. <code>cell_type_llm</code>,{" "}
        <code>cell_type_celltypist</code>), then check them against each other in Annotation cross-check below.
      </p>

      {/* Method */}
      <div className="mb-3 inline-flex rounded-lg border border-slate-200 p-0.5 text-xs">
        {tab("llm", llmAvailable === false ? "AI (needs an API key)" : "AI (any tissue)")}
        {tab("celltypist", "CellTypist (reference)")}
        {tab("marker_score", "Marker score (offline)")}
      </div>

      {method === "marker_score" ? (
        <div className="mb-3">
          <label htmlFor="marker-sets" className="mb-1 block text-xs font-medium text-slate-600">
            Marker sets
          </label>
          <select
            id="marker-sets"
            value={markerSets}
            onChange={(e) => setMarkerSets(e.target.value as "databases" | "builtin")}
            className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-sm focus:border-primary focus:outline-none"
          >
            <option value="databases">Marker databases (PanglaoDB, CellMarker)</option>
            <option value="builtin">Built-in immune sets (14 PBMC types)</option>
          </select>
          <p className="mt-1 text-[11px] text-slate-400">
            {markerSets === "databases"
              ? "Names each cluster after the database cell type its top markers match best, any tissue, human or mouse. Downloaded on first use; the built-in sets are used if they are unavailable."
              : "Scores 14 curated immune sets per cluster. Better than the databases on PBMC subtypes (CD4/CD8 T, monocytes), no use outside blood."}
          </p>
        </div>
      ) : method === "llm" ? (
        <div className="mb-3">
          <label className="mb-1 block text-xs font-medium text-slate-600">Tissue (optional)</label>
          <input
            aria-label="Tissue (optional)"
            type="text"
            value={tissue}
            onChange={(e) => setTissue(e.target.value)}
            placeholder="e.g. human PBMC, mouse brain"
            className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-sm focus:border-primary focus:outline-none"
          />
          <p className="mt-1 text-[11px] text-slate-400">
            No reference model needed: names cell types from each cluster's top marker genes.
          </p>
          {llmAvailable === false && (
            <p className="mt-1 flex items-start gap-1 text-[11px] text-amber-700">
              <AlertTriangle className="mt-px h-3 w-3 shrink-0" />
              No DeepInfra API key is set, so this runs CellTypist instead. Add DEEPINFRA_API_KEY to
              the .env file in the scView folder and restart scView to use the AI.
            </p>
          )}
        </div>
      ) : (
        <div className="mb-3">
          <label className="mb-1 block text-xs font-medium text-slate-600">
            Reference model (match your tissue)
          </label>
          <select
            value={model}
            onChange={(e) => {
              setModel(e.target.value);
              setModelPicked(true);
            }}
            aria-label="CellTypist model"
            className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-sm focus:border-primary focus:outline-none"
          >
            {models.length === 0 && <option value={model}>{model.replace(/\.pkl$/, "")}</option>}
            {model === "" && (
              <option value="" disabled>
                Pick the {species ?? ""} model nearest your tissue…
              </option>
            )}
            {groups.map((g) => (
              <optgroup
                key={g.species}
                label={`${SPECIES_GROUP[g.species]}${g.species === species ? " (this dataset)" : ""}${
                  g.species === "mouse" ? " · tissue-specific, no tissue match guaranteed" : ""
                }`}
              >
                {g.models.map((m) => (
                  <option key={m.model} value={m.model}>
                    {m.model.replace(/\.pkl$/, "")} — {m.description.slice(0, 64)}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
          {mismatch ? (
            <p role="alert" className="mt-1 flex gap-1 text-[11px] text-amber-700">
              <AlertTriangle className="mt-0.5 h-3 w-3 flex-shrink-0" />
              <span>{mismatch}</span>
            </p>
          ) : (
            <p className="mt-1 text-[11px] text-slate-400">
              CellTypist models are tissue-specific; a mismatch mislabels cells.
              {species === "mouse" &&
                " There is no general mouse model (none covers every tissue): pick the one nearest yours."}
            </p>
          )}
        </div>
      )}

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={handleAnnotate}
          disabled={busy || !groupby || (method === "celltypist" && !model)}
          title={method === "celltypist" && !model ? "Pick a reference model first" : undefined}
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
      {datasetId && summaryColumn && (
        <CellTypistSummary datasetId={datasetId} column={summaryColumn} refreshKey={ctRuns} />
      )}
    </div>
  );
}
