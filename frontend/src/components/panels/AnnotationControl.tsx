import { useEffect, useId, useMemo, useState } from "react";
import { AlertTriangle, Tags, Loader2, Check } from "lucide-react";
import { getCellTypistModels, getCellTypistSummary, type CellTypistModel } from "@/api/annotation";
import { getLlmStatus } from "@/api/assistant";
import { getProvenance } from "@/api/provenance";
import { CellTypistSummary } from "@/components/panels/CellTypistSummary";
import { ClusterEvidenceList } from "@/components/qc/ClusterEvidence";
import { AnnotationOntologyIds } from "@/components/annotation/AnnotationOntologyIds";
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
  const [llmProvider, setLlmProvider] = useState<string | null>(null);
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
  // The LLM reply ran out of tokens even after the retry: some clusters are Unknown.
  // From the step's record (`llm_reply_cut`), which History also shows.
  const [replyCut, setReplyCut] = useState<string | null>(null);
  // Clusters scView labelled itself, without the AI (`llm_decided_low_quality`, `llm_random_markers`,
  // `llm_high_mito_type_to_ai` in the step's record).
  const [decided, setDecided] = useState<Decided | null>(null);

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
      .then((s) => {
        if (!live) return;
        setLlmAvailable(s.available);
        setLlmProvider(s.available && s.provider ? s.provider : null);
      })
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
  // The labels' Cell Ontology ids: the column just written, else the output column if it
  // exists, else `cell_type` (what the pipeline's annotation step writes) if that exists.
  const hasColumn = (name: string) => !!dataset?.obs_columns?.some((c) => c.name === name);
  const ontologyColumn =
    lastWritten ?? (hasColumn(target.trim()) ? target.trim() : hasColumn("cell_type") ? "cell_type" : null);

  const handleAnnotate = async () => {
    // scView's own column (the server's default too), never a file's curated cell_type
    const tgt = target.trim() || (groupby ? `${groupby}_celltypeAnno` : "celltypeAnno");
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
    setReplyCut(null);
    setDecided(null);
    try {
      await onAnnotate(params);
      setLastWritten(tgt);
      if (method === "llm" && datasetId) {
        const notes = await lastLlmNotes(datasetId, tgt);
        setReplyCut(notes.cut);
        setDecided(notes.decided && { ...notes.decided, column: groupby });
      }
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
            {llmProvider && ` The marker lists are sent to ${llmProvider}.`}
          </p>
          {llmAvailable === false && (
            <p className="mt-1 flex items-start gap-1 text-[11px] text-amber-700">
              <AlertTriangle className="mt-px h-3 w-3 shrink-0" />
              No language model key is set, so this runs CellTypist instead. Add DEEPINFRA_API_KEY
              (or LLM_API_KEY for another provider) to the .env file in the scView folder and restart
              scView to use the AI.
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
      {lastWritten && !busy && replyCut && (
        <p
          role="alert"
          className="mt-2 flex items-start gap-1.5 rounded-md border border-amber-200 bg-amber-50 px-2 py-1.5 text-xs text-amber-800"
        >
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>
            The model&apos;s reply was cut short: {replyCut}. Run it again, annotate fewer clusters, or pick a
            model that reasons less.
          </span>
        </p>
      )}
      {lastWritten && !busy && decided && datasetId && <DecidedNote decided={decided} datasetId={datasetId} />}
      {lastWritten && !busy && hasColumn(`${lastWritten}_qualified`) && (
        <p className="text-[11px] text-slate-500">
          Also wrote{" "}
          <code className="rounded bg-slate-100 px-1 font-mono">{lastWritten}_qualified</code>: where clusters
          share a label, or a label is a mixture or no cell type, it adds the genes that set each cluster apart,
          e.g. &quot;Monocytes (CCL8+ RSAD2+)&quot; (+ higher, - lower than the clusters it is compared with).
        </p>
      )}
      {datasetId && summaryColumn && (
        <CellTypistSummary datasetId={datasetId} column={summaryColumn} refreshKey={ctRuns} />
      )}
      {datasetId && ontologyColumn && !busy && (
        <AnnotationOntologyIds datasetId={datasetId} column={ontologyColumn} refreshKey={ctRuns} />
      )}
    </div>
  );
}

interface Decided {
  lowQuality: string[];
  /** Labelled Unidentified by scView itself: annotations of v0.71.0 to v0.74.0 */
  unidentified: string[];
  /** Sent to the AI with a note that their markers look random (`llm_random_markers_noted`) */
  noted: string[];
  toAi: string[];
  /** The clustering the annotation was based on (its clusters' evidence). */
  column?: string;
}

const ids = (v: unknown): string[] => (Array.isArray(v) ? v.map(String) : []);
const clusterList = (g: string[]) => `${g.length === 1 ? "cluster" : "clusters"} ${g.join(", ")}`;

/** The LLM notes of the latest annotation step that wrote `target`: a cut reply, and the clusters
 * scView decided itself. Best effort: a failed history read shows nothing (History still has the chips). */
async function lastLlmNotes(datasetId: string, target: string): Promise<{ cut: string | null; decided: Decided | null }> {
  try {
    const prov = await getProvenance(datasetId);
    const step = [...(prov.recorded?.history ?? [])]
      .reverse()
      .find((h) => h.step === "cell_type_annotation");
    if (!step || (step.params?.annotation_target ?? "cell_type") !== target) return { cut: null, decided: null };
    const p = step.params ?? {};
    const cut = typeof p.llm_reply_cut === "string" && p.llm_reply_cut ? p.llm_reply_cut : null;
    const d = { lowQuality: ids(p.llm_decided_low_quality), unidentified: ids(p.llm_random_markers),
      noted: ids(p.llm_random_markers_noted), toAi: ids(p.llm_high_mito_type_to_ai) };
    const any = d.lowQuality.length || d.unidentified.length || d.noted.length || d.toAi.length;
    return { cut, decided: any ? d : null };
  } catch {
    return { cut: null, decided: null };
  }
}

/** Which clusters scView labelled without the AI, and why: the rule, then each cluster's evidence. */
function DecidedNote({ decided, datasetId }: { decided: Decided; datasetId: string }) {
  const [showEvidence, setShowEvidence] = useState(false);
  const groups = [...decided.lowQuality, ...decided.unidentified, ...decided.noted, ...decided.toAi];
  return (
    <div className="mt-2 space-y-0.5 rounded-md border border-slate-200 bg-slate-50 px-2 py-1.5 text-xs text-slate-600" data-testid="annotation-decided">
      {decided.lowQuality.length > 0 && (
        <p>
          scView labelled {clusterList(decided.lowQuality)} <b>Low-quality cells</b> itself: cluster QC flags
          them as damaged (high mitochondrial RNA, or few genes with it).
        </p>
      )}
      {decided.unidentified.length > 0 && (
        <p>
          scView labelled {clusterList(decided.unidentified)} <b>Unidentified</b> itself: their markers do not beat
          9 of 10 random gene lists of the same size against the marker databases.
        </p>
      )}
      {decided.noted.length > 0 && (
        <p>
          The AI was told that the markers of {clusterList(decided.noted)} do not beat 9 of 10 random gene lists
          of the same size against the marker databases: junk, or a state the databases lack.
        </p>
      )}
      {decided.toAi.length > 0 && (
        <p>
          {clusterList(decided.toAi).replace(/^c/, "C")} also{" "}
          {decided.toAi.length === 1 ? "has" : "have"} high mitochondrial RNA, but{" "}
          {decided.toAi.length === 1 ? "its" : "their"} markers match a cell type with much of it by nature
          (such as cardiomyocytes or hepatocytes), so the AI named {decided.toAi.length === 1 ? "it" : "them"}.
        </p>
      )}
      <p>The AI named the other clusters.</p>
      <button
        type="button"
        onClick={() => setShowEvidence(!showEvidence)}
        aria-expanded={showEvidence}
        className="font-medium text-primary hover:underline"
      >
        {showEvidence ? "Hide the evidence" : "Show the evidence"}
      </button>
      {showEvidence && (
        <div className="pt-1">
          <ClusterEvidenceList datasetId={datasetId} column={decided.column} only={groups} />
        </div>
      )}
    </div>
  );
}
