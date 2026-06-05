import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { apiFetch } from "@/api/client";
import { getDataset } from "@/api/datasets";
import { API_BASE } from "@/lib/constants";
import { QcPlots } from "@/components/panels/QcPlots";
import {
  ClipboardCheck,
  Check,
  AlertTriangle,
  Circle,
  Play,
  Sparkles,
  Loader2,
  ChevronDown,
  ChevronRight,
  RotateCcw,
} from "lucide-react";
import {
  MSigDBCollectionTree,
  DEFAULT_MSIGDB_COLLECTIONS,
} from "./MSigDBCollectionTree";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

interface StepStatus {
  done: boolean;
  confidence: "high" | "medium" | "low";
  details: string;
}

interface PreprocessingState {
  qc_metrics: StepStatus;
  doublet_detection: StepStatus;
  filtering: StepStatus;
  normalization: StepStatus;
  log_transform: StepStatus;
  highly_variable_genes: StepStatus;
  scaling: StepStatus;
  pca: StepStatus;
  batch_correction: StepStatus;
  neighbors: StepStatus;
  clustering: StepStatus;
  embeddings: StepStatus;
  marker_genes: StepStatus;
  enrichment: StepStatus;
  cell_cycle: StepStatus;
}

interface PipelineResult {
  steps_run: string[];
  steps_skipped: string[];
  errors: Record<string, string>;
  elapsed_seconds: number;
  output_path: string | null;
}

interface LLMSuggestion {
  step: string;
  recommended: boolean;
  reasoning: string;
  suggested_params: Record<string, unknown>;
}

interface AdvisorResponse {
  suggestions: LLMSuggestion[];
  raw_response: string;
}

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

const STEP_ORDER: {
  key: keyof PreprocessingState;
  label: string;
  description: string;
}[] = [
  { key: "qc_metrics", label: "QC Metrics", description: "Calculate quality control metrics" },
  { key: "doublet_detection", label: "Doublet Detection", description: "Flag likely doublets with Scrublet (doublet_score + predicted_doublet)" },
  { key: "filtering", label: "Cell/Gene Filtering", description: "Filter low-quality cells and rare genes" },
  { key: "normalization", label: "Normalization", description: "Normalize library size" },
  { key: "log_transform", label: "Log Transform", description: "Log-transform expression values" },
  { key: "highly_variable_genes", label: "Variable Genes", description: "Identify highly variable genes" },
  { key: "scaling", label: "Scaling", description: "Scale and center gene expression" },
  { key: "pca", label: "PCA", description: "Principal component analysis" },
  { key: "batch_correction", label: "Batch Correction", description: "Harmony integration to correct batch effects in PCA space" },
  { key: "neighbors", label: "Neighbor Graph", description: "Compute nearest neighbor graph" },
  { key: "clustering", label: "Clustering", description: "Cluster cells into groups" },
  { key: "embeddings", label: "UMAP/tSNE", description: "Compute embedding for visualization" },
  { key: "marker_genes", label: "Marker Genes", description: "Find differentially expressed genes per obs column" },
  { key: "enrichment", label: "Pathway Enrichment", description: "Run pathway enrichment on marker genes" },
  { key: "cell_cycle", label: "Cell Cycle", description: "Score cell cycle phases" },
];

interface ParamConfig {
  param: string;
  label: string;
  type: "number" | "select" | "checkbox_list" | "msigdb_tree" | "boolean";
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  default: unknown;
  dynamicOptions?: "categorical_obs_columns";
}

const STEP_PARAMS: Record<string, ParamConfig[]> = {
  doublet_detection: [
    { param: "expected_doublet_rate", label: "Expected doublet rate", type: "number", min: 0.0, max: 0.5, step: 0.01, default: 0.06 },
  ],
  filtering: [
    { param: "min_genes", label: "Min genes/cell", type: "number", min: 0, max: 5000, step: 50, default: 200 },
    { param: "min_cells", label: "Min cells/gene", type: "number", min: 0, max: 100, step: 1, default: 3 },
    { param: "max_pct_mt", label: "Max % mito", type: "number", min: 0, max: 100, step: 1, default: 20 },
    { param: "drop_doublets", label: "Drop predicted doublets", type: "boolean", default: false },
  ],
  normalization: [
    { param: "target_sum", label: "Target sum", type: "number", min: 1000, max: 100000, step: 1000, default: 10000 },
  ],
  highly_variable_genes: [
    { param: "n_top_genes", label: "N top genes", type: "number", min: 500, max: 10000, step: 100, default: 2000 },
  ],
  pca: [
    { param: "n_comps", label: "N components", type: "number", min: 10, max: 200, step: 5, default: 50 },
  ],
  batch_correction: [
    { param: "batch_key", label: "Batch Column", type: "select", options: [], default: "" },
  ],
  neighbors: [
    { param: "n_neighbors", label: "N neighbors", type: "number", min: 5, max: 100, step: 1, default: 15 },
  ],
  clustering: [
    { param: "clustering_resolution", label: "Resolution", type: "number", min: 0.1, max: 3.0, step: 0.1, default: 0.5 },
    { param: "clustering_method", label: "Method", type: "select", options: ["leiden", "louvain"], default: "leiden" },
  ],
  embeddings: [
    { param: "umap_min_dist", label: "UMAP min dist", type: "number", min: 0.0, max: 1.0, step: 0.05, default: 0.5 },
  ],
  marker_genes: [
    { param: "marker_columns", label: "Compute markers for these columns", type: "checkbox_list", dynamicOptions: "categorical_obs_columns", default: [] },
  ],
  enrichment: [
    { param: "enrichment_columns", label: "Compute enrichment for these columns", type: "checkbox_list", dynamicOptions: "categorical_obs_columns", default: [] },
    { param: "enrichment_n_genes", label: "Top N genes per group", type: "number", min: 10, max: 500, step: 10, default: 100 },
    { param: "enrichment_collections", label: "MSigDB Collections", type: "msigdb_tree", default: [] },
  ],
};

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function StepIcon({ status, isRunning }: { status: StepStatus; isRunning?: boolean }) {
  if (isRunning) {
    return (
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-100 text-blue-600">
        <Loader2 className="h-4 w-4 animate-spin" />
      </div>
    );
  }
  if (status.done && status.confidence === "high") {
    return (
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-emerald-600">
        <Check className="h-4 w-4" />
      </div>
    );
  }
  if (status.done && (status.confidence === "medium" || status.confidence === "low")) {
    return (
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-amber-100 text-amber-600">
        <AlertTriangle className="h-4 w-4" />
      </div>
    );
  }
  return (
    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-100 text-slate-400">
      <Circle className="h-4 w-4" />
    </div>
  );
}

function confidenceBadge(confidence: "high" | "medium" | "low") {
  const colors = {
    high: "bg-emerald-50 text-emerald-700",
    medium: "bg-amber-50 text-amber-700",
    low: "bg-slate-100 text-slate-500",
  };
  return (
    <span
      className={`mr-1.5 inline-block rounded px-1.5 py-0.5 text-[10px] font-medium uppercase ${colors[confidence]}`}
    >
      {confidence}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* Main component                                                      */
/* ------------------------------------------------------------------ */

export function DataAssessmentPanel() {
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const dataset = useDatasetStore((s) => s.currentDataset);
  const setPipelineRunning = useSettingsStore((s) => s.setPipelineRunning);
  const queryClient = useQueryClient();

  const [state, setState] = useState<PreprocessingState | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [running, setRunning] = useState(false);
  const [runResult, setRunResult] = useState<PipelineResult | null>(null);
  const [runningStep, setRunningStep] = useState<string | null>(null);
  const [runProgress, setRunProgress] = useState<{ current: number; total: number } | null>(null);
  const [completedRunSteps, setCompletedRunSteps] = useState<Set<string>>(new Set());
  const [substep, setSubstep] = useState<{ message: string; current: number; total: number } | null>(null);

  const [suggesting, setSuggesting] = useState(false);
  const [suggestions, setSuggestions] = useState<AdvisorResponse | null>(null);

  const [expandedStep, setExpandedStep] = useState<string | null>(null);
  const [selectedSteps, setSelectedSteps] = useState<Set<string>>(new Set());
  const [stepParams, setStepParams] = useState<Record<string, Record<string, unknown>>>({});

  /* ---- Categorical obs columns (shared by batch, markers, enrichment) ---- */
  const categoricalObsColumns = useMemo(() => {
    if (!dataset) return [];
    return dataset.obs_columns
      .filter((c) => {
        const isCat = c.dtype === "category" || c.dtype === "object" || c.dtype === "bool";
        if (!isCat) return false;
        if (c.n_unique < 2 || c.n_unique > 100) return false;
        const nCells = dataset.n_cells ?? 0;
        if (nCells > 0 && c.n_unique / nCells >= 0.9) return false;
        return true;
      })
      .map((c) => c.name);
  }, [dataset]);
  const batchColumnOptions = categoricalObsColumns;

  /* ---- Reprocess from scratch ---- */
  const reprocessFromScratch = () => {
    if (!confirm("This will reset to raw counts and reprocess everything from scratch. Continue?")) return;
    // All steps including reset_to_counts
    const allSteps = ["reset_to_counts", ...STEP_ORDER.filter((s) => s.key !== "batch_correction").map((s) => s.key)];
    // Flatten current params
    const flatParams: Record<string, unknown> = {};
    for (const stepKey of allSteps) {
      const sp = stepParams[stepKey];
      if (sp) Object.assign(flatParams, sp);
    }
    runSteps(allSteps, Object.keys(flatParams).length > 0 ? flatParams : null);
  };

  /* ---- Fetch assessment ---- */
  const fetchAssessment = useCallback(async () => {
    if (!datasetId) return;
    setLoading(true);
    setError(null);
    setRunResult(null);
    setSuggestions(null);
    try {
      const data = await apiFetch<PreprocessingState>(
        `/datasets/${datasetId}/assessment`
      );
      setState(data);
      // Auto-select all missing steps
      const skipByDefault = new Set(["batch_correction", "enrichment"]);
      const missing = STEP_ORDER.filter((s) => !data[s.key].done && !skipByDefault.has(s.key)).map((s) => s.key);
      setSelectedSteps(new Set(missing));
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to fetch assessment";
      setError(msg);
      setState(null);
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

  useEffect(() => {
    fetchAssessment();
  }, [fetchAssessment]);

  // Subset of categorical columns likely to be cluster/cell-type groupings (for marker default)
  const clusterLikeColumns = useMemo(() => {
    if (!dataset) return [];
    const patterns = ["leiden", "louvain", "cluster", "seurat_clusters", "cell_type", "celltype"];
    const candidates = dataset.obs_columns.filter((c) => {
      const lower = c.name.toLowerCase();
      return (
        (c.dtype === "category" || c.dtype === "object") &&
        c.n_unique >= 2 && c.n_unique <= 100 &&
        patterns.some((p) => lower.includes(p))
      );
    }).map((c) => c.name);
    // If active clustering exists, ensure it's included
    if (dataset.active_clustering && !candidates.includes(dataset.active_clustering)) {
      candidates.unshift(dataset.active_clustering);
    }
    // Fallback: if nothing matched, use just the first categorical column
    if (candidates.length === 0 && categoricalObsColumns.length > 0) {
      return [categoricalObsColumns[0]!];
    }
    return candidates;
  }, [dataset, categoricalObsColumns]);

  /* ---- Predicted clustering column (when clustering step is selected) ---- */
  const predictedClusteringCol = useMemo(() => {
    if (!selectedSteps.has("clustering")) return null;
    const method = (stepParams.clustering?.clustering_method as string) ?? "leiden";
    const res = stepParams.clustering?.clustering_resolution ?? 0.5;
    return `scview_${method}_r${res}`;
  }, [selectedSteps, stepParams.clustering?.clustering_method, stepParams.clustering?.clustering_resolution]);

  /** Categorical options augmented with predicted clustering column for checkbox_list */
  const augmentedCategoricalOptions = useMemo(() => {
    if (!predictedClusteringCol || categoricalObsColumns.includes(predictedClusteringCol)) {
      return categoricalObsColumns;
    }
    return [predictedClusteringCol, ...categoricalObsColumns];
  }, [predictedClusteringCol, categoricalObsColumns]);

  /* ---- Auto-initialize column checkboxes for marker_genes & enrichment ---- */
  useEffect(() => {
    if (!state || categoricalObsColumns.length === 0) return;
    setStepParams((prev) => {
      const next = { ...prev };
      // Markers default to cluster-like columns only (not all categorical)
      if (!next.marker_genes?.marker_columns) {
        next.marker_genes = { ...next.marker_genes, marker_columns: clusterLikeColumns };
      }
      // Enrichment defaults to same cluster-like columns
      if (!next.enrichment?.enrichment_columns) {
        next.enrichment = { ...next.enrichment, enrichment_columns: clusterLikeColumns };
      }
      if (!next.enrichment?.enrichment_collections) {
        next.enrichment = { ...next.enrichment, enrichment_collections: [...DEFAULT_MSIGDB_COLLECTIONS] };
      }
      return next;
    });
  }, [state, categoricalObsColumns, clusterLikeColumns]);

  /* ---- Sync predicted clustering column into marker/enrichment selections ---- */
  const prevPredictedRef = useRef<string | null>(null);
  useEffect(() => {
    const prev = prevPredictedRef.current;
    prevPredictedRef.current = predictedClusteringCol;

    // Remove old predicted column if it changed or clustering was unchecked
    if (prev && prev !== predictedClusteringCol) {
      setStepParams((p) => {
        const next = { ...p };
        const mc = (next.marker_genes?.marker_columns as string[]) || [];
        if (mc.includes(prev)) next.marker_genes = { ...next.marker_genes, marker_columns: mc.filter((c) => c !== prev) };
        const ec = (next.enrichment?.enrichment_columns as string[]) || [];
        if (ec.includes(prev)) next.enrichment = { ...next.enrichment, enrichment_columns: ec.filter((c) => c !== prev) };
        return next;
      });
    }

    // Add new predicted column when clustering is checked or params change
    if (predictedClusteringCol && predictedClusteringCol !== prev) {
      setStepParams((p) => {
        const next = { ...p };
        const mc = (next.marker_genes?.marker_columns as string[]) || [];
        if (!mc.includes(predictedClusteringCol)) next.marker_genes = { ...next.marker_genes, marker_columns: [predictedClusteringCol, ...mc] };
        const ec = (next.enrichment?.enrichment_columns as string[]) || [];
        if (!ec.includes(predictedClusteringCol)) next.enrichment = { ...next.enrichment, enrichment_columns: [predictedClusteringCol, ...ec] };
        return next;
      });
    }
  }, [predictedClusteringCol]);

  /* ---- Step selection helpers ---- */
  const toggleStep = (key: string) => {
    setSelectedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  const selectAllMissing = () => {
    if (!state) return;
    const missing = STEP_ORDER.filter((s) => !state[s.key].done).map((s) => s.key);
    setSelectedSteps(new Set(missing));
  };

  const deselectAll = () => setSelectedSteps(new Set());

  /* ---- Run steps (core) — uses SSE for progress ---- */
  const runSteps = async (steps: string[], params: Record<string, unknown> | null = null) => {
    if (!datasetId || !state || steps.length === 0) return;

    setRunning(true);
    setPipelineRunning(true);
    setRunResult(null);
    setRunningStep(null);
    setRunProgress(null);
    setCompletedRunSteps(new Set());
    setSubstep(null);
    setError(null);

    try {
      const res = await fetch(
        `${API_BASE}/datasets/${datasetId}/assessment/run-stream`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ steps, params }),
        }
      );

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`API error ${res.status}: ${errText}`);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response stream");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        // Keep the last partial line in the buffer
        buffer = lines.pop() ?? "";

        let currentEvent = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith("data: ") && currentEvent) {
            const data = JSON.parse(line.slice(6));
            if (currentEvent === "step_start") {
              setRunningStep(data.step);
              setRunProgress({ current: data.index, total: data.total });
              setSubstep(null);
            } else if (currentEvent === "step_done") {
              setCompletedRunSteps((prev) => new Set(prev).add(data.step));
              setRunProgress({ current: data.index + 1, total: data.total });
              setSubstep(null);
            } else if (currentEvent === "substep") {
              setSubstep({ message: data.message, current: data.current, total: data.total });
            } else if (currentEvent === "step_error") {
              setCompletedRunSteps((prev) => new Set(prev).add(data.step));
              setSubstep(null);
            } else if (currentEvent === "complete") {
              setRunResult(data as PipelineResult);
              setSubstep(null);
            } else if (currentEvent === "error") {
              throw new Error(data.error);
            }
            currentEvent = "";
          }
        }
      }

      setRunningStep(null);
      // Keep runProgress alive so the completed bar stays visible
      await fetchAssessment();
      // Refresh dataset metadata
      try {
        const updated = await getDataset(datasetId);
        useDatasetStore.getState().setCurrentDataset(updated);
        // Auto-select the active clustering column for colorBy
        if (updated.active_clustering) {
          useSettingsStore.getState().setColorBy(updated.active_clustering);
        }
      } catch { /* best-effort */ }
      queryClient.invalidateQueries({ queryKey: ["embedding", datasetId] });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Pipeline run failed";
      setError(msg);
    } finally {
      setPipelineRunning(false);
      setRunning(false);
      setRunningStep(null);
      // Note: runProgress is intentionally kept so the completed bar remains
      // visible until the next run starts (cleared in runSteps initialisation).
    }
  };

  /* ---- Run selected steps ---- */
  const runSelectedSteps = () => {
    const stepsToRun = STEP_ORDER
      .filter((s) => selectedSteps.has(s.key))
      .map((s) => s.key);
    // Flatten per-step params into a single dict for PipelineParams
    const flatParams: Record<string, unknown> = {};
    for (const stepKey of stepsToRun) {
      const sp = stepParams[stepKey];
      if (sp) {
        Object.assign(flatParams, sp);
      }
    }
    runSteps(stepsToRun, Object.keys(flatParams).length > 0 ? flatParams : null);
  };

  /* ---- Derived: re-run count ---- */
  const rerunCount = state
    ? STEP_ORDER.filter((s) => state[s.key].done && selectedSteps.has(s.key)).length
    : 0;

  /* ---- Set param for a step ---- */
  const setStepParam = (stepKey: string, param: string, value: unknown) => {
    setStepParams((prev) => ({
      ...prev,
      [stepKey]: { ...prev[stepKey], [param]: value },
    }));
  };

  /* ---- Apply AI suggestion params ---- */
  const applySuggestion = (suggestion: LLMSuggestion) => {
    const stepKey = suggestion.step;
    if (Object.keys(suggestion.suggested_params).length > 0) {
      setStepParams((prev) => ({
        ...prev,
        [stepKey]: { ...prev[stepKey], ...suggestion.suggested_params },
      }));
    }
    setSelectedSteps((prev) => new Set(prev).add(stepKey));
  };

  /* ---- Get suggestions ---- */
  const getSuggestions = async () => {
    if (!datasetId || !state) return;
    setSuggesting(true);
    setSuggestions(null);
    try {
      const result = await apiFetch<AdvisorResponse>(
        `/datasets/${datasetId}/assessment/suggest`,
        {
          method: "POST",
          body: JSON.stringify({ preprocessing_state: state }),
        }
      );
      setSuggestions(result);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to get suggestions";
      setError(msg);
    } finally {
      setSuggesting(false);
    }
  };

  /* ---- Derived values ---- */
  const missingCount = state
    ? STEP_ORDER.filter((s) => !state[s.key].done).length
    : 0;
  const doneCount = state ? STEP_ORDER.length - missingCount : 0;
  const selectedCount = selectedSteps.size;

  /* ---- Find suggestion for a step ---- */
  const suggestionForStep = (key: string): LLMSuggestion | undefined =>
    suggestions?.suggestions.find((s) => s.step === key);

  /* ---- No dataset selected ---- */
  if (!datasetId) {
    return (
      <div className="mx-auto max-w-4xl space-y-6">
        <div className="flex items-center gap-3">
          <ClipboardCheck className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-slate-900">Data Assessment</h2>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-8 text-center text-slate-400">
          Load a dataset first to assess its preprocessing state.
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <ClipboardCheck className="h-6 w-6 text-primary" />
        <h2 className="text-2xl font-bold text-slate-900">Data Assessment</h2>
      </div>
      <p className="text-sm text-slate-500">
        Automatically assess preprocessing state and run missing analysis steps.
      </p>

      {/* Two columns: controls + steps on the left, QC plots on the right */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start">
        {/* Left: assessment controls + steps */}
        <div className="min-w-0 flex-1 space-y-4">
      {/* Error banner */}
      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Loading state */}
      {loading && (
        <div className="flex items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white p-12 text-slate-400">
          <Loader2 className="h-5 w-5 animate-spin" />
          <span>Analyzing dataset...</span>
        </div>
      )}

      {/* Assessment results */}
      {state && !loading && (
        <>
          {/* Summary bar + actions */}
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex-1 text-sm text-slate-600">
                <span className="font-semibold text-emerald-600">{doneCount}</span>{" "}
                completed,{" "}
                <span className="font-semibold text-slate-500">{missingCount}</span>{" "}
                remaining
                {selectedCount > 0 && (
                  <span className="ml-1 text-primary">
                    ({selectedCount} selected{rerunCount > 0 ? `, ${rerunCount} re-run` : ""})
                  </span>
                )}
              </div>
              <button
                onClick={runSelectedSteps}
                disabled={running || selectedCount === 0}
                className="inline-flex items-center gap-1.5 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {running ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
                {running
                  ? runningStep
                    ? `Running: ${STEP_ORDER.find((s) => s.key === runningStep)?.label ?? runningStep}...`
                    : "Running..."
                  : selectedCount === 0
                    ? "All Steps Complete"
                    : rerunCount > 0 && rerunCount === selectedCount
                      ? `Re-run ${selectedCount} Step${selectedCount !== 1 ? "s" : ""}`
                      : `Run ${selectedCount} Step${selectedCount !== 1 ? "s" : ""}${rerunCount > 0 ? ` (${rerunCount} re-run)` : ""}`}
              </button>
              <button
                onClick={reprocessFromScratch}
                disabled={running}
                className="inline-flex items-center gap-1.5 rounded-lg border border-amber-300 bg-amber-50 px-4 py-2 text-sm font-medium text-amber-800 transition hover:bg-amber-100 disabled:cursor-not-allowed disabled:opacity-50"
                title="Reset to raw counts and run all preprocessing steps from scratch"
              >
                <RotateCcw className="h-4 w-4" />
                Reprocess from Scratch
              </button>
              <button
                onClick={getSuggestions}
                disabled={suggesting}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {suggesting ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Sparkles className="h-4 w-4 text-amber-500" />
                )}
                {suggesting ? "Thinking..." : "Get AI Suggestions"}
              </button>
            </div>
            {/* Select/Deselect links */}
            {missingCount > 0 && (
              <div className="mt-2 flex gap-3 text-xs">
                <button
                  onClick={selectAllMissing}
                  className="text-primary hover:underline"
                >
                  Select all missing
                </button>
                <button
                  onClick={deselectAll}
                  className="text-slate-400 hover:text-slate-600 hover:underline"
                >
                  Deselect all
                </button>
              </div>
            )}
          </div>

          {/* Pipeline progress bar — visible while running or just completed */}
          {(running || runResult) && runProgress && (
            <div className={`rounded-xl border p-4 ${runResult && !running ? "border-emerald-200 bg-emerald-50" : "border-blue-200 bg-blue-50"}`}>
              {/* Header row: step counter + percentage */}
              <div className="mb-2 flex items-center justify-between text-sm">
                <span className={`font-medium ${runResult && !running ? "text-emerald-800" : "text-blue-800"}`}>
                  {runResult && !running ? (
                    <>All {runProgress.total} step{runProgress.total !== 1 ? "s" : ""} complete</>
                  ) : runningStep ? (
                    <>
                      Step {runProgress.current + 1}/{runProgress.total}
                      {": "}
                      {STEP_ORDER.find((s) => s.key === runningStep)?.label ?? runningStep}
                    </>
                  ) : (
                    <>Step {runProgress.current}/{runProgress.total}</>
                  )}
                </span>
                <span className={`text-xs font-medium ${runResult && !running ? "text-emerald-600" : "text-blue-600"}`}>
                  {runResult && !running
                    ? "100%"
                    : `${Math.round((runProgress.current / runProgress.total) * 100)}%`}
                </span>
              </div>

              {/* Segmented progress bar */}
              <div className="flex gap-1">
                {Array.from({ length: runProgress.total }, (_, i) => {
                  const isDone = i < runProgress.current || (runResult && !running);
                  const isActive = i === runProgress.current && !!runningStep && running;
                  return (
                    <div
                      key={i}
                      className={`h-2.5 flex-1 rounded-full transition-all duration-300 ${
                        isDone
                          ? "bg-emerald-500"
                          : isActive
                            ? "animate-pulse bg-blue-400"
                            : "bg-blue-200"
                      }`}
                    />
                  );
                })}
              </div>

              {/* Sub-step progress (e.g. within marker_genes or enrichment) */}
              {substep && running && (
                <div className="mt-2 space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium text-blue-700 truncate">
                      {substep.message}
                    </span>
                    {substep.total > 0 && (
                      <span className="ml-2 flex-shrink-0 text-blue-500">
                        {substep.current + 1}/{substep.total}
                      </span>
                    )}
                  </div>
                  {substep.total > 1 && (
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-blue-100">
                      <div
                        className="h-full rounded-full bg-blue-400 transition-all duration-300"
                        style={{ width: `${Math.round(((substep.current + 1) / substep.total) * 100)}%` }}
                      />
                    </div>
                  )}
                </div>
              )}

              {/* Step labels (only when ≤ 13 steps so they fit) */}
              {runProgress.total <= 13 && (
                <div className="mt-2 flex gap-1">
                  {STEP_ORDER
                    .filter((s) => selectedSteps.has(s.key) || completedRunSteps.has(s.key))
                    .slice(0, runProgress.total)
                    .map((s) => {
                      const isDone = completedRunSteps.has(s.key) || (!running && !!runResult);
                      const isActive = s.key === runningStep && running;
                      return (
                        <div key={s.key} className="min-w-0 flex-1 text-center">
                          <span
                            className={`block truncate text-[9px] ${
                              isDone
                                ? "font-medium text-emerald-700"
                                : isActive
                                  ? "font-medium text-blue-700"
                                  : "text-blue-400"
                            }`}
                            title={s.label}
                          >
                            {isDone ? "✓ " : ""}{s.label}
                          </span>
                        </div>
                      );
                    })}
                </div>
              )}
            </div>
          )}

          {/* Pipeline run result */}
          {runResult && (
            <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-800">
              <p className="font-medium">
                Pipeline completed in {runResult.elapsed_seconds}s
              </p>
              {runResult.steps_run.length > 0 && (
                <p className="mt-1">
                  Steps run: {runResult.steps_run.join(", ")}
                </p>
              )}
              {Object.keys(runResult.errors).length > 0 && (
                <div className="mt-2 text-red-700">
                  <p className="font-medium">Errors:</p>
                  {Object.entries(runResult.errors).map(([step, err]) => (
                    <p key={step} className="ml-2">
                      {step}: {err}
                    </p>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* AI Suggestions card */}
          {suggestions && suggestions.suggestions.length > 0 && (
            <div className="rounded-xl border border-amber-200 bg-amber-50 p-4">
              <div className="mb-3 flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-amber-500" />
                <span className="text-sm font-medium text-amber-800">
                  AI Suggestions
                </span>
              </div>
              <div className="space-y-2">
                {suggestions.suggestions
                  .filter((s) => s.recommended)
                  .map((s) => (
                    <div
                      key={s.step}
                      className="rounded-lg border border-amber-100 bg-white/60 px-3 py-2"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs font-semibold uppercase text-amber-700">
                          {s.step.replace(/_/g, " ")}
                        </span>
                        <button
                          onClick={() => applySuggestion(s)}
                          className="rounded bg-amber-200/60 px-2 py-0.5 text-[10px] font-medium text-amber-800 transition hover:bg-amber-200"
                        >
                          Apply
                        </button>
                      </div>
                      <p className="mt-1 text-xs text-amber-900">{s.reasoning}</p>
                      {Object.keys(s.suggested_params).length > 0 && (
                        <p className="mt-1 text-[10px] text-amber-600">
                          Params:{" "}
                          {Object.entries(s.suggested_params)
                            .map(([k, v]) => `${k}=${String(v)}`)
                            .join(", ")}
                        </p>
                      )}
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Pipeline stepper */}
          <div className="rounded-xl border border-slate-200 bg-white">
            <div className="divide-y divide-slate-100">
              {STEP_ORDER.map((step, idx) => {
                const status = state[step.key];
                const isExpanded = expandedStep === step.key;
                const suggestion = suggestionForStep(step.key);
                const isSelected = selectedSteps.has(step.key);
                const isRerun = status.done && isSelected;
                const hasParams = step.key in STEP_PARAMS;
                return (
                  <div key={step.key}>
                    {/* Step row */}
                    <div className="flex w-full items-center gap-3 px-4 py-3 transition hover:bg-slate-50">
                      {/* Checkbox for all steps */}
                      <div className="flex w-5 shrink-0 items-center justify-center">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggleStep(step.key)}
                          disabled={running}
                          className={`h-4 w-4 cursor-pointer rounded border-slate-300 focus:ring-primary ${
                            isRerun ? "accent-amber-500 text-amber-500" : "accent-primary text-primary"
                          }`}
                        />
                      </div>

                      {/* Icon with connector line */}
                      <div className="relative flex flex-col items-center">
                        <StepIcon status={status} isRunning={runningStep === step.key} />
                        {idx < STEP_ORDER.length - 1 && (
                          <div className="absolute top-8 h-4 w-px bg-slate-200" />
                        )}
                      </div>

                      {/* Labels — clickable to expand */}
                      <button
                        onClick={() =>
                          setExpandedStep(isExpanded ? null : step.key)
                        }
                        className="min-w-0 flex-1 text-left"
                      >
                        <div className="flex items-center gap-1">
                          {confidenceBadge(status.confidence)}
                          <span
                            className={`text-sm font-medium ${
                              status.done ? "text-slate-900" : "text-slate-500"
                            }`}
                          >
                            {step.label}
                          </span>
                          {isRerun && (
                            <span className="rounded bg-amber-100 px-1.5 py-0.5 text-[9px] font-medium text-amber-700">
                              re-run
                            </span>
                          )}
                        </div>
                        <p className="mt-0.5 truncate text-xs text-slate-400">
                          {status.details}
                        </p>
                      </button>

                      {/* Expand chevron */}
                      <button
                        onClick={() =>
                          setExpandedStep(isExpanded ? null : step.key)
                        }
                        className="text-slate-300"
                      >
                        {isExpanded ? (
                          <ChevronDown className="h-4 w-4" />
                        ) : (
                          <ChevronRight className="h-4 w-4" />
                        )}
                      </button>
                    </div>

                    {/* Expanded details */}
                    {isExpanded && (
                      <div className="border-t border-slate-100 bg-slate-50 px-4 py-3 pl-16">
                        <p className="text-xs text-slate-500">
                          {step.description}
                        </p>
                        <p className="mt-2 text-xs text-slate-600">
                          <span className="font-medium">Status:</span>{" "}
                          {status.done ? "Completed" : "Not yet run"}
                        </p>
                        <p className="text-xs text-slate-600">
                          <span className="font-medium">Details:</span>{" "}
                          {status.details}
                        </p>
                        <p className="text-xs text-slate-600">
                          <span className="font-medium">Confidence:</span>{" "}
                          {status.confidence}
                        </p>
                        {/* Parameter inputs */}
                        {hasParams && (
                          <div className="mt-3 rounded border border-slate-200 bg-white px-3 py-2">
                            <p className="mb-2 text-[10px] font-medium uppercase text-slate-500">
                              Parameters
                            </p>
                            <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                              {STEP_PARAMS[step.key]!.map((pc) => {
                                const currentVal = stepParams[step.key]?.[pc.param] ?? pc.default;
                                const resolvedOptions =
                                  pc.dynamicOptions === "categorical_obs_columns"
                                    ? augmentedCategoricalOptions
                                    : (pc.options ?? []);
                                return (
                                  <div key={pc.param} className={`flex flex-col gap-0.5 ${pc.type === "checkbox_list" || pc.type === "msigdb_tree" ? "col-span-2" : ""}`}>
                                    <label className="text-[10px] font-medium text-slate-500">
                                      {pc.label}
                                    </label>
                                    {pc.type === "msigdb_tree" ? (
                                      <MSigDBCollectionTree
                                        selected={new Set((currentVal as string[]) || DEFAULT_MSIGDB_COLLECTIONS)}
                                        onChange={(sel) => setStepParam(step.key, pc.param, Array.from(sel))}
                                        compact
                                      />
                                    ) : pc.type === "checkbox_list" ? (
                                      <div className="flex max-h-40 flex-col gap-1 overflow-auto rounded border border-slate-100 bg-slate-50 p-2">
                                        <div className="mb-1 flex gap-2 text-[10px]">
                                          <button
                                            type="button"
                                            onClick={() => setStepParam(step.key, pc.param, resolvedOptions)}
                                            className="text-primary hover:underline"
                                          >
                                            Select all
                                          </button>
                                          <button
                                            type="button"
                                            onClick={() => setStepParam(step.key, pc.param, [])}
                                            className="text-slate-400 hover:text-slate-600 hover:underline"
                                          >
                                            Deselect all
                                          </button>
                                          <span className="text-slate-400">
                                            ({(currentVal as string[] | undefined)?.length ?? 0}/{resolvedOptions.length})
                                          </span>
                                        </div>
                                        {resolvedOptions.map((opt) => {
                                          const currentList = (currentVal as string[]) || [];
                                          const isChecked = currentList.includes(opt);
                                          return (
                                            <label key={opt} className="flex items-center gap-1.5 text-xs text-slate-700">
                                              <input
                                                type="checkbox"
                                                checked={isChecked}
                                                onChange={() => {
                                                  const next = isChecked
                                                    ? currentList.filter((c) => c !== opt)
                                                    : [...currentList, opt];
                                                  setStepParam(step.key, pc.param, next);
                                                }}
                                                disabled={running}
                                                className="h-3.5 w-3.5 rounded border-slate-300 accent-primary"
                                              />
                                              {opt}
                                              {opt === predictedClusteringCol && (
                                                <span className="rounded bg-blue-100 px-1.5 py-0.5 text-[9px] font-medium text-blue-600">
                                                  from Clustering step
                                                </span>
                                              )}
                                            </label>
                                          );
                                        })}
                                        {resolvedOptions.length === 0 && (
                                          <span className="text-[10px] text-slate-400">No categorical columns available</span>
                                        )}
                                      </div>
                                    ) : pc.type === "boolean" ? (
                                      <label className="flex items-center gap-2 text-xs text-slate-700">
                                        <input
                                          type="checkbox"
                                          checked={Boolean(currentVal)}
                                          onChange={(e) =>
                                            setStepParam(step.key, pc.param, e.target.checked)
                                          }
                                          disabled={running}
                                          className="h-3.5 w-3.5 rounded border-slate-300 text-blue-500 focus:ring-1 focus:ring-blue-100 disabled:opacity-50"
                                        />
                                        <span>{Boolean(currentVal) ? "Yes" : "No"}</span>
                                      </label>
                                    ) : pc.type === "number" ? (
                                      <input
                                        type="number"
                                        value={Number(currentVal)}
                                        min={pc.min}
                                        max={pc.max}
                                        step={pc.step}
                                        onChange={(e) =>
                                          setStepParam(step.key, pc.param, parseFloat(e.target.value) || pc.default)
                                        }
                                        disabled={running}
                                        className="w-full rounded border border-slate-200 px-2 py-1 text-xs text-slate-700 focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-100 disabled:opacity-50"
                                      />
                                    ) : (
                                      <select
                                        value={String(currentVal)}
                                        onChange={(e) =>
                                          setStepParam(step.key, pc.param, e.target.value)
                                        }
                                        disabled={running}
                                        className="w-full rounded border border-slate-200 px-2 py-1 text-xs text-slate-700 focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-100 disabled:opacity-50"
                                      >
                                        {pc.param === "batch_key" ? (
                                          <>
                                            <option value="">Select batch column...</option>
                                            {batchColumnOptions.map((opt) => (
                                              <option key={opt} value={opt}>{opt}</option>
                                            ))}
                                          </>
                                        ) : (
                                          pc.options?.map((opt) => (
                                            <option key={opt} value={opt}>{opt}</option>
                                          ))
                                        )}
                                      </select>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}
                        {/* AI suggestion for this step */}
                        {suggestion && (
                          <div className="mt-3 rounded border border-amber-100 bg-amber-50 px-3 py-2">
                            <div className="flex items-center justify-between">
                              <p className="text-[10px] font-medium uppercase text-amber-600">
                                AI Recommendation
                              </p>
                              {Object.keys(suggestion.suggested_params).length > 0 && (
                                <button
                                  onClick={() => applySuggestion(suggestion)}
                                  className="rounded bg-amber-200/60 px-2 py-0.5 text-[10px] font-medium text-amber-800 transition hover:bg-amber-200"
                                >
                                  Apply params
                                </button>
                              )}
                            </div>
                            <p className="mt-0.5 text-xs text-amber-900">
                              {suggestion.reasoning}
                            </p>
                            {Object.keys(suggestion.suggested_params).length > 0 && (
                              <p className="mt-1 text-[10px] text-amber-600">
                                Suggested params:{" "}
                                {Object.entries(suggestion.suggested_params)
                                  .map(([k, v]) => `${k}=${String(v)}`)
                                  .join(", ")}
                              </p>
                            )}
                          </div>
                        )}
                        {/* Per-step run button */}
                        <button
                          onClick={() => {
                            const sp = stepParams[step.key];
                            runSteps([step.key], sp && Object.keys(sp).length > 0 ? sp : null);
                          }}
                          disabled={running}
                          className={`mt-3 inline-flex items-center gap-1 rounded px-2.5 py-1 text-xs font-medium transition disabled:opacity-50 ${
                            status.done
                              ? "bg-amber-100 text-amber-700 hover:bg-amber-200"
                              : "bg-primary/10 text-primary hover:bg-primary/20"
                          }`}
                        >
                          {status.done ? (
                            <RotateCcw className="h-3 w-3" />
                          ) : (
                            <Play className="h-3 w-3" />
                          )}
                          {status.done ? "Re-run this step" : "Run this step only"}
                        </button>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}
        </div>
        {/* Right: QC distributions */}
        <div className="w-full flex-shrink-0 lg:sticky lg:top-0 lg:w-[44%] lg:self-start">
          {datasetId && <QcPlots datasetId={datasetId} />}
        </div>
      </div>
    </div>
  );
}
