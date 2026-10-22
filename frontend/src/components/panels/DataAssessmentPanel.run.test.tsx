import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { DatasetInfo } from "@/api/types";

const { apiFetch } = vi.hoisted(() => ({ apiFetch: vi.fn() }));
vi.mock("@/api/client", async () => ({ ...(await vi.importActual<object>("@/api/client")), apiFetch }));
// The cards beside the stepper fetch on their own; not what is tested here.
vi.mock("@/components/panels/QcPlots", () => ({ QcPlots: () => null }));
vi.mock("@/components/panels/ClusterQcTable", () => ({ ClusterQcTable: () => null }));
vi.mock("@/components/panels/QcReportCard", () => ({ QcReportCard: () => null }));
vi.mock("@/components/panels/AnnotationControl", () => ({ AnnotationControl: () => null }));
vi.mock("@/components/panels/AnnotationConfidenceCard", () => ({ AnnotationConfidenceCard: () => null }));
vi.mock("@/components/panels/MeanVariancePlot", () => ({ MeanVariancePlot: () => null }));
vi.mock("@/components/qc/SequencingQc", () => ({ SequencingQc: () => null }));

import { DataAssessmentPanel } from "./DataAssessmentPanel";
import { useDatasetStore } from "@/stores/datasetStore";

const STEPS = [
  "qc_metrics", "doublet_detection", "filtering", "normalization", "log_transform",
  "highly_variable_genes", "scaling", "pca", "batch_correction", "neighbors", "clustering",
  "embeddings", "marker_genes", "enrichment", "cell_cycle", "dissociation_stress",
];
const MISSING = new Set(["qc_metrics", "doublet_detection"]);

const DATASET = {
  id: "d1", name: "d1", filename: "d1.h5ad", n_cells: 10, n_genes: 5, status: "ready",
  available_embeddings: [], obs_columns: [], created_at: "",
} as unknown as DatasetInfo;

function assessment() {
  return {
    ...Object.fromEntries(
      STEPS.map((s) => [s, { done: !MISSING.has(s), confidence: "high", details: "" }]),
    ),
    normalization_method: null,
    qc_threshold_mode: null,
  };
}

/** A run-stream response the test writes events into, and closes. */
function controlledStream() {
  let ctrl!: ReadableStreamDefaultController<Uint8Array>;
  const body = new ReadableStream<Uint8Array>({ start: (c) => void (ctrl = c) });
  const enc = new TextEncoder();
  return {
    response: new Response(body, { status: 200 }),
    send: (event: string, data: unknown) => ctrl.enqueue(enc.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)),
    raw: (text: string) => ctrl.enqueue(enc.encode(text)),
    close: () => ctrl.close(),
  };
}

function renderPanel() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <DataAssessmentPanel />
    </QueryClientProvider>,
  );
}

const RESULT = { steps_skipped: [], errors: {}, elapsed_seconds: 1, output_path: null, unknown_steps: [] };

describe("DataAssessmentPanel run progress: watchdog, Cancel, outcome", () => {
  const fetchSpy = vi.fn();
  let stream: ReturnType<typeof controlledStream>;

  beforeEach(() => {
    stream = controlledStream();
    apiFetch.mockReset().mockImplementation(async (path: string) => {
      if (path.endsWith("/assessment")) return assessment();
      if (path.endsWith("/assessment/cancel")) return { cancelling: true, step: "doublet_detection" };
      if (path.includes("/qc/doublets")) return { suggestion: null };
      return DATASET;
    });
    fetchSpy.mockReset().mockImplementation(async () => stream.response);
    vi.stubGlobal("fetch", fetchSpy);
    useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: DATASET });
  });
  afterEach(() => vi.unstubAllGlobals());

  async function startRun() {
    renderPanel();
    const run = await screen.findByRole("button", { name: /Run 2 Steps/ });
    fireEvent.click(run);
    await waitFor(() => expect(fetchSpy).toHaveBeenCalled());
    expect(fetchSpy.mock.calls[0]?.[0]).toMatch(/\/assessment\/run-stream$/);
  }

  it("shows the watchdog's warnings in words, then a Cancel that ends as 'cancelled'", async () => {
    await startRun();
    stream.send("step_start", { step: "qc_metrics", index: 0, total: 2 });
    stream.send("step_done", { step: "qc_metrics", index: 0, elapsed: 1 });
    stream.send("step_start", { step: "doublet_detection", index: 1, total: 2 });
    stream.send("watchdog", {
      type: "watchdog", kind: "memory", threshold_pct: 90, used_gb: 7.3, limit_gb: 8, pct: 91,
      limited: true, step: "doublet_detection",
    });
    // An event split across two chunks still arrives.
    stream.raw("event: watchdog\n");
    stream.raw('data: {"type":"watchdog","kind":"slow_step","step":"doublet_detection","elapsed_s":185}\n\n');

    expect(await screen.findByText(/Memory at 91% of the 8.0 GB limit\. If it runs out, the system stops the backend; Cancel keeps the dataset as it was\./)).toBeTruthy();
    expect(screen.getByText("Doublet Detection: 3 min so far.")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(apiFetch).toHaveBeenCalledWith("/datasets/d1/assessment/cancel", { method: "POST" }));
    expect(await screen.findByText(/Cancelling: Doublet Detection finishes first, then the run stops without saving\./)).toBeTruthy();
    expect(screen.getByRole("button", { name: /Cancelling/ })).toBeDisabled();

    stream.send("complete", {
      ...RESULT, steps_run: ["qc_metrics", "doublet_detection"], status: "cancelled",
      stop_reason: "Cancelled before saving: nothing was saved, so the dataset is as it was before the run.",
      resources: { peak_gb: 7.4, limit_gb: 8, sample_seconds: 2, steps: {}, start_gb: 1, limited: true, write_s: null },
    });
    stream.close();

    expect(await screen.findByText("Run cancelled")).toBeTruthy();
    expect(screen.getByText("Cancelled before saving: nothing was saved, so the dataset is as it was before the run.")).toBeTruthy();
    expect(screen.getByText("Finished before the cancel, not saved: QC Metrics, Doublet Detection.")).toBeTruthy();
    expect(screen.getByText("Peak memory 7.4 of 8.0 GB (sampled every 2 s).")).toBeTruthy();
    expect(screen.getByText(/Cancelled: 2 of 2 steps finished/)).toBeTruthy();
    expect(screen.queryByText(/Pipeline completed/)).toBeNull();
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
  });

  it("says a run stopped before a step, and that the steps done were not saved", async () => {
    await startRun();
    stream.send("step_start", { step: "qc_metrics", index: 0, total: 2 });
    stream.send("step_done", { step: "qc_metrics", index: 0, elapsed: 1 });
    stream.send("complete", {
      ...RESULT, steps_run: ["qc_metrics"], status: "stopped",
      stop_reason: "Doublet detection needs about 5 GB and 2 GB is free.",
    });
    stream.close();
    expect(await screen.findByText("Run stopped")).toBeTruthy();
    expect(screen.getByText("Doublet detection needs about 5 GB and 2 GB is free.")).toBeTruthy();
    expect(screen.getByText("Finished before the stop, not saved: QC Metrics.")).toBeTruthy();
  });

  it("keeps a completed run's result after the assessment refreshes, with its peak memory", async () => {
    await startRun();
    stream.send("complete", {
      ...RESULT, steps_run: ["qc_metrics", "doublet_detection"], status: "complete",
      resources: { peak_gb: 2.5, limit_gb: 8, sample_seconds: 2, steps: {}, start_gb: 1, limited: true, write_s: 1 },
    });
    stream.close();
    expect(await screen.findByText(/Pipeline completed in 1s/)).toBeTruthy();
    await waitFor(() => expect(apiFetch.mock.calls.filter(([p]) => String(p).endsWith("/assessment")).length).toBe(2));
    expect(screen.getByText(/Pipeline completed in 1s/)).toBeTruthy();
    expect(screen.getByText("Peak memory 2.5 of 8.0 GB (sampled every 2 s).")).toBeTruthy();
  });

  it("shows a refused run's reason through readError, not its JSON", async () => {
    const detail = "Unknown pipeline step 'umap' (did you mean 'embeddings'?). Valid steps: qc_metrics, embeddings.";
    fetchSpy.mockImplementation(async () => new Response(JSON.stringify({ detail }), { status: 422 }));
    renderPanel();
    fireEvent.click(await screen.findByRole("button", { name: /Run 2 Steps/ }));
    expect(await screen.findByText(detail)).toBeTruthy();
    expect(screen.queryByText(/"detail"/)).toBeNull();
  });
});
