import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { DatasetInfo } from "@/api/types";

const { apiFetch } = vi.hoisted(() => ({ apiFetch: vi.fn() }));
vi.mock("@/api/client", () => ({ apiFetch }));
// The cards beside the stepper fetch on their own; not what is tested here.
vi.mock("@/components/panels/QcPlots", () => ({ QcPlots: () => null }));
vi.mock("@/components/panels/ClusterQcTable", () => ({ ClusterQcTable: () => null }));
vi.mock("@/components/panels/QcReportCard", () => ({ QcReportCard: () => null }));
vi.mock("@/components/panels/AnnotationControl", () => ({ AnnotationControl: () => null }));
vi.mock("@/components/panels/AnnotationConfidenceCard", () => ({ AnnotationConfidenceCard: () => null }));
vi.mock("@/components/panels/MeanVariancePlot", () => ({ MeanVariancePlot: () => null }));

import { DataAssessmentPanel } from "./DataAssessmentPanel";
import { useDatasetStore } from "@/stores/datasetStore";

const STEPS = [
  "qc_metrics", "doublet_detection", "filtering", "normalization", "log_transform",
  "highly_variable_genes", "scaling", "pca", "batch_correction", "neighbors", "clustering",
  "embeddings", "marker_genes", "enrichment", "cell_cycle", "dissociation_stress",
];

const DATASET = {
  id: "d1", name: "d1", filename: "d1.h5ad", n_cells: 10, n_genes: 5, status: "ready",
  available_embeddings: [], obs_columns: [], created_at: "",
} as DatasetInfo;

function assessment(method: string | null) {
  const done = { done: true, confidence: "high", details: "Run by scView (recorded)" };
  return { ...Object.fromEntries(STEPS.map((s) => [s, done])), normalization_method: method };
}

function streamResponse() {
  const body = new ReadableStream({
    start(c) {
      c.enqueue(new TextEncoder().encode('event: complete\ndata: {"steps_run":[],"steps_skipped":[],"errors":{},"elapsed_seconds":0,"output_path":null}\n\n'));
      c.close();
    },
  });
  return new Response(body, { status: 200 });
}

function renderPanel() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <DataAssessmentPanel />
    </QueryClientProvider>,
  );
}

describe("DataAssessmentPanel normalisation method", () => {
  const fetchSpy = vi.fn();

  beforeEach(() => {
    apiFetch.mockReset();
    fetchSpy.mockReset().mockImplementation(async () => streamResponse());
    vi.stubGlobal("fetch", fetchSpy);
    useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: DATASET });
  });
  afterEach(() => vi.unstubAllGlobals());

  it("keeps the recorded method when a step is re-run, and greys out the HVG flavour", async () => {
    apiFetch.mockImplementation(async (path: string) =>
      path.endsWith("/assessment") ? assessment("pearson_residuals") : DATASET,
    );
    renderPanel();
    await waitFor(() => screen.getByText("Variable Genes"));

    fireEvent.click(screen.getByText("Variable Genes"));
    expect(screen.getByText(/Not used: Pearson residuals/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /re-run this step/i }));
    await waitFor(() => expect(fetchSpy).toHaveBeenCalled());
    const body = JSON.parse(fetchSpy.mock.calls[0]![1].body as string);
    expect(body).toEqual({
      steps: ["highly_variable_genes"],
      params: { normalization_method: "pearson_residuals" },
    });
  });

  it("offers both methods by name on the Normalization step, the shifted log first", async () => {
    apiFetch.mockImplementation(async (path: string) =>
      path.endsWith("/assessment") ? assessment(null) : DATASET,
    );
    renderPanel();
    await waitFor(() => screen.getByText("Normalization"));
    fireEvent.click(screen.getByText("Normalization"));
    const options = screen.getAllByRole("option").map((o) => o.textContent);
    expect(options).toEqual(["Shifted log (default)", "Analytic Pearson residuals"]);
    expect(screen.getByText(/Markers, DE, scores and plots keep the log-normalised values/)).toBeTruthy();

    fireEvent.click(screen.getByText("Variable Genes"));
    expect(screen.queryByText(/Not used: Pearson residuals/)).toBeNull();
  });
});
