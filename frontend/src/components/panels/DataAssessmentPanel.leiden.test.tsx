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
vi.mock("@/components/qc/SequencingQc", () => ({ SequencingQc: () => null }));
vi.mock("@/components/qc/Hashing", () => ({ Hashing: () => null }));

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

const done = { done: true, confidence: "high", details: "Run by scView (recorded)" };
const ASSESSMENT = Object.fromEntries(STEPS.map((s) => [s, done]));

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

describe("DataAssessmentPanel: the Leiden optimisation setting", () => {
  const fetchSpy = vi.fn();

  beforeEach(() => {
    apiFetch.mockReset().mockImplementation(async (path: string) =>
      path.endsWith("/assessment") ? ASSESSMENT : DATASET,
    );
    fetchSpy.mockReset().mockImplementation(async () => streamResponse());
    vi.stubGlobal("fetch", fetchSpy);
    useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: DATASET });
  });
  afterEach(() => vi.unstubAllGlobals());

  it("is always shown on Clustering, full optimisation by default, with what fast Leiden changes", async () => {
    renderPanel();
    await waitFor(() => screen.getByText("Clustering"));
    fireEvent.click(screen.getByText("Clustering"));
    const select = screen.getByDisplayValue("Full optimisation (default)") as HTMLSelectElement;
    expect([...select.options].map((o) => o.textContent)).toEqual([
      "Full optimisation (default)",
      "Fast Leiden (2 passes)",
    ]);
    fireEvent.mouseEnter(screen.getByRole("button", { name: "What fast Leiden changes" }));
    expect(screen.getByRole("tooltip")).toHaveTextContent("ARI 0.93-0.97");
  });

  it("sends fast Leiden as the number 2 with the run", async () => {
    renderPanel();
    await waitFor(() => screen.getByText("Clustering"));
    fireEvent.click(screen.getByText("Clustering"));
    fireEvent.change(screen.getByDisplayValue("Full optimisation (default)"), { target: { value: "2" } });
    expect(screen.getByDisplayValue("Fast Leiden (2 passes)")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /re-run this step/i }));
    await waitFor(() => expect(fetchSpy).toHaveBeenCalled());
    const body = JSON.parse(fetchSpy.mock.calls[0]![1].body as string);
    expect(body).toEqual({ steps: ["clustering"], params: { leiden_iterations: 2 } });
  });

  it("is not used with Louvain", async () => {
    renderPanel();
    await waitFor(() => screen.getByText("Clustering"));
    fireEvent.click(screen.getByText("Clustering"));
    fireEvent.change(screen.getByDisplayValue("leiden"), { target: { value: "louvain" } });
    expect((screen.getByDisplayValue("Full optimisation (default)") as HTMLSelectElement).disabled).toBe(true);
    expect(screen.getByText("Not used: Louvain.")).toBeTruthy();
  });
});
