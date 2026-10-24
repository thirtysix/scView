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

function streamResponse() {
  const body = new ReadableStream({
    start(c) {
      c.enqueue(new TextEncoder().encode('event: complete\ndata: {"steps_run":[],"steps_skipped":[],"errors":{},"elapsed_seconds":0,"output_path":null}\n\n'));
      c.close();
    },
  });
  return new Response(body, { status: 200 });
}

describe("DataAssessmentPanel doublet detector choice", () => {
  const fetchSpy = vi.fn();

  beforeEach(() => {
    apiFetch.mockReset().mockImplementation(async (path: string) => {
      const done = { done: true, confidence: "high", details: "Run by scView (recorded)" };
      return path.endsWith("/assessment") ? Object.fromEntries(STEPS.map((s) => [s, done])) : DATASET;
    });
    fetchSpy.mockReset().mockImplementation(async () => streamResponse());
    vi.stubGlobal("fetch", fetchSpy);
    useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: DATASET });
  });
  afterEach(() => vi.unstubAllGlobals());

  it("offers Scrublet, the classifier or both, explains them, and sends the choice", async () => {
    render(
      <QueryClientProvider client={new QueryClient()}>
        <DataAssessmentPanel />
      </QueryClientProvider>,
    );
    await waitFor(() => screen.getByText("Doublet Detection"));
    fireEvent.click(screen.getByText("Doublet Detection"));
    const select = screen.getByDisplayValue("Scrublet (default)");
    expect(Array.from((select as HTMLSelectElement).options).map((o) => o.textContent)).toEqual([
      "Scrublet (default)", "scDblFinder-style classifier", "Both, and compare",
    ]);
    fireEvent.mouseEnter(screen.getByRole("button", { name: "About the doublet detectors" }));
    expect(screen.getByText(/trains a model on artificial doublets/)).toBeTruthy();
    fireEvent.change(select, { target: { value: "both" } });
    fireEvent.click(screen.getByRole("button", { name: /re-run this step/i }));
    await waitFor(() => expect(fetchSpy).toHaveBeenCalled());
    const body = JSON.parse(fetchSpy.mock.calls[0]![1].body as string);
    expect(body.steps).toEqual(["doublet_detection"]);
    expect(body.params.doublet_method).toBe("both");
  });
});
