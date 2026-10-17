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
  available_embeddings: [], obs_columns: [{ name: "Donor", dtype: "category", n_unique: 3 }], created_at: "",
} as unknown as DatasetInfo;

function assessment(mode: string | null) {
  const done = { done: true, confidence: "high", details: "Run by scView (recorded)" };
  return { ...Object.fromEntries(STEPS.map((s) => [s, done])), normalization_method: null, qc_threshold_mode: mode };
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

describe("DataAssessmentPanel filtering mode", () => {
  const fetchSpy = vi.fn();

  beforeEach(() => {
    apiFetch.mockReset();
    fetchSpy.mockReset().mockImplementation(async () => streamResponse());
    vi.stubGlobal("fetch", fetchSpy);
    useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: DATASET });
  });
  afterEach(() => vi.unstubAllGlobals());

  async function openFiltering(mode: string | null) {
    apiFetch.mockImplementation(async (path: string) => (path.endsWith("/assessment") ? assessment(mode) : DATASET));
    renderPanel();
    await waitFor(() => screen.getByText("Cell/Gene Filtering"));
    fireEvent.click(screen.getByText("Cell/Gene Filtering"));
  }

  const input = (label: string) =>
    screen.getByText(label).parentElement!.querySelector("input, select") as HTMLInputElement;

  it("defaults a dataset scView never filtered to per-sample MAD, greying out the fixed cut-offs", async () => {
    await openFiltering(null);
    expect(input("Cell thresholds")).toHaveValue("mad");
    expect(input("Min genes/cell")).toBeDisabled();
    expect(input("MADs for counts / genes")).not.toBeDisabled();
    expect(input("MADs for counts / genes")).toHaveValue(5);
    expect(screen.getAllByText("Not used: per-sample MAD mode.")).toHaveLength(3);
    const samples = input("Samples");
    expect([...samples.querySelectorAll("option")].map((o) => o.textContent)).toEqual([
      "Detect (donor / sample column)",
      "Donor",
    ]);
    // Untouched params send nothing: the backend's default is MAD too.
    fireEvent.click(screen.getByRole("button", { name: /re-run this step/i }));
    await waitFor(() => expect(fetchSpy).toHaveBeenCalled());
    expect(JSON.parse(fetchSpy.mock.calls[0]![1].body as string)).toEqual({ steps: ["filtering"], params: null });
  });

  it("keeps the fixed cut-offs of a dataset filtered with them before MAD mode", async () => {
    await openFiltering("fixed");
    expect(input("Cell thresholds")).toHaveValue("fixed");
    expect(input("Min genes/cell")).not.toBeDisabled();
    expect(input("MADs for % mito")).toBeDisabled();
    expect(screen.getAllByText("Not used: fixed cut-offs.")).toHaveLength(3);
    fireEvent.click(screen.getByRole("button", { name: /re-run this step/i }));
    await waitFor(() => expect(fetchSpy).toHaveBeenCalled());
    expect(JSON.parse(fetchSpy.mock.calls[0]![1].body as string)).toEqual({
      steps: ["filtering"],
      params: { qc_threshold_mode: "fixed" },
    });
  });

  it("switches mode from the step's own select", async () => {
    await openFiltering(null);
    fireEvent.change(input("Cell thresholds"), { target: { value: "fixed" } });
    expect(input("Min genes/cell")).not.toBeDisabled();
    expect(input("MADs for counts / genes")).toBeDisabled();
  });
});
