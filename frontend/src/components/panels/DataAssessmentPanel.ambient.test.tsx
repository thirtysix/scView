import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { DatasetInfo } from "@/api/types";

const { apiFetch } = vi.hoisted(() => ({ apiFetch: vi.fn() }));
vi.mock("@/api/client", async () => ({ ...(await vi.importActual<object>("@/api/client")), apiFetch }));
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
  "qc_metrics", "doublet_detection", "filtering", "ambient_correction", "normalization", "log_transform",
  "highly_variable_genes", "scaling", "pca", "batch_correction", "neighbors", "clustering",
  "embeddings", "marker_genes", "enrichment", "cell_cycle", "dissociation_stress",
];
const MISSING = new Set(["qc_metrics", "doublet_detection", "ambient_correction"]);

const DATASET = {
  id: "d1", name: "d1", filename: "d1.h5ad", n_cells: 10, n_genes: 5, status: "ready",
  available_embeddings: [], obs_columns: [], created_at: "",
} as unknown as DatasetInfo;

const AMBIENT = {
  available: true, approximate: false, column: "leiden", counts_from: "X",
  samples: [{
    sample: "s1", column: null, n_cells: 10, source: "empty_droplets", top_genes: [{ gene: "LYZ", share: 0.02 }],
    genes_covered: 1, umi_max: 100, rho: 0.05, rho_low: 0.03, rho_high: 0.07, rho_status: "ok",
    n_markers: 50, n_estimates: 200, rho_genes: [], notes: [],
  }],
  markers: { column: "leiden", approximate: false, groups: {}, n_flagged: 0 },
  hb_note: null, notes: [], correction: null, help: {}, method: "", url: "",
};

function assessment() {
  return {
    ...Object.fromEntries(STEPS.map((s) => [s, { done: !MISSING.has(s), confidence: "high", details: "" }])),
    normalization_method: null,
    qc_threshold_mode: null,
  };
}

function renderPanel() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <DataAssessmentPanel />
    </QueryClientProvider>,
  );
}

describe("DataAssessmentPanel ambient RNA correction", () => {
  const fetchSpy = vi.fn();

  beforeEach(() => {
    apiFetch.mockReset().mockImplementation(async (path: string) => {
      if (path.endsWith("/assessment")) return assessment();
      if (path.includes("/qc/ambient")) return AMBIENT;
      if (path.includes("/qc/doublets")) return { suggestion: null };
      return DATASET;
    });
    fetchSpy.mockReset().mockImplementation(async () => new Response("", { status: 200 }));
    vi.stubGlobal("fetch", fetchSpy);
    vi.stubGlobal("confirm", () => true);
    useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: DATASET });
  });
  afterEach(() => vi.unstubAllGlobals());

  it("is listed after filtering but never selected for the user", async () => {
    renderPanel();
    // Only the two missing default steps: the opt-in correction is left out.
    expect(await screen.findByRole("button", { name: /Run 2 Steps/ })).toBeTruthy();
    expect(screen.getByText("Ambient RNA Correction")).toBeTruthy();
  });

  it("the Ambient RNA card turns it on, and Reprocess from scratch includes it only then", async () => {
    renderPanel();
    await screen.findByRole("button", { name: /Run 2 Steps/ });
    fireEvent.click(screen.getByRole("button", { name: /Reprocess/ }));
    await waitFor(() => expect(fetchSpy).toHaveBeenCalled());
    const first = JSON.parse(String((fetchSpy.mock.calls[0]![1] as RequestInit).body)) as { steps: string[] };
    expect(first.steps).not.toContain("ambient_correction");
    expect(first.steps[0]).toBe("reset_to_counts");

    fetchSpy.mockClear();
    const toggle = (await screen.findByTestId("ambient-step-toggle")) as HTMLInputElement;
    fireEvent.click(toggle);
    expect(await screen.findByRole("button", { name: /Run 3 Steps/ })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Reprocess/ }));
    await waitFor(() => expect(fetchSpy).toHaveBeenCalled());
    const second = JSON.parse(String((fetchSpy.mock.calls[0]![1] as RequestInit).body)) as { steps: string[] };
    expect(second.steps).toContain("ambient_correction");
    expect(second.steps.indexOf("ambient_correction")).toBe(second.steps.indexOf("filtering") + 1);
  });
});
