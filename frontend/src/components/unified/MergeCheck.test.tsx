import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const api = vi.hoisted(() => ({
  getMergeCheckInfo: vi.fn(),
  runMergeCheck: vi.fn(),
  saveMergedClustering: vi.fn(),
}));
const refresh = vi.hoisted(() => vi.fn());
vi.mock("@/api/clustering", () => api);
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => refresh }));

import type { MergeCheckInfo, MergePair, MergeResult, SweepRow } from "@/api/clustering";
import { useSettingsStore } from "@/stores/settingsStore";
import { MergeCheck } from "./MergeCheck";

const info: MergeCheckInfo = {
  available: true,
  reason: null,
  n_cells: 1000,
  merge_max_genes: 10,
  padj: 0.05,
  log2fc: 1,
  max_cells_per_side: 500,
  first_run_seconds: 30,
  reference: "scview_leiden_r1.0",
  reference_exists: true,
  n_clusters: 6,
  estimate_seconds: 95,
};

const pair = (a: string, b: string, extra: Partial<MergePair> = {}): MergePair => ({
  a,
  b,
  adjacency: 0.2,
  n_cells_a: 300,
  n_cells_b: 200,
  evidence: "held_out",
  n_genes: 2,
  up: ["CCR7"],
  down: [],
  train_pairs: ["0+1"],
  candidate: true,
  ...extra,
});

const result = (maxGenes = 10, extra: Partial<MergeResult> = {}): MergeResult => ({
  reference: "scview_leiden_r1.0",
  reference_exists: true,
  method: "leiden",
  resolution: 1,
  train_resolution: 1.4,
  match_count: true,
  n_reference_clusters: 6,
  epsilon: 0.5,
  seed: 0,
  max_genes: maxGenes,
  overdispersion: { n_genes_fit: 2000, n_cells_fit: 1000, median_theta: 2, poisson_share: 0 },
  train: { n_clusters: 6, ari: 0.7, ami: 0.75, sizes: [], n_groups: 4, clusters: [] },
  groups: [["0", "1"], ["2"], ["3"]],
  pairs: [
    pair("0", "1"),
    pair("2", "3", { candidate: false, n_genes: 300, up: ["LYZ"], down: ["CD79A"] }),
    pair("4", "5", { evidence: "together", n_genes: null, candidate: false }),
  ],
  n_candidates: 1,
  thresholds: {
    padj: 0.05,
    log2fc: 1,
    min_pct: 0.1,
    merge_max_genes: maxGenes,
    max_cells_per_side: 500,
    min_adjacency: 0.02,
    host_share: 0.5,
  },
  seconds: {},
  ...extra,
});

const rows: SweepRow[] = [
  { resolution: 0.5, n_clusters: 4, sizes: [], silhouette: null, ari_previous: null, column: "scview_leiden_r0.5", exists: false },
];

describe("MergeCheck", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useSettingsStore.setState({ colorBy: "leiden" });
    api.getMergeCheckInfo.mockResolvedValue(info);
  });

  it("runs, ticks the candidates, and saves the merged clustering", async () => {
    api.runMergeCheck.mockImplementation(async (_id, body, h) => {
      h.onStart({ cached: false, estimate_seconds: 95 });
      h.onProgress({ step: "train", label: "Re-running the pipeline on the training half" });
      h.onResult(result(body.max_genes));
    });
    api.saveMergedClustering.mockResolvedValue({ column: "scview_leiden_r1.0_merged", n_clusters: 5, groups: [] });
    render(<MergeCheck datasetId="ds" rows={rows} method="leiden" selected={null} />);
    expect(await screen.findByText(/about 1 min 35 s/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Check merges" }));
    expect(await screen.findByText(/1 merge candidate/)).toBeInTheDocument();
    expect(api.runMergeCheck.mock.calls[0]![1]).toEqual({ max_genes: 10 });
    // Candidates and never-split pairs show; the kept-apart pair waits behind "show all".
    expect(screen.getByLabelText("Merge 0 and 1")).toBeChecked();
    expect(screen.getByLabelText("Merge 4 and 5")).not.toBeChecked();
    expect(screen.queryByLabelText("Merge 2 and 3")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Show all 3/ }));
    expect(screen.getByLabelText("Merge 2 and 3")).not.toBeChecked();

    fireEvent.click(screen.getByRole("button", { name: /Save merged clustering \(6 → 5\)/ }));
    await waitFor(() => expect(useSettingsStore.getState().colorBy).toBe("scview_leiden_r1.0_merged"));
    expect(api.saveMergedClustering).toHaveBeenCalledWith("ds", { reference: "scview_leiden_r1.0", pairs: [["0", "1"]] });
    expect(refresh).toHaveBeenCalledWith("ds");
  });

  it("re-reads the run at another threshold and checks a picked sweep resolution", async () => {
    api.runMergeCheck.mockImplementation(async (_id, body, h) => h.onResult(result(body.max_genes)));
    const { rerender } = render(<MergeCheck datasetId="ds" rows={rows} method="leiden" selected={null} />);
    await screen.findByText(/about 1 min 35 s/);
    fireEvent.click(screen.getByRole("button", { name: "Check merges" }));
    await screen.findByText(/1 merge candidate/);
    fireEvent.change(screen.getByLabelText(/Genes below which/), { target: { value: "20" } });
    await waitFor(() => expect(api.runMergeCheck).toHaveBeenCalledTimes(2));
    expect(api.runMergeCheck.mock.calls[1]![1]).toEqual({ max_genes: 20 });

    rerender(<MergeCheck datasetId="ds" rows={rows} method="leiden" selected={0.5} />);
    await waitFor(() => expect(api.getMergeCheckInfo).toHaveBeenLastCalledWith("ds", { resolution: 0.5, method: "leiden" }));
    expect(screen.getByLabelText("Clustering to check")).toHaveValue("0.5");
    expect(screen.queryByText(/merge candidate/)).not.toBeInTheDocument(); // the old result is gone
  });

  it("explains why it can't run", async () => {
    api.getMergeCheckInfo.mockResolvedValue({ ...info, available: false, reason: "The merge check splits the raw counts" });
    render(<MergeCheck datasetId="ds" rows={[]} method="leiden" selected={null} />);
    expect(await screen.findByText(/splits the raw counts/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Check merges" })).toBeDisabled();
  });
});
