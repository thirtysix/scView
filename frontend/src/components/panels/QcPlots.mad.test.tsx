import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import type { FilterThresholds } from "@/api/qc";

const { apiFetch, getQcByGroup, getFilterPreview } = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  getQcByGroup: vi.fn(),
  getFilterPreview: vi.fn(),
}));
vi.mock("@/api/client", () => ({ apiFetch }));
vi.mock("@/api/qc", () => ({ getQcByGroup, getFilterPreview }));
vi.mock("@/components/plots/Plot", () => ({ default: () => <div data-testid="plot" /> }));

import { histThresholds, QcPlots } from "./QcPlots";

const metric = { min: 0, max: 10, mean: 5, median: 5, q1: 3, q3: 7, hist: { bin_edges: [0, 5, 10], counts: [4, 6] } };
const POOLED = {
  n_cells: 10,
  computed_on_demand: false,
  has_mt: true,
  metrics: { n_genes_by_counts: metric, total_counts: metric, pct_counts_mt: metric },
  scatter: { x: [1], y: [2], color: [0.5], x_label: "total_counts", y_label: "n_genes_by_counts", color_label: "pct_counts_mt", n_shown: 10 },
};
const FIXED: FilterThresholds = { min_genes: 200, max_genes: 0, min_counts: 0, max_pct_mt: 20, drop_doublets: false };
const MAD: FilterThresholds = { ...FIXED, qc_threshold_mode: "mad", qc_mad_k: 5, qc_mad_k_mt: 3, qc_sample_column: "" };

describe("histThresholds", () => {
  it("carries every cut-off in fixed mode and only the % mito ceiling in MAD mode", () => {
    expect(histThresholds("n_genes_by_counts", FIXED)?.map((t) => [t.param, t.value])).toEqual([
      ["min_genes", 200],
      ["max_genes", 0],
    ]);
    expect(histThresholds("n_genes_by_counts", MAD)).toBeUndefined();
    expect(histThresholds("total_counts", MAD)).toBeUndefined();
    expect(histThresholds("pct_counts_mt", MAD)).toEqual([{ param: "max_pct_mt", label: "Ceiling", value: 20 }]);
    expect(histThresholds("pct_counts_mt", null)).toBeUndefined();
  });
});

describe("QcPlots in MAD mode", () => {
  beforeEach(() => {
    apiFetch.mockReset().mockResolvedValue(POOLED);
    getQcByGroup.mockReset().mockResolvedValue({
      column: null, reason: null, n_cells: 10, n_groups: 0, max_groups: 50, groups: [], metrics: {}, scatter: null,
    });
    getFilterPreview.mockReset().mockResolvedValue({
      n_cells: 10, n_removed: 1, n_kept: 9, fraction_removed: 0.1,
      thresholds: { ...FIXED, mode: "mad", mad_k: 5, mad_k_mt: 3 }, has_mt: true, doublet_calls: false,
      by_criterion: { max_counts: { label: "Max counts", n_removed: 1 } },
      mad: {
        column: null, reason: null, k: 5, k_mt: 3, ceiling_pct_mt: 20, min_sample_cells: 100, samples: [],
        pooled: { n_cells: 10, min_counts: 100, max_counts: 900, min_genes: 50, max_genes: 400, max_pct_mt: 8, skipped: [] },
        n_unlabelled: 0, reused_from: null,
      },
      samples: null, clusters: null, warnings: [], previous: null,
    });
  });

  it("puts only the % mito ceiling on the histograms and the MAD rule in the summary", async () => {
    render(<QcPlots datasetId="d1" filter={{ values: MAD, onChange: vi.fn() }} />);
    expect(await screen.findByLabelText("Ceiling max_pct_mt")).toHaveValue(20);
    expect(screen.queryByLabelText("Min min_genes")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Min min_counts")).not.toBeInTheDocument();
    expect(await screen.findByTestId("mad-controls")).toHaveTextContent(/each sample's outliers/);
    expect(getFilterPreview).toHaveBeenCalledWith("d1", MAD);
  });
});
