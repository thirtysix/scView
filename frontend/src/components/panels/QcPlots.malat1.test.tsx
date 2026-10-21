import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";

const { apiFetch, getQcByGroup, getFilterPreview } = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  getQcByGroup: vi.fn(),
  getFilterPreview: vi.fn(),
}));
vi.mock("@/api/client", () => ({ apiFetch }));
vi.mock("@/api/qc", () => ({ getQcByGroup, getFilterPreview }));
vi.mock("@/components/plots/Plot", () => ({ default: () => <div data-testid="plot" /> }));

import { QcPlots } from "./QcPlots";
import { useDatasetStore } from "@/stores/datasetStore";

const metric = { min: 0.2, max: 21.9, mean: 5.4, median: 5.52, q1: 3.69, q3: 7.21, hist: { bin_edges: [0, 10, 22], counts: [8, 2] } };

function qc(metrics: Record<string, typeof metric>) {
  return {
    n_cells: 10, computed_on_demand: false, has_mt: false, metrics,
    scatter: { x: [1], y: [2], color: [], x_label: "total_counts", y_label: "n_genes_by_counts", color_label: "", n_shown: 10 },
  };
}

beforeEach(() => {
  getQcByGroup.mockReset().mockResolvedValue({
    column: null, reason: null, n_cells: 10, n_groups: 0, max_groups: 50, groups: [], metrics: {}, scatter: null,
  });
  useDatasetStore.setState({
    currentDatasetId: "d1",
    currentDataset: { id: "d1", obs_columns: [] } as unknown as DatasetInfo,
  });
});

describe("QcPlots MALAT1 histogram", () => {
  it("shows % MALAT1 beside the others, with what it measures on hover", async () => {
    apiFetch.mockReset().mockResolvedValue(qc({ n_genes_by_counts: metric, total_counts: metric, pct_malat1: metric }));
    render(<QcPlots datasetId="d1" />);
    const title = await screen.findByText("% MALAT1");
    expect(title.getAttribute("title")).toMatch(/kept in the nucleus/);
    expect(title.getAttribute("title")).toMatch(/10\.1186\/s12864-024-11015-5/);
    expect(screen.getAllByText(/median 5\.52/).length).toBeGreaterThan(0);
  });

  it("has no MALAT1 histogram when the dataset has no MALAT1 gene", async () => {
    apiFetch.mockReset().mockResolvedValue(qc({ n_genes_by_counts: metric, total_counts: metric }));
    render(<QcPlots datasetId="d1" />);
    expect(await screen.findByText("Genes per cell")).toBeInTheDocument();
    expect(screen.queryByText("% MALAT1")).not.toBeInTheDocument();
  });
});
