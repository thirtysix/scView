import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";

const { apiFetch, getQcByGroup, getFilterPreview } = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  getQcByGroup: vi.fn(),
  getFilterPreview: vi.fn(),
}));
vi.mock("@/api/client", () => ({ apiFetch }));
vi.mock("@/api/qc", () => ({ getQcByGroup, getFilterPreview }));
// Plotly needs a real canvas: each plot reports its traces and y axis type.
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data, layout }: { data: { type: string }[]; layout?: { yaxis?: { type?: string } } }) => (
    <div data-testid="plot" data-types={data.map((t) => t.type).join(",")} data-ytype={layout?.yaxis?.type ?? ""} />
  ),
}));

import { QcPlots } from "./QcPlots";
import { useDatasetStore } from "@/stores/datasetStore";

const metric = { min: 0, max: 10, mean: 5, median: 5, q1: 3, q3: 7, hist: { bin_edges: [0, 5, 10], counts: [4, 6] } };

beforeEach(() => {
  apiFetch.mockReset().mockResolvedValue({
    n_cells: 10,
    computed_on_demand: false,
    has_mt: true,
    metrics: { n_genes_by_counts: metric, total_counts: metric },
    scatter: { x: [1], y: [2], color: [], x_label: "total_counts", y_label: "n_genes_by_counts", color_label: "", n_shown: 10 },
  });
  getQcByGroup.mockReset().mockResolvedValue({
    column: null, reason: null, n_cells: 10, n_groups: 0, max_groups: 50, groups: [], metrics: {}, scatter: null,
  });
  useDatasetStore.setState({
    currentDatasetId: "d1",
    currentDataset: { id: "d1", obs_columns: [] } as unknown as DatasetInfo,
  });
});

describe("QcPlots histogram log scale", () => {
  it("puts the histograms' cell counts on a log axis when ticked", async () => {
    render(<QcPlots datasetId="d1" />);
    const toggle = await screen.findByLabelText("Histograms on a log scale");
    const histograms = () => screen.getAllByTestId("plot").filter((p) => p.dataset.types === "bar,scatter");
    expect(histograms()).toHaveLength(2);
    expect(histograms().map((p) => p.dataset.ytype)).toEqual(["linear", "linear"]);
    fireEvent.click(toggle);
    expect(histograms().map((p) => p.dataset.ytype)).toEqual(["log", "log"]);
  });
});
