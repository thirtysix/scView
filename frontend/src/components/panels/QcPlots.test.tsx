import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { QcByGroup } from "@/api/qc";
import type { DatasetInfo } from "@/api/types";

const { apiFetch, getQcByGroup } = vi.hoisted(() => ({ apiFetch: vi.fn(), getQcByGroup: vi.fn() }));
vi.mock("@/api/client", () => ({ apiFetch }));
vi.mock("@/api/qc", () => ({ getQcByGroup }));
// Plotly needs a real canvas; the traces it is handed are what matters here.
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data }: { data: { name?: string; type: string }[] }) => (
    <div data-testid="plot" data-types={data.map((t) => t.type).join(",")}>
      {data.map((t) => t.name ?? "").join(",")}
    </div>
  ),
}));

import { QcPlots } from "./QcPlots";
import { useDatasetStore } from "@/stores/datasetStore";

const metric = { min: 0, max: 10, mean: 5, median: 5, q1: 3, q3: 7, hist: { bin_edges: [0, 5, 10], counts: [4, 6] } };

const POOLED = {
  n_cells: 10,
  computed_on_demand: false,
  has_mt: true,
  metrics: { n_genes_by_counts: metric, total_counts: metric, pct_counts_mt: metric },
  scatter: { x: [1], y: [2], color: [0.5], x_label: "total_counts", y_label: "n_genes_by_counts", color_label: "pct_counts_mt", n_shown: 10 },
};

function byGroup(column: string, reason: QcByGroup["reason"], names: string[]): QcByGroup {
  const values = Object.fromEntries(names.map((n) => [n, [1, 2, 3]]));
  return {
    column, reason, n_cells: 10, n_groups: names.length, max_groups: 50, n_unlabelled: 0,
    has_mt: true, groups: names.map((n, i) => ({ name: n, n_cells: 3 + i })),
    metrics: { n_genes_by_counts: values, total_counts: values, pct_counts_mt: values },
    scatter: { x: [1, 2], y: [3, 4], group: [0, 1], x_label: "total_counts", y_label: "n_genes_by_counts", n_shown: 2 },
  };
}

describe("QcPlots group by", () => {
  beforeEach(() => {
    apiFetch.mockReset().mockResolvedValue(POOLED);
    getQcByGroup.mockReset();
    useDatasetStore.setState({
      currentDatasetId: "d1",
      currentDataset: {
        id: "d1",
        obs_columns: [
          { name: "Donor", dtype: "category", n_unique: 2, values: ["d1", "d2"] },
          { name: "cluster", dtype: "category", n_unique: 3, values: ["0", "1", "2"] },
          { name: "fine", dtype: "category", n_unique: 70 },
        ],
      } as unknown as DatasetInfo,
    });
  });

  it("pre-selects the backend's donor column and draws violins and a scatter by donor", async () => {
    getQcByGroup.mockResolvedValue(byGroup("Donor", "donor", ["d1", "d2"]));
    render(<QcPlots datasetId="d1" />);
    expect(await screen.findByText(/the donor \/ sample column/)).toBeInTheDocument();
    expect(getQcByGroup).toHaveBeenCalledWith("d1", undefined);
    expect(screen.getByLabelText("Group by")).toHaveValue("Donor");
    expect(screen.getByText("Genes per cell by Donor")).toBeInTheDocument();
    expect(screen.getByText("% Mitochondrial by Donor")).toBeInTheDocument();
    expect(screen.getByText(/smallest d1 with 3 cells/)).toBeInTheDocument();
    const plots = screen.getAllByTestId("plot");
    expect(plots.filter((p) => p.dataset.types === "violin,violin")).toHaveLength(3);
    expect(plots.some((p) => p.dataset.types === "scattergl,scattergl" && p.textContent === "d1,d2")).toBe(true);
    // Over the cap: listed, but not pickable.
    expect(screen.getByRole("option", { name: /fine \(70 groups: over 50/ })).toBeDisabled();
  });

  it("goes back to pooled cells on None, and asks for a column the user picks", async () => {
    getQcByGroup.mockResolvedValue(byGroup("Donor", "donor", ["d1", "d2"]));
    render(<QcPlots datasetId="d1" />);
    await screen.findByText(/the donor \/ sample column/);

    fireEvent.change(screen.getByLabelText("Group by"), { target: { value: "" } });
    expect(screen.queryByText("Genes per cell by Donor")).not.toBeInTheDocument();
    expect(screen.getByText("Counts vs. genes per cell (colour = % mito)")).toBeInTheDocument();
    expect(getQcByGroup).toHaveBeenCalledTimes(1);

    getQcByGroup.mockResolvedValue(byGroup("cluster", "requested", ["0", "1", "2"]));
    fireEvent.change(screen.getByLabelText("Group by"), { target: { value: "cluster" } });
    expect(await screen.findByText("Genes per cell by cluster")).toBeInTheDocument();
    expect(getQcByGroup).toHaveBeenLastCalledWith("d1", "cluster");
  });

  it("shows the server's reason when a column is refused", async () => {
    getQcByGroup.mockResolvedValueOnce(byGroup("Donor", "donor", ["d1", "d2"]));
    render(<QcPlots datasetId="d1" />);
    await screen.findByText(/the donor \/ sample column/);
    getQcByGroup.mockRejectedValueOnce(new Error("'cluster' has 60 groups; QC by group shows at most 50."));
    fireEvent.change(screen.getByLabelText("Group by"), { target: { value: "cluster" } });
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("at most 50"));
    expect(screen.getByText("Counts vs. genes per cell (colour = % mito)")).toBeInTheDocument();
  });
});
