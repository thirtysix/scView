import { useState } from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { QcByGroup } from "@/api/qc";
import type { DatasetInfo } from "@/api/types";

const { apiFetch, getQcByGroup, getFilterPreview } = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  getQcByGroup: vi.fn(),
  getFilterPreview: vi.fn(),
}));
vi.mock("@/api/client", () => ({ apiFetch }));
vi.mock("@/api/qc", () => ({ getQcByGroup, getFilterPreview }));
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

describe("QcPlots thresholds", () => {
  const values = { min_genes: 200, max_genes: 0, min_counts: 0, max_pct_mt: 20, drop_doublets: false };

  beforeEach(() => {
    apiFetch.mockReset().mockResolvedValue(POOLED);
    getQcByGroup.mockReset().mockResolvedValue(byGroup("Donor", "donor", ["d1", "d2"]));
    getFilterPreview.mockReset().mockResolvedValue({
      n_cells: 10, n_removed: 2, n_kept: 8, fraction_removed: 0.2, thresholds: values, has_mt: true,
      doublet_calls: false, by_criterion: { min_genes: { label: "Min genes", n_removed: 2 } },
      samples: null, clusters: null, warnings: [],
    });
  });

  /** The panel's part: it owns the values, so they follow each change. */
  function Owner({ onChange }: { onChange: (patch: Partial<typeof values>) => void }) {
    const [v, setV] = useState(values);
    return (
      <QcPlots
        datasetId="d1"
        filter={{
          values: v,
          onChange: (patch) => {
            onChange(patch);
            setV((prev) => ({ ...prev, ...patch }));
          },
        }}
      />
    );
  }

  it("puts the filtering thresholds on the histograms and says what they remove", async () => {
    const onChange = vi.fn();
    render(<Owner onChange={onChange} />);
    expect(await screen.findByLabelText("Min min_genes")).toHaveValue(200);
    expect(screen.getByLabelText("Max max_genes")).toHaveValue(null); // off
    expect(screen.getByLabelText("Min min_counts")).toHaveValue(null);
    expect(screen.getByLabelText("Max max_pct_mt")).toHaveValue(20);
    expect(await screen.findByText(/2 of 10 cells \(20.0%\)/)).toBeInTheDocument();
    expect(getFilterPreview).toHaveBeenCalledWith("d1", values);

    fireEvent.change(screen.getByLabelText("Min min_genes"), { target: { value: "600" } });
    expect(onChange).toHaveBeenLastCalledWith({ min_genes: 600 });
    fireEvent.change(screen.getByLabelText("Max max_genes"), { target: { value: "4000" } });
    expect(onChange).toHaveBeenLastCalledWith({ max_genes: 4000 });
    fireEvent.change(screen.getByLabelText("Max max_genes"), { target: { value: "" } });
    expect(onChange).toHaveBeenLastCalledWith({ max_genes: 0 }); // emptied = off
    const calls = onChange.mock.calls.length;
    fireEvent.change(screen.getByLabelText("Max max_pct_mt"), { target: { value: "" } });
    expect(onChange).toHaveBeenCalledTimes(calls); // % mito has no off: wait for a number
  });

  it("shows a value set elsewhere in an input that still has the focus", async () => {
    const onChange = vi.fn();
    const { rerender } = render(<QcPlots datasetId="d1" filter={{ values, onChange }} />);
    const input = await screen.findByLabelText("Min min_genes");
    fireEvent.change(input, { target: { value: "600" } });
    rerender(<QcPlots datasetId="d1" filter={{ values: { ...values, min_genes: 600 }, onChange }} />);
    expect(input).toHaveValue(600);
    // A drag on the plot keeps the input focused and sets 750.
    rerender(<QcPlots datasetId="d1" filter={{ values: { ...values, min_genes: 750 }, onChange }} />);
    expect(input).toHaveValue(750);
  });

  it("has no thresholds without a filter to set", async () => {
    render(<QcPlots datasetId="d1" />);
    await screen.findByText("Genes per cell");
    expect(screen.queryByLabelText("Min min_genes")).not.toBeInTheDocument();
    expect(getFilterPreview).not.toHaveBeenCalled();
  });
});
