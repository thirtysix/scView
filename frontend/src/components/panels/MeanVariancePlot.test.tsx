import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { MeanVariance } from "@/api/normalization";

const { getMeanVariance } = vi.hoisted(() => ({ getMeanVariance: vi.fn() }));
vi.mock("@/api/normalization", () => ({ getMeanVariance }));
// Plotly needs a real canvas; the traces it is handed are what matters here.
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data }: { data: { name?: string }[] }) => (
    <div data-testid="plot">{data.map((t) => t.name ?? "").join(",")}</div>
  ),
}));

import { MeanVariancePlot } from "./MeanVariancePlot";

const DATA: MeanVariance = {
  n_cells: 30000,
  n_cells_used: 20000,
  n_genes: 3,
  n_genes_undetected: 0,
  theta: 100,
  clip: 141.4,
  counts_source: "layers['counts']",
  normalization_method: "pearson_residuals",
  genes: ["A", "B", "C"],
  mean: [0.1, 1, 10],
  var_counts: [0.1, 2, 30],
  var_log: [0.05, 0.3, 0.6],
  var_residuals: [1, 4, 1.1],
  hvg: [false, true, false],
};

describe("MeanVariancePlot", () => {
  beforeEach(() => {
    getMeanVariance.mockReset().mockResolvedValue(DATA);
  });

  it("computes only on request, then shows before and after", async () => {
    render(<MeanVariancePlot datasetId="d1" />);
    expect(getMeanVariance).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /show mean–variance/i }));
    await waitFor(() => expect(screen.getByText("Before: counts")).toBeTruthy());
    expect(getMeanVariance).toHaveBeenCalledWith("d1");
    expect(screen.getByText("After: Pearson residuals")).toBeTruthy();
    const plots = screen.getAllByTestId("plot").map((p) => p.textContent);
    expect(plots[0]).toContain("Poisson (var = mean)");
    expect(plots[1]).toContain("Sampling noise (1)");
    expect(screen.getByText(/20,000 sampled of 30,000/)).toBeTruthy();
    expect(screen.getByText(/Analytic Pearson residuals/)).toBeTruthy();
  });

  it("switches the before panel to the shifted log", async () => {
    render(<MeanVariancePlot datasetId="d1" />);
    fireEvent.click(screen.getByRole("button", { name: /show mean–variance/i }));
    await waitFor(() => screen.getByText("Before: counts"));
    fireEvent.click(screen.getByRole("button", { name: "Shifted log" }));
    expect(screen.getByText("Before: shifted log")).toBeTruthy();
    expect(screen.getAllByTestId("plot")[0]!.textContent).not.toContain("Poisson");
  });

  it("says why when the dataset has no counts", async () => {
    getMeanVariance.mockRejectedValue(new Error("Pearson residuals need raw counts"));
    render(<MeanVariancePlot datasetId="d1" />);
    fireEvent.click(screen.getByRole("button", { name: /show mean–variance/i }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toMatch(/need raw counts/));
  });
});
