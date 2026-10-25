import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { HeatmapCells } from "@/lib/cellHeatmap";

const fetchHeatmapCells = vi.fn();
vi.mock("@/api/expression", () => ({ fetchHeatmapCells: (...a: unknown[]) => fetchHeatmapCells(...a) }));
vi.mock("@/hooks/useCategoryColors", () => ({ useCategoryColors: () => () => [200, 0, 0] }));
// Plotly needs a real canvas; the traces it is handed are what matter here.
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data, config }: { data: { type: string; yaxis: string }[]; config: { toImageButtonOptions?: { filename?: string } } }) => (
    <div data-testid="plot" data-file={config.toImageButtonOptions?.filename ?? ""}>
      {data.map((t) => `${t.type}:${t.yaxis}`).join("|")}
    </div>
  ),
}));

import { CellHeatmap } from "./CellHeatmap";

const DATA: HeatmapCells = {
  meta: {
    groupby: "cell_type", layer: "raw", genes: ["CD3E", "MS4A1"], missing_genes: [], groups: ["B", "T"],
    n_sampled: [2, 1], n_total: [2, 40], n_cells: 42, scale: "none", value_range: [0, 4], seed: 0,
    per_group: 200, max_cells: 5000,
  },
  cells: Int32Array.from([10, 11, 1]),
  codes: Int32Array.from([0, 0, 1]),
  values: [Float32Array.from([0, 0.5, 3]), Float32Array.from([2, 1, 0])],
};

beforeEach(() => {
  fetchHeatmapCells.mockReset();
  fetchHeatmapCells.mockResolvedValue(DATA);
});

describe("CellHeatmap as tracks", () => {
  it("asks for the values (not z-scores) and draws a track per gene and group", async () => {
    render(<CellHeatmap datasetId="ds1" groupby="cell_type" groups={["B", "T"]} markers={{}} perGroup={5}
      genes={["CD3E", "MS4A1"]} kind="tracks" />);
    const plot = await screen.findByTestId("plot");
    expect(fetchHeatmapCells.mock.calls[0]![1]).toMatchObject({ genes: ["CD3E", "MS4A1"], scale: "none" });
    expect(plot.textContent).toBe("scatter:y|scatter:y|scatter:y2|scatter:y2|heatmap:y3");
    expect(screen.queryByLabelText("Colour:")).toBeNull();
    const caption = screen.getByTestId("cell-heatmap-caption").textContent ?? "";
    expect(caption).toContain("3 of 42 cells");
    expect(caption).toContain("in dataset order");
  });

  it("orders cells within each group by a gene", async () => {
    render(<CellHeatmap datasetId="ds1" groupby="cell_type" groups={["B", "T"]} markers={{}} perGroup={5}
      genes={["CD3E", "MS4A1"]} kind="tracks" />);
    await screen.findByTestId("plot");
    fireEvent.change(screen.getByLabelText("Cells in a group:"), { target: { value: "1" } });
    await waitFor(() =>
      expect(screen.getByTestId("cell-heatmap-caption").textContent).toContain("by MS4A1 (lowest first)"));
  });

  it("stays a heatmap by default, with its colour choice", async () => {
    render(<CellHeatmap datasetId="ds1" groupby="cell_type" groups={["B", "T"]} markers={{}} perGroup={5}
      genes={["CD3E"]} />);
    const plot = await screen.findByTestId("plot");
    expect(plot.textContent).toBe("heatmap:y|heatmap:y2");
    expect(fetchHeatmapCells.mock.calls[0]![1]).toMatchObject({ scale: "zscore" });
    expect(screen.getByLabelText("Colour:")).toBeTruthy();
  });
});
