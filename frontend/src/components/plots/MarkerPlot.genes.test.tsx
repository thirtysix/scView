import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

const apiFetch = vi.fn();
vi.mock("@/api/client", () => ({ apiFetch: (...a: unknown[]) => apiFetch(...a) }));
// Plotly needs a real browser; the test only checks what MarkerPlot asks for and says.
vi.mock("@/components/plots/Plot", () => ({ default: () => <div data-testid="plot" /> }));

import { MarkerPlot } from "./MarkerPlot";

const markers = { A: [{ gene: "TOP1", logfoldchange: 2 }], B: [{ gene: "TOP2", logfoldchange: 2 }] };

function renderPlot(genes?: string[]) {
  return render(
    <MarkerPlot
      datasetId="ds1"
      groupby="leiden"
      groups={["A", "B"]}
      markers={markers}
      kind="dotplot"
      perGroup={1}
      genes={genes}
    />,
  );
}

describe("MarkerPlot with chosen genes", () => {
  // Braces: a function returned from beforeEach would run as a teardown.
  beforeEach(() => {
    apiFetch.mockReset();
  });

  it("plots the top markers by default", async () => {
    apiFetch.mockResolvedValue({
      groupby: "leiden", layer: "X", groups: ["A", "B"], genes: ["TOP1", "TOP2"],
      mean: [[1, 0], [0, 1]], fraction: [[1, 0], [0, 1]],
    });
    renderPlot();
    expect(await screen.findByTestId("plot")).toBeInTheDocument();
    const url = String(apiFetch.mock.calls[0]![0]);
    expect(url).toContain("genes=TOP1");
    expect(url).toContain("genes=TOP2");
    expect(screen.getByText(/Top 1 up-regulated markers/)).toBeInTheDocument();
  });

  it("asks for the chosen genes and names the ones the dataset lacks", async () => {
    apiFetch.mockResolvedValue({
      groupby: "leiden", layer: "X", groups: ["A", "B"], genes: ["CD3E"],
      mean: [[1], [0]], fraction: [[1], [0]],
    });
    renderPlot(["CD3E", "cd4"]);
    expect(await screen.findByTestId("plot")).toBeInTheDocument();
    const url = String(apiFetch.mock.calls[0]![0]);
    expect(url).toContain("genes=CD3E");
    expect(url).toContain("genes=cd4");
    expect(url).not.toContain("TOP1");
    expect(screen.getByText(/not plotted/)).toHaveTextContent("cd4");
    expect(screen.getByText(/Your 1 gene across/)).toBeInTheDocument();
  });

  it("says so when none of the chosen genes are in the dataset", async () => {
    apiFetch.mockRejectedValue(new Error('API error 404: {"detail":"None of the requested genes are in this layer."}'));
    renderPlot(["FOO", "BAR"]);
    expect(await screen.findByText(/None of these genes are in this dataset/)).toHaveTextContent("FOO, BAR");
  });
});
