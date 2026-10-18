import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const apiFetch = vi.fn();
vi.mock("@/api/client", () => ({ apiFetch: (...a: unknown[]) => apiFetch(...a) }));
const plotProps = vi.fn();
vi.mock("@/components/plots/Plot", () => ({
  default: (p: unknown) => {
    plotProps(p);
    return <div data-testid="plot" />;
  },
}));

import { MarkerPlot } from "./MarkerPlot";

interface Props {
  data: { y: unknown[]; xaxis?: string }[];
  layout: { yaxis: { range?: number[]; categoryarray: string[] }; xaxis2?: unknown };
}
const last = () => plotProps.mock.calls.at(-1)![0] as Props;

const BY_GROUP = {
  groupby: "cluster", layer: "raw", groups: ["A", "B", "C"], genes: ["G1"],
  mean: [[1], [2], [3]], mean_expressing: [[1], [2], [3]], fraction: [[1], [1], [1]], scale_range: [0, 5],
};
// B and C close, A far: leaves A, B, C reversed to C, B, A by this order.
const DISTANCES = {
  column: "cluster", groups: ["A", "B", "C"], sizes: [10, 10, 10],
  distances: [[0, 9, 9], [9, 0, 1], [9, 1, 0]],
  order: [2, 1, 0], tree: [[1, 2, 1, 2], [0, 3, 9, 3]],
  representation: "X_pca", n_pcs: 30, rep_source: "graph", linkage: "complete", metric: "euclidean", n_left_out: 0,
  correlation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], correlation_method: "pearson",
};

function respond(distances: () => Promise<unknown>) {
  apiFetch.mockImplementation((url: string) =>
    url.includes("/clustering/distances") ? distances() : Promise.resolve(BY_GROUP),
  );
}

function renderPlot(groups = ["A", "B", "C"]) {
  return render(
    <MarkerPlot datasetId="ds1" groupby="cluster" groups={groups} markers={{}} kind="dotplot" perGroup={1} genes={["G1"]} />,
  );
}

describe("MarkerPlot ordered by dendrogram", () => {
  beforeEach(() => {
    apiFetch.mockReset();
    plotProps.mockReset();
  });

  it("asks for the tree only when asked, then orders the rows by it and draws it", async () => {
    respond(() => Promise.resolve(DISTANCES));
    renderPlot();
    await screen.findByTestId("plot");
    expect(last().layout.yaxis.categoryarray).toEqual(["A", "B", "C"]);
    expect(apiFetch.mock.calls.some(([u]) => String(u).includes("distances"))).toBe(false);

    fireEvent.click(screen.getByLabelText("Order groups by dendrogram"));
    await waitFor(() => expect(last().layout.yaxis.categoryarray).toEqual(["C", "B", "A"]));
    expect(String(apiFetch.mock.calls.at(-1)![0])).toContain("/datasets/ds1/clustering/distances?column=cluster");
    const tree = last().data.at(-1)!;
    expect(tree.xaxis).toBe("x2");
    expect(last().layout.yaxis.range).toEqual([2.5, -0.5]);
    expect(screen.getByText(/Groups ordered by the complete-linkage clustering/)).toHaveTextContent("tree on the right");

    fireEvent.click(screen.getByLabelText("Order groups by dendrogram"));
    expect(last().layout.yaxis.categoryarray).toEqual(["A", "B", "C"]);
    expect(last().layout.xaxis2).toBeUndefined();
  });

  it("says why it can't order, and keeps the plot", async () => {
    respond(() => Promise.reject(new Error('API error 400: {"detail":"No principal components (X_pca) in this dataset."}')));
    renderPlot();
    await screen.findByTestId("plot");
    fireEvent.click(screen.getByLabelText("Order groups by dendrogram"));
    expect(await screen.findByRole("alert")).toHaveTextContent("Can't order by dendrogram: No principal components");
    expect(last().layout.yaxis.categoryarray).toEqual(["A", "B", "C"]);
  });
});
