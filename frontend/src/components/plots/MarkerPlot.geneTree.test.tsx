import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const apiFetch = vi.fn();
vi.mock("@/api/client", () => ({ apiFetch: (...a: unknown[]) => apiFetch(...a) }));
// Plotly needs a real browser; the test checks what MarkerPlot draws.
const plotProps = vi.fn();
vi.mock("@/components/plots/Plot", () => ({
  default: (p: unknown) => {
    plotProps(p);
    return <div data-testid="plot" />;
  },
}));

import { MarkerPlot } from "./MarkerPlot";

interface Trace {
  type: string;
  x: unknown[];
  y: unknown[];
  z?: unknown[][];
  xaxis?: string;
  yaxis?: string;
  colorscale?: [number, string][];
}
interface Props {
  data: Trace[];
  layout: Record<string, { domain?: number[]; categoryarray?: string[] }>;
}
const last = () => plotProps.mock.calls.at(-1)![0] as Props;

// Profiles of G0..G3 across groups A, B, C (columns of `mean`): G1 = 2 x G0,
// G2 = G0 reversed, G3 = [1, 3, 2]. Average linkage on 1 - r joins G0 + G1
// (d 0), then G3 (d 0.5), then G2 (d 11/6): order G0, G1, G3, G2.
const BY_GROUP = {
  groupby: "cluster", layer: "raw", groups: ["A", "B", "C"], genes: ["G0", "G1", "G2", "G3"],
  mean: [[1, 2, 3, 1], [2, 4, 2, 3], [3, 6, 1, 2]],
  mean_expressing: [[1, 2, 3, 1], [2, 4, 2, 3], [3, 6, 1, 2]],
  fraction: [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
  scale_range: [0, 5],
};
const MARKERS = {
  A: [{ gene: "G0", logfoldchange: 2 }, { gene: "G1", logfoldchange: 1 }],
  B: [{ gene: "G2", logfoldchange: 2 }],
  C: [{ gene: "G3", logfoldchange: 2 }],
};

function renderPlot(kind: "heatmap" | "dotplot", genes?: string[]) {
  return render(
    <MarkerPlot datasetId="ds1" groupby="cluster" groups={["A", "B", "C"]} markers={MARKERS} kind={kind} perGroup={2} genes={genes} />,
  );
}

describe("MarkerPlot heatmap: marker strip and gene tree", () => {
  beforeEach(() => {
    apiFetch.mockReset();
    plotProps.mockReset();
    apiFetch.mockResolvedValue(BY_GROUP);
  });

  it("keeps top markers in blocks by group and marks each gene's group above", async () => {
    renderPlot("heatmap");
    await screen.findByTestId("plot");
    expect(last().layout.xaxis!.categoryarray).toEqual(["G0", "G1", "G2", "G3"]);
    const strip = last().data.find((t) => t.yaxis === "y4")!;
    expect(strip.x).toEqual(["G0", "G1", "G2", "G3"]);
    expect(strip.z).toEqual([[0, 0, 1, 2]]); // A, A, B, C
    expect(strip.colorscale).toHaveLength(6); // one flat band per group
    // above the heatmap, which no longer fills the height
    expect(last().layout.yaxis4!.domain![0]).toBeGreaterThan(last().layout.yaxis!.domain![1]!);
    expect(screen.getByText(/Strip above: the group each gene is a top marker of/)).toBeTruthy();
  });

  it("orders the genes by their clustering and draws the tree above them", async () => {
    renderPlot("heatmap");
    await screen.findByTestId("plot");
    fireEvent.click(screen.getByLabelText("Cluster genes"));
    await waitFor(() => expect(last().layout.xaxis!.categoryarray).toEqual(["G0", "G1", "G3", "G2"]));
    const heat = last().data[0]!;
    expect(heat.x).toEqual(["G0", "G1", "G3", "G2"]);
    // the columns move with their genes: row A is G0, G1, G3, G2 = 1, 2, 1, 3
    expect((heat.z as number[][])[0]!.map((v) => Math.round(v * 100) / 100)).toEqual([0, 0, 0, 1]);
    const tree = last().data.find((t) => t.xaxis === "x3")!;
    expect(tree.type).toBe("scatter");
    expect(last().layout.yaxis3!.domain![1]).toBe(1);
    // the strip follows the genes
    expect(last().data.find((t) => t.yaxis === "y4")!.z).toEqual([[0, 0, 2, 1]]);
    expect(screen.getByText(/average-linkage clustering of their mean over all cells/)).toBeTruthy();

    fireEvent.click(screen.getByLabelText("Cluster genes"));
    expect(last().layout.xaxis!.categoryarray).toEqual(["G0", "G1", "G2", "G3"]);
    expect(last().data.some((t) => t.xaxis === "x3")).toBe(false);
  });

  it("draws no strip for the user's genes, and keeps the dot plot as it was", async () => {
    renderPlot("heatmap", ["G0", "G1", "G2", "G3"]);
    await screen.findByTestId("plot");
    expect(last().data.some((t) => t.yaxis === "y4")).toBe(false);
    expect(last().layout.yaxis!.domain).toBeUndefined();
    fireEvent.click(screen.getByLabelText("Cluster genes")); // still clusters the user's genes
    await waitFor(() => expect(last().layout.xaxis!.categoryarray).toEqual(["G0", "G1", "G3", "G2"]));
  });

  it("offers no gene clustering on the dot plot", async () => {
    renderPlot("dotplot");
    await screen.findByTestId("plot");
    expect(screen.queryByLabelText("Cluster genes")).toBeNull();
    expect(last().data.some((t) => t.yaxis === "y4")).toBe(false);
  });
});
