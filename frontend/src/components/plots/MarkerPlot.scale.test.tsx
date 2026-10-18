import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const apiFetch = vi.fn();
vi.mock("@/api/client", () => ({ apiFetch: (...a: unknown[]) => apiFetch(...a) }));
// Plotly needs a real browser: keep the traces the plot was given.
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
  x: string[];
  y: string[];
  z?: (number | null)[][];
  zmin?: number;
  zmax?: number;
  marker?: { color?: unknown; cmin?: number; cmax?: number; symbol?: string; colorbar?: { title: { text: string } } };
  colorbar?: { title: { text: string } };
}
const lastTraces = () => (plotProps.mock.calls.at(-1)![0] as { data: Trace[] }).data;

const body = {
  groupby: "leiden", layer: "raw", groups: ["A", "B"], genes: ["G1", "G2"],
  mean: [[0.1, 2], [1.5, 0]],
  mean_expressing: [[2, 2.5], [3, 0]],
  fraction: [[0.03, 0.8], [0.5, 0]],
  scale_range: [0, 5],
};

function renderPlot(kind: "dotplot" | "heatmap") {
  return render(
    <MarkerPlot datasetId="ds1" groupby="leiden" groups={["A", "B"]} markers={{}} kind={kind} perGroup={1} genes={["G1", "G2"]} />,
  );
}

describe("MarkerPlot colour scale and dot sizes", () => {
  beforeEach(() => {
    apiFetch.mockReset().mockResolvedValue(body);
    plotProps.mockReset();
  });

  it("dot plot: hollow rings for too few expressing cells, and a fixed size legend", async () => {
    renderPlot("dotplot");
    await screen.findByTestId("plot");
    const [solid, hollow] = lastTraces();
    // A/G1 (3%) and B/G2 (0%) are hollow; the other two are coloured dots
    expect(solid!.x).toEqual(["G2", "G1"]);
    expect(solid!.y).toEqual(["A", "B"]);
    expect(hollow!.marker!.symbol).toBe("circle-open");
    expect(hollow!.x).toEqual(["G1", "G2"]);
    expect(screen.getByTestId("dot-size-legend")).toHaveTextContent("20%");
    expect(screen.getByTestId("dot-size-legend")).toHaveTextContent("100%");
    expect(solid!.marker!.colorbar!.title.text).toContain("scaled per gene");
  });

  it("switches to a fixed range from the server and lets the user change its top", async () => {
    renderPlot("heatmap");
    await screen.findByTestId("plot");
    fireEvent.change(screen.getByLabelText("Colour scale"), { target: { value: "fixed" } });
    let [heat] = lastTraces();
    expect([heat!.zmin, heat!.zmax]).toEqual([0, 5]);
    expect(heat!.z).toEqual(body.mean);
    expect(heat!.colorbar!.title.text).toContain("fixed 0-5");
    fireEvent.change(screen.getByLabelText("Top of the fixed colour range"), { target: { value: "3" } });
    [heat] = lastTraces();
    expect(heat!.zmax).toBe(3);
    fireEvent.click(screen.getByText(/reset \(5\)/));
    expect(lastTraces()[0]!.zmax).toBe(5);
  });

  it("colours by the mean in expressing cells, blank under the minimum share", async () => {
    renderPlot("heatmap");
    await screen.findByTestId("plot");
    fireEvent.change(screen.getByLabelText("Colour:"), { target: { value: "expressing" } });
    fireEvent.change(screen.getByLabelText("Colour scale"), { target: { value: "fixed" } });
    expect(lastTraces()[0]!.z).toEqual([[null, 2.5], [3, null]]);
    expect(screen.getByText(/Blank: under 5%/)).toBeInTheDocument();
  });
});
