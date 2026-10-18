import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";

const { apiFetch, downloadCsv, plotProps } = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  downloadCsv: vi.fn(),
  plotProps: { current: null as null | Record<string, unknown> },
}));
vi.mock("@/api/client", () => ({ apiFetch }));
vi.mock("@/lib/csv", () => ({ downloadCsv }));
// Plotly needs a real canvas; the traces and layout it is handed are what matters.
vi.mock("@/components/plots/Plot", () => ({
  default: (props: Record<string, unknown>) => {
    plotProps.current = props;
    return <div data-testid="plot" />;
  },
}));

import { ObservationsPanel } from "./ObservationsPanel";
import { useDatasetStore } from "@/stores/datasetStore";

interface Trace {
  name: string;
  x: unknown;
  y: number[];
  marker: { color: string };
}
const traces = () => (plotProps.current?.data ?? []) as Trace[];
const layout = () =>
  plotProps.current?.layout as { yaxis: { title: { text: string } }; xaxis: { type?: string }; legend: { orientation: string } };

const INFO = {
  id: "kang",
  name: "Kang",
  n_cells: 100,
  // Stored order: CD14 Mono before B (not alphabetical), donors d1, d2, d10.
  obs_columns: [
    { name: "cluster", dtype: "category", n_unique: 2, values: ["CD14 Mono", "B"] },
    { name: "Donor", dtype: "category", n_unique: 3, values: ["d1", "d2", "d10"] },
    { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] },
  ],
  active_clustering: "cluster",
} as unknown as DatasetInfo;

function respond(url: string) {
  if (url.includes("/metadata/summary")) return Promise.resolve({ groupby: "cluster", counts: { B: 40, "CD14 Mono": 60 } });
  if (url.includes("row=cluster&col=Donor")) {
    return Promise.resolve({ row_values: ["B", "CD14 Mono"], col_values: ["d1", "d10", "d2"], counts: [[10, 20, 10], [30, 10, 20]] });
  }
  if (url.includes("row=Donor&col=cluster&facet=stim")) {
    return Promise.resolve({
      row_values: ["d1", "d10", "d2"],
      col_values: ["B", "CD14 Mono"],
      counts: [[10, 30], [20, 10], [10, 20]],
      facet_values: ["CTRL", "STIM"],
      facet_counts: [
        [[5, 15], [10, 5], [10, 20]],
        [[5, 15], [10, 5], [0, 0]],
      ],
    });
  }
  return Promise.reject(new Error(`unexpected ${url}`));
}

describe("ObservationsPanel composition bars", () => {
  beforeEach(() => {
    apiFetch.mockReset().mockImplementation(respond);
    downloadCsv.mockReset();
    plotProps.current = null;
    useDatasetStore.setState({ currentDatasetId: "kang", currentDataset: INFO });
  });

  it("stacks in the stored category order with the stable colours, counts by default", async () => {
    render(<ObservationsPanel />);
    await waitFor(() => expect(traces()).toHaveLength(3));
    // Bars (x) in the cluster column's stored order, stacks in the donor order.
    expect(traces()[0]!.x).toEqual(["CD14 Mono", "B"]);
    expect(traces().map((t) => t.name)).toEqual(["d1", "d2", "d10"]);
    expect(traces()[0]!.y).toEqual([30, 10]);
    expect(layout().yaxis.title.text).toBe("Count");
    // Distinct colours per stacked category.
    expect(new Set(traces().map((t) => t.marker.color)).size).toBe(3);
  });

  it("switches to percent and sorts by name, size or a chosen share", async () => {
    render(<ObservationsPanel />);
    await waitFor(() => expect(traces()).toHaveLength(3));
    fireEvent.click(screen.getByRole("button", { name: "%" }));
    expect(layout().yaxis.title.text).toBe("% of cells");
    expect(traces()[0]!.y).toEqual([50, 25]); // d1: 30 of 60, 10 of 40

    fireEvent.change(screen.getByLabelText("Sort bars"), { target: { value: "name" } });
    expect(traces()[0]!.x).toEqual(["B", "CD14 Mono"]);

    fireEvent.change(screen.getByLabelText("Sort bars"), { target: { value: "share" } });
    fireEvent.change(screen.getByLabelText("Share of"), { target: { value: "d10" } });
    expect(traces()[0]!.x).toEqual(["B", "CD14 Mono"]); // d10: B 50%, CD14 Mono 17%
  });

  it("shows one bar per donor split by condition from the shortcut", async () => {
    render(<ObservationsPanel />);
    await waitFor(() => expect(traces()).toHaveLength(3));
    fireEvent.click(screen.getByRole("button", { name: "Per Donor by stim" }));
    await waitFor(() =>
      expect(apiFetch).toHaveBeenCalledWith("/datasets/kang/metadata/crosstab?row=Donor&col=cluster&facet=stim"),
    );
    await waitFor(() => expect(traces().map((t) => t.name)).toEqual(["CD14 Mono", "B"]));
    expect(layout().xaxis.type).toBe("multicategory");
    expect(layout().legend.orientation).toBe("v"); // clear of the two-row axis labels
    // d2 has no STIM cells, so no bar for it there; donors keep the stored order.
    expect(traces()[0]!.x).toEqual([
      ["CTRL", "CTRL", "CTRL", "STIM", "STIM"],
      ["d1", "d2", "d10", "d1", "d10"],
    ]);

    fireEvent.click(screen.getByTitle("Export the composition as shown (counts or %) as CSV"));
    const [, headers, rows] = downloadCsv.mock.calls.at(-1)!;
    expect(headers).toEqual(["stim", "Donor", "CD14 Mono", "B"]);
    expect(rows[0]).toEqual(["CTRL", "d1", 15, 5]);
  });

  it("draws a 10 x 10 waffle for the chosen bar", async () => {
    render(<ObservationsPanel />);
    await waitFor(() => expect(traces()).toHaveLength(3));
    fireEvent.change(screen.getByLabelText("Waffle of"), { target: { value: "B" } });
    expect(screen.getAllByTestId("waffle-square")).toHaveLength(100);
    expect(screen.getByRole("img").getAttribute("aria-label")).toBe("cluster B by Donor: d1 25%, d2 25%, d10 50%");
  });
});
