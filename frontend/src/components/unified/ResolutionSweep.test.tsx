import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const api = vi.hoisted(() => ({
  getSweepInfo: vi.fn(),
  runSweep: vi.fn(),
  keepResolution: vi.fn(),
  getDataset: vi.fn(),
}));
vi.mock("@/api/clustering", () => ({
  getSweepInfo: api.getSweepInfo,
  runSweep: api.runSweep,
  keepResolution: api.keepResolution,
}));
vi.mock("@/api/datasets", () => ({ getDataset: api.getDataset }));
// Plotly needs a real browser; the test checks what is asked for.
const plotProps = vi.hoisted(() => vi.fn());
vi.mock("@/lib/plotConfig", () => ({ plotConfig: () => ({}) }));
vi.mock("@/components/plots/Plot", () => ({
  default: (p: unknown) => {
    plotProps(p);
    return <div data-testid="plot" />;
  },
}));

import type { ClusterTree, SweepInfo, SweepRow } from "@/api/clustering";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { ResolutionSweepPanel } from "./ResolutionSweep";

const info: SweepInfo = {
  available: true,
  reason: null,
  method: "leiden",
  n_cells: 400,
  default_resolutions: [0.2, 0.4, 1],
  max_resolutions: 8,
  min_resolution: 0.01,
  max_resolution: 5,
  max_cells: 200000,
  silhouette_cells: 5000,
};

const row = (resolution: number, extra: Partial<SweepRow> = {}): SweepRow => ({
  resolution,
  n_clusters: Math.round(resolution * 10),
  sizes: [200, 150, 50],
  silhouette: resolution === 0.4 ? 0.3 : 0.1,
  ari_previous: resolution === 0.2 ? null : 0.8,
  column: `scview_leiden_r${resolution === 1 ? "1.0" : resolution}`,
  exists: false,
  ...extra,
});

function renderPanel() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <ResolutionSweepPanel datasetId="ds" onClose={() => {}} />
    </QueryClientProvider>,
  );
}

describe("ResolutionSweepPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useDatasetStore.setState({ currentDatasetId: "ds" });
    useSettingsStore.setState({ colorBy: "leiden" });
  });

  it("explains why a dataset can't be swept", async () => {
    api.getSweepInfo.mockResolvedValue({ ...info, available: false, reason: "This dataset has no neighbour graph." });
    renderPanel();
    expect(await screen.findByRole("alert")).toHaveTextContent("no neighbour graph");
    expect(screen.queryByRole("button", { name: "Run" })).not.toBeInTheDocument();
  });

  it("runs the default resolutions, lists them, and keeps one", async () => {
    api.getSweepInfo.mockResolvedValue(info);
    api.runSweep.mockImplementation(async (_id, _body, h) => {
      h.onStart({ method: "leiden", n_cells: 400, resolutions: [0.2, 0.4, 1], representation: "X_pca", silhouette_cells: 400 });
      h.onResult(row(0.2));
      h.onResult(row(0.4));
      h.onResult(row(1, { exists: true }));
    });
    api.keepResolution.mockResolvedValue({ column: "scview_leiden_r0.4", created: true, n_clusters: 4 });
    api.getDataset.mockResolvedValue({ id: "ds", obs_columns: [] });

    renderPanel();
    expect(await screen.findByDisplayValue("0.2, 0.4, 1.0")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(screen.getAllByRole("row")).toHaveLength(4)); // header + 3
    expect(api.runSweep.mock.calls[0]![1]).toEqual({ resolutions: [0.2, 0.4, 1], method: "leiden" });
    expect(screen.getByText(/Silhouette on 400 of 400 cells/)).toBeInTheDocument();

    // An existing column is shown, not kept again.
    fireEvent.click(screen.getByRole("button", { name: "Show" }));
    expect(useSettingsStore.getState().colorBy).toBe("scview_leiden_r1.0");

    fireEvent.click(screen.getAllByRole("button", { name: /Keep this one/ })[1]!);
    await waitFor(() => expect(useSettingsStore.getState().colorBy).toBe("scview_leiden_r0.4"));
    expect(api.keepResolution).toHaveBeenCalledWith("ds", 0.4, "leiden");
    expect(api.getDataset).toHaveBeenCalledWith("ds");
    expect(screen.getAllByRole("button", { name: "Show" })).toHaveLength(2);
  });

  it("draws the clustering tree and picks a resolution from a row or a node", async () => {
    const tree: ClusterTree = {
      prop_filter: 0.1,
      mixed_prop: 0.2,
      nodes: [
        { resolution: 0.2, cluster: "0", size: 400, sc3_stability: 0.9, n_parents: 0, mixed: false },
        { resolution: 0.4, cluster: "0", size: 250, sc3_stability: 0.6, n_parents: 1, mixed: false },
        { resolution: 0.4, cluster: "1", size: 150, sc3_stability: 0.4, n_parents: 2, mixed: true },
      ],
      edges: [
        { from_resolution: 0.2, from: "0", to_resolution: 0.4, to: "0", count: 250, in_prop: 1 },
        { from_resolution: 0.2, from: "0", to_resolution: 0.4, to: "1", count: 150, in_prop: 1 },
      ],
      n_mixed: 1,
    };
    api.getSweepInfo.mockResolvedValue(info);
    api.runSweep.mockImplementation(async (_id, _body, h) => {
      h.onStart({ method: "leiden", n_cells: 400, resolutions: [0.2, 0.4], representation: "X_pca", silhouette_cells: 400 });
      h.onResult(row(0.2));
      h.onResult(row(0.4));
      h.onTree(tree);
    });
    renderPanel();
    fireEvent.click(await screen.findByRole("button", { name: "Run" }));
    expect(await screen.findByTestId("plot")).toBeInTheDocument();
    expect(screen.getByText(/1 cluster draws at least 20% of its cells from each of two or more parents/)).toBeInTheDocument();

    const last = () => plotProps.mock.calls[plotProps.mock.calls.length - 1]![0] as {
      layout: { shapes: { y0: number }[] };
      onClick: (e: unknown) => void;
    };
    expect(last().layout.shapes).toHaveLength(0);
    fireEvent.click(screen.getByText("0.4"));
    await waitFor(() => expect(last().layout.shapes).toHaveLength(1));
    expect(last().layout.shapes[0]!.y0).toBeCloseTo(0.55);
    expect(screen.getAllByRole("row")[2]).toHaveAttribute("aria-selected", "true");

    // A node click picks its resolution in the table.
    act(() => last().onClick({ points: [{ customdata: 0.2 }] }));
    expect(screen.getAllByRole("row")[1]).toHaveAttribute("aria-selected", "true");
  });

  it("refuses resolutions it can't run", async () => {
    api.getSweepInfo.mockResolvedValue(info);
    renderPanel();
    const input = await screen.findByDisplayValue("0.2, 0.4, 1.0");
    fireEvent.change(input, { target: { value: "0.5, 9" } });
    expect(screen.getByText(/between 0.01 and 5/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
  });
});
