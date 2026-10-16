import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
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

import type { SweepInfo, SweepRow } from "@/api/clustering";
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

  it("refuses resolutions it can't run", async () => {
    api.getSweepInfo.mockResolvedValue(info);
    renderPanel();
    const input = await screen.findByDisplayValue("0.2, 0.4, 1.0");
    fireEvent.change(input, { target: { value: "0.5, 9" } });
    expect(screen.getByText(/between 0.01 and 5/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
  });
});
