import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { ActivitySummary } from "@/api/activity";
import type { DatasetInfo } from "@/api/types";

const { getActivitySummary, getActivityCells } = vi.hoisted(() => ({
  getActivitySummary: vi.fn(),
  getActivityCells: vi.fn(),
}));
vi.mock("@/api/activity", async (orig) => ({
  ...(await orig<typeof import("@/api/activity")>()),
  getActivitySummary,
  getActivityCells,
}));
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data }: { data: { type: string; y?: string[] }[] }) => (
    <div data-testid="plot">{data.map((t) => `${t.type}:${(t.y ?? []).join(",")}`).join(";")}</div>
  ),
}));

import { UnifiedActivitySubtab } from "./UnifiedActivitySubtab";
import { useDatasetStore } from "@/stores/datasetStore";

const SUMMARY: ActivitySummary = {
  network: "progeny",
  label: "PROGENy",
  kind: "pathway",
  citation: "Schubert et al. 2018",
  species: "human",
  matrix_source: "log-normalised .raw",
  n_cells: 4,
  n_genes_measured: 900,
  sources: ["JAK-STAT", "MAPK"],
  n_targets: [300, 280],
  groupby: "stim",
  groups: ["CTRL", "STIM"],
  group_sizes: [2, 2],
  mean: [
    [1, 0.5],
    [30, 0.2],
  ],
  elapsed_seconds: 0.2,
};

describe("UnifiedActivitySubtab", () => {
  beforeEach(() => {
    getActivitySummary.mockReset().mockResolvedValue(SUMMARY);
    getActivityCells.mockReset().mockResolvedValue({ network: "progeny", source: "JAK-STAT", n_targets: 300, scores: [1, 2, 30, 31] });
    useDatasetStore.setState({
      currentDatasetId: "d1",
      currentDataset: {
        id: "d1", name: "d1", filename: "", n_cells: 4, n_genes: 900, status: "ready", available_embeddings: [],
        created_at: "", obs_columns: [{ name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] }],
      } as DatasetInfo,
    });
  });

  it("averages per group, draws the heatmap and colours the cells by a pathway", async () => {
    const onScore = vi.fn();
    render(<UnifiedActivitySubtab onScore={onScore} groupByColumn="stim" setGroupByColumn={() => {}} />);
    await waitFor(() => screen.getByText("JAK-STAT"));
    expect(getActivitySummary).toHaveBeenCalledWith("d1", "progeny", "stim");
    expect(screen.getByTestId("plot").textContent).toBe("heatmap:MAPK,JAK-STAT");
    expect(screen.getByText(/2 pathways with at least 5 of their targets/)).toBeTruthy();

    fireEvent.click(screen.getByText("JAK-STAT"));
    await waitFor(() => expect(onScore).toHaveBeenCalled());
    expect(getActivityCells).toHaveBeenCalledWith("d1", "progeny", "JAK-STAT");
    const [values, name] = onScore.mock.calls[0]!;
    expect(Array.from(values as Float32Array)).toEqual([1, 2, 30, 31]);
    expect(name).toBe("JAK-STAT (pathway activity)");
  });

  it("switches to TFs and shows why a dataset cannot be scored", async () => {
    render(<UnifiedActivitySubtab onScore={vi.fn()} groupByColumn="" setGroupByColumn={() => {}} />);
    await waitFor(() => screen.getByText("JAK-STAT"));
    expect(getActivitySummary).toHaveBeenLastCalledWith("d1", "progeny", undefined);
    getActivitySummary.mockRejectedValueOnce(new Error("this looks like a mouse dataset"));
    fireEvent.click(screen.getByRole("button", { name: "TFs (CollecTRI)" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toMatch(/mouse dataset/));
    expect(getActivitySummary).toHaveBeenLastCalledWith("d1", "collectri", undefined);
  });
});
