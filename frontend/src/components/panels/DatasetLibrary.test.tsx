import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";

const api = vi.hoisted(() => ({
  listDatasets: vi.fn(),
  renameDataset: vi.fn(),
  getDataset: vi.fn(),
  deleteDataset: vi.fn(),
  pruneDatasets: vi.fn(),
}));
vi.mock("@/api/datasets", () => api);

import { DatasetLibrary } from "./DatasetLibrary";

const ds = (id: string, name: string, created_at = "2026-10-01T12:00:00+00:00") =>
  ({ id, name, n_cells: 1200, n_genes: 3000, status: "ready", created_at, obs_columns: [] }) as unknown as DatasetInfo;

describe("DatasetLibrary", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    const list = [ds("a", "pbmc_3k"), ds("b", "kang", "")];
    api.listDatasets.mockResolvedValue(list);
    useDatasetStore.setState({ availableDatasets: list, currentDataset: list[0]!, currentDatasetId: "a" });
  });

  it("shows when each dataset was added, and nothing for a missing date", async () => {
    render(<DatasetLibrary />);
    const added = await screen.findAllByText(/added/);
    expect(added).toHaveLength(1);
    expect(added[0]!.textContent).toContain("2026");
  });

  it("renames a dataset and updates the list and the current dataset", async () => {
    api.renameDataset.mockResolvedValue({ id: "a", name: "PBMC 3k" });
    render(<DatasetLibrary />);
    fireEvent.click(screen.getByLabelText("Rename pbmc_3k"));
    const input = screen.getByLabelText("Dataset name");
    fireEvent.change(input, { target: { value: "  PBMC 3k " } });
    fireEvent.keyDown(input, { key: "Enter" });
    fireEvent.blur(input); // the blur after Enter must not send a second request
    await waitFor(() => expect(useDatasetStore.getState().currentDataset?.name).toBe("PBMC 3k"));
    expect(api.renameDataset).toHaveBeenCalledTimes(1);
    expect(api.renameDataset).toHaveBeenCalledWith("a", "PBMC 3k");
    expect(useDatasetStore.getState().availableDatasets.map((d) => d.name)).toEqual(["PBMC 3k", "kang"]);
  });

  it("Escape cancels without a request; a rejected name shows the reason", async () => {
    render(<DatasetLibrary />);
    fireEvent.click(screen.getByLabelText("Rename kang"));
    fireEvent.keyDown(screen.getByLabelText("Dataset name"), { key: "Escape" });
    expect(api.renameDataset).not.toHaveBeenCalled();

    api.renameDataset.mockRejectedValue(new Error("Name must not contain control or invisible characters."));
    fireEvent.click(screen.getByLabelText("Rename kang"));
    const input = screen.getByLabelText("Dataset name");
    fireEvent.change(input, { target: { value: "bad‮name" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(await screen.findByText(/control or invisible/)).toBeInTheDocument();
    expect(useDatasetStore.getState().availableDatasets[1]!.name).toBe("kang");
  });
});
