import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";

const { uploadDataset, getDataset, listDatasets } = vi.hoisted(() => ({
  uploadDataset: vi.fn(),
  getDataset: vi.fn(),
  listDatasets: vi.fn(),
}));
vi.mock("@/api/datasets", () => ({ uploadDataset, getDataset, listDatasets }));
vi.mock("@/api/ingest", () => ({
  commitIngestSession: vi.fn(), createIngestSession: vi.fn(), discardIngestSession: vi.fn(),
  getMergePlan: vi.fn(), setIngestOptions: vi.fn(), uploadIngestFiles: vi.fn(),
}));
vi.mock("@/components/panels/DatasetLibrary", () => ({ DatasetLibrary: () => null }));
vi.mock("@/components/common/PreflightNote", () => ({ PreflightIngestNotes: () => null }));

import { DataPanel } from "./DataPanel";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";

const base = {
  id: "d1", name: "kang", filename: "kang.rds", n_cells: null, n_genes: null,
  available_embeddings: [], obs_columns: [], created_at: "",
} as unknown as DatasetInfo;

function drop(file: File) {
  const zone = screen.getByText("Drag files here, or").parentElement!;
  fireEvent.drop(zone, { dataTransfer: { files: [file] } });
}

describe("DataPanel: an R file", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    uploadDataset.mockReset();
    getDataset.mockReset();
    listDatasets.mockReset().mockResolvedValue([]);
    useViewStore.setState({ activePanel: "load" });
  });
  afterEach(() => vi.useRealTimers());

  it("names the class while it converts, then opens Data Assessment", async () => {
    uploadDataset.mockResolvedValue({ id: "d1", name: "kang", status: "converting" });
    getDataset
      .mockResolvedValueOnce({ ...base, status: "converting", conversion_progress: null })
      .mockResolvedValueOnce({
        ...base, status: "converting",
        conversion_progress: {
          stage: "writing", message: "Writing the counts assay", source_class: "SingleCellExperiment",
          n_cells: 13836, n_genes: 14053,
        },
      })
      .mockResolvedValueOnce({
        ...base, status: "ready", n_cells: 13836, n_genes: 14053,
        conversion: { source_class: "SingleCellExperiment", kept: [], skipped: [] },
      });

    render(<DataPanel />);
    await act(async () => drop(new File(["x"], "kang.rds")));
    expect(uploadDataset).toHaveBeenCalledTimes(1);
    expect(screen.getByText("Reading the R file… a large object takes a few minutes.")).toBeTruthy();

    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    expect(screen.getByText(
      "Converting a SingleCellExperiment (13,836 cells × 14,053 features): writing the counts assay…",
    )).toBeTruthy();

    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    expect(useDatasetStore.getState().currentDataset?.status).toBe("ready");
    expect(useViewStore.getState().activePanel).toBe("assessment");
  });

  it("shows the converter's own sentence when it refuses the object", async () => {
    const reason = "This file holds a list object, which scView cannot convert.";
    uploadDataset.mockResolvedValue({ id: "d1", name: "kang", status: "converting" });
    getDataset.mockResolvedValueOnce({ ...base, status: "error", error_message: reason });

    render(<DataPanel />);
    await act(async () => drop(new File(["x"], "obj.RData")));
    expect(screen.getByText(reason)).toBeTruthy();
    expect(useViewStore.getState().activePanel).toBe("load");
  });
});
