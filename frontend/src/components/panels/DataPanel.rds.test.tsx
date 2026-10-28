import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import type { BundleFile, IngestState } from "@/api/ingest";

const { createIngestSession, uploadIngestFiles, commitIngestSession, getIngestState } = vi.hoisted(() => ({
  createIngestSession: vi.fn(),
  uploadIngestFiles: vi.fn(),
  commitIngestSession: vi.fn(),
  getIngestState: vi.fn(),
}));
vi.mock("@/api/ingest", async (orig) => ({
  ...(await orig<typeof import("@/api/ingest")>()),
  createIngestSession, uploadIngestFiles, commitIngestSession, getIngestState,
  discardIngestSession: vi.fn(), getMergePlan: vi.fn().mockResolvedValue({ is_merge: false }),
  setIngestOptions: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@/api/datasets", () => ({
  uploadDataset: vi.fn(),
  getDataset: vi.fn().mockResolvedValue({ id: "d9", name: "merged", status: "ready" }),
  listDatasets: vi.fn().mockResolvedValue([]),
}));
vi.mock("@/components/panels/DatasetLibrary", () => ({ DatasetLibrary: () => null }));
vi.mock("@/components/common/PreflightNote", () => ({ PreflightIngestNotes: () => null }));

import { DataPanel } from "./DataPanel";

const bf = (name: string, extra: Partial<BundleFile>): BundleFile => ({
  path: `/data/ingest/s/files/${name}`, name, kind: "x", role: "data", confidence: 1, rel_path: name, ...extra,
});

const STATE: IngestState = {
  session_id: "s",
  options: { name: null, join: "inner", sample_label: "sample", apply_reconciliation: true, genes_in_rows: true },
  validation: {
    ok: true,
    issues: [{
      severity: "info", code: "seurat_converted_at_commit", unit_label: "obj",
      message: "obj.rds is an R object (Seurat, SingleCellExperiment). The R converter turns it into a dataset when you create it, which takes a few minutes for a large object.",
      suggestion: "Its cells and genes are read only then, so the merge plan here leaves it out.",
    }],
  },
  bundle: {
    is_merge: true, complete: true, issues: [],
    units: [
      { label: "obj", format: "seurat", complete: true, missing_roles: [], issues: [], side_files: [],
        files: [bf("obj.rds", { kind: "seurat_rds" })] },
      { label: "pbmc", format: "h5seurat", complete: true, missing_roles: [], issues: [], side_files: [],
        files: [bf("pbmc.h5Seurat", { kind: "h5seurat" })] },
    ],
  },
};

describe("DataPanel: an .rds among other files", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    createIngestSession.mockReset().mockResolvedValue({ session_id: "s" });
    uploadIngestFiles.mockReset().mockResolvedValue(STATE);
    getIngestState.mockReset();
    commitIngestSession.mockReset();
  });
  afterEach(() => vi.useRealTimers());

  it("goes to the wizard, labels both formats and names the .rds while it converts", async () => {
    let finish: (v: { dataset_id: string }) => void = () => {};
    commitIngestSession.mockReturnValue(new Promise((r) => { finish = r; }));
    getIngestState.mockResolvedValue({
      ...STATE,
      converting: {
        label: "obj", file: "obj.rds", index: 1, of: 1,
        progress: { stage: "converting", message: "Converting a Seurat object", source_class: "Seurat" },
      },
    });
    render(<DataPanel />);
    const input = screen.getByLabelText("Choose files");
    await act(async () => {
      fireEvent.change(input, { target: { files: [new File(["x"], "obj.rds"), new File(["y"], "pbmc.h5Seurat")] } });
    });
    expect(uploadIngestFiles).toHaveBeenCalledTimes(1); // not the lone-.rds converter path
    expect(screen.getByText("R object (.rds: Seurat, SingleCellExperiment), converted when the dataset is created")).toBeTruthy();
    expect(screen.getByText("Seurat (.h5Seurat)")).toBeTruthy();

    await act(async () => { fireEvent.click(screen.getByText("Create dataset")); });
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    expect(screen.getByRole("status")).toHaveTextContent("obj.rds: Converting a Seurat object…");

    await act(async () => { finish({ dataset_id: "d9" }); });
    expect(screen.queryByText("obj.rds: Converting a Seurat object…")).toBeNull();
  });
});
