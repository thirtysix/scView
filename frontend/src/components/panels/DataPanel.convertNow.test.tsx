import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import type { BundleFile, IngestState, MergePlan } from "@/api/ingest";

const { createIngestSession, uploadIngestFiles, getIngestState, convertIngestSession, getMergePlan } = vi.hoisted(
  () => ({
    createIngestSession: vi.fn(),
    uploadIngestFiles: vi.fn(),
    getIngestState: vi.fn(),
    convertIngestSession: vi.fn(),
    getMergePlan: vi.fn(),
  })
);
vi.mock("@/api/ingest", async (orig) => ({
  ...(await orig<typeof import("@/api/ingest")>()),
  createIngestSession, uploadIngestFiles, getIngestState, convertIngestSession, getMergePlan,
  commitIngestSession: vi.fn(), discardIngestSession: vi.fn(),
  setIngestOptions: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@/api/datasets", () => ({
  uploadDataset: vi.fn(),
  getDataset: vi.fn(),
  listDatasets: vi.fn().mockResolvedValue([]),
}));
vi.mock("@/components/panels/DatasetLibrary", () => ({ DatasetLibrary: () => null }));
vi.mock("@/components/common/PreflightNote", () => ({ PreflightIngestNotes: () => null }));

import { DataPanel } from "./DataPanel";

const bf = (name: string, kind: string): BundleFile => ({
  path: `/data/ingest/s/files/${name}`, name, kind, role: "data", confidence: 1, rel_path: name,
});

const STATE: IngestState = {
  session_id: "s",
  options: { name: null, join: "inner", sample_label: "sample", apply_reconciliation: true, genes_in_rows: true },
  validation: { ok: true, issues: [] },
  bundle: {
    is_merge: true, complete: true, issues: [],
    units: [
      { label: "obj", format: "seurat", complete: true, missing_roles: [], issues: [], side_files: [],
        files: [bf("obj.rds", "seurat_rds")] },
      { label: "a", format: "anndata", complete: true, missing_roles: [], issues: [], side_files: [],
        files: [bf("a.h5ad", "anndata_h5ad")] },
    ],
  },
  converted: [],
  conversion_running: false,
  conversion_error: null,
};

const plan = (samples: string[]): MergePlan => ({
  samples, per_sample_genes: Object.fromEntries(samples.map((s) => [s, 4])), bases: {},
  intersection: 4, union: 4, suspicious_low_overlap: false, reconciliation: null,
  recommended_join: "inner", est_cells: 11, est_genes: 4, warnings: [],
});

async function upload() {
  render(<DataPanel />);
  await act(async () => {
    fireEvent.change(screen.getByLabelText("Choose files"), {
      target: { files: [new File(["x"], "obj.rds"), new File(["y"], "a.h5ad")] },
    });
  });
}

describe("DataPanel: Convert now", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    createIngestSession.mockReset().mockResolvedValue({ session_id: "s" });
    uploadIngestFiles.mockReset().mockResolvedValue(STATE);
    getMergePlan.mockReset().mockResolvedValue({ is_merge: false });
    getIngestState.mockReset();
    convertIngestSession.mockReset();
  });
  afterEach(() => vi.useRealTimers());

  it("converts the .rds, shows its progress, then fetches the merge plan with it", async () => {
    convertIngestSession.mockResolvedValue({ ...STATE, conversion_running: true });
    getIngestState
      .mockResolvedValueOnce({
        ...STATE, conversion_running: true,
        converting: {
          label: "obj", file: "obj.rds", index: 1, of: 1,
          progress: { stage: "converting", message: "Converting a Seurat object", source_class: "Seurat" },
        },
      })
      .mockResolvedValue({ ...STATE, converted: [{ label: "obj", file: "obj.rds" }] });
    await upload();
    expect(getMergePlan).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/the merge plan then includes it/)).toBeTruthy();

    getMergePlan.mockResolvedValue(plan(["obj", "a"]));
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Convert now" })); });
    expect(convertIngestSession).toHaveBeenCalledWith("s");
    expect(screen.getByRole("button", { name: "Create dataset" })).toBeDisabled();
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    expect(screen.getByRole("status")).toHaveTextContent("obj.rds: Converting a Seurat object…");

    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    expect(getMergePlan).toHaveBeenCalledTimes(2); // refetched once converted
    expect(screen.queryByTestId("convert-now")).toBeNull(); // nothing left to convert
    expect(screen.getByText("R object (.rds: Seurat, SingleCellExperiment), converted")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Create dataset" })).not.toBeDisabled();
  });

  it("shows the session's conversion error", async () => {
    convertIngestSession.mockResolvedValue({ ...STATE, conversion_running: true });
    getIngestState.mockResolvedValue({
      ...STATE, conversion_error: "obj.rds could not be converted: This file holds a list object.",
    });
    await upload();
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Convert now" })); });
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    expect(screen.getByRole("alert")).toHaveTextContent("This file holds a list object.");
    expect(screen.getByRole("button", { name: "Convert now" })).not.toBeDisabled();
  });

  it("is not offered once converted", async () => {
    uploadIngestFiles.mockResolvedValue({ ...STATE, converted: [{ label: "obj", file: "obj.rds" }] });
    await upload();
    expect(screen.queryByRole("button", { name: "Convert now" })).toBeNull();
  });

  it("is not offered for an .rds with no other sample", async () => {
    uploadIngestFiles.mockResolvedValue({
      ...STATE, bundle: { ...STATE.bundle, is_merge: false, units: [STATE.bundle.units[0]!] },
    });
    await upload();
    expect(screen.getByText("obj")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Convert now" })).toBeNull();
  });
});
