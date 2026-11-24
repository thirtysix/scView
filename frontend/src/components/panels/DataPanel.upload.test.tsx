import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import type { IngestState } from "@/api/ingest";
import type { UploadOptions } from "@/api/upload";

const { uploadDataset, getDataset, listDatasets } = vi.hoisted(() => ({
  uploadDataset: vi.fn(),
  getDataset: vi.fn(),
  listDatasets: vi.fn(),
}));
const ingest = vi.hoisted(() => ({
  createIngestSession: vi.fn(),
  uploadIngestFiles: vi.fn(),
  discardIngestSession: vi.fn(),
  getIngestState: vi.fn(),
  commitIngestSession: vi.fn(),
  setIngestOptions: vi.fn(),
}));
vi.mock("@/api/datasets", () => ({ uploadDataset, getDataset, listDatasets }));
vi.mock("@/api/ingest", async (orig) => ({
  ...(await orig<typeof import("@/api/ingest")>()),
  ...ingest,
  getMergePlan: vi.fn().mockResolvedValue({ is_merge: false }),
}));
vi.mock("@/components/panels/DatasetLibrary", () => ({ DatasetLibrary: () => null }));
vi.mock("@/components/common/PreflightNote", () => ({ PreflightIngestNotes: () => null, PreflightNote: () => null }));

import { DataPanel } from "./DataPanel";

const MB = 1e6;
const abortError = () => new DOMException("Aborted", "AbortError");

/** An upload mock that stays open: the test drives its progress and its end. */
function pending() {
  let opts: UploadOptions = {};
  let finish: (v: unknown) => void = () => {};
  let fail: (e: unknown) => void = () => {};
  const call = (...args: unknown[]) => {
    opts = (args.at(-1) as UploadOptions) ?? {};
    return new Promise((res, rej) => {
      finish = res;
      fail = rej;
      opts.signal?.addEventListener("abort", () => rej(abortError()));
    });
  };
  return {
    call,
    progress: (loaded: number, total: number, fileIndex = 1, fileCount = 1, file = "sample.rds") =>
      opts.onProgress?.({ loaded, total, file, fileIndex, fileCount }),
    finish: (v: unknown) => finish(v),
    fail: (e: unknown) => fail(e),
  };
}

const zone = () => screen.getByText("Drag files here, or").parentElement!;
const drop = (files: File[]) => fireEvent.drop(zone(), { dataTransfer: { files } });

const STATE = {
  session_id: "s",
  options: { name: null, join: "inner", sample_label: "sample", apply_reconciliation: true, genes_in_rows: true },
  validation: { issues: [], ok: true },
  bundle: {
    is_merge: false, complete: true, issues: [],
    units: [{ label: "LineA", format: "anndata", complete: true, missing_roles: [], issues: [], side_files: [],
      files: [{ path: "/d/LineA.h5ad", name: "LineA.h5ad", kind: "h5ad", role: "data", confidence: 1 }] }],
  },
} as IngestState;

describe("DataPanel: the upload card", () => {
  beforeEach(() => {
    uploadDataset.mockReset();
    getDataset.mockReset();
    listDatasets.mockReset().mockResolvedValue([]);
    for (const f of Object.values(ingest)) f.mockReset();
    ingest.createIngestSession.mockResolvedValue({ session_id: "s" });
    ingest.discardIngestSession.mockResolvedValue(undefined);
  });

  it("replaces the drop zone with progress while an .rds goes up, then shows its conversion", async () => {
    const up = pending();
    uploadDataset.mockImplementation(up.call);
    getDataset
      .mockResolvedValueOnce({ id: "d1", status: "converting", conversion_progress: null })
      .mockReturnValue(new Promise(() => {})); // stays converting

    render(<DataPanel />);
    await act(async () => drop([new File(["x"], "sample.rds")]));
    expect(zone()).toHaveClass("hidden");
    expect(screen.getByTestId("upload-title")).toHaveTextContent("Uploading sample.rds");

    await act(async () => up.progress(152 * MB, 331 * MB));
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "45");
    expect(screen.getByTestId("upload-numbers")).toHaveTextContent("152 of 331 MB");

    await act(async () => up.finish({ id: "d1", name: "sample", status: "converting" }));
    expect(screen.getByTestId("upload-title")).toHaveTextContent("Converting sample.rds");
    expect(screen.getByRole("status")).toHaveTextContent("Reading the R file… a large object takes a few minutes.");
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
  });

  it("starts no second upload while one runs", async () => {
    const up = pending();
    uploadDataset.mockImplementation(up.call);
    render(<DataPanel />);
    await act(async () => drop([new File(["x"], "sample.rds")]));
    await act(async () => drop([new File(["y"], "other.rds")])); // on the hidden zone
    await act(async () => {
      fireEvent.drop(screen.getByTestId("upload-card"), { dataTransfer: { files: [new File(["z"], "LineA.h5ad")] } });
    });
    expect(uploadDataset).toHaveBeenCalledTimes(1);
    expect(ingest.uploadIngestFiles).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("One upload at a time");
  });

  it("Cancel aborts the .rds upload, says nothing was kept and brings the drop zone back", async () => {
    const up = pending();
    uploadDataset.mockImplementation(up.call);
    render(<DataPanel />);
    await act(async () => drop([new File(["x"], "sample.rds")]));
    await act(async () => up.progress(10 * MB, 331 * MB));
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Cancel" })));
    expect(screen.queryByTestId("upload-card")).toBeNull();
    expect(zone()).not.toHaveClass("hidden");
    expect(screen.getByTestId("upload-cancelled")).toHaveTextContent("Upload of sample.rds cancelled. Nothing was kept.");
    expect(getDataset).not.toHaveBeenCalled();
    expect(screen.queryByText(/could not be uploaded/)).toBeNull();
  });

  it("shows the server's sentence when the upload fails, and the drop zone again", async () => {
    const up = pending();
    uploadDataset.mockImplementation(up.call);
    render(<DataPanel />);
    await act(async () => drop([new File(["x"], "sample.rds")]));
    await act(async () => up.fail(new Error("File exceeds maximum upload size of 200 MB.")));
    expect(screen.getByText("File exceeds maximum upload size of 200 MB.")).toBeTruthy();
    expect(screen.queryByTestId("upload-card")).toBeNull();
    expect(zone()).not.toHaveClass("hidden");
  });

  it("files for the wizard: progress names the file of the set, then the wizard's next step", async () => {
    const up = pending();
    ingest.uploadIngestFiles.mockImplementation(up.call);
    render(<DataPanel />);
    const files = Array.from({ length: 3 }, (_, i) => new File(["x"], `LineA_${i}.h5ad`));
    await act(async () => drop(files));
    expect(screen.getByTestId("upload-title")).toHaveTextContent("Uploading 3 files");
    await act(async () => up.progress(40 * MB, 120 * MB, 2, 3, "LineA_1.h5ad"));
    expect(screen.getByTestId("upload-file")).toHaveTextContent("File 2 of 3: LineA_1.h5ad");
    await act(async () => up.finish(STATE));
    expect(screen.queryByTestId("upload-card")).toBeNull();
    expect(screen.getByText("Detected sample")).toBeTruthy();
  });

  it("a cancelled first upload discards its new import session", async () => {
    const up = pending();
    ingest.uploadIngestFiles.mockImplementation(up.call);
    render(<DataPanel />);
    await act(async () => drop([new File(["x"], "LineA.h5ad")]));
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Cancel" })));
    expect(ingest.discardIngestSession).toHaveBeenCalledWith("s");
    expect(screen.getByTestId("upload-cancelled")).toHaveTextContent("Upload cancelled. Nothing was kept.");
    expect(screen.queryByText("Detected sample")).toBeNull();
  });

  it("a cancelled upload into an import that has samples keeps them, as the server has them", async () => {
    ingest.uploadIngestFiles.mockResolvedValueOnce(STATE);
    render(<DataPanel />);
    await act(async () => drop([new File(["x"], "LineA.h5ad")]));
    const up = pending();
    ingest.uploadIngestFiles.mockImplementation(up.call);
    ingest.getIngestState.mockResolvedValue(STATE);
    await act(async () => drop([new File(["y"], "LineB.h5ad")]));
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Cancel" })));
    expect(ingest.discardIngestSession).not.toHaveBeenCalled();
    expect(ingest.getIngestState).toHaveBeenCalledWith("s");
    expect(screen.getByText("Detected sample")).toBeTruthy();
    expect(screen.getByTestId("upload-cancelled")).toHaveTextContent("Files sent before it stay below");
  });

  it("takes no files while the dataset is being created", async () => {
    ingest.uploadIngestFiles.mockResolvedValueOnce(STATE);
    ingest.setIngestOptions.mockResolvedValue(undefined);
    ingest.commitIngestSession.mockReturnValue(new Promise(() => {}));
    render(<DataPanel />);
    await act(async () => drop([new File(["x"], "LineA.h5ad")]));
    await act(async () => fireEvent.click(screen.getByText("Create dataset")));
    expect(zone()).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByRole("button", { name: "Choose files" })).toBeDisabled();
    await act(async () => drop([new File(["y"], "LineB.h5ad")]));
    expect(ingest.uploadIngestFiles).toHaveBeenCalledTimes(1);
  });
});
