import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import type { BundleFile, IngestState } from "@/api/ingest";

const { createIngestSession, uploadIngestFiles } = vi.hoisted(() => ({
  createIngestSession: vi.fn(),
  uploadIngestFiles: vi.fn(),
}));
vi.mock("@/api/ingest", async (orig) => ({
  ...(await orig<typeof import("@/api/ingest")>()),
  createIngestSession, uploadIngestFiles,
  commitIngestSession: vi.fn(), discardIngestSession: vi.fn(), getMergePlan: vi.fn(), setIngestOptions: vi.fn(),
}));
vi.mock("@/api/datasets", () => ({ uploadDataset: vi.fn(), getDataset: vi.fn(), listDatasets: vi.fn() }));
vi.mock("@/components/panels/DatasetLibrary", () => ({ DatasetLibrary: () => null }));
vi.mock("@/components/common/PreflightNote", () => ({ PreflightIngestNotes: () => null }));

import { DataPanel } from "./DataPanel";

function picked(path: string, size = 3): File {
  const f = new File(["x".repeat(size)], path.split("/").pop()!);
  Object.defineProperty(f, "webkitRelativePath", { value: path });
  return f;
}

const bf = (rel: string, extra: Partial<BundleFile>): BundleFile => ({
  path: `/data/ingest/s/files/${rel}`, name: rel.split("/").pop()!, kind: "x", role: "side", confidence: 1,
  rel_path: rel, ...extra,
});

const STATE: IngestState = {
  session_id: "s",
  options: { name: null, join: "inner", sample_label: "sample", apply_reconciliation: true, genes_in_rows: true },
  validation: { issues: [], ok: true },
  bundle: {
    is_merge: false, complete: true,
    issues: ["Skipped 1 file scView does not read: web_summary.html (Cell Ranger's HTML report)."],
    units: [{
      label: "pbmc", format: "tenx_mex", complete: true, missing_roles: [], issues: [],
      files: [
        bf("pbmc/outs/filtered_feature_bc_matrix/barcodes.tsv.gz", { role: "barcodes", kind: "tenx_barcodes" }),
        bf("pbmc/outs/filtered_feature_bc_matrix/features.tsv.gz", { role: "features", kind: "tenx_features" }),
        bf("pbmc/outs/filtered_feature_bc_matrix/matrix.mtx.gz", { role: "matrix", kind: "mtx_matrix" }),
      ],
      side_files: [
        bf("pbmc/outs/raw_feature_bc_matrix/barcodes.tsv.gz", { kind: "tenx_barcodes", side_kind: "raw" }),
        bf("pbmc/outs/metrics_summary.csv", { kind: "tenx_metrics", side_kind: "metrics" }),
        bf("pbmc/outs/vdj_t/filtered_contig_annotations.csv", { kind: "vdj_contigs", side_kind: "vdj", receptor: "T" }),
        bf("pbmc/outs/vdj_b/filtered_contig_annotations.csv", { kind: "vdj_contigs", side_kind: "vdj", receptor: "B" }),
      ],
    }],
  },
};

describe("DataPanel: a folder", () => {
  beforeEach(() => {
    createIngestSession.mockReset().mockResolvedValue({ session_id: "s" });
    uploadIngestFiles.mockReset().mockResolvedValue(STATE);
  });

  it("sends each file with its folder path, never the BAM, and shows which file is which", async () => {
    render(<DataPanel />);
    const input = screen.getByTestId("folder-input");
    const files = [
      picked("pbmc/outs/filtered_feature_bc_matrix/barcodes.tsv.gz"),
      picked("pbmc/outs/raw_feature_bc_matrix/barcodes.tsv.gz"),
      picked("pbmc/outs/possorted_genome_bam.bam", 2048),
      picked("pbmc/outs/vdj_t/filtered_contig_annotations.csv"),
    ];
    await act(async () => {
      fireEvent.change(input, { target: { files } });
    });
    expect(uploadIngestFiles).toHaveBeenCalledTimes(1);
    const sent = uploadIngestFiles.mock.calls[0]![1] as { file: File; path: string }[];
    expect(sent.map((i) => i.path)).toEqual([
      "pbmc/outs/filtered_feature_bc_matrix/barcodes.tsv.gz",
      "pbmc/outs/raw_feature_bc_matrix/barcodes.tsv.gz",
      "pbmc/outs/vdj_t/filtered_contig_annotations.csv",
    ]);
    expect(screen.getByTestId("not-sent")).toHaveTextContent(
      "Not uploaded: 1 file scView does not read, 2.0 KB: possorted_genome_bam.bam (read alignments",
    );
    // The two barcodes.tsv.gz are told apart by their folders; contigs say T or B.
    expect(screen.getByText("pbmc/outs/filtered_feature_bc_matrix/barcodes.tsv.gz")).toBeTruthy();
    const side = Array.from(screen.getByTestId("seqqc-side-files").querySelectorAll("li")).map((li) => li.textContent);
    expect(side).toEqual([
      "pbmc/outs/raw_feature_bc_matrix/barcodes.tsv.gz(raw matrix part): will be attached",
      "pbmc/outs/metrics_summary.csv(Cell Ranger metrics): will be attached",
    ]);
    const vdj = Array.from(screen.getByTestId("unit-vdj-files").querySelectorAll("li")).map((li) => li.textContent);
    expect(vdj).toEqual([
      "pbmc/outs/vdj_t/filtered_contig_annotations.csv (TCR contigs)",
      "pbmc/outs/vdj_b/filtered_contig_annotations.csv (BCR contigs)",
    ]);
    expect(screen.getByText(/Skipped 1 file scView does not read/)).toBeTruthy();
  });

  it("sends nothing when a folder holds only files scView does not read", async () => {
    render(<DataPanel />);
    await act(async () => {
      fireEvent.change(screen.getByTestId("folder-input"), { target: { files: [picked("o/web_summary.html")] } });
    });
    expect(uploadIngestFiles).not.toHaveBeenCalled();
    expect(screen.getByTestId("not-sent")).toHaveTextContent("web_summary.html (Cell Ranger's HTML report)");
  });

  const hdFiles = (um: string) => [
    picked(`binned_outputs/square_${um}um/filtered_feature_bc_matrix.h5`),
    picked(`binned_outputs/square_${um}um/raw_feature_bc_matrix.h5`),
    picked(`binned_outputs/square_${um}um/spatial/tissue_positions.parquet`),
    picked(`binned_outputs/square_${um}um/spatial/scalefactors_json.json`),
  ];

  it("asks which bin size of a Visium HD folder, then sends only that one's files", async () => {
    render(<DataPanel />);
    await act(async () => {
      fireEvent.change(screen.getByTestId("folder-input"), {
        target: { files: [...hdFiles("002"), ...hdFiles("008"), ...hdFiles("016")] },
      });
    });
    expect(uploadIngestFiles).not.toHaveBeenCalled();
    expect(screen.getByTestId("hd-bin-choice")).toBeTruthy();
    fireEvent.click(screen.getAllByRole("radio")[2]!);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Upload 16 µm bins/ }));
    });
    expect(screen.queryByTestId("hd-bin-choice")).toBeNull();
    const sent = uploadIngestFiles.mock.calls[0]![1] as { path: string }[];
    expect(sent.map((i) => i.path)).toEqual([
      "binned_outputs/square_016um/filtered_feature_bc_matrix.h5",
      "binned_outputs/square_016um/spatial/tissue_positions.parquet",
      "binned_outputs/square_016um/spatial/scalefactors_json.json",
    ]);
    expect(screen.getByTestId("not-sent")).toHaveTextContent(/another bin size of the run/);
    expect(screen.getByTestId("not-sent")).toHaveTextContent(/raw matrix/);
  });

  it("sends one bin folder without asking, and refuses a 2 um one or an archive with a sentence", async () => {
    render(<DataPanel />);
    await act(async () => {
      fireEvent.change(screen.getByTestId("folder-input"), { target: { files: hdFiles("002") } });
    });
    expect(uploadIngestFiles).not.toHaveBeenCalled();
    expect(screen.getByText(/use the square_008um or square_016um folder/)).toBeTruthy();
    await act(async () => {
      fireEvent.change(screen.getByTestId("folder-input"), { target: { files: [picked("binned_outputs.tar.gz")] } });
    });
    expect(uploadIngestFiles).not.toHaveBeenCalled();
    expect(screen.getByText(/unpack it and drop the binned_outputs folder/)).toBeTruthy();
    await act(async () => {
      fireEvent.change(screen.getByTestId("folder-input"), { target: { files: hdFiles("008") } });
    });
    expect((uploadIngestFiles.mock.calls[0]![1] as unknown[]).length).toBe(3);
  });

  it("shows the file's folder path on a single-file sample", async () => {
    uploadIngestFiles.mockResolvedValue({
      ...STATE,
      bundle: { ...STATE.bundle, issues: [], units: [{
        label: "pbmc", format: "tenx_h5", complete: true, missing_roles: [], issues: [], side_files: [],
        files: [bf("pbmc/outs/filtered_feature_bc_matrix.h5", { role: "data", kind: "tenx_h5" })],
      }] },
    });
    render(<DataPanel />);
    await act(async () => {
      fireEvent.change(screen.getByTestId("folder-input"), { target: { files: [picked("pbmc/outs/filtered_feature_bc_matrix.h5")] } });
    });
    expect(screen.getByTestId("unit-path")).toHaveTextContent("pbmc/outs/filtered_feature_bc_matrix.h5");
  });
});
