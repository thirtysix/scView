import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

const api = vi.hoisted(() => ({
  getRExportInfo: vi.fn(),
  startRExport: vi.fn(),
  getRExportStatus: vi.fn(),
  rExportFileUrl: (id: string, job: string) => `/api/v1/datasets/${id}/export/rds/${job}`,
}));
vi.mock("@/api/rdsExport", () => api);
const exp = vi.hoisted(() => ({ downloadUrl: vi.fn() }));
vi.mock("@/api/export", () => exp);

import { RExportDialog } from "./RExportDialog";

const info = {
  format: "seurat",
  label: "Seurat object",
  needs: "SeuratObject >= 5",
  filename: "scview_pbmc_seurat.rds",
  n_cells: 13836,
  n_genes: 14053,
  data_from: "X",
  counts_from: "layers['counts']",
  embeddings: ["X_pca", "X_umap"],
  n_proteins: 0,
  cluster_column: "leiden",
  notes: [],
  dropped: { obsp: ["connectivities", "distances"] },
  needs_bytes: 1_500_000_000,
  limit_bytes: 8_000_000_000,
  fits: true,
  converter: "ok",
  running_job: null,
};

const report = {
  kept: ["raw counts (from layers['counts']) as the RNA counts layer", "X_umap as reduction 'umap' (2 dimensions)"],
  skipped: ["obsp: connectivities, distances - neighbour graphs (rebuild with FindNeighbors / scran::buildSNNGraph)"],
  renamed_features: 1,
};

describe("RExportDialog", () => {
  beforeEach(() => vi.clearAllMocks());

  it("shows what goes in, runs the job, downloads the file and lists what was left out", async () => {
    api.getRExportInfo.mockResolvedValue(info);
    api.startRExport.mockResolvedValue({ job: "j1", format: "seurat", status: "writing", message: "Preparing the data", filename: info.filename });
    api.getRExportStatus
      .mockResolvedValueOnce({ job: "j1", format: "seurat", status: "converting", message: "Building the Seurat object", filename: info.filename })
      .mockResolvedValueOnce({ job: "j1", format: "seurat", status: "done", message: "Wrote", filename: info.filename, report, size_bytes: 45_900_000 });

    render(<RExportDialog datasetId="abc" format="seurat" onClose={() => {}} pollMs={5} />);
    expect(await screen.findByTestId("rexport-contents")).toHaveTextContent(
      "13,836 cells × 14,053 genes · raw counts from layers['counts'] · log-normalised values from X · layouts: X_pca, X_umap · Idents: leiden",
    );
    expect(screen.getByText(/obsp: connectivities, distances/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "How to open it in R" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Download" }));
    expect(await screen.findByText("Building the Seurat object")).toBeInTheDocument();
    await waitFor(() => expect(exp.downloadUrl).toHaveBeenCalledWith("/api/v1/datasets/abc/export/rds/j1", info.filename));
    expect(api.startRExport).toHaveBeenCalledWith("abc", "seurat");
    expect(screen.getByText(/Downloaded scview_pbmc_seurat\.rds/)).toBeInTheDocument();
    expect(screen.getByText("Left out (1)")).toBeInTheDocument();
    expect(screen.getByText(/rebuild with FindNeighbors/)).toBeInTheDocument();
    expect(screen.getByText(/readRDS\("file\.rds"\)/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Download" })).not.toBeInTheDocument();
  });

  it("says when the converter is not running and does not start", async () => {
    api.getRExportInfo.mockResolvedValue({ ...info, format: "sce", converter: "down" });
    render(<RExportDialog datasetId="abc" format="sce" onClose={() => {}} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("The R converter is not running");
    expect(screen.getByRole("button", { name: "Download" })).toBeDisabled();
    expect(screen.getByRole("dialog", { name: "SingleCellExperiment (.rds)" })).toBeInTheDocument();
  });

  it("refuses a dataset too big for the converter's memory", async () => {
    api.getRExportInfo.mockResolvedValue({ ...info, fits: false, too_big: "Building this R object needs about 40.0 GB in the R converter, more than its 8.0 GB limit." });
    render(<RExportDialog datasetId="abc" format="seurat" onClose={() => {}} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("more than its 8.0 GB limit");
    expect(screen.getByRole("button", { name: "Download" })).toBeDisabled();
  });

  it("shows the converter's error when the job fails", async () => {
    api.getRExportInfo.mockResolvedValue(info);
    api.startRExport.mockResolvedValue({ job: "j2", format: "seurat", status: "writing", message: "Preparing", filename: info.filename });
    api.getRExportStatus.mockResolvedValue({ job: "j2", format: "seurat", status: "error", message: "Failed", filename: info.filename, error: "Export failed: out of memory" });
    render(<RExportDialog datasetId="abc" format="seurat" onClose={() => {}} pollMs={5} />);
    fireEvent.click(await screen.findByRole("button", { name: "Download" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Export failed: out of memory");
    expect(exp.downloadUrl).not.toHaveBeenCalled();
  });

  it("follows a job started earlier instead of starting another", async () => {
    api.getRExportInfo.mockResolvedValue({ ...info, running_job: "j0" });
    api.getRExportStatus.mockResolvedValue({ job: "j0", format: "seurat", status: "done", message: "Wrote", filename: info.filename, report });
    render(<RExportDialog datasetId="abc" format="seurat" onClose={() => {}} pollMs={5} />);
    await waitFor(() => expect(exp.downloadUrl).toHaveBeenCalledWith("/api/v1/datasets/abc/export/rds/j0", info.filename));
    expect(api.startRExport).not.toHaveBeenCalled();
  });
});
