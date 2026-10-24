import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

const api = vi.hoisted(() => ({
  getExportFormatInfo: vi.fn(),
  buildExport: vi.fn(),
  exportFileUrl: (id: string, f: string) => `/api/v1/datasets/${id}/export/files/${f}`,
}));
vi.mock("@/api/exportFormats", () => api);
const exp = vi.hoisted(() => ({ downloadUrl: vi.fn() }));
vi.mock("@/api/export", () => exp);

import { ExportFormatDialog } from "./ExportFormatDialog";

const base = {
  filename: "scview_pbmc_cellxgene.h5ad",
  size_bytes: 27_000_000,
  large: false,
  n_cells: 13836,
  n_genes: 14053,
  embeddings: [
    { key: "X_umap", label: "UMAP", n_dims: 2, default: true, export_name: "X_umap" },
    { key: "X_pca", label: "PCA", n_dims: 50, default: false, export_name: "X_pca" },
  ],
  label_columns: [{ name: "leiden", n_values: 12, default: true }],
  has_variable_genes: true,
  x_from: "X",
  obs_columns: ["leiden"],
  exported_embeddings: ["X_umap"],
  dropped: { uns: ["scview_provenance"] },
  notes: [],
};

describe("ExportFormatDialog", () => {
  beforeEach(() => vi.clearAllMocks());

  it("shows size, readiness and builds then downloads the CELLxGENE file", async () => {
    api.getExportFormatInfo.mockResolvedValue({
      ...base,
      format: "cellxgene",
      has_counts: true,
      raw_from: null,
      readiness: {
        schema_version: "5.3.0",
        schema_url: "https://example.org/schema",
        items: [{ key: "donor_id", label: "donor id", status: "missing", detail: "Not in the cell metadata." }],
        n_ok: 0,
        n_items: 1,
        ready: false,
        hints: [],
      },
    });
    api.buildExport.mockImplementation(async (_id, _f, _o, onProgress) => {
      onProgress("Writing");
      return { file: "cellxgene-0123456789abcdef.h5ad", filename: "scview_pbmc_cellxgene.h5ad", size_bytes: 1 };
    });
    const onClose = vi.fn();
    render(<ExportFormatDialog datasetId="abc" format="cellxgene" onClose={onClose} />);
    expect(await screen.findByTestId("export-size")).toHaveTextContent("About 27.0 MB");
    expect(screen.getByText(/0 of 1 checks pass/)).toBeInTheDocument();
    expect(screen.getByText("missing")).toBeInTheDocument();
    expect(screen.getByText(/50 columns, first 2 exported/)).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText(/Include raw counts/));
    await waitFor(() =>
      expect(api.getExportFormatInfo).toHaveBeenLastCalledWith("abc", "cellxgene", {
        embeddings: ["X_umap"],
        include_raw: true,
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Download" }));
    await waitFor(() =>
      expect(exp.downloadUrl).toHaveBeenCalledWith(
        "/api/v1/datasets/abc/export/files/cellxgene-0123456789abcdef.h5ad",
        "scview_pbmc_cellxgene.h5ad",
      ),
    );
    expect(onClose).toHaveBeenCalled();
  });

  it("shows a refusal and keeps Download off without an embedding", async () => {
    api.getExportFormatInfo.mockResolvedValue({ ...base, format: "vitessce" });
    render(<ExportFormatDialog datasetId="abc" format="vitessce" onClose={() => {}} />);
    const umap = await screen.findByLabelText(/UMAP/);
    expect(screen.getByLabelText(/leiden/)).toBeChecked();
    fireEvent.click(umap);
    expect(screen.getByRole("button", { name: "Download" })).toBeDisabled();
    api.getExportFormatInfo.mockRejectedValueOnce(new Error("No variable genes"));
    fireEvent.click(umap);
    expect(await screen.findByRole("alert")).toHaveTextContent("No variable genes");
  });
});
