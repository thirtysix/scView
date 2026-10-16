import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

const api = vi.hoisted(() => ({
  getH5adInfo: vi.fn(),
  downloadTable: vi.fn(),
  downloadUrl: vi.fn(),
  h5adDownloadUrl: (id: string) => `/api/v1/datasets/${id}/export/h5ad`,
}));
vi.mock("@/api/export", () => api);

import { DatasetDownloadMenu } from "./DatasetDownloadMenu";

describe("DatasetDownloadMenu", () => {
  beforeEach(() => vi.clearAllMocks());

  it("shows the .h5ad size before downloading it", async () => {
    api.getH5adInfo.mockResolvedValue({
      filename: "scview_pbmc.h5ad",
      size_bytes: 1_200_000_000,
      source: "derived",
      source_label: "processed",
    });
    render(<DatasetDownloadMenu datasetId="abc" />);
    fireEvent.click(screen.getByRole("button", { name: /download/i }));
    expect(screen.getByText("Checking size…")).toBeInTheDocument();
    expect(await screen.findByText("Processed · 1.20 GB")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("menuitem", { name: /Dataset \(\.h5ad\)/ }));
    expect(api.downloadUrl).toHaveBeenCalledWith("/api/v1/datasets/abc/export/h5ad", "scview_pbmc.h5ad");
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });

  it("keeps the .h5ad item disabled when there is no file", async () => {
    api.getH5adInfo.mockRejectedValue(new Error("404"));
    render(<DatasetDownloadMenu datasetId="abc" compact />);
    fireEvent.click(screen.getByRole("button", { name: "Download" }));
    expect(await screen.findByText("No .h5ad file available yet")).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: /Dataset \(\.h5ad\)/ })).toBeDisabled();
  });

  it("asks for the marker table and shows a failure in the menu", async () => {
    api.getH5adInfo.mockResolvedValue({ filename: "f.h5ad", size_bytes: 10, source: "uploads", source_label: "as uploaded" });
    api.downloadTable.mockRejectedValue(new Error("No marker genes found in this dataset."));
    render(<DatasetDownloadMenu datasetId="abc" />);
    fireEvent.click(screen.getByRole("button", { name: /download/i }));
    fireEvent.click(screen.getByRole("menuitem", { name: /Marker genes \(\.xlsx\)/ }));
    expect(api.downloadTable).toHaveBeenCalledWith("abc", "markers", "xlsx");
    expect(await screen.findByRole("alert")).toHaveTextContent("No marker genes found");
    await waitFor(() => expect(screen.getByRole("menu")).toBeInTheDocument());
  });
});
