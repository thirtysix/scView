import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

const api = vi.hoisted(() => ({
  getLatestLinks: vi.fn(),
  getLinksEstimate: vi.fn(),
  getLinksLocus: vi.fn(),
  getLinksPage: vi.fn(),
  linksCsvUrl: vi.fn(() => "/csv"),
  runAtacLinks: vi.fn(),
}));
vi.mock("@/api/atacLinks", () => api);

import type { AtacState } from "@/api/atac";
import type { LinksPage } from "@/api/atacLinks";
import type { DatasetInfo } from "@/api/types";
import { useAtacRunStore } from "@/stores/atacRunStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { AtacLinksCard } from "./AtacLinksCard";

const STATE = { available: true, running: null } as unknown as AtacState;

const PAGE: LinksPage = {
  id: "r1",
  total: 2,
  offset: 0,
  limit: 100,
  sort: "p",
  descending: false,
  rows: [
    { gene: "LYZ", peak: "chr12:100-200", distance_to_tss: -1200, score: 0.41, z: 8.2, p: 1e-16, padj: 1e-14 },
    { gene: "MS4A1", peak: "chr11:300-400", distance_to_tss: 0, score: -0.12, z: -2.1, p: 0.018, padj: 0.04 },
  ],
  counts: { n_genes_asked: 3, n_genes_tested: 2, n_genes_few_cells: 1, n_genes_few_peaks: 0, n_pairs_cis: 50, n_pairs_tested: 20, n_links: 2,
            n_links_padj: 2, n_peaks_kept: 90, n_peaks_unknown: 51, n_cells: 100 },
  params: { genes: "variable", cells_column: null, cells_value: null },
  method: "Signac's LinkPeaks",
  caption: "A link is a correlation: a candidate regulatory element, never proof of regulation.",
  seconds: 12,
  created_at: "2026-11-02",
  missing: [],
  stale: false,
};

const ESTIMATE = {
  genes: "variable", n_genes: 2000, n_asked: 2000, missing: [], n_cells: 100, n_dataset_cells: 100, n_peaks: 90,
  estimate_seconds: 60, estimate_bytes: 1024 ** 3, estimate_disk_bytes: 1024 ** 3,
  genome: { genome: "GRCh38", source: "UCSC hg38", size: 835_393_456, downloaded: false },
};

beforeEach(() => {
  vi.clearAllMocks();
  useAtacRunStore.setState({ datasetId: null, kind: null, running: false, links: null, error: null, warnings: [], progress: null, version: 0 });
  useDatasetStore.setState({
    currentDataset: {
      id: "d", n_cells: 100,
      obs_columns: [{ name: "celltype", dtype: "category", n_unique: 2, values: ["B", "T"] }],
    } as unknown as DatasetInfo,
    currentDatasetId: "d",
  });
  api.getLatestLinks.mockResolvedValue({ result: null, runs: [] });
  api.getLinksEstimate.mockResolvedValue(ESTIMATE);
  api.getLinksPage.mockResolvedValue(PAGE);
});

describe("AtacLinksCard", () => {
  it("says what the choice costs and a first download before anything runs", async () => {
    render(<AtacLinksCard datasetId="d" state={STATE} columns={["celltype"]} onLocus={() => {}} />);
    const line = await screen.findByTestId("atac-links-estimate");
    expect(line.textContent).toContain("2,000 genes");
    expect(line.textContent).toContain("downloads the UCSC hg38 sequence");
    expect(api.getLinksEstimate).toHaveBeenCalledWith("d", { genes: "variable", cells_column: null, cells_value: null });
  });

  it("needs a list when a list is chosen and groups when a column is chosen", async () => {
    render(<AtacLinksCard datasetId="d" state={STATE} columns={["celltype"]} onLocus={() => {}} />);
    const button = screen.getByRole("button", { name: "Find links" });
    fireEvent.change(screen.getByLabelText("Genes"), { target: { value: "list" } });
    expect(button).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Gene list"), { target: { value: "LYZ, MS4A1" } });
    expect(button).not.toBeDisabled();
    fireEvent.change(screen.getByLabelText("Genes"), { target: { value: "variable" } });
    fireEvent.change(screen.getByLabelText("Cells from"), { target: { value: "celltype" } });
    expect(button).toBeDisabled();
    fireEvent.click(screen.getByLabelText("B"));
    expect(button).not.toBeDisabled();
    await waitFor(() =>
      expect(api.getLinksEstimate).toHaveBeenLastCalledWith("d", { genes: "variable", cells_column: "celltype", cells_value: ["B"] }));
    api.runAtacLinks.mockResolvedValue(PAGE);
    fireEvent.click(button);
    await waitFor(() => expect(api.runAtacLinks).toHaveBeenCalled());
    expect(api.runAtacLinks.mock.calls[0]![1]).toEqual({ genes: "variable", cells_column: "celltype", cells_value: ["B"] });
  });

  it("shows the table with its caption, sorts on the backend, and a row opens the locus", async () => {
    api.getLatestLinks.mockResolvedValue({ result: PAGE, runs: [{ id: "r1", label: "x", created_at: 1, stale: false }] });
    const locus = { gene: "LYZ", chrom: "chr12", tss: 5, start: 0, end: 10, arcs: [], peaks: [], genes: [], note: null };
    api.getLinksLocus.mockResolvedValue(locus);
    const onLocus = vi.fn();
    render(<AtacLinksCard datasetId="d" state={STATE} columns={[]} onLocus={onLocus} />);
    expect(await screen.findByText("LYZ")).toBeInTheDocument();
    expect(screen.getByText(/never proof of regulation/)).toBeInTheDocument();
    expect(screen.getByText(/51 peaks left out/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    await waitFor(() => expect(api.getLinksPage).toHaveBeenLastCalledWith("d", "r1", expect.objectContaining({ sort: "abs_score", descending: true })));
    fireEvent.click(screen.getByText("LYZ"));
    await waitFor(() => expect(onLocus).toHaveBeenCalledWith(locus, "chr12:100-200"));
    expect(api.getLinksLocus).toHaveBeenCalledWith("d", "r1", "LYZ");
  });

  it("says plainly when the result no longer matches the cells", async () => {
    api.getLatestLinks.mockResolvedValue({ result: PAGE, runs: [] });
    api.getLinksPage.mockResolvedValue({ ...PAGE, stale: true });
    render(<AtacLinksCard datasetId="d" state={STATE} columns={[]} onLocus={() => {}} />);
    expect(await screen.findByText(/changed since this ran/)).toBeInTheDocument();
  });

  it("offers kept runs and opens one, marking older data", async () => {
    api.getLatestLinks.mockResolvedValue({
      result: PAGE,
      runs: [{ id: "r1", label: "Variable", created_at: 2, stale: false }, { id: "r0", label: "All", created_at: 1, stale: true }],
    });
    render(<AtacLinksCard datasetId="d" state={STATE} columns={[]} onLocus={() => {}} />);
    const select = await screen.findByLabelText("Kept runs");
    expect(screen.getByText("All (older data)")).toBeInTheDocument();
    fireEvent.change(select, { target: { value: "r0" } });
    await waitFor(() => expect(api.getLinksPage).toHaveBeenCalledWith("d", "r0", expect.anything()));
  });

  it("names a backend refusal", async () => {
    api.getLinksEstimate.mockRejectedValue(new Error("This dataset has no variable genes yet: run the pipeline's variable genes step."));
    render(<AtacLinksCard datasetId="d" state={STATE} columns={[]} onLocus={() => {}} />);
    expect(await screen.findByText(/no variable genes yet/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Find links" })).toBeDisabled();
  });
});
