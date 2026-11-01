import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const api = vi.hoisted(() => ({
  getTranscriptStatus: vi.fn(),
  deleteTranscripts: vi.fn(),
  runTranscriptIndex: vi.fn(),
  uploadTranscripts: vi.fn(),
  uploadOutlines: vi.fn(),
  refresh: vi.fn(),
}));
vi.mock("@/api/transcripts", () => api);
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => api.refresh }));

import type { DatasetInfo, SpatialInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useTissueViewStore } from "@/stores/tissueViewStore";
import { TranscriptsSubtab } from "./TranscriptsSubtab";

const library = { id: "r1", n_obs: 3, images: {}, spot_diameter_fullres: null, bounds: [0, 0, 1, 1] as [number, number, number, number] };
const spatial = (over: Partial<SpatialInfo> = {}): SpatialInfo => ({
  technology: "xenium",
  unit: "cell",
  library_column: "library_id",
  libraries: [library],
  outlines: ["cell", "nucleus"],
  transcripts: false,
  ...over,
});
const open = (s: SpatialInfo) =>
  useDatasetStore.setState({ currentDataset: { id: "ds1", name: "x", spatial: s } as unknown as DatasetInfo, currentDatasetId: "ds1" });

const SUMMARY = {
  source: "transcripts.parquet",
  n_rows: 42638083,
  n_kept: 34442716,
  n_low_quality: 7990819,
  n_control: 204548,
  min_qv: 20,
  n_genes: 313,
  n_assigned: 32102954,
  bounds: [0, 0, 1, 1],
  check: { available: true, n_genes: 313, r: 1, median_abs_diff: 0, max_abs_diff: 0, max_diff_gene: "A", n_equal: 313 },
  seconds: 12,
};
const attached = {
  attached: true,
  uploaded: false,
  max_genes: 2,
  default_limit: 300000,
  min_qv: 20,
  summary: SUMMARY,
  genes: [{ name: "ERBB2", n: 5 }, { name: "KRT7", n: 3 }, { name: "CD3E", n: 1 }],
  disk_bytes: 312 * 1024 ** 2,
};

beforeEach(() => {
  vi.clearAllMocks();
  useTissueViewStore.setState({ transcriptGenes: [], transcriptShown: null, transcriptDotSize: 1 });
  useSettingsStore.setState({ embedding: "spatial" });
});

describe("TranscriptsSubtab", () => {
  it("without a store says what to upload and from where", async () => {
    open(spatial());
    api.getTranscriptStatus.mockResolvedValue({ attached: false, uploaded: false, max_genes: 8, min_qv: 20 });
    render(<TranscriptsSubtab />);
    await screen.findByLabelText("transcripts.parquet");
    expect(screen.getByTestId("transcripts-subtab").textContent).toContain("from the Xenium output folder");
    expect(screen.queryByText("Index transcripts")).toBeNull();
  });

  it("sends the file, says what indexing needs, then runs it", async () => {
    open(spatial());
    api.getTranscriptStatus.mockResolvedValueOnce({ attached: false, uploaded: false, max_genes: 8, min_qv: 20 });
    api.uploadTranscripts.mockResolvedValue({
      attached: false,
      uploaded: true,
      max_genes: 8,
      min_qv: 20,
      estimate: { rows: 42638083, disk_bytes: 311e6, peak_bytes: 0.4 * 1024 ** 3, seconds: 14, seconds_range: [10, 35] },
    });
    render(<TranscriptsSubtab />);
    const input = await screen.findByLabelText("transcripts.parquet");
    fireEvent.change(input, { target: { files: [new File(["x"], "transcripts.parquet")] } });
    const est = await screen.findByTestId("transcripts-estimate");
    expect(est.textContent).toContain("42,638,083 rows");
    expect(est.textContent).toContain("297 MB of disk");
    expect(est.textContent).toContain("about 0.4 GB of memory");

    api.runTranscriptIndex.mockImplementation(async (_id, on) => on.result?.(SUMMARY));
    api.getTranscriptStatus.mockResolvedValue(attached);
    fireEvent.click(screen.getByText("Index transcripts"));
    await screen.findByTestId("transcripts-summary");
    expect(api.refresh).toHaveBeenCalledWith("ds1");
  });

  it("states what was dropped and the check against the matrix", async () => {
    open(spatial({ transcripts: true }));
    api.getTranscriptStatus.mockResolvedValue(attached);
    render(<TranscriptsSubtab />);
    const s = await screen.findByTestId("transcripts-summary");
    expect(s.textContent).toContain("34,442,716 transcripts kept of 42,638,083 read");
    expect(s.textContent).toContain("7,990,819 under q 20");
    expect(s.textContent).toContain("204,548 of control features");
    expect(screen.getByTestId("transcripts-check").textContent).toContain("r = 1.0000");
    expect(screen.getByTestId("transcripts-subtab").textContent).toContain("An h5ad export carries neither outlines nor transcripts");
  });

  it("says plainly when the check could not run", async () => {
    open(spatial({ transcripts: true }));
    api.getTranscriptStatus.mockResolvedValue({
      ...attached,
      summary: { ...SUMMARY, check: { available: false, note: "No cell_feature_matrix.h5 among this dataset's originals to check against." } },
    });
    render(<TranscriptsSubtab />);
    expect((await screen.findByTestId("transcripts-check")).textContent).toContain("No cell_feature_matrix.h5");
  });

  it("adds genes of the panel only, up to the most it can draw, with a colour each", async () => {
    open(spatial({ transcripts: true }));
    api.getTranscriptStatus.mockResolvedValue(attached);
    render(<TranscriptsSubtab />);
    const input = await screen.findByLabelText("Add a gene");
    const add = (g: string) => {
      fireEvent.change(input, { target: { value: g } });
      fireEvent.submit(input.closest("form")!);
    };
    add("NOPE");
    expect(useTissueViewStore.getState().transcriptGenes).toEqual([]);
    add("erbb2");
    add("KRT7");
    expect(useTissueViewStore.getState().transcriptGenes).toEqual(["ERBB2", "KRT7"]);
    expect(input).toHaveProperty("disabled", true);
    fireEvent.click(screen.getByLabelText("Remove ERBB2"));
    expect(useTissueViewStore.getState().transcriptGenes).toEqual(["KRT7"]);
  });

  it("shows how many of those in view are drawn", async () => {
    open(spatial({ transcripts: true }));
    api.getTranscriptStatus.mockResolvedValue(attached);
    useTissueViewStore.setState({ transcriptGenes: ["ERBB2"], transcriptShown: { shown: 300000, total: 3348837, perGene: [3348837] } });
    render(<TranscriptsSubtab />);
    expect((await screen.findByTestId("transcripts-shown")).textContent).toContain("Showing 300,000 of 3,348,837");
  });

  it("a merged dataset is told why it has none", async () => {
    open(spatial({ libraries: [library, { ...library, id: "r2" }], outlines: [] }));
    api.getTranscriptStatus.mockResolvedValue({ attached: false, uploaded: false, max_genes: 8, min_qv: 20 });
    render(<TranscriptsSubtab />);
    await waitFor(() => expect(screen.getByTestId("transcripts-subtab").textContent).toContain("merges 2 sections"));
    expect(screen.queryByLabelText("transcripts.parquet")).toBeNull();
    expect(screen.queryByLabelText("Boundaries files")).toBeNull();
  });

  it("offers the boundaries files to a dataset without them, and refreshes it", async () => {
    open(spatial({ outlines: [] }));
    api.getTranscriptStatus.mockResolvedValue({ attached: false, uploaded: false, max_genes: 8, min_qv: 20 });
    api.uploadOutlines.mockResolvedValue({ kind: "cell", outlines: ["cell"] });
    render(<TranscriptsSubtab />);
    const input = await screen.findByLabelText("Boundaries files");
    expect(screen.getByLabelText("Cell outlines").textContent).toContain("cell_boundaries.parquet and nucleus_boundaries.parquet");
    fireEvent.change(input, { target: { files: [new File(["x"], "cell_boundaries.parquet")] } });
    await waitFor(() => expect(api.refresh).toHaveBeenCalledWith("ds1"));
  });
});
