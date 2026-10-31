import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

const api = vi.hoisted(() => ({
  getFragments: vi.fn(),
  deleteFragments: vi.fn(),
  getFragmentLengths: vi.fn(),
  getQcValues: vi.fn(),
  runFragmentQc: vi.fn(),
  uploadFragmentFile: vi.fn(),
}));
vi.mock("@/api/atacFragments", () => api);
const refreshDataset = vi.hoisted(() => vi.fn());
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => refreshDataset }));
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data }: { data: { type: string; name?: string }[] }) => (
    <div data-testid="plot">{data.map((t) => `${t.type}:${t.name ?? ""}`).join(";")}</div>
  ),
}));

import type { FragmentQc, FragmentStatus } from "@/api/atacFragments";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { AtacFragmentQc, uploadOrder } from "./AtacFragmentQc";

const QC: FragmentQc = {
  created_at: "2026-10-31T00:00:00Z",
  n_cells: 100,
  n_cells_with_fragments: 100,
  n_lines: 2000,
  n_fragments_in_cells: 1800,
  medians: { tss_enrichment: 5.79, nucleosome_signal: 0.67, n_fragments: 17305 },
  length_histogram: { counts: [0, 1, 2, 3], max: 3, n: 6 },
  tss_profile: { start: -1000, bin: 5, fold: [1, 2, 1], n_tss: 19934, chromosomes: 24 },
  params: { genome: "GRCh38", annotation: "GENCODE v32 basic", center_bp: 1001, flank_bp: 100, file: "f.tsv.gz" },
  seconds: 106,
  notes: ["A note."],
};

const empty: FragmentStatus = {
  attached: false, fragments: false, index: false, missing: null, name: null, size_bytes: null, barcode_share: null,
  lines_checked: null, cells_seen: null, genome: null,
};
const attached: FragmentStatus = {
  ...empty, attached: true, fragments: true, index: true, name: "atac_fragments.tsv.gz", barcode_share: 0.91,
  lines_checked: 1000000, genome: "GRCh38",
  estimate: { peak_bytes: 5e8, seconds: 106, seconds_range: [85, 160], note: "One pass over the whole file, roughly 1 to 3 min on 2 cores." },
};

const values = {
  group_by: "cell_type",
  max_per_group: 2000,
  groups: [{ name: "B", n: 40, n_shown: 40, tss_enrichment: [5, 6], nucleosome_signal: [0.5, null], n_fragments: [10, 20] },
           { name: "T", n: 60, n_shown: 60, tss_enrichment: [4, 5], nucleosome_signal: [0.7, 0.8], n_fragments: [10, 20] }],
};
const lengths = {
  group_by: "cell_type", bin: 5, x: [2.5, 7.5],
  groups: [{ name: "B", n_cells: 40, n_fragments: 5, share: [0.5, 0.5] }, { name: "T", n_cells: 60, n_fragments: 5, share: [0.5, 0.5] }],
};

beforeEach(() => {
  vi.clearAllMocks();
  refreshDataset.mockResolvedValue({});
  api.getQcValues.mockResolvedValue(values);
  api.getFragmentLengths.mockResolvedValue(lengths);
  useDatasetStore.setState({
    currentDataset: {
      id: "d",
      n_cells: 100,
      obs_columns: [{ name: "cell_type", dtype: "category", n_unique: 2, values: ["B", "T"] },
                    { name: "n_counts", dtype: "float", n_unique: 90 }],
    } as unknown as DatasetInfo,
    currentDatasetId: "d",
  });
  useSettingsStore.setState({ colorBy: "cell_type" });
});

describe("AtacFragmentQc", () => {
  it("asks for the two files when nothing is attached, and uploads them index last", async () => {
    api.getFragments.mockResolvedValue(empty);
    api.uploadFragmentFile.mockResolvedValue({ ...empty, index: true, fragments: true, attached: true });
    render(<AtacFragmentQc datasetId="d" />);
    expect(await screen.findByText(/Cell Ranger ARC writes/)).toBeTruthy();
    const input = screen.getByLabelText("Fragments file and index");
    const tbi = new File(["x"], "atac_fragments.tsv.gz.tbi");
    const frag = new File(["y"], "atac_fragments.tsv.gz");
    fireEvent.change(input, { target: { files: [tbi, frag] } });
    await waitFor(() => expect(api.uploadFragmentFile).toHaveBeenCalledTimes(2));
    expect(api.uploadFragmentFile.mock.calls.map((c) => (c[1] as File).name)).toEqual([frag.name, tbi.name]);
  });

  it("says what is missing and shows an upload's refusal", async () => {
    api.getFragments.mockResolvedValue({ ...empty, fragments: true, missing: "The tabix index is still missing." });
    api.uploadFragmentFile.mockRejectedValue(new Error("Only 0 of the first 1,000 fragments belong to this dataset."));
    render(<AtacFragmentQc datasetId="d" />);
    expect((await screen.findByTestId("atac-missing")).textContent).toContain("tabix index");
    fireEvent.change(screen.getByLabelText("Fragments file and index"), { target: { files: [new File(["y"], "x.tsv.gz")] } });
    expect(await screen.findByText(/Only 0 of the first/)).toBeTruthy();
  });

  it("says what a run needs, runs it, and cancels by aborting", async () => {
    api.getFragments.mockResolvedValueOnce(attached).mockResolvedValue({ ...attached, qc: QC });
    let signal: AbortSignal | undefined;
    let finish: () => void = () => undefined;
    api.runFragmentQc.mockImplementation((_d, on, s) => {
      signal = s;
      on.progress({ fraction: 0.5, message: "100 fragments read (50% of the file)" });
      return new Promise<void>((resolve) => { finish = resolve; });
    });
    render(<AtacFragmentQc datasetId="d" />);
    expect((await screen.findByTestId("atac-estimate")).textContent).toContain("One pass");
    expect(screen.getByTestId("atac-attached").textContent).toContain("91.0%");
    fireEvent.click(screen.getByRole("button", { name: "Run fragment QC" }));
    expect(await screen.findByText(/50% of the file/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(signal?.aborted).toBe(true);
    finish();
    expect(await screen.findByTestId("atac-qc")).toBeTruthy();
    expect(refreshDataset).toHaveBeenCalledWith("d", "atac_tss_enrichment");
  });

  it("shows the medians, histograms, violins by group and the ladder caption", async () => {
    api.getFragments.mockResolvedValue({ ...attached, qc: QC });
    render(<AtacFragmentQc datasetId="d" />);
    const box = await screen.findByTestId("atac-qc");
    expect(box.textContent).toContain("5.79");
    expect(box.textContent).toContain("0.67");
    expect(box.textContent).toContain("A note.");
    expect(box.textContent).toContain("ladder");
    await waitFor(() => expect(api.getQcValues).toHaveBeenCalledWith("d", "cell_type"));
    const plots = await screen.findAllByTestId("plot");
    const kinds = plots.map((p) => p.textContent).join("|");
    expect(kinds).toContain("bar");
    expect(kinds).toContain("violin:B");
    expect(kinds).toContain("scatter:T"); // lengths by group
  });

  it("Remove detaches", async () => {
    api.getFragments.mockResolvedValueOnce(attached).mockResolvedValue(empty);
    api.deleteFragments.mockResolvedValue({ deleted: true });
    render(<AtacFragmentQc datasetId="d" />);
    fireEvent.click(await screen.findByRole("button", { name: "Remove" }));
    await waitFor(() => expect(api.deleteFragments).toHaveBeenCalledWith("d"));
    expect(await screen.findByText(/Cell Ranger ARC writes/)).toBeTruthy();
  });

  it("orders uploads index last", () => {
    expect(uploadOrder([new File([], "a.TBI"), new File([], "a.tsv.gz")]).map((f) => f.name)).toEqual(["a.tsv.gz", "a.TBI"]);
  });
});
