import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { CnvHeatmap, CnvSummary } from "@/api/cnv";
import type { DatasetInfo } from "@/api/types";

const api = vi.hoisted(() => ({
  getCnv: vi.fn(),
  getCnvHeatmap: vi.fn(),
  runCnv: vi.fn(),
}));
vi.mock("@/api/cnv", async (orig) => ({ ...(await orig<typeof import("@/api/cnv")>()), ...api }));
const refresh = vi.hoisted(() => vi.fn(async () => ({})));
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => refresh }));
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data }: { data: { type: string; z?: number[][]; name?: string }[] }) => (
    <div data-testid="plot">
      {data.map((t) => (t.type === "heatmap" ? `heatmap:${t.z?.length}x${t.z?.[0]?.length}` : `${t.type}:${t.name}`)).join(";")}
    </div>
  ),
}));

import { CnvSubtab } from "./CnvSubtab";
import { useDatasetStore } from "@/stores/datasetStore";
import { useCnvRunStore } from "@/stores/cnvRunStore";
import { useSettingsStore } from "@/stores/settingsStore";

const DEFAULTS = {
  window: 100, step: 10, exclude: ["chrY", "chrMT"], min_frac: 0.01, threshold: 1.5, min_window: 11,
  max_window: 501, max_step: 100, min_ref_cells: 10, high_ratio: 2,
};
const RUNNABLE = {
  ok: true, reason: null, genome: "human" as const, assembly: "GRCh38", genome_why: "", source: "log-normalised .raw",
  n_placed: 12000, match: { n_genes: 14000, by_id: 0, by_symbol: 12000, unplaced: 2000, excluded: 0 },
};
const EMPTY: CnvSummary = {
  available: false, runnable: RUNNABLE, defaults: DEFAULTS, running: false, n_cells_total: 300,
  estimate: { need_bytes: 300 << 20, available_bytes: 8 << 30, level: "fits", seconds: 40 },
};
const SAVED: CnvSummary = {
  ...EMPTY,
  available: true,
  run_id: "r1",
  assembly: "GRCh38",
  reference_column: "cell_type",
  reference_groups: ["T cell", "B cell"],
  n_reference_cells: 200,
  n_cells: 300,
  n_genes: 9000,
  n_windows: 800,
  params: { window: 100, step: 10, exclude: ["chrY", "chrMT"], min_frac: 0.01, threshold: 1.5 },
  by_group: {
    column: "cell_type",
    groups: [
      { group: "T cell", n: 100, mean: 0.01, q1: 0.005, median: 0.01, q3: 0.015, lo: 0, hi: 0.02, reference: true, ratio: 1 },
      { group: "Epithelial", n: 100, mean: 0.06, q1: 0.05, median: 0.06, q3: 0.07, lo: 0.03, hi: 0.09, reference: false, ratio: 6 },
    ],
  },
  clusters: [
    { cluster: "2", n: 90, mean: 0.06, ratio: 6, reference_share: 0, gains: ["chr7"], losses: ["chr10"], high: true },
    { cluster: "0", n: 210, mean: 0.01, ratio: 1, reference_share: 0.95, gains: [], losses: [], high: false },
  ],
  method: "Copy number inferred from expression...",
};
const HEAT: CnvHeatmap = {
  column: "cell_type",
  groups: [
    { group: "T cell", n_sampled: 2, n_total: 100, reference: true },
    { group: "Epithelial", n_sampled: 1, n_total: 100, reference: false },
  ],
  n_rows: 3, n_cols: 2, chromosomes: [{ chromosome: "chr1", start: 0, end: 1 }, { chromosome: "chr2", start: 1, end: 2 }],
  vmax: 0.3, scale: 0.3 / 127, values: btoa(String.fromCharCode(0, 0, 1, 255, 127, 129)), clusters: ["0", "0", "2"],
  n_cells_total: 300,
};

function setDataset() {
  useDatasetStore.setState({
    currentDatasetId: "d1",
    currentDataset: {
      id: "d1", name: "d1", filename: "", n_cells: 300, n_genes: 14000, status: "ready", available_embeddings: [],
      created_at: "",
      obs_columns: [
        { name: "cell_type", dtype: "category", n_unique: 3, values: ["T cell", "B cell", "Epithelial"] },
        { name: "sample", dtype: "category", n_unique: 2, values: ["s1", "s2"] },
      ],
    } as DatasetInfo,
  });
}

describe("CnvSubtab", () => {
  beforeEach(() => {
    Object.values(api).forEach((f) => f.mockReset());
    api.getCnvHeatmap.mockResolvedValue(HEAT);
    useCnvRunStore.getState().reset();
    useCnvRunStore.setState({ version: 0 });
    setDataset();
  });

  it("suggests the normal groups, runs with them and shows progress, then the result", async () => {
    api.getCnv.mockResolvedValue(EMPTY);
    let finish: (s: CnvSummary) => void = () => {};
    api.runCnv.mockImplementation((_id, _body, h) => {
      h.onStart?.({ n_cells: 300, n_genes: 9000, n_windows: 800, genome: "human", assembly: "GRCh38",
        reference_column: "cell_type", reference_groups: ["T cell", "B cell"], n_reference_cells: 200,
        source: "log-normalised .raw", n_low: 10, notes: [], estimate_seconds: 40 });
      h.onProgress?.({ stage: "windows", done: 100, total: 300, elapsed_s: 6 });
      return new Promise((resolve) => { finish = resolve; });
    });
    render(<CnvSubtab />);
    await waitFor(() => screen.getByText("Infer copy number"));
    expect(screen.getByText(/About 40 s and 300 MB of memory/)).toBeTruthy();
    // The cell-type column and its immune groups are suggested; the tumour is left out.
    // The suggestion lands after the first render (an effect): wait for it, CI is slower.
    await waitFor(() => expect((screen.getByLabelText("T cell") as HTMLInputElement).checked).toBe(true));
    expect((screen.getByLabelText("Reference column") as HTMLSelectElement).value).toBe("cell_type");
    expect((screen.getByLabelText("Epithelial") as HTMLInputElement).checked).toBe(false);
    fireEvent.click(screen.getByLabelText("leave out X"));
    fireEvent.click(screen.getByText("Infer copy number"));
    expect(api.runCnv.mock.calls[0]![1]).toEqual({
      reference_column: "cell_type", reference_groups: ["T cell", "B cell"], window: 100, step: 10,
      exclude_chromosomes: ["chrY", "chrMT", "chrX"],
    });
    await waitFor(() => expect(screen.getByTestId("cnv-progress").textContent).toMatch(/100 of 300 cells smoothed/));
    expect(screen.getByText("Cancel")).toBeTruthy();
    api.getCnv.mockResolvedValue(SAVED);
    finish(SAVED);
    await waitFor(() => screen.getByTestId("cnv-heatmap"));
    expect(screen.getByText("Run again")).toBeTruthy();
    expect(refresh).toHaveBeenCalledWith("d1", "cnv_score"); // the new columns reach the pickers
  });

  it("draws the heatmap, the score boxes and the cluster table, and colours the cells", async () => {
    api.getCnv.mockResolvedValue(SAVED);
    render(<CnvSubtab />);
    await waitFor(() => screen.getByTestId("cnv-heatmap"));
    const plots = screen.getAllByTestId("plot").map((p) => p.textContent);
    expect(plots).toContain("heatmap:3x2");
    expect(plots).toContain("box:T cell;box:Epithelial");
    expect(screen.getByText("chr7 gain; chr10 loss")).toBeTruthy();
    expect(screen.getByText(/1 cluster with a high score/)).toBeTruthy();
    expect(screen.getByText(/relative to T cell, B cell/)).toBeTruthy();
    fireEvent.click(screen.getByText("CNV cluster"));
    expect(useSettingsStore.getState().colorBy).toBe("cnv_leiden");
    fireEvent.change(screen.getByLabelText("Group the rows by"), { target: { value: "sample" } });
    await waitFor(() => expect(api.getCnvHeatmap).toHaveBeenLastCalledWith("d1", "sample"));
    expect(api.getCnv).toHaveBeenLastCalledWith("d1", "sample");
  });

  it("says why a dataset cannot run and blocks an all-reference choice", async () => {
    api.getCnv.mockResolvedValueOnce({
      ...EMPTY, runnable: { ...RUNNABLE, ok: false, reason: "The species of this dataset's genes is unclear" },
    });
    const { unmount } = render(<CnvSubtab />);
    await waitFor(() => expect(screen.getByRole("alert").textContent).toMatch(/can't be inferred on this dataset: The species/));
    unmount();
    api.getCnv.mockResolvedValue(EMPTY);
    render(<CnvSubtab />);
    await waitFor(() => screen.getByText("Infer copy number"));
    fireEvent.click(screen.getByLabelText("Epithelial"));
    expect(screen.getByText(/cannot all be the reference/)).toBeTruthy();
    expect((screen.getByText("Infer copy number").closest("button") as HTMLButtonElement).disabled).toBe(true);
  });
});
