import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { CompositionResult } from "@/api/composition";
import type { DatasetInfo } from "@/api/types";

const { testComposition, getDataset, downloadCsv } = vi.hoisted(() => ({
  testComposition: vi.fn(),
  getDataset: vi.fn(),
  downloadCsv: vi.fn(),
}));
vi.mock("@/api/composition", () => ({ testComposition }));
vi.mock("@/api/datasets", () => ({ getDataset }));
vi.mock("@/lib/csv", () => ({ downloadCsv }));
vi.mock("@/components/plots/Plot", () => ({ default: () => <div data-testid="plot" /> }));

import { CompositionTest } from "./CompositionTest";

const KANG_INFO: DatasetInfo = {
  id: "kang",
  name: "Kang",
  filename: "kang.h5ad",
  n_cells: 13836,
  n_genes: 14053,
  status: "ready",
  available_embeddings: [],
  obs_columns: [
    { name: "n_counts", dtype: "float64", n_unique: 3413 },
    { name: "orig_ident", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] },
    { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] },
    { name: "cluster", dtype: "category", n_unique: 3, values: ["B", "CD14 Mono", "pDC"] },
    { name: "Donor", dtype: "category", n_unique: 8 },
  ],
  created_at: "2026-10-17T00:00:00Z",
};

const RESULT: CompositionResult = {
  condition_key: "stim",
  sample_key: "Donor",
  sample_key_detected: true,
  group_key: "cluster",
  conditions: ["CTRL", "STIM"],
  reference: "CTRL",
  transform: "logit",
  robust: true,
  n_outliers: 0,
  tested: true,
  design: "paired",
  test: "moderated t",
  df_residual: 7,
  df_prior: 2.06,
  n_samples: { CTRL: 8, STIM: 8 },
  n_cells: { CTRL: 6573, STIM: 7263 },
  n_significant: 1,
  fdr_threshold: 0.05,
  n_significant_cell_level: 3,
  groups: [
    { group: "B", props: { CTRL: 0.064, STIM: 0.072 }, n_cells: { CTRL: 420, STIM: 530 },
      fold_change: 1.13, stat: 1.87, pval: 0.094, fdr: 0.14, df_prior: 2.06 },
    { group: "CD14 Mono", props: { CTRL: 0.35, STIM: 0.3 }, n_cells: { CTRL: 2300, STIM: 1925 },
      fold_change: 0.86, stat: -3.82, pval: 0.004, fdr: 0.012, df_prior: 2.06 },
    { group: "pDC", props: { CTRL: 0.0057, STIM: 0.0108 }, n_cells: { CTRL: 38, STIM: 78 },
      fold_change: 1.91, stat: 1.14, pval: 0.28, fdr: 0.28, df_prior: 2.06 },
  ],
  samples: [
    { sample: "SNG-101", condition: "CTRL", n_cells: 498, props: [0.06, 0.35, 0.005] },
    { sample: "SNG-101", condition: "STIM", n_cells: 708, props: [0.07, 0.3, 0.01] },
  ],
  warnings: [],
  notes: [],
  dropped_samples: [],
};

describe("CompositionTest", () => {
  beforeEach(() => {
    testComposition.mockReset().mockResolvedValue(RESULT);
    getDataset.mockReset();
    downloadCsv.mockReset();
  });

  it("tests the condition-like column by the cell types, with the donors detected", async () => {
    render(<CompositionTest datasetId="kang" info={KANG_INFO} />);
    expect(await screen.findByText("1 of 3 groups shift (FDR < 0.05)")).toBeInTheDocument();
    expect(testComposition).toHaveBeenCalledTimes(1);
    expect(testComposition).toHaveBeenCalledWith("kang", {
      condition_key: "stim",
      group_key: "cluster",
      transform: "logit",
    });
    expect(screen.getByLabelText("Sample column")).toHaveDisplayValue("Auto (Donor)");
    // Best first, the shifted one marked; fold change as a ratio.
    const rows = screen.getAllByRole("row").slice(1);
    expect(rows[0]!.textContent).toContain("CD14 Mono");
    expect(rows[0]!.textContent).toContain("×0.86");
    expect(screen.getByText(/paired by Donor \(CTRL 8, STIM 8 samples\)/)).toBeInTheDocument();
    // The cell-level caution carries this result's own numbers.
    expect(screen.getByText(/would call 3 of 3 groups significant/)).toBeInTheDocument();
    expect(screen.getByTestId("plot")).toBeInTheDocument();
  });

  it("re-tests with a chosen sample column, a subset and a reference", async () => {
    const info = {
      ...KANG_INFO,
      obs_columns: [
        ...KANG_INFO.obs_columns.filter((c) => c.name !== "stim"),
        { name: "stim", dtype: "category", n_unique: 3, values: ["CTRL", "IFN", "LPS"] },
      ],
    };
    render(<CompositionTest datasetId="kang" info={info} />);
    await screen.findByText("1 of 3 groups shift (FDR < 0.05)");
    fireEvent.change(screen.getByLabelText("Sample column"), { target: { value: "Donor" } });
    await waitFor(() =>
      expect(testComposition).toHaveBeenLastCalledWith("kang", expect.objectContaining({ sample_key: "Donor" })),
    );
    fireEvent.click(screen.getByRole("button", { name: "LPS" }));
    await waitFor(() =>
      expect(testComposition).toHaveBeenLastCalledWith(
        "kang", expect.objectContaining({ conditions: ["CTRL", "IFN"] })),
    );
    fireEvent.change(screen.getByLabelText("Reference condition"), { target: { value: "IFN" } });
    await waitFor(() =>
      expect(testComposition).toHaveBeenLastCalledWith("kang", expect.objectContaining({ reference: "IFN" })),
    );
  });

  it("shows proportions only, with the warning, when a condition has one sample", async () => {
    testComposition.mockResolvedValue({
      ...RESULT,
      tested: false,
      design: null,
      test: null,
      sample_key: "orig_ident",
      sample_key_detected: false,
      n_samples: { CTRL: 1, STIM: 1 },
      n_significant: 0,
      n_significant_cell_level: null,
      groups: RESULT.groups.map((g) => ({ ...g, stat: null, pval: null, fdr: null, df_prior: null })),
      warnings: ["CTRL: 1, STIM: 1 sample(s) of 'orig_ident': a test needs at least 2 per condition."],
    });
    render(<CompositionTest datasetId="kang" info={KANG_INFO} />);
    expect(await screen.findByText("Proportions by cluster (not tested)")).toBeInTheDocument();
    expect(screen.getByRole("alert").textContent).toContain("at least 2 per condition");
    expect(screen.getByText(/read these differences as descriptive/)).toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "FDR" })).toBeNull();
  });

  it("sends the design and variance choices, and shows the auto design", async () => {
    render(<CompositionTest datasetId="kang" info={KANG_INFO} />);
    await screen.findByText("1 of 3 groups shift (FDR < 0.05)");
    expect(screen.getByLabelText("Design")).toHaveDisplayValue("Auto (paired)");
    fireEvent.change(screen.getByLabelText("Design"), { target: { value: "unpaired" } });
    await waitFor(() =>
      expect(testComposition).toHaveBeenLastCalledWith("kang", expect.objectContaining({ paired: false })),
    );
    fireEvent.change(screen.getByLabelText("Variance moderation"), { target: { value: "plain" } });
    await waitFor(() =>
      expect(testComposition).toHaveBeenLastCalledWith(
        "kang", expect.objectContaining({ paired: false, robust: false })),
    );
    // A new sample column: pairing back to auto.
    fireEvent.change(screen.getByLabelText("Sample column"), { target: { value: "Donor" } });
    await waitFor(() => expect(screen.getByLabelText("Design")).toHaveValue("auto"));
    const last = testComposition.mock.calls.at(-1)![1];
    expect(last).not.toHaveProperty("paired");
  });

  it("exports the table as CSV", async () => {
    render(<CompositionTest datasetId="kang" info={KANG_INFO} />);
    await screen.findByText("1 of 3 groups shift (FDR < 0.05)");
    fireEvent.click(screen.getByRole("button", { name: /CSV/ }));
    expect(downloadCsv).toHaveBeenCalledTimes(1);
    const [name, headers, rows] = downloadCsv.mock.calls[0]!;
    expect(name).toBe("composition_test_cluster_by_stim_kang.csv");
    expect(headers[0]).toBe("cluster");
    expect(rows[0][0]).toBe("CD14 Mono");
  });

  it("fetches the columns when it is not handed them (a joint dataset)", async () => {
    getDataset.mockResolvedValue({ ...KANG_INFO, id: "joint" });
    render(<CompositionTest datasetId="joint" defaultCondition="orig_ident" defaultGroup="cluster" />);
    await screen.findByText("1 of 3 groups shift (FDR < 0.05)");
    expect(getDataset).toHaveBeenCalledWith("joint");
    expect(testComposition).toHaveBeenCalledWith("joint", expect.objectContaining({ condition_key: "orig_ident" }));
  });

  it("reports a failed request", async () => {
    testComposition.mockRejectedValue(new Error("'stim' has fewer than two groups"));
    render(<CompositionTest datasetId="kang" info={KANG_INFO} />);
    expect(await screen.findByText("'stim' has fewer than two groups")).toBeInTheDocument();
  });
});
