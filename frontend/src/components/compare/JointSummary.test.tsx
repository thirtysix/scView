import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { BatchMixing, JointSummary as Summary } from "@/api/integration";

const { getJointSummary, apiFetch, getDataset, testComposition } = vi.hoisted(() => ({
  getJointSummary: vi.fn(),
  apiFetch: vi.fn(),
  getDataset: vi.fn(),
  testComposition: vi.fn(),
}));
vi.mock("@/api/integration", () => ({ getJointSummary }));
vi.mock("@/api/client", () => ({ apiFetch }));
vi.mock("@/api/datasets", () => ({ getDataset }));
vi.mock("@/api/composition", () => ({ testComposition }));
vi.mock("@/components/plots/Plot", () => ({ default: () => null }));

import { JointSummary, compositionRows, shareOfIdeal, type Crosstab } from "./JointSummary";

// Kang CTRL vs STIM, as scored by hand before this existed.
const KANG: BatchMixing = {
  k: 30,
  label_column: "dataset",
  n_cells: 13836,
  n_scored: 10000,
  ideal: 0.499,
  uncorrected: 0.107,
  corrected: 0.478,
  datasets: [
    { label: "CTRL", n_cells: 6960, ideal: 0.497, uncorrected: 0.1, corrected: 0.47 },
    { label: "STIM", n_cells: 6876, ideal: 0.503, uncorrected: 0.114, corrected: 0.486 },
  ],
};

function joint(mixing: BatchMixing | null): Summary {
  return { dataset_id: "j", joint: true, label_column: "dataset", cluster_column: "leiden", mixing };
}

describe("shareOfIdeal", () => {
  it("scales a score to the ideal and clamps it", () => {
    expect(shareOfIdeal(0.25, 0.5)).toBe(0.5);
    expect(shareOfIdeal(0.6, 0.5)).toBe(1);
    expect(shareOfIdeal(null, 0.5)).toBe(0);
    expect(shareOfIdeal(0.3, 0)).toBe(0);
  });
});

// Cluster 0 shared evenly, cluster 1 only in CTRL, cluster 2 empty after filtering.
const CROSSTAB: Crosstab = {
  row_values: ["0", "1", "2"],
  col_values: ["CTRL", "STIM"],
  counts: [
    [50, 50],
    [30, 0],
    [0, 0],
  ],
};

describe("compositionRows", () => {
  it("gives each dataset's share per cluster and over all cells", () => {
    const { all, rows } = compositionRows(CROSSTAB);
    expect(rows.map((r) => r.group)).toEqual(["0", "1"]);
    expect(rows[0]!.shares).toEqual([0.5, 0.5]);
    expect(rows[1]!.shares).toEqual([1, 0]);
    expect(all.n).toBe(130);
    expect(all.counts).toEqual([80, 50]);
    expect(all.shares[0]).toBeCloseTo(80 / 130);
  });
});

describe("JointSummary", () => {
  beforeEach(() => {
    getJointSummary.mockReset();
    apiFetch.mockReset().mockResolvedValue(CROSSTAB);
  });

  it("renders nothing for a dataset that isn't joint", async () => {
    getJointSummary.mockResolvedValue({ dataset_id: "x", joint: false });
    const { container } = render(<JointSummary datasetId="x" title="Checks" />);
    await act(async () => {}); // let the request settle
    expect(getJointSummary).toHaveBeenCalledWith("x");
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the score before and after correction against the ideal", async () => {
    getJointSummary.mockResolvedValue(joint(KANG));
    render(<JointSummary datasetId="j" title="Checks" />);
    expect(await screen.findByText("Checks")).toBeInTheDocument();
    expect(screen.getByText("Before correction")).toBeInTheDocument();
    expect(screen.getByText("0.107")).toBeInTheDocument();
    expect(screen.getByText("0.478")).toBeInTheDocument();
    expect(screen.getByText("0.499")).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "STIM" })).toBeInTheDocument();
    expect(screen.getByText(/10,000 of 13,836 cells/)).toBeInTheDocument();
  });

  it("breaks the clusters down by source dataset with the crosstab endpoint", async () => {
    getJointSummary.mockResolvedValue(joint(KANG));
    render(<JointSummary datasetId="j" title="Checks" />);
    expect(await screen.findByText(/Datasets per cluster/)).toBeInTheDocument();
    expect(apiFetch).toHaveBeenCalledWith("/datasets/j/metadata/crosstab?row=leiden&col=dataset");
    expect(screen.getByText("All cells")).toBeInTheDocument();
    expect(screen.getByTitle("CTRL: 100.0% (30 cells)")).toBeInTheDocument();
    expect(screen.getByTitle("STIM: 50.0% (50 cells)")).toBeInTheDocument();
  });

  it("says the bars count cells and opens a replicate-aware test of the clusters", async () => {
    getJointSummary.mockResolvedValue(joint(KANG));
    getDataset.mockResolvedValue({
      id: "j", name: "joint", filename: "j.h5ad", n_cells: 130, n_genes: 10, status: "ready",
      available_embeddings: [], created_at: "",
      obs_columns: [
        { name: "dataset", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] },
        { name: "leiden", dtype: "category", n_unique: 2, values: ["0", "1"] },
        { name: "Donor", dtype: "category", n_unique: 8 },
      ],
    });
    testComposition.mockReturnValue(new Promise(() => {}));
    render(<JointSummary datasetId="j" title="Checks" />);
    expect(await screen.findByText(/These bars count cells/)).toBeInTheDocument();
    expect(screen.getByText(/chi-square or Fisher test on the cell counts/)).toBeInTheDocument();
    expect(testComposition).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Test cluster shares across samples" }));
    await waitFor(() =>
      expect(testComposition).toHaveBeenCalledWith("j", {
        condition_key: "dataset", group_key: "leiden", transform: "logit",
      }),
    );
  });

  it("has no 'after' row when nothing was corrected", async () => {
    getJointSummary.mockResolvedValue(joint({ ...KANG, corrected: null }));
    render(<JointSummary datasetId="j" title="Checks" />);
    expect(await screen.findByText("Before correction")).toBeInTheDocument();
    expect(screen.queryByText("After correction")).toBeNull();
  });
});

describe("JointSummary integration metrics", () => {
  beforeEach(() => {
    getJointSummary.mockReset();
    apiFetch.mockReset();
    apiFetch.mockResolvedValue({ row_values: [], col_values: [], counts: [] });
  });

  it("shows iLISI, cell-type conservation and an over-correction warning", async () => {
    getJointSummary.mockResolvedValue(joint({
      ...KANG,
      ilisi: { ideal: 2, uncorrected: 1.2, corrected: 1.8 },
      conservation: {
        label_column: "cluster", kind: "shared", uncorrected: 0.9, corrected: 0.6,
        labels: [{ label: "CD14 Mono", n_scored: 3000, uncorrected: 0.95, corrected: 0.4 }],
      },
      overcorrection: "After correction a cell's nearest neighbours share its 'cluster' label 60% of the time.",
    }));
    await act(async () => {
      render(<JointSummary datasetId="j" title="Checks" />);
    });
    expect(screen.getByText(/iLISI/)).toBeTruthy();
    expect(screen.getByText(/Cell types kept apart/i)).toBeTruthy();
    expect(screen.getByRole("alert").textContent).toContain("60% of the time");
  });
});
