import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, render, screen } from "@testing-library/react";
import type { BatchMixing, JointSummary as Summary } from "@/api/integration";

const { getJointSummary } = vi.hoisted(() => ({ getJointSummary: vi.fn() }));
vi.mock("@/api/integration", () => ({ getJointSummary }));

import { JointSummary, shareOfIdeal } from "./JointSummary";

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

describe("JointSummary", () => {
  beforeEach(() => getJointSummary.mockReset());

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
    expect(screen.getByText("STIM")).toBeInTheDocument();
    expect(screen.getByText(/10,000 of 13,836 cells/)).toBeInTheDocument();
  });

  it("has no 'after' row when nothing was corrected", async () => {
    getJointSummary.mockResolvedValue(joint({ ...KANG, corrected: null }));
    render(<JointSummary datasetId="j" title="Checks" />);
    expect(await screen.findByText("Before correction")).toBeInTheDocument();
    expect(screen.queryByText("After correction")).toBeNull();
  });
});
