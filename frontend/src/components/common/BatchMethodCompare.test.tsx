import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { CompareHandlers, CompareInfo, CompareResult, CompareRow } from "@/api/batchCompare";

const getCompareInfo = vi.fn();
const runCompare = vi.fn();
vi.mock("@/api/batchCompare", async (orig) => ({
  ...(await orig<typeof import("@/api/batchCompare")>()),
  getCompareInfo: (...a: unknown[]) => getCompareInfo(...a),
  runCompare: (...a: unknown[]) => runCompare(...a),
}));

import { BatchMethodCompare } from "./BatchMethodCompare";

const INFO: CompareInfo = {
  available: true,
  reason: null,
  methods: ["none", "harmony", "scanorama", "bbknn"],
  background: "In a benchmark of 16 integration methods on 13 tasks (Luecken et al. 2022), ...",
  background_tip: "Luecken et al., Nat Methods 19:41 (2022) ...",
  n_cells: 1000,
  running: false,
  label_column: "cell_type",
  label_kind: "shared",
  estimate_seconds: 12.3,
};

const row = (method: CompareRow["method"], label: string, ratio: number, cons: number, over = false): CompareRow => ({
  method, label, seconds: 1.5, mixing: 0.4, ilisi: 1.5, mixing_ratio: ratio, conservation: cons,
  conservation_change: 0, overcorrection: over,
});

const RESULT: CompareResult = {
  rows: [row("none", "None", 0.1, 0.9), row("harmony", "Harmony", 0.7, 0.88), row("bbknn", "BBKNN", 0.6, 0.7, true)],
  recommended: "harmony",
  reason: "Harmony: among the methods that mix the batches acceptably ...",
  warnings: ["BBKNN: cells' neighbours share their label 70% of the time, down from 90% ..."],
  notes: ["BBKNN takes the same number of neighbours from every batch ..."],
  label_column: "cell_type",
  label_kind: "shared",
  k: 15,
  ideal: 0.5,
  ilisi_ideal: 2,
  background: INFO.background,
  background_tip: INFO.background_tip,
};

describe("BatchMethodCompare", () => {
  beforeEach(() => {
    getCompareInfo.mockReset().mockResolvedValue(INFO);
    runCompare.mockReset().mockImplementation(async (_id: string, _body: unknown, h: CompareHandlers) => {
      h.onStart?.({ methods: ["none", "harmony", "bbknn"], n_cells: 1000, n_scored: 1000, k: 15,
        batch_key: "batch", n_batches: 2, label_column: "cell_type", label_kind: "shared", ideal: 0.5, ilisi_ideal: 2 });
      RESULT.rows.forEach((r) => h.onMethod?.(r));
      h.onResult?.(RESULT);
    });
  });

  it("asks for a batch column first", () => {
    render(<BatchMethodCompare datasetId="d" batchKey={null} />);
    expect(screen.getByText(/Pick a batch column/)).toBeTruthy();
    expect(getCompareInfo).not.toHaveBeenCalled();
  });

  it("shows the background, runs, and applies the recommendation", async () => {
    const onApply = vi.fn();
    render(<BatchMethodCompare datasetId="d" batchKey="batch" onApply={onApply} applyLabel="Set the step to" />);
    await screen.findByText(/Luecken et al. 2022/);
    expect(getCompareInfo).toHaveBeenCalledWith("d", "batch", undefined);
    fireEvent.click(screen.getByRole("button", { name: /Compare/ }));
    await waitFor(() => expect(screen.getByTestId("batch-compare-table")).toBeTruthy());
    expect(runCompare.mock.calls[0]![1]).toEqual({ batch_key: "batch", label_column: null });
    expect(screen.getByText("recommended")).toBeTruthy();
    expect(screen.getAllByText("70%")).toHaveLength(2); // Harmony mixing, BBKNN kept apart
    expect(screen.getByTestId("batch-compare-reason").textContent).toContain("Harmony");
    expect(screen.getByText(/down from 90%/)).toBeTruthy();
    expect(screen.getByText(/1,000 cells scored/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Set the step to: Harmony/ }));
    expect(onApply).toHaveBeenCalledWith("harmony");
  });

  it("says why it cannot run", async () => {
    getCompareInfo.mockResolvedValue({ ...INFO, available: false, reason: "Run PCA first: ..." });
    render(<BatchMethodCompare datasetId="d" batchKey="batch" />);
    await screen.findByText(/Run PCA first/);
    expect(screen.queryByRole("button", { name: /Compare/ })).toBeNull();
  });

  it("shows a request error", async () => {
    getCompareInfo.mockRejectedValue(new Error("Comparing batch-correction methods is not available for large datasets yet"));
    render(<BatchMethodCompare datasetId="d" batchKey="batch" />);
    await screen.findByText(/not available for large datasets/);
  });
});
