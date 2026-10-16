import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { FilterPreview, FilterThresholds } from "@/api/qc";

const { getFilterPreview } = vi.hoisted(() => ({ getFilterPreview: vi.fn() }));
vi.mock("@/api/qc", () => ({ getFilterPreview }));

import { QcFilterSummary } from "./QcFilterSummary";

const T: FilterThresholds = { min_genes: 200, max_genes: 0, min_counts: 0, max_pct_mt: 20, drop_doublets: false };

function preview(nRemoved: number): FilterPreview {
  return {
    n_cells: 1000,
    n_removed: nRemoved,
    n_kept: 1000 - nRemoved,
    fraction_removed: nRemoved / 1000,
    thresholds: { ...T, max_pct_mt: 20 },
    has_mt: true,
    doublet_calls: true,
    by_criterion: {
      min_genes: { label: "Min genes", n_removed: nRemoved - 10 },
      max_pct_mt: { label: "Max % mito", n_removed: 10 },
    },
    samples: {
      column: "Donor",
      reason: "donor",
      groups: [
        { name: "d1", n_cells: 500, n_removed: 20, fraction: 0.04, by_criterion: {} },
        { name: "d2", n_cells: 500, n_removed: nRemoved - 20, fraction: (nRemoved - 20) / 500, by_criterion: {} },
      ],
    },
    clusters: {
      column: "leiden",
      reason: "clustering",
      groups: [
        { name: "0", n_cells: 900, n_removed: nRemoved - 100, fraction: (nRemoved - 100) / 900, by_criterion: {} },
        { name: "7", n_cells: 100, n_removed: 100, fraction: 1, by_criterion: {} },
      ],
    },
    warnings: [
      {
        kind: "cluster", column: "leiden", group: "7", n_cells: 100, n_removed: 100, fraction: 1, whole: true,
        criterion: "min_genes", message: "Removes all 100 cells of cluster 7, mostly by Min genes.",
      },
    ],
  };
}

describe("QcFilterSummary", () => {
  beforeEach(() => {
    getFilterPreview.mockReset().mockImplementation((_id: string, t: FilterThresholds) =>
      Promise.resolve(preview(t.min_genes > 200 ? 300 : 150)),
    );
  });

  it("says what the thresholds remove overall, per threshold, per sample and cluster", async () => {
    render(<QcFilterSummary datasetId="d1" thresholds={T} />);
    expect(await screen.findByText(/150 of 1,000 cells \(15.0%\)/)).toBeInTheDocument();
    expect(getFilterPreview).toHaveBeenCalledWith("d1", T);
    expect(screen.getByText(/min genes 140, max % mito 10/)).toBeInTheDocument();
    const warning = screen.getByRole("list", { name: "Cell-loss warnings" });
    expect(warning).toHaveTextContent("Removes all 100 cells of cluster 7");
    expect(warning.querySelector("li")!.className).toContain("bg-red-50");
    // Most affected first: cluster 7 (100 %) above cluster 0.
    const clusterRows = screen.getAllByRole("row").map((r) => r.textContent ?? "");
    expect(clusterRows.findIndex((t) => t.startsWith("7"))).toBeLessThan(
      clusterRows.findIndex((t) => t.startsWith("0")),
    );
    expect(screen.getByText(/Cell\/Gene Filtering step's parameters/)).toBeInTheDocument();
  });

  it("asks again when a threshold changes, and only once for a burst of changes", async () => {
    const { rerender } = render(<QcFilterSummary datasetId="d1" thresholds={T} />);
    await screen.findByText(/150 of 1,000/);
    rerender(<QcFilterSummary datasetId="d1" thresholds={{ ...T, min_genes: 300 }} />);
    rerender(<QcFilterSummary datasetId="d1" thresholds={{ ...T, min_genes: 400 }} />);
    expect(await screen.findByText(/300 of 1,000/)).toBeInTheDocument();
    expect(getFilterPreview).toHaveBeenCalledTimes(2);
    expect(getFilterPreview).toHaveBeenLastCalledWith("d1", { ...T, min_genes: 400 });
    // The same values in a new object: no new request.
    rerender(<QcFilterSummary datasetId="d1" thresholds={{ ...T, min_genes: 400 }} />);
    await new Promise((r) => setTimeout(r, 250));
    expect(getFilterPreview).toHaveBeenCalledTimes(2);
  });

  it("lists the first few warnings, the rest on request", async () => {
    const p = preview(150);
    const w = p.warnings[0]!;
    p.warnings = Array.from({ length: 6 }, (_, i) => ({ ...w, group: `g${i}`, message: `warning ${i}` }));
    getFilterPreview.mockResolvedValue(p);
    render(<QcFilterSummary datasetId="d1" thresholds={T} />);
    const list = await screen.findByRole("list", { name: "Cell-loss warnings" });
    expect(list.querySelectorAll("li")).toHaveLength(4);
    fireEvent.click(screen.getByRole("button", { name: "Show all 6 warnings" }));
    expect(list.querySelectorAll("li")).toHaveLength(6);
  });

  it("offers dropping the predicted doublets when there are calls", async () => {
    const onDropDoublets = vi.fn();
    render(<QcFilterSummary datasetId="d1" thresholds={T} onDropDoublets={onDropDoublets} />);
    fireEvent.click(await screen.findByLabelText("Also drop predicted doublets"));
    expect(onDropDoublets).toHaveBeenCalledWith(true);
  });

  it("shows the server's reason when the preview fails", async () => {
    getFilterPreview.mockRejectedValue(new Error("Column 'x' not found."));
    render(<QcFilterSummary datasetId="d1" thresholds={T} />);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Column 'x' not found."));
  });
});
