import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { FilterPreview, FilterThresholds } from "@/api/qc";
import type { DatasetInfo } from "@/api/types";

const { getFilterPreview } = vi.hoisted(() => ({ getFilterPreview: vi.fn() }));
vi.mock("@/api/qc", () => ({ getFilterPreview }));

import { QcFilterSummary } from "./QcFilterSummary";
import { useDatasetStore } from "@/stores/datasetStore";

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
    const onChange = vi.fn();
    render(<QcFilterSummary datasetId="d1" thresholds={T} onChange={onChange} />);
    fireEvent.click(await screen.findByLabelText("Also drop predicted doublets"));
    expect(onChange).toHaveBeenCalledWith({ drop_doublets: true });
  });

  it("shows the server's reason when the preview fails", async () => {
    getFilterPreview.mockRejectedValue(new Error("Column 'x' not found."));
    render(<QcFilterSummary datasetId="d1" thresholds={T} />);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Column 'x' not found."));
  });
});

describe("QcFilterSummary in MAD mode", () => {
  const M: FilterThresholds = { ...T, qc_threshold_mode: "mad", qc_mad_k: 5, qc_mad_k_mt: 3, qc_sample_column: "" };

  function madPreview(): FilterPreview {
    const p = preview(150);
    p.thresholds = { ...p.thresholds, mode: "mad", mad_k: 5, mad_k_mt: 3 };
    p.by_criterion = {
      max_counts: { label: "Max counts", n_removed: 40 },
      max_genes: { label: "Max genes", n_removed: 120 },
    };
    const bounds = { min_counts: 312.4, max_counts: 9870.2, min_genes: 210.6, max_genes: 3100, max_pct_mt: 6.12 };
    p.mad = {
      column: "Donor", reason: "donor", k: 5, k_mt: 3, ceiling_pct_mt: 20, min_sample_cells: 100,
      samples: [
        { name: "d1", n_cells: 500, pooled: false, ...bounds, skipped: [], n_removed: 20 },
        { name: "d2", n_cells: 60, pooled: true, ...bounds, max_pct_mt: null, skipped: ["pct_mt"], n_removed: 3 },
      ],
      pooled: { n_cells: 1000, ...bounds, skipped: [] },
      n_unlabelled: 4,
      reused_from: null,
    };
    return p;
  }

  beforeEach(() => {
    getFilterPreview.mockReset().mockResolvedValue(madPreview());
    useDatasetStore.setState({
      currentDataset: {
        id: "d1",
        obs_columns: [
          { name: "Donor", dtype: "category", n_unique: 2 },
          { name: "lane", dtype: "category", n_unique: 4 },
        ],
      } as unknown as DatasetInfo,
    });
  });

  it("lists each sample's thresholds and loss, and offers the mode, k and samples", async () => {
    const onChange = vi.fn();
    render(<QcFilterSummary datasetId="d1" thresholds={M} onChange={onChange} />);
    const table = await screen.findByRole("table", { name: "Thresholds per sample" });
    const rows = within(table).getAllByRole("row").map((r) => r.textContent ?? "");
    expect(rows[1]).toContain("312–9,870");
    expect(rows[1]).toContain("211–3,100");
    expect(rows[1]).toContain("< 6.1");
    expect(rows[1]).toContain("20 (4.0%)");
    expect(rows[2]).toContain("d2 (pooled)");
    expect(rows[2]).toContain("any"); // its mito MAD was 0: no bound from it
    expect(screen.getByText(/4 cells without a Donor value/)).toBeInTheDocument();
    // In MAD mode the per-sample loss sits in the thresholds table, not twice.
    expect(screen.queryByText("Per sample")).not.toBeInTheDocument();
    expect(screen.getByText(/fixed genes and counts cut-offs are not used/)).toBeInTheDocument();

    expect(screen.getByRole("button", { name: "Per-sample MAD" })).toHaveAttribute("aria-pressed", "true");
    fireEvent.change(screen.getByLabelText("MADs for counts and genes"), { target: { value: "4" } });
    expect(onChange).toHaveBeenLastCalledWith({ qc_mad_k: 4 });
    fireEvent.change(screen.getByLabelText("MADs for % mito"), { target: { value: "0" } });
    expect(onChange).toHaveBeenCalledTimes(1); // not a positive number: ignored
    expect(screen.getByRole("option", { name: "Detected: Donor" })).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Sample column for MAD thresholds"), { target: { value: "lane" } });
    expect(onChange).toHaveBeenLastCalledWith({ qc_sample_column: "lane" });
    fireEvent.click(screen.getByRole("button", { name: "Fixed cut-offs" }));
    expect(onChange).toHaveBeenLastCalledWith({ qc_threshold_mode: "fixed" });
  });

  it("says when the dataset was filtered before and its thresholds are used again", async () => {
    const p = madPreview();
    p.previous = { mode: "mad", timestamp: "2026-10-17T19:29:10+00:00" };
    p.mad!.reused_from = "2026-10-17T19:29:10+00:00";
    getFilterPreview.mockResolvedValue(p);
    render(<QcFilterSummary datasetId="d1" thresholds={M} />);
    expect(await screen.findByTestId("qc-previous")).toHaveTextContent(
      /Filtered before by scView on 2026-10-17 \(per-sample MAD\)\. With the same settings/,
    );
  });

  it("gives the pooled bounds when there is no sample column", async () => {
    const p = madPreview();
    p.mad = { ...p.mad!, column: null, reason: null, samples: [], n_unlabelled: 0 };
    p.samples = null;
    getFilterPreview.mockResolvedValue(p);
    render(<QcFilterSummary datasetId="d1" thresholds={M} />);
    expect(await screen.findByTestId("mad-pooled")).toHaveTextContent(
      "all cells are one group: kept with 312–9,870 counts and 211–3,100 genes per cell, under 6.1% mito",
    );
  });

  it("keeps the mode switch when the preview fails, so fixed cut-offs stay reachable", async () => {
    getFilterPreview.mockRejectedValue(new Error("'cell' has 900 groups; per-sample thresholds take at most 200."));
    const onChange = vi.fn();
    render(<QcFilterSummary datasetId="d1" thresholds={M} onChange={onChange} />);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("at most 200"));
    fireEvent.click(screen.getByRole("button", { name: "Fixed cut-offs" }));
    expect(onChange).toHaveBeenCalledWith({ qc_threshold_mode: "fixed" });
  });
});
