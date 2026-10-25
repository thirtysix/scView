import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import type { SplitGene, SplitGroup, SplitMarkersResult } from "@/api/splitMarkers";

const { getSplitMarkersInfo, runSplitMarkers, keepSplitResult } = vi.hoisted(() => ({
  getSplitMarkersInfo: vi.fn(),
  runSplitMarkers: vi.fn(),
  keepSplitResult: vi.fn(),
}));
vi.mock("@/api/splitMarkers", () => ({ getSplitMarkersInfo, runSplitMarkers, keepSplitResult }));
vi.mock("@/components/plots/Plot", () => ({
  default: (p: { data: { name: string }[] }) => <div data-testid="plot">{p.data.map((t) => t.name).join("|")}</div>,
}));

import { CountSplitCheck } from "./CountSplitCheck";

const gene = (name: string, marker: boolean, kept: boolean): SplitGene => ({
  gene: name, logfoldchange: 2, pval: 1e-10, pval_adj: 1e-8, split_logfoldchange: 1,
  split_pval: kept ? 1e-5 : 0.4, split_pval_adj: kept ? 1e-4 : 0.6, split_pct_in: 0.5, split_pct_out: 0.1,
  marker, kept,
});

const group = (name: string, n_kept: number, n_ordinary: number, over: Partial<SplitGroup> = {}): SplitGroup => ({
  group: name, n_cells: 120, train_cluster: "4", jaccard: 0.81, shared: false, tested: true, reason: null,
  n_ordinary, n_kept, n_lost: n_ordinary - n_kept, n_split: n_kept, n_split_only: 0,
  kept: [], lost: n_kept < n_ordinary ? ["FOS", "JUN"] : [], split_only: [],
  genes: [gene("CD14", true, true), gene("FOS", true, false), gene("ACTB", false, false)], ...over,
});

const RESULT: SplitMarkersResult = {
  reference: "leiden", reference_exists: true, method: "leiden", resolution: 1, train_resolution: 1.4,
  n_groups: 2, n_tested_groups: 2, n_keep_enough: 1, n_ordinary: 60, n_kept: 32,
  train: { n_clusters: 2, ari: 0.74, ami: 0.7 },
  groups: [group("0", 30, 40), group("1", 2, 20, { shared: true })],
  notes: [], epsilon: 0.5, seed: 0,
  thresholds: { padj: 0.05, min_logfc: 0.25, min_pts: 0.2, keeps_enough: 10, genes_per_group: 300, min_jaccard: 0.25 },
  seconds: { total: 30 },
};

function renderCheck(over: Partial<Parameters<typeof CountSplitCheck>[0]> = {}) {
  const props = {
    datasetId: "ds", column: "leiden", group: "1", onGroupChange: vi.fn(), showSplitP: false,
    onShowSplitPChange: vi.fn(), onResult: vi.fn(), ...over,
  };
  return { ...render(<CountSplitCheck {...props} />), props };
}

beforeEach(() => {
  getSplitMarkersInfo.mockReset().mockResolvedValue({
    available: true, reason: null, n_cells: 1000, first_run_seconds: 70, reference: "leiden", n_groups: 2,
    cached: false, estimate_seconds: 45,
  });
  runSplitMarkers.mockReset().mockImplementation(async (_d, _c, h) => {
    h.onStart?.({ cached: false, estimate_seconds: 45 });
    h.onResult(RESULT);
  });
  keepSplitResult.mockReset().mockResolvedValue({ result_id: "r1", label: "1 (count split, held-out half) vs rest" });
});

describe("CountSplitCheck", () => {
  it("runs the check and shows each group's kept and lost markers, the weakest first", async () => {
    const { props } = renderCheck();
    expect(await screen.findByText(/about 45 s/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Check with count splitting" }));
    expect(await screen.findByText(/1 of 2 groups keep at least 10 markers/)).toBeInTheDocument();
    expect(runSplitMarkers.mock.calls[0]![1]).toBe("leiden");
    expect(props.onResult).toHaveBeenLastCalledWith(RESULT);
    const rows = screen.getAllByRole("row").slice(1);
    expect(rows[0]).toHaveTextContent("1120202180.81*"); // group 1: 120 cells, keeps 2 of 20, lost 18; listed first, shared match
    expect(screen.getByText(/keeps 2 of 20 markers/)).toBeInTheDocument();
    expect(screen.getByText(/Lost, e.g. FOS, JUN/)).toBeInTheDocument();
    expect(screen.getByTestId("plot")).toHaveTextContent("Other genes (1)|Marker, lost (1)|Marker, kept (1)");
    expect(screen.getByText(/n = 3 genes/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "0" }));
    expect(props.onGroupChange).toHaveBeenCalledWith("0");
    fireEvent.click(screen.getByRole("checkbox", { name: "Split p-values in the table" }));
    expect(props.onShowSplitPChange).toHaveBeenCalledWith(true);
    fireEvent.click(screen.getByRole("button", { name: "Keep for GSEA / overlap" }));
    expect(await screen.findByText(/Kept with the DE results/)).toBeInTheDocument();
    expect(keepSplitResult).toHaveBeenCalledWith("ds", "leiden", "1");
  });

  it("explains why it cannot run", async () => {
    getSplitMarkersInfo.mockResolvedValue({ available: false, reason: "No raw counts here.", n_cells: 10, first_run_seconds: 70 });
    renderCheck();
    expect(await screen.findByText("No raw counts here.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Check with count splitting" })).toBeDisabled();
  });

  it("shows the server's error", async () => {
    runSplitMarkers.mockRejectedValue(new Error("A count-split check is already running on this dataset."));
    renderCheck();
    await screen.findByText(/about 45 s/);
    fireEvent.click(screen.getByRole("button", { name: "Check with count splitting" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("already running");
  });
});
