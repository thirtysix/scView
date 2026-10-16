import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { ClusterQc, ClusterQcReport } from "@/api/qc";

const { getClusterQc } = vi.hoisted(() => ({ getClusterQc: vi.fn() }));
vi.mock("@/api/qc", () => ({ getClusterQc }));

import { ClusterQcTable, askQuestion } from "./ClusterQcTable";
import { useViewStore } from "@/stores/viewStore";
import { useSelectionStore } from "@/stores/selectionStore";

function row(group: string, extra: Partial<ClusterQc> = {}): ClusterQc {
  return {
    group, n_cells: 150, fraction: 0.16, median_genes: 1400, median_counts: 4200,
    median_pct_mt: 2.1, median_pct_ribo: 20, doublet_fraction: 0.02,
    median_doublet_score: null, flags: [], reasons: [], ...extra,
  };
}

function report(clusters: ClusterQc[]): ClusterQcReport {
  return {
    column: "leiden", n_cells: 910, n_groups: clusters.length,
    metrics: ["genes", "counts", "pct_mt", "pct_ribo", "doublet_fraction"],
    typical: { median_genes: 1400, median_counts: 4200, median_pct_mt: 2.1, median_pct_ribo: 20, doublet_fraction: 0.02 },
    clusters, flag_help: { low_quality: "Damaged or dying cells." },
    computed_on_demand: false, note: "",
  };
}

const BAD = row("5", {
  median_genes: 400, median_pct_mt: 31,
  flags: ["low_quality"],
  reasons: ["Likely low quality: median 400 genes per cell vs 1,400 in a typical cluster, and 31.0% mitochondrial reads vs 2.1%."],
});

describe("ClusterQcTable", () => {
  beforeEach(() => {
    getClusterQc.mockReset();
    useViewStore.setState({ activePanel: "assessment", pendingAsk: null });
    useSelectionStore.setState({ highlightedGroup: null });
  });

  it("opens on a flagged cluster and explains the flag in plain words", async () => {
    getClusterQc.mockResolvedValue(report([row("0"), row("1"), BAD]));
    render(<ClusterQcTable datasetId="d1" />);
    expect(await screen.findByText("1 of 3 clusters stand out")).toBeInTheDocument();
    expect(screen.getByText("Low quality")).toHaveAttribute("title", BAD.reasons[0]);
    expect(screen.getByText(/Damaged or dying cells/)).toBeInTheDocument();
    expect(screen.getByText("31.0%")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Ask the co-pilot about cluster 5" }));
    expect(useViewStore.getState().pendingAsk).toBe(askQuestion("leiden", BAD));

    fireEvent.click(screen.getByRole("button", { name: "5" }));
    expect(useSelectionStore.getState().highlightedGroup).toEqual({ column: "leiden", value: "5" });
    expect(useViewStore.getState().activePanel).toBe("unified");
  });

  it("stays collapsed to one line when nothing stands out", async () => {
    getClusterQc.mockResolvedValue(report([row("0"), row("1"), row("2")]));
    render(<ClusterQcTable datasetId="d1" />);
    expect(await screen.findByText("No cluster stands out (3 clusters)")).toBeInTheDocument();
    expect(screen.queryByRole("table")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /cluster qc/i }));
    expect(screen.getByRole("table")).toBeInTheDocument();
  });

  it("explains a dataset with no clustering", async () => {
    getClusterQc.mockResolvedValue({
      ...report([]), column: null, n_groups: 0, note: "No clustering yet: run clustering.",
    });
    render(<ClusterQcTable datasetId="d1" />);
    await waitFor(() => expect(screen.getByText(/No clustering yet/)).toBeInTheDocument());
  });
});
