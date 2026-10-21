import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { ClusterQc, ClusterQcReport } from "@/api/qc";

const { getClusterQc } = vi.hoisted(() => ({ getClusterQc: vi.fn() }));
vi.mock("@/api/qc", () => ({ getClusterQc }));

import { ClusterQcTable } from "./ClusterQcTable";

function row(group: string, extra: Partial<ClusterQc> = {}): ClusterQc {
  return {
    group, n_cells: 150, fraction: 0.16, median_genes: 1400, median_counts: 4200,
    median_pct_mt: 2.1, median_pct_ribo: 20, doublet_fraction: null, median_doublet_score: null,
    median_pct_malat1: 5.2, flags: [], reasons: [], ...extra,
  };
}

const REASON =
  "Low MALAT1: median 0.12% of its reads vs 4.0% in the median cell (flagged below a third of it, 1.3%). " +
  "MALAT1 is kept in the nucleus, so these may be cell fragments or cytoplasm without its nucleus.";

function report(clusters: ClusterQc[], extra: Partial<ClusterQcReport> = {}): ClusterQcReport {
  return {
    column: "leiden", n_cells: 450, n_groups: clusters.length,
    metrics: ["genes", "counts", "pct_mt", "pct_malat1"],
    typical: { median_genes: 1400, median_counts: 4200, median_pct_mt: 2.1, median_pct_malat1: 5.2 },
    clusters, flag_help: { low_malat1: "Little MALAT1 RNA: below a third of the median cell's." },
    computed_on_demand: false, note: "", dataset_median_pct_malat1: 4.01, ...extra,
  };
}

describe("ClusterQcTable MALAT1", () => {
  beforeEach(() => getClusterQc.mockReset());

  it("shows the MALAT1 column, the median cell's share and the flag with its reason", async () => {
    getClusterQc.mockResolvedValue(report([
      row("0"), row("1"), row("8", { median_pct_malat1: 0.117, flags: ["low_malat1"], reasons: [REASON] }),
    ]));
    render(<ClusterQcTable datasetId="d1" />);
    expect(await screen.findByText("1 of 3 clusters stand out")).toBeInTheDocument();
    expect(screen.getByText("MALAT1").getAttribute("title")).toMatch(/nuclear-retained.*doi:10\.1186/);
    expect(screen.getByText("Low MALAT1")).toHaveAttribute("title", REASON);
    expect(screen.getByText("0.12%")).toBeInTheDocument();
    expect(screen.getByText("4.0%")).toHaveAttribute("title", expect.stringMatching(/median cell/));
    expect(screen.getByText(/below a third of the median cell's/)).toBeInTheDocument();
    expect(screen.queryByText("Intronic")).not.toBeInTheDocument();
  });

  it("shows the author's intronic share when obs has one", async () => {
    getClusterQc.mockResolvedValue(report([row("0", { median_pct_intronic: 31.5 }), row("1", { median_pct_intronic: 12 })], {
      metrics: ["genes", "counts", "pct_mt", "pct_malat1", "pct_intronic"], intronic_column: "fraction_unspliced",
    }));
    render(<ClusterQcTable datasetId="d1" />);
    await screen.findByText("No cluster stands out (2 clusters)");
    fireEvent.click(screen.getByRole("button", { name: /Cluster QC/ })); // collapsed: nothing flagged
    expect(screen.getByText("Intronic").getAttribute("title")).toMatch(/'fraction_unspliced'.*not computed/);
    expect(screen.getByText("31.5%")).toBeInTheDocument();
  });
});
