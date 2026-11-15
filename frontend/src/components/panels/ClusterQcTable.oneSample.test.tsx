import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { ClusterQc, ClusterQcReport } from "@/api/qc";

const { getClusterQc } = vi.hoisted(() => ({ getClusterQc: vi.fn() }));
vi.mock("@/api/qc", () => ({ getClusterQc }));

import { ClusterQcTable } from "./ClusterQcTable";

function row(group: string): ClusterQc {
  return {
    group, n_cells: 150, fraction: 0.16, median_genes: 1400, median_counts: 4200,
    median_pct_mt: 2.1, median_pct_ribo: 20, doublet_fraction: null, median_doublet_score: null,
    flags: [], reasons: [],
  };
}

const TEXT =
  "2 leiden clusters are almost all one sample (of 4 in all): 3 (40 cells, 100% from s1, which is 26% of the data); " +
  "5 (31 cells, 94% from s2, which is 24% of the data). A cluster made of one sample is often a batch effect, not a " +
  "cell type: check with crosstab(leiden x sample), or correct for sample (Harmony) before clustering.";

function report(extra: Partial<ClusterQcReport> = {}): ClusterQcReport {
  return {
    column: "leiden", n_cells: 450, n_groups: 3, metrics: ["genes", "counts", "pct_mt"],
    typical: { median_genes: 1400, median_counts: 4200, median_pct_mt: 2.1 },
    clusters: [row("0"), row("3"), row("5")], flag_help: {}, computed_on_demand: false, note: "", ...extra,
  };
}

describe("ClusterQcTable one-sample note", () => {
  beforeEach(() => getClusterQc.mockReset());

  it("counts the clusters in the header and shows the note, without flagging a cluster", async () => {
    getClusterQc.mockResolvedValue(report({
      one_sample: {
        sample_column: "sample", n_samples: 4, text: TEXT,
        clusters: [
          { cluster: "3", n_cells: 40, sample: "s1", share: 1, sample_share_of_data: 0.26 },
          { cluster: "5", n_cells: 31, sample: "s2", share: 0.94, sample_share_of_data: 0.24 },
        ],
      },
    }));
    render(<ClusterQcTable datasetId="d1" />);
    expect(await screen.findByText(/No cluster stands out \(3 clusters\) · 2 clusters from one sample/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Cluster QC/ }));
    expect(screen.getByTestId("one-sample-note")).toHaveTextContent(TEXT);
  });

  it("shows nothing when no cluster is one sample", async () => {
    getClusterQc.mockResolvedValue(report({ one_sample: null }));
    render(<ClusterQcTable datasetId="d1" />);
    await screen.findByText("No cluster stands out (3 clusters)");
    fireEvent.click(screen.getByRole("button", { name: /Cluster QC/ }));
    expect(screen.queryByTestId("one-sample-note")).not.toBeInTheDocument();
  });
});
