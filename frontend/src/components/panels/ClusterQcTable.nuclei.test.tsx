import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import type { ClusterQc, ClusterQcReport } from "@/api/qc";

const { getClusterQc } = vi.hoisted(() => ({ getClusterQc: vi.fn() }));
vi.mock("@/api/qc", () => ({ getClusterQc }));

import { ClusterQcTable } from "./ClusterQcTable";

function row(group: string, extra: Partial<ClusterQc> = {}): ClusterQc {
  return {
    group, n_cells: 900, fraction: 0.2, median_genes: 2400, median_counts: 4800,
    median_pct_mt: 0, median_pct_ribo: null, doublet_fraction: null, median_doublet_score: null,
    mean_pct_mt: 0.008, mt_detected: 0.04, flags: [], reasons: [], ...extra,
  };
}

const MITO = "Mitochondrial RNA in nuclei: 100% of its cells have mito reads vs 3% in a typical cluster.";
const LOW = "Low RNA for its identity: 13 of its top 25 markers are 1's.";

function report(clusters: ClusterQc[]): ClusterQcReport {
  return {
    column: "leiden", n_cells: 4000, n_groups: clusters.length,
    metrics: ["genes", "counts", "pct_mt", "mt_detected"],
    typical: { median_genes: 2400, median_counts: 4800, median_pct_mt: 0.1, mt_detected: 0.035, mean_pct_mt: 0.008 },
    clusters, flag_help: { nuclei_mito: "Mitochondrial RNA in far more of its cells.", low_content: "Shares the identity." },
    computed_on_demand: false, note: "", nuclei_like_mito: true,
  };
}

describe("ClusterQcTable in nuclei", () => {
  beforeEach(() => getClusterQc.mockReset());

  it("shows mito with the digits it needs, the share of cells with mito reads and the new flags", async () => {
    getClusterQc.mockResolvedValue(report([
      row("0", { median_pct_mt: 0.228 }), row("1"),
      row("10", { median_pct_mt: 0.814, mean_pct_mt: 1.105, mt_detected: 0.996, flags: ["nuclei_mito"], reasons: [MITO] }),
      row("12", { median_counts: 1718, flags: ["low_content", "sex_markers"], reasons: [LOW, "Sex genes lead."] }),
    ]));
    render(<ClusterQcTable datasetId="d1" />);
    expect(await screen.findByText("2 of 4 clusters stand out")).toBeInTheDocument();
    expect(screen.getByText("Mito cells")).toBeInTheDocument();
    expect(screen.getByText("0.23%")).toBeInTheDocument(); // not 0.2%
    expect(screen.getByText("0.10%")).toBeInTheDocument(); // the typical cluster
    expect(screen.getByText("100%")).toHaveAttribute("title", "Mean 1.1% of its reads");
    expect(screen.getByText("Mito in nuclei")).toHaveAttribute("title", MITO);
    expect(screen.getByText("Low RNA")).toHaveAttribute("title", LOW);
    expect(screen.getByText("Sex genes lead")).toBeInTheDocument();
  });

  it("leaves the share column out in cell data", async () => {
    const r = report([row("0", { median_pct_mt: 5.1 }), row("1", { median_pct_mt: 6.2 })]);
    r.metrics = ["genes", "counts", "pct_mt"];
    getClusterQc.mockResolvedValue(r);
    render(<ClusterQcTable datasetId="d1" />);
    await screen.findByText("No cluster stands out (2 clusters)");
    expect(screen.queryByText("Mito cells")).not.toBeInTheDocument();
  });
});
