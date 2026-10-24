import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { ClusterQc, ClusterQcReport } from "@/api/qc";

const { getClusterQc } = vi.hoisted(() => ({ getClusterQc: vi.fn() }));
vi.mock("@/api/qc", () => ({ getClusterQc }));

import { ClusterQcTable } from "./ClusterQcTable";

function row(group: string, extra: Partial<ClusterQc> = {}): ClusterQc {
  return {
    group, n_cells: 150, fraction: 0.33, median_genes: 1400, median_counts: 4200,
    median_pct_mt: 2.1, median_pct_ribo: 20, doublet_fraction: null, median_doublet_score: null,
    flags: [], reasons: [], ...extra,
  };
}

const HELP = "Ambient-looking marker: one of the top genes of the soup.";

function report(clusters: ClusterQc[], extra: Partial<ClusterQcReport> = {}): ClusterQcReport {
  return {
    column: "leiden", n_cells: 450, n_groups: clusters.length, metrics: ["genes", "counts", "pct_mt"],
    typical: { median_genes: 1400, median_counts: 4200, median_pct_mt: 2.1 }, clusters, flag_help: {},
    computed_on_demand: false, note: "", ambient_markers_help: HELP, ...extra,
  };
}

describe("ClusterQcTable ambient markers", () => {
  beforeEach(() => {
    getClusterQc.mockReset();
  });

  it("lists a cluster's ambient-looking markers as a note chip, not a flag", async () => {
    getClusterQc.mockResolvedValue(report([row("0", { ambient_markers: ["LYZ", "S100A9"] }), row("1"), row("2")]));
    render(<ClusterQcTable datasetId="d1" />);
    await screen.findByText("No cluster stands out (3 clusters)");
    fireEvent.click(screen.getByRole("button", { name: /Cluster QC/ }));
    const chip = screen.getByTestId("cluster-ambient-markers");
    expect(chip).toHaveTextContent("Ambient markers: 2");
    expect(chip.getAttribute("title")).toBe(`Ambient-looking markers: LYZ, S100A9. ${HELP}`);
    expect(screen.getAllByTestId("cluster-ambient-markers")).toHaveLength(1);
  });

  it("says when they come from the soup share alone", async () => {
    getClusterQc.mockResolvedValue(
      report([row("0", { ambient_markers: ["LYZ"] }), row("1"), row("2")], { ambient_markers_approximate: true }),
    );
    render(<ClusterQcTable datasetId="d1" />);
    await screen.findByText("No cluster stands out (3 clusters)");
    fireEvent.click(screen.getByRole("button", { name: /Cluster QC/ }));
    expect(screen.getByTestId("cluster-ambient-markers").getAttribute("title")).toMatch(/^Possibly ambient markers: LYZ/);
  });
});
