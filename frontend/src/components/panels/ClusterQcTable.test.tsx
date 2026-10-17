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

  it("shows the stress score column and a stress-driven flag", async () => {
    const STRESSED = row("4", {
      median_stress_score: 1.42,
      stress_markers: ["FOS", "JUN", "HSPA1A"],
      flags: ["stress_driven"],
      reasons: ["Possibly dissociation stress: median stress score 1.42 vs 0.21 in a typical cluster, and 3 of its top 20 markers are stress genes (FOS, JUN, HSPA1A)."],
    });
    getClusterQc.mockResolvedValue({
      ...report([row("0", { median_stress_score: 0.2 }), row("1", { median_stress_score: 0.21 }), STRESSED]),
      metrics: ["genes", "counts", "pct_mt", "pct_ribo", "doublet_fraction", "stress_score"],
      typical: { median_genes: 1400, median_counts: 4200, median_pct_mt: 2.1, median_pct_ribo: 20, doublet_fraction: 0.02, median_stress_score: 0.21 },
      flag_help: { stress_driven: "Possibly driven by dissociation stress." },
    });
    render(<ClusterQcTable datasetId="d1" />);
    expect(await screen.findByText("1 of 3 clusters stand out")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Stress" })).toBeInTheDocument();
    expect(screen.getByText("1.42")).toHaveAttribute("title", "Stress genes among its top markers: FOS, JUN, HSPA1A");
    expect(screen.getByText("Stress-driven")).toHaveAttribute("title", STRESSED.reasons[0]);
    expect(screen.getByText(/Possibly driven by dissociation stress/)).toBeInTheDocument();
    expect(askQuestion("leiden", STRESSED)).toContain("flagged as stress-driven");
  });

  it("tells an erythroid cluster from ambient haemoglobin and shows cycling", async () => {
    const ERY = row("9", {
      mean_pct_hb: 48.18, erythroid: true, erythroid_markers: ["HBB", "HBA2", "HBA1"], cycling_fraction: 0.02,
      notes: ["Erythroid: its top markers include red-cell genes (HBB, HBA2, HBA1), and haemoglobin is 48.2% of its reads vs 0.01% in a typical non-erythroid cluster. Real red-cell expression, not ambient contamination."],
    });
    const AMB = row("4", {
      mean_pct_hb: 3.4, cycling_fraction: 0.05, flags: ["hb_ambient"],
      reasons: ["Haemoglobin RNA without erythroid markers: 3.4% of its reads are haemoglobin vs 0.01% in a typical non-erythroid cluster."],
    });
    const CYC = row("3", {
      mean_pct_hb: 0.004, cycling_fraction: 0.93, flags: ["cycling"],
      reasons: ["Mostly cycling cells: 93% have a clear S or G2M score (above 0.1) vs 6% in a typical cluster."],
    });
    getClusterQc.mockResolvedValue({
      ...report([row("0", { mean_pct_hb: 0.012, cycling_fraction: 0.06 }), CYC, AMB, ERY]),
      metrics: ["genes", "counts", "pct_mt", "pct_ribo", "doublet_fraction", "pct_hb", "cycling_fraction"],
      typical: { median_genes: 1400, median_counts: 4200, median_pct_mt: 2.1, median_pct_ribo: 20, doublet_fraction: 0.02, mean_pct_hb: 0.012, cycling_fraction: 0.06 },
      flag_help: { hb_ambient: "Haemoglobin RNA without erythroid markers.", cycling: "Mostly cycling cells." },
    });
    render(<ClusterQcTable datasetId="d1" />);
    // The erythroid cluster is not counted as standing out.
    expect(await screen.findByText("2 of 4 clusters stand out")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Hb" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Cycling" })).toBeInTheDocument();
    expect(screen.getByText("48.2%")).toHaveAttribute("title", "Red-cell genes among its top markers: HBB, HBA2, HBA1");
    expect(screen.getByText("<0.01%")).toBeInTheDocument();
    expect(screen.getAllByText("0.01%").length).toBeGreaterThan(0); // typical + cluster 0
    expect(screen.getByText("Erythroid")).toHaveAttribute("title", ERY.notes![0]);
    expect(screen.getByText(/Real red-cell expression, not ambient contamination/)).toBeInTheDocument();
    expect(screen.getByText("Ambient Hb")).toHaveAttribute("title", AMB.reasons[0]);
    expect(screen.getByText("Cycling", { selector: "span" })).toHaveAttribute("title", CYC.reasons[0]);
    expect(screen.getByText("93%")).toBeInTheDocument();
    expect(askQuestion("leiden", AMB)).toContain("flagged as ambient hb");
  });

  it("opens on ambient haemoglobin across the data and lists technical drivers", async () => {
    const inside = {
      score: "log_counts", label: "Log total counts", pc: 2, r: 0.89, r_within: 0.95, pc_share: 0.12, inside_clusters: true,
      reason: "Log total counts correlates with PC2 (r = 0.89; PC2 carries 12% of the variance), and inside clusters too (r = 0.95).",
    };
    const between = {
      score: "pct_ribo", label: "Ribosomal %", pc: 2, r: -0.61, r_within: -0.45, pc_share: 0.12, inside_clusters: false,
      reason: "Ribosomal % correlates with PC2 (r = -0.61), but mostly between clusters.",
    };
    getClusterQc.mockResolvedValue({
      ...report([row("0"), row("1"), row("2")]),
      ambient_hb: "Haemoglobin RNA is spread across the non-erythroid clusters (0.8% of the reads in a typical one).",
      drivers: [inside, between], drivers_checked: ["Log total counts", "Ribosomal %"], drivers_space: "X_pca", drivers_n_pcs: 10,
    });
    render(<ClusterQcTable datasetId="d1" />);
    expect(await screen.findByText(/No cluster stands out \(3 clusters\) · 2 technical scores track a PC/)).toBeInTheDocument();
    expect(screen.getByRole("table")).toBeInTheDocument(); // opened by the ambient note + depth axis
    expect(screen.getByText(/spread across the non-erythroid clusters/)).toBeInTheDocument();
    expect(screen.getByText(/first 10 PCs of X_pca/)).toBeInTheDocument();
    expect(screen.getByText(inside.reason)).toBeInTheDocument();
    expect(screen.getByLabelText("Inside clusters too")).toBeInTheDocument();
    expect(screen.getByText(between.reason)).toBeInTheDocument();
  });

  it("stays shut for between-cluster drivers only and says when no score tracks a PC", async () => {
    getClusterQc.mockResolvedValue({
      ...report([row("0"), row("1"), row("2")]),
      drivers: [{
        score: "pct_ribo", label: "Ribosomal %", pc: 1, r: -0.78, r_within: -0.34, pc_share: 0.22, inside_clusters: false,
        reason: "Ribosomal % correlates with PC1 (r = -0.78), but mostly between clusters.",
      }],
      drivers_checked: ["Log total counts", "Ribosomal %"], drivers_space: "X_pca_harmony", drivers_n_pcs: 10,
    });
    const { unmount } = render(<ClusterQcTable datasetId="d1" />);
    expect(await screen.findByText(/· 1 technical score tracks a PC/)).toBeInTheDocument();
    expect(screen.queryByRole("table")).toBeNull();
    unmount();

    getClusterQc.mockResolvedValue({
      ...report([row("0"), row("1"), row("2")]),
      drivers: [], drivers_checked: ["Log total counts", "Ribosomal %", "G2M score"], drivers_space: "X_pca", drivers_n_pcs: 10,
    });
    render(<ClusterQcTable datasetId="d2" />);
    fireEvent.click(await screen.findByRole("button", { name: /cluster qc/i }));
    expect(screen.getByText(/None of Log total counts, Ribosomal %, G2M score correlates strongly/)).toBeInTheDocument();
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
