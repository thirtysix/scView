import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, render, screen, fireEvent } from "@testing-library/react";
import type { ClusterEvidenceReport, ClusterQcReport } from "@/api/qc";
import { hoverTip } from "@/test/hoverTip";

const { getClusterEvidence, getClusterQc } = vi.hoisted(() => ({
  getClusterEvidence: vi.fn(),
  getClusterQc: vi.fn(),
}));
vi.mock("@/api/qc", () => ({ getClusterEvidence, getClusterQc }));

import { ClusterEvidenceList } from "./ClusterEvidence";
import { ClusterQcTable } from "@/components/panels/ClusterQcTable";

const REPORT: ClusterEvidenceReport = {
  column: "leiden", n_groups: 3, markers: true, typing: false, annotation: null,
  notes: ["Database types are shown as shares of the evidence on named groupings only."],
  clusters: [
    { group: "0", n_cells: 1200, label: null, decided: false, flags: [], lines: [
      { kind: "markers", fired: false, text: "Markers match T cells better than random genes: best p 1e-17.6, while 1 in 10 random lists of 25 database genes reach 1e-5.7." },
      { kind: "qc", fired: false, text: "1,400 genes vs 1,370 typical; 5.5% vs 5.9% mito. No cluster QC flag." },
    ] },
    { group: "5", n_cells: 300, label: "Low-quality cells", decided: false, flags: ["low_quality"], lines: [
      { kind: "qc", fired: true, text: "1,115 genes vs 1,370 typical; 11.2% vs 5.9% mito. Cluster QC flags it low quality (few genes and high mito)." },
      { kind: "decision", fired: true, text: "AI annotation would label this Low-quality cells itself, without asking the AI." },
    ] },
    { group: "7", n_cells: 90, label: "Unidentified", decided: true, flags: [], lines: [
      { kind: "markers", fired: true, text: "Markers match no database type better than random genes: best p 1e-2.1 (NK cells), while 1 in 10 random lists of 18 database genes reach 1e-3.4." },
      { kind: "decision", fired: true, text: "scView labelled this Unidentified itself in 'cell_type', without asking the AI." },
    ] },
  ],
};

const details = () => screen.getAllByTestId("evidence-cluster") as HTMLDetailsElement[];

describe("ClusterEvidenceList", () => {
  beforeEach(() => {
    getClusterEvidence.mockReset().mockResolvedValue(REPORT);
  });

  it("shows each cluster's evidence, opening the ones where a rule fired", async () => {
    render(<ClusterEvidenceList datasetId="d1" column="leiden" />);
    expect(await screen.findByText(/11.2% vs 5.9% mito/)).toBeInTheDocument();
    expect(getClusterEvidence).toHaveBeenCalledWith("d1", "leiden");
    expect(details().map((d) => d.open)).toEqual([false, true, true]);
    // A recorded label and one the rules would give read differently; no probabilities anywhere.
    expect(screen.getByText("scView: Unidentified")).toBeInTheDocument();
    expect(screen.getByText("would be Low-quality cells")).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/probab/i);
    expect(screen.getByText(/named groupings only/)).toBeInTheDocument();
  });

  it("explains each kind of line on hover", async () => {
    render(<ClusterEvidenceList datasetId="d1" column="leiden" />);
    await screen.findByText(/11.2% vs 5.9% mito/);
    expect(await hoverTip(screen.getAllByText("Markers")[0]!)).toMatch(/random genes/);
  });

  it("keeps only the groups asked for, all open", async () => {
    render(<ClusterEvidenceList datasetId="d1" column="leiden" only={["7"]} />);
    expect(await screen.findByText(/best p 1e-2.1/)).toBeInTheDocument();
    expect(details().map((d) => d.open)).toEqual([true]);
    expect(screen.queryByText(/named groupings only/)).toBeNull();
  });

  it("says why when the evidence is unavailable", async () => {
    getClusterEvidence.mockRejectedValue(new Error('API error 400: {"detail":"Column \'x\' not found in obs."}'));
    render(<ClusterEvidenceList datasetId="d1" column="x" />);
    expect(await screen.findByText("Evidence unavailable: Column 'x' not found in obs.")).toBeInTheDocument();
  });

  it("says what it is doing only when the answer is slow (a first, uncached call)", async () => {
    vi.useFakeTimers();
    try {
      let answer!: (r: ClusterEvidenceReport) => void;
      getClusterEvidence.mockReturnValue(new Promise((r) => (answer = r)));
      render(<ClusterEvidenceList datasetId="d1" />);
      await act(async () => vi.advanceTimersByTime(900));
      expect(screen.queryByTestId("evidence-slow")).toBeNull();
      await act(async () => vi.advanceTimersByTime(200));
      expect(screen.getByTestId("evidence-slow")).toHaveTextContent(
        "Comparing each cluster's markers with random genes. The first time takes about 10 to 15 seconds.",
      );
      await act(async () => answer(REPORT));
      expect(screen.queryByTestId("evidence-slow")).toBeNull();
      expect(details()).toHaveLength(3);
    } finally {
      vi.useRealTimers();
    }
  });

  it("shows no waiting line for a cached (fast) answer", async () => {
    vi.useFakeTimers();
    try {
      render(<ClusterEvidenceList datasetId="d1" />);
      await act(async () => vi.advanceTimersByTime(50));
      expect(details()).toHaveLength(3);
      await act(async () => vi.advanceTimersByTime(2000));
      expect(screen.queryByTestId("evidence-slow")).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("Cluster QC > Evidence per cluster", () => {
  it("fetches the evidence only when opened, for the table's grouping", async () => {
    const qc: ClusterQcReport = {
      column: "leiden", n_cells: 1590, n_groups: 3, metrics: ["genes", "counts"],
      typical: { median_genes: 1370, median_counts: 4000 },
      clusters: ["0", "5", "7"].map((g) => ({
        group: g, n_cells: 100, fraction: 0.1, median_genes: 1370, median_counts: 4000, median_pct_mt: null,
        median_pct_ribo: null, doublet_fraction: null, median_doublet_score: null, flags: [], reasons: [],
      })),
      flag_help: {}, computed_on_demand: false, note: "",
    };
    getClusterQc.mockReset().mockResolvedValue(qc);
    getClusterEvidence.mockReset().mockResolvedValue(REPORT);
    render(<ClusterQcTable datasetId="d1" />);
    fireEvent.click(await screen.findByRole("button", { name: /Cluster QC/ }));
    const toggle = await screen.findByRole("button", { name: /Evidence per cluster/ });
    expect(getClusterEvidence).not.toHaveBeenCalled();
    fireEvent.click(toggle);
    expect(await screen.findByText(/11.2% vs 5.9% mito/)).toBeInTheDocument();
    expect(getClusterEvidence).toHaveBeenCalledWith("d1", "leiden");
  });
});
