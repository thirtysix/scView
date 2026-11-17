import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import type { ClusterQc, ClusterQcReport } from "@/api/qc";

const { getClusterQc } = vi.hoisted(() => ({ getClusterQc: vi.fn() }));
vi.mock("@/api/qc", () => ({ getClusterQc }));

import { ClusterQcTable, askQuestion } from "./ClusterQcTable";

function row(group: string, extra: Partial<ClusterQc> = {}): ClusterQc {
  return {
    group, n_cells: 97, fraction: 0.03, median_genes: 1400, median_counts: 4200,
    median_pct_mt: 2.1, median_pct_ribo: 20, doublet_fraction: null, median_doublet_score: null,
    flags: [], reasons: [], ...extra,
  };
}

const REASON =
  "Mostly one donor: 81% of its 97 cells are from CD_5 (donor_id), who holds 17% of all cells. Check it in the " +
  "other donors before calling it a cell type or a state of the condition.";

describe("ClusterQcTable one-donor flag", () => {
  beforeEach(() => getClusterQc.mockReset());

  it("shows the donor and share on the chip, the reason and the flag's help (QA-7: M1 CXCL5, 81% CD_5)", async () => {
    const nine = row("9", { flags: ["one_donor"], reasons: [REASON], top_donor: "CD_5", top_donor_share: 0.814,
      top_donor_share_of_data: 0.172 });
    const report: ClusterQcReport = {
      column: "leiden", n_cells: 3771, n_groups: 2, metrics: ["genes", "counts", "pct_mt"],
      typical: { median_genes: 1400, median_counts: 4200, median_pct_mt: 2.1 },
      clusters: [row("0"), nine], computed_on_demand: false, note: "", donor_column: "donor_id",
      flag_help: { one_donor: "Most of its cells come from one donor." },
    };
    getClusterQc.mockResolvedValue(report);
    render(<ClusterQcTable datasetId="d1" />);
    const chip = await screen.findByText("Mostly one donor (81% from CD_5)");
    expect(chip).toHaveAttribute("title", REASON);
    expect(screen.getAllByText(REASON).length).toBeGreaterThan(0);
    expect(screen.getByText("Most of its cells come from one donor.")).toBeInTheDocument();
    expect(askQuestion("leiden", nine)).toContain("flagged as mostly one donor. Mostly one donor: 81%");
  });
});
