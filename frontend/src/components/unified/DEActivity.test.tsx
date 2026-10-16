import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { DEResponse } from "@/api/de";

const { getDEActivity, getActivityCells } = vi.hoisted(() => ({
  getDEActivity: vi.fn(),
  getActivityCells: vi.fn(),
}));
vi.mock("@/api/activity", async (orig) => ({
  ...(await orig<typeof import("@/api/activity")>()),
  getDEActivity,
  getActivityCells,
}));

import { DEActivity } from "./DEActivity";

const RESULT: DEResponse = {
  n_selected: 10,
  n_rest: 10,
  label: "STIM",
  genes: [
    { gene: "ISG15", logfoldchange: 8, pval: 0, pval_adj: 0, stat: 56 },
    { gene: "CD14", logfoldchange: -2, pval: 1e-5, pval_adj: 1e-4, stat: -9 },
    { gene: "OLD", logfoldchange: 0.1, pval: 0.5, pval_adj: 0.9 },
  ],
};

const TF = {
  network: "collectri",
  label: "CollecTRI",
  kind: "tf",
  citation: "Müller-Dott et al. 2023",
  n_genes: 2,
  results: [
    { source: "STAT1", score: 19.1, pvalue: 1e-50, padj: 1e-48, n_targets: 200 },
    { source: "MYC", score: -7.4, pvalue: 1e-10, padj: 1e-9, n_targets: 400 },
  ],
};

describe("DEActivity", () => {
  beforeEach(() => {
    getDEActivity.mockReset().mockResolvedValue(TF);
    getActivityCells.mockReset().mockResolvedValue({ network: "collectri", source: "STAT1", n_targets: 200, scores: [0, 5] });
  });

  it("scores the DE statistic of every gene that has one", async () => {
    render(<DEActivity datasetId="d1" result={RESULT} labelA="STIM" labelB="CTRL" onScore={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /score/i }));
    await waitFor(() => screen.getByText("STAT1"));
    expect(getDEActivity).toHaveBeenCalledWith("d1", "collectri", ["ISG15", "CD14"], [56, -9]);
    expect(screen.getByText("19.10").className).toContain("text-red-600");
    expect(screen.getByText("-7.40").className).toContain("text-blue-600");
    expect(screen.getByText(/2 of 2 with adj. p < 0.05/)).toBeTruthy();
  });

  it("colours the cells by a row's source, and scores pathways on switching", async () => {
    const onScore = vi.fn();
    render(<DEActivity datasetId="d1" result={RESULT} labelA="STIM" labelB="CTRL" onScore={onScore} />);
    fireEvent.click(screen.getByRole("button", { name: /score/i }));
    await waitFor(() => screen.getByText("STAT1"));
    fireEvent.click(screen.getByText("STAT1"));
    await waitFor(() => expect(onScore).toHaveBeenCalled());
    expect(onScore.mock.calls[0]![1]).toBe("STAT1 (TF activity)");

    getDEActivity.mockResolvedValueOnce({ ...TF, network: "progeny", kind: "pathway", results: [] });
    fireEvent.click(screen.getByRole("button", { name: "Pathways (PROGENy)" }));
    await waitFor(() => expect(getDEActivity).toHaveBeenLastCalledWith("d1", "progeny", ["ISG15", "CD14"], [56, -9]));
  });

  it("cannot score a DE result without statistics (an older backend)", () => {
    const old = {
      ...RESULT,
      genes: RESULT.genes.map((g) => ({ gene: g.gene, logfoldchange: g.logfoldchange, pval: g.pval, pval_adj: g.pval_adj })),
    };
    render(<DEActivity datasetId="d1" result={old} labelA="STIM" labelB="CTRL" onScore={vi.fn()} />);
    expect((screen.getByRole("button", { name: /score/i }) as HTMLButtonElement).disabled).toBe(true);
  });
});
